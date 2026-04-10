"""
Per-frame video inpainting pipeline:
  1. Decode video sequentially for one episode
  2. FK -> mesh mask -> smooth mask per frame (with hand joint state)
  3. Render overlay per frame
  4. LaMa inpaint per frame
  5. Encode 4 output videos: original, mask, overlay, inpaint

Usage:
  python scripts/video_inpaint.py --episode 0
  python scripts/video_inpaint.py --episode 50 --no-grabcut
  python scripts/video_inpaint.py --episode 4 --start 5 --duration 5
"""

import sys
import os
import argparse
import time
import numpy as np
import pandas as pd
import cv2
import xml.etree.ElementTree as ET
import pinocchio as pin
from stl import mesh as stl_mesh

sys.stdout.reconfigure(line_buffering=True)

from config import (
    BASE_DIR, G1_URDF, MESH_DIR, BEST_PARAMS, SKIP_MESHES,
    ACTIVE_DATA_DIR, OUTPUT_DIR, get_hand_type,
)

# Aliases for backward compatibility with importers
URDF_PATH = G1_URDF


def build_q(model, rq, hand_state=None, hand_type="inspire"):
    """Map dataset rq (36) + hand_state (12) to Inspire FTP URDF q (60).

    Dataset rq layout:
      rq[0:3]   position
      rq[3:7]   quaternion (w,x,y,z)
      rq[7:29]  left leg(6) + right leg(6) + waist(3) + left arm(7)
      rq[29:36] right arm(7)

    hand_state layout (Inspire Dex5, 0=open 1=closed):
      [0] left index  [1] left middle  [2] left ring  [3] left little
      [4] left thumb close  [5] left thumb tilt
      [6] right index  [7] right middle  [8] right ring  [9] right little
      [10] right thumb close  [11] right thumb tilt

    hand_state layout (BrainCo, 0=open 1=closed):
      [0] left thumb close  [1] left thumb tilt
      [2] left index  [3] left middle  [4] left ring  [5] left little
      [6] right thumb close  [7] right thumb tilt
      [8] right index  [9] right middle  [10] right ring  [11] right little

    hand_type: "inspire" or "brainco" — controls hand_state index mapping.

    URDF q layout (nq=60):
      q[0:7]   freeflyer (pos + quat x,y,z,w)
      q[7:29]  left leg(6) + right leg(6) + waist(3) + left arm(7)
      q[29:41] left hand(12): index(2), little(2), middle(2), ring(2), thumb(4)
      q[41:48] right arm(7)
      q[48:60] right hand(12): index(2), little(2), middle(2), ring(2), thumb(4)
    """
    q = pin.neutral(model)
    q[0:3] = rq[0:3]
    q[3], q[4], q[5], q[6] = rq[4], rq[5], rq[6], rq[3]
    q[7:29] = rq[7:29]
    q[41:48] = rq[29:36]  # right arm

    if hand_state is not None:
        hs = hand_state
        if hand_type == "brainco":
            # Remap BrainCo -> Inspire layout per hand (6 values each)
            # BrainCo: [thumb_close, thumb_tilt, index, middle, ring, little]
            # Inspire: [index, middle, ring, little, thumb_close, thumb_tilt]
            hs = np.array(hs, dtype=np.float64)
            hs = np.concatenate([
                hs[2:6], hs[0:2],    # left: index,mid,ring,little,thumb_c,thumb_t
                hs[8:12], hs[6:8],   # right: same reorder
            ])
        # Finger joint limits: _1 max=1.4381, _2 mimics _1 * 1.0843
        # Thumb: _1 max=1.1641 (tilt), _2 max=0.5864 (close),
        #        _3 mimics _2 * 0.8024, _4 mimics _3 * 0.9487

        # Left hand q[29:41]
        q[29] = hs[0] * 1.4381                       # left_index_1
        q[30] = hs[0] * 1.4381 * 1.0843              # left_index_2 (mimic)
        q[31] = hs[3] * 1.4381                       # left_little_1
        q[32] = hs[3] * 1.4381 * 1.0843              # left_little_2 (mimic)
        q[33] = hs[1] * 1.4381                       # left_middle_1
        q[34] = hs[1] * 1.4381 * 1.0843              # left_middle_2 (mimic)
        q[35] = hs[2] * 1.4381                       # left_ring_1
        q[36] = hs[2] * 1.4381 * 1.0843              # left_ring_2 (mimic)
        q[37] = hs[5] * 1.1641                       # left_thumb_1 (tilt)
        q[38] = hs[4] * 0.5864                       # left_thumb_2 (close)
        q[39] = hs[4] * 0.5864 * 0.8024              # left_thumb_3 (mimic)
        q[40] = hs[4] * 0.5864 * 0.8024 * 0.9487     # left_thumb_4 (mimic)

        # Right hand q[48:60]
        q[48] = hs[6] * 1.4381                       # right_index_1
        q[49] = hs[6] * 1.4381 * 1.0843              # right_index_2 (mimic)
        q[50] = hs[9] * 1.4381                       # right_little_1
        q[51] = hs[9] * 1.4381 * 1.0843              # right_little_2 (mimic)
        q[52] = hs[7] * 1.4381                       # right_middle_1
        q[53] = hs[7] * 1.4381 * 1.0843              # right_middle_2 (mimic)
        q[54] = hs[8] * 1.4381                       # right_ring_1
        q[55] = hs[8] * 1.4381 * 1.0843              # right_ring_2 (mimic)
        q[56] = hs[11] * 1.1641                      # right_thumb_1 (tilt)
        q[57] = hs[10] * 0.5864                      # right_thumb_2 (close)
        q[58] = hs[10] * 0.5864 * 0.8024              # right_thumb_3 (mimic)
        q[59] = hs[10] * 0.5864 * 0.8024 * 0.9487     # right_thumb_4 (mimic)

    return q


def do_fk(model, data, q):
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    transforms = {}
    for i in range(model.nframes):
        name = model.frames[i].name
        T = data.oMf[i]
        transforms[name] = (T.translation.copy(), T.rotation.copy())
    return transforms


def parse_urdf_meshes(urdf_path):
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    link_meshes = {}
    for link_elem in root.findall("link"):
        name = link_elem.get("name")
        visual = link_elem.find("visual")
        if visual is None:
            continue
        geom = visual.find("geometry")
        if geom is None:
            continue
        mesh_elem = geom.find("mesh")
        if mesh_elem is None:
            continue
        filename = mesh_elem.get("filename")
        if filename:
            link_meshes[name] = os.path.basename(filename)
    return link_meshes


def preload_meshes(link_meshes, mesh_dir, skip_set=None, subsample=4):
    """Load all STL meshes once. Returns dict of link_name -> (triangles, unique_verts).

    subsample: keep every Nth triangle to reduce mesh complexity.
    """
    if skip_set is None:
        skip_set = SKIP_MESHES
    cache = {}
    for link_name, filename in link_meshes.items():
        if link_name in skip_set:
            continue
        path = os.path.join(mesh_dir, filename)
        if not os.path.exists(path):
            continue
        m = stl_mesh.Mesh.from_file(path)
        verts = m.vectors
        flat = verts.reshape(-1, 3)
        valid_per_vert = np.all(np.isfinite(flat), axis=1)
        valid_per_tri = valid_per_vert.reshape(-1, 3).all(axis=1)
        tris = verts[valid_per_tri]
        if subsample > 1:
            tris = tris[::subsample]
        flat_all = m.vectors.reshape(-1, 3)
        valid_all = np.all(np.isfinite(flat_all), axis=1)
        unique_verts = np.unique(flat_all[valid_all], axis=0)
        if subsample > 1:
            unique_verts = unique_verts[::subsample]
        if len(tris) > 0:
            cache[link_name] = (tris, unique_verts)
    return cache


def make_camera_const(params):
    """Precompute the camera rotation/intrinsics that are constant across frames."""
    p = params
    pitch = np.radians(p["pitch"])
    yaw = np.radians(p["yaw"])
    roll = np.radians(p["roll"])

    R_pitch = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    R_yaw = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw),  np.cos(yaw), 0],
        [0, 0, 1]
    ])
    R_roll = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll),  np.cos(roll)]
    ])
    R_body_to_cam = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]], dtype=np.float64)
    R_cam = R_body_to_cam @ R_roll @ R_yaw @ R_pitch

    K = np.array([[p["fx"], 0, p["cx"]],
                   [0, p["fy"], p["cy"]],
                   [0, 0, 1]], dtype=np.float64)
    D = np.array([p["k1"], p["k2"], p["k3"], p["k4"]], dtype=np.float64).reshape(4, 1)
    offset = np.array([p["dx"], p["dy"], p["dz"]], dtype=np.float64)
    return {"R_cam": R_cam, "K": K, "D": D, "offset": offset}


def make_camera(params, transforms, _const=None):
    """Build camera from params + per-frame torso transform.

    Optionally accepts precomputed constants from make_camera_const().
    """
    if _const is None:
        _const = make_camera_const(params)

    R_cam = _const["R_cam"]
    K = _const["K"]
    D = _const["D"]
    offset = _const["offset"]

    ref_t, ref_R = transforms["torso_link"]
    cam_pos = ref_t + ref_R @ offset
    R_w2c = (ref_R @ R_cam.T).T
    t_w2c = R_w2c @ (-cam_pos)

    rvec, _ = cv2.Rodrigues(R_w2c)
    tvec = t_w2c.reshape(3, 1)
    return K, D, rvec, tvec, R_w2c, t_w2c


def render_mask_and_overlay(img, mesh_cache, transforms, params, h, w, _cam_const=None):
    """Render both mask and overlay in a single pass (shared FK transform + projection)."""
    K, D, rvec, tvec, R_w2c, t_w2c = make_camera(params, transforms, _cam_const)
    t_w2c_flat = t_w2c.flatten()

    # --- Batch all triangle vertices from all links ---
    all_world_tris = []
    all_tri_counts = []
    # --- Batch unique verts per link for overlay ---
    overlay_links = []  # (link_name, world_verts)

    for link_name, (tris, unique_verts) in mesh_cache.items():
        if link_name not in transforms:
            continue
        t_link, R_link = transforms[link_name]

        # Triangle vertices → world coords
        flat = tris.reshape(-1, 3)
        world = (R_link @ flat.T).T + t_link
        all_world_tris.append(world)
        all_tri_counts.append(len(tris))

        # Unique verts for overlay
        if len(unique_verts) > 0:
            verts_w = (R_link @ unique_verts.T).T + t_link
            overlay_links.append((link_name, verts_w))

    # --- Mask: batched projection ---
    mask = np.zeros((h, w), dtype=np.uint8)
    if all_world_tris:
        all_world = np.concatenate(all_world_tris, axis=0).astype(np.float64)
        cam_pts = (R_w2c @ all_world.T).T + t_w2c_flat
        z_all = cam_pts[:, 2]

        pts2d, _ = cv2.fisheye.projectPoints(
            all_world.reshape(-1, 1, 3), rvec, tvec, K, D)
        pts2d = pts2d.reshape(-1, 2)

        # Split back by link and filter/draw triangles
        offset = 0
        for n_tris in all_tri_counts:
            n_pts = n_tris * 3
            z_seg = z_all[offset:offset + n_pts].reshape(n_tris, 3)
            pts_seg = pts2d[offset:offset + n_pts].reshape(n_tris, 3, 2)

            valid = (z_seg > 0.01).all(axis=1)
            pts_valid = pts_seg[valid]
            if len(pts_valid) > 0:
                finite = np.all(np.isfinite(pts_valid), axis=(1, 2))
                pts_valid = pts_valid[finite]
                if len(pts_valid) > 0:
                    cv2.fillPoly(mask, pts_valid.astype(np.int32), 255)

            offset += n_pts

        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

    # --- Overlay: per-link convex hull (needs per-link color) ---
    overlay = np.zeros_like(img)
    result = img.copy()
    for link_name, verts_w in overlay_links:
        verts_f64 = verts_w.astype(np.float64)
        depths = (R_w2c @ verts_f64.T).T + t_w2c_flat
        in_front = depths[:, 2] > 0.01
        if np.count_nonzero(in_front) < 3:
            continue

        pts2d_ov, _ = cv2.fisheye.projectPoints(
            verts_f64[in_front].reshape(-1, 1, 3), rvec, tvec, K, D)
        pts2d_ov = pts2d_ov.reshape(-1, 2)
        finite = np.all(np.isfinite(pts2d_ov), axis=1)
        pts2d_ov = pts2d_ov[finite]
        if len(pts2d_ov) < 3:
            continue

        hull = cv2.convexHull(pts2d_ov.astype(np.float32))
        if "left" in link_name:
            color = (255, 180, 0)
        elif "right" in link_name:
            color = (0, 180, 255)
        else:
            color = (0, 255, 180)
        cv2.fillConvexPoly(overlay, hull.astype(np.int32), color)
        cv2.polylines(result, [hull.astype(np.int32)], True, color, 1, cv2.LINE_AA)

    cv2.addWeighted(overlay, 0.35, result, 1.0, 0, result)
    return mask, result


# --- Backward-compatible wrappers (used by sam2_inpaint_pipeline, sam2_segment) ---

def render_mask(mesh_cache, transforms, params, h, w):
    """Render per-triangle mask using preloaded meshes."""
    K, D, rvec, tvec, R_w2c, t_w2c = make_camera(params, transforms)
    t_w2c_flat = t_w2c.flatten()

    all_world_tris = []
    all_tri_counts = []
    for link_name, (tris, _) in mesh_cache.items():
        if link_name not in transforms:
            continue
        t_link, R_link = transforms[link_name]
        flat = tris.reshape(-1, 3)
        world = (R_link @ flat.T).T + t_link
        all_world_tris.append(world)
        all_tri_counts.append(len(tris))

    mask = np.zeros((h, w), dtype=np.uint8)
    if not all_world_tris:
        return mask

    all_world = np.concatenate(all_world_tris, axis=0).astype(np.float64)
    cam_pts = (R_w2c @ all_world.T).T + t_w2c_flat
    z_all = cam_pts[:, 2]

    pts2d, _ = cv2.fisheye.projectPoints(
        all_world.reshape(-1, 1, 3), rvec, tvec, K, D)
    pts2d = pts2d.reshape(-1, 2)

    offset = 0
    for n_tris in all_tri_counts:
        n_pts = n_tris * 3
        z_seg = z_all[offset:offset + n_pts].reshape(n_tris, 3)
        pts_seg = pts2d[offset:offset + n_pts].reshape(n_tris, 3, 2)

        valid = (z_seg > 0.01).all(axis=1)
        pts_valid = pts_seg[valid]
        if len(pts_valid) > 0:
            finite = np.all(np.isfinite(pts_valid), axis=(1, 2))
            pts_valid = pts_valid[finite]
            if len(pts_valid) > 0:
                cv2.fillPoly(mask, pts_valid.astype(np.int32), 255)

        offset += n_pts

    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
    return mask


def render_overlay(img, mesh_cache, transforms, params):
    """Semi-transparent convex hull overlay."""
    K, D, rvec, tvec, R_w2c, t_w2c = make_camera(params, transforms)
    overlay = np.zeros_like(img)
    result = img.copy()

    for link_name, (_, unique_verts) in mesh_cache.items():
        if link_name not in transforms or len(unique_verts) == 0:
            continue

        t_link, R_link = transforms[link_name]
        verts3d = ((R_link @ unique_verts.T).T + t_link).astype(np.float64)

        depths = (R_w2c @ verts3d.T).T + t_w2c.flatten()
        in_front = depths[:, 2] > 0.01
        if np.count_nonzero(in_front) < 3:
            continue

        pts2d, _ = cv2.fisheye.projectPoints(
            verts3d[in_front].reshape(-1, 1, 3), rvec, tvec, K, D)
        pts2d = pts2d.reshape(-1, 2)
        finite = np.all(np.isfinite(pts2d), axis=1)
        pts2d = pts2d[finite]
        if len(pts2d) < 3:
            continue

        hull = cv2.convexHull(pts2d.astype(np.float32))
        if "left" in link_name:
            color = (255, 180, 0)
        elif "right" in link_name:
            color = (0, 180, 255)
        else:
            color = (0, 255, 180)
        cv2.fillConvexPoly(overlay, hull.astype(np.int32), color)
        cv2.polylines(result, [hull.astype(np.int32)], True, color, 1, cv2.LINE_AA)

    cv2.addWeighted(overlay, 0.35, result, 1.0, 0, result)
    return result


def grabcut_refine(img, mesh_mask, gc_iter=3):
    """GrabCut expansion (fewer iterations for speed)."""
    h, w = img.shape[:2]
    gc_mask = np.full((h, w), cv2.GC_BGD, dtype=np.uint8)
    gc_mask[mesh_mask > 0] = cv2.GC_FGD
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    dilated = cv2.dilate(mesh_mask, kernel)
    gc_mask[(dilated > 0) & (mesh_mask == 0)] = cv2.GC_PR_FGD
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    cv2.grabCut(img, gc_mask, None, bgdModel, fgdModel, gc_iter, cv2.GC_INIT_WITH_MASK)
    return np.where(
        (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0
    ).astype(np.uint8)


def postprocess_mask(mask):
    """平滑 → 膨胀 → 边缘模糊."""
    # 1. 平滑：去锯齿
    out = cv2.GaussianBlur(mask, (7, 7), 0)
    out = (out > 128).astype(np.uint8) * 255
    # 2. 膨胀：5px 安全边距
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    out = cv2.dilate(out, kernel)
    # 3. 边缘模糊：渐变带（LaMa 支持灰度 mask）
    blurred = cv2.GaussianBlur(out, (31, 31), 0)
    out = np.maximum(out, blurred)
    return out


def init_lama():
    from simple_lama_inpainting import SimpleLama
    return SimpleLama()


def run_lama(lama, img_bgr, mask):
    from PIL import Image
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    pil_mask = Image.fromarray(mask).convert("L")
    result = lama(pil_img, pil_mask)
    return cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)


def open_video_writer(path, w, h, fps=30):
    import av
    container = av.open(path, mode='w')
    stream = container.add_stream('libx264', rate=fps)
    stream.width = w
    stream.height = h
    stream.pix_fmt = 'yuv420p'
    stream.options = {'crf': '18', 'preset': 'medium'}
    return container, stream


def write_frame(container, stream, img_bgr):
    import av
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    frame = av.VideoFrame.from_ndarray(img_rgb, format='rgb24')
    for packet in stream.encode(frame):
        container.mux(packet)


def close_video(container, stream):
    for packet in stream.encode():
        container.mux(packet)
    container.close()


def load_episode_info(ep, data_dir=None):
    """Load episode meta and return (video_path, from_ts, to_ts, ep_df)."""
    if data_dir is None:
        data_dir = ACTIVE_DATA_DIR
    meta = pd.read_parquet(os.path.join(data_dir, "meta", "episodes",
                                         "chunk-000", "file-000.parquet"))
    ep_meta = meta[meta["episode_index"] == ep]
    if len(ep_meta) == 0:
        raise ValueError(f"Episode {ep} not found in meta")
    ep_meta = ep_meta.iloc[0]

    file_idx = int(ep_meta["videos/observation.images.head_stereo_left/file_index"])
    from_ts = float(ep_meta["videos/observation.images.head_stereo_left/from_timestamp"])
    to_ts = float(ep_meta["videos/observation.images.head_stereo_left/to_timestamp"])

    video_path = os.path.join(data_dir, "videos",
                               "observation.images.head_stereo_left",
                               "chunk-000", f"file-{file_idx:03d}.mp4")

    # Load parquet (determine which file)
    data_fi = int(ep_meta.get("data/file_index", 0))
    parquet_path = os.path.join(data_dir, "data", "chunk-000",
                                 f"file-{data_fi:03d}.parquet")
    df = pd.read_parquet(parquet_path)
    ep_df = df[df["episode_index"] == ep].sort_values("frame_index")

    return video_path, from_ts, to_ts, ep_df


def main():
    import av

    parser = argparse.ArgumentParser(description="Per-frame video inpainting")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--start", type=float, default=0,
                        help="Start offset in seconds within episode (default 0)")
    parser.add_argument("--duration", type=float, default=0,
                        help="Duration in seconds (default 0 = entire episode)")
    args = parser.parse_args()

    out_dir = os.path.join(OUTPUT_DIR, "inpaint_video")
    os.makedirs(out_dir, exist_ok=True)
    ep = args.episode

    # Load episode info
    print(f"Episode: {ep}")
    video_path, from_ts, to_ts, ep_df = load_episode_info(ep)
    n_total = len(ep_df)
    fps = 30
    ep_duration = n_total / fps
    print(f"Episode length: {n_total} frames ({ep_duration:.1f}s)")
    print(f"Video file: {os.path.basename(video_path)}, ts={from_ts:.1f}-{to_ts:.1f}")

    # Apply start/duration trim
    start_frame = int(args.start * fps) if args.start > 0 else 0
    if args.duration > 0:
        end_frame = min(start_frame + int(args.duration * fps), n_total)
    else:
        end_frame = n_total

    ep_df = ep_df.iloc[start_frame:end_frame]
    n_frames = len(ep_df)
    print(f"Processing frames {start_frame}-{end_frame - 1} ({n_frames} frames, {n_frames/fps:.1f}s)")

    # Build frame data lookup: frame_index -> (rq, hand_state)
    frame_data = {}
    for _, row in ep_df.iterrows():
        fi = int(row["frame_index"])
        rq = np.array(row["observation.state.robot_q_current"], dtype=np.float64)
        hs = np.array(row["observation.state.hand_state"], dtype=np.float64)
        frame_data[fi] = (rq, hs)

    # Compute video frame range (using timestamps)
    vid_from_frame = int(round((from_ts + start_frame / fps) * fps))
    vid_to_frame = int(round((from_ts + (end_frame - 1) / fps) * fps))

    # Load URDF + meshes
    print("Loading URDF and meshes...")
    model = pin.buildModelFromUrdf(G1_URDF, pin.JointModelFreeFlyer())
    data_pin = model.createData()
    link_meshes = parse_urdf_meshes(G1_URDF)
    mesh_cache = preload_meshes(link_meshes, MESH_DIR)
    print(f"Loaded {len(mesh_cache)} link meshes")

    # Init LaMa
    print("Loading LaMa model...")
    lama = init_lama()

    # Open input video and seek to episode start
    container_in = av.open(video_path)
    stream_in = container_in.streams.video[0]
    vid_fps = float(stream_in.average_rate)

    seek_ts = from_ts + start_frame / fps
    if seek_ts > 1.0:
        target_pts = int((seek_ts - 1.0) / stream_in.time_base)
        container_in.seek(max(target_pts, 0), stream=stream_in)

    # Open output videos
    tag = f"ep{ep:03d}"
    if args.start > 0 or args.duration > 0:
        tag += f"_{start_frame}-{end_frame}"

    writers_ready = False
    writers = {}

    processed = 0
    prev_inpainted = None  # EMA temporal blending
    ema_alpha = 0.6        # current frame weight (0.6 current + 0.4 previous)
    t_start = time.time()

    print(f"Processing {n_frames} frames...")
    for av_frame in container_in.decode(stream_in):
        pts_sec = float(av_frame.pts * stream_in.time_base)
        vid_fn = int(round(pts_sec * vid_fps))

        # Map video frame number to episode frame_index
        ep_fi = int(round((pts_sec - from_ts) * fps))

        if ep_fi < start_frame:
            continue
        if ep_fi >= end_frame:
            break

        if ep_fi not in frame_data:
            continue

        img = av_frame.to_ndarray(format='bgr24')

        if not writers_ready:
            h, w = img.shape[:2]
            for vname in ["original", "mask", "overlay", "inpaint"]:
                vpath = os.path.join(out_dir, f"{tag}_{vname}.mp4")
                c, s = open_video_writer(vpath, w, h, fps=int(vid_fps))
                writers[vname] = (c, s)
            writers_ready = True

        rq, hs = frame_data[ep_fi]

        # FK with hand state
        q = build_q(model, rq, hs, hand_type=get_hand_type())
        transforms = do_fk(model, data_pin, q)

        # Mask: raw → GrabCut → 平滑 → 膨胀 → 边缘模糊
        raw_mask = render_mask(mesh_cache, transforms, BEST_PARAMS, h, w)
        gc_mask = grabcut_refine(img, raw_mask)
        final_mask = postprocess_mask(gc_mask)

        # Overlay
        overlay_img = render_overlay(img, mesh_cache, transforms, BEST_PARAMS)

        # Inpaint
        inpainted = run_lama(lama, img, final_mask)

        # Temporal blending: EMA on mask region only
        if prev_inpainted is not None:
            mask_region = final_mask > 128
            blended = inpainted.copy()
            blended[mask_region] = (
                ema_alpha * inpainted[mask_region].astype(np.float32) +
                (1 - ema_alpha) * prev_inpainted[mask_region].astype(np.float32)
            ).astype(np.uint8)
            inpainted = blended
        prev_inpainted = inpainted.copy()

        # Write frames
        write_frame(*writers["original"], img)
        write_frame(*writers["mask"], cv2.cvtColor(final_mask, cv2.COLOR_GRAY2BGR))
        write_frame(*writers["overlay"], overlay_img)
        write_frame(*writers["inpaint"], inpainted)

        processed += 1
        if processed % 50 == 0 or processed == 1:
            elapsed = time.time() - t_start
            fps_proc = processed / elapsed
            eta = (n_frames - processed) / fps_proc if fps_proc > 0 else 0
            print(f"  {processed}/{n_frames} frames "
                  f"({fps_proc:.1f} fps, ETA {eta:.0f}s)")

    container_in.close()

    for vname, (c, s) in writers.items():
        close_video(c, s)

    elapsed = time.time() - t_start
    print(f"\nDone: {processed} frames in {elapsed:.1f}s "
          f"({processed/elapsed:.1f} fps)")
    print(f"Output: {out_dir}/{tag}_*.mp4")


if __name__ == "__main__":
    main()
