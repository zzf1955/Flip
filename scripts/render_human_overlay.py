"""
Render human arm overlay on G1 ego-centric video frames.

Uses the human_arm_overlay URDF with the same kinematic chain and build_q
mapping as the G1 Inspire hand URDF. Joint angles pass through directly
since G1 arm DOF structure matches human arm DOF exactly.

Usage:
  python scripts/render_human_overlay.py --episode 0
  python scripts/render_human_overlay.py --episode 0 --frame 50
  python scripts/render_human_overlay.py --episode 0 --frame 50 --side-by-side
"""

import sys
import os
import argparse
import numpy as np
import pandas as pd
import cv2
import xml.etree.ElementTree as ET
import pinocchio as pin
from stl import mesh as stl_mesh

sys.stdout.reconfigure(line_buffering=True)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Two URDFs: G1 robot (for mask) and human (for overlay)
G1_URDF = os.path.join(BASE_DIR, "data", "mesh",
                       "g1_29dof_rev_1_0_with_inspire_hand_FTP.urdf")
HUMAN_URDF = os.path.join(BASE_DIR, "data", "mesh",
                          "human_arm_overlay.urdf")
MESH_DIR = os.path.join(BASE_DIR, "data", "mesh", "meshes")
DATA_DIR = os.path.join(BASE_DIR, "data", "video", "G1_WBT_Brainco_Make_The_Bed")
OUTPUT_DIR = os.path.join(BASE_DIR, "test_results", "human_overlay")

G1_SKIP = {"head_link", "logo_link", "d435_link"}
HUMAN_SKIP = {"head_link"}  # only head_link exists without mesh anyway

SKIN_COLOR = (135, 165, 215)  # BGR: warm skin tone

# Best params from PSO (IoU=0.8970)
BEST_PARAMS = {
    "dx": 0.039, "dy": 0.052, "dz": 0.536,
    "pitch": -53.6, "yaw": 4.7, "roll": 3.0,
    "fx": 315, "fy": 302, "cx": 334, "cy": 230,
    "k1": 0.63, "k2": 0.17, "k3": 1.19, "k4": 0.25,
}


# --- Shared pipeline functions (from video_inpaint.py) ---

def build_q(model, rq, hand_state=None):
    """Map dataset rq (36) + hand_state (12) to URDF q (60).
    Works for both G1 Inspire and human URDF (same joint layout).
    """
    q = pin.neutral(model)
    q[0:3] = rq[0:3]
    q[3], q[4], q[5], q[6] = rq[4], rq[5], rq[6], rq[3]
    q[7:29] = rq[7:29]
    q[41:48] = rq[29:36]

    if hand_state is not None:
        hs = hand_state
        # Left hand q[29:41]
        q[29] = hs[0] * 1.4381
        q[30] = hs[0] * 1.4381 * 1.0843
        q[31] = hs[3] * 1.4381
        q[32] = hs[3] * 1.4381 * 1.0843
        q[33] = hs[1] * 1.4381
        q[34] = hs[1] * 1.4381 * 1.0843
        q[35] = hs[2] * 1.4381
        q[36] = hs[2] * 1.4381 * 1.0843
        q[37] = hs[5] * 1.1641
        q[38] = hs[4] * 0.5864
        q[39] = hs[4] * 0.5864 * 0.8024
        q[40] = hs[4] * 0.5864 * 0.8024 * 0.9487
        # Right hand q[48:60]
        q[48] = hs[6] * 1.4381
        q[49] = hs[6] * 1.4381 * 1.0843
        q[50] = hs[9] * 1.4381
        q[51] = hs[9] * 1.4381 * 1.0843
        q[52] = hs[7] * 1.4381
        q[53] = hs[7] * 1.4381 * 1.0843
        q[54] = hs[8] * 1.4381
        q[55] = hs[8] * 1.4381 * 1.0843
        q[56] = hs[11] * 1.1641
        q[57] = hs[10] * 0.5864
        q[58] = hs[10] * 0.5864 * 0.8024
        q[59] = hs[10] * 0.5864 * 0.8024 * 0.9487
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


def preload_meshes(link_meshes, mesh_dir, skip_set):
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
        flat_all = m.vectors.reshape(-1, 3)
        valid_all = np.all(np.isfinite(flat_all), axis=1)
        unique_verts = np.unique(flat_all[valid_all], axis=0)
        if len(tris) > 0:
            cache[link_name] = (tris, unique_verts)
    return cache


def make_camera(params, transforms):
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

    ref_t, ref_R = transforms["torso_link"]
    cam_pos = ref_t + ref_R @ np.array([p["dx"], p["dy"], p["dz"]])
    R_w2c = (ref_R @ R_cam.T).T
    t_w2c = R_w2c @ (-cam_pos)

    K = np.array([[p["fx"], 0, p["cx"]],
                   [0, p["fy"], p["cy"]],
                   [0, 0, 1]], dtype=np.float64)
    D = np.array([p["k1"], p["k2"], p["k3"], p["k4"]], dtype=np.float64).reshape(4, 1)
    rvec, _ = cv2.Rodrigues(R_w2c)
    tvec = t_w2c.reshape(3, 1)
    return K, D, rvec, tvec, R_w2c, t_w2c


# --- Human-specific rendering ---

def render_human_triangles(img, mesh_cache, transforms, params, color=SKIN_COLOR):
    """Render human arm mesh with per-triangle fill and simple shading."""
    K, D, rvec, tvec, R_w2c, t_w2c = make_camera(params, transforms)
    h, w = img.shape[:2]
    result = img.copy()

    # Collect all triangles with depth for painter's algorithm
    all_tris_2d = []
    all_depths = []
    all_normals_dot = []

    for link_name, (tris, _) in mesh_cache.items():
        if link_name not in transforms:
            continue

        t_link, R_link = transforms[link_name]
        flat = tris.reshape(-1, 3)
        world = (R_link @ flat.T).T + t_link

        cam_pts = (R_w2c @ world.T).T + t_w2c.flatten()
        z_cam = cam_pts[:, 2]

        pts2d, _ = cv2.fisheye.projectPoints(
            world.reshape(-1, 1, 3), rvec, tvec, K, D)
        pts2d = pts2d.reshape(-1, 2)

        n_tri = len(tris)
        z_tri = z_cam.reshape(n_tri, 3)
        pts_tri = pts2d.reshape(n_tri, 3, 2)

        valid = (z_tri > 0.01).all(axis=1)
        pts_tri = pts_tri[valid]
        z_tri = z_tri[valid]
        if len(pts_tri) == 0:
            continue

        finite = np.all(np.isfinite(pts_tri), axis=(1, 2))
        pts_tri = pts_tri[finite]
        z_tri = z_tri[finite]

        # Compute face normals in camera space for shading
        world_tris = world.reshape(n_tri, 3, 3)[valid][finite]
        v0, v1, v2 = world_tris[:, 0], world_tris[:, 1], world_tris[:, 2]
        normals = np.cross(v1 - v0, v2 - v0)
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        normals = normals / norms

        # View direction: from triangle center toward camera
        centers = world_tris.mean(axis=1)
        view_dirs = cam_pts.reshape(n_tri, 3, 3)[valid][finite].mean(axis=1)
        view_dirs = -view_dirs  # toward camera
        view_norms = np.linalg.norm(view_dirs, axis=1, keepdims=True)
        view_norms = np.maximum(view_norms, 1e-8)
        view_dirs = view_dirs / view_norms

        # Lambertian: abs dot product (abs to handle both face orientations)
        dots = np.abs(np.sum(normals * view_dirs, axis=1))

        avg_depths = z_tri.mean(axis=1)
        all_tris_2d.append(pts_tri)
        all_depths.append(avg_depths)
        all_normals_dot.append(dots)

    if not all_tris_2d:
        return result

    all_tris_2d = np.concatenate(all_tris_2d)
    all_depths = np.concatenate(all_depths)
    all_normals_dot = np.concatenate(all_normals_dot)

    # Painter's algorithm: sort far to near
    order = np.argsort(-all_depths)

    for idx in order:
        tri = all_tris_2d[idx].astype(np.int32)
        shade = 0.4 + 0.6 * all_normals_dot[idx]  # ambient 0.4 + diffuse 0.6
        shaded_color = tuple(int(c * shade) for c in color)
        cv2.fillPoly(result, [tri], shaded_color)

    return result


def render_g1_overlay(img, mesh_cache, transforms, params):
    """Semi-transparent G1 robot overlay (for comparison)."""
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
        color = (255, 180, 0) if "left" in link_name else (0, 180, 255)
        cv2.fillConvexPoly(overlay, hull.astype(np.int32), color)

    cv2.addWeighted(overlay, 0.35, result, 1.0, 0, result)
    return result


def load_episode_info(ep):
    meta = pd.read_parquet(os.path.join(DATA_DIR, "meta", "episodes",
                                         "chunk-000", "file-000.parquet"))
    ep_meta = meta[meta["episode_index"] == ep]
    if len(ep_meta) == 0:
        raise ValueError(f"Episode {ep} not found")
    ep_meta = ep_meta.iloc[0]

    file_idx = int(ep_meta["videos/observation.images.head_stereo_left/file_index"])
    from_ts = float(ep_meta["videos/observation.images.head_stereo_left/from_timestamp"])
    to_ts = float(ep_meta["videos/observation.images.head_stereo_left/to_timestamp"])

    video_path = os.path.join(DATA_DIR, "videos",
                               "observation.images.head_stereo_left",
                               "chunk-000", f"file-{file_idx:03d}.mp4")

    data_fi = int(ep_meta.get("data/file_index", 0))
    parquet_path = os.path.join(DATA_DIR, "data", "chunk-000",
                                 f"file-{data_fi:03d}.parquet")
    df = pd.read_parquet(parquet_path)
    ep_df = df[df["episode_index"] == ep].sort_values("frame_index")

    return video_path, from_ts, to_ts, ep_df


def extract_frame(video_path, from_ts, frame_idx, fps=30):
    """Extract a single frame using PyAV."""
    import av
    target_ts = from_ts + frame_idx / fps
    container = av.open(video_path)
    stream = container.streams.video[0]
    tb = float(stream.time_base)

    seek_pts = int(max(0, target_ts - 0.5) / tb)
    container.seek(seek_pts, stream=stream)

    for frame in container.decode(video=0):
        pts_sec = frame.pts * tb
        fi = int(round((pts_sec - from_ts) * fps))
        if fi == frame_idx:
            img = frame.to_ndarray(format='bgr24')
            container.close()
            return img

    container.close()
    return None


def main():
    parser = argparse.ArgumentParser(description="Human arm overlay test")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--frame", type=int, default=30,
                        help="Frame index within episode (default 30)")
    parser.add_argument("--side-by-side", action="store_true",
                        help="Show G1 overlay + human overlay side by side")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load episode
    print(f"Loading episode {args.episode}...")
    video_path, from_ts, to_ts, ep_df = load_episode_info(args.episode)
    print(f"  {len(ep_df)} frames, video: {os.path.basename(video_path)}")

    # Get frame data
    frame_row = ep_df[ep_df["frame_index"] == args.frame]
    if len(frame_row) == 0:
        print(f"Frame {args.frame} not found, using first frame")
        frame_row = ep_df.iloc[[0]]
        args.frame = int(frame_row.iloc[0]["frame_index"])

    rq = np.array(frame_row.iloc[0]["observation.state.robot_q_current"], dtype=np.float64)
    hs = np.array(frame_row.iloc[0]["observation.state.hand_state"], dtype=np.float64)

    # Extract video frame
    print(f"Extracting frame {args.frame}...")
    img = extract_frame(video_path, from_ts, args.frame)
    if img is None:
        print("Failed to extract frame")
        return

    h, w = img.shape[:2]
    print(f"  Frame size: {w}x{h}")

    # Load human URDF + meshes
    print("Loading human URDF...")
    model_h = pin.buildModelFromUrdf(HUMAN_URDF, pin.JointModelFreeFlyer())
    data_h = model_h.createData()
    link_meshes_h = parse_urdf_meshes(HUMAN_URDF)
    mesh_cache_h = preload_meshes(link_meshes_h, MESH_DIR, HUMAN_SKIP)
    print(f"  Human: {len(mesh_cache_h)} link meshes")

    # FK for human
    q = build_q(model_h, rq, hs)
    transforms_h = do_fk(model_h, data_h, q)

    # Render human overlay
    print("Rendering human overlay...")
    human_result = render_human_triangles(img, mesh_cache_h, transforms_h, BEST_PARAMS)

    if args.side_by_side:
        # Also load G1 for comparison
        print("Loading G1 URDF for comparison...")
        model_g = pin.buildModelFromUrdf(G1_URDF, pin.JointModelFreeFlyer())
        data_g = model_g.createData()
        link_meshes_g = parse_urdf_meshes(G1_URDF)
        mesh_cache_g = preload_meshes(link_meshes_g, MESH_DIR, G1_SKIP)
        print(f"  G1: {len(mesh_cache_g)} link meshes")

        q_g = build_q(model_g, rq, hs)
        transforms_g = do_fk(model_g, data_g, q_g)
        g1_result = render_g1_overlay(img, mesh_cache_g, transforms_g, BEST_PARAMS)

        # Side by side: original | G1 overlay | human overlay
        combined = np.hstack([img, g1_result, human_result])
        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(combined, "Original", (10, 30), font, 0.8, (255,255,255), 2)
        cv2.putText(combined, "G1 Robot", (w + 10, 30), font, 0.8, (255,255,255), 2)
        cv2.putText(combined, "Human Overlay", (2*w + 10, 30), font, 0.8, (255,255,255), 2)

        out_path = os.path.join(OUTPUT_DIR,
                                 f"ep{args.episode}_f{args.frame}_comparison.png")
        cv2.imwrite(out_path, combined)
        print(f"Saved: {out_path}")
    else:
        out_path = os.path.join(OUTPUT_DIR,
                                 f"ep{args.episode}_f{args.frame}_human.png")
        cv2.imwrite(out_path, human_result)
        print(f"Saved: {out_path}")

        # Also save just the original for reference
        out_orig = os.path.join(OUTPUT_DIR,
                                 f"ep{args.episode}_f{args.frame}_original.png")
        cv2.imwrite(out_orig, img)


if __name__ == "__main__":
    main()
