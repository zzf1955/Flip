"""
Unified mask pipeline: mesh overlay, raw mask, GrabCut, smooth, LaMa inpaint.
Outputs 7-panel comparison per frame.
"""

import sys
import numpy as np
import pandas as pd
import cv2
import os
import xml.etree.ElementTree as ET
import pinocchio as pin
from stl import mesh as stl_mesh
from PIL import Image

sys.stdout.reconfigure(line_buffering=True)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
URDF_PATH = os.path.join(BASE_DIR, "data", "unitree_ros", "robots",
                         "g1_description", "g1_29dof_with_hand_rev_1_0.urdf")
MESH_DIR = os.path.join(BASE_DIR, "data", "unitree_ros", "robots", "g1_description", "meshes")
OUTPUT_DIR = os.path.join(BASE_DIR, "test_results", "inpaint_mask")

SKIP_MESHES = {
    "head_link", "logo_link", "d435_link",
}

# Best params from PSO (IoU=0.8970)
BEST_PARAMS = {
    "dx": 0.039, "dy": 0.052, "dz": 0.536,
    "pitch": -53.6, "yaw": 4.7, "roll": 3.0,
    "fx": 315, "fy": 302, "cx": 334, "cy": 230,
    "k1": 0.63, "k2": 0.17, "k3": 1.19, "k4": 0.25,
}

TASKS = [
    {"name": "g1_wbt",       "episodes": [0, 2, 4, 6, 8, 9], "frames_per_ep": 3},
    {"name": "g1_wbt_task2",  "episodes": [0, 3, 6, 9, 12, 14], "frames_per_ep": 3},
    {"name": "g1_wbt_task3",  "episodes": [0, 5, 10, 15, 20, 25], "frames_per_ep": 3},
]


def extract_frame(video_path, frame_idx):
    import av
    container = av.open(video_path)
    stream = container.streams.video[0]
    fps = float(stream.average_rate)
    target_ts = int(frame_idx / fps / stream.time_base)
    container.seek(max(target_ts - int(2 / stream.time_base), 0), stream=stream)
    for f in container.decode(stream):
        pts_sec = float(f.pts * stream.time_base)
        fn = int(round(pts_sec * fps))
        if fn >= frame_idx:
            img = f.to_ndarray(format='bgr24')
            container.close()
            return img
    container.close()
    return None


def build_q(model, rq):
    """Map 36-element dataset rq to with_hand URDF q (50 elements).

    Dataset rq layout (old 29dof URDF order):
      rq[0:3]   position
      rq[3:7]   quaternion (w,x,y,z)
      rq[7:29]  left leg(6) + right leg(6) + waist(3) + left arm(7)
      rq[29:36] right arm(7)

    New URDF q layout:
      q[0:7]   freeflyer (pos + quat x,y,z,w)
      q[7:29]  left leg(6) + right leg(6) + waist(3) + left arm(7)
      q[29:36] left hand(7)  <- neutral
      q[36:43] right arm(7)
      q[43:50] right hand(7) <- neutral
    """
    q = pin.neutral(model)
    q[0:3] = rq[0:3]
    q[3], q[4], q[5], q[6] = rq[4], rq[5], rq[6], rq[3]
    q[7:29] = rq[7:29]
    q[36:43] = rq[29:36]
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


def render_triangle_mask(link_meshes, mesh_dir, transforms, params, h, w):
    """Raw per-triangle mask (morphological close only, no dilation)."""
    K, D, rvec, tvec, R_w2c, t_w2c = make_camera(params, transforms)
    mask = np.zeros((h, w), dtype=np.uint8)

    for link_name, filename in link_meshes.items():
        if link_name in SKIP_MESHES or link_name not in transforms:
            continue
        path = os.path.join(mesh_dir, filename)
        if not os.path.exists(path):
            continue

        m = stl_mesh.Mesh.from_file(path)
        verts = m.vectors
        flat = verts.reshape(-1, 3)
        valid_per_vert = np.all(np.isfinite(flat), axis=1)
        valid_per_tri = valid_per_vert.reshape(-1, 3).all(axis=1)
        verts = verts[valid_per_tri]
        if len(verts) == 0:
            continue

        t_link, R_link = transforms[link_name]
        flat = verts.reshape(-1, 3)
        world = (R_link @ flat.T).T + t_link

        cam_pts = (R_w2c @ world.T).T + t_w2c.flatten()
        z_cam = cam_pts[:, 2]

        pts2d, _ = cv2.fisheye.projectPoints(
            world.reshape(-1, 1, 3), rvec, tvec, K, D)
        pts2d = pts2d.reshape(-1, 2)

        n_tri = len(verts)
        z_tri = z_cam.reshape(n_tri, 3)
        pts_tri = pts2d.reshape(n_tri, 3, 2)

        valid = (z_tri > 0.01).all(axis=1)
        pts_tri = pts_tri[valid]
        if len(pts_tri) == 0:
            continue

        finite = np.all(np.isfinite(pts_tri), axis=(1, 2))
        pts_tri = pts_tri[finite]
        tris = pts_tri.astype(np.int32)

        if len(tris) > 0:
            cv2.fillPoly(mask, tris, 255)

    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
    return mask


def render_overlay(img, link_meshes, mesh_dir, transforms, params):
    """Semi-transparent convex hull overlay, colored by side."""
    K, D, rvec, tvec, R_w2c, t_w2c = make_camera(params, transforms)
    h, w = img.shape[:2]
    overlay = np.zeros_like(img)
    result = img.copy()

    for link_name, filename in link_meshes.items():
        if link_name in SKIP_MESHES or link_name not in transforms:
            continue
        path = os.path.join(mesh_dir, filename)
        if not os.path.exists(path):
            continue

        m = stl_mesh.Mesh.from_file(path)
        flat = m.vectors.reshape(-1, 3)
        valid = np.all(np.isfinite(flat), axis=1)
        flat = np.unique(flat[valid], axis=0)
        if len(flat) == 0:
            continue

        t_link, R_link = transforms[link_name]
        verts3d = ((R_link @ flat.T).T + t_link).astype(np.float64)

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


def grabcut_refine(img, mesh_mask, gc_iter=5):
    """GrabCut expansion from mesh mask seed."""
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


def smooth_mask(mask):
    """Gaussian smooth + 5px dilate."""
    out = cv2.GaussianBlur(mask, (7, 7), 0)
    out = (out > 128).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    return cv2.dilate(out, kernel)


def mask_overlay(img, mask, color=(0, 255, 0), alpha=0.4):
    """Overlay mask on image with semi-transparent color."""
    out = img.copy()
    roi = mask > 0
    out[roi] = ((1 - alpha) * out[roi] + alpha * np.array(color)).astype(np.uint8)
    return out


def run_lama(img_bgr, mask):
    from simple_lama_inpainting import SimpleLama
    global _lama
    if "_lama" not in globals():
        _lama = SimpleLama()
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    pil_mask = Image.fromarray(mask).convert("L")
    result = _lama(pil_img, pil_mask)
    return cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)


def make_panel(img, overlay_img, raw_mask, gc_mask, sm_mask, mask_ov, inpainted):
    """7-panel: original | overlay | raw mask | expanded mask | smooth mask | mask overlay | inpaint"""
    panels = [
        img,
        overlay_img,
        cv2.cvtColor(raw_mask, cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(gc_mask, cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(sm_mask, cv2.COLOR_GRAY2BGR),
        mask_ov,
        inpainted,
    ]
    return np.hstack(panels)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading URDF...")
    model = pin.buildModelFromUrdf(URDF_PATH, pin.JointModelFreeFlyer())
    data_pin = model.createData()
    link_meshes = parse_urdf_meshes(URDF_PATH)

    total = 0

    for task_cfg in TASKS:
        task_name = task_cfg["name"]
        data_dir = os.path.join(BASE_DIR, "data", task_name)
        video_path = os.path.join(data_dir, "videos",
                                   "observation.images.head_stereo_left",
                                   "chunk-000", "file-000.mp4")
        parquet_path = os.path.join(data_dir, "data", "chunk-000", "file-000.parquet")

        if not os.path.exists(parquet_path):
            print(f"SKIP {task_name}: no parquet")
            continue

        print(f"\n--- {task_name} ---")
        df = pd.read_parquet(parquet_path)
        available_eps = sorted(df["episode_index"].unique())

        for ep in task_cfg["episodes"]:
            if ep not in available_eps:
                print(f"  ep {ep}: not available, skip")
                continue

            ep_df = df[df["episode_index"] == ep]
            max_frame = ep_df["frame_index"].max()
            n_samples = task_cfg["frames_per_ep"]
            sample_frames = np.linspace(0, max_frame, n_samples + 2, dtype=int)[1:-1]

            for fi in sample_frames:
                row = ep_df[ep_df["frame_index"] == fi]
                if len(row) == 0:
                    nearest = ep_df.iloc[(ep_df["frame_index"] - fi).abs().argsort()[:1]]
                    row = nearest
                    fi = int(row["frame_index"].iloc[0])

                rq = np.array(row.iloc[0]["observation.state.robot_q_current"],
                              dtype=np.float64)
                video_frame_idx = int(row.iloc[0]["index"])
                img = extract_frame(video_path, video_frame_idx)
                if img is None:
                    print(f"  ep{ep} f{fi}: cannot extract frame")
                    continue

                h, w = img.shape[:2]
                q = build_q(model, rq)
                transforms = do_fk(model, data_pin, q)

                # 1. Mesh overlay (convex hull)
                overlay_img = render_overlay(
                    img, link_meshes, MESH_DIR, transforms, BEST_PARAMS)

                # 2. Raw triangle mask
                raw_mask = render_triangle_mask(
                    link_meshes, MESH_DIR, transforms, BEST_PARAMS, h, w)

                # 3. GrabCut expanded mask
                gc_mask = grabcut_refine(img, raw_mask)

                # 4. Smoothed mask
                sm_mask = smooth_mask(gc_mask)

                # 5. Mask overlay on original
                mask_ov = mask_overlay(img, sm_mask)

                # 6. LaMa inpaint
                inpainted = run_lama(img, sm_mask)

                # Save
                tag = f"{task_name}_ep{ep:03d}_f{fi:04d}"
                panel = make_panel(
                    img, overlay_img, raw_mask, gc_mask, sm_mask, mask_ov, inpainted)
                cv2.imwrite(os.path.join(OUTPUT_DIR, f"{tag}_panel.png"), panel)
                cv2.imwrite(os.path.join(OUTPUT_DIR, f"{tag}_mask.png"), sm_mask)

                print(f"  {tag}")
                total += 1

    print(f"\nDone: {total} frames -> {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
