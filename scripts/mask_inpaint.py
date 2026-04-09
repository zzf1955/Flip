"""
Render mesh mask, dilate 1-2 px, inpaint to erase robot arms.
Outputs: raw frame, binary mask, inpainted result.
"""

import sys
import numpy as np
import pandas as pd
import cv2
import os
import xml.etree.ElementTree as ET
import pinocchio as pin
from stl import mesh as stl_mesh

sys.stdout.reconfigure(line_buffering=True)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
URDF_PATH = os.path.join(BASE_DIR, "data", "g1_urdf", "g1_29dof_rev_1_0.urdf")
MESH_DIR = os.path.join(BASE_DIR, "data", "unitree_ros", "robots", "g1_description", "meshes")
OUTPUT_DIR = os.path.join(BASE_DIR, "test_results", "inpaint")

SKIP_MESHES = {
    "head_link", "logo_link", "d435_link",
    "left_rubber_hand", "right_rubber_hand",
}

# Best params from PSO (IoU=0.8970)
BEST_PARAMS = {
    "dx": 0.039, "dy": 0.052, "dz": 0.536,
    "pitch": -53.6, "yaw": 4.7, "roll": 3.0,
    "fx": 315, "fy": 302, "cx": 334, "cy": 230,
    "k1": 0.63, "k2": 0.17, "k3": 1.19, "k4": 0.25,
}

DILATE_PX = 2  # mask expansion in pixels

TASKS = [
    {"name": "g1_wbt",       "episodes": [0, 4, 8], "frames_per_ep": 2},
    {"name": "g1_wbt_task2", "episodes": [0, 6, 12], "frames_per_ep": 2},
    {"name": "g1_wbt_task3", "episodes": [0, 10, 20], "frames_per_ep": 2},
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
    q = pin.neutral(model)
    q[0:3] = rq[0:3]
    q[3], q[4], q[5], q[6] = rq[4], rq[5], rq[6], rq[3]
    q[7:36] = rq[7:36]
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
    """Build fisheye camera from params and torso FK."""
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
    """Render per-triangle mask using fisheye projection."""
    K, D, rvec, tvec, R_w2c, t_w2c = make_camera(params, transforms)
    mask = np.zeros((h, w), dtype=np.uint8)

    for link_name, filename in link_meshes.items():
        if link_name in SKIP_MESHES or link_name not in transforms:
            continue
        path = os.path.join(mesh_dir, filename)
        if not os.path.exists(path):
            continue

        m = stl_mesh.Mesh.from_file(path)
        verts = m.vectors  # (n_tri, 3, 3)
        flat = verts.reshape(-1, 3)
        valid_per_vert = np.all(np.isfinite(flat), axis=1)
        valid_per_tri = valid_per_vert.reshape(-1, 3).all(axis=1)
        verts = verts[valid_per_tri]
        if len(verts) == 0:
            continue

        t_link, R_link = transforms[link_name]
        flat = verts.reshape(-1, 3)
        world = (R_link @ flat.T).T + t_link

        # Depth in camera frame
        cam_pts = (R_w2c @ world.T).T + t_w2c.flatten()
        z_cam = cam_pts[:, 2]

        # Project with fisheye
        pts2d, _ = cv2.fisheye.projectPoints(
            world.reshape(-1, 1, 3), rvec, tvec, K, D)
        pts2d = pts2d.reshape(-1, 2)

        n_tri = len(verts)
        z_tri = z_cam.reshape(n_tri, 3)
        pts_tri = pts2d.reshape(n_tri, 3, 2)

        # Keep triangles fully in front of camera
        valid = (z_tri > 0.01).all(axis=1)
        pts_tri = pts_tri[valid]
        if len(pts_tri) == 0:
            continue

        finite = np.all(np.isfinite(pts_tri), axis=(1, 2))
        pts_tri = pts_tri[finite]
        tris = pts_tri.astype(np.int32)

        if len(tris) > 0:
            cv2.fillPoly(mask, tris, 255)

    # Morphological close to seal small gaps
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

    return mask


def dilate_mask(mask, px=2):
    """Expand mask by px pixels."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
    return cv2.dilate(mask, kernel)


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

                # 1. Render triangle mask
                mask_raw = render_triangle_mask(
                    link_meshes, MESH_DIR, transforms, BEST_PARAMS, h, w)

                # 2. Dilate mask
                mask_dilated = dilate_mask(mask_raw, DILATE_PX)

                # 3. Inpaint
                inpainted = cv2.inpaint(img, mask_dilated, inpaintRadius=5,
                                        flags=cv2.INPAINT_TELEA)

                # Save outputs
                tag = f"{task_name}_ep{ep:03d}_f{fi:04d}"

                # 3-panel: original | mask | inpainted
                mask_vis = cv2.cvtColor(mask_dilated, cv2.COLOR_GRAY2BGR)
                panel = np.hstack([img, mask_vis, inpainted])
                cv2.imwrite(os.path.join(OUTPUT_DIR, f"{tag}_panel.png"), panel)

                # Individual outputs
                cv2.imwrite(os.path.join(OUTPUT_DIR, f"{tag}_mask.png"), mask_dilated)
                cv2.imwrite(os.path.join(OUTPUT_DIR, f"{tag}_inpaint.png"), inpainted)

                print(f"  {tag}")
                total += 1

    print(f"\nDone: {total} frames -> {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
