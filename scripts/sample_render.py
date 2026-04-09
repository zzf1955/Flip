"""
Sample frames from multiple tasks/episodes, render semi-transparent mesh overlay
using the best fisheye camera parameters, to verify generalization.
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
URDF_PATH = os.path.join(BASE_DIR, "data", "unitree_ros", "robots",
                         "g1_description", "g1_29dof_with_hand_rev_1_0.urdf")
MESH_DIR = os.path.join(BASE_DIR, "data", "unitree_ros", "robots", "g1_description", "meshes")
RAW_DIR = os.path.join(BASE_DIR, "test_results", "raw_frames")
OVERLAY_DIR = os.path.join(BASE_DIR, "test_results", "overlay_frames")

SKIP_MESHES = {
    "head_link", "logo_link", "d435_link",
}

# Best params from Run 1 (IoU=0.8970)
BEST_PARAMS = {
    "dx": 0.039, "dy": 0.052, "dz": 0.536,
    "pitch": -53.6, "yaw": 4.7, "roll": 3.0,
    "fx": 315, "fy": 302, "cx": 334, "cy": 230,
    "k1": 0.63, "k2": 0.17, "k3": 1.19, "k4": 0.25,
}

# Tasks and sampling config — only episodes covered by video files
TASKS = [
    {"name": "g1_wbt",       "episodes": [0, 2, 4, 6, 8, 9], "frames_per_ep": 3},
    {"name": "g1_wbt_task2",  "episodes": [0, 3, 6, 9, 12, 14], "frames_per_ep": 3},
    {"name": "g1_wbt_task3",  "episodes": [0, 5, 10, 15, 20, 25], "frames_per_ep": 3},
]


def extract_frame(video_path, frame_idx):
    import av
    container = av.open(video_path)
    stream = container.streams.video[0]
    # Seek to nearby keyframe, then decode forward
    fps = float(stream.average_rate)
    target_ts = int(frame_idx / fps / stream.time_base)
    container.seek(max(target_ts - int(2 / stream.time_base), 0), stream=stream)
    for f in container.decode(stream):
        # pts to frame number
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


def load_link_verts(link_meshes, mesh_dir, transforms):
    link_verts = {}
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
        R, t = transforms[link_name][1], transforms[link_name][0]
        link_verts[link_name] = ((R @ flat.T).T + t).astype(np.float64)
    return link_verts


def render_overlay(img, link_verts, transforms, params):
    """Render semi-transparent mesh convex hulls on image."""
    p = params
    h, w = img.shape[:2]
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

    overlay = np.zeros_like(img)
    result = img.copy()

    for link_name, verts3d in link_verts.items():
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
        # Color by side
        if "left" in link_name:
            color = (255, 180, 0)    # cyan-ish (BGR)
        elif "right" in link_name:
            color = (0, 180, 255)    # orange-ish (BGR)
        else:
            color = (0, 255, 180)    # green-ish
        cv2.fillConvexPoly(overlay, hull.astype(np.int32), color)
        cv2.polylines(result, [hull.astype(np.int32)], True, color, 1, cv2.LINE_AA)

    # Blend: 35% overlay + 65% original
    cv2.addWeighted(overlay, 0.35, result, 1.0, 0, result)
    return result


def main():
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(OVERLAY_DIR, exist_ok=True)

    print("Loading URDF...")
    model = pin.buildModelFromUrdf(URDF_PATH, pin.JointModelFreeFlyer())
    data = model.createData()
    link_meshes = parse_urdf_meshes(URDF_PATH)

    total_rendered = 0

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

            sample_frames = np.linspace(0, max_frame, n_samples + 2,
                                         dtype=int)[1:-1]

            for fi in sample_frames:
                row = ep_df[ep_df["frame_index"] == fi]
                if len(row) == 0:
                    nearest = ep_df.iloc[(ep_df["frame_index"] - fi).abs().argsort()[:1]]
                    row = nearest
                    fi = int(row["frame_index"].iloc[0])

                rq = np.array(row.iloc[0]["observation.state.robot_q_current"],
                              dtype=np.float64)

                video_frame_idx = int(row.iloc[0]["index"])  # global video frame
                img = extract_frame(video_path, video_frame_idx)
                if img is None:
                    print(f"  ep{ep} f{fi}: cannot extract frame")
                    continue

                q = build_q(model, rq)
                transforms = do_fk(model, data, q)
                link_verts = load_link_verts(link_meshes, MESH_DIR, transforms)

                rendered = render_overlay(img, link_verts, transforms, BEST_PARAMS)

                out_name = f"{task_name}_ep{ep:03d}_f{fi:04d}.png"

                # Save raw frame
                cv2.imwrite(os.path.join(RAW_DIR, out_name), img)
                # Save overlay frame
                cv2.imwrite(os.path.join(OVERLAY_DIR, out_name), rendered)

                print(f"  {out_name}")
                total_rendered += 1

    print(f"\nDone: {total_rendered} frames")
    print(f"  Raw frames:     {RAW_DIR}")
    print(f"  Overlay frames: {OVERLAY_DIR}")


if __name__ == "__main__":
    main()
