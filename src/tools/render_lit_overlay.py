"""
Render per-triangle mesh overlay with Lambertian shading on ego-centric video.

Applies PSO-optimized camera params + fixed joint angle offsets to verify
whether joint errors are systematic across the entire episode.

Usage:
  python scripts/render_lit_overlay.py
  python scripts/render_lit_overlay.py --duration 10 --episode 0
  python scripts/render_lit_overlay.py --no-joint-offsets  # camera only, no joint correction
"""

import sys
import os
import math
import time
import argparse
import numpy as np
import cv2
import pinocchio as pin
import pandas as pd

sys.stdout.reconfigure(line_buffering=True)

from src.core.config import (G1_URDF, MESH_DIR, DATASET_ROOT, OUTPUT_DIR,
                     get_hand_type, get_skip_meshes, CAMERA_MODEL, BEST_PARAMS)
from src.core.camera import get_model, project_points_cv
from src.core.fk import (build_q, do_fk, parse_urdf_meshes, preload_meshes)
from src.core.camera import make_camera, make_camera_const
from src.core.data import load_episode_info, open_video_writer, write_frame, close_video

# ── PSO-optimized params (pinhole model, from auto_calibrate_mask.py) ──
PSO_PARAMS = {
    "dx": 0.0721, "dy": 0.0262, "dz": 0.4076,
    "pitch": -59.2610, "yaw": 2.4196, "roll": -0.3955,
    "fx": 285.4568, "fy": 282.2731, "cx": 325.7725, "cy": 316.8363,
}
# Best F1=0.9480 (baseline 0.8301), PSO 100 iters / 200 particles / pinhole + joint_dof=10

# ── Joint offset definitions (from auto_calibrate_mask.py / optimize_keypoints.py) ──
JOINT_Q_INDICES = list(range(7, 29)) + list(range(41, 48))  # 29 body joints
JOINT_NAMES = [
    "L_hip_pitch", "L_hip_roll", "L_hip_yaw",
    "L_knee", "L_ankle_pitch", "L_ankle_roll",
    "R_hip_pitch", "R_hip_roll", "R_hip_yaw",
    "R_knee", "R_ankle_pitch", "R_ankle_roll",
    "waist_yaw", "waist_roll", "waist_pitch",
    "L_shoulder_pitch", "L_shoulder_roll", "L_shoulder_yaw",
    "L_elbow", "L_elbow_roll", "L_wrist_yaw", "L_wrist_pitch",
    "R_shoulder_pitch", "R_shoulder_roll", "R_shoulder_yaw",
    "R_elbow", "R_elbow_roll", "R_wrist_yaw", "R_wrist_pitch",
]

# PSO-optimized joint offsets in degrees (from auto_calibrate_mask.py output).
# These are systematic corrections applied to every frame.
# Will be updated with final PSO output values.
PSO_JOINT_OFFSETS_DEG = {
    "L_hip_pitch": -2.15, "L_hip_roll": 1.18, "L_hip_yaw": -0.24,
    "L_knee": 0.09, "L_ankle_pitch": 0.00, "L_ankle_roll": 0.07,
    "R_hip_pitch": 0.45, "R_hip_roll": -1.25, "R_hip_yaw": -0.26,
    "R_knee": 0.81, "R_ankle_pitch": 0.07, "R_ankle_roll": -0.13,
    "waist_yaw": -0.07, "waist_roll": 0.02, "waist_pitch": -0.60,
    "L_shoulder_pitch": -0.28, "L_shoulder_roll": -0.58, "L_shoulder_yaw": 0.55,
    "L_elbow": -0.13, "L_elbow_roll": -0.03, "L_wrist_yaw": 2.59, "L_wrist_pitch": -3.42,
    "R_shoulder_pitch": 0.62, "R_shoulder_roll": -0.33, "R_shoulder_yaw": -0.11,
    "R_elbow": 1.88, "R_elbow_roll": -0.76, "R_wrist_yaw": 0.10, "R_wrist_pitch": 2.83,
}


def build_joint_offsets_rad():
    """Convert PSO_JOINT_OFFSETS_DEG to a numpy array in radians, ordered by JOINT_NAMES."""
    return np.array([math.radians(PSO_JOINT_OFFSETS_DEG[jn]) for jn in JOINT_NAMES])

# Per-side colors (BGR)
COLOR_LEFT  = (255, 160, 50)   # blue-ish
COLOR_RIGHT = (50, 160, 255)   # orange-ish
COLOR_BODY  = (50, 220, 180)   # cyan-green


def render_lit_mesh(img, mesh_cache, transforms, params, cam_const=None):
    """Per-triangle mesh rendering with Lambertian shading and per-link colors."""
    K, D, rvec, tvec, R_w2c, t_w2c, fisheye = make_camera(
        params, transforms, cam_const)
    h, w = img.shape[:2]
    t_w2c_flat = t_w2c.flatten()

    # Collect all world-space triangles with their link colors
    all_tris_world = []   # (N_total, 3, 3)
    all_colors = []       # (N_total, 3) BGR

    for link_name, (tris, _) in mesh_cache.items():
        if link_name not in transforms:
            continue
        t_link, R_link = transforms[link_name]
        flat = tris.reshape(-1, 3)
        world = (R_link @ flat.T).T + t_link
        world_tris = world.reshape(-1, 3, 3)

        if "left" in link_name:
            color = COLOR_LEFT
        elif "right" in link_name:
            color = COLOR_RIGHT
        else:
            color = COLOR_BODY

        all_tris_world.append(world_tris)
        all_colors.extend([color] * len(world_tris))

    if not all_tris_world:
        return img.copy()

    tri_world = np.concatenate(all_tris_world, axis=0)  # (N, 3, 3)
    colors = all_colors  # list of N tuples
    n_tri = len(tri_world)

    # Project all vertices
    flat = tri_world.reshape(-1, 3).astype(np.float64)
    cam_pts = (R_w2c @ flat.T).T + t_w2c_flat
    z_cam = cam_pts[:, 2]

    pts2d = project_points_cv(
        flat.reshape(-1, 1, 3), rvec, tvec, K, D, fisheye)
    pts2d = pts2d.reshape(-1, 2)

    z_tri = z_cam.reshape(n_tri, 3)
    pts_tri = pts2d.reshape(n_tri, 3, 2)

    # Filter valid triangles
    valid = (z_tri > 0.01).all(axis=1)
    finite = np.all(np.isfinite(pts_tri), axis=(1, 2))
    mask = valid & finite
    if mask.sum() == 0:
        return img.copy()

    indices = np.where(mask)[0]
    pts_sel = pts_tri[mask]
    z_sel = z_tri[mask]
    tri_sel = tri_world[mask]

    # Face normals for Lambertian shading
    v0, v1, v2 = tri_sel[:, 0], tri_sel[:, 1], tri_sel[:, 2]
    normals = np.cross(v1 - v0, v2 - v0)
    norms = np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-8)
    normals /= norms

    # View direction (from triangle centroid toward camera)
    centroids_cam = (R_w2c @ tri_sel.mean(axis=1).T).T + t_w2c_flat
    view_dirs = -centroids_cam
    view_norms = np.maximum(np.linalg.norm(view_dirs, axis=1, keepdims=True), 1e-8)
    view_dirs /= view_norms
    dots = np.abs(np.sum(normals * view_dirs, axis=1))

    # Depth-sort (painter's algorithm: far first)
    order = np.argsort(-z_sel.mean(axis=1))

    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    for rank in order:
        tri_2d = pts_sel[rank].astype(np.int32)
        shade = 0.3 + 0.7 * dots[rank]
        orig_idx = indices[rank]
        base_color = colors[orig_idx]
        shaded = tuple(int(c * shade) for c in base_color)
        cv2.fillPoly(canvas, [tri_2d], shaded)

    # Composite: mesh over original with slight transparency
    result = img.copy()
    mesh_mask = canvas.any(axis=2)
    alpha = 0.85
    result[mesh_mask] = (
        alpha * canvas[mesh_mask].astype(np.float32) +
        (1 - alpha) * result[mesh_mask].astype(np.float32)
    ).astype(np.uint8)

    return result


def main():
    parser = argparse.ArgumentParser(description="Lit mesh overlay video")
    parser.add_argument("--task", type=str,
                        default="G1_WBT_Inspire_Put_Clothes_into_Washing_Machine_MainCamOnly")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--start", type=float, default=0, help="Start seconds")
    parser.add_argument("--duration", type=float, default=5, help="Duration seconds")
    parser.add_argument("--mesh-subsample", type=int, default=2)
    parser.add_argument("--use-config-params", action="store_true",
                        help="Use BEST_PARAMS from config.py instead of PSO params")
    parser.add_argument("--no-joint-offsets", action="store_true",
                        help="Disable joint angle offsets (camera params only)")
    args = parser.parse_args()

    params = BEST_PARAMS if args.use_config_params else PSO_PARAMS
    cam_model_name = "pinhole" if not args.use_config_params else CAMERA_MODEL
    # Patch camera model if using pinhole
    import src.core.camera as _cam_mod
    _cam_mod.CAMERA_MODEL = cam_model_name

    # Joint offsets
    if args.no_joint_offsets:
        joint_offsets = None
        print("Joint offsets: DISABLED")
    else:
        joint_offsets = build_joint_offsets_rad()
        nonzero = [(JOINT_NAMES[i], math.degrees(joint_offsets[i]))
                    for i in range(len(JOINT_NAMES)) if abs(joint_offsets[i]) > 1e-6]
        print(f"Joint offsets: {len(nonzero)} active")
        for name, deg in sorted(nonzero, key=lambda x: -abs(x[1])):
            print(f"  {name}: {deg:+.1f}\u00b0")

    print(f"Camera model: {cam_model_name}")
    print(f"Params: {params}")

    task = args.task
    hand_type = get_hand_type(task)
    skip_set = get_skip_meshes(hand_type)

    print(f"Loading URDF + meshes (hand={hand_type}, subsample={args.mesh_subsample})...")
    model = pin.buildModelFromUrdf(G1_URDF, pin.JointModelFreeFlyer())
    data_pin = model.createData()
    link_meshes = parse_urdf_meshes(G1_URDF)
    mesh_cache = preload_meshes(link_meshes, MESH_DIR, skip_set=skip_set,
                                subsample=args.mesh_subsample)
    print(f"Loaded {len(mesh_cache)} link meshes")

    cam_const = make_camera_const(params)

    # Load episode
    data_dir = os.path.join(DATASET_ROOT, task)
    video_path, from_ts, to_ts, ep_df = load_episode_info(args.episode, data_dir)
    fps = 30

    start_frame = int(args.start * fps)
    n_frames = int(args.duration * fps)
    end_frame = min(start_frame + n_frames, len(ep_df))
    n_frames = end_frame - start_frame

    print(f"Task: {task}, ep={args.episode}")
    print(f"Frames: {start_frame}-{end_frame - 1} ({n_frames} frames, {n_frames/fps:.1f}s)")
    print(f"Video: {video_path}")

    # Build frame lookup
    fi_to_row = {}
    for _, row in ep_df.iterrows():
        fi_to_row[int(row["frame_index"])] = row

    # Open video
    import av
    container_in = av.open(video_path)
    stream_in = container_in.streams.video[0]
    vid_fps = float(stream_in.average_rate)

    seek_sec = from_ts + start_frame / fps
    if seek_sec > 1.0:
        target_pts = int((seek_sec - 1.0) / stream_in.time_base)
        container_in.seek(max(target_pts, 0), stream=stream_in)

    # Output
    out_dir = os.path.join(OUTPUT_DIR, "tmp", "lit_overlay")
    os.makedirs(out_dir, exist_ok=True)
    tag = f"{task.split('G1_WBT_')[-1]}_ep{args.episode}"
    if joint_offsets is not None:
        tag += "_joffset"
    out_path = os.path.join(out_dir, f"{tag}.mp4")

    writer = None
    frame_count = 0
    t_start = time.time()

    print(f"Rendering...")
    for av_frame in container_in.decode(stream_in):
        pts_sec = float(av_frame.pts * stream_in.time_base)
        ep_fi = int(round((pts_sec - from_ts) * vid_fps))

        if ep_fi < start_frame:
            continue
        if ep_fi >= end_frame:
            break
        if ep_fi not in fi_to_row:
            continue

        row = fi_to_row[ep_fi]
        img = av_frame.to_ndarray(format='bgr24')
        h, w = img.shape[:2]

        if writer is None:
            # Side-by-side: original | overlay
            writer = open_video_writer(out_path, w * 2, h, fps=vid_fps)

        rq = np.array(row["observation.state.robot_q_current"], dtype=np.float64)
        hs = np.array(row["observation.state.hand_state"], dtype=np.float64)

        q = build_q(model, rq, hs, hand_type=hand_type)
        if joint_offsets is not None:
            q[JOINT_Q_INDICES] += joint_offsets
        transforms = do_fk(model, data_pin, q)

        overlay = render_lit_mesh(img, mesh_cache, transforms, params, cam_const)

        # Info text
        cv2.putText(overlay, f"f={ep_fi}", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

        combined = np.hstack([img, overlay])
        write_frame(*writer, combined)

        frame_count += 1
        if frame_count % 30 == 0 or frame_count == 1:
            elapsed = time.time() - t_start
            fps_proc = frame_count / elapsed
            eta = (n_frames - frame_count) / fps_proc if fps_proc > 0 else 0
            print(f"  {frame_count}/{n_frames} ({fps_proc:.1f} fps, ETA {eta:.0f}s)")

    container_in.close()
    if writer is not None:
        close_video(*writer)

    elapsed = time.time() - t_start
    print(f"\nDone: {frame_count} frames in {elapsed:.1f}s "
          f"({frame_count/elapsed:.1f} fps)")
    print(f"Output: {out_path}")


if __name__ == "__main__":
    main()
