"""
Render SMPLH human body overlay via IK on G1 ego-centric video.

Shows side-by-side comparison:
  Original | G1 Robot Overlay | SMPLH IK Human Overlay

Usage:
  python scripts/render_smplh_ik.py --episode 0 --frame 30
  python scripts/render_smplh_ik.py --episode 4 --frame 153 --beta 2.0 -1.0
  python scripts/render_smplh_ik.py --episode 0 --start 5 --duration 2
"""

import sys
import os
import argparse
import time
import numpy as np
import cv2
import pinocchio as pin

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (G1_URDF, MESH_DIR, BEST_PARAMS, SKIP_MESHES, OUTPUT_DIR,
                     get_hand_type, get_skip_meshes, CAMERA_MODEL)
from camera_models import get_model, build_K, build_D, model_is_fisheye, project_points_cv
from video_inpaint import (build_q, do_fk, parse_urdf_meshes, preload_meshes,
                            make_camera, load_episode_info, render_overlay,
                            make_camera_const)
from smplh_ik import SMPLHForIK, IKSolver, extract_g1_targets

SKIN_COLOR = (135, 165, 215)  # BGR: warm skin tone


def render_mesh_on_image(img, v_world, faces, g1_transforms, params,
                         color=SKIN_COLOR, cam_const=None):
    """Render triangle mesh onto image with Lambertian shading."""
    K, D, rvec, tvec, R_w2c, t_w2c, fisheye = make_camera(
        params, g1_transforms, cam_const)
    h, w = img.shape[:2]
    canvas = np.zeros((h, w, 3), dtype=np.uint8)

    tri_verts = v_world[faces]  # (F, 3, 3)
    flat = tri_verts.reshape(-1, 3).astype(np.float64)

    cam_pts = (R_w2c @ flat.T).T + t_w2c.flatten()
    z_cam = cam_pts[:, 2]

    pts2d = project_points_cv(
        flat.reshape(-1, 1, 3), rvec, tvec, K, D, fisheye)
    pts2d = pts2d.reshape(-1, 2)

    n_tri = len(faces)
    z_tri = z_cam.reshape(n_tri, 3)
    pts_tri = pts2d.reshape(n_tri, 3, 2)

    valid = (z_tri > 0.01).all(axis=1)
    finite = np.all(np.isfinite(pts_tri), axis=(1, 2))
    mask = valid & finite
    if mask.sum() == 0:
        return img.copy()

    pts_tri = pts_tri[mask]
    z_tri = z_tri[mask]
    tri_world = tri_verts[mask]

    # Lambertian shading
    v0, v1, v2 = tri_world[:, 0], tri_world[:, 1], tri_world[:, 2]
    normals = np.cross(v1 - v0, v2 - v0)
    norms = np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-8)
    normals /= norms
    cam_tri_pts = (R_w2c @ tri_world.mean(axis=1).T).T + t_w2c.flatten()
    view_dirs = -cam_tri_pts
    view_norms = np.maximum(np.linalg.norm(view_dirs, axis=1, keepdims=True), 1e-8)
    view_dirs /= view_norms
    dots = np.abs(np.sum(normals * view_dirs, axis=1))

    # Depth-sort (painter's algorithm)
    order = np.argsort(-z_tri.mean(axis=1))
    for idx in order:
        tri = pts_tri[idx].astype(np.int32)
        shade = 0.3 + 0.7 * dots[idx]
        shaded = tuple(int(c * shade) for c in color)
        cv2.fillPoly(canvas, [tri], shaded)

    # Composite
    result = img.copy()
    mesh_mask = canvas.any(axis=2)
    result[mesh_mask] = canvas[mesh_mask]
    return result


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
    parser = argparse.ArgumentParser(description="SMPLH IK human overlay")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--frame", type=int, default=None,
                        help="Single frame index")
    parser.add_argument("--start", type=float, default=0,
                        help="Start offset in seconds (for sequence mode)")
    parser.add_argument("--duration", type=float, default=0,
                        help="Duration in seconds (0 = single frame)")
    parser.add_argument("--beta", type=float, nargs='*', default=None,
                        help="SMPLH shape params (e.g. --beta 2.0 -1.0)")
    parser.add_argument("--scale", type=float, default=None,
                        help="Body scale (default: auto-match G1, 1.0 = original SMPLH)")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    out_dir = os.path.join(OUTPUT_DIR, "smplh_ik")
    os.makedirs(out_dir, exist_ok=True)

    # ── 1. Load episode ──
    print(f"Loading episode {args.episode}...")
    video_path, from_ts, to_ts, ep_df = load_episode_info(args.episode)
    hand_type = get_hand_type()
    fps = 30

    # Determine frame range
    if args.frame is not None:
        frame_indices = [args.frame]
    elif args.duration > 0:
        start_frame = int(args.start * fps)
        end_frame = min(start_frame + int(args.duration * fps), len(ep_df))
        frame_indices = list(range(start_frame, end_frame))
    else:
        frame_indices = [int(args.start * fps) if args.start > 0 else 0]

    print(f"  Frames: {frame_indices[0]}-{frame_indices[-1]} "
          f"({len(frame_indices)} frames)")

    # ── 2. Load G1 model ──
    print("Loading G1 model...")
    model_g = pin.buildModelFromUrdf(G1_URDF, pin.JointModelFreeFlyer())
    data_g = model_g.createData()
    link_meshes_g = parse_urdf_meshes(G1_URDF)
    skip_set = get_skip_meshes(hand_type)
    mesh_cache_g = preload_meshes(link_meshes_g, MESH_DIR, skip_set=skip_set)

    # ── 3. Load SMPLH model ──
    print(f"Loading SMPLH model (device={args.device})...")
    smplh = SMPLHForIK(device=args.device)
    solver = IKSolver(smplh)

    betas = None
    if args.beta:
        betas = np.zeros(16)
        for i, b in enumerate(args.beta[:16]):
            betas[i] = b
        print(f"  Shape betas: {betas[:len(args.beta)]}")

    # Precompute shape (with body scale)
    J_shaped, v_shaped = smplh.shape_blend(betas, body_scale=args.scale)

    # Camera constants
    cam_const = make_camera_const(BEST_PARAMS)

    # ── 4. Process frames ──
    sequence_mode = len(frame_indices) > 1
    prev_pose = None
    init_pose = None
    results_for_video = []

    t_start = time.time()

    for fi_idx, fi in enumerate(frame_indices):
        # Get frame data
        frame_row = ep_df[ep_df["frame_index"] == fi]
        if len(frame_row) == 0:
            frame_row = ep_df.iloc[[min(fi, len(ep_df) - 1)]]
            fi = int(frame_row.iloc[0]["frame_index"])
        row = frame_row.iloc[0]
        rq = np.array(row["observation.state.robot_q_current"], dtype=np.float64)
        hs = np.array(row["observation.state.hand_state"], dtype=np.float64)

        # G1 FK
        q = build_q(model_g, rq, hs, hand_type=hand_type)
        transforms = do_fk(model_g, data_g, q)

        # Extract targets
        targets = extract_g1_targets(transforms)

        # Solve IK
        result = solver.solve_frame(targets, betas=betas,
                                     body_scale=args.scale,
                                     init_pose=init_pose,
                                     prev_pose=prev_pose)

        # Update warm start
        prev_pose = result.body_pose.copy()
        init_pose = {
            'root_trans': result.root_trans,
            'root_orient': result.root_orient,
            'body_pose': result.body_pose,
        }

        # Log
        max_contact = max(result.pos_errors[k] for k in
                          ["L_toe_pos", "R_toe_pos", "L_thumb_pos", "R_thumb_pos"])
        if fi_idx == 0 or (fi_idx + 1) % 10 == 0 or fi_idx == len(frame_indices) - 1:
            print(f"  Frame {fi}: loss={result.loss:.4f}, "
                  f"max_contact={max_contact*1000:.1f}mm, "
                  f"pelvis_off={result.pos_errors['pelvis_pos']*1000:.0f}mm, "
                  f"iters={result.n_iters}")

        results_for_video.append((fi, transforms, result))

    solve_time = time.time() - t_start
    print(f"IK solved {len(frame_indices)} frames in {solve_time:.1f}s "
          f"({len(frame_indices)/solve_time:.1f} fps)")

    # ── 5. Render ──
    print("Rendering...")
    import torch

    for fi, transforms, result in results_for_video:
        # Extract video frame
        img = extract_frame(video_path, from_ts, fi)
        if img is None:
            print(f"  WARNING: failed to extract frame {fi}")
            continue
        h, w = img.shape[:2]

        # G1 robot overlay
        g1_overlay = render_overlay(img, mesh_cache_g, transforms, BEST_PARAMS)

        # SMPLH IK mesh
        root_trans_t = torch.tensor(result.root_trans, dtype=torch.float64,
                                     device=args.device)
        root_orient_t = torch.tensor(result.root_orient, dtype=torch.float64,
                                      device=args.device)
        body_pose_t = torch.tensor(result.body_pose, dtype=torch.float64,
                                    device=args.device)
        v_g1 = smplh.lbs_to_g1(root_trans_t, root_orient_t, body_pose_t,
                                J_shaped, v_shaped)

        human_overlay = render_mesh_on_image(
            img, v_g1, smplh.faces, transforms, BEST_PARAMS,
            cam_const=cam_const)

        # Compose side-by-side
        panels = [img, g1_overlay, human_overlay]
        labels = ["Original", "G1 Robot", "SMPLH IK"]
        combined = np.hstack(panels)

        font = cv2.FONT_HERSHEY_SIMPLEX
        for i, label in enumerate(labels):
            cv2.putText(combined, label, (i * w + 10, 30),
                        font, 0.6, (255, 255, 255), 2)

        # Add IK info
        max_contact = max(result.pos_errors[k] for k in
                          ["L_toe_pos", "R_toe_pos", "L_thumb_pos", "R_thumb_pos"])
        info = (f"contact<{max_contact*1000:.1f}mm  "
                f"pelvis={result.pos_errors['pelvis_pos']*1000:.0f}mm")
        cv2.putText(combined, info, (2 * w + 10, h - 10),
                    font, 0.45, (200, 200, 200), 1)

        if sequence_mode:
            out_path = os.path.join(out_dir,
                                     f"ep{args.episode:03d}_f{fi:04d}.png")
        else:
            beta_tag = ""
            if args.beta:
                beta_tag = "_beta" + "_".join(f"{b:.1f}" for b in args.beta)
            out_path = os.path.join(
                out_dir,
                f"ep{args.episode}_f{fi:04d}{beta_tag}.png")

        cv2.imwrite(out_path, combined)

    print(f"Saved {len(results_for_video)} images to {out_dir}/")


if __name__ == "__main__":
    main()
