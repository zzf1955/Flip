"""
Render SMPLH human overlay on G1 ego-centric video using retarget_copy pipeline.

Produces a 3-panel MP4: [original | G1 robot overlay | SMPLH human overlay].

Reuses:
  - retarget_copy.retarget_frame / refine_arms / scale_hands / ...
  - smplh_ik.SMPLHForIK (FK + LBS)
  - video_inpaint.render_mask_and_overlay (G1 panel)
  - render_smplh_ik.render_mesh_on_image (SMPLH panel)
  - render_overlay_check.load_episode_info / open_video_writer / write_frame
  - config.BEST_PARAMS (pinhole_fixed camera params)

Usage:
  python scripts/render_retarget_video.py --task G1_WBT_Inspire_Pickup_Pillow_MainCamOnly --episode 0
  python scripts/render_retarget_video.py --task ... --start 2 --duration 5
  python scripts/render_retarget_video.py --task ... --no-refine  # faster, skip arm IK
"""

import sys
import os
import argparse
import time
import numpy as np
import cv2
import torch
import pinocchio as pin
import av

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (G1_URDF, MESH_DIR, DATASET_ROOT, OUTPUT_DIR,
                    BEST_PARAMS, CAMERA_MODEL, get_hand_type, get_skip_meshes)
from video_inpaint import (build_q, do_fk, parse_urdf_meshes, preload_meshes,
                            make_camera_const, render_mask_and_overlay)
from smplh_ik import SMPLHForIK, extract_g1_targets, R_SMPLH_TO_G1_NP
from retarget_copy import (retarget_frame, refine_arms, compute_g1_rest_transforms,
                            scale_hands, build_default_hand_pose)
from render_smplh_ik import render_mesh_on_image
from render_overlay_check import (load_episode_info, open_video_writer,
                                   write_frame, close_video)


def main():
    parser = argparse.ArgumentParser(description="Retarget SMPLH → video overlay")
    parser.add_argument("--task", type=str, required=True,
                        help="Task name, e.g. G1_WBT_Inspire_Pickup_Pillow_MainCamOnly")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--start", type=float, default=0.0,
                        help="Start offset in seconds")
    parser.add_argument("--duration", type=float, default=5.0,
                        help="Clip duration in seconds")
    parser.add_argument("--scale", type=float, default=0.75,
                        help="Body scale (default 0.75)")
    parser.add_argument("--hand-scale", type=float, default=1.3,
                        help="Hand scale multiplier (default 1.3)")
    parser.add_argument("--base-offset", type=float, default=0.0,
                        dest="base_offset",
                        help="Pelvis base offset in G1 +X (m). Default 0 "
                             "(validated via data/4points annotation: "
                             "bo=-0.10 caused ~100mm toe error in camera view; "
                             "bo=0 gives ~10mm toe error)")
    parser.add_argument("--no-refine", dest="refine", action="store_false",
                        default=True,
                        help="Disable IK arm refinement (default: on)")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--out", type=str, default=None,
                        help="Output mp4 path (default: test_results/retarget_video/<tag>.mp4)")
    args = parser.parse_args()

    hand_type = get_hand_type(args.task)

    # ── Load G1 URDF + meshes ──
    print("Loading G1 URDF...")
    model = pin.buildModelFromUrdf(G1_URDF, pin.JointModelFreeFlyer())
    data_pin = model.createData()
    link_meshes = parse_urdf_meshes(G1_URDF)
    skip_meshes = get_skip_meshes(hand_type)
    mesh_cache = preload_meshes(link_meshes, MESH_DIR,
                                 skip_set=skip_meshes, subsample=2)

    rest_transforms = compute_g1_rest_transforms()

    # ── Load SMPLH + apply body/hand scaling ──
    print(f"Loading SMPLH (device={args.device})...")
    smplh = SMPLHForIK(device=args.device)
    J_shaped, v_shaped = smplh.shape_blend(None, body_scale=args.scale)
    J_shaped, v_shaped = scale_hands(smplh, J_shaped, v_shaped, args.hand_scale)
    print(f"  body_scale={args.scale}  hand_scale={args.hand_scale}")

    # ── Camera ──
    params = BEST_PARAMS
    cam_const = make_camera_const(params)
    print(f"Camera model: {CAMERA_MODEL}")

    # ── Default hand pose (thumb opposition + straight fingers) ──
    hand_L_np, hand_R_np = build_default_hand_pose()
    hand_L_t = torch.tensor(hand_L_np, dtype=torch.float64, device=args.device)
    hand_R_t = torch.tensor(hand_R_np, dtype=torch.float64, device=args.device)

    # ── Face mask: drop neck/head triangles ──
    # SMPLH joints: 12 = neck, 15 = head. A face is considered "head" if
    # its max vertex weight on {neck, head} exceeds a threshold.
    HEAD_JOINTS = [12, 15]
    weights_np = smplh.weights.cpu().numpy()       # (V, 52)
    v_head_w = weights_np[:, HEAD_JOINTS].sum(axis=1)  # (V,)
    faces_all = smplh.faces                         # (F, 3)
    face_head_w = v_head_w[faces_all].max(axis=1)   # (F,)
    face_keep = face_head_w < 0.3                   # drop any face with significant head binding
    faces_nohead = faces_all[face_keep]
    print(f"  Faces: {len(faces_all)} → {len(faces_nohead)} "
          f"(dropped {len(faces_all) - len(faces_nohead)} head/neck)")

    # ── Episode ──
    data_dir = os.path.join(DATASET_ROOT, args.task)
    video_path, from_ts, to_ts, ep_df = load_episode_info(
        args.episode, data_dir=data_dir)
    print(f"  video={video_path}")
    print(f"  from_ts={from_ts:.3f}  to_ts={to_ts:.3f}  ep_len={len(ep_df)}")

    fi_to_row = {int(row["frame_index"]): row for _, row in ep_df.iterrows()}

    # ── Open input video ──
    container_in = av.open(video_path)
    stream_in = container_in.streams.video[0]
    fps = float(stream_in.average_rate)

    start_frame = int(round(args.start * fps))
    n_frames = int(round(args.duration * fps))
    end_frame = start_frame + n_frames

    # Seek near start
    seek_sec = from_ts + args.start
    if seek_sec > 1.0:
        tb = float(stream_in.time_base)
        target_pts = int(max(0, (seek_sec - 1.0) / tb))
        container_in.seek(target_pts, stream=stream_in)

    # ── Output ──
    out_dir = os.path.join(OUTPUT_DIR, "retarget_video")
    os.makedirs(out_dir, exist_ok=True)
    tag = (args.task.replace("G1_WBT_", "")
                    .replace("Inspire_", "")
                    .replace("Brainco_", ""))
    if args.out:
        out_path = args.out
    else:
        out_path = os.path.join(
            out_dir,
            f"{tag}_ep{args.episode}_s{int(args.start)}_d{int(args.duration)}.mp4")

    # ── Main loop ──
    prev_body_pose = None  # warm start for IK
    writer = None
    n_written = 0
    t_start = time.time()

    print(f"\nProcessing frames {start_frame}..{end_frame} "
          f"(duration {args.duration}s @ {fps:.0f}fps)")

    for av_frame in container_in.decode(stream_in):
        pts_sec = float(av_frame.pts * stream_in.time_base)
        ep_fi = int(round((pts_sec - from_ts) * fps))

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
            # 3 panels horizontally stacked
            writer = open_video_writer(out_path, w * 3, h, fps=fps)
            print(f"  output: {out_path}  ({w*3}x{h} @ {fps:.0f}fps)")

        # ── G1 FK ──
        rq = np.array(row["observation.state.robot_q_current"], dtype=np.float64)
        hs = np.array(row["observation.state.hand_state"], dtype=np.float64)
        q = build_q(model, rq, hs, hand_type=hand_type)
        transforms = do_fk(model, data_pin, q)
        targets = extract_g1_targets(transforms)

        # ── Step 1: Retarget ──
        root_trans_np, root_orient_np, body_pose_np = retarget_frame(
            transforms, rest_transforms, smplh, J_shaped, wrist_rot_deg=(0, 0, 0))

        if args.base_offset != 0.0:
            mesh_shift_g1 = np.array([-args.base_offset, 0.0, 0.0])
            mesh_shift_smplh = R_SMPLH_TO_G1_NP.T @ mesh_shift_g1
            root_trans_np = root_trans_np + mesh_shift_smplh

        root_t = torch.tensor(root_trans_np, dtype=torch.float64, device=args.device)
        root_o = torch.tensor(root_orient_np, dtype=torch.float64, device=args.device)
        body_p = torch.tensor(body_pose_np, dtype=torch.float64, device=args.device)

        # ── Step 2: Foot-plant Z ──
        with torch.no_grad():
            positions, rotations = smplh.forward_kinematics(
                root_t, root_o, body_p, J_shaped,
                left_hand_pose=hand_L_t, right_hand_pose=hand_R_t)
            ee_tmp = smplh.end_effector_positions(positions, rotations)
        g1_toe_mid = 0.5 * (targets['L_toe_pos'] + targets['R_toe_pos'])
        smplh_toe_mid = 0.5 * (ee_tmp['L_toe_pos'].cpu().numpy()
                               + ee_tmp['R_toe_pos'].cpu().numpy())
        shift_g1 = np.array([0.0, 0.0, g1_toe_mid[2] - smplh_toe_mid[2]])
        shift_smplh = R_SMPLH_TO_G1_NP.T @ shift_g1
        root_trans_np = root_trans_np + shift_smplh
        root_t = torch.tensor(root_trans_np, dtype=torch.float64, device=args.device)

        # ── Step 3: IK refinement with warm start ──
        if args.refine:
            init_pose = prev_body_pose if prev_body_pose is not None else body_pose_np
            body_pose_np = refine_arms(
                smplh, J_shaped, targets,
                root_trans_np, root_orient_np, init_pose,
                device=args.device, w_drift=10.0,
                hand_L=hand_L_np, hand_R=hand_R_np)
            body_p = torch.tensor(body_pose_np, dtype=torch.float64, device=args.device)
            prev_body_pose = body_pose_np.copy()

        # ── LBS → SMPLH vertices in G1 frame ──
        with torch.no_grad():
            v_g1 = smplh.lbs_to_g1(root_t, root_o, body_p, J_shaped, v_shaped,
                                    left_hand_pose=hand_L_t,
                                    right_hand_pose=hand_R_t)

        # ── Panel B: G1 mesh overlay on original frame ──
        _, g1_overlay = render_mask_and_overlay(
            img, mesh_cache, transforms, params, h, w, cam_const)

        # ── Panel C: SMPLH mesh overlay (head/neck dropped) ──
        smplh_overlay = render_mesh_on_image(
            img, v_g1, faces_nohead, transforms, params, cam_const=cam_const)

        # ── Compose 3 panels ──
        combined = np.hstack([img, g1_overlay, smplh_overlay])
        font = cv2.FONT_HERSHEY_SIMPLEX
        for i, label in enumerate(["Original", "G1", "SMPLH (retarget)"]):
            cv2.putText(combined, label, (i * w + 10, 25),
                        font, 0.6, (0, 255, 255), 2)
        cv2.putText(combined, f"ep{args.episode} frame={ep_fi}",
                    (10, h - 10), font, 0.45, (200, 200, 200), 1)

        write_frame(*writer, combined)
        n_written += 1

        if n_written == 1 or n_written % 10 == 0:
            elapsed = time.time() - t_start
            fps_proc = n_written / max(elapsed, 1e-6)
            eta = (n_frames - n_written) / max(fps_proc, 1e-6)
            print(f"  {n_written}/{n_frames}  ({fps_proc:.2f} fps, ETA {eta:.0f}s)")

    container_in.close()
    if writer is not None:
        close_video(*writer)
        elapsed = time.time() - t_start
        print(f"\nSaved: {out_path}  ({n_written} frames, {elapsed:.1f}s)")
    else:
        print("No frames processed")


if __name__ == "__main__":
    main()
