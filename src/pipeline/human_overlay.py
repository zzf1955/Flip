"""
Overlay SMPLH human mesh on inpainted (robot-removed) background video.

Reads an inpainted video produced by sam2_inpaint.py and the corresponding
episode parquet data to render the retargeted SMPLH mesh on the clean
background.  Output is a single-panel MP4.

Usage:
  python -m src.pipeline.human_overlay \
    --task G1_WBT_Inspire_Pickup_Pillow_MainCamOnly \
    --episode 0 --start 5 --duration 3 \
    --inpaint output/inpaint/sam2_propainter/.../inpaint.mp4 \
    --device cuda:2
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

from src.core.config import (G1_URDF, MESH_DIR, DATASET_ROOT, OUTPUT_DIR,
                              BEST_PARAMS, CAMERA_MODEL, get_hand_type)
from src.core.fk import build_q, do_fk
from src.core.camera import make_camera_const
from src.core.render import render_mesh_on_image
from src.core.smplh import SMPLHForIK, extract_g1_targets, R_SMPLH_TO_G1_NP
from src.core.retarget import (retarget_frame, refine_arms,
                                compute_g1_rest_transforms,
                                scale_hands, build_default_hand_pose)
from src.core.data import (load_episode_info, open_video_writer,
                            write_frame, close_video)


def main():
    parser = argparse.ArgumentParser(
        description="Overlay SMPLH on inpainted background")
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--start", type=float, default=0.0,
                        help="Start offset in seconds (must match inpaint)")
    parser.add_argument("--duration", type=float, default=3.0)
    parser.add_argument("--inpaint", type=str, required=True,
                        help="Path to inpaint.mp4 from sam2_inpaint")
    parser.add_argument("--scale", type=float, default=0.75)
    parser.add_argument("--hand-scale", type=float, default=1.3)
    parser.add_argument("--no-refine", dest="refine", action="store_false",
                        default=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--out", type=str, default=None,
                        help="Output path (default: output/human/long_overlay/<tag>.mp4)")
    args = parser.parse_args()

    fps = 30
    hand_type = get_hand_type(args.task)

    # ── Load G1 URDF ──
    print("Loading G1 URDF...")
    model = pin.buildModelFromUrdf(G1_URDF, pin.JointModelFreeFlyer())
    data_pin = model.createData()

    rest_transforms = compute_g1_rest_transforms()

    # ── SMPLH ──
    print(f"Loading SMPLH (device={args.device})...")
    smplh = SMPLHForIK(device=args.device)
    J_shaped, v_shaped = smplh.shape_blend(None, body_scale=args.scale)
    J_shaped, v_shaped = scale_hands(smplh, J_shaped, v_shaped, args.hand_scale)

    params = BEST_PARAMS
    cam_const = make_camera_const(params)

    hand_L_np, hand_R_np = build_default_hand_pose()
    hand_L_t = torch.tensor(hand_L_np, dtype=torch.float64, device=args.device)
    hand_R_t = torch.tensor(hand_R_np, dtype=torch.float64, device=args.device)

    # ── Drop head/neck faces ──
    HEAD_JOINTS = [12, 15]
    weights_np = smplh.weights.cpu().numpy()
    v_head_w = weights_np[:, HEAD_JOINTS].sum(axis=1)
    faces_all = smplh.faces
    face_head_w = v_head_w[faces_all].max(axis=1)
    faces_nohead = faces_all[face_head_w < 0.3]

    # ── Episode data ──
    data_dir = os.path.join(DATASET_ROOT, args.task)
    _, from_ts, to_ts, ep_df = load_episode_info(args.episode, data_dir=data_dir)

    start_frame = int(round(args.start * fps))
    n_frames = int(round(args.duration * fps))
    end_frame = start_frame + n_frames

    fi_to_row = {int(row["frame_index"]): row for _, row in ep_df.iterrows()}

    # ── Open inpaint video ──
    container_in = av.open(args.inpaint)
    stream_in = container_in.streams.video[0]
    inpaint_fps = float(stream_in.average_rate)
    print(f"Inpaint: {args.inpaint}  fps={inpaint_fps}")

    # ── Output ──
    out_dir = os.path.join(OUTPUT_DIR, "human/long_overlay")
    os.makedirs(out_dir, exist_ok=True)
    if args.out:
        out_path = args.out
    else:
        tag = (args.task.replace("G1_WBT_", "")
                        .replace("Inspire_", "")
                        .replace("Brainco_", ""))
        out_path = os.path.join(
            out_dir,
            f"{tag}_ep{args.episode}_s{int(args.start)}_d{int(args.duration)}.mp4")

    writer = None
    n_written = 0
    t_start = time.time()

    # Inpaint video frame index: sequential 0,1,2,...
    # Maps to parquet frame_index: start_frame, start_frame+1, ...
    for inpaint_idx, av_frame in enumerate(container_in.decode(stream_in)):
        ep_fi = start_frame + inpaint_idx
        if ep_fi >= end_frame:
            break
        if ep_fi not in fi_to_row:
            continue

        row = fi_to_row[ep_fi]
        bg = av_frame.to_ndarray(format='bgr24')
        h, w = bg.shape[:2]

        if writer is None:
            writer = open_video_writer(out_path, w, h, fps=fps)
            print(f"Output: {out_path}  ({w}x{h} @ {fps}fps)")

        # ── FK ──
        rq = np.array(row["observation.state.robot_q_current"], dtype=np.float64)
        hs = np.array(row["observation.state.hand_state"], dtype=np.float64)
        q = build_q(model, rq, hs, hand_type=hand_type)
        transforms = do_fk(model, data_pin, q)
        targets = extract_g1_targets(transforms)

        # ── Retarget ──
        root_trans_np, root_orient_np, body_pose_np = retarget_frame(
            transforms, rest_transforms, smplh, J_shaped, wrist_rot_deg=(0, 0, 0))

        root_t = torch.tensor(root_trans_np, dtype=torch.float64, device=args.device)
        root_o = torch.tensor(root_orient_np, dtype=torch.float64, device=args.device)
        body_p = torch.tensor(body_pose_np, dtype=torch.float64, device=args.device)

        # ── Foot-plant Z ──
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

        # ── IK refine ──
        if args.refine:
            body_pose_np = refine_arms(
                smplh, J_shaped, targets,
                root_trans_np, root_orient_np, body_pose_np,
                device=args.device, w_drift=10.0,
                hand_L=hand_L_np, hand_R=hand_R_np)
            body_p = torch.tensor(body_pose_np, dtype=torch.float64, device=args.device)

        # ── LBS + render on inpainted background ──
        with torch.no_grad():
            v_g1 = smplh.lbs_to_g1(root_t, root_o, body_p, J_shaped, v_shaped,
                                    left_hand_pose=hand_L_t,
                                    right_hand_pose=hand_R_t)

        frame_out = render_mesh_on_image(
            bg, v_g1, faces_nohead, transforms, params, cam_const=cam_const)

        write_frame(*writer, frame_out)
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
        print(f"Saved: {out_path}  ({n_written} frames, {elapsed:.1f}s)")
    else:
        print("No frames processed")


if __name__ == "__main__":
    main()
