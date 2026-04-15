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
import csv
import argparse
import time
import numpy as np
import cv2
import torch
import pinocchio as pin
import av

sys.stdout.reconfigure(line_buffering=True)

from src.core.config import (G1_URDF, MESH_DIR, DATASET_ROOT, OUTPUT_DIR,
                    BEST_PARAMS, CAMERA_MODEL, get_hand_type, get_skip_meshes)
from src.core.fk import (build_q, do_fk, parse_urdf_meshes, preload_meshes)
from src.core.camera import make_camera_const, make_camera, project_points_cv
from src.core.render import render_mask_and_overlay, render_mesh_on_image
from src.core.smplh import SMPLHForIK, extract_g1_targets, R_SMPLH_TO_G1_NP
from src.core.retarget import (retarget_frame, refine_arms, compute_g1_rest_transforms,
                            scale_hands, build_default_hand_pose)
from src.core.data import (load_episode_info, open_video_writer,
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
                        help="Output mp4 path (default: output/human/retarget_video/<tag>.mp4)")
    parser.add_argument("--no-debug-keypoints", dest="debug_keypoints",
                        action="store_false", default=True,
                        help="Disable per-frame keypoint debug overlay + CSV "
                             "(default: enabled).")
    parser.add_argument("--toe-px-threshold", type=float, default=20.0,
                        help="PASS/FAIL pixel threshold for toe alignment.")
    parser.add_argument("--toe-mm-threshold", type=float, default=50.0,
                        help="PASS/FAIL 3D mm threshold for toe alignment.")
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
    out_dir = os.path.join(OUTPUT_DIR, "human/retarget_video")
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

    # ── Debug CSV ──
    csv_path = out_path + ".metrics.csv"
    csv_file = None
    csv_writer = None
    debug_records = []
    KP_NAMES = ["L_toe", "R_toe", "L_thumb", "R_thumb"]
    if args.debug_keypoints:
        csv_file = open(csv_path, "w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(
            ["frame"]
            + [f"px_{k}" for k in KP_NAMES]
            + [f"mm_{k}" for k in KP_NAMES]
        )
        print(f"  debug CSV: {csv_path}")

    # ── Main loop ──
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

        # ── Step 3: IK refinement (fresh init every frame) ──
        # NOTE: refine_arms only updates arm joints (collar/shoulder/elbow);
        # all other joints (legs/spine/wrist) are passed through unchanged
        # from body_pose_init. Using the previous frame's pose as init would
        # freeze the legs at frame 0 forever — the bug we are fixing here.
        # Always seed from this frame's fresh retarget output.
        if args.refine:
            body_pose_np = refine_arms(
                smplh, J_shaped, targets,
                root_trans_np, root_orient_np, body_pose_np,
                device=args.device, w_drift=10.0,
                hand_L=hand_L_np, hand_R=hand_R_np)
            body_p = torch.tensor(body_pose_np, dtype=torch.float64, device=args.device)

        # ── LBS → SMPLH vertices in G1 frame ──
        with torch.no_grad():
            v_g1 = smplh.lbs_to_g1(root_t, root_o, body_p, J_shaped, v_shaped,
                                    left_hand_pose=hand_L_t,
                                    right_hand_pose=hand_R_t)

        # ── Keypoint instrumentation (final pose, post-refine) ──
        kp_metrics = None
        if args.debug_keypoints:
            with torch.no_grad():
                pos_final, rot_final = smplh.forward_kinematics(
                    root_t, root_o, body_p, J_shaped,
                    left_hand_pose=hand_L_t, right_hand_pose=hand_R_t)
                ee_final = smplh.end_effector_positions(pos_final, rot_final)
            g1_kps_3d = np.array([targets[k + "_pos"] for k in KP_NAMES])
            sm_kps_3d = np.array([ee_final[k + "_pos"].cpu().numpy()
                                  for k in KP_NAMES])
            mm_dist = np.linalg.norm(g1_kps_3d - sm_kps_3d, axis=1) * 1000.0
            K_, D_, rvec_, tvec_, _, _, fisheye_ = make_camera(
                params, transforms, cam_const)
            all_pts = np.concatenate([g1_kps_3d, sm_kps_3d], axis=0)
            pts2d = project_points_cv(
                all_pts.reshape(-1, 1, 3),
                rvec_, tvec_, K_, D_, fisheye_).reshape(-1, 2)
            g1_2d = pts2d[:4]
            sm_2d = pts2d[4:]
            px_dist = np.linalg.norm(g1_2d - sm_2d, axis=1)
            kp_metrics = {
                "g1_2d": g1_2d, "sm_2d": sm_2d,
                "px": px_dist, "mm": mm_dist,
                "g1_3d": g1_kps_3d, "sm_3d": sm_kps_3d,
            }

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

        # ── Draw keypoint markers on G1 + SMPLH panels ──
        if kp_metrics is not None:
            g1_2d = kp_metrics["g1_2d"]
            sm_2d = kp_metrics["sm_2d"]
            for panel_idx in (1, 2):  # G1, SMPLH
                ox = panel_idx * w
                for i, name in enumerate(KP_NAMES):
                    gp = (int(round(g1_2d[i, 0])) + ox,
                          int(round(g1_2d[i, 1])))
                    sp = (int(round(sm_2d[i, 0])) + ox,
                          int(round(sm_2d[i, 1])))
                    if 0 <= gp[0] - ox < w and 0 <= gp[1] < h \
                            and 0 <= sp[0] - ox < w and 0 <= sp[1] < h:
                        cv2.line(combined, gp, sp, (0, 255, 255), 1)
                    cv2.circle(combined, gp, 4, (0, 0, 255), -1)   # G1: red
                    cv2.circle(combined, sp, 4, (255, 0, 0), -1)   # SMPLH: blue
                    cv2.circle(combined, gp, 4, (255, 255, 255), 1)
                    cv2.circle(combined, sp, 4, (255, 255, 255), 1)
            px = kp_metrics["px"]
            mm = kp_metrics["mm"]
            txt = (f"toe L/R: {px[0]:5.1f}/{px[1]:5.1f}px  "
                   f"{mm[0]:5.0f}/{mm[1]:5.0f}mm   "
                   f"thumb L/R: {px[2]:5.1f}/{px[3]:5.1f}px")
            colour = (0, 255, 0)
            if px[0] > args.toe_px_threshold or px[1] > args.toe_px_threshold:
                colour = (0, 0, 255)
            cv2.putText(combined, txt, (10, h - 30),
                        font, 0.45, colour, 1)
            csv_writer.writerow(
                [ep_fi]
                + [f"{v:.3f}" for v in px]
                + [f"{v:.3f}" for v in mm]
            )
            debug_records.append((ep_fi, px.copy(), mm.copy()))

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

    if csv_file is not None:
        csv_file.close()

    if debug_records:
        all_px = np.array([r[1] for r in debug_records])  # (N, 4)
        all_mm = np.array([r[2] for r in debug_records])
        names = ["L_toe", "R_toe", "L_thumb", "R_thumb"]
        print("\n── Keypoint debug summary (G1 vs SMPLH) ──")
        print(f"  CSV: {csv_path}")
        print(f"  {'kp':<8} {'px_mean':>8} {'px_max':>8}  "
              f"{'mm_mean':>8} {'mm_max':>8}")
        for i, n in enumerate(names):
            print(f"  {n:<8} {all_px[:, i].mean():8.2f} "
                  f"{all_px[:, i].max():8.2f}  "
                  f"{all_mm[:, i].mean():8.1f} {all_mm[:, i].max():8.1f}")

        toe_px_max = max(all_px[:, 0].max(), all_px[:, 1].max())
        toe_mm_max = max(all_mm[:, 0].max(), all_mm[:, 1].max())
        toe_px_pass = toe_px_max < args.toe_px_threshold
        toe_mm_pass = toe_mm_max < args.toe_mm_threshold
        verdict_px = "PASS" if toe_px_pass else "FAIL"
        verdict_mm = "PASS" if toe_mm_pass else "FAIL"
        print(f"\n  toe px max = {toe_px_max:.1f}  "
              f"(threshold {args.toe_px_threshold}) [{verdict_px}]")
        print(f"  toe mm max = {toe_mm_max:.1f}  "
              f"(threshold {args.toe_mm_threshold}) [{verdict_mm}]")
        if toe_px_pass and toe_mm_pass:
            print("  VERDICT: aligned ✓")
        else:
            if not toe_mm_pass:
                print("  VERDICT: 3D mismatch — bug in retarget/pelvis/foot-Z "
                      "(pre-projection layer)")
            elif not toe_px_pass:
                print("  VERDICT: projection mismatch — 3D ok but pixels "
                      "diverge (camera/extrinsics layer)")


if __name__ == "__main__":
    main()
