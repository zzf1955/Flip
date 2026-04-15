"""
Debug SMPLH retarget by comparing user-annotated 4 keypoints against
G1 and SMPLH projections on a single frame.

Produces:
  - output/human/debug_retarget/<tag>.png  (overlay image)
  - stdout distance report

Three groups of points drawn on the same frame:
  - Red circle        : user annotation (ground truth)
  - Blue square       : G1 real keypoint projection (via extract_g1_targets)
  - Green cross       : SMPLH predicted keypoint projection (after retarget+IK)

Usage:
  python scripts/debug_retarget_points.py
  python scripts/debug_retarget_points.py --png data/4points/f015.png
  python scripts/debug_retarget_points.py --base-offset 0
  python scripts/debug_retarget_points.py --no-refine
"""

import sys
import os
import json
import argparse
import numpy as np
import cv2
import torch
import pinocchio as pin
import av

sys.stdout.reconfigure(line_buffering=True)

from src.core.config import (G1_URDF, MESH_DIR, DATASET_ROOT, OUTPUT_DIR,
                    BEST_PARAMS, CAMERA_MODEL, get_hand_type, get_skip_meshes)
from src.core.fk import (build_q, do_fk, parse_urdf_meshes, preload_meshes)
from src.core.camera import make_camera, make_camera_const, project_points_cv
from src.core.smplh import SMPLHForIK, extract_g1_targets, R_SMPLH_TO_G1_NP
from src.core.retarget import (retarget_frame, refine_arms, compute_g1_rest_transforms,
                            scale_hands, build_default_hand_pose)
from src.core.render import render_mesh_on_image
from src.core.data import detect_keypoints_from_alpha


# Manifest-ordered keypoint names (L→R pixel order in annotation)
KP_NAMES = ["L_thumb", "L_toe", "R_toe", "R_thumb"]


def extract_frame_by_index(video_path, frame_idx):
    """Decode `frame_idx`-th frame (0-based sequential count) from a video."""
    container = av.open(video_path)
    count = 0
    for frame in container.decode(video=0):
        if count == frame_idx:
            img = frame.to_ndarray(format='bgr24')
            container.close()
            return img
        count += 1
    container.close()
    raise RuntimeError(f"Frame {frame_idx} not found (video has {count} frames)")


def lookup_manifest(manifest_path, png_name):
    with open(manifest_path) as f:
        data = json.load(f)
    for entry in data["frames"]:
        if entry["image"] == png_name:
            return entry
    raise ValueError(f"{png_name} not found in {manifest_path}")


def load_fk_for_frame(task, episode, frame_idx, model, data_pin, hand_type):
    """Load parquet row for (task, episode, frame_idx) and compute FK."""
    import pandas as pd
    data_dir = os.path.join(DATASET_ROOT, task)
    meta = pd.read_parquet(os.path.join(
        data_dir, "meta", "episodes", "chunk-000", "file-000.parquet"))
    ep_meta = meta[meta["episode_index"] == episode].iloc[0]
    data_fi = int(ep_meta.get("data/file_index", 0))
    df = pd.read_parquet(os.path.join(
        data_dir, "data", "chunk-000", f"file-{data_fi:03d}.parquet"))
    ep_df = df[df["episode_index"] == episode].sort_values("frame_index")
    row = ep_df[ep_df["frame_index"] == frame_idx].iloc[0]

    rq = np.array(row["observation.state.robot_q_current"], dtype=np.float64)
    hs = np.array(row["observation.state.hand_state"], dtype=np.float64)
    q = build_q(model, rq, hs, hand_type=hand_type)
    transforms = do_fk(model, data_pin, q)
    return transforms, ep_meta


def find_video_path(task, episode):
    """Find video path, matching cam_0 / head_stereo_left."""
    import pandas as pd
    data_dir = os.path.join(DATASET_ROOT, task)
    meta = pd.read_parquet(os.path.join(
        data_dir, "meta", "episodes", "chunk-000", "file-000.parquet"))
    ep_meta = meta[meta["episode_index"] == episode].iloc[0]
    for cam_name in ["cam_0", "head_stereo_left"]:
        col = f"videos/observation.images.{cam_name}/file_index"
        if col in ep_meta.index:
            file_idx = int(ep_meta[col])
            vpath = os.path.join(data_dir, "videos",
                                  f"observation.images.{cam_name}",
                                  "chunk-000", f"file-{file_idx:03d}.mp4")
            if os.path.exists(vpath):
                return vpath
    raise FileNotFoundError(f"No video found for {task} ep{episode}")


def project_world_points(pts_world, params, transforms, cam_const):
    """Project (N, 3) world-frame points to (N, 2) pixel coords."""
    K, D, rvec, tvec, _R_w2c, _t_w2c, fisheye = make_camera(
        params, transforms, cam_const)
    pts3d = np.asarray(pts_world, dtype=np.float64).reshape(-1, 1, 3)
    px = project_points_cv(pts3d, rvec, tvec, K, D, fisheye)
    return px.reshape(-1, 2)


def draw_markers(img, pts_ann, pts_g1, pts_smplh, names):
    """Draw 3 sets of markers on a copy of img. Returns BGR image."""
    out = img.copy()
    h, w = out.shape[:2]

    def _in(p):
        return np.all(np.isfinite(p)) and 0 <= p[0] < w and 0 <= p[1] < h

    for i, name in enumerate(names):
        ann = pts_ann[i] if pts_ann is not None else None
        g1 = pts_g1[i] if pts_g1 is not None else None
        sm = pts_smplh[i] if pts_smplh is not None else None

        # connect annotation ↔ SMPLH with a thin yellow line
        if ann is not None and sm is not None and _in(ann) and _in(sm):
            cv2.line(out, tuple(ann.astype(int)), tuple(sm.astype(int)),
                     (0, 255, 255), 1, cv2.LINE_AA)

        # red circle: annotation (GT)
        if ann is not None and _in(ann):
            x, y = int(ann[0]), int(ann[1])
            cv2.circle(out, (x, y), 8, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.circle(out, (x, y), 2, (0, 0, 255), -1, cv2.LINE_AA)

        # blue square: G1 real
        if g1 is not None and _in(g1):
            x, y = int(g1[0]), int(g1[1])
            cv2.rectangle(out, (x - 6, y - 6), (x + 6, y + 6),
                          (255, 120, 0), 2, cv2.LINE_AA)

        # green cross: SMPLH retarget prediction
        if sm is not None and _in(sm):
            x, y = int(sm[0]), int(sm[1])
            cv2.line(out, (x - 7, y - 7), (x + 7, y + 7),
                     (0, 255, 0), 2, cv2.LINE_AA)
            cv2.line(out, (x - 7, y + 7), (x + 7, y - 7),
                     (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(out, name, (x + 10, y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1,
                        cv2.LINE_AA)

    # legend
    lx, ly = 10, h - 60
    cv2.circle(out, (lx + 8, ly), 6, (0, 0, 255), 2)
    cv2.putText(out, "annotation (GT)", (lx + 22, ly + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 255), 1)
    cv2.rectangle(out, (lx + 2, ly + 16), (lx + 14, ly + 28),
                  (255, 120, 0), 2)
    cv2.putText(out, "G1 projection", (lx + 22, ly + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 120, 0), 1)
    cv2.line(out, (lx + 2, ly + 40), (lx + 14, ly + 52), (0, 255, 0), 2)
    cv2.line(out, (lx + 2, ly + 52), (lx + 14, ly + 40), (0, 255, 0), 2)
    cv2.putText(out, "SMPLH projection", (lx + 22, ly + 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0), 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--png", type=str,
                    default="data/4points/f090.png",
                    help="Annotated PNG (alpha-marked keypoints)")
    ap.add_argument("--manifest", type=str,
                    default="data/4points/manifest.json")
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--scale", type=float, default=0.75)
    ap.add_argument("--hand-scale", type=float, default=1.3)
    ap.add_argument("--base-offset", type=float, default=-0.10,
                    dest="base_offset")
    ap.add_argument("--no-refine", dest="refine", action="store_false",
                    default=True)
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()

    png_name = os.path.basename(args.png)
    entry = lookup_manifest(args.manifest, png_name)
    task = entry["task"]
    episode = int(entry["episode"])
    frame_idx = int(entry["frame"])
    print(f"[{png_name}] task={task} ep={episode} frame={frame_idx}")

    hand_type = get_hand_type(task)

    # ── Detect annotated points ──
    annotated = [np.array(pt, dtype=np.float64) for pt in detect_keypoints_from_alpha(args.png)]
    print(f"  detected annotation px (L→R by X):")
    for name, pt in zip(KP_NAMES, annotated):
        print(f"    {name:8s}  ({pt[0]:7.1f}, {pt[1]:7.1f})")

    # ── Load G1 URDF + FK ──
    print("Loading G1 URDF...")
    model = pin.buildModelFromUrdf(G1_URDF, pin.JointModelFreeFlyer())
    data_pin = model.createData()
    _link_meshes = parse_urdf_meshes(G1_URDF)
    _skip = get_skip_meshes(hand_type)
    rest_transforms = compute_g1_rest_transforms()

    transforms, _ = load_fk_for_frame(
        task, episode, frame_idx, model, data_pin, hand_type)
    targets = extract_g1_targets(transforms)

    # ── Load video frame ──
    video_path = find_video_path(task, episode)
    print(f"  video={video_path}")
    img = extract_frame_by_index(video_path, frame_idx)
    h, w = img.shape[:2]
    print(f"  frame shape = {h}x{w}")

    # ── Load SMPLH with same params as render_retarget_video.py ──
    print(f"Loading SMPLH (device={args.device})...")
    smplh = SMPLHForIK(device=args.device)
    J_shaped, v_shaped = smplh.shape_blend(None, body_scale=args.scale)
    J_shaped, v_shaped = scale_hands(smplh, J_shaped, v_shaped, args.hand_scale)
    print(f"  body_scale={args.scale}  hand_scale={args.hand_scale}"
          f"  base_offset={args.base_offset}  refine={args.refine}")

    hand_L_np, hand_R_np = build_default_hand_pose()
    hand_L_t = torch.tensor(hand_L_np, dtype=torch.float64, device=args.device)
    hand_R_t = torch.tensor(hand_R_np, dtype=torch.float64, device=args.device)

    # Camera constants
    params = BEST_PARAMS
    cam_const = make_camera_const(params)
    print(f"Camera model: {CAMERA_MODEL}")

    # ── Retarget (matches render_retarget_video.py:186-222 exactly) ──
    root_trans_np, root_orient_np, body_pose_np = retarget_frame(
        transforms, rest_transforms, smplh, J_shaped, wrist_rot_deg=(0, 0, 0))

    if args.base_offset != 0.0:
        mesh_shift_g1 = np.array([-args.base_offset, 0.0, 0.0])
        mesh_shift_smplh = R_SMPLH_TO_G1_NP.T @ mesh_shift_g1
        root_trans_np = root_trans_np + mesh_shift_smplh

    root_t = torch.tensor(root_trans_np, dtype=torch.float64, device=args.device)
    root_o = torch.tensor(root_orient_np, dtype=torch.float64, device=args.device)
    body_p = torch.tensor(body_pose_np, dtype=torch.float64, device=args.device)

    # Foot-plant Z
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

    # IK refinement
    if args.refine:
        body_pose_np = refine_arms(
            smplh, J_shaped, targets,
            root_trans_np, root_orient_np, body_pose_np,
            device=args.device, w_drift=10.0,
            hand_L=hand_L_np, hand_R=hand_R_np)
        body_p = torch.tensor(body_pose_np, dtype=torch.float64, device=args.device)

    # Final FK → SMPLH keypoints in G1 frame + LBS vertices in G1 frame
    with torch.no_grad():
        positions, rotations = smplh.forward_kinematics(
            root_t, root_o, body_p, J_shaped,
            left_hand_pose=hand_L_t, right_hand_pose=hand_R_t)
        ee = smplh.end_effector_positions(positions, rotations)
        v_g1 = smplh.lbs_to_g1(root_t, root_o, body_p, J_shaped, v_shaped,
                                left_hand_pose=hand_L_t,
                                right_hand_pose=hand_R_t)

    smplh_kp_g1 = {
        "L_thumb": ee["L_thumb_pos"].cpu().numpy(),
        "L_toe":   ee["L_toe_pos"].cpu().numpy(),
        "R_toe":   ee["R_toe_pos"].cpu().numpy(),
        "R_thumb": ee["R_thumb_pos"].cpu().numpy(),
    }
    g1_kp_g1 = {
        "L_thumb": targets["L_thumb_pos"],
        "L_toe":   targets["L_toe_pos"],
        "R_toe":   targets["R_toe_pos"],
        "R_thumb": targets["R_thumb_pos"],
    }

    # ── Project both point sets to pixels ──
    smplh_pts_3d = np.array([smplh_kp_g1[n] for n in KP_NAMES])
    g1_pts_3d = np.array([g1_kp_g1[n] for n in KP_NAMES])
    smplh_px = project_world_points(smplh_pts_3d, params, transforms, cam_const)
    g1_px = project_world_points(g1_pts_3d, params, transforms, cam_const)

    # ── Render SMPLH mesh overlay (head dropped) ──
    HEAD_JOINTS = [12, 15]
    weights_np = smplh.weights.cpu().numpy()
    v_head_w = weights_np[:, HEAD_JOINTS].sum(axis=1)
    faces_all = smplh.faces
    face_head_w = v_head_w[faces_all].max(axis=1)
    faces_nohead = faces_all[face_head_w < 0.3]
    overlay = render_mesh_on_image(
        img, v_g1, faces_nohead, transforms, params, cam_const=cam_const)

    # ── Draw markers ──
    marked = draw_markers(overlay, annotated, g1_px, smplh_px, KP_NAMES)

    # ── Distance report ──
    print("\n" + "=" * 96)
    print(f"{'name':10s} {'annot px':>18s} {'G1 px':>18s} {'SMPLH px':>18s}"
          f" {'d(ann,G1)':>11s} {'d(ann,SM)':>11s} {'d(G1,SM)':>11s}"
          f" {'|G1-SM|3d':>11s}")
    print("-" * 96)
    d_ann_g1_list, d_ann_sm_list, d_g1_sm_list, d_3d_list = [], [], [], []
    for i, name in enumerate(KP_NAMES):
        a = annotated[i]
        g = g1_px[i]
        s = smplh_px[i]
        d_ag = np.linalg.norm(a - g)
        d_as = np.linalg.norm(a - s)
        d_gs = np.linalg.norm(g - s)
        d_3d = np.linalg.norm(g1_kp_g1[name] - smplh_kp_g1[name]) * 1000.0  # mm
        d_ann_g1_list.append(d_ag)
        d_ann_sm_list.append(d_as)
        d_g1_sm_list.append(d_gs)
        d_3d_list.append(d_3d)
        print(f"{name:10s}  ({a[0]:6.1f},{a[1]:6.1f})  "
              f"({g[0]:6.1f},{g[1]:6.1f})  "
              f"({s[0]:6.1f},{s[1]:6.1f})"
              f"  {d_ag:8.1f}px {d_as:8.1f}px {d_gs:8.1f}px "
              f"{d_3d:8.1f}mm")
    print("-" * 96)
    print(f"  mean d(ann,G1)    = {np.mean(d_ann_g1_list):7.2f} px"
          "   ← baseline: camera+annotation error")
    print(f"  mean d(ann,SMPLH) = {np.mean(d_ann_sm_list):7.2f} px"
          "   ← what user sees in video")
    print(f"  mean d(G1,SMPLH)  = {np.mean(d_g1_sm_list):7.2f} px"
          "   ← retarget pixel error")
    print(f"  mean |G1-SMPLH|3d = {np.mean(d_3d_list):7.2f} mm"
          "   ← retarget 3D error")
    print("=" * 96)

    # ── Save ──
    out_dir = os.path.join(OUTPUT_DIR, "human", "debug_retarget")
    os.makedirs(out_dir, exist_ok=True)
    if args.out:
        out_path = args.out
    else:
        stem = os.path.splitext(png_name)[0]
        suffix = []
        if abs(args.base_offset + 0.10) > 1e-9:
            suffix.append(f"bo{args.base_offset:+.2f}")
        if abs(args.scale - 0.75) > 1e-9:
            suffix.append(f"s{args.scale}")
        if abs(args.hand_scale - 1.3) > 1e-9:
            suffix.append(f"h{args.hand_scale}")
        if not args.refine:
            suffix.append("norefine")
        tag = ("_" + "_".join(suffix)) if suffix else ""
        out_path = os.path.join(out_dir, f"{stem}{tag}.png")
    cv2.imwrite(out_path, marked)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
