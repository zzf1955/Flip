"""
Prepare inputs for Cosmos Transfer 2.5 depth-guided human regeneration.

Takes inpainted background + SMPLH retarget → produces:
  1. composite.mp4    — inpainted BG + SMPLH overlay
  2. smplh_mask.mp4   — binary human body mask
  3. depth_raw.mp4    — VideoDepthAnything depth of composite
  4. depth_blurred.mp4 — depth with human region blurred
  5. guided_mask.mp4  — inverted mask (white=background to preserve)
  6. spec.json        — Cosmos inference configuration

Usage:
  python -m src.pipeline.cosmos_prepare \
      --task G1_WBT_Inspire_Pickup_Pillow_MainCamOnly \
      --episode 0 --start 5 --duration 1 --device cuda:1
"""

import sys
import os
import json
import argparse
import time
import numpy as np
import cv2
import torch
import pinocchio as pin
import av

sys.stdout.reconfigure(line_buffering=True)

from src.core.config import (
    G1_URDF, MESH_DIR, DATASET_ROOT, OUTPUT_DIR,
    BEST_PARAMS, CAMERA_MODEL, COSMOS_PREPARE_DIR, COSMOS25_ROOT,
    get_hand_type, get_skip_meshes,
)
from src.core.fk import build_q, do_fk, parse_urdf_meshes, preload_meshes
from src.core.camera import make_camera_const
from src.core.render import render_mesh_on_image, render_smplh_mask
from src.core.smplh import SMPLHForIK, extract_g1_targets
from src.core.retarget import (
    retarget_frame, refine_arms, compute_g1_rest_transforms,
    scale_hands, build_default_hand_pose,
)
from src.core.data import load_episode_info, open_video_writer, write_frame, close_video


# ── Depth extraction (Depth-Anything-V2 via transformers) ──

def extract_depth_video(frames_bgr, device="cuda:1"):
    """Extract depth maps from BGR frames using Depth-Anything-V2.

    Args:
        frames_bgr: list of (H, W, 3) uint8 BGR frames
        device: torch device string

    Returns:
        depth_maps: list of (H, W) uint8 grayscale depth maps (0=far, 255=near)
    """
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    from PIL import Image

    print("Loading Depth-Anything-V2 model...")
    model_name = "depth-anything/Depth-Anything-V2-Small-hf"
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForDepthEstimation.from_pretrained(model_name)
    model = model.to(device).eval()
    print(f"  Depth model loaded on {device}")

    depth_maps = []
    for i, bgr in enumerate(frames_bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        inputs = processor(images=pil_img, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            predicted_depth = outputs.predicted_depth

        # Resize to original resolution
        h, w = bgr.shape[:2]
        depth = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=(h, w),
            mode="bicubic",
            align_corners=False,
        ).squeeze().cpu().numpy()

        depth_maps.append(depth)

        if (i + 1) % 10 == 0 or i == 0:
            print(f"  Depth: {i + 1}/{len(frames_bgr)}")

    # Normalize globally to [0, 255]
    all_depths = np.stack(depth_maps)
    d_min, d_max = all_depths.min(), all_depths.max()
    if d_max - d_min > 0:
        all_depths = ((all_depths - d_min) / (d_max - d_min) * 255).astype(np.uint8)
    else:
        all_depths = np.zeros_like(all_depths, dtype=np.uint8)

    return [all_depths[i] for i in range(len(frames_bgr))]


# ── Depth blurring ──

def blur_depth_in_mask(depth_u8, mask_u8, blur_ratio=0.15):
    """Blur depth map in the masked region with feathered edges.

    Args:
        depth_u8: (H, W) uint8 depth map
        mask_u8: (H, W) uint8 binary mask (255=human)
        blur_ratio: blur kernel size as fraction of mask bbox height

    Returns:
        blurred depth (H, W) uint8
    """
    # Compute bbox height for adaptive kernel size
    ys = np.where(mask_u8 > 0)[0]
    if len(ys) == 0:
        return depth_u8
    bbox_h = ys.max() - ys.min() + 1
    ksize = max(31, int(bbox_h * blur_ratio) | 1)  # ensure odd

    # Heavy blur of full depth
    blurred = cv2.GaussianBlur(depth_u8, (ksize, ksize), ksize / 3.0)

    # Feathered transition: dilate mask + blur edges
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    dilated = cv2.dilate(mask_u8, kernel_dilate)
    feather = cv2.GaussianBlur(dilated.astype(np.float32), (21, 21), 7.0) / 255.0

    # Blend: sharp background + blurred human region
    depth_f = depth_u8.astype(np.float32)
    blurred_f = blurred.astype(np.float32)
    result = depth_f * (1.0 - feather) + blurred_f * feather
    return np.clip(result, 0, 255).astype(np.uint8)


# ── Frame subsampling (30fps → 16fps) ──

def subsample_indices(n_frames, src_fps=30, dst_fps=16):
    """Compute frame indices for uniform subsampling."""
    n_out = max(1, int(round(n_frames * dst_fps / src_fps)))
    return np.linspace(0, n_frames - 1, n_out).astype(int).tolist()


# ── Mask video writing helper ──

def write_mask_video(path, masks, fps):
    """Write binary mask frames as grayscale MP4."""
    h, w = masks[0].shape[:2]
    container, stream = open_video_writer(path, w, h, fps=fps)
    for m in masks:
        bgr = cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)
        write_frame(container, stream, bgr)
    close_video(container, stream)


def write_depth_video(path, depth_maps, fps):
    """Write depth maps (uint8 grayscale) as MP4."""
    write_mask_video(path, depth_maps, fps)


# ── Main ──

def main():
    parser = argparse.ArgumentParser(
        description="Prepare Cosmos Transfer 2.5 inputs (composite + depth + mask)")
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--start", type=float, default=0.0,
                        help="Start offset in seconds")
    parser.add_argument("--duration", type=float, default=1.0,
                        help="Clip duration in seconds")
    parser.add_argument("--scale", type=float, default=0.75,
                        help="SMPLH body scale")
    parser.add_argument("--hand-scale", type=float, default=1.3)
    parser.add_argument("--base-offset", type=float, default=0.0, dest="base_offset")
    parser.add_argument("--no-refine", dest="refine", action="store_false", default=True)
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--depth-blur-ratio", type=float, default=0.15, dest="depth_blur_ratio")
    parser.add_argument("--output-fps", type=int, default=16,
                        help="Output video FPS for Cosmos (default 16)")
    parser.add_argument("--inpaint-dir", type=str, default=None, dest="inpaint_dir",
                        help="Directory containing inpaint.mp4 (auto-detected if omitted)")
    parser.add_argument("--mask-dilate", type=int, default=20, dest="mask_dilate",
                        help="Dilate repaint mask by N pixels (default 20)")
    parser.add_argument("--mask-blur", type=int, default=21, dest="mask_blur",
                        help="Gaussian blur kernel for mask edges, 0=sharp (default 21)")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Text prompt for Cosmos generation")
    parser.add_argument("--out-dir", type=str, default=None, dest="out_dir")
    args = parser.parse_args()

    from src.core.smplh import R_SMPLH_TO_G1_NP

    hand_type = get_hand_type(args.task)

    # ── Tag for output directory ──
    tag = (args.task.replace("G1_WBT_", "")
                    .replace("Inspire_", "")
                    .replace("Brainco_", ""))
    tag = f"{tag}_ep{args.episode}_s{int(args.start)}_d{int(args.duration)}"

    out_dir = args.out_dir or os.path.join(COSMOS_PREPARE_DIR, tag)
    os.makedirs(out_dir, exist_ok=True)
    print(f"Output directory: {out_dir}")

    # ── Load inpainted background video ──
    if args.inpaint_dir:
        inpaint_path = os.path.join(args.inpaint_dir, "inpaint.mp4")
    else:
        # Auto-detect from existing inpaint outputs
        inpaint_base = os.path.join(OUTPUT_DIR, "inpaint", "sam2_propainter")
        candidates = [d for d in os.listdir(inpaint_base)
                      if tag.split("_ep")[0].replace("_MainCamOnly", "").lower()
                      in d.lower()] if os.path.isdir(inpaint_base) else []
        if candidates:
            inpaint_path = os.path.join(inpaint_base, candidates[0], "inpaint.mp4")
        else:
            print(f"WARNING: No inpaint output found. Using original video as background.")
            inpaint_path = None

    # ── Load G1 URDF + meshes ──
    print("Loading G1 URDF...")
    model = pin.buildModelFromUrdf(G1_URDF, pin.JointModelFreeFlyer())
    data_pin = model.createData()
    link_meshes = parse_urdf_meshes(G1_URDF)
    skip_meshes = get_skip_meshes(hand_type)
    mesh_cache = preload_meshes(link_meshes, MESH_DIR,
                                skip_set=skip_meshes, subsample=2)
    rest_transforms = compute_g1_rest_transforms()

    # ── Load SMPLH ──
    print(f"Loading SMPLH (device={args.device})...")
    smplh = SMPLHForIK(device=args.device)
    J_shaped, v_shaped = smplh.shape_blend(None, body_scale=args.scale)
    J_shaped, v_shaped = scale_hands(smplh, J_shaped, v_shaped, args.hand_scale)

    # Camera
    params = BEST_PARAMS
    cam_const = make_camera_const(params)

    # Default hand pose
    hand_L_np, hand_R_np = build_default_hand_pose()
    hand_L_t = torch.tensor(hand_L_np, dtype=torch.float64, device=args.device)
    hand_R_t = torch.tensor(hand_R_np, dtype=torch.float64, device=args.device)

    # Face mask: drop head/neck triangles
    HEAD_JOINTS = [12, 15]
    weights_np = smplh.weights.cpu().numpy()
    v_head_w = weights_np[:, HEAD_JOINTS].sum(axis=1)
    faces_all = smplh.faces
    face_head_w = v_head_w[faces_all].max(axis=1)
    faces_nohead = faces_all[face_head_w < 0.3]
    print(f"  Faces: {len(faces_all)} -> {len(faces_nohead)}")

    # ── Episode data ──
    data_dir = os.path.join(DATASET_ROOT, args.task)
    video_path, from_ts, to_ts, ep_df = load_episode_info(
        args.episode, data_dir=data_dir)
    fi_to_row = {int(row["frame_index"]): row for _, row in ep_df.iterrows()}

    # ── Open input videos ──
    container_in = av.open(video_path)
    stream_in = container_in.streams.video[0]
    fps = float(stream_in.average_rate)

    start_frame = int(round(args.start * fps))
    n_frames = int(round(args.duration * fps))
    end_frame = start_frame + n_frames

    # Seek
    seek_sec = from_ts + args.start
    if seek_sec > 1.0:
        tb = float(stream_in.time_base)
        target_pts = int(max(0, (seek_sec - 1.0) / tb))
        container_in.seek(target_pts, stream=stream_in)

    # Open inpaint video if available
    inpaint_frames = {}
    if inpaint_path and os.path.isfile(inpaint_path):
        print(f"Loading inpainted background: {inpaint_path}")
        cap = cv2.VideoCapture(inpaint_path)
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            inpaint_frames[idx] = frame
            idx += 1
        cap.release()
        print(f"  Loaded {len(inpaint_frames)} inpaint frames")
    else:
        print("No inpaint video — using original frames as background")

    # ── Phase 1: Render composite + mask ──
    print(f"\n=== Phase 1: Render composite + mask ({n_frames} frames) ===")
    composite_frames = []
    mask_frames = []
    n_written = 0
    t_start = time.time()

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

        # Use inpainted background if available
        local_idx = ep_fi - start_frame
        if local_idx in inpaint_frames:
            bg = inpaint_frames[local_idx]
        else:
            bg = img  # fallback to original

        # G1 FK
        rq = np.array(row["observation.state.robot_q_current"], dtype=np.float64)
        hs = np.array(row["observation.state.hand_state"], dtype=np.float64)
        q = build_q(model, rq, hs, hand_type=hand_type)
        transforms = do_fk(model, data_pin, q)
        targets = extract_g1_targets(transforms)

        # Retarget
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

        # LBS → vertices in G1 frame
        with torch.no_grad():
            v_g1 = smplh.lbs_to_g1(root_t, root_o, body_p, J_shaped, v_shaped,
                                    left_hand_pose=hand_L_t,
                                    right_hand_pose=hand_R_t)

        # Render composite (BG + SMPLH)
        composite = render_mesh_on_image(
            bg, v_g1, faces_nohead, transforms, params, cam_const=cam_const)
        composite_frames.append(composite)

        # Render mask
        mask = render_smplh_mask(
            (h, w), v_g1, faces_nohead, transforms, params, cam_const=cam_const)
        mask_frames.append(mask)

        n_written += 1
        if n_written == 1 or n_written % 10 == 0:
            elapsed = time.time() - t_start
            fps_proc = n_written / max(elapsed, 1e-6)
            print(f"  {n_written}/{n_frames}  ({fps_proc:.1f} fps)")

    container_in.close()
    print(f"  Rendered {n_written} frames in {time.time() - t_start:.1f}s")

    if n_written == 0:
        print("ERROR: No frames processed")
        return

    # ── Subsample to output FPS ──
    sub_idx = subsample_indices(len(composite_frames), fps, args.output_fps)
    composite_sub = [composite_frames[i] for i in sub_idx]
    mask_sub = [mask_frames[i] for i in sub_idx]
    print(f"  Subsampled {len(composite_frames)} -> {len(composite_sub)} frames "
          f"({fps:.0f}fps -> {args.output_fps}fps)")

    # ── Write composite.mp4 ──
    comp_path = os.path.join(out_dir, "composite.mp4")
    comp_writer = open_video_writer(comp_path, w, h, fps=args.output_fps)
    for frame in composite_sub:
        write_frame(*comp_writer, frame)
    close_video(*comp_writer)
    print(f"  Saved: {comp_path}")

    # ── Write smplh_mask.mp4 ──
    mask_path = os.path.join(out_dir, "smplh_mask.mp4")
    write_mask_video(mask_path, mask_sub, args.output_fps)
    print(f"  Saved: {mask_path}")

    # ── Write guided_mask.mp4 (inverted: white=background, black=repaint) ──
    # Dilate repaint region so Cosmos repaints slightly beyond the mesh edge
    repaint_masks = mask_sub
    if args.mask_dilate > 0:
        kern = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (args.mask_dilate * 2 + 1, args.mask_dilate * 2 + 1))
        repaint_masks = [cv2.dilate(m, kern) for m in repaint_masks]
        print(f"  Mask dilated by {args.mask_dilate}px")
    # Blur edges for soft transition
    if args.mask_blur > 0:
        ksize = args.mask_blur | 1  # ensure odd
        repaint_masks = [cv2.GaussianBlur(m, (ksize, ksize), ksize / 3.0)
                         for m in repaint_masks]
        print(f"  Mask edges blurred (kernel={ksize})")
    guided_masks = [cv2.bitwise_not(m) for m in repaint_masks]
    guided_path = os.path.join(out_dir, "guided_mask.mp4")
    write_mask_video(guided_path, guided_masks, args.output_fps)
    print(f"  Saved: {guided_path}")

    # ── Phase 2: Depth extraction ──
    print(f"\n=== Phase 2: Depth extraction ({len(composite_sub)} frames) ===")
    depth_maps = extract_depth_video(composite_sub, device=args.device)

    depth_raw_path = os.path.join(out_dir, "depth_raw.mp4")
    write_depth_video(depth_raw_path, depth_maps, args.output_fps)
    print(f"  Saved: {depth_raw_path}")

    # ── Phase 3: Blur depth in human region ──
    print(f"\n=== Phase 3: Depth blurring ===")
    depth_blurred = [blur_depth_in_mask(d, m, args.depth_blur_ratio)
                     for d, m in zip(depth_maps, mask_sub)]
    depth_blur_path = os.path.join(out_dir, "depth_blurred.mp4")
    write_depth_video(depth_blur_path, depth_blurred, args.output_fps)
    print(f"  Saved: {depth_blur_path}")

    # ── Phase 4: Generate spec.json ──
    print(f"\n=== Phase 4: Generate spec.json ===")
    prompt = args.prompt or (
        "First-person view of a person performing household tasks in an indoor room. "
        "The person has realistic skin, natural clothing, and proper lighting. "
        "The camera is mounted on the person's head, looking forward and slightly downward. "
        "The room has natural indoor lighting."
    )

    # Save prompt
    prompt_path = os.path.join(out_dir, "prompt.txt")
    with open(prompt_path, "w") as f:
        f.write(prompt)

    spec = {
        "name": f"{tag}_cosmos",
        "prompt": prompt,
        "video_path": os.path.abspath(comp_path),
        "guided_generation_mask": os.path.abspath(guided_path),
        "guided_generation_step_threshold": 25,
        "seed": 2025,
        "guidance": 3,
        "depth": {
            "control_path": os.path.abspath(depth_blur_path),
            "control_weight": 1.0,
        },
    }

    spec_path = os.path.join(out_dir, "spec.json")
    with open(spec_path, "w") as f:
        json.dump(spec, f, indent=2)
    print(f"  Saved: {spec_path}")

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"Cosmos preparation complete!")
    print(f"  Output:     {out_dir}")
    print(f"  Frames:     {len(composite_sub)} @ {args.output_fps}fps")
    print(f"  Resolution: {w}x{h}")
    print(f"  Files:")
    for f in ["composite.mp4", "smplh_mask.mp4", "guided_mask.mp4",
              "depth_raw.mp4", "depth_blurred.mp4", "spec.json", "prompt.txt"]:
        fpath = os.path.join(out_dir, f)
        size = os.path.getsize(fpath) if os.path.exists(fpath) else 0
        print(f"    {f:<25s} {size/1024:.0f} KB")
    print(f"\nNext: python -m src.pipeline.cosmos_regen --prepare-dir {out_dir}")


if __name__ == "__main__":
    main()
