"""
Full pipeline: SAM2 segmentation (box prompt) → mask postprocess → LaMa inpaint.

Outputs all intermediate results for debugging:
  1. original.mp4         — raw video frames
  2. fk_overlay.mp4       — FK mesh overlay (convex hull)
  3. fk_mask.mp4          — raw FK triangle mask
  4. sam2_mask.mp4         — SAM2 segmentation mask (colored by part)
  5. sam2_overlay.mp4      — SAM2 mask overlaid on original
  6. final_mask.mp4        — postprocessed binary mask (smooth + dilate + edge blur)
  7. inpaint.mp4           — LaMa inpaint result
  8. prompt_vis/           — prompt visualization frames (JPEG)

Usage:
  python -m src.pipeline.sam2_inpaint --episode 4 --start 5 --duration 5
"""

import sys
import os
import argparse
import json
import time
import shutil
import fnmatch
import numpy as np
import pandas as pd
import cv2
import pinocchio as pin
import torch

sys.stdout.reconfigure(line_buffering=True)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.config import G1_URDF, MESH_DIR, BEST_PARAMS, CAMERA_MODEL, SKIP_MESHES, OUTPUT_DIR, get_hand_type, get_skip_meshes
from src.core.camera import project_points_cv, make_camera, make_camera_const
from src.core.fk import build_q, do_fk, parse_urdf_meshes, preload_meshes
from src.core.render import render_mask, render_overlay, render_mask_and_overlay
from src.core.mask import postprocess_mask, init_lama, run_lama
from src.core.data import load_episode_info, open_video_writer, write_frame, close_video

INPAINT_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "inpaint/sam2_propainter")

MODEL_IDS = {
    "tiny": "facebook/sam2.1-hiera-tiny",
    "small": "facebook/sam2.1-hiera-small",
    "base": "facebook/sam2.1-hiera-base-plus",
    "large": "facebook/sam2.1-hiera-large",
}

BODY_PARTS = {
    "left_arm":  ["left_shoulder_*", "left_elbow_*", "left_wrist_*"],
    "left_hand": ["left_base_link", "left_palm_force_sensor",
                  "left_thumb_*", "left_index_*", "left_middle_*",
                  "left_ring_*", "left_little_*"],
    "right_arm": ["right_shoulder_*", "right_elbow_*", "right_wrist_*"],
    "right_hand":["right_base_link", "right_palm_force_sensor",
                  "right_thumb_*", "right_index_*", "right_middle_*",
                  "right_ring_*", "right_little_*"],
    "left_leg":  ["left_hip_*", "left_knee_*", "left_ankle_*"],
    "right_leg": ["right_hip_*", "right_knee_*", "right_ankle_*"],
    "torso":     ["pelvis", "pelvis_contour_link", "waist_*", "torso_link"],
}

PART_IDS = {name: i + 1 for i, name in enumerate(BODY_PARTS)}

PART_COLORS = {
    "left_arm":  (0, 100, 255),
    "left_hand": (0, 200, 255),
    "right_arm": (255, 100, 0),
    "right_hand":(255, 200, 0),
    "left_leg":  (200, 0, 200),
    "right_leg": (100, 0, 200),
    "torso":     (100, 200, 0),
}

def match_links(mesh_cache, patterns):
    matched = {}
    for link_name, data in mesh_cache.items():
        if link_name in SKIP_MESHES:
            continue
        for pat in patterns:
            if fnmatch.fnmatch(link_name, pat):
                matched[link_name] = data
                break
    return matched


def render_mask_for_links(filtered_cache, transforms, params, h, w):
    if not filtered_cache:
        return np.zeros((h, w), dtype=np.uint8)
    K, D, rvec, tvec, R_w2c, t_w2c, _fisheye = make_camera(params, transforms)
    mask = np.zeros((h, w), dtype=np.uint8)
    for link_name, (tris, _) in filtered_cache.items():
        if link_name not in transforms:
            continue
        t_link, R_link = transforms[link_name]
        flat = tris.reshape(-1, 3)
        world = (R_link @ flat.T).T + t_link
        cam_pts = (R_w2c @ world.T).T + t_w2c.flatten()
        z_cam = cam_pts[:, 2]
        pts2d = project_points_cv(
            world.reshape(-1, 1, 3), rvec, tvec, K, D, _fisheye)
        pts2d = pts2d.reshape(-1, 2)
        n_tri = len(tris)
        z_tri = z_cam.reshape(n_tri, 3)
        pts_tri = pts2d.reshape(n_tri, 3, 2)
        valid = (z_tri > 0.01).all(axis=1)
        pts_tri = pts_tri[valid]
        if len(pts_tri) == 0:
            continue
        finite = np.all(np.isfinite(pts_tri), axis=(1, 2))
        pts_tri = pts_tri[finite].astype(np.int32)
        if len(pts_tri) > 0:
            cv2.fillPoly(mask, pts_tri, 255)
    return mask


def mask_to_bbox(mask, margin=0):
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    h, w = mask.shape
    return np.array([
        max(0, int(xs.min()) - margin),
        max(0, int(ys.min()) - margin),
        min(w, int(xs.max()) + margin),
        min(h, int(ys.max()) + margin),
    ], dtype=np.float32)


def draw_box_prompt(img, part_name, box):
    color = PART_COLORS[part_name]
    x1, y1, x2, y2 = box.astype(int)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    cv2.putText(img, part_name, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)


def main():
    import av

    parser = argparse.ArgumentParser(description="SAM2 + LaMa inpaint pipeline")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--start", type=float, default=0)
    parser.add_argument("--duration", type=float, default=0)
    parser.add_argument("--sam2-model", default="small",
                        choices=["tiny", "small", "base", "large"])
    parser.add_argument("--prompt-interval", type=int, default=30)
    parser.add_argument("--inpaint-method", default="propainter",
                        choices=["lama", "propainter"],
                        help="Inpainting backend (default: propainter)")
    parser.add_argument("--task", type=str, default=None,
                        help="Task name (default: ACTIVE_TASK from config)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override output directory")
    parser.add_argument("--camera-params", type=str, default=None,
                        help="Path to best_params.json (overrides BEST_PARAMS from config)")
    parser.add_argument("--bbox-margin", type=int, default=0,
                        help="Pixels to expand each FK-derived part bbox on every side "
                             "before sending as SAM2 prompt (default: 0)")
    parser.add_argument("--min-visible-area", type=int, default=50,
                        help="Minimum FK-rendered mask pixels for a body part to count "
                             "as visible and get its own SAM2 prompt (default: 50)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (e.g. cuda:0, cuda:2, cpu)")
    args = parser.parse_args()

    # Resolve task data directory and hand type
    hand_type = get_hand_type(args.task)
    if args.task:
        from src.core.config import DATASET_ROOT
        task_data_dir = os.path.join(DATASET_ROOT, args.task)
    else:
        task_data_dir = None  # will use ACTIVE_DATA_DIR default

    ep = args.episode
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Device: {device}")

    # Output directory
    task_short = args.task.replace("G1_WBT_", "") if args.task else "default"
    tag = f"{task_short}_ep{ep:03d}"
    fps = 30
    start_frame = int(args.start * fps) if args.start > 0 else 0

    video_path, from_ts, to_ts, ep_df = load_episode_info(ep, data_dir=task_data_dir)
    n_total = len(ep_df)
    end_frame = min(start_frame + int(args.duration * fps), n_total) if args.duration > 0 else n_total
    ep_df = ep_df.iloc[start_frame:end_frame]
    n_frames = len(ep_df)

    if args.start > 0 or args.duration > 0:
        tag += f"_{start_frame}-{end_frame}"

    base_out = args.output_dir if args.output_dir else INPAINT_OUTPUT_DIR
    out_dir = os.path.join(base_out, tag)
    prompt_vis_dir = os.path.join(out_dir, "prompt_vis")
    os.makedirs(prompt_vis_dir, exist_ok=True)

    print(f"Episode {ep}: frames {start_frame}-{end_frame-1} ({n_frames} frames, {n_frames/fps:.1f}s)")

    # Frame data lookup
    frame_data = {}
    for _, row in ep_df.iterrows():
        fi = int(row["frame_index"])
        rq = np.array(row["observation.state.robot_q_current"], dtype=np.float64)
        hs = np.array(row["observation.state.hand_state"], dtype=np.float64)
        frame_data[fi] = (rq, hs)

    # Load URDF + meshes
    skip_set = get_skip_meshes(hand_type)
    print(f"Loading URDF and meshes... (hand_type={hand_type}, skipping {len(skip_set)} links)")
    model_pin = pin.buildModelFromUrdf(G1_URDF, pin.JointModelFreeFlyer())
    data_pin = model_pin.createData()
    link_meshes = parse_urdf_meshes(G1_URDF)
    mesh_cache = preload_meshes(link_meshes, MESH_DIR, skip_set=skip_set)

    part_caches = {name: match_links(mesh_cache, pats) for name, pats in BODY_PARTS.items()}

    cam_params = BEST_PARAMS
    if args.camera_params:
        with open(args.camera_params) as f:
            payload = json.load(f)
        loaded = payload.get("params", payload)
        model_tag = payload.get("camera_model", None)
        if model_tag and model_tag != CAMERA_MODEL:
            print(f"[WARN] camera-params model={model_tag} "
                  f"differs from config CAMERA_MODEL={CAMERA_MODEL}")
        cam_params = {k: loaded[k] for k in BEST_PARAMS if k in loaded}
        missing = set(BEST_PARAMS) - set(cam_params)
        if missing:
            print(f"[WARN] camera-params missing keys {missing}, "
                  f"falling back to config for those")
            for k in missing:
                cam_params[k] = BEST_PARAMS[k]
        print(f"Camera params loaded from {args.camera_params}: {cam_params}")
    cam_const = make_camera_const(cam_params)

    # ==========================================
    # Stage 1: Extract frames + FK + prompts
    # ==========================================
    print(f"\n=== Stage 1: Extract frames + FK ===")

    tmp_dir = os.path.join(out_dir, ".tmp")
    jpeg_dir = os.path.join(tmp_dir, "frames")
    os.makedirs(jpeg_dir, exist_ok=True)

    container_in = av.open(video_path)
    stream_in = container_in.streams.video[0]
    vid_fps = float(stream_in.average_rate)

    seek_ts = from_ts + start_frame / fps
    if seek_ts > 1.0:
        container_in.seek(int((seek_ts - 1.0) / stream_in.time_base), stream=stream_in)

    # Writers for FK intermediate outputs
    h, w = 480, 640
    writers = {}

    all_prompts = []      # (seq_idx, part_name, bbox)
    prev_visible = set()
    processed = 0
    t_start = time.time()
    frame_ep_fi_map = []
    fk_transforms_cache = {}  # seq_idx -> transforms (for overlay in stage 3)

    for av_frame in container_in.decode(stream_in):
        pts_sec = float(av_frame.pts * stream_in.time_base)
        ep_fi = int(round((pts_sec - from_ts) * fps))

        if ep_fi < start_frame:
            continue
        if ep_fi >= end_frame:
            break
        if ep_fi not in frame_data:
            continue

        img = av_frame.to_ndarray(format='bgr24')
        h, w = img.shape[:2]

        # Init writers on first frame
        if not writers:
            for vname in ["original", "fk_overlay", "fk_mask"]:
                vpath = os.path.join(out_dir, f"{vname}.mp4")
                c, s = open_video_writer(vpath, w, h, int(vid_fps))
                writers[vname] = (c, s)

        # Save JPEG for SAM2
        cv2.imwrite(os.path.join(jpeg_dir, f"{processed:05d}.jpg"), img)
        frame_ep_fi_map.append(ep_fi)

        # FK
        rq, hs = frame_data[ep_fi]
        q = build_q(model_pin, rq, hs, hand_type=hand_type)
        transforms = do_fk(model_pin, data_pin, q)

        # FK mask + overlay (combined single-pass)
        fk_mask, fk_overlay = render_mask_and_overlay(
            img, mesh_cache, transforms, BEST_PARAMS, h, w, cam_const)

        write_frame(*writers["original"], img)
        write_frame(*writers["fk_overlay"], fk_overlay)
        write_frame(*writers["fk_mask"], cv2.cvtColor(fk_mask, cv2.COLOR_GRAY2BGR))

        # Per-part visibility + prompts
        cur_visible = set()
        part_masks = {}
        for part_name, pcache in part_caches.items():
            if not pcache:
                continue
            m = render_mask_for_links(pcache, transforms, BEST_PARAMS, h, w)
            if np.count_nonzero(m) >= args.min_visible_area:
                cur_visible.add(part_name)
                part_masks[part_name] = m

        is_periodic = (processed == 0 or processed % args.prompt_interval == 0)
        newly_appeared = cur_visible - prev_visible
        parts_to_prompt = set()
        if is_periodic:
            parts_to_prompt = cur_visible.copy()
        if newly_appeared:
            parts_to_prompt |= newly_appeared

        if parts_to_prompt:
            vis_img = img.copy()
            for part_name in parts_to_prompt:
                bbox = mask_to_bbox(part_masks[part_name], margin=args.bbox_margin)
                if bbox is not None:
                    all_prompts.append((processed, part_name, bbox))
                    draw_box_prompt(vis_img, part_name, bbox)

            reason = "periodic" if is_periodic else "new"
            if is_periodic and newly_appeared:
                reason = "periodic+new"
            cv2.putText(vis_img, f"f{processed} ({reason})", (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.imwrite(os.path.join(prompt_vis_dir, f"{processed:05d}.jpg"), vis_img)

        prev_visible = cur_visible
        processed += 1
        if processed % 50 == 0 or processed == 1:
            print(f"  {processed}/{n_frames}")

    container_in.close()
    for c, s in writers.values():
        close_video(c, s)

    elapsed = time.time() - t_start
    print(f"Stage 1 done: {processed} frames in {elapsed:.1f}s, "
          f"{len(all_prompts)} prompts")

    # ==========================================
    # Stage 2: SAM2 segmentation
    # ==========================================
    print(f"\n=== Stage 2: SAM2 ({args.sam2_model}) ===")

    from sam2.sam2_video_predictor import SAM2VideoPredictor

    model_id = MODEL_IDS[args.sam2_model]
    print(f"Loading {model_id}...")
    predictor = SAM2VideoPredictor.from_pretrained(model_id, device=device)

    id_to_part = {v: k for k, v in PART_IDS.items()}
    frame_part_masks = {}  # seq_idx -> {part_name: mask_uint8}

    with torch.inference_mode():
        state = predictor.init_state(
            video_path=jpeg_dir, offload_video_to_cpu=True)

        for seq_idx, part_name, bbox in all_prompts:
            predictor.add_new_points_or_box(
                inference_state=state,
                frame_idx=seq_idx,
                obj_id=PART_IDS[part_name],
                box=bbox,
            )

        print(f"Added {len(all_prompts)} box prompts, propagating...")
        t_prop = time.time()
        count = 0
        for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(state):
            masks = (mask_logits > 0.0).cpu().numpy()
            parts = {}
            for i, oid in enumerate(obj_ids):
                pn = id_to_part.get(oid)
                if pn is None:
                    continue
                m = masks[i, 0].astype(np.uint8) * 255
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)
                parts[pn] = m
            frame_part_masks[frame_idx] = parts
            count += 1
            if count % 50 == 0:
                print(f"  {count}/{processed}")

    elapsed_prop = time.time() - t_prop
    print(f"SAM2 done: {count} frames in {elapsed_prop:.1f}s ({count/elapsed_prop:.1f} fps)")

    # ==========================================
    # Stage 3: Postprocess masks + inpaint
    # ==========================================
    inpaint_method = args.inpaint_method
    print(f"\n=== Stage 3: Postprocess + {inpaint_method} inpaint ===")

    # 3a. Generate mask videos + save binary masks for inpainting
    sam2_mask_c, sam2_mask_s = open_video_writer(
        os.path.join(out_dir, "sam2_mask.mp4"), w, h, int(vid_fps))
    sam2_overlay_c, sam2_overlay_s = open_video_writer(
        os.path.join(out_dir, "sam2_overlay.mp4"), w, h, int(vid_fps))
    final_mask_c, final_mask_s = open_video_writer(
        os.path.join(out_dir, "final_mask.mp4"), w, h, int(vid_fps))

    # Save binary masks for ProPainter (or LaMa)
    mask_png_dir = os.path.join(tmp_dir, "masks")
    os.makedirs(mask_png_dir, exist_ok=True)

    t_mask = time.time()
    for i in range(processed):
        img = cv2.imread(os.path.join(jpeg_dir, f"{i:05d}.jpg"))
        parts = frame_part_masks.get(i, {})

        sam2_mask_img = np.zeros_like(img)
        sam2_overlay_img = img.copy()
        combined_binary = np.zeros((h, w), dtype=np.uint8)

        for pn, m in parts.items():
            color = PART_COLORS.get(pn, (128, 128, 128))
            roi = m > 128
            sam2_mask_img[roi] = color
            sam2_overlay_img[roi] = (
                0.6 * sam2_overlay_img[roi] + 0.4 * np.array(color)
            ).astype(np.uint8)
            contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(sam2_overlay_img, contours, -1, color, 1, cv2.LINE_AA)
            combined_binary = np.maximum(combined_binary, m)

        final_mask = postprocess_mask(combined_binary)

        # Save binary mask PNG
        mask_binary = (final_mask > 128).astype(np.uint8) * 255
        cv2.imwrite(os.path.join(mask_png_dir, f"{i:05d}.png"), mask_binary)

        write_frame(sam2_mask_c, sam2_mask_s, sam2_mask_img)
        write_frame(sam2_overlay_c, sam2_overlay_s, sam2_overlay_img)
        write_frame(final_mask_c, final_mask_s, cv2.cvtColor(final_mask, cv2.COLOR_GRAY2BGR))

    close_video(sam2_mask_c, sam2_mask_s)
    close_video(sam2_overlay_c, sam2_overlay_s)
    close_video(final_mask_c, final_mask_s)
    print(f"Masks saved: {time.time() - t_mask:.1f}s")

    # 3b. Inpainting
    t_inpaint = time.time()

    if inpaint_method == "propainter":
        import subprocess
        propainter_dir = os.path.join(BASE_DIR, "ProPainter")

        # ProPainter loads all frames to GPU at once → OOM on long videos.
        # Split into segments of max_seg frames with overlap for blending.
        max_seg = 200
        overlap = 10

        if processed <= max_seg:
            segments = [(0, processed)]
        else:
            segments = []
            s = 0
            while s < processed:
                e = min(s + max_seg, processed)
                segments.append((s, e))
                s = e - overlap
                if processed - s <= overlap:
                    break

        print(f"Running ProPainter ({processed} frames, {len(segments)} segment(s))...")

        seg_results = []  # list of (start, end, output_dir)
        for si, (seg_start, seg_end) in enumerate(segments):
            n_seg = seg_end - seg_start
            seg_frames = os.path.join(tmp_dir, f"seg_frames_{si}")
            seg_masks = os.path.join(tmp_dir, f"seg_masks_{si}")
            seg_out = os.path.join(tmp_dir, f"pp_seg_{si}")
            os.makedirs(seg_frames, exist_ok=True)
            os.makedirs(seg_masks, exist_ok=True)

            # Symlink frames/masks for this segment (re-index from 0)
            for i in range(seg_start, seg_end):
                new_idx = f"{i - seg_start:05d}"
                old_idx = f"{i:05d}"
                os.symlink(os.path.abspath(os.path.join(jpeg_dir, f"{old_idx}.jpg")),
                           os.path.join(seg_frames, f"{new_idx}.jpg"))
                os.symlink(os.path.abspath(os.path.join(mask_png_dir, f"{old_idx}.png")),
                           os.path.join(seg_masks, f"{new_idx}.png"))

            cmd = [
                sys.executable,
                os.path.join(propainter_dir, "inference_propainter.py"),
                "--video", os.path.abspath(seg_frames),
                "--mask", os.path.abspath(seg_masks),
                "--output", os.path.abspath(seg_out),
                "--save_fps", str(int(vid_fps)),
                "--mask_dilation", "4",
                "--subvideo_length", "80",
                "--fp16", "--save_frames",
            ]
            print(f"  Segment {si+1}/{len(segments)}: frames {seg_start}-{seg_end-1} ({n_seg} frames)")
            # Propagate GPU selection to ProPainter subprocess.
            # If parent already has CUDA_VISIBLE_DEVICES set (parallel launcher),
            # inherit it so child sees the same physical GPU as slot 0.
            # Otherwise fall back to pinning via --device.
            sub_env = os.environ.copy()
            if "CUDA_VISIBLE_DEVICES" not in sub_env and device.startswith("cuda:"):
                sub_env["CUDA_VISIBLE_DEVICES"] = device.split(":", 1)[1]
            result = subprocess.run(cmd, cwd=propainter_dir, env=sub_env)

            # Cleanup symlinks
            shutil.rmtree(seg_frames, ignore_errors=True)
            shutil.rmtree(seg_masks, ignore_errors=True)

            if result.returncode != 0:
                print(f"  ProPainter segment {si+1} failed (code {result.returncode})")
                print(f"  Temp files at: {tmp_dir}")
                return

            seg_results.append((seg_start, seg_end, seg_out))

        # Stitch segments into final video (blend overlap regions)
        inpaint_out = os.path.join(out_dir, "inpaint.mp4")
        inpaint_c, inpaint_s = open_video_writer(inpaint_out, w, h, int(vid_fps))

        written = 0
        for si, (seg_start, seg_end, seg_out) in enumerate(seg_results):
            # Find the frames dir inside ProPainter output
            pp_frames_dir = None
            for root, dirs, files in os.walk(seg_out):
                pngs = [f for f in files if f.endswith('.png')]
                if pngs:
                    pp_frames_dir = root
                    break

            if pp_frames_dir is None:
                print(f"  WARNING: No frames found in segment {si+1}")
                continue

            n_seg = seg_end - seg_start
            for i in range(n_seg):
                global_idx = seg_start + i

                # Skip frames already written (overlap region from prev segment)
                if global_idx < written:
                    continue

                frame_path = os.path.join(pp_frames_dir, f"{i:04d}.png")
                if not os.path.exists(frame_path):
                    continue

                frame = cv2.imread(frame_path)
                if frame is None:
                    continue

                # In overlap region, blend with decreasing weight
                # (simple: just use the new segment's frames for overlap)
                write_frame(inpaint_c, inpaint_s, frame)
                written += 1

        close_video(inpaint_c, inpaint_s)
        print(f"ProPainter done: {written} frames, {time.time() - t_inpaint:.1f}s")

    else:  # lama
        print("Loading LaMa...")
        lama = init_lama()
        inpaint_c, inpaint_s = open_video_writer(
            os.path.join(out_dir, "inpaint.mp4"), w, h, int(vid_fps))

        for i in range(processed):
            img = cv2.imread(os.path.join(jpeg_dir, f"{i:05d}.jpg"))
            mask = cv2.imread(os.path.join(mask_png_dir, f"{i:05d}.png"),
                              cv2.IMREAD_GRAYSCALE)
            inpainted = run_lama(lama, img, mask)
            write_frame(inpaint_c, inpaint_s, inpainted)

            if (i + 1) % 50 == 0 or i == 0:
                elapsed = time.time() - t_inpaint
                fps_p = (i + 1) / elapsed
                eta = (processed - i - 1) / fps_p if fps_p > 0 else 0
                print(f"  {i+1}/{processed} ({fps_p:.1f} fps, ETA {eta:.0f}s)")

        close_video(inpaint_c, inpaint_s)

    # Cleanup tmp
    shutil.rmtree(tmp_dir, ignore_errors=True)

    total = time.time() - t_start
    print(f"\nDone in {total:.1f}s ({inpaint_method})")
    print(f"Output: {out_dir}/")
    for f in ["original", "fk_overlay", "fk_mask", "sam2_mask",
              "sam2_overlay", "final_mask", "inpaint"]:
        print(f"  {f}.mp4")
    print(f"  prompt_vis/ ({len(os.listdir(prompt_vis_dir))} frames)")


if __name__ == "__main__":
    main()
