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
  python scripts/sam2_inpaint_pipeline.py --episode 4 --start 5 --duration 5
"""

import sys
import os
import argparse
import time
import shutil
import fnmatch
import numpy as np
import pandas as pd
import cv2
import pinocchio as pin
import torch

sys.stdout.reconfigure(line_buffering=True)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "scripts"))

from video_inpaint import (
    URDF_PATH, MESH_DIR, BEST_PARAMS, SKIP_MESHES,
    build_q, do_fk, parse_urdf_meshes, preload_meshes,
    make_camera, render_mask, render_overlay,
    postprocess_mask, init_lama, run_lama,
    open_video_writer, write_frame, close_video,
    load_episode_info,
)

OUTPUT_DIR = os.path.join(BASE_DIR, "test_results", "inpaint_video")

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

MIN_VISIBLE_AREA = 50


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
    K, D, rvec, tvec, R_w2c, t_w2c = make_camera(params, transforms)
    mask = np.zeros((h, w), dtype=np.uint8)
    for link_name, (tris, _) in filtered_cache.items():
        if link_name not in transforms:
            continue
        t_link, R_link = transforms[link_name]
        flat = tris.reshape(-1, 3)
        world = (R_link @ flat.T).T + t_link
        cam_pts = (R_w2c @ world.T).T + t_w2c.flatten()
        z_cam = cam_pts[:, 2]
        pts2d, _ = cv2.fisheye.projectPoints(
            world.reshape(-1, 1, 3), rvec, tvec, K, D)
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


def mask_to_bbox(mask, margin=20):
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
    args = parser.parse_args()

    ep = args.episode
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    # Output directory
    tag = f"ep{ep:03d}"
    fps = 30
    start_frame = int(args.start * fps) if args.start > 0 else 0

    video_path, from_ts, to_ts, ep_df = load_episode_info(ep)
    n_total = len(ep_df)
    end_frame = min(start_frame + int(args.duration * fps), n_total) if args.duration > 0 else n_total
    ep_df = ep_df.iloc[start_frame:end_frame]
    n_frames = len(ep_df)

    if args.start > 0 or args.duration > 0:
        tag += f"_{start_frame}-{end_frame}"

    out_dir = os.path.join(OUTPUT_DIR, tag)
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
    print("Loading URDF and meshes...")
    model_pin = pin.buildModelFromUrdf(URDF_PATH, pin.JointModelFreeFlyer())
    data_pin = model_pin.createData()
    link_meshes = parse_urdf_meshes(URDF_PATH)
    mesh_cache = preload_meshes(link_meshes, MESH_DIR)

    part_caches = {name: match_links(mesh_cache, pats) for name, pats in BODY_PARTS.items()}

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
        q = build_q(model_pin, rq, hs)
        transforms = do_fk(model_pin, data_pin, q)

        # FK overlay
        fk_overlay = render_overlay(img, mesh_cache, transforms, BEST_PARAMS)
        # FK mask
        fk_mask = render_mask(mesh_cache, transforms, BEST_PARAMS, h, w)

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
            if np.count_nonzero(m) >= MIN_VISIBLE_AREA:
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
                bbox = mask_to_bbox(part_masks[part_name])
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
    # Stage 3: Postprocess masks + LaMa inpaint
    # ==========================================
    print(f"\n=== Stage 3: Postprocess + LaMa inpaint ===")

    print("Loading LaMa...")
    lama = init_lama()

    sam2_mask_c, sam2_mask_s = open_video_writer(
        os.path.join(out_dir, "sam2_mask.mp4"), w, h, int(vid_fps))
    sam2_overlay_c, sam2_overlay_s = open_video_writer(
        os.path.join(out_dir, "sam2_overlay.mp4"), w, h, int(vid_fps))
    final_mask_c, final_mask_s = open_video_writer(
        os.path.join(out_dir, "final_mask.mp4"), w, h, int(vid_fps))
    inpaint_c, inpaint_s = open_video_writer(
        os.path.join(out_dir, "inpaint.mp4"), w, h, int(vid_fps))

    t_inpaint = time.time()
    for i in range(processed):
        img = cv2.imread(os.path.join(jpeg_dir, f"{i:05d}.jpg"))
        parts = frame_part_masks.get(i, {})

        # Colored SAM2 mask
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

        # Postprocess: smooth + dilate + edge blur
        final_mask = postprocess_mask(combined_binary)

        # LaMa inpaint
        inpainted = run_lama(lama, img, final_mask)

        write_frame(sam2_mask_c, sam2_mask_s, sam2_mask_img)
        write_frame(sam2_overlay_c, sam2_overlay_s, sam2_overlay_img)
        write_frame(final_mask_c, final_mask_s, cv2.cvtColor(final_mask, cv2.COLOR_GRAY2BGR))
        write_frame(inpaint_c, inpaint_s, inpainted)

        if (i + 1) % 50 == 0 or i == 0:
            elapsed = time.time() - t_inpaint
            fps_p = (i + 1) / elapsed
            eta = (processed - i - 1) / fps_p if fps_p > 0 else 0
            print(f"  {i+1}/{processed} ({fps_p:.1f} fps, ETA {eta:.0f}s)")

    close_video(sam2_mask_c, sam2_mask_s)
    close_video(sam2_overlay_c, sam2_overlay_s)
    close_video(final_mask_c, final_mask_s)
    close_video(inpaint_c, inpaint_s)

    # Cleanup tmp
    shutil.rmtree(tmp_dir, ignore_errors=True)

    total = time.time() - t_start
    print(f"\nDone in {total:.1f}s")
    print(f"Output: {out_dir}/")
    for f in ["original", "fk_overlay", "fk_mask", "sam2_mask",
              "sam2_overlay", "final_mask", "inpaint"]:
        print(f"  {f}.mp4")
    print(f"  prompt_vis/ ({len(os.listdir(prompt_vis_dir))} frames)")


if __name__ == "__main__":
    main()
