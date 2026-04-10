"""
SAM2 fine-grained video segmentation with FK-based prompts.

Segments robot body parts independently (arms, hands, legs, torso).
Supports box prompt and point prompt modes.

Stage 1: Extract JPEG frames + per-part FK masks → bbox/centroid + visibility tracking
Stage 2: SAM2 video prediction with multi-object prompts
Stage 3: Encode output videos with colored per-part masks + prompt visualizations

Usage:
  python scripts/sam2_segment.py --episode 4 --start 5 --duration 5 --mode box
  python scripts/sam2_segment.py --episode 4 --start 5 --duration 5 --mode point
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

from config import G1_URDF, MESH_DIR, BEST_PARAMS, SKIP_MESHES, OUTPUT_DIR, get_hand_type
from video_inpaint import (
    build_q, do_fk, parse_urdf_meshes, preload_meshes,
    make_camera,
    open_video_writer, write_frame, close_video,
    load_episode_info,
)

SAM2_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "sam2_segment")

MODEL_IDS = {
    "tiny": "facebook/sam2.1-hiera-tiny",
    "small": "facebook/sam2.1-hiera-small",
    "base": "facebook/sam2.1-hiera-base-plus",
    "large": "facebook/sam2.1-hiera-large",
}

# Body part grouping: name -> list of link name patterns
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

# obj_id for SAM2 (1-based)
PART_IDS = {name: i + 1 for i, name in enumerate(BODY_PARTS)}

# Colors for visualization (BGR)
PART_COLORS = {
    "left_arm":  (0, 100, 255),    # orange
    "left_hand": (0, 200, 255),    # yellow
    "right_arm": (255, 100, 0),    # blue
    "right_hand":(255, 200, 0),    # cyan
    "left_leg":  (200, 0, 200),    # purple
    "right_leg": (100, 0, 200),    # pink
    "torso":     (100, 200, 0),    # green
}

MIN_VISIBLE_AREA = 50  # pixels


def match_links(mesh_cache, patterns):
    """Filter mesh_cache keys matching any of the glob patterns."""
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
    """Render triangle mask for a subset of links."""
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
    x1 = max(0, int(xs.min()) - margin)
    y1 = max(0, int(ys.min()) - margin)
    x2 = min(w, int(xs.max()) + margin)
    y2 = min(h, int(ys.max()) + margin)
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def mask_to_centroid(mask):
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    cx = int(xs.mean())
    cy = int(ys.mean())
    return np.array([[cx, cy]], dtype=np.float32)


def draw_prompt_vis(img, part_name, prompt_data, mode):
    """Draw prompt visualization on image."""
    color = PART_COLORS[part_name]
    if mode == "box" and prompt_data is not None:
        x1, y1, x2, y2 = prompt_data.astype(int)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, part_name, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    elif mode == "point" and prompt_data is not None:
        cx, cy = prompt_data[0].astype(int)
        cv2.circle(img, (cx, cy), 6, color, -1)
        cv2.circle(img, (cx, cy), 8, (255, 255, 255), 1)
        cv2.putText(img, part_name, (cx + 10, cy - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)


def main():
    import av

    parser = argparse.ArgumentParser(description="SAM2 fine-grained segmentation")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--start", type=float, default=0)
    parser.add_argument("--duration", type=float, default=0)
    parser.add_argument("--model", default="small",
                        choices=["tiny", "small", "base", "large"])
    parser.add_argument("--mode", default="box", choices=["box", "point"])
    parser.add_argument("--prompt-interval", type=int, default=30)
    args = parser.parse_args()

    ep = args.episode
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}, Mode: {args.mode}")

    # Output subdirectory per mode
    out_dir = os.path.join(SAM2_OUTPUT_DIR, f"ep{ep:03d}_{args.mode}")
    prompt_vis_dir = os.path.join(out_dir, "prompt_vis")
    os.makedirs(prompt_vis_dir, exist_ok=True)

    # Load episode
    print(f"Episode: {ep}")
    video_path, from_ts, to_ts, ep_df = load_episode_info(ep)
    n_total = len(ep_df)
    fps = 30

    start_frame = int(args.start * fps) if args.start > 0 else 0
    end_frame = min(start_frame + int(args.duration * fps), n_total) if args.duration > 0 else n_total
    ep_df = ep_df.iloc[start_frame:end_frame]
    n_frames = len(ep_df)
    print(f"Frames: {start_frame}-{end_frame-1} ({n_frames} frames, {n_frames/fps:.1f}s)")

    frame_data = {}
    for _, row in ep_df.iterrows():
        fi = int(row["frame_index"])
        rq = np.array(row["observation.state.robot_q_current"], dtype=np.float64)
        hs = np.array(row["observation.state.hand_state"], dtype=np.float64)
        frame_data[fi] = (rq, hs)

    # Load URDF + meshes
    print("Loading URDF and meshes...")
    model_pin = pin.buildModelFromUrdf(G1_URDF, pin.JointModelFreeFlyer())
    data_pin = model_pin.createData()
    link_meshes = parse_urdf_meshes(G1_URDF)
    mesh_cache = preload_meshes(link_meshes, MESH_DIR)

    # Pre-filter mesh cache per body part
    part_caches = {}
    for part_name, patterns in BODY_PARTS.items():
        part_caches[part_name] = match_links(mesh_cache, patterns)
    print(f"Body parts: {', '.join(f'{k}({len(v)})' for k, v in part_caches.items() if v)}")

    # ====== Stage 1: Extract frames + per-part FK prompts ======
    print(f"\n=== Stage 1: Extract frames + FK prompts ===")

    tmp_dir = os.path.join(out_dir, ".tmp")
    jpeg_dir = os.path.join(tmp_dir, "frames")
    os.makedirs(jpeg_dir, exist_ok=True)

    container_in = av.open(video_path)
    stream_in = container_in.streams.video[0]
    vid_fps = float(stream_in.average_rate)

    seek_ts = from_ts + start_frame / fps
    if seek_ts > 1.0:
        container_in.seek(int((seek_ts - 1.0) / stream_in.time_base), stream=stream_in)

    # prompts: list of (seq_idx, part_name, prompt_data)
    all_prompts = []
    prev_visible = set()
    processed = 0
    h, w = 480, 640
    t_start = time.time()
    frame_ep_fi_map = []  # seq_idx -> ep_fi

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

        cv2.imwrite(os.path.join(jpeg_dir, f"{processed:05d}.jpg"), img)
        frame_ep_fi_map.append(ep_fi)

        # FK on every frame to track visibility
        rq, hs = frame_data[ep_fi]
        q = build_q(model_pin, rq, hs, hand_type=get_hand_type())
        transforms = do_fk(model_pin, data_pin, q)

        cur_visible = set()
        part_masks = {}
        for part_name, pcache in part_caches.items():
            if not pcache:
                continue
            m = render_mask_for_links(pcache, transforms, BEST_PARAMS, h, w)
            area = np.count_nonzero(m)
            if area >= MIN_VISIBLE_AREA:
                cur_visible.add(part_name)
                part_masks[part_name] = m

        # Determine which parts need prompting
        is_periodic = (processed == 0 or processed % args.prompt_interval == 0)
        newly_appeared = cur_visible - prev_visible

        parts_to_prompt = set()
        if is_periodic:
            parts_to_prompt = cur_visible.copy()
        if newly_appeared:
            parts_to_prompt |= newly_appeared

        # Generate prompts and visualization
        if parts_to_prompt:
            vis_img = img.copy()
            for part_name in parts_to_prompt:
                m = part_masks[part_name]
                if args.mode == "box":
                    prompt_data = mask_to_bbox(m)
                else:
                    prompt_data = mask_to_centroid(m)
                if prompt_data is not None:
                    all_prompts.append((processed, part_name, prompt_data))
                    draw_prompt_vis(vis_img, part_name, prompt_data, args.mode)

            # Save prompt visualization
            reason = "periodic" if is_periodic else "new"
            if is_periodic and newly_appeared:
                reason = "periodic+new"
            cv2.putText(vis_img, f"f{processed} ({reason})", (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.imwrite(os.path.join(prompt_vis_dir, f"{processed:05d}.jpg"), vis_img)

        prev_visible = cur_visible
        processed += 1
        if processed % 50 == 0 or processed == 1:
            vis_parts = ", ".join(sorted(cur_visible)) if cur_visible else "none"
            print(f"  {processed}/{n_frames} | visible: {vis_parts}")

    container_in.close()
    elapsed = time.time() - t_start
    print(f"Stage 1 done: {processed} frames in {elapsed:.1f}s, "
          f"{len(all_prompts)} prompts on "
          f"{len(set(p[0] for p in all_prompts))} frames")

    # ====== Stage 2: SAM2 video prediction ======
    print(f"\n=== Stage 2: SAM2 segmentation ({args.model}, {args.mode}) ===")

    from sam2.sam2_video_predictor import SAM2VideoPredictor

    model_id = MODEL_IDS[args.model]
    print(f"Loading {model_id}...")
    predictor = SAM2VideoPredictor.from_pretrained(model_id, device=device)

    # Store per-frame per-part masks
    frame_part_masks = {}  # frame_idx -> {part_name: mask_uint8}

    with torch.inference_mode():
        state = predictor.init_state(
            video_path=jpeg_dir,
            offload_video_to_cpu=True,
        )

        # Add prompts
        for seq_idx, part_name, prompt_data in all_prompts:
            obj_id = PART_IDS[part_name]
            if args.mode == "box":
                predictor.add_new_points_or_box(
                    inference_state=state,
                    frame_idx=seq_idx,
                    obj_id=obj_id,
                    box=prompt_data,
                )
            else:
                predictor.add_new_points_or_box(
                    inference_state=state,
                    frame_idx=seq_idx,
                    obj_id=obj_id,
                    points=prompt_data,
                    labels=np.array([1], dtype=np.int32),
                )

        unique_frames = len(set(p[0] for p in all_prompts))
        unique_parts = len(set(p[1] for p in all_prompts))
        print(f"Added {len(all_prompts)} prompts ({unique_parts} parts, {unique_frames} frames)")

        # Propagate
        print("Propagating masks...")
        t_prop = time.time()
        frame_count = 0

        # Build reverse map: obj_id -> part_name
        id_to_part = {v: k for k, v in PART_IDS.items()}

        for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(state):
            masks = (mask_logits > 0.0).cpu().numpy()  # [n_obj, 1, H, W]
            parts = {}
            for i, oid in enumerate(obj_ids):
                part_name = id_to_part.get(oid)
                if part_name is None:
                    continue
                m = masks[i, 0].astype(np.uint8) * 255
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)
                parts[part_name] = m
            frame_part_masks[frame_idx] = parts
            frame_count += 1
            if frame_count % 50 == 0:
                print(f"  {frame_count}/{processed} frames")

    elapsed_prop = time.time() - t_prop
    print(f"Propagation done: {frame_count} frames in {elapsed_prop:.1f}s "
          f"({frame_count/elapsed_prop:.1f} fps)")

    # ====== Stage 3: Encode output videos ======
    print(f"\n=== Stage 3: Encoding videos ===")

    orig_c, orig_s = open_video_writer(
        os.path.join(out_dir, "original.mp4"), w, h, int(vid_fps))
    mask_c, mask_s = open_video_writer(
        os.path.join(out_dir, "mask.mp4"), w, h, int(vid_fps))
    overlay_c, overlay_s = open_video_writer(
        os.path.join(out_dir, "overlay.mp4"), w, h, int(vid_fps))

    for i in range(processed):
        img = cv2.imread(os.path.join(jpeg_dir, f"{i:05d}.jpg"))
        parts = frame_part_masks.get(i, {})

        # Colored mask image
        mask_img = np.zeros_like(img)
        overlay = img.copy()

        for part_name, m in parts.items():
            color = PART_COLORS.get(part_name, (128, 128, 128))
            roi = m > 128
            mask_img[roi] = color
            overlay[roi] = (0.6 * overlay[roi] + 0.4 * np.array(color)).astype(np.uint8)
            # Contour
            contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, color, 1, cv2.LINE_AA)

        write_frame(orig_c, orig_s, img)
        write_frame(mask_c, mask_s, mask_img)
        write_frame(overlay_c, overlay_s, overlay)

    close_video(orig_c, orig_s)
    close_video(mask_c, mask_s)
    close_video(overlay_c, overlay_s)

    # Cleanup tmp frames (keep prompt_vis)
    shutil.rmtree(tmp_dir, ignore_errors=True)

    total = time.time() - t_start
    print(f"\nDone in {total:.1f}s")
    print(f"Output: {out_dir}/")
    print(f"  original.mp4, mask.mp4, overlay.mp4")
    print(f"  prompt_vis/ ({len(os.listdir(prompt_vis_dir))} frames)")


if __name__ == "__main__":
    main()
