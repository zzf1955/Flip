"""Generate per-sample hand-region weight maps for Mitty training.

Uses FK mesh projection to locate robot hands in each frame, converts
pixel-space bounding boxes to latent-space weight maps.  Output is a
directory of .pth files with the same naming as the mitty_cache dir,
ready to be loaded by train_mitty.py via --patch-dir.

Usage:
  # Generate weights for train split
  python -m src.pipeline.hand_patch \
      --cache-dir output/mitty_cache_1s/train \
      --output    output/hand_patch_1s/train \
      --hand-weight 3.0

  # With debug overlay videos
  python -m src.pipeline.hand_patch \
      --cache-dir output/mitty_cache_1s/eval \
      --output    output/hand_patch_1s/eval \
      --debug-dir output/hand_patch_1s/debug/eval

  # All three splits at once
  for split in train eval ood_eval; do
    python -m src.pipeline.hand_patch \
        --cache-dir output/mitty_cache_1s/$split \
        --output    output/hand_patch_1s/$split
  done
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pinocchio as pin
import torch

sys.stdout.reconfigure(line_buffering=True)

from src.core.config import (
    BEST_PARAMS, G1_URDF, MAIN_ROOT, MESH_DIR, SEGMENT_DIR,
    TRAINING_DATA_ROOT, get_skip_meshes,
)
from src.core.camera import make_camera_const
from src.core.fk import build_q, do_fk, parse_urdf_meshes, preload_meshes
from src.pipeline.sam2_segment import (
    BODY_PARTS, mask_to_bbox, match_links, render_mask_for_links,
)


DEFAULT_SOURCE_MAP = os.path.join(TRAINING_DATA_ROOT, "pair", "1s", "source_map.json")
SEGMENT_FPS = 30
CLIP_FRAMES_1S = 30  # 1s at 30fps
VIDEO_H, VIDEO_W = 480, 640
LATENT_H, LATENT_W = 30, 40
LATENT_F = 5
VAE_SPATIAL = 16  # VAE spatial downsampling factor


# ── FK helpers ────────────────────────────────────────────────────────

def load_fk_model():
    model = pin.buildModelFromUrdf(str(G1_URDF), pin.JointModelFreeFlyer())
    data = model.createData()
    link_meshes = parse_urdf_meshes(str(G1_URDF))
    skip_set = get_skip_meshes("inspire")
    mesh_cache = preload_meshes(link_meshes, str(MESH_DIR), skip_set=skip_set)
    hand_caches = {
        "left_hand": match_links(mesh_cache, BODY_PARTS["left_hand"]),
        "right_hand": match_links(mesh_cache, BODY_PARTS["right_hand"]),
    }
    cam_const = make_camera_const(BEST_PARAMS)
    return model, data, mesh_cache, hand_caches, cam_const


def compute_hand_bboxes(
    model, data, hand_caches, cam_const, clip_df, margin=20,
):
    """Compute union bounding box for each hand across all frames in clip_df."""
    all_bboxes = {"left_hand": [], "right_hand": []}

    for _, row in clip_df.iterrows():
        rq = np.array(row["observation.state.robot_q_current"], dtype=np.float64)
        hs = np.array(row["observation.state.hand_state"], dtype=np.float64)
        q = build_q(model, rq, hs, hand_type="inspire")
        transforms = do_fk(model, data, q)

        for hand_name in ("left_hand", "right_hand"):
            mask = render_mask_for_links(
                hand_caches[hand_name], transforms, BEST_PARAMS,
                VIDEO_H, VIDEO_W,
            )
            bbox = mask_to_bbox(mask, margin=margin)
            if bbox is not None:
                all_bboxes[hand_name].append(bbox)

    union = {}
    for hand_name, bboxes in all_bboxes.items():
        if not bboxes:
            union[hand_name] = None
            continue
        arr = np.stack(bboxes)
        union[hand_name] = [
            int(arr[:, 0].min()),
            int(arr[:, 1].min()),
            int(arr[:, 2].max()),
            int(arr[:, 3].max()),
        ]
    return union


def pixel_bbox_to_latent(bbox):
    """Convert pixel-space [x1,y1,x2,y2] to latent-space, clamped."""
    if bbox is None:
        return None
    x1, y1, x2, y2 = bbox
    lx1 = max(0, x1 // VAE_SPATIAL)
    ly1 = max(0, y1 // VAE_SPATIAL)
    lx2 = min(LATENT_W, math.ceil(x2 / VAE_SPATIAL))
    ly2 = min(LATENT_H, math.ceil(y2 / VAE_SPATIAL))
    if lx2 <= lx1 or ly2 <= ly1:
        return None
    return [lx1, ly1, lx2, ly2]


def build_weight_map(bboxes_latent, hand_weight):
    """Build (LATENT_F, LATENT_H, LATENT_W) weight tensor."""
    weights = torch.ones(LATENT_F, LATENT_H, LATENT_W, dtype=torch.float32)
    for bbox in bboxes_latent.values():
        if bbox is None:
            continue
        lx1, ly1, lx2, ly2 = bbox
        weights[:, ly1:ly2, lx1:lx2] = hand_weight
    return weights


# ── Debug overlay ─────────────────────────────────────────────────────

def draw_debug_frame(frame, bboxes_pixel, bboxes_latent, hand_weight):
    """Draw hand bboxes + latent grid + highlighted cells on a frame."""
    out = frame.copy()

    # Draw latent grid (every 16px)
    for x in range(0, VIDEO_W + 1, VAE_SPATIAL):
        cv2.line(out, (x, 0), (x, VIDEO_H - 1), (60, 60, 60), 1)
    for y in range(0, VIDEO_H + 1, VAE_SPATIAL):
        cv2.line(out, (0, y), (VIDEO_W - 1, y), (60, 60, 60), 1)

    colors = {"left_hand": (0, 200, 255), "right_hand": (255, 200, 0)}

    # Highlight latent cells with elevated weight
    overlay = out.copy()
    for hand_name, lbbox in bboxes_latent.items():
        if lbbox is None:
            continue
        lx1, ly1, lx2, ly2 = lbbox
        px1, py1 = lx1 * VAE_SPATIAL, ly1 * VAE_SPATIAL
        px2, py2 = lx2 * VAE_SPATIAL, ly2 * VAE_SPATIAL
        cv2.rectangle(overlay, (px1, py1), (px2, py2), colors[hand_name], -1)
    cv2.addWeighted(overlay, 0.25, out, 0.75, 0, out)

    # Draw pixel bboxes
    for hand_name, pbbox in bboxes_pixel.items():
        if pbbox is None:
            continue
        x1, y1, x2, y2 = pbbox
        cv2.rectangle(out, (x1, y1), (x2, y2), colors[hand_name], 2)
        cv2.putText(out, f"{hand_name} w={hand_weight:.1f}",
                    (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    colors[hand_name], 1)

    return out


def write_debug_video(
    robot_video_path, clip_start_sec, out_path,
    bboxes_pixel, bboxes_latent, hand_weight,
    model, data, hand_caches, clip_df,
):
    """Write a debug overlay video for one pair."""
    import av
    container = av.open(robot_video_path)
    stream = container.streams.video[0]

    fps = float(stream.average_rate or 30)
    start_frame = int(clip_start_sec * fps)
    end_frame = start_frame + CLIP_FRAMES_1S

    frames_bgr = []
    for i, av_frame in enumerate(container.decode(video=0)):
        if i < start_frame:
            continue
        if i >= end_frame:
            break
        img = av_frame.to_ndarray(format="bgr24")
        frames_bgr.append(img)
    container.close()

    if not frames_bgr:
        return

    # Compute per-frame FK masks for overlay
    rows = list(clip_df.iterrows())
    h, w = frames_bgr[0].shape[:2]

    writer = cv2.VideoWriter(
        out_path, cv2.VideoWriter_fourcc(*"mp4v"),
        min(fps, 30), (w, h),
    )

    for fi, frame in enumerate(frames_bgr):
        # FK overlay for this frame
        if fi < len(rows):
            _, row = rows[fi]
            rq = np.array(row["observation.state.robot_q_current"], dtype=np.float64)
            hs = np.array(row["observation.state.hand_state"], dtype=np.float64)
            q = build_q(model, rq, hs, hand_type="inspire")
            transforms = do_fk(model, data, q)

            # Draw hand mesh masks as semi-transparent overlay
            colors_bgr = {"left_hand": (0, 200, 255), "right_hand": (255, 200, 0)}
            mesh_overlay = frame.copy()
            for hand_name, cache in hand_caches.items():
                mask = render_mask_for_links(
                    cache, transforms, BEST_PARAMS, h, w)
                if mask.any():
                    mesh_overlay[mask > 0] = colors_bgr[hand_name]
            cv2.addWeighted(mesh_overlay, 0.3, frame, 0.7, 0, frame)

        out_frame = draw_debug_frame(
            frame, bboxes_pixel, bboxes_latent, hand_weight)
        writer.write(out_frame)

    writer.release()


# ── Main ──────────────────────────────────────────────────────────────

def process_pair(
    pair_name, split, source_map, model, data, hand_caches, cam_const,
    hand_weight, margin, debug_dir=None,
):
    key = f"{split}/{pair_name}.mp4"
    if key not in source_map:
        return None

    info = source_map[key]
    task = info["task"]
    ep = info["episode"]
    seg = info["seg"]
    clip_idx = info.get("clip_idx")
    if clip_idx is None:
        clip_idx = 0

    # Load segment parquet
    parquet_path = os.path.join(SEGMENT_DIR, task, ep, f"{seg}_joints.parquet")
    if not os.path.isfile(parquet_path):
        print(f"  WARN: parquet not found: {parquet_path}")
        return None

    df = pd.read_parquet(parquet_path)
    seg_frame_min = int(df["frame_index"].min())
    frame_start = seg_frame_min + clip_idx * CLIP_FRAMES_1S
    frame_end = frame_start + CLIP_FRAMES_1S
    clip_df = df[(df["frame_index"] >= frame_start) & (df["frame_index"] < frame_end)]

    if clip_df.empty:
        print(f"  WARN: no frames for {pair_name} clip_idx={clip_idx}")
        return None

    # Compute hand bboxes
    bboxes_pixel = compute_hand_bboxes(
        model, data, hand_caches, cam_const, clip_df, margin=margin)

    bboxes_latent = {
        k: pixel_bbox_to_latent(v) for k, v in bboxes_pixel.items()
    }

    weights = build_weight_map(bboxes_latent, hand_weight)

    # Debug overlay
    if debug_dir:
        robot_video = os.path.join(SEGMENT_DIR, task, ep, f"{seg}_video.mp4")
        if os.path.isfile(robot_video):
            clip_start_sec = clip_idx * 1.0
            debug_path = os.path.join(debug_dir, f"{pair_name}.mp4")
            write_debug_video(
                robot_video, clip_start_sec, debug_path,
                bboxes_pixel, bboxes_latent, hand_weight,
                model, data, hand_caches, clip_df,
            )

    return {
        "weights": weights,
        "bboxes_pixel": {k: v for k, v in bboxes_pixel.items()},
        "bboxes_latent": bboxes_latent,
        "hand_weight": hand_weight,
        "source_id": info.get("source_id", ""),
    }


def main():
    ap = argparse.ArgumentParser(
        description="Generate hand-region weight maps for Mitty training")
    ap.add_argument("--cache-dir", required=True,
                    help="mitty_cache dir (reads pair filenames + source_id)")
    ap.add_argument("--output", required=True,
                    help="output weight map directory")
    ap.add_argument("--source-map", default=DEFAULT_SOURCE_MAP,
                    help="path to source_map.json")
    ap.add_argument("--hand-weight", type=float, default=3.0,
                    help="weight multiplier for hand regions (default: 3.0)")
    ap.add_argument("--margin", type=int, default=20,
                    help="pixel margin around FK hand bbox (default: 20)")
    ap.add_argument("--debug-dir", default="",
                    help="if set, write debug overlay videos here")
    ap.add_argument("--resume", action="store_true",
                    help="skip pairs whose .pth already exists")
    args = ap.parse_args()

    # Resolve relative paths
    for attr in ("cache_dir", "output", "source_map", "debug_dir"):
        val = getattr(args, attr)
        if val and not os.path.isabs(val):
            setattr(args, attr, os.path.join(MAIN_ROOT, val))

    cache_dir = Path(args.cache_dir).resolve()
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.debug_dir:
        Path(args.debug_dir).mkdir(parents=True, exist_ok=True)

    # Detect split name from cache_dir basename
    split = cache_dir.name  # e.g. "train", "eval", "ood_eval"

    # Load source map
    source_map = json.load(open(args.source_map))
    print(f"Source map: {len(source_map)} entries from {args.source_map}")

    # Enumerate pairs from cache dir
    pair_files = sorted(cache_dir.glob("pair_*.pth"))
    if not pair_files:
        print(f"No pair_*.pth files in {cache_dir}")
        return
    print(f"Pairs:       {len(pair_files)} in {cache_dir}")
    print(f"Output:      {output_dir}")
    print(f"Split:       {split}")
    print(f"Hand weight: {args.hand_weight}")
    print(f"Margin:      {args.margin}px")
    if args.debug_dir:
        print(f"Debug dir:   {args.debug_dir}")

    # Load FK model once
    print("Loading FK model + meshes...")
    t0 = time.time()
    model, data, mesh_cache, hand_caches, cam_const = load_fk_model()
    print(f"FK model loaded in {time.time() - t0:.1f}s")

    skipped = 0
    processed = 0
    no_source = 0
    t_start = time.time()

    for idx, pth_path in enumerate(pair_files):
        pair_name = pth_path.stem  # e.g. "pair_0042"
        out_path = output_dir / f"{pair_name}.pth"

        if args.resume and out_path.exists():
            skipped += 1
            continue

        result = process_pair(
            pair_name, split, source_map,
            model, data, hand_caches, cam_const,
            args.hand_weight, args.margin,
            debug_dir=args.debug_dir if args.debug_dir else None,
        )

        if result is None:
            no_source += 1
            continue

        torch.save(result, str(out_path))
        processed += 1

        if (idx + 1) % 10 == 0 or idx == len(pair_files) - 1:
            lh = result["bboxes_latent"].get("left_hand")
            rh = result["bboxes_latent"].get("right_hand")
            print(f"  [{idx+1}/{len(pair_files)}] {pair_name}"
                  f"  L={lh}  R={rh}")

    elapsed = time.time() - t_start
    print(f"\nDone. {processed} processed, {skipped} skipped, "
          f"{no_source} no source. {elapsed:.1f}s")


if __name__ == "__main__":
    main()
