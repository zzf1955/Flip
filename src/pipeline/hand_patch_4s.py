"""
Precompute per-frame hand FK bounding boxes at the 4s segment level.

For each segment, runs G1 FK on every frame, renders hand masks, and stores
per-frame pixel-space bounding boxes as a parquet file.  These are later
consumed by make_pair.py to generate latent-space weight maps for each clip.

Input:   training_data/segment/<task>/ep<N>/seg<M>_joints.parquet
Output:  training_data/hand_patch/4s/<task>/ep<N>/seg<M>_hands.parquet

Usage:
  python -m src.pipeline.hand_patch_4s --task all --resume
  python -m src.pipeline.hand_patch_4s --task Inspire_Pickup_Pillow_MainCamOnly
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.stdout.reconfigure(line_buffering=True)

from src.core.config import BEST_PARAMS, MAIN_ROOT, get_hand_type
from src.pipeline.hand_patch import (
    VIDEO_H, VIDEO_W,
    load_fk_model,
)
from src.core.fk import build_q, do_fk
from src.pipeline.sam2_segment import mask_to_bbox, render_mask_for_links

_TRAINING_DATA = os.path.join(MAIN_ROOT, "training_data")
SEGMENT_ROOT = os.path.join(_TRAINING_DATA, "segment")
OUTPUT_ROOT = os.path.join(_TRAINING_DATA, "hand_patch", "4s")


def compute_per_frame_bboxes(model, data, hand_caches, seg_df,
                             hand_type, margin=20):
    """Compute L/R hand bboxes for every frame in a segment.

    Returns DataFrame with columns:
      frame_idx, left_x1, left_y1, left_x2, left_y2,
                 right_x1, right_y1, right_x2, right_y2
    """
    rows = []
    for _, row in seg_df.iterrows():
        rq = np.array(row["observation.state.robot_q_current"], dtype=np.float64)
        hs = np.array(row["observation.state.hand_state"], dtype=np.float64)
        q = build_q(model, rq, hs, hand_type=hand_type)
        transforms = do_fk(model, data, q)

        entry = {"frame_idx": int(row["frame_index"])}
        for hand_name in ("left_hand", "right_hand"):
            prefix = hand_name.replace("_hand", "")
            mask = render_mask_for_links(
                hand_caches[hand_name], transforms, BEST_PARAMS,
                VIDEO_H, VIDEO_W,
            )
            bbox = mask_to_bbox(mask, margin=margin)
            if bbox is not None:
                entry[f"{prefix}_x1"] = int(bbox[0])
                entry[f"{prefix}_y1"] = int(bbox[1])
                entry[f"{prefix}_x2"] = int(bbox[2])
                entry[f"{prefix}_y2"] = int(bbox[3])
            else:
                entry[f"{prefix}_x1"] = None
                entry[f"{prefix}_y1"] = None
                entry[f"{prefix}_x2"] = None
                entry[f"{prefix}_y2"] = None
        rows.append(entry)

    return pd.DataFrame(rows)


def process_segment(task_short, ep_dir, seg_name, model, data, hand_caches,
                    hand_type, margin, resume):
    """Process one segment. Returns True if produced output."""
    parquet_path = os.path.join(SEGMENT_ROOT, task_short, ep_dir,
                                f"{seg_name}_joints.parquet")
    if not os.path.isfile(parquet_path):
        return False

    out_dir = os.path.join(OUTPUT_ROOT, task_short, ep_dir)
    out_path = os.path.join(out_dir, f"{seg_name}_hands.parquet")

    if resume and os.path.isfile(out_path):
        return False

    seg_df = pd.read_parquet(parquet_path)
    if seg_df.empty:
        return False

    bbox_df = compute_per_frame_bboxes(
        model, data, hand_caches, seg_df, hand_type, margin)

    os.makedirs(out_dir, exist_ok=True)
    bbox_df.to_parquet(out_path, index=False)
    return True


def discover_segments(task_short):
    """Scan segment directory for all (ep_dir, seg_name) pairs."""
    task_dir = os.path.join(SEGMENT_ROOT, task_short)
    if not os.path.isdir(task_dir):
        return []
    results = []
    for ep_dir in sorted(os.listdir(task_dir)):
        ep_path = os.path.join(task_dir, ep_dir)
        if not os.path.isdir(ep_path):
            continue
        for f in sorted(os.listdir(ep_path)):
            if f.endswith("_joints.parquet"):
                seg_name = f.replace("_joints.parquet", "")
                results.append((ep_dir, seg_name))
    return results


def main():
    ap = argparse.ArgumentParser(
        description="Precompute per-frame hand FK bboxes at 4s segment level")
    ap.add_argument("--task", default="all",
                    help="task short name, comma-separated, or 'all'")
    ap.add_argument("--margin", type=int, default=20,
                    help="pixel margin around FK hand bbox (default: 20)")
    ap.add_argument("--resume", action="store_true",
                    help="skip segments whose output already exists")
    args = ap.parse_args()

    if args.task == "all":
        tasks = sorted(
            d for d in os.listdir(SEGMENT_ROOT)
            if os.path.isdir(os.path.join(SEGMENT_ROOT, d))
        )
    else:
        tasks = [t.strip() for t in args.task.split(",")]

    print(f"Hand Patch 4s Precomputation")
    print(f"  segment root: {SEGMENT_ROOT}")
    print(f"  output root:  {OUTPUT_ROOT}")
    print(f"  tasks:        {', '.join(tasks)}")
    print(f"  margin:       {args.margin}px")
    print(f"  resume:       {args.resume}")

    current_hand_type = None
    model = data = hand_caches = None

    total_processed = 0
    total_skipped = 0
    t_start = time.time()

    for task_short in tasks:
        full_task_name = f"G1_WBT_{task_short}"
        hand_type = get_hand_type(full_task_name)

        if hand_type != current_hand_type:
            print(f"\nLoading FK model for hand_type={hand_type}...")
            t0 = time.time()
            model, data, _mc, hand_caches, _cam = load_fk_model(hand_type)
            print(f"  loaded in {time.time() - t0:.1f}s")
            current_hand_type = hand_type

        segments = discover_segments(task_short)
        seg_count = 0
        skip_count = 0

        for ep_dir, seg_name in segments:
            produced = process_segment(
                task_short, ep_dir, seg_name,
                model, data, hand_caches,
                hand_type, args.margin, args.resume,
            )
            if produced:
                seg_count += 1
            else:
                skip_count += 1

        total_processed += seg_count
        total_skipped += skip_count
        print(f"  {task_short}: {seg_count} processed, {skip_count} skipped")

    elapsed = time.time() - t_start
    print(f"\nDone. {total_processed} processed, {total_skipped} skipped. "
          f"{elapsed:.1f}s")


if __name__ == "__main__":
    main()
