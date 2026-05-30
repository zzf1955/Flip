"""Cut episode videos into fixed-length segments with synced joint data.

Reads LeRobot-format episodes, splits each into non-overlapping (or
overlapping) segments of a fixed duration, writes per-segment MP4 + parquet,
and produces manifest files for downstream pipeline stages.

Usage:
  python -m src.pipeline.segment_episodes --tasks all --workers 4
  python -m src.pipeline.segment_episodes --tasks G1_WBT_Inspire_Collect_Clothes_MainCamOnly --dry-run
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from multiprocessing import Pool

import av
import cv2
import numpy as np
import pandas as pd

sys.stdout.reconfigure(line_buffering=True)

from src.core.config import (
    ALL_TASKS, DATASET_ROOT, SEGMENT_DIR, get_hand_type,
)
from src.core.data import (
    load_all_episode_meta, load_data_parquet,
    open_video_writer, write_frame, close_video,
)

DEFAULT_OUTPUT_ROOT = SEGMENT_DIR


# ── Segment planning ──

def plan_task_segments(task_name, segment_frames, stride_frames):
    """Compute segment boundaries for all episodes in a task.

    Returns:
        segments_by_file: dict mapping video file_index to list of
            (episode, seg_idx, ep_frame_start, ep_frame_end,
             from_ts, to_ts, data_file_index)
        cam_key: detected camera key
    """
    meta_df, cam_key = load_all_episode_meta(task_name)
    cam_prefix = f"videos/observation.images.{cam_key}"

    segments_by_file = defaultdict(list)

    for _, row in meta_df.iterrows():
        ep = int(row["episode_index"])
        file_idx = int(row[f"{cam_prefix}/file_index"])
        from_ts = float(row[f"{cam_prefix}/from_timestamp"])
        to_ts = float(row[f"{cam_prefix}/to_timestamp"])
        data_fi = int(row.get("data/file_index", 0))
        ep_length = int(row["length"])

        seg_idx = 0
        seg_start = 0
        while seg_start + segment_frames <= ep_length:
            seg_end = seg_start + segment_frames
            segments_by_file[file_idx].append(
                (ep, seg_idx, seg_start, seg_end, from_ts, to_ts, data_fi))
            seg_idx += 1
            seg_start += stride_frames

    # Sort segments within each file by (from_ts, seg_start) for sequential decode
    for fi in segments_by_file:
        segments_by_file[fi].sort(key=lambda s: (s[4], s[2]))

    return dict(segments_by_file), cam_key


# ── Single video file processor ──

def process_video_file(task_name, file_index, segments, cam_key,
                       output_base, task_short, codec, crf, resume, fps=30):
    """Decode one source MP4 and extract all planned segments.

    Returns list of completed segment metadata dicts.
    """
    video_path = os.path.join(
        DATASET_ROOT, task_name, "videos",
        f"observation.images.{cam_key}", "chunk-000",
        f"file-{file_index:03d}.mp4")

    if not os.path.isfile(video_path):
        print(f"  WARN: video not found: {video_path}")
        return []

    # Pre-load data parquet files needed by segments in this video file
    data_fi_set = set(s[6] for s in segments)
    data_dfs = {}
    for dfi in data_fi_set:
        data_dfs[dfi] = load_data_parquet(task_name, dfi)

    # Build episode ranges and segment lookup
    # episode_info[ep] = (from_ts, to_ts, data_fi, [(seg_idx, start, end)])
    episode_info = {}
    for (ep, seg_idx, seg_start, seg_end, from_ts, to_ts, data_fi) in segments:
        if ep not in episode_info:
            episode_info[ep] = (from_ts, to_ts, data_fi, [])
        episode_info[ep][3].append((seg_idx, seg_start, seg_end))

    # Sort episode list by from_ts for efficient matching
    sorted_episodes = sorted(episode_info.items(), key=lambda x: x[1][0])

    # Prepare output paths and check resume
    seg_output_info = {}  # (ep, seg_idx) -> output_path
    segments_to_write = set()
    for (ep, seg_idx, seg_start, seg_end, from_ts, to_ts, data_fi) in segments:
        ep_dir = os.path.join(output_base, task_short, f"ep{ep:03d}")
        video_out = os.path.join(ep_dir, f"seg{seg_idx:02d}_video.mp4")
        seg_output_info[(ep, seg_idx)] = (video_out, ep_dir)
        if resume and os.path.isfile(video_out) and os.path.getsize(video_out) > 0:
            continue
        segments_to_write.add((ep, seg_idx))

    if not segments_to_write:
        # All segments already done
        results = []
        for (ep, seg_idx, seg_start, seg_end, from_ts, to_ts, data_fi) in segments:
            video_out, _ = seg_output_info[(ep, seg_idx)]
            results.append({
                "episode": ep,
                "segment_index": seg_idx,
                "video": os.path.relpath(video_out, os.path.join(output_base, task_short)),
                "joints": os.path.relpath(
                    video_out.replace("_video.mp4", "_joints.parquet"),
                    os.path.join(output_base, task_short)),
                "ep_frame_start": seg_start,
                "ep_frame_end": seg_end,
            })
        return results

    # Open source video
    container_in = av.open(video_path)
    stream_in = container_in.streams.video[0]

    # Sequential frame counter per episode (avoids PTS rounding issues)
    ep_frame_counters = {}  # ep_idx -> sequential count of frames seen

    # Active segment writers: (ep, seg_idx) -> (writer_tuple, frame_count)
    active_writers = {}
    completed = []

    # Track which episodes are fully done (all segments extracted)
    ep_done = set()

    for av_frame in container_in.decode(stream_in):
        pts_sec = float(av_frame.pts * stream_in.time_base)

        # Find matching episode by PTS range
        matched_ep_idx = None
        for ep_idx, (from_ts, to_ts, data_fi, seg_list) in sorted_episodes:
            if pts_sec < from_ts - 0.5 / fps:
                break
            if from_ts - 0.5 / fps <= pts_sec < to_ts + 0.5 / fps:
                matched_ep_idx = ep_idx
                break

        if matched_ep_idx is None:
            continue
        if matched_ep_idx in ep_done:
            continue

        # Sequential frame index within this episode
        if matched_ep_idx not in ep_frame_counters:
            ep_frame_counters[matched_ep_idx] = 0
        ep_fi = ep_frame_counters[matched_ep_idx]
        ep_frame_counters[matched_ep_idx] += 1

        from_ts, to_ts, data_fi, seg_list = episode_info[matched_ep_idx]

        # Find matching segment for this frame index
        for (seg_idx, seg_start, seg_end) in seg_list:
            if seg_start <= ep_fi < seg_end:
                key = (matched_ep_idx, seg_idx)
                if key not in segments_to_write:
                    break

                img = av_frame.to_ndarray(format='bgr24')
                h, w = img.shape[:2]

                if key not in active_writers:
                    video_out, ep_dir = seg_output_info[key]
                    os.makedirs(ep_dir, exist_ok=True)
                    writer = open_video_writer(video_out, w, h, fps=fps)
                    active_writers[key] = (writer, 0, seg_start, seg_end, data_fi)

                writer_tuple, count, s_start, s_end, d_fi = active_writers[key]
                write_frame(*writer_tuple, img)
                count += 1
                active_writers[key] = (writer_tuple, count, s_start, s_end, d_fi)

                if count >= (s_end - s_start):
                    close_video(*writer_tuple)
                    completed.append(key)
                    del active_writers[key]
                break

        # Check if all segments for this episode are done
        all_done = all(
            (matched_ep_idx, si) not in segments_to_write or
            (matched_ep_idx, si) in set(completed)
            for si, _, _ in seg_list
        )
        if all_done:
            ep_done.add(matched_ep_idx)

    # Close any still-open writers (incomplete segments)
    for key, (writer_tuple, count, s_start, s_end, d_fi) in active_writers.items():
        close_video(*writer_tuple)
        if count < (s_end - s_start):
            video_out, _ = seg_output_info[key]
            if os.path.isfile(video_out):
                os.remove(video_out)
            print(f"  WARN: incomplete segment {key}, got {count}/{s_end-s_start} frames, removed")

    container_in.close()

    # Write parquet for completed segments
    for key in completed:
        ep_idx, seg_idx = key
        from_ts_ep, to_ts_ep, data_fi, seg_list = episode_info[ep_idx]

        # Find seg_start/seg_end for this segment
        seg_start = seg_end = None
        for (si, ss, se) in seg_list:
            if si == seg_idx:
                seg_start, seg_end = ss, se
                break

        video_out, _ = seg_output_info[key]
        parquet_out = video_out.replace("_video.mp4", "_joints.parquet")

        # Slice parquet data
        df = data_dfs[data_fi]
        ep_df = df[df["episode_index"] == ep_idx]
        seg_df = ep_df[(ep_df["frame_index"] >= seg_start) &
                       (ep_df["frame_index"] < seg_end)]

        # Keep only essential columns
        cols_to_keep = ["episode_index", "frame_index"]
        for c in ep_df.columns:
            if "robot_q_current" in c or "hand_state" in c:
                cols_to_keep.append(c)
        cols_to_keep = [c for c in cols_to_keep if c in seg_df.columns]
        seg_df = seg_df[cols_to_keep].reset_index(drop=True)
        seg_df.to_parquet(parquet_out, index=False)

    # Build results for all segments in this file (including resumed ones)
    results = []
    task_dir = os.path.join(output_base, task_short)
    for (ep, seg_idx, seg_start, seg_end, from_ts, to_ts, data_fi) in segments:
        video_out, _ = seg_output_info[(ep, seg_idx)]
        if not os.path.isfile(video_out):
            continue
        results.append({
            "episode": ep,
            "segment_index": seg_idx,
            "video": os.path.relpath(video_out, task_dir),
            "joints": os.path.relpath(
                video_out.replace("_video.mp4", "_joints.parquet"), task_dir),
            "ep_frame_start": seg_start,
            "ep_frame_end": seg_end,
        })

    return results


# ── Wrapper for multiprocessing (top-level function) ──

def _process_video_file_wrapper(args):
    """Unpack args tuple for Pool.map."""
    return process_video_file(*args)


# ── Task orchestrator ──

def process_task(task_name, args):
    """Process all episodes for one task: plan segments, extract, write manifest."""
    segment_frames = int(args.segment_duration * 30)
    stride_frames = int(args.stride * 30)
    task_short = task_name.replace("G1_WBT_", "")
    hand_type = get_hand_type(task_name)

    print(f"\n{'='*60}")
    print(f"Task: {task_name}")
    print(f"  hand_type={hand_type}, segment={args.segment_duration}s, stride={args.stride}s")

    segments_by_file, cam_key = plan_task_segments(
        task_name, segment_frames, stride_frames)

    total_segments = sum(len(v) for v in segments_by_file.values())
    n_episodes = len(set(s[0] for segs in segments_by_file.values() for s in segs))

    print(f"  {n_episodes} episodes -> {total_segments} segments across "
          f"{len(segments_by_file)} video files")

    if args.dry_run:
        return total_segments

    t_start = time.time()

    # Build work items for pool
    work_items = [
        (task_name, fi, segs, cam_key,
         args.output_root, task_short, args.codec, args.crf, args.resume)
        for fi, segs in sorted(segments_by_file.items())
    ]

    # Process
    all_results = []
    if args.workers > 1 and len(work_items) > 1:
        with Pool(processes=min(args.workers, len(work_items))) as pool:
            for i, result in enumerate(
                    pool.imap_unordered(_process_video_file_wrapper, work_items)):
                all_results.extend(result)
                done_segs = len(all_results)
                print(f"  [{i+1}/{len(work_items)} files] "
                      f"{done_segs}/{total_segments} segments")
    else:
        for i, item in enumerate(work_items):
            result = process_video_file(*item)
            all_results.extend(result)
            print(f"  [{i+1}/{len(work_items)} files] "
                  f"{len(all_results)}/{total_segments} segments")

    elapsed = time.time() - t_start
    print(f"  Done: {len(all_results)} segments in {elapsed:.1f}s")

    # Sort results by (episode, segment_index)
    all_results.sort(key=lambda r: (r["episode"], r["segment_index"]))

    # Write per-task manifest
    task_dir = os.path.join(args.output_root, task_short)
    os.makedirs(task_dir, exist_ok=True)
    manifest = {
        "task": task_name,
        "task_short": task_short,
        "hand_type": hand_type,
        "segment_duration_sec": args.segment_duration,
        "stride_sec": args.stride,
        "segment_frames": segment_frames,
        "fps": 30,
        "num_episodes": n_episodes,
        "num_segments": len(all_results),
        "segments": all_results,
    }
    manifest_path = os.path.join(task_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"  Manifest: {manifest_path}")

    return len(all_results)


# ── CLI ──

def parse_args():
    p = argparse.ArgumentParser(
        description="Cut episode videos into fixed-length segments")
    p.add_argument("--tasks", nargs="+", default=["all"],
                   help="Task names, or 'all' for all tasks")
    p.add_argument("--segment-duration", type=float, default=4.0,
                   help="Segment length in seconds (default: 4.0)")
    p.add_argument("--stride", type=float, default=None,
                   help="Stride in seconds (default: same as segment-duration)")
    p.add_argument("--output-root", type=str, default=DEFAULT_OUTPUT_ROOT,
                   help="Output root directory")
    p.add_argument("--workers", type=int, default=4,
                   help="Number of parallel workers (default: 4)")
    p.add_argument("--codec", type=str, default="libx264")
    p.add_argument("--crf", type=int, default=18)
    p.add_argument("--resume", action="store_true",
                   help="Skip segments whose output already exists")
    p.add_argument("--dry-run", action="store_true",
                   help="Print segment plan without writing files")
    args = p.parse_args()

    if args.stride is None:
        args.stride = args.segment_duration

    if "all" in args.tasks:
        args.tasks = list(ALL_TASKS)

    return args


def main():
    args = parse_args()

    print(f"Segment Episodes Pipeline")
    print(f"  Tasks: {len(args.tasks)}")
    print(f"  Segment: {args.segment_duration}s, stride: {args.stride}s")
    print(f"  Output: {args.output_root}")
    print(f"  Workers: {args.workers}")
    if args.dry_run:
        print(f"  ** DRY RUN **")

    os.makedirs(args.output_root, exist_ok=True)

    grand_total = 0
    task_stats = {}
    t_total = time.time()

    for task in args.tasks:
        n_segs = process_task(task, args)
        grand_total += n_segs
        task_stats[task] = n_segs

    # Write global manifest
    if not args.dry_run:
        global_manifest = {
            "version": 1,
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "segment_duration_sec": args.segment_duration,
            "stride_sec": args.stride,
            "segment_frames": int(args.segment_duration * 30),
            "fps": 30,
            "resolution": [640, 480],
            "total_segments": grand_total,
            "tasks": {},
        }
        for task in args.tasks:
            task_short = task.replace("G1_WBT_", "")
            global_manifest["tasks"][task_short] = {
                "full_name": task,
                "hand_type": get_hand_type(task),
                "num_segments": task_stats[task],
            }
        gm_path = os.path.join(args.output_root, "manifest.json")
        with open(gm_path, "w") as f:
            json.dump(global_manifest, f, indent=2, ensure_ascii=False)
        print(f"\nGlobal manifest: {gm_path}")

    elapsed = time.time() - t_total
    print(f"\n{'='*60}")
    print(f"Total: {grand_total} segments, {elapsed:.1f}s")
    if args.dry_run:
        print("(dry run, no files written)")


if __name__ == "__main__":
    main()
