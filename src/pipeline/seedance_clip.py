"""
Cut Seedance 4s videos into 1s and 2s clips.

Reads from training_data/seedance_direct/4s/<task>/ and writes clips to
training_data/seedance_direct/{1s,2s}/<task>/.

Usage:
  python -m src.pipeline.seedance_clip \
    --task Brainco_Collect_Plates_Into_Dishwasher
  python -m src.pipeline.seedance_clip --task all
"""

import argparse
import json
import os
import subprocess
import sys
import time

sys.stdout.reconfigure(line_buffering=True)

from src.core.config import SEEDANCE_DIRECT_DIR, ALL_TASKS

SEEDANCE_4S_DIR = os.path.join(SEEDANCE_DIRECT_DIR, "4s")
SEEDANCE_CLIP_DIR = SEEDANCE_DIRECT_DIR

FFMPEG = os.environ.get(
    "FFMPEG_BIN",
    "/home/leadtek/miniconda3/envs/flip/bin/ffmpeg",
)

# Seedance outputs 24fps; 1s = 24 frames, 2s = 48 frames
FPS = 24
CLIP_SPECS = {
    "1s": 1.0,
    "2s": 2.0,
}


def collect_videos(task: str) -> list[str]:
    """Collect all _human.mp4 files for a task, sorted."""
    task_dir = os.path.join(SEEDANCE_4S_DIR, task)
    if not os.path.isdir(task_dir):
        raise FileNotFoundError(f"not found: {task_dir}")
    videos = []
    for root, _, files in os.walk(task_dir):
        for f in files:
            if f.endswith("_human.mp4"):
                videos.append(os.path.join(root, f))
    videos.sort()
    return videos


def cut_clips(src: str, task: str, clip_dur: float, label: str,
              resume: bool) -> list[dict]:
    """Cut one source video into non-overlapping clips of clip_dur seconds."""
    # e.g. ep000/seg00_human.mp4 → ep000/seg00_clip00.mp4
    rel = os.path.relpath(src, os.path.join(SEEDANCE_4S_DIR, task))
    base = rel.replace("_human.mp4", "")  # ep000/seg00

    out_dir = os.path.join(SEEDANCE_CLIP_DIR, label, task,
                           os.path.dirname(base))
    os.makedirs(out_dir, exist_ok=True)

    seg_name = os.path.basename(base)  # seg00
    n_clips = int(4.0 / clip_dur)  # 4s source → 4 x 1s or 2 x 2s
    results = []

    for ci in range(n_clips):
        start = ci * clip_dur
        out_name = f"{seg_name}_clip{ci:02d}.mp4"
        out_path = os.path.join(out_dir, out_name)

        if resume and os.path.isfile(out_path) and os.path.getsize(out_path) > 0:
            results.append(out_path)
            continue

        subprocess.check_call([
            FFMPEG, "-y",
            "-ss", f"{start:.3f}",
            "-i", src,
            "-t", f"{clip_dur:.3f}",
            "-c:v", "libx264", "-crf", "18", "-preset", "fast",
            "-an",  # drop audio for training
            out_path,
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        results.append(out_path)

    return results


def process_task(task: str, resume: bool):
    """Process all videos for one task."""
    videos = collect_videos(task)
    if not videos:
        print(f"  no videos found for {task}")
        return {}

    print(f"\n{'='*60}")
    print(f"Task: {task}  ({len(videos)} videos)")

    stats = {}
    for label, dur in CLIP_SPECS.items():
        t0 = time.time()
        total_clips = 0
        for v in videos:
            clips = cut_clips(v, task, dur, label, resume)
            total_clips += len(clips)
        elapsed = time.time() - t0
        stats[label] = total_clips
        print(f"  {label}: {total_clips} clips ({elapsed:.1f}s)")

    return stats


def main():
    p = argparse.ArgumentParser(
        description="Cut Seedance 4s videos into 1s/2s clips")
    p.add_argument("--task", required=True,
                   help="task short name, or 'all'")
    p.add_argument("--resume", action="store_true",
                   help="skip clips that already exist")
    args = p.parse_args()

    if args.task == "all":
        tasks = [t.replace("G1_WBT_", "") for t in ALL_TASKS]
    else:
        tasks = [args.task]

    print(f"Seedance Clip Generator")
    print(f"  source: {SEEDANCE_4S_DIR}")
    print(f"  output: {SEEDANCE_CLIP_DIR}")
    print(f"  clips:  {list(CLIP_SPECS.keys())}")

    t_total = time.time()
    all_stats = {}
    for task in tasks:
        task_dir = os.path.join(SEEDANCE_4S_DIR, task)
        if not os.path.isdir(task_dir):
            continue
        all_stats[task] = process_task(task, args.resume)

    elapsed = time.time() - t_total
    print(f"\n{'='*60}")
    print(f"Done in {elapsed:.1f}s")
    for task, stats in all_stats.items():
        print(f"  {task}: " + ", ".join(f"{k}={v}" for k, v in stats.items()))


if __name__ == "__main__":
    main()
