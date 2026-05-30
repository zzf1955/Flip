"""
Post-process Seedance direct 4s videos into augmented 1s clips.

Reads from training_data/seedance_direct/4s/<task>/ and writes normal plus
horizontal-flip sliding-window clips to training_data/seedance_direct/1s/<task>/.
The original 4s videos are never modified.

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

SOURCE_DURATION = 4.0
CLIP_LABEL = "1s"
CLIP_DURATION = 1.0
CLIP_STRIDE = 0.5
CLIP_WINDOWS = [round(i * CLIP_STRIDE, 3)
                for i in range(int((SOURCE_DURATION - CLIP_DURATION) /
                                   CLIP_STRIDE) + 1)]
MANIFEST_NAME = "manifest.jsonl"


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


def cut_clips(src: str, task: str, resume: bool) -> list[dict]:
    """Cut one source video into normal and horizontally flipped 1s clips."""
    # e.g. ep000/seg00_human.mp4 → ep000/seg00_clip00.mp4
    rel = os.path.relpath(src, os.path.join(SEEDANCE_4S_DIR, task))
    base = rel.replace("_human.mp4", "")  # ep000/seg00

    out_dir = os.path.join(SEEDANCE_CLIP_DIR, CLIP_LABEL, task,
                           os.path.dirname(base))
    os.makedirs(out_dir, exist_ok=True)

    seg_name = os.path.basename(base)  # seg00
    results = []

    for augment_idx, augment in enumerate(("normal", "hflip")):
        for window_idx, start in enumerate(CLIP_WINDOWS):
            clip_idx = augment_idx * len(CLIP_WINDOWS) + window_idx
            out_name = f"{seg_name}_clip{clip_idx:02d}.mp4"
            out_path = os.path.join(out_dir, out_name)

            if not (resume and os.path.isfile(out_path) and
                    os.path.getsize(out_path) > 0):
                command = [
                    FFMPEG, "-y",
                    "-ss", f"{start:.3f}",
                    "-i", src,
                    "-t", f"{CLIP_DURATION:.3f}",
                ]
                if augment == "hflip":
                    command.extend(["-vf", "hflip"])
                command.extend([
                    "-c:v", "libx264", "-crf", "18", "-preset", "fast",
                    "-an",  # drop audio for training
                    out_path,
                ])
                subprocess.check_call(
                    command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            results.append({
                "task": task,
                "episode": os.path.dirname(base),
                "seg": seg_name,
                "clip_idx": clip_idx,
                "window_idx": window_idx,
                "augment": augment,
                "source_4s": os.path.relpath(src, SEEDANCE_4S_DIR),
                "clip": os.path.relpath(out_path, SEEDANCE_CLIP_DIR),
                "start": start,
                "duration": CLIP_DURATION,
                "stride": CLIP_STRIDE,
            })

    return results


def process_task(task: str, resume: bool):
    """Process all videos for one task."""
    videos = collect_videos(task)
    if not videos:
        print(f"  no videos found for {task}")
        return {}

    print(f"\n{'='*60}")
    print(f"Task: {task}  ({len(videos)} videos)")

    t0 = time.time()
    records = []
    for v in videos:
        records.extend(cut_clips(v, task, resume))

    manifest_path = os.path.join(SEEDANCE_CLIP_DIR, CLIP_LABEL, task,
                                 MANIFEST_NAME)
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    with open(manifest_path, "w") as f:
        for record in sorted(records, key=lambda r: r["clip"]):
            f.write(json.dumps(record, sort_keys=True) + "\n")

    elapsed = time.time() - t0
    stats = {CLIP_LABEL: len(records)}
    print(f"  {CLIP_LABEL}: {len(records)} clips ({elapsed:.1f}s)")
    print(f"  manifest: {manifest_path}")

    return stats


def main():
    p = argparse.ArgumentParser(
        description="Post-process Seedance direct 4s videos into 1s clips")
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
    print(f"  clip:   {CLIP_LABEL}, duration={CLIP_DURATION}s, stride={CLIP_STRIDE}s")
    print(f"  aug:    normal,hflip ({len(CLIP_WINDOWS)} windows each)")

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
