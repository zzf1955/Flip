"""Batch downsample videos from 30fps to 16fps.

Creates videos_16fps/ alongside videos/ for each task, only for the
pipeline-relevant camera (head_stereo_left or cam_0).

Usage:
  python scripts/downsample_videos.py
  python scripts/downsample_videos.py --workers 8
  python scripts/downsample_videos.py --dry-run
"""

import os
import sys
import argparse
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
from config import DATASET_ROOT, ALL_TASKS

FFMPEG = os.path.join(os.path.dirname(sys.executable), "ffmpeg")
TARGET_FPS = 16

# Camera key per task
CAMERA_KEYS = {
    "G1_WBT_Inspire_Collect_Clothes_MainCamOnly": "observation.images.cam_0",
    "G1_WBT_Inspire_Pickup_Pillow_MainCamOnly": "observation.images.cam_0",
    "G1_WBT_Inspire_Put_Clothes_into_Washing_Machine_MainCamOnly": "observation.images.cam_0",
}
DEFAULT_CAMERA = "observation.images.head_stereo_left"


def get_camera_key(task):
    return CAMERA_KEYS.get(task, DEFAULT_CAMERA)


def transcode_one(src, dst):
    """Transcode a single video to target fps. Returns (src, dst, ok, msg)."""
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    cmd = [
        FFMPEG, "-y", "-i", src,
        "-r", str(TARGET_FPS),
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-pix_fmt", "yuv420p",
        "-an",  # no audio
        dst,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=300)
        if result.returncode == 0:
            return (src, dst, True, "ok")
        else:
            return (src, dst, False, result.stderr.decode()[-200:])
    except subprocess.TimeoutExpired:
        return (src, dst, False, "timeout")


def collect_jobs():
    """Collect all (src, dst) pairs."""
    jobs = []
    for task in ALL_TASKS:
        task_dir = os.path.join(DATASET_ROOT, task)
        cam = get_camera_key(task)

        src_dir = os.path.join(task_dir, "videos", cam, "chunk-000")
        dst_dir = os.path.join(task_dir, "videos_16fps", cam, "chunk-000")

        if not os.path.isdir(src_dir):
            print(f"SKIP {task}: {src_dir} not found")
            continue

        for f in sorted(os.listdir(src_dir)):
            if not f.endswith(".mp4"):
                continue
            src = os.path.join(src_dir, f)
            dst = os.path.join(dst_dir, f)
            if os.path.exists(dst):
                continue  # already done
            jobs.append((src, dst))

    return jobs


def main():
    parser = argparse.ArgumentParser(description="Downsample videos to 16fps")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    jobs = collect_jobs()
    print(f"Found {len(jobs)} videos to transcode ({args.workers} workers)")

    if args.dry_run:
        for src, dst in jobs:
            print(f"  {src} -> {dst}")
        return

    if not jobs:
        print("Nothing to do.")
        return

    t0 = time.time()
    done = 0
    failed = 0

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(transcode_one, s, d): (s, d) for s, d in jobs}
        for fut in as_completed(futures):
            src, dst, ok, msg = fut.result()
            done += 1
            if ok:
                name = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(src)))))
                print(f"  [{done}/{len(jobs)}] {name}/{os.path.basename(src)}")
            else:
                failed += 1
                print(f"  [{done}/{len(jobs)}] FAIL {src}: {msg}")

    elapsed = time.time() - t0
    print(f"\nDone: {done - failed}/{len(jobs)} ok, {failed} failed, {elapsed:.0f}s")


if __name__ == "__main__":
    main()
