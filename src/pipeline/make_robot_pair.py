"""
Generate robot→robot reconstruction pairs from segment data.

For each 4s segment, creates adjacent 1s clip pairs (context, target):
  clip_0→clip_1, clip_1→clip_2, clip_2→clip_3

Output format is identical to make_pair.py so mitty_cache.py can be reused
directly without modification.

Usage:
  python -m src.pipeline.make_robot_pair --task all --max-segments 500 --clean

  python -m src.pipeline.make_robot_pair --task all --max-segments 50 \
    --per-task-eval 3 --workers 8
"""

import argparse
import csv
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.stdout.reconfigure(line_buffering=True)

from src.core.config import ALL_TASKS, SEGMENT_DIR, ROBOT_PAIR_DIR

FFMPEG = os.environ.get(
    "FFMPEG_BIN",
    "/home/leadtek/miniconda3/envs/flip/bin/ffmpeg",
)

TARGET_FPS = 16
NUM_FRAMES = 17
PROMPT = "A first-person view robot arm performing household tasks flip_v2v"
SEGMENT_DURATION = 4.0
CLIP_DURATION = 1.0
CLIPS_PER_SEG = int(SEGMENT_DURATION / CLIP_DURATION)  # 4


def _ffmpeg(args: list[str]):
    subprocess.check_call(
        [FFMPEG, "-y"] + args,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def cut_clip(segment_video: str, out_path: str, start: float):
    _ffmpeg([
        "-ss", f"{start:.3f}", "-i", segment_video,
        "-t", f"{CLIP_DURATION + 0.5:.3f}",
        "-r", str(TARGET_FPS),
        "-frames:v", str(NUM_FRAMES),
        "-c:v", "libx264", "-crf", "18", "-preset", "fast",
        "-an", out_path,
    ])


# ── segment discovery ────────────────────────────────────────────────

def find_segments(tasks: list[str]) -> list[dict]:
    segments = []
    for task in tasks:
        task_dir = os.path.join(SEGMENT_DIR, task)
        if not os.path.isdir(task_dir):
            continue
        for root, _, files in os.walk(task_dir):
            for fname in sorted(files):
                m = re.match(r"(seg\d+)_video\.mp4$", fname)
                if not m:
                    continue
                seg = m.group(1)
                ep_dir = os.path.relpath(root, task_dir)
                segments.append({
                    "task": task,
                    "episode": ep_dir,
                    "seg": seg,
                    "path": os.path.join(root, fname),
                })
    segments.sort(key=lambda s: f"{s['task']}/{s['episode']}/{s['seg']}")
    return segments


def make_pairs_from_segments(segments: list[dict]) -> list[dict]:
    pairs = []
    for s in segments:
        for ci in range(CLIPS_PER_SEG - 1):
            pairs.append({
                "task": s["task"],
                "episode": s["episode"],
                "seg": s["seg"],
                "seg_path": s["path"],
                "ctx_clip": ci,
                "tgt_clip": ci + 1,
                "ctx_start": ci * CLIP_DURATION,
                "tgt_start": (ci + 1) * CLIP_DURATION,
                "source_id": f"{s['task']}/{s['episode']}/{s['seg']}_c{ci}t{ci+1}",
            })
    return pairs


# ── splitting ────────────────────────────────────────────────────────

def split_pairs(all_pairs: list[dict], per_task_eval: int,
                seed: int) -> dict[str, list[dict]]:
    by_task: dict[str, list[dict]] = {}
    for p in all_pairs:
        by_task.setdefault(p["task"], []).append(p)

    seg_to_pairs: dict[str, list[dict]] = {}
    for p in all_pairs:
        key = f"{p['task']}/{p['episode']}/{p['seg']}"
        seg_to_pairs.setdefault(key, []).append(p)

    splits: dict[str, list[dict]] = {"train": [], "eval": []}

    for task in sorted(by_task.keys()):
        task_segs = sorted({f"{p['task']}/{p['episode']}/{p['seg']}"
                            for p in by_task[task]})
        rng = random.Random(f"{seed}:{task}")
        rng.shuffle(task_segs)
        n_eval = min(per_task_eval, max(0, len(task_segs) - 1))
        eval_segs = set(task_segs[:n_eval])
        for seg_key in task_segs:
            bucket = "eval" if seg_key in eval_segs else "train"
            splits[bucket].extend(seg_to_pairs[seg_key])

    for name in splits:
        splits[name].sort(key=lambda p: p["source_id"])
    return splits


# ── process one pair ─────────────────────────────────────────────────

def process_pair(p: dict) -> dict:
    os.makedirs(os.path.dirname(p["out_tgt"]), exist_ok=True)
    os.makedirs(os.path.dirname(p["out_ctx"]), exist_ok=True)

    if not os.path.isfile(p["out_tgt"]):
        cut_clip(p["seg_path"], p["out_tgt"], p["tgt_start"])
    if not os.path.isfile(p["out_ctx"]):
        cut_clip(p["seg_path"], p["out_ctx"], p["ctx_start"])

    return {
        "video": p["rel_tgt"],
        "prompt": PROMPT,
        "control_video": p["rel_ctx"],
    }


# ── main ─────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Generate robot→robot reconstruction pairs from segments")
    ap.add_argument("--task", required=True,
                    help="task short name, comma-separated, or 'all'")
    ap.add_argument("--max-segments", type=int, default=0,
                    help="max segments per task (0 = unlimited)")
    ap.add_argument("--per-task-eval", type=int, default=5,
                    help="segments per task reserved for eval")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--clean", action="store_true",
                    help="remove existing robot_pair/1s/ before writing")
    args = ap.parse_args()

    key = args.task.lower()
    task_groups = {"all": None, "inspire": "Inspire_", "brainco": "Brainco_"}
    if key in task_groups:
        prefix = task_groups[key]
        short = [t.replace("G1_WBT_", "") for t in ALL_TASKS]
        tasks = [t for t in short if prefix is None or t.startswith(prefix)]
    else:
        tasks = [t.strip() for t in args.task.split(",")]

    sec_dir = os.path.join(ROBOT_PAIR_DIR, "1s")
    if args.clean and os.path.isdir(sec_dir):
        shutil.rmtree(sec_dir)

    print("Make Robot Pair (adjacent clips)")
    print(f"  tasks:         {tasks}")
    print(f"  max-segments:  {args.max_segments or 'unlimited'}")
    print(f"  per-task-eval: {args.per_task_eval}")
    print(f"  seed:          {args.seed}")
    print(f"  workers:       {args.workers}")

    all_segs = find_segments(tasks)
    print(f"  found {len(all_segs)} segments across {len(tasks)} tasks")

    if args.max_segments > 0:
        by_task: dict[str, list[dict]] = {}
        for s in all_segs:
            by_task.setdefault(s["task"], []).append(s)
        sampled = []
        for task in sorted(by_task.keys()):
            segs = by_task[task]
            rng = random.Random(f"{args.seed}:sample:{task}")
            rng.shuffle(segs)
            sampled.extend(segs[:args.max_segments])
        all_segs = sampled
        print(f"  sampled → {len(all_segs)} segments")

    all_pairs = make_pairs_from_segments(all_segs)
    if not all_pairs:
        print("No pairs generated, exiting")
        return

    splits = split_pairs(all_pairs, args.per_task_eval, args.seed)
    print(f"  {len(all_pairs)} pairs → "
          f"train={len(splits['train'])} eval={len(splits['eval'])}")

    source_map: dict[str, dict] = {}
    to_process: list[dict] = []

    for split_name, split_pairs_list in splits.items():
        if not split_pairs_list:
            continue
        split_dir = os.path.join(sec_dir, split_name)
        video_dir = os.path.join(split_dir, "video")
        ctrl_dir = os.path.join(split_dir, "control_video")
        os.makedirs(video_dir, exist_ok=True)
        os.makedirs(ctrl_dir, exist_ok=True)

        for idx, p in enumerate(split_pairs_list):
            name = f"pair_{idx:04d}.mp4"
            p["split"] = split_name
            p["out_tgt"] = os.path.join(video_dir, name)
            p["out_ctx"] = os.path.join(ctrl_dir, name)
            p["rel_tgt"] = f"video/{name}"
            p["rel_ctx"] = f"control_video/{name}"
            source_map[f"{split_name}/{name}"] = {
                "task": p["task"],
                "episode": p["episode"],
                "seg": p["seg"],
                "ctx_clip": p["ctx_clip"],
                "tgt_clip": p["tgt_clip"],
                "source_id": p["source_id"],
                "seg_path": p["seg_path"],
            }
            to_process.append(p)

    t0 = time.time()
    split_meta: dict[str, list[dict]] = {k: [] for k in splits}

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(process_pair, p): p for p in to_process
        }
        done = 0
        total = len(to_process)
        for fut in as_completed(futures):
            p = futures[fut]
            meta = fut.result()
            split_meta[p["split"]].append(meta)
            done += 1
            if done % 50 == 0 or done == total:
                print(f"  {done}/{total}", flush=True)

    print(f"  clips done in {time.time() - t0:.1f}s")

    for split_name, metas in split_meta.items():
        if not metas:
            continue
        metas.sort(key=lambda m: m["video"])
        csv_path = os.path.join(sec_dir, split_name, "metadata.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["video", "prompt", "control_video"])
            writer.writeheader()
            writer.writerows(metas)
        print(f"  metadata: {csv_path} ({len(metas)} rows)")

    map_path = os.path.join(sec_dir, "source_map.json")
    with open(map_path, "w") as f:
        json.dump(source_map, f, indent=2, sort_keys=True)
    print(f"  source_map: {map_path} ({len(source_map)} entries)")

    print(f"\nDone")


if __name__ == "__main__":
    main()
