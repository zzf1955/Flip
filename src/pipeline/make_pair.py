"""
Generate training pairs and comparison videos, split into train / eval / ood_eval.

Matches segment (robot) videos with human videos (seedance_direct, overlay, or
seedance_advance), resamples both to 16fps, and writes three independent
subdirectories per duration:
  pair/<second>/train/          <- majority of clips for training
  pair/<second>/eval/           <- in-task eval, N clips per non-OOD task
  pair/<second>/ood_eval/       <- OOD eval, all clips from --ood-tasks

Each split gets its own pair_NNNN.mp4 numbering and metadata.csv. A top-level
pair/<second>/source_map.json records the mapping back to (task, episode, seg).

Usage:
  python -m src.pipeline.make_pair --task all --second 1s --clean

  python -m src.pipeline.make_pair --task all --second 1s \
    --human-source seedance_advance --hand-patch --hand-weight 3.0 \
    --ood-tasks Inspire_Pickup_Pillow_MainCamOnly --per-task-eval 1
"""

import argparse
import csv
import glob
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import torch

sys.stdout.reconfigure(line_buffering=True)

from src.core.config import (
    ALL_TASKS, SEGMENT_DIR, SEEDANCE_DIRECT_DIR, OVERLAY_DIR,
    SEEDANCE_ADVANCE_DIR, HAND_PATCH_4S_DIR,
    PAIR_DIR, TRAINING_DATA_ROOT,
)

FFMPEG = os.environ.get(
    "FFMPEG_BIN",
    "/home/leadtek/miniconda3/envs/flip/bin/ffmpeg",
)

TARGET_FPS = 16
COMPARE_DIR = os.path.join(TRAINING_DATA_ROOT, "compare")
PROMPT = "A first-person view robot arm performing household tasks flip_v2v"

# 4k+1 frame counts at 16fps for each clip duration
FRAMES_4K1 = {"1s": 17, "2s": 33, "4s": 65}

HUMAN_SOURCE_MAP = {
    "seedance_direct": SEEDANCE_DIRECT_DIR,
    "overlay": OVERLAY_DIR,
    "seedance_advance": SEEDANCE_ADVANCE_DIR,
}

SEGMENT_FPS = 30

# ── helpers ────────────────────────────────────────────────────────────

def _ffmpeg(args: list[str]):
    """Run ffmpeg silently."""
    subprocess.check_call(
        [FFMPEG, "-y"] + args,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def make_robot_clip(segment_video: str, out_path: str,
                    start: float, duration: float,
                    num_frames: int | None = None):
    """Cut a segment from the robot video and resample to TARGET_FPS."""
    args = [
        "-ss", f"{start:.3f}", "-i", segment_video,
        "-t", f"{duration + 0.5:.3f}",  # generous cut, rely on -frames:v
        "-r", str(TARGET_FPS),
    ]
    if num_frames is not None:
        args += ["-frames:v", str(num_frames)]
    args += ["-c:v", "libx264", "-crf", "18", "-preset", "fast",
             "-an", out_path]
    _ffmpeg(args)


def make_human_clip(human_video: str, out_path: str,
                    start: float = 0.0, duration: float | None = None,
                    num_frames: int | None = None):
    """Cut and resample human video to TARGET_FPS."""
    args = ["-ss", f"{start:.3f}", "-i", human_video]
    if duration is not None:
        args += ["-t", f"{duration + 0.5:.3f}"]
    args += ["-r", str(TARGET_FPS)]
    if num_frames is not None:
        args += ["-frames:v", str(num_frames)]
    args += ["-c:v", "libx264", "-crf", "18", "-preset", "fast",
             "-an", out_path]
    _ffmpeg(args)


def make_compare(robot_path: str, human_path: str, out_path: str):
    """Side-by-side comparison: left=robot, right=human."""
    _ffmpeg([
        "-i", robot_path, "-i", human_path,
        "-filter_complex", "[0:v][1:v]hstack=inputs=2",
        "-c:v", "libx264", "-crf", "18", "-preset", "fast",
        out_path,
    ])


# ── hand patch helpers ─────────────────────────────────────────────────

def load_hand_patch_data(task: str, ep_dir: str, seg: str) -> pd.DataFrame | None:
    """Load precomputed per-frame hand bboxes from hand_patch/4s/."""
    path = os.path.join(HAND_PATCH_4S_DIR, task, ep_dir, f"{seg}_hands.parquet")
    if not os.path.isfile(path):
        return None
    return pd.read_parquet(path)


def compute_clip_weight_map(hand_df: pd.DataFrame,
                            clip_start_frame: int, clip_frames: int,
                            hand_weight: float) -> torch.Tensor | None:
    """Slice per-frame bboxes for clip window, union, convert to latent weight map."""
    from src.pipeline.hand_patch import (
        LATENT_F, LATENT_H, LATENT_W, VAE_SPATIAL,
        pixel_bbox_to_latent, build_weight_map,
    )

    clip_df = hand_df[
        (hand_df["frame_idx"] >= clip_start_frame) &
        (hand_df["frame_idx"] < clip_start_frame + clip_frames)
    ]
    if clip_df.empty:
        return None

    bboxes_pixel = {}
    for hand in ("left", "right"):
        cols = [f"{hand}_x1", f"{hand}_y1", f"{hand}_x2", f"{hand}_y2"]
        valid = clip_df[cols].dropna()
        if valid.empty:
            bboxes_pixel[f"{hand}_hand"] = None
        else:
            bboxes_pixel[f"{hand}_hand"] = [
                int(valid[f"{hand}_x1"].min()),
                int(valid[f"{hand}_y1"].min()),
                int(valid[f"{hand}_x2"].max()),
                int(valid[f"{hand}_y2"].max()),
            ]

    bboxes_latent = {
        k: pixel_bbox_to_latent(v) for k, v in bboxes_pixel.items()
    }
    return build_weight_map(bboxes_latent, hand_weight)


# ── matching ───────────────────────────────────────────────────────────

def collect_pairs(task: str, second: str, human_root: str,
                  source_second: str | None = None) -> list[dict]:
    """Find all (human, robot) pairs for a given task and clip duration.

    When *source_second* differs from *second* (e.g. source_second="4s",
    second="1s"), walks the source_second directory and cuts each source
    video into floor(source_dur/target_dur) non-overlapping clips.

    Returns list of dicts with keys:
      task, human_src, robot_src, clip_start, clip_dur, source_id, episode, seg, clip_idx
    Source_id is used for sorting/dedup; flat output paths are assigned later.
    """
    src_sec = source_second or second
    human_dir = os.path.join(human_root, src_sec, task)
    if not os.path.isdir(human_dir):
        return []

    dur_sec = {"1s": 1.0, "2s": 2.0, "4s": 4.0}[second]
    src_dur = {"1s": 1.0, "2s": 2.0, "4s": 4.0}[src_sec]
    cross_cut = (src_sec != second)
    clips_per_seg = int(src_dur / dur_sec) if cross_cut else 1
    pairs = []

    for root, _, files in os.walk(human_dir):
        for fname in sorted(files):
            if not fname.endswith(".mp4"):
                continue

            human_src = os.path.join(root, fname)
            rel_from_task = os.path.relpath(root, human_dir)
            ep_dir = rel_from_task

            if cross_cut:
                m = re.match(r"(seg\d+)_human\.mp4$", fname)
                if not m:
                    continue
                seg = m.group(1)
                robot_src = os.path.join(SEGMENT_DIR, task, ep_dir,
                                         f"{seg}_video.mp4")
                if not os.path.isfile(robot_src):
                    continue
                for ci in range(clips_per_seg):
                    pairs.append({
                        "task": task,
                        "episode": ep_dir,
                        "seg": seg,
                        "clip_idx": ci,
                        "human_src": human_src,
                        "robot_src": robot_src,
                        "clip_start": ci * dur_sec,
                        "clip_dur": dur_sec,
                        "source_id": f"{task}/{ep_dir}/{seg}_clip{ci:02d}",
                    })
            elif src_sec == "4s":
                m = re.match(r"(seg\d+)_human\.mp4$", fname)
                if not m:
                    continue
                seg = m.group(1)
                robot_src = os.path.join(SEGMENT_DIR, task, ep_dir,
                                         f"{seg}_video.mp4")
                if not os.path.isfile(robot_src):
                    continue
                pairs.append({
                    "task": task,
                    "episode": ep_dir,
                    "seg": seg,
                    "clip_idx": None,
                    "human_src": human_src,
                    "robot_src": robot_src,
                    "clip_start": 0.0,
                    "clip_dur": dur_sec,
                    "source_id": f"{task}/{ep_dir}/{seg}",
                })
            else:
                m = re.match(r"(seg\d+)_clip(\d+)\.mp4$", fname)
                if not m:
                    continue
                seg = m.group(1)
                clip_idx = int(m.group(2))
                clip_start = clip_idx * dur_sec
                robot_src = os.path.join(SEGMENT_DIR, task, ep_dir,
                                         f"{seg}_video.mp4")
                if not os.path.isfile(robot_src):
                    continue
                pairs.append({
                    "task": task,
                    "episode": ep_dir,
                    "seg": seg,
                    "clip_idx": clip_idx,
                    "human_src": human_src,
                    "robot_src": robot_src,
                    "clip_start": clip_start,
                    "clip_dur": dur_sec,
                    "source_id": f"{task}/{ep_dir}/{seg}_clip{clip_idx:02d}",
                })

    pairs.sort(key=lambda p: p["source_id"])
    return pairs


# ── splitting ──────────────────────────────────────────────────────────

def split_by_task(all_pairs: list[dict], ood_tasks: set[str],
                  per_task_eval: int, split_seed: int
                  ) -> dict[str, list[dict]]:
    """Partition pairs into {train, eval, ood_eval} by task.

    - OOD tasks: every clip → ood_eval
    - Non-OOD tasks: seed-deterministic shuffle, first `per_task_eval` → eval,
      rest → train
    """
    by_task: dict[str, list[dict]] = {}
    for p in all_pairs:
        by_task.setdefault(p["task"], []).append(p)

    splits: dict[str, list[dict]] = {"train": [], "eval": [], "ood_eval": []}
    for task, clips in sorted(by_task.items()):
        if task in ood_tasks:
            splits["ood_eval"].extend(clips)
            continue
        rng = random.Random(f"{split_seed}:{task}")
        shuffled = list(clips)
        rng.shuffle(shuffled)
        n_eval = min(per_task_eval, max(0, len(shuffled) - 1))  # keep >=1 for train
        splits["eval"].extend(shuffled[:n_eval])
        splits["train"].extend(shuffled[n_eval:])

    for name in splits:
        splits[name].sort(key=lambda p: p["source_id"])
    return splits


# ── process one pair ───────────────────────────────────────────────────

def process_pair(p: dict, do_compare: bool, resume: bool,
                 num_frames: int | None = None,
                 hand_patch: bool = False,
                 hand_weight: float = 3.0) -> dict:
    """Process a single pair. Returns metadata dict."""
    os.makedirs(os.path.dirname(p["out_robot"]), exist_ok=True)
    os.makedirs(os.path.dirname(p["out_human"]), exist_ok=True)

    # robot → video/ (target)
    if not (resume and os.path.isfile(p["out_robot"])):
        make_robot_clip(p["robot_src"], p["out_robot"],
                        p["clip_start"], p["clip_dur"],
                        num_frames=num_frames)

    # human → control_video/ (condition)
    if not (resume and os.path.isfile(p["out_human"])):
        make_human_clip(p["human_src"], p["out_human"],
                        start=p["clip_start"], duration=p["clip_dur"],
                        num_frames=num_frames)

    # compare
    if do_compare:
        os.makedirs(os.path.dirname(p["compare"]), exist_ok=True)
        if not (resume and os.path.isfile(p["compare"])):
            make_compare(p["out_robot"], p["out_human"], p["compare"])

    # hand patch weight map
    if hand_patch and p.get("hand_df") is not None:
        clip_idx = p.get("clip_idx") or 0
        clip_start_frame = int(clip_idx * p["clip_dur"] * SEGMENT_FPS)
        clip_frames = int(p["clip_dur"] * SEGMENT_FPS)
        weights = compute_clip_weight_map(
            p["hand_df"], clip_start_frame, clip_frames, hand_weight)
        if weights is not None:
            split_dir = os.path.dirname(os.path.dirname(p["out_robot"]))
            patch_dir = os.path.join(split_dir, "hand_patch")
            os.makedirs(patch_dir, exist_ok=True)
            pair_name = os.path.basename(p["out_robot"]).replace(".mp4", ".pth")
            torch.save({
                "weights": weights,
                "hand_weight": hand_weight,
            }, os.path.join(patch_dir, pair_name))

    return {
        "video": p["rel_robot"],
        "prompt": PROMPT,
        "control_video": p["rel_human"],
    }


# ── main ───────────────────────────────────────────────────────────────

def _clean_sec_dir(sec_dir: str):
    """Remove old pair outputs (flat video/, control_video/, and split dirs)."""
    for sub in ("video", "control_video", "compare",
                "train", "eval", "ood_eval"):
        p = os.path.join(sec_dir, sub)
        if os.path.isdir(p):
            shutil.rmtree(p)
    for f in ("metadata.csv", "source_map.json"):
        p = os.path.join(sec_dir, f)
        if os.path.isfile(p):
            os.remove(p)


def main():
    ap = argparse.ArgumentParser(
        description="Generate train/eval/ood_eval training pair splits")
    ap.add_argument("--task", required=True,
                    help="task short name or 'all'")
    ap.add_argument("--second", default="all",
                    choices=["1s", "2s", "4s", "all"])
    ap.add_argument("--human-source", default="seedance_direct",
                    choices=list(HUMAN_SOURCE_MAP.keys()),
                    help="human video source (default: seedance_direct)")
    ap.add_argument("--source-second", default=None,
                    choices=["1s", "2s", "4s"],
                    help="source clip duration to cut from (default: same as --second)")
    ap.add_argument("--compare", dest="compare", action="store_true",
                    default=False, help="generate compare videos")
    ap.add_argument("--no-compare", dest="compare", action="store_false")
    ap.add_argument("--resume", action="store_true",
                    help="skip files that already exist")
    ap.add_argument("--clean", action="store_true",
                    help="remove existing pair/<sec>/* before writing")
    ap.add_argument("--ood-tasks",
                    default="Inspire_Pickup_Pillow_MainCamOnly",
                    help="comma-separated task short names routed entirely to "
                         "ood_eval split")
    ap.add_argument("--per-task-eval", type=int, default=1,
                    help="clips per non-OOD task to reserve for in-task eval")
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--hand-patch", action="store_true",
                    help="generate hand patch weight maps from precomputed 4s bbox data")
    ap.add_argument("--hand-weight", type=float, default=3.0,
                    help="weight multiplier for hand regions (default: 3.0)")
    args = ap.parse_args()

    human_root = HUMAN_SOURCE_MAP[args.human_source]
    seconds = ["1s", "2s", "4s"] if args.second == "all" else [args.second]

    TASK_GROUPS = {"all": None, "inspire": "Inspire_", "brainco": "Brainco_"}
    key = args.task.lower()
    if key in TASK_GROUPS:
        prefix = TASK_GROUPS[key]
        short = [t.replace("G1_WBT_", "") for t in ALL_TASKS]
        tasks = [t for t in short if prefix is None or t.startswith(prefix)]
    else:
        tasks = [t.strip() for t in args.task.split(",")]

    ood_tasks = {t.strip() for t in args.ood_tasks.split(",") if t.strip()}

    print(f"Make Pair (split-aware)")
    print(f"  tasks:         {tasks}")
    print(f"  seconds:       {seconds}")
    print(f"  human:         {args.human_source} ({human_root})")
    print(f"  fps:           {TARGET_FPS}")
    print(f"  compare:       {args.compare}")
    print(f"  workers:       {args.workers}")
    print(f"  ood tasks:     {sorted(ood_tasks)}")
    print(f"  per-task eval: {args.per_task_eval}")
    print(f"  split seed:    {args.split_seed}")
    print(f"  source-second: {args.source_second or '(same as --second)'}")
    print(f"  clean:         {args.clean}")
    print(f"  hand-patch:    {args.hand_patch}")
    if args.hand_patch:
        print(f"  hand-weight:   {args.hand_weight}")

    t_total = time.time()

    for sec in seconds:
        source_second = args.source_second or sec
        num_frames = FRAMES_4K1[sec]
        sec_dir = os.path.join(PAIR_DIR, sec)
        if args.clean:
            _clean_sec_dir(sec_dir)
        os.makedirs(sec_dir, exist_ok=True)

        # Collect all pairs across tasks for this duration
        all_pairs = []
        for task in tasks:
            pairs = collect_pairs(task, sec, human_root,
                                  source_second=source_second)
            all_pairs.extend(pairs)

        if not all_pairs:
            print(f"\n[{sec}] no pairs found, skipping")
            continue

        # Load hand patch data if enabled (shared across clips from same segment)
        if args.hand_patch:
            _hand_cache: dict[str, pd.DataFrame | None] = {}
            n_loaded = 0
            for p in all_pairs:
                cache_key = f"{p['task']}/{p['episode']}/{p['seg']}"
                if cache_key not in _hand_cache:
                    _hand_cache[cache_key] = load_hand_patch_data(
                        p["task"], p["episode"], p["seg"])
                    if _hand_cache[cache_key] is not None:
                        n_loaded += 1
                p["hand_df"] = _hand_cache[cache_key]
            print(f"\n[{sec}] hand patch: {n_loaded} segments loaded "
                  f"from {HAND_PATCH_4S_DIR}")

        splits = split_by_task(
            all_pairs, ood_tasks, args.per_task_eval, args.split_seed,
        )

        print(f"\n[{sec}] {len(all_pairs)} pairs -> "
              f"train={len(splits['train'])} "
              f"eval={len(splits['eval'])} "
              f"ood_eval={len(splits['ood_eval'])} "
              f"({num_frames} frames, 4k+1)")

        source_map: dict[str, dict] = {}
        to_process: list[dict] = []

        for split_name, split_pairs in splits.items():
            if not split_pairs:
                continue
            split_dir = os.path.join(sec_dir, split_name)
            video_dir = os.path.join(split_dir, "video")
            control_dir = os.path.join(split_dir, "control_video")
            os.makedirs(video_dir, exist_ok=True)
            os.makedirs(control_dir, exist_ok=True)

            for idx, p in enumerate(split_pairs):
                name = f"pair_{idx:04d}.mp4"
                p["split"] = split_name
                p["out_robot"] = os.path.join(video_dir, name)
                p["out_human"] = os.path.join(control_dir, name)
                p["rel_robot"] = f"video/{name}"
                p["rel_human"] = f"control_video/{name}"
                p["compare"] = os.path.join(split_dir, "compare", name)
                source_map[f"{split_name}/{name}"] = {
                    "task": p["task"],
                    "episode": p["episode"],
                    "seg": p["seg"],
                    "clip_idx": p["clip_idx"],
                    "source_id": p["source_id"],
                    "human_src": p["human_src"],
                    "robot_src": p["robot_src"],
                }
                to_process.append(p)

        t0 = time.time()
        split_meta: dict[str, list[dict]] = {k: [] for k in splits}

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(process_pair, p, args.compare, args.resume,
                            num_frames, args.hand_patch, args.hand_weight): p
                for p in to_process
            }
            done = 0
            total = len(to_process)
            for fut in as_completed(futures):
                p = futures[fut]
                meta = fut.result()
                split_meta[p["split"]].append(meta)
                done += 1
                if done % 20 == 0 or done == total:
                    print(f"  {done}/{total}", flush=True)

        print(f"  done in {time.time() - t0:.1f}s")

        # Write per-split metadata.csv
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

        # Write source_map.json
        map_path = os.path.join(sec_dir, "source_map.json")
        with open(map_path, "w") as f:
            json.dump(source_map, f, indent=2, sort_keys=True)
        print(f"  source_map: {map_path} ({len(source_map)} entries)")

    elapsed = time.time() - t_total
    print(f"\nDone in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
