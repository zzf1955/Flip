"""Generate full-body robot patch data for appearance learning.

For each 4s segment, cuts 4 non-overlapping 1s clips and produces:
  video/pair_NNNN.mp4         clean robot video (17 frames at 16fps)
  control_video/pair_NNNN.mp4 degraded robot video (robot body region damaged)
  patch/pair_NNNN.pth         binary latent-space mask (5, 30, 40)

Three degradation modes:
  blur  — Gaussian blur on robot body region
  noise — Gaussian noise on robot body region
  mean  — fill robot body region with its mean color

Output is compatible with mitty_cache.py (metadata.csv) and train.py
(patch .pth with "weights" key for _load_patch_weights).

Usage:
  python -m src.pipeline.robot_patch --task all --degrade blur
  python -m src.pipeline.robot_patch --task all --degrade noise --noise-std 60
  python -m src.pipeline.robot_patch --task all --degrade mean --max-segments 50
"""

import argparse
import csv
import json
import os
import random
import re
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import av
import cv2
import numpy as np
import pandas as pd
import pinocchio as pin
import torch

sys.stdout.reconfigure(line_buffering=True)

from src.core.config import (
    ALL_TASKS, BEST_PARAMS, G1_URDF, MAIN_ROOT, MESH_DIR,
    get_hand_type, get_skip_meshes,
)
from src.core.camera import make_camera_const
from src.core.data import close_video, open_video_writer, write_frame
from src.core.fk import build_q, do_fk, parse_urdf_meshes, preload_meshes
from src.core.render import render_mask

_MAIN_TRAINING_DATA = os.path.join(MAIN_ROOT, "training_data")
_SEGMENT_DIR = os.path.join(_MAIN_TRAINING_DATA, "segment")
_ROBOT_PATCH_DIR = os.path.join(_MAIN_TRAINING_DATA, "robot_patch")

SEGMENT_FPS = 30
TARGET_FPS = 16
NUM_FRAMES = 17
SEGMENT_DURATION = 4.0
CLIP_DURATION = 1.0
CLIPS_PER_SEG = int(SEGMENT_DURATION / CLIP_DURATION)

VIDEO_H, VIDEO_W = 480, 640
LATENT_H, LATENT_W = 30, 40
LATENT_F = 5
VAE_SPATIAL = 16

PROMPT = "A first-person view robot arm performing household tasks flip_v2v"

RESAMPLE_INDICES = [min(round(i * SEGMENT_FPS / TARGET_FPS), 29)
                    for i in range(NUM_FRAMES)]


# ── Segment discovery ────────────────────────────────────────────────

def find_segments(tasks: list[str]) -> list[dict]:
    segments = []
    for task in tasks:
        task_dir = os.path.join(_SEGMENT_DIR, task)
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
                    "video_path": os.path.join(root, fname),
                    "parquet_path": os.path.join(root, f"{seg}_joints.parquet"),
                })
    segments.sort(key=lambda s: f"{s['task']}/{s['episode']}/{s['seg']}")
    return segments


def make_pairs(segments: list[dict]) -> list[dict]:
    pairs = []
    for s in segments:
        for ci in range(CLIPS_PER_SEG):
            pairs.append({
                "task": s["task"],
                "episode": s["episode"],
                "seg": s["seg"],
                "video_path": s["video_path"],
                "parquet_path": s["parquet_path"],
                "clip": ci,
                "clip_start_30fps": ci * int(CLIP_DURATION * SEGMENT_FPS),
                "source_id": f"{s['task']}/{s['episode']}/{s['seg']}_c{ci}",
            })
    return pairs


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


# ── FK model ─────────────────────────────────────────────────────────

def load_fk_model(hand_type: str):
    model = pin.buildModelFromUrdf(str(G1_URDF), pin.JointModelFreeFlyer())
    data = model.createData()
    link_meshes = parse_urdf_meshes(str(G1_URDF))
    skip_set = get_skip_meshes(hand_type)
    mesh_cache = preload_meshes(link_meshes, str(MESH_DIR), skip_set=skip_set)
    cam_const = make_camera_const(BEST_PARAMS)
    return model, data, mesh_cache, cam_const


# ── Mask processing ──────────────────────────────────────────────────

def soften_mask(mask: np.ndarray) -> np.ndarray:
    """Smooth + dilate + soft-edge for degradation blending."""
    out = cv2.GaussianBlur(mask, (7, 7), 0)
    out = (out > 128).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    out = cv2.dilate(out, kernel)
    out = cv2.GaussianBlur(out, (21, 21), 0)
    return out


def pixel_mask_to_latent(mask_hw: np.ndarray) -> np.ndarray:
    """(H, W) uint8 mask -> (LATENT_H, LATENT_W) binary float."""
    hard = (mask_hw > 128).astype(np.float32)
    latent = hard.reshape(LATENT_H, VAE_SPATIAL, LATENT_W, VAE_SPATIAL)
    latent = latent.max(axis=(1, 3))
    return latent


def build_latent_mask(masks: list[np.ndarray]) -> torch.Tensor:
    """17 per-frame pixel masks -> (LATENT_F, LATENT_H, LATENT_W) binary tensor."""
    latent = torch.zeros(LATENT_F, LATENT_H, LATENT_W, dtype=torch.float32)
    for j in range(LATENT_F):
        start = j * 4
        end = min(start + 4, NUM_FRAMES)
        group = np.maximum.reduce(masks[start:end])
        latent[j] = torch.from_numpy(pixel_mask_to_latent(group))
    return latent


# ── Degradation ──────────────────────────────────────────────────────

def degrade_blur(frame: np.ndarray, soft_mask: np.ndarray,
                 ksize: int) -> np.ndarray:
    blurred = cv2.GaussianBlur(frame, (ksize, ksize), 0)
    alpha = soft_mask.astype(np.float32) / 255.0
    out = alpha[..., None] * blurred + (1.0 - alpha[..., None]) * frame
    return np.clip(out, 0, 255).astype(np.uint8)


def degrade_noise(frame: np.ndarray, soft_mask: np.ndarray,
                  noise_std: float, rng: np.random.Generator) -> np.ndarray:
    noise = rng.standard_normal(frame.shape).astype(np.float32) * noise_std
    noisy = np.clip(frame.astype(np.float32) + noise, 0, 255)
    alpha = soft_mask.astype(np.float32) / 255.0
    out = alpha[..., None] * noisy + (1.0 - alpha[..., None]) * frame
    return np.clip(out, 0, 255).astype(np.uint8)


def degrade_mean(frame: np.ndarray, soft_mask: np.ndarray,
                 hard_mask: np.ndarray) -> np.ndarray:
    robot_pixels = frame[hard_mask > 128]
    if len(robot_pixels) == 0:
        return frame.copy()
    mean_val = robot_pixels.mean(axis=0)
    mean_img = np.full_like(frame, mean_val, dtype=np.float32)
    alpha = soft_mask.astype(np.float32) / 255.0
    out = alpha[..., None] * mean_img + (1.0 - alpha[..., None]) * frame
    return np.clip(out, 0, 255).astype(np.uint8)


# ── Video I/O ────────────────────────────────────────────────────────

def read_all_frames(video_path: str) -> list[np.ndarray]:
    """Read all frames from a video file."""
    container = av.open(video_path)
    stream = container.streams.video[0]
    frames = [f.to_ndarray(format="bgr24") for f in container.decode(stream)]
    container.close()
    return frames


def write_video_from_frames(frames: list[np.ndarray], path: str,
                            fps: int = TARGET_FPS):
    h, w = frames[0].shape[:2]
    container, stream = open_video_writer(path, w, h, fps=fps)
    for f in frames:
        write_frame(container, stream, f)
    close_video(container, stream)


# ── Per-clip processing ──────────────────────────────────────────────

def process_clip(pair: dict, seg_frames: list[np.ndarray],
                 seg_df: pd.DataFrame, seg_frame_min: int,
                 fk_model, fk_data, mesh_cache, cam_const,
                 hand_type: str, degrade_mode: str,
                 blur_ksize: int, noise_std: float) -> dict | None:
    """Process one 1s clip from pre-loaded segment data.

    Returns dict with keys: clean_frames, degraded_frames, latent_mask.
    Returns None if the clip has insufficient data.
    """
    clip_start = pair["clip_start_30fps"]
    clip_end = clip_start + 30
    clip_frames_30 = seg_frames[clip_start:clip_end]
    if len(clip_frames_30) < 17:
        return None

    clip_abs_start = seg_frame_min + clip_start
    clip_df = seg_df[
        (seg_df["frame_index"] >= clip_abs_start) &
        (seg_df["frame_index"] < clip_abs_start + 30)
    ]
    if len(clip_df) < 17:
        return None

    clean_frames = []
    degraded_frames = []
    hard_masks = []
    rng = np.random.default_rng(seed=hash(pair["source_id"]) & 0xFFFFFFFF)

    for _, src_idx in enumerate(RESAMPLE_INDICES):
        if src_idx >= len(clip_frames_30):
            src_idx = len(clip_frames_30) - 1
        frame = clip_frames_30[src_idx]

        abs_frame = clip_abs_start + src_idx
        row = clip_df[clip_df["frame_index"] == abs_frame]
        if row.empty:
            nearest_idx = (clip_df["frame_index"] - abs_frame).abs().idxmin()
            row = clip_df.loc[[nearest_idx]]

        row = row.iloc[0]
        rq = np.array(row["observation.state.robot_q_current"], dtype=np.float64)
        hs = np.array(row["observation.state.hand_state"], dtype=np.float64)
        q = build_q(fk_model, rq, hs, hand_type=hand_type)
        transforms = do_fk(fk_model, fk_data, q)

        hard_mask = render_mask(mesh_cache, transforms, BEST_PARAMS,
                                VIDEO_H, VIDEO_W, cam_const)
        soft_mask = soften_mask(hard_mask)
        hard_masks.append(hard_mask)

        if degrade_mode == "blur":
            degraded = degrade_blur(frame, soft_mask, blur_ksize)
        elif degrade_mode == "noise":
            degraded = degrade_noise(frame, soft_mask, noise_std, rng)
        elif degrade_mode == "mean":
            degraded = degrade_mean(frame, soft_mask, hard_mask)

        clean_frames.append(frame)
        degraded_frames.append(degraded)

    latent_mask = build_latent_mask(hard_masks)

    return {
        "clean_frames": clean_frames,
        "degraded_frames": degraded_frames,
        "latent_mask": latent_mask,
    }


def write_pair_outputs(pair: dict, result: dict):
    """Write video files and patch .pth to disk."""
    os.makedirs(os.path.dirname(pair["out_video"]), exist_ok=True)
    os.makedirs(os.path.dirname(pair["out_ctrl"]), exist_ok=True)
    os.makedirs(os.path.dirname(pair["out_patch"]), exist_ok=True)

    write_video_from_frames(result["clean_frames"], pair["out_video"])
    write_video_from_frames(result["degraded_frames"], pair["out_ctrl"])

    latent_mask = result["latent_mask"]
    torch.save({
        "mask": latent_mask,
        "weights": 1.0 + 2.0 * latent_mask,
    }, pair["out_patch"])


# ── Main ─────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Generate full-body robot patch data for appearance learning")
    ap.add_argument("--task", required=True,
                    help="task short name, comma-separated, or 'all'/'inspire'/'brainco'")
    ap.add_argument("--degrade", choices=["blur", "noise", "mean"], default="blur",
                    help="degradation mode for robot body region")
    ap.add_argument("--blur-ksize", type=int, default=51,
                    help="Gaussian blur kernel size for blur mode (must be odd)")
    ap.add_argument("--noise-std", type=float, default=50.0,
                    help="Gaussian noise std for noise mode (0-255 scale)")
    ap.add_argument("--max-segments", type=int, default=0,
                    help="max segments per task (0 = unlimited)")
    ap.add_argument("--per-task-eval", type=int, default=5,
                    help="segments per task reserved for eval")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=4,
                    help="thread pool workers for video I/O")
    ap.add_argument("--resume", action="store_true",
                    help="skip pairs whose outputs already exist")
    ap.add_argument("--clean", action="store_true",
                    help="remove existing robot_patch/1s/ before writing")
    args = ap.parse_args()

    key = args.task.lower()
    task_groups = {"all": None, "inspire": "Inspire_", "brainco": "Brainco_"}
    if key in task_groups:
        prefix = task_groups[key]
        short = [t.replace("G1_WBT_", "") for t in ALL_TASKS]
        tasks = [t for t in short if prefix is None or t.startswith(prefix)]
    else:
        tasks = [t.strip() for t in args.task.split(",")]

    sec_dir = os.path.join(_ROBOT_PATCH_DIR, "1s")
    if args.clean and os.path.isdir(sec_dir):
        shutil.rmtree(sec_dir)

    print("Robot Patch (full-body degradation for appearance learning)")
    print(f"  tasks:         {tasks}")
    print(f"  degrade:       {args.degrade}")
    if args.degrade == "blur":
        print(f"  blur-ksize:    {args.blur_ksize}")
    elif args.degrade == "noise":
        print(f"  noise-std:     {args.noise_std}")
    print(f"  max-segments:  {args.max_segments or 'unlimited'}")
    print(f"  per-task-eval: {args.per_task_eval}")
    print(f"  seed:          {args.seed}")
    print(f"  workers:       {args.workers}")
    print(f"  resume:        {args.resume}")

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

    all_pairs = make_pairs(all_segs)
    if not all_pairs:
        print("No pairs generated, exiting")
        return

    splits = split_pairs(all_pairs, args.per_task_eval, args.seed)
    print(f"  {len(all_pairs)} pairs → "
          f"train={len(splits['train'])} eval={len(splits['eval'])}")

    # Assign output paths
    source_map: dict[str, dict] = {}
    all_to_process: list[dict] = []

    for split_name, split_pairs_list in splits.items():
        if not split_pairs_list:
            continue
        split_dir = os.path.join(sec_dir, split_name)
        video_dir = os.path.join(split_dir, "video")
        ctrl_dir = os.path.join(split_dir, "control_video")
        patch_dir = os.path.join(split_dir, "patch")
        os.makedirs(video_dir, exist_ok=True)
        os.makedirs(ctrl_dir, exist_ok=True)
        os.makedirs(patch_dir, exist_ok=True)

        for idx, p in enumerate(split_pairs_list):
            name = f"pair_{idx:04d}"
            p["split"] = split_name
            p["out_video"] = os.path.join(video_dir, f"{name}.mp4")
            p["out_ctrl"] = os.path.join(ctrl_dir, f"{name}.mp4")
            p["out_patch"] = os.path.join(patch_dir, f"{name}.pth")
            p["rel_video"] = f"video/{name}.mp4"
            p["rel_ctrl"] = f"control_video/{name}.mp4"
            source_map[f"{split_name}/{name}.mp4"] = {
                "task": p["task"],
                "episode": p["episode"],
                "seg": p["seg"],
                "clip": p["clip"],
                "source_id": p["source_id"],
                "video_path": p["video_path"],
            }
            all_to_process.append(p)

    # Load FK model (reloaded when hand_type changes)
    current_hand_type = None
    fk_model = fk_data = mesh_cache = cam_const = None

    t_start = time.time()
    n_done = 0
    n_skip = 0
    n_fail = 0
    last_report = 0

    # Group pairs by (task, episode, seg) for segment-level batching
    seg_groups: dict[str, list[dict]] = {}
    for p in all_to_process:
        seg_key = f"{p['task']}/{p['episode']}/{p['seg']}"
        seg_groups.setdefault(seg_key, []).append(p)

    write_pool = ThreadPoolExecutor(max_workers=args.workers)
    pending_futures: list = []

    for seg_key in sorted(seg_groups.keys()):
        seg_pairs = seg_groups[seg_key]
        task = seg_pairs[0]["task"]
        full_task_name = f"G1_WBT_{task}"
        hand_type = get_hand_type(full_task_name)

        if hand_type != current_hand_type:
            print(f"\nLoading FK model for hand_type={hand_type}...")
            t0 = time.time()
            fk_model, fk_data, mesh_cache, cam_const = load_fk_model(hand_type)
            print(f"  loaded in {time.time() - t0:.1f}s")
            current_hand_type = hand_type

        # Check if all clips in this segment can be skipped
        active_pairs = []
        for p in seg_pairs:
            if args.resume and all(
                os.path.isfile(p[k])
                for k in ("out_video", "out_ctrl", "out_patch")
            ):
                n_skip += 1
            else:
                active_pairs.append(p)
        if not active_pairs:
            continue

        # Read video + parquet once for the entire segment
        ref = active_pairs[0]
        seg_frames = read_all_frames(ref["video_path"])
        if not os.path.isfile(ref["parquet_path"]):
            n_fail += len(active_pairs)
            continue
        seg_df = pd.read_parquet(ref["parquet_path"])
        seg_frame_min = int(seg_df["frame_index"].min())

        for pair in active_pairs:
            result = process_clip(
                pair, seg_frames, seg_df, seg_frame_min,
                fk_model, fk_data, mesh_cache, cam_const,
                hand_type, args.degrade, args.blur_ksize, args.noise_std,
            )

            if result is None:
                n_fail += 1
                continue

            fut = write_pool.submit(write_pair_outputs, pair, result)
            pending_futures.append(fut)
            n_done += 1

        if n_done - last_report >= 100:
            elapsed = time.time() - t_start
            rate = n_done / elapsed if elapsed > 0 else 0
            print(f"  [{n_done} done, {n_skip} skipped, {n_fail} failed] "
                  f"({rate:.1f} pairs/s)")
            last_report = n_done

    # Wait for all writes to complete
    for fut in pending_futures:
        fut.result()
    write_pool.shutdown(wait=True)

    # Write metadata.csv per split
    split_meta: dict[str, list[dict]] = {"train": [], "eval": []}
    for p in all_to_process:
        if p["split"] not in split_meta:
            continue
        split_meta[p["split"]].append({
            "video": p["rel_video"],
            "prompt": PROMPT,
            "control_video": p["rel_ctrl"],
        })

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

    elapsed = time.time() - t_start
    print(f"\nDone. {n_done} processed, {n_skip} skipped, {n_fail} failed. "
          f"{elapsed:.1f}s")


if __name__ == "__main__":
    main()
