"""
Generate split-free task-organized training pairs and comparison videos.

Matches segment (robot) videos with human videos (seedance_direct, overlay, or
seedance_advance), resamples both to 16fps, and writes three independent
subdirectories per semantic data type, duration, and robot task:
  pair/<data_type>/<second>/<task>/video/
  pair/<data_type>/<second>/<task>/control_video/

Each task directory gets its own pair_NNNN.mp4 numbering, metadata.csv, and
manifest.jsonl. In-task/OOD splits are decided by training runtime.

Usage:
  # Formal training data: --task all expands to TRAINING_TASKS only.
  python -m src.pipeline.make_pair --task all --second 1s --data-type h2r --clean

  python -m src.pipeline.make_pair --task all --second 1s \
    --data-type h2r --human-source seedance_advance \
    --hand-patch --hand-weight 3.0

  # Historical/debug data outside TRAINING_TASKS must be requested explicitly.
  python -m src.pipeline.make_pair --task inspire --second 1s --clean
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

import av
import cv2
import numpy as np
import pandas as pd
import torch

sys.stdout.reconfigure(line_buffering=True)

from src.core.config import (
    ALL_TASKS, MAIN_ROOT,
    PAIR_DIR, TRAINING_DATA_ROOT,
    TRAINING_TASKS,
)

_MAIN_TRAINING_DATA = os.path.join(MAIN_ROOT, "training_data")
_MAIN_SEGMENT = os.path.join(_MAIN_TRAINING_DATA, "segment")
_MAIN_HAND_PATCH_4S = os.path.join(_MAIN_TRAINING_DATA, "hand_patch", "4s")
_MAIN_SEEDANCE_DIRECT = os.path.join(_MAIN_TRAINING_DATA, "seedance_direct")
_MAIN_OVERLAY = os.path.join(_MAIN_TRAINING_DATA, "overlay")
_MAIN_SEEDANCE_ADVANCE = os.path.join(_MAIN_TRAINING_DATA, "seedance_advance")
_MAIN_SAM2_MASK = os.path.join(_MAIN_TRAINING_DATA, "sam2_mask")

FFMPEG = os.environ.get(
    "FFMPEG_BIN",
    "/home/leadtek/miniconda3/envs/flip/bin/ffmpeg",
)

TARGET_FPS = 16
COMPARE_DIR = os.path.join(TRAINING_DATA_ROOT, "compare")
PROMPT = "A first-person view robot arm performing household tasks flip_v2v"
DEFAULT_BLUR_KSIZE = 51
DEFAULT_BLUR_PIXEL_EXPAND = 16

# 4k+1 frame counts at 16fps for each clip duration
FRAMES_4K1 = {"1s": 17, "2s": 33, "4s": 65}

HUMAN_SOURCE_MAP = {
    "seedance_direct": _MAIN_SEEDANCE_DIRECT,
    "overlay": _MAIN_OVERLAY,
    "seedance_advance": _MAIN_SEEDANCE_ADVANCE,
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
                    num_frames: int | None = None,
                    hflip: bool = False):
    """Cut a segment from the robot video and resample to TARGET_FPS."""
    args = [
        "-ss", f"{start:.3f}", "-i", segment_video,
        "-t", f"{duration + 0.5:.3f}",  # generous cut, rely on -frames:v
        "-r", str(TARGET_FPS),
    ]
    if hflip:
        args += ["-vf", "hflip"]
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


def read_video_bgr(video_path: str) -> list[np.ndarray]:
    """Read all frames from a video as BGR uint8 arrays."""
    container = av.open(video_path)
    stream = container.streams.video[0]
    frames = [f.to_ndarray(format="bgr24") for f in container.decode(stream)]
    container.close()
    if not frames:
        raise RuntimeError(f"video has no frames: {video_path}")
    return frames


def write_video_bgr(frames: list[np.ndarray], out_path: str,
                    fps: int = TARGET_FPS):
    """Write BGR uint8 frames as an H.264 video."""
    from src.core.data import close_video, open_video_writer, write_frame

    if not frames:
        raise ValueError(f"no frames to write: {out_path}")
    h, w = frames[0].shape[:2]
    container, stream = open_video_writer(out_path, w, h, fps=fps)
    for frame in frames:
        write_frame(container, stream, frame)
    close_video(container, stream)


def soften_mask(mask: np.ndarray, pixel_expand: int) -> np.ndarray:
    """Match robot_patch blur-mask feathering for full-body degradation."""
    out = cv2.GaussianBlur(mask, (7, 7), 0)
    out = (out > 128).astype(np.uint8) * 255
    d = 15 + 2 * pixel_expand
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d, d))
    out = cv2.dilate(out, kernel)
    blur_k = 21 + 2 * (pixel_expand // 2)
    if blur_k % 2 == 0:
        blur_k += 1
    out = cv2.GaussianBlur(out, (blur_k, blur_k), 0)
    return out


def blur_frame_in_mask(frame: np.ndarray, soft_mask: np.ndarray,
                       ksize: int) -> np.ndarray:
    """Blur only the robot mask region and alpha-blend soft boundaries."""
    blurred = cv2.GaussianBlur(frame, (ksize, ksize), 0)
    alpha = soft_mask.astype(np.float32) / 255.0
    out = alpha[..., None] * blurred + (1.0 - alpha[..., None]) * frame
    return np.clip(out, 0, 255).astype(np.uint8)


def clip_mask_indices(clip_start: float, clip_dur: float,
                      num_frames: int, mask_count: int) -> list[int]:
    """Map output frames to original 30fps segment mask frame indices."""
    base = int(round(clip_start * SEGMENT_FPS))
    clip_frames = max(1, int(round(clip_dur * SEGMENT_FPS)))
    indices = []
    for i in range(num_frames):
        offset = min(round(i * SEGMENT_FPS / TARGET_FPS), clip_frames - 1)
        indices.append(min(max(base + offset, 0), mask_count - 1))
    return indices


def make_blurred_robot_clip(clean_clip_path: str, out_path: str, p: dict,
                            mask_root: str, blur_ksize: int,
                            pixel_expand: int):
    """Create blur_r2r control video from the clean robot target and SAM2 masks."""
    mask_path = os.path.join(mask_root, p["task"], p["episode"],
                             f"{p['seg']}.npz")
    if not os.path.isfile(mask_path):
        raise FileNotFoundError(f"SAM2 mask not found for blur_r2r: {mask_path}")

    with np.load(mask_path) as mask_npz:
        masks = mask_npz["masks"]
    if masks.ndim != 3:
        raise ValueError(f"invalid SAM2 mask shape in {mask_path}: {masks.shape}")

    frames = read_video_bgr(clean_clip_path)
    mask_indices = clip_mask_indices(
        p["clip_start"], p["clip_dur"], len(frames), len(masks))
    degraded_frames = []
    for frame, mask_idx in zip(frames, mask_indices):
        mask = masks[mask_idx]
        if p.get("augment") == "hflip":
            mask = cv2.flip(mask, 1)
        if mask.shape[:2] != frame.shape[:2]:
            raise ValueError(
                f"mask/frame shape mismatch for {mask_path}: "
                f"mask={mask.shape[:2]}, frame={frame.shape[:2]}")
        soft = soften_mask(mask, pixel_expand=pixel_expand)
        degraded_frames.append(blur_frame_in_mask(frame, soft, blur_ksize))

    write_video_bgr(degraded_frames, out_path, fps=TARGET_FPS)


# ── hand patch helpers ─────────────────────────────────────────────────

def load_hand_patch_data(task: str, ep_dir: str, seg: str) -> pd.DataFrame | None:
    """Load precomputed per-frame hand bboxes from hand_patch/4s/."""
    path = os.path.join(_MAIN_HAND_PATCH_4S, task, ep_dir, f"{seg}_hands.parquet")
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

def _load_clip_manifest(human_dir: str) -> dict[str, dict]:
    """Load optional seedance_clip manifest keyed by task-relative clip path."""
    manifest_path = os.path.join(human_dir, "manifest.jsonl")
    if not os.path.isfile(manifest_path):
        return {}

    records = {}
    with open(manifest_path) as f:
        for line in f:
            record = json.loads(line)
            clip_rel = record["clip"]
            parts = clip_rel.split(os.sep, 2)
            if len(parts) != 3:
                raise ValueError(f"invalid clip path in manifest: {clip_rel}")
            records[parts[2]] = record
    return records


def _expand_task_spec(task_spec: str, default_tasks: list[str]) -> list[str]:
    """Resolve CLI task spec to short task names used by pair generators."""
    task_groups = {
        "all": default_tasks,
        "training": default_tasks,
        "inspire": [t for t in ALL_TASKS if "G1_WBT_Inspire_" in t],
        "brainco": [t for t in ALL_TASKS if "G1_WBT_Brainco_" in t],
    }
    key = task_spec.lower()
    if key in task_groups:
        return [t.replace("G1_WBT_", "") for t in task_groups[key]]
    return [t.strip() for t in task_spec.split(",") if t.strip()]


def collect_pairs(task: str, second: str, human_root: str,
                  source_second: str | None = None) -> list[dict]:
    """Find all (human, robot) pairs for a given task and clip duration.

    For pre-cut Seedance direct 1s clips, reads manifest.jsonl when present so
    overlapping windows and hflip augmentation keep their true source time.

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
    manifest = _load_clip_manifest(human_dir) if not cross_cut else {}
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
                robot_src = os.path.join(_MAIN_SEGMENT, task, ep_dir,
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
                    "human_clip_start": ci * dur_sec,
                    "clip_dur": dur_sec,
                    "source_segment_id": f"{task}/{ep_dir}/{seg}",
                    "source_id": f"{task}/{ep_dir}/{seg}_clip{ci:02d}",
                })
            elif src_sec == "4s":
                m = re.match(r"(seg\d+)_human\.mp4$", fname)
                if not m:
                    continue
                seg = m.group(1)
                robot_src = os.path.join(_MAIN_SEGMENT, task, ep_dir,
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
                    "human_clip_start": 0.0,
                    "clip_dur": dur_sec,
                    "source_segment_id": f"{task}/{ep_dir}/{seg}",
                    "source_id": f"{task}/{ep_dir}/{seg}",
                })
            else:
                m = re.match(r"(seg\d+)_clip(\d+)\.mp4$", fname)
                if not m:
                    continue
                seg = m.group(1)
                clip_idx = int(m.group(2))
                clip_rel = os.path.relpath(human_src, human_dir)
                manifest_record = manifest.get(clip_rel)
                if manifest_record:
                    clip_start = float(manifest_record["start"])
                    clip_dur = float(manifest_record["duration"])
                    augment = manifest_record.get("augment", "normal")
                    window_idx = manifest_record.get("window_idx", clip_idx)
                else:
                    clip_start = clip_idx * dur_sec
                    clip_dur = dur_sec
                    augment = "normal"
                    window_idx = clip_idx
                robot_src = os.path.join(_MAIN_SEGMENT, task, ep_dir,
                                         f"{seg}_video.mp4")
                if not os.path.isfile(robot_src):
                    continue
                source_suffix = f"clip{clip_idx:02d}"
                if augment != "normal":
                    source_suffix += f"_{augment}"
                pairs.append({
                    "task": task,
                    "episode": ep_dir,
                    "seg": seg,
                    "clip_idx": clip_idx,
                    "window_idx": window_idx,
                    "augment": augment,
                    "human_src": human_src,
                    "robot_src": robot_src,
                    "clip_start": clip_start,
                    "human_clip_start": 0.0 if manifest_record else clip_start,
                    "clip_dur": clip_dur,
                    "source_segment_id": f"{task}/{ep_dir}/{seg}",
                    "source_id": f"{task}/{ep_dir}/{seg}_{source_suffix}",
                })

    pairs.sort(key=lambda p: p["source_id"])
    return pairs


# ── splitting ──────────────────────────────────────────────────────────

def split_by_task(all_pairs: list[dict], ood_tasks: set[str],
                  per_task_eval: int, split_seed: int,
                  max_ood_per_task: int = 0,
                  per_task_eval_clips: int = 0,
                  ) -> dict[str, list[dict]]:
    """Partition pairs into {train, eval, ood_eval} by task.

    - OOD tasks: every source segment → ood_eval
    - Non-OOD tasks: split by source segment, so overlapping windows and hflip
      variants from one 4s source never leak across train/eval
    """
    by_task: dict[str, dict[str, list[dict]]] = {}
    for p in all_pairs:
        segment_id = p.get("source_segment_id") or f"{p['task']}/{p['episode']}/{p['seg']}"
        by_task.setdefault(p["task"], {}).setdefault(segment_id, []).append(p)

    splits: dict[str, list[dict]] = {"train": [], "eval": [], "ood_eval": []}
    for task, segment_groups in sorted(by_task.items()):
        if task in ood_tasks:
            clips = [
                clip
                for segment_id in sorted(segment_groups)
                for clip in sorted(segment_groups[segment_id],
                                   key=lambda p: p["source_id"])
            ]
            if max_ood_per_task > 0:
                clips = clips[:max_ood_per_task]
            splits["ood_eval"].extend(clips)
            continue
        rng = random.Random(f"{split_seed}:{task}")
        segment_ids = sorted(segment_groups)
        rng.shuffle(segment_ids)
        if per_task_eval_clips > 0:
            remaining_eval = per_task_eval_clips
            heldout_segments = set()
            for segment_id in segment_ids:
                if remaining_eval == 0:
                    break
                segment_clips = sorted(segment_groups[segment_id],
                                       key=lambda p: p["source_id"])
                take = min(remaining_eval, len(segment_clips))
                splits["eval"].extend(segment_clips[:take])
                heldout_segments.add(segment_id)
                remaining_eval -= take
            if remaining_eval != 0:
                raise RuntimeError(
                    f"task {task} has insufficient clips for "
                    f"per_task_eval_clips={per_task_eval_clips}")
            for segment_id in segment_ids:
                if segment_id in heldout_segments:
                    continue
                splits["train"].extend(segment_groups[segment_id])
        else:
            n_eval = min(per_task_eval, max(0, len(segment_ids) - 1))
            eval_segments = set(segment_ids[:n_eval])
            for segment_id in segment_ids:
                bucket = "eval" if segment_id in eval_segments else "train"
                splits[bucket].extend(segment_groups[segment_id])

    for name in splits:
        splits[name].sort(key=lambda p: p["source_id"])
    return splits


# ── process one pair ───────────────────────────────────────────────────

def process_pair(p: dict, do_compare: bool, resume: bool,
                 num_frames: int | None = None,
                 hand_patch: bool = False,
                 hand_weight: float = 3.0,
                 blur_mask_root: str = _MAIN_SAM2_MASK,
                 blur_ksize: int = DEFAULT_BLUR_KSIZE,
                 blur_pixel_expand: int = DEFAULT_BLUR_PIXEL_EXPAND) -> dict:
    """Process a single pair. Returns metadata dict."""
    os.makedirs(os.path.dirname(p["out_target"]), exist_ok=True)
    os.makedirs(os.path.dirname(p["out_control"]), exist_ok=True)

    if not (resume and os.path.isfile(p["out_target"])):
        if p["target_role"] == "robot":
            make_robot_clip(p["robot_src"], p["out_target"],
                            p["clip_start"], p["clip_dur"],
                            num_frames=num_frames,
                            hflip=p.get("augment") == "hflip")
        elif p["target_role"] == "human":
            make_human_clip(p["human_src"], p["out_target"],
                            start=p.get("human_clip_start", p["clip_start"]),
                            duration=p["clip_dur"],
                            num_frames=num_frames)
        else:
            raise ValueError(f"Unknown target_role: {p['target_role']}")

    if not (resume and os.path.isfile(p["out_control"])):
        if p["input_role"] == "human":
            make_human_clip(p["human_src"], p["out_control"],
                            start=p.get("human_clip_start", p["clip_start"]),
                            duration=p["clip_dur"],
                            num_frames=num_frames)
        elif p["input_role"] == "robot":
            if p.get("control_degrade") == "sam2_blur":
                if p["target_role"] != "robot":
                    raise ValueError(
                        "sam2_blur control requires robot target_role")
                make_blurred_robot_clip(
                    p["out_target"], p["out_control"], p,
                    blur_mask_root, blur_ksize, blur_pixel_expand)
            else:
                make_robot_clip(p["robot_src"], p["out_control"],
                                p["clip_start"], p["clip_dur"],
                                num_frames=num_frames,
                                hflip=p.get("augment") == "hflip")
        else:
            raise ValueError(f"Unknown input_role: {p['input_role']}")

    # compare
    if do_compare:
        os.makedirs(os.path.dirname(p["compare"]), exist_ok=True)
        if not (resume and os.path.isfile(p["compare"])):
            make_compare(p["out_target"], p["out_control"], p["compare"])

    # hand patch weight map
    if hand_patch and p.get("hand_df") is not None:
        hand_df = p["hand_df"]
        seg_start = int(hand_df["frame_idx"].min())
        clip_start_frame = seg_start + int(p["clip_start"] * SEGMENT_FPS)
        clip_frames = int(p["clip_dur"] * SEGMENT_FPS)
        weights = compute_clip_weight_map(
            p["hand_df"], clip_start_frame, clip_frames, hand_weight)
        if weights is not None:
            if p.get("augment") == "hflip":
                weights = torch.flip(weights, dims=[2])
            split_dir = os.path.dirname(os.path.dirname(p["out_target"]))
            patch_dir = os.path.join(split_dir, "hand_patch")
            os.makedirs(patch_dir, exist_ok=True)
            pair_name = os.path.basename(p["out_target"]).replace(".mp4", ".pth")
            torch.save({
                "weights": weights,
                "hand_weight": hand_weight,
            }, os.path.join(patch_dir, pair_name))

    return {
        "video": p["rel_target"],
        "prompt": PROMPT,
        "control_video": p["rel_control"],
    }


# ── main ───────────────────────────────────────────────────────────────

def _clean_sec_dir(sec_dir: str):
    """Remove old pair outputs for one data_type/duration directory."""
    for sub in os.listdir(sec_dir) if os.path.isdir(sec_dir) else []:
        p = os.path.join(sec_dir, sub)
        if os.path.isdir(p):
            shutil.rmtree(p)
    for f in ("metadata.csv", "source_map.json", "index.jsonl"):
        p = os.path.join(sec_dir, f)
        if os.path.isfile(p):
            os.remove(p)


def _write_jsonl(path: str, rows: list[dict]):
    with open(path, "w") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")


def main():
    ap = argparse.ArgumentParser(
        description="Generate split-free task-organized training pairs")
    ap.add_argument("--task", required=True,
                    help="task short name or 'all'")
    ap.add_argument("--second", default="all",
                    choices=["1s", "2s", "4s", "all"])
    ap.add_argument("--human-source", default="seedance_direct",
                    choices=list(HUMAN_SOURCE_MAP.keys()),
                    help="human video source (default: seedance_direct)")
    ap.add_argument("--data-type", default="h2r",
                    choices=["h2r", "r2h", "blur_r2r"],
                    help="semantic pair type for output layout")
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
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--hand-patch", action="store_true",
                    help="generate hand patch weight maps from precomputed 4s bbox data")
    ap.add_argument("--hand-weight", type=float, default=3.0,
                    help="weight multiplier for hand regions (default: 3.0)")
    ap.add_argument("--blur-mask-root", default=_MAIN_SAM2_MASK,
                    help="SAM2 mask root for blur_r2r control degradation")
    ap.add_argument("--blur-ksize", type=int, default=DEFAULT_BLUR_KSIZE,
                    help="Gaussian blur kernel for blur_r2r control videos")
    ap.add_argument("--blur-pixel-expand", type=int,
                    default=DEFAULT_BLUR_PIXEL_EXPAND,
                    help="pixel-space mask dilation for blur_r2r soft mask")
    args = ap.parse_args()

    if args.blur_ksize <= 0 or args.blur_ksize % 2 == 0:
        raise ValueError("--blur-ksize must be a positive odd integer")
    if args.blur_pixel_expand < 0:
        raise ValueError("--blur-pixel-expand must be non-negative")

    human_root = HUMAN_SOURCE_MAP[args.human_source]
    seconds = ["1s", "2s", "4s"] if args.second == "all" else [args.second]

    tasks = _expand_task_spec(args.task, TRAINING_TASKS)

    print(f"Make Pair (task-organized, split-free)")
    print(f"  tasks:         {tasks}")
    print(f"  seconds:       {seconds}")
    print(f"  data type:     {args.data_type}")
    print(f"  human:         {args.human_source} ({human_root})")
    print(f"  fps:           {TARGET_FPS}")
    print(f"  compare:       {args.compare}")
    print(f"  workers:       {args.workers}")
    print(f"  source-second: {args.source_second or '(same as --second)'}")
    print(f"  clean:         {args.clean}")
    print(f"  hand-patch:    {args.hand_patch}")
    if args.hand_patch:
        print(f"  hand-weight:   {args.hand_weight}")
    if args.data_type == "blur_r2r":
        print(f"  blur-mask-root: {args.blur_mask_root}")
        print(f"  blur-ksize:    {args.blur_ksize}")
        print(f"  blur-pixel-expand: {args.blur_pixel_expand}")

    t_total = time.time()

    for sec in seconds:
        source_second = args.source_second or sec
        num_frames = FRAMES_4K1[sec]
        sec_dir = os.path.join(PAIR_DIR, args.data_type, sec)
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
                  f"from {_MAIN_HAND_PATCH_4S}")

        print(f"\n[{sec}] {len(all_pairs)} pairs ({num_frames} frames, 4k+1)")

        to_process: list[dict] = []
        by_task: dict[str, list[dict]] = {}
        for p in all_pairs:
            by_task.setdefault(p["task"], []).append(p)

        for task, task_pairs in sorted(by_task.items()):
            if not task_pairs:
                continue
            task_dir = os.path.join(sec_dir, task)
            video_dir = os.path.join(task_dir, "video")
            control_dir = os.path.join(task_dir, "control_video")
            os.makedirs(video_dir, exist_ok=True)
            os.makedirs(control_dir, exist_ok=True)

            for idx, p in enumerate(sorted(task_pairs, key=lambda x: x["source_id"])):
                name = f"pair_{idx:04d}.mp4"
                input_role = "human" if args.data_type == "h2r" else "robot"
                target_role = "human" if args.data_type == "r2h" else "robot"
                p["input_role"] = input_role
                p["target_role"] = target_role
                if args.data_type == "blur_r2r":
                    p["control_degrade"] = "sam2_blur"
                p["out_target"] = os.path.join(video_dir, name)
                p["out_control"] = os.path.join(control_dir, name)
                p["rel_target"] = f"video/{name}"
                p["rel_control"] = f"control_video/{name}"
                p["compare"] = os.path.join(task_dir, "compare", name)
                p["manifest"] = {
                    "data_type": args.data_type,
                    "duration": sec,
                    "robot_task": p["task"],
                    "task": p["task"],
                    "episode": p["episode"],
                    "seg": p["seg"],
                    "clip_idx": p["clip_idx"],
                    "window_idx": p.get("window_idx"),
                    "augment": p.get("augment", "normal"),
                    "clip_start": p["clip_start"],
                    "clip_dur": p["clip_dur"],
                    "source_segment_id": p.get("source_segment_id"),
                    "source_id": p["source_id"],
                    "human_src": p["human_src"],
                    "robot_src": p["robot_src"],
                    "video": f"video/{name}",
                    "control_video": f"control_video/{name}",
                    "input_role": input_role,
                    "target_role": target_role,
                }
                if args.data_type == "blur_r2r":
                    p["manifest"].update({
                        "control_degrade": "sam2_blur",
                        "blur_ksize": args.blur_ksize,
                        "blur_pixel_expand": args.blur_pixel_expand,
                    })
                to_process.append(p)

        t0 = time.time()
        task_meta: dict[str, list[dict]] = {task: [] for task in by_task}
        task_manifest: dict[str, list[dict]] = {task: [] for task in by_task}

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(process_pair, p, args.compare, args.resume,
                            num_frames, args.hand_patch, args.hand_weight,
                            args.blur_mask_root, args.blur_ksize,
                            args.blur_pixel_expand): p
                for p in to_process
            }
            done = 0
            total = len(to_process)
            for fut in as_completed(futures):
                p = futures[fut]
                meta = fut.result()
                task_meta[p["task"]].append(meta)
                task_manifest[p["task"]].append(p["manifest"])
                done += 1
                if done % 20 == 0 or done == total:
                    print(f"  {done}/{total}", flush=True)

        print(f"  done in {time.time() - t0:.1f}s")

        index_rows: list[dict] = []
        for task, metas in task_meta.items():
            if not metas:
                continue
            metas.sort(key=lambda m: m["video"])
            task_dir = os.path.join(sec_dir, task)
            csv_path = os.path.join(task_dir, "metadata.csv")
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(
                    f, fieldnames=["video", "prompt", "control_video"])
                writer.writeheader()
                writer.writerows(metas)
            print(f"  metadata: {csv_path} ({len(metas)} rows)")
            manifest_rows = sorted(task_manifest[task], key=lambda m: m["source_id"])
            manifest_path = os.path.join(task_dir, "manifest.jsonl")
            _write_jsonl(manifest_path, manifest_rows)
            index_rows.extend(manifest_rows)
            print(f"  manifest: {manifest_path} ({len(manifest_rows)} rows)")

        index_path = os.path.join(sec_dir, "index.jsonl")
        _write_jsonl(index_path, sorted(index_rows, key=lambda m: m["source_id"]))
        print(f"  index: {index_path} ({len(index_rows)} entries)")

    elapsed = time.time() - t_total
    print(f"\nDone in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
