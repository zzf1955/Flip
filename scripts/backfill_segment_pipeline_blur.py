#!/usr/bin/env python3
"""Backfill original and blur_r2r preview videos for segment_pipeline output.

This script does not run FK, SAM2, inpaint, or human overlay. It only reads:
  output/segment_pipeline/<task>/epXXX/segXX/05_sam2_postproc.mp4
  training_data/segment/<task>/epXXX/segXX_video.mp4

and writes:
  output/segment_pipeline/<task>/epXXX/segXX/00_original.mp4
  output/segment_pipeline/<task>/epXXX/segXX/08_blur_r2r_control.mp4
"""

from __future__ import annotations

import argparse
import os
import re
import sys

import av
import cv2
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.core.config import SEGMENT_DIR, SEGMENT_PIPELINE_DIR
from src.core.data import close_video, open_video_writer, write_frame

DEFAULT_BLUR_R2R_KSIZE = 51
DEFAULT_BLUR_R2R_PIXEL_EXPAND = 16
MASK_NAME = "05_sam2_postproc.mp4"
ORIGINAL_NAME = "00_original.mp4"
BLUR_NAME = "08_blur_r2r_control.mp4"


def _exists(path: str) -> bool:
    return os.path.isfile(path) and os.path.getsize(path) > 0


def _parse_csv_set(values: list[str] | None) -> set[str] | None:
    if not values:
        return None
    out: set[str] = set()
    for value in values:
        out.update(v.strip() for v in value.split(",") if v.strip())
    return out or None


def read_video_bgr(path: str, max_frames: int | None = None) -> list[np.ndarray]:
    container = av.open(path)
    stream = container.streams.video[0]
    frames = []
    for i, frame in enumerate(container.decode(stream)):
        if max_frames is not None and i >= max_frames:
            break
        frames.append(frame.to_ndarray(format="bgr24"))
    container.close()
    if not frames:
        raise RuntimeError(f"video has no frames: {path}")
    return frames


def read_binary_masks(mask_video: str) -> list[np.ndarray]:
    frames = read_video_bgr(mask_video)
    masks = []
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        masks.append((gray > 128).astype(np.uint8) * 255)
    return masks


def write_video_bgr(frames: list[np.ndarray], path: str, fps: int):
    if not frames:
        raise ValueError(f"no frames to write: {path}")
    h, w = frames[0].shape[:2]
    container, stream = open_video_writer(path, w, h, fps=fps)
    for frame in frames:
        write_frame(container, stream, frame)
    close_video(container, stream)


def soften_blur_mask(mask: np.ndarray, pixel_expand: int) -> np.ndarray:
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
    blurred = cv2.GaussianBlur(frame, (ksize, ksize), 0)
    alpha = soft_mask.astype(np.float32) / 255.0
    out = alpha[..., None] * blurred + (1.0 - alpha[..., None]) * frame
    return np.clip(out, 0, 255).astype(np.uint8)


def write_blur_video(frames: list[np.ndarray], masks: list[np.ndarray],
                     path: str, fps: int, blur_ksize: int,
                     pixel_expand: int):
    if len(frames) != len(masks):
        raise ValueError(
            f"frame/mask count mismatch: frames={len(frames)}, "
            f"masks={len(masks)}, out={path}")
    if not frames:
        raise ValueError(f"no frames to write: {path}")

    h, w = frames[0].shape[:2]
    container, stream = open_video_writer(path, w, h, fps=fps)
    for frame, mask in zip(frames, masks):
        if mask.shape[:2] != frame.shape[:2]:
            raise ValueError(
                f"mask/frame shape mismatch: mask={mask.shape[:2]}, "
                f"frame={frame.shape[:2]}, out={path}")
        soft = soften_blur_mask(mask, pixel_expand=pixel_expand)
        write_frame(container, stream, blur_frame_in_mask(frame, soft, blur_ksize))
    close_video(container, stream)


def iter_mask_records(pipeline_root: str, tasks: set[str] | None,
                      episodes: set[str] | None):
    ep_re = re.compile(r"^ep\d+$")
    seg_re = re.compile(r"^seg\d+$")

    for task in sorted(os.listdir(pipeline_root)):
        if tasks is not None and task not in tasks:
            continue
        task_dir = os.path.join(pipeline_root, task)
        if not os.path.isdir(task_dir):
            continue
        for ep in sorted(os.listdir(task_dir)):
            if not ep_re.match(ep):
                continue
            if episodes is not None and ep not in episodes:
                continue
            ep_dir = os.path.join(task_dir, ep)
            for seg in sorted(os.listdir(ep_dir)):
                if not seg_re.match(seg):
                    continue
                seg_dir = os.path.join(ep_dir, seg)
                mask_path = os.path.join(seg_dir, MASK_NAME)
                if _exists(mask_path):
                    yield {
                        "task": task,
                        "episode": ep,
                        "seg": seg,
                        "seg_dir": seg_dir,
                        "mask_path": mask_path,
                    }


def process_one(record: dict, segment_root: str, fps: int,
                blur_ksize: int, pixel_expand: int,
                overwrite: bool, dry_run: bool) -> tuple[bool, str]:
    task = record["task"]
    ep = record["episode"]
    seg = record["seg"]
    seg_video = os.path.join(segment_root, task, ep, f"{seg}_video.mp4")
    original_out = os.path.join(record["seg_dir"], ORIGINAL_NAME)
    blur_out = os.path.join(record["seg_dir"], BLUR_NAME)

    if not _exists(seg_video):
        raise FileNotFoundError(f"segment video not found: {seg_video}")

    need_original = overwrite or not _exists(original_out)
    need_blur = overwrite or not _exists(blur_out)
    if not need_original and not need_blur:
        return False, f"{task}/{ep}/{seg} already complete"

    if dry_run:
        actions = []
        if need_original:
            actions.append(ORIGINAL_NAME)
        if need_blur:
            actions.append(BLUR_NAME)
        return True, f"{task}/{ep}/{seg} would write {', '.join(actions)}"

    masks = read_binary_masks(record["mask_path"])
    frames = read_video_bgr(seg_video, max_frames=len(masks))
    if len(frames) != len(masks):
        raise ValueError(
            f"segment/mask frame count mismatch for {task}/{ep}/{seg}: "
            f"frames={len(frames)}, masks={len(masks)}")

    if need_original:
        write_video_bgr(frames, original_out, fps=fps)
    if need_blur:
        write_blur_video(
            frames, masks, blur_out, fps=fps,
            blur_ksize=blur_ksize, pixel_expand=pixel_expand)

    actions = []
    if need_original:
        actions.append(ORIGINAL_NAME)
    if need_blur:
        actions.append(BLUR_NAME)
    return True, f"{task}/{ep}/{seg} wrote {', '.join(actions)}"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Backfill 00_original.mp4 and 08_blur_r2r_control.mp4 "
                    "from existing segment_pipeline masks.")
    parser.add_argument("--pipeline-root", default=SEGMENT_PIPELINE_DIR,
                        help="Root of existing segment_pipeline outputs")
    parser.add_argument("--segment-root", default=SEGMENT_DIR,
                        help="Root of source segment videos")
    parser.add_argument("--tasks", nargs="+", default=None,
                        help="Task names to process; accepts space or comma separated values")
    parser.add_argument("--episodes", nargs="+", default=None,
                        help="Episode dirs to process, e.g. ep000 or 000")
    parser.add_argument("--limit", type=int, default=0,
                        help="Maximum records to process after filtering; 0 means all")
    parser.add_argument("--fps", type=int, default=30,
                        help="Output fps for generated videos")
    parser.add_argument("--blur-ksize", type=int,
                        default=DEFAULT_BLUR_R2R_KSIZE,
                        help="Odd Gaussian kernel size for blur control video")
    parser.add_argument("--blur-pixel-expand", type=int,
                        default=DEFAULT_BLUR_R2R_PIXEL_EXPAND,
                        help="Pixel mask expansion before blur feathering")
    parser.add_argument("--overwrite", action="store_true",
                        help="Regenerate outputs even if they already exist")
    parser.add_argument("--dry-run", action="store_true",
                        help="List work without writing videos")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.blur_ksize <= 0 or args.blur_ksize % 2 == 0:
        raise ValueError(
            f"--blur-ksize must be a positive odd integer: {args.blur_ksize}")
    if args.blur_pixel_expand < 0:
        raise ValueError(
            f"--blur-pixel-expand must be non-negative: {args.blur_pixel_expand}")
    if not os.path.isdir(args.pipeline_root):
        raise FileNotFoundError(f"pipeline root not found: {args.pipeline_root}")
    if not os.path.isdir(args.segment_root):
        raise FileNotFoundError(f"segment root not found: {args.segment_root}")

    tasks = _parse_csv_set(args.tasks)
    episodes = _parse_csv_set(args.episodes)
    if episodes is not None:
        episodes = {ep if ep.startswith("ep") else f"ep{int(ep):03d}"
                    for ep in episodes}

    records = list(iter_mask_records(args.pipeline_root, tasks, episodes))
    if args.limit > 0:
        records = records[:args.limit]

    print("Backfill segment_pipeline original + blur_r2r")
    print(f"  Pipeline root: {args.pipeline_root}")
    print(f"  Segment root: {args.segment_root}")
    print(f"  Records: {len(records)}")
    print(f"  Blur R2R: ksize={args.blur_ksize}, pixel_expand={args.blur_pixel_expand}")
    if args.dry_run:
        print("  Dry run: yes")
    if args.overwrite:
        print("  Overwrite: yes")

    done = 0
    skipped = 0
    for i, record in enumerate(records, 1):
        changed, message = process_one(
            record, args.segment_root, args.fps, args.blur_ksize,
            args.blur_pixel_expand, args.overwrite, args.dry_run)
        if changed:
            done += 1
        else:
            skipped += 1
        print(f"[{i}/{len(records)}] {message}")

    print(f"Done: {done} changed, {skipped} skipped")


if __name__ == "__main__":
    main()
