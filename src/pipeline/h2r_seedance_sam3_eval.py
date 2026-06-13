"""Evaluate H2R Seedance robot-to-human edits with SAM3.1 hand masks.

This evaluator is intentionally separate from the Seedance API caller. It reads
Seedance result jsonl files, runs SAM3.1 text segmentation on the generated
``final`` videos with a human-hand prompt, and compares the resulting hand mask
against the original guided marker region used for the edit.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from src.core.config import MAIN_ROOT
from src.pipeline.g1_sam3_precompute import (
    patch_sam31_start_session,
    run_prompt_request_once,
)
from src.pipeline.h2r_sam3_precompute import safe_name, write_json


DEFAULT_OUTPUT_ROOT = Path(MAIN_ROOT) / "tmp" / "h2r_seedance_sam3_eval"
DEFAULT_TMP_DIR = Path(MAIN_ROOT) / "tmp" / "h2r_seedance_sam3_eval_frames"
DEFAULT_RESULT_JSONL = (
    Path(MAIN_ROOT)
    / "tmp"
    / "h2r_seedance_sam3_red_task_prompts"
    / "seedance_results.jsonl"
)
DEFAULT_PROMPT_LIST = "human hand,bare human hand,human fingers"
DEFAULT_PROMPT_FRAME_POSITIONS = "middle,first"


@dataclass(frozen=True)
class VideoInfo:
    frame_count: int
    fps: float
    width: int
    height: int


@dataclass(frozen=True)
class EvalJob:
    sample_id: str
    final_video: Path
    target_video: Path
    target_mode: str
    prompt: str
    source_result: dict[str, Any]


def parse_csv_list(value: str) -> list[str]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise ValueError(f"expected comma-separated value, got {value!r}")
    return items


def probe_video(path: Path) -> VideoInfo:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if frame_count <= 0:
        raise ValueError(f"video has no frames: {path}")
    if fps <= 0.0:
        raise ValueError(f"video has invalid fps={fps}: {path}")
    if width <= 0 or height <= 0:
        raise ValueError(f"video has invalid size {width}x{height}: {path}")
    return VideoInfo(frame_count=frame_count, fps=fps, width=width, height=height)


def selected_indices(frame_count: int, stride: int, max_frames: int) -> tuple[int, ...]:
    if stride <= 0:
        raise ValueError(f"--frame-stride must be positive, got {stride}")
    if max_frames < 0:
        raise ValueError(f"--max-frames must be non-negative, got {max_frames}")
    indices = tuple(range(0, frame_count, stride))
    if max_frames > 0:
        indices = indices[:max_frames]
    if not indices:
        raise ValueError(f"no frames selected from frame_count={frame_count}")
    return indices


def build_chunks(indices: tuple[int, ...], chunk_size: int) -> list[tuple[int, ...]]:
    if chunk_size <= 0:
        raise ValueError(f"--chunk-size must be positive, got {chunk_size}")
    return [indices[start : start + chunk_size] for start in range(0, len(indices), chunk_size)]


def extract_frames(video_path: Path, source_indices: tuple[int, ...], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.jpg"):
        old.unlink()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    for local_index, source_index in enumerate(source_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(source_index))
        ok, bgr = cap.read()
        if not ok:
            cap.release()
            raise RuntimeError(f"failed to read frame {source_index} from {video_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        Image.fromarray(rgb).save(out_dir / f"{local_index:05d}.jpg", quality=95)
    cap.release()


def read_video_bgr(path: Path) -> tuple[np.ndarray, VideoInfo]:
    info = probe_video(path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if len(frames) != info.frame_count:
        raise ValueError(
            f"OpenCV decoded {len(frames)} frames but probe reported {info.frame_count}: {path}"
        )
    return np.stack(frames, axis=0), info


def color_marker_mask_from_frame(frame_bgr: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    r = rgb[:, :, 0].astype(np.float32)
    g = rgb[:, :, 1].astype(np.float32)
    b = rgb[:, :, 2].astype(np.float32)
    red = (r > 145.0) & (g < 125.0) & (b < 125.0) & (r > g * 1.35) & (r > b * 1.35)
    magenta = (r > 130.0) & (b > 130.0) & (g < 130.0)
    cyan = (g > 130.0) & (b > 130.0) & (r < 130.0)
    yellow = (r > 150.0) & (g > 130.0) & (b < 120.0)
    green = (g > 145.0) & (r < 130.0) & (b < 130.0)
    return (red | magenta | cyan | yellow | green).astype(np.uint8)


def binary_mask_from_frame(frame_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return (gray > 127).astype(np.uint8)


def target_mask_from_video(
    target_video: Path,
    mode: str,
    output_frame_count: int,
    output_hw: tuple[int, int],
) -> np.ndarray:
    frames, info = read_video_bgr(target_video)
    out_h, out_w = output_hw
    masks = []
    for output_index in range(output_frame_count):
        if output_frame_count == 1:
            target_index = 0
        else:
            target_index = int(round(output_index * (info.frame_count - 1) / (output_frame_count - 1)))
        frame = frames[target_index]
        resized = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
        if mode == "binary":
            masks.append(binary_mask_from_frame(resized))
        elif mode == "color_marker":
            masks.append(color_marker_mask_from_frame(resized))
        else:
            raise ValueError(f"unsupported target mask mode: {mode}")
    return np.stack(masks, axis=0)


def prompt_frame_indices(frame_count: int, positions: list[str]) -> list[int]:
    out = []
    for item in positions:
        key = item.strip().lower()
        if key == "first":
            value = 0
        elif key == "middle":
            value = frame_count // 2
        elif key == "last":
            raise ValueError(
                "--prompt-frame-positions=last is disabled because SAM3.1 backward "
                "propagation from the final frame can fail; use middle or an integer "
                "before the last frame"
            )
        else:
            value = int(key)
        if value < 0 or value >= frame_count - 1:
            raise ValueError(
                f"prompt frame {value} out of range for stable SAM3.1 propagation "
                f"with chunk length {frame_count}; expected [0, {frame_count - 2}]"
            )
        if value not in out:
            out.append(value)
    return out


def run_chunk_hand_mask(
    predictor,
    frame_dir: Path,
    prompts: list[str],
    prompt_positions: list[str],
    frame_count: int,
    output_prob_thresh: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    attempts = []
    for prompt in prompts:
        for prompt_frame in prompt_frame_indices(frame_count, prompt_positions):
            masks, summary = run_prompt_request_once(
                predictor,
                frame_dir,
                {"text": prompt},
                prompt_frame,
                frame_count,
                output_prob_thresh,
            )
            area_sum = int(masks.astype(bool).sum())
            attempts.append(
                {
                    "prompt": prompt,
                    "prompt_frame": int(prompt_frame),
                    "area_sum": area_sum,
                    "nonempty_frames": int(
                        np.count_nonzero(masks.reshape(len(masks), -1).sum(axis=1))
                    ),
                }
            )
            if area_sum > 0:
                return masks, {
                    "selected_prompt": prompt,
                    "selected_prompt_frame": int(prompt_frame),
                    "attempts": attempts,
                    "frames": summary,
                }
    return np.zeros((frame_count, *Image.open(frame_dir / "00000.jpg").size[::-1]), dtype=np.uint8), {
        "selected_prompt": "",
        "selected_prompt_frame": -1,
        "attempts": attempts,
        "frames": [],
    }


def load_predictor(args: argparse.Namespace):
    from sam3.model_builder import build_sam3_predictor

    kwargs = {
        "version": args.sam3_version,
        "compile": False,
        "warm_up": False,
        "use_fa3": False,
        "use_rope_real": False,
        "async_loading_frames": False,
    }
    if args.sam3_version == "sam3.1":
        kwargs.update(
            {
                "max_num_objects": args.max_num_objects,
                "multiplex_count": args.multiplex_count,
                "default_output_prob_thresh": args.output_prob_thresh,
            }
        )
    predictor = build_sam3_predictor(**kwargs)
    if args.sam3_version == "sam3.1":
        patch_sam31_start_session(predictor)
    return predictor


def load_jobs(result_jsonl: Path, limit: int) -> list[EvalJob]:
    if not result_jsonl.is_file():
        raise FileNotFoundError(f"result jsonl not found: {result_jsonl}")
    jobs = []
    with result_jsonl.open("r", encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            if row.get("status") != "ok":
                continue
            final_path = Path(row["final_output_path"])
            target_mode = "color_marker"
            target_value = row["input_path"]
            for key in ("target_mask_video_path", "mask_video_path"):
                if key in row and row[key]:
                    target_value = row[key]
                    target_mode = "binary"
                    break
            target_path = Path(target_value)
            if not final_path.is_absolute():
                final_path = Path(MAIN_ROOT) / final_path
            if not target_path.is_absolute():
                target_path = Path(MAIN_ROOT) / target_path
            jobs.append(
                EvalJob(
                    sample_id=str(row["sample_id"]),
                    final_video=final_path,
                    target_video=target_path,
                    target_mode=target_mode,
                    prompt=str(row.get("prompt", "")),
                    source_result=row,
                )
            )
            if limit > 0 and len(jobs) >= limit:
                break
    if not jobs:
        raise ValueError(f"no ok Seedance rows found in {result_jsonl}")
    return jobs


def mask_metrics(
    hand_masks: np.ndarray,
    target_masks: np.ndarray,
    indices: tuple[int, ...],
) -> dict[str, Any]:
    hand = hand_masks[list(indices)].astype(bool)
    target = target_masks[list(indices)].astype(bool)
    hand_area = hand.reshape(len(hand), -1).sum(axis=1)
    target_area = target.reshape(len(target), -1).sum(axis=1)
    inter = np.logical_and(hand, target).reshape(len(hand), -1).sum(axis=1)
    union = np.logical_or(hand, target).reshape(len(hand), -1).sum(axis=1)
    hand_nonzero = hand_area > 0
    target_nonzero = target_area > 0
    valid_iou = union > 0
    return {
        "frame_count": int(len(indices)),
        "hand_nonzero_frames": int(hand_nonzero.sum()),
        "target_nonzero_frames": int(target_nonzero.sum()),
        "both_nonzero_frames": int(np.logical_and(hand_nonzero, target_nonzero).sum()),
        "hand_nonzero_ratio": float(hand_nonzero.mean()) if len(hand_nonzero) else 0.0,
        "target_nonzero_ratio": float(target_nonzero.mean()) if len(target_nonzero) else 0.0,
        "both_given_target_ratio": float(
            np.logical_and(hand_nonzero, target_nonzero).sum() / target_nonzero.sum()
        )
        if target_nonzero.sum() > 0
        else 0.0,
        "mean_hand_area_px": float(hand_area.mean()) if len(hand_area) else 0.0,
        "mean_target_area_px": float(target_area.mean()) if len(target_area) else 0.0,
        "mean_intersection_px": float(inter.mean()) if len(inter) else 0.0,
        "mean_iou": float(np.mean(inter[valid_iou] / union[valid_iou])) if valid_iou.any() else 0.0,
        "hand_on_target_ratio": float(inter.sum() / hand_area.sum()) if hand_area.sum() > 0 else 0.0,
        "target_covered_by_hand_ratio": float(inter.sum() / target_area.sum())
        if target_area.sum() > 0
        else 0.0,
    }


def overlay_frames(
    frames_bgr: np.ndarray,
    hand_masks: np.ndarray,
    target_masks: np.ndarray,
    indices: tuple[int, ...],
) -> list[np.ndarray]:
    frames = []
    for frame_index in indices:
        frame = frames_bgr[frame_index].copy()
        hand = hand_masks[frame_index].astype(bool)
        target = target_masks[frame_index].astype(np.uint8)
        frame[hand] = (
            frame[hand].astype(np.float32) * 0.45 + np.array([0, 255, 0], dtype=np.float32) * 0.55
        ).astype(np.uint8)
        contours, _ = cv2.findContours(target, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(frame, contours, -1, (0, 0, 255), 1)
        frames.append(frame)
    return frames


def write_mp4(frames_bgr: list[np.ndarray], out_path: Path, fps: float) -> None:
    if not frames_bgr:
        raise ValueError(f"no frames to write: {out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames_bgr[0].shape[:2]
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        max(1, int(round(fps))),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"cannot open video writer: {out_path}")
    for frame in frames_bgr:
        if frame.shape[:2] != (height, width):
            writer.release()
            raise ValueError(f"frame shape mismatch while writing {out_path}")
        writer.write(frame)
    writer.release()


def evaluate_job(
    predictor,
    job: EvalJob,
    args: argparse.Namespace,
    prompts: list[str],
    prompt_positions: list[str],
) -> dict[str, Any]:
    final_frames, final_info = read_video_bgr(job.final_video)
    indices = selected_indices(final_info.frame_count, args.frame_stride, args.max_frames)
    chunks = build_chunks(indices, args.chunk_size)
    target_masks = target_mask_from_video(
        job.target_video,
        job.target_mode,
        final_info.frame_count,
        (final_info.height, final_info.width),
    )
    hand_masks = np.zeros((final_info.frame_count, final_info.height, final_info.width), dtype=np.uint8)
    chunk_rows = []

    sample_dir = args.output_root / safe_name(job.sample_id)
    for chunk_index, chunk_indices in enumerate(chunks):
        frame_dir = args.tmp_dir / safe_name(job.sample_id) / f"chunk_{chunk_index:04d}"
        extract_frames(job.final_video, chunk_indices, frame_dir)
        masks, chunk_summary = run_chunk_hand_mask(
            predictor,
            frame_dir,
            prompts,
            prompt_positions,
            len(chunk_indices),
            args.output_prob_thresh,
        )
        for local_index, source_index in enumerate(chunk_indices):
            hand_masks[source_index] = np.maximum(hand_masks[source_index], masks[local_index])
        chunk_rows.append(
            {
                "chunk_index": int(chunk_index),
                "source_indices": [int(index) for index in chunk_indices],
                "selected_prompt": chunk_summary["selected_prompt"],
                "selected_prompt_frame": int(chunk_summary["selected_prompt_frame"]),
                "attempts": chunk_summary["attempts"],
            }
        )
    metrics = mask_metrics(hand_masks, target_masks, indices)

    npz_path = sample_dir / "sam3_hand_eval.npz"
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        npz_path,
        hand_masks=hand_masks,
        target_masks=target_masks,
        selected_frames=np.array(indices, dtype=np.int32),
        prompts=np.array(prompts),
        prompt_positions=np.array(prompt_positions),
        sample_id=np.array(job.sample_id),
        final_video=np.array(str(job.final_video)),
        target_video=np.array(str(job.target_video)),
        target_mode=np.array(job.target_mode),
    )
    overlay_path = sample_dir / "sam3_hand_green_target_red_contour.mp4"
    if args.write_overlay:
        write_mp4(
            overlay_frames(final_frames, hand_masks, target_masks, indices),
            overlay_path,
            final_info.fps / args.frame_stride,
        )

    summary = {
        "sample_id": job.sample_id,
        "status": "ok",
        "final_video": str(job.final_video),
        "target_video": str(job.target_video),
        "target_mode": job.target_mode,
        "source_prompt": job.prompt,
        "final_info": {
            "frame_count": final_info.frame_count,
            "fps": final_info.fps,
            "width": final_info.width,
            "height": final_info.height,
        },
        "selected_frames": [int(index) for index in indices],
        "frame_stride": args.frame_stride,
        "chunk_size": args.chunk_size,
        "sam3_prompt_list": prompts,
        "sam3_prompt_frame_positions": prompt_positions,
        "metrics": metrics,
        "npz_path": str(npz_path),
        "overlay_path": str(overlay_path) if args.write_overlay else "",
        "chunks": chunk_rows,
    }
    write_json(sample_dir / "sam3_hand_eval.json", summary)
    if not args.keep_frames:
        shutil.rmtree(args.tmp_dir / safe_name(job.sample_id), ignore_errors=True)
    return summary


def write_summary_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "sample_id",
        "hand_nonzero_ratio",
        "both_given_target_ratio",
        "mean_iou",
        "hand_on_target_ratio",
        "target_covered_by_hand_ratio",
        "mean_hand_area_px",
        "mean_target_area_px",
        "overlay_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            metrics = row["metrics"]
            writer.writerow(
                {
                    "sample_id": row["sample_id"],
                    "hand_nonzero_ratio": metrics["hand_nonzero_ratio"],
                    "both_given_target_ratio": metrics["both_given_target_ratio"],
                    "mean_iou": metrics["mean_iou"],
                    "hand_on_target_ratio": metrics["hand_on_target_ratio"],
                    "target_covered_by_hand_ratio": metrics["target_covered_by_hand_ratio"],
                    "mean_hand_area_px": metrics["mean_hand_area_px"],
                    "mean_target_area_px": metrics["mean_target_area_px"],
                    "overlay_path": row["overlay_path"],
                }
            )


def print_gpu_memory(label: str) -> None:
    query = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.used,memory.free,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    print(f"GPU memory {label}:", flush=True)
    result = subprocess.run(query, check=True, capture_output=True, text=True)
    for line in result.stdout.strip().splitlines():
        print(f"  {line}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run SAM3.1 human-hand segmentation on H2R Seedance outputs and compare with marker masks."
    )
    parser.add_argument("--result-jsonl", type=Path, default=DEFAULT_RESULT_JSONL)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--tmp-dir", type=Path, default=DEFAULT_TMP_DIR)
    parser.add_argument("--prompt-list", default=DEFAULT_PROMPT_LIST)
    parser.add_argument("--prompt-frame-positions", default=DEFAULT_PROMPT_FRAME_POSITIONS)
    parser.add_argument("--sam3-version", default="sam3.1", choices=["sam3", "sam3.1"])
    parser.add_argument("--max-num-objects", type=int, default=2)
    parser.add_argument("--multiplex-count", type=int, default=16)
    parser.add_argument("--output-prob-thresh", type=float, default=0.5)
    parser.add_argument("--frame-stride", type=int, default=5)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--chunk-size", type=int, default=17)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--write-overlay", action="store_true")
    parser.add_argument("--keep-frames", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.result_jsonl = args.result_jsonl if args.result_jsonl.is_absolute() else Path(MAIN_ROOT) / args.result_jsonl
    args.output_root = args.output_root if args.output_root.is_absolute() else Path(MAIN_ROOT) / args.output_root
    args.tmp_dir = args.tmp_dir if args.tmp_dir.is_absolute() else Path(MAIN_ROOT) / args.tmp_dir
    if args.max_num_objects <= 0:
        raise ValueError(f"--max-num-objects must be positive, got {args.max_num_objects}")
    if args.multiplex_count <= 0:
        raise ValueError(f"--multiplex-count must be positive, got {args.multiplex_count}")
    if args.output_prob_thresh <= 0.0 or args.output_prob_thresh >= 1.0:
        raise ValueError(f"--output-prob-thresh must be in (0,1), got {args.output_prob_thresh}")

    prompts = parse_csv_list(args.prompt_list)
    prompt_positions = parse_csv_list(args.prompt_frame_positions)
    jobs = load_jobs(args.result_jsonl, args.limit)
    print("H2R Seedance SAM3.1 hand evaluation")
    print(f"  jobs:        {len(jobs)}")
    print(f"  result jsonl:{args.result_jsonl}")
    print(f"  output:      {args.output_root}")
    print(f"  prompts:     {prompts}")
    print(f"  prompt pos:  {prompt_positions}")
    print(f"  stride/chunk:{args.frame_stride}/{args.chunk_size}")
    print(f"  dry_run:     {args.dry_run}")

    plan = {
        "status": "dry_run" if args.dry_run else "planned",
        "result_jsonl": str(args.result_jsonl),
        "output_root": str(args.output_root),
        "prompt_list": prompts,
        "prompt_frame_positions": prompt_positions,
        "frame_stride": args.frame_stride,
        "chunk_size": args.chunk_size,
        "jobs": [
            {
                "sample_id": job.sample_id,
                "final_video": str(job.final_video),
                "target_video": str(job.target_video),
                "target_mode": job.target_mode,
            }
            for job in jobs
        ],
    }
    write_json(args.output_root / "run_plan.json", plan)
    if args.dry_run:
        print(f"run plan: {args.output_root / 'run_plan.json'}")
        return

    print_gpu_memory("before SAM3 load")
    predictor = load_predictor(args)
    print_gpu_memory("after SAM3 load")
    t0 = time.time()
    rows = [evaluate_job(predictor, job, args, prompts, prompt_positions) for job in jobs]
    summary = {
        "status": "complete",
        "elapsed_sec": round(time.time() - t0, 1),
        "result_jsonl": str(args.result_jsonl),
        "output_root": str(args.output_root),
        "prompt_list": prompts,
        "prompt_frame_positions": prompt_positions,
        "frame_stride": args.frame_stride,
        "chunk_size": args.chunk_size,
        "ok": len(rows),
        "rows": [
            {
                "sample_id": row["sample_id"],
                "metrics": row["metrics"],
                "overlay_path": row["overlay_path"],
            }
            for row in rows
        ],
    }
    write_json(args.output_root / "summary.json", summary)
    write_summary_csv(rows, args.output_root / "summary.csv")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
