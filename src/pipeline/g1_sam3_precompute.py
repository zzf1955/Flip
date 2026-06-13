"""Precompute SAM3/SAM3.1 masks for G1 segment videos.

This is the G1 counterpart of ``h2r_sam3_precompute``. It reads maintained
G1 segment videos:

  training_data/segment/<task>/<episode>/seg*_video.mp4

and writes explicit SAM3 mask artifacts:

  training_data/g1_sam3_mask/<task>/<episode>/<seg>.npz

SAM3 inference is deliberately an explicit precompute step. Downstream blur
builders should consume these artifacts and fail when required frames are not
covered instead of falling back to SAM2.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from src.core.config import ALL_TASKS, MAIN_ROOT, TRAINING_TASKS
from src.pipeline.h2r_sam3_precompute import (
    outputs_to_mask,
    patch_sam31_start_session,
    write_json,
)


DEFAULT_TRAINING_ROOT = Path(MAIN_ROOT) / "training_data"
DEFAULT_SEGMENT_ROOT = DEFAULT_TRAINING_ROOT / "segment"
DEFAULT_OUTPUT_ROOT = DEFAULT_TRAINING_ROOT / "g1_sam3_mask"
DEFAULT_TMP_DIR = Path(MAIN_ROOT) / "tmp" / "g1_sam3_precompute_frames"
DEFAULT_SAM2_MASK_ROOT = DEFAULT_TRAINING_ROOT / "sam2_mask"


@dataclass(frozen=True)
class VideoInfo:
    frame_count: int
    fps: float
    width: int
    height: int


@dataclass(frozen=True)
class SegmentInfo:
    task: str
    episode: str
    seg: str
    video_path: Path
    frame_count: int
    fps: float
    height: int
    width: int


@dataclass(frozen=True)
class FrameChunk:
    chunk_index: int
    source_indices: tuple[int, ...]


@dataclass(frozen=True)
class BBoxPrompt:
    frame_index: int
    xywh: tuple[float, float, float, float]
    area_px: int


def short_task_name(task: str) -> str:
    return task.strip().replace("G1_WBT_", "")


def parse_csv(value: str) -> list[str]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise ValueError(f"expected comma-separated value, got {value!r}")
    return items


def expand_tasks(task_spec: str) -> list[str]:
    groups = {
        "all": TRAINING_TASKS,
        "training": TRAINING_TASKS,
        "canonical": TRAINING_TASKS,
        "inspire": [task for task in ALL_TASKS if "G1_WBT_Inspire_" in task],
        "brainco": [task for task in ALL_TASKS if "G1_WBT_Brainco_" in task],
    }
    key = task_spec.lower()
    if key in groups:
        return [short_task_name(task) for task in groups[key]]
    return [short_task_name(task) for task in parse_csv(task_spec)]


def safe_name(value: str) -> str:
    name = re.sub(r"[^a-zA-Z0-9._-]+", "_", value.strip()).strip("_")
    if not name:
        raise ValueError(f"value cannot be converted to a safe name: {value!r}")
    return name


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


def discover_segments(args: argparse.Namespace, tasks: list[str]) -> list[SegmentInfo]:
    segments: list[SegmentInfo] = []
    for task in tasks:
        if args.max_segments > 0 and len(segments) >= args.max_segments:
            break
        task_dir = args.segment_root / task
        if not task_dir.is_dir():
            raise FileNotFoundError(f"G1 segment task dir not found: {task_dir}")
        task_count = 0
        for video_path in sorted(task_dir.glob("ep*/seg*_video.mp4")):
            if args.max_segments > 0 and len(segments) >= args.max_segments:
                break
            if args.max_segments_per_task > 0 and task_count >= args.max_segments_per_task:
                break
            match = re.match(r"(seg\d+)_video\.mp4$", video_path.name)
            if not match:
                continue
            info = probe_video(video_path)
            segments.append(
                SegmentInfo(
                    task=task,
                    episode=video_path.parent.name,
                    seg=match.group(1),
                    video_path=video_path,
                    frame_count=info.frame_count,
                    fps=info.fps,
                    height=info.height,
                    width=info.width,
                )
            )
            task_count += 1
    segments.sort(key=lambda item: f"{item.task}/{item.episode}/{item.seg}")
    if not segments:
        raise ValueError("no G1 segment videos selected")
    return segments


def selected_frame_indices(segment: SegmentInfo, args: argparse.Namespace) -> tuple[int, ...]:
    if args.frame_stride <= 0:
        raise ValueError(f"--frame-stride must be positive, got {args.frame_stride}")
    start = args.frame_start
    if start < 0:
        raise ValueError(f"--frame-start must be non-negative, got {start}")
    if start >= segment.frame_count:
        raise ValueError(
            f"--frame-start={start} exceeds segment frame count {segment.frame_count}: "
            f"{segment.video_path}"
        )
    if args.frame_count < 0:
        raise ValueError(f"--frame-count must be non-negative, got {args.frame_count}")
    if args.frame_count == 0:
        indices = tuple(range(start, segment.frame_count, args.frame_stride))
        if not indices:
            raise ValueError(f"no frames selected from {segment.video_path}")
        return indices
    count = args.frame_count
    if count <= 0:
        raise ValueError(f"--frame-count must be positive or zero for all remaining frames, got {count}")
    stop = min(segment.frame_count, start + count * args.frame_stride)
    indices = tuple(range(start, stop, args.frame_stride))
    if len(indices) != count:
        raise ValueError(
            f"selected only {len(indices)} frames but requested {count}; "
            f"segment has {segment.frame_count} frames: {segment.video_path}"
        )
    return indices


def build_chunks(indices: tuple[int, ...], chunk_size: int, chunk_overlap: int) -> list[FrameChunk]:
    if chunk_size <= 0:
        raise ValueError(f"--chunk-size must be positive, got {chunk_size}")
    if chunk_overlap < 0:
        raise ValueError(f"--chunk-overlap must be non-negative, got {chunk_overlap}")
    if chunk_overlap >= chunk_size:
        raise ValueError("--chunk-overlap must be smaller than --chunk-size")
    step = chunk_size - chunk_overlap
    chunks: list[FrameChunk] = []
    cursor = 0
    while cursor < len(indices):
        chunk = indices[cursor : cursor + chunk_size]
        chunks.append(FrameChunk(chunk_index=len(chunks), source_indices=chunk))
        if cursor + chunk_size >= len(indices):
            break
        cursor += step
    return chunks


def extract_frames(video_path: Path, source_indices: tuple[int, ...], out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.jpg"):
        old.unlink()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    frame_paths: list[Path] = []
    for out_index, source_index in enumerate(source_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, source_index)
        ok, bgr = cap.read()
        if not ok:
            cap.release()
            raise RuntimeError(f"failed to read frame {source_index} from {video_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        frame_path = out_dir / f"{out_index:05d}.jpg"
        Image.fromarray(rgb).save(frame_path, quality=95)
        frame_paths.append(frame_path)
    cap.release()
    return frame_paths


def output_path_for_segment(output_root: Path, segment: SegmentInfo) -> Path:
    return output_root / segment.task / segment.episode / f"{segment.seg}.npz"


def existing_mask_covers(
    out_path: Path,
    segment: SegmentInfo,
    required_frames: set[int],
    args: argparse.Namespace,
) -> bool:
    with np.load(out_path) as data:
        if "masks" not in data:
            raise ValueError(f"existing SAM3 npz missing 'masks': {out_path}")
        if "covered_frames" not in data:
            return False
        for key in ("prompt", "prompt_mode", "sam3_model"):
            if key not in data:
                return False
        if str(data["prompt"]) != args.prompt:
            return False
        if str(data["prompt_mode"]) != args.prompt_mode:
            return False
        if str(data["sam3_model"]) != args.sam3_version:
            return False
        masks = data["masks"]
        covered = {int(index) for index in data["covered_frames"].tolist()}
    if masks.shape != (segment.frame_count, segment.height, segment.width):
        return False
    return required_frames <= covered


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


def collect_masks_from_outputs(
    frame_dir: Path,
    outputs_by_frame: dict[int, dict],
    frame_count: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    first_frame = np.asarray(Image.open(frame_dir / "00000.jpg").convert("RGB"))
    height, width = first_frame.shape[:2]
    masks = []
    summary = []
    for frame_index in range(frame_count):
        outputs = outputs_by_frame.get(frame_index)
        mask = outputs_to_mask(outputs, height, width)
        masks.append(mask)
        row_outputs = outputs or {}
        obj_ids = row_outputs.get("out_obj_ids", [])
        if hasattr(obj_ids, "detach"):
            obj_ids = obj_ids.detach().cpu().numpy()
        obj_ids = np.asarray(obj_ids)
        summary.append(
            {
                "frame_index": frame_index,
                "num_objects": int(len(obj_ids)),
                "area": int(mask.astype(bool).sum()),
            }
        )
    return np.stack(masks, axis=0), summary


def run_prompt_request_once(
    predictor,
    frame_dir: Path,
    prompt_request: dict[str, Any],
    prompt_frame_index: int,
    frame_count: int,
    output_prob_thresh: float,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    if not 0 <= prompt_frame_index < frame_count:
        raise ValueError(
            f"prompt frame {prompt_frame_index} out of range for {frame_count} frames"
        )
    response = predictor.handle_request(
        {
            "type": "start_session",
            "resource_path": str(frame_dir),
            "offload_video_to_cpu": True,
        }
    )
    session_id = response["session_id"]
    try:
        request = {
            "type": "add_prompt",
            "session_id": session_id,
            "frame_index": prompt_frame_index,
            "output_prob_thresh": output_prob_thresh,
        }
        request.update(prompt_request)
        first = predictor.handle_request(request)
        outputs_by_frame = {int(first["frame_index"]): first["outputs"]}

        forward_count = frame_count - prompt_frame_index
        if forward_count > 0:
            for item in predictor.handle_stream_request(
                {
                    "type": "propagate_in_video",
                    "session_id": session_id,
                    "propagation_direction": "forward",
                    "start_frame_index": prompt_frame_index,
                    "max_frame_num_to_track": forward_count,
                    "output_prob_thresh": output_prob_thresh,
                }
            ):
                outputs_by_frame[int(item["frame_index"])] = item["outputs"]

        backward_count = prompt_frame_index + 1
        if prompt_frame_index > 0:
            for item in predictor.handle_stream_request(
                {
                    "type": "propagate_in_video",
                    "session_id": session_id,
                    "propagation_direction": "backward",
                    "start_frame_index": prompt_frame_index,
                    "max_frame_num_to_track": backward_count,
                    "output_prob_thresh": output_prob_thresh,
                }
            ):
                outputs_by_frame[int(item["frame_index"])] = item["outputs"]

        return collect_masks_from_outputs(frame_dir, outputs_by_frame, frame_count)
    finally:
        predictor.handle_request({"type": "close_session", "session_id": session_id})


def run_text_once(
    predictor,
    frame_dir: Path,
    prompt: str,
    frame_count: int,
    output_prob_thresh: float,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    return run_prompt_request_once(
        predictor,
        frame_dir,
        {"text": prompt},
        0,
        frame_count,
        output_prob_thresh,
    )


def run_text_clip(
    predictor,
    frame_dir: Path,
    prompt: str,
    backup_prompt: str,
    frame_count: int,
    output_prob_thresh: float,
) -> tuple[np.ndarray, str, list[dict[str, Any]]]:
    masks, summary = run_text_once(
        predictor, frame_dir, prompt, frame_count, output_prob_thresh
    )
    if int(masks.astype(bool).sum()) > 0 or not backup_prompt:
        return masks, prompt, summary
    backup_masks, backup_summary = run_text_once(
        predictor, frame_dir, backup_prompt, frame_count, output_prob_thresh
    )
    return backup_masks, backup_prompt, backup_summary


def expanded_mask_bbox(
    mask: np.ndarray,
    expand_ratio: float,
    min_area_ratio: float,
) -> BBoxPrompt | None:
    if expand_ratio < 0.0:
        raise ValueError(f"--bbox-expand-ratio must be non-negative, got {expand_ratio}")
    if min_area_ratio < 0.0:
        raise ValueError(f"--bbox-min-area-ratio must be non-negative, got {min_area_ratio}")
    height, width = mask.shape[:2]
    ys, xs = np.nonzero(mask.astype(bool))
    area = int(len(xs))
    min_area = int(round(height * width * min_area_ratio))
    if area <= 0 or area < min_area:
        return None

    x0 = float(xs.min())
    x1 = float(xs.max() + 1)
    y0 = float(ys.min())
    y1 = float(ys.max() + 1)
    box_w = max(1.0, x1 - x0)
    box_h = max(1.0, y1 - y0)
    x0 = max(0.0, x0 - box_w * expand_ratio)
    x1 = min(float(width), x1 + box_w * expand_ratio)
    y0 = max(0.0, y0 - box_h * expand_ratio)
    y1 = min(float(height), y1 + box_h * expand_ratio)
    norm = (
        float(x0 / width),
        float(y0 / height),
        float((x1 - x0) / width),
        float((y1 - y0) / height),
    )
    return BBoxPrompt(frame_index=0, xywh=norm, area_px=area)


def choose_bbox_prompt(masks: np.ndarray, args: argparse.Namespace) -> BBoxPrompt | None:
    if masks.ndim != 3:
        raise ValueError(f"expected masks shape [T,H,W], got {masks.shape}")
    if args.bbox_source == "first-frame":
        prompt = expanded_mask_bbox(
            masks[0],
            args.bbox_expand_ratio,
            args.bbox_min_area_ratio,
        )
        if prompt is None:
            return None
        return BBoxPrompt(frame_index=0, xywh=prompt.xywh, area_px=prompt.area_px)
    if args.bbox_source == "first-nonempty":
        for frame_index, mask in enumerate(masks):
            prompt = expanded_mask_bbox(
                mask,
                args.bbox_expand_ratio,
                args.bbox_min_area_ratio,
            )
            if prompt is not None:
                return BBoxPrompt(
                    frame_index=frame_index,
                    xywh=prompt.xywh,
                    area_px=prompt.area_px,
                )
        return None
    if args.bbox_source == "union":
        prompt = expanded_mask_bbox(
            np.any(masks.astype(bool), axis=0),
            args.bbox_expand_ratio,
            args.bbox_min_area_ratio,
        )
        if prompt is None:
            return None
        return BBoxPrompt(frame_index=0, xywh=prompt.xywh, area_px=prompt.area_px)
    raise ValueError(f"unsupported --bbox-source={args.bbox_source!r}")


def run_bbox_once(
    predictor,
    frame_dir: Path,
    bbox: BBoxPrompt,
    text_prompt: str | None,
    frame_count: int,
    output_prob_thresh: float,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    prompt_request: dict[str, Any] = {
        "bounding_boxes": [list(bbox.xywh)],
        "bounding_box_labels": [1],
    }
    if text_prompt is not None:
        prompt_request["text"] = text_prompt
    return run_prompt_request_once(
        predictor,
        frame_dir,
        prompt_request,
        bbox.frame_index,
        frame_count,
        output_prob_thresh,
    )


def run_sam3_chunk(
    predictor,
    frame_dir: Path,
    args: argparse.Namespace,
    frame_count: int,
) -> dict[str, Any]:
    text_masks, used_prompt, text_summary = run_text_clip(
        predictor,
        frame_dir,
        args.prompt,
        args.backup_prompt,
        frame_count,
        args.output_prob_thresh,
    )
    if args.prompt_mode == "text":
        return {
            "masks": text_masks,
            "used_prompt": used_prompt,
            "final_stage": "text",
            "frames": text_summary,
            "text_area_sum": int(text_masks.astype(bool).sum()),
        }
    if args.prompt_mode != "text_bbox":
        raise ValueError(f"unsupported --prompt-mode={args.prompt_mode!r}")

    bbox = choose_bbox_prompt(text_masks, args)
    if bbox is None:
        return {
            "masks": text_masks,
            "used_prompt": used_prompt,
            "final_stage": "text_no_bbox",
            "frames": text_summary,
            "text_area_sum": int(text_masks.astype(bool).sum()),
            "bbox": None,
        }

    bbox_masks, bbox_summary = run_bbox_once(
        predictor,
        frame_dir,
        bbox,
        used_prompt if args.bbox_include_text else None,
        frame_count,
        args.output_prob_thresh,
    )
    return {
        "masks": bbox_masks,
        "used_prompt": used_prompt,
        "final_stage": "bbox",
        "frames": bbox_summary,
        "text_frames": text_summary,
        "text_area_sum": int(text_masks.astype(bool).sum()),
            "bbox": {
                "source": args.bbox_source,
                "include_text": bool(args.bbox_include_text),
                "prompt_frame_index": bbox.frame_index,
                "xywh": list(bbox.xywh),
                "seed_area_px": bbox.area_px,
            },
        }


def frame_stats(masks: np.ndarray, selected_indices: tuple[int, ...]) -> dict[str, Any]:
    selected = masks[list(selected_indices)]
    flat_area = selected.astype(bool).reshape(len(selected), -1).sum(axis=1)
    nonempty = flat_area > 0
    mean_area = float(flat_area.mean()) if len(flat_area) else 0.0
    std_area = float(flat_area.std()) if len(flat_area) else 0.0
    return {
        "selected_frame_count": int(len(selected_indices)),
        "nonempty_frame_count": int(nonempty.sum()),
        "nonempty_frame_ratio": float(nonempty.mean()) if len(nonempty) else 0.0,
        "mean_area_px": mean_area,
        "std_area_px": std_area,
        "area_cv": float(std_area / mean_area) if mean_area > 0 else 0.0,
        "mean_area_ratio": float(mean_area / (selected.shape[1] * selected.shape[2]))
        if selected.size
        else 0.0,
    }


def sam2_metrics(
    sam3_masks: np.ndarray,
    selected_indices: tuple[int, ...],
    sam2_path: Path,
) -> dict[str, Any]:
    if not sam2_path.is_file():
        return {"sam2_mask_path": str(sam2_path), "sam2_available": False}
    with np.load(sam2_path) as data:
        if "masks" not in data:
            raise ValueError(f"SAM2 mask npz missing 'masks': {sam2_path}")
        sam2_masks = data["masks"]
    if sam2_masks.ndim != 3:
        raise ValueError(f"SAM2 masks must be [T,H,W], got {sam2_masks.shape}: {sam2_path}")
    if max(selected_indices) >= len(sam2_masks):
        raise ValueError(
            f"SAM2 mask count too short for {sam2_path}: need {max(selected_indices)}, "
            f"count={len(sam2_masks)}"
        )
    ious = []
    sam3_in_sam2 = []
    sam2_in_sam3 = []
    area_ratios = []
    for frame_index in selected_indices:
        pred = sam3_masks[frame_index].astype(bool)
        ref = sam2_masks[frame_index].astype(bool)
        inter = np.logical_and(pred, ref).sum()
        pred_area = pred.sum()
        ref_area = ref.sum()
        union = np.logical_or(pred, ref).sum()
        ious.append(float(inter / union) if union else 1.0)
        sam3_in_sam2.append(float(inter / pred_area) if pred_area else 0.0)
        sam2_in_sam3.append(float(inter / ref_area) if ref_area else 0.0)
        area_ratios.append(float(pred_area / ref_area) if ref_area else 0.0)
    return {
        "sam2_mask_path": str(sam2_path),
        "sam2_available": True,
        "sam2_mean_iou": float(np.mean(ious)),
        "sam2_mean_sam3_in_sam2": float(np.mean(sam3_in_sam2)),
        "sam2_mean_sam2_in_sam3": float(np.mean(sam2_in_sam3)),
        "sam2_mean_area_ratio": float(np.mean(area_ratios)),
    }


def make_overlay_frame(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = frame.copy()
    roi = mask.astype(bool)
    out[roi] = (0.45 * out[roi].astype(np.float32) + np.array([0, 0, 255]) * 0.55).astype(np.uint8)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, (0, 255, 255), 1)
    return out


def write_overlay_video(
    video_path: Path,
    masks: np.ndarray,
    selected_indices: tuple[int, ...],
    out_path: Path,
    fps: float,
) -> None:
    frames = []
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    for frame_index in selected_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok:
            cap.release()
            raise RuntimeError(f"failed to read frame {frame_index} from {video_path}")
        frames.append(make_overlay_frame(frame, (masks[frame_index] > 0).astype(np.uint8) * 255))
    cap.release()
    if not frames:
        raise ValueError(f"no overlay frames for {out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        max(1, int(round(fps))),
        (w, h),
    )
    if not writer.isOpened():
        raise RuntimeError(f"cannot open overlay video writer: {out_path}")
    for frame in frames:
        writer.write(frame)
    writer.release()


def process_segment(args: argparse.Namespace, predictor, segment: SegmentInfo) -> dict[str, Any]:
    selected_indices = selected_frame_indices(segment, args)
    chunks = build_chunks(selected_indices, args.chunk_size, args.chunk_overlap)
    required_frames = set(selected_indices)
    out_path = output_path_for_segment(args.output_root, segment)
    summary_path = out_path.with_suffix(".json")

    if args.resume and out_path.is_file() and existing_mask_covers(
        out_path, segment, required_frames, args
    ):
        with np.load(out_path) as data:
            masks = data["masks"]
        stats = frame_stats(masks, selected_indices)
        return {
            "task": segment.task,
            "episode": segment.episode,
            "seg": segment.seg,
            "video_path": str(segment.video_path),
            "output_path": str(out_path),
            "summary_path": str(summary_path),
            "skipped": True,
            **stats,
        }

    segment_tmp = (
        args.tmp_dir
        / safe_name(segment.task)
        / safe_name(segment.episode)
        / safe_name(segment.seg)
    )
    masks_full = np.zeros((segment.frame_count, segment.height, segment.width), dtype=np.uint8)
    covered_frames: set[int] = set()
    chunk_rows = []
    for chunk in chunks:
        frame_dir = segment_tmp / f"chunk_{chunk.chunk_index:04d}"
        frame_paths = extract_frames(segment.video_path, chunk.source_indices, frame_dir)
        chunk_result = run_sam3_chunk(predictor, frame_dir, args, len(frame_paths))
        chunk_masks = chunk_result["masks"]
        for local_index, source_index in enumerate(chunk.source_indices):
            masks_full[source_index] = np.maximum(masks_full[source_index], chunk_masks[local_index])
            covered_frames.add(source_index)
        chunk_row = {
            "chunk_index": chunk.chunk_index,
            "source_indices": list(chunk.source_indices),
            "prompt": chunk_result["used_prompt"],
            "prompt_mode": args.prompt_mode,
            "final_stage": chunk_result["final_stage"],
            "nonempty_frames": int(
                np.count_nonzero(chunk_masks.reshape(len(chunk_masks), -1).sum(axis=1))
            ),
            "area_sum": int(chunk_masks.astype(bool).sum()),
            "text_area_sum": int(chunk_result["text_area_sum"]),
            "frames": chunk_result["frames"],
        }
        if "text_frames" in chunk_result:
            chunk_row["text_frames"] = chunk_result["text_frames"]
        if "bbox" in chunk_result:
            chunk_row["bbox"] = chunk_result["bbox"]
        chunk_rows.append(chunk_row)

    stats = frame_stats(masks_full, selected_indices)
    sam2_path = args.sam2_mask_root / segment.task / segment.episode / f"{segment.seg}.npz"
    sam2 = sam2_metrics(masks_full, selected_indices, sam2_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        masks=masks_full,
        covered_frames=np.array(sorted(covered_frames), dtype=np.int32),
        selected_frames=np.array(selected_indices, dtype=np.int32),
        chunk_starts=np.array([chunk.source_indices[0] for chunk in chunks], dtype=np.int32),
        task=np.array(segment.task),
        episode=np.array(segment.episode),
        seg=np.array(segment.seg),
        video_path=np.array(str(segment.video_path)),
        source_fps=np.array(float(segment.fps), dtype=np.float32),
        source_frame_count=np.array(int(segment.frame_count), dtype=np.int32),
        sam3_model=np.array(args.sam3_version),
        prompt_mode=np.array(args.prompt_mode),
        prompt=np.array(args.prompt),
        backup_prompt=np.array(args.backup_prompt),
        bbox_source=np.array(args.bbox_source),
        bbox_expand_ratio=np.array(float(args.bbox_expand_ratio), dtype=np.float32),
        bbox_min_area_ratio=np.array(float(args.bbox_min_area_ratio), dtype=np.float32),
        bbox_include_text=np.array(bool(args.bbox_include_text)),
        chunk_size=np.array(int(args.chunk_size), dtype=np.int32),
        chunk_overlap=np.array(int(args.chunk_overlap), dtype=np.int32),
        output_prob_thresh=np.array(float(args.output_prob_thresh), dtype=np.float32),
    )
    summary = {
        "task": segment.task,
        "episode": segment.episode,
        "seg": segment.seg,
        "video_path": str(segment.video_path),
        "output_path": str(out_path),
        "summary_path": str(summary_path),
        "frame_count": segment.frame_count,
        "fps": segment.fps,
        "height": segment.height,
        "width": segment.width,
        "sam3_model": args.sam3_version,
        "prompt_mode": args.prompt_mode,
        "prompt": args.prompt,
        "backup_prompt": args.backup_prompt,
        "bbox_source": args.bbox_source,
        "bbox_expand_ratio": args.bbox_expand_ratio,
        "bbox_min_area_ratio": args.bbox_min_area_ratio,
        "bbox_include_text": bool(args.bbox_include_text),
        "chunk_size": args.chunk_size,
        "chunk_overlap": args.chunk_overlap,
        "output_prob_thresh": args.output_prob_thresh,
        "selected_frames": list(selected_indices),
        "covered_frame_count": int(len(covered_frames)),
        "chunks": chunk_rows,
        **stats,
        **sam2,
    }
    write_json(summary_path, summary)
    if args.write_overlay:
        overlay_path = summary_path.with_name(f"{summary_path.stem}_overlay.mp4")
        write_overlay_video(segment.video_path, masks_full, selected_indices, overlay_path, segment.fps)
        summary["overlay_path"] = str(overlay_path)
        write_json(summary_path, summary)
    if not args.keep_frames:
        shutil.rmtree(segment_tmp, ignore_errors=True)
    return summary


def aggregate_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_task: dict[str, int] = {}
    by_final_stage: dict[str, int] = {}
    for row in rows:
        by_task[row["task"]] = by_task.get(row["task"], 0) + 1
        for chunk in row.get("chunks", []):
            final_stage = str(chunk.get("final_stage", "unknown"))
            by_final_stage[final_stage] = by_final_stage.get(final_stage, 0) + 1
    nonempty_ratios = [float(row["nonempty_frame_ratio"]) for row in rows]
    mean_area_ratios = [float(row["mean_area_ratio"]) for row in rows]
    sam2_ious = [
        float(row["sam2_mean_iou"])
        for row in rows
        if row.get("sam2_available") and "sam2_mean_iou" in row
    ]
    return {
        "segments": len(rows),
        "segments_by_task": by_task,
        "chunks_by_final_stage": by_final_stage,
        "mean_nonempty_frame_ratio": float(np.mean(nonempty_ratios)) if nonempty_ratios else 0.0,
        "mean_area_ratio": float(np.mean(mean_area_ratios)) if mean_area_ratios else 0.0,
        "mean_sam2_iou": float(np.mean(sam2_ious)) if sam2_ious else None,
        "rows": rows,
    }


def selected_prompts(args: argparse.Namespace) -> list[str]:
    return parse_csv(args.prompt_list) if args.prompt_list else [args.prompt]


def output_root_for_prompt(args: argparse.Namespace, prompt: str, prompt_count: int) -> Path:
    if prompt_count == 1:
        return args.output_root
    return args.output_root / safe_name(f"{args.prompt_mode}_{prompt}")


def run_prompt_job(
    base_args: argparse.Namespace,
    predictor,
    segments: list[SegmentInfo],
    tasks: list[str],
    prompt: str,
    prompt_count: int,
) -> dict[str, Any]:
    args = argparse.Namespace(**vars(base_args))
    args.prompt = prompt
    args.output_root = output_root_for_prompt(base_args, prompt, prompt_count)

    print(
        f"Prompt job: {prompt!r} mode={args.prompt_mode} output={args.output_root}",
        flush=True,
    )
    rows: list[dict[str, Any]] = []
    t0 = time.time()
    for index, segment in enumerate(segments, start=1):
        print(
            f"[{index}/{len(segments)}] {segment.task}/{segment.episode}/{segment.seg}",
            flush=True,
        )
        rows.append(process_segment(args, predictor, segment))
        if args.print_gpu_memory:
            print_gpu_memory(f"after {segment.task}/{segment.episode}/{segment.seg}")

    summary = aggregate_summary(rows)
    summary.update(
        {
            "task_spec": args.task,
            "tasks": tasks,
            "output_root": str(args.output_root),
            "elapsed_seconds": time.time() - t0,
            "sam3_model": args.sam3_version,
            "prompt_mode": args.prompt_mode,
            "prompt": args.prompt,
            "backup_prompt": args.backup_prompt,
            "bbox_source": args.bbox_source,
            "bbox_expand_ratio": args.bbox_expand_ratio,
            "bbox_min_area_ratio": args.bbox_min_area_ratio,
            "bbox_include_text": bool(args.bbox_include_text),
            "chunk_size": args.chunk_size,
            "chunk_overlap": args.chunk_overlap,
            "frame_start": args.frame_start,
            "frame_count": args.frame_count,
        }
    )
    index_path = args.output_root / "index.json"
    write_json(index_path, summary)
    print(f"Done: {index_path}")
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute SAM3 masks for G1 segment videos.")
    parser.add_argument("--task", default="all", help="task short name, CSV, or all/training/canonical")
    parser.add_argument("--segment-root", type=Path, default=DEFAULT_SEGMENT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--tmp-dir", type=Path, default=DEFAULT_TMP_DIR)
    parser.add_argument("--sam2-mask-root", type=Path, default=DEFAULT_SAM2_MASK_ROOT)
    parser.add_argument("--prompt", default="robot arm")
    parser.add_argument(
        "--prompt-list",
        default="",
        help="comma-separated robot prompts to sweep; writes one subdir per prompt",
    )
    parser.add_argument("--backup-prompt", default="robotic arm")
    parser.add_argument("--prompt-mode", default="text", choices=["text", "text_bbox"])
    parser.add_argument(
        "--bbox-source",
        default="first-nonempty",
        choices=["first-nonempty", "first-frame", "union"],
        help="text mask source used to build the second-stage bbox prompt",
    )
    parser.add_argument("--bbox-expand-ratio", type=float, default=0.15)
    parser.add_argument("--bbox-min-area-ratio", type=float, default=0.0005)
    parser.add_argument(
        "--bbox-include-text",
        action="store_true",
        help="include the text prompt again when running the second-stage bbox prompt",
    )
    parser.add_argument("--sam3-version", default="sam3.1", choices=["sam3", "sam3.1"])
    parser.add_argument("--max-num-objects", type=int, default=1)
    parser.add_argument("--multiplex-count", type=int, default=16)
    parser.add_argument("--output-prob-thresh", type=float, default=0.5)
    parser.add_argument("--frame-start", type=int, default=0)
    parser.add_argument(
        "--frame-count",
        type=int,
        default=0,
        help="number of source frames to process; 0 means all remaining frames",
    )
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--chunk-size", type=int, default=17)
    parser.add_argument("--chunk-overlap", type=int, default=0)
    parser.add_argument("--max-segments-per-task", type=int, default=0)
    parser.add_argument("--max-segments", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--keep-frames", action="store_true")
    parser.add_argument("--write-overlay", action="store_true")
    parser.add_argument("--print-gpu-memory", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.max_num_objects <= 0:
        raise ValueError(f"--max-num-objects must be positive, got {args.max_num_objects}")
    if args.multiplex_count <= 0:
        raise ValueError(f"--multiplex-count must be positive, got {args.multiplex_count}")
    if args.max_segments_per_task < 0:
        raise ValueError("--max-segments-per-task must be non-negative")
    if args.max_segments < 0:
        raise ValueError("--max-segments must be non-negative")
    if args.bbox_expand_ratio < 0.0:
        raise ValueError("--bbox-expand-ratio must be non-negative")
    if args.bbox_min_area_ratio < 0.0:
        raise ValueError("--bbox-min-area-ratio must be non-negative")

    tasks = expand_tasks(args.task)
    segments = discover_segments(args, tasks)
    prompts = selected_prompts(args)
    planned = []
    for segment in segments:
        indices = selected_frame_indices(segment, args)
        chunks = build_chunks(indices, args.chunk_size, args.chunk_overlap)
        planned.append(
            {
                "task": segment.task,
                "episode": segment.episode,
                "seg": segment.seg,
                "video_path": str(segment.video_path),
                "output_path": str(output_path_for_segment(args.output_root, segment)),
                "frame_count": segment.frame_count,
                "size": [segment.height, segment.width],
                "selected_frame_count": len(indices),
                "chunk_count": len(chunks),
                "chunk_size": args.chunk_size,
            }
        )

    print("G1 SAM3 mask precompute")
    print(f"  tasks:        {tasks}")
    print(f"  segments:     {len(planned)}")
    print(f"  output:       {args.output_root}")
    print(f"  prompt_mode:  {args.prompt_mode}")
    print(f"  prompts:      {prompts}")
    print(f"  backup:       {args.backup_prompt or '<none>'}")
    print(f"  bbox_source:  {args.bbox_source}")
    print(f"  bbox_text:    {bool(args.bbox_include_text)}")
    print(f"  frame_start:  {args.frame_start}")
    print(f"  frame_count:  {args.frame_count or 'all'}")
    print(f"  chunk_size:   {args.chunk_size}")
    print(f"  chunk_overlap:{args.chunk_overlap}")
    if args.dry_run:
        by_task: dict[str, dict[str, int]] = {}
        for row in planned:
            item = by_task.setdefault(row["task"], {"segments": 0, "chunks": 0, "frames": 0})
            item["segments"] += 1
            item["chunks"] += int(row["chunk_count"])
            item["frames"] += int(row["selected_frame_count"])
        print("  planned:")
        for task, item in sorted(by_task.items()):
            print(
                f"    {task}: segments={item['segments']} chunks={item['chunks']} "
                f"selected_frames={item['frames']}"
            )
        return

    if args.print_gpu_memory:
        print_gpu_memory("before SAM3 load")
    predictor = load_predictor(args)
    if args.print_gpu_memory:
        print_gpu_memory("after SAM3 load")

    sweep_t0 = time.time()
    summaries = [
        run_prompt_job(args, predictor, segments, tasks, prompt, len(prompts))
        for prompt in prompts
    ]
    if len(prompts) > 1:
        sweep_summary = {
            "task_spec": args.task,
            "tasks": tasks,
            "output_root": str(args.output_root),
            "elapsed_seconds": time.time() - sweep_t0,
            "sam3_model": args.sam3_version,
            "prompt_mode": args.prompt_mode,
            "prompts": prompts,
            "backup_prompt": args.backup_prompt,
            "bbox_source": args.bbox_source,
            "bbox_expand_ratio": args.bbox_expand_ratio,
            "bbox_min_area_ratio": args.bbox_min_area_ratio,
            "bbox_include_text": bool(args.bbox_include_text),
            "runs": [
                {k: v for k, v in summary.items() if k != "rows"}
                for summary in summaries
            ],
        }
        sweep_index_path = args.output_root / "prompt_sweep_index.json"
        write_json(sweep_index_path, sweep_summary)
        print(f"Prompt sweep done: {sweep_index_path}")
        print(json.dumps(sweep_summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
