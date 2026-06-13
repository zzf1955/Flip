"""Seedance H2R robot-arm to human-hand smoke with explicit SAM3 mask markers.

This entry keeps the direct H2R Seedance smoke separate from the SAM3-guided
variant.  It consumes precomputed SAM3/SAM3.1 robot-arm masks, paints the
masked robot region in the Seedance reference video, and asks Seedance to
replace the marked region with a bare human hand/forearm while preserving the
rest of the scene.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np

from src.core.config import MAIN_ROOT
from src.pipeline.h2r_seedance_edit import (
    DEFAULT_DATA_ROOT,
    DEFAULT_FPS,
    DEFAULT_NUM_FRAMES,
    DEFAULT_SAMPLES,
    SampleSpec,
    aspect_fill_resize_crop,
    hdf5_path,
    load_env_file,
    parse_hxw,
    parse_sample,
    prepare_input_video,
    process_one,
    read_h2r_camera_clip,
    validate_seedance_input,
    write_jsonl,
    write_video_rgb,
)
from src.pipeline.seedance_gen import MODEL_FAST, MODEL_STANDARD, get_video_info


DEFAULT_OUTPUT_ROOT = Path(MAIN_ROOT) / "tmp" / "h2r_seedance_sam3_red_edit"
DEFAULT_MASK_ROOT = DEFAULT_OUTPUT_ROOT / "sam3_mask"
DEFAULT_PROMPT_TEMPLATE = (
    "把视频中{marker_desc}标出的移动装置替换成裸露的人类胳膊。"
    "保持背景不变，人手和该装置的动作轨迹保持一致。"
    "人类手臂{task_name}"
)
DEFAULT_TASK_NAMES = {
    "grab_both_cubes_v1": "抓起物块。",
    "grab_cup_v1": "抓起杯子。",
    "roll": "滚动物体。",
}
RED_RGB = np.array([255, 0, 0], dtype=np.float32)
MARKER_COLORS_RGB = {
    "red": np.array([255, 0, 0], dtype=np.float32),
    "magenta": np.array([255, 0, 255], dtype=np.float32),
    "cyan": np.array([0, 255, 255], dtype=np.float32),
    "yellow": np.array([255, 255, 0], dtype=np.float32),
    "green": np.array([0, 255, 0], dtype=np.float32),
    "skin": np.array([232, 174, 132], dtype=np.float32),
}
MARKER_COLORS_CN = {
    "red": "红色",
    "magenta": "紫色",
    "cyan": "青色",
    "yellow": "黄色",
    "green": "绿色",
    "skin": "肤色",
}


def episode_mask_path(mask_root: Path, sample: SampleSpec) -> Path:
    return mask_root / sample.task / f"episode_{sample.episode}.npz"


def prompt_for_sample(sample: SampleSpec, args: argparse.Namespace) -> tuple[str, str]:
    if args.prompt:
        return args.prompt, DEFAULT_TASK_NAMES.get(sample.task, "")
    task_name = DEFAULT_TASK_NAMES.get(sample.task)
    if task_name is None:
        raise ValueError(
            f"missing Chinese task name for {sample.task!r}; add it to DEFAULT_TASK_NAMES "
            "or pass --prompt to use one prompt for all samples"
        )
    return args.prompt_template.format(
        task_name=task_name,
        marker_desc=args.marker_desc,
    ), task_name


def postprocess_mask(mask: np.ndarray, dilate_pixels: int, close_pixels: int) -> np.ndarray:
    out = (mask > 0).astype(np.uint8)
    if close_pixels > 0:
        close_kernel = np.ones((close_pixels, close_pixels), dtype=np.uint8)
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, close_kernel)
    if dilate_pixels > 0:
        dilate_kernel = np.ones((dilate_pixels, dilate_pixels), dtype=np.uint8)
        out = cv2.dilate(out, dilate_kernel, iterations=1)
    return out


def nearest_covered_index(
    frame_index: int,
    covered_frames: np.ndarray,
    *,
    max_gap: int,
    source: Path,
) -> int:
    insert = int(np.searchsorted(covered_frames, frame_index))
    candidates = []
    if insert < len(covered_frames):
        candidates.append(int(covered_frames[insert]))
    if insert > 0:
        candidates.append(int(covered_frames[insert - 1]))
    if not candidates:
        raise ValueError(f"SAM3 mask has no covered frames: {source}")
    nearest = min(candidates, key=lambda value: abs(value - frame_index))
    gap = abs(nearest - frame_index)
    if gap > max_gap:
        raise ValueError(
            f"nearest SAM3 covered frame is too far for frame {frame_index}: "
            f"nearest={nearest} gap={gap} max_gap={max_gap} source={source}"
        )
    return nearest


def read_sam3_mask_clip(
    mask_path: Path,
    start_frame: int,
    num_frames: int,
    *,
    max_nearest_gap: int,
    dilate_pixels: int,
    close_pixels: int,
) -> tuple[np.ndarray, dict]:
    if not mask_path.is_file():
        raise FileNotFoundError(f"SAM3 mask npz not found: {mask_path}")
    with np.load(mask_path) as data:
        if "masks" not in data:
            raise ValueError(f"SAM3 mask npz missing 'masks': {mask_path}")
        if "covered_frames" not in data:
            raise ValueError(f"SAM3 mask npz missing 'covered_frames': {mask_path}")
        masks = data["masks"]
        covered_frames = np.asarray(data["covered_frames"], dtype=np.int32)
    if masks.ndim == 4:
        masks = masks.max(axis=1)
    if masks.ndim != 3:
        raise ValueError(f"SAM3 masks must be [T,H,W] or [T,N,H,W], got {masks.shape}")
    if covered_frames.ndim != 1 or len(covered_frames) == 0:
        raise ValueError(f"SAM3 covered_frames must be a non-empty 1D array: {mask_path}")
    covered_frames = np.unique(covered_frames)
    if int(covered_frames[-1]) >= len(masks):
        raise ValueError(
            f"SAM3 covered frame index exceeds mask count: max={int(covered_frames[-1])} "
            f"count={len(masks)} source={mask_path}"
        )

    source_indices = np.arange(start_frame, start_frame + num_frames, dtype=np.int32)
    selected_indices = [
        nearest_covered_index(
            int(frame_index),
            covered_frames,
            max_gap=max_nearest_gap,
            source=mask_path,
        )
        for frame_index in source_indices
    ]
    out = np.stack(
        [
            postprocess_mask(masks[index], dilate_pixels, close_pixels)
            for index in selected_indices
        ],
        axis=0,
    ).astype(np.uint8)
    per_frame_area = out.reshape(len(out), -1).sum(axis=1).astype(np.int64)
    info = {
        "mask_path": str(mask_path),
        "mask_shape": list(masks.shape),
        "covered_frame_count": int(len(covered_frames)),
        "source_frame_indices": source_indices.tolist(),
        "selected_mask_indices": [int(index) for index in selected_indices],
        "max_selected_gap": int(
            max(abs(int(a) - int(b)) for a, b in zip(source_indices, selected_indices))
        ),
        "mask_nonzero_frames": int(np.count_nonzero(per_frame_area)),
        "mask_area_min": int(per_frame_area.min()),
        "mask_area_max": int(per_frame_area.max()),
        "mask_area_mean": float(per_frame_area.mean()),
        "mask_dilate_pixels": int(dilate_pixels),
        "mask_close_pixels": int(close_pixels),
        "mask_max_nearest_gap": int(max_nearest_gap),
    }
    return out, info


def mask_area_info(masks: np.ndarray) -> dict:
    per_frame_area = masks.reshape(len(masks), -1).sum(axis=1).astype(np.int64)
    return {
        "nonzero_frames": int(np.count_nonzero(per_frame_area)),
        "area_min": int(per_frame_area.min()),
        "area_max": int(per_frame_area.max()),
        "area_mean": float(per_frame_area.mean()),
    }


def marker_description(
    marker_color: str,
    annotation_mode: str,
    bbox_marker_color: str = "",
) -> str:
    color_cn = MARKER_COLORS_CN[marker_color]
    if annotation_mode == "fill":
        return f"{color_cn}区域"
    if annotation_mode == "outline":
        return f"{color_cn}轮廓线"
    if annotation_mode == "bbox":
        return f"{color_cn}方框"
    if annotation_mode == "dual_bbox":
        bbox_color_cn = MARKER_COLORS_CN[bbox_marker_color] if bbox_marker_color else color_cn
        if bbox_color_cn == color_cn:
            return f"{color_cn}双层方框"
        return f"{color_cn}方框和{bbox_color_cn}方框"
    if annotation_mode == "fill_bbox":
        bbox_color_cn = MARKER_COLORS_CN[bbox_marker_color] if bbox_marker_color else color_cn
        if bbox_color_cn == color_cn:
            return f"{color_cn}区域和{color_cn}方框"
        return f"{color_cn}区域和{bbox_color_cn}方框"
    raise ValueError(f"unsupported annotation mode: {annotation_mode}")


def sam3_input_mode(args: argparse.Namespace) -> str:
    if args.original_input:
        return "original"
    if args.annotation_mode == "dual_bbox" and args.bbox_marker_color:
        return (
            f"sam3_{args.mask_filter}_{args.marker_color}_bbox_"
            f"{args.bbox_marker_color}_bbox"
        )
    if args.annotation_mode == "fill_bbox" and args.bbox_marker_color:
        return (
            f"sam3_{args.mask_filter}_{args.marker_color}_fill_"
            f"{args.bbox_marker_color}_bbox"
        )
    return f"sam3_{args.mask_filter}_{args.marker_color}_{args.annotation_mode}"


def filter_target_masks(
    frames_rgb: np.ndarray,
    masks: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict]:
    if args.mask_filter == "full":
        return masks, {
            "mask_filter": args.mask_filter,
            **mask_area_info(masks),
        }
    if args.mask_filter not in {"dark", "distal_dark"}:
        raise ValueError(f"unsupported --mask-filter={args.mask_filter!r}")

    dark = np.max(frames_rgb, axis=3) <= args.dark_threshold
    filtered = (masks.astype(bool) & dark).astype(np.uint8)
    if args.mask_filter == "distal_dark":
        filtered, distal_info = select_distal_dark_components(filtered, masks, args)
    else:
        distal_info = {}
    processed = np.stack(
        [
            postprocess_mask(mask, args.filter_dilate_pixels, args.filter_close_pixels)
            for mask in filtered
        ],
        axis=0,
    ).astype(np.uint8)
    info = {
        "mask_filter": args.mask_filter,
        "dark_threshold": int(args.dark_threshold),
        "filter_dilate_pixels": int(args.filter_dilate_pixels),
        "filter_close_pixels": int(args.filter_close_pixels),
        **distal_info,
        **mask_area_info(processed),
    }
    if info["nonzero_frames"] == 0:
        raise ValueError(
            "dark mask filter produced zero target frames; adjust --dark-threshold "
            "or use a different SAM3 mask root"
        )
    return processed, info


def select_distal_dark_components(
    dark_masks: np.ndarray,
    arm_masks: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict]:
    out = np.zeros_like(dark_masks, dtype=np.uint8)
    selected_counts = []
    candidate_counts = []
    selected_areas = []
    previous_centroid: np.ndarray | None = None
    frame_diag = float(np.hypot(dark_masks.shape[2], dark_masks.shape[1]))
    for frame_index, (dark_mask, arm_mask) in enumerate(zip(dark_masks, arm_masks)):
        base = postprocess_mask(dark_mask, dilate_pixels=0, close_pixels=1)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            base.astype(np.uint8), 8
        )
        arm_ys, arm_xs = np.nonzero(arm_mask.astype(bool))
        if len(arm_xs) == 0:
            arm_center = np.array(
                [dark_mask.shape[1] / 2.0, dark_mask.shape[0] / 2.0],
                dtype=np.float32,
            )
        else:
            arm_center = np.array([arm_xs.mean(), arm_ys.mean()], dtype=np.float32)

        candidates = []
        for label in range(1, num_labels):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < args.distal_min_area or area > args.distal_max_area:
                continue
            x = int(stats[label, cv2.CC_STAT_LEFT])
            y = int(stats[label, cv2.CC_STAT_TOP])
            width = int(stats[label, cv2.CC_STAT_WIDTH])
            height = int(stats[label, cv2.CC_STAT_HEIGHT])
            aspect = max(width / max(1, height), height / max(1, width))
            if aspect > args.distal_max_aspect:
                continue
            centroid = np.asarray(centroids[label], dtype=np.float32)
            distance = float(np.linalg.norm(centroid - arm_center))
            touches_border = (
                x == 0
                or y == 0
                or x + width >= dark_mask.shape[1]
                or y + height >= dark_mask.shape[0]
            )
            border_weight = args.distal_border_penalty if touches_border else 1.0
            score = distance * float(np.sqrt(area)) * border_weight
            candidates.append(
                {
                    "label": label,
                    "area": area,
                    "base_score": score,
                    "centroid": centroid,
                    "touches_border": touches_border,
                }
            )
        candidate_counts.append(len(candidates))
        if not candidates:
            selected_counts.append(0)
            continue

        max_base_score = max(float(row["base_score"]) for row in candidates)
        for row in candidates:
            if previous_centroid is None:
                row["score"] = float(row["base_score"])
            else:
                previous_distance = float(np.linalg.norm(row["centroid"] - previous_centroid))
                row["score"] = (
                    float(row["base_score"]) / max_base_score
                    - args.distal_temporal_weight * previous_distance / frame_diag
                )
        candidates.sort(key=lambda row: row["score"], reverse=True)
        selected = candidates[: args.distal_components]
        for row in selected:
            out[frame_index][labels == row["label"]] = 1
            selected_areas.append(row["area"])
        selected_counts.append(len(selected))
        previous_centroid = np.mean(
            np.stack([row["centroid"] for row in selected], axis=0),
            axis=0,
        )

    info = {
        "distal_min_area": int(args.distal_min_area),
        "distal_max_area": int(args.distal_max_area),
        "distal_components": int(args.distal_components),
        "distal_border_penalty": float(args.distal_border_penalty),
        "distal_max_aspect": float(args.distal_max_aspect),
        "distal_temporal_weight": float(args.distal_temporal_weight),
        "distal_candidate_frames": int(sum(1 for count in candidate_counts if count > 0)),
        "distal_selected_frames": int(sum(1 for count in selected_counts if count > 0)),
        "distal_selected_component_count": int(sum(selected_counts)),
        "distal_selected_area_mean": float(np.mean(selected_areas)) if selected_areas else 0.0,
    }
    return out, info


def annotation_masks_from_target(target_masks: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    if args.annotation_mode == "fill":
        return target_masks
    out = np.zeros_like(target_masks, dtype=np.uint8)
    if args.annotation_mode == "outline":
        kernel = np.ones(
            (args.annotation_thickness, args.annotation_thickness),
            dtype=np.uint8,
        )
        for index, mask in enumerate(target_masks):
            dilated = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)
            eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=1)
            out[index] = np.clip(dilated - eroded, 0, 1).astype(np.uint8)
        return out
    if args.annotation_mode in {"bbox", "dual_bbox", "fill_bbox"}:
        for index, mask in enumerate(target_masks):
            if args.annotation_mode == "fill_bbox":
                out[index] = np.maximum(out[index], mask.astype(np.uint8))
            ys, xs = np.nonzero(mask.astype(bool))
            if len(xs) == 0:
                continue
            x0 = max(0, int(xs.min()) - args.bbox_expand_pixels)
            x1 = min(mask.shape[1] - 1, int(xs.max()) + args.bbox_expand_pixels)
            y0 = max(0, int(ys.min()) - args.bbox_expand_pixels)
            y1 = min(mask.shape[0] - 1, int(ys.max()) + args.bbox_expand_pixels)
            cv2.rectangle(
                out[index],
                (x0, y0),
                (x1, y1),
                1,
                thickness=args.annotation_thickness,
            )
        return out
    raise ValueError(f"unsupported --annotation-mode={args.annotation_mode!r}")


def bbox_masks_from_target(
    target_masks: np.ndarray,
    args: argparse.Namespace,
    *,
    extra_expand_pixels: int = 0,
) -> np.ndarray:
    out = np.zeros_like(target_masks, dtype=np.uint8)
    for index, mask in enumerate(target_masks):
        ys, xs = np.nonzero(mask.astype(bool))
        if len(xs) == 0:
            continue
        expand = args.bbox_expand_pixels + extra_expand_pixels
        x0 = max(0, int(xs.min()) - expand)
        x1 = min(mask.shape[1] - 1, int(xs.max()) + expand)
        y0 = max(0, int(ys.min()) - expand)
        y1 = min(mask.shape[0] - 1, int(ys.max()) + expand)
        cv2.rectangle(
            out[index],
            (x0, y0),
            (x1, y1),
            1,
            thickness=args.annotation_thickness,
        )
    return out


def apply_marker_annotation(
    frames_rgb: np.ndarray,
    masks: np.ndarray,
    *,
    alpha: float,
    color_rgb: np.ndarray,
) -> np.ndarray:
    if frames_rgb.shape[:3] != masks.shape:
        raise ValueError(
            f"frame/mask shape mismatch: {frames_rgb.shape[:3]} != {masks.shape}"
        )
    if alpha < 0.0 or alpha > 1.0:
        raise ValueError(f"--red-alpha/marker alpha must be in [0,1], got {alpha}")
    out = frames_rgb.astype(np.float32).copy()
    mask_bool = masks.astype(bool)
    out[mask_bool] = out[mask_bool] * (1.0 - alpha) + color_rgb * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def prepare_marker_mask_input_video(
    sample: SampleSpec,
    args: argparse.Namespace,
) -> dict:
    h5_path = hdf5_path(args.data_root, sample)
    frames, dtype_info = read_h2r_camera_clip(
        h5_path, args.camera, sample.start_frame, args.num_frames
    )
    if frames.shape[1:3] != (240, 426):
        raise ValueError(
            f"expected H2R source shape 240x426, got {frames.shape[1:3]} "
            f"for {h5_path}:{args.camera}"
        )
    mask_path = episode_mask_path(args.mask_root, sample)
    masks, mask_info = read_sam3_mask_clip(
        mask_path,
        sample.start_frame,
        args.num_frames,
        max_nearest_gap=args.mask_max_nearest_gap,
        dilate_pixels=args.mask_dilate_pixels,
        close_pixels=args.mask_close_pixels,
    )
    target_masks, target_mask_info = filter_target_masks(frames, masks, args)
    if args.annotation_mode == "dual_bbox" and args.bbox_marker_color:
        annotation_masks = annotation_masks_from_target(target_masks, args)
        annotated_frames = apply_marker_annotation(
            frames,
            annotation_masks,
            alpha=args.red_alpha,
            color_rgb=MARKER_COLORS_RGB[args.marker_color],
        )
        annotated_frames = apply_marker_annotation(
            annotated_frames,
            bbox_masks_from_target(
                target_masks,
                args,
                extra_expand_pixels=args.secondary_bbox_extra_pixels,
            ),
            alpha=args.bbox_alpha,
            color_rgb=MARKER_COLORS_RGB[args.bbox_marker_color],
        )
    elif args.annotation_mode == "fill_bbox" and args.bbox_marker_color:
        annotated_frames = apply_marker_annotation(
            frames,
            target_masks,
            alpha=args.red_alpha,
            color_rgb=MARKER_COLORS_RGB[args.marker_color],
        )
        annotated_frames = apply_marker_annotation(
            annotated_frames,
            bbox_masks_from_target(target_masks, args),
            alpha=args.bbox_alpha,
            color_rgb=MARKER_COLORS_RGB[args.bbox_marker_color],
        )
    else:
        annotation_masks = annotation_masks_from_target(target_masks, args)
        annotated_frames = apply_marker_annotation(
            frames,
            annotation_masks,
            alpha=args.red_alpha,
            color_rgb=MARKER_COLORS_RGB[args.marker_color],
        )
    input_frames, geom = aspect_fill_resize_crop(
        annotated_frames, args.api_size[0], args.api_size[1]
    )
    original_frames, original_geom = aspect_fill_resize_crop(
        frames, args.api_size[0], args.api_size[1]
    )
    mask_rgb = np.repeat((target_masks[:, :, :, None] * 255).astype(np.uint8), 3, axis=3)
    mask_frames, mask_geom = aspect_fill_resize_crop(
        mask_rgb, args.api_size[0], args.api_size[1]
    )

    sample_dir = args.output_root / sample.sample_id
    marker_tag = f"{args.mask_filter}_{args.marker_color}_{args.annotation_mode}"
    if args.annotation_mode == "dual_bbox" and args.bbox_marker_color:
        marker_tag = (
            f"{args.mask_filter}_{args.marker_color}_bbox_"
            f"{args.bbox_marker_color}_bbox"
        )
    if args.annotation_mode == "fill_bbox" and args.bbox_marker_color:
        marker_tag = (
            f"{args.mask_filter}_{args.marker_color}_fill_"
            f"{args.bbox_marker_color}_bbox"
        )
    input_path = sample_dir / "input" / f"{sample.sample_id}_{args.camera}_sam3_{marker_tag}_ref_864x480.mp4"
    original_path = sample_dir / "input" / f"{sample.sample_id}_{args.camera}_original_ref_864x480.mp4"
    mask_video_path = sample_dir / "mask" / f"{sample.sample_id}_{args.camera}_sam3_{args.mask_filter}_target_mask_864x480.mp4"
    write_video_rgb(input_frames, input_path, args.fps)
    write_video_rgb(original_frames, original_path, args.fps)
    write_video_rgb(mask_frames, mask_video_path, args.fps)

    info = get_video_info(str(input_path))
    validate_seedance_input(info, expected_ratio=args.ratio)
    prompt, task_name_cn = prompt_for_sample(sample, args)
    return {
        "sample_id": sample.sample_id,
        "task": sample.task,
        "task_name_cn": task_name_cn,
        "episode": sample.episode,
        "start_frame": sample.start_frame,
        "num_frames": args.num_frames,
        "fps": args.fps,
        "camera": args.camera,
        "hdf5_path": str(h5_path),
        "input_mode": f"sam3_{marker_tag}",
        "input_path": str(input_path),
        "original_input_path": str(original_path),
        "mask_video_path": str(mask_video_path),
        "target_mask_video_path": str(mask_video_path),
        "mask_filter": args.mask_filter,
        "target_mask_info": target_mask_info,
        "marker_color": args.marker_color,
        "bbox_marker_color": args.bbox_marker_color,
        "annotation_mode": args.annotation_mode,
        "marker_desc": args.marker_desc,
        "dtype_info": dtype_info,
        "geometry": geom,
        "original_geometry": original_geom,
        "mask_geometry": mask_geom,
        "secondary_bbox_extra_pixels": args.secondary_bbox_extra_pixels,
        "mask_info": mask_info,
        "input_info": info,
        "prompt": prompt,
        "source_frame_indices": list(
            range(sample.start_frame, sample.start_frame + args.num_frames)
        ),
    }


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def process_one_with_record_prompt(
    index: int,
    total: int,
    record: dict,
    args: argparse.Namespace,
    api_key: str,
) -> dict:
    local_args = argparse.Namespace(**vars(args))
    local_args.prompt = record["prompt"]
    return process_one(index, total, record, local_args, api_key)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare and run H2R Seedance smoke using SAM3 marker-mask references"
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--mask-root", type=Path, default=DEFAULT_MASK_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--sample", action="append", default=[],
                        help="task:episode:start_frame; repeatable")
    parser.add_argument("--camera", default="robot_camera")
    parser.add_argument("--num-frames", type=int, default=DEFAULT_NUM_FRAMES)
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--api-size", default="480x864",
                        help="Seedance reference input HxW; default 480x864")
    parser.add_argument("--final-size", default="256x488",
                        help="final local review output HxW; default follows current H2R review size")
    parser.add_argument("--prompt", default="",
                        help="override prompt for all samples; default uses task-specific prompts")
    parser.add_argument("--prompt-template", default=DEFAULT_PROMPT_TEMPLATE,
                        help="task-specific template with {task_name} and optional {marker_desc}; ignored when --prompt is set")
    parser.add_argument("--mask-filter", default="full", choices=["full", "dark", "distal_dark"],
                        help="full uses the SAM3 mask as-is; dark keeps all dark pixels inside SAM3; distal_dark keeps small distal dark components")
    parser.add_argument("--dark-threshold", type=int, default=80,
                        help="RGB max-channel threshold for --mask-filter dark")
    parser.add_argument("--filter-dilate-pixels", type=int, default=2,
                        help="post-filter dilation for --mask-filter dark")
    parser.add_argument("--filter-close-pixels", type=int, default=1,
                        help="post-filter close for --mask-filter dark")
    parser.add_argument("--distal-min-area", type=int, default=20,
                        help="minimum source-frame connected-component area for --mask-filter distal_dark")
    parser.add_argument("--distal-max-area", type=int, default=2500,
                        help="maximum source-frame connected-component area for --mask-filter distal_dark")
    parser.add_argument("--distal-components", type=int, default=2,
                        help="number of distal dark components to keep per frame")
    parser.add_argument("--distal-border-penalty", type=float, default=0.6,
                        help="score multiplier for components touching the image border")
    parser.add_argument("--distal-max-aspect", type=float, default=3.5,
                        help="maximum component long-side/short-side ratio for --mask-filter distal_dark")
    parser.add_argument("--distal-temporal-weight", type=float, default=1.0,
                        help="weight for continuity with the previous selected component in --mask-filter distal_dark")
    parser.add_argument("--marker-color", default="red",
                        choices=sorted(MARKER_COLORS_RGB.keys()))
    parser.add_argument("--bbox-marker-color", default="",
                        choices=["", *sorted(MARKER_COLORS_RGB.keys())],
                        help="optional separate bbox color for --annotation-mode fill_bbox or dual_bbox")
    parser.add_argument("--annotation-mode", default="fill",
                        choices=["fill", "outline", "bbox", "dual_bbox", "fill_bbox"])
    parser.add_argument("--annotation-thickness", type=int, default=5)
    parser.add_argument("--bbox-expand-pixels", type=int, default=8)
    parser.add_argument("--secondary-bbox-extra-pixels", type=int, default=8,
                        help="extra expansion for the second bbox in --annotation-mode dual_bbox")
    parser.add_argument("--red-alpha", type=float, default=0.85)
    parser.add_argument("--bbox-alpha", type=float, default=0.95)
    parser.add_argument("--mask-max-nearest-gap", type=int, default=2)
    parser.add_argument("--mask-dilate-pixels", type=int, default=3)
    parser.add_argument("--mask-close-pixels", type=int, default=3)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--resolution", default="480p",
                        choices=["480p", "720p", "1080p", "2k"])
    parser.add_argument("--ratio", default="16:9", choices=["16:9"])
    parser.add_argument("--duration", type=int, default=4)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--api-key", default="")
    parser.add_argument("--env-file", type=Path, default=Path(MAIN_ROOT) / ".env")
    parser.add_argument("--poll-interval", type=float, default=10.0)
    parser.add_argument("--poll-timeout", type=float, default=900.0)
    parser.add_argument("--original-input", action="store_true",
                        help="debug: use the direct unmasked input instead of SAM3 red masks")
    parser.add_argument("--dry-run", action="store_true",
                        help="prepare inputs and plan only; do not call Seedance")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.num_frames <= 0:
        raise ValueError("--num-frames must be positive")
    if args.fps <= 0:
        raise ValueError("--fps must be positive")
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    if args.mask_max_nearest_gap < 0:
        raise ValueError("--mask-max-nearest-gap must be non-negative")
    if args.mask_dilate_pixels < 0:
        raise ValueError("--mask-dilate-pixels must be non-negative")
    if args.mask_close_pixels < 0:
        raise ValueError("--mask-close-pixels must be non-negative")
    if args.dark_threshold < 0 or args.dark_threshold > 255:
        raise ValueError("--dark-threshold must be in [0,255]")
    if args.filter_dilate_pixels < 0:
        raise ValueError("--filter-dilate-pixels must be non-negative")
    if args.filter_close_pixels < 0:
        raise ValueError("--filter-close-pixels must be non-negative")
    if args.distal_min_area <= 0:
        raise ValueError("--distal-min-area must be positive")
    if args.distal_max_area < args.distal_min_area:
        raise ValueError("--distal-max-area must be >= --distal-min-area")
    if args.distal_components <= 0:
        raise ValueError("--distal-components must be positive")
    if args.distal_border_penalty < 0.0:
        raise ValueError("--distal-border-penalty must be non-negative")
    if args.distal_max_aspect < 1.0:
        raise ValueError("--distal-max-aspect must be >= 1")
    if args.distal_temporal_weight < 0.0:
        raise ValueError("--distal-temporal-weight must be non-negative")
    if args.annotation_thickness <= 0:
        raise ValueError("--annotation-thickness must be positive")
    if args.bbox_expand_pixels < 0:
        raise ValueError("--bbox-expand-pixels must be non-negative")
    if args.secondary_bbox_extra_pixels < 0:
        raise ValueError("--secondary-bbox-extra-pixels must be non-negative")
    if args.bbox_alpha < 0.0 or args.bbox_alpha > 1.0:
        raise ValueError("--bbox-alpha must be in [0,1]")
    if args.annotation_mode == "dual_bbox" and not args.bbox_marker_color:
        raise ValueError("--annotation-mode dual_bbox requires --bbox-marker-color")
    args.api_size = parse_hxw(args.api_size)
    args.final_size = parse_hxw(args.final_size)
    args.data_root = (
        args.data_root if args.data_root.is_absolute() else Path(MAIN_ROOT) / args.data_root
    )
    args.mask_root = (
        args.mask_root if args.mask_root.is_absolute() else Path(MAIN_ROOT) / args.mask_root
    )
    args.output_root = (
        args.output_root if args.output_root.is_absolute() else Path(MAIN_ROOT) / args.output_root
    )
    args.model = MODEL_FAST if args.fast else MODEL_STANDARD
    args.env_file = (
        args.env_file if args.env_file.is_absolute() else Path(MAIN_ROOT) / args.env_file
    )
    args.marker_desc = marker_description(
        args.marker_color,
        args.annotation_mode,
        args.bbox_marker_color,
    )

    samples = [parse_sample(value) for value in (args.sample or DEFAULT_SAMPLES)]
    if not samples:
        raise ValueError("at least one --sample is required")

    input_mode = sam3_input_mode(args)
    print("H2R Seedance SAM3 marker-mask robot-arm -> human-hand smoke")
    print(f"  samples:     {[sample.sample_id for sample in samples]}")
    print(f"  camera:      {args.camera}")
    print(f"  frames/fps:  {args.num_frames} @ {args.fps}fps")
    print(f"  api size:    {args.api_size[0]}x{args.api_size[1]} HxW")
    print(f"  final size:  {args.final_size[0]}x{args.final_size[1]} HxW")
    print(f"  mask root:   {args.mask_root}")
    print(f"  input mode:  {input_mode}")
    print(f"  mask filter: {args.mask_filter}")
    print(f"  marker:      {args.marker_color}/{args.annotation_mode} ({args.marker_desc})")
    print(f"  model:       {args.model}")
    print(f"  workers:     {args.workers}")
    print(f"  prompt mode: {'override' if args.prompt else 'task_specific'}")
    print(f"  dry_run:     {args.dry_run}")

    if args.original_input:
        prepared = [prepare_input_video(sample, args) for sample in samples]
    else:
        prepared = [prepare_marker_mask_input_video(sample, args) for sample in samples]
    for sample, record in zip(samples, prepared):
        if "prompt" not in record:
            prompt, task_name_cn = prompt_for_sample(sample, args)
            record["prompt"] = prompt
            record["task_name_cn"] = task_name_cn
    write_jsonl(args.output_root / "prepared_inputs.jsonl", prepared)

    plan = {
        "status": "dry_run" if args.dry_run else "planned",
        "model": args.model,
        "resolution": args.resolution,
        "ratio": args.ratio,
        "duration": args.duration,
        "workers": args.workers,
        "fps": args.fps,
        "num_frames": args.num_frames,
        "api_size_hxw": list(args.api_size),
        "final_size_hxw": list(args.final_size),
        "input_mode": input_mode,
        "mask_root": str(args.mask_root),
        "mask_filter": args.mask_filter,
        "dark_threshold": args.dark_threshold,
        "filter_dilate_pixels": args.filter_dilate_pixels,
        "filter_close_pixels": args.filter_close_pixels,
        "distal_min_area": args.distal_min_area,
        "distal_max_area": args.distal_max_area,
        "distal_components": args.distal_components,
        "distal_border_penalty": args.distal_border_penalty,
        "distal_max_aspect": args.distal_max_aspect,
        "distal_temporal_weight": args.distal_temporal_weight,
        "marker_color": args.marker_color,
        "bbox_marker_color": args.bbox_marker_color,
        "annotation_mode": args.annotation_mode,
        "marker_desc": args.marker_desc,
        "red_alpha": args.red_alpha,
        "bbox_alpha": args.bbox_alpha,
        "secondary_bbox_extra_pixels": args.secondary_bbox_extra_pixels,
        "prompt": args.prompt,
        "prompt_template": args.prompt_template,
        "prompts_by_sample": {
            record["sample_id"]: record["prompt"] for record in prepared
        },
        "prepared_count": len(prepared),
    }
    write_json(args.output_root / "run_plan.json", plan)

    if args.dry_run:
        print(f"prepared inputs: {args.output_root / 'prepared_inputs.jsonl'}")
        print(f"run plan:        {args.output_root / 'run_plan.json'}")
        return

    env_values = load_env_file(args.env_file)
    api_key = args.api_key or os.environ.get("ARK_API_KEY") or env_values.get("ARK_API_KEY", "")
    if not api_key:
        raise ValueError(f"provide --api-key, set ARK_API_KEY, or add it to {args.env_file}")

    t0 = time.time()
    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(process_one_with_record_prompt, index, len(prepared), record, args, api_key)
            for index, record in enumerate(prepared)
        ]
        for future in as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda row: row["sample_id"])
    write_jsonl(args.output_root / "seedance_results.jsonl", results)
    summary = {
        "status": "complete",
        "ok": sum(1 for row in results if row["status"] == "ok"),
        "failed": sum(1 for row in results if row["status"] == "failed"),
        "elapsed_sec": round(time.time() - t0, 1),
        "model": args.model,
        "resolution": args.resolution,
        "ratio": args.ratio,
        "duration": args.duration,
        "workers": args.workers,
        "input_mode": input_mode,
        "mask_root": str(args.mask_root),
        "mask_filter": args.mask_filter,
        "dark_threshold": args.dark_threshold,
        "filter_dilate_pixels": args.filter_dilate_pixels,
        "filter_close_pixels": args.filter_close_pixels,
        "distal_min_area": args.distal_min_area,
        "distal_max_area": args.distal_max_area,
        "distal_components": args.distal_components,
        "distal_border_penalty": args.distal_border_penalty,
        "distal_max_aspect": args.distal_max_aspect,
        "distal_temporal_weight": args.distal_temporal_weight,
        "marker_color": args.marker_color,
        "bbox_marker_color": args.bbox_marker_color,
        "annotation_mode": args.annotation_mode,
        "marker_desc": args.marker_desc,
        "red_alpha": args.red_alpha,
        "bbox_alpha": args.bbox_alpha,
        "secondary_bbox_extra_pixels": args.secondary_bbox_extra_pixels,
        "prompt": args.prompt,
        "prompt_template": args.prompt_template,
        "prompts_by_sample": {
            row["sample_id"]: row["prompt"] for row in results
        },
    }
    write_json(args.output_root / "seedance_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
