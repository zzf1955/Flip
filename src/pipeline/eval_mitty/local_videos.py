"""Local metric crop and overlay video outputs for Mitty evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from src.core.train_utils import save_video
from src.tools.eval_metrics import (
    load_clip_mask_stack,
    mask_bbox,
    read_video_frames,
    resolve_sam2_mask_path,
)


def _resize_crop(frame: np.ndarray, bbox: tuple[int, int, int, int], size: int) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    crop = frame[y1:y2, x1:x2]
    resized = Image.fromarray(crop).resize((size, size), Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.uint8)


def _union_bbox(bboxes: list[tuple[int, int, int, int]]) -> tuple[int, int, int, int]:
    if not bboxes:
        raise ValueError("Cannot compute union bbox from an empty bbox list")
    return (
        min(b[0] for b in bboxes),
        min(b[1] for b in bboxes),
        max(b[2] for b in bboxes),
        max(b[3] for b in bboxes),
    )


def _local_video_bboxes(
    masks: np.ndarray,
    margin: int,
    bbox_mode: str,
) -> tuple[list[tuple[int, int, int, int]], tuple[int, int, int, int]]:
    frame_bboxes = [mask_bbox(mask, margin=margin) for mask in masks]
    union = _union_bbox(frame_bboxes)
    if bbox_mode == "frame":
        return frame_bboxes, union
    if bbox_mode == "union":
        return [union for _ in frame_bboxes], union
    raise ValueError(f"Unsupported local video bbox mode: {bbox_mode}")


def _crop_frames(
    frames: np.ndarray,
    bboxes: list[tuple[int, int, int, int]],
    output_size: int,
) -> list[Image.Image]:
    if len(frames) != len(bboxes):
        raise ValueError(f"Frame/bbox count mismatch: {len(frames)} vs {len(bboxes)}")
    return [
        Image.fromarray(_resize_crop(frame, bbox, output_size))
        for frame, bbox in zip(frames, bboxes)
    ]


def _draw_metric_bbox(
    frame: np.ndarray,
    bbox: tuple[int, int, int, int],
    mask: np.ndarray,
) -> Image.Image:
    image = Image.fromarray(frame).convert("RGB")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    mask_alpha = (mask.astype(np.uint8) * 88)
    mask_rgba = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
    mask_rgba[..., 0] = 255
    mask_rgba[..., 1] = 216
    mask_rgba[..., 3] = mask_alpha
    overlay.alpha_composite(Image.fromarray(mask_rgba, mode="RGBA"))
    image = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
    draw = ImageDraw.Draw(image)
    x1, y1, x2, y2 = bbox
    for inset in range(3):
        draw.rectangle(
            (x1 + inset, y1 + inset, x2 - 1 - inset, y2 - 1 - inset),
            outline=(255, 216, 0),
        )
    return image


def _overlay_frames(
    frames: np.ndarray,
    masks: np.ndarray,
    bboxes: list[tuple[int, int, int, int]],
) -> list[Image.Image]:
    if not (len(frames) == len(masks) == len(bboxes)):
        raise ValueError(
            f"Overlay count mismatch: frames={len(frames)} masks={len(masks)} "
            f"bboxes={len(bboxes)}"
        )
    return [
        _draw_metric_bbox(frame, bbox, mask)
        for frame, mask, bbox in zip(frames, masks, bboxes)
    ]


def _stack_frame_lists(frame_lists: list[list[Image.Image]]) -> list[Image.Image]:
    if not frame_lists:
        raise ValueError("No frame lists to stack")
    n_frames = len(frame_lists[0])
    if any(len(frames) != n_frames for frames in frame_lists):
        raise ValueError("Local compare video frame count mismatch")
    stacked = []
    for frames in zip(*frame_lists):
        arrays = [np.asarray(frame) for frame in frames]
        stacked.append(Image.fromarray(np.concatenate(arrays, axis=1)))
    return stacked


def write_local_videos(
    split_out: Path,
    records: list[dict],
    sam2_mask_root: str | Path,
    *,
    margin: int,
    output_size: int,
    bbox_mode: str,
    show_progress: bool = True,
) -> None:
    """Write Local metric crop videos, overlay videos, and a patch index."""
    if margin < 0:
        raise ValueError(f"local video margin must be non-negative, got {margin}")
    if output_size <= 0:
        raise ValueError(f"local video size must be positive, got {output_size}")
    if bbox_mode not in {"frame", "union"}:
        raise ValueError(f"Unsupported local video bbox mode: {bbox_mode}")

    local_dir = split_out / "local_fid"
    local_dir.mkdir(parents=True, exist_ok=True)
    index_rows = []

    for idx, record in enumerate(records):
        sample_id = f"{idx:05d}"
        if show_progress:
            print(
                f"  Local videos {split_out.name}: {idx + 1}/{len(records)} "
                f"sample={sample_id}",
                flush=True,
            )
        paths = {
            "gen": split_out / f"gen_{sample_id}.mp4",
            "gt": split_out / f"gt_{sample_id}.mp4",
            "ctrl": split_out / f"ctrl_{sample_id}.mp4",
        }
        for label, path in paths.items():
            if not path.is_file():
                raise FileNotFoundError(f"Missing {label} video for Local output: {path}")

        gen_frames = read_video_frames(str(paths["gen"]))
        gt_frames = read_video_frames(str(paths["gt"]))
        ctrl_frames = read_video_frames(str(paths["ctrl"]))
        if not (len(gen_frames) == len(gt_frames) == len(ctrl_frames)):
            raise ValueError(
                f"Local video frame count mismatch for sample {sample_id}: "
                f"gen={len(gen_frames)} gt={len(gt_frames)} ctrl={len(ctrl_frames)}"
            )

        masks = load_clip_mask_stack(
            record,
            sam2_mask_root,
            len(gen_frames),
            gen_frames.shape[1:3],
        )
        bboxes, union = _local_video_bboxes(masks, margin=margin, bbox_mode=bbox_mode)
        gen_local = _crop_frames(gen_frames, bboxes, output_size)
        gt_local = _crop_frames(gt_frames, bboxes, output_size)
        ctrl_local = _crop_frames(ctrl_frames, bboxes, output_size)
        gen_overlay = _overlay_frames(gen_frames, masks, bboxes)
        gt_overlay = _overlay_frames(gt_frames, masks, bboxes)
        ctrl_overlay = _overlay_frames(ctrl_frames, masks, bboxes)

        rel_paths = {
            "local_gen_video": f"local_fid/gen_{sample_id}.mp4",
            "local_gt_video": f"local_fid/gt_{sample_id}.mp4",
            "local_ctrl_video": f"local_fid/ctrl_{sample_id}.mp4",
            "local_compare_video": f"local_fid/compare_{sample_id}.mp4",
            "overlay_gen_video": f"local_fid/gen_overlay_{sample_id}.mp4",
            "overlay_gt_video": f"local_fid/gt_overlay_{sample_id}.mp4",
            "overlay_ctrl_video": f"local_fid/ctrl_overlay_{sample_id}.mp4",
            "overlay_compare_video": f"local_fid/compare_overlay_{sample_id}.mp4",
        }
        save_video(gen_local, str(split_out / rel_paths["local_gen_video"]))
        save_video(gt_local, str(split_out / rel_paths["local_gt_video"]))
        save_video(ctrl_local, str(split_out / rel_paths["local_ctrl_video"]))
        save_video(
            _stack_frame_lists([gt_local, gen_local, ctrl_local]),
            str(split_out / rel_paths["local_compare_video"]),
        )
        save_video(gen_overlay, str(split_out / rel_paths["overlay_gen_video"]))
        save_video(gt_overlay, str(split_out / rel_paths["overlay_gt_video"]))
        save_video(ctrl_overlay, str(split_out / rel_paths["overlay_ctrl_video"]))
        save_video(
            _stack_frame_lists([gt_overlay, gen_overlay, ctrl_overlay]),
            str(split_out / rel_paths["overlay_compare_video"]),
        )

        mask_path = resolve_sam2_mask_path(record, sam2_mask_root)
        index_row = {
            "sample_id": sample_id,
            "split_dir": str(split_out),
            "gen_video": f"gen_{sample_id}.mp4",
            "gt_video": f"gt_{sample_id}.mp4",
            "ctrl_video": f"ctrl_{sample_id}.mp4",
            **rel_paths,
            "robot_task": record.get("robot_task", record.get("task", "")),
            "pair_id": record.get("pair_id", ""),
            "order_index": record.get("order_index"),
            "pair_order_path": record.get("pair_order_path", ""),
            "source_id": record.get("source_id", ""),
            "source_segment_id": record.get("source_segment_id", ""),
            "episode": record.get("episode", ""),
            "seg": record.get("seg", ""),
            "clip_idx": record.get("clip_idx"),
            "clip_start": record.get("clip_start"),
            "clip_dur": record.get("clip_dur"),
            "augment": record.get("augment", "normal"),
            "mask_path": str(mask_path),
            "local_margin_px": margin,
            "local_output_size": output_size,
            "local_bbox_mode": bbox_mode,
            "overlay_mask_rgba": [255, 216, 0, 88],
            "overlay_bbox_rgb": [255, 216, 0],
            "frame_bboxes_xyxy": [list(bbox) for bbox in bboxes],
            "union_bbox_xyxy": list(union),
        }
        for required in ("pair_id", "order_index", "pair_order_path"):
            if index_row[required] in ("", None):
                raise ValueError(f"Selected record missing {required}: {record}")
        index_rows.append(index_row)

    with (local_dir / "patch_index.jsonl").open("w") as fh:
        for row in index_rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")

