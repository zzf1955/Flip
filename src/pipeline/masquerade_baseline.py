"""Masquerade-style direct render baseline for existing h2r pairs.

This baseline consumes the existing `training_data/pair/h2r/1s/<task>/`
layout, reconstructs the underlying robot clip from `training_data/segment/`
with the robot model and calibrated camera, and emits a compare package that
keeps the human input, baseline render, and GT robot clip aligned.

The human side is not hand-labeled.  Instead, it is estimated directly from the
`control_video` clip by a lightweight foreground / skin / half-frame heuristic
that produces per-frame masks, left/right arm-hint boxes, centroids, and a
trajectory trace.  The human mask is also used to inpaint the control video
background before an opaque robot mesh is rendered back into the same view.
These annotations are written alongside the rendered robot baseline so the
output can be inspected or reused downstream.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from src.core.config import (
    BEST_PARAMS,
    G1_URDF,
    MAIN_ROOT,
    MESH_DIR,
    OUTPUT_DIR,
    TRAINING_TASKS,
    get_hand_type,
    get_skip_meshes,
)

PAIR_ROOT = Path(MAIN_ROOT) / "training_data" / "pair" / "h2r" / "1s"
SEGMENT_ROOT = Path(MAIN_ROOT) / "training_data" / "segment"
DEFAULT_OUTPUT_ROOT = Path(OUTPUT_DIR) / "masquerade_baseline"
TARGET_FPS = 16
SOURCE_FPS = 30
EXPECTED_OUTPUT_FRAMES = 17
SOURCE_VIDEO_FRAMES = 120
PAIR_DIRNAME = "video"
HUMAN_DIRNAME = "control_video"
GT_DIRNAME = "gt_video"
BACKGROUND_DIRNAME = "background"
HUMAN_OVERLAY_DIRNAME = "human_overlay"
ANNOTATION_DIRNAME = "human_annotations"
COMPARE_DIRNAME = "compare"
COLOR_LEFT = (235, 166, 52)
COLOR_RIGHT = (43, 145, 236)
COLOR_BODY = (110, 190, 110)


@dataclass(frozen=True)
class PairRecord:
    pair_id: str
    pair_index: int
    source_id: str
    source_segment_id: str
    task: str
    episode: str
    seg: str
    clip_idx: int | None
    clip_start: float
    clip_dur: float
    augment: str
    human_src: Path
    robot_src: Path
    control_video: Path
    robot_video: Path
    segment_joints: Path
    manifest: dict


def _short_task_name(task: str) -> str:
    return task.strip().replace("G1_WBT_", "")


def parse_task_list(value: str | Iterable[str], *, allow_empty: bool = False) -> list[str]:
    if isinstance(value, str):
        items = [item.strip() for item in value.split(",")]
    else:
        items = [str(item).strip() for item in value]
    tasks = [_short_task_name(item) for item in items if item]
    if not tasks:
        if allow_empty:
            return []
        raise ValueError("Task list must not be empty")
    return tasks


def expand_task_spec(task_spec: str) -> list[str]:
    key = task_spec.strip().lower()
    if key in {"all", "training"}:
        return [_short_task_name(task) for task in TRAINING_TASKS]
    return parse_task_list(task_spec)


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Required JSONL not found: {path}")
    rows = []
    with path.open() as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_no}") from exc
    if not rows:
        raise ValueError(f"Manifest is empty: {path}")
    return rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")


def _render_robot_opaque(
    background: np.ndarray,
    mesh_cache,
    transforms,
    params,
    *,
    cam_const=None,
    hflip: bool = False,
) -> np.ndarray:
    from src.core.camera import make_camera, project_points_cv

    K, D, rvec, tvec, R_w2c, t_w2c, fisheye = make_camera(params, transforms, cam_const)
    h, w = background.shape[:2]
    t_w2c_flat = t_w2c.flatten()

    all_tri_world = []
    all_colors = []
    for link_name, (tris, _) in mesh_cache.items():
        if link_name not in transforms:
            continue
        t_link, R_link = transforms[link_name]
        flat = tris.reshape(-1, 3)
        world = (R_link @ flat.T).T + t_link
        tri_world = world.reshape(-1, 3, 3)
        if "left" in link_name:
            color = COLOR_LEFT
        elif "right" in link_name:
            color = COLOR_RIGHT
        else:
            color = COLOR_BODY
        all_tri_world.append(tri_world)
        all_colors.extend([color] * len(tri_world))

    if not all_tri_world:
        return background.copy()

    tri_world = np.concatenate(all_tri_world, axis=0)
    flat = tri_world.reshape(-1, 3).astype(np.float64)
    cam_pts = (R_w2c @ flat.T).T + t_w2c_flat
    z_cam = cam_pts[:, 2]
    pts2d = project_points_cv(flat.reshape(-1, 1, 3), rvec, tvec, K, D, fisheye)
    pts2d = pts2d.reshape(-1, 2)

    n_tri = len(tri_world)
    z_tri = z_cam.reshape(n_tri, 3)
    pts_tri = pts2d.reshape(n_tri, 3, 2)
    valid = (z_tri > 0.01).all(axis=1)
    finite = np.all(np.isfinite(pts_tri), axis=(1, 2))
    vis_mask = valid & finite
    if vis_mask.sum() == 0:
        return background.copy()

    tri_sel = tri_world[vis_mask]
    pts_sel = pts_tri[vis_mask]
    z_sel = z_tri[vis_mask]
    colors = [all_colors[idx] for idx, flag in enumerate(vis_mask) if flag]

    v0, v1, v2 = tri_sel[:, 0], tri_sel[:, 1], tri_sel[:, 2]
    normals = np.cross(v1 - v0, v2 - v0)
    norms = np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-8)
    normals /= norms
    centroids_cam = (R_w2c @ tri_sel.mean(axis=1).T).T + t_w2c_flat
    view_dirs = -centroids_cam
    view_norms = np.maximum(np.linalg.norm(view_dirs, axis=1, keepdims=True), 1e-8)
    view_dirs /= view_norms
    dots = np.abs(np.sum(normals * view_dirs, axis=1))

    order = np.argsort(-z_sel.mean(axis=1))
    canvas = np.zeros_like(background)
    for rank in order:
        tri = pts_sel[rank].astype(np.int32)
        shade = 0.3 + 0.7 * dots[rank]
        shaded = tuple(int(c * shade) for c in colors[rank])
        cv2.fillPoly(canvas, [tri], shaded)

    if hflip:
        canvas = cv2.flip(canvas, 1)
    mesh_mask = canvas.any(axis=2)
    result = background.copy()
    result[mesh_mask] = canvas[mesh_mask]
    return result


def _inpaint_human_background(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    if not np.any(mask):
        raise ValueError("Cannot inpaint with an empty mask")
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    clean_mask = cv2.dilate(mask, kernel)
    return cv2.inpaint(frame, clean_mask, 5, cv2.INPAINT_TELEA)


def _resolve_pair_task_dir(pair_root: Path, task: str) -> Path:
    task_dir = pair_root / _short_task_name(task)
    if not task_dir.is_dir():
        raise FileNotFoundError(f"Pair task directory not found: {task_dir}")
    return task_dir


def _validate_manifest_record(record: dict, task_dir: Path, task: str) -> None:
    record_task = _short_task_name(str(record.get("task", record.get("robot_task", ""))))
    if record_task != task:
        raise ValueError(
            f"Manifest task mismatch in {task_dir}: {record_task!r} != {task!r}"
        )
    if str(record.get("data_type")) != "h2r":
        raise ValueError(f"Unsupported data_type in {task_dir}: {record.get('data_type')!r}")
    if str(record.get("duration")) != "1s":
        raise ValueError(
            f"Masquerade baseline only supports 1s pairs, got {record.get('duration')!r}"
        )
    if str(record.get("input_role")) != "human" or str(record.get("target_role")) != "robot":
        raise ValueError(
            f"Unexpected pair roles in {task_dir}: "
            f"input={record.get('input_role')!r} target={record.get('target_role')!r}"
        )


def _pair_records(pair_root: Path, task: str) -> tuple[Path, list[PairRecord]]:
    task_dir = _resolve_pair_task_dir(pair_root, task)
    manifest_rows = _read_jsonl(task_dir / "manifest.jsonl")
    parsed = []
    for row in manifest_rows:
        _validate_manifest_record(row, task_dir, task)
        human_src = Path(str(row.get("human_src", "")))
        robot_src = Path(str(row.get("robot_src", "")))
        if not human_src.is_file():
            raise FileNotFoundError(f"Human source not found: {human_src}")
        if not robot_src.is_file():
            raise FileNotFoundError(f"Robot source not found: {robot_src}")
        episode = str(row.get("episode", ""))
        seg = str(row.get("seg", ""))
        if not episode or not seg:
            raise ValueError(f"Manifest missing episode/seg in {task_dir}: {row}")
        segment_joints = SEGMENT_ROOT / task / episode / f"{seg}_joints.parquet"
        if not segment_joints.is_file():
            raise FileNotFoundError(f"Segment joints parquet not found: {segment_joints}")
        source_id = str(row.get("source_id", ""))
        if not source_id:
            raise ValueError(f"Manifest row missing source_id in {task_dir}: {row}")
        source_segment_id = str(row.get("source_segment_id", ""))
        if not source_segment_id:
            raise ValueError(f"Manifest row missing source_segment_id in {task_dir}: {row}")
        clip_idx = row.get("clip_idx")
        clip_idx = None if clip_idx is None else int(clip_idx)
        parsed.append({
            "source_id": source_id,
            "record": PairRecord(
                pair_id="",
                pair_index=-1,
                source_id=source_id,
                source_segment_id=source_segment_id,
                task=task,
                episode=episode,
                seg=seg,
                clip_idx=clip_idx,
                clip_start=float(row.get("clip_start", 0.0)),
                clip_dur=float(row.get("clip_dur", 1.0)),
                augment=str(row.get("augment", "normal")),
                human_src=human_src,
                robot_src=robot_src,
                control_video=task_dir / str(row.get("control_video", "")),
                robot_video=task_dir / str(row.get("video", "")),
                segment_joints=segment_joints,
                manifest=dict(row),
            ),
        })

    parsed.sort(key=lambda item: item["source_id"])
    records: list[PairRecord] = []
    for index, item in enumerate(parsed):
        record = item["record"]
        pair_id = f"pair_{index:04d}"
        control_video = task_dir / f"control_video/{pair_id}.mp4"
        robot_video = task_dir / f"video/{pair_id}.mp4"
        if not control_video.is_file():
            raise FileNotFoundError(f"Pair control video not found: {control_video}")
        if not robot_video.is_file():
            raise FileNotFoundError(f"Pair target video not found: {robot_video}")
        records.append(
            PairRecord(
                pair_id=pair_id,
                pair_index=index,
                source_id=record.source_id,
                source_segment_id=record.source_segment_id,
                task=record.task,
                episode=record.episode,
                seg=record.seg,
                clip_idx=record.clip_idx,
                clip_start=record.clip_start,
                clip_dur=record.clip_dur,
                augment=record.augment,
                human_src=record.human_src,
                robot_src=record.robot_src,
                control_video=control_video,
                robot_video=robot_video,
                segment_joints=record.segment_joints,
                manifest=record.manifest,
            )
        )
    return task_dir, records


def _parse_range(value: str, total: int) -> tuple[int, int]:
    match = re.match(r"^(\d*):(\d*)$", value)
    if not match:
        raise ValueError(f"--range must use START:END syntax, got {value!r}")
    start = int(match.group(1)) if match.group(1) else 0
    end = int(match.group(2)) if match.group(2) else total
    if start < 0 or end < start or end > total:
        raise ValueError(f"Invalid --range {value!r} for {total} pairs")
    return start, end


def select_records(
    records: list[PairRecord],
    *,
    num_samples: int = 0,
    head: int = 0,
    tail: int = 0,
    range_spec: str = "",
    pair_ids: set[str] | None = None,
) -> list[PairRecord]:
    selected = list(records)
    if pair_ids is not None:
        selected = [record for record in selected if record.pair_id in pair_ids]
    modes = sum(
        1
        for enabled in (num_samples > 0, head > 0, tail > 0, bool(range_spec))
        if enabled
    )
    if modes > 1:
        raise ValueError("Use only one of --num-samples, --head, --tail, or --range")
    if num_samples > 0:
        if num_samples > len(selected):
            raise ValueError(f"Requested {num_samples} pairs, only {len(selected)} available")
        return selected[:num_samples]
    if head > 0:
        if head > len(selected):
            raise ValueError(f"Requested head {head} pairs, only {len(selected)} available")
        return selected[:head]
    if tail > 0:
        if tail > len(selected):
            raise ValueError(f"Requested tail {tail} pairs, only {len(selected)} available")
        return selected[-tail:]
    if range_spec:
        start, end = _parse_range(range_spec, len(selected))
        return selected[start:end]
    return selected


def _read_video_frames(video_path: Path) -> tuple[list[np.ndarray], float, tuple[int, int]]:
    import av

    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")
    container = av.open(str(video_path))
    stream = container.streams.video[0]
    fps = float(stream.average_rate or TARGET_FPS)
    frames = [frame.to_ndarray(format="bgr24") for frame in container.decode(video=0)]
    container.close()
    if not frames:
        raise ValueError(f"Video has no frames: {video_path}")
    shape = frames[0].shape[:2]
    for frame in frames[1:]:
        if frame.shape[:2] != shape:
            raise ValueError(f"Frame shape changes inside video: {video_path}")
    return frames, fps, shape


def _frame_index_for_output(
    clip_start_sec: float,
    clip_dur_sec: float,
    output_index: int,
) -> int:
    clip_frames = max(1, int(round(clip_dur_sec * SOURCE_FPS)))
    base = int(round(clip_start_sec * SOURCE_FPS))
    offset = min(round(output_index * SOURCE_FPS / TARGET_FPS), clip_frames - 1)
    frame_idx = base + int(offset)
    if frame_idx < 0 or frame_idx >= SOURCE_VIDEO_FRAMES:
        raise IndexError(
            f"Source frame index out of range: clip_start={clip_start_sec} "
            f"clip_dur={clip_dur_sec} output_index={output_index} frame_idx={frame_idx}"
        )
    return frame_idx


def _load_segment_rows(joints_path: Path) -> "np.ndarray":
    import pandas as pd

    df = pd.read_parquet(joints_path)
    if "frame_idx" in df.columns:
        frame_col = "frame_idx"
    elif "frame_index" in df.columns:
        frame_col = "frame_index"
    else:
        raise ValueError(f"Missing frame index column in {joints_path}")
    required = {"observation.state.robot_q_current", "observation.state.hand_state"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in {joints_path}: {missing}")
    frame_map = {}
    frame_start = int(df[frame_col].min())
    for _, row in df.iterrows():
        frame_idx = int(row[frame_col]) - frame_start
        if frame_idx in frame_map:
            raise ValueError(f"Duplicate normalized frame index {frame_idx} in {joints_path}")
        frame_map[frame_idx] = row
    if not frame_map:
        raise ValueError(f"Empty joints parquet: {joints_path}")
    if min(frame_map) != 0:
        raise ValueError(f"Normalized segment frame indices must start at 0 in {joints_path}")
    return frame_map


def _build_robot_renderer(task: str):
    import pinocchio as pin
    from src.core.camera import make_camera_const
    from src.core.fk import parse_urdf_meshes, preload_meshes

    hand_type = get_hand_type(task)
    model = pin.buildModelFromUrdf(str(G1_URDF), pin.JointModelFreeFlyer())
    data = model.createData()
    link_meshes = parse_urdf_meshes(str(G1_URDF))
    mesh_cache = preload_meshes(link_meshes, str(MESH_DIR), skip_set=get_skip_meshes(hand_type))
    cam_const = make_camera_const(BEST_PARAMS)
    return model, data, mesh_cache, cam_const


def _mask_from_frame(frame: np.ndarray, background: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    bg_gray = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray, bg_gray)
    _, fg = cv2.threshold(diff, 24, 255, cv2.THRESH_BINARY)

    skin = cv2.inRange(
        cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb),
        np.array([0, 133, 77], dtype=np.uint8),
        np.array([255, 173, 127], dtype=np.uint8),
    )

    mask = cv2.bitwise_or(fg, skin)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    _, mask = cv2.threshold(mask, 80, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def _bbox_from_mask(mask: np.ndarray, *, x_offset: int = 0) -> list[int] | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return [
        int(xs.min()) + x_offset,
        int(ys.min()),
        int(xs.max()) + x_offset,
        int(ys.max()),
    ]


def _centroid_from_mask(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return [int(round(xs.mean())), int(round(ys.mean()))]


def _component_bbox(mask: np.ndarray) -> list[int] | None:
    if mask.size == 0 or not np.any(mask):
        return None
    return _bbox_from_mask(mask)


def _estimate_human_annotations(frames: list[np.ndarray]) -> dict:
    if not frames:
        raise ValueError("Human control video has no frames")
    if len(frames) != EXPECTED_OUTPUT_FRAMES:
        raise ValueError(
            f"Human control video frame count mismatch: {len(frames)} != {EXPECTED_OUTPUT_FRAMES}"
        )
    background = np.median(np.stack(frames, axis=0), axis=0).astype(np.uint8)
    masks = []
    bboxes = []
    left_bboxes = []
    right_bboxes = []
    centroids = []
    annotations = []

    for frame_idx, frame in enumerate(frames):
        mask = _mask_from_frame(frame, background)
        if not np.any(mask):
            raise ValueError(f"Empty human mask for frame {frame_idx}")
        h, w = mask.shape
        mid = w // 2
        left_mask = mask[:, :mid]
        right_mask = mask[:, mid:]
        bbox = _bbox_from_mask(mask)
        left_bbox = _component_bbox(left_mask)
        right_bbox = _bbox_from_mask(right_mask, x_offset=mid)
        centroid = _centroid_from_mask(mask)
        if bbox is None or centroid is None:
            raise ValueError(f"Failed to estimate human annotations for frame {frame_idx}")
        masks.append(mask)
        bboxes.append(bbox)
        left_bboxes.append(left_bbox if left_bbox is not None else [-1, -1, -1, -1])
        right_bboxes.append(right_bbox if right_bbox is not None else [-1, -1, -1, -1])
        centroids.append(centroid)
        left_center = None
        right_center = None
        if left_bbox is not None:
            left_center = [
                int(round((left_bbox[0] + left_bbox[2]) / 2.0)),
                int(round((left_bbox[1] + left_bbox[3]) / 2.0)),
            ]
        if right_bbox is not None:
            right_center = [
                int(round((right_bbox[0] + right_bbox[2]) / 2.0)),
                int(round((right_bbox[1] + right_bbox[3]) / 2.0)),
            ]
        annotations.append({
            "frame_index": frame_idx,
            "bbox_xyxy": bbox,
            "left_bbox_xyxy": left_bbox,
            "right_bbox_xyxy": right_bbox,
            "centroid_xy": centroid,
            "mask_area": int(np.count_nonzero(mask)),
            "keypoints_xy": {
                "centroid": centroid,
                "left_center": left_center,
                "right_center": right_center,
                "top": [centroid[0], bbox[1]],
                "bottom": [centroid[0], bbox[3]],
            },
        })

    trajectory = [
        {"frame_index": idx, "xy": centroid}
        for idx, centroid in enumerate(centroids)
    ]
    return {
        "background": background,
        "masks": np.stack(masks, axis=0).astype(np.uint8),
        "bboxes": np.asarray(bboxes, dtype=np.int32),
        "left_bboxes": np.asarray(left_bboxes, dtype=np.int32),
        "right_bboxes": np.asarray(right_bboxes, dtype=np.int32),
        "centroids": np.asarray(centroids, dtype=np.int32),
        "annotations": annotations,
        "trajectory": trajectory,
    }


def _draw_human_overlay(frame: np.ndarray, annotation: dict, mask: np.ndarray) -> np.ndarray:
    out = frame.copy()
    overlay = out.copy()
    overlay[mask > 0] = (0, 180, 255)
    cv2.addWeighted(overlay, 0.35, out, 0.65, 0, out)

    for bbox, color in (
        (annotation["bbox_xyxy"], (0, 255, 0)),
        (annotation["left_bbox_xyxy"], (255, 180, 0)),
        (annotation["right_bbox_xyxy"], (0, 180, 255)),
    ):
        if bbox is None:
            continue
        x1, y1, x2, y2 = bbox
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

    c_x, c_y = annotation["centroid_xy"]
    cv2.circle(out, (c_x, c_y), 4, (255, 255, 255), -1)
    cv2.putText(
        out,
        f"frame={annotation['frame_index']}",
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return out


def _write_video(frames: list[np.ndarray], out_path: Path, fps: int = TARGET_FPS) -> None:
    from src.core.data import close_video, open_video_writer, write_frame

    if not frames:
        raise ValueError(f"No frames to write: {out_path}")
    h, w = frames[0].shape[:2]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    container, stream = open_video_writer(str(out_path), w, h, fps=fps)
    for frame in frames:
        write_frame(container, stream, frame)
    close_video(container, stream)


def _stack_compare(
    human_frames: list[np.ndarray],
    baseline_frames: list[np.ndarray],
    gt_frames: list[np.ndarray],
    out_path: Path,
) -> None:
    if not (len(human_frames) == len(baseline_frames) == len(gt_frames)):
        raise ValueError(
            "Compare frame count mismatch: "
            f"human={len(human_frames)} baseline={len(baseline_frames)} gt={len(gt_frames)}"
        )
    compare_frames = []
    for idx, (human, baseline, gt) in enumerate(zip(human_frames, baseline_frames, gt_frames)):
        h, w = human.shape[:2]
        if baseline.shape[:2] != (h, w) or gt.shape[:2] != (h, w):
            raise ValueError(f"Compare frame shape mismatch at index {idx}")
        combined = np.hstack([human, baseline, gt])
        cv2.putText(
            combined,
            "human control",
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            combined,
            "baseline render",
            (w + 10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            combined,
            "GT robot",
            (2 * w + 10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        compare_frames.append(combined)
    _write_video(compare_frames, out_path, fps=TARGET_FPS)


def _build_output_paths(output_root: Path, data_type: str, duration: str, task: str) -> dict[str, Path]:
    task_dir = output_root / data_type / duration / task
    return {
        "task_dir": task_dir,
        "video_dir": task_dir / PAIR_DIRNAME,
        "human_dir": task_dir / HUMAN_DIRNAME,
        "gt_dir": task_dir / GT_DIRNAME,
        "background_dir": task_dir / BACKGROUND_DIRNAME,
        "human_overlay_dir": task_dir / HUMAN_OVERLAY_DIRNAME,
        "annotation_dir": task_dir / ANNOTATION_DIRNAME,
        "compare_dir": task_dir / COMPARE_DIRNAME,
        "manifest": task_dir / "manifest.jsonl",
        "summary": task_dir / "summary.json",
    }


def _select_pair_ids(value: str) -> set[str]:
    if not value:
        return set()
    ids = {item.strip() for item in value.split(",") if item.strip()}
    if not ids:
        raise ValueError("--pair-id must not be empty")
    for pair_id in ids:
        if not re.match(r"^pair_\d{4}$", pair_id):
            raise ValueError(f"Invalid pair_id: {pair_id!r}")
    return ids


def process_pair(record: PairRecord, output_root: Path, *, background_mode: str) -> dict:
    output_paths = _build_output_paths(output_root, "h2r", "1s", record.task)
    for key in ("video_dir", "human_dir", "gt_dir", "background_dir", "human_overlay_dir", "annotation_dir", "compare_dir"):
        output_paths[key].mkdir(parents=True, exist_ok=True)

    human_frames, human_fps, human_shape = _read_video_frames(record.control_video)
    gt_frames, gt_fps, gt_shape = _read_video_frames(record.robot_video)
    if len(human_frames) != EXPECTED_OUTPUT_FRAMES:
        raise ValueError(
            f"Human clip frame count mismatch for {record.control_video}: "
            f"{len(human_frames)} != {EXPECTED_OUTPUT_FRAMES}"
        )
    if len(gt_frames) != EXPECTED_OUTPUT_FRAMES:
        raise ValueError(
            f"GT robot clip frame count mismatch for {record.robot_video}: "
            f"{len(gt_frames)} != {EXPECTED_OUTPUT_FRAMES}"
        )
    if human_shape != gt_shape:
        raise ValueError(
            f"Human/GT frame shape mismatch for {record.pair_id}: "
            f"human={human_shape} gt={gt_shape}"
        )
    if int(round(human_fps)) != TARGET_FPS or int(round(gt_fps)) != TARGET_FPS:
        raise ValueError(
            f"Expected 16fps pair clips, got human={human_fps} gt={gt_fps} for {record.pair_id}"
        )

    annotations = _estimate_human_annotations(human_frames)
    background_frames = [
        _inpaint_human_background(frame, mask)
        if background_mode == "inpaint"
        else np.zeros_like(frame)
        for frame, mask in zip(human_frames, annotations["masks"])
    ]
    human_overlay_frames = [
        _draw_human_overlay(frame, anno, mask)
        for frame, anno, mask in zip(human_frames, annotations["annotations"], annotations["masks"])
    ]

    renderer = _build_robot_renderer(record.task)
    model, data, mesh_cache, cam_const = renderer
    from src.core.fk import build_q, do_fk
    segment_rows = _load_segment_rows(record.segment_joints)
    source_frames, source_fps, source_shape = _read_video_frames(record.robot_src)
    if int(round(source_fps)) != SOURCE_FPS:
        raise ValueError(
            f"Expected 30fps source segment video, got {source_fps} for {record.robot_src}"
        )
    if len(source_frames) != SOURCE_VIDEO_FRAMES:
        raise ValueError(
            f"Expected 120 frames in source segment video, got {len(source_frames)} for {record.robot_src}"
        )
    if source_shape != human_shape:
        raise ValueError(
            f"Source video shape mismatch for {record.robot_src}: {source_shape} != {human_shape}"
        )

    baseline_frames: list[np.ndarray] = []
    source_frame_indices = []
    for out_idx in range(EXPECTED_OUTPUT_FRAMES):
        frame_idx = _frame_index_for_output(record.clip_start, record.clip_dur, out_idx)
        source_frame_indices.append(frame_idx)
        if frame_idx not in segment_rows:
            raise KeyError(f"Missing joints row for source frame {frame_idx} in {record.segment_joints}")
        row = segment_rows[frame_idx]
        rq = np.array(row["observation.state.robot_q_current"], dtype=np.float64)
        hs = np.array(row["observation.state.hand_state"], dtype=np.float64)
        hand_type = get_hand_type(record.task)
        q = build_q(model, rq, hs, hand_type=hand_type)
        transforms = do_fk(model, data, q)
        rendered = _render_robot_opaque(
            background_frames[out_idx],
            mesh_cache,
            transforms,
            BEST_PARAMS,
            cam_const=cam_const,
            hflip=record.augment == "hflip",
        )
        baseline_frames.append(rendered)

    video_out = output_paths["video_dir"] / f"{record.pair_id}.mp4"
    control_out = output_paths["human_dir"] / f"{record.pair_id}.mp4"
    background_out = output_paths["background_dir"] / f"{record.pair_id}.mp4"
    human_overlay_out = output_paths["human_overlay_dir"] / f"{record.pair_id}.mp4"
    compare_out = output_paths["compare_dir"] / f"{record.pair_id}.mp4"
    human_annotation_out = output_paths["annotation_dir"] / f"{record.pair_id}.jsonl"
    mask_out = output_paths["annotation_dir"] / f"{record.pair_id}.npz"
    gt_out = output_paths["gt_dir"] / f"{record.pair_id}.mp4"

    _write_video(baseline_frames, video_out, fps=TARGET_FPS)
    _write_video(human_frames, control_out, fps=TARGET_FPS)
    _write_video(background_frames, background_out, fps=TARGET_FPS)
    _write_video(human_overlay_frames, human_overlay_out, fps=TARGET_FPS)
    _write_video(gt_frames, gt_out, fps=TARGET_FPS)
    _stack_compare(human_frames, baseline_frames, gt_frames, compare_out)
    _write_jsonl(human_annotation_out, annotations["annotations"])
    np.savez_compressed(
        mask_out,
        masks=annotations["masks"],
        bboxes=annotations["bboxes"],
        left_bboxes=annotations["left_bboxes"],
        right_bboxes=annotations["right_bboxes"],
        centroids=annotations["centroids"],
        trajectory=np.asarray([item["xy"] for item in annotations["trajectory"]], dtype=np.int32),
    )

    manifest_row = {
        "pair_id": record.pair_id,
        "source_id": record.source_id,
        "source_segment_id": record.source_segment_id,
        "data_type": "h2r",
        "duration": "1s",
        "task": record.task,
        "robot_task": record.task,
        "episode": record.episode,
        "seg": record.seg,
        "clip_idx": record.clip_idx,
        "clip_start": record.clip_start,
        "clip_dur": record.clip_dur,
        "augment": record.augment,
        "human_src": str(record.human_src),
        "robot_src": str(record.robot_src),
        "control_video_source": str(record.control_video),
        "gt_robot_source": str(record.robot_video),
        "segment_joints": str(record.segment_joints),
        "baseline_video": str(video_out.relative_to(output_paths["task_dir"])),
        "control_video": str(control_out.relative_to(output_paths["task_dir"])),
        "background_video": str(background_out.relative_to(output_paths["task_dir"])),
        "human_overlay": str(human_overlay_out.relative_to(output_paths["task_dir"])),
        "compare_video": str(compare_out.relative_to(output_paths["task_dir"])),
        "human_annotation_jsonl": str(human_annotation_out.relative_to(output_paths["task_dir"])),
        "human_mask_npz": str(mask_out.relative_to(output_paths["task_dir"])),
        "gt_video": str(gt_out.relative_to(output_paths["task_dir"])),
        "background_mode": background_mode,
        "source_frame_indices": source_frame_indices,
        "frame_count": EXPECTED_OUTPUT_FRAMES,
        "frame_size": [human_shape[1], human_shape[0]],
        "human_fps": human_fps,
        "robot_fps": gt_fps,
        "source_fps": source_fps,
    }
    return manifest_row


def main() -> None:
    parser = argparse.ArgumentParser(description="Masquerade-style direct render baseline for h2r pairs")
    parser.add_argument("--task", default="all", help="task short name, comma list, or all/training")
    parser.add_argument("--pair-root", default=str(PAIR_ROOT), help="root containing h2r/1s/<task>/ manifests")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="baseline output root")
    parser.add_argument("--num-samples", type=int, default=0, help="select the first N pairs after sorting")
    parser.add_argument("--head", type=int, default=0, help="select the first N pairs after filtering")
    parser.add_argument("--tail", type=int, default=0, help="select the last N pairs after filtering")
    parser.add_argument("--range", dest="range_spec", default="", help="select a closed-open slice START:END")
    parser.add_argument("--pair-id", default="", help="comma-separated pair ids such as pair_0000,pair_0007")
    parser.add_argument("--background-mode", choices=["inpaint", "black"], default="inpaint")
    parser.add_argument("--list-only", action="store_true", help="only validate and print selected pairs")
    args = parser.parse_args()

    if args.num_samples < 0 or args.head < 0 or args.tail < 0:
        raise ValueError("Selection counts must be non-negative")
    if args.background_mode not in {"inpaint", "black"}:
        raise ValueError(f"Unsupported background mode: {args.background_mode}")

    pair_root = Path(args.pair_root)
    output_root = Path(args.output_root)
    tasks = expand_task_spec(args.task)
    pair_ids = _select_pair_ids(args.pair_id)

    all_manifests = []
    print("Masquerade baseline")
    print(f"  pair root:    {pair_root}")
    print(f"  output root:  {output_root}")
    print(f"  tasks:        {tasks}")
    print(f"  background:   {args.background_mode}")
    print(f"  selection:    head={args.head} tail={args.tail} num={args.num_samples} range={args.range_spec!r}")
    if pair_ids:
        print(f"  pair ids:     {sorted(pair_ids)}")

    for task in tasks:
        task_dir, records = _pair_records(pair_root, task)
        selected = select_records(
            records,
            num_samples=args.num_samples,
            head=args.head,
            tail=args.tail,
            range_spec=args.range_spec,
            pair_ids=pair_ids or None,
        )
        if not selected:
            raise ValueError(f"No pairs selected for task {task}")
        print(f"  {task}: {len(selected)} / {len(records)} pairs from {task_dir}")
        if args.list_only:
            for record in selected:
                print(f"    {record.pair_id}  {record.source_id}")
            continue

        output_paths = _build_output_paths(output_root, "h2r", "1s", task)
        output_paths["task_dir"].mkdir(parents=True, exist_ok=True)
        task_rows = []
        for record in selected:
            print(f"    processing {record.pair_id}  {record.source_id}")
            row = process_pair(record, output_root, background_mode=args.background_mode)
            task_rows.append(row)
            all_manifests.append(row)
        task_rows.sort(key=lambda row: row["pair_id"])
        _write_jsonl(output_paths["manifest"], task_rows)
        _write_json(
            output_paths["summary"],
            {
                "task": task,
                "data_type": "h2r",
                "duration": "1s",
                "pair_count": len(task_rows),
                "output_root": str(output_root),
                "pair_root": str(pair_root),
                "background_mode": args.background_mode,
            },
        )
        print(f"  wrote: {output_paths['manifest']}")

    if args.list_only:
        return

    if all_manifests:
        print(f"Done: {len(all_manifests)} baseline pairs written under {output_root}")


if __name__ == "__main__":
    main()
