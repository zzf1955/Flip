"""Precompute visible body-part and action masks for Wan VAE IDM.

The precompute is FK/mesh based and does not invoke SAM2.  For each robot
segment it renders the target-mode body parts frame by frame, checks
that at least one link origin projects inside the image, and writes both
part-level visibility and the derived IDM action mask.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import re
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pinocchio as pin

from src.core.camera import make_camera, project_points_cv
from src.core.config import (
    ALL_TASKS,
    BEST_PARAMS,
    G1_URDF,
    MAIN_ROOT,
    MESH_DIR,
    get_hand_type,
    get_skip_meshes,
)
from src.core.fk import build_q, do_fk, parse_urdf_meshes, preload_meshes
from src.pipeline.action_mask import (
    ACTION_MASK_VERSION,
    action_mask_from_part_visibility,
    action_dim_names_for_mode,
    action_dim_parts_for_mode,
    body_part_names_for_mode,
    default_action_mask_root,
    validate_target_mode,
)
from src.pipeline.sam2_precompute import BODY_PARTS, match_links, render_mask_for_links

DEFAULT_SEGMENT_ROOT = Path(MAIN_ROOT) / "training_data" / "segment"
SEGMENT_FPS = 30.0
_WORKER_FK_MODEL = None
_WORKER_FK_DATA = None
_WORKER_LINK_MESHES = None
_WORKER_CACHES_BY_HAND: dict[str, dict] = {}
_WORKER_TARGET_MODE = ""


def parse_tasks(value: str) -> list[str]:
    key = value.strip().lower()
    short_tasks = [task.replace("G1_WBT_", "") for task in ALL_TASKS]
    if key == "all":
        return short_tasks
    if key == "h2r":
        allowed = {
            "Inspire_Collect_Clothes_MainCamOnly",
            "Inspire_Put_Clothes_into_Washing_Machine",
            "Inspire_Put_Clothes_into_Washing_Machine_MainCamOnly",
            "Inspire_Pickup_Pillow_MainCamOnly",
        }
        return [task for task in short_tasks if task in allowed]
    return [item.strip() for item in value.split(",") if item.strip()]


def find_segments(
    segment_root: Path,
    tasks: list[str],
    episodes: set[str] | None,
) -> list[dict]:
    segments = []
    for task in tasks:
        task_dir = segment_root / task
        if not task_dir.is_dir():
            raise FileNotFoundError(f"Segment task directory not found: {task_dir}")
        for video_path in sorted(task_dir.glob("ep*/seg*_video.mp4")):
            match = re.match(r"(seg\d+)_video\.mp4$", video_path.name)
            if match is None:
                continue
            episode = video_path.parent.name
            if episodes is not None and episode not in episodes:
                continue
            segment = match.group(1)
            parquet_path = video_path.with_name(f"{segment}_joints.parquet")
            if not parquet_path.is_file():
                raise FileNotFoundError(f"Segment joints parquet not found: {parquet_path}")
            segments.append({
                "task": task,
                "episode": episode,
                "segment": segment,
                "video_path": video_path,
                "parquet_path": parquet_path,
            })
    segments.sort(key=lambda row: (row["task"], row["episode"], row["segment"]))
    return segments


def load_fk_model():
    model = pin.buildModelFromUrdf(str(G1_URDF), pin.JointModelFreeFlyer())
    data = model.createData()
    link_meshes = parse_urdf_meshes(str(G1_URDF))
    return model, data, link_meshes


def init_action_mask_worker(target_mode: str) -> None:
    global _WORKER_FK_MODEL
    global _WORKER_FK_DATA
    global _WORKER_LINK_MESHES
    global _WORKER_CACHES_BY_HAND
    global _WORKER_TARGET_MODE
    _WORKER_FK_MODEL, _WORKER_FK_DATA, _WORKER_LINK_MESHES = load_fk_model()
    _WORKER_CACHES_BY_HAND = {}
    _WORKER_TARGET_MODE = validate_target_mode(target_mode)


def build_action_part_caches(link_meshes, hand_type: str, target_mode: str):
    skip_set = get_skip_meshes(hand_type)
    mesh_cache = preload_meshes(link_meshes, str(MESH_DIR), skip_set)
    return {
        part_name: match_links(mesh_cache, BODY_PARTS[part_name], skip_set)
        for part_name in body_part_names_for_mode(target_mode)
    }


def video_shape(video_path: Path) -> tuple[int, int, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open segment video: {video_path}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if frame_count <= 0 or width <= 0 or height <= 0:
        raise RuntimeError(
            f"Invalid segment video metadata for {video_path}: "
            f"frames={frame_count} width={width} height={height}"
        )
    return frame_count, height, width


def clip_frame_indices(
    clip_start: float,
    clip_dur: float,
    num_frames: int,
    target_fps: float,
) -> list[int]:
    if clip_start < 0:
        raise ValueError(f"clip_start must be non-negative, got {clip_start}")
    if clip_dur <= 0:
        raise ValueError(f"clip_dur must be positive, got {clip_dur}")
    if num_frames <= 0:
        raise ValueError(f"num_frames must be positive, got {num_frames}")
    if target_fps <= 0:
        raise ValueError(f"target_fps must be positive, got {target_fps}")
    base = int(round(clip_start * SEGMENT_FPS))
    clip_frames = max(1, int(round(clip_dur * SEGMENT_FPS)))
    return [
        base + min(int(round(i * SEGMENT_FPS / target_fps)), clip_frames - 1)
        for i in range(num_frames)
    ]


def clip_middle_frame_indices(
    n_frames: int,
    *,
    clip_dur: float,
    clip_stride: float,
    num_frames: int,
    target_fps: float,
) -> list[int]:
    if clip_stride <= 0:
        raise ValueError(f"clip_stride must be positive, got {clip_stride}")
    max_start = (n_frames / SEGMENT_FPS) - clip_dur
    if max_start < -1e-6:
        return []
    middle_indices = []
    value = 0.0
    while value <= max_start + 1e-6:
        indices = clip_frame_indices(value, clip_dur, num_frames, target_fps)
        middle_indices.append(indices[len(indices) // 2])
        value += clip_stride
    return sorted(set(int(idx) for idx in middle_indices))


def link_origin_in_frame_counts(
    filtered_cache: dict,
    transforms: dict,
    params: dict,
    height: int,
    width: int,
) -> int:
    if not filtered_cache:
        return 0
    K, D, rvec, tvec, r_w2c, t_w2c, fisheye = make_camera(params, transforms)
    origins = []
    for link_name in filtered_cache:
        if link_name not in transforms:
            continue
        t_link, _r_link = transforms[link_name]
        origins.append(np.asarray(t_link, dtype=np.float64).reshape(3))
    if not origins:
        return 0
    origins_np = np.asarray(origins, dtype=np.float64)
    cam = (r_w2c @ origins_np.T).T + t_w2c.flatten()
    in_front = cam[:, 2] > 0.01
    world = origins_np.reshape(-1, 1, 3)
    pts2d = project_points_cv(world, rvec, tvec, K, D, fisheye).reshape(-1, 2)
    finite = np.isfinite(pts2d).all(axis=1)
    inside = (
        finite
        & in_front
        & (pts2d[:, 0] >= 0.0)
        & (pts2d[:, 0] < float(width))
        & (pts2d[:, 1] >= 0.0)
        & (pts2d[:, 1] < float(height))
    )
    return int(inside.sum())


def segment_masks(
    seg: dict,
    fk_model,
    fk_data,
    part_caches: dict,
    *,
    min_part_pixels: int,
    require_projected_origin: bool,
    target_mode: str,
    clip_middle_only: bool,
    clip_duration: float,
    clip_stride: float,
    num_frames: int,
    target_fps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    frame_count, height, width = video_shape(seg["video_path"])
    df = pd.read_parquet(seg["parquet_path"])
    required = {
        "observation.state.robot_q_current",
        "observation.state.hand_state",
        "frame_index",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Segment parquet missing {missing}: {seg['parquet_path']}")
    if df.empty:
        raise ValueError(f"Segment parquet is empty: {seg['parquet_path']}")

    n_frames = min(frame_count, len(df))
    if n_frames <= 0:
        raise ValueError(f"No aligned frames for segment: {seg['video_path']}")
    part_names = list(body_part_names_for_mode(target_mode))
    part_pixel_counts = np.zeros((n_frames, len(part_names)), dtype=np.int32)
    part_origin_counts = np.zeros((n_frames, len(part_names)), dtype=np.int16)
    part_visibility = np.zeros((n_frames, len(part_names)), dtype=bool)
    if clip_middle_only:
        rendered_frame_indices = clip_middle_frame_indices(
            n_frames,
            clip_dur=clip_duration,
            clip_stride=clip_stride,
            num_frames=num_frames,
            target_fps=target_fps,
        )
        if not rendered_frame_indices:
            raise ValueError(f"No clip middle frames for segment: {seg['video_path']}")
    else:
        rendered_frame_indices = list(range(n_frames))

    task_full = f"G1_WBT_{seg['task']}"
    hand_type = get_hand_type(task_full)
    for frame_idx in rendered_frame_indices:
        row = df.iloc[frame_idx]
        robot_q = np.asarray(row["observation.state.robot_q_current"], dtype=np.float64)
        hand_state = np.asarray(row["observation.state.hand_state"], dtype=np.float64)
        q = build_q(fk_model, robot_q, hand_state, hand_type=hand_type)
        transforms = do_fk(fk_model, fk_data, q)
        for part_idx, part_name in enumerate(part_names):
            cache = part_caches[part_name]
            mask = render_mask_for_links(cache, transforms, BEST_PARAMS, height, width)
            pixel_count = int(np.count_nonzero(mask))
            origin_count = link_origin_in_frame_counts(
                cache, transforms, BEST_PARAMS, height, width,
            )
            part_pixel_counts[frame_idx, part_idx] = pixel_count
            part_origin_counts[frame_idx, part_idx] = origin_count
            origin_ok = origin_count > 0 or not require_projected_origin
            part_visibility[frame_idx, part_idx] = (
                pixel_count >= min_part_pixels and origin_ok
            )

    metadata = {
        "version": ACTION_MASK_VERSION,
        "target_mode": target_mode,
        "task": seg["task"],
        "task_full": task_full,
        "episode": seg["episode"],
        "segment": seg["segment"],
        "source_video_path": str(seg["video_path"]),
        "source_parquet_path": str(seg["parquet_path"]),
        "fps": SEGMENT_FPS,
        "num_frames": int(n_frames),
        "image_size": [int(height), int(width)],
        "min_part_pixels": int(min_part_pixels),
        "require_projected_origin": bool(require_projected_origin),
        "camera_params": {key: float(value) for key, value in BEST_PARAMS.items()},
        "part_names": part_names,
        "action_dim_names": list(action_dim_names_for_mode(target_mode)),
        "action_dim_parts": list(action_dim_parts_for_mode(target_mode)),
        "clip_middle_only": bool(clip_middle_only),
        "rendered_frame_indices": [int(idx) for idx in rendered_frame_indices],
        "clip_duration": float(clip_duration),
        "clip_stride": float(clip_stride),
        "num_clip_frames": int(num_frames),
        "target_fps": float(target_fps),
    }
    return part_visibility, part_pixel_counts, part_origin_counts, metadata


def write_index(rows: list[dict], output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    index_path = output_root / "index.jsonl"
    with index_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def validate_existing_artifact(path: Path, seg: dict, target_mode: str) -> dict:
    with np.load(path, allow_pickle=False) as data:
        if "metadata_json" not in data:
            raise KeyError(f"metadata_json missing from artifact: {path}")
        if "action_mask" not in data:
            raise KeyError(f"action_mask missing from artifact: {path}")
        action_mask = data["action_mask"].astype(bool)
        metadata = json.loads(str(data["metadata_json"]))
    checks = {
        "version": ACTION_MASK_VERSION,
        "target_mode": target_mode,
        "task": seg["task"],
        "episode": seg["episode"],
        "segment": seg["segment"],
    }
    for key, expected in checks.items():
        if metadata.get(key) != expected:
            raise ValueError(
                f"Existing action mask metadata mismatch for {path}: "
                f"{key} expected {expected!r}, got {metadata.get(key)!r}"
            )
    rendered_indices = metadata.get("rendered_frame_indices")
    if rendered_indices:
        visible_action_mask = action_mask[np.asarray(rendered_indices, dtype=np.int64)]
    else:
        visible_action_mask = action_mask
    visible_counts = visible_action_mask.sum(axis=0)
    if target_mode == "arm_hand":
        arm_mask = visible_action_mask[:, :12]
        hand_mask = visible_action_mask[:, 12:]
    else:
        arm_mask = visible_action_mask[:, 22:36]
        hand_mask = visible_action_mask[:, 36:48]
    return {
        "task": seg["task"],
        "episode": seg["episode"],
        "segment": seg["segment"],
        "path": str(path),
        "target_mode": target_mode,
        "status": "skipped_existing",
        "clip_middle_only": bool(metadata.get("clip_middle_only", False)),
        "rendered_frame_count": int(visible_action_mask.shape[0]),
        "visible_action_ratio_mean": float(visible_action_mask.mean()),
        "visible_arm_ratio_mean": float(arm_mask.mean()),
        "visible_hand_ratio_mean": float(hand_mask.mean()),
        "visible_dim_frame_counts": [int(v) for v in visible_counts.tolist()],
    }


def process_segment_job(job: dict) -> dict:
    if _WORKER_FK_MODEL is None or _WORKER_FK_DATA is None or _WORKER_LINK_MESHES is None:
        raise RuntimeError("Action mask worker was not initialized")
    seg = job["segment"]
    output_root = Path(job["output_root"])
    target_mode = str(job["target_mode"])
    if target_mode != _WORKER_TARGET_MODE:
        raise ValueError(
            f"Worker target_mode mismatch: expected {_WORKER_TARGET_MODE!r}, got {target_mode!r}"
        )
    task_full = f"G1_WBT_{seg['task']}"
    hand_type = get_hand_type(task_full)
    if hand_type not in _WORKER_CACHES_BY_HAND:
        _WORKER_CACHES_BY_HAND[hand_type] = build_action_part_caches(
            _WORKER_LINK_MESHES,
            hand_type,
            target_mode,
        )
    out_dir = output_root / seg["task"] / seg["episode"]
    out_path = out_dir / f"{seg['segment']}.npz"
    if bool(job["resume"]) and out_path.is_file():
        return validate_existing_artifact(out_path, seg, target_mode)

    part_visibility, part_pixels, part_origins, metadata = segment_masks(
        seg,
        _WORKER_FK_MODEL,
        _WORKER_FK_DATA,
        _WORKER_CACHES_BY_HAND[hand_type],
        min_part_pixels=int(job["min_part_pixels"]),
        require_projected_origin=bool(job["require_projected_origin"]),
        target_mode=target_mode,
        clip_middle_only=bool(job["clip_middle_only"]),
        clip_duration=float(job["clip_duration"]),
        clip_stride=float(job["clip_stride"]),
        num_frames=int(job["num_frames"]),
        target_fps=float(job["target_fps"]),
    )
    action_mask = action_mask_from_part_visibility(
        part_visibility,
        metadata["part_names"],
        target_mode=target_mode,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        part_names=np.asarray(metadata["part_names"]),
        action_dim_names=np.asarray(action_dim_names_for_mode(target_mode)),
        action_dim_parts=np.asarray(action_dim_parts_for_mode(target_mode)),
        part_visibility=part_visibility.astype(np.uint8),
        part_pixel_counts=part_pixels,
        part_origin_in_frame_counts=part_origins,
        action_mask=action_mask.astype(np.uint8),
        metadata_json=json.dumps(metadata, ensure_ascii=False),
    )
    rendered_indices = np.asarray(metadata["rendered_frame_indices"], dtype=np.int64)
    visible_action_mask = action_mask[rendered_indices]
    visible_counts = visible_action_mask.sum(axis=0)
    if target_mode == "arm_hand":
        arm_mask = visible_action_mask[:, :12]
        hand_mask = visible_action_mask[:, 12:]
    else:
        arm_mask = visible_action_mask[:, 22:36]
        hand_mask = visible_action_mask[:, 36:48]
    return {
        "task": seg["task"],
        "episode": seg["episode"],
        "segment": seg["segment"],
        "path": str(out_path),
        "target_mode": target_mode,
        "status": "written",
        "num_frames": int(action_mask.shape[0]),
        "image_size": metadata["image_size"],
        "clip_middle_only": bool(job["clip_middle_only"]),
        "rendered_frame_count": len(metadata["rendered_frame_indices"]),
        "visible_action_ratio_mean": float(visible_action_mask.mean()),
        "visible_arm_ratio_mean": float(arm_mask.mean()),
        "visible_hand_ratio_mean": float(hand_mask.mean()),
        "visible_dim_frame_counts": [int(v) for v in visible_counts.tolist()],
    }


def process_segments(args: argparse.Namespace) -> None:
    segment_root = Path(args.segment_root)
    output_root = Path(args.output)
    tasks = parse_tasks(args.task)
    target_mode = validate_target_mode(args.target_mode)
    episodes = None
    if args.episodes:
        episodes = {
            item if item.startswith("ep") else f"ep{int(item):03d}"
            for item in args.episodes.split(",")
            if item.strip()
        }
    segments = find_segments(segment_root, tasks, episodes)
    if args.max_segments > 0:
        segments = segments[: args.max_segments]
    if not segments:
        raise ValueError(
            f"No segments found under {segment_root} for task={args.task!r}"
        )

    jobs = [
        {
            "segment": seg,
            "output_root": str(output_root),
            "target_mode": target_mode,
            "resume": bool(args.resume),
            "min_part_pixels": int(args.min_part_pixels),
            "require_projected_origin": not args.no_require_projected_origin,
            "clip_middle_only": bool(args.clip_middle_only),
            "clip_duration": float(args.clip_duration),
            "clip_stride": float(args.clip_stride),
            "num_frames": int(args.num_frames),
            "target_fps": float(args.target_fps),
        }
        for seg in segments
    ]
    index_rows = []
    t0 = time.time()

    if args.workers == 1:
        init_action_mask_worker(target_mode)
        row_iter = (process_segment_job(job) for job in jobs)
    else:
        ctx = mp.get_context("spawn")
        pool = ctx.Pool(
            processes=args.workers,
            initializer=init_action_mask_worker,
            initargs=(target_mode,),
        )
        row_iter = pool.imap_unordered(process_segment_job, jobs)
    try:
        for idx, row in enumerate(row_iter, start=1):
            index_rows.append(row)
            elapsed = time.time() - t0
            visible = row.get("visible_action_ratio_mean")
            visible_text = "n/a" if visible is None else f"{float(visible):.3f}"
            print(
                f"[{idx}/{len(segments)}] {row['task']}/{row['episode']}/{row['segment']} "
                f"status={row['status']} visible={visible_text} elapsed={elapsed:.1f}s",
                flush=True,
            )
    finally:
        if args.workers != 1:
            pool.close()
            pool.join()

    index_rows.sort(key=lambda row: (str(row["task"]), str(row["episode"]), str(row["segment"])))

    write_index(index_rows, output_root)
    print(
        json.dumps(
            {
                "segments": len(segments),
                "output": str(output_root),
                "index": str(output_root / "index.jsonl"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Precompute FK visible part masks and IDM action masks"
    )
    parser.add_argument(
        "--task",
        required=True,
        help="task filter: all, h2r, or comma-separated task-short names",
    )
    parser.add_argument("--episodes", default="", help="comma-separated ep000 or integer ids")
    parser.add_argument("--segment-root", default=str(DEFAULT_SEGMENT_ROOT))
    parser.add_argument("--output", default=str(default_action_mask_root()))
    parser.add_argument("--max-segments", type=int, default=0)
    parser.add_argument("--target-mode", choices=["arm_hand", "full_body"], default="arm_hand")
    parser.add_argument("--min-part-pixels", type=int, default=50)
    parser.add_argument(
        "--clip-middle-only",
        action="store_true",
        help="render only middle frames used by Wan VAE IDM clips for faster training masks",
    )
    parser.add_argument("--clip-duration", type=float, default=1.0)
    parser.add_argument("--clip-stride", type=float, default=1.0)
    parser.add_argument("--num-frames", type=int, default=17)
    parser.add_argument("--target-fps", type=float, default=16.0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--no-require-projected-origin",
        action="store_true",
        help="use mesh area only instead of requiring an in-frame link origin",
    )
    parser.add_argument("--resume", action="store_true")
    parser.set_defaults(func=process_segments)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.min_part_pixels <= 0:
        raise ValueError(f"--min-part-pixels must be positive, got {args.min_part_pixels}")
    if args.workers <= 0:
        raise ValueError(f"--workers must be positive, got {args.workers}")
    args.func(args)


if __name__ == "__main__":
    main()
