"""Precompute SAM3/SAM3.1 robot-arm masks for raw H2R robot-camera videos.

The output is intentionally simple and directly consumable by
``src.pipeline.h2r_sam3_blur_pair``:

  training_data/h2r_sam3_mask/<task>/episode_<id>.npz

Each npz contains a full-resolution uint8 ``masks`` array with shape
``[source_frame_count, H, W]``. Only frames used by the selected 1s clips are
filled; other frames remain zero. The blur-pair converter reads the same source
frame indices, so no implicit interpolation or fallback is needed.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MethodType
from typing import Any

import cv2
import numpy as np
from PIL import Image

from src.core.config import MAIN_ROOT, TRAINING_DATA_ROOT


DEFAULT_DATA_ROOT = Path(MAIN_ROOT) / "data" / "h2r" / "v1"
DEFAULT_OUTPUT_ROOT = Path(TRAINING_DATA_ROOT) / "h2r_sam3_mask"
DEFAULT_TMP_DIR = Path(MAIN_ROOT) / "tmp" / "h2r_sam3_precompute_frames"
TARGET_FPS = 16


@dataclass(frozen=True)
class EpisodeInfo:
    task: str
    episode: str
    video_path: Path
    frame_count: int
    fps: float
    height: int
    width: int


@dataclass(frozen=True)
class ClipInfo:
    clip_index: int
    start_frame: int
    source_indices: tuple[int, ...]


def safe_name(value: str) -> str:
    name = re.sub(r"[^a-zA-Z0-9._-]+", "_", value.strip()).strip("_")
    if not name:
        raise ValueError(f"value cannot be converted to a safe name: {value!r}")
    return name


def parse_csv(value: str) -> list[str]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise ValueError(f"expected comma-separated value, got {value!r}")
    return items


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def patch_sam31_start_session(predictor) -> None:
    """Patch SAM3.1 start_session to pass async_loading_frames explicitly.

    The local SAM3.1 checkout used for this project needs this patch for the
    multiplex predictor path when starting sessions from a JPEG folder.
    """

    def start_session(
        self,
        resource_path,
        session_id=None,
        offload_video_to_cpu=False,
        offload_state_to_cpu=False,
    ):
        inference_state = self.model.init_state(
            resource_path=resource_path,
            offload_video_to_cpu=offload_video_to_cpu,
            async_loading_frames=getattr(self, "async_loading_frames", False),
        )
        if not session_id:
            session_id = str(uuid.uuid4())
        self._all_inference_states[session_id] = {
            "state": inference_state,
            "session_id": session_id,
            "start_time": time.time(),
            "last_use_time": time.time(),
        }
        return {"session_id": session_id}

    predictor.start_session = MethodType(start_session, predictor)


def discover_episodes(args: argparse.Namespace) -> list[EpisodeInfo]:
    video_root = args.data_root / "video"
    if not video_root.is_dir():
        raise FileNotFoundError(f"H2R video root not found: {video_root}")
    tasks = None if args.tasks == "all" else parse_csv(args.tasks)
    search_roots = [video_root] if tasks is None else [video_root / task for task in tasks]
    for search_root in search_roots:
        if not search_root.is_dir():
            raise FileNotFoundError(f"H2R task video dir not found: {search_root}")

    video_paths: list[Path] = []
    for search_root in search_roots:
        video_paths.extend(sorted(search_root.glob(f"**/{args.camera_key}.mp4")))

    episodes: list[EpisodeInfo] = []
    per_task_counts: dict[str, int] = {}
    for video_path in video_paths:
        episode_dir = video_path.parent
        if not episode_dir.name.startswith("episode_"):
            raise ValueError(f"H2R camera video is not under episode_* dir: {video_path}")
        task = episode_dir.parent.relative_to(video_root).as_posix()
        if args.max_episodes_per_task > 0:
            count = per_task_counts.get(task, 0)
            if count >= args.max_episodes_per_task:
                continue
            per_task_counts[task] = count + 1

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"cannot open H2R video: {video_path}")
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        if frame_count <= 0:
            raise ValueError(f"H2R video has no frames: {video_path}")
        if fps <= 0:
            raise ValueError(f"H2R video has invalid fps={fps}: {video_path}")
        if height <= 0 or width <= 0:
            raise ValueError(f"H2R video has invalid size {width}x{height}: {video_path}")
        episodes.append(
            EpisodeInfo(
                task=task,
                episode=episode_dir.name,
                video_path=video_path,
                frame_count=frame_count,
                fps=fps,
                height=height,
                width=width,
            )
        )
    if not episodes:
        raise ValueError(f"no H2R videos found under {video_root}")
    return episodes


def clip_indices(start_frame: int, fps: float, clip_seconds: float) -> tuple[int, ...]:
    return tuple(
        int(round(start_frame + i * fps / TARGET_FPS))
        for i in range(int(round(clip_seconds * TARGET_FPS)) + 1)
    )


def build_episode_clips(args: argparse.Namespace, episode: EpisodeInfo) -> list[ClipInfo]:
    explicit_starts = getattr(args, "clip_starts_by_episode", None)
    stride_frames = int(round(args.clip_stride * episode.fps))
    if stride_frames <= 0:
        raise ValueError(f"--clip-stride is too small for fps={episode.fps}")
    clip_seconds = float(args.clip_seconds)
    last_start = episode.frame_count - int(round(clip_seconds * episode.fps)) - 1
    if last_start < 0:
        return []

    if explicit_starts is not None:
        task_starts = explicit_starts.get(episode.task, {})
        starts = task_starts.get(episode.episode)
        if starts is None and episode.episode.startswith("episode_"):
            starts = task_starts.get(episode.episode.removeprefix("episode_"))
        if starts is None:
            return []

        clips = []
        for clip_index, start_frame in enumerate(starts):
            if start_frame < 0 or start_frame > last_start:
                raise ValueError(
                    f"explicit clip start {start_frame} is outside [0, {last_start}] "
                    f"for {episode.task}/{episode.episode}"
                )
            indices = clip_indices(start_frame, episode.fps, clip_seconds)
            if max(indices) >= episode.frame_count:
                raise ValueError(
                    f"explicit clip start {start_frame} produces source index "
                    f"{max(indices)} >= frame_count {episode.frame_count} "
                    f"for {episode.task}/{episode.episode}"
                )
            clips.append(
                ClipInfo(
                    clip_index=clip_index,
                    start_frame=start_frame,
                    source_indices=indices,
                )
            )
        return clips

    clips = []
    for clip_index, start_frame in enumerate(range(0, last_start + 1, stride_frames)):
        indices = clip_indices(start_frame, episode.fps, clip_seconds)
        if max(indices) < episode.frame_count:
            clips.append(
                ClipInfo(
                    clip_index=clip_index,
                    start_frame=start_frame,
                    source_indices=indices,
                )
            )
    if args.max_clips_per_episode > 0:
        clips = clips[: args.max_clips_per_episode]
    return clips


def load_clip_starts_file(path: Path) -> dict[str, dict[str, list[int]]]:
    if not path.is_file():
        raise FileNotFoundError(f"clip starts file not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"clip starts file must be a JSON object: {path}")
    out: dict[str, dict[str, list[int]]] = {}
    for task, task_payload in payload.items():
        if not isinstance(task, str) or not task:
            raise ValueError(f"clip starts task key must be a non-empty string: {path}")
        if not isinstance(task_payload, dict):
            raise ValueError(f"clip starts for task {task!r} must be an object: {path}")
        episode_map: dict[str, list[int]] = {}
        for episode, starts_payload in task_payload.items():
            if not isinstance(episode, str) or not episode:
                raise ValueError(
                    f"clip starts episode key for task {task!r} must be a non-empty string: {path}"
                )
            if not isinstance(starts_payload, list) or not starts_payload:
                raise ValueError(
                    f"clip starts for {task}/{episode} must be a non-empty list: {path}"
                )
            starts: list[int] = []
            for value in starts_payload:
                if not isinstance(value, int):
                    raise ValueError(
                        f"clip start for {task}/{episode} must be an integer, got {value!r}"
                    )
                starts.append(value)
            if len(set(starts)) != len(starts):
                raise ValueError(f"duplicate clip starts for {task}/{episode}: {starts}")
            episode_map[episode] = starts
        out[task] = episode_map
    return out


def extract_clip_frames(
    video_path: Path,
    source_indices: tuple[int, ...],
    out_dir: Path,
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.jpg"):
        old.unlink()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    frame_paths = []
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


def outputs_to_mask(outputs: dict | None, height: int, width: int) -> np.ndarray:
    if outputs is None:
        return np.zeros((height, width), dtype=np.uint8)
    masks = outputs.get("out_binary_masks")
    if masks is None:
        raise ValueError("SAM3 output missing out_binary_masks")
    if hasattr(masks, "detach"):
        masks = masks.detach().cpu().numpy()
    masks = np.asarray(masks)
    if masks.size == 0:
        return np.zeros((height, width), dtype=np.uint8)
    if masks.ndim == 2:
        merged = masks.astype(bool)
    elif masks.ndim == 3:
        merged = np.any(masks.astype(bool), axis=0)
    else:
        raise ValueError(f"unsupported SAM3 mask shape: {masks.shape}")
    if merged.shape != (height, width):
        raise ValueError(f"SAM3 mask shape mismatch: {merged.shape} != {(height, width)}")
    return merged.astype(np.uint8)


def run_prompt_once(
    predictor,
    frame_dir: Path,
    prompt: str,
    frame_count: int,
    output_prob_thresh: float,
    prompt_frame_index: int,
) -> tuple[np.ndarray, list[dict]]:
    if prompt_frame_index < 0 or prompt_frame_index >= frame_count - 1:
        raise ValueError(
            f"prompt_frame_index must be in [0, {frame_count - 2}] for stable SAM3.1 "
            f"forward/backward propagation, got {prompt_frame_index}"
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
        first = predictor.handle_request(
            {
                "type": "add_prompt",
                "session_id": session_id,
                "frame_index": prompt_frame_index,
                "text": prompt,
                "output_prob_thresh": output_prob_thresh,
            }
        )
        outputs_by_frame = {first["frame_index"]: first["outputs"]}
        forward_count = frame_count - prompt_frame_index
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
            outputs_by_frame[item["frame_index"]] = item["outputs"]

        if prompt_frame_index > 0:
            backward_count = prompt_frame_index + 1
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
                outputs_by_frame[item["frame_index"]] = item["outputs"]

        first_frame = np.asarray(Image.open(frame_dir / "00000.jpg").convert("RGB"))
        height, width = first_frame.shape[:2]
        masks = []
        summary = []
        for frame_index in range(frame_count):
            mask = outputs_to_mask(outputs_by_frame.get(frame_index), height, width)
            masks.append(mask)
            outputs = outputs_by_frame.get(frame_index) or {}
            obj_ids = outputs.get("out_obj_ids", [])
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
    finally:
        predictor.handle_request({"type": "close_session", "session_id": session_id})


def run_sam3_clip(
    predictor,
    frame_dir: Path,
    prompt: str,
    backup_prompt: str,
    frame_count: int,
    output_prob_thresh: float,
    prompt_frame_index: int,
) -> tuple[np.ndarray, str, list[dict]]:
    masks, summary = run_prompt_once(
        predictor, frame_dir, prompt, frame_count, output_prob_thresh, prompt_frame_index
    )
    if int(masks.astype(bool).sum()) > 0 or not backup_prompt:
        return masks, prompt, summary
    backup_masks, backup_summary = run_prompt_once(
        predictor,
        frame_dir,
        backup_prompt,
        frame_count,
        output_prob_thresh,
        prompt_frame_index,
    )
    return backup_masks, backup_prompt, backup_summary


def resolve_prompt_frame_index(frame_count: int, position: str) -> int:
    key = position.strip().lower()
    if key == "first":
        index = 0
    elif key == "middle":
        index = frame_count // 2
    else:
        index = int(key)
    if index < 0 or index >= frame_count - 1:
        raise ValueError(
            f"--prompt-frame-position={position!r} resolves to {index}, but stable "
            f"SAM3.1 propagation requires a prompt frame in [0, {frame_count - 2}]"
        )
    return index


def output_path_for_episode(output_root: Path, episode: EpisodeInfo) -> Path:
    return output_root / episode.task / f"{episode.episode}.npz"


def existing_mask_covers_clips(
    out_path: Path,
    episode: EpisodeInfo,
    clips: list[ClipInfo],
) -> bool:
    with np.load(out_path) as data:
        if "masks" not in data:
            raise ValueError(f"Existing SAM3 mask npz missing 'masks': {out_path}")
        masks = data["masks"]
        if "covered_frames" not in data:
            return False
        covered_frames = set(int(index) for index in data["covered_frames"].tolist())
    if masks.shape != (episode.frame_count, episode.height, episode.width):
        return False
    required_frames = {
        int(source_index)
        for clip in clips
        for source_index in clip.source_indices
    }
    return required_frames <= covered_frames


def process_episode(args: argparse.Namespace, predictor, episode: EpisodeInfo) -> dict:
    clips = build_episode_clips(args, episode)
    if not clips:
        raise ValueError(f"no clips selected for episode: {episode.video_path}")

    out_path = output_path_for_episode(args.output_root, episode)
    if args.resume and out_path.is_file() and existing_mask_covers_clips(
        out_path, episode, clips,
    ):
        return {
            "task": episode.task,
            "episode": episode.episode,
            "video_path": str(episode.video_path),
            "output_path": str(out_path),
            "skipped": True,
            "num_clips": len(clips),
        }
    if args.resume and out_path.is_file():
        print(
            f"  existing mask does not cover requested clips; recomputing {out_path}",
            flush=True,
        )

    episode_tmp = args.tmp_dir / safe_name(episode.task) / safe_name(episode.episode)
    masks_full = np.zeros(
        (episode.frame_count, episode.height, episode.width),
        dtype=np.uint8,
    )
    covered_frames: set[int] = set()
    clip_rows = []

    for clip in clips:
        frame_dir = episode_tmp / f"clip_{clip.clip_index:04d}"
        frame_paths = extract_clip_frames(
            episode.video_path, clip.source_indices, frame_dir
        )
        prompt_frame_index = resolve_prompt_frame_index(
            len(frame_paths), args.prompt_frame_position
        )
        clip_masks, used_prompt, frame_summary = run_sam3_clip(
            predictor,
            frame_dir,
            args.prompt,
            args.backup_prompt,
            len(frame_paths),
            args.output_prob_thresh,
            prompt_frame_index,
        )
        for local_index, source_index in enumerate(clip.source_indices):
            masks_full[source_index] = np.maximum(
                masks_full[source_index], clip_masks[local_index]
            )
            covered_frames.add(source_index)
        clip_rows.append(
            {
                "clip_index": clip.clip_index,
                "start_frame": clip.start_frame,
                "source_indices": list(clip.source_indices),
                "prompt_frame_index": int(prompt_frame_index),
                "prompt": used_prompt,
                "nonempty_frames": int(np.count_nonzero(clip_masks.reshape(len(clip_masks), -1).sum(axis=1))),
                "area_sum": int(clip_masks.astype(bool).sum()),
                "frames": frame_summary,
            }
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    covered = np.array(sorted(covered_frames), dtype=np.int32)
    np.savez_compressed(
        out_path,
        masks=masks_full,
        covered_frames=covered,
        clip_starts=np.array([clip.start_frame for clip in clips], dtype=np.int32),
        prompt=np.array(args.prompt),
        backup_prompt=np.array(args.backup_prompt),
        prompt_frame_position=np.array(args.prompt_frame_position),
        task=np.array(episode.task),
        episode=np.array(episode.episode),
        video_path=np.array(str(episode.video_path)),
        source_fps=np.array(float(episode.fps), dtype=np.float32),
        target_fps=np.array(TARGET_FPS, dtype=np.int32),
        clip_seconds=np.array(float(args.clip_seconds), dtype=np.float32),
        clip_stride=np.array(float(args.clip_stride), dtype=np.float32),
    )
    summary_path = out_path.with_suffix(".json")
    summary = {
        "task": episode.task,
        "episode": episode.episode,
        "video_path": str(episode.video_path),
        "output_path": str(out_path),
        "summary_path": str(summary_path),
        "frame_count": episode.frame_count,
        "height": episode.height,
        "width": episode.width,
        "source_fps": episode.fps,
        "target_fps": TARGET_FPS,
        "num_clips": len(clips),
        "covered_frame_count": int(len(covered_frames)),
        "mask_nonzero_frame_count": int(np.count_nonzero(masks_full.reshape(len(masks_full), -1).sum(axis=1))),
        "mask_area_sum": int(masks_full.astype(bool).sum()),
        "clips": clip_rows,
    }
    write_json(summary_path, summary)
    if not args.keep_frames:
        shutil.rmtree(episode_tmp, ignore_errors=True)
    return summary


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Precompute SAM3 robot-arm masks for H2R robot-camera videos."
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--tmp-dir", type=Path, default=DEFAULT_TMP_DIR)
    parser.add_argument("--tasks", default="grab_cup_v1,grab_cube2_v1,push_box_random_v1")
    parser.add_argument("--camera-key", default="robot_camera")
    parser.add_argument("--prompt", default="robot arm")
    parser.add_argument("--backup-prompt", default="robotic arm")
    parser.add_argument(
        "--prompt-frame-position",
        default="first",
        help="local SAM3 prompt frame inside each 17-frame clip: first, middle, or an integer before the last frame",
    )
    parser.add_argument("--sam3-version", default="sam3.1", choices=["sam3", "sam3.1"])
    parser.add_argument("--max-num-objects", type=int, default=1)
    parser.add_argument("--multiplex-count", type=int, default=16)
    parser.add_argument("--output-prob-thresh", type=float, default=0.5)
    parser.add_argument("--clip-seconds", type=float, default=1.0)
    parser.add_argument("--clip-stride", type=float, default=1.0)
    parser.add_argument(
        "--clip-starts-file",
        type=Path,
        default=None,
        help=(
            "optional JSON mapping task -> episode -> explicit 1s clip start frames; "
            "used for tail-aligned Seedance batches that need mask coverage outside "
            "the regular clip stride"
        ),
    )
    parser.add_argument("--max-episodes-per-task", type=int, default=0)
    parser.add_argument("--max-clips-per-episode", type=int, default=0)
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--keep-frames", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.clip_seconds != 1.0:
        raise ValueError("only 1s clips are supported by the maintained stage2 preset")
    if args.clip_stride <= 0:
        raise ValueError(f"--clip-stride must be positive, got {args.clip_stride}")
    if args.max_num_objects <= 0:
        raise ValueError(f"--max-num-objects must be positive, got {args.max_num_objects}")
    if args.multiplex_count <= 0:
        raise ValueError(f"--multiplex-count must be positive, got {args.multiplex_count}")
    if args.clip_starts_file is not None:
        args.clip_starts_file = (
            args.clip_starts_file
            if args.clip_starts_file.is_absolute()
            else Path(MAIN_ROOT) / args.clip_starts_file
        )
        args.clip_starts_by_episode = load_clip_starts_file(args.clip_starts_file)
    else:
        args.clip_starts_by_episode = None

    episodes = discover_episodes(args)
    if args.clip_starts_by_episode is not None:
        episodes = [episode for episode in episodes if build_episode_clips(args, episode)]
    if args.max_episodes > 0:
        episodes = episodes[: args.max_episodes]
    planned = []
    for episode in episodes:
        clips = build_episode_clips(args, episode)
        planned.append(
            {
                "task": episode.task,
                "episode": episode.episode,
                "video_path": str(episode.video_path),
                "output_path": str(output_path_for_episode(args.output_root, episode)),
                "frame_count": episode.frame_count,
                "fps": episode.fps,
                "size": [episode.height, episode.width],
                "num_clips": len(clips),
            }
        )
    if not planned:
        raise ValueError("no H2R episodes selected")

    print("H2R SAM3 mask precompute")
    print(f"  episodes: {len(planned)}")
    print(f"  output:   {args.output_root}")
    print(f"  prompt:   {args.prompt}")
    print(f"  backup:   {args.backup_prompt or '<none>'}")
    print(f"  prompt frame: {args.prompt_frame_position}")
    if args.clip_starts_file is not None:
        print(f"  explicit starts: {args.clip_starts_file}")
    if args.dry_run:
        by_task: dict[str, int] = {}
        for row in planned:
            by_task[row["task"]] = by_task.get(row["task"], 0) + row["num_clips"]
        print("  planned clips:")
        for task, count in sorted(by_task.items()):
            print(f"    {task}: {count}")
        return

    predictor = load_predictor(args)
    summaries = []
    for index, episode in enumerate(episodes, start=1):
        print(f"[{index}/{len(episodes)}] {episode.task}/{episode.episode}", flush=True)
        summaries.append(process_episode(args, predictor, episode))

    summary_path = args.output_root / "index.json"
    write_json(
        summary_path,
        {
            "data_root": str(args.data_root),
            "output_root": str(args.output_root),
            "sam3_version": args.sam3_version,
            "prompt": args.prompt,
            "backup_prompt": args.backup_prompt,
            "prompt_frame_position": args.prompt_frame_position,
            "clip_starts_file": str(args.clip_starts_file) if args.clip_starts_file else "",
            "target_fps": TARGET_FPS,
            "clip_seconds": args.clip_seconds,
            "clip_stride": args.clip_stride,
            "episodes": summaries,
        },
    )
    print(f"Done: {summary_path}")


if __name__ == "__main__":
    main()
