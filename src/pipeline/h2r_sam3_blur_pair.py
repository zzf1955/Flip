"""Build blur_r2r pair data from human2robot robot-camera videos and SAM3 masks.

This converts the raw human2robot video layout. The historical local data path
is still ``data/h2r/v1``:

  data/h2r/v1/video/<task>/episode_<id>/robot_camera.mp4

plus precomputed SAM3 masks:

  training_data/h2r_sam3_mask/<task>/episode_<id>.npz

into the maintained Mitty pair layout:

  training_data/pair/blur_r2r/1s/<task>/{video,control_video}

The target video is the clear robot clip. The control video is the same clip
with the SAM3 robot-arm mask region blurred, matching the stage-2 appearance
training objective. SAM3 inference is intentionally a separate precompute step;
this script only consumes mask artifacts and fails if they are missing.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from src.core.config import MAIN_ROOT, TRAINING_DATA_ROOT
from src.pipeline.make_pair import (
    PROMPT,
    TARGET_FPS,
    blur_frame_in_mask,
    soften_mask,
    write_video_bgr,
)


DEFAULT_DATA_ROOT = Path(MAIN_ROOT) / "data" / "h2r" / "v1"
DEFAULT_MASK_ROOT = Path(TRAINING_DATA_ROOT) / "h2r_sam3_mask"
DEFAULT_PAIR_ROOT = Path(TRAINING_DATA_ROOT) / "pair"
DEFAULT_DURATION = "1s"
DEFAULT_RESIZE = "224x416"


@dataclass(frozen=True)
class EpisodeVideo:
    task: str
    episode: str
    video_path: Path
    mask_path: Path
    frame_count: int
    fps: float


@dataclass(frozen=True)
class ClipSpec:
    task: str
    episode: str
    clip_index: int
    source_start_frame: int
    source_indices: tuple[int, ...]
    video_path: Path
    mask_path: Path


def parse_csv(value: str) -> list[str]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise ValueError(f"expected comma-separated value, got {value!r}")
    return items


def parse_resize(value: str) -> tuple[int, int]:
    parts = value.lower().replace(" ", "").split("x")
    if len(parts) != 2:
        raise ValueError(f"resize must be formatted as HxW, got {value!r}")
    height, width = int(parts[0]), int(parts[1])
    if height <= 0 or width <= 0:
        raise ValueError(f"resize dimensions must be positive, got {value!r}")
    if height % 32 != 0 or width % 32 != 0:
        raise ValueError(
            "resize dimensions must be divisible by 32 so Wan VAE latents "
            f"have even spatial grids, got {value!r}"
        )
    return height, width


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


def write_metadata(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["video", "prompt", "control_video"]
        )
        writer.writeheader()
        writer.writerows(rows)


def mask_candidates(mask_root: Path, task: str, episode: str) -> list[Path]:
    return [
        mask_root / task / f"{episode}.npz",
        mask_root / task / episode / "robot_camera.npz",
        mask_root / task / episode / "robot_camera_mask.npz",
        mask_root / task / episode / "robot_camera_mask.mp4",
    ]


def resolve_mask_path(mask_root: Path, task: str, episode: str) -> Path:
    candidates = mask_candidates(mask_root, task, episode)
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        "SAM3 mask not found for H2R episode. Expected one of: "
        + ", ".join(str(path) for path in candidates)
    )


def discover_videos(args: argparse.Namespace) -> list[EpisodeVideo]:
    video_root = args.data_root / "video"
    if not video_root.is_dir():
        raise FileNotFoundError(f"H2R video root not found: {video_root}")
    tasks = None if args.tasks == "all" else parse_csv(args.tasks)
    episodes: list[EpisodeVideo] = []
    search_roots = [video_root] if tasks is None else [video_root / task for task in tasks]
    for search_root in search_roots:
        if not search_root.is_dir():
            raise FileNotFoundError(f"H2R task video dir not found: {search_root}")
    video_paths = []
    for search_root in search_roots:
        video_paths.extend(sorted(search_root.glob(f"**/{args.camera_key}.mp4")))
    for video_path in video_paths:
        episode_dir = video_path.parent
        if not episode_dir.name.startswith("episode_"):
            raise ValueError(f"H2R camera video is not under episode_* dir: {video_path}")
        task = episode_dir.parent.relative_to(video_root).as_posix()
        task_dir = video_root / task
        if not task_dir.is_dir():
            raise FileNotFoundError(f"H2R task video dir not found: {task_dir}")
        if args.max_episodes_per_task > 0:
            existing = sum(1 for episode in episodes if episode.task == task)
            if existing >= args.max_episodes_per_task:
                continue
        mask_path = resolve_mask_path(args.mask_root, task, episode_dir.name)
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"cannot open H2R video: {video_path}")
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        cap.release()
        if frame_count <= 0:
            raise ValueError(f"H2R video has no frames: {video_path}")
        if fps <= 0:
            raise ValueError(f"H2R video has invalid fps={fps}: {video_path}")
        episodes.append(
            EpisodeVideo(
                task=task,
                episode=episode_dir.name,
                video_path=video_path,
                mask_path=mask_path,
                frame_count=frame_count,
                fps=fps,
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


def build_clips(args: argparse.Namespace, episodes: list[EpisodeVideo]) -> list[ClipSpec]:
    clips: list[ClipSpec] = []
    for episode in episodes:
        clip_seconds = float(args.clip_seconds)
        stride_frames = int(round(args.clip_stride * episode.fps))
        if stride_frames <= 0:
            raise ValueError(f"--clip-stride is too small for fps={episode.fps}")
        last_start = episode.frame_count - int(round(clip_seconds * episode.fps)) - 1
        if last_start < 0:
            continue
        episode_clips = []
        for clip_index, start_frame in enumerate(range(0, last_start + 1, stride_frames)):
            indices = clip_indices(start_frame, episode.fps, clip_seconds)
            if max(indices) >= episode.frame_count:
                continue
            episode_clips.append(
                ClipSpec(
                    task=episode.task,
                    episode=episode.episode,
                    clip_index=clip_index,
                    source_start_frame=start_frame,
                    source_indices=indices,
                    video_path=episode.video_path,
                    mask_path=episode.mask_path,
                )
            )
        if args.max_clips_per_episode > 0:
            episode_clips = episode_clips[: args.max_clips_per_episode]
        clips.extend(episode_clips)
    if args.max_clips > 0:
        clips = clips[: args.max_clips]
    if not clips:
        raise ValueError("no clips selected; check video lengths and clip settings")
    return clips


def read_video_frames(video_path: Path, indices: tuple[int, ...],
                      resize_hw: tuple[int, int]) -> list[np.ndarray]:
    index_set = set(indices)
    frames_by_index: dict[int, np.ndarray] = {}
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    frame_index = 0
    while frame_index <= max(index_set):
        ok, frame = cap.read()
        if not ok:
            break
        if frame_index in index_set:
            height, width = resize_hw
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            frames_by_index[frame_index] = frame
        frame_index += 1
    cap.release()
    missing = sorted(index_set - set(frames_by_index))
    if missing:
        raise ValueError(f"video ended before required frames {missing[:5]}: {video_path}")
    return [frames_by_index[index] for index in indices]


def read_mask_video(mask_path: Path, source_indices: tuple[int, ...],
                    mask_stride: int, resize_hw: tuple[int, int]) -> list[np.ndarray]:
    mask_indices = tuple(int(round(index / mask_stride)) for index in source_indices)
    index_set = set(mask_indices)
    masks_by_index: dict[int, np.ndarray] = {}
    cap = cv2.VideoCapture(str(mask_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open SAM3 mask video: {mask_path}")
    frame_index = 0
    while frame_index <= max(index_set):
        ok, frame = cap.read()
        if not ok:
            break
        if frame_index in index_set:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            height, width = resize_hw
            gray = cv2.resize(gray, (width, height), interpolation=cv2.INTER_NEAREST)
            masks_by_index[frame_index] = (gray > 127).astype(np.uint8) * 255
        frame_index += 1
    cap.release()
    missing = sorted(index_set - set(masks_by_index))
    if missing:
        raise ValueError(
            f"SAM3 mask video ended before required frames {missing[:5]}: {mask_path}"
        )
    return [masks_by_index[index] for index in mask_indices]


def read_mask_npz(mask_path: Path, source_indices: tuple[int, ...],
                  mask_stride: int, resize_hw: tuple[int, int]) -> list[np.ndarray]:
    with np.load(mask_path) as data:
        if "masks" not in data:
            raise ValueError(f"SAM3 mask npz missing 'masks': {mask_path}")
        masks = data["masks"]
    if masks.ndim == 4:
        masks = masks.max(axis=1)
    if masks.ndim != 3:
        raise ValueError(f"SAM3 masks must be [T,H,W] or [T,N,H,W], got {masks.shape}")
    mask_indices = [int(round(index / mask_stride)) for index in source_indices]
    if max(mask_indices) >= len(masks):
        raise ValueError(
            f"SAM3 mask count too short for {mask_path}: "
            f"need index {max(mask_indices)}, count={len(masks)}"
        )
    out = []
    height, width = resize_hw
    for index in mask_indices:
        mask = masks[index]
        mask = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
        out.append((mask > 0).astype(np.uint8) * 255)
    return out


def read_masks(mask_path: Path, source_indices: tuple[int, ...],
               mask_stride: int, resize_hw: tuple[int, int]) -> list[np.ndarray]:
    if mask_path.suffix == ".npz":
        return read_mask_npz(mask_path, source_indices, mask_stride, resize_hw)
    if mask_path.suffix == ".mp4":
        return read_mask_video(mask_path, source_indices, mask_stride, resize_hw)
    raise ValueError(f"unsupported SAM3 mask artifact type: {mask_path}")


def blur_frames(frames: list[np.ndarray], masks: list[np.ndarray],
                blur_ksize: int, blur_pixel_expand: int) -> list[np.ndarray]:
    if len(frames) != len(masks):
        raise ValueError(f"frame/mask count mismatch: {len(frames)} != {len(masks)}")
    blurred = []
    for frame, mask in zip(frames, masks):
        if frame.shape[:2] != mask.shape[:2]:
            raise ValueError(
                f"frame/mask shape mismatch: {frame.shape[:2]} != {mask.shape[:2]}"
            )
        soft = soften_mask(mask, pixel_expand=blur_pixel_expand)
        blurred.append(blur_frame_in_mask(frame, soft, blur_ksize))
    return blurred


def process_clip(
    clip: ClipSpec,
    pair_name: str,
    task_dir: Path,
    resize_hw: tuple[int, int],
    args: argparse.Namespace,
) -> dict:
    target_path = task_dir / "video" / f"{pair_name}.mp4"
    control_path = task_dir / "control_video" / f"{pair_name}.mp4"
    if args.resume and target_path.is_file() and control_path.is_file():
        pass
    else:
        frames = read_video_frames(clip.video_path, clip.source_indices, resize_hw)
        masks = read_masks(
            clip.mask_path, clip.source_indices, args.mask_stride, resize_hw
        )
        control_frames = blur_frames(
            frames, masks, args.blur_ksize, args.blur_pixel_expand
        )
        write_video_bgr(frames, str(target_path), fps=TARGET_FPS)
        write_video_bgr(control_frames, str(control_path), fps=TARGET_FPS)
    return {
        "data_type": "blur_r2r",
        "duration": args.duration,
        "robot_task": clip.task,
        "task": clip.task,
        "episode": clip.episode,
        "clip_idx": clip.clip_index,
        "clip_start_frame": clip.source_start_frame,
        "source_frame_indices": list(clip.source_indices),
        "source_id": f"{clip.task}/{clip.episode}/clip{clip.clip_index:04d}",
        "source_segment_id": f"{clip.task}/{clip.episode}",
        "robot_src": str(clip.video_path),
        "sam3_mask_path": str(clip.mask_path),
        "sam3_mask_stride": args.mask_stride,
        "sam3_prompt": args.sam3_prompt,
        "sam3_model": args.sam3_model,
        "control_degrade": "sam3_blur",
        "blur_ksize": args.blur_ksize,
        "blur_pixel_expand": args.blur_pixel_expand,
        "resize": args.resize,
        "fps": TARGET_FPS,
        "video": f"video/{pair_name}.mp4",
        "control_video": f"control_video/{pair_name}.mp4",
        "input_role": "robot",
        "target_role": "robot",
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Convert H2R robot videos + SAM3 masks to blur_r2r pair data."
    )
    ap.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    ap.add_argument("--mask-root", type=Path, default=DEFAULT_MASK_ROOT)
    ap.add_argument("--pair-root", type=Path, default=DEFAULT_PAIR_ROOT)
    ap.add_argument("--tasks", default="grab_cup_v1,grab_cube2_v1,push_box_random_v1")
    ap.add_argument("--duration", default=DEFAULT_DURATION, choices=["1s"])
    ap.add_argument("--camera-key", default="robot_camera")
    ap.add_argument("--resize", default=DEFAULT_RESIZE, help="output HxW, divisible by 32")
    ap.add_argument("--clip-seconds", type=float, default=1.0)
    ap.add_argument("--clip-stride", type=float, default=1.0)
    ap.add_argument("--max-episodes-per-task", type=int, default=0)
    ap.add_argument("--max-clips-per-episode", type=int, default=0)
    ap.add_argument("--max-clips", type=int, default=0)
    ap.add_argument("--mask-stride", type=int, default=1)
    ap.add_argument("--blur-ksize", type=int, default=51)
    ap.add_argument("--blur-pixel-expand", type=int, default=16)
    ap.add_argument("--sam3-prompt", default="robot arm")
    ap.add_argument("--sam3-model", default="sam3.1")
    ap.add_argument("--clean", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.mask_stride <= 0:
        raise ValueError(f"--mask-stride must be positive, got {args.mask_stride}")
    if args.clip_seconds != 1.0:
        raise ValueError("only 1s clips are supported by the maintained train preset")
    if args.clip_stride <= 0:
        raise ValueError(f"--clip-stride must be positive, got {args.clip_stride}")
    if args.blur_ksize <= 0 or args.blur_ksize % 2 == 0:
        raise ValueError("--blur-ksize must be a positive odd integer")
    if args.blur_pixel_expand < 0:
        raise ValueError("--blur-pixel-expand must be non-negative")

    resize_hw = parse_resize(args.resize)
    episodes = discover_videos(args)
    clips = build_clips(args, episodes)
    print("H2R SAM3 blur_r2r pair build")
    print(f"  episodes: {len(episodes)}")
    print(f"  clips:    {len(clips)}")
    print(f"  output:   {args.pair_root / 'blur_r2r' / args.duration}")
    if args.dry_run:
        by_task: dict[str, int] = {}
        for clip in clips:
            by_task[clip.task] = by_task.get(clip.task, 0) + 1
        print("  per task:")
        for task, count in sorted(by_task.items()):
            print(f"    {task}: {count}")
        return

    sec_dir = args.pair_root / "blur_r2r" / args.duration
    if args.clean and sec_dir.exists():
        for task in sorted({clip.task for clip in clips}):
            task_dir = sec_dir / task
            if task_dir.exists():
                import shutil
                shutil.rmtree(task_dir)
    sec_dir.mkdir(parents=True, exist_ok=True)

    all_manifest_rows: list[dict] = []
    by_task: dict[str, list[ClipSpec]] = {}
    for clip in clips:
        by_task.setdefault(clip.task, []).append(clip)

    for task, task_clips in sorted(by_task.items()):
        task_dir = sec_dir / task
        (task_dir / "video").mkdir(parents=True, exist_ok=True)
        (task_dir / "control_video").mkdir(parents=True, exist_ok=True)
        metadata_rows = []
        manifest_rows = []
        for pair_index, clip in enumerate(task_clips):
            pair_name = f"pair_{pair_index:04d}"
            record = process_clip(clip, pair_name, task_dir, resize_hw, args)
            metadata_rows.append({
                "video": record["video"],
                "prompt": PROMPT,
                "control_video": record["control_video"],
            })
            manifest_rows.append(record)
        write_metadata(task_dir / "metadata.csv", metadata_rows)
        write_jsonl(task_dir / "manifest.jsonl", manifest_rows)
        all_manifest_rows.extend(manifest_rows)
        print(f"  {task}: {len(manifest_rows)} pairs")

    write_jsonl(sec_dir / "index.jsonl", sorted(all_manifest_rows, key=lambda row: row["source_id"]))
    print(f"Done: {sec_dir / 'index.jsonl'} ({len(all_manifest_rows)} entries)")


if __name__ == "__main__":
    main()
