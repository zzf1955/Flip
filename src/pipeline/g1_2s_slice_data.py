"""Build G1 2s/30fps slice artifacts for staged Mitty training.

This script is intentionally separate from the older 1s/16fps pair builders.
It writes explicit slice outputs plus pair-layout datasets:

  training_data/slice/g1_<format_label>/{original,seedance_direct,sam2_blur}/...
  training_data/pair/{identity_r2r,blur_r2r,h2r}/<format_label>/<task>/...

The default ``2s61f30`` format keeps Wan's 4k+1 frame convention while staying
close to 2 seconds on the original 30Hz G1 timeline.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from src.core.config import ALL_TASKS, MAIN_ROOT, TRAINING_TASKS
from src.core.data import close_video, open_video_writer, write_frame


PROMPT = "A first-person view robot arm performing household tasks flip_v2v"
DEFAULT_FORMAT_LABEL = "2s61f30"
DEFAULT_FPS = 30
DEFAULT_NUM_FRAMES = 61
DEFAULT_CLIP_SECONDS = 2.0
DEFAULT_CLIP_STRIDE = 2.0
DEFAULT_BLUR_KSIZE = 51
DEFAULT_BLUR_PIXEL_EXPAND = 16

TRAINING_ROOT = Path(MAIN_ROOT) / "training_data"
DEFAULT_SEGMENT_ROOT = TRAINING_ROOT / "segment"
DEFAULT_SEEDANCE_4S_ROOT = TRAINING_ROOT / "seedance_direct" / "4s"
DEFAULT_SAM2_MASK_ROOT = TRAINING_ROOT / "sam2_mask"
DEFAULT_SLICE_ROOT = TRAINING_ROOT / "slice"
DEFAULT_PAIR_ROOT = TRAINING_ROOT / "pair"


@dataclass(frozen=True)
class VideoInfo:
    frame_count: int
    fps: float
    width: int
    height: int


@dataclass(frozen=True)
class SegmentSpec:
    task: str
    episode: str
    seg: str
    video_path: Path
    mask_path: Path
    info: VideoInfo


@dataclass(frozen=True)
class ClipSpec:
    task: str
    episode: str
    seg: str
    clip_idx: int
    clip_start_frame: int
    source_frame_indices: tuple[int, ...]
    source_video: Path
    mask_path: Path
    source_info: VideoInfo
    tail_aligned: bool

    @property
    def source_segment_id(self) -> str:
        return f"{self.task}/{self.episode}/{self.seg}"

    @property
    def source_id(self) -> str:
        return f"{self.source_segment_id}_clip{self.clip_idx:02d}"


@dataclass(frozen=True)
class SeedanceSpec:
    task: str
    episode: str
    seg: str
    clip_idx: int
    clip_start_frame: int
    robot_frame_indices: tuple[int, ...]
    human_frame_indices: tuple[int, ...]
    source_video: Path
    source_info: VideoInfo
    tail_aligned: bool

    @property
    def source_segment_id(self) -> str:
        return f"{self.task}/{self.episode}/{self.seg}"

    @property
    def source_id(self) -> str:
        return f"{self.source_segment_id}_clip{self.clip_idx:02d}"


def short_task_name(task: str) -> str:
    return task.strip().replace("G1_WBT_", "")


def expand_tasks(task_spec: str) -> list[str]:
    groups = {
        "all": TRAINING_TASKS,
        "training": TRAINING_TASKS,
        "inspire": [task for task in ALL_TASKS if "G1_WBT_Inspire_" in task],
        "brainco": [task for task in ALL_TASKS if "G1_WBT_Brainco_" in task],
    }
    key = task_spec.lower()
    if key in groups:
        return [short_task_name(task) for task in groups[key]]
    tasks = [short_task_name(part) for part in task_spec.split(",") if part.strip()]
    if not tasks:
        raise ValueError("--task must not be empty")
    return tasks


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


def clip_starts(
    frame_count: int,
    *,
    num_frames: int,
    stride_frames: int,
    include_tail: bool,
) -> list[tuple[int, bool]]:
    if num_frames <= 0:
        raise ValueError("--num-frames must be positive")
    if stride_frames <= 0:
        raise ValueError("--clip-stride is too small")
    if frame_count < num_frames:
        return []
    starts = list(range(0, frame_count - num_frames + 1, stride_frames))
    if include_tail:
        tail = frame_count - num_frames
        if tail >= 0 and tail not in starts:
            starts.append(tail)
    starts = sorted(set(starts))
    return [(start, start % stride_frames != 0) for start in starts]


def segment_specs(args: argparse.Namespace, tasks: list[str]) -> list[SegmentSpec]:
    specs: list[SegmentSpec] = []
    for task in tasks:
        task_dir = args.segment_root / task
        if not task_dir.is_dir():
            raise FileNotFoundError(f"segment task dir not found: {task_dir}")
        task_count = 0
        for video_path in sorted(task_dir.glob("ep*/seg*_video.mp4")):
            if args.max_segments_per_task > 0 and task_count >= args.max_segments_per_task:
                break
            match = re.match(r"(seg\d+)_video\.mp4$", video_path.name)
            if not match:
                continue
            episode = video_path.parent.name
            seg = match.group(1)
            mask_path = args.sam2_mask_root / task / episode / f"{seg}.npz"
            if not mask_path.is_file():
                raise FileNotFoundError(f"SAM2 mask not found: {mask_path}")
            specs.append(
                SegmentSpec(
                    task=task,
                    episode=episode,
                    seg=seg,
                    video_path=video_path,
                    mask_path=mask_path,
                    info=probe_video(video_path),
                )
            )
            task_count += 1
    specs.sort(key=lambda item: f"{item.task}/{item.episode}/{item.seg}")
    return specs


def build_clip_specs(
    segments: list[SegmentSpec],
    *,
    num_frames: int,
    target_fps: int,
    clip_stride: float,
    include_tail: bool,
    max_segments_per_task: int,
) -> list[ClipSpec]:
    by_task_seen: dict[str, int] = {}
    specs: list[ClipSpec] = []
    for segment in segments:
        if max_segments_per_task > 0:
            count = by_task_seen.get(segment.task, 0)
            if count >= max_segments_per_task:
                continue
            by_task_seen[segment.task] = count + 1
        if round(segment.info.fps) != target_fps:
            raise ValueError(
                f"robot segment must be {target_fps}fps, got {segment.info.fps}: "
                f"{segment.video_path}"
            )
        stride_frames = int(round(clip_stride * segment.info.fps))
        starts = clip_starts(
            segment.info.frame_count,
            num_frames=num_frames,
            stride_frames=stride_frames,
            include_tail=include_tail,
        )
        for clip_idx, (start_frame, tail_aligned) in enumerate(starts):
            indices = tuple(range(start_frame, start_frame + num_frames))
            specs.append(
                ClipSpec(
                    task=segment.task,
                    episode=segment.episode,
                    seg=segment.seg,
                    clip_idx=clip_idx,
                    clip_start_frame=start_frame,
                    source_frame_indices=indices,
                    source_video=segment.video_path,
                    mask_path=segment.mask_path,
                    source_info=segment.info,
                    tail_aligned=tail_aligned,
                )
            )
    return specs


def read_frames_by_index(path: Path, indices: tuple[int, ...]) -> list[np.ndarray]:
    if not indices:
        raise ValueError("indices must not be empty")
    unique_indices = sorted(set(indices))
    frames_by_index: dict[int, np.ndarray] = {}
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, unique_indices[0])
    cursor = unique_indices[0]
    wanted = set(unique_indices)
    while cursor <= unique_indices[-1]:
        ok, frame = cap.read()
        if not ok:
            break
        if cursor in wanted:
            frames_by_index[cursor] = frame
        cursor += 1
    cap.release()
    missing = sorted(wanted - set(frames_by_index))
    if missing:
        raise ValueError(f"video ended before frames {missing[:5]}: {path}")
    return [frames_by_index[index] for index in indices]


def write_video_bgr(frames: list[np.ndarray], out_path: Path, fps: int) -> None:
    if not frames:
        raise ValueError(f"no frames to write: {out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    container, stream = open_video_writer(str(out_path), width, height, fps=fps)
    for frame in frames:
        if frame.shape[:2] != (height, width):
            raise ValueError(
                f"frame size mismatch for {out_path}: {frame.shape[:2]} != "
                f"{(height, width)}"
            )
        write_frame(container, stream, frame)
    close_video(container, stream)


def load_masks(mask_path: Path, indices: tuple[int, ...]) -> list[np.ndarray]:
    with np.load(mask_path) as data:
        if "masks" not in data:
            raise ValueError(f"SAM2 mask npz missing 'masks': {mask_path}")
        masks = data["masks"]
    if masks.ndim != 3:
        raise ValueError(f"SAM2 masks must have shape [T,H,W], got {masks.shape}: {mask_path}")
    if max(indices) >= len(masks):
        raise ValueError(
            f"SAM2 mask too short for {mask_path}: need {max(indices)}, count={len(masks)}"
        )
    return [masks[index].astype(np.uint8) for index in indices]


def soften_mask(mask: np.ndarray, pixel_expand: int) -> np.ndarray:
    out = cv2.GaussianBlur(mask, (7, 7), 0)
    out = (out > 128).astype(np.uint8) * 255
    diameter = 15 + 2 * pixel_expand
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diameter, diameter))
    out = cv2.dilate(out, kernel)
    blur_k = 21 + 2 * (pixel_expand // 2)
    if blur_k % 2 == 0:
        blur_k += 1
    return cv2.GaussianBlur(out, (blur_k, blur_k), 0)


def blur_frame_in_mask(frame: np.ndarray, mask: np.ndarray, ksize: int) -> np.ndarray:
    blurred = cv2.GaussianBlur(frame, (ksize, ksize), 0)
    alpha = mask.astype(np.float32) / 255.0
    out = alpha[..., None] * blurred + (1.0 - alpha[..., None]) * frame
    return np.clip(out, 0, 255).astype(np.uint8)


def clip_filename(clip_idx: int, seg: str) -> str:
    return f"{seg}_clip{clip_idx:02d}.mp4"


def base_record(
    clip: ClipSpec,
    *,
    kind: str,
    rel_video: str,
    args: argparse.Namespace,
) -> dict:
    return {
        "kind": kind,
        "duration": args.format_label,
        "fps": args.target_fps,
        "num_frames": args.num_frames,
        "clip_seconds": args.clip_seconds,
        "clip_stride": args.clip_stride,
        "tail_aligned": clip.tail_aligned,
        "task": clip.task,
        "robot_task": clip.task,
        "episode": clip.episode,
        "seg": clip.seg,
        "clip_idx": clip.clip_idx,
        "clip_start": clip.clip_start_frame / clip.source_info.fps,
        "clip_start_frame": clip.clip_start_frame,
        "source_frame_indices": list(clip.source_frame_indices),
        "source_segment_id": clip.source_segment_id,
        "source_id": clip.source_id,
        "source_video": str(clip.source_video),
        "source_fps": clip.source_info.fps,
        "source_frame_count": clip.source_info.frame_count,
        "video": rel_video,
    }


def process_robot_clip(clip: ClipSpec, args: argparse.Namespace) -> tuple[dict, dict]:
    rel = f"{clip.task}/{clip.episode}/{clip_filename(clip.clip_idx, clip.seg)}"
    original_path = args.slice_root / f"g1_{args.format_label}" / "original" / rel
    blur_path = args.slice_root / f"g1_{args.format_label}" / "sam2_blur" / rel

    write_original = "original" in args.outputs and not (
        args.resume and original_path.is_file()
    )
    write_blur = "sam2_blur" in args.outputs and not (
        args.resume and blur_path.is_file()
    )
    frames = None
    if write_original or write_blur:
        frames = read_frames_by_index(clip.source_video, clip.source_frame_indices)
    if write_original:
        write_video_bgr(frames, original_path, fps=args.target_fps)
    if write_blur:
        masks = load_masks(clip.mask_path, clip.source_frame_indices)
        blurred_frames = []
        for frame, mask in zip(frames, masks):
            if frame.shape[:2] != mask.shape[:2]:
                raise ValueError(
                    f"frame/mask shape mismatch for {clip.mask_path}: "
                    f"{frame.shape[:2]} != {mask.shape[:2]}"
                )
            soft = soften_mask(mask, args.blur_pixel_expand)
            blurred_frames.append(blur_frame_in_mask(frame, soft, args.blur_ksize))
        write_video_bgr(blurred_frames, blur_path, fps=args.target_fps)

    original_record = base_record(
        clip,
        kind="original",
        rel_video=str(original_path.relative_to(args.slice_root / f"g1_{args.format_label}" / "original")),
        args=args,
    )
    blur_record = base_record(
        clip,
        kind="sam2_blur",
        rel_video=str(blur_path.relative_to(args.slice_root / f"g1_{args.format_label}" / "sam2_blur")),
        args=args,
    )
    blur_record["sam2_mask_path"] = str(clip.mask_path)
    blur_record["blur_ksize"] = args.blur_ksize
    blur_record["blur_pixel_expand"] = args.blur_pixel_expand
    return original_record, blur_record


def discover_seedance_specs(
    args: argparse.Namespace,
    clip_by_segment: dict[str, list[ClipSpec]],
) -> list[SeedanceSpec]:
    specs: list[SeedanceSpec] = []
    for task in sorted({key.split("/", 1)[0] for key in clip_by_segment}):
        task_dir = args.seedance_4s_root / task
        if not task_dir.is_dir():
            continue
        for source_video in sorted(task_dir.glob("ep*/seg*_human.mp4")):
            match = re.match(r"(seg\d+)_human\.mp4$", source_video.name)
            if not match:
                continue
            episode = source_video.parent.name
            seg = match.group(1)
            segment_id = f"{task}/{episode}/{seg}"
            robot_clips = clip_by_segment.get(segment_id, [])
            if not robot_clips:
                continue
            info = probe_video(source_video)
            for clip in robot_clips:
                start_sec = clip.clip_start_frame / args.target_fps
                human_indices = tuple(
                    min(
                        max(int(round((start_sec + i / args.target_fps) * info.fps)), 0),
                        info.frame_count - 1,
                    )
                    for i in range(args.num_frames)
                )
                specs.append(
                    SeedanceSpec(
                        task=task,
                        episode=episode,
                        seg=seg,
                        clip_idx=clip.clip_idx,
                        clip_start_frame=clip.clip_start_frame,
                        robot_frame_indices=clip.source_frame_indices,
                        human_frame_indices=human_indices,
                        source_video=source_video,
                        source_info=info,
                        tail_aligned=clip.tail_aligned,
                    )
                )
    specs.sort(key=lambda item: item.source_id)
    return specs


def process_seedance_clip(spec: SeedanceSpec, args: argparse.Namespace) -> dict:
    rel = f"{spec.task}/{spec.episode}/{clip_filename(spec.clip_idx, spec.seg)}"
    out_path = args.slice_root / f"g1_{args.format_label}" / "seedance_direct" / rel
    if not (args.resume and out_path.is_file()):
        frames = read_frames_by_index(spec.source_video, spec.human_frame_indices)
        write_video_bgr(frames, out_path, fps=args.target_fps)
    return {
        "kind": "seedance_direct",
        "duration": args.format_label,
        "fps": args.target_fps,
        "num_frames": args.num_frames,
        "clip_seconds": args.clip_seconds,
        "clip_stride": args.clip_stride,
        "tail_aligned": spec.tail_aligned,
        "task": spec.task,
        "robot_task": spec.task,
        "episode": spec.episode,
        "seg": spec.seg,
        "clip_idx": spec.clip_idx,
        "clip_start": spec.clip_start_frame / args.target_fps,
        "clip_start_frame": spec.clip_start_frame,
        "source_frame_indices": list(spec.robot_frame_indices),
        "human_source_frame_indices": list(spec.human_frame_indices),
        "source_segment_id": spec.source_segment_id,
        "source_id": spec.source_id,
        "source_video": str(spec.source_video),
        "source_fps": spec.source_info.fps,
        "source_frame_count": spec.source_info.frame_count,
        "video": str(out_path.relative_to(args.slice_root / f"g1_{args.format_label}" / "seedance_direct")),
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")


def write_manifest_group(root: Path, rows: list[dict]) -> None:
    by_task: dict[str, list[dict]] = {}
    for row in rows:
        by_task.setdefault(row["task"], []).append(row)
    for task, task_rows in sorted(by_task.items()):
        write_jsonl(root / task / "manifest.jsonl", sorted(task_rows, key=lambda row: row["source_id"]))
    write_jsonl(root / "index.jsonl", sorted(rows, key=lambda row: row["source_id"]))


def write_metadata(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["video", "prompt", "control_video"])
        writer.writeheader()
        writer.writerows(rows)


def hardlink_file(src: Path, dst: Path, *, resume: bool) -> None:
    if resume and dst.is_file():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    os.link(src, dst)


def pair_record(
    source: dict,
    *,
    data_type: str,
    pair_name: str,
    video_rel: str,
    control_rel: str,
    input_role: str,
    target_role: str,
) -> dict:
    record = dict(source)
    record["data_type"] = data_type
    record["duration"] = source["duration"]
    record["robot_task"] = source["task"]
    record["video"] = video_rel
    record["control_video"] = control_rel
    record["input_role"] = input_role
    record["target_role"] = target_role
    record["pair_id"] = pair_name
    return record


def build_pair_layout(
    args: argparse.Namespace,
    *,
    data_type: str,
    left_records: list[dict],
    right_records: list[dict],
    left_root: Path,
    right_root: Path,
    input_role: str,
    target_role: str,
) -> list[dict]:
    if data_type not in {"identity_r2r", "blur_r2r", "h2r"}:
        raise ValueError(f"unsupported pair data_type: {data_type}")
    right_by_id = {row["source_id"]: row for row in right_records}
    missing = sorted(row["source_id"] for row in left_records if row["source_id"] not in right_by_id)
    if missing:
        raise ValueError(f"{data_type} missing right-side records: {missing[:5]}")

    all_rows: list[dict] = []
    pair_base = args.pair_root / data_type / args.format_label
    by_task: dict[str, list[dict]] = {}
    for row in left_records:
        by_task.setdefault(row["task"], []).append(row)

    for task, task_rows in sorted(by_task.items()):
        pair_dir = pair_base / task
        metadata_rows = []
        manifest_rows = []
        for pair_index, left in enumerate(sorted(task_rows, key=lambda row: row["source_id"])):
            right = right_by_id[left["source_id"]]
            pair_name = f"pair_{pair_index:04d}"
            video_rel = f"video/{pair_name}.mp4"
            control_rel = f"control_video/{pair_name}.mp4"
            if data_type == "identity_r2r":
                video_src = left_root / left["video"]
                control_src = video_src
            elif data_type == "blur_r2r":
                video_src = left_root / left["video"]
                control_src = right_root / right["video"]
            else:
                video_src = left_root / left["video"]
                control_src = right_root / right["video"]
            hardlink_file(video_src, pair_dir / video_rel, resume=args.resume)
            hardlink_file(control_src, pair_dir / control_rel, resume=args.resume)
            metadata_rows.append({
                "video": video_rel,
                "prompt": PROMPT,
                "control_video": control_rel,
            })
            record = pair_record(
                left,
                data_type=data_type,
                pair_name=pair_name,
                video_rel=video_rel,
                control_rel=control_rel,
                input_role=input_role,
                target_role=target_role,
            )
            if data_type == "blur_r2r":
                record["control_degrade"] = "sam2_blur"
                record["sam2_blur_video"] = right["video"]
                record["sam2_mask_path"] = right["sam2_mask_path"]
                record["blur_ksize"] = right["blur_ksize"]
                record["blur_pixel_expand"] = right["blur_pixel_expand"]
            if data_type == "h2r":
                record["human_src"] = str(right_root / right["video"])
                record["human_source_4s"] = right["source_video"]
                record["human_source_frame_indices"] = right["human_source_frame_indices"]
            manifest_rows.append(record)
        write_metadata(pair_dir / "metadata.csv", metadata_rows)
        write_jsonl(pair_dir / "manifest.jsonl", manifest_rows)
        all_rows.extend(manifest_rows)
    write_jsonl(pair_base / "index.jsonl", sorted(all_rows, key=lambda row: row["source_id"]))
    return all_rows


def filter_records_by_source_ids(records: list[dict], source_ids: set[str]) -> list[dict]:
    return [record for record in records if record["source_id"] in source_ids]


def clean_outputs(args: argparse.Namespace) -> None:
    slice_dir = args.slice_root / f"g1_{args.format_label}"
    if slice_dir.exists():
        shutil.rmtree(slice_dir)
    for data_type in ("identity_r2r", "blur_r2r", "h2r"):
        path = args.pair_root / data_type / args.format_label
        if path.exists():
            shutil.rmtree(path)


def parse_outputs(value: str) -> set[str]:
    allowed = {
        "original",
        "seedance_direct",
        "sam2_blur",
        "identity_pair",
        "blur_pair",
        "h2r_pair",
    }
    outputs = {part.strip() for part in value.split(",") if part.strip()}
    unknown = sorted(outputs - allowed)
    if unknown:
        raise ValueError(f"unknown outputs: {unknown}; allowed={sorted(allowed)}")
    if not outputs:
        raise ValueError("--outputs must not be empty")
    if "identity_pair" in outputs:
        outputs.add("original")
    if "blur_pair" in outputs:
        outputs.update({"original", "sam2_blur"})
    if "h2r_pair" in outputs:
        outputs.update({"original", "seedance_direct"})
    return outputs


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)

    parser = argparse.ArgumentParser(
        description="Generate G1 2s/30fps original, Seedance, and SAM2-blur slices."
    )
    parser.add_argument("--task", default="all", help="task short name, CSV, or all/training")
    parser.add_argument("--format-label", default=DEFAULT_FORMAT_LABEL)
    parser.add_argument("--target-fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--num-frames", type=int, default=DEFAULT_NUM_FRAMES)
    parser.add_argument("--clip-seconds", type=float, default=DEFAULT_CLIP_SECONDS)
    parser.add_argument("--clip-stride", type=float, default=DEFAULT_CLIP_STRIDE)
    parser.add_argument("--no-include-tail", dest="include_tail", action="store_false", default=True)
    parser.add_argument(
        "--outputs",
        default="original,seedance_direct,sam2_blur,identity_pair,blur_pair,h2r_pair",
        help="CSV subset: original,seedance_direct,sam2_blur,identity_pair,blur_pair,h2r_pair",
    )
    parser.add_argument("--segment-root", type=Path, default=DEFAULT_SEGMENT_ROOT)
    parser.add_argument("--seedance-4s-root", type=Path, default=DEFAULT_SEEDANCE_4S_ROOT)
    parser.add_argument("--sam2-mask-root", type=Path, default=DEFAULT_SAM2_MASK_ROOT)
    parser.add_argument("--slice-root", type=Path, default=DEFAULT_SLICE_ROOT)
    parser.add_argument("--pair-root", type=Path, default=DEFAULT_PAIR_ROOT)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-segments-per-task", type=int, default=0)
    parser.add_argument("--blur-ksize", type=int, default=DEFAULT_BLUR_KSIZE)
    parser.add_argument("--blur-pixel-expand", type=int, default=DEFAULT_BLUR_PIXEL_EXPAND)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.target_fps <= 0:
        raise ValueError("--target-fps must be positive")
    if args.num_frames <= 0:
        raise ValueError("--num-frames must be positive")
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    if args.max_segments_per_task < 0:
        raise ValueError("--max-segments-per-task must be non-negative")
    if args.blur_ksize <= 0 or args.blur_ksize % 2 == 0:
        raise ValueError("--blur-ksize must be a positive odd integer")
    if args.blur_pixel_expand < 0:
        raise ValueError("--blur-pixel-expand must be non-negative")

    args.outputs = parse_outputs(args.outputs)
    tasks = expand_tasks(args.task)
    if args.clean and args.dry_run:
        raise ValueError("--clean and --dry-run cannot be used together")
    if args.clean:
        clean_outputs(args)

    segments = segment_specs(args, tasks)
    clips = build_clip_specs(
        segments,
        num_frames=args.num_frames,
        target_fps=args.target_fps,
        clip_stride=args.clip_stride,
        include_tail=args.include_tail,
        max_segments_per_task=args.max_segments_per_task,
    )
    clip_by_segment: dict[str, list[ClipSpec]] = {}
    for clip in clips:
        clip_by_segment.setdefault(clip.source_segment_id, []).append(clip)
    seedance_specs = discover_seedance_specs(args, clip_by_segment)

    print("G1 2s/30fps slice data")
    print(f"  tasks:        {tasks}")
    print(f"  format:       {args.format_label}")
    print(f"  fps:          {args.target_fps}")
    print(f"  num_frames:   {args.num_frames}")
    print(f"  clip_stride:  {args.clip_stride}s")
    print(f"  outputs:      {sorted(args.outputs)}")
    print(f"  segments:     {len(segments)}")
    print(f"  robot clips:  {len(clips)}")
    print(f"  seedance:     {len(seedance_specs)}")
    print(f"  slice root:   {args.slice_root / ('g1_' + args.format_label)}")
    print(f"  pair root:    {args.pair_root}")
    if args.dry_run:
        return

    original_rows: list[dict] = []
    blur_rows: list[dict] = []
    if "original" in args.outputs or "sam2_blur" in args.outputs:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(process_robot_clip, clip, args) for clip in clips]
            for done, future in enumerate(as_completed(futures), start=1):
                original, blur = future.result()
                original_rows.append(original)
                blur_rows.append(blur)
                if done % 200 == 0 or done == len(futures):
                    print(f"  robot/sam2 clips: {done}/{len(futures)}", flush=True)

    seedance_rows: list[dict] = []
    if "seedance_direct" in args.outputs:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(process_seedance_clip, spec, args) for spec in seedance_specs]
            for done, future in enumerate(as_completed(futures), start=1):
                seedance_rows.append(future.result())
                if done % 50 == 0 or done == len(futures):
                    print(f"  seedance clips: {done}/{len(futures)}", flush=True)

    slice_base = args.slice_root / f"g1_{args.format_label}"
    original_root = slice_base / "original"
    blur_root = slice_base / "sam2_blur"
    seedance_root = slice_base / "seedance_direct"
    if original_rows:
        write_manifest_group(original_root, original_rows)
    if blur_rows:
        write_manifest_group(blur_root, blur_rows)
    if seedance_rows:
        write_manifest_group(seedance_root, seedance_rows)

    if "identity_pair" in args.outputs:
        rows = build_pair_layout(
            args,
            data_type="identity_r2r",
            left_records=original_rows,
            right_records=original_rows,
            left_root=original_root,
            right_root=original_root,
            input_role="robot",
            target_role="robot",
        )
        print(f"  identity_r2r pairs: {len(rows)}")
    if "blur_pair" in args.outputs:
        rows = build_pair_layout(
            args,
            data_type="blur_r2r",
            left_records=original_rows,
            right_records=blur_rows,
            left_root=original_root,
            right_root=blur_root,
            input_role="robot",
            target_role="robot",
        )
        print(f"  blur_r2r pairs: {len(rows)}")
    if "h2r_pair" in args.outputs:
        seedance_source_ids = {row["source_id"] for row in seedance_rows}
        h2r_original_rows = filter_records_by_source_ids(original_rows, seedance_source_ids)
        rows = build_pair_layout(
            args,
            data_type="h2r",
            left_records=h2r_original_rows,
            right_records=seedance_rows,
            left_root=original_root,
            right_root=seedance_root,
            input_role="human",
            target_role="robot",
        )
        print(f"  h2r pairs: {len(rows)}")


if __name__ == "__main__":
    main()
