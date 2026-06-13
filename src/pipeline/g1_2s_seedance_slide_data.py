"""Build G1 2s Seedance sliding-window data and pair layout.

This is the dense-Seedance counterpart to ``g1_2s_slice_data``.  It does not
modify the existing ``2s61f30`` slice or pair directories.  Instead it writes a
separate step layout and a separate duration label:

  training_data/g1_2s61f30_seedance_slide/
  training_data/pair/{identity_r2r,blur_r2r,h2r}/2s61f30_slide/

Step2 reuses task076's full robot-only 2s data via hardlinks.  Step1 cuts
Seedance 4s videos with a dense 0.5s stride and creates matching robot-origin
clips from the original robot segment videos.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from src.core.config import MAIN_ROOT
from src.pipeline import g1_2s_slice_data as g1_2s


DEFAULT_BASE_FORMAT_LABEL = "2s61f30"
DEFAULT_DURATION_LABEL = "2s61f30_slide"
DEFAULT_STEP_LAYOUT_NAME = "g1_2s61f30_seedance_slide"
DEFAULT_OUTPUT_ROOT = Path(MAIN_ROOT) / "training_data" / DEFAULT_STEP_LAYOUT_NAME


@dataclass(frozen=True)
class SlideSpec:
    task: str
    episode: str
    seg: str
    clip_idx: int
    clip_start_frame: int
    robot_frame_indices: tuple[int, ...]
    human_frame_indices: tuple[int, ...]
    robot_video: Path
    robot_info: g1_2s.VideoInfo
    seedance_video: Path
    seedance_info: g1_2s.VideoInfo
    tail_aligned: bool
    stride_frames: int

    @property
    def source_segment_id(self) -> str:
        return f"{self.task}/{self.episode}/{self.seg}"

    @property
    def source_id(self) -> str:
        return (
            f"{self.source_segment_id}_slide{self.clip_idx:02d}"
            f"_f{self.clip_start_frame:04d}"
        )

    @property
    def filename(self) -> str:
        return f"{self.seg}_slide{self.clip_idx:02d}_f{self.clip_start_frame:04d}.mp4"

    @property
    def rel_video(self) -> str:
        return f"{self.task}/{self.episode}/{self.filename}"


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"JSONL file not found: {path}")
    rows: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_no}") from exc
    if not rows:
        raise ValueError(f"JSONL file is empty: {path}")
    return rows


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def count_by_task(rows: list[dict]) -> dict[str, int]:
    return dict(sorted(Counter(row["task"] for row in rows).items()))


def limit_rows_per_task(rows: list[dict], limit: int) -> list[dict]:
    if limit <= 0:
        return rows
    kept: list[dict] = []
    seen: dict[str, int] = {}
    for row in sorted(rows, key=lambda item: item["source_id"]):
        task = row["task"]
        count = seen.get(task, 0)
        if count >= limit:
            continue
        kept.append(row)
        seen[task] = count + 1
    return kept


def load_base_rows(args: argparse.Namespace, kind: str) -> list[dict]:
    path = args.base_slice_root / f"g1_{args.base_format_label}" / kind / "index.jsonl"
    rows = read_jsonl(path)
    for row in rows:
        duration = row.get("duration")
        if duration != args.base_format_label:
            raise ValueError(
                f"Base {kind} row has duration={duration!r}, expected "
                f"{args.base_format_label!r}: {row.get('source_id')}"
            )
        video_rel = row.get("video")
        if not video_rel:
            raise ValueError(f"Base {kind} row missing video: {row}")
        video_path = args.base_slice_root / f"g1_{args.base_format_label}" / kind / video_rel
        if not video_path.is_file():
            raise FileNotFoundError(f"Base {kind} video not found: {video_path}")
    return rows


def frame_key(row: dict) -> tuple[str, tuple[int, ...]]:
    return row["source_segment_id"], tuple(row["source_frame_indices"])


def human_frame_key(row: dict) -> tuple[str, tuple[int, ...], tuple[int, ...]]:
    return (
        row["source_segment_id"],
        tuple(row["source_frame_indices"]),
        tuple(row["human_source_frame_indices"]),
    )


def map_by_frame_key(rows: list[dict], label: str) -> dict[tuple, dict]:
    mapped: dict[tuple, dict] = {}
    for row in rows:
        key = frame_key(row)
        if key in mapped:
            raise ValueError(f"Duplicate {label} frame key: {key}")
        mapped[key] = row
    return mapped


def map_seedance_by_frame_key(rows: list[dict]) -> dict[tuple, dict]:
    mapped: dict[tuple, dict] = {}
    for row in rows:
        key = human_frame_key(row)
        if key in mapped:
            raise ValueError(f"Duplicate seedance frame key: {key}")
        mapped[key] = row
    return mapped


def prepare_output_file(path: Path, args: argparse.Namespace) -> bool:
    """Return True when the caller should write ``path``."""
    if args.resume and path.is_file() and path.stat().st_size > 0:
        return False
    if path.exists():
        if not args.allow_overwrite:
            raise FileExistsError(
                f"Output already exists: {path}. Use --resume to keep it or "
                "--allow-overwrite to replace it."
            )
        if path.is_dir():
            raise IsADirectoryError(f"Refusing to overwrite directory: {path}")
        path.unlink()
    path.parent.mkdir(parents=True, exist_ok=True)
    return True


def hardlink_or_skip(src: Path, dst: Path, args: argparse.Namespace) -> bool:
    if not src.is_file():
        raise FileNotFoundError(f"Hardlink source not found: {src}")
    if not prepare_output_file(dst, args):
        return False
    os.link(src, dst)
    return True


def write_video_or_skip(src_video: Path, frame_indices: tuple[int, ...], dst: Path, args: argparse.Namespace) -> bool:
    if not prepare_output_file(dst, args):
        return False
    frames = g1_2s.read_frames_by_index(src_video, frame_indices)
    g1_2s.write_video_bgr(frames, dst, fps=args.target_fps)
    return True


def step_record_from_base(
    row: dict,
    *,
    role: str,
    dst_rel: str,
    src_path: Path,
    args: argparse.Namespace,
) -> dict:
    record = dict(row)
    record["base_duration"] = row["duration"]
    record["duration"] = args.duration_label
    record["source_kind"] = row.get("kind", "")
    record["kind"] = f"step2_{role}"
    record["step"] = "step2"
    record["role"] = role
    record["video"] = dst_rel
    record["step_layout_source"] = str(args.output_root)
    record["source_slice_video"] = str(src_path)
    return record


def build_step2_layout(
    rows: list[dict],
    *,
    role: str,
    base_kind: str,
    dst_root: Path,
    args: argparse.Namespace,
) -> list[dict]:
    base_root = args.base_slice_root / f"g1_{args.base_format_label}" / base_kind
    out_rows: list[dict] = []
    for row in sorted(rows, key=lambda item: item["source_id"]):
        rel = row["video"]
        src = base_root / rel
        dst = dst_root / rel
        if not args.dry_run:
            hardlink_or_skip(src, dst, args)
        out_rows.append(
            step_record_from_base(row, role=role, dst_rel=rel, src_path=src, args=args)
        )
    return out_rows


def stride_frames_from_args(args: argparse.Namespace) -> int:
    if args.stride_frames > 0:
        return args.stride_frames
    stride_frames = int(round(args.stride_seconds * args.target_fps))
    if stride_frames <= 0:
        raise ValueError("--stride-seconds is too small")
    return stride_frames


def discover_slide_specs(args: argparse.Namespace, tasks: list[str]) -> list[SlideSpec]:
    stride_frames = stride_frames_from_args(args)
    specs: list[SlideSpec] = []
    for task in tasks:
        task_dir = args.seedance_4s_root / task
        if not task_dir.is_dir():
            raise FileNotFoundError(f"Seedance 4s task dir not found: {task_dir}")
        task_source_count = 0
        for seedance_video in sorted(task_dir.glob("ep*/seg*_human.mp4")):
            if (
                args.max_seedance_sources_per_task > 0
                and task_source_count >= args.max_seedance_sources_per_task
            ):
                break
            match = re.match(r"(seg\d+)_human\.mp4$", seedance_video.name)
            if not match:
                continue
            episode = seedance_video.parent.name
            seg = match.group(1)
            robot_video = args.segment_root / task / episode / f"{seg}_video.mp4"
            if not robot_video.is_file():
                raise FileNotFoundError(
                    f"Robot segment for Seedance source not found: {robot_video}"
                )
            robot_info = g1_2s.probe_video(robot_video)
            seedance_info = g1_2s.probe_video(seedance_video)
            if round(robot_info.fps) != args.target_fps:
                raise ValueError(
                    f"Robot segment must be {args.target_fps}fps, got "
                    f"{robot_info.fps}: {robot_video}"
                )
            starts = g1_2s.clip_starts(
                robot_info.frame_count,
                num_frames=args.num_frames,
                stride_frames=stride_frames,
                include_tail=args.include_tail,
            )
            if not starts:
                raise ValueError(
                    f"Robot segment has no valid {args.num_frames}-frame windows: "
                    f"{robot_video}"
                )
            for clip_idx, (start_frame, tail_aligned) in enumerate(starts):
                robot_indices = tuple(range(start_frame, start_frame + args.num_frames))
                start_sec = start_frame / args.target_fps
                human_indices = tuple(
                    min(
                        max(int(round((start_sec + i / args.target_fps) * seedance_info.fps)), 0),
                        seedance_info.frame_count - 1,
                    )
                    for i in range(args.num_frames)
                )
                specs.append(
                    SlideSpec(
                        task=task,
                        episode=episode,
                        seg=seg,
                        clip_idx=clip_idx,
                        clip_start_frame=start_frame,
                        robot_frame_indices=robot_indices,
                        human_frame_indices=human_indices,
                        robot_video=robot_video,
                        robot_info=robot_info,
                        seedance_video=seedance_video,
                        seedance_info=seedance_info,
                        tail_aligned=tail_aligned,
                        stride_frames=stride_frames,
                    )
                )
            task_source_count += 1
    return sorted(specs, key=lambda item: item.source_id)


def step1_base_record(
    spec: SlideSpec,
    *,
    role: str,
    kind: str,
    rel_video: str,
    source_video: Path,
    source_info: g1_2s.VideoInfo,
    args: argparse.Namespace,
) -> dict:
    return {
        "kind": kind,
        "step": "step1",
        "role": role,
        "duration": args.duration_label,
        "base_duration": args.base_format_label,
        "fps": args.target_fps,
        "num_frames": args.num_frames,
        "clip_seconds": args.clip_seconds,
        "clip_stride": args.stride_seconds if args.stride_frames <= 0 else spec.stride_frames / args.target_fps,
        "stride_frames": spec.stride_frames,
        "tail_aligned": spec.tail_aligned,
        "task": spec.task,
        "robot_task": spec.task,
        "episode": spec.episode,
        "seg": spec.seg,
        "clip_idx": spec.clip_idx,
        "clip_start": spec.clip_start_frame / args.target_fps,
        "clip_start_frame": spec.clip_start_frame,
        "source_frame_indices": list(spec.robot_frame_indices),
        "source_segment_id": spec.source_segment_id,
        "source_id": spec.source_id,
        "source_video": str(source_video),
        "source_fps": source_info.fps,
        "source_frame_count": source_info.frame_count,
        "video": rel_video,
        "step_layout_source": str(args.output_root),
    }


def process_step1_spec(
    spec: SlideSpec,
    args: argparse.Namespace,
    base_original_by_frames: dict[tuple, dict],
    base_seedance_by_frames: dict[tuple, dict],
) -> tuple[dict, dict]:
    origin_root = args.output_root / "step1" / "origin"
    human_root = args.output_root / "step1" / "human"
    origin_path = origin_root / spec.rel_video
    human_path = human_root / spec.rel_video

    origin_record = step1_base_record(
        spec,
        role="origin",
        kind="step1_origin",
        rel_video=spec.rel_video,
        source_video=spec.robot_video,
        source_info=spec.robot_info,
        args=args,
    )
    human_record = step1_base_record(
        spec,
        role="human",
        kind="step1_human",
        rel_video=spec.rel_video,
        source_video=spec.seedance_video,
        source_info=spec.seedance_info,
        args=args,
    )
    human_record["human_source_frame_indices"] = list(spec.human_frame_indices)
    human_record["seedance_source_video"] = str(spec.seedance_video)
    human_record["seedance_source_fps"] = spec.seedance_info.fps
    human_record["seedance_source_frame_count"] = spec.seedance_info.frame_count

    original_key = (spec.source_segment_id, spec.robot_frame_indices)
    base_origin = base_original_by_frames.get(original_key)
    if base_origin is not None:
        src = args.base_slice_root / f"g1_{args.base_format_label}" / "original" / base_origin["video"]
        if not args.dry_run:
            hardlink_or_skip(src, origin_path, args)
        origin_record["source_slice_video"] = str(src)
        origin_record["write_strategy"] = "hardlink_base_original"
    else:
        if not args.dry_run:
            write_video_or_skip(spec.robot_video, spec.robot_frame_indices, origin_path, args)
        origin_record["write_strategy"] = "decode_robot_segment"

    seedance_key = (spec.source_segment_id, spec.robot_frame_indices, spec.human_frame_indices)
    base_human = base_seedance_by_frames.get(seedance_key)
    if base_human is not None:
        src = args.base_slice_root / f"g1_{args.base_format_label}" / "seedance_direct" / base_human["video"]
        if not args.dry_run:
            hardlink_or_skip(src, human_path, args)
        human_record["source_slice_video"] = str(src)
        human_record["write_strategy"] = "hardlink_base_seedance"
    else:
        if not args.dry_run:
            write_video_or_skip(spec.seedance_video, spec.human_frame_indices, human_path, args)
        human_record["write_strategy"] = "decode_seedance_source"

    return origin_record, human_record


def build_pair_layout(
    args: argparse.Namespace,
    *,
    data_type: str,
    target_records: list[dict],
    control_records: list[dict],
    target_root: Path,
    control_root: Path,
    input_role: str,
    target_role: str,
    input_step_role: str,
    target_step_role: str,
) -> list[dict]:
    control_by_id = {row["source_id"]: row for row in control_records}
    missing = sorted(row["source_id"] for row in target_records if row["source_id"] not in control_by_id)
    if missing:
        raise ValueError(f"{data_type} missing control records: {missing[:5]}")

    pair_base = args.pair_root / data_type / args.duration_label
    all_rows: list[dict] = []
    by_task: dict[str, list[dict]] = {}
    for row in target_records:
        by_task.setdefault(row["task"], []).append(row)

    for task, task_rows in sorted(by_task.items()):
        pair_dir = pair_base / task
        metadata_rows: list[dict] = []
        manifest_rows: list[dict] = []
        for pair_index, target in enumerate(sorted(task_rows, key=lambda row: row["source_id"])):
            control = control_by_id[target["source_id"]]
            pair_name = f"pair_{pair_index:04d}"
            video_rel = f"video/{pair_name}.mp4"
            control_rel = f"control_video/{pair_name}.mp4"
            if not args.dry_run:
                hardlink_or_skip(target_root / target["video"], pair_dir / video_rel, args)
                hardlink_or_skip(control_root / control["video"], pair_dir / control_rel, args)
            metadata_rows.append(
                {
                    "video": video_rel,
                    "prompt": g1_2s.PROMPT,
                    "control_video": control_rel,
                }
            )

            record = dict(target)
            record["data_type"] = data_type
            record["duration"] = args.duration_label
            record["robot_task"] = target["task"]
            record["video"] = video_rel
            record["control_video"] = control_rel
            record["input_role"] = input_role
            record["target_role"] = target_role
            record["input_step_role"] = input_step_role
            record["target_step_role"] = target_step_role
            record["pair_id"] = pair_name
            record["prompt"] = g1_2s.PROMPT
            record["step_layout_source"] = str(args.output_root)
            record["input_step_video"] = control["video"]
            record["target_step_video"] = target["video"]
            if data_type == "blur_r2r":
                record["control_degrade"] = "sam2_blur"
                record["sam2_blur_video"] = control["video"]
                record["sam2_mask_path"] = control["sam2_mask_path"]
                record["blur_ksize"] = control["blur_ksize"]
                record["blur_pixel_expand"] = control["blur_pixel_expand"]
            if data_type == "h2r":
                record["human_src"] = str(control_root / control["video"])
                record["human_source_4s"] = control["seedance_source_video"]
                record["human_source_frame_indices"] = control["human_source_frame_indices"]
                record["seedance_source_fps"] = control["seedance_source_fps"]
            manifest_rows.append(record)
        if not args.dry_run:
            g1_2s.write_metadata(pair_dir / "metadata.csv", metadata_rows)
            g1_2s.write_jsonl(pair_dir / "manifest.jsonl", manifest_rows)
        all_rows.extend(manifest_rows)

    if not args.dry_run:
        g1_2s.write_jsonl(pair_base / "index.jsonl", sorted(all_rows, key=lambda row: row["source_id"]))
    return all_rows


def print_plan(
    *,
    tasks: list[str],
    args: argparse.Namespace,
    step2_origin_rows: list[dict],
    step2_blur_rows: list[dict],
    slide_specs: list[SlideSpec],
) -> None:
    starts = Counter(spec.clip_start_frame for spec in slide_specs)
    by_task = Counter(spec.task for spec in slide_specs)
    source_ids = {spec.source_segment_id for spec in slide_specs}
    print("G1 2s Seedance sliding-window data")
    print(f"  tasks:             {tasks}")
    print(f"  base format:       {args.base_format_label}")
    print(f"  duration label:    {args.duration_label}")
    print(f"  fps / frames:      {args.target_fps} / {args.num_frames}")
    print(f"  stride frames:     {stride_frames_from_args(args)}")
    print(f"  stride seconds:    {stride_frames_from_args(args) / args.target_fps:.3f}")
    print(f"  step layout root:  {args.output_root}")
    print(f"  pair root:         {args.pair_root}")
    print(f"  Step2 origin:      {len(step2_origin_rows)} {count_by_task(step2_origin_rows)}")
    print(f"  Step2 blur:        {len(step2_blur_rows)} {count_by_task(step2_blur_rows)}")
    print(f"  Seedance sources:  {len(source_ids)}")
    print(f"  Step1 windows:     {len(slide_specs)} {dict(sorted(by_task.items()))}")
    print(f"  window starts:     {dict(sorted(starts.items()))}")


def write_summary(
    args: argparse.Namespace,
    *,
    tasks: list[str],
    step2_origin_rows: list[dict],
    step2_blur_rows: list[dict],
    step1_origin_rows: list[dict],
    step1_human_rows: list[dict],
    identity_rows: list[dict],
    blur_rows: list[dict],
    h2r_rows: list[dict],
) -> None:
    summary = {
        "duration_label": args.duration_label,
        "base_format_label": args.base_format_label,
        "target_fps": args.target_fps,
        "num_frames": args.num_frames,
        "clip_seconds": args.clip_seconds,
        "stride_frames": stride_frames_from_args(args),
        "stride_seconds": stride_frames_from_args(args) / args.target_fps,
        "tasks": tasks,
        "output_root": str(args.output_root),
        "pair_root": str(args.pair_root),
        "counts": {
            "step2_origin": len(step2_origin_rows),
            "step2_blur": len(step2_blur_rows),
            "step1_origin": len(step1_origin_rows),
            "step1_human": len(step1_human_rows),
            "identity_r2r_pair": len(identity_rows),
            "blur_r2r_pair": len(blur_rows),
            "h2r_pair": len(h2r_rows),
        },
        "counts_by_task": {
            "step2_origin": count_by_task(step2_origin_rows),
            "step2_blur": count_by_task(step2_blur_rows),
            "step1_origin": count_by_task(step1_origin_rows),
            "step1_human": count_by_task(step1_human_rows),
            "identity_r2r_pair": count_by_task(identity_rows),
            "blur_r2r_pair": count_by_task(blur_rows),
            "h2r_pair": count_by_task(h2r_rows),
        },
        "window_starts": dict(
            sorted(Counter(row["clip_start_frame"] for row in step1_human_rows).items())
        ),
        "write_strategies": dict(
            sorted(Counter(row.get("write_strategy", "") for row in step1_origin_rows + step1_human_rows).items())
        ),
    }
    write_json(args.output_root / "summary.json", summary)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate G1 2s Seedance sliding-window step layout and pair data."
    )
    parser.add_argument("--task", default="all", help="task short name, CSV, or all/training")
    parser.add_argument("--base-format-label", default=DEFAULT_BASE_FORMAT_LABEL)
    parser.add_argument("--duration-label", default=DEFAULT_DURATION_LABEL)
    parser.add_argument("--target-fps", type=int, default=g1_2s.DEFAULT_FPS)
    parser.add_argument("--num-frames", type=int, default=g1_2s.DEFAULT_NUM_FRAMES)
    parser.add_argument("--clip-seconds", type=float, default=g1_2s.DEFAULT_CLIP_SECONDS)
    parser.add_argument("--stride-seconds", type=float, default=0.5)
    parser.add_argument("--stride-frames", type=int, default=0)
    parser.add_argument("--no-include-tail", dest="include_tail", action="store_false", default=True)
    parser.add_argument("--segment-root", type=Path, default=g1_2s.DEFAULT_SEGMENT_ROOT)
    parser.add_argument("--seedance-4s-root", type=Path, default=g1_2s.DEFAULT_SEEDANCE_4S_ROOT)
    parser.add_argument("--base-slice-root", type=Path, default=g1_2s.DEFAULT_SLICE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--pair-root", type=Path, default=g1_2s.DEFAULT_PAIR_ROOT)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-seedance-sources-per-task", type=int, default=0)
    parser.add_argument("--max-step2-records-per-task", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.target_fps <= 0:
        raise ValueError("--target-fps must be positive")
    if args.num_frames <= 0:
        raise ValueError("--num-frames must be positive")
    if args.clip_seconds <= 0.0:
        raise ValueError("--clip-seconds must be positive")
    if args.stride_seconds <= 0.0 and args.stride_frames <= 0:
        raise ValueError("--stride-seconds or --stride-frames must be positive")
    if args.stride_frames < 0:
        raise ValueError("--stride-frames must be non-negative")
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    if args.max_seedance_sources_per_task < 0:
        raise ValueError("--max-seedance-sources-per-task must be non-negative")
    if args.max_step2_records_per_task < 0:
        raise ValueError("--max-step2-records-per-task must be non-negative")


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    args = parse_args()
    validate_args(args)

    tasks = g1_2s.expand_tasks(args.task)
    base_original_rows = limit_rows_per_task(
        load_base_rows(args, "original"), args.max_step2_records_per_task
    )
    base_blur_rows = limit_rows_per_task(
        load_base_rows(args, "sam2_blur"), args.max_step2_records_per_task
    )
    base_seedance_rows = load_base_rows(args, "seedance_direct")

    original_ids = {row["source_id"] for row in base_original_rows}
    blur_ids = {row["source_id"] for row in base_blur_rows}
    if original_ids != blur_ids:
        missing_blur = sorted(original_ids - blur_ids)
        missing_origin = sorted(blur_ids - original_ids)
        raise ValueError(
            "Base Step2 original/blur source_id mismatch: "
            f"missing_blur={missing_blur[:5]} missing_origin={missing_origin[:5]}"
        )

    slide_specs = discover_slide_specs(args, tasks)
    step2_origin_root = args.output_root / "step2" / "origin"
    step2_blur_root = args.output_root / "step2" / "blur"
    step1_origin_root = args.output_root / "step1" / "origin"
    step1_human_root = args.output_root / "step1" / "human"

    print_plan(
        tasks=tasks,
        args=args,
        step2_origin_rows=base_original_rows,
        step2_blur_rows=base_blur_rows,
        slide_specs=slide_specs,
    )
    if args.dry_run:
        return

    step2_origin_rows = build_step2_layout(
        base_original_rows,
        role="origin",
        base_kind="original",
        dst_root=step2_origin_root,
        args=args,
    )
    step2_blur_rows = build_step2_layout(
        base_blur_rows,
        role="blur",
        base_kind="sam2_blur",
        dst_root=step2_blur_root,
        args=args,
    )
    g1_2s.write_manifest_group(step2_origin_root, step2_origin_rows)
    g1_2s.write_manifest_group(step2_blur_root, step2_blur_rows)
    print(f"  wrote Step2 origin/blur: {len(step2_origin_rows)} / {len(step2_blur_rows)}")

    base_original_by_frames = map_by_frame_key(load_base_rows(args, "original"), "original")
    base_seedance_by_frames = map_seedance_by_frame_key(base_seedance_rows)
    step1_origin_rows: list[dict] = []
    step1_human_rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(process_step1_spec, spec, args, base_original_by_frames, base_seedance_by_frames)
            for spec in slide_specs
        ]
        for done, future in enumerate(as_completed(futures), start=1):
            origin_row, human_row = future.result()
            step1_origin_rows.append(origin_row)
            step1_human_rows.append(human_row)
            if done % 25 == 0 or done == len(futures):
                print(f"  Step1 clips: {done}/{len(futures)}", flush=True)

    step1_origin_rows = sorted(step1_origin_rows, key=lambda row: row["source_id"])
    step1_human_rows = sorted(step1_human_rows, key=lambda row: row["source_id"])
    g1_2s.write_manifest_group(step1_origin_root, step1_origin_rows)
    g1_2s.write_manifest_group(step1_human_root, step1_human_rows)

    identity_rows = build_pair_layout(
        args,
        data_type="identity_r2r",
        target_records=step2_origin_rows,
        control_records=step2_origin_rows,
        target_root=step2_origin_root,
        control_root=step2_origin_root,
        input_role="robot",
        target_role="robot",
        input_step_role="step2_origin",
        target_step_role="step2_origin",
    )
    blur_rows = build_pair_layout(
        args,
        data_type="blur_r2r",
        target_records=step2_origin_rows,
        control_records=step2_blur_rows,
        target_root=step2_origin_root,
        control_root=step2_blur_root,
        input_role="robot",
        target_role="robot",
        input_step_role="step2_blur",
        target_step_role="step2_origin",
    )
    h2r_rows = build_pair_layout(
        args,
        data_type="h2r",
        target_records=step1_origin_rows,
        control_records=step1_human_rows,
        target_root=step1_origin_root,
        control_root=step1_human_root,
        input_role="human",
        target_role="robot",
        input_step_role="step1_human",
        target_step_role="step1_origin",
    )
    write_summary(
        args,
        tasks=tasks,
        step2_origin_rows=step2_origin_rows,
        step2_blur_rows=step2_blur_rows,
        step1_origin_rows=step1_origin_rows,
        step1_human_rows=step1_human_rows,
        identity_rows=identity_rows,
        blur_rows=blur_rows,
        h2r_rows=h2r_rows,
    )
    print(f"  identity_r2r pairs: {len(identity_rows)}")
    print(f"  blur_r2r pairs:     {len(blur_rows)}")
    print(f"  h2r pairs:          {len(h2r_rows)}")
    print(f"  summary:            {args.output_root / 'summary.json'}")


if __name__ == "__main__":
    main()
