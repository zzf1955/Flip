#!/usr/bin/env python3
"""Slice ep000-ep003 robot clips and generate r2h syn videos for analysis.

This is intentionally separate from ``src.pipeline.r2h_synthesize`` because the
outputs are analysis artifacts under ``output/syn_error_analysis`` rather than
training pairs under ``training_data/pair``.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.core.config import (  # noqa: E402
    DEFAULT_OOD_TASKS,
    DEFAULT_TRAIN_TASKS,
    MAIN_ROOT,
    OUTPUT_DIR,
    TRAINING_TASKS,
)
from src.core.train_utils import load_t5_cache  # noqa: E402
from src.pipeline.r2h_synthesize import (  # noqa: E402
    DEFAULT_DIT_DIR,
    DEFAULT_TOKENIZER,
    DEFAULT_VAE,
    DURATION_SECONDS,
    FRAMES_4K1,
    PROMPT,
    collect_segment_clips,
    cut_robot_clip,
    make_compare,
    read_video_info,
    validate_selected_segments,
    video_complete,
    _generate_human_video,
    _load_generator,
)
from src.pipeline.runtime_data import short_task_name  # noqa: E402


DEFAULT_EPISODES = ("ep000", "ep001", "ep002", "ep003")


def _csv(value: str) -> list[str]:
    return [item.strip() for item in value.replace(";", ",").split(",") if item.strip()]


def _normalise_episode(value: str) -> str:
    token = value.strip()
    if not token:
        raise ValueError("empty episode token")
    if token.lower() == "ep0123":
        raise ValueError("ep0123 is a shorthand list, not one episode token")
    if token.lower().startswith("ep"):
        digits = token[2:]
    else:
        digits = token
    if not digits.isdigit():
        raise ValueError(f"invalid episode token: {value!r}")
    return f"ep{int(digits):03d}"


def parse_episodes(value: str) -> list[str]:
    if value.strip().lower() == "ep0123":
        return list(DEFAULT_EPISODES)
    episodes = [_normalise_episode(item) for item in _csv(value)]
    if not episodes:
        raise ValueError("--episodes must not be empty")
    return sorted(set(episodes))


def expand_source_tasks(value: str) -> list[str]:
    groups = {
        "in_task": [short_task_name(task) for task in DEFAULT_TRAIN_TASKS],
        "intask": [short_task_name(task) for task in DEFAULT_TRAIN_TASKS],
        "train": [short_task_name(task) for task in DEFAULT_TRAIN_TASKS],
        "ood": [short_task_name(task) for task in DEFAULT_OOD_TASKS],
        "all": [short_task_name(task) for task in TRAINING_TASKS],
        "training": [short_task_name(task) for task in TRAINING_TASKS],
    }
    tasks: list[str] = []
    for item in _csv(value):
        key = item.lower().replace("-", "_")
        if key in groups:
            tasks.extend(groups[key])
        else:
            tasks.append(short_task_name(item))
    tasks = list(dict.fromkeys(tasks))
    if not tasks:
        raise ValueError("--source-task must not be empty")
    return tasks


def resolve_main_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return Path(MAIN_ROOT) / path


def rel(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def clip_output_paths(output_root: Path, duration: str, clip) -> tuple[Path, Path, Path]:
    name = f"{clip.seg}_clip{clip.clip_index:02d}.mp4"
    subdir = Path(duration) / clip.task / clip.episode
    robot_path = output_root / "robot" / subdir / name
    syn_path = output_root / "syn" / subdir / name
    compare_path = output_root / "compare" / subdir / name
    return robot_path, syn_path, compare_path


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")


def write_metadata(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["video", "prompt", "control_video"])
        writer.writeheader()
        writer.writerows(rows)


def build_record(
    *,
    clip,
    output_root: Path,
    robot_path: Path,
    syn_path: Path,
    compare_path: Path,
    args: argparse.Namespace,
    run_name: str,
    checkpoint_path: Path,
) -> dict:
    return {
        "clip_dur": clip.clip_dur,
        "clip_idx": clip.clip_index,
        "clip_start": clip.clip_start,
        "compare_video": rel(compare_path, output_root) if compare_path.exists() else "",
        "duration": args.duration,
        "episode": clip.episode,
        "generator_cfg_scale": args.cfg_scale,
        "generator_checkpoint": str(checkpoint_path),
        "generator_num_inference_steps": args.num_inference_steps,
        "generator_run": run_name,
        "prompt": args.prompt,
        "robot_source_key": clip.robot_source_key,
        "robot_video": rel(robot_path, output_root),
        "seg": clip.seg,
        "slice_policy": "non_overlapping_segment_windows",
        "source_robot_clip_id": clip.source_robot_clip_id,
        "source_robot_task": clip.task,
        "source_segment_path": str(clip.segment_path),
        "syn_video": rel(syn_path, output_root),
        "task": clip.task,
    }


def print_summary(label: str, clips: list) -> None:
    counts: dict[str, int] = {}
    for clip in clips:
        key = f"{clip.task}/{clip.episode}"
        counts[key] = counts.get(key, 0) + 1
    print(f"{label}: {len(clips)} clips")
    for key in sorted(counts):
        print(f"  {key}: {counts[key]}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Cut non-overlapping 1s robot clips from ep000-ep003 and generate "
            "r2h syn videos under output/syn_error_analysis."
        )
    )
    parser.add_argument("--source-task", default="in_task,ood",
                        help="task CSV or groups: in_task, ood, all/training")
    parser.add_argument("--episodes", default=",".join(DEFAULT_EPISODES),
                        help="episode CSV, e.g. ep000,ep001,ep002,ep003 or ep0123")
    parser.add_argument("--duration", default="1s", choices=sorted(DURATION_SECONDS))
    parser.add_argument("--segment-root",
                        default=str(Path(MAIN_ROOT) / "training_data" / "segment"))
    parser.add_argument("--output-root",
                        default=str(Path(OUTPUT_DIR) / "syn_error_analysis"))
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--head", type=int, default=0,
                        help="optional debug cap after task/episode selection")
    parser.add_argument("--list-only", action="store_true",
                        help="print selected clips and exit before slicing/model loading")
    parser.add_argument("--overwrite", action="store_true",
                        help="overwrite existing robot/syn clips instead of reusing complete files")
    parser.add_argument("--compare", action="store_true",
                        help="also write robot|syn side-by-side comparison videos")

    parser.add_argument("--run", default="",
                        help="r2h training_data/log run name or run directory")
    parser.add_argument("--checkpoint", default="latest",
                        help="checkpoint filename under ckpt/, absolute path, or latest")
    parser.add_argument("--no-auto-merge-lora", action="store_true")
    parser.add_argument("--t5-cache-dir", default="",
                        help="default: MAIN_ROOT/training_data/cache/t5/r2h/<duration>")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-inference-steps", type=int, default=30)
    parser.add_argument("--cfg-scale", type=float, default=5.0)
    parser.add_argument("--lora-rank", type=int, default=None)
    parser.add_argument("--lora-target-modules", default=None)
    parser.add_argument("--lora-attn-types", default="self,cross")
    parser.add_argument("--lora-attn-projections", default="q,k,v,o")
    parser.add_argument("--dit-dir", default=DEFAULT_DIT_DIR)
    parser.add_argument("--vae-path", default=DEFAULT_VAE)
    parser.add_argument("--tokenizer-dir", default=DEFAULT_TOKENIZER)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tasks = expand_source_tasks(args.source_task)
    episodes = set(parse_episodes(args.episodes))
    args.segment_root = resolve_main_path(args.segment_root)
    args.output_root = resolve_main_path(args.output_root)
    if args.t5_cache_dir:
        args.t5_cache_dir = str(resolve_main_path(args.t5_cache_dir))
    else:
        args.t5_cache_dir = str(
            Path(MAIN_ROOT) / "training_data" / "cache" / "t5" / "r2h" / args.duration
        )

    clips = collect_segment_clips(
        args.segment_root,
        tasks,
        args.duration,
        clip_stride=None,
        validate_videos=False,
    )
    selected = [clip for clip in clips if clip.episode in episodes]
    if args.head > 0:
        selected = selected[:args.head]
    if not selected:
        raise ValueError(
            f"No clips found under {args.segment_root} for tasks={tasks} "
            f"episodes={sorted(episodes)}"
        )

    print(f"source tasks: {tasks}")
    print(f"episodes: {sorted(episodes)}")
    print(f"duration: {args.duration} (non-overlap stride={DURATION_SECONDS[args.duration]:g}s)")
    print(f"segment root: {args.segment_root}")
    print(f"output root: {args.output_root}")
    print_summary("selected", selected)

    if args.list_only:
        for clip in selected[:20]:
            print(f"  {clip.source_order_index}: {clip.robot_source_key}")
        if len(selected) > 20:
            print(f"  ... {len(selected) - 20} more")
        return
    if not args.run:
        raise ValueError("--run is required unless --list-only is used")

    args.output_root.mkdir(parents=True, exist_ok=True)
    validate_selected_segments(selected)

    expected_frames = FRAMES_4K1[args.duration]
    robot_shapes: dict[Path, tuple[int, int]] = {}
    print("cutting robot clips", flush=True)
    for index, clip in enumerate(selected):
        robot_path, _, _ = clip_output_paths(args.output_root, args.duration, clip)
        robot_complete = False
        if not args.overwrite and robot_path.exists():
            robot_complete = video_complete(robot_path, expected_frames)
        if not robot_complete:
            cut_robot_clip(clip, robot_path, expected_frames)

        robot_frames, robot_shape = read_video_info(robot_path)
        if robot_frames != expected_frames:
            raise ValueError(
                f"Robot clip frame count mismatch for {robot_path}: "
                f"{robot_frames} != {expected_frames}"
            )
        robot_shapes[robot_path] = robot_shape
        status = "reused" if robot_complete else "cut"
        print(f"[slice {index + 1}/{len(selected)}] {status} {clip.robot_source_key}",
              flush=True)

    print("loading r2h generator", flush=True)
    t5_pos, t5_neg = load_t5_cache(args.t5_cache_dir, device="cpu")
    run, model, spec = _load_generator(args)

    manifest_rows: list[dict] = []
    metadata_rows: list[dict] = []
    manifest_path = args.output_root / "manifest.jsonl"
    metadata_path = args.output_root / "metadata.csv"

    for index, clip in enumerate(selected):
        robot_path, syn_path, compare_path = clip_output_paths(
            args.output_root, args.duration, clip)
        robot_shape = robot_shapes[robot_path]

        syn_complete = False
        if not args.overwrite and syn_path.exists():
            syn_complete = video_complete(syn_path, expected_frames, robot_shape)
        if not syn_complete:
            _generate_human_video(
                robot_path,
                syn_path,
                model=model,
                spec=spec,
                prompt=args.prompt,
                t5_pos=t5_pos,
                t5_neg=t5_neg,
                device=args.device,
                num_inference_steps=args.num_inference_steps,
                cfg_scale=args.cfg_scale,
            )
            video_complete(syn_path, expected_frames, robot_shape)

        if args.compare and (args.overwrite or not compare_path.exists()):
            make_compare(robot_path, syn_path, compare_path)

        manifest_rows.append(build_record(
            clip=clip,
            output_root=args.output_root,
            robot_path=robot_path,
            syn_path=syn_path,
            compare_path=compare_path,
            args=args,
            run_name=run.name,
            checkpoint_path=run.checkpoint,
        ))
        metadata_rows.append({
            "video": rel(robot_path, args.output_root),
            "prompt": args.prompt,
            "control_video": rel(syn_path, args.output_root),
        })
        write_jsonl(manifest_path, manifest_rows)
        write_metadata(metadata_path, metadata_rows)

        status = "reused" if syn_complete else "generated"
        print(f"[syn {index + 1}/{len(selected)}] {status} {clip.robot_source_key}",
              flush=True)

    print(f"wrote {len(manifest_rows)} analysis clips -> {args.output_root}")
    print(f"manifest: {manifest_path}")
    print(f"metadata: {metadata_path}")


if __name__ == "__main__":
    main()
