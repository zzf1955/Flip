"""Generate h2r synthetic-human pairs from a trained r2h Mitty model.

The source robot clips are enumerated directly from
training_data/segment/<task>/<episode>/seg*_video.mp4.  Existing Seedance or
original h2r manifests are used only as a coverage/exclusion source so the same
robot clip is not represented by both Seedance human and self-generated human.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from src.core.config import MAIN_ROOT, TRAINING_DATA_ROOT, TRAINING_TASKS


FFMPEG = os.environ.get(
    "FFMPEG_BIN",
    "/home/leadtek/miniconda3/envs/flip/bin/ffmpeg",
)

PROMPT = "A first-person view robot arm performing household tasks flip_v2v"
TARGET_FPS = 16
SEGMENT_SECONDS = 4.0
DURATION_SECONDS = {"1s": 1.0, "2s": 2.0, "4s": 4.0}
FRAMES_4K1 = {"1s": 17, "2s": 33, "4s": 65}
DEFAULT_EXCLUDE_EPISODES = ("ep000", "ep001", "ep002", "ep003")
ALLOCATION_MODES = ("global_head", "proportional")

_MANUAL_DIR = os.path.join(
    "/disk_n/zzf/.cache/huggingface/hub",
    "models--Wan-AI--Wan2.2-TI2V-5B", "manual",
)
DEFAULT_DIT_DIR = _MANUAL_DIR
DEFAULT_VAE = os.path.join(_MANUAL_DIR, "Wan2.2_VAE.pth")
DEFAULT_TOKENIZER = os.path.join(_MANUAL_DIR, "google", "umt5-xxl")


@dataclass(frozen=True)
class RobotClip:
    task: str
    episode: str
    seg: str
    segment_path: Path
    clip_index: int
    clip_start: float
    clip_dur: float
    source_order_index: int
    robot_source_key: str
    source_robot_clip_id: str


def _short_task_name(task: str) -> str:
    return task.strip().replace("G1_WBT_", "")


def expand_task_spec(task_spec: str) -> list[str]:
    groups = {
        "all": [_short_task_name(task) for task in TRAINING_TASKS],
        "training": [_short_task_name(task) for task in TRAINING_TASKS],
    }
    key = task_spec.lower()
    if key in groups:
        return groups[key]
    tasks = [_short_task_name(item) for item in task_spec.split(",") if item.strip()]
    if not tasks:
        raise ValueError("--source-task must not be empty")
    return tasks


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def robot_source_key(
    task: str,
    episode: str,
    seg: str,
    clip_start: float,
    clip_dur: float,
) -> str:
    return (
        f"{task}/{episode}/{seg}"
        f"_start{clip_start:.3f}_dur{clip_dur:.3f}"
    )


def source_robot_clip_id(
    task: str,
    episode: str,
    seg: str,
    clip_index: int,
) -> str:
    return f"{task}/{episode}/{seg}_clip{clip_index:02d}"


def _segment_clip_starts(duration: str, clip_stride: float | None) -> list[float]:
    dur = DURATION_SECONDS[duration]
    stride = dur if clip_stride is None else clip_stride
    if stride <= 0.0:
        raise ValueError(f"--clip-stride must be positive, got {stride}")
    starts = []
    start = 0.0
    while start + dur <= SEGMENT_SECONDS + 1e-6:
        starts.append(round(start, 6))
        start += stride
    if not starts:
        raise ValueError(
            f"duration={duration} with stride={stride:g} yields no clips "
            f"inside {SEGMENT_SECONDS:g}s segments"
        )
    return starts


def _validate_segment_video(path: Path) -> None:
    read_video_info(path)


def collect_segment_clips(
    segment_root: Path,
    tasks: list[str],
    duration: str,
    *,
    clip_stride: float | None = None,
    validate_videos: bool = True,
) -> list[RobotClip]:
    if duration not in DURATION_SECONDS:
        raise ValueError(f"Unsupported duration: {duration}")
    starts = _segment_clip_starts(duration, clip_stride)
    dur = DURATION_SECONDS[duration]
    clips: list[RobotClip] = []

    for task in sorted(tasks):
        task_dir = segment_root / task
        if not task_dir.is_dir():
            raise FileNotFoundError(f"Segment task directory not found: {task_dir}")
        for ep_dir in sorted(path for path in task_dir.iterdir() if path.is_dir()):
            for segment_path in sorted(ep_dir.glob("seg*_video.mp4")):
                match = re.match(r"(seg\d+)_video\.mp4$", segment_path.name)
                if not match:
                    continue
                if validate_videos:
                    _validate_segment_video(segment_path)
                seg = match.group(1)
                for clip_index, start in enumerate(starts):
                    order_index = len(clips)
                    key = robot_source_key(task, ep_dir.name, seg, start, dur)
                    clips.append(RobotClip(
                        task=task,
                        episode=ep_dir.name,
                        seg=seg,
                        segment_path=segment_path,
                        clip_index=clip_index,
                        clip_start=start,
                        clip_dur=dur,
                        source_order_index=order_index,
                        robot_source_key=key,
                        source_robot_clip_id=source_robot_clip_id(
                            task, ep_dir.name, seg, clip_index),
                    ))

    if not clips:
        raise ValueError(f"No segment clips found under {segment_root} for {tasks}")
    return clips


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"JSONL not found: {path}")
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
    return rows


def _record_robot_source_key(record: dict) -> str:
    existing = record.get("robot_source_key")
    if existing:
        return str(existing)
    task = record.get("source_robot_task") or record.get("task") or record.get("robot_task")
    episode = record.get("episode")
    seg = record.get("seg")
    clip_dur = record.get("clip_dur") or record.get("duration_seconds")
    clip_start = record.get("clip_start")
    if task and episode and seg and clip_start is not None and clip_dur is not None:
        return robot_source_key(
            _short_task_name(str(task)),
            str(episode),
            str(seg),
            float(clip_start),
            float(clip_dur),
        )
    source_id = record.get("source_id")
    if source_id:
        return str(source_id)
    source_segment_id = record.get("source_segment_id")
    if source_segment_id:
        return str(source_segment_id)
    raise ValueError(f"Cannot derive robot_source_key from manifest record: {record}")


def load_covered_robot_sources(paths: list[Path]) -> set[str]:
    covered: set[str] = set()
    for path in paths:
        for record in _read_jsonl(path):
            covered.add(_record_robot_source_key(record))
    return covered


def filter_excluded_clips(
    clips: list[RobotClip],
    exclude_episodes: set[str],
    covered_sources: set[str],
) -> list[RobotClip]:
    eligible = [
        clip for clip in clips
        if clip.episode not in exclude_episodes
        and clip.robot_source_key not in covered_sources
        and clip.source_robot_clip_id not in covered_sources
    ]
    if not eligible:
        raise ValueError("No eligible clips remain after Seedance/source exclusion")
    return eligible


def _parse_range(value: str, total: int) -> tuple[int, int]:
    match = re.match(r"^(\d*):(\d*)$", value)
    if not match:
        raise ValueError(f"--range must use START:END syntax, got {value!r}")
    start = int(match.group(1)) if match.group(1) else 0
    end = int(match.group(2)) if match.group(2) else total
    if start < 0 or end < start or end > total:
        raise ValueError(f"Invalid --range {value!r} for {total} clips")
    return start, end


def select_clips(
    clips: list[RobotClip],
    *,
    num_samples: int = 0,
    head: int = 0,
    tail: int = 0,
    range_spec: str = "",
) -> list[RobotClip]:
    modes = sum(1 for enabled in (num_samples > 0, head > 0, tail > 0, bool(range_spec)) if enabled)
    if modes > 1:
        raise ValueError("Use only one of --num-samples, --head, --tail, or --range")
    if num_samples > 0:
        if num_samples > len(clips):
            raise ValueError(f"Requested {num_samples} clips, only {len(clips)} available")
        return clips[:num_samples]
    if head > 0:
        if head > len(clips):
            raise ValueError(f"Requested head {head} clips, only {len(clips)} available")
        return clips[:head]
    if tail > 0:
        if tail > len(clips):
            raise ValueError(f"Requested tail {tail} clips, only {len(clips)} available")
        return clips[-tail:]
    if range_spec:
        start, end = _parse_range(range_spec, len(clips))
        return clips[start:end]
    return list(clips)


def _allocate_counts(capacity_by_task: dict[str, int], size: int) -> dict[str, int]:
    total = sum(capacity_by_task.values())
    if size <= 0:
        raise ValueError(f"proportional allocation requires a positive size, got {size}")
    if total == 0:
        raise ValueError("No clips available for proportional allocation")
    if size > total:
        raise ValueError(
            f"Requested {size} clips, only {total} clips are available")

    raw = {
        task: size * capacity / total
        for task, capacity in capacity_by_task.items()
    }
    counts = {task: int(value) for task, value in raw.items()}
    remaining = size - sum(counts.values())
    remainders = sorted(
        capacity_by_task,
        key=lambda task: (raw[task] - counts[task], capacity_by_task[task], task),
        reverse=True,
    )
    for task in remainders:
        if remaining == 0:
            break
        if counts[task] < capacity_by_task[task]:
            counts[task] += 1
            remaining -= 1
    if remaining:
        raise RuntimeError(f"Unable to allocate {remaining} clips")
    return counts


def select_clips_proportional(
    clips: list[RobotClip],
    num_samples: int,
) -> tuple[list[RobotClip], dict[str, int]]:
    grouped: dict[str, list[RobotClip]] = {}
    for clip in clips:
        grouped.setdefault(clip.task, []).append(clip)
    grouped = {
        task: sorted(task_clips, key=lambda clip: clip.source_order_index)
        for task, task_clips in sorted(grouped.items())
    }
    counts = _allocate_counts(
        {task: len(task_clips) for task, task_clips in grouped.items()},
        num_samples,
    )
    selected = [
        clip
        for task, task_clips in grouped.items()
        for clip in task_clips[:counts[task]]
    ]
    selected.sort(key=lambda clip: clip.source_order_index)
    return selected, counts


def select_clips_for_args(
    clips: list[RobotClip],
    args,
) -> tuple[list[RobotClip], dict[str, int]]:
    if args.allocate_by_task == "proportional":
        if args.head > 0 or args.tail > 0 or args.range_spec:
            raise ValueError(
                "--allocate-by-task proportional only supports --num-samples")
        selected, counts = select_clips_proportional(clips, args.num_samples)
        return selected, counts

    selected = select_clips(
        clips,
        num_samples=args.num_samples,
        head=args.head,
        tail=args.tail,
        range_spec=args.range_spec,
    )
    counts: dict[str, int] = {}
    for clip in selected:
        counts[clip.task] = counts.get(clip.task, 0) + 1
    return selected, dict(sorted(counts.items()))


def read_video_info(path: Path) -> tuple[int, tuple[int, int]]:
    if not path.is_file():
        raise FileNotFoundError(f"Video not found: {path}")
    import av

    container = av.open(str(path))
    frames = 0
    shape = None
    for frame in container.decode(video=0):
        arr = frame.to_ndarray(format="rgb24")
        frames += 1
        current_shape = arr.shape[:2]
        if shape is None:
            shape = current_shape
        elif shape != current_shape:
            raise ValueError(f"Frame shape changes inside video: {path}")
    container.close()
    if frames == 0 or shape is None:
        raise ValueError(f"Video has no frames: {path}")
    return frames, shape


def video_complete(
    path: Path,
    expected_frames: int,
    expected_shape: tuple[int, int] | None = None,
) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    frames, shape = read_video_info(path)
    if frames != expected_frames:
        raise ValueError(
            f"Unexpected frame count for {path}: {frames} != {expected_frames}"
        )
    if expected_shape is not None and shape != expected_shape:
        raise ValueError(f"Unexpected video shape for {path}: {shape} != {expected_shape}")
    return True


def _ffmpeg(args: list[str]) -> None:
    subprocess.check_call(
        [FFMPEG, "-y"] + args,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def cut_robot_clip(clip: RobotClip, out_path: Path, expected_frames: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _ffmpeg([
        "-ss", f"{clip.clip_start:.3f}",
        "-i", str(clip.segment_path),
        "-t", f"{clip.clip_dur + 0.5:.3f}",
        "-r", str(TARGET_FPS),
        "-frames:v", str(expected_frames),
        "-c:v", "libx264",
        "-crf", "18",
        "-preset", "fast",
        "-an",
        str(out_path),
    ])


def make_compare(robot_path: Path, human_path: Path, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _ffmpeg([
        "-i", str(robot_path),
        "-i", str(human_path),
        "-filter_complex", "[0:v][1:v]hstack=inputs=2",
        "-c:v", "libx264",
        "-crf", "18",
        "-preset", "fast",
        str(out_path),
    ])


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")


def _write_metadata(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["video", "prompt", "control_video"])
        writer.writeheader()
        writer.writerows(rows)


def build_manifest_record(
    clip: RobotClip,
    pair_id: str,
    syn_task: str,
    rel_video: str,
    rel_control: str,
    args,
    run_name: str,
    checkpoint_path: Path,
) -> dict:
    source_id = f"{clip.robot_source_key}_syn"
    return {
        "augment": "normal",
        "clip_dur": clip.clip_dur,
        "clip_idx": clip.clip_index,
        "clip_start": clip.clip_start,
        "control_video": rel_control,
        "data_type": "h2r",
        "duration": args.duration,
        "episode": clip.episode,
        "generator_cfg_scale": args.cfg_scale,
        "generator_checkpoint": str(checkpoint_path),
        "generator_num_inference_steps": args.num_inference_steps,
        "generator_run": run_name,
        "input_role": "human",
        "pair_id": pair_id,
        "robot_source_key": clip.robot_source_key,
        "robot_task": syn_task,
        "seg": clip.seg,
        "seedance_exclusion_checked": True,
        "source_clip_dur": clip.clip_dur,
        "source_clip_index": clip.clip_index,
        "source_clip_start": clip.clip_start,
        "source_id": source_id,
        "source_order_index": clip.source_order_index,
        "source_robot_clip_id": clip.source_robot_clip_id,
        "source_robot_task": clip.task,
        "source_segment_id": f"{clip.task}/{clip.episode}/{clip.seg}",
        "source_segment_path": str(clip.segment_path),
        "synthetic_source": "mitty_r2h",
        "target_role": "robot",
        "task": syn_task,
        "video": rel_video,
        "window_idx": clip.clip_index,
    }


def build_pair_order_entry(record: dict, order_index: int) -> dict:
    return {
        "control_video": record["control_video"],
        "data_type": record["data_type"],
        "duration": record["duration"],
        "order_index": order_index,
        "pair_id": record["pair_id"],
        "robot_source_key": record["robot_source_key"],
        "robot_task": record["robot_task"],
        "source_id": record["source_id"],
        "source_order_index": record["source_order_index"],
        "source_robot_task": record["source_robot_task"],
        "source_segment_id": record["source_segment_id"],
        "video": record["video"],
    }


def _resolve_run_spec(run_arg: str, checkpoint: str, auto_merge_lora: bool):
    from src.pipeline.eval_mitty.run_specs import (
        RunSpec, find_latest_checkpoint, read_train_args,
    )

    raw = Path(run_arg)
    candidates = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.extend([
            Path(MAIN_ROOT) / raw,
            Path(MAIN_ROOT) / "training_data" / "log" / raw,
            Path(TRAINING_DATA_ROOT) / "log" / raw,
        ])
    run_dir = next((path for path in candidates if path.exists()), None)
    if run_dir is None:
        tried = ", ".join(str(path) for path in candidates)
        raise FileNotFoundError(f"Run directory not found: {run_arg}; tried {tried}")

    ckpt_raw = Path(checkpoint)
    if checkpoint == "latest":
        ckpt = find_latest_checkpoint(run_dir)
    elif ckpt_raw.is_absolute():
        ckpt = ckpt_raw
    else:
        ckpt = run_dir / "ckpt" / checkpoint
    if not ckpt.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    train_args = read_train_args(run_dir)
    merge_paths = ()
    if auto_merge_lora and train_args.get("merge_lora"):
        merge_paths = tuple(_resolve_main_path(path) for path in train_args["merge_lora"])
        for path in merge_paths:
            if not path.is_file():
                raise FileNotFoundError(f"Merged LoRA not found: {path}")
    return RunSpec(
        name=run_dir.name,
        run_dir=run_dir,
        checkpoint=ckpt,
        merge_lora_paths=merge_paths,
        merge_lora_rank=int(train_args.get("merge_lora_rank", 96)),
    )


def _resolve_main_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return Path(MAIN_ROOT) / path


def _load_generator(args):
    from src.pipeline.eval_mitty.generation import load_model

    run = _resolve_run_spec(
        args.run, args.checkpoint, auto_merge_lora=not args.no_auto_merge_lora)
    if run.merge_lora_paths:
        print(
            "Replaying merged LoRA: "
            + ", ".join(str(path) for path in run.merge_lora_paths),
            flush=True,
        )
    model, spec = load_model(
        run=run,
        device=args.device,
        lora_rank=args.lora_rank,
        lora_target_modules=args.lora_target_modules,
        lora_attn_types=args.lora_attn_types,
        lora_attn_projections=args.lora_attn_projections,
        dit_dir=args.dit_dir,
        vae_path=args.vae_path,
        tokenizer_dir=args.tokenizer_dir,
    )
    return run, model, spec


def _generate_human_video(
    robot_path: Path,
    human_path: Path,
    *,
    model,
    spec,
    prompt: str,
    t5_pos,
    t5_neg,
    device: str,
    num_inference_steps: int,
    cfg_scale: float,
) -> None:
    import torch
    from diffsynth.diffusion.flow_match import FlowMatchScheduler

    from src.core.train_utils import save_video, tensor_to_frames
    from src.pipeline.mitty_cache import (
        encode_video_array_batch, load_video_as_rgb_array,
    )

    if prompt not in t5_pos:
        raise KeyError(f"Prompt not found in T5 cache: {prompt}")
    if t5_neg is None:
        raise FileNotFoundError("negative T5 cache is required for generation")

    robot_video = load_video_as_rgb_array(str(robot_path))
    pipe = model.pipe
    pipe.load_models_to_device(["vae"])
    robot_latent = encode_video_array_batch(pipe.vae, [robot_video], device).cpu()
    sample = {
        "human_latent": robot_latent,
        "robot_latent": torch.zeros_like(robot_latent),
        "context_posi": t5_pos[prompt],
        "context_nega": t5_neg,
        "prompt": prompt,
    }

    sched = FlowMatchScheduler("Wan")
    sched.set_timesteps(
        num_inference_steps=num_inference_steps,
        denoising_strength=1.0,
        shift=5.0,
    )
    with torch.no_grad():
        denoised = spec.eval_denoise_fn(
            pipe=pipe,
            sample=sample,
            sched=sched,
            device=device,
            cfg_scale=cfg_scale,
            num_inference_steps=num_inference_steps,
        )
        pipe.load_models_to_device(["vae"])
        generated = pipe.vae.decode(denoised, device=device, tiled=False)
    human_path.parent.mkdir(parents=True, exist_ok=True)
    save_video(tensor_to_frames(generated), str(human_path), fps=TARGET_FPS)


def _process_task_group(
    task: str,
    clips: list[RobotClip],
    args,
    *,
    run,
    model,
    spec,
    t5_pos,
    t5_neg,
) -> None:
    syn_task = args.output_task_suffix.format(task=task)
    task_dir = args.output_pair_root / "h2r" / args.duration / syn_task
    video_dir = task_dir / "video"
    control_dir = task_dir / "control_video"
    compare_dir = task_dir / "compare"
    expected_frames = FRAMES_4K1[args.duration]
    metadata_rows: list[dict] = []
    manifest_rows: list[dict] = []

    for pair_index, clip in enumerate(clips):
        pair_id = f"pair_{pair_index:04d}"
        rel_video = f"video/{pair_id}.mp4"
        rel_control = f"control_video/{pair_id}.mp4"
        robot_path = video_dir / f"{pair_id}.mp4"
        human_path = control_dir / f"{pair_id}.mp4"
        compare_path = compare_dir / f"{pair_id}.mp4"

        skip = False
        if args.resume_existing and robot_path.exists() and human_path.exists():
            robot_frames, robot_shape = read_video_info(robot_path)
            human_frames, human_shape = read_video_info(human_path)
            if (
                robot_frames == expected_frames
                and human_frames == expected_frames
                and robot_shape == human_shape
            ):
                skip = True

        if not skip:
            cut_robot_clip(clip, robot_path, expected_frames)
            robot_frames, robot_shape = read_video_info(robot_path)
            if robot_frames != expected_frames:
                raise ValueError(
                    f"Robot clip frame count mismatch for {robot_path}: "
                    f"{robot_frames} != {expected_frames}"
                )
            _generate_human_video(
                robot_path,
                human_path,
                model=model,
                spec=spec,
                prompt=args.prompt,
                t5_pos=t5_pos,
                t5_neg=t5_neg,
                device=args.device,
                num_inference_steps=args.num_inference_steps,
                cfg_scale=args.cfg_scale,
            )
            video_complete(human_path, expected_frames, robot_shape)
            if args.compare:
                make_compare(robot_path, human_path, compare_path)
        elif args.compare and not compare_path.exists():
            make_compare(robot_path, human_path, compare_path)

        metadata_rows.append({
            "video": rel_video,
            "prompt": args.prompt,
            "control_video": rel_control,
        })
        manifest_rows.append(build_manifest_record(
            clip=clip,
            pair_id=pair_id,
            syn_task=syn_task,
            rel_video=rel_video,
            rel_control=rel_control,
            args=args,
            run_name=run.name,
            checkpoint_path=run.checkpoint,
        ))
        if (pair_index + 1) % 10 == 0 or pair_index == len(clips) - 1:
            print(
                f"[{syn_task}] {pair_index + 1}/{len(clips)} "
                f"{pair_id} {'skip' if skip else 'generated'}",
                flush=True,
            )

    overlap = {row["robot_source_key"] for row in manifest_rows} & args.covered_sources
    if overlap:
        raise ValueError(f"Syn output overlaps covered robot sources: {sorted(overlap)[:5]}")

    _write_metadata(task_dir / "metadata.csv", metadata_rows)
    _write_jsonl(task_dir / "manifest.jsonl", manifest_rows)
    order_rows = [
        build_pair_order_entry(record, order_index)
        for order_index, record in enumerate(manifest_rows)
    ]
    _write_jsonl(task_dir / "pair_order.jsonl", order_rows)
    print(f"[{syn_task}] wrote {len(manifest_rows)} pairs -> {task_dir}", flush=True)


def _group_by_task(clips: list[RobotClip]) -> dict[str, list[RobotClip]]:
    grouped: dict[str, list[RobotClip]] = {}
    for clip in clips:
        grouped.setdefault(clip.task, []).append(clip)
    return dict(sorted(grouped.items()))


def _print_clip_summary(label: str, clips: list[RobotClip]) -> None:
    counts: dict[str, int] = {}
    for clip in clips:
        counts[clip.task] = counts.get(clip.task, 0) + 1
    print(f"{label}: {len(clips)} clips {json.dumps(counts, sort_keys=True)}")


def validate_selected_segments(clips: list[RobotClip]) -> None:
    seen: set[Path] = set()
    for clip in clips:
        if clip.segment_path in seen:
            continue
        _validate_segment_video(clip.segment_path)
        seen.add(clip.segment_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate h2r _syn pairs from training_data/segment via r2h Mitty"
    )
    parser.add_argument("--source-task", required=True,
                        help="source task short name, comma list, or all/training")
    parser.add_argument("--duration", default="1s", choices=sorted(DURATION_SECONDS))
    parser.add_argument("--segment-root",
                        default=str(Path(MAIN_ROOT) / "training_data" / "segment"))
    parser.add_argument("--output-pair-root",
                        default=str(Path(TRAINING_DATA_ROOT) / "pair"))
    parser.add_argument("--output-task-suffix", default="{task}_syn",
                        help="format string for output task, receives {task}")
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--clip-stride", type=float, default=None,
                        help="seconds between source windows; default equals duration")
    parser.add_argument("--num-samples", type=int, default=0)
    parser.add_argument("--head", type=int, default=0)
    parser.add_argument("--tail", type=int, default=0)
    parser.add_argument("--range", dest="range_spec", default="",
                        help="select eligible source clips by START:END after filtering")
    parser.add_argument("--allocate-by-task", choices=ALLOCATION_MODES,
                        default="global_head",
                        help="global_head keeps legacy global ordering; "
                             "proportional splits --num-samples by per-task "
                             "eligible clip counts")
    parser.add_argument("--exclude-episodes",
                        default=",".join(DEFAULT_EXCLUDE_EPISODES),
                        help="comma-separated episodes to skip by default")
    parser.add_argument("--seedance-covered-manifest", action="append", default=[],
                        help="manifest.jsonl with already covered robot sources; repeatable")
    parser.add_argument("--resume-existing", action="store_true")
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--list-only", action="store_true",
                        help="print selected clips and exit before model loading")

    parser.add_argument("--run", default="",
                        help="r2h training_data/log run name or run directory")
    parser.add_argument("--checkpoint", default="latest",
                        help="checkpoint filename under ckpt/, absolute path, or latest")
    parser.add_argument("--no-auto-merge-lora", action="store_true",
                        help="do not replay merge_lora paths recorded in train.log")
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
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    args.segment_root = _resolve_main_path(args.segment_root)
    args.output_pair_root = _resolve_main_path(args.output_pair_root)
    if args.t5_cache_dir:
        args.t5_cache_dir = str(_resolve_main_path(args.t5_cache_dir))
    else:
        args.t5_cache_dir = str(
            Path(MAIN_ROOT) / "training_data" / "cache" / "t5" / "r2h" / args.duration
        )

    tasks = expand_task_spec(args.source_task)
    exclude_episodes = set(parse_csv(args.exclude_episodes))
    covered_paths = [_resolve_main_path(path) for path in args.seedance_covered_manifest]
    covered_sources = load_covered_robot_sources(covered_paths) if covered_paths else set()
    args.covered_sources = covered_sources

    clips = collect_segment_clips(
        args.segment_root,
        tasks,
        args.duration,
        clip_stride=args.clip_stride,
        validate_videos=False,
    )
    eligible = filter_excluded_clips(clips, exclude_episodes, covered_sources)
    try:
        selected, selected_counts = select_clips_for_args(eligible, args)
    except ValueError as exc:
        parser.error(str(exc))

    print(f"r2h synth source tasks: {tasks}")
    print(f"segment root: {args.segment_root}")
    print(f"output pair root: {args.output_pair_root}")
    print(f"exclude episodes: {sorted(exclude_episodes)}")
    print(f"covered manifest sources: {len(covered_sources)}")
    print(f"allocation mode: {args.allocate_by_task}")
    _print_clip_summary("all segment clips", clips)
    _print_clip_summary("eligible clips", eligible)
    _print_clip_summary("selected clips", selected)
    print(f"selected allocation: {json.dumps(selected_counts, sort_keys=True)}")

    if args.list_only:
        for clip in selected[:20]:
            print(f"  {clip.source_order_index}: {clip.robot_source_key}")
        if len(selected) > 20:
            print(f"  ... {len(selected) - 20} more")
        return
    if not args.run:
        parser.error("--run is required unless --list-only is used")
    validate_selected_segments(selected)

    from src.core.train_utils import load_t5_cache

    t5_pos, t5_neg = load_t5_cache(args.t5_cache_dir, device="cpu")
    run, model, spec = _load_generator(args)
    for task, task_clips in _group_by_task(selected).items():
        _process_task_group(
            task,
            task_clips,
            args,
            run=run,
            model=model,
            spec=spec,
            t5_pos=t5_pos,
            t5_neg=t5_neg,
        )


if __name__ == "__main__":
    main()
