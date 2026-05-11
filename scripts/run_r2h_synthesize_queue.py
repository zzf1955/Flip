#!/usr/bin/env python3
"""Queue r2h_synthesize jobs across a user-provided CUDA list.

The script computes per-task final target counts from one global
``--num-samples`` value, writes the expanded command queue, then runs one
single-task ``src.pipeline.r2h_synthesize`` command per free CUDA device.
Each child is pinned through ``scripts/flip_run.sh r2h_synthesize --cuda GPU``,
so inside the child process the generator device remains ``cuda:0``.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.core.config import DEFAULT_TRAIN_TASKS, MAIN_ROOT, TRAINING_TASKS
from src.pipeline.r2h_synthesize import (
    collect_segment_clips,
    filter_excluded_clips,
    load_covered_robot_sources,
    parse_csv,
    select_clips_proportional,
)
from src.pipeline.runtime_data import short_task_name


DEFAULT_RUNNER = REPO_ROOT / "scripts" / "flip_run.sh"


@dataclass(frozen=True)
class QueueItem:
    task: str
    target_count: int
    cuda: str
    log_path: Path
    command: tuple[str, ...]


def _csv(value: str) -> list[str]:
    return [item.strip() for item in value.replace(";", ",").split(",") if item.strip()]


def _expand_source_tasks(value: str) -> list[str]:
    key = value.strip().lower().replace("-", "_")
    if key in {"in_task", "intask", "train", "default_train"}:
        return [short_task_name(task) for task in DEFAULT_TRAIN_TASKS]
    if key in {"all", "training"}:
        return [short_task_name(task) for task in TRAINING_TASKS]
    tasks = [short_task_name(task) for task in _csv(value)]
    if not tasks:
        raise ValueError("--source-task must not be empty")
    return tasks


def _resolve_main_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return Path(MAIN_ROOT) / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build and run a per-task r2h_synthesize queue across available CUDA ids. "
            "--num-samples is the final global target, not the number added in this run."
        )
    )
    parser.add_argument("--cuda", required=True,
                        help="comma-separated physical CUDA ids, e.g. 0,1,3")
    parser.add_argument("--source-task", default="in_task",
                        help="task CSV, all/training, or in_task for default train tasks")
    parser.add_argument("--duration", default="1s", choices=["1s", "2s", "4s"])
    parser.add_argument("--run", required=True,
                        help="r2h run name or run directory")
    parser.add_argument("--checkpoint", default="latest")
    parser.add_argument("--num-samples", type=int, required=True,
                        help="final global _syn target count across selected tasks")
    parser.add_argument("--segment-root",
                        default=str(Path(MAIN_ROOT) / "training_data" / "segment"))
    parser.add_argument("--output-pair-root",
                        default=str(Path(MAIN_ROOT) / "training_data" / "pair"))
    parser.add_argument("--output-task-suffix", default="{task}_syn")
    parser.add_argument("--clip-stride", type=float, default=None)
    parser.add_argument("--exclude-episodes", default="ep000,ep001,ep002,ep003")
    parser.add_argument("--seedance-covered-manifest", action="append", default=[])
    parser.add_argument("--num-inference-steps", type=int, default=30)
    parser.add_argument("--cfg-scale", type=float, default=5.0)
    parser.add_argument("--t5-cache-dir", default="")
    parser.add_argument("--lora-rank", type=int, default=None)
    parser.add_argument("--lora-target-modules", default="")
    parser.add_argument("--lora-attn-types", default="self,cross")
    parser.add_argument("--lora-attn-projections", default="q,k,v,o")
    parser.add_argument("--dit-dir", default="")
    parser.add_argument("--vae-path", default="")
    parser.add_argument("--tokenizer-dir", default="")
    parser.add_argument("--no-auto-merge-lora", action="store_true")
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--runner", default=str(DEFAULT_RUNNER),
                        help="runner script; default scripts/flip_run.sh")
    parser.add_argument("--log-dir", default="",
                        help="queue log dir; default training_data/log/r2h_synthesize_queue/<timestamp>")
    parser.add_argument("--timestamp", default="")
    parser.add_argument("--execute", action="store_true",
                        help="run the queue; default only writes/prints commands")
    return parser.parse_args()


def _compute_counts(args: argparse.Namespace, tasks: list[str]) -> dict[str, int]:
    segment_root = _resolve_main_path(args.segment_root)
    covered_paths = [_resolve_main_path(path) for path in args.seedance_covered_manifest]
    covered_sources = load_covered_robot_sources(covered_paths) if covered_paths else set()
    clips = collect_segment_clips(
        segment_root,
        tasks,
        args.duration,
        clip_stride=args.clip_stride,
        validate_videos=False,
    )
    eligible = filter_excluded_clips(
        clips,
        set(parse_csv(args.exclude_episodes)),
        covered_sources,
    )
    _, counts = select_clips_proportional(eligible, args.num_samples)
    return counts


def _append_optional(command: list[str], flag: str, value: object) -> None:
    if value is None:
        return
    if isinstance(value, str) and not value:
        return
    command.extend([flag, str(value)])


def _build_queue(args: argparse.Namespace, counts: dict[str, int], log_dir: Path) -> list[QueueItem]:
    cuda_devices = _csv(args.cuda)
    if not cuda_devices:
        raise ValueError("--cuda is empty")
    runner = str(Path(args.runner))
    queue: list[QueueItem] = []
    for index, task in enumerate(sorted(counts)):
        target_count = counts[task]
        if target_count <= 0:
            continue
        cuda = cuda_devices[index % len(cuda_devices)]
        log_path = log_dir / f"{index:03d}_{task}.log"
        command = [
            runner,
            "r2h_synthesize",
            "--cuda",
            cuda,
            "--",
            "--source-task",
            task,
            "--duration",
            args.duration,
            "--run",
            args.run,
            "--checkpoint",
            args.checkpoint,
            "--num-samples",
            str(target_count),
            "--output-task-suffix",
            args.output_task_suffix,
            "--segment-root",
            args.segment_root,
            "--output-pair-root",
            args.output_pair_root,
            "--exclude-episodes",
            args.exclude_episodes,
            "--num-inference-steps",
            str(args.num_inference_steps),
            "--cfg-scale",
            str(args.cfg_scale),
            "--device",
            "cuda:0",
            "--resume-existing",
        ]
        if args.clip_stride is not None:
            command.extend(["--clip-stride", str(args.clip_stride)])
        for path in args.seedance_covered_manifest:
            command.extend(["--seedance-covered-manifest", path])
        _append_optional(command, "--t5-cache-dir", args.t5_cache_dir)
        _append_optional(command, "--lora-rank", args.lora_rank)
        _append_optional(command, "--lora-target-modules", args.lora_target_modules)
        _append_optional(command, "--lora-attn-types", args.lora_attn_types)
        _append_optional(command, "--lora-attn-projections", args.lora_attn_projections)
        _append_optional(command, "--dit-dir", args.dit_dir)
        _append_optional(command, "--vae-path", args.vae_path)
        _append_optional(command, "--tokenizer-dir", args.tokenizer_dir)
        if args.no_auto_merge_lora:
            command.append("--no-auto-merge-lora")
        if args.compare:
            command.append("--compare")
        queue.append(QueueItem(
            task=task,
            target_count=target_count,
            cuda=cuda,
            log_path=log_path,
            command=tuple(command),
        ))
    return queue


def _write_queue_files(queue: list[QueueItem], log_dir: Path, args: argparse.Namespace) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = log_dir / "queue.jsonl"
    with manifest_path.open("w") as fh:
        for item in queue:
            fh.write(json.dumps({
                "task": item.task,
                "target_count": item.target_count,
                "cuda_initial": item.cuda,
                "log_path": str(item.log_path),
                "command": list(item.command),
            }, sort_keys=True) + "\n")
    commands_path = log_dir / "commands.sh"
    with commands_path.open("w") as fh:
        fh.write("#!/usr/bin/env bash\nset -euo pipefail\n\n")
        for item in queue:
            fh.write(f"# task={item.task} target={item.target_count} initial_cuda={item.cuda}\n")
            fh.write(shlex.join(item.command) + f" 2>&1 | tee -a {shlex.quote(str(item.log_path))}\n\n")
    commands_path.chmod(0o755)
    config_path = log_dir / "config.json"
    config_path.write_text(json.dumps({
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "cuda": args.cuda,
        "source_task": args.source_task,
        "duration": args.duration,
        "run": args.run,
        "checkpoint": args.checkpoint,
        "num_samples": args.num_samples,
        "output_task_suffix": args.output_task_suffix,
        "queue_size": len(queue),
    }, indent=2, sort_keys=True) + "\n")


def _run_queue(queue: list[QueueItem], cuda_devices: list[str]) -> int:
    pending = list(queue)
    running: dict[subprocess.Popen, tuple[QueueItem, object]] = {}
    free = list(cuda_devices)
    failures: list[tuple[QueueItem, int]] = []
    while pending or running:
        while pending and free:
            cuda = free.pop(0)
            item = pending.pop(0)
            command = list(item.command)
            command[3] = cuda
            item = QueueItem(item.task, item.target_count, cuda, item.log_path, tuple(command))
            item.log_path.parent.mkdir(parents=True, exist_ok=True)
            log_fh = item.log_path.open("a")
            log_fh.write(f"\n[{datetime.now().strftime('%F %T')}] START cuda={cuda} task={item.task} target={item.target_count}\n")
            log_fh.write("$ " + shlex.join(command) + "\n\n")
            log_fh.flush()
            print(
                f"[{datetime.now().strftime('%F %T')}] START cuda={cuda} "
                f"task={item.task} target={item.target_count} log={item.log_path}",
                flush=True,
            )
            proc = subprocess.Popen(
                command,
                cwd=REPO_ROOT,
                stdout=log_fh,
                stderr=subprocess.STDOUT,
            )
            running[proc] = (item, log_fh)
        time.sleep(5)
        for proc in list(running):
            ret = proc.poll()
            if ret is None:
                continue
            item, log_fh = running.pop(proc)
            log_fh.write(f"\n[{datetime.now().strftime('%F %T')}] END exit={ret}\n")
            log_fh.close()
            free.append(item.cuda)
            status = "DONE" if ret == 0 else "FAIL"
            print(
                f"[{datetime.now().strftime('%F %T')}] {status} cuda={item.cuda} "
                f"task={item.task} exit={ret}",
                flush=True,
            )
            if ret != 0:
                failures.append((item, ret))
    if failures:
        print("Failed queue item(s):", flush=True)
        for item, ret in failures:
            print(f"  task={item.task} cuda={item.cuda} exit={ret} log={item.log_path}", flush=True)
        return failures[0][1]
    return 0


def main() -> None:
    args = parse_args()
    try:
        tasks = _expand_source_tasks(args.source_task)
        counts = _compute_counts(args, tasks)
        timestamp = args.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = Path(args.log_dir) if args.log_dir else (
            Path(MAIN_ROOT) / "training_data" / "log" / "r2h_synthesize_queue" / timestamp
        )
        if not log_dir.is_absolute():
            log_dir = Path(MAIN_ROOT) / log_dir
        queue = _build_queue(args, counts, log_dir)
        _write_queue_files(queue, log_dir, args)
    except (FileNotFoundError, ValueError) as exc:
        print(f"run_r2h_synthesize_queue.py: {exc}", flush=True)
        raise SystemExit(2) from None

    print(f"Expanded {len(queue)} r2h_synthesize queue item(s).", flush=True)
    print(f"Queue dir: {log_dir}", flush=True)
    for item in queue:
        print(
            f"  task={item.task} target={item.target_count} "
            f"initial_cuda={item.cuda} log={item.log_path}",
            flush=True,
        )
        print("    " + shlex.join(item.command), flush=True)
    if not args.execute:
        print("Dry run only. Add --execute to launch the queue.", flush=True)
        return
    raise SystemExit(_run_queue(queue, _csv(args.cuda)))


if __name__ == "__main__":
    main()
