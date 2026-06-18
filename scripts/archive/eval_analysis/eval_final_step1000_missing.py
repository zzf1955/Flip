#!/usr/bin/env python3
"""Run offline metrics for completed final* step-1000 checkpoints.

The script discovers training_data/log/final* runs, skips r2h by default, keeps
only runs that have finished 1000 training steps and still lack a complete
offline summary under <run>/full_eval/, then invokes scripts/flip_run.sh
eval_mitty sequentially with the legacy 80 in-task + 42 OOD eval sizes, Local
metrics, and Patch FID enabled.
"""

from __future__ import annotations

import argparse
import ast
import csv
import os
import queue
import shlex
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from src.pipeline.train_config import TRAIN_TASKS

DEFAULT_LOG_ROOT = REPO_ROOT / "training_data" / "log"
DEFAULT_CHECKPOINT = "step-1000.safetensors"
DEFAULT_CUDA = "2"
DEFAULT_IN_TASK_EVAL_SIZE = 80
DEFAULT_OOD_EVAL_SIZE = 42
DEFAULT_OUTPUT_SUBDIR = "full_eval"
SKIP_OURS_STAGE_PREFIXES = (
    "final_ours_step1",
    "final_ours_step2",
)
RUNNER_SCRIPTS = {
    "flip_run": "flip_run.sh",
    "flip_run_2": "flip_run_2.sh",
}
COMPLETED_STEP_MARKER = "step=1000/1000"
REQUIRED_METRIC_COLUMNS = (
    "n_samples",
    "mse",
    "psnr",
    "ssim",
    "lpips",
    "fid",
    "fvd",
)
REGION_METRIC_COLUMNS = (
    "foreground_mse",
    "foreground_psnr",
    "foreground_ssim",
    "background_mse",
    "background_psnr",
    "background_ssim",
    "foreground_local_fid",
    "foreground_local_fvd",
)
PATCH_METRIC_COLUMNS = (
    "foreground_patch_fid",
    "foreground_patch_count",
    "foreground_patch_size",
    "foreground_patch_stride",
    "foreground_patch_coverage_threshold",
    "foreground_patch_min_mask_pixels",
    "foreground_patch_max_per_frame",
    "foreground_patch_max_per_video",
)
TASK_NAME_BY_DATA = {
    "h2r": "h2r_1s",
    "r2h": "r2h_1s",
    "blur_r2r": "blur_r2r_1s",
    "identity_r2r": "identity_r2r_1s",
}


@dataclass(frozen=True)
class EvalTarget:
    run_dir: Path
    train_args: dict
    splits: tuple[str, ...]
    output_dir: Path
    log_path: Path


@dataclass(frozen=True)
class SkippedRun:
    name: str
    reason: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Discover completed final* step-1000 checkpoints missing offline "
            "metrics and run src.pipeline.evaluate_mitty_models for them."
        )
    )
    parser.add_argument("--log-root", default=str(DEFAULT_LOG_ROOT))
    parser.add_argument("--prefix", default="final")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--cuda", default=os.environ.get("CUDA_ID", DEFAULT_CUDA))
    parser.add_argument("--cuda-list", default="",
                        help="comma-separated CUDA ids to poll for idle eval slots, "
                             "e.g. 0,2,3; defaults to the single --cuda value")
    parser.add_argument("--poll-interval", type=float, default=30.0,
                        help="seconds between CUDA idle checks when --execute is used")
    parser.add_argument("--runner", choices=sorted(RUNNER_SCRIPTS), default="flip_run",
                        help="GPU launcher script to use")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--in-task-eval-size", type=int, default=DEFAULT_IN_TASK_EVAL_SIZE)
    parser.add_argument("--ood-eval-size", type=int, default=DEFAULT_OOD_EVAL_SIZE)
    parser.add_argument("--output-subdir", default=DEFAULT_OUTPUT_SUBDIR,
                        help="per-run directory name for videos, summaries, and eval.log")
    parser.add_argument("--force", action="store_true",
                        help="evaluate matching runs even if summary files look complete")
    parser.add_argument("--include-r2h", action="store_true",
                        help="include robot-to-human runs; skipped by default")
    parser.add_argument("--include-ours-step1-step2", action="store_true",
                        help="include final_ours_step1* and final_ours_step2* runs")
    parser.add_argument("--no-local-metrics", action="store_true",
                        help="do not force mask-region Local metrics for non-r2h runs")
    parser.add_argument("--no-patch-fid", action="store_true",
                        help="do not force Patch FID for non-r2h runs")
    parser.add_argument("--include-incomplete-train-log", action="store_true",
                        help="allow runs with the checkpoint but without step=1000/1000 in train.log")
    parser.add_argument("--execute", action="store_true",
                        help="execute commands; without this flag commands are only printed")
    parser.add_argument("--no-resume-existing", action="store_true",
                        help="do not pass --resume-existing to evaluate_mitty_models")
    parser.add_argument("--metric-workers", type=int, default=None)
    parser.add_argument("--lpips-batch-size", type=int, default=None)
    parser.add_argument("--feature-batch-size", type=int, default=None)
    parser.add_argument("--fvd-batch-size", type=int, default=None)
    parser.add_argument("--num-inference-steps", type=int, default=None,
                        help="override the value recorded in train.log")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def parse_cuda_values(args: argparse.Namespace) -> list[str]:
    raw = args.cuda_list if args.cuda_list else args.cuda
    values = [item.strip() for item in str(raw).split(",") if item.strip()]
    if not values:
        raise ValueError("At least one CUDA id must be provided")
    if len(set(values)) != len(values):
        raise ValueError(f"Duplicate CUDA ids are not allowed: {values}")
    return values


def cuda_uuid_map(cuda_values: list[str]) -> dict[str, str]:
    proc = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid",
            "--format=csv,noheader,nounits",
        ],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    by_index = {}
    for line in proc.stdout.splitlines():
        if not line.strip():
            continue
        index, uuid = [part.strip() for part in line.split(",", 1)]
        by_index[index] = uuid
    missing = [cuda for cuda in cuda_values if cuda not in by_index]
    if missing:
        raise ValueError(
            f"CUDA ids not found by nvidia-smi: {missing}; available={sorted(by_index)}"
        )
    return {cuda: by_index[cuda] for cuda in cuda_values}


def busy_compute_cuda_values(cuda_to_uuid: dict[str, str]) -> set[str]:
    proc = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid",
            "--format=csv,noheader,nounits",
        ],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    busy_uuids = set()
    for line in proc.stdout.splitlines():
        if not line.strip() or "No running processes found" in line:
            continue
        uuid = line.split(",", 1)[0].strip()
        if uuid:
            busy_uuids.add(uuid)
    return {
        cuda
        for cuda, uuid in cuda_to_uuid.items()
        if uuid in busy_uuids
    }


def read_train_args(run_dir: Path) -> dict:
    log_path = run_dir / "train.log"
    if not log_path.is_file():
        raise FileNotFoundError(f"Missing train.log: {log_path}")
    with log_path.open() as fh:
        for line in fh:
            marker = " Args: "
            if marker in line:
                return ast.literal_eval(line.split(marker, 1)[1].strip())
    raise ValueError(f"Could not find Args line in {log_path}")


def train_log_has_completed_step(run_dir: Path) -> bool:
    log_path = run_dir / "train.log"
    if not log_path.is_file():
        return False
    with log_path.open() as fh:
        return any(COMPLETED_STEP_MARKER in line for line in fh)


def positive_int(value: object) -> bool:
    if value is None:
        return False
    return int(value) > 0


def nonempty_task_csv(value: object) -> bool:
    if value is None:
        return False
    return any(item.strip() for item in str(value).split(","))


def eval_splits(train_args: dict, args: argparse.Namespace) -> tuple[str, ...]:
    splits = []
    if positive_int(args.in_task_eval_size):
        splits.append("in_task_eval")
    ood_tasks = train_args.get("ood_tasks") or train_args.get("ood_eval_tasks")
    if positive_int(args.ood_eval_size) and nonempty_task_csv(ood_tasks):
        splits.append("ood_eval")
    if not splits:
        raise ValueError(f"No eval splits found in train args: {train_args}")
    return tuple(splits)


def required_columns(
    train_args: dict,
    local_metrics: bool,
    patch_fid: bool,
) -> tuple[str, ...]:
    columns = list(REQUIRED_METRIC_COLUMNS)
    if local_metrics and train_args.get("data_type") != "r2h":
        columns.extend(REGION_METRIC_COLUMNS)
    if patch_fid and train_args.get("data_type") != "r2h":
        columns.extend(PATCH_METRIC_COLUMNS)
    return tuple(columns)


def summary_complete(
    output_dir: Path,
    checkpoint: str,
    train_args: dict,
    splits: tuple[str, ...],
    local_metrics: bool,
    patch_fid: bool,
) -> bool:
    csv_path = output_dir / "summary.csv"
    json_path = output_dir / "summary.json"
    if not csv_path.is_file() or not json_path.is_file():
        return False

    with csv_path.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    rows_by_split = {row.get("split"): row for row in rows}
    needed = required_columns(train_args, local_metrics, patch_fid)
    for split in splits:
        row = rows_by_split.get(split)
        if row is None:
            return False
        if row.get("checkpoint") != checkpoint:
            return False
        for key in needed:
            if row.get(key, "") == "":
                return False
    return True


def eval_task_name(train_args: dict) -> str:
    task_name = str(train_args.get("task_name", ""))
    if task_name in TRAIN_TASKS:
        return task_name
    data_type = str(train_args.get("data_type", ""))
    mapped = TASK_NAME_BY_DATA.get(data_type)
    if mapped and mapped in TRAIN_TASKS:
        return mapped
    available = ", ".join(sorted(TRAIN_TASKS))
    raise ValueError(
        f"Cannot map train task '{task_name}' with data_type '{data_type}' "
        f"to an eval task. Available eval task names: {available}"
    )


def add_option(command: list[str], name: str, value: object) -> None:
    if value is None:
        return
    command.extend([name, str(value)])


def build_command(
    run_dir: Path,
    output_dir: Path,
    checkpoint: str,
    train_args: dict,
    splits: tuple[str, ...],
    args: argparse.Namespace,
    cuda_value: str,
) -> tuple[str, ...]:
    train_tasks = train_args.get("original_train_tasks") or train_args.get("train_tasks")
    ood_tasks = train_args.get("ood_eval_tasks") or train_args.get("ood_tasks")
    if not nonempty_task_csv(train_tasks):
        raise ValueError(f"Missing train tasks in {run_dir}")
    runner_path = REPO_ROOT / "scripts" / RUNNER_SCRIPTS[args.runner]
    if not runner_path.is_file():
        raise FileNotFoundError(f"Runner script not found: {runner_path}")

    command = [
        str(runner_path),
        "eval_mitty",
        "--cuda",
        cuda_value,
        "--",
        "--device",
        args.device,
        "--runs",
        run_dir.name,
        "--checkpoint",
        checkpoint,
        "--output-exact-dir",
        str(output_dir),
        "--splits",
        *splits,
        "--task-name",
        eval_task_name(train_args),
        "--data-type",
        str(train_args["data_type"]),
        "--duration",
        str(train_args.get("duration", "1s")),
        "--train-tasks",
        str(train_tasks),
        "--ood-tasks",
        str(ood_tasks or ""),
        "--cache-root",
        str(train_args["cache_root"]),
        "--pair-root",
        str(train_args["pair_root"]),
        "--t5-cache-dir",
        str(train_args["t5_cache_dir"]),
        "--data-seed",
        str(train_args.get("data_seed", 42)),
    ]
    if positive_int(args.in_task_eval_size):
        command.extend(["--in-task-eval-size", str(args.in_task_eval_size)])
    if "ood_eval" in splits and positive_int(args.ood_eval_size):
        command.extend(["--ood-eval-size", str(args.ood_eval_size)])
    add_option(command, "--dit-dir", train_args.get("dit_dir"))
    add_option(command, "--vae-path", train_args.get("vae_path"))
    add_option(command, "--tokenizer-dir", train_args.get("tokenizer_dir"))
    add_option(
        command,
        "--num-inference-steps",
        args.num_inference_steps
        if args.num_inference_steps is not None
        else train_args.get("num_inference_steps"),
    )
    add_option(command, "--metric-workers", args.metric_workers)
    add_option(command, "--lpips-batch-size", args.lpips_batch_size)
    add_option(command, "--feature-batch-size", args.feature_batch_size)
    add_option(command, "--fvd-batch-size", args.fvd_batch_size)
    if not args.no_local_metrics and train_args.get("data_type") != "r2h":
        command.extend(["--mask-region-metrics", "on"])
    if not args.no_patch_fid and train_args.get("data_type") != "r2h":
        command.append("--patch-fid")
    if not args.no_resume_existing:
        command.append("--resume-existing")
    if args.no_progress:
        command.append("--no-progress")
    return tuple(command)


def discover_targets(args: argparse.Namespace) -> tuple[list[EvalTarget], list[SkippedRun]]:
    log_root = Path(args.log_root)
    if not log_root.is_absolute():
        log_root = REPO_ROOT / log_root
    if not log_root.is_dir():
        raise FileNotFoundError(f"Log root not found: {log_root}")

    targets = []
    skipped = []
    for run_dir in sorted(log_root.glob(f"{args.prefix}*")):
        run_name = run_dir.name
        if not run_dir.is_dir():
            continue
        ckpt = run_dir / "ckpt" / args.checkpoint
        if not ckpt.is_file():
            skipped.append(SkippedRun(run_name, f"missing {args.checkpoint}"))
            continue
        if not args.include_incomplete_train_log and not train_log_has_completed_step(run_dir):
            skipped.append(SkippedRun(run_name, "train.log has no step=1000/1000"))
            continue
        train_args = read_train_args(run_dir)
        if train_args.get("data_type") == "r2h" and not args.include_r2h:
            skipped.append(SkippedRun(run_name, "r2h skipped"))
            continue
        if (
            not args.include_ours_step1_step2
            and run_name.startswith(SKIP_OURS_STAGE_PREFIXES)
        ):
            skipped.append(SkippedRun(run_name, "final_ours_step1/step2 skipped"))
            continue
        splits = eval_splits(train_args, args)
        output_dir = run_dir / args.output_subdir
        log_path = output_dir / "eval.log"
        if not args.force and summary_complete(
            output_dir,
            args.checkpoint,
            train_args,
            splits,
            local_metrics=not args.no_local_metrics,
            patch_fid=not args.no_patch_fid,
        ):
            skipped.append(SkippedRun(run_name, "full_eval summary already complete"))
            continue
        targets.append(EvalTarget(
            run_dir=run_dir,
            train_args=train_args,
            splits=splits,
            output_dir=output_dir,
            log_path=log_path,
        ))
        if args.limit > 0 and len(targets) >= args.limit:
            break
    return targets, skipped


def command_for_target(
    target: EvalTarget,
    args: argparse.Namespace,
    cuda_value: str,
) -> tuple[str, ...]:
    return build_command(
        target.run_dir,
        target.output_dir,
        args.checkpoint,
        target.train_args,
        target.splits,
        args,
        cuda_value,
    )


def run_with_log(
    target: EvalTarget,
    args: argparse.Namespace,
    cuda_value: str,
    worker_name: str,
) -> None:
    command = command_for_target(target, args, cuda_value)
    target.output_dir.mkdir(parents=True, exist_ok=True)
    with target.log_path.open("a", buffering=1) as log:
        log.write("\n" + "=" * 80 + "\n")
        log.write(f"Run: {target.run_dir}\n")
        log.write(f"Output: {target.output_dir}\n")
        log.write(f"CUDA: {cuda_value}\n")
        log.write("Command: " + shlex.join(command) + "\n")
        log.write("=" * 80 + "\n")
        proc = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(f"[{worker_name}][{target.run_dir.name}] {line}", end="")
            log.write(line)
        return_code = proc.wait()
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, command)


def execute_queue(
    targets: list[EvalTarget],
    args: argparse.Namespace,
    cuda_values: list[str],
) -> None:
    pending: list[tuple[int, EvalTarget]] = list(enumerate(targets, start=1))
    cuda_to_uuid = cuda_uuid_map(cuda_values)
    active: dict[str, tuple[int, EvalTarget, threading.Thread]] = {}
    done_queue: queue.Queue[tuple[str, str, int, EvalTarget, BaseException | None]] = queue.Queue()
    failures: list[tuple[int, EvalTarget, BaseException]] = []

    def worker(cuda_value: str, index: int, target: EvalTarget) -> None:
        worker_name = f"cuda:{cuda_value}"
        try:
            run_with_log(target, args, cuda_value, worker_name)
            done_queue.put(("done", cuda_value, index, target, None))
        except BaseException as exc:
            done_queue.put(("failed", cuda_value, index, target, exc))

    while pending or active:
        while True:
            try:
                status, cuda_value, index, target, exc = done_queue.get_nowait()
            except queue.Empty:
                break
            active.pop(cuda_value, None)
            if status == "done":
                print(
                    f"\n[cuda:{cuda_value}] DONE [{index}/{len(targets)}] "
                    f"{target.run_dir.name}",
                    flush=True,
                )
            else:
                assert exc is not None
                failures.append((index, target, exc))
                print(
                    f"\n[cuda:{cuda_value}] FAILED [{index}/{len(targets)}] "
                    f"{target.run_dir.name}: {exc}",
                    flush=True,
                )

        if pending:
            busy = busy_compute_cuda_values(cuda_to_uuid)
            idle = [
                cuda
                for cuda in cuda_values
                if cuda not in active and cuda not in busy
            ]
            for cuda_value in idle:
                if not pending:
                    break
                index, target = pending.pop(0)
                print(
                    f"\n[cuda:{cuda_value}] START [{index}/{len(targets)}] "
                    f"{target.run_dir.name}",
                    flush=True,
                )
                thread = threading.Thread(
                    target=worker,
                    args=(cuda_value, index, target),
                    daemon=False,
                )
                active[cuda_value] = (index, target, thread)
                thread.start()
            if pending:
                active_desc = ", ".join(
                    f"cuda:{cuda}={target.run_dir.name}"
                    for cuda, (_, target, _) in sorted(active.items())
                ) or "none"
                busy_desc = ", ".join(f"cuda:{cuda}" for cuda in sorted(busy)) or "none"
                print(
                    f"Waiting for idle CUDA slot; pending={len(pending)} "
                    f"active={active_desc} busy_compute={busy_desc}",
                    flush=True,
                )

        if pending or active:
            time.sleep(args.poll_interval)

    while True:
        try:
            status, cuda_value, index, target, exc = done_queue.get_nowait()
        except queue.Empty:
            break
        if status == "failed":
            assert exc is not None
            failures.append((index, target, exc))

    if failures:
        details = "\n".join(
            f"  [{index}] {target.run_dir.name}: {exc}"
            for index, target, exc in failures
        )
        raise RuntimeError(f"{len(failures)} eval job(s) failed:\n{details}")


def print_skip_summary(skipped: list[SkippedRun]) -> None:
    if not skipped:
        return
    counts: dict[str, int] = {}
    for item in skipped:
        counts[item.reason] = counts.get(item.reason, 0) + 1
    print("\nSkipped runs:")
    for reason, count in sorted(counts.items()):
        print(f"  {count}: {reason}")
    print("Skipped detail:")
    for item in skipped:
        print(f"  {item.name}: {item.reason}")


def main() -> None:
    args = parse_args()
    cuda_values = parse_cuda_values(args)
    targets, skipped = discover_targets(args)
    if not targets:
        print("No matching final* step-1000 checkpoints need offline metrics.")
        print_skip_summary(skipped)
        return

    action = "EXECUTE" if args.execute else "DRY-RUN"
    print(f"{action}: {len(targets)} target(s)")
    print("CUDA poll list: " + ", ".join(cuda_values))
    for index, target in enumerate(targets, start=1):
        preview_cuda = cuda_values[(index - 1) % len(cuda_values)]
        command = command_for_target(target, args, preview_cuda)
        print(f"\n[{index}/{len(targets)}] {target.run_dir.name}")
        print(f"  preview cuda: {preview_cuda}")
        print(f"  splits: {', '.join(target.splits)}")
        print(f"  output: {target.output_dir}")
        print(f"  log: {target.log_path}")
        print("  command:")
        print("    " + shlex.join(command))
    print_skip_summary(skipped)
    if args.execute:
        execute_queue(targets, args, cuda_values)


if __name__ == "__main__":
    main()
