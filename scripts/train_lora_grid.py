#!/usr/bin/env python3
"""Launch FLIP LoRA layout/rank grid training runs.

The launcher expands ``layout × rank`` into one-GPU training commands and runs
them sequentially, rotating through the requested CUDA ids.  It delegates
actual training to ``scripts/flip_run.sh train`` so the standard FLIP runtime
environment is preserved.
"""

from __future__ import annotations

import argparse
import shlex
import socket
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAIN_ROOT = PROJECT_ROOT if PROJECT_ROOT.name != "t038" else PROJECT_ROOT.parents[1]
DEFAULT_FLIP_RUNNER = PROJECT_ROOT / "scripts" / "flip_run.sh"
IP_RUNNERS = {
    "10.20.1.2": PROJECT_ROOT / "scripts" / "flip_run_2.sh",
}

LAYOUT_TARGETS: dict[str, list[str]] = {
    "self_qkv": ["self_attn.q", "self_attn.k", "self_attn.v"],
    "self_qkvo": ["self_attn.q", "self_attn.k", "self_attn.v", "self_attn.o"],
    "cross_qkv": ["cross_attn.q", "cross_attn.k", "cross_attn.v"],
    "cross_qkvo": ["cross_attn.q", "cross_attn.k", "cross_attn.v", "cross_attn.o"],
    "self_qkv_cross_qkv": [
        "self_attn.q", "self_attn.k", "self_attn.v",
        "cross_attn.q", "cross_attn.k", "cross_attn.v",
    ],
    "self_qkvo_cross_qkvo": [
        "self_attn.q", "self_attn.k", "self_attn.v", "self_attn.o",
        "cross_attn.q", "cross_attn.k", "cross_attn.v", "cross_attn.o",
    ],
    "ffn": ["ffn.0", "ffn.2"],
    "self_qkv_ffn": ["self_attn.q", "self_attn.k", "self_attn.v", "ffn.0", "ffn.2"],
    "self_qkvo_ffn": [
        "self_attn.q", "self_attn.k", "self_attn.v", "self_attn.o", "ffn.0", "ffn.2",
    ],
    "self_qkv_cross_qkv_ffn": [
        "self_attn.q", "self_attn.k", "self_attn.v",
        "cross_attn.q", "cross_attn.k", "cross_attn.v",
        "ffn.0", "ffn.2",
    ],
    "self_qkvo_cross_qkvo_ffn": [
        "self_attn.q", "self_attn.k", "self_attn.v", "self_attn.o",
        "cross_attn.q", "cross_attn.k", "cross_attn.v", "cross_attn.o",
        "ffn.0", "ffn.2",
    ],
}


@dataclass(frozen=True)
class RunSpec:
    layout: str
    rank: int | None
    cuda: str
    run_name: str
    command: list[str]


def _csv(value: str) -> list[str]:
    parts = []
    for chunk in value.replace(";", ",").split(","):
        item = chunk.strip()
        if item:
            parts.append(item)
    return parts


def _merge_lora_items(values: list[str]) -> list[str]:
    items: list[str] = []
    for value in values:
        for chunk in value.split(","):
            item = chunk.strip()
            if item:
                items.append(item)
    return items


def _resolve_existing_file(path_text: str) -> str:
    path = Path(path_text).expanduser()
    candidates = [path] if path.is_absolute() else [PROJECT_ROOT / path, MAIN_ROOT / path]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    path = candidates[-1]
    raise FileNotFoundError(f"LoRA checkpoint not found: {path}")


def _local_ip_addresses() -> set[str]:
    addresses = set()
    hostname = socket.gethostname()
    for host in (hostname, socket.getfqdn(), ""):
        try:
            infos = socket.getaddrinfo(host, None, family=socket.AF_INET)
        except socket.gaierror:
            continue
        for info in infos:
            addresses.add(info[4][0])
    return addresses


def _default_runner() -> Path:
    addresses = _local_ip_addresses()
    for ip_address, runner in IP_RUNNERS.items():
        if ip_address in addresses:
            return runner
    return DEFAULT_FLIP_RUNNER


def _parse_layout(value: str) -> tuple[str, list[str]]:
    if "=" in value:
        name, raw_targets = value.split("=", 1)
        layout = name.strip()
        targets = _csv(raw_targets)
        if not layout:
            raise ValueError(f"Custom layout has empty name: {value}")
        if not targets:
            raise ValueError(f"Custom layout has empty targets: {value}")
        return layout, targets
    layout = value.strip()
    if layout not in LAYOUT_TARGETS:
        available = ", ".join(sorted(LAYOUT_TARGETS))
        raise ValueError(f"Unknown LoRA layout '{layout}'. Available: {available}")
    return layout, LAYOUT_TARGETS[layout]


def _positive_int_csv(value: str, *, label: str) -> list[int]:
    out = []
    for item in _csv(value):
        number = int(item)
        if number <= 0:
            raise ValueError(f"{label} must be positive: {number}")
        out.append(number)
    if not out:
        raise ValueError(f"{label} is empty")
    return out


def _optional_int_flag(cmd: list[str], flag: str, value: int) -> None:
    if value != 0:
        cmd.extend([flag, str(value)])


def _build_runs(args: argparse.Namespace) -> list[RunSpec]:
    cuda_devices = _csv(args.cuda)
    if not cuda_devices:
        raise ValueError("--cuda is empty")

    train_lora_args = [
        (flag, value)
        for flag, value in [
            ("--init-lora", args.init_lora),
            ("--continue-lora", args.continue_lora),
            ("--train-lora", args.train_lora),
        ]
        if value
    ]
    init_lora = ""
    if train_lora_args:
        init_lora = _resolve_existing_file(train_lora_args[0][1])
        for flag, value in train_lora_args[1:]:
            path = _resolve_existing_file(value)
            if path != init_lora:
                raise ValueError(
                    f"{flag} points to a different file than {train_lora_args[0][0]}"
                )
    merge_loras = [_resolve_existing_file(path) for path in _merge_lora_items(args.merge_lora)]
    if init_lora and not args.layouts_explicit:
        layouts = [("continued", [])]
    else:
        layouts = [_parse_layout(item) for item in _csv(args.layouts)]
    if init_lora and not args.ranks_explicit:
        ranks = [None]
    else:
        ranks = _positive_int_csv(args.ranks, label="rank")

    timestamp = args.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = args.name_prefix or args.task_name
    runner = str(Path(args.runner).expanduser()) if args.runner else str(_default_runner())

    runs: list[RunSpec] = []
    index = 0
    for layout, targets in layouts:
        for rank in ranks:
            cuda = cuda_devices[index % len(cuda_devices)]
            rank_part = f"r{rank}" if rank is not None else "auto"
            run_name = f"{prefix}_{layout}_{rank_part}_{timestamp}"
            cmd = [runner, "train", "--cuda", cuda, "--nproc", "1", "--"]
            cmd.extend(["--task-name", args.task_name])
            if args.data_type:
                cmd.extend(["--data-type", args.data_type])
            if args.duration:
                cmd.extend(["--duration", args.duration])
            if args.train_tasks:
                cmd.extend(["--train-tasks", args.train_tasks])
            if args.ood_tasks:
                cmd.extend(["--ood-tasks", args.ood_tasks])
            if args.cache_root:
                cmd.extend(["--cache-root", args.cache_root])
            if args.t5_cache_dir:
                cmd.extend(["--t5-cache-dir", args.t5_cache_dir])
            if args.output_dir:
                cmd.extend(["--output-dir", args.output_dir])
            for lora_path in merge_loras:
                cmd.extend(["--merge-lora", lora_path])
            if init_lora:
                cmd.extend(["--train-lora", init_lora])
            if rank is not None:
                cmd.extend(["--lora-rank", str(rank)])
            if targets:
                cmd.extend(["--lora-target-modules", ",".join(targets)])
            cmd.extend(["--batch-size", str(args.batch_size)])
            _optional_int_flag(cmd, "--train-size", args.train_size)
            _optional_int_flag(cmd, "--in-task-eval-size", args.in_task_eval_size)
            _optional_int_flag(cmd, "--ood-eval-size", args.ood_eval_size)
            _optional_int_flag(cmd, "--in-task-video-size", args.in_task_video_size)
            _optional_int_flag(cmd, "--ood-video-size", args.ood_video_size)
            cmd.extend(["--data-seed", str(args.data_seed)])
            cmd.extend(["--max-steps", str(args.max_steps)])
            cmd.extend(["--save-steps", str(args.save_steps)])
            cmd.extend(["--eval-steps", str(args.eval_steps)])
            cmd.extend(["--eval-video-steps", str(args.eval_video_steps)])
            cmd.extend(["--lr", str(args.lr)])
            cmd.extend(["--lr-min", str(args.lr_min)])
            cmd.extend(["--warmup-steps", str(args.warmup_steps)])
            cmd.extend(["--weight-decay", str(args.weight_decay)])
            cmd.extend(["--wandb-project", args.wandb_project])
            cmd.extend(["--wandb-run-name", run_name])
            cmd.extend([
                "--wandb-tags",
                args.task_name,
                f"layout:{layout}",
                f"grid:{timestamp}",
                f"cuda:{cuda}",
            ])
            if rank is not None:
                cmd.append(f"rank:{rank}")
            else:
                cmd.append("rank:auto")
            cmd.extend(args.extra_train_arg)
            runs.append(RunSpec(layout=layout, rank=rank, cuda=cuda, run_name=run_name, command=cmd))
            index += 1
    return runs


def _print_run(run: RunSpec) -> None:
    print(
        f"\n[{datetime.now().strftime('%F %T')}] "
        f"cuda={run.cuda} layout={run.layout} rank={run.rank} run={run.run_name}",
        flush=True,
    )
    print("Command:", shlex.join(run.command), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run LoRA layout × rank grid search with sequential CUDA assignment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--merge-lora", nargs="*", default=[], help="one or more LoRA checkpoints merged before training")
    parser.add_argument("--init-lora", default="", help="optional LoRA checkpoint used to initialize trainable LoRA")
    parser.add_argument("--continue-lora", default="", help="preferred alias for --init-lora when continuing one trainable LoRA")
    parser.add_argument("--train-lora", default="", help="trainable LoRA checkpoint to continue")
    parser.add_argument("--task-name", default="h2r_1s", help="src.pipeline.train task preset")
    parser.add_argument("--data-type", default="", help="override semantic data type")
    parser.add_argument("--duration", default="", help="override data duration")
    parser.add_argument("--train-tasks", default="", help="override train/in-task robot task short names")
    parser.add_argument("--ood-tasks", default="", help="override OOD robot task short names")
    parser.add_argument("--cache-root", default="", help="override VAE cache root")
    parser.add_argument("--t5-cache-dir", default="", help="override T5 cache dir")
    parser.add_argument("--output-dir", default="", help="override training log/output root")
    parser.add_argument("--train-size", type=int, default=0, help="runtime train clip count; 0 means train.py default/all")
    parser.add_argument("--in-task-eval-size", type=int, default=16)
    parser.add_argument("--ood-eval-size", type=int, default=16)
    parser.add_argument("--in-task-video-size", type=int, default=8)
    parser.add_argument("--ood-video-size", type=int, default=8)
    parser.add_argument("--data-seed", type=int, default=42)
    parser.add_argument("--layouts", default="self_qkv,self_qkvo,cross_qkv,cross_qkvo,self_qkv_cross_qkv,self_qkvo_cross_qkvo,ffn,self_qkv_cross_qkv_ffn,self_qkvo_cross_qkvo_ffn")
    parser.add_argument("--ranks", default="64,128,256")
    parser.add_argument("--cuda", required=True, help="comma-separated CUDA ids; runs rotate through this list")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--eval-video-steps", type=int, default=100)
    parser.add_argument("--lr", default="1e-4")
    parser.add_argument("--lr-min", default="1e-6")
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--weight-decay", default="0.01")
    parser.add_argument("--wandb-project", default="Flip")
    parser.add_argument("--name-prefix", default="", help="prefix for W&B run name; default is task-name")
    parser.add_argument("--timestamp", default="", help="fixed YYYYMMDD_HHMMSS-like run timestamp")
    parser.add_argument("--runner", default="", help="override launcher; default uses flip_run_2.sh on 10.20.1.2, otherwise flip_run.sh")
    parser.add_argument("--dry-run", action="store_true", help="print commands without launching training")
    parser.add_argument("extra_train_arg", nargs=argparse.REMAINDER, help="extra train.py args after --")
    args = parser.parse_args()
    args.layouts_explicit = any(
        arg == "--layouts" or arg.startswith("--layouts=") for arg in sys.argv
    )
    args.ranks_explicit = any(
        arg == "--ranks" or arg.startswith("--ranks=") for arg in sys.argv
    )
    if args.extra_train_arg and args.extra_train_arg[0] == "--":
        args.extra_train_arg = args.extra_train_arg[1:]
    return args


def main() -> None:
    args = parse_args()
    try:
        runs = _build_runs(args)
    except (FileNotFoundError, ValueError) as exc:
        print(f"train_lora_grid.py: {exc}", flush=True)
        raise SystemExit(2) from None
    print(f"Expanded {len(runs)} single-card run(s).", flush=True)
    for run in runs:
        _print_run(run)
        if not args.dry_run:
            proc = subprocess.run(run.command, cwd=PROJECT_ROOT, check=False)
            if proc.returncode != 0:
                print(
                    "\ntrain_lora_grid.py: run failed\n"
                    f"  run: {run.run_name}\n"
                    f"  cuda: {run.cuda}\n"
                    f"  layout: {run.layout}\n"
                    f"  rank: {run.rank}\n"
                    f"  exit_code: {proc.returncode}\n"
                    f"  command: {shlex.join(run.command)}",
                    flush=True,
                )
                raise SystemExit(proc.returncode) from None


if __name__ == "__main__":
    main()
