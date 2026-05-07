"""Mitty training entry for mixed original/synthetic h2r data."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from src.core.config import MAIN_ROOT
from src.pipeline import train_mitty
from src.pipeline.runtime_mixed_h2r import (
    MixedH2RSplit,
    build_mixed_h2r_split,
    write_mixed_h2r_split,
)


def _resolve_project_path(value: str, *, default: str = "") -> str:
    raw = value or default
    if not raw:
        return ""
    path = Path(raw)
    if not path.is_absolute():
        path = Path(MAIN_ROOT) / path
    return str(path)


def _link_record_cache_files(out_dir: Path, records: list[dict]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for idx, record in enumerate(records):
        src = Path(record["cache_path"])
        if not src.is_file():
            raise FileNotFoundError(f"Cache file does not exist: {src}")
        task = str(record["robot_task"]).replace("/", "_")
        mix_source = str(record.get("mix_source", "sample"))
        pair_id = str(record["pair_id"]).replace("/", "_")
        link = out_dir / f"{idx:06d}_{mix_source}_{task}_{pair_id}.pth"
        if link.exists() or link.is_symlink():
            if link.is_symlink() and Path(os.readlink(link)) == src:
                continue
            raise FileExistsError(f"Split cache path already exists: {link}")
        os.symlink(src, link)


def materialize_split_cache(run_dir: Path, split: MixedH2RSplit) -> tuple[str, str, str]:
    cache_dir = run_dir / "mixed_cache"
    train_dir = cache_dir / "train"
    eval_dir = cache_dir / "in_task_eval"
    ood_dir = cache_dir / "ood_eval"
    _link_record_cache_files(train_dir, split.train_records)
    _link_record_cache_files(eval_dir, split.eval_records)
    _link_record_cache_files(ood_dir, split.ood_records)
    return str(train_dir), str(eval_dir), str(ood_dir)


def _distributed_rank() -> int:
    raw = os.environ.get("RANK", "0")
    if not raw.isdigit():
        raise ValueError(f"RANK must be an integer when set, got {raw!r}")
    return int(raw)


def _sync_key() -> str:
    master_addr = os.environ.get("MASTER_ADDR", "single").replace("/", "_")
    master_port = os.environ.get("MASTER_PORT", "0")
    return f"{master_addr}_{master_port}"


def _wait_for_text(path: Path, *, label: str, timeout_s: float = 300.0) -> str:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if path.is_file():
            text = path.read_text().strip()
            if text:
                return text
        time.sleep(0.2)
    raise TimeoutError(f"Timed out waiting for {label}: {path}")


def build_arg_parser() -> argparse.ArgumentParser:
    ap = train_mitty.build_arg_parser(
        description=(
            "Mitty LoRA training with a stable mixed h2r split "
            "(original h2r + r2h-synthesized h2r)"
        ),
        require_cache=False,
    )
    ap.add_argument("--data-type", default="h2r",
                    help="data type; this mixed entry only accepts h2r")
    ap.add_argument("--duration", default="1s")
    ap.add_argument("--original-train-tasks", required=True,
                    help="comma-separated original h2r tasks used for train and stable in-task eval")
    ap.add_argument("--syn-train-tasks", required=True,
                    help="comma-separated synthetic h2r tasks used only for train")
    ap.add_argument("--ood-eval-tasks", default="",
                    help="comma-separated original h2r tasks used for stable OOD eval")
    ap.add_argument("--original-train-size", type=int, required=True,
                    help="number of original h2r training samples; 0 uses all non-eval capacity")
    ap.add_argument("--syn-train-size", type=int, required=True,
                    help="number of synthetic h2r training samples; 0 uses all syn samples")
    ap.add_argument("--in-task-eval-size", type=int, default=80,
                    help="stable in-task eval samples from original task pair_order tails")
    ap.add_argument("--ood-eval-size", type=int, default=42,
                    help="stable OOD eval samples from original OOD task pair_order tails")
    ap.add_argument("--cache-root", default="",
                    help="VAE cache root; default MAIN_ROOT/training_data/cache/vae")
    ap.add_argument("--pair-root", default="",
                    help="pair root; default MAIN_ROOT/training_data/pair")
    ap.add_argument("--data-seed", type=int, default=42,
                    help="seed used only if a task pair_order.jsonl must be created")
    return ap


def main() -> None:
    ap = build_arg_parser()
    args = ap.parse_args()
    user_cache_args = [
        flag for flag, attr in [
            ("--cache-train", "cache_train"),
            ("--cache-eval", "cache_eval"),
            ("--cache-ood", "cache_ood"),
        ]
        if getattr(args, attr)
    ]
    if user_cache_args:
        ap.error(
            "mixed h2r owns cache split construction; do not pass "
            + ", ".join(user_cache_args)
        )

    train_mitty.normalize_train_args(ap, args)
    args.cache_root = _resolve_project_path(
        args.cache_root, default="training_data/cache/vae")
    args.pair_root = _resolve_project_path(
        args.pair_root, default="training_data/pair")

    split = build_mixed_h2r_split(args)
    if not getattr(args, "run_prefix", ""):
        args.run_prefix = "MittyMixedH2R"

    rank = _distributed_rank()
    sync_dir = Path(args.output_dir) / ".mixed_h2r_sync"
    sync_dir.mkdir(parents=True, exist_ok=True)
    run_name_path = sync_dir / f"{_sync_key()}_run_name.txt"
    if args.wandb_run_name:
        if rank == 0:
            run_name_path.write_text(args.wandb_run_name + "\n")
    elif rank == 0:
        args.wandb_run_name = train_mitty.build_run_name(
            "mitty", args, n_train=len(split.train_records))
        run_name_path.write_text(args.wandb_run_name + "\n")
    else:
        args.wandb_run_name = _wait_for_text(run_name_path, label="mixed h2r run name")

    run_dir = Path(args.output_dir) / args.wandb_run_name
    ready_path = run_dir / "data_split" / ".mixed_h2r_ready"
    if rank == 0:
        train_dir, eval_dir, ood_dir = materialize_split_cache(run_dir, split)
        write_mixed_h2r_split(run_dir, args, split)
        ready_path.write_text("ok\n")
    else:
        _wait_for_text(ready_path, label="mixed h2r split")
        train_dir = str(run_dir / "mixed_cache" / "train")
        eval_dir = str(run_dir / "mixed_cache" / "in_task_eval")
        ood_dir = str(run_dir / "mixed_cache" / "ood_eval")

    args.cache_train = train_dir
    args.cache_eval = eval_dir
    args.cache_ood = ood_dir if split.ood_records else ""
    if "mixed_h2r" not in args.wandb_tags:
        args.wandb_tags = [*args.wandb_tags, "mixed_h2r"]

    train_mitty.train(args)


if __name__ == "__main__":
    main()
