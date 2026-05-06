"""Run, split, and selected-record helpers for offline Mitty evaluation."""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from pathlib import Path

from src.core.config import MAIN_ROOT, TRAINING_DATA_ROOT
from src.pipeline.eval_mitty.constants import SPLIT_ALIASES
from src.pipeline.runtime_data import (
    RuntimeSplit,
    build_count_eval_split,
    build_tail_eval_split,
)


@dataclass(frozen=True)
class RunSpec:
    name: str
    run_dir: Path
    checkpoint: Path
    merge_lora_paths: tuple[Path, ...]
    merge_lora_rank: int


def read_train_args(run_dir: Path) -> dict:
    log_path = run_dir / "train.log"
    if not log_path.is_file():
        return {}
    with log_path.open() as f:
        for line in f:
            marker = " Args: "
            if marker in line:
                return ast.literal_eval(line.split(marker, 1)[1].strip())
    return {}


def resolve_path(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return Path(MAIN_ROOT) / p


def normalize_splits(values: list[str]) -> list[str]:
    splits = []
    for value in values:
        if value not in SPLIT_ALIASES:
            raise ValueError(
                f"Unknown split '{value}'. Available: {sorted(SPLIT_ALIASES)}"
            )
        split = SPLIT_ALIASES[value]
        if split not in splits:
            splits.append(split)
    return splits


def find_latest_checkpoint(run_dir: Path) -> Path:
    ckpts = sorted(run_dir.glob("ckpt/step-*.safetensors"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found under {run_dir / 'ckpt'}")
    return max(ckpts, key=lambda p: int(p.stem.split("-")[-1]))


def parse_run_specs(
    run_args: list[str],
    checkpoint: str,
    auto_merge_lora: bool,
) -> list[RunSpec]:
    specs = []
    for run_arg in run_args:
        run_dir = resolve_path(run_arg)
        if not run_dir.exists():
            run_dir = Path(TRAINING_DATA_ROOT) / "log" / run_arg
        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {run_arg}")

        if checkpoint == "latest":
            ckpt = find_latest_checkpoint(run_dir)
        else:
            ckpt = run_dir / "ckpt" / checkpoint
            if not ckpt.exists():
                raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
        train_args = read_train_args(run_dir)
        merge_paths = ()
        if auto_merge_lora and train_args.get("merge_lora"):
            merge_paths = tuple(resolve_path(p) for p in train_args["merge_lora"])
            for p in merge_paths:
                if not p.is_file():
                    raise FileNotFoundError(
                        f"Merged LoRA not found for {run_dir.name}: {p}"
                    )
        specs.append(RunSpec(
            name=run_dir.name,
            run_dir=run_dir,
            checkpoint=ckpt,
            merge_lora_paths=merge_paths,
            merge_lora_rank=int(train_args.get("merge_lora_rank", 96)),
        ))
    return specs


def records_for_split(split: str, runtime_split: RuntimeSplit) -> list[dict]:
    if split == "in_task_eval":
        return runtime_split.eval_records
    if split == "ood_eval":
        return runtime_split.ood_records
    raise ValueError(f"Unsupported split: {split}")


def resolve_pair_media(record: dict, kind: str) -> Path:
    raw_path = record.get(kind)
    if not raw_path:
        raise ValueError(f"Selected record missing {kind}: {record}")
    path = Path(str(raw_path))
    if path.is_absolute():
        return path
    pair_dir = record.get("pair_dir")
    if not pair_dir:
        raise ValueError(f"Selected record missing pair_dir for relative {kind}: {record}")
    return Path(str(pair_dir)) / path


def eval_base_dir(run: RunSpec, output_dir: str) -> Path:
    if output_dir:
        return resolve_path(output_dir) / run.name / run.checkpoint.stem
    return run.checkpoint.parent / f"{run.checkpoint.stem}_eval"


def build_eval_split(args) -> RuntimeSplit:
    if args.in_task_eval_size is not None or args.ood_eval_size is not None:
        return build_count_eval_split(args)
    return build_tail_eval_split(args)


def write_selected_records(
    base_dir: Path,
    splits: list[str],
    runtime_split: RuntimeSplit,
    args,
) -> None:
    split_dir = base_dir / "data_split"
    split_dir.mkdir(parents=True, exist_ok=True)
    payloads = {
        "in_task_eval": runtime_split.eval_records,
        "ood_eval": runtime_split.ood_records,
    }
    for split in splits:
        with (split_dir / f"{split}.jsonl").open("w") as fh:
            for record in payloads[split]:
                fh.write(json.dumps(record, sort_keys=True) + "\n")
    config = {
        "task_name": args.task_name,
        "data_type": args.data_type,
        "duration": args.duration,
        "train_tasks": args.train_tasks,
        "ood_tasks": args.ood_tasks,
        "in_task_eval_size": args.in_task_eval_size,
        "ood_eval_size": args.ood_eval_size,
        "eval_tail_percent": args.eval_tail_percent,
        "mask_region_metrics": args.mask_region_metrics,
        "sam2_mask_root": args.sam2_mask_root,
        "write_local_videos": args.write_local_videos,
        "local_video_margin": args.local_video_margin,
        "local_video_size": args.local_video_size,
        "local_video_bbox_mode": args.local_video_bbox_mode,
        "metric_workers": args.metric_workers,
        "lpips_batch_size": args.lpips_batch_size,
        "feature_batch_size": args.feature_batch_size,
        "fvd_batch_size": args.fvd_batch_size,
        "data_seed": args.data_seed,
        "cache_root": args.cache_root,
        "pair_root": args.pair_root,
        "split_counts": runtime_split.split_counts,
        "pair_order_paths": runtime_split.order_paths,
    }
    (split_dir / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n"
    )

