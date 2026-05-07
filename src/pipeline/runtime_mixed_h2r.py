"""Mixed h2r runtime split builder.

This module is intentionally separate from :mod:`src.pipeline.runtime_data`.
It builds the experimental split for original h2r + r2h-synthesized h2r
training while keeping the maintained training split behavior unchanged.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from src.core.config import MAIN_ROOT
from src.pipeline.runtime_data import (
    DATA_TYPES,
    RuntimeSplit,
    _allocate_counts,
    _load_ordered_task_records,
    _task_counts,
    parse_task_list,
)


@dataclass(frozen=True)
class MixedH2RSplit(RuntimeSplit):
    original_train_records: list[dict]
    syn_train_records: list[dict]
    task_counts: dict[str, dict[str, int]]


def _load_tasks(
    cache_task_root: Path,
    pair_root: Path,
    data_type: str,
    duration: str,
    tasks: Iterable[str],
    data_seed: int,
    order_paths: dict[str, str],
    order_label: str,
) -> dict[str, list[dict]]:
    records_by_task = {}
    for task in tasks:
        records, order_path = _load_ordered_task_records(
            cache_task_root, pair_root, data_type, duration, task, data_seed,
        )
        records_by_task[task] = records
        order_paths[f"{order_label}:{task}"] = str(order_path)
    return records_by_task


def _is_syn_task(task: str, syn_tasks: set[str]) -> bool:
    return task in syn_tasks or task.endswith("_syn")


def _robot_source_key(record: dict) -> str:
    existing = record.get("robot_source_key")
    if existing:
        return str(existing)

    task = (
        record.get("source_robot_task")
        or record.get("task")
        or record.get("robot_task")
    )
    episode = record.get("episode")
    seg = record.get("seg")
    clip_start = record.get("clip_start")
    clip_dur = record.get("clip_dur") or record.get("duration_seconds")
    if task and episode and seg and clip_start is not None and clip_dur is not None:
        return (
            f"{parse_task_list([str(task)])[0]}/{episode}/{seg}"
            f"_start{float(clip_start):.3f}_dur{float(clip_dur):.3f}"
        )

    source_segment_id = record.get("source_segment_id")
    if source_segment_id:
        clip_parts = []
        for key in ("clip_start", "clip_idx", "window_idx", "source_clip_start"):
            value = record.get(key)
            if value is not None:
                clip_parts.append(f"{key}={value}")
                break
        if clip_parts:
            return f"{source_segment_id}|{clip_parts[0]}"
        return str(source_segment_id)

    source_id = record.get("source_id")
    if source_id:
        return str(source_id)

    raise ValueError(
        "Cannot derive robot_source_key from manifest/cache record "
        f"for pair_id={record.get('pair_id')!r}"
    )


def _tag_records(records: list[dict], mix_source: str) -> list[dict]:
    tagged = []
    for record in records:
        out = dict(record)
        out["mix_source"] = mix_source
        out["robot_source_key"] = _robot_source_key(out)
        tagged.append(out)
    return tagged


def _record_keys(records: list[dict]) -> set[tuple[str, str]]:
    return {
        (str(record["robot_task"]), str(record["pair_id"]))
        for record in records
    }


def _pair_id_sort_key(record: dict) -> tuple:
    pair_id = str(record["pair_id"])
    parts = re.split(r"(\d+)", pair_id)
    key = tuple(int(part) if part.isdigit() else part for part in parts)
    return key


def _pair_id_ordered(records: list[dict]) -> list[dict]:
    return sorted(records, key=_pair_id_sort_key)


def build_mixed_h2r_split(args) -> MixedH2RSplit:
    data_type = getattr(args, "data_type", "h2r")
    duration = args.duration
    if data_type != "h2r":
        raise ValueError(f"mixed h2r split only supports data_type='h2r', got {data_type!r}")
    if data_type not in DATA_TYPES:
        raise ValueError(f"Unknown data type '{data_type}'. Available: {sorted(DATA_TYPES)}")

    original_tasks = parse_task_list(args.original_train_tasks)
    syn_tasks = parse_task_list(args.syn_train_tasks)
    ood_tasks = parse_task_list(args.ood_eval_tasks, allow_empty=True)
    syn_task_set = set(syn_tasks)

    bad_eval_tasks = [
        task
        for task in [*original_tasks, *ood_tasks]
        if _is_syn_task(task, syn_task_set)
    ]
    if bad_eval_tasks:
        raise ValueError(f"Syn tasks cannot be used for stable eval: {bad_eval_tasks}")
    overlaps = sorted((set(original_tasks) | set(ood_tasks)) & syn_task_set)
    if overlaps:
        raise ValueError(f"Tasks cannot be both syn train and original/eval: {overlaps}")

    if args.original_train_size < 0 or args.syn_train_size < 0:
        raise ValueError("--original-train-size and --syn-train-size must be >= 0")
    if args.in_task_eval_size <= 0:
        raise ValueError(f"--in-task-eval-size must be positive, got {args.in_task_eval_size}")
    if args.ood_eval_size < 0:
        raise ValueError(f"--ood-eval-size must be >= 0, got {args.ood_eval_size}")
    if args.ood_eval_size > 0 and not ood_tasks:
        raise ValueError("--ood-eval-size is positive but --ood-eval-tasks is empty")

    cache_root = getattr(args, "cache_root", "") or str(
        Path(MAIN_ROOT) / "training_data" / "cache" / "vae"
    )
    pair_root = getattr(args, "pair_root", "") or str(
        Path(MAIN_ROOT) / "training_data" / "pair"
    )
    cache_task_root = Path(cache_root) / data_type / duration
    pair_root_path = Path(pair_root)
    order_paths: dict[str, str] = {}

    original_by_task = _load_tasks(
        cache_task_root, pair_root_path, data_type, duration,
        original_tasks, args.data_seed, order_paths, "original",
    )
    syn_by_task = _load_tasks(
        cache_task_root, pair_root_path, data_type, duration,
        syn_tasks, args.data_seed, order_paths, "syn",
    )
    ood_by_task = _load_tasks(
        cache_task_root, pair_root_path, data_type, duration,
        ood_tasks, args.data_seed, order_paths, "ood",
    )

    original_sizes = {task: len(original_by_task[task]) for task in original_tasks}
    eval_counts = _allocate_counts(
        original_sizes, args.in_task_eval_size, "stable in-task eval",
    )
    original_capacity = {
        task: original_sizes[task] - eval_counts[task]
        for task in original_tasks
    }
    if sum(original_capacity.values()) <= 0:
        raise ValueError(
            "Stable in-task eval consumes all original samples; no original train data remains"
        )
    original_train_counts = _allocate_counts(
        original_capacity, args.original_train_size, "original train",
    )

    syn_sizes = {task: len(syn_by_task[task]) for task in syn_tasks}
    syn_train_counts = _allocate_counts(syn_sizes, args.syn_train_size, "syn train")

    eval_records: list[dict] = []
    for task in original_tasks:
        task_records = original_by_task[task]
        n_eval = eval_counts[task]
        eval_start = len(task_records) - n_eval
        eval_records.extend(task_records[eval_start:])

    eval_keys_for_train_exclusion = _record_keys(eval_records)

    original_train_records: list[dict] = []
    for task in original_tasks:
        n_train = original_train_counts[task]
        candidates = [
            record
            for record in _pair_id_ordered(original_by_task[task])
            if (str(record["robot_task"]), str(record["pair_id"]))
            not in eval_keys_for_train_exclusion
        ]
        original_train_records.extend(candidates[:n_train])

    syn_train_records: list[dict] = []
    for task in syn_tasks:
        syn_train_records.extend(
            _pair_id_ordered(syn_by_task[task])[:syn_train_counts[task]]
        )

    ood_records: list[dict] = []
    if ood_tasks and args.ood_eval_size > 0:
        ood_sizes = {task: len(ood_by_task[task]) for task in ood_tasks}
        ood_counts = _allocate_counts(ood_sizes, args.ood_eval_size, "stable OOD eval")
        for task in ood_tasks:
            n_ood = ood_counts[task]
            if n_ood:
                ood_records.extend(ood_by_task[task][-n_ood:])

    original_train_records = _tag_records(original_train_records, "original")
    syn_train_records = _tag_records(syn_train_records, "syn")
    eval_records = _tag_records(eval_records, "original")
    ood_records = _tag_records(ood_records, "original")

    train_records = [*original_train_records, *syn_train_records]
    train_keys = _record_keys(train_records)
    eval_keys = _record_keys(eval_records) | _record_keys(ood_records)
    overlap_keys = sorted(train_keys & eval_keys)
    if overlap_keys:
        raise ValueError(f"Train records overlap stable eval records: {overlap_keys[:5]}")

    original_sources = {record["robot_source_key"] for record in original_train_records}
    syn_sources = {record["robot_source_key"] for record in syn_train_records}
    robot_overlap = sorted(original_sources & syn_sources)
    if robot_overlap:
        raise ValueError(
            "Original and syn train records share robot_source_key values: "
            f"{robot_overlap[:5]}"
        )

    return MixedH2RSplit(
        train_files=[record["cache_path"] for record in train_records],
        eval_files=[record["cache_path"] for record in eval_records],
        ood_files=[record["cache_path"] for record in ood_records],
        train_records=train_records,
        eval_records=eval_records,
        ood_records=ood_records,
        split_counts={
            "train": _task_counts(train_records),
            "in_task_eval": _task_counts(eval_records),
            "ood_eval": _task_counts(ood_records),
        },
        order_paths=order_paths,
        original_train_records=original_train_records,
        syn_train_records=syn_train_records,
        task_counts={
            "original_train": _task_counts(original_train_records),
            "syn_train": _task_counts(syn_train_records),
            "in_task_eval": _task_counts(eval_records),
            "ood_eval": _task_counts(ood_records),
        },
    )


def write_mixed_h2r_split(run_dir: Path, args, split: MixedH2RSplit) -> None:
    out_dir = run_dir / "data_split"
    out_dir.mkdir(parents=True, exist_ok=True)
    payloads = {
        "train.jsonl": split.train_records,
        "in_task_eval.jsonl": split.eval_records,
        "ood_eval.jsonl": split.ood_records,
    }
    for name, records in payloads.items():
        with (out_dir / name).open("w") as fh:
            for record in records:
                fh.write(json.dumps(record, sort_keys=True) + "\n")

    config = {
        "mode": "mixed_h2r",
        "data_type": "h2r",
        "duration": args.duration,
        "original_train_tasks": parse_task_list(args.original_train_tasks),
        "syn_train_tasks": parse_task_list(args.syn_train_tasks),
        "ood_eval_tasks": parse_task_list(args.ood_eval_tasks, allow_empty=True),
        "original_train_size": args.original_train_size,
        "syn_train_size": args.syn_train_size,
        "in_task_eval_size": args.in_task_eval_size,
        "ood_eval_size": args.ood_eval_size,
        "actual_counts": split.task_counts,
        "split_counts": split.split_counts,
        "pair_order_paths": split.order_paths,
        "train_selection_order": "pair_id_ascending",
        "eval_selection_order": "pair_order_tail",
        "cache_root": getattr(args, "cache_root", ""),
        "pair_root": getattr(args, "pair_root", ""),
        "data_seed": args.data_seed,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True))
