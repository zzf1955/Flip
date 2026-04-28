"""Runtime data selection for FLIP training.

The on-disk data layout is split-free:
  training_data/cache/vae/<data_type>/<duration>/<robot_task>/manifest.jsonl

In-task/OOD splits are experiment-time choices. This module builds deterministic
train/eval/video pools from per-task manifests and writes the exact selection
into each run directory.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from src.core.config import MAIN_ROOT

DATA_TYPES = {"identity_r2r", "blur_r2r", "h2r", "r2h"}


@dataclass(frozen=True)
class RuntimeSplit:
    train_files: list[str]
    eval_files: list[str]
    ood_files: list[str]
    train_records: list[dict]
    eval_records: list[dict]
    ood_records: list[dict]


def short_task_name(task: str) -> str:
    return task.strip().replace("G1_WBT_", "")


def parse_task_list(value: str | Iterable[str], *, allow_empty: bool = False) -> list[str]:
    if isinstance(value, str):
        items = [item.strip() for item in value.split(",")]
    else:
        items = [str(item).strip() for item in value]
    tasks = [short_task_name(item) for item in items if item]
    if not tasks:
        if allow_empty:
            return []
        raise ValueError("Task list must not be empty")
    return tasks


def _sample(records: list[dict], size: int, seed: str, label: str) -> list[dict]:
    if size < 0:
        return list(records)
    if size == 0:
        return list(records)
    if size > len(records):
        raise ValueError(
            f"Requested {label} size {size}, but only {len(records)} samples are available"
        )
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    return sorted(shuffled[:size], key=_record_sort_key)


def _record_sort_key(record: dict) -> tuple[str, str]:
    return (str(record.get("source_id", "")), str(record.get("cache_path", "")))


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Required manifest not found: {path}")
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
    if not rows:
        raise ValueError(f"Manifest is empty: {path}")
    return rows


def _resolve_cache_path(task_dir: Path, record: dict) -> str:
    raw_path = record.get("cache_path")
    if not raw_path:
        raise ValueError(f"Manifest record missing cache_path in {task_dir}")
    path = Path(raw_path)
    if not path.is_absolute():
        path = task_dir / path
    if not path.is_file():
        raise FileNotFoundError(f"Cache file from manifest does not exist: {path}")
    return str(path)


def _load_task_records(root: Path, data_type: str, duration: str, task: str) -> list[dict]:
    task_dir = root / task
    rows = _read_jsonl(task_dir / "manifest.jsonl")
    records = []
    for row in rows:
        record = dict(row)
        record_data_type = record.get("data_type", data_type)
        record_duration = record.get("duration", duration)
        record_task = short_task_name(record.get("robot_task", record.get("task", task)))
        if record_data_type != data_type:
            raise ValueError(
                f"Manifest data_type mismatch in {task_dir}: {record_data_type} != {data_type}"
            )
        if record_duration != duration:
            raise ValueError(
                f"Manifest duration mismatch in {task_dir}: {record_duration} != {duration}"
            )
        if record_task != task:
            raise ValueError(
                f"Manifest robot_task mismatch in {task_dir}: {record_task} != {task}"
            )
        record["data_type"] = data_type
        record["duration"] = duration
        record["robot_task"] = task
        record["cache_path"] = _resolve_cache_path(task_dir, record)
        record.setdefault("source_segment_id", record.get("source_id", record["cache_path"]))
        records.append(record)
    return sorted(records, key=_record_sort_key)


def _heldout_eval_pool(records: list[dict], seed: int, task: str) -> tuple[list[dict], list[dict]]:
    by_segment: dict[str, list[dict]] = {}
    for record in records:
        by_segment.setdefault(str(record["source_segment_id"]), []).append(record)
    segment_ids = sorted(by_segment)
    if len(segment_ids) < 2:
        if len(records) == 1:
            return list(records), list(records)
        raise ValueError(
            f"Task {task} needs at least 2 source_segment_id groups for runtime train/eval split"
        )
    random.Random(f"{seed}:segment:{task}").shuffle(segment_ids)
    n_eval_segments = max(1, int(len(segment_ids) * 0.1))
    n_eval_segments = min(n_eval_segments, len(segment_ids) - 1)
    eval_segments = set(segment_ids[:n_eval_segments])
    train_records = []
    eval_records = []
    for segment_id in segment_ids:
        bucket = eval_records if segment_id in eval_segments else train_records
        bucket.extend(by_segment[segment_id])
    return sorted(train_records, key=_record_sort_key), sorted(eval_records, key=_record_sort_key)


def build_runtime_split(args) -> RuntimeSplit:
    data_type = args.data_type
    duration = args.duration
    if data_type not in DATA_TYPES:
        raise ValueError(f"Unknown data type '{data_type}'. Available: {sorted(DATA_TYPES)}")
    train_tasks = parse_task_list(args.train_tasks)
    ood_tasks = parse_task_list(args.ood_tasks, allow_empty=True)
    overlap = sorted(set(train_tasks) & set(ood_tasks))
    if overlap:
        raise ValueError(f"Tasks cannot be both train and OOD: {overlap}")

    cache_root = getattr(args, "cache_root", "") or str(
        Path(MAIN_ROOT) / "training_data" / "cache" / "vae"
    )
    root = Path(cache_root) / data_type / duration
    train_pool: list[dict] = []
    eval_pool: list[dict] = []
    ood_pool: list[dict] = []

    for task in train_tasks:
        records = _load_task_records(root, data_type, duration, task)
        task_train, task_eval = _heldout_eval_pool(records, args.data_seed, task)
        train_pool.extend(task_train)
        eval_pool.extend(task_eval)

    for task in ood_tasks:
        ood_pool.extend(_load_task_records(root, data_type, duration, task))

    train_records = _sample(
        sorted(train_pool, key=_record_sort_key), args.train_size,
        f"{args.data_seed}:train", "train",
    )
    eval_records = _sample(
        sorted(eval_pool, key=_record_sort_key), args.in_task_eval_size,
        f"{args.data_seed}:in_task_eval", "in-task eval",
    )
    ood_records = _sample(
        sorted(ood_pool, key=_record_sort_key), args.ood_eval_size,
        f"{args.data_seed}:ood_eval", "OOD eval",
    )

    return RuntimeSplit(
        train_files=[record["cache_path"] for record in train_records],
        eval_files=[record["cache_path"] for record in eval_records],
        ood_files=[record["cache_path"] for record in ood_records],
        train_records=train_records,
        eval_records=eval_records,
        ood_records=ood_records,
    )


def sample_eval_video_files(
    files: list[str], size: int, data_seed: int, step: int, split_name: str,
) -> list[str]:
    if size < 0 or size == 0:
        return list(files)
    if size > len(files):
        raise ValueError(
            f"Requested {split_name} eval video size {size}, but only {len(files)} eval samples are available"
        )
    sampled = list(files)
    random.Random(f"{data_seed}:video:{split_name}:{step}").shuffle(sampled)
    return sorted(sampled[:size])


def write_runtime_split(run_dir: Path, args, split: RuntimeSplit) -> None:
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
        "data_type": args.data_type,
        "duration": args.duration,
        "train_tasks": parse_task_list(args.train_tasks),
        "ood_tasks": parse_task_list(args.ood_tasks, allow_empty=True),
        "train_size": args.train_size,
        "in_task_eval_size": args.in_task_eval_size,
        "in_task_video_size": args.in_task_video_size,
        "ood_eval_size": args.ood_eval_size,
        "ood_video_size": args.ood_video_size,
        "data_seed": args.data_seed,
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True))
