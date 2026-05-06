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
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from src.core.config import MAIN_ROOT

DATA_TYPES = {"identity_r2r", "blur_r2r", "h2r", "r2h"}
PAIR_ORDER_FILENAME = "pair_order.jsonl"


@dataclass(frozen=True)
class RuntimeSplit:
    train_files: list[str]
    eval_files: list[str]
    ood_files: list[str]
    train_records: list[dict]
    eval_records: list[dict]
    ood_records: list[dict]
    split_counts: dict[str, dict[str, int]]
    order_paths: dict[str, str]


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


def _pair_id_from_path(value: str) -> str:
    if not value:
        raise ValueError("Pair/cache path is empty")
    return Path(value).stem


def _pair_id_from_pair_record(record: dict, task_dir: Path) -> str:
    raw_path = record.get("video") or record.get("cache_path")
    if not raw_path:
        raise ValueError(f"Pair manifest record missing video in {task_dir}")
    return _pair_id_from_path(str(raw_path))


def _pair_id_from_cache_record(record: dict, task_dir: Path) -> str:
    raw_path = record.get("cache_path")
    if not raw_path:
        raise ValueError(f"Cache manifest record missing cache_path in {task_dir}")
    return _pair_id_from_path(str(raw_path))


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


def _validate_manifest_record(
    record: dict, task_dir: Path, data_type: str, duration: str, task: str,
) -> None:
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


def _load_cache_task_records(root: Path, data_type: str, duration: str, task: str) -> list[dict]:
    task_dir = root / task
    rows = _read_jsonl(task_dir / "manifest.jsonl")
    records = []
    for row in rows:
        record = dict(row)
        _validate_manifest_record(record, task_dir, data_type, duration, task)
        record["data_type"] = data_type
        record["duration"] = duration
        record["robot_task"] = task
        record["cache_path"] = _resolve_cache_path(task_dir, record)
        record["pair_id"] = _pair_id_from_cache_record(record, task_dir)
        record.setdefault("source_segment_id", record.get("source_id", record["cache_path"]))
        records.append(record)
    pair_ids = [record["pair_id"] for record in records]
    duplicates = sorted(pair_id for pair_id, n in Counter(pair_ids).items() if n > 1)
    if duplicates:
        raise ValueError(f"Duplicate cache pair_id values in {task_dir}: {duplicates[:5]}")
    return sorted(records, key=lambda record: record["pair_id"])


def _load_pair_manifest_records(
    pair_root: Path, data_type: str, duration: str, task: str,
) -> tuple[Path, list[dict]]:
    task_dir = pair_root / data_type / duration / task
    rows = _read_jsonl(task_dir / "manifest.jsonl")
    records = []
    for row in rows:
        record = dict(row)
        _validate_manifest_record(record, task_dir, data_type, duration, task)
        record["data_type"] = data_type
        record["duration"] = duration
        record["robot_task"] = task
        record["pair_id"] = _pair_id_from_pair_record(record, task_dir)
        record.setdefault("source_segment_id", record.get("source_id", record["pair_id"]))
        records.append(record)
    pair_ids = [record["pair_id"] for record in records]
    duplicates = sorted(pair_id for pair_id, n in Counter(pair_ids).items() if n > 1)
    if duplicates:
        raise ValueError(f"Duplicate pair_id values in {task_dir}: {duplicates[:5]}")
    return task_dir, sorted(records, key=lambda record: record["pair_id"])


def _pair_order_entry(record: dict, index: int, seed: int) -> dict:
    return {
        "order_index": index,
        "pair_id": record["pair_id"],
        "source_id": record.get("source_id", record["pair_id"]),
        "source_segment_id": record.get("source_segment_id", ""),
        "video": record.get("video", ""),
        "control_video": record.get("control_video", ""),
        "data_type": record["data_type"],
        "duration": record["duration"],
        "robot_task": record["robot_task"],
        "order_seed": seed,
    }


def _write_order_table_once(order_path: Path, entries: list[dict]) -> None:
    tmp_path = order_path.with_name(f".{order_path.name}.{os.getpid()}.tmp")
    with tmp_path.open("w") as fh:
        for entry in entries:
            fh.write(json.dumps(entry, sort_keys=True) + "\n")
    try:
        os.link(tmp_path, order_path)
    except FileExistsError:
        pass
    finally:
        tmp_path.unlink(missing_ok=True)


def _read_or_create_pair_order(
    pair_dir: Path,
    pair_records: list[dict],
    data_seed: int,
    data_type: str,
    duration: str,
    task: str,
) -> tuple[Path, list[dict]]:
    order_path = pair_dir / PAIR_ORDER_FILENAME
    if not order_path.exists():
        shuffled = list(pair_records)
        random.Random(f"{data_seed}:pair_order:{data_type}:{duration}:{task}").shuffle(shuffled)
        entries = [
            _pair_order_entry(record, index, data_seed)
            for index, record in enumerate(shuffled)
        ]
        _write_order_table_once(order_path, entries)

    entries = _read_jsonl(order_path)
    pair_ids = [str(entry.get("pair_id", "")) for entry in entries]
    if any(not pair_id for pair_id in pair_ids):
        raise ValueError(f"Pair order table has empty pair_id values: {order_path}")
    duplicates = sorted(pair_id for pair_id, n in Counter(pair_ids).items() if n > 1)
    if duplicates:
        raise ValueError(f"Duplicate pair_id values in {order_path}: {duplicates[:5]}")
    expected_pair_ids = {record["pair_id"] for record in pair_records}
    actual_pair_ids = set(pair_ids)
    if actual_pair_ids != expected_pair_ids:
        missing = sorted(expected_pair_ids - actual_pair_ids)
        extra = sorted(actual_pair_ids - expected_pair_ids)
        raise ValueError(
            f"Pair order table does not match manifest: {order_path}; "
            f"missing={missing[:5]} extra={extra[:5]}"
        )
    return order_path, entries


def _load_ordered_task_records(
    cache_root: Path,
    pair_root: Path,
    data_type: str,
    duration: str,
    task: str,
    data_seed: int,
) -> tuple[list[dict], Path]:
    pair_dir, pair_records = _load_pair_manifest_records(pair_root, data_type, duration, task)
    order_path, order_entries = _read_or_create_pair_order(
        pair_dir, pair_records, data_seed, data_type, duration, task,
    )
    cache_records = _load_cache_task_records(cache_root, data_type, duration, task)
    cache_by_pair = {record["pair_id"]: record for record in cache_records}
    order_pair_ids = [entry["pair_id"] for entry in order_entries]
    missing = sorted(set(order_pair_ids) - set(cache_by_pair))
    extra = sorted(set(cache_by_pair) - set(order_pair_ids))
    if missing or extra:
        raise ValueError(
            f"Cache manifest does not match pair order for {task}: "
            f"missing_cache={missing[:5]} extra_cache={extra[:5]}"
        )
    ordered_records = []
    for order_index, entry in enumerate(order_entries):
        record = dict(cache_by_pair[entry["pair_id"]])
        record["order_index"] = order_index
        record["pair_order_path"] = str(order_path)
        ordered_records.append(record)
    return ordered_records, order_path


def _auto_in_task_eval_size(total: int) -> int:
    if total <= 1:
        raise ValueError("In-task data needs at least 2 samples for train/eval separation")
    return min(max(1, int(total * 0.1)), total - 1)


def _allocate_counts(capacity_by_task: dict[str, int], size: int, label: str) -> dict[str, int]:
    total = sum(capacity_by_task.values())
    if size < 0 or size == 0:
        return dict(capacity_by_task)
    if size > total:
        raise ValueError(
            f"Requested {label} size {size}, but only {total} samples are available"
        )
    if total == 0:
        raise ValueError(f"No samples are available for {label}")
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
        raise RuntimeError(f"Unable to allocate {remaining} {label} samples")
    return counts


def _task_counts(records: list[dict]) -> dict[str, int]:
    return dict(sorted(Counter(record["robot_task"] for record in records).items()))


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
    pair_root = getattr(args, "pair_root", "") or str(
        Path(MAIN_ROOT) / "training_data" / "pair"
    )
    cache_task_root = Path(cache_root) / data_type / duration
    pair_root_path = Path(pair_root)
    records_by_task: dict[str, list[dict]] = {}
    order_paths: dict[str, str] = {}

    for task in train_tasks:
        records, order_path = _load_ordered_task_records(
            cache_task_root, pair_root_path, data_type, duration, task, args.data_seed,
        )
        records_by_task[task] = records
        order_paths[task] = str(order_path)

    for task in ood_tasks:
        records, order_path = _load_ordered_task_records(
            cache_task_root, pair_root_path, data_type, duration, task, args.data_seed,
        )
        records_by_task[task] = records
        order_paths[task] = str(order_path)

    train_task_sizes = {task: len(records_by_task[task]) for task in train_tasks}
    in_task_total = sum(train_task_sizes.values())
    if args.in_task_eval_size < 0 or args.in_task_eval_size == 0:
        in_task_eval_size = _auto_in_task_eval_size(in_task_total)
    else:
        in_task_eval_size = args.in_task_eval_size
    if in_task_eval_size >= in_task_total:
        raise ValueError(
            f"Requested in-task eval size {in_task_eval_size}, but only "
            f"{in_task_total} in-task samples are available; at least one sample "
            "must remain available for training"
        )
    eval_counts = _allocate_counts(
        train_task_sizes, in_task_eval_size, "in-task eval",
    )
    train_capacity = {
        task: train_task_sizes[task] - eval_counts[task]
        for task in train_tasks
    }
    train_counts = _allocate_counts(train_capacity, args.train_size, "train")

    train_records: list[dict] = []
    eval_records: list[dict] = []
    for task in train_tasks:
        task_records = records_by_task[task]
        n_eval = eval_counts[task]
        n_train = train_counts[task]
        train_limit = len(task_records) - n_eval
        train_records.extend(task_records[:n_train])
        eval_records.extend(task_records[train_limit:])

    ood_records: list[dict] = []
    if ood_tasks:
        ood_task_sizes = {task: len(records_by_task[task]) for task in ood_tasks}
        ood_counts = _allocate_counts(ood_task_sizes, args.ood_eval_size, "OOD eval")
        for task in ood_tasks:
            n_ood = ood_counts[task]
            if n_ood:
                ood_records.extend(records_by_task[task][-n_ood:])

    return RuntimeSplit(
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
    random.Random(f"{data_seed}:video:{split_name}").shuffle(sampled)
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
        "pair_root": getattr(args, "pair_root", ""),
        "split_counts": split.split_counts,
        "pair_order_paths": split.order_paths,
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True))
