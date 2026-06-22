"""Runtime data selection for FLIP training.

Two split sources are supported:
  - runtime: split-free per-task manifests plus deterministic pair_order tails.
  - explicit: root-level train/eval/test JSONL files, used by human2robot.

This module writes the exact train/eval/test selection into each run directory.
"""

from __future__ import annotations

import json
import os
import random
from collections import Counter
from dataclasses import dataclass, field
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
    test_files: list[str] = field(default_factory=list)
    test_records: list[dict] = field(default_factory=list)
    split_source: str = "runtime"
    split_root: str = ""


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


def _tail_percent_count(total: int, percent: float, label: str) -> int:
    if total <= 0:
        raise ValueError(f"No samples are available for {label}")
    if percent <= 0.0 or percent > 100.0:
        raise ValueError(f"{label} tail percent must be in (0, 100], got {percent}")
    count = int(total * percent / 100.0)
    if count * 100.0 < total * percent:
        count += 1
    return min(max(1, count), total)


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


def _allocate_counts_balanced(
    capacity_by_task: dict[str, int],
    size: int,
    label: str,
    seed_key: str,
) -> dict[str, int]:
    total = sum(capacity_by_task.values())
    if size < 0 or size == 0:
        return dict(capacity_by_task)
    if size > total:
        raise ValueError(
            f"Requested {label} size {size}, but only {total} samples are available"
        )
    if total == 0:
        raise ValueError(f"No samples are available for {label}")

    counts = {task: 0 for task in capacity_by_task}
    tasks = sorted(capacity_by_task)
    random.Random(seed_key).shuffle(tasks)
    remaining = size
    while remaining:
        progressed = False
        for task in tasks:
            if remaining == 0:
                break
            if counts[task] >= capacity_by_task[task]:
                continue
            counts[task] += 1
            remaining -= 1
            progressed = True
        if not progressed:
            raise RuntimeError(f"Unable to allocate {remaining} {label} samples")
    return counts


def _task_counts(records: list[dict]) -> dict[str, int]:
    return dict(sorted(Counter(record["robot_task"] for record in records).items()))


def _split_row_task(row: dict, split_path: Path) -> str:
    raw_task = row.get("robot_task", row.get("task", ""))
    task = short_task_name(str(raw_task))
    if not task:
        raise ValueError(f"Explicit split row missing robot_task/task in {split_path}")
    return task


def _split_row_pair_id(row: dict, split_path: Path) -> str:
    pair_id = str(row.get("pair_id", "")).strip()
    if pair_id:
        return pair_id
    raw_path = row.get("video") or row.get("cache_path")
    if not raw_path:
        raise ValueError(f"Explicit split row missing pair_id/video in {split_path}")
    return _pair_id_from_path(str(raw_path))


def _validate_split_row(
    row: dict,
    split_path: Path,
    expected_split: str,
    data_type: str,
    duration: str,
) -> tuple[str, str]:
    record_split = row.get("split", expected_split)
    if record_split != expected_split:
        raise ValueError(
            f"Explicit split mismatch in {split_path}: {record_split} != {expected_split}"
        )
    record_data_type = row.get("data_type", data_type)
    record_duration = row.get("duration", duration)
    if record_data_type != data_type:
        raise ValueError(
            f"Explicit split data_type mismatch in {split_path}: "
            f"{record_data_type} != {data_type}"
        )
    if record_duration != duration:
        raise ValueError(
            f"Explicit split duration mismatch in {split_path}: "
            f"{record_duration} != {duration}"
        )
    return _split_row_task(row, split_path), _split_row_pair_id(row, split_path)


def _resolve_split_root(args, pair_root_path: Path, data_type: str, duration: str) -> Path:
    raw_root = getattr(args, "split_root", "")
    if raw_root:
        return Path(raw_root)
    return pair_root_path / data_type / duration


def _load_explicit_split_rows(
    split_root: Path,
    split_name: str,
    data_type: str,
    duration: str,
) -> list[dict]:
    split_path = split_root / f"{split_name}.jsonl"
    rows = _read_jsonl(split_path)
    seen: set[tuple[str, str]] = set()
    out = []
    for row in rows:
        record = dict(row)
        task, pair_id = _validate_split_row(
            record, split_path, split_name, data_type, duration,
        )
        key = (task, pair_id)
        if key in seen:
            raise ValueError(
                f"Duplicate explicit split pair in {split_path}: {task}/{pair_id}"
            )
        seen.add(key)
        record["data_type"] = data_type
        record["duration"] = duration
        record["robot_task"] = task
        record["pair_id"] = pair_id
        record["split"] = split_name
        record.setdefault("source_segment_id", record.get("source_id", pair_id))
        out.append(record)
    return out


def _load_cache_records_for_split(
    cache_task_root: Path,
    data_type: str,
    duration: str,
    tasks: Iterable[str],
) -> dict[tuple[str, str], dict]:
    cache_by_key: dict[tuple[str, str], dict] = {}
    for task in sorted(set(tasks)):
        task_records = _load_cache_task_records(cache_task_root, data_type, duration, task)
        for record in task_records:
            key = (task, record["pair_id"])
            if key in cache_by_key:
                raise ValueError(f"Duplicate cache pair for explicit split: {task}/{record['pair_id']}")
            cache_by_key[key] = record
    return cache_by_key


def _attach_cache_records(
    rows: list[dict],
    cache_by_key: dict[tuple[str, str], dict],
    split_name: str,
    split_root: Path,
) -> list[dict]:
    records = []
    for row in rows:
        key = (row["robot_task"], row["pair_id"])
        if key not in cache_by_key:
            raise FileNotFoundError(
                f"Explicit {split_name} split references uncached pair "
                f"{row['robot_task']}/{row['pair_id']} under {split_root}"
            )
        cache_record = cache_by_key[key]
        cache_split = cache_record.get("split")
        if cache_split and cache_split != split_name:
            raise ValueError(
                f"Cache split mismatch for {row['robot_task']}/{row['pair_id']}: "
                f"{cache_split} != {split_name}"
            )
        record = dict(cache_record)
        for field_name, value in row.items():
            if field_name != "cache_path":
                record[field_name] = value
        record["cache_path"] = cache_record["cache_path"]
        records.append(record)
    return records


def _records_by_task(records: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for record in records:
        out.setdefault(record["robot_task"], []).append(record)
    return dict(sorted(out.items()))


def _select_balanced_records(
    records: list[dict],
    size: int,
    label: str,
    seed_key: str,
) -> list[dict]:
    records_by_task = _records_by_task(records)
    capacities = {task: len(task_records) for task, task_records in records_by_task.items()}
    counts = _allocate_counts_balanced(capacities, size, label, seed_key)
    selected = []
    for task, task_records in records_by_task.items():
        count = counts[task]
        if not count:
            continue
        shuffled = list(task_records)
        random.Random(f"{seed_key}:{task}").shuffle(shuffled)
        selected.extend(shuffled[:count])
    return selected


def build_explicit_runtime_split(args) -> RuntimeSplit:
    data_type = args.data_type
    duration = args.duration
    if data_type not in DATA_TYPES:
        raise ValueError(f"Unknown data type '{data_type}'. Available: {sorted(DATA_TYPES)}")

    cache_root = getattr(args, "cache_root", "") or str(
        Path(MAIN_ROOT) / "training_data" / "cache" / "vae"
    )
    pair_root = getattr(args, "pair_root", "") or str(
        Path(MAIN_ROOT) / "training_data" / "pair"
    )
    cache_task_root = Path(cache_root) / data_type / duration
    pair_root_path = Path(pair_root)
    split_root = _resolve_split_root(args, pair_root_path, data_type, duration)

    train_rows = _load_explicit_split_rows(split_root, "train", data_type, duration)
    eval_rows = _load_explicit_split_rows(split_root, "eval", data_type, duration)
    test_rows = _load_explicit_split_rows(split_root, "test", data_type, duration)

    all_rows = [*train_rows, *eval_rows, *test_rows]
    row_keys = [(row["robot_task"], row["pair_id"]) for row in all_rows]
    duplicate_keys = [
        key for key, count in Counter(row_keys).items()
        if count > 1
    ]
    if duplicate_keys:
        first = duplicate_keys[0]
        raise ValueError(
            f"Explicit train/eval/test splits overlap at {first[0]}/{first[1]}"
        )

    tasks = [row["robot_task"] for row in all_rows]
    cache_by_key = _load_cache_records_for_split(
        cache_task_root, data_type, duration, tasks,
    )
    train_pool = _attach_cache_records(train_rows, cache_by_key, "train", split_root)
    eval_pool = _attach_cache_records(eval_rows, cache_by_key, "eval", split_root)
    test_records = _attach_cache_records(test_rows, cache_by_key, "test", split_root)

    train_records = _select_balanced_records(
        train_pool,
        args.train_size,
        "explicit train",
        f"{args.data_seed}:explicit_train:{data_type}:{duration}",
    )

    eval_records = []
    ood_records = []
    for record in eval_pool:
        eval_role = str(record.get("eval_role", "")).strip()
        domain = str(record.get("domain", "")).strip()
        if eval_role == "ood" or domain == "ood":
            ood_records.append(record)
        elif eval_role == "in_task" or domain == "id":
            eval_records.append(record)
        else:
            raise ValueError(
                f"Explicit eval row must set eval_role=in_task/ood or domain=id/ood: "
                f"{record['robot_task']}/{record['pair_id']}"
            )

    eval_records = _select_balanced_records(
        eval_records,
        args.in_task_eval_size,
        "explicit in-task eval",
        f"{args.data_seed}:explicit_eval_in_task:{data_type}:{duration}",
    )
    ood_records = _select_balanced_records(
        ood_records,
        args.ood_eval_size,
        "explicit OOD eval",
        f"{args.data_seed}:explicit_eval_ood:{data_type}:{duration}",
    )

    return RuntimeSplit(
        train_files=[record["cache_path"] for record in train_records],
        eval_files=[record["cache_path"] for record in eval_records],
        ood_files=[record["cache_path"] for record in ood_records],
        train_records=train_records,
        eval_records=eval_records,
        ood_records=ood_records,
        test_files=[record["cache_path"] for record in test_records],
        test_records=test_records,
        split_counts={
            "train": _task_counts(train_records),
            "in_task_eval": _task_counts(eval_records),
            "ood_eval": _task_counts(ood_records),
            "test": _task_counts(test_records),
        },
        order_paths={},
        split_source="explicit",
        split_root=str(split_root),
    )


def build_runtime_split(args) -> RuntimeSplit:
    split_source = getattr(args, "split_source", "") or "runtime"
    if split_source == "explicit":
        return build_explicit_runtime_split(args)
    if split_source != "runtime":
        raise ValueError(f"Unknown split source '{split_source}'")

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


def build_tail_eval_split(args) -> RuntimeSplit:
    """Select eval records from the tail of each task-level pair order table."""
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

    eval_records: list[dict] = []
    for task in train_tasks:
        task_records = records_by_task[task]
        n_eval = _tail_percent_count(
            len(task_records), args.eval_tail_percent, f"in-task eval for {task}",
        )
        eval_records.extend(task_records[-n_eval:])

    ood_records: list[dict] = []
    for task in ood_tasks:
        task_records = records_by_task[task]
        n_ood = _tail_percent_count(
            len(task_records), args.eval_tail_percent, f"OOD eval for {task}",
        )
        ood_records.extend(task_records[-n_ood:])

    return RuntimeSplit(
        train_files=[],
        eval_files=[record["cache_path"] for record in eval_records],
        ood_files=[record["cache_path"] for record in ood_records],
        train_records=[],
        eval_records=eval_records,
        ood_records=ood_records,
        split_counts={
            "train": {},
            "in_task_eval": _task_counts(eval_records),
            "ood_eval": _task_counts(ood_records),
        },
        order_paths=order_paths,
    )


def build_count_eval_split(args) -> RuntimeSplit:
    """Select fixed-size eval records from task-level pair order table tails.

    In-task counts are allocated across train tasks in proportion to each
    task's available pair count. OOD counts use the same allocator, which
    reduces to taking the requested tail count when a single OOD task is used.
    """
    data_type = args.data_type
    duration = args.duration
    if data_type not in DATA_TYPES:
        raise ValueError(f"Unknown data type '{data_type}'. Available: {sorted(DATA_TYPES)}")
    train_tasks = parse_task_list(args.train_tasks)
    ood_tasks = parse_task_list(args.ood_tasks, allow_empty=True)
    overlap = sorted(set(train_tasks) & set(ood_tasks))
    if overlap:
        raise ValueError(f"Tasks cannot be both train and OOD: {overlap}")

    if args.in_task_eval_size is None and args.ood_eval_size is None:
        raise ValueError("Count eval split requires at least one explicit eval size")
    if args.in_task_eval_size is not None and args.in_task_eval_size <= 0:
        raise ValueError(
            f"--in-task-eval-size must be positive when set, got {args.in_task_eval_size}"
        )
    if args.ood_eval_size is not None and args.ood_eval_size <= 0:
        raise ValueError(
            f"--ood-eval-size must be positive when set, got {args.ood_eval_size}"
        )

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

    eval_records: list[dict] = []
    if args.in_task_eval_size is None:
        for task in train_tasks:
            task_records = records_by_task[task]
            n_eval = _tail_percent_count(
                len(task_records), args.eval_tail_percent, f"in-task eval for {task}",
            )
            eval_records.extend(task_records[-n_eval:])
    else:
        train_task_sizes = {task: len(records_by_task[task]) for task in train_tasks}
        eval_counts = _allocate_counts(
            train_task_sizes, args.in_task_eval_size, "in-task eval",
        )
        for task in train_tasks:
            n_eval = eval_counts[task]
            if n_eval:
                eval_records.extend(records_by_task[task][-n_eval:])

    ood_records: list[dict] = []
    if ood_tasks:
        if args.ood_eval_size is None:
            for task in ood_tasks:
                task_records = records_by_task[task]
                n_ood = _tail_percent_count(
                    len(task_records), args.eval_tail_percent, f"OOD eval for {task}",
                )
                ood_records.extend(task_records[-n_ood:])
        else:
            ood_task_sizes = {task: len(records_by_task[task]) for task in ood_tasks}
            ood_counts = _allocate_counts(ood_task_sizes, args.ood_eval_size, "OOD eval")
            for task in ood_tasks:
                n_ood = ood_counts[task]
                if n_ood:
                    ood_records.extend(records_by_task[task][-n_ood:])

    return RuntimeSplit(
        train_files=[],
        eval_files=[record["cache_path"] for record in eval_records],
        ood_files=[record["cache_path"] for record in ood_records],
        train_records=[],
        eval_records=eval_records,
        ood_records=ood_records,
        split_counts={
            "train": {},
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
    if split.test_records:
        payloads["test.jsonl"] = split.test_records
    for name, records in payloads.items():
        with (out_dir / name).open("w") as fh:
            for record in records:
                fh.write(json.dumps(record, sort_keys=True) + "\n")
    config = {
        "data_type": args.data_type,
        "duration": args.duration,
        "train_tasks": (
            sorted(split.split_counts.get("train", {}))
            if split.split_source == "explicit"
            else parse_task_list(args.train_tasks)
        ),
        "ood_tasks": (
            sorted(split.split_counts.get("ood_eval", {}))
            if split.split_source == "explicit"
            else parse_task_list(args.ood_tasks, allow_empty=True)
        ),
        "train_size": args.train_size,
        "in_task_eval_size": args.in_task_eval_size,
        "in_task_video_size": args.in_task_video_size,
        "ood_eval_size": args.ood_eval_size,
        "ood_video_size": args.ood_video_size,
        "data_seed": args.data_seed,
        "pair_root": getattr(args, "pair_root", ""),
        "split_source": split.split_source,
        "split_root": split.split_root,
        "split_counts": split.split_counts,
        "pair_order_paths": split.order_paths,
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True))
