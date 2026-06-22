"""Derive a human-to-human identity cache from an r2h cache.

The r2h cache already stores the target human video latent in ``human_latent``.
For a human identity stage we reuse that latent on both Mitty sides:

  human_latent = source human_latent
  robot_latent = source human_latent

The output keeps the existing training data layout and uses ``identity_r2r`` as
the data type because the runtime trainer currently has no separate ``h2h``
data type.  The role metadata marks the actual semantics as human -> human.
"""

from __future__ import annotations

import argparse
import csv
import errno
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch

from src.core.config import MAIN_ROOT
from src.pipeline.runtime_data import short_task_name


DEFAULT_SOURCE_DATA_TYPE = "r2h"
DEFAULT_TARGET_DATA_TYPE = "identity_r2r"
DEFAULT_SOURCE_DURATION = "2s61f30_human2robot_hdf5_2x_v1"
DEFAULT_TARGET_DURATION = "2s61f30_human2robot_hdf5_2x_human_identity_v1"
DEFAULT_PROMPT = "A first-person view robot arm performing household tasks flip_v2v"


@dataclass(frozen=True)
class PairSpec:
    task: str
    pair_id: str
    source_pair_dir: Path
    source_cache_dir: Path
    target_pair_dir: Path
    target_cache_dir: Path
    source_pair_record: dict
    source_cache_record: dict
    target_pair_record: dict
    target_cache_record: dict


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Required JSONL file not found: {path}")
    rows: list[dict] = []
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
        raise ValueError(f"JSONL file is empty: {path}")
    return rows


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")


def _pair_id(record: dict) -> str:
    if record.get("pair_id"):
        return str(record["pair_id"])
    raw_path = record.get("video") or record.get("cache_path")
    if not raw_path:
        raise ValueError(f"Record has no pair_id/video/cache_path: {record}")
    return Path(str(raw_path)).stem


def _resolve_relative_file(base: Path, value: str, label: str) -> Path:
    if not value:
        raise ValueError(f"{label} path is empty in {base}")
    path = Path(value)
    if not path.is_absolute():
        path = base / path
    if not path.is_file():
        raise FileNotFoundError(f"{label} file not found: {path}")
    return path


def _install_same_file(src: Path, dst: Path, *, overwrite: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        if not overwrite:
            return
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError as exc:
        if exc.errno != errno.EXDEV:
            raise
        shutil.copy2(src, dst)


def _relative_task_dirs(root: Path) -> list[str]:
    if not root.is_dir():
        raise NotADirectoryError(f"Source data root does not exist: {root}")
    tasks = [
        str(path.parent.relative_to(root))
        for path in root.rglob("manifest.jsonl")
    ]
    return sorted(short_task_name(task) for task in tasks)


def _parse_tasks(value: str, source_pair_root: Path) -> list[str]:
    if not value or value.strip() == "all":
        return _relative_task_dirs(source_pair_root)
    return [
        short_task_name(item.strip())
        for item in value.split(",")
        if item.strip()
    ]


def _target_pair_record(
    source_record: dict,
    *,
    task: str,
    pair_id: str,
    source_data_type: str,
    source_duration: str,
    target_data_type: str,
    target_duration: str,
) -> dict:
    record = dict(source_record)
    record["data_type"] = target_data_type
    record["duration"] = target_duration
    record["robot_task"] = task
    record["task"] = task
    record["pair_id"] = pair_id
    record["video"] = f"video/{pair_id}.mp4"
    record["control_video"] = f"control_video/{pair_id}.mp4"
    record["input_role"] = "human"
    record["target_role"] = "human"
    record["identity_mapping"] = "human_to_human"
    record["identity_source_data_type"] = source_data_type
    record["identity_source_duration"] = source_duration
    record["prompt"] = source_record.get("prompt") or DEFAULT_PROMPT
    return record


def _target_cache_record(
    pair_record: dict,
    source_cache_record: dict,
    *,
    target_pair_dir: Path,
) -> dict:
    record = dict(pair_record)
    for key, value in source_cache_record.items():
        if key not in record:
            record[key] = value
    record["data_type"] = pair_record["data_type"]
    record["duration"] = pair_record["duration"]
    record["robot_task"] = pair_record["robot_task"]
    record["task"] = pair_record["task"]
    record["pair_id"] = pair_record["pair_id"]
    record["cache_path"] = f"{pair_record['pair_id']}.pth"
    record["pair_dir"] = str(target_pair_dir)
    record["input_role"] = "human"
    record["target_role"] = "human"
    record["identity_mapping"] = "human_to_human"
    return record


def _load_task_specs(
    *,
    task: str,
    source_pair_root: Path,
    source_cache_root: Path,
    target_pair_root: Path,
    target_cache_root: Path,
    source_data_type: str,
    source_duration: str,
    target_data_type: str,
    target_duration: str,
) -> list[PairSpec]:
    source_pair_dir = source_pair_root / task
    source_cache_dir = source_cache_root / task
    target_pair_dir = target_pair_root / task
    target_cache_dir = target_cache_root / task

    pair_rows = _read_jsonl(source_pair_dir / "manifest.jsonl")
    cache_rows = _read_jsonl(source_cache_dir / "manifest.jsonl")
    cache_by_id = {_pair_id(row): row for row in cache_rows}
    if len(cache_by_id) != len(cache_rows):
        raise ValueError(f"Duplicate cache pair IDs in {source_cache_dir}")

    specs: list[PairSpec] = []
    seen: set[str] = set()
    for pair_row in pair_rows:
        pair_id = _pair_id(pair_row)
        if pair_id in seen:
            raise ValueError(f"Duplicate pair ID {pair_id} in {source_pair_dir}")
        seen.add(pair_id)
        if pair_id not in cache_by_id:
            raise ValueError(
                f"Missing source cache for pair_id={pair_id} in {source_cache_dir}"
            )
        target_pair = _target_pair_record(
            pair_row,
            task=task,
            pair_id=pair_id,
            source_data_type=source_data_type,
            source_duration=source_duration,
            target_data_type=target_data_type,
            target_duration=target_duration,
        )
        target_cache = _target_cache_record(
            target_pair, cache_by_id[pair_id], target_pair_dir=target_pair_dir,
        )
        specs.append(
            PairSpec(
                task=task,
                pair_id=pair_id,
                source_pair_dir=source_pair_dir,
                source_cache_dir=source_cache_dir,
                target_pair_dir=target_pair_dir,
                target_cache_dir=target_cache_dir,
                source_pair_record=pair_row,
                source_cache_record=cache_by_id[pair_id],
                target_pair_record=target_pair,
                target_cache_record=target_cache,
            )
        )
    extra_cache = sorted(set(cache_by_id) - seen)
    if extra_cache:
        raise ValueError(
            f"Source cache has rows absent from pair manifest in {task}: "
            f"{extra_cache[:5]}"
        )
    return specs


def _write_pair_files(spec: PairSpec, *, overwrite: bool) -> None:
    source_human_video = _resolve_relative_file(
        spec.source_pair_dir, str(spec.source_pair_record["video"]), "source human video",
    )
    target_video = spec.target_pair_dir / str(spec.target_pair_record["video"])
    target_control = spec.target_pair_dir / str(spec.target_pair_record["control_video"])
    _install_same_file(source_human_video, target_video, overwrite=overwrite)
    _install_same_file(source_human_video, target_control, overwrite=overwrite)


def _write_cache_file(spec: PairSpec, *, overwrite: bool) -> None:
    target_cache_path = spec.target_cache_dir / f"{spec.pair_id}.pth"
    if target_cache_path.exists() and not overwrite:
        return
    target_cache_path.parent.mkdir(parents=True, exist_ok=True)
    source_cache_path = _resolve_relative_file(
        spec.source_cache_dir,
        str(spec.source_cache_record.get("cache_path", f"{spec.pair_id}.pth")),
        "source cache",
    )
    source_cache = torch.load(str(source_cache_path), map_location="cpu")
    if "human_latent" not in source_cache:
        raise KeyError(f"human_latent missing in source cache: {source_cache_path}")
    human_latent = source_cache["human_latent"]
    if human_latent.ndim != 5 or human_latent.shape[0] != 1:
        raise ValueError(
            f"Unexpected human_latent shape in {source_cache_path}: "
            f"{tuple(human_latent.shape)}"
        )

    target_cache = dict(source_cache)
    target_cache["human_latent"] = human_latent.clone()
    target_cache["robot_latent"] = human_latent.clone()
    for key, value in spec.target_cache_record.items():
        if key in {
            "cache_path",
            "control_video",
            "pair_dir",
            "video",
        }:
            continue
        target_cache[key] = value
    torch.save(target_cache, str(target_cache_path))


def _write_task_tables(task: str, specs: list[PairSpec]) -> None:
    if not specs:
        raise ValueError(f"No specs for task {task}")
    pair_dir = specs[0].target_pair_dir
    cache_dir = specs[0].target_cache_dir
    pair_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    pair_rows = [spec.target_pair_record for spec in specs]
    cache_rows = [spec.target_cache_record for spec in specs]
    _write_jsonl(pair_dir / "manifest.jsonl", pair_rows)
    _write_jsonl(cache_dir / "manifest.jsonl", cache_rows)

    with (pair_dir / "metadata.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["video", "prompt", "control_video"])
        writer.writeheader()
        for row in pair_rows:
            writer.writerow(
                {
                    "video": row["video"],
                    "prompt": row["prompt"],
                    "control_video": row["control_video"],
                }
            )


def _copy_t5_cache(source_dir: Path, target_dir: Path, *, overwrite: bool) -> int:
    if not source_dir.is_dir():
        raise NotADirectoryError(f"T5 source dir does not exist: {source_dir}")
    target_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for src in sorted(path for path in source_dir.iterdir() if path.is_file()):
        dst = target_dir / src.name
        _install_same_file(src, dst, overwrite=overwrite)
        copied += 1
    return copied


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Derive human identity pair/cache data from an r2h VAE cache."
    )
    parser.add_argument("--pair-root", default="training_data/pair")
    parser.add_argument("--cache-root", default="training_data/cache/vae")
    parser.add_argument("--source-data-type", default=DEFAULT_SOURCE_DATA_TYPE)
    parser.add_argument("--source-duration", default=DEFAULT_SOURCE_DURATION)
    parser.add_argument("--target-data-type", default=DEFAULT_TARGET_DATA_TYPE)
    parser.add_argument("--target-duration", default=DEFAULT_TARGET_DURATION)
    parser.add_argument("--tasks", default="all")
    parser.add_argument("--source-t5-dir", default="")
    parser.add_argument("--target-t5-dir", default="")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _resolve_root(path_value: str) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = Path(MAIN_ROOT) / path
    return path.resolve()


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")
    if args.target_data_type != DEFAULT_TARGET_DATA_TYPE:
        raise ValueError(
            "The training runtime only supports this derived identity cache via "
            f"{DEFAULT_TARGET_DATA_TYPE!r}; got {args.target_data_type!r}"
        )

    pair_root = _resolve_root(args.pair_root)
    cache_root = _resolve_root(args.cache_root)
    source_pair_root = pair_root / args.source_data_type / args.source_duration
    source_cache_root = cache_root / args.source_data_type / args.source_duration
    target_pair_root = pair_root / args.target_data_type / args.target_duration
    target_cache_root = cache_root / args.target_data_type / args.target_duration
    source_t5_dir = _resolve_root(
        args.source_t5_dir
        or f"training_data/cache/t5/{args.source_data_type}/{args.source_duration}"
    )
    target_t5_dir = _resolve_root(
        args.target_t5_dir
        or f"training_data/cache/t5/{args.target_data_type}/{args.target_duration}"
    )

    tasks = _parse_tasks(args.tasks, source_pair_root)
    if not tasks:
        raise ValueError(f"No tasks found under {source_pair_root}")

    specs_by_task: dict[str, list[PairSpec]] = {}
    for task in tasks:
        specs_by_task[task] = _load_task_specs(
            task=task,
            source_pair_root=source_pair_root,
            source_cache_root=source_cache_root,
            target_pair_root=target_pair_root,
            target_cache_root=target_cache_root,
            source_data_type=args.source_data_type,
            source_duration=args.source_duration,
            target_data_type=args.target_data_type,
            target_duration=args.target_duration,
        )
    all_specs = [spec for specs in specs_by_task.values() for spec in specs]

    print(f"Source pair:  {source_pair_root}")
    print(f"Source cache: {source_cache_root}")
    print(f"Target pair:  {target_pair_root}")
    print(f"Target cache: {target_cache_root}")
    print(f"Source T5:    {source_t5_dir}")
    print(f"Target T5:    {target_t5_dir}")
    print(f"Tasks:        {len(tasks)}")
    print(f"Pairs:        {len(all_specs)}")
    print(f"Workers:      {args.workers}")
    print(f"Overwrite:    {args.overwrite}")
    if args.dry_run:
        return

    for task, specs in specs_by_task.items():
        specs[0].target_pair_dir.mkdir(parents=True, exist_ok=True)
        specs[0].target_cache_dir.mkdir(parents=True, exist_ok=True)
        _write_task_tables(task, specs)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(lambda spec: _write_pair_files(spec, overwrite=args.overwrite), all_specs))
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(lambda spec: _write_cache_file(spec, overwrite=args.overwrite), all_specs))

    t5_count = _copy_t5_cache(source_t5_dir, target_t5_dir, overwrite=args.overwrite)

    print("Done")
    print(f"  pair/cache rows: {len(all_specs)}")
    print(f"  t5 files:        {t5_count}")
    for task in tasks:
        print(f"  {task}: {len(specs_by_task[task])}")


if __name__ == "__main__":
    main()
