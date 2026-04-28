#!/usr/bin/env python3
"""Migrate legacy split pair/cache dirs into split-free task layout.

This copies files inside training_data and builds per-task manifest.jsonl files.
It does not decide in-task/OOD; train.py performs runtime splits from manifests.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

from src.core.config import MAIN_ROOT
from src.pipeline.runtime_data import short_task_name


LEGACY_PAIR_MAP = {
    "h2r": "1s",
    "blur_r2r": "1s_r2h",
    "identity_r2r": "1s_identity",
}

LEGACY_CACHE_MAP = {
    "h2r": "pair_1s",
    "blur_r2r": "pair_1s_r2h",
    "identity_r2r": "robot_1s",
}


def read_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    return json.loads(path.read_text())


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for row in sorted(rows, key=lambda r: (r.get("robot_task", ""), r.get("source_id", ""), r.get("cache_path", ""))):
            fh.write(json.dumps(row, sort_keys=True) + "\n")


def migrate_pairs(data_type: str, duration: str, *, clean: bool) -> None:
    root = Path(MAIN_ROOT)
    legacy_name = LEGACY_PAIR_MAP[data_type]
    src_root = root / "training_data" / "pair" / legacy_name
    dst_root = root / "training_data" / "pair" / data_type / duration
    if not src_root.is_dir():
        print(f"skip pair {data_type}: missing {src_root}")
        return
    if clean and dst_root.exists():
        shutil.rmtree(dst_root)

    per_task: dict[str, list[dict]] = {}
    for split in ("train", "eval", "ood_eval"):
        split_dir = src_root / split
        metadata_path = split_dir / "metadata.csv"
        source_map = read_json(src_root / "source_map.json")
        if not metadata_path.is_file():
            continue
        for row in csv.DictReader(metadata_path.open()):
            video_name = Path(row["video"]).name
            source = source_map.get(f"{split}/{video_name}", {})
            task = short_task_name(source.get("task", ""))
            if not task:
                raise ValueError(f"missing task for {split}/{video_name} in {src_root}")
            task_dir = dst_root / task
            dst_video = task_dir / "video" / video_name
            dst_control = task_dir / "control_video" / video_name
            dst_video.parent.mkdir(parents=True, exist_ok=True)
            dst_control.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(split_dir / row["video"], dst_video)
            shutil.copy2(split_dir / row["control_video"], dst_control)
            manifest = {
                **source,
                "data_type": data_type,
                "duration": duration,
                "robot_task": task,
                "task": task,
                "video": f"video/{video_name}",
                "control_video": f"control_video/{video_name}",
                "input_role": "human" if data_type == "h2r" else "robot",
                "target_role": "human" if data_type == "r2h" else "robot",
            }
            manifest.setdefault("source_segment_id", source.get("source_id", ""))
            per_task.setdefault(task, []).append(manifest)

    index_rows = []
    for task, records in per_task.items():
        task_dir = dst_root / task
        with (task_dir / "metadata.csv").open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=["video", "prompt", "control_video"])
            writer.writeheader()
            for record in sorted(records, key=lambda r: r["video"]):
                writer.writerow({
                    "video": record["video"],
                    "prompt": "A first-person view robot arm performing household tasks flip_v2v",
                    "control_video": record["control_video"],
                })
        write_jsonl(task_dir / "manifest.jsonl", records)
        index_rows.extend(records)
    write_jsonl(dst_root / "index.jsonl", index_rows)
    print(f"pair {data_type}: {len(index_rows)} rows -> {dst_root}")


def migrate_cache(data_type: str, duration: str, *, clean: bool) -> None:
    root = Path(MAIN_ROOT)
    legacy_name = LEGACY_CACHE_MAP[data_type]
    src_root = root / "training_data" / "cache" / "vae" / legacy_name
    dst_root = root / "training_data" / "cache" / "vae" / data_type / duration
    pair_root = root / "training_data" / "pair" / data_type / duration
    legacy_pair_root = root / "training_data" / "pair" / LEGACY_PAIR_MAP[data_type]
    if not src_root.is_dir():
        print(f"skip cache {data_type}: missing {src_root}")
        return
    if clean and dst_root.exists():
        shutil.rmtree(dst_root)

    legacy_source_map = read_json(legacy_pair_root / "source_map.json")
    pair_by_source_id = {}
    for manifest_path in pair_root.glob("*/manifest.jsonl"):
        for line in manifest_path.read_text().splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            pair_by_source_id[record.get("source_id", "")] = record

    per_task: dict[str, list[dict]] = {}
    for split in ("train", "eval", "ood_eval"):
        split_dir = src_root / split
        if not split_dir.is_dir():
            continue
        for src_file in sorted(split_dir.glob("*.pth")):
            legacy_record = legacy_source_map.get(f"{split}/{src_file.name.replace('.pth', '.mp4')}", {})
            source_id = legacy_record.get("source_id", "")
            record = pair_by_source_id.get(source_id)
            if record is None:
                raise ValueError(
                    f"Cannot map legacy cache {src_file} to migrated pair manifest; "
                    f"source_id={source_id!r}"
                )
            task = record["robot_task"]
            dst_file = dst_root / task / src_file.name
            dst_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_file, dst_file)
            cache_record = dict(record)
            cache_record["cache_path"] = dst_file.name
            per_task.setdefault(task, []).append(cache_record)

    index_rows = []
    for task, records in per_task.items():
        write_jsonl(dst_root / task / "manifest.jsonl", records)
        index_rows.extend(records)
    write_jsonl(dst_root / "index.jsonl", index_rows)
    print(f"cache {data_type}: {len(index_rows)} rows -> {dst_root}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Migrate FLIP data to task-organized split-free layout")
    parser.add_argument("--data-type", choices=sorted(LEGACY_PAIR_MAP), required=True)
    parser.add_argument("--duration", default="1s")
    parser.add_argument("--pairs", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    if args.pairs:
        migrate_pairs(args.data_type, args.duration, clean=args.clean)
    if args.cache:
        migrate_cache(args.data_type, args.duration, clean=args.clean)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
