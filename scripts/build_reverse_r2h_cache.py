#!/usr/bin/env python3
"""Build a 2s robot-to-human cache view from an existing H2R cache.

The maintained Mitty trainer interprets ``human_latent`` as the clean condition
and ``robot_latent`` as the denoise target.  Existing H2R caches already contain
both sides, so an R2H view can be built without re-running VAE encoding:

* pair/control_video hardlinks to the original robot target video
* pair/video hardlinks to the original human control video
* cache/human_latent is copied from the original robot_latent
* cache/robot_latent is copied from the original human_latent
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.core.config import MAIN_ROOT


DEFAULT_TASKS = [
    "Inspire_Collect_Clothes_MainCamOnly",
    "Inspire_Put_Clothes_into_Washing_Machine",
    "Inspire_Pickup_Pillow_MainCamOnly",
]


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"missing JSONL: {path}")
    rows = []
    with path.open() as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON in {path}:{line_no}") from exc
    if not rows:
        raise ValueError(f"empty JSONL: {path}")
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
    os.replace(tmp, path)


def read_metadata(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"missing metadata: {path}")
    rows = list(csv.DictReader(path.open()))
    if not rows:
        raise ValueError(f"empty metadata: {path}")
    expected = {"video", "prompt", "control_video"}
    if set(rows[0]) != expected:
        raise ValueError(f"unexpected metadata columns in {path}: {sorted(rows[0])}")
    return rows


def write_metadata(path: Path, rows: list[dict]) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["video", "prompt", "control_video"])
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def link_file(src: Path, dst: Path, *, resume: bool) -> None:
    if not src.is_file():
        raise FileNotFoundError(f"source file not found: {src}")
    if dst.exists():
        if resume:
            return
        raise FileExistsError(f"destination exists: {dst}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    os.link(src, dst)


def reverse_pair_record(record: dict) -> dict:
    out = dict(record)
    pair_id = record["pair_id"]
    out["data_type"] = "r2h"
    out["input_role"] = "robot"
    out["target_role"] = "human"
    out["control_video"] = f"control_video/{pair_id}.mp4"
    out["video"] = f"video/{pair_id}.mp4"
    if "input_step_role" in out or "target_step_role" in out:
        out["input_step_role"] = record.get("target_step_role", "")
        out["target_step_role"] = record.get("input_step_role", "")
    if "input_step_video" in out or "target_step_video" in out:
        out["input_step_video"] = record.get("target_step_video", "")
        out["target_step_video"] = record.get("input_step_video", "")
    out["reverse_source_data_type"] = record.get("data_type", "")
    out["reverse_source_input_role"] = record.get("input_role", "")
    out["reverse_source_target_role"] = record.get("target_role", "")
    out["write_strategy"] = "reverse_h2r_hardlink"
    return out


def reverse_pair_order_record(record: dict) -> dict:
    out = dict(record)
    pair_id = record["pair_id"]
    out["data_type"] = "r2h"
    out["control_video"] = f"control_video/{pair_id}.mp4"
    out["video"] = f"video/{pair_id}.mp4"
    return out


def reverse_cache_record(record: dict) -> dict:
    out = dict(record)
    pair_id = record.get("pair_id", Path(record["cache_path"]).stem)
    out["data_type"] = "r2h"
    out["input_role"] = "robot"
    out["target_role"] = "human"
    out["control_video"] = f"control_video/{pair_id}.mp4"
    out["video"] = f"video/{pair_id}.mp4"
    if "input_step_role" in out or "target_step_role" in out:
        out["input_step_role"] = record.get("target_step_role", "")
        out["target_step_role"] = record.get("input_step_role", "")
    if "input_step_video" in out or "target_step_video" in out:
        out["input_step_video"] = record.get("target_step_video", "")
        out["target_step_video"] = record.get("input_step_video", "")
    out["reverse_source_data_type"] = record.get("data_type", "")
    out["reverse_source_input_role"] = record.get("input_role", "")
    out["reverse_source_target_role"] = record.get("target_role", "")
    out["write_strategy"] = "reverse_h2r_cache_swap"
    return out


def reverse_cache_file(src: Path, dst: Path, *, resume: bool) -> None:
    if dst.exists():
        if resume:
            return
        raise FileExistsError(f"destination exists: {dst}")
    data = torch.load(src, map_location="cpu", weights_only=False)
    if "human_latent" not in data or "robot_latent" not in data:
        raise KeyError(f"cache missing human_latent or robot_latent: {src}")
    out = dict(data)
    out["human_latent"] = data["robot_latent"]
    out["robot_latent"] = data["human_latent"]
    out["data_type"] = "r2h"
    out["input_role"] = "robot"
    out["target_role"] = "human"
    out["reverse_source_data_type"] = data.get("data_type", "")
    out["reverse_source_input_role"] = data.get("input_role", "")
    out["reverse_source_target_role"] = data.get("target_role", "")
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(f".{dst.name}.{os.getpid()}.tmp")
    torch.save(out, tmp)
    os.replace(tmp, dst)


def build_task(args: argparse.Namespace, task: str) -> dict:
    src_pair_task = args.src_pair_root / args.src_data_type / args.duration / task
    dst_pair_task = args.dst_pair_root / args.dst_data_type / args.duration / task
    src_cache_task = args.src_cache_root / args.src_data_type / args.duration / task
    dst_cache_task = args.dst_cache_root / args.dst_data_type / args.duration / task

    src_pair_rows = read_jsonl(src_pair_task / "manifest.jsonl")
    src_order_rows = read_jsonl(src_pair_task / "pair_order.jsonl")
    src_meta_rows = read_metadata(src_pair_task / "metadata.csv")
    src_cache_rows = read_jsonl(src_cache_task / "manifest.jsonl")

    if len(src_pair_rows) != len(src_cache_rows):
        raise ValueError(
            f"pair/cache row count mismatch for {task}: "
            f"{len(src_pair_rows)} != {len(src_cache_rows)}"
        )

    dst_pair_task.mkdir(parents=True, exist_ok=True)
    (dst_pair_task / "video").mkdir(exist_ok=True)
    (dst_pair_task / "control_video").mkdir(exist_ok=True)
    dst_cache_task.mkdir(parents=True, exist_ok=True)

    dst_pair_rows = []
    for record in src_pair_rows:
        dst_pair_rows.append(reverse_pair_record(record))
        pair_id = record["pair_id"]
        link_file(
            src_pair_task / record["video"],
            dst_pair_task / "control_video" / f"{pair_id}.mp4",
            resume=args.resume,
        )
        link_file(
            src_pair_task / record["control_video"],
            dst_pair_task / "video" / f"{pair_id}.mp4",
            resume=args.resume,
        )

    dst_meta_rows = []
    for row in src_meta_rows:
        pair_id = Path(row["video"]).stem
        dst_meta_rows.append({
            "video": f"video/{pair_id}.mp4",
            "prompt": row["prompt"],
            "control_video": f"control_video/{pair_id}.mp4",
        })

    dst_cache_rows = []
    for record in src_cache_rows:
        dst_cache_record = reverse_cache_record(record)
        dst_cache_record["pair_dir"] = str(dst_pair_task)
        dst_cache_rows.append(dst_cache_record)
        reverse_cache_file(
            src_cache_task / record["cache_path"],
            dst_cache_task / record["cache_path"],
            resume=args.resume,
        )

    dst_order_rows = [reverse_pair_order_record(record) for record in src_order_rows]
    write_metadata(dst_pair_task / "metadata.csv", dst_meta_rows)
    write_jsonl(dst_pair_task / "manifest.jsonl", dst_pair_rows)
    write_jsonl(dst_pair_task / "pair_order.jsonl", dst_order_rows)
    write_jsonl(dst_cache_task / "manifest.jsonl", dst_cache_rows)

    return {
        "task": task,
        "pairs": len(dst_pair_rows),
        "caches": len(dst_cache_rows),
        "pair_dir": str(dst_pair_task),
        "cache_dir": str(dst_cache_task),
    }


def link_t5_cache(args: argparse.Namespace) -> list[str]:
    src = args.src_t5_root / args.src_data_type / args.duration
    dst = args.dst_t5_root / args.dst_data_type / args.duration
    if not src.is_dir():
        raise NotADirectoryError(f"source T5 cache dir not found: {src}")
    dst.mkdir(parents=True, exist_ok=True)
    linked = []
    for path in sorted(src.glob("*.pth")):
        link_file(path, dst / path.name, resume=args.resume)
        linked.append(path.name)
    if not linked:
        raise FileNotFoundError(f"no T5 .pth files in {src}")
    return linked


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Build r2h 2s cache by reversing existing h2r pair/cache.")
    root = Path(MAIN_ROOT)
    ap.add_argument("--duration", default="2s61f30_slide")
    ap.add_argument("--src-data-type", default="h2r")
    ap.add_argument("--dst-data-type", default="r2h")
    ap.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    ap.add_argument("--src-pair-root", type=Path, default=root / "training_data" / "pair")
    ap.add_argument("--dst-pair-root", type=Path, default=root / "training_data" / "pair")
    ap.add_argument("--src-cache-root", type=Path, default=root / "training_data" / "cache" / "vae")
    ap.add_argument("--dst-cache-root", type=Path, default=root / "training_data" / "cache" / "vae")
    ap.add_argument("--src-t5-root", type=Path, default=root / "training_data" / "cache" / "t5")
    ap.add_argument("--dst-t5-root", type=Path, default=root / "training_data" / "cache" / "t5")
    ap.add_argument("--resume", action="store_true",
                    help="skip existing linked/cache files and rewrite manifests")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    summaries = [build_task(args, task) for task in args.tasks]
    linked_t5 = link_t5_cache(args)
    print(json.dumps({
        "duration": args.duration,
        "src_data_type": args.src_data_type,
        "dst_data_type": args.dst_data_type,
        "tasks": summaries,
        "t5_files": linked_t5,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
