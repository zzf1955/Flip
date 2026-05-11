#!/usr/bin/env python3
"""Compute foreground/background Patch FID from an existing Mitty eval log.

The script reuses generated ``gen_*.mp4`` / ``gt_*.mp4`` files and the
``data_split/*.jsonl`` records written by ``src.pipeline.evaluate_mitty_models``.
It does not generate videos. Results are written under
``output/background_fid/<log-name>/`` by default.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.pipeline.eval_mitty.constants import DEFAULT_SAM2_MASK_ROOT
from src.tools.eval_metrics import (
    DEFAULT_FEATURE_BATCH_SIZE,
    DEFAULT_METRIC_WORKERS,
    PATCH_FID_COVERAGE_THRESHOLD,
    PATCH_FID_MAX_PATCHES_PER_FRAME,
    PATCH_FID_MAX_PATCHES_PER_VIDEO,
    PATCH_FID_MIN_MASK_PIXELS,
    PATCH_FID_SIZE,
    PATCH_FID_STRIDE,
    InceptionFeatureExtractor,
    compute_background_patch_fid,
    compute_patch_fid,
    find_pairs,
    load_clip_mask_stack,
    make_print_progress,
    read_video_frames,
)

DEFAULT_LOG_ROOT = REPO_ROOT / "training_data" / "log"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "output" / "background_fid"
DEFAULT_EVAL_SUBDIR = "full_eval"


@dataclass(frozen=True)
class SplitPairData:
    idx: int
    gen_frames: object
    gt_frames: object
    masks: object


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute foreground Patch FID and background Patch FID for an "
            "existing evaluate_mitty_models output directory."
        )
    )
    parser.add_argument("log_name",
                        help="training_data/log run name or path to a run directory")
    parser.add_argument("--log-root", default=str(DEFAULT_LOG_ROOT))
    parser.add_argument("--eval-dir", default="",
                        help="exact eval output directory; default auto-detects full_eval")
    parser.add_argument("--eval-subdir", default=DEFAULT_EVAL_SUBDIR,
                        help="run subdirectory to read when --eval-dir is omitted")
    parser.add_argument("--splits", nargs="*", default=None,
                        help="splits to evaluate; default uses data_split/*.jsonl")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--sam2-mask-root", default="",
                        help="override SAM2 mask root; default reads data_split/config.json")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--metric-workers", type=int, default=DEFAULT_METRIC_WORKERS)
    parser.add_argument("--feature-batch-size", type=int, default=DEFAULT_FEATURE_BATCH_SIZE)
    parser.add_argument("--patch-size", type=int, default=None)
    parser.add_argument("--patch-stride", type=int, default=None)
    parser.add_argument("--patch-coverage-threshold", type=float, default=None)
    parser.add_argument("--patch-min-mask-pixels", type=int, default=None)
    parser.add_argument("--patch-max-per-frame", type=int, default=None)
    parser.add_argument("--patch-max-per-video", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=0,
                        help="limit videos per split for smoke tests; 0 uses all")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def resolve_path(path: str | Path, base: Path = REPO_ROOT) -> Path:
    value = Path(path)
    if value.is_absolute():
        return value
    return base / value


def resolve_run_dir(log_name: str, log_root: str) -> Path:
    raw = Path(log_name)
    if raw.exists():
        return raw.resolve()
    root = resolve_path(log_root)
    run_dir = root / log_name
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    return run_dir


def resolve_eval_dir(args: argparse.Namespace, run_dir: Path) -> Path:
    if args.eval_dir:
        eval_dir = resolve_path(args.eval_dir)
        if not eval_dir.is_dir():
            raise FileNotFoundError(f"Eval directory not found: {eval_dir}")
        return eval_dir

    candidates = [run_dir / args.eval_subdir]
    ckpt_dir = run_dir / "ckpt"
    if ckpt_dir.is_dir():
        candidates.extend(sorted(ckpt_dir.glob("*_eval")))
    for candidate in candidates:
        if (candidate / "data_split").is_dir():
            return candidate
    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"No eval output with data_split found for {run_dir}; searched: {searched}"
    )


def read_json(path: Path) -> dict:
    with path.open() as fh:
        return json.load(fh)


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def split_names(eval_dir: Path, requested: list[str] | None) -> list[str]:
    if requested:
        return requested
    data_split = eval_dir / "data_split"
    names = sorted(path.stem for path in data_split.glob("*.jsonl"))
    return [name for name in names if (eval_dir / name).is_dir()]


def config_value(config: dict, key: str, default):
    value = config.get(key)
    if value is None:
        return default
    return value


def patch_config(args: argparse.Namespace, config: dict) -> dict:
    return {
        "patch_size": int(
            args.patch_size
            if args.patch_size is not None
            else config_value(config, "patch_size", PATCH_FID_SIZE)
        ),
        "stride": int(
            args.patch_stride
            if args.patch_stride is not None
            else config_value(config, "patch_stride", PATCH_FID_STRIDE)
        ),
        "coverage_threshold": float(
            args.patch_coverage_threshold
            if args.patch_coverage_threshold is not None
            else config_value(
                config,
                "patch_coverage_threshold",
                PATCH_FID_COVERAGE_THRESHOLD,
            )
        ),
        "min_mask_pixels": int(
            args.patch_min_mask_pixels
            if args.patch_min_mask_pixels is not None
            else config_value(config, "patch_min_mask_pixels", PATCH_FID_MIN_MASK_PIXELS)
        ),
        "max_patches_per_frame": int(
            args.patch_max_per_frame
            if args.patch_max_per_frame is not None
            else config_value(config, "patch_max_per_frame", PATCH_FID_MAX_PATCHES_PER_FRAME)
        ),
        "max_patches_per_video": int(
            args.patch_max_per_video
            if args.patch_max_per_video is not None
            else config_value(config, "patch_max_per_video", PATCH_FID_MAX_PATCHES_PER_VIDEO)
        ),
    }


def load_pair(
    pair: tuple[str, str, int],
    records_by_index: dict[int, dict],
    sam2_mask_root: Path,
) -> SplitPairData:
    gen_path, gt_path, idx = pair
    if idx not in records_by_index:
        raise ValueError(f"No selected record for sample index {idx}")
    gen_frames = read_video_frames(gen_path)
    gt_frames = read_video_frames(gt_path)
    if len(gen_frames) != len(gt_frames):
        raise ValueError(
            f"Frame count mismatch: {gen_path} has {len(gen_frames)} frames, "
            f"{gt_path} has {len(gt_frames)} frames"
        )
    masks = load_clip_mask_stack(
        records_by_index[idx],
        sam2_mask_root,
        len(gen_frames),
        gen_frames.shape[1:3],
    )
    return SplitPairData(idx, gen_frames, gt_frames, masks)


def load_split_data(
    split_dir: Path,
    records: list[dict],
    sam2_mask_root: Path,
    max_samples: int,
    metric_workers: int,
    progress_prefix: str,
    progress_enabled: bool,
) -> list[SplitPairData]:
    pairs = find_pairs(str(split_dir))
    if max_samples > 0:
        pairs = pairs[:max_samples]
    if not pairs:
        raise FileNotFoundError(f"No gen_*.mp4 / gt_*.mp4 pairs found in {split_dir}")
    if metric_workers <= 0:
        raise ValueError(f"metric_workers must be positive, got {metric_workers}")

    records_by_index = {idx: record for idx, record in enumerate(records)}
    loaded = []
    with ThreadPoolExecutor(max_workers=metric_workers) as executor:
        futures = [
            executor.submit(load_pair, pair, records_by_index, sam2_mask_root)
            for pair in pairs
        ]
        for done, future in enumerate(as_completed(futures), 1):
            loaded.append(future.result())
            if progress_enabled:
                print(f"  {progress_prefix} decode: {done}/{len(pairs)}", flush=True)
    loaded.sort(key=lambda item: item.idx)
    return loaded


def metric_row(
    *,
    run_name: str,
    eval_dir: Path,
    split: str,
    data: list[SplitPairData],
    extractor: InceptionFeatureExtractor,
    device: torch.device,
    feature_batch_size: int,
    patch_cfg: dict,
    progress_enabled: bool,
) -> dict:
    gen_videos = [item.gen_frames for item in data]
    gt_videos = [item.gt_frames for item in data]
    masks = [item.masks for item in data]
    progress = None if not progress_enabled else make_print_progress(f"{split}")

    foreground_fid, foreground_count = compute_patch_fid(
        gen_videos,
        gt_videos,
        masks,
        extractor,
        device,
        batch_size=feature_batch_size,
        progress_callback=progress,
        **patch_cfg,
    )
    background_fid, background_count = compute_background_patch_fid(
        gen_videos,
        gt_videos,
        masks,
        extractor,
        device,
        batch_size=feature_batch_size,
        progress_callback=progress,
        **patch_cfg,
    )
    row = {
        "run": run_name,
        "eval_dir": str(eval_dir),
        "split": split,
        "n_samples": len(data),
        "foreground_patch_count": foreground_count,
        "background_patch_count": background_count,
        "patch_size": patch_cfg["patch_size"],
        "patch_stride": patch_cfg["stride"],
        "patch_coverage_threshold": patch_cfg["coverage_threshold"],
        "patch_min_mask_pixels": patch_cfg["min_mask_pixels"],
        "patch_max_per_frame": patch_cfg["max_patches_per_frame"],
        "patch_max_per_video": patch_cfg["max_patches_per_video"],
    }
    if foreground_fid is not None:
        row["foreground_patch_fid"] = foreground_fid
    if background_fid is not None:
        row["background_patch_fid"] = background_fid
    return row


def write_outputs(output_dir: Path, rows: list[dict], config: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": config,
        "results": rows,
    }
    with (output_dir / "summary.json").open("w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)

    headers = [
        "run", "eval_dir", "split", "n_samples",
        "foreground_patch_fid", "background_patch_fid",
        "foreground_patch_count", "background_patch_count",
        "patch_size", "patch_stride", "patch_coverage_threshold",
        "patch_min_mask_pixels", "patch_max_per_frame", "patch_max_per_video",
    ]
    with (output_dir / "summary.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            formatted = dict(row)
            for key in ("foreground_patch_fid", "background_patch_fid"):
                if key in formatted:
                    formatted[key] = f"{float(formatted[key]):.6f}"
            writer.writerow({key: formatted.get(key, "") for key in headers})


def main() -> None:
    args = parse_args()
    run_dir = resolve_run_dir(args.log_name, args.log_root)
    eval_dir = resolve_eval_dir(args, run_dir)
    config_path = eval_dir / "data_split" / "config.json"
    config = read_json(config_path) if config_path.is_file() else {}
    sam2_mask_root = resolve_path(
        args.sam2_mask_root
        or str(config_value(config, "sam2_mask_root", DEFAULT_SAM2_MASK_ROOT))
    )
    if not sam2_mask_root.is_dir():
        raise FileNotFoundError(f"SAM2 mask root not found: {sam2_mask_root}")

    splits = split_names(eval_dir, args.splits)
    if not splits:
        raise FileNotFoundError(f"No split jsonl/video directories found in {eval_dir}")

    output_dir = resolve_path(args.output_root) / run_dir.name
    patch_cfg = patch_config(args, config)
    device = torch.device(args.device)

    print(f"Run: {run_dir.name}", flush=True)
    print(f"Eval dir: {eval_dir}", flush=True)
    print(f"SAM2 masks: {sam2_mask_root}", flush=True)
    print(f"Output: {output_dir}", flush=True)
    print(f"Loading InceptionV3 on {device} ...", flush=True)
    extractor = InceptionFeatureExtractor().to(device).eval()

    rows = []
    for split in splits:
        split_dir = eval_dir / split
        record_path = eval_dir / "data_split" / f"{split}.jsonl"
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Split video directory not found: {split_dir}")
        if not record_path.is_file():
            raise FileNotFoundError(f"Split record file not found: {record_path}")
        records = read_jsonl(record_path)
        print(f"\n[{split}] loading videos and masks", flush=True)
        data = load_split_data(
            split_dir,
            records,
            sam2_mask_root,
            args.max_samples,
            args.metric_workers,
            split,
            not args.no_progress,
        )
        print(f"[{split}] computing foreground/background Patch FID", flush=True)
        row = metric_row(
            run_name=run_dir.name,
            eval_dir=eval_dir,
            split=split,
            data=data,
            extractor=extractor,
            device=device,
            feature_batch_size=args.feature_batch_size,
            patch_cfg=patch_cfg,
            progress_enabled=not args.no_progress,
        )
        rows.append(row)
        fg = row.get("foreground_patch_fid", "")
        bg = row.get("background_patch_fid", "")
        print(
            f"[{split}] foreground_patch_fid={fg} "
            f"background_patch_fid={bg} "
            f"fg_patches={row['foreground_patch_count']} "
            f"bg_patches={row['background_patch_count']}",
            flush=True,
        )

    run_config = {
        "run_dir": str(run_dir),
        "eval_dir": str(eval_dir),
        "sam2_mask_root": str(sam2_mask_root),
        "device": str(device),
        "max_samples": args.max_samples,
        "feature_batch_size": args.feature_batch_size,
        **patch_cfg,
    }
    write_outputs(output_dir, rows, run_config)
    print(f"\nWrote {output_dir / 'summary.csv'}", flush=True)
    print(f"Wrote {output_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
