"""Metric model setup and summary writing for offline Mitty evaluation."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import torch

from src.pipeline.eval_mitty.run_specs import (
    RunSpec,
    eval_base_dir,
    records_for_split,
)
from src.pipeline.runtime_data import RuntimeSplit
from src.tools.eval_metrics import (
    InceptionFeatureExtractor,
    LPIPS,
    PATCH_FID_COVERAGE_THRESHOLD,
    PATCH_FID_MAX_PATCHES_PER_FRAME,
    PATCH_FID_MAX_PATCHES_PER_VIDEO,
    PATCH_FID_MIN_MASK_PIXELS,
    PATCH_FID_SIZE,
    PATCH_FID_STRIDE,
    VideoFeatureExtractor,
    make_print_progress,
    process_step,
)


def metric_models(
    device: torch.device,
    no_lpips: bool,
    no_fid: bool,
    patch_fid: bool,
    patch_fid_only: bool,
):
    lpips_model = None if no_lpips or patch_fid_only else LPIPS().to(device).eval()
    needs_inception = not no_fid or patch_fid or patch_fid_only
    inception = InceptionFeatureExtractor().to(device).eval() if needs_inception else None
    video_extractor = None if no_fid or patch_fid_only else VideoFeatureExtractor().to(device).eval()
    return lpips_model, inception, video_extractor


def compute_rows(
    run_specs: list[RunSpec],
    splits: list[str],
    output_dir: str,
    device: torch.device,
    no_lpips: bool,
    no_fid: bool,
    runtime_split: RuntimeSplit,
    sam2_mask_root: str | None,
    metric_workers: int,
    lpips_batch_size: int,
    feature_batch_size: int,
    fvd_batch_size: int,
    show_progress: bool,
    patch_fid: bool = False,
    patch_fid_only: bool = False,
    patch_size: int = PATCH_FID_SIZE,
    patch_stride: int = PATCH_FID_STRIDE,
    patch_coverage_threshold: float = PATCH_FID_COVERAGE_THRESHOLD,
    patch_min_mask_pixels: int = PATCH_FID_MIN_MASK_PIXELS,
    patch_max_per_frame: int = PATCH_FID_MAX_PATCHES_PER_FRAME,
    patch_max_per_video: int = PATCH_FID_MAX_PATCHES_PER_VIDEO,
) -> list[dict]:
    lpips_model, inception, video_extractor = metric_models(
        device, no_lpips, no_fid, patch_fid, patch_fid_only,
    )
    rows = []
    for run in run_specs:
        base_dir = eval_base_dir(run, output_dir)
        for split in splits:
            split_out = base_dir / split
            selected_records = records_for_split(split, runtime_split) if sam2_mask_root else None
            print(f"[{run.name}] metrics {split}: {split_out}", flush=True)
            metrics = process_step(
                str(split_out),
                lpips_model,
                inception,
                video_extractor,
                device,
                selected_records=selected_records,
                sam2_mask_root=sam2_mask_root,
                metric_workers=metric_workers,
                lpips_batch_size=lpips_batch_size,
                feature_batch_size=feature_batch_size,
                fvd_batch_size=fvd_batch_size,
                patch_fid=patch_fid or patch_fid_only,
                patch_fid_only=patch_fid_only,
                patch_size=patch_size,
                patch_stride=patch_stride,
                patch_coverage_threshold=patch_coverage_threshold,
                patch_min_mask_pixels=patch_min_mask_pixels,
                patch_max_per_frame=patch_max_per_frame,
                patch_max_per_video=patch_max_per_video,
                progress_callback=(
                    make_print_progress(f"[{run.name}] {split}")
                    if show_progress else None
                ),
            )
            if not metrics:
                raise RuntimeError(f"No gen/gt pairs found in {split_out}")
            row = {
                "run": run.name,
                "checkpoint": run.checkpoint.name,
                "split": split,
                "out_dir": str(split_out),
                "summary_dir": str(base_dir),
                **{k: v for k, v in metrics.items() if k != "per_sample"},
            }
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False, indent=2), flush=True)
    return rows


def write_csv(rows: list[dict], path: Path):
    headers = [
        "run", "checkpoint", "split", "n_samples",
        "mse", "psnr", "ssim", "lpips", "fid", "fvd",
        "foreground_mse", "foreground_psnr", "foreground_ssim",
        "background_mse", "background_psnr", "background_ssim",
        "foreground_local_fid", "foreground_local_fvd",
        "foreground_patch_fid", "foreground_patch_count",
        "foreground_patch_size", "foreground_patch_stride",
        "foreground_patch_coverage_threshold",
        "foreground_patch_min_mask_pixels",
        "foreground_patch_max_per_frame", "foreground_patch_max_per_video",
        "out_dir", "summary_dir",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
