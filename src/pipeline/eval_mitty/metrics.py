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
    VideoFeatureExtractor,
    make_print_progress,
    process_step,
)


def metric_models(device: torch.device, no_lpips: bool, no_fid: bool):
    lpips_model = None if no_lpips else LPIPS().to(device).eval()
    inception = None if no_fid else InceptionFeatureExtractor().to(device).eval()
    video_extractor = None if no_fid else VideoFeatureExtractor().to(device).eval()
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
) -> list[dict]:
    lpips_model, inception, video_extractor = metric_models(device, no_lpips, no_fid)
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
        "out_dir", "summary_dir",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

