"""CLI parser for offline Mitty evaluation."""

from __future__ import annotations

import argparse

from src.pipeline.eval_mitty.constants import (
    DEFAULT_LOCAL_VIDEO_SIZE,
    DEFAULT_RUNS,
    DEFAULT_SAM2_MASK_ROOT,
    DEFAULT_SPLITS,
)
from src.pipeline.eval_mitty.generation import (
    DEFAULT_DIT_DIR,
    DEFAULT_TOKENIZER,
    DEFAULT_VAE,
)
from src.tools.eval_metrics import (
    DEFAULT_FEATURE_BATCH_SIZE,
    DEFAULT_FVD_BATCH_SIZE,
    DEFAULT_LPIPS_BATCH_SIZE,
    DEFAULT_METRIC_WORKERS,
    LOCAL_FID_MARGIN,
    PATCH_FID_COVERAGE_THRESHOLD,
    PATCH_FID_MAX_PATCHES_PER_FRAME,
    PATCH_FID_MAX_PATCHES_PER_VIDEO,
    PATCH_FID_MIN_MASK_PIXELS,
    PATCH_FID_SIZE,
    PATCH_FID_STRIDE,
)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Generate Mitty eval videos and compute PSNR/SSIM/LPIPS/FID/FVD."
    )
    ap.add_argument("--runs", nargs="+", default=DEFAULT_RUNS,
                    help="training_data/log run names or run directories")
    ap.add_argument("--checkpoint", default="step-2000.safetensors",
                    help="checkpoint filename under ckpt/, or 'latest'")
    ap.add_argument("--no-auto-merge-lora", action="store_true",
                    help="do not replay merge_lora paths recorded in train.log")
    ap.add_argument("--splits", nargs="+", default=DEFAULT_SPLITS,
                    help="splits to evaluate: in_task_eval/eval and/or ood_eval/ood")
    ap.add_argument("--task-name", default="pair_1s",
                    help="training data preset used to fill data/cache/pair defaults")
    ap.add_argument("--data-type", default="")
    ap.add_argument("--duration", default="")
    ap.add_argument("--train-tasks", default="")
    ap.add_argument("--ood-tasks", default="")
    ap.add_argument("--cache-root", default="",
                    help="VAE cache root; default comes from --task-name")
    ap.add_argument("--pair-root", default="",
                    help="pair root; default comes from --task-name")
    ap.add_argument("--t5-cache-dir", default="",
                    help="T5 cache dir; default comes from --task-name")
    ap.add_argument("--data-seed", type=int, default=42,
                    help="seed used only when pair_order.jsonl must be created")
    ap.add_argument("--in-task-eval-size", type=int, default=None,
                    help="fixed total in-task eval sample count; allocated across "
                         "in-task tasks by data volume and selected from pair_order tails")
    ap.add_argument("--ood-eval-size", type=int, default=None,
                    help="fixed total OOD eval sample count; selected from OOD "
                         "pair_order tails")
    ap.add_argument("--eval-tail-percent", type=float, default=10.0,
                    help="legacy tail percentage to read from each task pair_order.jsonl "
                         "when fixed eval sizes are not provided")
    ap.add_argument("--output-dir", default="",
                    help="optional output root; default writes to <run>/ckpt/<step>_eval")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--lora-rank", type=int, default=None,
                    help="LoRA rank; auto-detected from each checkpoint by default")
    ap.add_argument("--lora-target-modules", default=None,
                    help="explicit comma-separated PEFT target suffixes; "
                         "auto-detected from each checkpoint by default")
    ap.add_argument("--lora-attn-types", default="self,cross",
                    help="attention blocks used only when target modules are not "
                         "auto-detected from a checkpoint")
    ap.add_argument("--lora-attn-projections", default="q,k,v,o",
                    help="attention projections used only when target modules are "
                         "not auto-detected from a checkpoint")
    ap.add_argument("--num-inference-steps", type=int, default=30)
    ap.add_argument("--cfg-scale", type=float, default=5.0)
    ap.add_argument("--dit-dir", default=DEFAULT_DIT_DIR)
    ap.add_argument("--vae-path", default=DEFAULT_VAE)
    ap.add_argument("--tokenizer-dir", default=DEFAULT_TOKENIZER)
    ap.add_argument("--no-generate", action="store_true",
                    help="skip generation and compute metrics from existing mp4 files")
    ap.add_argument("--resume-existing", action="store_true",
                    help="reuse complete gen/gt/ctrl video triplets and generate only missing samples")
    ap.add_argument("--generate-only", action="store_true",
                    help="generate videos only and skip metric computation")
    ap.add_argument("--no-lpips", action="store_true")
    ap.add_argument("--no-fid", action="store_true",
                    help="skip FID, FVD, Local FID, and Local FVD; "
                         "--patch-fid/--patch-fid-only still load Inception")
    ap.add_argument("--mask-region-metrics",
                    choices=["auto", "on", "off"], default="auto",
                    help="compute mask-region local metrics, foreground Local FID, "
                         "and foreground Local FVD from SAM2 masks; auto enables "
                         "this for blur_r2r")
    ap.add_argument("--mask-region-frechet-metrics",
                    choices=["auto", "on", "off"], default=None,
                    help="deprecated alias for --mask-region-metrics")
    ap.add_argument("--sam2-mask-root", default=str(DEFAULT_SAM2_MASK_ROOT),
                    help="SAM2 mask root for mask-region metrics")
    ap.add_argument("--write-local-videos", action="store_true",
                    help="write Local metric crop/overlay videos and patch_index.jsonl")
    ap.add_argument("--local-video-margin", type=int, default=LOCAL_FID_MARGIN,
                    help="pixel margin around the robot mask bbox for Local videos")
    ap.add_argument("--local-video-size", type=int, default=DEFAULT_LOCAL_VIDEO_SIZE,
                    help="square output size for Local crop videos")
    ap.add_argument("--local-video-bbox-mode", choices=["frame", "union"], default="frame",
                    help="frame uses per-frame mask bbox; union uses one clip-level bbox")
    ap.add_argument("--patch-fid", action="store_true",
                    help="compute foreground Patch FID from mask-selected frame patches")
    ap.add_argument("--patch-fid-only", action="store_true",
                    help="compute only foreground Patch FID from existing/generated videos")
    ap.add_argument("--write-patch-overlays", action="store_true",
                    help="write videos that overlay selected Patch FID patches per frame")
    ap.add_argument("--patch-size", type=int, default=PATCH_FID_SIZE,
                    help="square patch size in pixels for Patch FID")
    ap.add_argument("--patch-stride", type=int, default=PATCH_FID_STRIDE,
                    help="patch grid stride in pixels for Patch FID")
    ap.add_argument("--patch-coverage-threshold", type=float,
                    default=PATCH_FID_COVERAGE_THRESHOLD,
                    help="minimum mask coverage ratio for selecting a patch")
    ap.add_argument("--patch-min-mask-pixels", type=int,
                    default=PATCH_FID_MIN_MASK_PIXELS,
                    help="select patches with strictly more than this many mask pixels")
    ap.add_argument("--patch-max-per-frame", type=int,
                    default=PATCH_FID_MAX_PATCHES_PER_FRAME,
                    help="maximum selected patches per frame; 0 keeps all selected patches")
    ap.add_argument("--patch-max-per-video", type=int,
                    default=PATCH_FID_MAX_PATCHES_PER_VIDEO,
                    help="maximum selected patches per video; 0 keeps all per-frame selections")
    ap.add_argument("--metric-workers", type=int, default=DEFAULT_METRIC_WORKERS,
                    help="parallel workers for video decode and CPU pairwise metrics")
    ap.add_argument("--lpips-batch-size", type=int, default=DEFAULT_LPIPS_BATCH_SIZE,
                    help="GPU batch size for LPIPS frame batches")
    ap.add_argument("--feature-batch-size", type=int, default=DEFAULT_FEATURE_BATCH_SIZE,
                    help="GPU batch size for Inception/FID frame features")
    ap.add_argument("--fvd-batch-size", type=int, default=DEFAULT_FVD_BATCH_SIZE,
                    help="GPU batch size for S3D/FVD video features")
    ap.add_argument("--no-progress", action="store_true",
                    help="disable generation/local/metric progress printing")
    return ap
