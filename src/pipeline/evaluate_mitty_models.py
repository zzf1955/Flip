"""Offline generation + metric evaluation for trained Mitty LoRA runs.

The entrypoint evaluates one or more ``training_data/log/<run>`` folders on
runtime task splits. It generates ``gen_XX.mp4`` videos, copies matched
``gt_XX.mp4`` / ``ctrl_XX.mp4`` videos, then computes MSE / PSNR / SSIM /
LPIPS / FID / FVD with ``src.tools.eval_metrics``.

When SAM2 masks are enabled, region pairwise metrics are computed on mask
pixels and Frechet region metrics use the Local protocol: crop gen/GT videos by
the robot mask bbox, then compute foreground Local FID and foreground Local FVD.
"""

from __future__ import annotations

import json

import torch

from src.core.train_utils import load_t5_cache
from src.pipeline.eval_mitty.cli import build_parser
from src.pipeline.eval_mitty.generation import generate_split, load_model
from src.pipeline.eval_mitty.local_videos import write_local_videos, write_patch_overlays
from src.pipeline.eval_mitty.metrics import compute_rows, write_csv
from src.pipeline.eval_mitty.run_specs import (
    build_eval_split,
    eval_base_dir,
    normalize_splits,
    parse_run_specs,
    records_for_split,
    resolve_path,
    write_selected_records,
)
from src.pipeline.train_config import apply_train_task_config


def _configure_args(args, parser, output_dir_provided: bool) -> None:
    try:
        apply_train_task_config(args)
        if not output_dir_provided:
            args.output_dir = ""
        args.splits = normalize_splits(args.splits)
    except ValueError as exc:
        parser.error(str(exc))


def _resolve_mask_region(args, parser) -> str | None:
    if args.no_generate and args.generate_only:
        parser.error("--no-generate and --generate-only are mutually exclusive")
    if args.mask_region_frechet_metrics is not None:
        args.mask_region_metrics = args.mask_region_frechet_metrics
    if args.patch_fid_only:
        args.patch_fid = True
    mask_region_enabled = (
        args.mask_region_metrics == "on"
        or (
            args.mask_region_metrics == "auto"
            and args.data_type == "blur_r2r"
        )
        or args.patch_fid
        or args.write_patch_overlays
    )
    sam2_mask_root = str(resolve_path(args.sam2_mask_root)) if mask_region_enabled else None
    if args.write_local_videos and sam2_mask_root is None:
        parser.error("--write-local-videos requires mask-region metrics to be enabled")
    if args.write_patch_overlays and sam2_mask_root is None:
        parser.error("--write-patch-overlays requires patch/mask metrics to be enabled")
    return sam2_mask_root


def _validate_splits(args, runtime_split, parser) -> None:
    for split in args.splits:
        if not records_for_split(split, runtime_split):
            parser.error(f"Split '{split}' selected no eval samples")


def _generate_outputs(args, run_specs, runtime_split, t5_dir) -> None:
    t5_pos, t5_neg = load_t5_cache(str(t5_dir), device="cpu")
    if t5_neg is None:
        raise FileNotFoundError(f"negative T5 cache not found in {t5_dir}")

    for run in run_specs:
        print(f"\n=== {run.name} | {run.checkpoint.name} ===", flush=True)
        if run.merge_lora_paths:
            print(
                "replaying merged LoRA: "
                + ", ".join(str(p) for p in run.merge_lora_paths),
                flush=True,
            )
        model, spec = load_model(
            run=run,
            device=args.device,
            lora_rank=args.lora_rank,
            lora_target_modules=args.lora_target_modules,
            lora_attn_types=args.lora_attn_types,
            lora_attn_projections=args.lora_attn_projections,
            dit_dir=args.dit_dir,
            vae_path=args.vae_path,
            tokenizer_dir=args.tokenizer_dir,
        )

        base_dir = eval_base_dir(run, args.output_dir, args.output_exact_dir)
        write_selected_records(base_dir, args.splits, runtime_split, args)
        for split in args.splits:
            split_records = records_for_split(split, runtime_split)
            split_out = base_dir / split
            print(
                f"[{run.name}] {split}: {len(split_records)} samples -> {split_out}",
                flush=True,
            )
            generate_split(
                model=model,
                spec=spec,
                records=split_records,
                out_dir=split_out,
                device=args.device,
                num_inference_steps=args.num_inference_steps,
                cfg_scale=args.cfg_scale,
                t5_pos=t5_pos,
                t5_neg=t5_neg,
                resume_existing=args.resume_existing,
                show_progress=not args.no_progress,
            )

        del model
        torch.cuda.empty_cache()


def _write_local_outputs(args, run_specs, runtime_split, sam2_mask_root: str) -> None:
    for run in run_specs:
        base_dir = eval_base_dir(run, args.output_dir, args.output_exact_dir)
        for split in args.splits:
            split_records = records_for_split(split, runtime_split)
            split_out = base_dir / split
            print(
                f"[{run.name}] writing Local videos for {split} "
                f"-> {split_out / 'local_fid'}",
                flush=True,
            )
            write_local_videos(
                split_out,
                split_records,
                sam2_mask_root,
                margin=args.local_video_margin,
                output_size=args.local_video_size,
                bbox_mode=args.local_video_bbox_mode,
                show_progress=not args.no_progress,
            )


def _write_patch_outputs(args, run_specs, runtime_split, sam2_mask_root: str) -> None:
    for run in run_specs:
        base_dir = eval_base_dir(run, args.output_dir, args.output_exact_dir)
        for split in args.splits:
            split_records = records_for_split(split, runtime_split)
            split_out = base_dir / split
            print(
                f"[{run.name}] writing Patch FID overlays for {split} "
                f"-> {split_out / 'patch_fid'}",
                flush=True,
            )
            write_patch_overlays(
                split_out,
                split_records,
                sam2_mask_root,
                patch_size=args.patch_size,
                patch_stride=args.patch_stride,
                coverage_threshold=args.patch_coverage_threshold,
                min_mask_pixels=args.patch_min_mask_pixels,
                max_patches_per_frame=args.patch_max_per_frame,
                max_patches_per_video=args.patch_max_per_video,
                show_progress=not args.no_progress,
            )


def _write_summaries(args, run_specs, rows, runtime_split) -> None:
    for run in run_specs:
        base_dir = eval_base_dir(run, args.output_dir, args.output_exact_dir)
        run_rows = [row for row in rows if row["run"] == run.name]
        write_selected_records(base_dir, args.splits, runtime_split, args)
        csv_path = base_dir / "summary.csv"
        json_path = base_dir / "summary.json"
        write_csv(run_rows, csv_path)
        json_path.write_text(json.dumps(run_rows, ensure_ascii=False, indent=2) + "\n")
        print(f"\nSaved summary: {csv_path}")
        print(f"Saved summary: {json_path}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    output_dir_provided = bool(args.output_dir)
    _configure_args(args, parser, output_dir_provided)

    run_specs = parse_run_specs(
        args.runs,
        args.checkpoint,
        auto_merge_lora=not args.no_auto_merge_lora,
    )
    if args.output_exact_dir and len(run_specs) != 1:
        parser.error("--output-exact-dir requires exactly one --runs value")
    t5_dir = resolve_path(args.t5_cache_dir)
    args.cache_root = str(resolve_path(args.cache_root))
    args.pair_root = str(resolve_path(args.pair_root))

    runtime_split = build_eval_split(args)
    _validate_splits(args, runtime_split, parser)
    print(
        "Selected eval samples from pair_order tails: "
        + json.dumps(runtime_split.split_counts, ensure_ascii=False, sort_keys=True),
        flush=True,
    )

    device = torch.device(args.device)
    sam2_mask_root = _resolve_mask_region(args, parser)

    if not args.no_generate:
        _generate_outputs(args, run_specs, runtime_split, t5_dir)

    if args.write_local_videos:
        _write_local_outputs(args, run_specs, runtime_split, sam2_mask_root)

    if args.write_patch_overlays:
        _write_patch_outputs(args, run_specs, runtime_split, sam2_mask_root)

    if args.generate_only:
        print("\nGeneration finished; metric computation skipped (--generate-only).")
        return

    rows = compute_rows(
        run_specs=run_specs,
        splits=args.splits,
        output_dir=args.output_dir,
        output_exact_dir=args.output_exact_dir,
        device=device,
        no_lpips=args.no_lpips,
        no_fid=args.no_fid,
        runtime_split=runtime_split,
        sam2_mask_root=sam2_mask_root,
        metric_workers=args.metric_workers,
        lpips_batch_size=args.lpips_batch_size,
        feature_batch_size=args.feature_batch_size,
        fvd_batch_size=args.fvd_batch_size,
        patch_fid=args.patch_fid,
        patch_fid_only=args.patch_fid_only,
        patch_size=args.patch_size,
        patch_stride=args.patch_stride,
        patch_coverage_threshold=args.patch_coverage_threshold,
        patch_min_mask_pixels=args.patch_min_mask_pixels,
        patch_max_per_frame=args.patch_max_per_frame,
        patch_max_per_video=args.patch_max_per_video,
        show_progress=not args.no_progress,
    )
    _write_summaries(args, run_specs, rows, runtime_split)


if __name__ == "__main__":
    main()
