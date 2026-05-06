"""Offline generation + metric evaluation for trained Mitty LoRA runs.

The script evaluates one or more trained ``training_data/log/<run>`` folders on
the task-level runtime data split. For each model and split it first generates
``gen_XX.mp4`` videos, writes matched ``gt_XX.mp4`` / ``ctrl_XX.mp4`` videos, then computes
PSNR / SSIM / LPIPS / FID / FVD with ``src.tools.eval_metrics``.

Default target:
  - Mitty-transfer-124d_r128_2000s_0425_1456
  - Mitty-transfer2LoRA-124d_r128_2000s_0425_1425
  - tail 10% from each configured in-task and OOD task ``pair_order.jsonl``
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import torch
from diffsynth.diffusion.flow_match import FlowMatchScheduler

from src.core.config import MAIN_ROOT, TRAINING_DATA_ROOT
from src.core.train_utils import (
    load_sample,
    load_t5_cache,
    save_video,
    tensor_to_frames,
)
from src.pipeline.backbones import get_mitty_spec
from src.pipeline.runtime_data import RuntimeSplit, build_tail_eval_split
from src.pipeline.train_config import apply_train_task_config
from src.pipeline.train_mitty import DEFAULT_DIT_DIR, DEFAULT_TOKENIZER, DEFAULT_VAE
from src.tools.eval_metrics import (
    InceptionFeatureExtractor,
    LPIPS,
    VideoFeatureExtractor,
    process_step,
)


DEFAULT_RUNS = [
    "Mitty-transfer-124d_r128_2000s_0425_1456",
    "Mitty-transfer2LoRA-124d_r128_2000s_0425_1425",
]
DEFAULT_SPLITS = ["in_task_eval", "ood_eval"]
SPLIT_ALIASES = {
    "eval": "in_task_eval",
    "in_task": "in_task_eval",
    "in_task_eval": "in_task_eval",
    "ood": "ood_eval",
    "ood_eval": "ood_eval",
}


@dataclass(frozen=True)
class RunSpec:
    name: str
    run_dir: Path
    checkpoint: Path
    merge_lora_paths: tuple[Path, ...]
    merge_lora_rank: int


def read_train_args(run_dir: Path) -> dict:
    log_path = run_dir / "train.log"
    if not log_path.is_file():
        return {}
    with log_path.open() as f:
        for line in f:
            marker = " Args: "
            if marker in line:
                return ast.literal_eval(line.split(marker, 1)[1].strip())
    return {}


def resolve_path(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return Path(MAIN_ROOT) / p


def normalize_splits(values: list[str]) -> list[str]:
    splits = []
    for value in values:
        if value not in SPLIT_ALIASES:
            raise ValueError(
                f"Unknown split '{value}'. Available: {sorted(SPLIT_ALIASES)}"
            )
        split = SPLIT_ALIASES[value]
        if split not in splits:
            splits.append(split)
    return splits


def find_latest_checkpoint(run_dir: Path) -> Path:
    ckpts = sorted(run_dir.glob("ckpt/step-*.safetensors"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found under {run_dir / 'ckpt'}")
    return max(ckpts, key=lambda p: int(p.stem.split("-")[-1]))


def parse_run_specs(
    run_args: list[str],
    checkpoint: str,
    auto_merge_lora: bool,
) -> list[RunSpec]:
    specs = []
    for run_arg in run_args:
        run_dir = resolve_path(run_arg)
        if not run_dir.exists():
            run_dir = Path(TRAINING_DATA_ROOT) / "log" / run_arg
        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {run_arg}")

        if checkpoint == "latest":
            ckpt = find_latest_checkpoint(run_dir)
        else:
            ckpt = run_dir / "ckpt" / checkpoint
            if not ckpt.exists():
                raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
        train_args = read_train_args(run_dir)
        merge_paths = ()
        if auto_merge_lora and train_args.get("merge_lora"):
            merge_paths = tuple(resolve_path(p) for p in train_args["merge_lora"])
            for p in merge_paths:
                if not p.is_file():
                    raise FileNotFoundError(f"Merged LoRA not found for {run_dir.name}: {p}")
        specs.append(RunSpec(
            name=run_dir.name,
            run_dir=run_dir,
            checkpoint=ckpt,
            merge_lora_paths=merge_paths,
            merge_lora_rank=int(train_args.get("merge_lora_rank", 96)),
        ))
    return specs


def load_model(
    run: RunSpec,
    device: str,
    lora_rank: int | None,
    lora_target_modules: str | None,
    lora_attn_types: str,
    lora_attn_projections: str,
    dit_dir: str,
    vae_path: str,
    tokenizer_dir: str,
):
    spec = get_mitty_spec()
    extra_kwargs = {}
    if run.merge_lora_paths:
        extra_kwargs["merge_lora_paths"] = [str(p) for p in run.merge_lora_paths]
    model = spec.training_module_factory(
        device=device,
        dit_dir=dit_dir,
        vae_path=vae_path,
        tokenizer_dir=tokenizer_dir,
        lora_rank=lora_rank,
        lora_target_modules=lora_target_modules,
        lora_attn_types=lora_attn_types,
        lora_attn_projections=lora_attn_projections,
        use_gradient_checkpointing=False,
        load_vae=True,
        init_lora_path=str(run.checkpoint),
        **extra_kwargs,
    )
    model.eval()
    model.pipe.dit.eval()
    return model, spec


def records_for_split(split: str, runtime_split: RuntimeSplit) -> list[dict]:
    if split == "in_task_eval":
        return runtime_split.eval_records
    if split == "ood_eval":
        return runtime_split.ood_records
    raise ValueError(f"Unsupported split: {split}")


def resolve_pair_media(record: dict, kind: str) -> Path:
    raw_path = record.get(kind)
    if not raw_path:
        raise ValueError(f"Selected record missing {kind}: {record}")
    path = Path(str(raw_path))
    if path.is_absolute():
        return path
    pair_dir = record.get("pair_dir")
    if not pair_dir:
        raise ValueError(f"Selected record missing pair_dir for relative {kind}: {record}")
    return Path(str(pair_dir)) / path


def write_selected_records(base_dir: Path, splits: list[str], runtime_split: RuntimeSplit, args) -> None:
    split_dir = base_dir / "data_split"
    split_dir.mkdir(parents=True, exist_ok=True)
    payloads = {
        "in_task_eval": runtime_split.eval_records,
        "ood_eval": runtime_split.ood_records,
    }
    for split in splits:
        with (split_dir / f"{split}.jsonl").open("w") as fh:
            for record in payloads[split]:
                fh.write(json.dumps(record, sort_keys=True) + "\n")
    config = {
        "task_name": args.task_name,
        "data_type": args.data_type,
        "duration": args.duration,
        "train_tasks": args.train_tasks,
        "ood_tasks": args.ood_tasks,
        "eval_tail_percent": args.eval_tail_percent,
        "data_seed": args.data_seed,
        "cache_root": args.cache_root,
        "pair_root": args.pair_root,
        "split_counts": runtime_split.split_counts,
        "pair_order_paths": runtime_split.order_paths,
    }
    (split_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")


def eval_base_dir(run: RunSpec, output_dir: str) -> Path:
    if output_dir:
        return resolve_path(output_dir) / run.name / run.checkpoint.stem
    return run.checkpoint.parent / f"{run.checkpoint.stem}_eval"


@torch.no_grad()
def generate_split(
    model,
    spec,
    records: list[dict],
    out_dir: Path,
    device: str,
    num_inference_steps: int,
    cfg_scale: float,
    t5_pos: dict[str, torch.Tensor],
    t5_neg: torch.Tensor,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    pipe = model.pipe
    sched = FlowMatchScheduler("Wan")
    sched.set_timesteps(
        num_inference_steps=num_inference_steps,
        denoising_strength=1.0,
        shift=5.0,
    )

    for idx, record in enumerate(records):
        path = record["cache_path"]
        sample = load_sample(path, device=device, t5_pos=t5_pos, t5_neg=t5_neg)
        denoised = spec.eval_denoise_fn(
            pipe=pipe,
            sample=sample,
            sched=sched,
            device=device,
            cfg_scale=cfg_scale,
            num_inference_steps=num_inference_steps,
        )

        pipe.load_models_to_device(["vae"])
        gen_video = pipe.vae.decode(denoised, device=device, tiled=False)

        sample_id = f"{idx:05d}"
        save_video(tensor_to_frames(gen_video), str(out_dir / f"gen_{sample_id}.mp4"))

        gt_path = resolve_pair_media(record, "video")
        ctrl_path = resolve_pair_media(record, "control_video")
        if not gt_path.is_file():
            raise FileNotFoundError(f"GT video not found: {gt_path}")
        if not ctrl_path.is_file():
            raise FileNotFoundError(f"Control video not found: {ctrl_path}")
        shutil.copy2(gt_path, out_dir / f"gt_{sample_id}.mp4")
        shutil.copy2(ctrl_path, out_dir / f"ctrl_{sample_id}.mp4")


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
) -> list[dict]:
    lpips_model, inception, video_extractor = metric_models(device, no_lpips, no_fid)
    rows = []
    for run in run_specs:
        base_dir = eval_base_dir(run, output_dir)
        for split in splits:
            split_out = base_dir / split
            metrics = process_step(
                str(split_out), lpips_model, inception, video_extractor, device)
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
        "psnr", "ssim", "lpips", "fid", "fvd", "out_dir", "summary_dir",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
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
    ap.add_argument("--eval-tail-percent", type=float, default=10.0,
                    help="tail percentage to read from each task pair_order.jsonl")
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
    ap.add_argument("--generate-only", action="store_true",
                    help="generate videos only and skip metric computation")
    ap.add_argument("--no-lpips", action="store_true")
    ap.add_argument("--no-fid", action="store_true",
                    help="skip both FID and FVD")
    args = ap.parse_args()

    try:
        apply_train_task_config(args)
        args.splits = normalize_splits(args.splits)
    except ValueError as exc:
        ap.error(str(exc))

    run_specs = parse_run_specs(
        args.runs,
        args.checkpoint,
        auto_merge_lora=not args.no_auto_merge_lora,
    )
    t5_dir = resolve_path(args.t5_cache_dir)
    args.cache_root = str(resolve_path(args.cache_root))
    args.pair_root = str(resolve_path(args.pair_root))

    runtime_split = build_tail_eval_split(args)

    print(
        "Selected eval samples from pair_order tails: "
        + json.dumps(runtime_split.split_counts, ensure_ascii=False, sort_keys=True),
        flush=True,
    )

    device = torch.device(args.device)

    if args.no_generate and args.generate_only:
        ap.error("--no-generate and --generate-only are mutually exclusive")

    if not args.no_generate:
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

            base_dir = eval_base_dir(run, args.output_dir)
            write_selected_records(base_dir, args.splits, runtime_split, args)
            for split in args.splits:
                split_records = records_for_split(split, runtime_split)
                split_out = base_dir / split
                print(f"[{run.name}] {split}: {len(split_records)} samples -> {split_out}", flush=True)
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
                )

            del model
            torch.cuda.empty_cache()

    if args.generate_only:
        print("\nGeneration finished; metric computation skipped (--generate-only).")
        return

    rows = compute_rows(
        run_specs=run_specs,
        splits=args.splits,
        output_dir=args.output_dir,
        device=device,
        no_lpips=args.no_lpips,
        no_fid=args.no_fid,
    )

    for run in run_specs:
        base_dir = eval_base_dir(run, args.output_dir)
        run_rows = [row for row in rows if row["run"] == run.name]
        write_selected_records(base_dir, args.splits, runtime_split, args)
        csv_path = base_dir / "summary.csv"
        json_path = base_dir / "summary.json"
        write_csv(run_rows, csv_path)
        json_path.write_text(json.dumps(run_rows, ensure_ascii=False, indent=2) + "\n")
        print(f"\nSaved summary: {csv_path}")
        print(f"Saved summary: {json_path}")


if __name__ == "__main__":
    main()
