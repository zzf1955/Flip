"""Offline generation + metric evaluation for trained Mitty LoRA runs.

The script evaluates one or more trained ``training_data/log/<run>`` folders on
the 1s pair cache. For each model and split it first generates ``gen_XX.mp4``
videos, writes matched ``gt_XX.mp4`` / ``ctrl_XX.mp4`` videos, then computes
PSNR / SSIM / LPIPS / FID / FVD with ``src.tools.eval_metrics``.

Default target:
  - Mitty-transfer-124d_r128_2000s_0425_1456
  - Mitty-transfer2LoRA-124d_r128_2000s_0425_1425
  - 32 samples from ``eval`` + 32 samples from ``ood_eval``
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

from src.core.config import MAIN_ROOT, T5_CACHE_DIR, TRAINING_DATA_ROOT
from src.core.train_utils import (
    load_cached_files,
    load_sample,
    load_t5_cache,
    save_video,
    tensor_to_frames,
)
from src.pipeline.backbones import get_mitty_spec
from src.pipeline.train_mitty import DEFAULT_DIT_DIR, DEFAULT_TOKENIZER, DEFAULT_VAE
from src.tools.eval_metrics import InceptionFeatureExtractor, LPIPS, process_step


DEFAULT_RUNS = [
    "Mitty-transfer-124d_r128_2000s_0425_1456",
    "Mitty-transfer2LoRA-124d_r128_2000s_0425_1425",
]
DEFAULT_SPLITS = ["eval", "ood_eval"]


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
    lora_rank: int,
    lora_target_modules: str,
    dit_dir: str,
    vae_path: str,
    tokenizer_dir: str,
):
    spec = get_mitty_spec()
    extra_kwargs = {}
    if run.merge_lora_paths:
        extra_kwargs["merge_lora_paths"] = [str(p) for p in run.merge_lora_paths]
        extra_kwargs["merge_lora_rank"] = run.merge_lora_rank
    model = spec.training_module_factory(
        device=device,
        dit_dir=dit_dir,
        vae_path=vae_path,
        tokenizer_dir=tokenizer_dir,
        lora_rank=lora_rank,
        lora_target_modules=lora_target_modules,
        use_gradient_checkpointing=False,
        load_vae=True,
        init_lora_path=str(run.checkpoint),
        **extra_kwargs,
    )
    model.eval()
    model.pipe.dit.eval()
    return model, spec


@torch.no_grad()
def generate_split(
    model,
    spec,
    files: list[str],
    pair_split_dir: Path,
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

    for idx, path in enumerate(files):
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

        pair_name = Path(path).stem + ".mp4"
        gt_path = pair_split_dir / "video" / pair_name
        ctrl_path = pair_split_dir / "control_video" / pair_name
        if not gt_path.is_file():
            raise FileNotFoundError(f"GT video not found: {gt_path}")
        if not ctrl_path.is_file():
            raise FileNotFoundError(f"Control video not found: {ctrl_path}")
        shutil.copy2(gt_path, out_dir / f"gt_{sample_id}.mp4")
        shutil.copy2(ctrl_path, out_dir / f"ctrl_{sample_id}.mp4")


def metric_models(device: torch.device, no_lpips: bool, no_fid: bool):
    lpips_model = None if no_lpips else LPIPS().to(device).eval()
    inception = None if no_fid else InceptionFeatureExtractor().to(device).eval()
    return lpips_model, inception


def compute_rows(
    run_specs: list[RunSpec],
    splits: list[str],
    out_root: Path,
    device: torch.device,
    no_lpips: bool,
    no_fid: bool,
) -> list[dict]:
    lpips_model, inception = metric_models(device, no_lpips, no_fid)
    rows = []
    for run in run_specs:
        for split in splits:
            split_out = out_root / run.name / run.checkpoint.stem / split
            metrics = process_step(str(split_out), lpips_model, inception, device)
            if not metrics:
                raise RuntimeError(f"No gen/gt pairs found in {split_out}")
            row = {
                "run": run.name,
                "checkpoint": run.checkpoint.name,
                "split": split,
                "out_dir": str(split_out),
                **{k: v for k, v in metrics.items() if k != "per_sample"},
            }
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False, indent=2), flush=True)
    return rows


def write_csv(rows: list[dict], path: Path):
    headers = [
        "run", "checkpoint", "split", "n_samples",
        "psnr", "ssim", "lpips", "fid", "fvd", "out_dir",
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
                    choices=["train", "eval", "ood_eval"])
    ap.add_argument("--cache-root", default="training_data/cache/vae/pair_1s",
                    help="VAE cache root containing split subdirs")
    ap.add_argument("--pair-root", default="training_data/pair/1s",
                    help="pair root containing split/video and split/control_video")
    ap.add_argument("--t5-cache-dir", default=T5_CACHE_DIR)
    ap.add_argument("--output-dir", default="training_data/eval/mitty_pair_1s")
    ap.add_argument("--samples-per-split", type=int, default=32,
                    help="samples per split; -1 means all cached samples")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--lora-rank", type=int, default=128)
    ap.add_argument("--lora-target-modules", default="q,k,v,o")
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

    run_specs = parse_run_specs(
        args.runs,
        args.checkpoint,
        auto_merge_lora=not args.no_auto_merge_lora,
    )
    cache_root = resolve_path(args.cache_root)
    pair_root = resolve_path(args.pair_root)
    out_root = resolve_path(args.output_dir)
    t5_dir = resolve_path(args.t5_cache_dir)

    t5_pos, t5_neg = load_t5_cache(str(t5_dir), device="cpu")
    if t5_neg is None:
        raise FileNotFoundError(f"negative T5 cache not found in {t5_dir}")

    device = torch.device(args.device)

    if args.no_generate and args.generate_only:
        ap.error("--no-generate and --generate-only are mutually exclusive")

    if not args.no_generate:
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
                dit_dir=args.dit_dir,
                vae_path=args.vae_path,
                tokenizer_dir=args.tokenizer_dir,
            )

            for split in args.splits:
                split_files = load_cached_files(str(cache_root / split))
                if args.samples_per_split >= 0:
                    split_files = split_files[:args.samples_per_split]
                split_out = out_root / run.name / run.checkpoint.stem / split
                print(f"[{run.name}] {split}: {len(split_files)} samples -> {split_out}", flush=True)
                generate_split(
                    model=model,
                    spec=spec,
                    files=split_files,
                    pair_split_dir=pair_root / split,
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
        out_root=out_root,
        device=device,
        no_lpips=args.no_lpips,
        no_fid=args.no_fid,
    )

    csv_path = out_root / "summary.csv"
    json_path = out_root / "summary.json"
    write_csv(rows, csv_path)
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n")
    print(f"\nSaved summary: {csv_path}")
    print(f"Saved summary: {json_path}")


if __name__ == "__main__":
    main()
