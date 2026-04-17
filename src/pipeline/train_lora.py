"""
Custom LoRA training for Wan 2.1 FunControl.

Reuses DiffSynth's WanTrainingModule for model loading + LoRA + forward pass,
but adds a proper training loop with:
- Train / Eval data split (deterministic, fixed seed)
- Per-step train loss logging
- Periodic eval loss computation
- Organized output: ckpt/, eval/, train.log

Usage:
  python -m src.pipeline.train_lora \
    --cache-dir output/data_cache_80 \
    --device cuda:3 --epochs 1 --repeat 1 \
    --save-steps 5 --eval-steps 5
"""

import argparse
import glob
import logging
import os
import random
import sys
import time
from datetime import datetime

# Reduce CUDA memory fragmentation — must precede torch import
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.stdout.reconfigure(line_buffering=True)

import torch
from safetensors.torch import save_file

# Import WanTrainingModule from DiffSynth-Studio examples (not part of the
# installed diffsynth package — it lives in the examples/ directory).
_DIFFSYNTH_TRAIN_DIR = os.path.join(
    os.environ.get("DIFFSYNTH_ROOT", "/disk_n/zzf/DiffSynth-Studio"),
    "examples", "wanvideo", "model_training",
)
sys.path.insert(0, _DIFFSYNTH_TRAIN_DIR)
from train import WanTrainingModule  # noqa: E402

from src.core.config import MAIN_ROOT, TRAINING_DATA_ROOT  # noqa: E402

# ── Defaults ──────────────────────────────────────────────────────────

DEFAULT_DIT_PATH = os.path.join(
    "/disk_n/zzf/.cache/huggingface/hub",
    "models--alibaba-pai--Wan2.1-Fun-V1.1-14B-Control",
    "manual", "diffusion_pytorch_model.safetensors",
)


# ── Helpers ───────────────────────────────────────────────────────────

def setup_logging(log_path: str) -> logging.Logger:
    logger = logging.getLogger("train_lora")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger


def load_cached_files(cache_dir: str) -> list[str]:
    """Scan cache directory for .pth files (output of DiffSynth data_process)."""
    files = sorted(glob.glob(os.path.join(cache_dir, "**", "*.pth"), recursive=True))
    if not files:
        raise FileNotFoundError(f"No .pth files in {cache_dir}")
    return files


def split_train_eval(files: list[str], eval_ratio: float, seed: int):
    """Deterministic split into train and eval sets."""
    if eval_ratio <= 0:
        return list(files), []
    rng = random.Random(seed)
    shuffled = list(files)
    rng.shuffle(shuffled)
    n_eval = max(1, int(len(shuffled) * eval_ratio))
    return shuffled[n_eval:], shuffled[:n_eval]


def load_sample(path: str):
    """Load a cached .pth sample → (inputs_shared, inputs_posi, inputs_nega)."""
    return torch.load(path, map_location="cpu", weights_only=False)


def save_lora_ckpt(model, path: str) -> int:
    """Extract LoRA params and save as safetensors. Returns number of tensors."""
    sd = model.state_dict()
    sd = model.export_trainable_state_dict(sd, remove_prefix="pipe.dit.")
    save_file({k: v.contiguous() for k, v in sd.items()}, path)
    return len(sd)


# ── Training ──────────────────────────────────────────────────────────

def train(args):
    # ── Output directory ──
    run_name = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_dir = os.path.join(args.output_dir, run_name)
    ckpt_dir = os.path.join(run_dir, "ckpt")
    eval_dir = os.path.join(run_dir, "eval")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)

    log = setup_logging(os.path.join(run_dir, "train.log"))
    log.info(f"Run: {run_dir}")
    log.info(f"Args: {vars(args)}")

    # ── Data ──
    all_files = load_cached_files(args.cache_dir)
    train_files, eval_files = split_train_eval(all_files, args.eval_ratio, args.seed)
    log.info(f"Data: {len(all_files)} total -> {len(train_files)} train, {len(eval_files)} eval")

    # ── Model ──
    log.info("Loading model...")
    t0 = time.time()
    model = WanTrainingModule(
        model_paths=f'["{args.dit_path}"]',
        fp8_models=args.dit_path,
        lora_base_model="dit",
        lora_target_modules="q,k,v,o,ffn.0,ffn.2",
        lora_rank=args.lora_rank,
        use_gradient_checkpointing=True,
        task="sft:train",
        device=args.device,
    )
    log.info(f"Model loaded in {time.time() - t0:.1f}s")

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    log.info(f"Params: {n_total:,} total, {n_trainable:,} trainable ({n_trainable / 1e6:.1f}M)")

    # ── Optimizer ──
    optimizer = torch.optim.AdamW(
        model.trainable_modules(), lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)

    # ── Training loop ──
    total_steps = args.epochs * len(train_files) * args.repeat
    log.info(
        f"Plan: {args.epochs} ep x {len(train_files)} train x {args.repeat} repeat"
        f" = {total_steps} steps"
    )
    log.info(f"Save every {args.save_steps} steps, eval every {args.eval_steps} steps")
    log.info("=" * 60)

    step = 0
    train_t0 = time.time()

    for epoch in range(args.epochs):
        rng = random.Random(args.seed + epoch)
        epoch_files = train_files * args.repeat
        rng.shuffle(epoch_files)

        for fpath in epoch_files:
            step += 1
            st = time.time()

            data = load_sample(fpath)
            optimizer.zero_grad()
            loss = model({}, inputs=data)
            loss.backward()
            optimizer.step()
            scheduler.step()

            lr = optimizer.param_groups[0]["lr"]
            log.info(
                f"step={step}/{total_steps} train_loss={loss.item():.4f}"
                f" lr={lr:.1e} time={time.time() - st:.1f}s"
            )

            # Checkpoint
            if args.save_steps and step % args.save_steps == 0:
                p = os.path.join(ckpt_dir, f"step-{step:04d}.safetensors")
                n = save_lora_ckpt(model, p)
                log.info(f"  SAVE {p} ({n} tensors)")

            # Eval
            if args.eval_steps and step % args.eval_steps == 0 and eval_files:
                losses = []
                with torch.no_grad():
                    for ef in eval_files:
                        ed = load_sample(ef)
                        el = model({}, inputs=ed)
                        losses.append(el.item())
                avg = sum(losses) / len(losses)
                log.info(f"  EVAL eval_loss={avg:.4f} ({len(losses)} samples)")

    # Final checkpoint if not already saved at last step
    if args.save_steps and step > 0 and step % args.save_steps != 0:
        p = os.path.join(ckpt_dir, f"step-{step:04d}.safetensors")
        save_lora_ckpt(model, p)
        log.info(f"  SAVE (final) {p}")

    elapsed = time.time() - train_t0
    log.info("=" * 60)
    log.info(f"Done. {step} steps in {elapsed:.0f}s ({elapsed / step:.1f}s/step)")


# ── CLI ───────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Custom LoRA training for Wan 2.1 FunControl")

    # Data
    ap.add_argument("--cache-dir", required=True,
                    help="Cached .pth directory (from DiffSynth data_process)")
    ap.add_argument("--output-dir",
                    default=os.path.join(TRAINING_DATA_ROOT, "log"),
                    help="Output root (default: training_data/log)")
    ap.add_argument("--eval-ratio", type=float, default=0.1,
                    help="Fraction of data held out for eval")
    ap.add_argument("--seed", type=int, default=42)

    # Model
    ap.add_argument("--dit-path", default=DEFAULT_DIT_PATH,
                    help="Path to DiT safetensors")
    ap.add_argument("--device", default="cuda:3")
    ap.add_argument("--lora-rank", type=int, default=16)

    # Training
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--repeat", type=int, default=10,
                    help="Repeat train set per epoch")
    ap.add_argument("--save-steps", type=int, default=50)
    ap.add_argument("--eval-steps", type=int, default=50)

    args = ap.parse_args()

    # Resolve relative cache-dir against MAIN_ROOT
    if not os.path.isabs(args.cache_dir):
        args.cache_dir = os.path.join(MAIN_ROOT, args.cache_dir)

    train(args)


if __name__ == "__main__":
    main()
