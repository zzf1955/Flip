"""
Custom LoRA training for Wan 2.1 FunControl.

Reuses DiffSynth's WanTrainingModule for model loading + LoRA + forward pass,
but adds a proper training loop with:
- Train / Eval data split (deterministic, fixed seed)
- Per-step train loss logging
- Periodic eval loss computation
- Eval video generation (ground truth + control + generated)
- Organized output: ckpt/, eval/, train.log

Usage:
  python -m src.pipeline.train_lora \
    --cache-dir output/data_cache_80 \
    --device cuda:0 --epochs 1 --repeat 1 \
    --save-steps 5 --eval-steps 5 --eval-video-steps 50
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

import numpy as np
import subprocess
import torch
from PIL import Image
from safetensors.torch import save_file

# Import WanTrainingModule from DiffSynth-Studio examples (not part of the
# installed diffsynth package — it lives in the examples/ directory).
_DIFFSYNTH_TRAIN_DIR = os.path.join(
    os.environ.get("DIFFSYNTH_ROOT", "/disk_n/zzf/DiffSynth-Studio"),
    "examples", "wanvideo", "model_training",
)
sys.path.insert(0, _DIFFSYNTH_TRAIN_DIR)
from train import WanTrainingModule  # noqa: E402

from diffsynth.diffusion.flow_match import FlowMatchScheduler  # noqa: E402
from src.core.config import MAIN_ROOT, TRAINING_DATA_ROOT  # noqa: E402

# ── Defaults ──────────────────────────────────────────────────────────

_MODEL_HUB = os.path.join(
    "/disk_n/zzf/.cache/huggingface/hub",
    "models--alibaba-pai--Wan2.1-Fun-V1.1-14B-Control",
)

DEFAULT_DIT_PATH = os.path.join(_MODEL_HUB, "manual", "diffusion_pytorch_model.safetensors")
DEFAULT_VAE_PATH = os.path.join(_MODEL_HUB, "manual", "Wan2.1_VAE.pth")

# UMT5 tokenizer — already cached from data_process stage.
DEFAULT_TOKENIZER_PATH = os.path.join(
    _MODEL_HUB, "snapshots",
    "d4d4513ee56cc9db003780fb1e63feb1b4e0c5d8",
    "google", "umt5-xxl",
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


# ── Video helpers ─────────────────────────────────────────────────────

FFMPEG = os.environ.get(
    "FFMPEG_BIN",
    "/home/leadtek/miniconda3/envs/flip/bin/ffmpeg",
)


def save_video(frames, path, fps=16):
    """Save list of PIL.Image frames as H.264 MP4 via ffmpeg pipe."""
    h, w = frames[0].size[1], frames[0].size[0]
    cmd = [
        FFMPEG, "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{w}x{h}", "-r", str(fps),
        "-i", "pipe:0",
        "-c:v", "libx264", "-crf", "18", "-preset", "fast",
        "-pix_fmt", "yuv420p", path,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for f in frames:
        proc.stdin.write(np.array(f).tobytes())
    proc.stdin.close()
    proc.wait()


def tensor_to_frames(video_tensor):
    """Convert VAE output [B, 3, T, H, W] range [-1,1] to list of PIL.Image."""
    video = video_tensor[0]  # [3, T, H, W]
    video = (video.float().clamp(-1, 1) + 1) / 2 * 255
    video = video.to(torch.uint8).cpu().numpy()  # [3, T, H, W]
    video = video.transpose(1, 2, 3, 0)  # [T, H, W, 3]
    return [Image.fromarray(frame) for frame in video]


def generate_eval_videos(model, eval_files, eval_dir, step, device,
                         num_samples=2, num_inference_steps=30,
                         cfg_scale=5.0, log=None):
    """Generate eval videos: ground truth + control + model output.

    Runs denoising loop with current LoRA weights + CFG, decodes with VAE.
    """
    vae = getattr(model.pipe, "vae", None)
    if vae is None:
        if log:
            log.info("  EVAL VIDEO skipped (no VAE loaded)")
        return

    # Separate scheduler for inference — must use "Wan" template
    # (default is "FLUX.1" which has different sigma schedule and shift)
    inf_scheduler = FlowMatchScheduler("Wan")
    inf_scheduler.set_timesteps(
        num_inference_steps=num_inference_steps,
        denoising_strength=1.0,
        shift=5.0,
    )

    step_dir = os.path.join(eval_dir, f"step-{step:04d}")
    os.makedirs(step_dir, exist_ok=True)

    n = min(num_samples, len(eval_files))
    for idx in range(n):
        t0 = time.time()
        sample = load_sample(eval_files[idx])
        inputs_shared, inputs_posi, inputs_nega = sample

        # ── Save ground truth (from cached PIL images) ──
        gt_frames = inputs_shared.get("input_video")
        if gt_frames:
            save_video(gt_frames, os.path.join(step_dir, f"gt_{idx:02d}.mp4"))

        # ── Save control / condition ──
        ctrl_frames = inputs_shared.get("control_video")
        if ctrl_frames:
            save_video(ctrl_frames, os.path.join(step_dir, f"ctrl_{idx:02d}.mp4"))

        # ── Generate via denoising with CFG ──
        context_posi = inputs_posi["context"].to(device=device, dtype=torch.bfloat16)
        context_nega = inputs_nega["context"].to(device=device, dtype=torch.bfloat16)
        clip_feature = inputs_shared.get("clip_feature")
        if clip_feature is not None:
            clip_feature = clip_feature.to(device=device, dtype=torch.bfloat16)
        y = inputs_shared.get("y")
        if y is not None:
            y = y.to(device=device, dtype=torch.bfloat16)

        latent_shape = inputs_shared["input_latents"].shape  # [1, 16, 5, 60, 80]
        latents = torch.randn(latent_shape, device=device, dtype=torch.bfloat16)

        model_fn = model.pipe.model_fn
        dit = model.pipe.dit
        shared_kwargs = dict(
            clip_feature=clip_feature, y=y,
            use_gradient_checkpointing=False,
        )

        with torch.no_grad():
            for pid, ts in enumerate(inf_scheduler.timesteps):
                t_tensor = ts.unsqueeze(0).to(device=device, dtype=torch.bfloat16)

                # Positive prediction
                pred_posi = model_fn(
                    dit=dit, latents=latents, timestep=t_tensor,
                    context=context_posi, **shared_kwargs,
                )

                # CFG: blend positive and negative predictions
                if cfg_scale != 1.0:
                    pred_nega = model_fn(
                        dit=dit, latents=latents, timestep=t_tensor,
                        context=context_nega, **shared_kwargs,
                    )
                    noise_pred = pred_nega + cfg_scale * (pred_posi - pred_nega)
                else:
                    noise_pred = pred_posi

                latents = inf_scheduler.step(
                    noise_pred, inf_scheduler.timesteps[pid], latents,
                )

            # VAE decode
            video_tensor = vae.decode(latents, device=device, tiled=False)

        gen_frames = tensor_to_frames(video_tensor)
        save_video(gen_frames, os.path.join(step_dir, f"gen_{idx:02d}.mp4"))

        if log:
            log.info(
                f"  EVAL VIDEO [{idx+1}/{n}] -> {step_dir} "
                f"({time.time()-t0:.0f}s)"
            )


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

    # ── Model (DiT + optional VAE) ──
    log.info("Loading model...")
    t0 = time.time()

    # Build model_paths JSON — DiT always, VAE if eval videos requested
    if args.vae_path and args.eval_video_steps > 0:
        model_paths = f'["{args.dit_path}", "{args.vae_path}"]'
        log.info("VAE included for eval video generation")
    else:
        model_paths = f'["{args.dit_path}"]'

    model = WanTrainingModule(
        model_paths=model_paths,
        fp8_models=args.dit_path,
        tokenizer_path=args.tokenizer_path,
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
    lr_scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)

    # ── Training loop ──
    total_steps = args.epochs * len(train_files) * args.repeat
    log.info(
        f"Plan: {args.epochs} ep x {len(train_files)} train x {args.repeat} repeat"
        f" = {total_steps} steps"
    )
    log.info(f"Save every {args.save_steps}, eval loss every {args.eval_steps}"
             f", eval video every {args.eval_video_steps} steps")
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
            lr_scheduler.step()

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

            # Eval loss
            if args.eval_steps and step % args.eval_steps == 0 and eval_files:
                losses = []
                with torch.no_grad():
                    for ef in eval_files:
                        ed = load_sample(ef)
                        el = model({}, inputs=ed)
                        losses.append(el.item())
                avg = sum(losses) / len(losses)
                log.info(f"  EVAL eval_loss={avg:.4f} ({len(losses)} samples)")

            # Eval videos (separate, usually less frequent)
            if (args.eval_video_steps and step % args.eval_video_steps == 0
                    and eval_files):
                generate_eval_videos(
                    model, eval_files, eval_dir, step, args.device,
                    num_samples=args.eval_video_samples,
                    num_inference_steps=args.num_inference_steps,
                    log=log,
                )

    # Final checkpoint if not already saved at last step
    if args.save_steps and step > 0 and step % args.save_steps != 0:
        p = os.path.join(ckpt_dir, f"step-{step:04d}.safetensors")
        save_lora_ckpt(model, p)
        log.info(f"  SAVE (final) {p}")

    # Final eval videos
    if (args.eval_video_steps and step > 0
            and step % args.eval_video_steps != 0 and eval_files):
        generate_eval_videos(
            model, eval_files, eval_dir, step, args.device,
            num_samples=args.eval_video_samples,
            num_inference_steps=args.num_inference_steps,
            log=log,
        )

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
    ap.add_argument("--vae-path", default=DEFAULT_VAE_PATH,
                    help="Path to VAE weights (for eval video generation)")
    ap.add_argument("--tokenizer-path", default=DEFAULT_TOKENIZER_PATH,
                    help="Path to UMT5 tokenizer (avoids re-download)")
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

    # Eval video
    ap.add_argument("--eval-video-steps", type=int, default=0,
                    help="Generate eval videos every N steps (0=off, uses VAE)")
    ap.add_argument("--eval-video-samples", type=int, default=2,
                    help="Number of eval videos to generate per eval point")
    ap.add_argument("--num-inference-steps", type=int, default=30,
                    help="Denoising steps for eval video generation")

    args = ap.parse_args()

    # Resolve relative cache-dir against MAIN_ROOT
    if not os.path.isabs(args.cache_dir):
        args.cache_dir = os.path.join(MAIN_ROOT, args.cache_dir)

    train(args)


if __name__ == "__main__":
    main()
