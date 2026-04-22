"""
Custom LoRA training for Wan 2.1 FunControl.

Reuses DiffSynth's WanTrainingModule for model loading + LoRA + forward pass,
but adds a proper training loop with:
- Train / Eval data split (deterministic, fixed seed)
- Per-step train loss logging
- Periodic eval loss computation
- Eval video generation (ground truth + control + generated)
- Multi-GPU DDP support (torchrun)
- Organized output: ckpt/, eval/, train.log

Usage:
  # Single GPU
  python -m src.pipeline.train_lora \
    --cache-dir output/data_cache_80 \
    --device cuda:0 --max-steps 50 \
    --save-steps 5 --eval-steps 5 --eval-video-steps 50

  # Multi-GPU DDP (4 GPUs)
  torchrun --nproc_per_node=4 -m src.pipeline.train_lora \
    --cache-dir output/data_cache_80 --max-steps 400 \
    --save-steps 50 --eval-steps 50 --eval-video-steps 50
"""

import argparse
import os
import random
import sys
import time

# Reduce CUDA memory fragmentation — must precede torch import
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.stdout.reconfigure(line_buffering=True)

import torch
import torch.distributed as dist

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
from src.core.train_utils import (  # noqa: E402
    CsvLogger,
    WandbLogger,
    build_run_name,
    build_wandb_tags,
    cleanup_distributed,
    infinite_file_batches,
    load_cached_files,
    load_sample,
    log_step_eval_videos,
    reduce_scalar,
    save_lora_ckpt,
    save_video,
    setup_distributed,
    setup_logging,
    sync_gradients,
    tensor_to_frames,
)


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

def split_train_eval(files: list[str], eval_ratio: float, seed: int):
    """Deterministic split into train and eval sets."""
    if eval_ratio <= 0:
        return list(files), []
    rng = random.Random(seed)
    shuffled = list(files)
    rng.shuffle(shuffled)
    n_eval = max(1, int(len(shuffled) * eval_ratio))
    return shuffled[n_eval:], shuffled[:n_eval]


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
    # ── DDP setup ──
    rank, world_size, ddp_device = setup_distributed()
    is_main = (rank == 0)
    if ddp_device is not None:
        args.device = ddp_device  # override --device with LOCAL_RANK

    # ── Data (loaded early so n_train is available for run_name) ──
    all_files = load_cached_files(args.cache_dir, recursive=True)
    train_files, eval_files = split_train_eval(all_files, args.eval_ratio, args.seed)

    # ── Run name + dirs ──
    run_name = build_run_name("lora", args, n_train=len(train_files))
    run_dir = os.path.join(args.output_dir, run_name)
    ckpt_dir = os.path.join(run_dir, "ckpt")
    eval_dir = os.path.join(run_dir, "eval")
    if is_main:
        os.makedirs(ckpt_dir, exist_ok=True)
        os.makedirs(eval_dir, exist_ok=True)
    if world_size > 1:
        dist.barrier()

    log = setup_logging(os.path.join(run_dir, "train.log"),
                        name="train_lora") if is_main else None

    # ── CSV ──
    csv_headers = [
        "step", "train_loss", "lr", "time_s",
        "eval_loss", "save_ckpt", "eval_video",
    ]
    csv_logger = CsvLogger(os.path.join(run_dir, "train.csv"),
                           csv_headers) if is_main else None

    # ── W&B (rank 0 only; no-op if --wandb-project not set) ──
    wb = WandbLogger(
        project=args.wandb_project if is_main else None,
        run_name=args.wandb_run_name or run_name,
        config=vars(args),
        tags=build_wandb_tags("lora", args,
                              n_train=len(train_files),
                              world_size=world_size,
                              extra_tags=args.wandb_tags),
        dir=run_dir,
    )

    def info(msg):
        if log:
            log.info(msg)

    def write_csv_row(**fields):
        if csv_logger is not None:
            csv_logger.write(**fields)

    info(f"Run: {run_dir}")
    info(f"Args: {vars(args)}")
    info(f"DDP: rank={rank}, world_size={world_size}, device={args.device}")
    info(f"Data: {len(all_files)} total -> {len(train_files)} train, {len(eval_files)} eval")

    # ── Model (DiT + optional VAE) ──
    info("Loading model...")
    t0 = time.time()

    # Build model_paths JSON — DiT always, VAE if eval videos requested (rank 0 only)
    if args.vae_path and args.eval_video_steps > 0 and is_main:
        model_paths = f'["{args.dit_path}", "{args.vae_path}"]'
        info("VAE included for eval video generation (rank 0 only)")
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
    info(f"Model loaded in {time.time() - t0:.1f}s")

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    info(f"Params: {n_total:,} total, {n_trainable:,} trainable ({n_trainable / 1e6:.1f}M)")

    # ── Sync initial LoRA weights across ranks ──
    if world_size > 1:
        for p in model.parameters():
            if p.requires_grad:
                dist.broadcast(p.data, src=0)
        info("LoRA weights synced across ranks")

    # ── Optimizer ──
    optimizer = torch.optim.AdamW(
        model.trainable_modules(), lr=args.lr, weight_decay=args.weight_decay,
    )
    lr_scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)

    # ── Training loop ──
    total_steps = args.max_steps
    info(f"Plan: {total_steps} steps, {len(train_files)} train files, cycling infinitely")
    info(f"DDP: world_size={world_size}, effective_bs={world_size}")
    info(f"Save every {args.save_steps}, eval loss every {args.eval_steps}"
         f", eval video every {args.eval_video_steps} steps")
    info("=" * 60)

    step = 0
    train_t0 = time.time()

    data_iter = infinite_file_batches(
        train_files, batch_size=1,
        world_size=world_size, rank=rank, seed=args.seed,
    )

    for batch in data_iter:
        if step >= total_steps:
            break
        fpath = batch[0]
        step += 1
        st = time.time()

        data = load_sample(fpath)
        optimizer.zero_grad()
        loss = model({}, inputs=data)
        loss.backward()

        if world_size > 1:
            sync_gradients(model)

        optimizer.step()
        lr_scheduler.step()

        loss_val = loss.item()
        if world_size > 1:
            loss_val = reduce_scalar(loss_val, args.device)

        step_time = time.time() - st
        lr = optimizer.param_groups[0]["lr"]
        info(
            f"step={step}/{total_steps} train_loss={loss_val:.4f}"
            f" lr={lr:.1e} time={step_time:.1f}s"
        )

        row_fields = {
            "step": step,
            "train_loss": f"{loss_val:.4f}",
            "lr": f"{lr:.3e}",
            "time_s": f"{step_time:.2f}",
        }
        if is_main:
            wb.log({
                "train/loss": loss_val,
                "train/lr": lr,
                "train/time_s": step_time,
            }, step=step)

        hit_eval = (args.eval_steps
                    and step % args.eval_steps == 0 and eval_files)
        hit_eval_video = (args.eval_video_steps
                          and step % args.eval_video_steps == 0 and eval_files)

        # Checkpoint (rank 0 only)
        if args.save_steps and step % args.save_steps == 0 and is_main:
            p = os.path.join(ckpt_dir, f"step-{step:04d}.safetensors")
            n = save_lora_ckpt(model, p)
            info(f"  SAVE {p} ({n} tensors)")
            row_fields["save_ckpt"] = os.path.basename(p)

        # Eval loss (rank 0 only)
        if hit_eval and is_main:
            losses = []
            with torch.no_grad():
                for ef in eval_files:
                    ed = load_sample(ef)
                    el = model({}, inputs=ed)
                    losses.append(el.item())
            avg = sum(losses) / len(losses)
            info(f"  EVAL eval_loss={avg:.4f} ({len(losses)} samples)")
            row_fields["eval_loss"] = f"{avg:.4f}"
            wb.log({"train/eval_loss_in_task": avg}, step=step)

        # Eval videos (rank 0 only)
        if hit_eval_video and is_main:
            generate_eval_videos(
                model, eval_files, eval_dir, step, args.device,
                num_samples=args.eval_video_samples,
                num_inference_steps=args.num_inference_steps,
                log=log,
            )
            row_fields["eval_video"] = f"step-{step:04d}"
            if args.wandb_log_videos:
                log_step_eval_videos(
                    wb, os.path.join(eval_dir, f"step-{step:04d}"), step,
                )

        write_csv_row(**row_fields)

        if world_size > 1 and (hit_eval or hit_eval_video):
            dist.barrier()

    # Final checkpoint if not already saved at last step
    if args.save_steps and step > 0 and step % args.save_steps != 0 and is_main:
        p = os.path.join(ckpt_dir, f"step-{step:04d}.safetensors")
        save_lora_ckpt(model, p)
        info(f"  SAVE (final) {p}")

    # Final eval videos (rank 0 only)
    if (args.eval_video_steps and step > 0
            and step % args.eval_video_steps != 0 and eval_files and is_main):
        generate_eval_videos(
            model, eval_files, eval_dir, step, args.device,
            num_samples=args.eval_video_samples,
            num_inference_steps=args.num_inference_steps,
            log=log,
        )
        if args.wandb_log_videos:
            _log_step_videos(wb, "eval/video_in_task",
                             os.path.join(eval_dir, f"step-{step:04d}"), step)

    elapsed = time.time() - train_t0
    info("=" * 60)
    if step > 0:
        info(f"Done. {step} steps in {elapsed:.0f}s ({elapsed / step:.1f}s/step)")

    if csv_logger is not None:
        csv_logger.close()
    wb.finish()

    cleanup_distributed()


# ── CLI ───────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Custom LoRA training for Wan 2.1 FunControl")

    ap.add_argument("--task-name", default="",
                    help="training task label for run name & W&B "
                         "(e.g. identity, directly_transfer, appearance)")

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
    ap.add_argument("--max-steps", type=int, required=True,
                    help="Total training steps (data cycles infinitely)")
    ap.add_argument("--save-steps", type=int, default=50)
    ap.add_argument("--eval-steps", type=int, default=50)

    # Eval video
    ap.add_argument("--eval-video-steps", type=int, default=0,
                    help="Generate eval videos every N steps (0=off, uses VAE)")
    ap.add_argument("--eval-video-samples", type=int, default=2,
                    help="Number of eval videos to generate per eval point")
    ap.add_argument("--num-inference-steps", type=int, default=30,
                    help="Denoising steps for eval video generation")

    # W&B
    ap.add_argument("--wandb-project", default="Flip",
                    help="W&B project name (default: 'Flip'; set to '' to disable)")
    ap.add_argument("--wandb-run-name", default=None,
                    help="W&B run name (default: timestamp)")
    ap.add_argument("--wandb-tags", nargs="+", default=[],
                    help="extra W&B tags (in addition to 'fun-control')")
    ap.add_argument("--wandb-log-videos", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="upload eval videos to W&B (--no-wandb-log-videos to skip)")

    args = ap.parse_args()

    # Resolve relative cache-dir against MAIN_ROOT
    if not os.path.isabs(args.cache_dir):
        args.cache_dir = os.path.join(MAIN_ROOT, args.cache_dir)

    train(args)


if __name__ == "__main__":
    main()
