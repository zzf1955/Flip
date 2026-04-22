"""Rectified Flow (Route A) LoRA training for Wan 2.2 TI2V-5B.

Uses ``rf_model_fn_wan_video`` + ``RFFlowMatchLoss``:
  noise = source_latent (not Gaussian)
  noisy = (1-σ)*target + σ*source  →  5 frames, uniform timestep
  loss = MSE on all frames

Usage:
  # Single GPU smoke
  CUDA_VISIBLE_DEVICES=2 python -m src.pipeline.train_rf \\
    --cache-train output/mitty_cache_1s/train \\
    --cache-eval  output/mitty_cache_1s/eval \\
    --cache-ood   output/mitty_cache_1s/ood_eval \\
    --max-steps 50 --save-steps 10 --eval-steps 10 --eval-video-steps 20

  # DDP 2 GPUs
  torchrun --nproc_per_node=2 -m src.pipeline.train_rf \\
    --cache-train output/mitty_cache_1s/train \\
    --cache-eval  output/mitty_cache_1s/eval \\
    --cache-ood   output/mitty_cache_1s/ood_eval \\
    --max-steps 400 --save-steps 50 --eval-steps 50 --eval-video-steps 100
"""

import argparse
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.stdout.reconfigure(line_buffering=True)

import torch
import torch.distributed as dist

from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from diffsynth.diffusion.flow_match import FlowMatchScheduler
from diffsynth.diffusion.training_module import DiffusionTrainingModule

from src.core.config import MAIN_ROOT, TRAINING_DATA_ROOT
from src.core.train_utils import (
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
from src.pipeline.rf_model_fn import rf_model_fn_wan_video, RFFlowMatchLoss

# Reuse model loading + data helpers from Mitty (identical logic)
from src.pipeline.train_mitty import (
    build_pipe,
    _load_patch_weights,
    prepare_sample,
    collate_batch,
    MANUAL_DIR,
    DEFAULT_DIT_DIR,
    DEFAULT_VAE,
    DEFAULT_TOKENIZER,
)


# ── Model ──────────────────────────────────────────────────────────────

class RFTrainingModule(DiffusionTrainingModule):
    """Training wrapper: loads pipe, injects LoRA on DiT, installs RF model_fn."""

    def __init__(
        self,
        device: str,
        dit_dir: str = DEFAULT_DIT_DIR,
        vae_path: str = DEFAULT_VAE,
        tokenizer_dir: str = DEFAULT_TOKENIZER,
        lora_rank: int = 96,
        lora_target_modules: str = "q,k,v,o",
        use_gradient_checkpointing: bool = True,
        load_vae: bool = True,
        init_lora_path: str = "",
        skip_dit_load: bool = False,
    ):
        super().__init__()
        self.pipe = build_pipe(device, dit_dir, vae_path, tokenizer_dir,
                               load_vae=load_vae, skip_dit_load=skip_dit_load)
        self.pipe.scheduler.set_timesteps(1000, training=True)

        for name, module in self.pipe.named_children():
            for p in module.parameters():
                p.requires_grad_(False)

        target_modules = [m.strip() for m in lora_target_modules.split(",")]
        if len(target_modules) == 1:
            target_modules = target_modules[0]
        self.pipe.dit = self.add_lora_to_model(
            self.pipe.dit,
            target_modules=target_modules,
            lora_rank=lora_rank,
            upcast_dtype=torch.bfloat16,
        )

        if init_lora_path:
            from safetensors.torch import load_file
            sd = load_file(init_lora_path, device=str(device))
            result = self.pipe.dit.load_state_dict(sd, strict=False)
            if result.unexpected_keys:
                raise ValueError(
                    f"LoRA checkpoint has unexpected keys: "
                    f"{result.unexpected_keys[:5]}")
            self._init_lora_n = len(sd)
        else:
            self._init_lora_n = 0

        # Install RF forward (uniform timestep, no concat)
        self.pipe.model_fn = rf_model_fn_wan_video

        self.pipe.dit.train()
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.device = device

    def forward(self, sample: dict) -> torch.Tensor:
        sample = dict(sample)
        sample["use_gradient_checkpointing"] = self.use_gradient_checkpointing
        return RFFlowMatchLoss(self.pipe, **sample)


# ── Eval ───────────────────────────────────────────────────────────────

@torch.no_grad()
def eval_loss(model: RFTrainingModule, files: list[str], device: str,
              num_t_samples: int = 5, seed_base: int = 12345,
              patch_dir: str = "") -> float:
    """Eval loss averaged over files × num_t_samples timesteps."""
    torch_state = torch.get_rng_state()
    cuda_state = (torch.cuda.get_rng_state(device)
                  if torch.cuda.is_available() else None)
    try:
        losses = []
        for i, f in enumerate(files):
            s = load_sample(f, device=device)
            if patch_dir:
                _load_patch_weights(s, f, patch_dir, device=device)
            s = prepare_sample(s, device)
            sub = []
            for k in range(num_t_samples):
                torch.manual_seed(seed_base + i * num_t_samples + k)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed_base + i * num_t_samples + k)
                sub.append(model(s).item())
            losses.append(sum(sub) / num_t_samples)
        return sum(losses) / max(1, len(losses))
    finally:
        torch.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state(cuda_state, device)


@torch.no_grad()
def generate_eval_videos(
    model: RFTrainingModule,
    files: list[str],
    out_dir: str,
    step: int,
    device: str,
    num_samples: int = 2,
    num_inference_steps: int = 30,
    cfg_scale: float = 5.0,
    log=None,
):
    """RF inference: start from source latent, denoise to target."""
    pipe = model.pipe
    vae = getattr(pipe, "vae", None)
    if vae is None:
        if log:
            log.info("  EVAL VIDEO skipped (no VAE)")
        return

    sched = FlowMatchScheduler("Wan")
    sched.set_timesteps(num_inference_steps=num_inference_steps,
                        denoising_strength=1.0, shift=5.0)

    step_dir = Path(out_dir) / f"step-{step:04d}"
    step_dir.mkdir(parents=True, exist_ok=True)

    was_training = pipe.dit.training
    pipe.dit.eval()

    n = min(num_samples, len(files))
    for idx in range(n):
        t0 = time.time()
        s = load_sample(files[idx], device=device)
        source_lat = s["human_latent"].to(device=device, dtype=torch.bfloat16)
        ctx_posi = s["context_posi"].to(device=device, dtype=torch.bfloat16)
        ctx_nega = s["context_nega"].to(device=device, dtype=torch.bfloat16)

        if s.get("robot_frames"):
            save_video(s["robot_frames"], str(step_dir / f"gt_{idx:02d}.mp4"))
        if s.get("human_frames"):
            save_video(s["human_frames"], str(step_dir / f"ctrl_{idx:02d}.mp4"))

        # RF: initialize from source latent (not noise)
        latents = source_lat.clone()

        for ts in sched.timesteps:
            t_tensor = ts.unsqueeze(0).to(dtype=torch.bfloat16, device=device)

            pred_posi = pipe.model_fn(
                dit=pipe.dit, latents=latents, timestep=t_tensor,
                context=ctx_posi,
                use_gradient_checkpointing=False,
            )
            if cfg_scale != 1.0:
                pred_nega = pipe.model_fn(
                    dit=pipe.dit, latents=latents, timestep=t_tensor,
                    context=ctx_nega,
                    use_gradient_checkpointing=False,
                )
                noise_pred = pred_nega + cfg_scale * (pred_posi - pred_nega)
            else:
                noise_pred = pred_posi

            # Update ALL frames (no segment slicing)
            latents = sched.step(noise_pred, ts, latents)

        pipe.load_models_to_device(["vae"])
        video = vae.decode(latents, device=device, tiled=False)
        frames = tensor_to_frames(video)
        save_video(frames, str(step_dir / f"gen_{idx:02d}.mp4"))

        if log:
            log.info(f"  EVAL VIDEO [{idx+1}/{n}] → {step_dir} ({time.time() - t0:.0f}s)")

    pipe.dit.train(was_training)


# ── Main training loop ─────────────────────────────────────────────────

def train(args):
    rank, world_size, ddp_device = setup_distributed()
    is_main = rank == 0
    if ddp_device is not None:
        args.device = ddp_device

    # ── Data (loaded early so n_train is available for run_name) ──
    if args.eval_video_steps == -1:
        args.eval_video_steps = args.eval_steps

    train_files = load_cached_files(args.cache_train)
    eval_files = load_cached_files(args.cache_eval) if args.cache_eval else []
    if args.max_eval_files and len(eval_files) > args.max_eval_files:
        eval_files = eval_files[:args.max_eval_files]
    ood_files = load_cached_files(args.cache_ood) if args.cache_ood else []

    # ── Run name + dirs ──
    run_name = build_run_name("rf", args, n_train=len(train_files))
    run_dir = Path(args.output_dir) / run_name
    ckpt_dir = run_dir / "ckpt"
    eval_dir = run_dir / "eval"
    if is_main:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        eval_dir.mkdir(parents=True, exist_ok=True)
    if world_size > 1:
        dist.barrier()

    log = setup_logging(str(run_dir / "train.log"),
                        name="train_rf") if is_main else None

    csv_headers = [
        "step", "train_loss", "lr", "time_s",
        "eval_loss_in_task", "eval_loss_ood",
        "save_ckpt", "eval_video",
    ]
    csv_logger = CsvLogger(str(run_dir / "train.csv"),
                           csv_headers) if is_main else None

    wb = WandbLogger(
        project=args.wandb_project if is_main else None,
        run_name=args.wandb_run_name or run_name,
        config=vars(args),
        tags=build_wandb_tags("rf", args,
                              n_train=len(train_files),
                              world_size=world_size,
                              extra_tags=args.wandb_tags),
        dir=str(run_dir),
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
    info(f"Data: train={len(train_files)} eval={len(eval_files)} ood={len(ood_files)}")

    # ── Patch weights ──
    patch_dir_eval = ""
    patch_dir_ood = ""
    if args.patch_dir:
        patch_leaf = os.path.basename(args.patch_dir)
        patch_parent = os.path.dirname(os.path.dirname(args.patch_dir))
        patch_dir_eval = os.path.join(patch_parent, "eval", patch_leaf)
        patch_dir_ood = os.path.join(patch_parent, "ood_eval", patch_leaf)
        info(f"Patch weights: train={args.patch_dir}"
             f" eval={patch_dir_eval} ood={patch_dir_ood}")
    else:
        info("Patch weights: disabled (uniform loss)")

    # ── Model (safetensors reads directly to GPU — no CPU staging) ──
    info("Loading model...")
    t0 = time.time()
    load_vae = is_main and args.eval_video_steps > 0
    model = RFTrainingModule(
        device=args.device,
        lora_rank=args.lora_rank,
        lora_target_modules=args.lora_target_modules,
        use_gradient_checkpointing=True,
        load_vae=load_vae,
        init_lora_path=args.init_lora,
    )
    info(f"Model loaded in {time.time() - t0:.1f}s (load_vae={load_vae})")
    if model._init_lora_n:
        info(f"Loaded {model._init_lora_n} LoRA tensors from {args.init_lora}")

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    info(f"Params: {n_total:,} total, {n_trainable:,} trainable ({n_trainable / 1e6:.1f}M)")

    if world_size > 1:
        for p in model.parameters():
            if p.requires_grad:
                dist.broadcast(p.data, src=0)
        info("LoRA weights synced across ranks")

    optimizer = torch.optim.AdamW(
        model.trainable_modules(), lr=args.lr, weight_decay=args.weight_decay,
    )

    total_steps = args.max_steps
    effective_bs = args.batch_size * world_size

    if args.warmup_steps > 0 and total_steps > args.warmup_steps:
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1e-3, end_factor=1.0,
            total_iters=args.warmup_steps,
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps - args.warmup_steps,
            eta_min=args.lr_min,
        )
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, cosine],
            milestones=[args.warmup_steps],
        )
    elif total_steps > 0:
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps, eta_min=args.lr_min,
        )
    else:
        lr_scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)

    info(f"Plan: {total_steps} steps, {len(train_files)} train files, cycling infinitely")
    info(f"DDP: world_size={world_size}, batch_size={args.batch_size},"
         f" effective_bs={effective_bs}")
    info(f"LR: warmup {args.warmup_steps} steps → cosine to {args.lr_min:.1e}"
         f" (lr={args.lr:.1e})")
    info(f"Save every {args.save_steps}, eval every {args.eval_steps}"
         f", eval video every {args.eval_video_steps}")
    info("=" * 60)

    step = 0
    train_t0 = time.time()

    data_iter = infinite_file_batches(
        train_files, batch_size=args.batch_size,
        world_size=world_size, rank=rank, seed=args.seed,
    )

    def _prefetch(files):
        out = []
        for f in files:
            s = load_sample(f, device=args.device)
            if args.patch_dir:
                _load_patch_weights(s, f, args.patch_dir, device=args.device)
            out.append(s)
        return out

    prefetch_ex = ThreadPoolExecutor(max_workers=1)
    next_batch = next(data_iter)
    pending = prefetch_ex.submit(_prefetch, next_batch)

    while step < total_steps:
        samples = pending.result()
        next_batch = next(data_iter)
        pending = prefetch_ex.submit(_prefetch, next_batch)

        step += 1
        st = time.time()
        batch = collate_batch(samples, args.device)
        optimizer.zero_grad()
        loss = model(batch)
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
        info(f"step={step}/{total_steps} train_loss={loss_val:.4f}"
             f" lr={lr:.1e} time={step_time:.1f}s")

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

        hit_eval = bool(args.eval_steps) and (
            step == 1 or step % args.eval_steps == 0)
        hit_eval_video = bool(args.eval_video_steps) and (
            step == 1 or step % args.eval_video_steps == 0)

        if args.save_steps and step % args.save_steps == 0 and is_main:
            p = ckpt_dir / f"step-{step:04d}.safetensors"
            n = save_lora_ckpt(model, str(p))
            info(f"  SAVE {p} ({n} tensors)")
            row_fields["save_ckpt"] = p.name

        if hit_eval and is_main:
            eval_payload = {}
            if eval_files:
                el = eval_loss(model, eval_files, args.device,
                               num_t_samples=args.eval_t_samples,
                               patch_dir=patch_dir_eval)
                info(f"  EVAL eval_loss_in_task={el:.4f} "
                     f"({len(eval_files)} samples × {args.eval_t_samples} t)")
                row_fields["eval_loss_in_task"] = f"{el:.4f}"
                eval_payload["train/eval_loss_in_task"] = el
            if ood_files:
                ol = eval_loss(model, ood_files, args.device,
                               num_t_samples=args.eval_t_samples,
                               patch_dir=patch_dir_ood)
                info(f"  EVAL eval_loss_ood={ol:.4f} "
                     f"({len(ood_files)} samples × {args.eval_t_samples} t)")
                row_fields["eval_loss_ood"] = f"{ol:.4f}"
                eval_payload["train/eval_loss_ood"] = ol
            if eval_payload:
                wb.log(eval_payload, step=step)

        if hit_eval_video and is_main:
            n_in = (args.eval_video_samples_in_task
                    if args.eval_video_samples_in_task > 0
                    else len(eval_files))
            n_ood = (args.eval_video_samples_ood
                     if args.eval_video_samples_ood > 0
                     else len(ood_files))
            if eval_files:
                generate_eval_videos(
                    model, eval_files, str(eval_dir / "in_task"), step,
                    args.device,
                    num_samples=n_in,
                    num_inference_steps=args.num_inference_steps,
                    log=log,
                )
                if args.wandb_log_videos:
                    log_step_eval_videos(
                        wb, str(eval_dir / "in_task" / f"step-{step:04d}"),
                        step, split_tag="in_task",
                    )
            if ood_files:
                generate_eval_videos(
                    model, ood_files, str(eval_dir / "ood"), step,
                    args.device,
                    num_samples=n_ood,
                    num_inference_steps=args.num_inference_steps,
                    log=log,
                )
                if args.wandb_log_videos:
                    log_step_eval_videos(
                        wb, str(eval_dir / "ood" / f"step-{step:04d}"),
                        step, split_tag="ood",
                    )
            row_fields["eval_video"] = f"step-{step:04d}"

        write_csv_row(**row_fields)

        if world_size > 1 and (hit_eval or hit_eval_video):
            dist.barrier()

    prefetch_ex.shutdown(wait=False)

    if args.save_steps and step > 0 and step % args.save_steps != 0 and is_main:
        p = ckpt_dir / f"step-{step:04d}.safetensors"
        save_lora_ckpt(model, str(p))
        info(f"  SAVE (final) {p}")

    elapsed = time.time() - train_t0
    info("=" * 60)
    if step > 0:
        info(f"Done. {step} steps in {elapsed:.0f}s ({elapsed / step:.1f}s/step)")

    if csv_logger is not None:
        csv_logger.close()
    wb.finish()

    cleanup_distributed()


def main():
    ap = argparse.ArgumentParser(
        description="Rectified Flow (Route A) LoRA training — Wan 2.2 TI2V-5B")

    ap.add_argument("--cache-train", required=True)
    ap.add_argument("--cache-eval", default="")
    ap.add_argument("--cache-ood", default="")
    ap.add_argument("--patch-dir", default="",
                    help="hand_patch weight dir for train split "
                         "(eval/ood auto-derived; empty = uniform loss)")
    ap.add_argument("--output-dir",
                    default=os.path.join(TRAINING_DATA_ROOT, "log"))

    ap.add_argument("--dit-dir", default=DEFAULT_DIT_DIR)
    ap.add_argument("--vae-path", default=DEFAULT_VAE)
    ap.add_argument("--tokenizer-dir", default=DEFAULT_TOKENIZER)

    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--lora-rank", type=int, default=96)
    ap.add_argument("--lora-target-modules", default="q,k,v,o")
    ap.add_argument("--init-lora", default="",
                    help="path to .safetensors LoRA checkpoint to initialize from")

    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lr-min", type=float, default=1e-6,
                    help="cosine annealing end LR")
    ap.add_argument("--warmup-steps", type=int, default=50,
                    help="linear warmup steps (0 = no warmup)")
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--max-steps", type=int, required=True,
                    help="Total training steps (data cycles infinitely)")
    ap.add_argument("--batch-size", type=int, default=1,
                    help="per-rank training batch size")
    ap.add_argument("--save-steps", type=int, default=50)
    ap.add_argument("--eval-steps", type=int, default=50)
    ap.add_argument("--eval-t-samples", type=int, default=5,
                    help="timesteps to sample per eval sample (reduces variance)")

    ap.add_argument("--eval-video-steps", type=int, default=-1,
                    help="generate eval videos every N steps "
                         "(-1=follow eval-steps, 0=off, needs VAE)")
    ap.add_argument("--eval-video-samples-in-task", type=int, default=4,
                    help="N in-task eval videos per trigger (-1 = all)")
    ap.add_argument("--eval-video-samples-ood", type=int, default=2,
                    help="N OOD eval videos per trigger (-1 = all)")
    ap.add_argument("--max-eval-files", type=int, default=0,
                    help="cap eval file count (0=no cap)")
    ap.add_argument("--num-inference-steps", type=int, default=30)

    ap.add_argument("--wandb-project", default="Flip",
                    help="W&B project name (default: 'Flip'; set to '' to disable)")
    ap.add_argument("--wandb-run-name", default=None,
                    help="W&B run name (default: timestamp)")
    ap.add_argument("--wandb-tags", nargs="+", default=[],
                    help="extra W&B tags (in addition to 'rectflow')")
    ap.add_argument("--wandb-log-videos", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="upload eval videos to W&B (--no-wandb-log-videos to skip)")

    args = ap.parse_args()

    for attr in ("cache_train", "cache_eval", "cache_ood", "patch_dir", "output_dir",
                 "init_lora"):
        val = getattr(args, attr)
        if val and not os.path.isabs(val):
            setattr(args, attr, os.path.join(MAIN_ROOT, val))

    train(args)


if __name__ == "__main__":
    main()
