"""Unified LoRA training entry for Wan 2.2 TI2V-5B.

Supports two backbones and two loss variants via CLI flags:

  --backbone {mitty,rectflow}   # which forward + loss formulation
  --loss     {uniform,hand_patch}  # weighting of the MSE reduction

Hand-patch loss requires ``--patch-dir``; uniform loss rejects it.

This script is a behavior-equivalent superset of
``src/pipeline/train_mitty.py`` and ``src/pipeline/train_rf.py`` — those
legacy scripts are retained for backward compatibility but new runs should
go through here for ablation convenience.

Usage:
  # Single GPU smoke: Mitty, uniform loss
  CUDA_VISIBLE_DEVICES=2 python -m src.pipeline.train \\
    --backbone mitty --loss uniform \\
    --cache-train output/mitty_cache_1s/train \\
    --cache-eval  output/mitty_cache_1s/eval \\
    --cache-ood   output/mitty_cache_1s/ood_eval \\
    --max-steps 10 --save-steps 10 --eval-steps 10 --eval-video-steps 20

  # DDP 4 GPUs: Mitty + hand_patch weighting
  torchrun --nproc_per_node=4 -m src.pipeline.train \\
    --backbone mitty --loss hand_patch \\
    --patch-dir output/hand_patch_1s/train \\
    --cache-train output/mitty_cache_1s/train \\
    --cache-eval  output/mitty_cache_1s/eval \\
    --max-steps 400

  # RectFlow + uniform
  torchrun --nproc_per_node=4 -m src.pipeline.train \\
    --backbone rectflow --loss uniform \\
    --cache-train output/mitty_cache_1s/train \\
    --max-steps 400
"""

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.stdout.reconfigure(line_buffering=True)

import torch
import torch.distributed as dist

from diffsynth.diffusion.flow_match import FlowMatchScheduler

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
from src.pipeline.backbones import BackboneSpec, all_names, get as get_backbone

# Reuse helpers from the legacy Mitty script (these are generic pipe/IO utils
# that happen to live there; keeping them there avoids touching legacy scripts).
from src.pipeline.train_mitty import (
    DEFAULT_DIT_DIR,
    DEFAULT_TOKENIZER,
    DEFAULT_VAE,
    _load_patch_weights,
    collate_batch,
    prepare_sample,
)


# ── Eval ───────────────────────────────────────────────────────────────

@torch.no_grad()
def eval_loss(model, files: list[str], device: str,
              num_t_samples: int = 5, seed_base: int = 12345,
              patch_dir: str = "") -> float:
    """Eval MSE loss averaged over every file × ``num_t_samples`` timesteps.

    The loss function inside the training module samples one random timestep
    per call, so a single eval pass on 5-8 samples has high variance. We call
    it ``num_t_samples`` times per sample with deterministic seeds so the
    metric is both low-variance and reproducible across checkpoints.

    RNG state is saved and restored so this does not perturb training.
    """
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
    spec: BackboneSpec,
    model,
    files: list[str],
    out_dir: str,
    step: int,
    device: str,
    num_samples: int = 2,
    num_inference_steps: int = 30,
    cfg_scale: float = 5.0,
    log=None,
):
    """Shared eval-video shell: backbone supplies the inner denoise loop."""
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

        denoised = spec.eval_denoise_fn(
            pipe=pipe, sample=s, sched=sched, device=device,
            cfg_scale=cfg_scale, num_inference_steps=num_inference_steps,
        )

        pipe.load_models_to_device(["vae"])
        video = vae.decode(denoised, device=device, tiled=False)
        save_video(tensor_to_frames(video), str(step_dir / f"gen_{idx:02d}.mp4"))

        if s.get("robot_frames"):
            save_video(s["robot_frames"], str(step_dir / f"gt_{idx:02d}.mp4"))
        else:
            gt_vid = vae.decode(s["robot_latent"], device=device, tiled=False)
            save_video(tensor_to_frames(gt_vid), str(step_dir / f"gt_{idx:02d}.mp4"))

        if s.get("human_frames"):
            save_video(s["human_frames"], str(step_dir / f"ctrl_{idx:02d}.mp4"))
        else:
            ctrl_vid = vae.decode(s["human_latent"], device=device, tiled=False)
            save_video(tensor_to_frames(ctrl_vid), str(step_dir / f"ctrl_{idx:02d}.mp4"))

        if log:
            log.info(f"  EVAL VIDEO [{idx+1}/{n}] → {step_dir} "
                     f"({time.time() - t0:.0f}s)")

    pipe.dit.train(was_training)


# ── Main training loop ─────────────────────────────────────────────────

def train(args, spec: BackboneSpec):
    rank, world_size, ddp_device = setup_distributed()
    is_main = rank == 0
    if ddp_device is not None:
        args.device = ddp_device

    # Deterministic timestep sampling inside the loss — the legacy train_mitty /
    # train_rf left torch.randint seeded by interpreter start time, so loss
    # values differed run-to-run. Ablation runs need reproducibility across
    # seeds, so pin the global torch RNG here (rank offset keeps DDP workers
    # decorrelated).
    torch.manual_seed(args.seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed + rank)

    # ── Data (loaded early so n_train is available for run_name) ──
    if args.eval_video_steps == -1:
        args.eval_video_steps = args.eval_steps

    train_files = load_cached_files(args.cache_train)
    eval_files = load_cached_files(args.cache_eval) if args.cache_eval else []
    if args.max_eval_files and len(eval_files) > args.max_eval_files:
        eval_files = eval_files[:args.max_eval_files]
    ood_files = load_cached_files(args.cache_ood) if args.cache_ood else []

    # ── Run name + dirs ──
    run_name = build_run_name(spec.name, args, n_train=len(train_files))
    run_dir = Path(args.output_dir) / run_name
    ckpt_dir = run_dir / "ckpt"
    eval_dir = run_dir / "eval"
    if is_main:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        eval_dir.mkdir(parents=True, exist_ok=True)
    if world_size > 1:
        dist.barrier()

    log = setup_logging(str(run_dir / "train.log"),
                        name=spec.log_name) if is_main else None

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
        tags=build_wandb_tags(spec.wandb_tag, args,
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

    info(f"Backbone: {spec.name} — {spec.description}")
    info(f"Loss: {args.loss} (patch_dir={args.patch_dir or '<none>'})")
    info(f"Run: {run_dir}")
    info(f"Args: {vars(args)}")
    info(f"DDP: rank={rank}, world_size={world_size}, device={args.device}")
    info(f"Data: train={len(train_files)} eval={len(eval_files)} ood={len(ood_files)}")

    # ── Patch weights (derived paths for eval/ood mirror the train layout) ──
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

    # ── Model (rank 0 loads from disk; others allocate empty + receive via broadcast) ──
    # All ranks print during loading so we can diagnose stalls.
    def info_all(msg):
        print(f"[rank {rank}] {msg}", flush=True)

    info_all("Loading model...")
    t0 = time.time()
    load_vae = is_main and args.eval_video_steps > 0
    skip_dit = world_size > 1 and rank != 0
    info_all(f"skip_dit={skip_dit}, load_vae={load_vae}, device={args.device}")
    extra_kwargs = {}
    if getattr(args, "merge_lora", ""):
        extra_kwargs["merge_lora_path"] = args.merge_lora
        extra_kwargs["merge_lora_rank"] = args.merge_lora_rank
    model = spec.training_module_factory(
        device=args.device,
        lora_rank=args.lora_rank,
        lora_target_modules=args.lora_target_modules,
        use_gradient_checkpointing=True,
        load_vae=load_vae,
        init_lora_path=args.init_lora,
        skip_dit_load=skip_dit,
        **extra_kwargs,
    )
    info_all(f"Model {'allocated (empty)' if skip_dit else 'loaded from disk'}"
             f" in {time.time() - t0:.1f}s")
    if getattr(model, "_merge_n", 0):
        info(f"Merged {model._merge_n} LoRA pairs into base weights"
             f" from {args.merge_lora}")
    if getattr(model, "_init_lora_n", 0):
        info(f"Loaded {model._init_lora_n} LoRA tensors from {args.init_lora}")

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    info_all(f"Params: {n_total:,} total, {n_trainable:,} trainable "
             f"({n_trainable / 1e6:.1f}M)")

    if world_size > 1:
        info_all("Waiting at broadcast barrier...")
        dist.barrier()
        info_all("Starting DiT broadcast...")
        t_bc = time.time()
        dit = model.pipe.dit
        n_params = sum(1 for _ in dit.parameters())
        for i, p in enumerate(dit.parameters()):
            dist.broadcast(p.data, src=0)
            if (i + 1) % 50 == 0:
                info_all(f"  broadcast {i+1}/{n_params} params")
        for b in dit.buffers():
            dist.broadcast(b, src=0)
        bc_gb = sum(p.numel() for p in dit.parameters()) * 2 / 1e9
        info_all(f"DiT broadcast done in {time.time() - t_bc:.1f}s "
                 f"({bc_gb:.2f} GB bf16)")

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

    info(f"Plan: {len(train_files)} train files, {total_steps} steps,"
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
                    spec, model, eval_files,
                    str(eval_dir / "in_task"), step,
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
                    spec, model, ood_files,
                    str(eval_dir / "ood"), step,
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
        description="Unified LoRA training for Wan 2.2 TI2V-5B "
                    "(Mitty / Rectified Flow backbones × uniform / hand_patch loss)"
    )

    # Backbone + loss (the ablation axes)
    ap.add_argument("--backbone", choices=all_names(), default="mitty",
                    help="training backbone (forward + loss formulation)")
    ap.add_argument("--loss", choices=["uniform", "hand_patch"], default="uniform",
                    help="loss weighting scheme; hand_patch requires --patch-dir")

    ap.add_argument("--task-name", required=True,
                    help="training task label for run name & W&B "
                         "(e.g. identity, directly_transfer, appearance)")

    ap.add_argument("--cache-train", required=True)
    ap.add_argument("--cache-eval", default="")
    ap.add_argument("--cache-ood", default="")
    ap.add_argument("--patch-dir", default="",
                    help="hand_patch weight dir for train split "
                         "(eval/ood auto-derived)")
    ap.add_argument("--output-dir",
                    default=os.path.join(TRAINING_DATA_ROOT, "log"))

    # Model paths
    ap.add_argument("--dit-dir", default=DEFAULT_DIT_DIR)
    ap.add_argument("--vae-path", default=DEFAULT_VAE)
    ap.add_argument("--tokenizer-dir", default=DEFAULT_TOKENIZER)

    # Device
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)

    # LoRA
    ap.add_argument("--lora-rank", type=int, default=96)
    ap.add_argument("--lora-target-modules", default="q,k,v,o")
    ap.add_argument("--init-lora", default="",
                    help="path to .safetensors LoRA checkpoint to initialize from")
    ap.add_argument("--merge-lora", default="",
                    help="path to LoRA checkpoint to merge into base weights "
                         "before training (e.g. identity LoRA from stage 1)")
    ap.add_argument("--merge-lora-rank", type=int, default=96,
                    help="rank of the LoRA to merge (for alpha/rank scaling)")

    # Training
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lr-min", type=float, default=1e-6,
                    help="cosine annealing end LR")
    ap.add_argument("--warmup-steps", type=int, default=50,
                    help="linear warmup steps (0 = no warmup)")
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--max-steps", type=int, required=True,
                    help="total training steps (pure step-based control)")
    ap.add_argument("--batch-size", type=int, default=1,
                    help="per-rank training batch size")
    ap.add_argument("--save-steps", type=int, default=50)
    ap.add_argument("--eval-steps", type=int, default=50)
    ap.add_argument("--eval-t-samples", type=int, default=5,
                    help="timesteps to sample per eval sample (reduces variance)")

    # Eval video
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

    # W&B
    ap.add_argument("--wandb-project", default="Flip",
                    help="W&B project name (default: 'Flip'; set to '' to disable)")
    ap.add_argument("--wandb-run-name", default=None,
                    help="W&B run name (default: timestamp)")
    ap.add_argument("--wandb-tags", nargs="+", default=[],
                    help="extra W&B tags (backbone + loss tags added automatically)")
    ap.add_argument("--wandb-log-videos", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="upload eval videos to W&B (--no-wandb-log-videos to skip)")

    args = ap.parse_args()

    # --- Ablation-flag validation ---
    if args.loss == "hand_patch" and not args.patch_dir:
        ap.error("--loss hand_patch requires --patch-dir")
    if args.loss == "uniform" and args.patch_dir:
        ap.error("--patch-dir is only valid with --loss hand_patch")

    # Resolve relative paths against MAIN_ROOT (worktree-safe; see CLAUDE.md)
    for attr in ("cache_train", "cache_eval", "cache_ood", "patch_dir",
                 "output_dir", "init_lora", "merge_lora"):
        val = getattr(args, attr)
        if val and not os.path.isabs(val):
            setattr(args, attr, os.path.join(MAIN_ROOT, val))

    spec = get_backbone(args.backbone)
    train(args, spec)


if __name__ == "__main__":
    main()
