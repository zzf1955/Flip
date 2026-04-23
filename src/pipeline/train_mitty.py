"""Mitty-style LoRA training for Wan 2.2 TI2V-5B.

Uses `mitty_model_fn_wan_video` + `MittyFlowMatchLoss` for in-context training:
  latents = cat([human_lat (clean), robot_noisy], dim=2)
  loss = MSE only on the robot segment.

Usage:
  # Single GPU smoke
  CUDA_VISIBLE_DEVICES=2 python -m src.pipeline.train_mitty \\
    --cache-train output/mitty_cache_1s/train \\
    --cache-eval  output/mitty_cache_1s/eval \\
    --cache-ood   output/mitty_cache_1s/ood_eval \\
    --max-steps 50 --save-steps 10 --eval-steps 10 --eval-video-steps 20

  # DDP 4 GPUs (equivalent bs=4 for Mitty)
  torchrun --nproc_per_node=4 -m src.pipeline.train_mitty \\
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

from diffsynth.diffusion.flow_match import FlowMatchScheduler
from diffsynth.diffusion.training_module import DiffusionTrainingModule

from src.core.config import MAIN_ROOT, T5_CACHE_DIR, TRAINING_DATA_ROOT
from src.core.wan_loader import (
    SimplePipe,
    build_dit_shard_list,
    load_dit,
    load_vae as _load_vae,
)
from src.core.train_utils import (
    CsvLogger,
    WandbLogger,
    build_run_name,
    build_wandb_tags,
    cleanup_distributed,
    infinite_file_batches,
    load_cached_files,
    load_sample,
    load_t5_cache,
    log_step_eval_videos,
    reduce_scalar,
    save_lora_ckpt,
    save_video,
    setup_distributed,
    setup_logging,
    sync_gradients,
    tensor_to_frames,
)
from src.pipeline.mitty_model_fn import (
    mitty_model_fn_wan_video, MittyFlowMatchLoss,
)


# ── Defaults ────────────────────────────────────────────────────────────

MANUAL_DIR = os.path.join(
    "/disk_n/zzf/.cache/huggingface/hub",
    "models--Wan-AI--Wan2.2-TI2V-5B", "manual",
)
DEFAULT_T5 = os.path.join(MANUAL_DIR, "models_t5_umt5-xxl-enc-bf16.pth")
DEFAULT_DIT_DIR = MANUAL_DIR
DEFAULT_VAE = os.path.join(MANUAL_DIR, "Wan2.2_VAE.pth")
DEFAULT_TOKENIZER = os.path.join(MANUAL_DIR, "google", "umt5-xxl")


# ── LoRA merge ─────────────────────────────────────────────────────────

def merge_lora_into_weights(model, lora_path: str, lora_rank: int,
                            lora_alpha: int | None = None) -> int:
    """Merge a pre-trained LoRA checkpoint into the model's base weights.

    Computes delta = (lora_alpha / lora_rank) * lora_B @ lora_A for each
    LoRA pair and adds it to the corresponding base Linear weight in-place.
    Returns the number of merged LoRA pairs.
    """
    if lora_alpha is None:
        lora_alpha = lora_rank
    scaling = lora_alpha / lora_rank

    from safetensors.torch import load_file
    device_str = str(next(model.parameters()).device)
    sd = load_file(lora_path, device=device_str)

    if any(".module." in k for k in sd):
        sd = {k.replace(".module.", ".", 1): v for k, v in sd.items()}

    base_params = dict(model.named_parameters())
    merged = 0
    for key in sd:
        if "lora_A" not in key:
            continue
        base_key = key.replace(".lora_A.default.weight", ".weight")
        lora_B_key = key.replace("lora_A", "lora_B")
        assert base_key in base_params, \
            f"Cannot find base param {base_key} for LoRA key {key}"
        assert lora_B_key in sd, \
            f"Missing lora_B for {key}"
        lora_A = sd[key]
        lora_B = sd[lora_B_key]
        delta = (lora_B @ lora_A) * scaling
        base_params[base_key].data.add_(delta.to(base_params[base_key].dtype))
        merged += 1
    return merged


# ── Model ──────────────────────────────────────────────────────────────

def build_pipe(device: str, dit_dir: str, vae_path: str,
               tokenizer_dir: str, load_vae: bool = True,
               skip_dit_load: bool = False) -> SimplePipe:
    """Direct TI2V-5B DiT (+VAE) loader (see src/core/wan_loader.py):
    - DiT resident on `device` (bf16, ~10 GB)
    - VAE on `device` (bf16, ~0.67 GB) — no CPU parking
    - Text encoder / tokenizer not loaded (embeddings are pre-cached)
    """
    del tokenizer_dir  # unused (kept for signature compat)
    shards = build_dit_shard_list(dit_dir)
    pipe = SimplePipe(device)
    pipe.dit = load_dit(shards, device, torch.bfloat16, skip_load=skip_dit_load)
    if load_vae:
        pipe.vae = _load_vae(vae_path, torch.bfloat16, home_device=device)
    return pipe


class MittyTrainingModule(DiffusionTrainingModule):
    """Training wrapper: loads pipe, injects LoRA on DiT, installs Mitty model_fn."""

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
        merge_lora_paths: list[str] | None = None,
        merge_lora_rank: int = 96,
    ):
        super().__init__()
        self.pipe = build_pipe(device, dit_dir, vae_path, tokenizer_dir,
                               load_vae=load_vae, skip_dit_load=skip_dit_load)
        self.pipe.scheduler.set_timesteps(1000, training=True)

        # Merge pre-trained LoRA(s) into base weights before freeze + new LoRA
        self._merge_n = 0
        for p in (merge_lora_paths or []):
            self._merge_n += merge_lora_into_weights(
                self.pipe.dit, p, merge_lora_rank)

        # Freeze everything
        for name, module in self.pipe.named_children():
            for p in module.parameters():
                p.requires_grad_(False)

        # Inject LoRA into DiT
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
            # Legacy ckpts from DiffSynth's AutoWrappedLinear path carry a
            # `.module.` prefix on every LoRA key (e.g. `blocks.0.self_attn.q` →
            # `blocks.0.module.self_attn.q`). The new loader uses bare Linears,
            # so strip that prefix when present to keep resume-from-old working.
            if any(".module." in k for k in sd):
                sd = {k.replace(".module.", ".", 1): v for k, v in sd.items()}
            result = self.pipe.dit.load_state_dict(sd, strict=False)
            if result.unexpected_keys:
                raise ValueError(
                    f"LoRA checkpoint has unexpected keys: "
                    f"{result.unexpected_keys[:5]}")
            self._init_lora_n = len(sd)
        else:
            self._init_lora_n = 0

        # Install Mitty forward
        self.pipe.model_fn = mitty_model_fn_wan_video

        # Training mode + grad ckpt
        self.pipe.dit.train()
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.device = device

    def forward(self, sample: dict) -> torch.Tensor:
        sample = dict(sample)  # shallow copy
        sample["use_gradient_checkpointing"] = self.use_gradient_checkpointing
        return MittyFlowMatchLoss(self.pipe, **sample)


# ── Sample IO ──────────────────────────────────────────────────────────

def _load_patch_weights(sample: dict, cache_path: str, patch_dir: str,
                        device: str = "cpu"):
    """Attach patch_weights from a companion .pth in patch_dir."""
    patch_path = os.path.join(patch_dir, os.path.basename(cache_path))
    if os.path.isfile(patch_path):
        pd_ = torch.load(patch_path, map_location=device, weights_only=False)
        sample["patch_weights"] = pd_["weights"]


def prepare_sample(sample: dict, device: str,
                   dtype=torch.bfloat16) -> dict:
    """Move cached latents + context to device, keep PIL/str fields as-is."""
    out = {}
    for k, v in sample.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device=device, dtype=dtype if v.is_floating_point() else v.dtype)
        else:
            out[k] = v
    # Rename robot_latent → input_latents for MittyFlowMatchLoss compatibility
    if "robot_latent" in out:
        out["input_latents"] = out.pop("robot_latent")
    if "context_posi" in out and "context" not in out:
        out["context"] = out["context_posi"]
    return out


def collate_batch(samples: list[dict], device: str,
                  dtype=torch.bfloat16) -> dict:
    """Stack multiple cached samples into a true batch.

    WanModel supports batch>1 in DiTBlock: t_mod shape (1, N, 6, dim) broadcasts
    to (B, N, dim) after squeeze(2). The whole batch shares one timestep (sampled
    once per batch), consistent with Mitty's flow-match training.
    """
    batch = {
        "human_latent": torch.cat([s["human_latent"] for s in samples], dim=0).to(device, dtype),
        "input_latents": torch.cat([s["robot_latent"] for s in samples], dim=0).to(device, dtype),
        "context": torch.cat([s["context_posi"] for s in samples], dim=0).to(device, dtype),
    }
    if any("patch_weights" in s for s in samples):
        pw = []
        for s in samples:
            if "patch_weights" in s:
                pw.append(s["patch_weights"])
            else:
                f_R = s["robot_latent"].shape[2]
                h, w = s["robot_latent"].shape[3], s["robot_latent"].shape[4]
                pw.append(torch.ones(f_R, h, w, dtype=dtype))
        batch["patch_weights"] = torch.stack(pw, dim=0).to(device, dtype)
    return batch


# ── Eval ───────────────────────────────────────────────────────────────

@torch.no_grad()
def eval_loss(model: MittyTrainingModule, files: list[str], device: str,
              num_t_samples: int = 5, seed_base: int = 12345,
              patch_dir: str = "",
              t5_pos: dict = None, t5_neg: torch.Tensor = None) -> float:
    """Eval MSE loss averaged over every file × `num_t_samples` timesteps.

    MittyFlowMatchLoss samples one random timestep + noise per call, so a single
    eval pass on 5-8 samples has high variance. We call it `num_t_samples` times
    per sample with deterministic seeds so the eval metric is both low-variance
    and reproducible across checkpoints.

    RNG state is saved and restored so this does not perturb training.
    """
    torch_state = torch.get_rng_state()
    cuda_state = (torch.cuda.get_rng_state(device)
                  if torch.cuda.is_available() else None)
    try:
        losses = []
        for i, f in enumerate(files):
            s = load_sample(f, device=device,
                            t5_pos=t5_pos, t5_neg=t5_neg)
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
    model: MittyTrainingModule,
    files: list[str],
    out_dir: str,
    step: int,
    device: str,
    num_samples: int = 2,
    num_inference_steps: int = 30,
    cfg_scale: float = 5.0,
    log=None,
    t5_pos: dict = None,
    t5_neg: torch.Tensor = None,
):
    """Mitty H2R zero-frame denoising: cat(clean_human, noise) → denoise robot only."""
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
        s = load_sample(files[idx], device=device,
                        t5_pos=t5_pos, t5_neg=t5_neg)
        human_lat = s["human_latent"].to(device=device, dtype=torch.bfloat16)
        robot_lat_shape = s["robot_latent"].shape
        ctx_posi = s["context_posi"].to(device=device, dtype=torch.bfloat16)
        ctx_nega = s["context_nega"].to(device=device, dtype=torch.bfloat16)
        f_H = human_lat.shape[2]

        # Initialize robot latent as noise
        robot_noisy = torch.randn(robot_lat_shape, device=device, dtype=torch.bfloat16)

        for ts in sched.timesteps:
            t_tensor = ts.unsqueeze(0).to(dtype=torch.bfloat16, device=device)
            latents = torch.concat([human_lat, robot_noisy], dim=2)

            pred_posi = pipe.model_fn(
                dit=pipe.dit, latents=latents, timestep=t_tensor,
                context=ctx_posi, mitty_human_frames=f_H,
                use_gradient_checkpointing=False,
            )
            if cfg_scale != 1.0:
                pred_nega = pipe.model_fn(
                    dit=pipe.dit, latents=latents, timestep=t_tensor,
                    context=ctx_nega, mitty_human_frames=f_H,
                    use_gradient_checkpointing=False,
                )
                noise_pred = pred_nega + cfg_scale * (pred_posi - pred_nega)
            else:
                noise_pred = pred_posi

            # Update only the robot segment
            noise_pred_robot = noise_pred[:, :, f_H:]
            robot_noisy = sched.step(noise_pred_robot, ts, robot_noisy)

        # VAE decode
        pipe.load_models_to_device(["vae"])
        video = vae.decode(robot_noisy, device=device, tiled=False)
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
    run_name = build_run_name("mitty", args, n_train=len(train_files))
    run_dir = Path(args.output_dir) / run_name
    ckpt_dir = run_dir / "ckpt"
    eval_dir = run_dir / "eval"
    if is_main:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        eval_dir.mkdir(parents=True, exist_ok=True)
    if world_size > 1:
        dist.barrier()

    log = setup_logging(str(run_dir / "train.log"),
                        name="train_mitty") if is_main else None

    # ── CSV ──
    csv_headers = [
        "step", "train_loss", "lr", "time_s",
        "eval_loss_in_task", "eval_loss_ood",
        "save_ckpt", "eval_video",
    ]
    csv_logger = CsvLogger(str(run_dir / "train.csv"),
                           csv_headers) if is_main else None

    # ── W&B (rank 0 only; no-op if --wandb-project not set) ──
    wb = WandbLogger(
        project=args.wandb_project if is_main else None,
        run_name=args.wandb_run_name or run_name,
        config=vars(args),
        tags=build_wandb_tags("mitty", args,
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

    # ── T5 cache ──
    t5_pos, t5_neg = {}, None
    if args.t5_cache_dir and os.path.isdir(args.t5_cache_dir):
        t5_pos, t5_neg = load_t5_cache(args.t5_cache_dir, device="cpu")
        info(f"T5 cache: {len(t5_pos)} prompts + negative from {args.t5_cache_dir}")
    elif args.t5_cache_dir:
        info(f"T5 cache dir not found: {args.t5_cache_dir} (old-format cache assumed)")

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
    model = MittyTrainingModule(
        device=args.device,
        lora_rank=args.lora_rank,
        lora_target_modules=args.lora_target_modules,
        use_gradient_checkpointing=True,
        load_vae=load_vae,
        init_lora_path=args.init_lora,
        merge_lora_paths=args.merge_lora,
        merge_lora_rank=args.merge_lora_rank,
    )
    info(f"Model loaded in {time.time() - t0:.1f}s (load_vae={load_vae})")
    if model._merge_n:
        info(f"Merged {model._merge_n} LoRA pairs into base weights"
             f" from {len(args.merge_lora)} checkpoint(s)")
    if model._init_lora_n:
        info(f"Loaded {model._init_lora_n} LoRA tensors from {args.init_lora}")

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    info(f"Params: {n_total:,} total, {n_trainable:,} trainable ({n_trainable / 1e6:.1f}M)")

    # Sync initial LoRA weights across ranks
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

    # LR schedule: warmup (linear 0→lr) + cosine (lr→lr_min)
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
            s = load_sample(f, device=args.device,
                            t5_pos=t5_pos, t5_neg=t5_neg)
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

        # Hit eval / eval-video at step 1 too (baseline snapshot)
        hit_eval = bool(args.eval_steps) and (
            step == 1 or step % args.eval_steps == 0)
        hit_eval_video = bool(args.eval_video_steps) and (
            step == 1 or step % args.eval_video_steps == 0)

        # Checkpoint (rank 0)
        if args.save_steps and step % args.save_steps == 0 and is_main:
            p = ckpt_dir / f"step-{step:04d}.safetensors"
            n = save_lora_ckpt(model, str(p))
            info(f"  SAVE {p} ({n} tensors)")
            row_fields["save_ckpt"] = p.name

        # Eval loss (rank 0 only); averaged over files × num_t_samples t
        if hit_eval and is_main:
            eval_payload = {}
            if eval_files:
                el = eval_loss(model, eval_files, args.device,
                               num_t_samples=args.eval_t_samples,
                               patch_dir=patch_dir_eval,
                               t5_pos=t5_pos, t5_neg=t5_neg)
                info(f"  EVAL eval_loss_in_task={el:.4f} "
                     f"({len(eval_files)} samples × {args.eval_t_samples} t)")
                row_fields["eval_loss_in_task"] = f"{el:.4f}"
                eval_payload["train/eval_loss_in_task"] = el
            if ood_files:
                ol = eval_loss(model, ood_files, args.device,
                               num_t_samples=args.eval_t_samples,
                               patch_dir=patch_dir_ood,
                               t5_pos=t5_pos, t5_neg=t5_neg)
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
                    t5_pos=t5_pos, t5_neg=t5_neg,
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
                    t5_pos=t5_pos, t5_neg=t5_neg,
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

    # Final checkpoint (if not at save boundary)
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
    ap = argparse.ArgumentParser(description="Mitty LoRA training (Wan 2.2 TI2V-5B)")

    ap.add_argument("--task-name", required=True,
                    help="training task label for run name & W&B "
                         "(e.g. identity, directly_transfer, appearance)")

    ap.add_argument("--cache-train", required=True)
    ap.add_argument("--cache-eval", default="")
    ap.add_argument("--cache-ood", default="")
    ap.add_argument("--t5-cache-dir", default=T5_CACHE_DIR,
                    help="shared T5 embedding cache dir "
                         "(default: training_data/cache/t5/)")
    ap.add_argument("--patch-dir", default="",
                    help="hand_patch weight dir for train split "
                         "(eval/ood auto-derived; empty = uniform loss)")
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
    ap.add_argument("--merge-lora", action="append", default=None,
                    help="LoRA checkpoint to merge into base weights "
                         "(can be specified multiple times)")
    ap.add_argument("--merge-lora-rank", type=int, default=96,
                    help="rank of the LoRA(s) to merge (for alpha/rank scaling)")

    # Training
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lr-min", type=float, default=1e-6,
                    help="cosine annealing end LR")
    ap.add_argument("--warmup-steps", type=int, default=50,
                    help="linear warmup steps (0 = no warmup)")
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--max-steps", type=int, required=True,
                    help="Total training steps (data cycles infinitely)")
    ap.add_argument("--batch-size", type=int, default=1,
                    help="per-rank training batch size (real batch; the whole "
                         "batch shares one sampled timestep)")
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
                    help="extra W&B tags (in addition to 'mitty')")
    ap.add_argument("--wandb-log-videos", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="upload eval videos to W&B (--no-wandb-log-videos to skip)")

    args = ap.parse_args()

    # Resolve relative paths
    for attr in ("cache_train", "cache_eval", "cache_ood", "t5_cache_dir",
                 "patch_dir", "output_dir", "init_lora"):
        val = getattr(args, attr)
        if val and not os.path.isabs(val):
            setattr(args, attr, os.path.join(MAIN_ROOT, val))
    if args.merge_lora:
        args.merge_lora = [
            os.path.join(MAIN_ROOT, p) if not os.path.isabs(p) else p
            for p in args.merge_lora
        ]

    train(args)


if __name__ == "__main__":
    main()
