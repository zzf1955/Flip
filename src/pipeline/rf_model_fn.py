"""Rectified Flow forward and loss for Wan 2.2 TI2V-5B.

Route A: replace Gaussian noise with source video latent in the flow-match
formulation.  No temporal concatenation — the DiT processes a single 5-frame
latent that is a σ-interpolation between target and source.

    noisy = (1-σ) * target + σ * source
    velocity_target = source - target
    inference: start from source, denoise to target

Usage:
    from src.pipeline.rf_model_fn import (
        rf_model_fn_wan_video, RFFlowMatchLoss,
    )
    pipe.model_fn = rf_model_fn_wan_video
"""

from __future__ import annotations

import torch
from einops import rearrange

from diffsynth.core import gradient_checkpoint_forward
from diffsynth.models.wan_video_dit import WanModel, sinusoidal_embedding_1d


# ── forward ──────────────────────────────────────────────────────────────

def rf_model_fn_wan_video(
    dit: WanModel,
    latents: torch.Tensor,
    timestep: torch.Tensor,
    context: torch.Tensor,
    use_gradient_checkpointing: bool = False,
    use_gradient_checkpointing_offload: bool = False,
    **ignored,
):
    """Uniform-timestep forward for Rectified Flow.

    Unlike Mitty (first f_H frames at t=0), ALL patches receive the same
    sampled timestep.  Still uses the ``seperated_timestep`` code path so
    the per-patch AdaLN modulation matches the pre-trained weight layout.

    Args:
        dit: WanModel with seperated_timestep=True
        latents: (B, 48, f, h, w) — σ-interpolated latent (5 frames)
        timestep: scalar flow-match timestep tensor (shape ``(1,)``)
        context: T5 text embedding (B, 512, text_dim)
    """
    assert dit.seperated_timestep, \
        "rf_model_fn requires DiT with seperated_timestep=True (Wan 2.2 TI2V-5B)"

    ps = dit.patch_size  # (1, 2, 2)
    patch_f = latents.shape[2] // ps[0]
    spatial_per_frame = (latents.shape[3] // ps[1]) * (latents.shape[4] // ps[2])
    num_patches = patch_f * spatial_per_frame

    # Uniform timestep for every patch
    ts = torch.full(
        (num_patches,), timestep.item(),
        dtype=latents.dtype, device=latents.device,
    )

    t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, ts).unsqueeze(0))
    t_mod = dit.time_projection(t).unflatten(2, (6, dit.dim))
    context = dit.text_embedding(context)

    x = latents
    if x.shape[0] != context.shape[0]:
        x = torch.concat([x] * context.shape[0], dim=0)

    x = dit.patchify(x)
    f, h, w = x.shape[2:]
    x = rearrange(x, "b c f h w -> b (f h w) c").contiguous()

    freqs = torch.cat([
        dit.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
        dit.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
        dit.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
    ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)

    for block in dit.blocks:
        if dit.training:
            x = gradient_checkpoint_forward(
                block,
                use_gradient_checkpointing,
                use_gradient_checkpointing_offload,
                x, context, t_mod, freqs,
            )
        else:
            x = block(x, context, t_mod, freqs)

    x = dit.head(x, t)
    x = dit.unpatchify(x, (f, h, w))
    return x


# ── loss ────────────────────────────────────────────────────────────────

def RFFlowMatchLoss(pipe, **inputs):
    """Rectified Flow loss: source latent replaces Gaussian noise.

    Expected ``inputs`` keys:
        human_latent  (B, 48, f, h, w)  source VAE latent (the "noise" endpoint)
        input_latents (B, 48, f, h, w)  target VAE latent (clean robot)
        context       (B, 512, text_dim) T5 embedding
    """
    max_t = int(inputs.get("max_timestep_boundary", 1) * len(pipe.scheduler.timesteps))
    min_t = int(inputs.get("min_timestep_boundary", 0) * len(pipe.scheduler.timesteps))

    source_lat: torch.Tensor = inputs["human_latent"]
    target_lat: torch.Tensor = inputs["input_latents"]
    assert source_lat.shape == target_lat.shape, \
        f"source/target shape mismatch: {source_lat.shape} vs {target_lat.shape}"

    # Sample timestep
    tid = torch.randint(min_t, max_t, (1,))
    timestep = pipe.scheduler.timesteps[tid].to(
        dtype=pipe.torch_dtype, device=pipe.device)

    # Rectified Flow: source latent IS the noise
    noisy_latents = pipe.scheduler.add_noise(target_lat, source_lat, timestep)
    training_target = pipe.scheduler.training_target(target_lat, source_lat, timestep)

    mf_inputs = dict(inputs)
    mf_inputs["latents"] = noisy_latents
    mf_inputs.pop("human_latent", None)
    mf_inputs.pop("input_latents", None)
    patch_weights = mf_inputs.pop("patch_weights", None)

    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    noise_pred = pipe.model_fn(**models, **mf_inputs, timestep=timestep)

    # Loss on ALL frames (no segment slicing)
    if patch_weights is not None:
        if patch_weights.dim() == 3:
            patch_weights = patch_weights.unsqueeze(0)
        w = patch_weights.unsqueeze(1).float()
        mse = (noise_pred.float() - training_target.float()).pow(2)
        loss = (mse * w).mean()
    else:
        loss = torch.nn.functional.mse_loss(
            noise_pred.float(), training_target.float())
    loss = loss * pipe.scheduler.training_weight(timestep)
    return loss
