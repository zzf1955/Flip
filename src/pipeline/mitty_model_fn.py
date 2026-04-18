"""Mitty-style forward and flow-match loss for Wan 2.2 TI2V-5B.

This is a minimal fork of DiffSynth's `model_fn_wan_video` that only supports
the branch used by Wan 2.2 TI2V-5B (seperated_timestep + fuse_vae_embedding),
with the "first-frame clean" mask extended to cover the first `f_H` frames
(Mitty in-context: human demonstration is kept clean, robot latent is denoised).

Usage (wire into a WanVideoPipeline):

    from src.pipeline.mitty_model_fn import (
        mitty_model_fn_wan_video, MittyFlowMatchLoss,
    )
    pipe.model_fn = mitty_model_fn_wan_video
    # In training loop, pass latents = cat([human_lat, robot_noisy], dim=2)
    # and set inputs["mitty_human_frames"] = human_lat.shape[2]
"""

from __future__ import annotations

import torch
from einops import rearrange

from diffsynth.core import gradient_checkpoint_forward
from diffsynth.models.wan_video_dit import WanModel, sinusoidal_embedding_1d


# ── forward ──────────────────────────────────────────────────────────────

def mitty_model_fn_wan_video(
    dit: WanModel,
    latents: torch.Tensor,
    timestep: torch.Tensor,
    context: torch.Tensor,
    mitty_human_frames: int = 0,
    use_gradient_checkpointing: bool = False,
    use_gradient_checkpointing_offload: bool = False,
    **ignored,
):
    """Minimal forward: only the TI2V-5B (seperated_timestep) branch.

    Args:
        dit: WanModel with seperated_timestep=True, fuse_vae_embedding_in_latents=True
        latents: (B, 48, f_H + f_R, h, w) — clean human + noisy robot, concat on temporal
        timestep: scalar flow-match timestep tensor (shape `(1,)`)
        context: T5 text embedding (B, 512, text_dim)
        mitty_human_frames: number of **latent** frames that must be treated as
            clean (t=0). Must be >0; typical value 5 for 17-frame human video.
    """
    assert mitty_human_frames > 0, "mitty_human_frames must be set (>0)"
    assert dit.seperated_timestep, \
        "mitty_model_fn requires DiT with seperated_timestep=True (Wan 2.2 TI2V-5B)"

    f_H = mitty_human_frames
    ps = dit.patch_size  # (1, 2, 2) for TI2V-5B
    patch_f_H = f_H // ps[0]
    patch_f_total = latents.shape[2] // ps[0]
    spatial_per_frame = (latents.shape[3] // ps[1]) * (latents.shape[4] // ps[2])

    # Per-patch timestep: first `patch_f_H` frames → 0 (clean), rest → sampled t
    ts = torch.concat([
        torch.zeros((patch_f_H, spatial_per_frame),
                    dtype=latents.dtype, device=latents.device),
        torch.ones((patch_f_total - patch_f_H, spatial_per_frame),
                   dtype=latents.dtype, device=latents.device) * timestep,
    ]).flatten()

    t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, ts).unsqueeze(0))
    t_mod = dit.time_projection(t).unflatten(2, (6, dit.dim))
    context = dit.text_embedding(context)

    x = latents
    # CFG merge (if context batch > latents batch)
    if x.shape[0] != context.shape[0]:
        x = torch.concat([x] * context.shape[0], dim=0)

    # Patchify
    x = dit.patchify(x)  # (B, dim, f, h, w)
    f, h, w = x.shape[2:]
    x = rearrange(x, "b c f h w -> b (f h w) c").contiguous()

    # 3D RoPE
    freqs = torch.cat([
        dit.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
        dit.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
        dit.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
    ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)

    # Transformer blocks
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

    # Head + unpatchify
    x = dit.head(x, t)
    x = dit.unpatchify(x, (f, h, w))
    return x


# ── loss ────────────────────────────────────────────────────────────────

def MittyFlowMatchLoss(pipe, **inputs):
    """Flow-match loss with Mitty in-context denoising.

    Expected `inputs` keys (set by training loop):
        human_latent (1, 48, f_H, h, w)  clean VAE latent, MUST be bf16
        input_latents (1, 48, f_R, h, w) clean robot latent (the denoise target)
        context      (1, 512, text_dim)  T5 embedding

    The loss is MSE only on the robot segment of the DiT output.
    """
    max_t = int(inputs.get("max_timestep_boundary", 1) * len(pipe.scheduler.timesteps))
    min_t = int(inputs.get("min_timestep_boundary", 0) * len(pipe.scheduler.timesteps))

    human_lat: torch.Tensor = inputs["human_latent"]
    robot_lat: torch.Tensor = inputs["input_latents"]
    assert human_lat.dim() == 5 and robot_lat.dim() == 5, \
        f"latents must be (B,C,F,H,W), got {human_lat.shape} / {robot_lat.shape}"
    f_H = human_lat.shape[2]

    # Sample timestep
    tid = torch.randint(min_t, max_t, (1,))
    timestep = pipe.scheduler.timesteps[tid].to(
        dtype=pipe.torch_dtype, device=pipe.device)

    # Noise only the robot latent
    noise = torch.randn_like(robot_lat)
    robot_noisy = pipe.scheduler.add_noise(robot_lat, noise, timestep)
    training_target = pipe.scheduler.training_target(robot_lat, noise, timestep)

    # Concat clean human + noisy robot along temporal
    latents = torch.concat([human_lat, robot_noisy], dim=2)

    # Build inputs for model_fn
    mf_inputs = dict(inputs)
    mf_inputs["latents"] = latents
    mf_inputs["mitty_human_frames"] = f_H
    # Drop keys our minimal model_fn doesn't use, but keep anything the wrapper
    # might expect. ignored kwargs are swallowed by **ignored.
    mf_inputs.pop("human_latent", None)
    mf_inputs.pop("input_latents", None)
    patch_weights = mf_inputs.pop("patch_weights", None)

    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    noise_pred = pipe.model_fn(**models, **mf_inputs, timestep=timestep)

    noise_pred_robot = noise_pred[:, :, f_H:]
    if patch_weights is not None:
        w = patch_weights.unsqueeze(1).float()  # (B,1,f_R,H,W)
        mse = (noise_pred_robot.float() - training_target.float()).pow(2)
        loss = (mse * w).mean()
    else:
        loss = torch.nn.functional.mse_loss(
            noise_pred_robot.float(), training_target.float())
    loss = loss * pipe.scheduler.training_weight(timestep)
    return loss
