"""Mitty in-context method spec."""

from __future__ import annotations

import torch

from src.pipeline.train_mitty import MittyTrainingModule

from . import MethodSpec


@torch.no_grad()
def _eval_denoise(pipe, sample, sched, device, cfg_scale, num_inference_steps):
    """H2R zero-frame denoising: cat(clean_human, noisy_robot) → denoise robot only."""
    del num_inference_steps  # scheduler already primed by caller
    human_lat = sample["human_latent"].to(device=device, dtype=torch.bfloat16)
    robot_lat_shape = sample["robot_latent"].shape
    ctx_posi = sample["context_posi"].to(device=device, dtype=torch.bfloat16)
    ctx_nega = sample["context_nega"].to(device=device, dtype=torch.bfloat16)
    f_H = human_lat.shape[2]

    robot_noisy = torch.randn(
        robot_lat_shape, device=device, dtype=torch.bfloat16,
    )

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

        noise_pred_robot = noise_pred[:, :, f_H:]
        robot_noisy = sched.step(noise_pred_robot, ts, robot_noisy)

    return robot_noisy


SPEC = MethodSpec(
    name="mitty",
    wandb_tag="mitty",
    log_name="train_mitty",
    description="Mitty LoRA training (Wan 2.2 TI2V-5B)",
    training_module_factory=MittyTrainingModule,
    eval_denoise_fn=_eval_denoise,
)
