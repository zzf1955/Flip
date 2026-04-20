"""Rectified Flow (Route A) backbone spec.

Wraps the existing ``RFTrainingModule`` (living in ``train_rf.py``) and
supplies the eval-time denoise loop extracted from ``train_rf.py``.
"""

from __future__ import annotations

import torch

from src.pipeline.train_rf import RFTrainingModule

from . import BackboneSpec, register


@torch.no_grad()
def _eval_denoise(pipe, sample, sched, device, cfg_scale, num_inference_steps):
    """RF inference: init from source latent, denoise full sequence in place."""
    del num_inference_steps  # scheduler already primed by caller
    source_lat = sample["human_latent"].to(device=device, dtype=torch.bfloat16)
    ctx_posi = sample["context_posi"].to(device=device, dtype=torch.bfloat16)
    ctx_nega = sample["context_nega"].to(device=device, dtype=torch.bfloat16)

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

        latents = sched.step(noise_pred, ts, latents)

    return latents


register(BackboneSpec(
    name="rectflow",
    wandb_tag="rectflow",
    log_name="train_rf",
    description="Rectified Flow (Route A) LoRA training — Wan 2.2 TI2V-5B",
    training_module_factory=RFTrainingModule,
    eval_denoise_fn=_eval_denoise,
))
