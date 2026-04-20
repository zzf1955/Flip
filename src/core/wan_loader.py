"""Fast direct loader for Wan2.2 TI2V-5B DiT + Wan2.2 VAE (training path only).

Replaces DiffSynth's `WanVideoPipeline.from_pretrained` for train_mitty:
- Skips `hash_model_file` / `DiskMap` metadata scans
- Skips `enable_vram_management` wrapping (AutoWrappedLinear is overhead when
  offload_device == onload_device == cuda)
- Skips the 25 PipelineUnit instantiations
- DiT shards read in parallel with ThreadPoolExecutor
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Iterable

import torch
from safetensors import safe_open
from torch import nn

from diffsynth.diffusion.flow_match import FlowMatchScheduler
from diffsynth.models.wan_video_dit import WanModel
from diffsynth.models.wan_video_vae import WanVideoVAE38
from diffsynth.core.vram.initialization import skip_model_initialization
from diffsynth.utils.state_dict_converters.wan_video_vae import (
    WanVideoVAEStateDictConverter,
)


# Config vendored from diffsynth/configs/model_configs.py:290-295
# (model_hash = "1f5ab7703c6fc803fdded85ff040c316" — Wan2.2 TI2V-5B DiT).
WAN22_TI2V_5B_DIT_CONFIG = {
    "has_image_input": False,
    "patch_size": [1, 2, 2],
    "in_dim": 48,
    "dim": 3072,
    "ffn_dim": 14336,
    "freq_dim": 256,
    "text_dim": 4096,
    "out_dim": 48,
    "num_heads": 24,
    "num_layers": 30,
    "eps": 1e-06,
    "seperated_timestep": True,
    "require_clip_embedding": False,
    "require_vae_embedding": False,
    "fuse_vae_embedding_in_latents": True,
}


def _read_safetensors(
    path: str, dtype: torch.dtype, device: str = "cpu",
) -> dict[str, torch.Tensor]:
    sd: dict[str, torch.Tensor] = {}
    with safe_open(path, framework="pt", device=device) as f:
        for k in f.keys():
            sd[k] = f.get_tensor(k).to(dtype)
    return sd


def _load_shards(
    paths: Iterable[str], dtype: torch.dtype, device: str = "cpu",
) -> dict[str, torch.Tensor]:
    paths = list(paths)
    sd: dict[str, torch.Tensor] = {}
    if len(paths) == 1:
        return _read_safetensors(paths[0], dtype, device)
    if device == "cpu":
        with ThreadPoolExecutor(max_workers=min(len(paths), 4)) as ex:
            for part in ex.map(lambda p: _read_safetensors(p, dtype, device), paths):
                sd.update(part)
    else:
        for p in paths:
            sd.update(_read_safetensors(p, dtype, device))
    return sd


def load_dit(
    shard_paths: Iterable[str],
    device: str | torch.device,
    dtype: torch.dtype = torch.bfloat16,
) -> WanModel:
    """Load Wan2.2 TI2V-5B DiT directly to `device` via safetensors.

    Reads tensors straight to GPU — no CPU staging, no staggered DDP loading needed.
    """
    with skip_model_initialization():
        model = WanModel(**WAN22_TI2V_5B_DIT_CONFIG)
    sd = _load_shards(shard_paths, dtype, device=str(device))
    result = model.load_state_dict(sd, assign=True, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(
            f"DiT state_dict mismatch: missing={result.missing_keys[:5]}, "
            f"unexpected={result.unexpected_keys[:5]}"
        )
    return model.eval()


def load_vae(
    path: str,
    dtype: torch.dtype = torch.bfloat16,
    home_device: str | torch.device = "cpu",
) -> WanVideoVAE38:
    """Load Wan2.2 VAE. Parked on `home_device` (default cpu). Actual compute
    device is chosen later via SimplePipe.load_models_to_device(["vae"]).
    """
    with skip_model_initialization():
        vae = WanVideoVAE38(z_dim=48, dim=160)

    if path.endswith(".safetensors"):
        sd = _read_safetensors(path, dtype)
    else:
        sd = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(sd, dict) and len(sd) == 1 and "model_state" in sd:
            sd = sd["model_state"]
        sd = WanVideoVAEStateDictConverter(sd)
        for k, v in list(sd.items()):
            if isinstance(v, torch.Tensor) and v.dtype != dtype:
                sd[k] = v.to(dtype)

    result = vae.load_state_dict(sd, assign=True, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(
            f"VAE state_dict mismatch: missing={result.missing_keys[:5]}, "
            f"unexpected={result.unexpected_keys[:5]}"
        )
    return vae.to(device=home_device, dtype=dtype).eval()


class SimplePipe(nn.Module):
    """Minimal replacement for WanVideoPipeline used by train_mitty + mitty_model_fn.

    Only exposes the attributes train_mitty actually touches:
      .device, .dit, .vae, .scheduler, .model_fn, .load_models_to_device(names)
    """

    # Mirrors WanVideoPipeline: names fetched and passed as kwargs to model_fn.
    # Mitty only uses DiT; other slots from DiffSynth (motion_controller/vace/...)
    # are irrelevant and would be None anyway.
    in_iteration_models = ("dit",)

    def __init__(self, device: str | torch.device,
                 torch_dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.device = device
        self.torch_dtype = torch_dtype
        self.dit: WanModel | None = None
        self.vae: WanVideoVAE38 | None = None
        self.scheduler = FlowMatchScheduler("Wan")
        self.model_fn = None  # set by caller

    def load_models_to_device(self, names: list[str]):
        """Move VAE between CPU home and GPU. DiT is resident on GPU (no-op)."""
        if self.vae is not None:
            if "vae" in names:
                if str(next(self.vae.parameters()).device) != str(self.device):
                    self.vae.to(self.device)
            else:
                if str(next(self.vae.parameters()).device) != "cpu":
                    self.vae.to("cpu")


def build_dit_shard_list(dit_dir: str) -> list[str]:
    """Return sorted list of `diffusion_pytorch_model-*.safetensors` in `dit_dir`."""
    shards = sorted(
        os.path.join(dit_dir, f)
        for f in os.listdir(dit_dir)
        if f.startswith("diffusion_pytorch_model-") and f.endswith(".safetensors")
    )
    if not shards:
        raise FileNotFoundError(f"No DiT shards in {dit_dir}")
    return shards
