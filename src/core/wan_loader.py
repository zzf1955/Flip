"""Fast direct loader for Wan2.2 TI2V-5B DiT + VAE + T5 (training path only).

Replaces DiffSynth's `WanVideoPipeline.from_pretrained`:
- Skips `hash_model_file` / `DiskMap` metadata scans
- Skips `enable_vram_management` wrapping (AutoWrappedLinear is overhead when
  offload_device == onload_device == cuda)
- Skips the 25 PipelineUnit instantiations
- All models load directly to target GPU — no CPU staging
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
from diffsynth.models.wan_video_text_encoder import WanTextEncoder
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
    progress_prefix: str = "",
) -> dict[str, torch.Tensor]:
    sd: dict[str, torch.Tensor] = {}
    with safe_open(path, framework="pt", device=device) as f:
        keys = list(f.keys())
        n = len(keys)
        for i, k in enumerate(keys):
            sd[k] = f.get_tensor(k).to(dtype)
            if progress_prefix and (i + 1) % 100 == 0:
                print(f"  {progress_prefix} {i+1}/{n} tensors", flush=True)
    if progress_prefix:
        print(f"  {progress_prefix} {n}/{n} tensors done", flush=True)
    return sd


def _load_shards(
    paths: Iterable[str], dtype: torch.dtype, device: str = "cpu",
) -> dict[str, torch.Tensor]:
    paths = list(paths)
    sd: dict[str, torch.Tensor] = {}
    if len(paths) == 1:
        return _read_safetensors(
            paths[0], dtype, device,
            progress_prefix=os.path.basename(paths[0]),
        )
    if device == "cpu":
        with ThreadPoolExecutor(max_workers=min(len(paths), 4)) as ex:
            for part in ex.map(lambda p: _read_safetensors(p, dtype, device), paths):
                sd.update(part)
    else:
        for i, p in enumerate(paths):
            sd.update(_read_safetensors(
                p, dtype, device,
                progress_prefix=f"[{i+1}/{len(paths)}] {os.path.basename(p)}",
            ))
    return sd


def load_dit(
    shard_paths: Iterable[str],
    device: str | torch.device,
    dtype: torch.dtype = torch.bfloat16,
    skip_load: bool = False,
) -> WanModel:
    """Load Wan2.2 TI2V-5B DiT directly to `device` via safetensors.

    When ``skip_load=True`` the model structure is allocated on ``device``
    with dtype ``dtype`` but no weights are read from disk.  Used for DDP
    broadcast: rank 0 loads normally, others call with ``skip_load=True``
    and receive weights via NCCL.
    """
    with skip_model_initialization():
        model = WanModel(**WAN22_TI2V_5B_DIT_CONFIG)
    if skip_load:
        model.to_empty(device=device)
        for p in model.parameters():
            p.data = p.data.to(dtype=dtype)
        for b in model.buffers():
            b.data = b.data.to(dtype=dtype)
    else:
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
    """Load Wan2.2 VAE directly to `home_device`."""
    dev = str(home_device)
    with skip_model_initialization():
        vae = WanVideoVAE38(z_dim=48, dim=160)

    if path.endswith(".safetensors"):
        sd = _read_safetensors(path, dtype, device=dev)
    else:
        sd = torch.load(path, map_location=dev, weights_only=True)
        if isinstance(sd, dict) and len(sd) == 1 and "model_state" in sd:
            sd = sd["model_state"]
        sd = WanVideoVAEStateDictConverter(sd)
        for k, v in list(sd.items()):
            if isinstance(v, torch.Tensor) and v.dtype != dtype:
                sd[k] = v.to(dtype=dtype)

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
    """Return DiT weight paths in ``dit_dir``.

    Prefers a single pre-converted bf16 file over the original FP32 shards.
    """
    bf16_path = os.path.join(dit_dir, "diffusion_pytorch_model-bf16.safetensors")
    if os.path.isfile(bf16_path):
        return [bf16_path]
    shards = sorted(
        os.path.join(dit_dir, f)
        for f in os.listdir(dit_dir)
        if f.startswith("diffusion_pytorch_model-") and f.endswith(".safetensors")
    )
    if not shards:
        raise FileNotFoundError(f"No DiT shards in {dit_dir}")
    return shards


# ── T5 text encoder ──────────────────────────────────────────────────

WAN22_T5_CONFIG = {
    "vocab": 256384,
    "dim": 4096,
    "dim_attn": 4096,
    "dim_ffn": 10240,
    "num_heads": 64,
    "num_layers": 24,
    "num_buckets": 32,
    "shared_pos": False,
    "dropout": 0.1,
}


def load_text_encoder(
    path: str,
    device: str | torch.device,
    dtype: torch.dtype = torch.bfloat16,
) -> WanTextEncoder:
    """Load Wan2.2 T5 text encoder directly to `device`."""
    with skip_model_initialization():
        model = WanTextEncoder(**WAN22_T5_CONFIG)
    sd = torch.load(path, map_location=str(device), weights_only=True)
    for k, v in list(sd.items()):
        if isinstance(v, torch.Tensor) and v.dtype != dtype:
            sd[k] = v.to(dtype=dtype)
    result = model.load_state_dict(sd, assign=True, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(
            f"T5 state_dict mismatch: missing={result.missing_keys[:5]}, "
            f"unexpected={result.unexpected_keys[:5]}"
        )
    return model.eval()


def load_tokenizer(tokenizer_dir: str, seq_len: int = 512):
    """Load HuggingFace T5 tokenizer (no DiffSynth wrapper)."""
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(tokenizer_dir), seq_len
