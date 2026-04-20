"""Backbone registry for the unified training entry (`src/pipeline/train.py`).

Each backbone is a ``BackboneSpec`` bundling:
  - a training module factory (wraps DiT + LoRA + forward+loss)
  - an eval-time denoise function (differs between Mitty-cat and RectFlow)
  - logger/tag metadata so the train loop can be backbone-agnostic

Adding a new backbone: create ``src/pipeline/backbones/<name>.py`` that
imports ``BackboneSpec`` + ``register`` from here, then eager-import it in
this file's bottom block so it auto-registers at package load.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol

import torch

from diffsynth.diffusion.training_module import DiffusionTrainingModule


class EvalDenoiseFn(Protocol):
    def __call__(
        self,
        pipe: Any,
        sample: dict,
        sched: Any,
        device: str,
        cfg_scale: float,
        num_inference_steps: int,
    ) -> torch.Tensor:
        """Run the backbone-specific denoising loop and return the final
        latent (ready for VAE decode). Outer shell handles IO + decode."""


@dataclass(frozen=True)
class BackboneSpec:
    name: str                                        # "mitty" | "rectflow"
    wandb_tag: str                                   # "mitty" | "rectflow"
    log_name: str                                    # "train_mitty" | "train_rf"
    description: str                                 # info line printed at startup
    training_module_factory: Callable[..., DiffusionTrainingModule]
    eval_denoise_fn: EvalDenoiseFn


_REGISTRY: dict[str, BackboneSpec] = {}


def register(spec: BackboneSpec) -> None:
    if spec.name in _REGISTRY:
        raise ValueError(f"backbone {spec.name!r} already registered")
    _REGISTRY[spec.name] = spec


def get(name: str) -> BackboneSpec:
    if name not in _REGISTRY:
        raise ValueError(
            f"unknown backbone {name!r}; known: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name]


def all_names() -> list[str]:
    return sorted(_REGISTRY)


from . import mitty, rectflow  # noqa: E402,F401
