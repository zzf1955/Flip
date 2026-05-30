"""Mitty training method descriptor for the canonical training entry.

The repository no longer exposes experimental RectFlow/FunControl backbones as
maintained training methods.  Keep this module intentionally small so
``src.pipeline.train`` can stay method-aware without depending on legacy entry
scripts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol

import torch


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
        """Run method-specific denoising and return the decoded-ready latent."""


@dataclass(frozen=True)
class MethodSpec:
    name: str
    wandb_tag: str
    log_name: str
    description: str
    training_module_factory: Callable[..., Any]
    eval_denoise_fn: EvalDenoiseFn


from .mitty import SPEC as MITTY_SPEC  # noqa: E402


def get_mitty_spec() -> MethodSpec:
    return MITTY_SPEC
