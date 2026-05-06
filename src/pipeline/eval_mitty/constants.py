"""Shared defaults for offline Mitty evaluation."""

from __future__ import annotations

from pathlib import Path

from src.core.config import TRAINING_DATA_ROOT

DEFAULT_RUNS = [
    "Mitty-transfer-124d_r128_2000s_0425_1456",
    "Mitty-transfer2LoRA-124d_r128_2000s_0425_1425",
]
DEFAULT_SPLITS = ["in_task_eval", "ood_eval"]
DEFAULT_SAM2_MASK_ROOT = Path(TRAINING_DATA_ROOT) / "sam2_mask"
DEFAULT_LOCAL_VIDEO_SIZE = 300
SPLIT_ALIASES = {
    "eval": "in_task_eval",
    "in_task": "in_task_eval",
    "in_task_eval": "in_task_eval",
    "ood": "ood_eval",
    "ood_eval": "ood_eval",
}

