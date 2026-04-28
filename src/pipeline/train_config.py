"""Fixed task configuration for the maintained training entry.

`src.pipeline.train` keeps frequently changed training knobs on the CLI, but
data/cache roots are selected by task name here so experiment commands stay
short and reproducible.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from src.core.config import BASE_DIR, MAIN_ROOT, TRAINING_DATA_ROOT


@dataclass(frozen=True)
class TrainTaskConfig:
    task_name: str
    cache_train: str
    cache_eval: str = ""
    cache_ood: str = ""
    t5_cache_dir: str = ""
    output_dir: str = ""
    description: str = ""


def _main(*parts: str) -> str:
    return os.path.join(MAIN_ROOT, *parts)


def _base(*parts: str) -> str:
    return os.path.join(BASE_DIR, *parts)


def _vae_cache(name: str, *, description: str) -> TrainTaskConfig:
    root = _main("training_data", "cache", "vae", name)
    return TrainTaskConfig(
        task_name=name,
        cache_train=os.path.join(root, "train"),
        cache_eval=os.path.join(root, "eval"),
        cache_ood=os.path.join(root, "ood_eval"),
        t5_cache_dir=_main("training_data", "cache", "t5"),
        output_dir=os.path.join(TRAINING_DATA_ROOT, "log"),
        description=description,
    )


TRAIN_TASKS: dict[str, TrainTaskConfig] = {
    "pair_1s": _vae_cache(
        "pair_1s",
        description="Canonical 1s synthetic-human/real-robot paired cache.",
    ),
    "pair_1s_r2h": _vae_cache(
        "pair_1s_r2h",
        description="Robot-to-human 1s paired cache variant.",
    ),
    "pair_1s_train3": _vae_cache(
        "pair_1s_train3",
        description="Selected train3 1s paired cache variant.",
    ),
    "pair_1s_16": _vae_cache(
        "pair_1s_16",
        description="16-frame 1s paired cache variant.",
    ),
    "robot_1s": TrainTaskConfig(
        task_name="robot_1s",
        cache_train=_main("training_data", "cache", "vae", "robot_1s", "train"),
        cache_eval=_main("training_data", "cache", "vae", "robot_1s", "eval"),
        cache_ood="",
        t5_cache_dir=_main("training_data", "cache", "t5"),
        output_dir=os.path.join(TRAINING_DATA_ROOT, "log"),
        description="Robot-to-robot identity cache for staged LoRA training.",
    ),
    "attn_ffn_selected": TrainTaskConfig(
        task_name="attn_ffn_selected",
        cache_train=_main("output", "mitty_cache_1s", "train"),
        cache_eval=_main("output", "mitty_cache_1s", "eval"),
        cache_ood=_main("output", "mitty_cache_1s", "ood_eval"),
        t5_cache_dir=_main("training_data", "cache", "t5"),
        output_dir=os.path.join(TRAINING_DATA_ROOT, "log"),
        description="Selected Mitty cache used by attn+ffn LoRA experiments.",
    ),
    "smoke_t032_e2e": TrainTaskConfig(
        task_name="smoke_t032_e2e",
        cache_train=_base("tmp", "t032", "gpu_smoke", "cache_generated", "train"),
        cache_eval=_base("tmp", "t032", "gpu_smoke", "cache", "eval"),
        cache_ood="",
        t5_cache_dir=_base("tmp", "t032", "gpu_smoke", "t5"),
        output_dir=_base("tmp", "t032", "gpu_smoke", "e2e_train_run"),
        description="Local one-sample GPU smoke task.",
    ),
}


def resolve_train_task(task_name: str) -> TrainTaskConfig:
    """Return the fixed config for `task_name`, failing on unknown names."""
    try:
        return TRAIN_TASKS[task_name]
    except KeyError as exc:
        available = ", ".join(sorted(TRAIN_TASKS))
        raise ValueError(
            f"Unknown training task '{task_name}'. Available tasks: {available}"
        ) from exc


def apply_train_task_config(args) -> None:
    """Attach fixed data/cache/output paths from the selected task to args."""
    cfg = resolve_train_task(args.task_name)
    args.cache_train = cfg.cache_train
    args.cache_eval = cfg.cache_eval
    args.cache_ood = cfg.cache_ood
    args.t5_cache_dir = cfg.t5_cache_dir
    args.output_dir = cfg.output_dir
    args.task_description = cfg.description
