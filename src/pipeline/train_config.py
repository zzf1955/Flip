"""Runtime data presets for the maintained training entry.

`src.pipeline.train` keeps model/training knobs on the CLI. Dataset presets here
only provide defaults for data type, duration, runtime task split, T5 cache, and
output root; explicit CLI values override them.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from src.core.config import (
    BASE_DIR,
    DEFAULT_OOD_TASKS,
    DEFAULT_TRAIN_TASKS,
    MAIN_ROOT,
    TRAINING_DATA_ROOT,
)
from src.pipeline.runtime_data import short_task_name


def _csv_tasks(tasks: list[str]) -> str:
    return ",".join(short_task_name(task) for task in tasks)


@dataclass(frozen=True)
class TrainTaskConfig:
    task_name: str
    data_type: str
    duration: str = "1s"
    train_tasks: str = _csv_tasks(DEFAULT_TRAIN_TASKS)
    ood_tasks: str = _csv_tasks(DEFAULT_OOD_TASKS)
    cache_root: str = ""
    t5_cache_dir: str = ""
    output_dir: str = ""
    description: str = ""


def _main(*parts: str) -> str:
    return os.path.join(MAIN_ROOT, *parts)


def _base(*parts: str) -> str:
    return os.path.join(BASE_DIR, *parts)


def _task_t5_cache(*parts: str) -> str:
    return _main("training_data", "cache", "t5", *parts)


def _preset(
    name: str,
    data_type: str,
    *,
    duration: str = "1s",
    description: str,
) -> TrainTaskConfig:
    return TrainTaskConfig(
        task_name=name,
        data_type=data_type,
        duration=duration,
        t5_cache_dir=_task_t5_cache(data_type, duration),
        cache_root=_main("training_data", "cache", "vae"),
        output_dir=os.path.join(TRAINING_DATA_ROOT, "log"),
        description=description,
    )


TRAIN_TASKS: dict[str, TrainTaskConfig] = {
    "h2r_1s": _preset(
        "h2r_1s",
        "h2r",
        description="Human-to-robot 1s cache with runtime task split.",
    ),
    "r2h_1s": _preset(
        "r2h_1s",
        "r2h",
        description="Robot-to-human 1s cache with runtime task split.",
    ),
    "blur_r2r_1s": _preset(
        "blur_r2r_1s",
        "blur_r2r",
        description="Blurred-robot to clear-robot 1s cache with runtime task split.",
    ),
    "identity_r2r_1s": _preset(
        "identity_r2r_1s",
        "identity_r2r",
        description="Robot-to-robot identity 1s cache with runtime task split.",
    ),
    # Compatibility aliases for existing commands; they now map to semantic data types.
    "pair_1s": _preset(
        "pair_1s",
        "h2r",
        description="Compatibility alias for human-to-robot 1s runtime data.",
    ),
    "pair_1s_r2h": _preset(
        "pair_1s_r2h",
        "blur_r2r",
        description="Compatibility alias for blurred-robot to clear-robot 1s runtime data.",
    ),
    "pair_1s_train3": _preset(
        "pair_1s_train3",
        "h2r",
        description="Compatibility alias for selected train3 human-to-robot runtime data.",
    ),
    "pair_1s_16": _preset(
        "pair_1s_16",
        "h2r",
        description="Compatibility alias for 16-frame human-to-robot runtime data.",
    ),
    "attn_ffn_selected": _preset(
        "attn_ffn_selected",
        "h2r",
        description="Compatibility alias for selected attn+ffn runtime data.",
    ),
    "robot_1s": _preset(
        "robot_1s",
        "identity_r2r",
        description="Compatibility alias for robot-to-robot identity runtime data.",
    ),
    "smoke_test": TrainTaskConfig(
        task_name="smoke_test",
        data_type="h2r",
        duration="1s",
        train_tasks=short_task_name(DEFAULT_TRAIN_TASKS[0]),
        ood_tasks="",
        cache_root=_base("tmp", "smoke_test", "gpu", "cache_generated"),
        t5_cache_dir=_base("tmp", "smoke_test", "gpu", "t5"),
        output_dir=_base("tmp", "smoke_test", "gpu", "e2e_train_run"),
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


def _apply_default(args, attr: str, value: str) -> None:
    if not getattr(args, attr):
        setattr(args, attr, value)


def apply_train_task_config(args) -> None:
    """Attach runtime data defaults from the selected preset to args."""
    cfg = resolve_train_task(args.task_name)
    _apply_default(args, "data_type", cfg.data_type)
    _apply_default(args, "duration", cfg.duration)
    _apply_default(args, "train_tasks", cfg.train_tasks)
    _apply_default(args, "ood_tasks", cfg.ood_tasks)
    _apply_default(args, "cache_root", cfg.cache_root)
    _apply_default(args, "t5_cache_dir", cfg.t5_cache_dir)
    _apply_default(args, "output_dir", cfg.output_dir)
    args.task_description = cfg.description
