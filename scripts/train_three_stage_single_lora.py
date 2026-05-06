#!/usr/bin/env python3
"""Run FLIP training stages with explicit merged and trainable LoRA choices.

Each stage launches ``src.pipeline.train`` through ``scripts/flip_run.sh``.
``--merge-lora`` checkpoints are frozen into the base model.  ``--train-lora``
selects the one adapter that remains trainable; omit it to create a fresh LoRA,
or use the previous stage checkpoint to keep training one adapter across tasks.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNNER = PROJECT_ROOT / "scripts" / "flip_run.sh"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "training_data" / "log"
DEFAULT_STAGES = (
    "identity_r2r_1s:1000",
    "blur_r2r_1s:1000",
    "h2r_1s:1000",
)


@dataclass(frozen=True)
class StageSpec:
    index: int
    task_name: str
    max_steps: int
    merge_loras: tuple[str, ...] = ()
    train_lora: str = "previous"
    lora_rank: int | None = None
    lora_target_modules: str = ""
    lora_attn_types: str = ""
    lora_attn_projections: str = ""


def _resolve_path(value: str, *, must_exist: bool = False) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if must_exist and not path.exists():
        raise FileNotFoundError(f"Path not found: {path}")
    return path


def _split_stage_paths(value: str) -> tuple[str, ...]:
    out = []
    for chunk in value.replace(",", ";").split(";"):
        item = chunk.strip()
        if item:
            out.append(item)
    return tuple(out)


def _parse_stage_kv(value: str) -> dict[str, str]:
    items = {}
    for chunk in value.split(";"):
        if not chunk.strip():
            continue
        if "=" not in chunk:
            raise ValueError(
                f"Invalid --stage item '{chunk}', expected key=value"
            )
        key, raw = chunk.split("=", 1)
        key = key.strip().replace("-", "_")
        raw = raw.strip()
        if not key or not raw:
            raise ValueError(f"Invalid --stage item '{chunk}'")
        if key in items:
            raise ValueError(f"Duplicate --stage key '{key}' in {value}")
        items[key] = raw
    return items


def _parse_stage(index: int, value: str) -> StageSpec:
    if ";" in value or "=" in value:
        items = _parse_stage_kv(value)
        task_name = items.get("task") or items.get("task_name")
        steps_text = items.get("steps") or items.get("max_steps")
        if not task_name or not steps_text:
            raise ValueError(
                f"Invalid --stage '{value}', expected task=...;steps=..."
            )
        train_lora = items.get("train") or items.get("train_lora") or "previous"
        max_steps = int(steps_text)
        if max_steps <= 0:
            raise ValueError(f"Stage max_steps must be positive: {value}")
        lora_rank = int(items["rank"]) if "rank" in items else None
        return StageSpec(
            index=index,
            task_name=task_name,
            max_steps=max_steps,
            merge_loras=_split_stage_paths(items.get("merge", "")),
            train_lora=train_lora,
            lora_rank=lora_rank,
            lora_target_modules=items.get("targets", "")
            or items.get("target_modules", ""),
            lora_attn_types=items.get("attn_types", ""),
            lora_attn_projections=items.get("attn_projections", ""),
        )

    parts = [part.strip() for part in value.split(":")]
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError(
            f"Invalid --stage '{value}', expected task_name:max_steps"
        )
    max_steps = int(parts[1])
    if max_steps <= 0:
        raise ValueError(f"Stage max_steps must be positive: {value}")
    return StageSpec(index=index, task_name=parts[0], max_steps=max_steps)


def _list_run_dirs(output_dir: Path) -> set[Path]:
    if not output_dir.exists():
        return set()
    return {item for item in output_dir.iterdir() if item.is_dir()}


def _checkpoint_step(path: Path) -> int:
    stem = path.stem
    if not stem.startswith("step-"):
        raise ValueError(f"Unsupported checkpoint name: {path}")
    return int(stem.removeprefix("step-"))


def _latest_checkpoint(run_dir: Path) -> Path:
    ckpts = sorted(
        (run_dir / "ckpt").glob("step-*.safetensors"),
        key=_checkpoint_step,
    )
    if not ckpts:
        raise FileNotFoundError(f"No stage checkpoint found under {run_dir / 'ckpt'}")
    return ckpts[-1]


def _new_stage_run_dir(before: set[Path], after: set[Path], prefix: str) -> Path:
    new_dirs = sorted(
        (item for item in after - before if item.name.startswith(prefix)),
        key=lambda item: item.stat().st_mtime,
    )
    if len(new_dirs) != 1:
        names = [item.name for item in new_dirs]
        raise RuntimeError(
            f"Expected one new run dir with prefix '{prefix}', found {names}"
        )
    return new_dirs[0]


def _append_optional_int(cmd: list[str], flag: str, value: int) -> None:
    if value != 0:
        cmd.extend([flag, str(value)])


def _resolve_train_lora_for_stage(
    stage: StageSpec,
    previous_lora: Path | None,
) -> Path | None:
    value = stage.train_lora.strip()
    if value in ("", "previous", "prev", "auto"):
        return previous_lora
    if value in ("fresh", "new", "none"):
        return None
    return _resolve_path(value, must_exist=True)


def _build_stage_command(
    args: argparse.Namespace,
    stage: StageSpec,
    runner: Path,
    output_dir: Path,
    train_lora: Path | None,
) -> tuple[str, list[str]]:
    stage_prefix = f"{args.run_prefix}_s{stage.index}_{stage.task_name}"
    cmd = [
        str(runner),
        "train",
        "--cuda",
        args.cuda,
        "--nproc",
        str(args.nproc),
        "--",
        "--task-name",
        stage.task_name,
        "--run-prefix",
        stage_prefix,
        "--output-dir",
        str(output_dir),
        "--max-steps",
        str(stage.max_steps),
        "--save-steps",
        str(args.save_steps),
        "--eval-steps",
        str(args.eval_steps),
        "--eval-video-steps",
        str(args.eval_video_steps),
        "--batch-size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
        "--lr-min",
        str(args.lr_min),
        "--warmup-steps",
        str(args.warmup_steps),
        "--weight-decay",
        str(args.weight_decay),
        "--data-seed",
        str(args.data_seed),
        "--wandb-project",
        args.wandb_project,
        "--wandb-tags",
        "single_lora",
        f"stage:{stage.index}",
        f"task:{stage.task_name}",
    ]
    for lora_path in args.merge_lora:
        cmd.extend(["--merge-lora", str(lora_path)])
    for lora_path in stage.merge_loras:
        cmd.extend(["--merge-lora", str(_resolve_path(lora_path, must_exist=True))])
    _append_optional_int(cmd, "--train-size", args.train_size)
    _append_optional_int(cmd, "--in-task-eval-size", args.in_task_eval_size)
    _append_optional_int(cmd, "--ood-eval-size", args.ood_eval_size)
    _append_optional_int(cmd, "--in-task-video-size", args.in_task_video_size)
    _append_optional_int(cmd, "--ood-video-size", args.ood_video_size)
    stage_rank = stage.lora_rank if stage.lora_rank is not None else args.train_lora_rank
    if stage_rank:
        cmd.extend(["--train-lora-rank", str(stage_rank)])
    stage_targets = stage.lora_target_modules or args.train_lora_target_modules
    if stage_targets:
        cmd.extend(["--train-lora-target-modules", stage_targets])
    else:
        cmd.extend(["--lora-attn-types", stage.lora_attn_types or args.lora_attn_types])
        cmd.extend([
            "--lora-attn-projections",
            stage.lora_attn_projections or args.lora_attn_projections,
        ])
    if train_lora is not None:
        cmd.extend(["--train-lora", str(train_lora)])
    if args.no_wandb_log_videos:
        cmd.append("--no-wandb-log-videos")
    if args.no_eval_video_frechet_metrics:
        cmd.append("--no-eval-video-frechet-metrics")
    cmd.extend(args.extra_train_arg)
    return stage_prefix, cmd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run staged training with explicit merged and trainable LoRA choices.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--stage", action="append", default=[],
                        help="stage spec task_name:max_steps, or "
                             "task=...;steps=...;merge=a,b;train=path|fresh|previous")
    parser.add_argument("--initial-lora", default="",
                        help="optional checkpoint to continue from before stage 1")
    parser.add_argument("--train-lora", default="",
                        help="alias for --initial-lora")
    parser.add_argument("--merge-lora", action="append", default=[],
                        help="LoRA checkpoint merged into every stage; repeatable")
    parser.add_argument("--runner", default=str(DEFAULT_RUNNER))
    parser.add_argument("--cuda", required=True)
    parser.add_argument("--nproc", type=int, default=1)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--run-prefix", default="single_lora3")
    parser.add_argument("--train-lora-rank", "--lora-rank",
                        dest="train_lora_rank", type=int, default=0)
    parser.add_argument("--train-lora-target-modules", "--lora-target-modules",
                        dest="train_lora_target_modules", default="")
    parser.add_argument("--lora-attn-types", default="self,cross")
    parser.add_argument("--lora-attn-projections", default="q,k,v,o")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--train-size", type=int, default=0)
    parser.add_argument("--in-task-eval-size", type=int, default=16)
    parser.add_argument("--ood-eval-size", type=int, default=16)
    parser.add_argument("--in-task-video-size", type=int, default=4)
    parser.add_argument("--ood-video-size", type=int, default=2)
    parser.add_argument("--data-seed", type=int, default=42)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--eval-video-steps", type=int, default=100)
    parser.add_argument("--lr", default="1e-4")
    parser.add_argument("--lr-min", default="1e-6")
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--weight-decay", default="0.01")
    parser.add_argument("--wandb-project", default="Flip")
    parser.add_argument("--no-wandb-log-videos", action="store_true")
    parser.add_argument("--no-eval-video-frechet-metrics", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("extra_train_arg", nargs=argparse.REMAINDER,
                        help="extra train.py args after --")
    args = parser.parse_args()
    if args.extra_train_arg and args.extra_train_arg[0] == "--":
        args.extra_train_arg = args.extra_train_arg[1:]
    if args.save_steps <= 0:
        parser.error("--save-steps must be positive so later stages can continue")
    if args.nproc <= 0:
        parser.error("--nproc must be positive")
    return args


def main() -> None:
    args = parse_args()
    stage_values = args.stage or list(DEFAULT_STAGES)
    stages = [_parse_stage(i + 1, value) for i, value in enumerate(stage_values)]
    runner = _resolve_path(args.runner, must_exist=True)
    output_dir = _resolve_path(args.output_dir)
    if args.initial_lora and args.train_lora and args.initial_lora != args.train_lora:
        raise ValueError("--initial-lora and --train-lora point to different files")
    initial_lora = args.train_lora or args.initial_lora
    previous_lora = (
        _resolve_path(initial_lora, must_exist=True)
        if initial_lora else None
    )
    args.merge_lora = [
        _resolve_path(path, must_exist=True)
        for raw in args.merge_lora
        for path in _split_stage_paths(raw)
    ]

    for stage in stages:
        train_lora = _resolve_train_lora_for_stage(stage, previous_lora)
        prefix, cmd = _build_stage_command(
            args, stage, runner, output_dir, train_lora,
        )
        print(
            f"\nStage {stage.index}/{len(stages)} "
            f"task={stage.task_name} steps={stage.max_steps}",
            flush=True,
        )
        if args.merge_lora or stage.merge_loras:
            merged = [str(path) for path in args.merge_lora] + list(stage.merge_loras)
            print("Merge LoRA:", ", ".join(merged), flush=True)
        else:
            print("Merge LoRA: <none>", flush=True)
        print("Train LoRA:", train_lora or "<fresh>", flush=True)
        print("Command:", shlex.join(cmd), flush=True)
        if args.dry_run:
            previous_lora = Path(f"<stage-{stage.index}-checkpoint>")
            continue

        before = _list_run_dirs(output_dir)
        proc = subprocess.run(cmd, cwd=PROJECT_ROOT, check=False)
        if proc.returncode != 0:
            print(
                "\ntrain_three_stage_single_lora.py: stage failed\n"
                f"  stage: {stage.index}\n"
                f"  task: {stage.task_name}\n"
                f"  exit_code: {proc.returncode}\n"
                f"  command: {shlex.join(cmd)}",
                flush=True,
            )
            raise SystemExit(proc.returncode) from None
        after = _list_run_dirs(output_dir)
        run_dir = _new_stage_run_dir(before, after, prefix)
        previous_lora = _latest_checkpoint(run_dir)
        print(f"Stage checkpoint: {previous_lora}", flush=True)


if __name__ == "__main__":
    main()
