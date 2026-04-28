#!/usr/bin/env python3
"""Minimal GPU smoke test for Mitty cache + 1-step training.

Writes all artifacts under ./tmp/smoke_test/gpu.  The script copies one pair
video, one matching eval VAE cache, and matching T5 cache from MAIN_ROOT so it
does not modify shared training_data.  It records NVIDIA-SMI before GPU work
and reports whether the train step used a single-card or multi-card launch.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAIN_ROOT = Path(os.environ.get("FLIP_MAIN_ROOT", "/disk_n/zzf/flip"))
PYTHON = Path(sys.executable)
SMOKE_ROOT = ROOT / "tmp" / "smoke_test" / "gpu"
SMOKE_TASK = "Inspire_Put_Clothes_Into_Basket"
PAIR_SRC = MAIN_ROOT / "training_data" / "pair" / "h2r" / "1s" / SMOKE_TASK
T5_SRC = MAIN_ROOT / "training_data" / "cache" / "t5" / "h2r" / "1s"


class SmokeFailure(RuntimeError):
    pass


def visible_device_count(cuda_devices: str) -> int:
    return len([part for part in cuda_devices.split(",") if part.strip()])


def smoke_scope(nproc: int) -> str:
    return "dual-card" if nproc == 2 else ("single-card" if nproc == 1 else f"{nproc}-card")


def build_env(cuda_devices: str) -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = cuda_devices
    env.setdefault("HF_HOME", "/disk_n/zzf/.cache/huggingface")
    env.setdefault("PIP_CACHE_DIR", "/disk_n/zzf/.pip_cache")
    env.setdefault("no_proxy", "localhost,127.0.0.1")
    libjpeg = "/home/leadtek/miniconda3/envs/flip/lib/libjpeg.so.8"
    env["LD_PRELOAD"] = libjpeg + ((":" + env["LD_PRELOAD"]) if env.get("LD_PRELOAD") else "")
    return env


def run(cmd: list[str], log_path: Path, *, env: dict[str, str]) -> None:
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(proc.stdout)
    print(proc.stdout, end="")
    if proc.returncode != 0:
        raise SmokeFailure(f"command failed ({proc.returncode}): {' '.join(cmd)}; see {log_path}")


def record_gpu_status(log_path: Path) -> None:
    proc = subprocess.run(
        ["nvidia-smi"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(proc.stdout)
    print(proc.stdout, end="")
    if proc.returncode != 0:
        raise SmokeFailure(f"nvidia-smi failed ({proc.returncode}); see {log_path}")


def prepare_inputs() -> None:
    pair_dir = SMOKE_ROOT / "pair_src"
    cache_task = SMOKE_ROOT / "cache_generated" / "h2r" / "1s" / SMOKE_TASK
    t5_dir = SMOKE_ROOT / "t5"
    for path in (pair_dir, SMOKE_ROOT / "cache_generated", t5_dir):
        if path.exists():
            shutil.rmtree(path)
    (pair_dir / "video").mkdir(parents=True)
    (pair_dir / "control_video").mkdir(parents=True)
    cache_task.mkdir(parents=True)
    t5_dir.mkdir(parents=True)

    shutil.copy2(PAIR_SRC / "video" / "pair_0000.mp4", pair_dir / "video" / "pair_0000.mp4")
    shutil.copy2(PAIR_SRC / "control_video" / "pair_0000.mp4", pair_dir / "control_video" / "pair_0000.mp4")
    with open(PAIR_SRC / "metadata.csv", newline="") as src, open(pair_dir / "metadata.csv", "w", newline="") as dst:
        reader = csv.reader(src)
        writer = csv.writer(dst)
        writer.writerow(next(reader))
        writer.writerow(next(reader))
    manifest_src = PAIR_SRC / "manifest.jsonl"
    if manifest_src.is_file():
        with open(manifest_src) as src, open(pair_dir / "manifest.jsonl", "w") as dst:
            first = json.loads(next(line for line in src if line.strip()))
            first["video"] = "video/pair_0000.mp4"
            first["control_video"] = "control_video/pair_0000.mp4"
            dst.write(json.dumps(first, sort_keys=True) + "\n")
    for item in T5_SRC.glob("*.pth"):
        shutil.copy2(item, t5_dir / item.name)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run minimal GPU smoke test")
    parser.add_argument("--cuda", default="2", help="CUDA_VISIBLE_DEVICES for GPU smoke, e.g. 2 or 2,3")
    parser.add_argument("--nproc", type=int, default=0, help="train worker count; default equals --cuda device count")
    parser.add_argument("--skip-prepare", action="store_true", help="reuse existing tmp inputs")
    args = parser.parse_args()

    nproc = args.nproc or visible_device_count(args.cuda)
    if nproc < 1:
        raise ValueError("--nproc must be >= 1")
    if nproc > visible_device_count(args.cuda):
        raise ValueError("--nproc cannot exceed the number of --cuda devices")

    if not args.skip_prepare:
        prepare_inputs()

    record_gpu_status(SMOKE_ROOT / "nvidia_smi_before.log")
    env = build_env(args.cuda)

    run(
        [
            str(PYTHON), "-m", "src.pipeline.mitty_cache",
            "--pair-dir", str(SMOKE_ROOT / "pair_src"),
            "--output", str(SMOKE_ROOT / "cache_generated" / "h2r" / "1s" / SMOKE_TASK),
            "--t5-cache-dir", str(SMOKE_ROOT / "t5"),
            "--device", "cuda:0",
            "--batch-size", "1",
            "--prefetch-workers", "0",
            "--save-workers", "1",
        ],
        SMOKE_ROOT / "mitty_cache_1sample.log",
        env=env,
    )

    train_args = [
        "-m", "src.pipeline.train",
        "--task-name", "smoke_test",
        "--device", "cuda:0",
        "--max-steps", "1",
        "--save-steps", "1",
        "--eval-steps", "1",
        "--eval-t-samples", "1",
        "--eval-video-steps", "0",
        "--train-size", "1",
        "--in-task-eval-size", "1",
        "--ood-tasks", "",
        "--ood-eval-size", "0",
        "--lora-rank", "4",
        "--warmup-steps", "0",
        "--wandb-project", "",
    ]
    train_cmd = [str(PYTHON), *train_args]
    if nproc > 1:
        train_cmd = ["torchrun", f"--nproc_per_node={nproc}", *train_args]
    run(train_cmd, SMOKE_ROOT / "train_e2e_1step.log", env=env)

    summary = {
        "status": "passed",
        "cuda_visible_devices": args.cuda,
        "nproc": nproc,
        "scope": smoke_scope(nproc),
        "logs": str(SMOKE_ROOT.relative_to(ROOT)),
    }
    (SMOKE_ROOT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(f"GPU smoke test passed ({summary['scope']}); logs: {SMOKE_ROOT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
