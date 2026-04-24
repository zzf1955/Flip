#!/usr/bin/env python3
"""T032 minimal GPU smoke for Mitty cache + 1-step training.

Uses CUDA_VISIBLE_DEVICES=2 by default and writes all artifacts under
./tmp/t032/gpu_smoke.  The script copies one pair video and one eval cache from
MAIN_ROOT so it does not modify shared training_data.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAIN_ROOT = Path(os.environ.get("FLIP_MAIN_ROOT", "/disk_n/zzf/flip"))
PYTHON = Path(sys.executable)
SMOKE_ROOT = ROOT / "tmp" / "t032" / "gpu_smoke"
PAIR_SRC = MAIN_ROOT / "training_data" / "pair" / "1s" / "train"
EVAL_CACHE = MAIN_ROOT / "training_data" / "cache" / "1s_patch_sam2" / "train" / "pair_0001.pth"
T5_SRC = MAIN_ROOT / "training_data" / "cache" / "t5"


class SmokeFailure(RuntimeError):
    pass


def run(cmd: list[str], log_path: Path) -> None:
    env = os.environ.copy()
    env.setdefault("CUDA_VISIBLE_DEVICES", "2")
    env.setdefault("HF_HOME", "/disk_n/zzf/.cache/huggingface")
    env.setdefault("PIP_CACHE_DIR", "/disk_n/zzf/.pip_cache")
    env.setdefault("no_proxy", "localhost,127.0.0.1")
    libjpeg = "/home/leadtek/miniconda3/envs/flip/lib/libjpeg.so.8"
    env["LD_PRELOAD"] = libjpeg + ((":" + env["LD_PRELOAD"]) if env.get("LD_PRELOAD") else "")
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


def prepare_inputs() -> None:
    pair_dir = SMOKE_ROOT / "pair_src"
    cache_train = SMOKE_ROOT / "cache_generated" / "train"
    cache_eval = SMOKE_ROOT / "cache" / "eval"
    t5_dir = SMOKE_ROOT / "t5"
    for path in (pair_dir, cache_train, cache_eval, t5_dir):
        if path.exists():
            shutil.rmtree(path)
    (pair_dir / "video").mkdir(parents=True)
    (pair_dir / "control_video").mkdir(parents=True)
    cache_eval.mkdir(parents=True)
    t5_dir.mkdir(parents=True)

    shutil.copy2(PAIR_SRC / "video" / "pair_0000.mp4", pair_dir / "video" / "pair_0000.mp4")
    shutil.copy2(PAIR_SRC / "control_video" / "pair_0000.mp4", pair_dir / "control_video" / "pair_0000.mp4")
    with open(PAIR_SRC / "metadata.csv", newline="") as src, open(pair_dir / "metadata.csv", "w", newline="") as dst:
        reader = csv.reader(src)
        writer = csv.writer(dst)
        writer.writerow(next(reader))
        writer.writerow(next(reader))
    shutil.copy2(EVAL_CACHE, cache_eval / "pair_0001.pth")
    for item in T5_SRC.glob("*.pth"):
        shutil.copy2(item, t5_dir / item.name)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run T032 minimal GPU smoke on card 2")
    parser.add_argument("--skip-prepare", action="store_true", help="reuse existing tmp inputs")
    args = parser.parse_args()

    if not args.skip_prepare:
        prepare_inputs()

    run(
        [
            str(PYTHON), "-m", "src.pipeline.mitty_cache",
            "--pair-dir", str(SMOKE_ROOT / "pair_src"),
            "--output", str(SMOKE_ROOT / "cache_generated" / "train"),
            "--t5-cache-dir", str(SMOKE_ROOT / "t5"),
            "--device", "cuda:0",
            "--batch-size", "1",
            "--prefetch-workers", "0",
            "--save-workers", "1",
        ],
        SMOKE_ROOT / "mitty_cache_1sample.log",
    )
    run(
        [
            str(PYTHON), "-m", "src.pipeline.train",
            "--task-name", "t032_e2e_smoke",
            "--loss", "uniform",
            "--cache-train", str(SMOKE_ROOT / "cache_generated" / "train"),
            "--cache-eval", str(SMOKE_ROOT / "cache" / "eval"),
            "--t5-cache-dir", str(SMOKE_ROOT / "t5"),
            "--output-dir", str(SMOKE_ROOT / "e2e_train_run"),
            "--device", "cuda:0",
            "--max-steps", "1",
            "--save-steps", "1",
            "--eval-steps", "1",
            "--eval-t-samples", "1",
            "--eval-video-steps", "0",
            "--lora-rank", "4",
            "--warmup-steps", "0",
            "--wandb-project", "",
        ],
        SMOKE_ROOT / "train_e2e_1step.log",
    )
    print(f"T032 GPU smoke passed; logs: {SMOKE_ROOT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
