#!/usr/bin/env python3
"""Lightweight smoke tests for the maintained FLIP workflow.

This script intentionally avoids real model/data execution. It verifies that
maintained pipeline entry points import, expose CLI help, and write smoke logs
under ./tmp/smoke_test/light.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(sys.executable)
SMOKE_DIR = ROOT / "tmp" / "smoke_test" / "light"

MODULE_HELPS = [
    # Data segmentation and robot removal
    "src.pipeline.segment_episodes",
    "src.pipeline.sam2_precompute",
    "src.pipeline.batch_sam2_precompute",
    "src.pipeline.sam2_segment",
    "src.pipeline.sam2_inpaint",
    "src.pipeline.batch_inpaint",
    "src.pipeline.video_inpaint",
    "src.pipeline.segment_pipeline",
    # Retarget / human rendering / pair construction
    "src.pipeline.retarget_video",
    "src.pipeline.human_overlay",
    "src.pipeline.seedance_clip",
    "src.pipeline.make_pair",
    "src.pipeline.make_robot_pair",
    "src.pipeline.robot_patch",
    "src.pipeline.hand_patch",
    "src.pipeline.hand_patch_4s",
    # Local regeneration utilities kept outside the training mainline
    "src.pipeline.cosmos_prepare",
    "src.pipeline.cosmos_regen",
    "src.pipeline.wan_regen",
    # Training mainline
    "src.pipeline.mitty_cache",
    "src.pipeline.train",
    "src.pipeline.train_mitty",
    # Maintained tools used by training/eval
    "src.tools.train_log_to_csv",
    "src.tools.eval_metrics",
]

DELETED_MODULES = [
    "src.pipeline.train_lora",
    "src.pipeline.train_rf",
    "src.pipeline.rf_model_fn",
    "src.pipeline.backbones.rectflow",
]


class SmokeFailure(RuntimeError):
    pass


def run(cmd: list[str], *, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    merged_env = os.environ.copy()
    merged_env.setdefault("CUDA_VISIBLE_DEVICES", "2")
    if env:
        merged_env.update(env)
    return subprocess.run(
        cmd,
        cwd=ROOT,
        env=merged_env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def write_log(name: str, text: str) -> Path:
    SMOKE_DIR.mkdir(parents=True, exist_ok=True)
    path = SMOKE_DIR / name
    path.write_text(text)
    return path


def check_compileall(results: list[dict]) -> None:
    cmd = [
        str(PYTHON), "-m", "compileall", "-q", "src",
        "scripts/smoke_wan22_ti2v5b.py",
        "scripts/smoke_test.py",
        "scripts/smoke_test_light.py",
        "scripts/smoke_test_gpu.py",
    ]
    proc = run(cmd)
    write_log("compileall.log", proc.stdout)
    results.append({"name": "compileall", "returncode": proc.returncode})
    if proc.returncode != 0:
        raise SmokeFailure("compileall failed; see tmp/smoke_test/light/compileall.log")


def check_help(module: str, results: list[dict]) -> None:
    proc = run([str(PYTHON), "-m", module, "--help"])
    safe_name = module.replace(".", "_") + "__help.log"
    write_log(safe_name, proc.stdout)
    ok = proc.returncode == 0 and "usage:" in proc.stdout.lower()
    results.append({"name": module, "kind": "help", "returncode": proc.returncode, "ok": ok})
    if not ok:
        raise SmokeFailure(f"--help failed for {module}; see tmp/smoke_test/light/{safe_name}")


def check_deleted_module(module: str, results: list[dict]) -> None:
    proc = run([str(PYTHON), "-c", f"import {module}"])
    safe_name = module.replace(".", "_") + "__deleted_import.log"
    write_log(safe_name, proc.stdout)
    ok = proc.returncode != 0 and "ModuleNotFoundError" in proc.stdout
    results.append({"name": module, "kind": "deleted_import", "returncode": proc.returncode, "ok": ok})
    if not ok:
        raise SmokeFailure(f"deleted module still importable or failed unexpectedly: {module}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run lightweight smoke tests")
    parser.add_argument("--skip-help", action="store_true", help="only run compile/deletion checks")
    args = parser.parse_args()

    results: list[dict] = []
    try:
        check_compileall(results)
        if not args.skip_help:
            for module in MODULE_HELPS:
                check_help(module, results)
        for module in DELETED_MODULES:
            check_deleted_module(module, results)
    finally:
        write_log("summary.json", json.dumps(results, indent=2, ensure_ascii=False) + "\n")

    print(f"Lightweight smoke test passed; logs: {SMOKE_DIR.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
