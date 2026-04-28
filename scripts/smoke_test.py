#!/usr/bin/env python3
"""Run the complete FLIP smoke test sequence.

The sequence always runs the lightweight smoke first, records GPU status through
the GPU smoke script, then runs the real GPU smoke.  The final report states
whether the GPU training step used single-card or multi-card execution.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(sys.executable)
SMOKE_ROOT = ROOT / "tmp" / "smoke_test"


class SmokeFailure(RuntimeError):
    pass


def run(cmd: list[str], log_path: Path) -> None:
    proc = subprocess.run(
        cmd,
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
        raise SmokeFailure(f"command failed ({proc.returncode}): {' '.join(cmd)}; see {log_path}")


def gpu_scope(nproc: int) -> str:
    return "dual-card" if nproc == 2 else ("single-card" if nproc == 1 else f"{nproc}-card")


def visible_device_count(cuda_devices: str) -> int:
    return len([part for part in cuda_devices.split(",") if part.strip()])


def main() -> int:
    parser = argparse.ArgumentParser(description="Run lightweight + GPU smoke tests")
    parser.add_argument("--cuda", default="2", help="CUDA_VISIBLE_DEVICES for GPU smoke, e.g. 2 or 2,3")
    parser.add_argument("--nproc", type=int, default=0, help="GPU train worker count; default equals --cuda device count")
    parser.add_argument("--skip-help", action="store_true", help="pass through to lightweight smoke")
    parser.add_argument("--skip-prepare", action="store_true", help="pass through to GPU smoke")
    args = parser.parse_args()

    nproc = args.nproc or visible_device_count(args.cuda)
    if nproc < 1:
        raise ValueError("--nproc must be >= 1")

    light_cmd = [str(PYTHON), "scripts/smoke_test_light.py"]
    if args.skip_help:
        light_cmd.append("--skip-help")
    run(light_cmd, SMOKE_ROOT / "smoke_test_light.log")

    gpu_cmd = [str(PYTHON), "scripts/smoke_test_gpu.py", "--cuda", args.cuda, "--nproc", str(nproc)]
    if args.skip_prepare:
        gpu_cmd.append("--skip-prepare")
    run(gpu_cmd, SMOKE_ROOT / "smoke_test_gpu.log")

    summary = {
        "status": "passed",
        "lightweight": "passed",
        "gpu": "passed",
        "cuda_visible_devices": args.cuda,
        "nproc": nproc,
        "scope": gpu_scope(nproc),
        "logs": str(SMOKE_ROOT.relative_to(ROOT)),
    }
    (SMOKE_ROOT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(f"Smoke test passed: lightweight + GPU ({summary['scope']}); logs: {SMOKE_ROOT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
