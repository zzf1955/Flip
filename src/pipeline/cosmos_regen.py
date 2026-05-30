"""
Launch Cosmos Transfer 2.5 inference for depth-guided human regeneration.

Reads spec.json from cosmos_prepare output and runs Cosmos via torchrun.

Usage:
  python -m src.pipeline.cosmos_regen \
      --prepare-dir output/human/cosmos_prepare/Pickup_Pillow_MainCamOnly_ep0_s5_d1
  python -m src.pipeline.cosmos_regen \
      --prepare-dir output/human/cosmos_prepare/... --gpus 0 1 2 3
"""

import sys
import os
import json
import argparse
import subprocess
import shutil

sys.stdout.reconfigure(line_buffering=True)

from src.core.config import COSMOS25_ROOT, COSMOS_REGEN_DIR


def main():
    parser = argparse.ArgumentParser(
        description="Run Cosmos Transfer 2.5 depth-guided inference")
    parser.add_argument("--prepare-dir", type=str, required=True, dest="prepare_dir",
                        help="cosmos_prepare output directory containing spec.json")
    parser.add_argument("--gpus", type=int, nargs="+", default=[1, 2, 3],
                        help="GPU IDs to use (default: 1 2 3)")
    parser.add_argument("--resolution", type=str, default="480",
                        choices=["480", "720"],
                        help="Generation resolution (default: 480)")
    parser.add_argument("--num-steps", type=int, default=35, dest="num_steps",
                        help="Diffusion steps (default: 35)")
    parser.add_argument("--out-dir", type=str, default=None, dest="out_dir",
                        help="Output directory (default: auto from tag)")
    parser.add_argument("--dry-run", action="store_true", dest="dry_run",
                        help="Print command without executing")
    args = parser.parse_args()

    spec_path = os.path.join(args.prepare_dir, "spec.json")
    if not os.path.isfile(spec_path):
        print(f"ERROR: spec.json not found at {spec_path}")
        sys.exit(1)

    with open(spec_path) as f:
        spec = json.load(f)

    tag = spec.get("name", "cosmos_output").replace("_cosmos", "")
    out_dir = args.out_dir or os.path.join(COSMOS_REGEN_DIR, tag)
    os.makedirs(out_dir, exist_ok=True)

    # Copy spec to output dir for reference
    out_spec = os.path.join(out_dir, "spec.json")
    shutil.copy2(spec_path, out_spec)

    # Build torchrun command
    n_gpus = len(args.gpus)
    cuda_devices = ",".join(str(g) for g in args.gpus)
    inference_script = os.path.join(COSMOS25_ROOT, "examples", "inference.py")

    if not os.path.isfile(inference_script):
        print(f"ERROR: Cosmos inference script not found: {inference_script}")
        sys.exit(1)

    cmd = [
        "torchrun",
        f"--nproc_per_node={n_gpus}",
        "--master_port=29501",
        inference_script,
        "-i", os.path.abspath(spec_path),
        "--output-dir", os.path.abspath(out_dir),
        f"--num-steps={args.num_steps}",
        f"--resolution={args.resolution}",
        "--disable-guardrail",
        "control:depth",
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = cuda_devices

    print(f"{'='*60}")
    print(f"Cosmos Transfer 2.5 — Depth-Guided Generation")
    print(f"  GPUs:       {cuda_devices} ({n_gpus} devices)")
    print(f"  Resolution: {args.resolution}p")
    print(f"  Steps:      {args.num_steps}")
    print(f"  Spec:       {spec_path}")
    print(f"  Output:     {out_dir}")
    print(f"  Command:")
    print(f"    CUDA_VISIBLE_DEVICES={cuda_devices} \\")
    print(f"    {' '.join(cmd)}")
    print(f"{'='*60}")

    if args.dry_run:
        print("\n[DRY RUN] Command printed above. Not executing.")
        return

    print("\nStarting Cosmos inference...\n")
    result = subprocess.run(
        cmd,
        env=env,
        cwd=COSMOS25_ROOT,
    )

    if result.returncode != 0:
        print(f"\nERROR: Cosmos inference failed with exit code {result.returncode}")
        sys.exit(result.returncode)

    print(f"\nCosmos inference complete!")
    print(f"Output: {out_dir}")

    # List output files
    for f in sorted(os.listdir(out_dir)):
        fpath = os.path.join(out_dir, f)
        size = os.path.getsize(fpath) if os.path.isfile(fpath) else 0
        print(f"  {f:<40s} {size/1024:.0f} KB")


if __name__ == "__main__":
    main()
