"""Convert Wan2.2 TI2V-5B DiT FP32 shards → single bf16 safetensors.

The upstream shards store all 5B params in FP32 (20 GB on disk).
Training only uses bf16, so this one-time conversion halves disk read
time from ~20 GB to ~10 GB.

Usage:
    python -m src.tools.convert_dit_bf16 [--dit-dir DIR] [--output PATH]
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def convert(dit_dir: str, output: str) -> None:
    shards = sorted(
        os.path.join(dit_dir, f)
        for f in os.listdir(dit_dir)
        if f.startswith("diffusion_pytorch_model-") and f.endswith(".safetensors")
    )
    if not shards:
        print(f"No FP32 shards found in {dit_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(shards)} shard(s) in {dit_dir}")
    sd: dict[str, torch.Tensor] = {}
    t0 = time.time()
    for i, path in enumerate(shards):
        with safe_open(path, framework="pt", device="cpu") as f:
            for k in f.keys():
                sd[k] = f.get_tensor(k).to(torch.bfloat16)
        print(f"  [{i+1}/{len(shards)}] {os.path.basename(path)} "
              f"→ {len(sd)} tensors so far ({time.time()-t0:.1f}s)")

    total_bytes = sum(t.numel() * t.element_size() for t in sd.values())
    print(f"Saving {len(sd)} tensors ({total_bytes / 1e9:.2f} GB bf16) → {output}")
    save_file(sd, output)
    print(f"Done in {time.time()-t0:.1f}s  "
          f"(file size: {os.path.getsize(output) / 1e9:.2f} GB)")


def main():
    from src.pipeline.train_mitty import DEFAULT_DIT_DIR

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dit-dir", default=DEFAULT_DIT_DIR,
                        help="Directory containing FP32 shards")
    parser.add_argument("--output", default="",
                        help="Output path (default: <dit-dir>/diffusion_pytorch_model-bf16.safetensors)")
    args = parser.parse_args()

    if not args.output:
        args.output = os.path.join(args.dit_dir,
                                   "diffusion_pytorch_model-bf16.safetensors")

    if os.path.exists(args.output):
        print(f"Output already exists: {args.output}")
        print("Delete it first if you want to regenerate.")
        sys.exit(1)

    convert(args.dit_dir, args.output)


if __name__ == "__main__":
    main()
