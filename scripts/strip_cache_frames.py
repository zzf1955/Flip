"""Strip PIL frames from .pth cache files to reduce IO during training.

Removes human_frames and robot_frames keys (used only for eval video display).
Reduces file size from ~55MB to ~9MB per file.

Usage:
    python scripts/strip_cache_frames.py output/mitty_cache_robot/train/
"""

import sys
import time
from multiprocessing import Pool
from pathlib import Path

import torch


def strip_one(path_str: str) -> tuple[str, int, int]:
    path = Path(path_str)
    d = torch.load(path, map_location="cpu", weights_only=False)
    removed = []
    for k in ("human_frames", "robot_frames"):
        if k in d:
            del d[k]
            removed.append(k)
    if removed:
        torch.save(d, str(path))
    old_size = path.stat().st_size
    return path.name, len(removed), old_size


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <cache_dir>")
        sys.exit(1)

    cache_dir = Path(sys.argv[1])
    files = sorted(cache_dir.glob("*.pth"))
    if not files:
        print(f"No .pth files in {cache_dir}")
        sys.exit(1)

    print(f"Stripping PIL frames from {len(files)} files in {cache_dir}")
    t0 = time.time()

    with Pool(4) as pool:
        for i, (name, n_removed, _) in enumerate(
            pool.imap_unordered(strip_one, [str(f) for f in files])
        ):
            if (i + 1) % 200 == 0 or i + 1 == len(files):
                print(f"  [{i+1}/{len(files)}] {name} removed={n_removed}")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s. Stripped {len(files)} files.")


if __name__ == "__main__":
    main()
