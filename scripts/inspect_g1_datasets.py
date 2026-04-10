#!/usr/bin/env python3
"""Inspect all G1 WBT datasets: print summary table and extract first frame per camera."""

import json
import os
import subprocess
import sys
from glob import glob
from pathlib import Path


DATA_ROOT = Path("data/unitree_G1_WBT")
OUTPUT_DIR = DATA_ROOT / "_inspect"


def find_meta_dir(dataset_dir: Path) -> Path | None:
    """Find the meta/ directory (handles nested MainCamOnly structure)."""
    info_files = list(dataset_dir.rglob("meta/info.json"))
    return info_files[0].parent if info_files else None


def extract_first_frame(video_path: Path, output_path: Path) -> bool:
    """Extract the first frame from a video using ffmpeg."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(video_path),
            "-vframes", "1", "-q:v", "2",
            str(output_path),
        ],
        capture_output=True,
    )
    return result.returncode == 0 and output_path.exists()


def short_cam_name(cam_key: str) -> str:
    """observation.images.head_stereo_left -> head_stereo_left"""
    return cam_key.replace("observation.images.", "")


def short_dataset_name(name: str) -> str:
    """Shorten dataset directory name for output."""
    return name.replace("G1_WBT_", "")


def inspect_dataset(dataset_dir: Path) -> dict | None:
    meta_dir = find_meta_dir(dataset_dir)
    if meta_dir is None:
        print(f"  SKIP {dataset_dir.name}: no meta/info.json found")
        return None

    with open(meta_dir / "info.json") as f:
        info = json.load(f)

    # Identify camera keys
    cam_keys = [k for k in info["features"] if "images" in k]
    # Identify state/action keys
    state_keys = {k: info["features"][k]["shape"] for k in info["features"] if k.startswith("observation.state.")}
    action_keys = {k: info["features"][k]["shape"] for k in info["features"] if k.startswith("action.")}

    data_root = meta_dir.parent  # the directory containing meta/, data/, videos/

    # Count data parquet files
    data_parquets = list(data_root.rglob("data/chunk-*/file-*.parquet"))
    # Count video files
    video_files = list(data_root.rglob("videos/*/chunk-*/file-*.mp4"))

    result = {
        "name": dataset_dir.name,
        "short_name": short_dataset_name(dataset_dir.name),
        "total_episodes": info["total_episodes"],
        "total_frames": info["total_frames"],
        "fps": info["fps"],
        "cameras": cam_keys,
        "state_keys": state_keys,
        "action_keys": action_keys,
        "n_data_parquets": len(data_parquets),
        "n_video_files": len(video_files),
        "data_root": data_root,
    }

    # Extract first frame per camera
    out_dir = OUTPUT_DIR / result["short_name"]
    for cam_key in cam_keys:
        cam_short = short_cam_name(cam_key)
        # Find first video file for this camera
        video_dir = data_root / "videos" / cam_key / "chunk-000"
        first_video = video_dir / "file-000.mp4"
        if not first_video.exists():
            print(f"  WARNING: {first_video} not found")
            continue

        out_path = out_dir / f"{cam_short}.png"
        ok = extract_first_frame(first_video, out_path)
        status = "OK" if ok else "FAIL"
        print(f"  [{status}] {result['short_name']}/{cam_short}.png")

    return result


def main():
    datasets = sorted(DATA_ROOT.iterdir())
    datasets = [d for d in datasets if d.is_dir() and not d.name.startswith("_")]

    print(f"Found {len(datasets)} datasets under {DATA_ROOT}\n")

    results = []
    for d in datasets:
        print(f"--- {d.name} ---")
        r = inspect_dataset(d)
        if r:
            results.append(r)
        print()

    # Print summary table
    print("=" * 120)
    print(f"{'Dataset':<55} {'Eps':>5} {'Frames':>8} {'Cams':>4} {'Videos':>6} {'Parq':>4} {'State Dims':>12} {'Action Dims':>12}")
    print("-" * 120)
    for r in results:
        state_dim = sum(s[0] for s in r["state_keys"].values())
        action_dim = sum(s[0] for s in r["action_keys"].values())
        cam_names = ",".join(short_cam_name(c) for c in r["cameras"])
        print(
            f"{r['short_name']:<55} {r['total_episodes']:>5} {r['total_frames']:>8} {len(r['cameras']):>4} "
            f"{r['n_video_files']:>6} {r['n_data_parquets']:>4} {state_dim:>12} {action_dim:>12}"
        )
    print("=" * 120)

    # Print camera details
    print("\nCamera keys per dataset:")
    for r in results:
        cam_names = ", ".join(short_cam_name(c) for c in r["cameras"])
        print(f"  {r['short_name']}: [{cam_names}]")

    # Count output frames
    pngs = list(OUTPUT_DIR.rglob("*.png"))
    print(f"\nExtracted {len(pngs)} sample frames to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
