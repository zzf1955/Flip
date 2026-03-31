"""Extract keyframes from LEVERB dataset videos.

Scans data/leverb/videos/chunk-{000..003}/observation.images.tpv_cam/
and extracts first, middle, and last frames from each video.

Output: data/leverb_frames/ with PNG files + manifest.json
"""
import argparse
import json
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parent.parent
LEVERB_VIDEO_ROOT = PROJECT_ROOT / "data" / "leverb" / "videos"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "leverb_frames"


def extract_frames(video_path: Path, output_dir: Path, episode_id: str, chunk: str):
    """Extract first, middle, and last frames from a video."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  WARNING: cannot open {video_path}")
        return []

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        print(f"  WARNING: 0 frames in {video_path.name}")
        return []

    frame_indices = {
        "first": 0,
        "mid": total // 2,
        "last": max(0, total - 1),
    }

    results = []
    for label, idx in frame_indices.items():
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            print(f"  WARNING: cannot read frame {idx} from {video_path.name}")
            continue
        out_name = f"{episode_id}_f{label}.png"
        out_path = output_dir / out_name
        cv2.imwrite(str(out_path), frame)
        results.append({
            "file": out_name,
            "episode": episode_id,
            "chunk": chunk,
            "frame_index": idx,
            "frame_label": label,
            "total_frames": total,
            "source": str(video_path.relative_to(PROJECT_ROOT)),
        })
        print(f"  Saved: {out_name} (frame {idx}/{total})")

    cap.release()
    return results


def main():
    parser = argparse.ArgumentParser(description="Extract keyframes from LEVERB videos")
    parser.add_argument(
        "--output-dir", default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory for extracted frames",
    )
    parser.add_argument(
        "--num-per-chunk", type=int, default=1,
        help="Number of videos to process per chunk (default: 1, use -1 for all)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    chunks = sorted(LEVERB_VIDEO_ROOT.glob("chunk-*"))
    if not chunks:
        print(f"No chunks found in {LEVERB_VIDEO_ROOT}")
        return

    for chunk_dir in chunks:
        chunk_name = chunk_dir.name  # e.g. "chunk-000"
        tpv_dir = chunk_dir / "observation.images.tpv_cam"
        if not tpv_dir.exists():
            print(f"\n=== {chunk_name}: tpv_cam not found, skipping ===")
            continue

        videos = sorted(tpv_dir.glob("*.mp4"))
        if args.num_per_chunk > 0:
            videos = videos[:args.num_per_chunk]

        print(f"\n=== {chunk_name}: processing {len(videos)} video(s) ===")
        for vid in videos:
            episode_id = vid.stem  # e.g. "episode_000025"
            results = extract_frames(vid, output_dir, episode_id, chunk_name)
            manifest.extend(results)

    # Write manifest
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"\n=== Done: {len(manifest)} frames saved to {output_dir} ===")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
