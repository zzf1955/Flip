"""
Extract frames from inpaint_v2 SAM2 results for camera calibration.

For each folder in data/output/inpaint_v2/:
  - sam2_overlay.mp4 → extracted/overlay/*.jpg  (visual browsing)
  - sam2_mask.mp4    → extracted/mask/*.png     (binary GT for PSO)
  - Generates a contact sheet (thumbnail grid) for quick browsing
  - Generates a template calib_frames.json

Usage:
  python scripts/extract_sam2_frames.py
  python scripts/extract_sam2_frames.py --input-dir data/output/inpaint_v2/Brainco_Make_The_Bed_ep150
"""

import sys
import os
import argparse
import json
import numpy as np
import cv2

sys.stdout.reconfigure(line_buffering=True)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPAINT_V2_DIR = os.path.join(BASE_DIR, "data", "output", "inpaint_v2")

# Map folder names to (task, episode)
FOLDER_MAP = {
    "Brainco_Make_The_Bed_ep150": ("G1_WBT_Brainco_Make_The_Bed", 150),
    "Brainco_Pickup_Pillow_ep150": ("G1_WBT_Brainco_Pickup_Pillow", 150),
    "ep000_0-150": ("G1_WBT_Brainco_Make_The_Bed", 0),
}


def extract_folder(folder_path, folder_name):
    """Extract overlay JPEGs and binary mask PNGs from one folder."""
    import av

    overlay_dir = os.path.join(folder_path, "extracted", "overlay")
    mask_dir = os.path.join(folder_path, "extracted", "mask")
    os.makedirs(overlay_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    overlay_path = os.path.join(folder_path, "sam2_overlay.mp4")
    mask_path = os.path.join(folder_path, "sam2_mask.mp4")

    if not os.path.exists(overlay_path) or not os.path.exists(mask_path):
        print(f"  SKIP {folder_name}: missing sam2_overlay.mp4 or sam2_mask.mp4")
        return 0

    # Extract overlay frames
    print(f"  Extracting overlay frames...")
    n_overlay = 0
    container = av.open(overlay_path)
    for frame in container.decode(container.streams.video[0]):
        img = frame.to_ndarray(format='bgr24')
        cv2.imwrite(os.path.join(overlay_dir, f"{n_overlay:05d}.jpg"), img,
                     [cv2.IMWRITE_JPEG_QUALITY, 95])
        n_overlay += 1
    container.close()

    # Extract mask frames (binarize)
    print(f"  Extracting + binarizing mask frames...")
    n_mask = 0
    mask_coverages = []
    container = av.open(mask_path)
    for frame in container.decode(container.streams.video[0]):
        img = frame.to_ndarray(format='bgr24')
        # Any channel > 10 → robot pixel (removes H.264 compression noise)
        binary = np.any(img > 10, axis=2).astype(np.uint8) * 255
        cv2.imwrite(os.path.join(mask_dir, f"{n_mask:05d}.png"), binary)
        coverage = np.count_nonzero(binary) / binary.size * 100
        mask_coverages.append(coverage)
        n_mask += 1
    container.close()

    assert n_overlay == n_mask, \
        f"Frame count mismatch: overlay={n_overlay}, mask={n_mask}"

    print(f"  {n_overlay} frames extracted")
    print(f"  Mask coverage: min={min(mask_coverages):.1f}%, "
          f"max={max(mask_coverages):.1f}%, mean={np.mean(mask_coverages):.1f}%")

    # Generate contact sheet
    generate_contact_sheet(overlay_dir, n_overlay, folder_path, folder_name)

    return n_overlay


def generate_contact_sheet(overlay_dir, n_frames, folder_path, folder_name):
    """Create a thumbnail grid for quick visual browsing."""
    thumb_w, thumb_h = 160, 120
    cols = 10
    # Sample every Nth frame to fit ~100 thumbnails
    step = max(1, n_frames // 100)
    indices = list(range(0, n_frames, step))
    rows = (len(indices) + cols - 1) // cols

    sheet = np.zeros((rows * (thumb_h + 20), cols * thumb_w, 3), dtype=np.uint8)

    for i, idx in enumerate(indices):
        r, c = divmod(i, cols)
        img = cv2.imread(os.path.join(overlay_dir, f"{idx:05d}.jpg"))
        if img is None:
            continue
        thumb = cv2.resize(img, (thumb_w, thumb_h))
        y = r * (thumb_h + 20)
        x = c * thumb_w
        sheet[y:y + thumb_h, x:x + thumb_w] = thumb
        # Frame index label
        cv2.putText(sheet, f"{idx}", (x + 2, y + thumb_h + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    out_path = os.path.join(folder_path, "extracted", "contact_sheet.jpg")
    cv2.imwrite(out_path, sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"  Contact sheet: {out_path} ({len(indices)} thumbnails, step={step})")


def generate_template_json(folders_info, out_path):
    """Generate template calib_frames.json for user to fill in."""
    frames = []
    for folder_name, n_frames in folders_info:
        if folder_name not in FOLDER_MAP:
            continue
        task, episode = FOLDER_MAP[folder_name]
        frames.append({
            "source_dir": folder_name,
            "task": task,
            "episode": episode,
            "total_frames": n_frames,
            "frame_indices": [],
        })

    template = {"frames": frames}
    with open(out_path, 'w') as f:
        json.dump(template, f, indent=2)
    print(f"\nTemplate manifest: {out_path}")
    print("  Edit frame_indices lists after browsing contact sheets.")


def main():
    parser = argparse.ArgumentParser(
        description="Extract SAM2 frames for camera calibration")
    parser.add_argument("--input-dir", type=str, default=None,
                        help="Single folder to process (default: all in inpaint_v2/)")
    args = parser.parse_args()

    if args.input_dir:
        folders = [(os.path.basename(args.input_dir), args.input_dir)]
    else:
        if not os.path.isdir(INPAINT_V2_DIR):
            print(f"Not found: {INPAINT_V2_DIR}")
            sys.exit(1)
        folders = []
        for name in sorted(os.listdir(INPAINT_V2_DIR)):
            path = os.path.join(INPAINT_V2_DIR, name)
            if os.path.isdir(path):
                folders.append((name, path))

    print(f"Processing {len(folders)} folder(s)\n")
    folders_info = []

    for folder_name, folder_path in folders:
        print(f"=== {folder_name} ===")
        n = extract_folder(folder_path, folder_name)
        folders_info.append((folder_name, n))
        print()

    # Generate template JSON
    json_path = os.path.join(BASE_DIR, "data", "calib_frames.json")
    generate_template_json(folders_info, json_path)


if __name__ == "__main__":
    main()
