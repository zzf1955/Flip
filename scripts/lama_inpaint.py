"""
Run LaMa inpainting using pre-generated GrabCut masks.
Reads raw frames from test_results/raw_frames/ and masks from test_results/inpaint_mask/.
Outputs panels (original | mask | inpainted) to test_results/lama_results/.
"""

import sys
import os
import glob
import numpy as np
import cv2
from PIL import Image

sys.stdout.reconfigure(line_buffering=True)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "test_results", "raw_frames")
MASK_DIR = os.path.join(BASE_DIR, "test_results", "inpaint_mask")
OUTPUT_DIR = os.path.join(BASE_DIR, "test_results", "lama_results")


def run_lama(img_bgr, mask):
    from simple_lama_inpainting import SimpleLama
    global _lama
    if "_lama" not in globals():
        _lama = SimpleLama()
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    pil_mask = Image.fromarray(mask).convert("L")
    result = _lama(pil_img, pil_mask)
    return cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    mask_files = sorted(glob.glob(os.path.join(MASK_DIR, "*_mask.png")))
    print(f"Found {len(mask_files)} masks")

    total = 0
    for mask_path in mask_files:
        tag = os.path.basename(mask_path).replace("_mask.png", "")
        raw_path = os.path.join(RAW_DIR, f"{tag}.png")

        if not os.path.exists(raw_path):
            print(f"  {tag}: raw frame not found, skip")
            continue

        img = cv2.imread(raw_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        print(f"  {tag}: running LaMa ({mask.sum() // 255} px)...")
        inpainted = run_lama(img, mask)

        # 3-panel: original | mask | inpainted
        mask_vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        panel = np.hstack([img, mask_vis, inpainted])
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{tag}_panel.png"), panel)
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{tag}_inpaint.png"), inpainted)

        total += 1

    print(f"\nDone: {total} frames -> {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
