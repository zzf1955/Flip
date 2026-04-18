"""Mask post-processing and inpainting backends (LaMa, ProPainter)."""

import numpy as np
import cv2


def postprocess_mask(mask):
    """Smooth -> dilate -> edge blur.

    1. Gaussian blur (7x7) to remove aliasing
    2. Dilate (41x41 ellipse) for 20px safety margin
    3. Gaussian blur (31x31) for soft edges (LaMa supports grayscale masks)
    """
    out = cv2.GaussianBlur(mask, (7, 7), 0)
    out = (out > 128).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))
    out = cv2.dilate(out, kernel)
    blurred = cv2.GaussianBlur(out, (31, 31), 0)
    out = np.maximum(out, blurred)
    return out


def grabcut_refine(img, mesh_mask, gc_iter=3):
    """GrabCut expansion from FK mesh mask."""
    h, w = img.shape[:2]
    gc_mask = np.full((h, w), cv2.GC_BGD, dtype=np.uint8)
    gc_mask[mesh_mask > 0] = cv2.GC_FGD
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    dilated = cv2.dilate(mesh_mask, kernel)
    gc_mask[(dilated > 0) & (mesh_mask == 0)] = cv2.GC_PR_FGD
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    cv2.grabCut(img, gc_mask, None, bgdModel, fgdModel, gc_iter, cv2.GC_INIT_WITH_MASK)
    return np.where(
        (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0
    ).astype(np.uint8)


def init_lama():
    """Lazy-load SimpleLama model."""
    from simple_lama_inpainting import SimpleLama
    return SimpleLama()


def run_lama(lama, img_bgr, mask):
    """Run LaMa inpainting on a single frame.

    Args:
        lama: SimpleLama instance
        img_bgr: input image (BGR, uint8)
        mask: binary mask (uint8, 255=inpaint region)

    Returns:
        Inpainted image (BGR, uint8)
    """
    from PIL import Image
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    pil_mask = Image.fromarray(mask).convert("L")
    result = lama(pil_img, pil_mask)
    return cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)
