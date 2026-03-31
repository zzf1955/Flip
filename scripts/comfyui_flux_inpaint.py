"""Flux Fill inpainting via ComfyUI REST API.

Supports two modes:
  - With mask: provide --mask for targeted inpainting
  - Without mask: auto-generates full white mask (entire image inpaint)

Usage:
  python scripts/comfyui_flux_inpaint.py --image data/leverb_frames/episode_000025_fmid.png
  python scripts/comfyui_flux_inpaint.py --image img.png --mask mask.png --prompt "a human standing"
"""
import argparse
import random
import tempfile
from pathlib import Path

from comfyui_client import ComfyUIClient

try:
    from PIL import Image
except ImportError:
    Image = None

DEFAULT_PROMPT = (
    "a human performing the same action, realistic, third person view, photorealistic"
)


def create_white_mask(image_path, output_path):
    """Create a full white mask matching the image dimensions."""
    if Image is not None:
        img = Image.open(image_path)
        mask = Image.new("RGB", img.size, (255, 255, 255))
        mask.save(output_path)
    else:
        import cv2
        img = cv2.imread(str(image_path))
        h, w = img.shape[:2]
        import numpy as np
        mask = np.ones((h, w, 3), dtype=np.uint8) * 255
        cv2.imwrite(str(output_path), mask)


def build_flux_fill_workflow(
    image_name,
    mask_name,
    prompt,
    steps=28,
    cfg=1.0,
    guidance=30.0,
    denoise=0.85,
    mask_grow=6,
    seed=None,
    output_prefix="flux_inpaint",
):
    """Build a Flux Fill inpainting workflow in ComfyUI API format."""
    if seed is None:
        seed = random.randint(0, 2**32 - 1)

    return {
        "1": {
            "class_type": "UnetLoaderGGUF",
            "inputs": {"unet_name": "flux1-fill-dev-Q8_0.gguf"},
        },
        "2": {
            "class_type": "DualCLIPLoader",
            "inputs": {
                "clip_name1": "clip_l.safetensors",
                "clip_name2": "t5xxl_fp8_e4m3fn.safetensors",
                "type": "flux",
            },
        },
        "3": {
            "class_type": "CLIPTextEncodeFlux",
            "inputs": {
                "clip": ["2", 0],
                "clip_l": prompt,
                "t5xxl": prompt,
                "guidance": guidance,
            },
        },
        "4": {
            "class_type": "CLIPTextEncodeFlux",
            "inputs": {
                "clip": ["2", 0],
                "clip_l": "",
                "t5xxl": "",
                "guidance": guidance,
            },
        },
        "5": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": "ae.safetensors"},
        },
        "6": {
            "class_type": "LoadImage",
            "inputs": {"image": image_name},
        },
        "7": {
            "class_type": "LoadImage",
            "inputs": {"image": mask_name},
        },
        "8": {
            "class_type": "ImageToMask",
            "inputs": {"image": ["7", 0], "channel": "red"},
        },
        "9": {
            "class_type": "GrowMask",
            "inputs": {
                "mask": ["8", 0],
                "expand": mask_grow,
                "tapered_corners": True,
            },
        },
        "10": {
            "class_type": "InpaintModelConditioning",
            "inputs": {
                "positive": ["3", 0],
                "negative": ["4", 0],
                "vae": ["5", 0],
                "pixels": ["6", 0],
                "mask": ["9", 0],
                "noise_mask": True,
            },
        },
        "11": {
            "class_type": "DifferentialDiffusion",
            "inputs": {"model": ["1", 0]},
        },
        "12": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["11", 0],
                "seed": seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": "euler",
                "scheduler": "simple",
                "positive": ["10", 0],
                "negative": ["10", 1],
                "latent_image": ["10", 2],
                "denoise": denoise,
            },
        },
        "13": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["12", 0], "vae": ["5", 0]},
        },
        "14": {
            "class_type": "SaveImage",
            "inputs": {
                "images": ["13", 0],
                "filename_prefix": output_prefix,
            },
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Flux Fill inpainting via ComfyUI")
    parser.add_argument("--image", required=True, help="Source image path")
    parser.add_argument("--mask", default=None, help="Mask image (white=inpaint). Omit for full-image.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--output-dir", default="data/leverb_edited/flux/")
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--cfg", type=float, default=1.0)
    parser.add_argument("--guidance", type=float, default=30.0)
    parser.add_argument("--denoise", type=float, default=0.85)
    parser.add_argument("--mask-grow", type=int, default=6)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    image_path = Path(args.image)
    client = ComfyUIClient(args.host, args.port)

    # Handle mask
    tmp_mask = None
    if args.mask:
        mask_path = Path(args.mask)
    else:
        tmp_mask = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        create_white_mask(image_path, tmp_mask.name)
        mask_path = Path(tmp_mask.name)
        print(f"  Auto-generated full white mask")

    print(f"\n--- Flux Fill Inpaint ---")
    print(f"  Image: {image_path}")
    print(f"  Mask:  {mask_path}")
    print(f"  Prompt: {args.prompt}")

    # Upload
    img_result = client.upload_image(image_path)
    mask_result = client.upload_image(mask_path)

    # Build workflow
    output_prefix = f"flux_{image_path.stem}"
    workflow = build_flux_fill_workflow(
        image_name=img_result.get("name", image_path.name),
        mask_name=mask_result.get("name", mask_path.name),
        prompt=args.prompt,
        steps=args.steps,
        cfg=args.cfg,
        guidance=args.guidance,
        denoise=args.denoise,
        mask_grow=args.mask_grow,
        seed=args.seed,
        output_prefix=output_prefix,
    )

    # Execute
    prompt_id = client.queue_prompt(workflow)
    history = client.wait_for_completion(prompt_id)
    client.download_results(history, output_dir=args.output_dir)

    # Cleanup temp mask
    if tmp_mask:
        Path(tmp_mask.name).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
