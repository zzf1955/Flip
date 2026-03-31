"""Qwen-Image-Edit text-guided image editing (no mask) via ComfyUI REST API.

Uses Qwen2.5-VL as text encoder for visual understanding + text instruction.
No mask needed — the model infers what to change from the text.

Usage:
  python scripts/comfyui_qwen_edit.py --image data/leverb_frames/episode_000025_fmid.png
  python scripts/comfyui_qwen_edit.py --batch-dir data/leverb_frames/
"""
import argparse
import random
from pathlib import Path

from comfyui_client import ComfyUIClient

DEFAULT_PROMPT = (
    "Replace the robot with a realistic human performing the same action. "
    "Keep the background, furniture, and all other objects exactly unchanged. "
    "The human should have natural skin, realistic proportions, and appropriate clothing."
)


def build_qwen_edit_workflow(
    image_name,
    prompt,
    width=1024,
    height=576,
    steps=28,
    cfg=1.0,
    denoise=1.0,
    seed=None,
    output_prefix="qwen_edit",
):
    """Build a Qwen-Image-Edit workflow in ComfyUI API format."""
    if seed is None:
        seed = random.randint(0, 2**32 - 1)

    return {
        "1": {
            "class_type": "UnetLoaderGGUF",
            "inputs": {"unet_name": "qwen-image-edit/Qwen_Image_Edit-Q4_K_S.gguf"},
        },
        "2": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": "qwen/qwen_2.5_vl_7b_fp8_scaled.safetensors",
                "type": "qwen_image",
            },
        },
        "3": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": "qwen-image/qwen_image_vae.safetensors"},
        },
        "4": {
            "class_type": "LoadImage",
            "inputs": {"image": image_name},
        },
        "5": {
            "class_type": "TextEncodeQwenImageEdit",
            "inputs": {
                "clip": ["2", 0],
                "prompt": prompt,
                "vae": ["3", 0],
                "image": ["4", 0],
            },
        },
        "6": {
            "class_type": "TextEncodeQwenImageEdit",
            "inputs": {
                "clip": ["2", 0],
                "prompt": "",
            },
        },
        "7": {
            "class_type": "EmptySD3LatentImage",
            "inputs": {"width": width, "height": height, "batch_size": 1},
        },
        "8": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["1", 0],
                "seed": seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": "euler",
                "scheduler": "simple",
                "positive": ["5", 0],
                "negative": ["6", 0],
                "latent_image": ["7", 0],
                "denoise": denoise,
            },
        },
        "9": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["8", 0], "vae": ["3", 0]},
        },
        "10": {
            "class_type": "SaveImage",
            "inputs": {
                "images": ["9", 0],
                "filename_prefix": output_prefix,
            },
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Qwen-Image-Edit via ComfyUI")
    parser.add_argument("--image", help="Source image path (single mode)")
    parser.add_argument("--batch-dir", help="Process all images in directory")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--output-dir", default="data/leverb_edited/qwen/")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=576)
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--cfg", type=float, default=1.0)
    parser.add_argument("--denoise", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    client = ComfyUIClient(args.host, args.port)

    # Collect images
    images = []
    if args.image:
        images.append(Path(args.image))
    elif args.batch_dir:
        batch_dir = Path(args.batch_dir)
        images = sorted(
            p for p in batch_dir.iterdir()
            if p.suffix.lower() in (".jpg", ".jpeg", ".png")
        )
    else:
        parser.error("Provide --image or --batch-dir")

    print(f"Processing {len(images)} image(s)")

    for img_path in images:
        print(f"\n--- Qwen-Image-Edit ---")
        print(f"  Image: {img_path}")
        print(f"  Prompt: {args.prompt[:80]}...")

        img_result = client.upload_image(img_path)
        image_name = img_result.get("name", img_path.name)

        output_prefix = f"qwen_{img_path.stem}"
        workflow = build_qwen_edit_workflow(
            image_name=image_name,
            prompt=args.prompt,
            width=args.width,
            height=args.height,
            steps=args.steps,
            cfg=args.cfg,
            denoise=args.denoise,
            seed=args.seed,
            output_prefix=output_prefix,
        )

        prompt_id = client.queue_prompt(workflow)
        history = client.wait_for_completion(prompt_id)
        client.download_results(history, output_dir=args.output_dir)

    print(f"\nDone. {len(images)} image(s) processed.")


if __name__ == "__main__":
    main()
