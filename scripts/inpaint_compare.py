"""
Compare inpainting methods for robot arm removal:
  - LaMa (local, fast, no prompt)
  - Flux Fill (ComfyUI, slow, prompt-guided)

Usage:
  python scripts/inpaint_compare.py --lama-only
  python scripts/inpaint_compare.py --port 8001
"""

import sys
import os
import argparse
import tempfile
import random
import numpy as np
import pandas as pd
import cv2

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mask_inpaint import (
    extract_frame, build_q, do_fk, parse_urdf_meshes,
    make_camera, render_triangle_mask, dilate_mask,
    BEST_PARAMS, URDF_PATH, MESH_DIR, SKIP_MESHES,
)
import pinocchio as pin

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "test_results", "inpaint_compare")

DILATE_PX = 25  # aggressive dilation to cover mask gaps

# 3 samples: one per task
SAMPLES = [
    {"name": "g1_wbt",       "episode": 0, "frames_per_ep": 1},
    {"name": "g1_wbt_task2", "episode": 0, "frames_per_ep": 1},
    {"name": "g1_wbt_task3", "episode": 0, "frames_per_ep": 1},
]

FLUX_PROMPT = "clean background, indoor scene, no people, no robot, photorealistic"


def run_lama(img_bgr, mask):
    """Run LaMa inpainting. img_bgr: BGR numpy, mask: uint8 (255=inpaint)."""
    from PIL import Image
    from simple_lama_inpainting import SimpleLama

    lama = SimpleLama()
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    pil_mask = Image.fromarray(mask).convert("L")
    result = lama(pil_img, pil_mask)
    result_bgr = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)
    return result_bgr


def build_flux_fill_workflow(
    image_name, mask_name, prompt=FLUX_PROMPT,
    steps=28, cfg=1.0, guidance=20.0, denoise=0.7,
    mask_grow=4, seed=None, output_prefix="flux_inpaint",
):
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


def run_flux_fill(client, img_bgr, mask, tag):
    """Run Flux Fill inpainting via ComfyUI. Returns BGR numpy or None."""
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        img_path = os.path.join(tmpdir, f"{tag}_src.png")
        mask_path = os.path.join(tmpdir, f"{tag}_mask.png")
        cv2.imwrite(img_path, img_bgr)
        cv2.imwrite(mask_path, mask)

        img_result = client.upload_image(img_path)
        mask_result = client.upload_image(mask_path)

        workflow = build_flux_fill_workflow(
            image_name=img_result.get("name", f"{tag}_src.png"),
            mask_name=mask_result.get("name", f"{tag}_mask.png"),
            output_prefix=f"compare_{tag}",
        )

        prompt_id = client.queue_prompt(workflow)
        history = client.wait_for_completion(prompt_id)

        output_images = client.get_output_images(history)
        if not output_images:
            print(f"  WARNING: No Flux output for {tag}")
            return None

        img_data = client.download_image(
            output_images[0]["filename"],
            output_images[0].get("subfolder", ""),
        )
        arr = np.frombuffer(img_data, dtype=np.uint8)
        result = cv2.imdecode(arr, cv2.IMREAD_COLOR)

        # Flux may output at different resolution, resize to match
        h, w = img_bgr.shape[:2]
        if result.shape[:2] != (h, w):
            result = cv2.resize(result, (w, h), interpolation=cv2.INTER_LANCZOS4)

        return result


def main():
    parser = argparse.ArgumentParser(description="Compare LaMa vs Flux Fill inpainting")
    parser.add_argument("--lama-only", action="store_true",
                        help="Only run LaMa (no ComfyUI needed)")
    parser.add_argument("--flux-only", action="store_true",
                        help="Only run Flux Fill via ComfyUI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    run_lama_flag = not args.flux_only
    run_flux_flag = not args.lama_only

    # Setup ComfyUI client if needed
    client = None
    if run_flux_flag:
        from comfyui_client import ComfyUIClient
        client = ComfyUIClient(args.host, args.port)
        print(f"ComfyUI: {client.base}")

    # Load URDF model
    print("Loading URDF...")
    model = pin.buildModelFromUrdf(URDF_PATH, pin.JointModelFreeFlyer())
    data_pin = model.createData()
    link_meshes = parse_urdf_meshes(URDF_PATH)

    total = 0

    for sample in SAMPLES:
        task_name = sample["name"]
        data_dir = os.path.join(BASE_DIR, "data", task_name)
        video_path = os.path.join(data_dir, "videos",
                                   "observation.images.head_stereo_left",
                                   "chunk-000", "file-000.mp4")
        parquet_path = os.path.join(data_dir, "data", "chunk-000", "file-000.parquet")

        if not os.path.exists(parquet_path):
            print(f"SKIP {task_name}: no parquet")
            continue

        print(f"\n--- {task_name} ---")
        df = pd.read_parquet(parquet_path)
        ep = sample["episode"]
        ep_df = df[df["episode_index"] == ep]
        if len(ep_df) == 0:
            print(f"  ep {ep}: not available")
            continue

        max_frame = ep_df["frame_index"].max()
        n = sample["frames_per_ep"]
        sample_frames = np.linspace(0, max_frame, n + 2, dtype=int)[1:-1]

        for fi in sample_frames:
            row = ep_df[ep_df["frame_index"] == fi]
            if len(row) == 0:
                nearest = ep_df.iloc[(ep_df["frame_index"] - fi).abs().argsort()[:1]]
                row = nearest
                fi = int(row["frame_index"].iloc[0])

            rq = np.array(row.iloc[0]["observation.state.robot_q_current"],
                          dtype=np.float64)
            video_frame_idx = int(row.iloc[0]["index"])
            img = extract_frame(video_path, video_frame_idx)
            if img is None:
                print(f"  ep{ep} f{fi}: cannot extract frame")
                continue

            h, w = img.shape[:2]
            q = build_q(model, rq)
            transforms = do_fk(model, data_pin, q)

            # Render and dilate mask
            mask_raw = render_triangle_mask(
                link_meshes, MESH_DIR, transforms, BEST_PARAMS, h, w)
            mask = dilate_mask(mask_raw, DILATE_PX)

            tag = f"{task_name}_ep{ep:03d}_f{fi:04d}"
            print(f"  {tag}: mask ready ({mask.sum() // 255} px)")

            # Run methods
            lama_result = None
            flux_result = None

            if run_lama_flag:
                print(f"  {tag}: running LaMa...")
                lama_result = run_lama(img, mask)
                cv2.imwrite(os.path.join(OUTPUT_DIR, f"{tag}_lama.png"), lama_result)
                print(f"  {tag}: LaMa done")

            if run_flux_flag and client is not None:
                print(f"  {tag}: running Flux Fill...")
                try:
                    flux_result = run_flux_fill(client, img, mask, tag)
                    if flux_result is not None:
                        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{tag}_flux.png"), flux_result)
                        print(f"  {tag}: Flux done")
                except Exception as e:
                    print(f"  {tag}: Flux FAILED: {e}")

            # Save mask
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"{tag}_mask.png"), mask)

            # Build comparison panel
            mask_vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            panels = [img, mask_vis]
            labels = ["Original", "Mask"]

            if lama_result is not None:
                panels.append(lama_result)
                labels.append("LaMa")
            if flux_result is not None:
                panels.append(flux_result)
                labels.append("Flux Fill")

            # Add labels
            labeled = []
            for panel_img, label in zip(panels, labels):
                p = panel_img.copy()
                cv2.putText(p, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 255, 0), 2)
                labeled.append(p)

            panel = np.hstack(labeled)
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"{tag}_panel.png"), panel)
            total += 1

    print(f"\nDone: {total} frames -> {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
