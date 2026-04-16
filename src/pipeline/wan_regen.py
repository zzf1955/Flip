"""
Wan 2.1 VACE depth+mask human regeneration via ComfyUI API.

Takes cosmos_prepare outputs (composite, depth, mask) and generates
realistic human video using Wan 2.1 VACE model through ComfyUI.

Prerequisites:
  - ComfyUI running on specified port (default 8001)
  - Wan 2.1 VACE model + umt5 text encoder + VAE downloaded

Usage:
  # Start ComfyUI first:
  cd /disk_n/zzf/ComfyUI && python main.py --port 8001 --cuda-device 2

  # Run regeneration:
  python -m src.pipeline.wan_regen \
      --prepare-dir output/human/cosmos_prepare/Pickup_Pillow_MainCamOnly_ep0_s5_d1
"""

import sys
import os
import json
import argparse
import time
import uuid
import urllib.request
import urllib.error
import cv2
import numpy as np

sys.stdout.reconfigure(line_buffering=True)

from src.core.config import WAN_REGEN_DIR, COMFYUI_ROOT
from src.core.data import open_video_writer, write_frame, close_video


# ── ComfyUI API helpers ──

def comfyui_post(url, data):
    """POST JSON to ComfyUI API."""
    req = urllib.request.Request(
        url,
        data=json.dumps(data).encode('utf-8'),
        headers={'Content-Type': 'application/json'},
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


def comfyui_get(url):
    """GET from ComfyUI API."""
    with urllib.request.urlopen(url) as resp:
        return json.loads(resp.read())


def upload_image(server, filepath, subfolder="", image_type="input"):
    """Upload an image file to ComfyUI."""
    import io
    import mimetypes

    filename = os.path.basename(filepath)
    content_type = mimetypes.guess_type(filepath)[0] or 'image/png'

    with open(filepath, 'rb') as f:
        file_data = f.read()

    boundary = uuid.uuid4().hex
    body = (
        f'--{boundary}\r\n'
        f'Content-Disposition: form-data; name="image"; filename="{filename}"\r\n'
        f'Content-Type: {content_type}\r\n\r\n'
    ).encode('utf-8') + file_data + (
        f'\r\n--{boundary}\r\n'
        f'Content-Disposition: form-data; name="subfolder"\r\n\r\n'
        f'{subfolder}\r\n'
        f'--{boundary}\r\n'
        f'Content-Disposition: form-data; name="type"\r\n\r\n'
        f'{image_type}\r\n'
        f'--{boundary}--\r\n'
    ).encode('utf-8')

    req = urllib.request.Request(
        f'{server}/upload/image',
        data=body,
        headers={'Content-Type': f'multipart/form-data; boundary={boundary}'},
        method='POST',
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


def wait_for_prompt(server, prompt_id, timeout=600):
    """Poll ComfyUI until prompt execution completes."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            history = comfyui_get(f'{server}/history/{prompt_id}')
            if prompt_id in history:
                return history[prompt_id]
        except Exception:
            pass
        time.sleep(2)
    raise TimeoutError(f"ComfyUI prompt {prompt_id} did not complete in {timeout}s")


def download_output(server, filename, subfolder, output_type="output"):
    """Download output file from ComfyUI."""
    url = f'{server}/view?filename={filename}&subfolder={subfolder}&type={output_type}'
    with urllib.request.urlopen(url) as resp:
        return resp.read()


# ── Video frame I/O ──

def load_video_frames(video_path):
    """Load all frames from a video as list of BGR numpy arrays."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


def save_frames_as_png(frames, out_dir, prefix="frame"):
    """Save frames as numbered PNG files. Returns list of paths."""
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for i, frame in enumerate(frames):
        path = os.path.join(out_dir, f"{prefix}_{i:04d}.png")
        cv2.imwrite(path, frame)
        paths.append(path)
    return paths


# ── Workflow builder ──

def build_vace_workflow(
    control_frames_dir,
    mask_frames_dir,
    n_frames,
    width, height,
    prompt_text,
    negative_prompt="",
    steps=30,
    cfg=6.0,
    strength=1.0,
    seed=2025,
    model_name="wan2.1_vace_1.3B_fp16.safetensors",
    text_encoder="umt5_xxl_fp8_e4m3fn_scaled.safetensors",
    vae_name="wan_2.1_vae.safetensors",
):
    """Build a ComfyUI workflow JSON for WanVaceToVideo.

    This constructs the node graph programmatically.
    """
    # Wan 2.1 VACE requires length = 4k+1
    length = ((n_frames - 1) // 4) * 4 + 1
    if length < 5:
        length = 5

    workflow = {
        # Load UNET (diffusion model)
        "1": {
            "class_type": "UNETLoader",
            "inputs": {
                "unet_name": model_name,
                "weight_dtype": "default",
            },
        },
        # Load text encoder
        "2": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": text_encoder,
                "type": "wan",
            },
        },
        # Load VAE
        "3": {
            "class_type": "VAELoader",
            "inputs": {
                "vae_name": vae_name,
            },
        },
        # Positive prompt
        "4": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": prompt_text,
                "clip": ["2", 0],
            },
        },
        # Negative prompt
        "5": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": negative_prompt,
                "clip": ["2", 0],
            },
        },
        # Load control video (depth maps) as image sequence
        "6": {
            "class_type": "LoadImageSequence" if os.path.isdir(control_frames_dir) else "VHS_LoadVideo",
            "inputs": {
                "image": control_frames_dir if os.path.isdir(control_frames_dir) else control_frames_dir,
            },
        },
        # Load mask sequence
        "7": {
            "class_type": "LoadImageSequence" if os.path.isdir(mask_frames_dir) else "VHS_LoadVideo",
            "inputs": {
                "image": mask_frames_dir if os.path.isdir(mask_frames_dir) else mask_frames_dir,
            },
        },
        # WanVaceToVideo conditioning
        "10": {
            "class_type": "WanVaceToVideo",
            "inputs": {
                "positive": ["4", 0],
                "negative": ["5", 0],
                "vae": ["3", 0],
                "width": width,
                "height": height,
                "length": length,
                "batch_size": 1,
                "strength": strength,
                "control_video": ["6", 0],
                "control_masks": ["7", 0],
            },
        },
        # KSampler
        "11": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["1", 0],
                "positive": ["10", 0],
                "negative": ["10", 1],
                "latent_image": ["10", 2],
                "seed": seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": "euler",
                "scheduler": "normal",
                "denoise": 1.0,
            },
        },
        # Trim latent (remove reference frame padding)
        "12": {
            "class_type": "TrimVideoLatent",
            "inputs": {
                "samples": ["11", 0],
                "trim_amount": ["10", 3],
            },
        },
        # VAE Decode
        "13": {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["12", 0],
                "vae": ["3", 0],
            },
        },
        # Save output
        "14": {
            "class_type": "SaveAnimatedWEBP",
            "inputs": {
                "images": ["13", 0],
                "filename_prefix": "wan_regen",
                "fps": 16,
                "lossless": False,
                "quality": 90,
                "method": "default",
            },
        },
    }
    return workflow


# ── Main ──

def main():
    parser = argparse.ArgumentParser(
        description="Wan 2.1 VACE human regeneration via ComfyUI")
    parser.add_argument("--prepare-dir", type=str, required=True, dest="prepare_dir",
                        help="cosmos_prepare output directory")
    parser.add_argument("--server", type=str, default="http://localhost:8001",
                        help="ComfyUI server URL")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--cfg", type=float, default=6.0)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default=None, dest="out_dir")
    parser.add_argument("--no-composite", dest="composite", action="store_false", default=True,
                        help="Skip post-processing composite (bg + generated)")
    args = parser.parse_args()

    # ── Validate inputs ──
    prepare_dir = args.prepare_dir
    composite_path = os.path.join(prepare_dir, "composite.mp4")
    depth_path = os.path.join(prepare_dir, "depth_blurred.mp4")
    mask_path = os.path.join(prepare_dir, "smplh_mask.mp4")

    for p, name in [(composite_path, "composite"), (depth_path, "depth_blurred"), (mask_path, "smplh_mask")]:
        if not os.path.isfile(p):
            print(f"ERROR: {name}.mp4 not found at {p}")
            sys.exit(1)

    # ── Check ComfyUI is running ──
    try:
        stats = comfyui_get(f'{args.server}/system_stats')
        print(f"ComfyUI connected: {args.server}")
    except Exception as e:
        print(f"ERROR: Cannot connect to ComfyUI at {args.server}")
        print(f"Start it first: cd {COMFYUI_ROOT} && python main.py --port 8001 --cuda-device 2")
        sys.exit(1)

    # ── Load video frames ──
    print("Loading input videos...")
    composite_frames = load_video_frames(composite_path)
    depth_frames = load_video_frames(depth_path)
    mask_frames = load_video_frames(mask_path)
    n_frames = len(composite_frames)
    h, w = composite_frames[0].shape[:2]
    print(f"  {n_frames} frames, {w}x{h}")

    # ── Output directory ──
    tag = os.path.basename(prepare_dir)
    out_dir = args.out_dir or os.path.join(WAN_REGEN_DIR, tag)
    os.makedirs(out_dir, exist_ok=True)

    # ── Save frames as PNG for ComfyUI upload ──
    tmp_dir = os.path.join(out_dir, "tmp_frames")
    print("Saving frames for ComfyUI...")
    depth_frame_paths = save_frames_as_png(depth_frames, os.path.join(tmp_dir, "depth"), "depth")
    mask_gray_frames = []
    for m in mask_frames:
        gray = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY) if m.ndim == 3 else m
        mask_gray_frames.append(gray)
    mask_frame_paths = save_frames_as_png(mask_gray_frames, os.path.join(tmp_dir, "mask"), "mask")

    # ── Upload frames to ComfyUI ──
    print("Uploading to ComfyUI...")
    subfolder = f"wan_regen_{tag}"
    for p in depth_frame_paths:
        upload_image(args.server, p, subfolder=subfolder)
    for p in mask_frame_paths:
        upload_image(args.server, p, subfolder=subfolder)
    print(f"  Uploaded {len(depth_frame_paths)} depth + {len(mask_frame_paths)} mask frames")

    # ── Build and submit workflow ──
    prompt_text = args.prompt or (
        "First-person view of a person performing household tasks in an indoor room. "
        "The person has realistic skin, natural clothing, and proper lighting."
    )

    workflow = build_vace_workflow(
        control_frames_dir=os.path.join(tmp_dir, "depth"),
        mask_frames_dir=os.path.join(tmp_dir, "mask"),
        n_frames=n_frames,
        width=w, height=h,
        prompt_text=prompt_text,
        steps=args.steps,
        cfg=args.cfg,
        strength=args.strength,
        seed=args.seed,
    )

    print(f"\nSubmitting workflow to ComfyUI...")
    print(f"  Steps: {args.steps}, CFG: {args.cfg}, Strength: {args.strength}")

    client_id = str(uuid.uuid4())
    result = comfyui_post(f'{args.server}/prompt', {
        "prompt": workflow,
        "client_id": client_id,
    })

    if "error" in result:
        print(f"ERROR: {result['error']}")
        # Save workflow for debugging
        wf_path = os.path.join(out_dir, "workflow_debug.json")
        with open(wf_path, 'w') as f:
            json.dump(workflow, f, indent=2)
        print(f"  Workflow saved to: {wf_path}")
        sys.exit(1)

    prompt_id = result["prompt_id"]
    print(f"  Prompt ID: {prompt_id}")
    print(f"  Waiting for completion...")

    # ── Wait for result ──
    t0 = time.time()
    history = wait_for_prompt(args.server, prompt_id, timeout=600)
    elapsed = time.time() - t0
    print(f"  Completed in {elapsed:.0f}s")

    # ── Download output ──
    outputs = history.get("outputs", {})
    output_node = outputs.get("14", {})  # SaveAnimatedWEBP node
    if not output_node:
        print("WARNING: No output found in node 14, checking all nodes...")
        for nid, nout in outputs.items():
            if "images" in nout or "gifs" in nout:
                output_node = nout
                print(f"  Found output in node {nid}")
                break

    if "images" in output_node:
        for img_info in output_node["images"]:
            filename = img_info["filename"]
            subfolder = img_info.get("subfolder", "")
            data = download_output(args.server, filename, subfolder)
            save_path = os.path.join(out_dir, filename)
            with open(save_path, 'wb') as f:
                f.write(data)
            print(f"  Saved: {save_path} ({len(data)/1024:.0f} KB)")
    elif "gifs" in output_node:
        for gif_info in output_node["gifs"]:
            filename = gif_info["filename"]
            subfolder = gif_info.get("subfolder", "")
            data = download_output(args.server, filename, subfolder)
            save_path = os.path.join(out_dir, filename)
            with open(save_path, 'wb') as f:
                f.write(data)
            print(f"  Saved: {save_path} ({len(data)/1024:.0f} KB)")

    # ── Post-processing: composite with original background ──
    if args.composite:
        print("\nPost-processing: compositing with original background...")
        # Load original background from composite (inpaint BG + SMPLH)
        # We want: keep bg where mask=0, use generated where mask=1
        # But first we need the generated frames — for now skip if output is WEBP
        print("  (Post-processing will be done after converting WEBP output to frames)")

    # ── Save workflow for reference ──
    wf_path = os.path.join(out_dir, "workflow.json")
    with open(wf_path, 'w') as f:
        json.dump(workflow, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Wan 2.1 VACE regeneration complete!")
    print(f"  Output: {out_dir}")
    print(f"  Workflow: {wf_path}")


if __name__ == "__main__":
    main()
