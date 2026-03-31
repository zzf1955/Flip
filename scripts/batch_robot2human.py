"""Batch Robot→Human: auto mask + ControlNet pose + Flux Fill inpaint.

Pipeline:
  1. Resize frames to 480P (856×480)
  2. GroundingDINO detects robot bounding box
  3. Create rectangular mask from bbox
  4. Submit to ComfyUI: SDPose + ControlNet + Flux Fill inpaint
  5. Output 5 images per frame: original, bbox, mask, pose, result

Usage:
  python scripts/batch_robot2human.py
  python scripts/batch_robot2human.py --input-dir data/leverb_frames/ --width 856 --height 480
"""
import argparse
import random
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from comfyui_client import ComfyUIClient


# ---------------------------------------------------------------------------
# Robot Detection (GroundingDINO via HuggingFace transformers)
# ---------------------------------------------------------------------------

_detector = None
_processor = None
_sam_predictor = None


def _load_detector(device="cpu"):
    """Lazy-load GroundingDINO model."""
    global _detector, _processor
    if _detector is None:
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        model_id = "IDEA-Research/grounding-dino-tiny"
        print(f"Loading GroundingDINO from {model_id}...")
        _processor = AutoProcessor.from_pretrained(model_id)
        _detector = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
        print("  GroundingDINO loaded.")
    return _detector, _processor


def _load_sam2(device="cpu"):
    """Lazy-load SAM2 model."""
    global _sam_predictor
    if _sam_predictor is None:
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        print("Loading SAM2 (sam2-hiera-small)...")
        _sam_predictor = SAM2ImagePredictor.from_pretrained(
            "facebook/sam2-hiera-small", device=device
        )
        print("  SAM2 loaded.")
    return _sam_predictor


def detect_robot_bbox(image_bgr, text="robot . humanoid", threshold=0.25, device="cpu"):
    """Detect robot bounding box using GroundingDINO.

    Returns: (x1, y1, x2, y2, score) or None if not detected.
    """
    from PIL import Image as PILImage

    model, processor = _load_detector(device)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_image = PILImage.fromarray(image_rgb)

    inputs = processor(images=pil_image, text=text, return_tensors="pt").to(device)
    with __import__("torch").no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=threshold,
        text_threshold=threshold,
        target_sizes=[pil_image.size[::-1]],
    )

    if len(results) == 0 or len(results[0]["boxes"]) == 0:
        return None

    # Pick highest confidence detection
    scores = results[0]["scores"]
    best_idx = scores.argmax().item()
    box = results[0]["boxes"][best_idx].cpu().numpy()
    score = scores[best_idx].item()

    x1, y1, x2, y2 = box.astype(int)
    return (int(x1), int(y1), int(x2), int(y2), float(score))


def create_sam2_mask(image_bgr, bbox, mask_grow=10, device="cpu"):
    """Create pixel-level mask using SAM2 with bbox prompt.

    Args:
        image_bgr: input image (BGR)
        bbox: (x1, y1, x2, y2, ...) from GroundingDINO
        mask_grow: dilate mask by this many pixels for better inpaint blending
        device: "cpu" or "mps"

    Returns: mask as numpy array (h, w, 3), white=255 for robot, black=0 for background
    """
    predictor = _load_sam2(device)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    predictor.set_image(image_rgb)

    input_box = np.array(bbox[:4], dtype=np.float32).reshape(1, 4)
    masks, scores, _ = predictor.predict(box=input_box, multimask_output=False)
    mask_bool = masks[0].astype(bool)

    # Dilate mask for smoother inpaint edges
    if mask_grow > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (mask_grow * 2, mask_grow * 2))
        mask_uint8 = mask_bool.astype(np.uint8) * 255
        mask_uint8 = cv2.dilate(mask_uint8, kernel, iterations=1)
        mask_bool = mask_uint8 > 0

    mask_bgr = np.zeros((*mask_bool.shape, 3), dtype=np.uint8)
    mask_bgr[mask_bool] = 255
    return mask_bgr


def draw_bbox_on_image(image_bgr, bbox):
    """Draw detection bbox on image. Returns copy with green rectangle + score."""
    vis = image_bgr.copy()
    x1, y1, x2, y2 = bbox[:4]
    score = bbox[4] if len(bbox) > 4 else 0.0
    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
    label = f"robot {score:.2f}"
    cv2.putText(vis, label, (x1, max(y1 - 8, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return vis


def fallback_center_bbox(image_shape, ratio=0.6):
    """Fallback: center 60% of image as bbox."""
    h, w = image_shape[:2]
    margin_x = int(w * (1 - ratio) / 2)
    margin_y = int(h * (1 - ratio) / 2)
    return (margin_x, margin_y, w - margin_x, h - margin_y, 0.0)


# ---------------------------------------------------------------------------
# ComfyUI Workflow (API prompt format)
# ---------------------------------------------------------------------------

def build_controlnet_pose_workflow(
    image_name,
    mask_name,
    prompt_positive="a human",
    prompt_t5xxl="a human is walking",
    steps=16,
    cfg=1.0,
    denoise=0.85,
    cn_strength=0.8,
    guidance=30.0,
    seed=None,
    output_prefix="result",
    pose_prefix="pose",
):
    """Build ControlNet + Pose + Flux Fill inpaint workflow (API format).

    Replicates the user-verified flux_controlnet_pose_robot2human workflow.
    """
    if seed is None:
        seed = random.randint(0, 2**32 - 1)

    return {
        # Flux Fill model
        "1": {
            "class_type": "UnetLoaderGGUF",
            "inputs": {"unet_name": "flux1-fill-dev-Q8_0.gguf"},
        },
        # CLIP
        "2": {
            "class_type": "DualCLIPLoader",
            "inputs": {
                "clip_name1": "clip_l.safetensors",
                "clip_name2": "t5xxl_fp8_e4m3fn.safetensors",
                "type": "flux",
            },
        },
        # VAE
        "3": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": "ae.safetensors"},
        },
        # ControlNet Union Pro
        "4": {
            "class_type": "ControlNetLoader",
            "inputs": {"control_net_name": "flux-controlnet-union-pro/diffusion_pytorch_model.safetensors"},
        },
        # Set ControlNet type to openpose
        "5": {
            "class_type": "SetUnionControlNetType",
            "inputs": {"control_net": ["4", 0], "type": "openpose"},
        },
        # Positive prompt
        "6": {
            "class_type": "CLIPTextEncodeFlux",
            "inputs": {
                "clip": ["2", 0],
                "clip_l": prompt_positive,
                "t5xxl": prompt_t5xxl,
                "guidance": guidance,
            },
        },
        # Negative prompt (empty)
        "7": {
            "class_type": "CLIPTextEncodeFlux",
            "inputs": {
                "clip": ["2", 0],
                "clip_l": "",
                "t5xxl": "",
                "guidance": guidance,
            },
        },
        # Source image
        "8": {
            "class_type": "LoadImage",
            "inputs": {"image": image_name},
        },
        # Mask image
        "9": {
            "class_type": "LoadImage",
            "inputs": {"image": mask_name},
        },
        # SDPose model
        "10": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": "sdpose_wholebody_fp16.safetensors"},
        },
        # Pose extraction
        "11": {
            "class_type": "SDPoseKeypointExtractor",
            "inputs": {
                "model": ["10", 0],
                "vae": ["10", 2],
                "image": ["8", 0],
                "batch_size": 16,
            },
        },
        # Draw pose keypoints
        "12": {
            "class_type": "SDPoseDrawKeypoints",
            "inputs": {
                "keypoints": ["11", 0],
                "draw_body": True,
                "draw_hands": True,
                "draw_face": True,
                "draw_feet": False,
                "stick_width": 4,
                "face_point_size": 3,
                "score_threshold": 0.3,
            },
        },
        # ControlNet Apply (pose image → conditioning)
        "13": {
            "class_type": "ControlNetApplyAdvanced",
            "inputs": {
                "positive": ["6", 0],
                "negative": ["7", 0],
                "control_net": ["5", 0],
                "image": ["12", 0],
                "vae": ["3", 0],
                "strength": cn_strength,
                "start_percent": 0.0,
                "end_percent": 1.0,
            },
        },
        # Mask: ImageToMask (red channel)
        "14": {
            "class_type": "ImageToMask",
            "inputs": {"image": ["9", 0], "channel": "red"},
        },
        # GrowMask
        "15": {
            "class_type": "GrowMask",
            "inputs": {"mask": ["14", 0], "expand": 5, "tapered_corners": True},
        },
        # Inpaint conditioning (ControlNet-enhanced + mask)
        "16": {
            "class_type": "InpaintModelConditioning",
            "inputs": {
                "positive": ["13", 0],
                "negative": ["13", 1],
                "vae": ["3", 0],
                "pixels": ["8", 0],
                "mask": ["15", 0],
                "noise_mask": True,
            },
        },
        # Differential Diffusion
        "17": {
            "class_type": "DifferentialDiffusion",
            "inputs": {"model": ["1", 0]},
        },
        # KSampler
        "18": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["17", 0],
                "seed": seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": "ddim",
                "scheduler": "ddim_uniform",
                "positive": ["16", 0],
                "negative": ["16", 1],
                "latent_image": ["16", 2],
                "denoise": denoise,
            },
        },
        # VAE Decode
        "19": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["18", 0], "vae": ["3", 0]},
        },
        # Save result
        "20": {
            "class_type": "SaveImage",
            "inputs": {"images": ["19", 0], "filename_prefix": output_prefix},
        },
        # Save pose debug
        "21": {
            "class_type": "SaveImage",
            "inputs": {"images": ["12", 0], "filename_prefix": pose_prefix},
        },
    }


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

def resize_image(image_bgr, width, height):
    """Resize image to target dimensions."""
    return cv2.resize(image_bgr, (width, height), interpolation=cv2.INTER_AREA)


def process_single_frame(
    client, image_path, output_dir, width, height,
    detect_text, bbox_padding, threshold, device, **workflow_kwargs
):
    """Process a single frame: detect → mask → ComfyUI → save 5 outputs."""
    frame_name = image_path.stem
    frame_dir = output_dir / frame_name
    frame_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Processing: {image_path.name}")

    # 1. Load and resize
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"  ERROR: Cannot read {image_path}")
        return False
    image = resize_image(image, width, height)
    original_path = frame_dir / "original.png"
    cv2.imwrite(str(original_path), image)
    print(f"  Saved: original.png ({width}x{height})")

    # 2. Detect robot bbox
    print(f"  Detecting robot...")
    bbox = detect_robot_bbox(image, text=detect_text, threshold=threshold, device=device)
    if bbox is None:
        print(f"  WARNING: No detection, using center fallback")
        bbox = fallback_center_bbox(image.shape)
    else:
        print(f"  Detected: bbox=({bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}) score={bbox[4]:.2f}")

    # 3. Create bbox visualization
    bbox_vis = draw_bbox_on_image(image, bbox)
    bbox_path = frame_dir / "bbox.png"
    cv2.imwrite(str(bbox_path), bbox_vis)
    print(f"  Saved: bbox.png")

    # 4. Create pixel-level mask via SAM2
    print(f"  Segmenting with SAM2...")
    mask = create_sam2_mask(image, bbox, mask_grow=bbox_padding, device=device)
    mask_path = frame_dir / "mask.png"
    cv2.imwrite(str(mask_path), mask)
    print(f"  Saved: mask.png (SAM2 pixel-level, dilated {bbox_padding}px)")

    # 5. Upload to ComfyUI and run workflow
    print(f"  Uploading to ComfyUI...")
    img_result = client.upload_image(str(original_path))
    mask_result = client.upload_image(str(mask_path))

    image_name = img_result.get("name", original_path.name)
    mask_name = mask_result.get("name", mask_path.name)

    result_prefix = f"result_{frame_name}"
    pose_prefix = f"pose_{frame_name}"

    workflow = build_controlnet_pose_workflow(
        image_name=image_name,
        mask_name=mask_name,
        output_prefix=result_prefix,
        pose_prefix=pose_prefix,
        **workflow_kwargs,
    )

    prompt_id = client.queue_prompt(workflow)
    print(f"  Waiting for ComfyUI...")
    history = client.wait_for_completion(prompt_id, timeout=1800)

    # 6. Download outputs
    output_images = client.get_output_images(history)
    downloaded = set()
    for img_info in output_images:
        filename = img_info["filename"]
        subfolder = img_info.get("subfolder", "")
        data = client.download_image(filename, subfolder)

        if filename.startswith(pose_prefix):
            save_path = frame_dir / "pose.png"
        elif filename.startswith(result_prefix):
            save_path = frame_dir / "result.png"
        else:
            continue

        save_path.write_bytes(data)
        downloaded.add(save_path.name)
        print(f"  Saved: {save_path.name}")

    if "pose.png" not in downloaded:
        print("  WARNING: pose.png not found in ComfyUI outputs")

    print(f"  Done: {frame_dir}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Batch Robot→Human with auto mask + ControlNet")
    parser.add_argument("--input-dir", default="data/leverb_frames/")
    parser.add_argument("--output-dir", default="data/leverb_edited/batch/")
    parser.add_argument("--width", type=int, default=856)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--prompt", default="a human")
    parser.add_argument("--prompt-t5", default="a human is walking")
    parser.add_argument("--detect-text", default="robot . humanoid")
    parser.add_argument("--detect-threshold", type=float, default=0.25)
    parser.add_argument("--bbox-padding", type=int, default=1)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--denoise", type=float, default=0.85)
    parser.add_argument("--cn-strength", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="cpu", help="Device for GroundingDINO (cpu/mps/cuda)")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(
        p for p in input_dir.glob("episode_*_f*.png")
        if not p.name.startswith(".") and "_inpaint" not in p.name
    )
    if not images:
        print(f"No PNG files in {input_dir}")
        return

    print(f"Found {len(images)} images in {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Resolution: {args.width}x{args.height}")
    print(f"ComfyUI: {args.host}:{args.port}")

    client = ComfyUIClient(args.host, args.port)

    workflow_kwargs = dict(
        prompt_positive=args.prompt,
        prompt_t5xxl=args.prompt_t5,
        steps=args.steps,
        denoise=args.denoise,
        cn_strength=args.cn_strength,
        seed=args.seed,
    )

    success = 0
    for img_path in images:
        frame_name = img_path.stem
        frame_dir = output_dir / frame_name
        if (frame_dir / "result.png").exists():
            print(f"Skipping {img_path.name} (already done)")
            success += 1
            continue
        ok = process_single_frame(
            client, img_path, output_dir,
            args.width, args.height,
            args.detect_text, args.bbox_padding, args.detect_threshold,
            args.device, **workflow_kwargs,
        )
        if ok:
            success += 1

    print(f"\n{'='*60}")
    print(f"Done: {success}/{len(images)} frames processed")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
