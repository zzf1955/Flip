"""
Mitty cache: encode (human_video, robot_video, prompt) → .pth for each pair.

Input:  training_data/pair/1s/<split>/{video,control_video}/pair_NNNN.mp4
Output: output/mitty_cache_1s/<split>/NNNN.pth with keys:
    human_latent  (1, 48, 5, 30, 40)  clean VAE latent of human (control) video
    robot_latent  (1, 48, 5, 30, 40)  clean VAE latent of robot (target) video
    context_posi  (1, 512, 4096)       T5 positive
    context_nega  (1, 512, 4096)       T5 negative
    human_frames  list[PIL.Image]      original frames for eval video display
    robot_frames  list[PIL.Image]      GT frames for eval video display
    prompt        str                   training prompt
    source_id     str                   source_map lookup (task/ep/seg/clip)

Usage:
    python -m src.pipeline.mitty_cache \\
        --pair-dir training_data/pair/1s/train \\
        --output   output/mitty_cache_1s/train \\
        --device   cuda:0
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.stdout.reconfigure(line_buffering=True)

import av
import torch
from PIL import Image

from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from src.core.config import MAIN_ROOT


MANUAL_DIR = os.path.join(
    "/disk_n/zzf/.cache/huggingface/hub",
    "models--Wan-AI--Wan2.2-TI2V-5B", "manual",
)

NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，"
    "整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，"
    "画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，"
    "静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)


def load_video_as_pil(path: str) -> list[Image.Image]:
    c = av.open(path)
    frames = [f.to_image() for f in c.decode(video=0)]
    c.close()
    return frames


def build_pipe(device: str) -> WanVideoPipeline:
    """Build pipeline with T5 + VAE only (no DiT)."""
    t5_path = os.path.join(MANUAL_DIR, "models_t5_umt5-xxl-enc-bf16.pth")
    vae_path = os.path.join(MANUAL_DIR, "Wan2.2_VAE.pth")
    tokenizer_dir = os.path.join(MANUAL_DIR, "google", "umt5-xxl")

    # T5: ~5.6GB bf16, VAE: ~0.5GB. 两个加一起 GPU 也够，保持 CPU offload 一致行为
    cpu_offload = {
        "offload_dtype": torch.bfloat16,
        "offload_device": "cpu",
        "onload_dtype": torch.bfloat16,
        "onload_device": "cpu",
        "preparing_dtype": torch.bfloat16,
        "preparing_device": device,
        "computation_dtype": torch.bfloat16,
        "computation_device": device,
    }

    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=[
            ModelConfig(path=t5_path, **cpu_offload),
            ModelConfig(path=vae_path, **cpu_offload),
        ],
        tokenizer_config=ModelConfig(path=tokenizer_dir),
    )
    return pipe


@torch.no_grad()
def encode_prompt(pipe: WanVideoPipeline, prompt: str) -> torch.Tensor:
    """Encode prompt to T5 embedding. Returns (1, 512, 4096) bf16."""
    pipe.load_models_to_device(["text_encoder"])
    ids, mask = pipe.tokenizer(prompt, return_mask=True, add_special_tokens=True)
    ids = ids.to(pipe.device)
    mask = mask.to(pipe.device)
    seq_lens = mask.gt(0).sum(dim=1).long()
    prompt_emb = pipe.text_encoder(ids, mask)
    for i, v in enumerate(seq_lens):
        prompt_emb[:, v:] = 0
    return prompt_emb


@torch.no_grad()
def encode_video(pipe: WanVideoPipeline, frames: list[Image.Image]) -> torch.Tensor:
    """VAE-encode frames. Wan 2.2 VAE: spatial 16x, temporal 4x, z_dim=48.

    480x640 / 17 frames → (1, 48, 5, 30, 40)
    """
    pipe.load_models_to_device(["vae"])
    video_tensor = pipe.preprocess_video(frames)   # (1, 3, T, H, W), bf16, [-1, 1]
    # vae.encode accepts list of (C,T,H,W) or tensor with batch dim; iterates over batch
    latent = pipe.vae.encode(video_tensor, device=pipe.device)
    return latent.to(dtype=torch.bfloat16)


def main():
    ap = argparse.ArgumentParser(
        description="Encode pair videos + prompt to .pth cache for Mitty training")
    ap.add_argument("--pair-dir", required=True,
                    help="pair split dir with video/ + control_video/ + metadata.csv")
    ap.add_argument("--output", required=True, help="output cache dir")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--resume", action="store_true",
                    help="skip pairs whose .pth already exists")
    ap.add_argument("--no-frames", action="store_true",
                    help="skip saving PIL frames to reduce file size (~55MB→~9MB)")
    args = ap.parse_args()

    # Resolve relative paths against MAIN_ROOT
    if not os.path.isabs(args.pair_dir):
        args.pair_dir = os.path.join(MAIN_ROOT, args.pair_dir)
    if not os.path.isabs(args.output):
        args.output = os.path.join(MAIN_ROOT, args.output)

    pair_dir = Path(args.pair_dir).resolve()
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = pair_dir / "metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"{metadata_path} not found")

    # Load source_map from sec dir (parent of pair_dir)
    source_map_path = pair_dir.parent / "source_map.json"
    source_map = json.load(open(source_map_path)) if source_map_path.exists() else {}
    split_name = pair_dir.name

    rows = list(csv.DictReader(open(metadata_path)))
    print(f"Pairs:    {len(rows)} from {pair_dir}")
    print(f"Output:   {output_dir}")
    print(f"Device:   {args.device}")

    pipe = build_pipe(args.device)
    print("Pipeline loaded (T5 + VAE)")

    # Pre-encode unique prompts (all pairs share the same prompt in current dataset)
    prompts = sorted(set(r["prompt"] for r in rows))
    print(f"Prompts:  {len(prompts)} unique")
    t0 = time.time()
    prompt_emb_cache = {p: encode_prompt(pipe, p).cpu() for p in prompts}
    nega_emb = encode_prompt(pipe, NEGATIVE_PROMPT).cpu()
    print(f"T5 encode done in {time.time() - t0:.1f}s")

    skipped = 0
    for idx, row in enumerate(rows):
        name = Path(row["video"]).stem  # pair_NNNN
        out_path = output_dir / f"{name}.pth"
        if args.resume and out_path.exists():
            skipped += 1
            continue

        t0 = time.time()
        human_path = pair_dir / row["control_video"]
        robot_path = pair_dir / row["video"]

        human_frames = load_video_as_pil(str(human_path))
        robot_frames = load_video_as_pil(str(robot_path))

        human_lat = encode_video(pipe, human_frames).cpu()
        robot_lat = encode_video(pipe, robot_frames).cpu()

        source_key = f"{split_name}/{Path(row['video']).name}"
        source_id = source_map.get(source_key, {}).get("source_id", "")

        data = {
            "human_latent": human_lat,
            "robot_latent": robot_lat,
            "context_posi": prompt_emb_cache[row["prompt"]],
            "context_nega": nega_emb,
            "prompt": row["prompt"],
            "source_id": source_id,
        }
        if not args.no_frames:
            data["human_frames"] = human_frames
            data["robot_frames"] = robot_frames
        torch.save(data, str(out_path))

        dt = time.time() - t0
        if (idx + 1) % 10 == 0 or idx == len(rows) - 1:
            print(f"  [{idx+1}/{len(rows)}] {name} shapes "
                  f"h={tuple(human_lat.shape)} r={tuple(robot_lat.shape)} "
                  f"ctx={tuple(data['context_posi'].shape)} ({dt:.1f}s)")

    print(f"\nDone. {len(rows) - skipped} encoded, {skipped} skipped → {output_dir}")


if __name__ == "__main__":
    main()
