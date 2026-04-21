"""
Mitty cache: encode (human_video, robot_video, prompt) → separate VAE + T5 caches.

VAE cache (per sample):
    Output: training_data/cache/vae/<dataset>/<split>/pair_NNNN.pth
    Keys:   human_latent, robot_latent, prompt, source_id

T5 cache (per unique prompt, shared across all datasets):
    Output: training_data/cache/t5/prompt_NNN.pth  (positive)
            training_data/cache/t5/negative.pth
    Keys:   embedding (1, 512, 4096), prompt (str)

Usage:
    python -m src.pipeline.mitty_cache \\
        --pair-dir training_data/pair/1s/train \\
        --output   training_data/cache/vae/pair_1s/train \\
        --device   cuda:0

    # T5 cache is written to --t5-cache-dir (default: training_data/cache/t5/)
    # and only encoded once per unique prompt.

Legacy (old format with embedded T5 + frames):
    python -m src.pipeline.mitty_cache \\
        --pair-dir training_data/pair/1s/train \\
        --output   output/mitty_cache_1s/train \\
        --legacy --device cuda:0
"""

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.stdout.reconfigure(line_buffering=True)

import av
import numpy as np
import torch
from PIL import Image

from src.core.config import MAIN_ROOT, T5_CACHE_DIR
from src.core.wan_loader import load_text_encoder, load_tokenizer, load_vae


MANUAL_DIR = os.path.join(
    "/disk_n/zzf/.cache/huggingface/hub",
    "models--Wan-AI--Wan2.2-TI2V-5B", "manual",
)
T5_PATH = os.path.join(MANUAL_DIR, "models_t5_umt5-xxl-enc-bf16.pth")
VAE_PATH = os.path.join(MANUAL_DIR, "Wan2.2_VAE.pth")
TOKENIZER_DIR = os.path.join(MANUAL_DIR, "google", "umt5-xxl")

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


def build_models(device: str, skip_t5: bool = False):
    """Load T5 + VAE directly to GPU (no DiffSynth pipeline)."""
    t5 = tokenizer = seq_len = None
    if not skip_t5:
        t5 = load_text_encoder(T5_PATH, device, torch.bfloat16)
        tokenizer, seq_len = load_tokenizer(TOKENIZER_DIR)
    vae = load_vae(VAE_PATH, torch.bfloat16, home_device=device)
    return t5, vae, tokenizer, seq_len


def _preprocess_video(frames: list[Image.Image], device: str,
                      dtype=torch.bfloat16) -> torch.Tensor:
    """PIL frames → (1, 3, T, H, W) in [-1, 1], matching DiffSynth's preprocess_video."""
    tensors = []
    for f in frames:
        t = torch.from_numpy(np.array(f, dtype=np.float32))
        t = t.to(dtype=dtype, device=device)
        t = t * (2.0 / 255.0) - 1.0
        t = t.permute(2, 0, 1)  # (C, H, W)
        tensors.append(t)
    return torch.stack(tensors, dim=1).unsqueeze(0)  # (1, C, T, H, W)


@torch.no_grad()
def encode_prompt(t5, tokenizer, seq_len: int, prompt: str,
                  device: str) -> torch.Tensor:
    """Encode prompt to T5 embedding. Returns (1, 512, 4096) bf16."""
    inputs = tokenizer(
        [prompt], return_tensors="pt",
        padding="max_length", truncation=True,
        max_length=seq_len, add_special_tokens=True,
    )
    ids = inputs.input_ids.to(device)
    mask = inputs.attention_mask.to(device)
    seq_lens = mask.gt(0).sum(dim=1).long()
    emb = t5(ids, mask)
    for i, v in enumerate(seq_lens):
        emb[:, v:] = 0
    return emb


@torch.no_grad()
def encode_video(vae, frames: list[Image.Image], device: str) -> torch.Tensor:
    """VAE-encode frames. Wan 2.2 VAE: spatial 16x, temporal 4x, z_dim=48.

    480x640 / 17 frames → (1, 48, 5, 30, 40)
    """
    video_tensor = _preprocess_video(frames, device)
    latent = vae.encode(video_tensor, device=device)
    return latent.to(dtype=torch.bfloat16)


def _prompt_filename(prompt: str) -> str:
    """Deterministic filename for a prompt: prompt_<first8chars_of_sha256>.pth"""
    h = hashlib.sha256(prompt.encode()).hexdigest()[:8]
    return f"prompt_{h}.pth"


def _save_t5_cache(t5_dir: str, prompts: list[str], neg_prompt: str,
                   t5, tokenizer, seq_len: int, device: str):
    """Encode and save T5 embeddings to shared cache dir (skip if already exist)."""
    t5_dir = Path(t5_dir)
    t5_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    for prompt in prompts:
        fname = _prompt_filename(prompt)
        out_path = t5_dir / fname
        if out_path.exists():
            print(f"  T5 cache exists: {fname}")
            continue
        emb = encode_prompt(t5, tokenizer, seq_len, prompt, device).cpu()
        torch.save({"embedding": emb, "prompt": prompt}, str(out_path))
        print(f"  T5 cache saved: {fname} shape={tuple(emb.shape)}")
        saved += 1

    neg_path = t5_dir / "negative.pth"
    if not neg_path.exists():
        emb = encode_prompt(t5, tokenizer, seq_len, neg_prompt, device).cpu()
        torch.save({"embedding": emb, "prompt": neg_prompt}, str(neg_path))
        print(f"  T5 cache saved: negative.pth shape={tuple(emb.shape)}")
        saved += 1
    else:
        print(f"  T5 cache exists: negative.pth")

    return saved


def main():
    ap = argparse.ArgumentParser(
        description="Encode pair videos + prompt to .pth cache for Mitty training")
    ap.add_argument("--pair-dir", required=True,
                    help="pair split dir with video/ + control_video/ + metadata.csv")
    ap.add_argument("--output", required=True, help="VAE output cache dir")
    ap.add_argument("--t5-cache-dir", default="",
                    help="T5 cache dir (default: training_data/cache/t5/)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--resume", action="store_true",
                    help="skip pairs whose .pth already exists")
    ap.add_argument("--legacy", action="store_true",
                    help="write old format (embedded T5 + optional PIL frames)")
    ap.add_argument("--no-frames", action="store_true",
                    help="(legacy mode only) skip saving PIL frames")
    args = ap.parse_args()

    # Resolve relative paths against MAIN_ROOT
    if not os.path.isabs(args.pair_dir):
        args.pair_dir = os.path.join(MAIN_ROOT, args.pair_dir)
    if not os.path.isabs(args.output):
        args.output = os.path.join(MAIN_ROOT, args.output)
    t5_dir = args.t5_cache_dir
    if not t5_dir:
        t5_dir = T5_CACHE_DIR
    elif not os.path.isabs(t5_dir):
        t5_dir = os.path.join(MAIN_ROOT, t5_dir)

    pair_dir = Path(args.pair_dir).resolve()
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = pair_dir / "metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"{metadata_path} not found")

    source_map_path = pair_dir.parent / "source_map.json"
    source_map = json.load(open(source_map_path)) if source_map_path.exists() else {}
    split_name = pair_dir.name

    rows = list(csv.DictReader(open(metadata_path)))
    prompts = sorted(set(r["prompt"] for r in rows))
    print(f"Pairs:    {len(rows)} from {pair_dir}")
    print(f"Output:   {output_dir}")
    print(f"T5 dir:   {t5_dir}")
    print(f"Format:   {'legacy (embedded T5)' if args.legacy else 'new (separate T5)'}")
    print(f"Prompts:  {len(prompts)} unique")
    print(f"Device:   {args.device}")

    # In new mode, T5 is only needed if cache files don't exist yet
    need_t5 = args.legacy or not all(
        (Path(t5_dir) / _prompt_filename(p)).exists() for p in prompts
    ) or not (Path(t5_dir) / "negative.pth").exists()

    t5, vae, tokenizer, seq_len = build_models(args.device, skip_t5=not need_t5)
    if need_t5:
        print(f"Models loaded: T5 + VAE on {args.device}")
    else:
        print(f"Models loaded: VAE on {args.device} (T5 cache already exists)")

    # T5 encoding
    t0 = time.time()
    if args.legacy:
        prompt_emb_cache = {
            p: encode_prompt(t5, tokenizer, seq_len, p, args.device).cpu()
            for p in prompts
        }
        nega_emb = encode_prompt(
            t5, tokenizer, seq_len, NEGATIVE_PROMPT, args.device).cpu()
        print(f"T5 encode done in {time.time() - t0:.1f}s (in-memory, legacy)")
    else:
        if need_t5:
            _save_t5_cache(t5_dir, prompts, NEGATIVE_PROMPT,
                           t5, tokenizer, seq_len, args.device)
            print(f"T5 cache saved in {time.time() - t0:.1f}s → {t5_dir}")
            # Free T5 memory — only VAE needed from here
            del t5, tokenizer
            torch.cuda.empty_cache()

    # VAE encoding
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

        human_lat = encode_video(vae, human_frames, args.device).cpu()
        robot_lat = encode_video(vae, robot_frames, args.device).cpu()

        source_key = f"{split_name}/{Path(row['video']).name}"
        source_id = source_map.get(source_key, {}).get("source_id", "")

        if args.legacy:
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
        else:
            data = {
                "human_latent": human_lat,
                "robot_latent": robot_lat,
                "prompt": row["prompt"],
                "source_id": source_id,
            }
        torch.save(data, str(out_path))

        dt = time.time() - t0
        if (idx + 1) % 10 == 0 or idx == len(rows) - 1:
            print(f"  [{idx+1}/{len(rows)}] {name} shapes "
                  f"h={tuple(human_lat.shape)} r={tuple(robot_lat.shape)} ({dt:.1f}s)")

    print(f"\nDone. {len(rows) - skipped} encoded, {skipped} skipped → {output_dir}")


if __name__ == "__main__":
    main()
