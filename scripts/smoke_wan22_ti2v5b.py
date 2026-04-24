"""Wan 2.2 TI2V-5B 冒烟测试。

用途（T004）：
- 验证从本地 manual/ 目录加载权重能工作
- 跑一条 T2V inference（不给 input_image，17 帧 @ 480×640，30 步）
- 跑一条 TI2V inference（给 input_image）
- 测峰值显存（在 GPU 2 或 3 上跑，避开 ComfyUI 的 GPU 0）

跑法：
    LD_PRELOAD=/home/leadtek/miniconda3/envs/flip/lib/libjpeg.so.8 \
        CUDA_VISIBLE_DEVICES=2 \
        python scripts/smoke_wan22_ti2v5b.py
"""
import argparse
import os
import time

import torch

from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from diffsynth.utils.data import save_video


MANUAL_DIR = "/disk_n/zzf/.cache/huggingface/hub/models--Wan-AI--Wan2.2-TI2V-5B/manual"


def log_vram(tag: str) -> None:
    if not torch.cuda.is_available():
        return
    alloc = torch.cuda.memory_allocated() / (1024**3)
    reserved = torch.cuda.memory_reserved() / (1024**3)
    peak = torch.cuda.max_memory_allocated() / (1024**3)
    print(f"[VRAM {tag:<18}] allocated={alloc:.2f}G reserved={reserved:.2f}G peak={peak:.2f}G")


def build_pipe(device: str, use_low_vram: bool) -> WanVideoPipeline:
    dit_files = sorted(
        f for f in os.listdir(MANUAL_DIR)
        if f.startswith("diffusion_pytorch_model-") and f.endswith(".safetensors")
    )
    dit_paths = [os.path.join(MANUAL_DIR, f) for f in dit_files]

    t5_path = os.path.join(MANUAL_DIR, "models_t5_umt5-xxl-enc-bf16.pth")
    vae_path = os.path.join(MANUAL_DIR, "Wan2.2_VAE.pth")
    tokenizer_dir = os.path.join(MANUAL_DIR, "google", "umt5-xxl")

    # 分层 VRAM 策略（对齐 T005 训练需求）：
    # - T5 文本编码器：disk offload（只 encode 一次，用完 disk 保留）
    # - VAE：CPU offload（encode/decode 时搬 GPU，占用低）
    # - DiT：常驻 GPU（核心计算，搬进搬出开销大）
    t5_kwargs = {
        "offload_dtype": "disk",
        "offload_device": "disk",
        "onload_dtype": torch.bfloat16,
        "onload_device": "cpu",
        "preparing_dtype": torch.bfloat16,
        "preparing_device": device,
        "computation_dtype": torch.bfloat16,
        "computation_device": device,
    }
    vae_kwargs = {
        "offload_dtype": torch.bfloat16,
        "offload_device": "cpu",
        "onload_dtype": torch.bfloat16,
        "onload_device": "cpu",
        "preparing_dtype": torch.bfloat16,
        "preparing_device": device,
        "computation_dtype": torch.bfloat16,
        "computation_device": device,
    }
    dit_kwargs = {
        "offload_dtype": torch.bfloat16,
        "offload_device": device,
        "onload_dtype": torch.bfloat16,
        "onload_device": device,
        "preparing_dtype": torch.bfloat16,
        "preparing_device": device,
        "computation_dtype": torch.bfloat16,
        "computation_device": device,
    }
    # 保留 use_low_vram 参数为兼容，但忽略（默认即"T5 卸载、DiT 常驻"）
    _ = use_low_vram

    total_gb = torch.cuda.mem_get_info(device)[1] / (1024**3)
    vram_limit = total_gb - 3
    print(f"vram_limit={vram_limit:.1f} GB (total {total_gb:.1f} GB)")

    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=[
            ModelConfig(path=t5_path, **t5_kwargs),
            ModelConfig(path=dit_paths, **dit_kwargs),
            ModelConfig(path=vae_path, **vae_kwargs),
        ],
        tokenizer_config=ModelConfig(path=tokenizer_dir),
        vram_limit=vram_limit,
    )
    return pipe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--num-frames", type=int, default=17)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--low-vram", action="store_true",
                    help="use disk offload (slower but avoids OOM)")
    ap.add_argument("--skip-i2v", action="store_true")
    ap.add_argument("--output-dir", default="tmp/wan22_smoke")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Building pipeline on {args.device}, low_vram={args.low_vram}...")
    t0 = time.time()
    pipe = build_pipe(args.device, args.low_vram)
    print(f"Loaded in {time.time() - t0:.1f}s")
    log_vram("after-load")

    prompt = "A first-person view robot arm performing household tasks"
    negative = (
        "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，"
        "整体发灰，最差质量，低质量，JPEG压缩残留"
    )

    # ── T2V ──
    print(f"\n[T2V] {args.width}x{args.height}, {args.num_frames} frames, {args.steps} steps...")
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    video = pipe(
        prompt=prompt,
        negative_prompt=negative,
        seed=0,
        tiled=True,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        num_inference_steps=args.steps,
    )
    dt = time.time() - t0
    log_vram("T2V-peak")
    out_t2v = os.path.join(args.output_dir, "smoke_t2v.mp4")
    save_video(video, out_t2v, fps=16, quality=5)
    print(f"T2V done in {dt:.1f}s → {out_t2v}")

    # ── TI2V ──
    if not args.skip_i2v:
        from PIL import Image
        # 复用训练数据的 1 帧作为 input_image
        human_first = "/disk_n/zzf/flip/training_data/pair/1s/control_video/pair_0000.mp4"
        import av
        c = av.open(human_first)
        img = next(c.decode(video=0)).to_image().resize((args.width, args.height))
        c.close()

        print(f"\n[TI2V] with first-frame from pair_0000 human...")
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        video = pipe(
            prompt=prompt,
            negative_prompt=negative,
            seed=0,
            tiled=True,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            num_inference_steps=args.steps,
            input_image=img,
        )
        dt = time.time() - t0
        log_vram("TI2V-peak")
        out_ti2v = os.path.join(args.output_dir, "smoke_ti2v.mp4")
        save_video(video, out_ti2v, fps=16, quality=5)
        print(f"TI2V done in {dt:.1f}s → {out_ti2v}")

    print("\nSmoke test complete.")


if __name__ == "__main__":
    main()
