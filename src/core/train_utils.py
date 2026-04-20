"""Shared utilities for LoRA training pipelines.

Used by `src/pipeline/train_lora.py` (Wan 2.1 Fun-Control) and
`src/pipeline/train_mitty.py` (Wan 2.2 TI2V-5B Mitty in-context).

Contains:
  - DDP helpers: setup_distributed / cleanup_distributed / sync_gradients / reduce_scalar
  - Logging: setup_logging, CsvLogger
  - IO: load_cached_files / load_sample / save_lora_ckpt
  - Video IO: save_video / tensor_to_frames (+ FFMPEG constant)
"""

from __future__ import annotations

import csv
import glob
import logging
import os
import random
import subprocess
import sys
from collections.abc import Iterator
from datetime import datetime

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from safetensors.torch import save_file


FFMPEG = os.environ.get(
    "FFMPEG_BIN",
    "/home/leadtek/miniconda3/envs/flip/bin/ffmpeg",
)


# ── DDP ────────────────────────────────────────────────────────────────

def setup_distributed():
    """Init DDP if launched via torchrun. Returns (rank, world_size, device_override).

    When RANK is not set (single-GPU mode), returns (0, 1, None).
    """
    if "RANK" not in os.environ:
        return 0, 1, None
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    device = f"cuda:{local_rank}"
    torch.cuda.set_device(device)
    return rank, world_size, device


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def sync_gradients(model):
    """All-reduce trainable (LoRA) gradients across ranks."""
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)


def reduce_scalar(value, device):
    """All-reduce a scalar value across ranks and return the mean."""
    t = torch.tensor(value, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.AVG)
    return t.item()


# ── Logging ────────────────────────────────────────────────────────────

def setup_logging(log_path: str, name: str = "train") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger


class CsvLogger:
    """Per-step CSV writer. Only rank 0 should instantiate.

    Rows passed to `write()` are filtered to the declared headers; missing
    columns are written as empty strings.
    """

    def __init__(self, path: str, headers: list[str]):
        self.headers = list(headers)
        self.file = open(path, "w", newline="")
        self.writer = csv.DictWriter(self.file, fieldnames=self.headers)
        self.writer.writeheader()
        self.file.flush()

    def write(self, **fields):
        row = {h: "" for h in self.headers}
        row.update({k: v for k, v in fields.items() if k in row})
        self.writer.writerow(row)
        self.file.flush()

    def close(self):
        if not self.file.closed:
            self.file.close()


# ── Cache IO ───────────────────────────────────────────────────────────

def load_cached_files(cache_dir: str, recursive: bool = False) -> list[str]:
    """Scan cache directory for .pth files.

    `recursive=True` mirrors DiffSynth `sft:data_process` layout (cache_dir/0/*.pth);
    `recursive=False` mirrors Mitty's flat-dir layout.
    """
    if recursive:
        files = sorted(glob.glob(os.path.join(cache_dir, "**", "*.pth"), recursive=True))
    else:
        files = sorted(glob.glob(os.path.join(cache_dir, "*.pth")))
    if not files:
        raise FileNotFoundError(f"No .pth files in {cache_dir}")
    return files


def load_sample(path: str):
    return torch.load(path, map_location="cpu", weights_only=False)


def infinite_file_batches(
    files: list[str],
    batch_size: int,
    world_size: int,
    rank: int,
    seed: int = 42,
) -> Iterator[list[str]]:
    """Yield batches of file paths infinitely, reshuffling each pass.

    All ranks use the same seed so sharding is deterministic.
    Incomplete trailing batch is dropped.
    """
    pass_idx = 0
    while True:
        rng = random.Random(seed + pass_idx)
        shuffled = list(files)
        rng.shuffle(shuffled)
        my_files = shuffled[rank::world_size]
        for i in range(0, len(my_files) - batch_size + 1, batch_size):
            yield my_files[i:i + batch_size]
        pass_idx += 1


def save_lora_ckpt(model, path: str) -> int:
    """Extract LoRA params and save as safetensors. Returns number of tensors."""
    sd = model.state_dict()
    sd = model.export_trainable_state_dict(sd, remove_prefix="pipe.dit.")
    save_file({k: v.contiguous() for k, v in sd.items()}, path)
    return len(sd)


# ── Video IO ───────────────────────────────────────────────────────────

def save_video(frames: list[Image.Image], path: str, fps: int = 16):
    """Save list of PIL.Image frames as H.264 MP4 via ffmpeg pipe."""
    w, h = frames[0].size
    cmd = [
        FFMPEG, "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{w}x{h}", "-r", str(fps),
        "-i", "pipe:0",
        "-c:v", "libx264", "-crf", "18", "-preset", "fast",
        "-pix_fmt", "yuv420p", path,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for f in frames:
        proc.stdin.write(np.array(f).tobytes())
    proc.stdin.close()
    proc.wait()


def tensor_to_frames(video_tensor: torch.Tensor) -> list[Image.Image]:
    """(B, 3, T, H, W) in [-1, 1] → list of PIL.Image (first batch item)."""
    v = video_tensor[0].float().clamp(-1, 1)
    v = ((v + 1) / 2 * 255).to(torch.uint8).cpu().numpy()  # (3, T, H, W)
    v = v.transpose(1, 2, 3, 0)  # (T, H, W, 3)
    return [Image.fromarray(f) for f in v]


# ── W&B ────────────────────────────────────────────────────────────────

def _fmt_lr(lr: float) -> str:
    """Format learning rate compactly: 1e-4 → '1e-4', 3e-5 → '3e-5'."""
    s = f"{lr:.0e}"
    base, exp = s.split("e")
    return f"{base}e{int(exp)}"


def build_run_name(prefix: str, args) -> str:
    """Build a descriptive run name: {prefix}-r{rank}-lr{lr}-{timestamp}."""
    lr = getattr(args, "lr", None)
    lora_rank = getattr(args, "lora_rank", None)
    parts = [prefix]
    if lora_rank is not None:
        parts.append(f"r{lora_rank}")
    if lr is not None:
        parts.append(f"lr{_fmt_lr(lr)}")
    parts.append(datetime.now().strftime("%Y-%m-%d_%H%M%S"))
    return "-".join(parts)


def build_wandb_tags(method_tag: str, args,
                     extra_tags: list[str] | None = None) -> list[str]:
    """Build W&B tags from args: [method, r{rank}, lr{lr}, bs{N}, ...].

    Uses getattr for safe extraction across different scripts' arg namespaces.
    """
    tags = [method_tag]
    lora_rank = getattr(args, "lora_rank", None)
    if lora_rank is not None:
        tags.append(f"r{lora_rank}")
    lr = getattr(args, "lr", None)
    if lr is not None:
        tags.append(f"lr{_fmt_lr(lr)}")
    bs = getattr(args, "batch_size", None)
    if bs is not None and bs > 1:
        tags.append(f"bs{bs}")
    warmup = getattr(args, "warmup_steps", None)
    if warmup is not None and warmup > 0:
        tags.append(f"warmup{warmup}")
    max_steps = getattr(args, "max_steps", None)
    if max_steps is not None:
        tags.append(f"steps{max_steps}")
    if getattr(args, "init_lora", ""):
        tags.append("init-lora")
    if extra_tags:
        tags.extend(extra_tags)
    return tags


class WandbLogger:
    """Thin W&B wrapper. No-op when `project` is falsy (e.g. user didn't pass
    `--wandb-project`), so training scripts can unconditionally call `.log()`
    without checking a flag.

    Lazy-imports `wandb` only when enabled, so the package stays optional.
    """

    def __init__(self, project: str = None, run_name: str = None,
                 config: dict = None, tags: list[str] = None,
                 dir: str = None):
        self.enabled = bool(project)
        self._wandb = None
        if not self.enabled:
            return
        import wandb  # lazy import
        wandb.init(
            project=project,
            name=run_name,
            tags=list(tags) if tags else [],
            config=config or {},
            dir=dir,
        )
        self._wandb = wandb

    def log(self, data: dict, step: int = None):
        if self.enabled:
            self._wandb.log(data, step=step)

    def log_videos(self, prefix: str, paths: dict, step: int, fps: int = 16):
        """Upload mp4 files from `paths` ({key: filepath}) under `prefix/key`."""
        if not self.enabled:
            return
        payload = {}
        for k, p in paths.items():
            if os.path.exists(p):
                payload[f"{prefix}/{k}"] = self._wandb.Video(p, format="mp4", fps=fps)
        if payload:
            self._wandb.log(payload, step=step)

    def finish(self):
        if self.enabled:
            self._wandb.finish()


# File-kind → W&B subgroup name. Eval videos are saved as `{kind}_NN.mp4`.
_EVAL_VIDEO_GROUPS = {"gt": "GT", "ctrl": "Control", "gen": "Gen"}


def log_step_eval_videos(wb: "WandbLogger", step_dir: str, step: int,
                         split_tag: str = "",
                         section: str = "Video sample"):
    """Upload eval videos grouped by kind.

    Keys:  `{section}/GT/{split_tag}_NN`, `{section}/Control/...`, `{section}/Gen/...`
    If `split_tag` is empty the key is just the sample index `NN`.
    """
    if not wb.enabled or not os.path.isdir(step_dir):
        return
    for kind, group in _EVAL_VIDEO_GROUPS.items():
        paths = {}
        for p in sorted(glob.glob(os.path.join(step_dir, f"{kind}_*.mp4"))):
            idx = os.path.splitext(os.path.basename(p))[0].split("_", 1)[1]
            key = f"{split_tag}_{idx}" if split_tag else idx
            paths[key] = p
        wb.log_videos(f"{section}/{group}", paths, step=step)
