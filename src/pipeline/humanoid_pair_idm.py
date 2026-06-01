"""Humanoid Everyday H1 two-frame RGB inverse dynamics training.

This module reads the Humanoid Everyday LeRobot layout directly:

  - ``data/chunk-*/episode_*.parquet`` stores the 26-dim H1 ``action`` label.
  - ``videos/chunk-*/egocentric/episode_*.mp4`` stores the RGB frames.

Each sample uses RGB frames ``(frame_t, frame_{t+d})`` and predicts the mean
action over the half-open interval ``action[t:t+d]``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.core.config import MAIN_ROOT
from src.pipeline.wan_pair_idm import (
    action_regression_metrics,
    SmallPairCnn,
    count_trainable_parameters,
    mse_np,
    parse_resize,
    read_video_pair_frames,
    tensor_float,
    write_json,
    write_rows,
)

HUMANOID_H1_ACTION_DIM = 26


@dataclass(frozen=True)
class HumanoidPairSample:
    video_path: str
    parquet_path: str
    episode: int
    chunk: int
    rel_frame_t: int
    rel_frame_tpd: int
    action_target: tuple[float, ...]


def discover_humanoid_pairs(
    data_root: Path,
    *,
    max_samples: int,
    seed: int,
    action_dim: int,
    frame_stride: int,
    frame_delta: int,
    max_pairs_per_episode: int,
) -> list[HumanoidPairSample]:
    if frame_stride <= 0:
        raise ValueError(f"frame_stride must be positive, got {frame_stride}")
    if frame_delta <= 0:
        raise ValueError(f"frame_delta must be positive, got {frame_delta}")
    if max_pairs_per_episode < 0:
        raise ValueError(
            f"max_pairs_per_episode must be non-negative, got {max_pairs_per_episode}"
        )
    data_dir = data_root / "data"
    video_dir = data_root / "videos"
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Humanoid LeRobot data directory not found: {data_dir}")
    if not video_dir.is_dir():
        raise FileNotFoundError(f"Humanoid LeRobot video directory not found: {video_dir}")

    samples: list[HumanoidPairSample] = []
    parquet_paths = sorted(data_dir.glob("chunk-*/episode_*.parquet"))
    early_stop = max_samples > 0 and max_pairs_per_episode > 0
    if early_stop:
        random.Random(seed).shuffle(parquet_paths)
    for parquet_path in parquet_paths:
        chunk_name = parquet_path.parent.name
        if not chunk_name.startswith("chunk-"):
            raise ValueError(f"Unexpected chunk directory name: {parquet_path.parent}")
        chunk = int(chunk_name.removeprefix("chunk-"))
        episode = int(parquet_path.stem.removeprefix("episode_"))
        video_path = (
            video_dir
            / f"chunk-{chunk:03d}"
            / "egocentric"
            / f"episode_{episode:06d}.mp4"
        )
        if not video_path.is_file():
            raise FileNotFoundError(f"Humanoid egocentric video not found: {video_path}")

        df = pd.read_parquet(parquet_path, columns=["action", "frame_index", "next.done"])
        if df.empty:
            raise ValueError(f"Humanoid episode parquet is empty: {parquet_path}")
        frame_indices = df["frame_index"].to_numpy(dtype=np.int64)
        if not np.array_equal(frame_indices, np.arange(len(df), dtype=np.int64)):
            raise ValueError(
                f"Humanoid frame_index must be contiguous 0..N-1 for video alignment: "
                f"{parquet_path}"
            )

        episode_pair_count = 0
        for row_idx in range(0, len(df) - frame_delta, frame_stride):
            window = df.iloc[row_idx:row_idx + frame_delta]
            if bool(window["next.done"].any()):
                continue
            action_window = [
                np.asarray(value, dtype=np.float32) for value in window["action"].tolist()
            ]
            for offset, action in enumerate(action_window):
                if action.shape != (action_dim,):
                    raise ValueError(
                        f"Bad Humanoid action shape at episode={episode} frame={row_idx + offset}: "
                        f"{action.shape}, expected {(action_dim,)}"
                    )
            action = np.stack(action_window, axis=0).mean(axis=0)
            samples.append(
                HumanoidPairSample(
                    video_path=str(video_path),
                    parquet_path=str(parquet_path),
                    episode=episode,
                    chunk=chunk,
                    rel_frame_t=row_idx,
                    rel_frame_tpd=row_idx + frame_delta,
                    action_target=tuple(float(v) for v in action.tolist()),
                )
            )
            episode_pair_count += 1
            if early_stop and len(samples) >= max_samples:
                break
            if max_pairs_per_episode > 0 and episode_pair_count >= max_pairs_per_episode:
                break
        if early_stop and len(samples) >= max_samples:
            break

    random.Random(seed).shuffle(samples)
    if max_samples > 0:
        samples = samples[:max_samples]
    if not samples:
        raise ValueError(f"No Humanoid interval frame pairs discovered under {data_root}")
    return samples


def split_humanoid_samples(
    samples: list[HumanoidPairSample],
    train_ratio: float,
    split_by: str,
    seed: int,
    *,
    train_samples_count: int = 0,
    val_samples_count: int = 0,
) -> tuple[list[HumanoidPairSample], list[HumanoidPairSample]]:
    if split_by not in {"sample", "episode"}:
        raise ValueError(f"Unsupported split_by={split_by!r}")
    if train_samples_count > 0 or val_samples_count > 0:
        if train_samples_count <= 0 or val_samples_count <= 0:
            raise ValueError(
                "train_samples_count and val_samples_count must be both positive "
                "when explicit sample counts are used"
            )
        if split_by == "sample":
            total = train_samples_count + val_samples_count
            if total > len(samples):
                raise ValueError(
                    f"Requested train/val counts exceed available samples: "
                    f"{total} > {len(samples)}"
                )
            return (
                samples[:train_samples_count],
                samples[train_samples_count:train_samples_count + val_samples_count],
            )
        episode_groups: dict[int, list[HumanoidPairSample]] = {}
        for sample in samples:
            episode_groups.setdefault(sample.episode, []).append(sample)
        ordered_episodes = sorted(episode_groups)
        random.Random(seed).shuffle(ordered_episodes)
        train_samples: list[HumanoidPairSample] = []
        val_samples: list[HumanoidPairSample] = []
        train_episode_boundary = len(ordered_episodes)
        for episode_idx, episode in enumerate(ordered_episodes):
            episode_samples = episode_groups[episode]
            if len(train_samples) < train_samples_count:
                remaining = train_samples_count - len(train_samples)
                take = min(remaining, len(episode_samples))
                train_samples.extend(episode_samples[:take])
                if take < len(episode_samples):
                    train_episode_boundary = episode_idx + 1
                    break
                train_episode_boundary = episode_idx + 1
                continue
            break
        if len(train_samples) < train_samples_count:
            raise ValueError(
                f"Requested train sample count exceeds available samples: "
                f"{train_samples_count} > {len(train_samples)}"
            )
        for episode in ordered_episodes[train_episode_boundary:]:
            episode_samples = episode_groups[episode]
            if len(val_samples) < val_samples_count:
                remaining = val_samples_count - len(val_samples)
                take = min(remaining, len(episode_samples))
                val_samples.extend(episode_samples[:take])
                if take < len(episode_samples):
                    break
                continue
            break
        if len(val_samples) < val_samples_count:
            raise ValueError(
                f"Requested eval sample count exceeds available samples: "
                f"{val_samples_count} > {len(val_samples)}"
            )
        if not train_samples or not val_samples:
            raise ValueError(
                f"Invalid episode split with explicit counts: train={len(train_samples)} "
                f"val={len(val_samples)}"
            )
        return train_samples, val_samples

    if not 0.0 < train_ratio < 1.0:
        raise ValueError(f"train_ratio must be in (0,1), got {train_ratio}")
    if split_by == "sample":
        n_train = max(1, min(len(samples) - 1, int(round(len(samples) * train_ratio))))
        return samples[:n_train], samples[n_train:]
    if split_by != "episode":
        raise ValueError(f"Unsupported split_by={split_by!r}")
    episodes = sorted({sample.episode for sample in samples})
    if len(episodes) < 2:
        raise ValueError("Episode split requires at least two episodes")
    random.Random(seed).shuffle(episodes)
    n_train_eps = max(1, min(len(episodes) - 1, int(round(len(episodes) * train_ratio))))
    train_eps = set(episodes[:n_train_eps])
    train_samples = [sample for sample in samples if sample.episode in train_eps]
    val_samples = [sample for sample in samples if sample.episode not in train_eps]
    if not train_samples or not val_samples:
        raise ValueError(
            f"Invalid episode split: train={len(train_samples)} val={len(val_samples)}"
        )
    return train_samples, val_samples


def apply_humanoid_checkpoint_config(args: argparse.Namespace, ckpt: dict) -> None:
    config = ckpt.get("config")
    if not isinstance(config, dict):
        raise ValueError("Checkpoint does not contain a config dict")
    required = {
        "data_root": config.get("data_root"),
        "resize": config.get("resize"),
        "split_by": config.get("split_by"),
        "train_ratio": config.get("train_ratio"),
        "max_samples": config.get("max_samples"),
        "seed": config.get("seed"),
        "action_dim": config.get("action_dim"),
        "frame_stride": config.get("frame_stride"),
        "frame_delta": config.get("frame_delta"),
        "max_pairs_per_episode": config.get("max_pairs_per_episode"),
        "train_samples": config.get("train_samples"),
        "eval_samples": config.get("eval_samples"),
    }
    missing = [key for key, value in required.items() if value is None]
    if missing and not bool(getattr(args, "allow_cli_split", False)):
        raise ValueError(
            "Checkpoint is missing humanoid validation config fields "
            f"{missing}. Re-train with the updated humanoid pair IDM code, or "
            "pass --allow-cli-split to explicitly use the CLI split arguments "
            "for this legacy checkpoint."
        )
    for key, value in required.items():
        if value is not None:
            setattr(args, key, value)


class HumanoidPairDataset(torch.utils.data.Dataset):
    def __init__(self, samples: list[HumanoidPairSample], resize: tuple[int, int] | None):
        self.samples = samples
        self.resize = resize

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        frames = read_video_pair_frames(
            sample.video_path,
            sample.rel_frame_t,
            self.resize,
            sample.rel_frame_tpd,
        )
        return {
            "frames": frames,
            "action_target": np.asarray(sample.action_target, dtype=np.float32),
            "sample_index": idx,
        }


def collate_humanoid_pair_batch(items: list[dict]) -> dict:
    frames_np = np.stack([item["frames"] for item in items], axis=0)
    frames = torch.from_numpy(frames_np).to(dtype=torch.float32)
    frames = frames.permute(0, 1, 4, 2, 3).contiguous()
    b, t, c, h, w = frames.shape
    if t != 2 or c != 3:
        raise ValueError(f"Expected pair frames [B,2,3,H,W], got {frames.shape}")
    frames = frames.view(b, t * c, h, w).mul_(2.0 / 255.0).sub_(1.0)
    return {
        "frames": frames,
        "action_target": torch.from_numpy(
            np.stack([item["action_target"] for item in items], axis=0)
        ),
        "sample_index": [item["sample_index"] for item in items],
    }


def build_1d_sincos_embedding(positions: torch.Tensor, dim: int) -> torch.Tensor:
    if dim % 2 != 0:
        raise ValueError(f"sincos embedding dim must be even, got {dim}")
    if positions.ndim != 1:
        raise ValueError(f"positions must be 1D, got {positions.shape}")
    half = dim // 2
    omega = torch.arange(half, device=positions.device, dtype=torch.float32)
    omega = 1.0 / (10000 ** (omega / max(half, 1)))
    angles = positions.to(dtype=torch.float32)[:, None] * omega[None, :]
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)


def build_2d_sincos_embedding(grid_h: int, grid_w: int, dim: int, device: torch.device) -> torch.Tensor:
    if dim % 4 != 0:
        raise ValueError(f"2D sincos embedding dim must be divisible by 4, got {dim}")
    ys, xs = torch.meshgrid(
        torch.arange(grid_h, device=device, dtype=torch.float32),
        torch.arange(grid_w, device=device, dtype=torch.float32),
        indexing="ij",
    )
    y_embed = build_1d_sincos_embedding(ys.reshape(-1), dim // 2)
    x_embed = build_1d_sincos_embedding(xs.reshape(-1), dim // 2)
    return torch.cat([y_embed, x_embed], dim=1)


class HumanoidPairTransformer(nn.Module):
    def __init__(
        self,
        action_dim: int,
        *,
        patch_size: int = 32,
        embed_dim: int = 256,
        depth: int = 4,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        attn_dropout: float = 0.0,
    ):
        super().__init__()
        if action_dim <= 0:
            raise ValueError(f"action_dim must be positive, got {action_dim}")
        if patch_size <= 0:
            raise ValueError(f"patch_size must be positive, got {patch_size}")
        if embed_dim <= 0:
            raise ValueError(f"embed_dim must be positive, got {embed_dim}")
        if depth <= 0:
            raise ValueError(f"depth must be positive, got {depth}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim must be divisible by num_heads, got {embed_dim} / {num_heads}"
            )
        if mlp_ratio <= 0.0:
            raise ValueError(f"mlp_ratio must be positive, got {mlp_ratio}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if dropout < 0.0:
            raise ValueError(f"dropout must be non-negative, got {dropout}")
        if attn_dropout < 0.0:
            raise ValueError(f"attn_dropout must be non-negative, got {attn_dropout}")
        self.action_dim = int(action_dim)
        self.patch_size = int(patch_size)
        self.embed_dim = int(embed_dim)
        self.depth = int(depth)
        self.num_heads = int(num_heads)
        self.mlp_ratio = float(mlp_ratio)
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)
        self.attn_dropout = float(attn_dropout)

        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.frame_embed = nn.Parameter(torch.zeros(1, 2, 1, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=max(dropout, attn_dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=depth,
            enable_nested_tensor=False,
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, action_dim),
        )
        self.dropout_layer = nn.Dropout(dropout)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.frame_embed, std=0.02)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.ndim != 4 or frames.shape[1] != 6:
            raise ValueError(f"Expected [B,6,H,W] frame pairs, got {frames.shape}")
        b, _, h, w = frames.shape
        if h % self.patch_size != 0 or w % self.patch_size != 0:
            raise ValueError(
                f"Input spatial size {h}x{w} must be divisible by patch_size={self.patch_size}"
            )
        pair = frames.float().view(b, 2, 3, h, w)
        x = pair.reshape(b * 2, 3, h, w)
        x = self.patch_embed(x)
        ph, pw = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2).contiguous()
        x = x.view(b, 2, ph * pw, self.embed_dim)
        pos = build_2d_sincos_embedding(ph, pw, self.embed_dim, x.device).to(dtype=x.dtype)
        x = x + pos.unsqueeze(0).unsqueeze(0)
        x = x + self.frame_embed[:, :2, :, :].to(dtype=x.dtype, device=x.device)
        x = x.view(b, 2 * ph * pw, self.embed_dim)
        cls = self.cls_token.to(dtype=x.dtype, device=x.device).expand(b, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.dropout_layer(x)
        x = self.encoder(x)
        x = self.norm(x[:, 0])
        return self.head(x)


class HumanoidPairMotionTransformer(nn.Module):
    def __init__(
        self,
        action_dim: int,
        *,
        patch_size: int = 32,
        embed_dim: int = 256,
        depth: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        hidden_dim: int = 256,
        dropout: float = 0.05,
        attn_dropout: float = 0.0,
    ):
        super().__init__()
        if action_dim <= 0:
            raise ValueError(f"action_dim must be positive, got {action_dim}")
        if patch_size <= 0:
            raise ValueError(f"patch_size must be positive, got {patch_size}")
        if embed_dim <= 0:
            raise ValueError(f"embed_dim must be positive, got {embed_dim}")
        if depth <= 0:
            raise ValueError(f"depth must be positive, got {depth}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim must be divisible by num_heads, got {embed_dim} / {num_heads}"
            )
        if mlp_ratio <= 0.0:
            raise ValueError(f"mlp_ratio must be positive, got {mlp_ratio}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if dropout < 0.0:
            raise ValueError(f"dropout must be non-negative, got {dropout}")
        if attn_dropout < 0.0:
            raise ValueError(f"attn_dropout must be non-negative, got {attn_dropout}")
        self.action_dim = int(action_dim)
        self.patch_size = int(patch_size)
        self.embed_dim = int(embed_dim)
        self.depth = int(depth)
        self.num_heads = int(num_heads)
        self.mlp_ratio = float(mlp_ratio)
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)
        self.attn_dropout = float(attn_dropout)

        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.motion_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.frame_embed = nn.Parameter(torch.zeros(1, 2, 1, embed_dim))
        self.motion_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.motion_proj = nn.Sequential(
            nn.LayerNorm(embed_dim * 4),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=max(dropout, attn_dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=depth,
            enable_nested_tensor=False,
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.readout_norm = nn.LayerNorm(embed_dim * 5)
        self.head = nn.Sequential(
            nn.Linear(embed_dim * 5, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, action_dim),
        )
        self.dropout_layer = nn.Dropout(dropout)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.motion_token, std=0.02)
        nn.init.trunc_normal_(self.frame_embed, std=0.02)
        nn.init.trunc_normal_(self.motion_embed, std=0.02)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.ndim != 4 or frames.shape[1] != 6:
            raise ValueError(f"Expected [B,6,H,W] frame pairs, got {frames.shape}")
        b, _, h, w = frames.shape
        if h % self.patch_size != 0 or w % self.patch_size != 0:
            raise ValueError(
                f"Input spatial size {h}x{w} must be divisible by patch_size={self.patch_size}"
            )
        pair = frames.float().view(b, 2, 3, h, w)
        x = pair.reshape(b * 2, 3, h, w)
        x = self.patch_embed(x)
        ph, pw = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2).contiguous()
        x = x.view(b, 2, ph * pw, self.embed_dim)

        x0 = x[:, 0]
        x1 = x[:, 1]
        motion = self.motion_proj(torch.cat([x0, x1, x1 - x0, torch.abs(x1 - x0)], dim=-1))
        pos = build_2d_sincos_embedding(ph, pw, self.embed_dim, x.device).to(dtype=x.dtype)
        frame_tokens = x + pos.unsqueeze(0).unsqueeze(0)
        frame_tokens = frame_tokens + self.frame_embed[:, :2, :, :].to(
            dtype=x.dtype,
            device=x.device,
        )
        motion_tokens = motion + pos.unsqueeze(0)
        motion_tokens = motion_tokens + self.motion_embed.to(dtype=x.dtype, device=x.device)

        cls = self.cls_token.to(dtype=x.dtype, device=x.device).expand(b, -1, -1)
        motion_cls = self.motion_token.to(dtype=x.dtype, device=x.device).expand(b, -1, -1)
        tokens = torch.cat(
            [
                cls,
                motion_cls,
                frame_tokens[:, 0],
                frame_tokens[:, 1],
                motion_tokens,
            ],
            dim=1,
        )
        tokens = self.dropout_layer(tokens)
        tokens = self.encoder(tokens)
        tokens = self.norm(tokens)

        n = ph * pw
        cls_out = tokens[:, 0]
        motion_cls_out = tokens[:, 1]
        frame0_pool = tokens[:, 2:2 + n].mean(dim=1)
        frame1_pool = tokens[:, 2 + n:2 + 2 * n].mean(dim=1)
        motion_pool = tokens[:, 2 + 2 * n:2 + 3 * n].mean(dim=1)
        readout = torch.cat(
            [cls_out, motion_cls_out, frame0_pool, frame1_pool, motion_pool],
            dim=1,
        )
        return self.head(self.readout_norm(readout))


def make_humanoid_model(args: argparse.Namespace) -> nn.Module:
    if args.model_arch == "small_cnn":
        return SmallPairCnn(
            int(args.action_dim),
            base_channels=args.base_channels,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
        )
    if args.model_arch == "transformer":
        return HumanoidPairTransformer(
            int(args.action_dim),
            patch_size=args.transformer_patch_size,
            embed_dim=args.transformer_embed_dim,
            depth=args.transformer_depth,
            num_heads=args.transformer_num_heads,
            mlp_ratio=args.transformer_mlp_ratio,
            hidden_dim=args.hidden_dim,
            dropout=args.transformer_dropout,
            attn_dropout=args.transformer_attn_dropout,
        )
    if args.model_arch == "motion_transformer":
        return HumanoidPairMotionTransformer(
            int(args.action_dim),
            patch_size=args.transformer_patch_size,
            embed_dim=args.transformer_embed_dim,
            depth=args.transformer_depth,
            num_heads=args.transformer_num_heads,
            mlp_ratio=args.transformer_mlp_ratio,
            hidden_dim=args.hidden_dim,
            dropout=args.transformer_dropout,
            attn_dropout=args.transformer_attn_dropout,
        )
    raise ValueError(f"Unsupported model_arch={args.model_arch!r}")


def humanoid_model_payload(args: argparse.Namespace) -> dict:
    if args.model_arch == "small_cnn":
        return {
            "model_arch": "small_cnn",
            "input_channels": 6,
            "base_channels": args.base_channels,
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "action_dim": args.action_dim,
            "alignment": "humanoid_frame_pair_t_to_t_plus_d_predict_mean_action_t_to_t_plus_d",
        }
    if args.model_arch == "transformer":
        return {
            "model_arch": "transformer",
            "action_dim": args.action_dim,
            "patch_size": args.transformer_patch_size,
            "embed_dim": args.transformer_embed_dim,
            "depth": args.transformer_depth,
            "num_heads": args.transformer_num_heads,
            "mlp_ratio": args.transformer_mlp_ratio,
            "hidden_dim": args.hidden_dim,
            "dropout": args.transformer_dropout,
            "attn_dropout": args.transformer_attn_dropout,
            "alignment": "humanoid_frame_pair_t_to_t_plus_d_predict_mean_action_t_to_t_plus_d",
        }
    if args.model_arch == "motion_transformer":
        return {
            "model_arch": "motion_transformer",
            "action_dim": args.action_dim,
            "patch_size": args.transformer_patch_size,
            "embed_dim": args.transformer_embed_dim,
            "depth": args.transformer_depth,
            "num_heads": args.transformer_num_heads,
            "mlp_ratio": args.transformer_mlp_ratio,
            "hidden_dim": args.hidden_dim,
            "dropout": args.transformer_dropout,
            "attn_dropout": args.transformer_attn_dropout,
            "readout": "cls_motion_cls_frame0_mean_frame1_mean_motion_mean",
            "motion_tokens": "patch_feature_pair_concat_delta_abs_delta",
            "alignment": "humanoid_frame_pair_t_to_t_plus_d_predict_mean_action_t_to_t_plus_d",
        }
    raise ValueError(f"Unsupported model_arch={args.model_arch!r}")


def humanoid_target_stats(samples: list[HumanoidPairSample]) -> tuple[torch.Tensor, torch.Tensor]:
    action_arr = np.asarray([sample.action_target for sample in samples], dtype=np.float32)
    action_mean = action_arr.mean(axis=0)
    action_std = np.maximum(action_arr.std(axis=0), 1e-4)
    return torch.from_numpy(action_mean), torch.from_numpy(action_std)


@torch.no_grad()
def validate_humanoid_samples(
    model: nn.Module,
    samples: list[HumanoidPairSample],
    resize: tuple[int, int] | None,
    action_mean: torch.Tensor,
    action_std: torch.Tensor,
    device: str,
    args: argparse.Namespace,
    *,
    prediction_path: Path | None,
) -> dict[str, float]:
    if not samples:
        raise ValueError("validate_humanoid_samples received no samples")
    model.eval()
    subset = samples if args.val_max_samples <= 0 else samples[: args.val_max_samples]
    loader = torch.utils.data.DataLoader(
        HumanoidPairDataset(subset, resize),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate_humanoid_pair_batch,
        drop_last=False,
    )

    pred_chunks = []
    target_chunks = []
    pred_rows = []
    for batch in loader:
        frames = batch["frames"].to(device)
        target = batch["action_target"].to(device)
        pred = model(frames) * action_std + action_mean
        pred_cpu = pred.detach().cpu().numpy()
        target_cpu = target.detach().cpu().numpy()
        pred_chunks.append(pred_cpu)
        target_chunks.append(target_cpu)
        for local_idx, sample_idx in enumerate(batch["sample_index"]):
            sample = subset[int(sample_idx)]
            row = {
                "sample_index": int(sample_idx),
                "episode": sample.episode,
                "chunk": sample.chunk,
                "rel_frame_t": sample.rel_frame_t,
                "rel_frame_tpd": sample.rel_frame_tpd,
                "video_path": sample.video_path,
                "parquet_path": sample.parquet_path,
            }
            for dim in range(args.action_dim):
                row[f"action_target_{dim:02d}"] = float(target_cpu[local_idx, dim])
                row[f"action_pred_{dim:02d}"] = float(pred_cpu[local_idx, dim])
                row[f"action_err_{dim:02d}"] = float(
                    pred_cpu[local_idx, dim] - target_cpu[local_idx, dim]
                )
            pred_rows.append(row)

    model.train()
    pred_all = np.concatenate(pred_chunks, axis=0)
    target_all = np.concatenate(target_chunks, axis=0)
    action_mean_np = action_mean.detach().cpu().numpy()
    action_std_np = action_std.detach().cpu().numpy()
    baseline = action_mean_np[None, :]
    action_mse = mse_np(pred_all, target_all)
    baseline_mse = mse_np(np.broadcast_to(baseline, target_all.shape), target_all)
    total_sse = float(np.square(pred_all - target_all).sum())
    target_sse = float(np.square(target_all).sum())
    metrics = {
        "n_samples": len(subset),
        "action_mse": action_mse,
        "mean_baseline_action_mse": baseline_mse,
        "relative_l2_error": float(math.sqrt(total_sse / max(target_sse, 1e-12))),
        "pred_std_mean": float(pred_all.std(axis=0).mean()),
        "target_std_mean": float(target_all.std(axis=0).mean()),
    }
    metrics.update(
        action_regression_metrics(
            pred_all,
            target_all,
            action_mean_np,
            action_std_np,
            prefix="action",
            dim_prefix="action_dim",
        )
    )
    if prediction_path is not None:
        write_rows(pred_rows, prediction_path)
    return metrics


def save_humanoid_checkpoint(
    model: nn.Module,
    action_mean: torch.Tensor,
    action_std: torch.Tensor,
    args: argparse.Namespace,
    out_dir: Path,
    val_metrics: dict[str, float],
    *,
    filename: str = "checkpoint.pt",
) -> None:
    payload = {
        "model_state": model.state_dict(),
        "action_mean": action_mean,
        "action_std": action_std,
        "model": humanoid_model_payload(args),
        "config": {
            "data_root": args.data_root,
            "resize": args.resize,
            "max_samples": args.max_samples,
            "seed": args.seed,
            "action_dim": args.action_dim,
            "model_arch": args.model_arch,
            "split_by": args.split_by,
            "train_ratio": args.train_ratio,
            "train_samples": args.train_samples,
            "eval_samples": args.eval_samples,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "adam_beta1": args.adam_beta1,
            "adam_beta2": args.adam_beta2,
            "lr_scheduler": args.lr_scheduler,
            "min_lr_ratio": args.min_lr_ratio,
            "lr_warmup_steps": args.lr_warmup_steps,
            "lr_warmup_steps_effective": int(
                getattr(args, "effective_lr_warmup_steps", args.lr_warmup_steps)
            ),
            "lr_warmup_ratio": args.lr_warmup_ratio,
            "dataset": "humanoid_everyday_lerobot",
            "robot_type": "h1",
            "frame_stride": args.frame_stride,
            "frame_delta": args.frame_delta,
            "max_pairs_per_episode": args.max_pairs_per_episode,
            "target_semantics": "mean(action[t:t+frame_delta])",
        },
        "val_metrics": val_metrics,
    }
    torch.save(payload, out_dir / filename)


def load_humanoid_pair_idm(checkpoint: Path, device: str) -> tuple[nn.Module, torch.Tensor, torch.Tensor, dict]:
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model_cfg = ckpt["model"]
    model_arch = str(model_cfg.get("model_arch", "small_cnn"))
    if model_arch == "small_cnn":
        model = SmallPairCnn(
            int(model_cfg["action_dim"]),
            base_channels=int(model_cfg["base_channels"]),
            hidden_dim=int(model_cfg["hidden_dim"]),
            dropout=float(model_cfg["dropout"]),
        ).to(device)
    elif model_arch == "transformer":
        model = HumanoidPairTransformer(
            int(model_cfg["action_dim"]),
            patch_size=int(model_cfg["patch_size"]),
            embed_dim=int(model_cfg["embed_dim"]),
            depth=int(model_cfg["depth"]),
            num_heads=int(model_cfg["num_heads"]),
            mlp_ratio=float(model_cfg["mlp_ratio"]),
            hidden_dim=int(model_cfg["hidden_dim"]),
            dropout=float(model_cfg["dropout"]),
            attn_dropout=float(model_cfg["attn_dropout"]),
        ).to(device)
    elif model_arch == "motion_transformer":
        model = HumanoidPairMotionTransformer(
            int(model_cfg["action_dim"]),
            patch_size=int(model_cfg["patch_size"]),
            embed_dim=int(model_cfg["embed_dim"]),
            depth=int(model_cfg["depth"]),
            num_heads=int(model_cfg["num_heads"]),
            mlp_ratio=float(model_cfg["mlp_ratio"]),
            hidden_dim=int(model_cfg["hidden_dim"]),
            dropout=float(model_cfg["dropout"]),
            attn_dropout=float(model_cfg["attn_dropout"]),
        ).to(device)
    else:
        raise ValueError(f"Unsupported checkpoint model_arch={model_arch!r}: {checkpoint}")
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()
    return model, ckpt["action_mean"].to(device), ckpt["action_std"].to(device), ckpt


def plot_humanoid_loss_curves(out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    train_path = out_dir / "train_loss.csv"
    eval_path = out_dir / "eval_loss.csv"
    if not train_path.is_file():
        raise FileNotFoundError(f"Training loss CSV not found: {train_path}")
    if not eval_path.is_file():
        raise FileNotFoundError(f"Eval loss CSV not found: {eval_path}")
    train_rows = [
        {key: float(value) for key, value in row.items()}
        for row in csv.DictReader(train_path.open())
    ]
    eval_rows = [
        {key: float(value) for key, value in row.items()}
        for row in csv.DictReader(eval_path.open())
    ]
    if not train_rows or not eval_rows:
        raise ValueError(f"Empty Humanoid loss CSV under {out_dir}")
    train_steps = np.asarray([row["step"] for row in train_rows], dtype=np.float64)
    eval_steps = np.asarray([row["step"] for row in eval_rows], dtype=np.float64)

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), dpi=160, sharex=False)
    axes[0].plot(
        train_steps,
        np.asarray([row["loss"] for row in train_rows], dtype=np.float64),
        color="#1f77b4",
        linewidth=1.2,
        label="train normalized action MSE",
    )
    axes[0].set_ylabel("normalized train MSE")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(frameon=False)

    action_mse = np.asarray([row["action_mse"] for row in eval_rows], dtype=np.float64)
    baseline = np.asarray(
        [row["mean_baseline_action_mse"] for row in eval_rows],
        dtype=np.float64,
    )
    axes[1].plot(eval_steps, action_mse, marker="o", color="#1f77b4", label="eval action MSE")
    axes[1].plot(eval_steps, baseline, color="#555555", linestyle="--", label="mean baseline")
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("action MSE")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(frameon=False)
    fig.suptitle("Humanoid Everyday H1 two-frame pair IDM")
    fig.tight_layout()
    fig.savefig(out_dir / "loss_curve.png")
    plt.close(fig)


def prepare_humanoid_samples(args: argparse.Namespace) -> list[HumanoidPairSample]:
    if args.data_root is None:
        raise ValueError("--data-root is required unless it is replayed from checkpoint config")
    return discover_humanoid_pairs(
        Path(args.data_root),
        max_samples=args.max_samples,
        seed=int(args.seed),
        action_dim=int(args.action_dim),
        frame_stride=int(args.frame_stride),
        frame_delta=int(args.frame_delta),
        max_pairs_per_episode=int(args.max_pairs_per_episode),
    )


def effective_lr_warmup_steps(args: argparse.Namespace) -> int:
    if args.steps <= 0:
        raise ValueError(f"steps must be positive, got {args.steps}")
    if args.lr_warmup_steps < 0:
        raise ValueError(f"lr_warmup_steps must be non-negative, got {args.lr_warmup_steps}")
    if args.lr_warmup_ratio < 0.0:
        raise ValueError(f"lr_warmup_ratio must be non-negative, got {args.lr_warmup_ratio}")
    warmup_steps = int(args.lr_warmup_steps)
    if warmup_steps == 0 and args.lr_warmup_ratio > 0.0:
        warmup_steps = int(round(args.steps * args.lr_warmup_ratio))
    if warmup_steps > args.steps:
        raise ValueError(
            f"lr warmup steps must not exceed total steps, got {warmup_steps} > {args.steps}"
        )
    return warmup_steps


def build_humanoid_optimizer(model: nn.Module, args: argparse.Namespace) -> torch.optim.Optimizer:
    if args.lr <= 0.0:
        raise ValueError(f"lr must be positive, got {args.lr}")
    if args.weight_decay < 0.0:
        raise ValueError(f"weight_decay must be non-negative, got {args.weight_decay}")
    if not 0.0 < args.adam_beta1 < 1.0:
        raise ValueError(f"adam_beta1 must be in (0,1), got {args.adam_beta1}")
    if not 0.0 < args.adam_beta2 < 1.0:
        raise ValueError(f"adam_beta2 must be in (0,1), got {args.adam_beta2}")
    if args.adam_beta1 >= args.adam_beta2:
        raise ValueError(
            f"adam_beta1 should be smaller than adam_beta2, got "
            f"{args.adam_beta1} >= {args.adam_beta2}"
        )
    return torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.weight_decay,
    )


def build_humanoid_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
):
    if not 0.0 <= args.min_lr_ratio <= 1.0:
        raise ValueError(f"min_lr_ratio must be in [0,1], got {args.min_lr_ratio}")
    warmup_steps = effective_lr_warmup_steps(args)
    args.effective_lr_warmup_steps = warmup_steps
    if args.lr_scheduler == "none" and warmup_steps == 0:
        return None
    if args.lr_scheduler not in {"none", "cosine"}:
        raise ValueError(f"Unsupported lr_scheduler={args.lr_scheduler!r}")

    def lr_lambda(step_index: int) -> float:
        if warmup_steps > 0 and step_index < warmup_steps:
            return float(step_index + 1) / float(warmup_steps)
        if args.lr_scheduler == "none":
            return 1.0
        decay_steps = max(1, args.steps - warmup_steps)
        progress = min(1.0, max(0.0, float(step_index - warmup_steps) / float(decay_steps)))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return args.min_lr_ratio + (1.0 - args.min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def train_humanoid(args: argparse.Namespace) -> None:
    seed = int(args.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    resize = parse_resize(args.resize)
    samples = prepare_humanoid_samples(args)
    train_samples, val_samples = split_humanoid_samples(
        samples,
        args.train_ratio,
        args.split_by,
        seed,
        train_samples_count=int(args.train_samples),
        val_samples_count=int(args.eval_samples),
    )
    action_mean, action_std = humanoid_target_stats(train_samples)
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = str(
            Path(MAIN_ROOT) / "output" / "humanoid_pair_idm" / "humanoid_everyday_h1"
        )
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.write_samples_json:
        write_json(out_dir / "samples.json", [asdict(sample) for sample in samples])
        write_json(out_dir / "train_samples.json", [asdict(sample) for sample in train_samples])
        write_json(out_dir / "val_samples.json", [asdict(sample) for sample in val_samples])

    print(
        f"alignment=(frame_t,frame_t+{args.frame_delta})->mean(action_t:t+{args.frame_delta}) "
        f"dataset=humanoid_everyday_h1 "
        f"model_arch={args.model_arch} "
        f"split_by={args.split_by} train_samples={len(train_samples)} "
        f"val_samples={len(val_samples)} train_episodes={len({s.episode for s in train_samples})} "
        f"val_episodes={len({s.episode for s in val_samples})}",
        flush=True,
    )

    device = args.device
    model = make_humanoid_model(args).to(device)
    print(
        f"trainable_params={count_trainable_parameters(model)} action_dim={args.action_dim}",
        flush=True,
    )
    optimizer = build_humanoid_optimizer(model, args)
    scheduler = build_humanoid_lr_scheduler(optimizer, args)
    print(
        f"optimizer=AdamW lr={args.lr:g} betas=({args.adam_beta1:g},{args.adam_beta2:g}) "
        f"weight_decay={args.weight_decay:g} lr_scheduler={args.lr_scheduler} "
        f"warmup_steps={args.effective_lr_warmup_steps} min_lr_ratio={args.min_lr_ratio:g}",
        flush=True,
    )

    train_loader = torch.utils.data.DataLoader(
        HumanoidPairDataset(train_samples, resize),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate_humanoid_pair_batch,
        drop_last=False,
    )
    action_mean_dev = action_mean.to(device)
    action_std_dev = action_std.to(device)
    history = []
    eval_history = []
    step = 0
    last_eval_step: int | None = None
    best_eval_action = float("inf")

    def run_eval(eval_step: int, *, write_predictions: bool) -> dict[str, float]:
        nonlocal best_eval_action, last_eval_step
        prediction_path = out_dir / "val_predictions.csv" if write_predictions else None
        metrics = validate_humanoid_samples(
            model,
            val_samples,
            resize,
            action_mean_dev,
            action_std_dev,
            device,
            args,
            prediction_path=prediction_path,
        )
        row = {"step": eval_step, **metrics}
        if last_eval_step == eval_step:
            eval_history[-1] = row
        else:
            eval_history.append(row)
        last_eval_step = eval_step
        write_rows(eval_history, out_dir / "eval_loss.csv")
        if metrics["action_mse"] < best_eval_action:
            best_eval_action = metrics["action_mse"]
            save_humanoid_checkpoint(
                model,
                action_mean,
                action_std,
                args,
                out_dir,
                metrics,
                filename="best_checkpoint.pt",
            )
        print(
            f"eval_step={eval_step:04d} action_mse={metrics['action_mse']:.6f} "
            f"baseline={metrics['mean_baseline_action_mse']:.6f} "
            f"best_action_mse={best_eval_action:.6f}",
            flush=True,
        )
        return metrics

    model.train()
    if args.eval_every > 0:
        run_eval(0, write_predictions=False)
    while step < args.steps:
        for batch in train_loader:
            if step >= args.steps:
                break
            frames = batch["frames"].to(device)
            target = batch["action_target"].to(device)
            norm_target = (target - action_mean_dev) / action_std_dev
            pred = model(frames)
            loss = F.mse_loss(pred, norm_target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            step += 1
            row = {
                "step": step,
                "loss": tensor_float(loss),
                "lr": float(optimizer.param_groups[0]["lr"]),
            }
            history.append(row)
            if step == 1 or step % args.log_every == 0 or step == args.steps:
                print(f"step={step:04d} loss={row['loss']:.6f}", flush=True)
            if args.eval_every > 0 and (step % args.eval_every == 0 or step == args.steps):
                run_eval(step, write_predictions=False)

    val_metrics = run_eval(step, write_predictions=True)
    save_humanoid_checkpoint(model, action_mean, action_std, args, out_dir, val_metrics)
    best_ckpt_path = out_dir / "best_checkpoint.pt"
    if best_ckpt_path.is_file():
        best_model, best_action_mean, best_action_std, _ = load_humanoid_pair_idm(
            best_ckpt_path,
            device,
        )
        best_metrics = validate_humanoid_samples(
            best_model,
            val_samples,
            resize,
            best_action_mean,
            best_action_std,
            device,
            args,
            prediction_path=out_dir / "best_val_predictions.csv",
        )
        write_json(out_dir / "best_val_metrics.json", best_metrics)
    write_rows(history, out_dir / "train_loss.csv")
    plot_humanoid_loss_curves(out_dir)
    print(json.dumps({"val": val_metrics, "out_dir": str(out_dir)}, indent=2), flush=True)


@torch.no_grad()
def validate_humanoid_checkpoint(args: argparse.Namespace) -> None:
    device = args.device
    model, action_mean, action_std, ckpt = load_humanoid_pair_idm(
        Path(args.checkpoint),
        device,
    )
    apply_humanoid_checkpoint_config(args, ckpt)
    seed = int(args.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    resize = parse_resize(args.resize)
    samples = prepare_humanoid_samples(args)
    _, val_samples = split_humanoid_samples(
        samples,
        args.train_ratio,
        args.split_by,
        seed,
        train_samples_count=int(args.train_samples),
        val_samples_count=int(args.eval_samples),
    )
    _ = ckpt
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = validate_humanoid_samples(
        model,
        val_samples,
        resize,
        action_mean,
        action_std,
        device,
        args,
        prediction_path=out_dir / "val_predictions.csv",
    )
    write_json(out_dir / "val_metrics.json", metrics)
    print(json.dumps({"val": metrics, "out_dir": str(out_dir)}, indent=2), flush=True)


@torch.no_grad()
def eval_humanoid_pairs(args: argparse.Namespace) -> None:
    device = args.device
    model, action_mean, action_std, ckpt = load_humanoid_pair_idm(
        Path(args.checkpoint),
        device,
    )
    apply_humanoid_checkpoint_config(args, ckpt)
    seed = int(args.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    resize = parse_resize(args.resize)
    samples = prepare_humanoid_samples(args)
    _ = ckpt
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = validate_humanoid_samples(
        model,
        samples,
        resize,
        action_mean,
        action_std,
        device,
        args,
        prediction_path=out_dir / "predictions.csv",
    )
    write_json(out_dir / "metrics.json", metrics)
    print(json.dumps({"eval": metrics, "out_dir": str(out_dir)}, indent=2), flush=True)


def add_humanoid_data_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resize", default="256x256")
    parser.add_argument(
        "--model-arch",
        choices=["small_cnn", "transformer", "motion_transformer"],
        default="motion_transformer",
    )
    parser.add_argument("--max-samples", type=int, default=0,
                        help="maximum discovered interval frame pairs; 0 keeps all")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--action-dim", type=int, default=HUMANOID_H1_ACTION_DIM)
    parser.add_argument("--frame-stride", type=int, default=1,
                        help="sample every Nth candidate start frame within each episode")
    parser.add_argument("--frame-delta", type=int, default=1,
                        help="frame interval d for (frame_t, frame_t+d) and mean action[t:t+d]")
    parser.add_argument("--max-pairs-per-episode", type=int, default=0,
                        help="maximum pair samples to keep per episode; 0 keeps all")
    parser.add_argument("--transformer-patch-size", type=int, default=32)
    parser.add_argument("--transformer-embed-dim", type=int, default=256)
    parser.add_argument("--transformer-depth", type=int, default=6)
    parser.add_argument("--transformer-num-heads", type=int, default=8)
    parser.add_argument("--transformer-mlp-ratio", type=float, default=4.0)
    parser.add_argument("--transformer-dropout", type=float, default=0.05)
    parser.add_argument("--transformer-attn-dropout", type=float, default=0.0)


def add_humanoid_split_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--train-samples", type=int, default=0,
                        help="explicit train sample count; 0 uses train-ratio")
    parser.add_argument("--eval-samples", type=int, default=0,
                        help="explicit eval sample count; 0 uses train-ratio")
    parser.add_argument("--train-ratio", type=float, default=0.875)
    parser.add_argument("--split-by", choices=["sample", "episode"], default="episode")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    train_p = sub.add_parser(
        "train",
        help="train Humanoid Everyday H1 two-frame pair IDM",
    )
    add_humanoid_data_args(train_p)
    add_humanoid_split_args(train_p)
    train_p.add_argument("--output-dir", default=None)
    train_p.add_argument("--steps", type=int, default=1000)
    train_p.add_argument("--batch-size", type=int, default=16)
    train_p.add_argument("--workers", type=int, default=4)
    train_p.add_argument("--base-channels", type=int, default=32)
    train_p.add_argument("--hidden-dim", type=int, default=256)
    train_p.add_argument("--dropout", type=float, default=0.0)
    train_p.add_argument("--lr", type=float, default=3e-4)
    train_p.add_argument("--weight-decay", type=float, default=1e-2)
    train_p.add_argument("--adam-beta1", type=float, default=0.9)
    train_p.add_argument("--adam-beta2", type=float, default=0.95)
    train_p.add_argument("--lr-scheduler", choices=["none", "cosine"], default="cosine")
    train_p.add_argument("--min-lr-ratio", type=float, default=0.02)
    train_p.add_argument("--lr-warmup-steps", type=int, default=0,
                         help="explicit warmup steps; 0 derives from lr-warmup-ratio")
    train_p.add_argument("--lr-warmup-ratio", type=float, default=0.05,
                         help="fraction of total steps used for linear warmup")
    train_p.add_argument("--grad-clip-norm", type=float, default=1.0)
    train_p.add_argument("--log-every", type=int, default=50)
    train_p.add_argument("--eval-every", type=int, default=100)
    train_p.add_argument("--val-max-samples", type=int, default=0,
                         help="maximum eval samples; <=0 evaluates all")
    train_p.add_argument("--write-samples-json", action="store_true")
    train_p.set_defaults(func=train_humanoid)

    val_p = sub.add_parser(
        "validate",
        help="validate a Humanoid Everyday H1 checkpoint on held-out samples",
    )
    add_humanoid_data_args(val_p)
    add_humanoid_split_args(val_p)
    val_p.add_argument("--checkpoint", required=True)
    val_p.add_argument("--output-dir", required=True)
    val_p.add_argument("--allow-cli-split", action="store_true",
                       help="explicitly use CLI split args for legacy checkpoints missing split config")
    val_p.add_argument("--batch-size", type=int, default=16)
    val_p.add_argument("--workers", type=int, default=2)
    val_p.add_argument("--val-max-samples", type=int, default=0,
                       help="maximum eval samples; <=0 evaluates all")
    val_p.set_defaults(func=validate_humanoid_checkpoint)

    eval_p = sub.add_parser(
        "eval",
        help="evaluate a Humanoid Everyday H1 checkpoint on discovered samples",
    )
    add_humanoid_data_args(eval_p)
    eval_p.add_argument("--checkpoint", required=True)
    eval_p.add_argument("--output-dir", required=True)
    eval_p.add_argument("--allow-cli-split", action="store_true",
                       help="explicitly use CLI data args for legacy checkpoints missing config")
    eval_p.add_argument("--batch-size", type=int, default=16)
    eval_p.add_argument("--workers", type=int, default=2)
    eval_p.add_argument("--val-max-samples", type=int, default=0,
                        help="maximum eval samples; <=0 evaluates all")
    eval_p.set_defaults(func=eval_humanoid_pairs)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
