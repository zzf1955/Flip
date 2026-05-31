"""Humanoid Everyday H1 two-frame RGB inverse dynamics training.

This module reads the Humanoid Everyday LeRobot layout directly:

  - ``data/chunk-*/episode_*.parquet`` stores the 26-dim H1 ``action`` label.
  - ``videos/chunk-*/egocentric/episode_*.mp4`` stores the RGB frames.

Each sample uses adjacent RGB frames ``(frame_t, frame_{t+1})`` and predicts
the action recorded in the parquet row with ``frame_index=t``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.core.config import MAIN_ROOT
from src.pipeline.wan_pair_idm import (
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
class HumanoidTaskInfo:
    task_index: int
    task: str
    category: str
    description: str


@dataclass(frozen=True)
class HumanoidEpisodeInfo:
    episode_index: int
    task_indexes: tuple[int, ...]
    length: int
    instruction: str


@dataclass(frozen=True)
class HumanoidPairSample:
    video_path: str
    parquet_path: str
    episode: int
    chunk: int
    task_index: int
    task: str
    category: str
    rel_frame_t: int
    rel_frame_tp1: int
    action_target: tuple[float, ...]


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Required Humanoid metadata JSONL not found: {path}")
    rows = []
    for line_no, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
    if not rows:
        raise ValueError(f"Humanoid metadata JSONL is empty: {path}")
    return rows


def load_humanoid_metadata(
    data_root: Path,
) -> tuple[dict[int, HumanoidTaskInfo], dict[int, HumanoidEpisodeInfo]]:
    meta_dir = data_root / "meta"
    task_rows = read_jsonl(meta_dir / "tasks.jsonl")
    episode_rows = read_jsonl(meta_dir / "episodes.jsonl")

    tasks: dict[int, HumanoidTaskInfo] = {}
    for row in task_rows:
        missing = sorted({"task_index", "task", "category", "description"} - set(row))
        if missing:
            raise ValueError(f"Humanoid task metadata missing {missing}: {row}")
        task_index = int(row["task_index"])
        if task_index in tasks:
            raise ValueError(f"Duplicate Humanoid task_index in metadata: {task_index}")
        tasks[task_index] = HumanoidTaskInfo(
            task_index=task_index,
            task=str(row["task"]),
            category=str(row["category"]),
            description=str(row["description"]),
        )

    episodes: dict[int, HumanoidEpisodeInfo] = {}
    for row in episode_rows:
        missing = sorted({"episode_index", "tasks", "length", "instruction"} - set(row))
        if missing:
            raise ValueError(f"Humanoid episode metadata missing {missing}: {row}")
        episode_index = int(row["episode_index"])
        if episode_index in episodes:
            raise ValueError(f"Duplicate Humanoid episode_index in metadata: {episode_index}")
        task_indexes = tuple(int(value) for value in row["tasks"])
        if not task_indexes:
            raise ValueError(f"Humanoid episode has no task indexes: {row}")
        for task_index in task_indexes:
            if task_index not in tasks:
                raise ValueError(
                    f"Episode {episode_index} references unknown task_index={task_index}"
                )
        episodes[episode_index] = HumanoidEpisodeInfo(
            episode_index=episode_index,
            task_indexes=task_indexes,
            length=int(row["length"]),
            instruction=str(row["instruction"]),
        )
    return tasks, episodes


def parse_csv_selector(value: str) -> set[str]:
    if not value:
        return set()
    items = {item.strip() for item in value.split(",") if item.strip()}
    if not items:
        raise ValueError(f"Selector must contain at least one non-empty value: {value!r}")
    return items


def parse_task_index_selector(value: str) -> set[int]:
    if not value:
        return set()
    indexes: set[int] = set()
    for item in value.split(","):
        part = item.strip()
        if not part:
            continue
        if "-" in part:
            bounds = part.split("-")
            if len(bounds) != 2 or not bounds[0] or not bounds[1]:
                raise ValueError(f"Bad task index range: {part!r}")
            start = int(bounds[0])
            end = int(bounds[1])
            if end < start:
                raise ValueError(f"Bad descending task index range: {part!r}")
            indexes.update(range(start, end + 1))
        else:
            indexes.add(int(part))
    if not indexes:
        raise ValueError(f"Task index selector did not contain indexes: {value!r}")
    return indexes


def sample_matches_task_filters(
    task: HumanoidTaskInfo,
    *,
    task_indexes: set[int],
    tasks: set[str],
    categories: set[str],
) -> bool:
    if task_indexes and task.task_index not in task_indexes:
        return False
    if tasks and task.task not in tasks:
        return False
    if categories and task.category not in categories:
        return False
    return True


def discover_humanoid_pairs(
    data_root: Path,
    *,
    max_samples: int,
    seed: int,
    action_dim: int,
    frame_stride: int,
    max_pairs_per_episode: int,
    task_indexes: set[int],
    tasks: set[str],
    categories: set[str],
) -> list[HumanoidPairSample]:
    if frame_stride <= 0:
        raise ValueError(f"frame_stride must be positive, got {frame_stride}")
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
    task_info_by_index, episode_info_by_index = load_humanoid_metadata(data_root)

    samples: list[HumanoidPairSample] = []
    for parquet_path in sorted(data_dir.glob("chunk-*/episode_*.parquet")):
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

        df = pd.read_parquet(
            parquet_path,
            columns=["action", "frame_index", "episode_index", "task_index", "next.done"],
        )
        if df.empty:
            raise ValueError(f"Humanoid episode parquet is empty: {parquet_path}")
        episode_values = df["episode_index"].dropna().unique()
        if len(episode_values) != 1 or int(episode_values[0]) != episode:
            raise ValueError(
                f"Humanoid episode_index mismatch for {parquet_path}: "
                f"filename episode={episode}, parquet values={episode_values.tolist()}"
            )
        if episode not in episode_info_by_index:
            raise ValueError(f"Humanoid episode metadata not found for episode={episode}")
        episode_info = episode_info_by_index[episode]
        if int(episode_info.length) != len(df):
            raise ValueError(
                f"Humanoid episode length mismatch for {parquet_path}: "
                f"metadata={episode_info.length} parquet={len(df)}"
            )
        frame_indices = df["frame_index"].to_numpy(dtype=np.int64)
        if not np.array_equal(frame_indices, np.arange(len(df), dtype=np.int64)):
            raise ValueError(
                f"Humanoid frame_index must be contiguous 0..N-1 for video alignment: "
                f"{parquet_path}"
            )

        episode_pair_count = 0
        for row_idx in range(0, len(df) - 1, frame_stride):
            if bool(df.iloc[row_idx]["next.done"]):
                continue
            task_index = int(df.iloc[row_idx]["task_index"])
            if task_index not in episode_info.task_indexes:
                raise ValueError(
                    f"Humanoid row task_index={task_index} is not listed in episode "
                    f"metadata tasks={episode_info.task_indexes}: {parquet_path}"
                )
            if task_index not in task_info_by_index:
                raise ValueError(f"Humanoid task metadata not found for task_index={task_index}")
            task_info = task_info_by_index[task_index]
            if not sample_matches_task_filters(
                task_info,
                task_indexes=task_indexes,
                tasks=tasks,
                categories=categories,
            ):
                continue
            action = np.asarray(df.iloc[row_idx]["action"], dtype=np.float32)
            if action.shape != (action_dim,):
                raise ValueError(
                    f"Bad Humanoid action shape at episode={episode} frame={row_idx}: "
                    f"{action.shape}, expected {(action_dim,)}"
                )
            samples.append(
                HumanoidPairSample(
                    video_path=str(video_path),
                    parquet_path=str(parquet_path),
                    episode=episode,
                    chunk=chunk,
                    task_index=task_index,
                    task=task_info.task,
                    category=task_info.category,
                    rel_frame_t=row_idx,
                    rel_frame_tp1=row_idx + 1,
                    action_target=tuple(float(v) for v in action.tolist()),
                )
            )
            episode_pair_count += 1
            if max_pairs_per_episode > 0 and episode_pair_count >= max_pairs_per_episode:
                break

    random.Random(seed).shuffle(samples)
    if max_samples > 0:
        samples = samples[:max_samples]
    if not samples:
        raise ValueError(f"No Humanoid adjacent frame pairs discovered under {data_root}")
    return samples


def split_humanoid_samples(
    samples: list[HumanoidPairSample],
    train_ratio: float,
    split_by: str,
    seed: int,
    *,
    train_samples_count: int = 0,
    val_samples_count: int = 0,
    val_task_indexes: set[int] | None = None,
) -> tuple[list[HumanoidPairSample], list[HumanoidPairSample]]:
    val_task_indexes = val_task_indexes or set()
    if train_samples_count > 0 or val_samples_count > 0:
        if val_task_indexes:
            raise ValueError(
                "explicit train/val sample counts cannot be combined with "
                "--val-task-indexes"
            )
        if train_samples_count <= 0 or val_samples_count <= 0:
            raise ValueError(
                "train_samples_count and val_samples_count must be both positive "
                "when explicit sample counts are used"
            )
        if split_by != "sample":
            raise ValueError(
                "explicit train/val sample counts require split_by=sample"
            )
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

    if val_task_indexes:
        known_task_indexes = {sample.task_index for sample in samples}
        missing = sorted(val_task_indexes - known_task_indexes)
        if missing:
            raise ValueError(f"--val-task-indexes selected absent task indexes: {missing}")
        train_samples = [
            sample for sample in samples if sample.task_index not in val_task_indexes
        ]
        val_samples = [
            sample for sample in samples if sample.task_index in val_task_indexes
        ]
        if not train_samples or not val_samples:
            raise ValueError(
                f"Invalid task holdout split: train={len(train_samples)} "
                f"val={len(val_samples)}"
            )
        return train_samples, val_samples

    if not 0.0 < train_ratio < 1.0:
        raise ValueError(f"train_ratio must be in (0,1), got {train_ratio}")
    if split_by == "sample":
        n_train = max(1, min(len(samples) - 1, int(round(len(samples) * train_ratio))))
        return samples[:n_train], samples[n_train:]
    if split_by == "episode":
        group_values = sorted({sample.episode for sample in samples})
        group_for_sample = lambda sample: sample.episode
    elif split_by == "task":
        group_values = sorted({sample.task_index for sample in samples})
        group_for_sample = lambda sample: sample.task_index
    elif split_by == "category":
        group_values = sorted({sample.category for sample in samples})
        group_for_sample = lambda sample: sample.category
    else:
        raise ValueError(f"Unsupported split_by={split_by!r}")
    if len(group_values) < 2:
        raise ValueError(f"{split_by} split requires at least two groups")
    random.Random(seed).shuffle(group_values)
    n_train_groups = max(
        1,
        min(len(group_values) - 1, int(round(len(group_values) * train_ratio))),
    )
    train_groups = set(group_values[:n_train_groups])
    train_samples = [
        sample for sample in samples if group_for_sample(sample) in train_groups
    ]
    val_samples = [
        sample for sample in samples if group_for_sample(sample) not in train_groups
    ]
    if not train_samples or not val_samples:
        raise ValueError(
            f"Invalid {split_by} split: train={len(train_samples)} val={len(val_samples)}"
        )
    return train_samples, val_samples


def task_distribution_rows(samples: list[HumanoidPairSample]) -> list[dict]:
    episode_sets: dict[tuple[int, str, str], set[int]] = {}
    sample_counts: Counter[tuple[int, str, str]] = Counter()
    for sample in samples:
        key = (sample.task_index, sample.task, sample.category)
        sample_counts[key] += 1
        episode_sets.setdefault(key, set()).add(sample.episode)
    rows = []
    for task_index, task, category in sorted(sample_counts):
        key = (task_index, task, category)
        rows.append(
            {
                "task_index": task_index,
                "task": task,
                "category": category,
                "n_samples": int(sample_counts[key]),
                "n_episodes": len(episode_sets[key]),
            }
        )
    return rows


def split_summary(
    args: argparse.Namespace,
    samples: list[HumanoidPairSample],
    train_samples: list[HumanoidPairSample],
    val_samples: list[HumanoidPairSample],
) -> dict:
    return {
        "dataset": "humanoid_everyday_lerobot",
        "robot_type": "h1",
        "data_root": args.data_root,
        "seed": int(args.seed),
        "frame_stride": int(args.frame_stride),
        "max_samples": int(args.max_samples),
        "max_pairs_per_episode": int(args.max_pairs_per_episode),
        "filters": {
            "task_indexes": sorted(parse_task_index_selector(args.task_indexes)),
            "tasks": sorted(parse_csv_selector(args.tasks)),
            "categories": sorted(parse_csv_selector(args.categories)),
        },
        "split": {
            "split_by": args.split_by,
            "train_ratio": float(args.train_ratio),
            "train_samples_arg": int(args.train_samples),
            "eval_samples_arg": int(args.eval_samples),
            "val_task_indexes": sorted(parse_task_index_selector(args.val_task_indexes)),
        },
        "all": summarize_sample_set(samples),
        "train": summarize_sample_set(train_samples),
        "val": summarize_sample_set(val_samples),
    }


def summarize_sample_set(samples: list[HumanoidPairSample]) -> dict:
    return {
        "n_samples": len(samples),
        "n_episodes": len({sample.episode for sample in samples}),
        "n_tasks": len({sample.task_index for sample in samples}),
        "n_categories": len({sample.category for sample in samples}),
        "tasks": task_distribution_rows(samples),
    }


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
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
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


def humanoid_target_stats(samples: list[HumanoidPairSample]) -> tuple[torch.Tensor, torch.Tensor]:
    action_arr = np.asarray([sample.action_target for sample in samples], dtype=np.float32)
    action_mean = action_arr.mean(axis=0)
    action_std = np.maximum(action_arr.std(axis=0), 1e-4)
    return torch.from_numpy(action_mean), torch.from_numpy(action_std)


def humanoid_action_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    baseline: np.ndarray,
) -> dict[str, float]:
    action_mse = mse_np(pred, target)
    baseline_mse = mse_np(np.broadcast_to(baseline, target.shape), target)
    total_sse = float(np.square(pred - target).sum())
    target_sse = float(np.square(target).sum())
    variance_ratio = action_mse / max(baseline_mse, 1e-12)
    return {
        "action_mse": action_mse,
        "mean_baseline_action_mse": baseline_mse,
        "variance_ratio": float(variance_ratio),
        "relative_l2_error": float(math.sqrt(total_sse / max(target_sse, 1e-12))),
        "pred_std_mean": float(pred.std(axis=0).mean()),
        "target_std_mean": float(target.std(axis=0).mean()),
    }


def humanoid_task_metric_rows(
    pred: np.ndarray,
    target: np.ndarray,
    baseline: np.ndarray,
    samples: list[HumanoidPairSample],
) -> list[dict]:
    indexes_by_task: dict[tuple[int, str, str], list[int]] = {}
    for idx, sample in enumerate(samples):
        key = (sample.task_index, sample.task, sample.category)
        indexes_by_task.setdefault(key, []).append(idx)
    rows = []
    for task_index, task, category in sorted(indexes_by_task):
        indexes = indexes_by_task[(task_index, task, category)]
        metrics = humanoid_action_metrics(pred[indexes], target[indexes], baseline)
        rows.append(
            {
                "task_index": task_index,
                "task": task,
                "category": category,
                "n_samples": len(indexes),
                **metrics,
            }
        )
    return rows


def humanoid_model_arch(args: argparse.Namespace) -> str:
    return str(args.model_arch)


def make_humanoid_model(args: argparse.Namespace) -> nn.Module:
    model_arch = humanoid_model_arch(args)
    if model_arch == "small_cnn":
        return SmallPairCnn(
            int(args.action_dim),
            base_channels=args.base_channels,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
        )
    if model_arch == "transformer":
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
    raise ValueError(f"Unsupported model_arch={model_arch!r}")


def humanoid_model_payload(args: argparse.Namespace) -> dict:
    model_arch = humanoid_model_arch(args)
    if model_arch == "small_cnn":
        return {
            "model_arch": model_arch,
            "input_channels": 6,
            "base_channels": args.base_channels,
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "action_dim": args.action_dim,
            "alignment": "humanoid_frame_pair_t_to_t_plus_1_predict_action_t",
        }
    if model_arch == "transformer":
        return {
            "model_arch": model_arch,
            "action_dim": args.action_dim,
            "patch_size": args.transformer_patch_size,
            "embed_dim": args.transformer_embed_dim,
            "depth": args.transformer_depth,
            "num_heads": args.transformer_num_heads,
            "mlp_ratio": args.transformer_mlp_ratio,
            "hidden_dim": args.hidden_dim,
            "dropout": args.transformer_dropout,
            "attn_dropout": args.transformer_attn_dropout,
            "alignment": "humanoid_frame_pair_t_to_t_plus_1_predict_action_t",
        }
    raise ValueError(f"Unsupported model_arch={model_arch!r}")


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
    task_metrics_path: Path | None = None,
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
                "task_index": sample.task_index,
                "task": sample.task,
                "category": sample.category,
                "rel_frame_t": sample.rel_frame_t,
                "rel_frame_tp1": sample.rel_frame_tp1,
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
    baseline = action_mean.detach().cpu().numpy()[None, :]
    metrics = {
        "n_samples": len(subset),
        "n_tasks": len({sample.task_index for sample in subset}),
        "n_categories": len({sample.category for sample in subset}),
        **humanoid_action_metrics(pred_all, target_all, baseline),
    }
    task_rows = humanoid_task_metric_rows(pred_all, target_all, baseline, subset)
    if prediction_path is not None:
        write_rows(pred_rows, prediction_path)
    if task_metrics_path is not None:
        write_rows(task_rows, task_metrics_path)
    return metrics


def save_humanoid_checkpoint(
    model: SmallPairCnn,
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
            "split_by": args.split_by,
            "train_ratio": args.train_ratio,
            "task_indexes": sorted(parse_task_index_selector(args.task_indexes)),
            "tasks": sorted(parse_csv_selector(args.tasks)),
            "categories": sorted(parse_csv_selector(args.categories)),
            "val_task_indexes": sorted(parse_task_index_selector(args.val_task_indexes)),
            "lr_scheduler": args.lr_scheduler,
            "min_lr_ratio": args.min_lr_ratio,
            "dataset": "humanoid_everyday_lerobot",
            "robot_type": "h1",
            "frame_stride": args.frame_stride,
            "max_pairs_per_episode": args.max_pairs_per_episode,
        },
        "val_metrics": val_metrics,
    }
    torch.save(payload, out_dir / filename)


def load_humanoid_pair_idm(checkpoint: Path, device: str) -> tuple[nn.Module, torch.Tensor, torch.Tensor, dict]:
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model_cfg = ckpt["model"]
    model_arch = str(model_cfg.get("model_arch", "small_cnn"))
    if model_arch == "small_pair_cnn":
        model_arch = "small_cnn"
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
    return discover_humanoid_pairs(
        Path(args.data_root),
        max_samples=args.max_samples,
        seed=int(args.seed),
        action_dim=int(args.action_dim),
        frame_stride=int(args.frame_stride),
        max_pairs_per_episode=int(args.max_pairs_per_episode),
        task_indexes=parse_task_index_selector(args.task_indexes),
        tasks=parse_csv_selector(args.tasks),
        categories=parse_csv_selector(args.categories),
    )


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
        val_task_indexes=parse_task_index_selector(args.val_task_indexes),
    )
    action_mean, action_std = humanoid_target_stats(train_samples)
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = str(
            Path(MAIN_ROOT) / "output" / "humanoid_pair_idm" / "humanoid_everyday_h1"
        )
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "split_manifest.json", split_summary(args, samples, train_samples, val_samples))
    if args.write_samples_json:
        write_json(out_dir / "samples.json", [asdict(sample) for sample in samples])
        write_json(out_dir / "train_samples.json", [asdict(sample) for sample in train_samples])
        write_json(out_dir / "val_samples.json", [asdict(sample) for sample in val_samples])

    print(
        f"alignment=(frame_t,frame_t+1)->action_t dataset=humanoid_everyday_h1 "
        f"model_arch={args.model_arch} "
        f"split_by={args.split_by} train_samples={len(train_samples)} "
        f"val_samples={len(val_samples)} train_episodes={len({s.episode for s in train_samples})} "
        f"val_episodes={len({s.episode for s in val_samples})} "
        f"train_tasks={len({s.task_index for s in train_samples})} "
        f"val_tasks={len({s.task_index for s in val_samples})}",
        flush=True,
    )

    device = args.device
    model = make_humanoid_model(args).to(device)
    print(
        f"trainable_params={count_trainable_parameters(model)} action_dim={args.action_dim}",
        flush=True,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.min_lr_ratio < 0.0:
        raise ValueError(f"min_lr_ratio must be non-negative, got {args.min_lr_ratio}")
    if args.lr_scheduler == "none":
        scheduler = None
    elif args.lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.steps,
            eta_min=args.lr * args.min_lr_ratio,
        )
    else:
        raise ValueError(f"Unsupported lr_scheduler={args.lr_scheduler!r}")

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
        task_metrics_path = out_dir / "val_task_metrics.csv" if write_predictions else None
        metrics = validate_humanoid_samples(
            model,
            val_samples,
            resize,
            action_mean_dev,
            action_std_dev,
            device,
            args,
            prediction_path=prediction_path,
            task_metrics_path=task_metrics_path,
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
            task_metrics_path=out_dir / "best_val_task_metrics.csv",
        )
        write_json(out_dir / "best_val_metrics.json", best_metrics)
    write_rows(history, out_dir / "train_loss.csv")
    plot_humanoid_loss_curves(out_dir)
    print(json.dumps({"val": val_metrics, "out_dir": str(out_dir)}, indent=2), flush=True)


@torch.no_grad()
def validate_humanoid_checkpoint(args: argparse.Namespace) -> None:
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
        val_task_indexes=parse_task_index_selector(args.val_task_indexes),
    )
    device = args.device
    model, action_mean, action_std, ckpt = load_humanoid_pair_idm(
        Path(args.checkpoint),
        device,
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
        task_metrics_path=out_dir / "val_task_metrics.csv",
    )
    write_json(out_dir / "val_metrics.json", metrics)
    print(json.dumps({"val": metrics, "out_dir": str(out_dir)}, indent=2), flush=True)


@torch.no_grad()
def eval_humanoid_pairs(args: argparse.Namespace) -> None:
    seed = int(args.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    resize = parse_resize(args.resize)
    samples = prepare_humanoid_samples(args)
    device = args.device
    model, action_mean, action_std, ckpt = load_humanoid_pair_idm(
        Path(args.checkpoint),
        device,
    )
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
        task_metrics_path=out_dir / "task_metrics.csv",
    )
    write_json(out_dir / "metrics.json", metrics)
    print(json.dumps({"eval": metrics, "out_dir": str(out_dir)}, indent=2), flush=True)


def add_humanoid_data_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resize", default="256x256")
    parser.add_argument("--model-arch", choices=["small_cnn", "transformer"], default="transformer")
    parser.add_argument("--max-samples", type=int, default=0,
                        help="maximum discovered adjacent frame pairs; 0 keeps all")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--action-dim", type=int, default=HUMANOID_H1_ACTION_DIM)
    parser.add_argument("--frame-stride", type=int, default=1,
                        help="use every Nth adjacent frame pair within each episode")
    parser.add_argument("--max-pairs-per-episode", type=int, default=0,
                        help="maximum pair samples to keep per episode; 0 keeps all")
    parser.add_argument("--task-indexes", default="",
                        help="comma/range task_index filter, e.g. 0,3,8-12; empty keeps all")
    parser.add_argument("--tasks", default="",
                        help="comma-separated exact meta/tasks.jsonl task names; empty keeps all")
    parser.add_argument("--categories", default="",
                        help="comma-separated task categories/groups; empty keeps all")
    parser.add_argument("--transformer-patch-size", type=int, default=32)
    parser.add_argument("--transformer-embed-dim", type=int, default=256)
    parser.add_argument("--transformer-depth", type=int, default=4)
    parser.add_argument("--transformer-num-heads", type=int, default=8)
    parser.add_argument("--transformer-mlp-ratio", type=float, default=4.0)
    parser.add_argument("--transformer-dropout", type=float, default=0.1)
    parser.add_argument("--transformer-attn-dropout", type=float, default=0.0)


def add_humanoid_split_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--train-samples", type=int, default=0,
                        help="explicit train sample count; 0 uses train-ratio")
    parser.add_argument("--eval-samples", type=int, default=0,
                        help="explicit eval sample count; 0 uses train-ratio")
    parser.add_argument("--train-ratio", type=float, default=0.875)
    parser.add_argument(
        "--split-by",
        choices=["sample", "episode", "task", "category"],
        default="sample",
    )
    parser.add_argument("--val-task-indexes", default="",
                        help="explicit held-out task_index selector for validation")


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
    train_p.add_argument("--hidden-dim", type=int, default=128)
    train_p.add_argument("--dropout", type=float, default=0.0)
    train_p.add_argument("--lr", type=float, default=1e-4)
    train_p.add_argument("--weight-decay", type=float, default=1e-4)
    train_p.add_argument("--lr-scheduler", choices=["none", "cosine"], default="cosine")
    train_p.add_argument("--min-lr-ratio", type=float, default=0.05)
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
