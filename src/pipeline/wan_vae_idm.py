"""Wan VAE based inverse dynamics model for action consistency.

This module trains a lightweight Video2Action head on top of frozen Wan 2.2
VAE latents.  The model predicts the middle-frame action for a 17-frame clip:

  - ``action.ee_action`` + ``action.hand_cmd`` for 24-dim arm-hand mode
  - ``action.robot_q_desired`` + ``action.hand_cmd`` for 48-dim full-body mode

Evaluation compares IDM-predicted actions from generated and GT videos against
the same ground-truth action target.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.core.config import DATASET_ROOT, MAIN_ROOT
from src.core.wan_loader import load_vae
from src.pipeline.action_mask import (
    ActionMaskResolver,
    ClipActionMask,
    action_dim_names_for_mode,
    action_dim_parts_for_mode,
    default_action_mask_root,
)
from src.pipeline.mitty_cache import encode_video_array_batch
from src.pipeline.train_mitty import DEFAULT_VAE

DEFAULT_TASK_SHORT = "Inspire_Collect_Clothes_MainCamOnly"
SEGMENT_FPS = 30.0
DEFAULT_TARGET_FPS = 16.0
DEFAULT_NUM_FRAMES = 17
ARM_DIM = 12
HAND_DIM = 12
ACTION_DIM = ARM_DIM + HAND_DIM
ROBOT_Q_DIM = 36
FULL_BODY_ACTION_DIM = ROBOT_Q_DIM + HAND_DIM
TARGET_MODES = ("arm_hand", "full_body")


def validate_target_mode(target_mode: str) -> str:
    if target_mode not in TARGET_MODES:
        raise ValueError(f"Unsupported target_mode={target_mode!r}, expected {TARGET_MODES}")
    return target_mode


def target_action_dim(target_mode: str) -> int:
    target_mode = validate_target_mode(target_mode)
    return ACTION_DIM if target_mode == "arm_hand" else FULL_BODY_ACTION_DIM


def target_action_slices(target_mode: str) -> dict[str, slice]:
    target_mode = validate_target_mode(target_mode)
    if target_mode == "arm_hand":
        return {
            "total": slice(0, ACTION_DIM),
            "arm": slice(0, ARM_DIM),
            "hand": slice(ARM_DIM, ACTION_DIM),
        }
    return {
        "total": slice(0, FULL_BODY_ACTION_DIM),
        "root": slice(0, 7),
        "left_leg": slice(7, 13),
        "right_leg": slice(13, 19),
        "waist": slice(19, 22),
        "left_arm": slice(22, 29),
        "right_arm": slice(29, 36),
        "arm": slice(22, 36),
        "left_hand": slice(36, 42),
        "right_hand": slice(42, 48),
        "hand": slice(36, 48),
    }

COLLECT_TASK = "Inspire_Collect_Clothes_MainCamOnly"
WASH_TASK = "Inspire_Put_Clothes_into_Washing_Machine"
WASH_MAINCAM_TASK = "Inspire_Put_Clothes_into_Washing_Machine_MainCamOnly"
PILLOW_TASK = "Inspire_Pickup_Pillow_MainCamOnly"


@dataclass(frozen=True)
class H2RTaskConfig:
    record_task: str
    canonical_task: str
    checkpoint_key: str
    target_task_short: str
    model_task_short: str


@dataclass
class H2RTaskBundle:
    config: H2RTaskConfig
    checkpoint: Path
    model: "WanVaeActionHead"
    mean: torch.Tensor
    std: torch.Tensor
    ckpt: dict
    resolver: "ActionResolver"
    segment_root: Path


H2R_TASK_CONFIGS = {
    COLLECT_TASK: H2RTaskConfig(
        record_task=COLLECT_TASK,
        canonical_task=COLLECT_TASK,
        checkpoint_key="collect",
        target_task_short=COLLECT_TASK,
        model_task_short=COLLECT_TASK,
    ),
    WASH_TASK: H2RTaskConfig(
        record_task=WASH_TASK,
        canonical_task=WASH_MAINCAM_TASK,
        checkpoint_key="wash",
        target_task_short=WASH_TASK,
        model_task_short=WASH_MAINCAM_TASK,
    ),
    WASH_MAINCAM_TASK: H2RTaskConfig(
        record_task=WASH_MAINCAM_TASK,
        canonical_task=WASH_MAINCAM_TASK,
        checkpoint_key="wash",
        target_task_short=WASH_MAINCAM_TASK,
        model_task_short=WASH_MAINCAM_TASK,
    ),
    PILLOW_TASK: H2RTaskConfig(
        record_task=PILLOW_TASK,
        canonical_task=PILLOW_TASK,
        checkpoint_key="pillow",
        target_task_short=PILLOW_TASK,
        model_task_short=PILLOW_TASK,
    ),
}


@dataclass(frozen=True)
class ClipSample:
    video_path: str
    episode: int
    episode_name: str
    segment: str
    frame_start: int
    clip_start: float
    clip_dur: float
    rel_frame_indices: tuple[int, ...]
    abs_frame_indices: tuple[int, ...]
    target: tuple[float, ...]
    action_mask: tuple[float, ...] | None = None
    action_mask_frame_ratio: tuple[float, ...] | None = None
    visible_action_count: int = ACTION_DIM
    visible_arm_count: int = ARM_DIM
    visible_hand_count: int = HAND_DIM
    visible_action_ratio: float = 1.0
    visible_arm_ratio: float = 1.0
    visible_hand_ratio: float = 1.0
    action_mask_path: str = ""


def parse_resize(value: str) -> tuple[int, int] | None:
    if value.lower() in {"none", "native", "0"}:
        return None
    parts = value.lower().split("x")
    if len(parts) != 2:
        raise ValueError(f"resize must be WIDTHxHEIGHT or none, got {value!r}")
    width, height = int(parts[0]), int(parts[1])
    if width <= 0 or height <= 0:
        raise ValueError(f"resize dimensions must be positive, got {value!r}")
    if width % 16 != 0 or height % 16 != 0:
        raise ValueError(f"Wan VAE resize dimensions must be multiples of 16, got {value!r}")
    return width, height


def default_task_full(task_short: str) -> str:
    return f"G1_WBT_{task_short}"


def default_task_data_root(task_short: str, task_full: str | None = None) -> Path:
    return Path(DATASET_ROOT) / (task_full or default_task_full(task_short))


def default_task_segment_root(task_short: str) -> Path:
    return Path(MAIN_ROOT) / "training_data" / "segment" / task_short


def resolve_task_args(args: argparse.Namespace) -> None:
    task_short = str(args.task_short)
    if task_short.startswith("G1_WBT_"):
        raise ValueError(
            f"--task-short must be the segment task name without G1_WBT_ prefix, got {task_short!r}"
        )
    task_full = str(args.task_full) if args.task_full else default_task_full(task_short)
    args.task_full = task_full
    if args.data_root is None:
        args.data_root = str(default_task_data_root(task_short, task_full))
    if args.segment_root is None:
        args.segment_root = str(default_task_segment_root(task_short))


def resolve_action_mask_root_arg(args: argparse.Namespace) -> Path | None:
    root = getattr(args, "action_mask_root", None)
    if root is None or str(root).strip() == "":
        return None
    if str(root).lower() == "default":
        return default_action_mask_root()
    return Path(root)


def load_action_frame_table(data_root: Path) -> pd.DataFrame:
    data_dir = data_root / "data"
    parquet_paths = sorted(data_dir.glob("chunk-*/*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(f"Action parquet files not found under: {data_dir}")
    df = pd.concat([pd.read_parquet(path) for path in parquet_paths], ignore_index=True)
    required = {
        "episode_index",
        "frame_index",
        "action.ee_action",
        "action.hand_cmd",
        "action.robot_q_desired",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Action parquet missing required columns {missing}: {data_root}")
    return df


class ActionResolver:
    """Resolve clip-level arm-hand action labels from raw LeRobot parquet."""

    def __init__(self, data_root: Path):
        self.data_root = data_root
        self.df = load_action_frame_table(self.data_root)
        self._episode_rows = {
            int(ep): group.set_index("frame_index", drop=False)
            for ep, group in self.df.groupby("episode_index", sort=False)
        }

    def target_for_indices(
        self,
        episode: int,
        frame_indices: list[int],
        *,
        target_mode: str,
    ) -> np.ndarray:
        if episode not in self._episode_rows:
            raise ValueError(f"Episode {episode} not found in {self.data_root}")
        rows = self._episode_rows[episode]
        missing = [idx for idx in frame_indices if idx not in rows.index]
        if missing:
            raise ValueError(
                f"Missing action frames for episode {episode}: {missing[:8]}"
            )
        idx = frame_indices[len(frame_indices) // 2]
        row = rows.loc[idx]
        ee = np.asarray(row["action.ee_action"], dtype=np.float32)
        hand = np.asarray(row["action.hand_cmd"], dtype=np.float32)
        robot_q = np.asarray(row["action.robot_q_desired"], dtype=np.float32)
        if ee.shape != (ARM_DIM,) or hand.shape != (HAND_DIM,):
            raise ValueError(
                f"Bad action shape at episode={episode} frame={idx}: "
                f"ee={ee.shape}, hand={hand.shape}"
            )
        if target_mode == "arm_hand":
            return np.concatenate([ee, hand], axis=0).astype(np.float32)
        if target_mode == "full_body":
            if robot_q.shape != (ROBOT_Q_DIM,):
                raise ValueError(
                    f"Bad full-body action shape at episode={episode} frame={idx}: "
                    f"robot_q={robot_q.shape}, hand={hand.shape}"
                )
            return np.concatenate([robot_q, hand], axis=0).astype(np.float32)
        raise ValueError(f"Unsupported target_mode={target_mode!r}")


def clip_frame_indices(
    clip_start: float,
    clip_dur: float,
    num_frames: int,
    target_fps: float,
) -> list[int]:
    if clip_start < 0:
        raise ValueError(f"clip_start must be non-negative, got {clip_start}")
    if clip_dur <= 0:
        raise ValueError(f"clip_dur must be positive, got {clip_dur}")
    if num_frames <= 0:
        raise ValueError(f"num_frames must be positive, got {num_frames}")
    if target_fps <= 0:
        raise ValueError(f"target_fps must be positive, got {target_fps}")
    base = int(round(clip_start * SEGMENT_FPS))
    clip_frames = max(1, int(round(clip_dur * SEGMENT_FPS)))
    return [
        base + min(int(round(i * SEGMENT_FPS / target_fps)), clip_frames - 1)
        for i in range(num_frames)
    ]


def read_video_selected_frames(
    path: str | Path,
    indices: list[int],
    resize: tuple[int, int] | None,
) -> np.ndarray:
    if not indices:
        raise ValueError("indices must not be empty")
    video_path = str(path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    frames = []
    wanted = set(indices)
    max_idx = max(indices)
    current = 0
    ok = True
    while current <= max_idx:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        if current in wanted:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            if resize is not None:
                frame_rgb = cv2.resize(frame_rgb, resize, interpolation=cv2.INTER_AREA)
            frames.append(frame_rgb)
        current += 1
    cap.release()
    if not ok and current <= max_idx:
        raise RuntimeError(
            f"Video ended before requested frame {max_idx}: {video_path}"
        )
    if len(frames) != len(indices):
        raise RuntimeError(
            f"Decoded {len(frames)} frames but requested {len(indices)} from {video_path}"
        )
    return np.stack(frames, axis=0).astype(np.uint8)


def read_video_uniform_frames(
    path: str | Path,
    num_frames: int,
    resize: tuple[int, int] | None,
) -> np.ndarray:
    video_path = str(path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    frames = []
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        if resize is not None:
            frame_rgb = cv2.resize(frame_rgb, resize, interpolation=cv2.INTER_AREA)
        frames.append(frame_rgb)
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames decoded from {video_path}")
    if len(frames) == num_frames:
        selected = frames
    else:
        selected_indices = np.linspace(0, len(frames) - 1, num_frames).round().astype(int)
        selected = [frames[int(idx)] for idx in selected_indices]
    return np.stack(selected, axis=0).astype(np.uint8)


def clip_sample_action_mask_fields(mask: ClipActionMask) -> dict:
    return {
        "action_mask": tuple(float(v) for v in mask.mask.tolist()),
        "action_mask_frame_ratio": tuple(float(v) for v in mask.frame_ratios.tolist()),
        "visible_action_count": mask.visible_action_count,
        "visible_arm_count": mask.visible_arm_count,
        "visible_hand_count": mask.visible_hand_count,
        "visible_action_ratio": mask.visible_action_ratio,
        "visible_arm_ratio": mask.visible_arm_ratio,
        "visible_hand_ratio": mask.visible_hand_ratio,
        "action_mask_path": mask.mask_path,
    }


def require_nonempty_clip_mask(sample_id: str, mask: ClipActionMask, policy: str) -> bool:
    if mask.visible_action_count > 0:
        return True
    if policy == "drop":
        return False
    if policy == "error":
        raise ValueError(f"Clip has no visible action dimensions: {sample_id}")
    raise ValueError(f"Unsupported empty action mask policy: {policy!r}")


def discover_segment_clips(
    resolver: ActionResolver,
    segment_root: Path,
    *,
    max_samples: int,
    clip_dur: float,
    clip_stride: float,
    num_frames: int,
    target_fps: float,
    seed: int,
    task_short: str,
    target_mode: str,
    action_mask_root: Path | None,
    action_mask_min_frame_ratio: float,
    empty_action_mask_policy: str,
) -> list[ClipSample]:
    if not segment_root.is_dir():
        raise FileNotFoundError(f"Segment root not found: {segment_root}")
    if clip_stride <= 0:
        raise ValueError(f"clip_stride must be positive, got {clip_stride}")
    samples: list[ClipSample] = []
    mask_resolver = None
    if action_mask_root is not None:
        mask_resolver = ActionMaskResolver(
            action_mask_root,
            task_short,
            min_frame_ratio=action_mask_min_frame_ratio,
            target_mode=target_mode,
        )
    for joints_path in sorted(segment_root.glob("ep*/seg*_joints.parquet")):
        seg_df = pd.read_parquet(joints_path)
        required = {"episode_index", "frame_index"}
        missing = sorted(required - set(seg_df.columns))
        if missing:
            raise ValueError(f"Segment parquet missing {missing}: {joints_path}")
        if seg_df.empty:
            continue
        episode_values = seg_df["episode_index"].unique()
        if len(episode_values) != 1:
            raise ValueError(f"Segment parquet spans multiple episodes: {joints_path}")
        episode = int(episode_values[0])
        frame_start = int(seg_df["frame_index"].min())
        frame_count = int(seg_df["frame_index"].max()) - frame_start + 1
        max_start = (frame_count / SEGMENT_FPS) - clip_dur
        if max_start < -1e-6:
            continue
        starts = []
        value = 0.0
        while value <= max_start + 1e-6:
            starts.append(round(value, 6))
            value += clip_stride
        video_path = joints_path.with_name(joints_path.name.replace("_joints.parquet", "_video.mp4"))
        if not video_path.is_file():
            raise FileNotFoundError(f"Segment video not found: {video_path}")
        segment = joints_path.stem.replace("_joints", "")
        episode_name = joints_path.parent.name
        for clip_start in starts:
            rel_indices = clip_frame_indices(clip_start, clip_dur, num_frames, target_fps)
            abs_indices = [frame_start + idx for idx in rel_indices]
            target = resolver.target_for_indices(
                episode,
                abs_indices,
                target_mode=target_mode,
            )
            mask_fields = {}
            if mask_resolver is not None:
                clip_id = f"{task_short}/{episode_name}/{segment}@{clip_start:.3f}"
                clip_mask = mask_resolver.load_clip(episode_name, segment, rel_indices)
                if not require_nonempty_clip_mask(
                    clip_id,
                    clip_mask,
                    empty_action_mask_policy,
                ):
                    continue
                mask_fields = clip_sample_action_mask_fields(clip_mask)
            samples.append(
                ClipSample(
                    video_path=str(video_path),
                    episode=episode,
                    episode_name=episode_name,
                    segment=segment,
                    frame_start=frame_start,
                    clip_start=float(clip_start),
                    clip_dur=float(clip_dur),
                    rel_frame_indices=tuple(rel_indices),
                    abs_frame_indices=tuple(abs_indices),
                    target=tuple(float(v) for v in target.tolist()),
                    **mask_fields,
                )
            )
    random.Random(seed).shuffle(samples)
    if max_samples > 0:
        samples = samples[:max_samples]
    if not samples:
        raise ValueError(f"No clips discovered under {segment_root}")
    return samples


class ClipDataset(torch.utils.data.Dataset):
    def __init__(self, samples: list[ClipSample], resize: tuple[int, int] | None):
        self.samples = samples
        self.resize = resize

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        frames = read_video_selected_frames(
            sample.video_path,
            list(sample.rel_frame_indices),
            self.resize,
        )
        target = np.asarray(sample.target, dtype=np.float32)
        item = {
            "frames": frames,
            "target": target,
            "sample_index": idx,
        }
        if sample.action_mask is not None:
            action_mask = np.asarray(sample.action_mask, dtype=np.float32)
            if action_mask.shape != target.shape:
                raise ValueError(
                    f"Bad action mask shape for sample {idx}: {action_mask.shape}"
                )
            item["action_mask"] = action_mask
        return item


def collate_clip_batch(items: list[dict]) -> dict:
    batch = {
        "frames": [item["frames"] for item in items],
        "target": torch.from_numpy(np.stack([item["target"] for item in items], axis=0)),
        "sample_index": [item["sample_index"] for item in items],
    }
    has_mask = ["action_mask" in item for item in items]
    if any(has_mask) and not all(has_mask):
        raise ValueError("Batch mixes masked and unmasked samples")
    if all(has_mask):
        batch["action_mask"] = torch.from_numpy(
            np.stack([item["action_mask"] for item in items], axis=0)
        )
    return batch


class WanVaeActionHead(nn.Module):
    def __init__(
        self,
        latent_channels: int = 48,
        action_dim: int = ACTION_DIM,
        target_mode: str = "arm_hand",
        head_type: str = "cnn_mlp",
        conv_channels: int = 256,
        conv_blocks: int = 4,
        readout_dim: int = 1024,
        hidden_dim: int = 1024,
        mlp_layers: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.action_dim = int(action_dim)
        self.target_mode = validate_target_mode(target_mode)
        expected_dim = target_action_dim(self.target_mode)
        if self.action_dim != expected_dim:
            raise ValueError(
                f"action_dim={self.action_dim} does not match "
                f"target_mode={self.target_mode!r} expected_dim={expected_dim}"
            )
        if head_type != "cnn_mlp":
            raise ValueError(f"Unsupported head_type={head_type!r}; expected 'cnn_mlp'")
        if conv_channels <= 0:
            raise ValueError(f"conv_channels must be positive, got {conv_channels}")
        if conv_blocks < 2:
            raise ValueError(f"conv_blocks must be at least 2, got {conv_blocks}")
        if readout_dim <= 0:
            raise ValueError(f"readout_dim must be positive, got {readout_dim}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if mlp_layers <= 0:
            raise ValueError(f"mlp_layers must be positive, got {mlp_layers}")
        if dropout < 0.0:
            raise ValueError(f"dropout must be non-negative, got {dropout}")

        first_channels = max(1, conv_channels // 2)
        conv_layers: list[nn.Module] = [
            nn.Conv3d(
                latent_channels,
                first_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.SiLU(),
            nn.Conv3d(
                first_channels,
                conv_channels,
                kernel_size=3,
                stride=(1, 2, 2),
                padding=1,
            ),
            nn.SiLU(),
        ]
        for _ in range(conv_blocks - 2):
            conv_layers.extend([
                nn.Conv3d(conv_channels, conv_channels, kernel_size=3, padding=1),
                nn.SiLU(),
            ])
        self.encoder = nn.Sequential(*conv_layers)
        self.readout = nn.Sequential(
            nn.Linear(conv_channels * 5, readout_dim),
            nn.SiLU(),
        )
        mlp_layers_list: list[nn.Module] = []
        in_dim = readout_dim
        for layer_idx in range(mlp_layers - 1):
            out_dim = max(self.action_dim * 2, hidden_dim // (2 ** layer_idx))
            mlp_layers_list.extend([
                nn.Linear(in_dim, out_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
            ])
            in_dim = out_dim
        mlp_layers_list.append(nn.Linear(in_dim, self.action_dim))
        self.mlp = nn.Sequential(*mlp_layers_list)

    def forward(self, latent: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.encoder(latent.float())
        if x.ndim != 5:
            raise ValueError(f"Expected 5D latent features, got {x.shape}")
        if x.shape[2:] != (5, 8, 8):
            raise ValueError(f"Expected CNN output [B, C, 5, 8, 8], got {x.shape}")
        x = x.mean(dim=(-1, -2)).permute(0, 2, 1).contiguous()
        if x.shape[1:] != (5, x.shape[-1]):
            raise ValueError(f"Expected pooled shape [B, 5, C], got {x.shape}")
        x = x.flatten(start_dim=1)
        x = self.readout(x)
        action = self.mlp(x)
        return {"action": action}


def split_samples(
    samples: list[ClipSample],
    train_ratio: float,
    split_by: str,
    seed: int,
) -> tuple[list[ClipSample], list[ClipSample]]:
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


def target_stats(samples: list[ClipSample]) -> tuple[torch.Tensor, torch.Tensor]:
    arr = np.asarray([sample.target for sample in samples], dtype=np.float32)
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    std = np.maximum(std, 1e-4)
    return torch.from_numpy(mean), torch.from_numpy(std)


def mse_parts(
    pred_action: torch.Tensor,
    target_action: torch.Tensor,
    *,
    target_mode: str,
) -> dict[str, torch.Tensor]:
    slices = target_action_slices(target_mode)
    parts = {
        name: F.mse_loss(pred_action[:, slc], target_action[:, slc])
        for name, slc in slices.items()
        if name != "total"
    }
    parts["total"] = F.mse_loss(pred_action, target_action)
    return parts


def _masked_mse_or_none(
    pred_action: torch.Tensor,
    target_action: torch.Tensor,
    mask: torch.Tensor,
    slc: slice,
) -> torch.Tensor | None:
    part_mask = mask[:, slc].to(dtype=pred_action.dtype)
    denom = part_mask.sum()
    if float(denom.detach().cpu()) <= 0.0:
        return None
    diff = (pred_action[:, slc] - target_action[:, slc]) ** 2
    return (diff * part_mask).sum() / denom


def masked_mse_parts(
    pred_action: torch.Tensor,
    target_action: torch.Tensor,
    mask: torch.Tensor,
    *,
    target_mode: str,
) -> dict[str, torch.Tensor | None]:
    if mask.shape != pred_action.shape:
        raise ValueError(f"mask shape {mask.shape} does not match prediction {pred_action.shape}")
    slices = target_action_slices(target_mode)
    total = _masked_mse_or_none(pred_action, target_action, mask, slices["total"])
    if total is None:
        raise ValueError("Masked MSE received a batch with zero visible action dimensions")
    parts: dict[str, torch.Tensor | None] = {"total": total}
    for name, slc in slices.items():
        if name == "total":
            continue
        parts[name] = _masked_mse_or_none(pred_action, target_action, mask, slc)
    return parts


def masked_sse_counts(
    pred_action: torch.Tensor,
    target_action: torch.Tensor,
    mask: torch.Tensor,
    *,
    target_mode: str,
) -> dict[str, tuple[float, float]]:
    if mask.shape != pred_action.shape:
        raise ValueError(f"mask shape {mask.shape} does not match prediction {pred_action.shape}")
    diff = (pred_action - target_action) ** 2
    mask = mask.to(dtype=diff.dtype)
    slices = target_action_slices(target_mode)

    def part(slc: slice) -> tuple[float, float]:
        part_mask = mask[:, slc]
        return (
            float((diff[:, slc] * part_mask).sum().detach().cpu()),
            float(part_mask.sum().detach().cpu()),
        )

    return {name: part(slc) for name, slc in slices.items()}


def sse_target_sse_parts(
    pred_action: torch.Tensor,
    target_action: torch.Tensor,
    *,
    target_mode: str,
) -> dict[str, tuple[float, float]]:
    diff = (pred_action - target_action) ** 2
    target_sq = target_action ** 2
    slices = target_action_slices(target_mode)
    return {
        name: (
            float(diff[:, slc].sum().detach().cpu()),
            float(target_sq[:, slc].sum().detach().cpu()),
        )
        for name, slc in slices.items()
    }


def masked_sse_target_sse_parts(
    pred_action: torch.Tensor,
    target_action: torch.Tensor,
    mask: torch.Tensor,
    *,
    target_mode: str,
) -> dict[str, tuple[float, float]]:
    if mask.shape != pred_action.shape:
        raise ValueError(f"mask shape {mask.shape} does not match prediction {pred_action.shape}")
    diff = (pred_action - target_action) ** 2
    target_sq = target_action ** 2
    mask = mask.to(dtype=diff.dtype)
    slices = target_action_slices(target_mode)
    return {
        name: (
            float((diff[:, slc] * mask[:, slc]).sum().detach().cpu()),
            float((target_sq[:, slc] * mask[:, slc]).sum().detach().cpu()),
        )
        for name, slc in slices.items()
    }


def tensor_or_nan(value: torch.Tensor | None) -> float:
    if value is None:
        return float("nan")
    return float(value.detach().cpu())


def weighted_action_loss(parts: dict[str, torch.Tensor],
                         arm_weight: float,
                         hand_weight: float) -> torch.Tensor:
    if arm_weight <= 0.0 or hand_weight <= 0.0:
        raise ValueError(
            f"loss weights must be positive, got arm={arm_weight}, hand={hand_weight}"
        )
    return (
        parts["arm"] * arm_weight + parts["hand"] * hand_weight
    ) / (arm_weight + hand_weight)


def weighted_masked_action_loss(
    parts: dict[str, torch.Tensor | None],
    arm_weight: float,
    hand_weight: float,
) -> torch.Tensor:
    if arm_weight <= 0.0 or hand_weight <= 0.0:
        raise ValueError(
            f"loss weights must be positive, got arm={arm_weight}, hand={hand_weight}"
        )
    weighted = []
    weights = []
    if parts["arm"] is not None:
        weighted.append(parts["arm"] * arm_weight)
        weights.append(arm_weight)
    if parts["hand"] is not None:
        weighted.append(parts["hand"] * hand_weight)
        weights.append(hand_weight)
    if not weighted:
        raise ValueError("Masked loss has neither visible arm nor visible hand dimensions")
    return sum(weighted) / sum(weights)


def encode_batch(vae, frames: list[np.ndarray], device: str) -> torch.Tensor:
    with torch.no_grad():
        latent = encode_video_array_batch(vae, frames, device)
    return latent


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def train(args: argparse.Namespace) -> None:
    resolve_task_args(args)
    seed = int(args.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    resize = parse_resize(args.resize)
    target_mode = validate_target_mode(args.target_mode)
    action_dim = target_action_dim(target_mode)
    resolver = ActionResolver(Path(args.data_root))
    action_mask_root = resolve_action_mask_root_arg(args)
    samples = discover_segment_clips(
        resolver,
        Path(args.segment_root),
        max_samples=args.max_samples,
        clip_dur=args.clip_duration,
        clip_stride=args.clip_stride,
        num_frames=args.num_frames,
        target_fps=args.target_fps,
        seed=seed,
        task_short=args.task_short,
        target_mode=target_mode,
        action_mask_root=action_mask_root,
        action_mask_min_frame_ratio=args.action_mask_min_frame_ratio,
        empty_action_mask_policy=args.empty_action_mask_policy,
    )
    train_samples, val_samples = split_samples(
        samples, args.train_ratio, args.split_by, seed,
    )
    mean, std = target_stats(train_samples)
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = str(Path(MAIN_ROOT) / "output" / "wan_vae_idm" / args.task_short)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "samples.json").write_text(
        json.dumps([asdict(sample) for sample in samples], ensure_ascii=False, indent=2)
    )
    (out_dir / "train_samples.json").write_text(
        json.dumps([asdict(sample) for sample in train_samples], ensure_ascii=False, indent=2)
    )
    (out_dir / "val_samples.json").write_text(
        json.dumps([asdict(sample) for sample in val_samples], ensure_ascii=False, indent=2)
    )
    print(
        f"split_by={args.split_by} train_samples={len(train_samples)} "
        f"val_samples={len(val_samples)} train_episodes={len({s.episode for s in train_samples})} "
        f"val_episodes={len({s.episode for s in val_samples})}",
        flush=True,
    )
    if action_mask_root is not None:
        visible_counts = np.asarray([sample.visible_action_count for sample in samples], dtype=np.float32)
        print(
            f"action_mask_root={action_mask_root} "
            f"min_frame_ratio={args.action_mask_min_frame_ratio:.3f} "
            f"visible_action_count_mean={visible_counts.mean():.2f} "
            f"visible_action_count_min={visible_counts.min():.0f} "
            f"visible_action_count_max={visible_counts.max():.0f}",
            flush=True,
        )

    device = args.device
    vae = load_vae(args.vae_path, torch.bfloat16, home_device=device)
    for param in vae.parameters():
        param.requires_grad_(False)
    model = WanVaeActionHead(
        action_dim=action_dim,
        target_mode=target_mode,
        head_type=args.head_type,
        conv_channels=args.conv_channels,
        conv_blocks=args.conv_blocks,
        readout_dim=args.readout_dim,
        hidden_dim=args.hidden_dim,
        mlp_layers=args.mlp_layers,
        dropout=args.dropout,
    ).to(device)
    print(f"trainable_action_head_params={count_trainable_parameters(model)}", flush=True)
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
        ClipDataset(train_samples, resize),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate_clip_batch,
        drop_last=False,
    )
    mean_dev = mean.to(device)
    std_dev = std.to(device)
    step = 0
    history = []
    eval_history = []
    last_eval_step: int | None = None
    best_eval_total = float("inf")

    def run_eval(eval_step: int, *, write_predictions: bool) -> dict[str, float]:
        nonlocal best_eval_total, last_eval_step
        prediction_path = out_dir / "val_predictions.csv" if write_predictions else None
        metrics = validate(
            model,
            vae,
            val_samples,
            resize,
            mean_dev,
            std_dev,
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
        write_metric_rows(eval_history, out_dir / "eval_loss.csv")
        selection_key = "masked_total_mse" if "masked_total_mse" in metrics else "total_mse"
        if metrics[selection_key] < best_eval_total:
            best_eval_total = metrics[selection_key]
            save_checkpoint(
                model,
                mean,
                std,
                args,
                out_dir,
                metrics,
                filename="best_checkpoint.pt",
            )
        print(
            f"eval_step={eval_step:04d} "
            f"total_mse={metrics['total_mse']:.6f} "
            f"arm_mse={metrics['arm_mse']:.6f} "
            f"hand_mse={metrics['hand_mse']:.6f} "
            f"select_{selection_key}={metrics[selection_key]:.6f} "
            f"mean_baseline_total_mse={metrics['mean_baseline_total_mse']:.6f} "
            f"best_select_mse={best_eval_total:.6f}",
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
            target = batch["target"].to(device)
            norm_target = (target - mean_dev) / std_dev
            latent = encode_batch(vae, batch["frames"], device)
            pred = model(latent)["action"]
            unmasked_parts = mse_parts(pred, norm_target, target_mode=target_mode)
            action_mask = batch.get("action_mask")
            if action_mask is not None:
                action_mask = action_mask.to(device)
                parts = masked_mse_parts(
                    pred,
                    norm_target,
                    action_mask,
                    target_mode=target_mode,
                )
                if target_mode == "arm_hand":
                    loss = weighted_masked_action_loss(
                        parts, args.arm_loss_weight, args.hand_loss_weight,
                    )
                else:
                    loss = parts["total"]
                visible_count = float(action_mask.sum(dim=1).mean().detach().cpu())
            else:
                parts = unmasked_parts
                if target_mode == "arm_hand":
                    loss = weighted_action_loss(parts, args.arm_loss_weight, args.hand_loss_weight)
                else:
                    loss = parts["total"]
                visible_count = float(action_dim)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            step += 1
            row = {
                "step": step,
                "loss": float(loss.detach().cpu()),
                "unweighted_loss": tensor_or_nan(parts["total"]),
                "arm_loss": tensor_or_nan(parts["arm"]),
                "hand_loss": tensor_or_nan(parts["hand"]),
                "unmasked_total_loss": float(unmasked_parts["total"].detach().cpu()),
                "unmasked_arm_loss": float(unmasked_parts["arm"].detach().cpu()),
                "unmasked_hand_loss": float(unmasked_parts["hand"].detach().cpu()),
                "visible_action_count": visible_count,
                "lr": float(optimizer.param_groups[0]["lr"]),
            }
            for key, value in parts.items():
                if key not in {"total", "arm", "hand"}:
                    row[f"{key}_loss"] = tensor_or_nan(value)
            for key, value in unmasked_parts.items():
                if key not in {"total", "arm", "hand"}:
                    row[f"unmasked_{key}_loss"] = float(value.detach().cpu())
            history.append(row)
            if step == 1 or step % args.log_every == 0 or step == args.steps:
                print(
                    f"step={step:04d} loss={row['loss']:.6f} "
                    f"arm={row['arm_loss']:.6f} hand={row['hand_loss']:.6f} "
                    f"visible={row['visible_action_count']:.1f}",
                    flush=True,
                )
            if args.eval_every > 0 and (step % args.eval_every == 0 or step == args.steps):
                run_eval(step, write_predictions=False)

    val_metrics = run_eval(step, write_predictions=True)
    save_checkpoint(model, mean, std, args, out_dir, val_metrics)
    best_ckpt_path = out_dir / "best_checkpoint.pt"
    if best_ckpt_path.is_file():
        best_payload = torch.load(best_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(best_payload["model_state"], strict=True)
        best_metrics = validate(
            model,
            vae,
            val_samples,
            resize,
            mean_dev,
            std_dev,
            device,
            args,
            prediction_path=out_dir / "best_val_predictions.csv",
        )
        (out_dir / "best_val_metrics.json").write_text(
            json.dumps(best_metrics, ensure_ascii=False, indent=2)
        )
    write_metric_rows(history, out_dir / "train_loss.csv")
    plot_loss_curves(out_dir)
    print(json.dumps({"val": val_metrics, "out_dir": str(out_dir)}, indent=2), flush=True)


@torch.no_grad()
def validate(
    model: WanVaeActionHead,
    vae,
    samples: list[ClipSample],
    resize: tuple[int, int] | None,
    mean: torch.Tensor,
    std: torch.Tensor,
    device: str,
    args: argparse.Namespace,
    prediction_path: Path | None,
) -> dict[str, float]:
    if not samples:
        return {}
    target_mode = validate_target_mode(args.target_mode)
    model.eval()
    val_subset = samples if args.val_max_samples <= 0 else samples[: args.val_max_samples]
    loader = torch.utils.data.DataLoader(
        ClipDataset(val_subset, resize),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate_clip_batch,
        drop_last=False,
    )
    metric_keys = list(target_action_slices(target_mode))
    sums = {key: 0.0 for key in metric_keys}
    baseline_sums = {key: 0.0 for key in metric_keys}
    relative_sse = {key: 0.0 for key in metric_keys}
    relative_target_sse = {key: 0.0 for key in metric_keys}
    masked_sums = {key: 0.0 for key in metric_keys}
    masked_counts = {key: 0.0 for key in metric_keys}
    masked_baseline_sums = {key: 0.0 for key in metric_keys}
    masked_baseline_counts = {key: 0.0 for key in metric_keys}
    masked_relative_sse = {key: 0.0 for key in metric_keys}
    masked_relative_target_sse = {key: 0.0 for key in metric_keys}
    visible_action_counts = []
    visible_arm_ratios = []
    visible_hand_ratios = []
    pred_rows = []
    pred_chunks = []
    target_chunks = []
    count = 0
    for batch in loader:
        target = batch["target"].to(device)
        latent = encode_batch(vae, batch["frames"], device)
        pred_norm = model(latent)["action"]
        pred = pred_norm * std + mean
        parts = mse_parts(pred, target, target_mode=target_mode)
        rel_parts = sse_target_sse_parts(pred, target, target_mode=target_mode)
        baseline = mean.unsqueeze(0).expand_as(target)
        baseline_parts = mse_parts(baseline, target, target_mode=target_mode)
        action_mask = batch.get("action_mask")
        n = target.shape[0]
        count += n
        for key in sums:
            sums[key] += float(parts[key].detach().cpu()) * n
            baseline_sums[key] += float(baseline_parts[key].detach().cpu()) * n
            part_sse, part_target_sse = rel_parts[key]
            relative_sse[key] += part_sse
            relative_target_sse[key] += part_target_sse
        if action_mask is not None:
            action_mask = action_mask.to(device)
            pred_masked = masked_sse_counts(
                pred,
                target,
                action_mask,
                target_mode=target_mode,
            )
            baseline_masked = masked_sse_counts(
                baseline,
                target,
                action_mask,
                target_mode=target_mode,
            )
            rel_masked = masked_sse_target_sse_parts(
                pred,
                target,
                action_mask,
                target_mode=target_mode,
            )
            for key in masked_sums:
                sse, denom = pred_masked[key]
                base_sse, base_denom = baseline_masked[key]
                part_sse, part_target_sse = rel_masked[key]
                masked_sums[key] += sse
                masked_counts[key] += denom
                masked_baseline_sums[key] += base_sse
                masked_baseline_counts[key] += base_denom
                masked_relative_sse[key] += part_sse
                masked_relative_target_sse[key] += part_target_sse
            mask_cpu = action_mask.detach().cpu().numpy()
            visible_action_counts.extend(mask_cpu.sum(axis=1).astype(float).tolist())
            if target_mode == "arm_hand":
                visible_arm_ratios.extend(mask_cpu[:, :ARM_DIM].mean(axis=1).astype(float).tolist())
                visible_hand_ratios.extend(mask_cpu[:, ARM_DIM:].mean(axis=1).astype(float).tolist())
            else:
                visible_arm_ratios.extend(mask_cpu[:, 22:36].mean(axis=1).astype(float).tolist())
                visible_hand_ratios.extend(mask_cpu[:, 36:48].mean(axis=1).astype(float).tolist())
        pred_cpu = pred.detach().cpu().numpy()
        target_cpu = target.detach().cpu().numpy()
        pred_chunks.append(pred_cpu)
        target_chunks.append(target_cpu)
        for local_idx, sample_idx in enumerate(batch["sample_index"]):
            sample = val_subset[int(sample_idx)]
            row = {
                "sample_index": int(sample_idx),
                "episode": sample.episode,
                "segment": sample.segment,
                "clip_start": sample.clip_start,
                "clip_dur": sample.clip_dur,
                "video_path": sample.video_path,
                "visible_action_count": sample.visible_action_count,
                "visible_arm_count": sample.visible_arm_count,
                "visible_hand_count": sample.visible_hand_count,
                "visible_action_ratio": sample.visible_action_ratio,
                "visible_arm_ratio": sample.visible_arm_ratio,
                "visible_hand_ratio": sample.visible_hand_ratio,
                "action_mask_path": sample.action_mask_path,
            }
            for dim in range(target_action_dim(target_mode)):
                row[f"target_{dim:02d}"] = float(target_cpu[local_idx, dim])
                row[f"pred_{dim:02d}"] = float(pred_cpu[local_idx, dim])
                row[f"err_{dim:02d}"] = float(pred_cpu[local_idx, dim] - target_cpu[local_idx, dim])
                if sample.action_mask is not None:
                    row[f"mask_{dim:02d}"] = float(sample.action_mask[dim])
                    row[f"mask_ratio_{dim:02d}"] = float(sample.action_mask_frame_ratio[dim])
            pred_rows.append(row)
    model.train()
    metrics = {f"{key}_mse": value / count for key, value in sums.items()}
    metrics.update({
        f"mean_baseline_{key}_mse": value / count
        for key, value in baseline_sums.items()
    })
    for key in metric_keys:
        denom = max(relative_target_sse[key], 1e-12)
        metrics[f"{key}_relative_l2_error"] = float(math.sqrt(relative_sse[key] / denom))
    if masked_counts["total"] > 0.0:
        for key in masked_sums:
            if masked_counts[key] > 0.0:
                metrics[f"masked_{key}_mse"] = masked_sums[key] / masked_counts[key]
                metrics[f"mean_baseline_masked_{key}_mse"] = (
                    masked_baseline_sums[key] / masked_baseline_counts[key]
                )
                denom = max(masked_relative_target_sse[key], 1e-12)
                metrics[f"masked_{key}_relative_l2_error"] = float(
                    math.sqrt(masked_relative_sse[key] / denom)
                )
            else:
                metrics[f"masked_{key}_mse"] = float("nan")
                metrics[f"mean_baseline_masked_{key}_mse"] = float("nan")
                metrics[f"masked_{key}_relative_l2_error"] = float("nan")
        metrics["visible_action_count_mean"] = float(np.mean(visible_action_counts))
        metrics["visible_action_ratio_mean"] = float(
            np.mean(visible_action_counts) / target_action_dim(target_mode)
        )
        metrics["visible_arm_ratio_mean"] = float(np.mean(visible_arm_ratios))
        metrics["visible_hand_ratio_mean"] = float(np.mean(visible_hand_ratios))
    pred_all = np.concatenate(pred_chunks, axis=0)
    target_all = np.concatenate(target_chunks, axis=0)
    pred_std = pred_all.std(axis=0)
    target_std = target_all.std(axis=0)
    metrics["pred_std_mean"] = float(pred_std.mean())
    metrics["target_std_mean"] = float(target_std.mean())
    arm_slice = target_action_slices(target_mode)["arm"]
    hand_slice = target_action_slices(target_mode)["hand"]
    metrics["pred_arm_std_mean"] = float(pred_std[arm_slice].mean())
    metrics["target_arm_std_mean"] = float(target_std[arm_slice].mean())
    metrics["pred_hand_std_mean"] = float(pred_std[hand_slice].mean())
    metrics["target_hand_std_mean"] = float(target_std[hand_slice].mean())
    if prediction_path is not None:
        write_prediction_rows(pred_rows, prediction_path)
    return metrics


def write_metric_rows(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_prediction_rows(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_loss_curves(out_dir: Path) -> None:
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
        {key: float(value) for key, value in row.items() if value != ""}
        for row in csv.DictReader(eval_path.open())
    ]
    if not train_rows:
        raise ValueError(f"Training loss CSV is empty: {train_path}")
    if not eval_rows:
        raise ValueError(f"Eval loss CSV is empty: {eval_path}")

    train_steps = np.asarray([row["step"] for row in train_rows], dtype=np.float64)
    train_total = np.asarray([row["loss"] for row in train_rows], dtype=np.float64)
    train_arm = np.asarray([row["arm_loss"] for row in train_rows], dtype=np.float64)
    train_hand = np.asarray([row["hand_loss"] for row in train_rows], dtype=np.float64)
    eval_steps = np.asarray([row["step"] for row in eval_rows], dtype=np.float64)
    eval_total = np.asarray([row["total_mse"] for row in eval_rows], dtype=np.float64)
    eval_arm = np.asarray([row["arm_mse"] for row in eval_rows], dtype=np.float64)
    eval_hand = np.asarray([row["hand_mse"] for row in eval_rows], dtype=np.float64)
    eval_baseline = np.asarray(
        [row["mean_baseline_total_mse"] for row in eval_rows],
        dtype=np.float64,
    )

    def smooth(values: np.ndarray, window: int = 50) -> np.ndarray:
        if len(values) < window:
            return values
        kernel = np.ones(window, dtype=np.float64) / window
        smoothed = np.convolve(values, kernel, mode="valid")
        return np.concatenate([np.full(window - 1, np.nan), smoothed])

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), dpi=160, sharex=True)
    axes[0].plot(train_steps, train_total, color="#1f77b4", alpha=0.16, linewidth=1)
    axes[0].plot(train_steps, train_arm, color="#ff7f0e", alpha=0.10, linewidth=1)
    axes[0].plot(train_steps, train_hand, color="#2ca02c", alpha=0.10, linewidth=1)
    axes[0].plot(train_steps, smooth(train_total), color="#1f77b4", linewidth=2.2, label="train total")
    axes[0].plot(train_steps, smooth(train_arm), color="#ff7f0e", linewidth=1.8, label="train arm")
    axes[0].plot(train_steps, smooth(train_hand), color="#2ca02c", linewidth=1.8, label="train hand")
    axes[0].set_ylabel("normalized train MSE")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(frameon=False, ncol=3)

    axes[1].plot(eval_steps, eval_total, marker="o", color="#1f77b4", label="eval total")
    axes[1].plot(eval_steps, eval_arm, marker="o", color="#ff7f0e", label="eval arm")
    axes[1].plot(eval_steps, eval_hand, marker="o", color="#2ca02c", label="eval hand")
    axes[1].plot(eval_steps, eval_baseline, color="#555555", linestyle="--", label="eval mean baseline")
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("action MSE")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(frameon=False, ncol=4)
    fig.suptitle("Wan VAE Video2Action training and eval loss")
    fig.tight_layout()
    fig.savefig(out_dir / "loss_curve.png")
    plt.close(fig)


def save_checkpoint(
    model: WanVaeActionHead,
    mean: torch.Tensor,
    std: torch.Tensor,
    args: argparse.Namespace,
    out_dir: Path,
    val_metrics: dict[str, float],
    filename: str = "checkpoint.pt",
) -> None:
    payload = {
        "model_state": model.state_dict(),
        "target_mean": mean,
        "target_std": std,
        "model": {
            "latent_channels": 48,
            "head_type": args.head_type,
            "conv_channels": args.conv_channels,
            "conv_blocks": args.conv_blocks,
            "readout_dim": args.readout_dim,
            "hidden_dim": args.hidden_dim,
            "mlp_layers": args.mlp_layers,
            "dropout": args.dropout,
            "action_dim": target_action_dim(args.target_mode),
            "target_mode": args.target_mode,
            "arm_dim": ARM_DIM,
            "hand_dim": HAND_DIM,
        },
        "config": {
            "task": args.task_full,
            "task_short": args.task_short,
            "data_root": args.data_root,
            "segment_root": args.segment_root,
            "resize": args.resize,
            "num_frames": args.num_frames,
            "target_fps": args.target_fps,
            "clip_duration": args.clip_duration,
            "clip_stride": args.clip_stride,
            "split_by": args.split_by,
            "train_ratio": args.train_ratio,
            "arm_loss_weight": args.arm_loss_weight,
            "hand_loss_weight": args.hand_loss_weight,
            "lr_scheduler": args.lr_scheduler,
            "min_lr_ratio": args.min_lr_ratio,
            "target_mode": args.target_mode,
            "action_mask_root": getattr(args, "action_mask_root", None),
            "action_mask_min_frame_ratio": getattr(args, "action_mask_min_frame_ratio", None),
            "empty_action_mask_policy": getattr(args, "empty_action_mask_policy", None),
            "action_dim_names": list(action_dim_names_for_mode(args.target_mode)),
            "action_dim_parts": list(action_dim_parts_for_mode(args.target_mode)),
        },
        "val_metrics": val_metrics,
    }
    torch.save(payload, out_dir / filename)


def load_action_model(checkpoint: Path, device: str) -> tuple[WanVaeActionHead, torch.Tensor, torch.Tensor, dict]:
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model_cfg = ckpt["model"]
    head_type = str(model_cfg.get("head_type", "residual"))
    target_mode = str(model_cfg.get("target_mode", "arm_hand"))
    action_dim = int(model_cfg.get("action_dim", target_action_dim(target_mode)))
    model = WanVaeActionHead(
        latent_channels=int(model_cfg["latent_channels"]),
        action_dim=action_dim,
        target_mode=target_mode,
        head_type=head_type,
        conv_channels=int(model_cfg["conv_channels"]),
        conv_blocks=int(model_cfg["conv_blocks"]),
        readout_dim=int(model_cfg.get("readout_dim", 1024)),
        hidden_dim=int(model_cfg["hidden_dim"]),
        mlp_layers=int(model_cfg["mlp_layers"]),
        dropout=float(model_cfg["dropout"]),
    ).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()
    mean = ckpt["target_mean"].to(device)
    std = ckpt["target_std"].to(device)
    return model, mean, std, ckpt


def record_action_target(
    record: dict,
    resolver: ActionResolver,
    num_frames: int,
    target_fps: float,
    segment_root: Path,
    task_short: str,
    task_full: str,
    target_mode: str,
) -> np.ndarray:
    return record_action_target_for_expected_tasks(
        record,
        resolver,
        num_frames,
        target_fps,
        segment_root,
        {task_short, task_full},
    )


def record_action_target_for_expected_tasks(
    record: dict,
    resolver: ActionResolver,
    num_frames: int,
    target_fps: float,
    segment_root: Path,
    expected_tasks: set[str],
) -> np.ndarray:
    task = record.get("robot_task") or record.get("task")
    if task not in expected_tasks:
        raise ValueError(f"Expected eval task in {sorted(expected_tasks)}, got {task!r}")
    episode_raw = record.get("episode")
    seg = record.get("seg")
    if not episode_raw or not seg:
        raise ValueError(f"Eval record missing episode/seg: {record}")
    episode = int(str(episode_raw).replace("ep", ""))
    joints_path = segment_root / str(episode_raw) / f"{seg}_joints.parquet"
    if not joints_path.is_file():
        raise FileNotFoundError(f"Segment joints not found for eval record: {joints_path}")
    seg_df = pd.read_parquet(joints_path)
    frame_start = int(seg_df["frame_index"].min())
    clip_start = float(record["clip_start"])
    clip_dur = float(record["clip_dur"])
    rel = clip_frame_indices(clip_start, clip_dur, num_frames, target_fps)
    abs_indices = [frame_start + idx for idx in rel]
    return resolver.target_for_indices(episode, abs_indices, target_mode=target_mode)


def record_action_mask(
    record: dict,
    mask_resolver: ActionMaskResolver,
    num_frames: int,
    target_fps: float,
) -> ClipActionMask:
    episode_raw = record.get("episode")
    seg = record.get("seg")
    if not episode_raw or not seg:
        raise ValueError(f"Eval record missing episode/seg: {record}")
    clip_start = float(record["clip_start"])
    clip_dur = float(record["clip_dur"])
    rel = clip_frame_indices(clip_start, clip_dur, num_frames, target_fps)
    return mask_resolver.load_clip(str(episode_raw), str(seg), rel)


@torch.no_grad()
def predict_video_action(
    video_path: Path,
    model: WanVaeActionHead,
    vae,
    mean: torch.Tensor,
    std: torch.Tensor,
    *,
    num_frames: int,
    resize: tuple[int, int] | None,
    device: str,
) -> np.ndarray:
    frames = read_video_uniform_frames(video_path, num_frames, resize)
    latent = encode_batch(vae, [frames], device)
    pred_norm = model(latent)["action"]
    pred = pred_norm * std + mean
    return pred[0].detach().cpu().numpy().astype(np.float32)


def eval_existing(args: argparse.Namespace) -> None:
    resolve_task_args(args)
    resize = parse_resize(args.resize)
    eval_dir = Path(args.eval_dir)
    records_path = Path(args.records_jsonl)
    if not eval_dir.is_dir():
        raise FileNotFoundError(f"Eval dir not found: {eval_dir}")
    if not records_path.is_file():
        raise FileNotFoundError(f"Eval records jsonl not found: {records_path}")
    records = [
        json.loads(line)
        for line in records_path.read_text().splitlines()
        if line.strip()
    ]
    if args.max_samples > 0:
        records = records[: args.max_samples]
    if not records:
        raise ValueError(f"No eval records loaded from {records_path}")

    device = args.device
    model, mean, std, ckpt = load_action_model(Path(args.checkpoint), device)
    ckpt_target_mode = str(ckpt.get("model", {}).get("target_mode", "arm_hand"))
    if args.target_mode != ckpt_target_mode:
        raise ValueError(
            f"--target-mode {args.target_mode!r} does not match checkpoint "
            f"target_mode {ckpt_target_mode!r}"
        )
    vae = load_vae(args.vae_path, torch.bfloat16, home_device=device)
    resolver = ActionResolver(Path(args.data_root))
    segment_root = Path(args.segment_root)
    action_mask_root = resolve_action_mask_root_arg(args)
    mask_resolver = None
    if action_mask_root is not None:
        mask_resolver = ActionMaskResolver(
            action_mask_root,
            args.task_short,
            min_frame_ratio=args.action_mask_min_frame_ratio,
            target_mode=args.target_mode,
        )

    rows = []
    for idx, record in enumerate(records):
        sample_id = f"{idx:05d}"
        target = record_action_target(
            record,
            resolver,
            args.num_frames,
            args.target_fps,
            segment_root,
            args.task_short,
            args.task_full,
            args.target_mode,
        )
        gt_pred = predict_video_action(
            eval_dir / f"gt_{sample_id}.mp4", model, vae, mean, std,
            num_frames=args.num_frames, resize=resize, device=device,
        )
        gen_pred = predict_video_action(
            eval_dir / f"gen_{sample_id}.mp4", model, vae, mean, std,
            num_frames=args.num_frames, resize=resize, device=device,
        )
        clip_mask = None
        if mask_resolver is not None:
            clip_mask = record_action_mask(
                record,
                mask_resolver,
                args.num_frames,
                args.target_fps,
            )
            require_nonempty_clip_mask(sample_id, clip_mask, args.empty_action_mask_policy)
        row = action_metric_row(
            sample_id,
            target,
            gt_pred,
            gen_pred,
            clip_mask=clip_mask,
            target_mode=args.target_mode,
        )
        rows.append(row)
        if idx == 0 or (idx + 1) % args.log_every == 0 or idx + 1 == len(records):
            print(f"eval {idx + 1}/{len(records)} sample={sample_id}", flush=True)

    summary = summarize_action_rows(rows)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    write_eval_csv(rows, output_csv)
    output_json = output_csv.with_suffix(".json")
    output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    _ = ckpt


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return Path(MAIN_ROOT) / path


def parse_labeled_path(value: str) -> tuple[str, Path]:
    if "=" in value:
        label, raw_path = value.split("=", 1)
        label = label.strip()
        if not label:
            raise ValueError(f"Run label must not be empty in {value!r}")
        if not raw_path:
            raise ValueError(f"Run path must not be empty in {value!r}")
        return label, resolve_project_path(raw_path)
    path = resolve_project_path(value)
    return path.name, path


def load_h2r_task_bundles(
    checkpoint_paths: dict[str, Path],
    device: str,
) -> dict[str, H2RTaskBundle]:
    model_cache: dict[str, tuple[WanVaeActionHead, torch.Tensor, torch.Tensor, dict]] = {}
    resolver_cache: dict[str, ActionResolver] = {}
    bundles: dict[str, H2RTaskBundle] = {}
    for record_task, config in H2R_TASK_CONFIGS.items():
        checkpoint = checkpoint_paths[config.checkpoint_key]
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"{config.checkpoint_key} IDM checkpoint not found: {checkpoint}"
            )
        if config.checkpoint_key not in model_cache:
            model_cache[config.checkpoint_key] = load_action_model(checkpoint, device)
        if config.target_task_short not in resolver_cache:
            resolver_cache[config.target_task_short] = ActionResolver(
                default_task_data_root(config.target_task_short)
            )
        model, mean, std, ckpt = model_cache[config.checkpoint_key]
        bundles[record_task] = H2RTaskBundle(
            config=config,
            checkpoint=checkpoint,
            model=model,
            mean=mean,
            std=std,
            ckpt=ckpt,
            resolver=resolver_cache[config.target_task_short],
            segment_root=default_task_segment_root(config.target_task_short),
        )
    return bundles


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"JSONL file not found: {path}")
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def action_part_mse(a: np.ndarray, b: np.ndarray, part: str) -> float:
    if part == "arm":
        return float(np.mean((a[:ARM_DIM] - b[:ARM_DIM]) ** 2))
    if part == "hand":
        return float(np.mean((a[ARM_DIM:] - b[ARM_DIM:]) ** 2))
    if part == "arm_hand":
        return float(np.mean((a - b) ** 2))
    raise ValueError(f"Unknown action part: {part}")


def h2r_action_metric_row(
    *,
    run_label: str,
    run_name: str,
    split: str,
    sample_id: str,
    record: dict,
    canonical_task: str,
    target: np.ndarray,
    gt_pred: np.ndarray,
    gen_pred: np.ndarray,
) -> dict[str, float | int | str]:
    row: dict[str, float | int | str] = {
        "run_label": run_label,
        "run_name": run_name,
        "split": split,
        "sample_id": sample_id,
        "sample_index": int(sample_id),
        "robot_task": str(record.get("robot_task") or record.get("task")),
        "canonical_task": canonical_task,
        "episode": str(record.get("episode", "")),
        "seg": str(record.get("seg", "")),
        "clip_start": float(record["clip_start"]),
        "clip_dur": float(record["clip_dur"]),
        "augment": str(record.get("augment", "")),
        "source_id": str(record.get("source_id", "")),
    }
    for part in ("arm", "hand", "arm_hand"):
        gt_to_target = action_part_mse(gt_pred, target, part)
        gen_to_target = action_part_mse(gen_pred, target, part)
        gen_to_gt = action_part_mse(gen_pred, gt_pred, part)
        row[f"gt_video_to_target_{part}_mse"] = gt_to_target
        row[f"gen_video_to_target_{part}_mse"] = gen_to_target
        row[f"gen_video_to_gt_video_{part}_mse"] = gen_to_gt
        row[f"gt_idm_{part}_mse"] = gt_to_target
        row[f"gen_idm_{part}_mse"] = gen_to_target
        row[f"gen_to_gt_idm_{part}_mse"] = gen_to_gt
        row[f"idm_{part}_gap"] = gen_to_target - gt_to_target
        row[f"idm_{part}_ratio"] = gen_to_target / max(gt_to_target, 1e-12)
    return row


def h2r_action_vector_row(
    *,
    metric_row: dict[str, float | int | str],
    target: np.ndarray,
    gt_pred: np.ndarray,
    gen_pred: np.ndarray,
) -> dict:
    return {
        "run_label": metric_row["run_label"],
        "run_name": metric_row["run_name"],
        "split": metric_row["split"],
        "sample_id": metric_row["sample_id"],
        "robot_task": metric_row["robot_task"],
        "canonical_task": metric_row["canonical_task"],
        "episode": metric_row["episode"],
        "seg": metric_row["seg"],
        "clip_start": metric_row["clip_start"],
        "clip_dur": metric_row["clip_dur"],
        "augment": metric_row["augment"],
        "source_id": metric_row["source_id"],
        "target_action": [float(v) for v in target.tolist()],
        "gt_video_pred_action": [float(v) for v in gt_pred.tolist()],
        "gen_video_pred_action": [float(v) for v in gen_pred.tolist()],
    }


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def summarize_h2r_metric_rows(rows: list[dict[str, float | int | str]]) -> list[dict]:
    if not rows:
        raise ValueError("No H2R metric rows to summarize")
    df = pd.DataFrame(rows)
    group_cols = ["run_label", "run_name", "split", "canonical_task"]
    metric_cols = [
        col for col in df.columns
        if col.endswith("_mse") or col.endswith("_gap") or col.endswith("_ratio")
    ]
    summary_rows = []
    for keys, group in df.groupby(group_cols, sort=True):
        row = dict(zip(group_cols, keys))
        row["n_samples"] = int(len(group))
        row["robot_tasks"] = ",".join(sorted(str(v) for v in group["robot_task"].unique()))
        for col in metric_cols:
            values = group[col].astype(float)
            row[f"{col}_mean"] = float(values.mean())
            row[f"{col}_median"] = float(values.median())
        summary_rows.append(row)
    return summary_rows


def compare_h2r_summary_rows(
    summary_rows: list[dict],
    baseline_label: str,
    ours_label: str,
) -> list[dict]:
    by_key = {
        (row["run_label"], row["split"], row["canonical_task"]): row
        for row in summary_rows
    }
    groups = sorted({(row["split"], row["canonical_task"]) for row in summary_rows})
    compare_rows = []
    for split, canonical_task in groups:
        base_key = (baseline_label, split, canonical_task)
        ours_key = (ours_label, split, canonical_task)
        if base_key not in by_key or ours_key not in by_key:
            continue
        base = by_key[base_key]
        ours = by_key[ours_key]
        row = {
            "baseline_label": baseline_label,
            "ours_label": ours_label,
            "split": split,
            "canonical_task": canonical_task,
            "baseline_n_samples": int(base["n_samples"]),
            "ours_n_samples": int(ours["n_samples"]),
        }
        metric_cols = sorted(
            col for col in base
            if col.endswith("_mean") and isinstance(base[col], (float, int))
        )
        for col in metric_cols:
            base_value = float(base[col])
            ours_value = float(ours[col])
            row[f"baseline_{col}"] = base_value
            row[f"ours_{col}"] = ours_value
            row[f"delta_{col}"] = ours_value - base_value
        compare_rows.append(row)
    return compare_rows


def write_dict_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def eval_h2r_runs(args: argparse.Namespace) -> None:
    if not args.run:
        raise ValueError("eval-h2r requires at least one --run LABEL=PATH")
    run_specs = [parse_labeled_path(value) for value in args.run]
    duplicate_labels = {
        label for label in [label for label, _ in run_specs]
        if [item[0] for item in run_specs].count(label) > 1
    }
    if duplicate_labels:
        raise ValueError(f"Duplicate run labels: {sorted(duplicate_labels)}")

    checkpoint_paths = {
        "collect": resolve_project_path(args.collect_checkpoint),
        "wash": resolve_project_path(args.wash_checkpoint),
        "pillow": resolve_project_path(args.pillow_checkpoint),
    }
    output_dir = resolve_project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    resize = parse_resize(args.resize)
    device = args.device
    bundles = load_h2r_task_bundles(checkpoint_paths, device)
    vae = load_vae(args.vae_path, torch.bfloat16, home_device=device)

    all_metric_rows: list[dict[str, float | int | str]] = []
    config = {
        "runs": [
            {"label": label, "path": str(path), "name": path.name}
            for label, path in run_specs
        ],
        "splits": args.splits,
        "augment_filter": "normal",
        "max_samples_per_split": args.max_samples_per_split,
        "checkpoints": {key: str(path) for key, path in checkpoint_paths.items()},
        "task_configs": {key: asdict(value) for key, value in H2R_TASK_CONFIGS.items()},
        "resize": args.resize,
        "num_frames": args.num_frames,
        "target_fps": args.target_fps,
        "vae_path": args.vae_path,
    }
    (output_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2))

    for run_label, run_path in run_specs:
        full_eval_dir = run_path / "full_eval"
        if not full_eval_dir.is_dir():
            raise FileNotFoundError(f"full_eval dir not found: {full_eval_dir}")
        run_dir = output_dir / run_path.name
        run_dir.mkdir(parents=True, exist_ok=True)
        run_metric_rows: list[dict[str, float | int | str]] = []
        run_vector_rows: list[dict] = []
        for split in args.splits:
            eval_dir = full_eval_dir / split
            records_path = full_eval_dir / "data_split" / f"{split}.jsonl"
            if not eval_dir.is_dir():
                raise FileNotFoundError(f"Eval split dir not found: {eval_dir}")
            raw_records = read_jsonl(records_path)
            records = [
                (idx, record)
                for idx, record in enumerate(raw_records)
                if str(record.get("augment", "normal")) == "normal"
            ]
            skipped = len(raw_records) - len(records)
            if not records:
                raise ValueError(
                    f"No normal-augment records loaded from {records_path}; skipped {skipped}"
                )
            if args.max_samples_per_split > 0:
                records = records[: args.max_samples_per_split]
            if not records:
                raise ValueError(f"No records loaded from {records_path}")
            if skipped > 0:
                print(
                    f"{run_label}/{split} skipped_non_normal={skipped} "
                    f"used={len(records)}",
                    flush=True,
                )

            for idx, record in records:
                record_task = str(record.get("robot_task") or record.get("task"))
                if record_task not in bundles:
                    raise ValueError(
                        f"Unexpected eval task {record_task!r} in {records_path}; "
                        f"allowed tasks are {sorted(bundles)}"
                    )
                bundle = bundles[record_task]
                sample_id = f"{idx:05d}"
                target = record_action_target_for_expected_tasks(
                    record,
                    bundle.resolver,
                    args.num_frames,
                    args.target_fps,
                    bundle.segment_root,
                    {record_task},
                )
                gt_path = eval_dir / f"gt_{sample_id}.mp4"
                gen_path = eval_dir / f"gen_{sample_id}.mp4"
                if not gt_path.is_file():
                    raise FileNotFoundError(f"GT eval video not found: {gt_path}")
                if not gen_path.is_file():
                    raise FileNotFoundError(f"Generated eval video not found: {gen_path}")
                gt_pred = predict_video_action(
                    gt_path, bundle.model, vae, bundle.mean, bundle.std,
                    num_frames=args.num_frames, resize=resize, device=device,
                )
                gen_pred = predict_video_action(
                    gen_path, bundle.model, vae, bundle.mean, bundle.std,
                    num_frames=args.num_frames, resize=resize, device=device,
                )
                metric_row = h2r_action_metric_row(
                    run_label=run_label,
                    run_name=run_path.name,
                    split=split,
                    sample_id=sample_id,
                    record=record,
                    canonical_task=bundle.config.canonical_task,
                    target=target,
                    gt_pred=gt_pred,
                    gen_pred=gen_pred,
                )
                run_metric_rows.append(metric_row)
                run_vector_rows.append(
                    h2r_action_vector_row(
                        metric_row=metric_row,
                        target=target,
                        gt_pred=gt_pred,
                        gen_pred=gen_pred,
                    )
                )
                n_done = idx + 1
                if (
                    (len(records) > 0 and idx == records[0][0])
                    or n_done % args.log_every == 0
                    or idx == records[-1][0]
                ):
                    print(
                        f"{run_label}/{split} {n_done}/{len(raw_records)} "
                        f"task={record_task} sample={sample_id}",
                        flush=True,
                    )
        write_dict_csv(run_metric_rows, run_dir / "per_sample_metrics.csv")
        write_jsonl(run_vector_rows, run_dir / "per_sample_actions.jsonl")
        run_summary = summarize_h2r_metric_rows(run_metric_rows)
        write_dict_csv(run_summary, run_dir / "summary_by_task.csv")
        (run_dir / "config.json").write_text(
            json.dumps({**config, "run_label": run_label, "run_path": str(run_path)},
                       ensure_ascii=False, indent=2)
        )
        all_metric_rows.extend(run_metric_rows)

    write_dict_csv(all_metric_rows, output_dir / "per_sample_metrics.csv")
    summary_rows = summarize_h2r_metric_rows(all_metric_rows)
    write_dict_csv(summary_rows, output_dir / "summary_by_task.csv")
    compare_rows = compare_h2r_summary_rows(
        summary_rows,
        args.baseline_label,
        args.ours_label,
    )
    write_dict_csv(compare_rows, output_dir / "summary_compare_baseline_ours.csv")
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "n_samples": len(all_metric_rows),
                "n_summary_rows": len(summary_rows),
                "n_compare_rows": len(compare_rows),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


@torch.no_grad()
def validate_checkpoint(args: argparse.Namespace) -> None:
    resolve_task_args(args)
    seed = int(args.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    resize = parse_resize(args.resize)
    target_mode = validate_target_mode(args.target_mode)
    resolver = ActionResolver(Path(args.data_root))
    action_mask_root = resolve_action_mask_root_arg(args)
    samples = discover_segment_clips(
        resolver,
        Path(args.segment_root),
        max_samples=args.max_samples,
        clip_dur=args.clip_duration,
        clip_stride=args.clip_stride,
        num_frames=args.num_frames,
        target_fps=args.target_fps,
        seed=seed,
        task_short=args.task_short,
        target_mode=target_mode,
        action_mask_root=action_mask_root,
        action_mask_min_frame_ratio=args.action_mask_min_frame_ratio,
        empty_action_mask_policy=args.empty_action_mask_policy,
    )
    _, val_samples = split_samples(samples, args.train_ratio, args.split_by, seed)

    device = args.device
    model, mean, std, ckpt = load_action_model(Path(args.checkpoint), device)
    ckpt_target_mode = str(ckpt.get("model", {}).get("target_mode", "arm_hand"))
    if args.target_mode != ckpt_target_mode:
        raise ValueError(
            f"--target-mode {args.target_mode!r} does not match checkpoint "
            f"target_mode {ckpt_target_mode!r}"
        )
    vae = load_vae(args.vae_path, torch.bfloat16, home_device=device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = validate(
        model,
        vae,
        val_samples,
        resize,
        mean,
        std,
        device,
        args,
        prediction_path=out_dir / "val_predictions.csv",
    )
    (out_dir / "val_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2)
    )
    print(json.dumps({"val": metrics, "out_dir": str(out_dir)}, indent=2), flush=True)
    _ = ckpt


def action_metric_row(
    sample_id: str,
    target: np.ndarray,
    gt_pred: np.ndarray,
    gen_pred: np.ndarray,
    *,
    clip_mask: ClipActionMask | None = None,
    target_mode: str = "arm_hand",
) -> dict[str, float | str]:
    def mse(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.mean((a - b) ** 2))

    def masked_mse(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
        visible = mask.astype(bool)
        if int(visible.sum()) == 0:
            raise ValueError(f"Empty action mask for eval sample {sample_id}")
        return float(np.mean((a[visible] - b[visible]) ** 2))

    slices = target_action_slices(target_mode)
    action_dim = target_action_dim(target_mode)
    row: dict[str, float | str] = {"sample_id": sample_id}
    if target_mode == "arm_hand":
        eval_parts = {
            "arm": slices["arm"],
            "hand": slices["hand"],
            "arm_hand": slices["total"],
        }
    else:
        eval_parts = slices
    for part, slc in eval_parts.items():
        row[f"gt_idm_{part}_mse"] = mse(gt_pred[slc], target[slc])
        row[f"gen_idm_{part}_mse"] = mse(gen_pred[slc], target[slc])
    for part in eval_parts:
        gt_key = f"gt_idm_{part}_mse"
        gen_key = f"gen_idm_{part}_mse"
        gap = float(row[gen_key]) - float(row[gt_key])
        ratio = float(row[gen_key]) / max(float(row[gt_key]), 1e-12)
        row[f"idm_{part}_gap"] = gap
        row[f"idm_{part}_ratio"] = ratio
    if clip_mask is not None:
        mask = clip_mask.mask.astype(bool)
        row.update({
            "visible_action_count": clip_mask.visible_action_count,
            "visible_arm_count": clip_mask.visible_arm_count,
            "visible_hand_count": clip_mask.visible_hand_count,
            "visible_action_ratio": clip_mask.visible_action_ratio,
            "visible_arm_ratio": clip_mask.visible_arm_ratio,
            "visible_hand_ratio": clip_mask.visible_hand_ratio,
            "action_mask_path": clip_mask.mask_path,
        })
        for part, slc in eval_parts.items():
            part_mask = mask[slc]
            if int(part_mask.sum()) > 0:
                row[f"gt_idm_masked_{part}_mse"] = masked_mse(
                    gt_pred[slc], target[slc], part_mask,
                )
                row[f"gen_idm_masked_{part}_mse"] = masked_mse(
                    gen_pred[slc], target[slc], part_mask,
                )
            else:
                row[f"gt_idm_masked_{part}_mse"] = float("nan")
                row[f"gen_idm_masked_{part}_mse"] = float("nan")
        for part in eval_parts:
            gt_key = f"gt_idm_masked_{part}_mse"
            gen_key = f"gen_idm_masked_{part}_mse"
            gap = float(row[gen_key]) - float(row[gt_key])
            ratio = float(row[gen_key]) / max(float(row[gt_key]), 1e-12)
            row[f"idm_masked_{part}_gap"] = gap
            row[f"idm_masked_{part}_ratio"] = ratio
        for dim in range(action_dim):
            row[f"mask_{dim:02d}"] = float(mask[dim])
            row[f"mask_ratio_{dim:02d}"] = float(clip_mask.frame_ratios[dim])
    return row


def summarize_action_rows(rows: list[dict[str, float | str]]) -> dict[str, float | int]:
    keys = [
        key for key, value in rows[0].items()
        if key != "sample_id" and isinstance(value, (int, float, np.integer, np.floating))
    ]
    summary: dict[str, float | int] = {"n_samples": len(rows)}
    for key in keys:
        values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        summary[key] = float(np.nanmean(values))
    return summary


def write_eval_csv(rows: list[dict[str, float | str]], path: Path) -> None:
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Wan VAE Video2Action IDM")
    sub = parser.add_subparsers(dest="cmd", required=True)

    train_p = sub.add_parser("train", help="train IDM on one MainCamOnly task")
    add_common_data_args(train_p)
    train_p.add_argument(
        "--output-dir",
        default=None,
        help="output directory; defaults to output/wan_vae_idm/<task-short>",
    )
    train_p.add_argument("--max-samples", type=int, default=64,
                         help="maximum discovered clip samples; 0 keeps all")
    train_p.add_argument("--train-ratio", type=float, default=0.8)
    train_p.add_argument("--split-by", choices=["episode", "sample"], default="episode",
                         help="episode keeps held-out episodes out of training")
    train_p.add_argument("--clip-duration", type=float, default=1.0)
    train_p.add_argument("--clip-stride", type=float, default=1.0)
    train_p.add_argument("--steps", type=int, default=80)
    train_p.add_argument("--batch-size", type=int, default=1)
    train_p.add_argument("--workers", type=int, default=0)
    train_p.add_argument("--head-type", choices=["cnn_mlp"], default="cnn_mlp")
    train_p.add_argument("--conv-channels", type=int, default=256)
    train_p.add_argument("--conv-blocks", type=int, default=4,
                         help="3D CNN blocks; second block downsamples Wan VAE 16x16 latent to 8x8")
    train_p.add_argument("--readout-dim", type=int, default=1024)
    train_p.add_argument("--hidden-dim", type=int, default=1024)
    train_p.add_argument("--mlp-layers", type=int, default=3)
    train_p.add_argument("--dropout", type=float, default=0.0)
    train_p.add_argument("--lr", type=float, default=1e-3)
    train_p.add_argument("--weight-decay", type=float, default=1e-4)
    train_p.add_argument("--lr-scheduler", choices=["none", "cosine"], default="none")
    train_p.add_argument("--min-lr-ratio", type=float, default=0.05,
                         help="cosine scheduler final lr ratio relative to --lr")
    train_p.add_argument("--arm-loss-weight", type=float, default=1.0)
    train_p.add_argument("--hand-loss-weight", type=float, default=1.0)
    train_p.add_argument("--val-max-samples", type=int, default=16,
                         help="maximum held-out samples for eval; <=0 evaluates all")
    train_p.add_argument("--seed", type=int, default=42)
    train_p.add_argument("--log-every", type=int, default=10)
    train_p.add_argument("--eval-every", type=int, default=200,
                         help="run held-out eval every N train steps; <=0 disables periodic eval")
    train_p.set_defaults(func=train)

    val_p = sub.add_parser("validate", help="validate a trained IDM checkpoint on held-out task clips")
    add_common_data_args(val_p)
    val_p.add_argument("--checkpoint", required=True)
    val_p.add_argument("--output-dir", required=True)
    val_p.add_argument("--max-samples", type=int, default=0,
                       help="maximum discovered clip samples; 0 keeps all")
    val_p.add_argument("--train-ratio", type=float, default=0.8)
    val_p.add_argument("--split-by", choices=["episode", "sample"], default="episode")
    val_p.add_argument("--clip-duration", type=float, default=1.0)
    val_p.add_argument("--clip-stride", type=float, default=1.0)
    val_p.add_argument("--batch-size", type=int, default=4)
    val_p.add_argument("--workers", type=int, default=2)
    val_p.add_argument("--val-max-samples", type=int, default=0,
                       help="maximum held-out samples for eval; <=0 evaluates all")
    val_p.add_argument("--seed", type=int, default=42)
    val_p.set_defaults(func=validate_checkpoint)

    eval_p = sub.add_parser("eval", help="evaluate existing gen/gt videos with a trained IDM")
    add_common_data_args(eval_p)
    eval_p.add_argument("--checkpoint", required=True)
    eval_p.add_argument("--eval-dir", required=True,
                        help="directory containing gen_00000.mp4 and gt_00000.mp4")
    eval_p.add_argument("--records-jsonl", required=True,
                        help="matching data_split jsonl for eval-dir")
    eval_p.add_argument("--output-csv", required=True)
    eval_p.add_argument("--max-samples", type=int, default=0)
    eval_p.add_argument("--log-every", type=int, default=8)
    eval_p.set_defaults(func=eval_existing)

    h2r_p = sub.add_parser(
        "eval-h2r",
        help="evaluate Baseline/Ours H2R full_eval videos with task-specific IDM checkpoints",
    )
    h2r_p.add_argument(
        "--run",
        action="append",
        default=[],
        help="run label and path as LABEL=training_data/log/<run>; repeat for Baseline and Ours",
    )
    h2r_p.add_argument("--baseline-label", default="Baseline")
    h2r_p.add_argument("--ours-label", default="Ours")
    h2r_p.add_argument("--collect-checkpoint", required=True)
    h2r_p.add_argument("--wash-checkpoint", required=True)
    h2r_p.add_argument("--pillow-checkpoint", required=True)
    h2r_p.add_argument(
        "--output-dir",
        default="output/idm_h2r_action_eval",
        help="root output directory for per-run metrics and Baseline/Ours summary",
    )
    h2r_p.add_argument(
        "--splits",
        nargs="+",
        default=["in_task_eval", "ood_eval"],
        help="full_eval split names to process",
    )
    h2r_p.add_argument("--max-samples-per-split", type=int, default=0)
    h2r_p.add_argument("--vae-path", default=DEFAULT_VAE)
    h2r_p.add_argument("--device", default="cuda:0")
    h2r_p.add_argument("--resize", default="256x256",
                       help="WIDTHxHEIGHT for Wan VAE input, or 'native'")
    h2r_p.add_argument("--num-frames", type=int, default=DEFAULT_NUM_FRAMES)
    h2r_p.add_argument("--target-fps", type=float, default=DEFAULT_TARGET_FPS)
    h2r_p.add_argument("--log-every", type=int, default=8)
    h2r_p.set_defaults(func=eval_h2r_runs)
    return parser


def add_common_data_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--task-short", default=DEFAULT_TASK_SHORT,
                        help="segment task name, for example Inspire_Collect_Clothes_MainCamOnly")
    parser.add_argument("--task-full", default=None,
                        help="raw dataset task dir; defaults to G1_WBT_<task-short>")
    parser.add_argument("--data-root", default=None,
                        help="raw LeRobot task root; defaults from --task-full")
    parser.add_argument("--segment-root", default=None,
                        help="segmented task root; defaults from --task-short")
    parser.add_argument("--vae-path", default=DEFAULT_VAE)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resize", default="256x256",
                        help="WIDTHxHEIGHT for Wan VAE input, or 'native'")
    parser.add_argument("--num-frames", type=int, default=DEFAULT_NUM_FRAMES)
    parser.add_argument("--target-fps", type=float, default=DEFAULT_TARGET_FPS)
    parser.add_argument(
        "--target-mode",
        choices=list(TARGET_MODES),
        default="arm_hand",
        help=(
            "arm_hand predicts action.ee_action + action.hand_cmd (24 dims); "
            "full_body predicts action.robot_q_desired + action.hand_cmd (48 dims)"
        ),
    )
    parser.add_argument(
        "--action-mask-root",
        default=None,
        help=(
            "optional visible action mask root; use 'default' for "
            "training_data/action_mask. If set, missing or mismatched masks error."
        ),
    )
    parser.add_argument(
        "--action-mask-min-frame-ratio",
        type=float,
        default=0.25,
        help="clip dimension is visible when this ratio of selected frames is visible",
    )
    parser.add_argument(
        "--empty-action-mask-policy",
        choices=["error", "drop"],
        default="error",
        help="error on clips with zero visible dimensions, or drop them during discovery",
    )


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
