"""Two-frame inverse dynamics model for arm and hand action consistency.

This module trains a pair-level IDM on raw RGB frames instead of Wan VAE
latents.  Each sample uses adjacent segment frames ``(s_t, s_{t+1})`` and
predicts the action recorded at frame ``t``:

  - ``action.ee_action`` for a 12-dim arm action network
  - ``action.hand_cmd`` for a separate 12-dim hand action network

The arm and hand networks share the same data loader and optimizer step but do
not share parameters.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
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

DEFAULT_TASK_SHORT = "Inspire_Collect_Clothes_MainCamOnly"
ARM_DIM = 12
HAND_DIM = 12
ACTION_DIM = ARM_DIM + HAND_DIM


@dataclass(frozen=True)
class PairSample:
    video_path: str
    episode: int
    episode_name: str
    segment: str
    frame_start: int
    rel_frame_t: int
    abs_frame_t: int
    abs_frame_tp1: int
    arm_target: tuple[float, ...]
    hand_target: tuple[float, ...]


def parse_resize(value: str) -> tuple[int, int] | None:
    if value.lower() in {"none", "native", "0"}:
        return None
    parts = value.lower().split("x")
    if len(parts) != 2:
        raise ValueError(f"resize must be WIDTHxHEIGHT or none, got {value!r}")
    width, height = int(parts[0]), int(parts[1])
    if width <= 0 or height <= 0:
        raise ValueError(f"resize dimensions must be positive, got {value!r}")
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
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Action parquet missing required columns {missing}: {data_root}")
    return df


class ActionResolver:
    """Resolve ``a_t`` arm and hand labels from raw LeRobot parquet."""

    def __init__(self, data_root: Path):
        self.data_root = data_root
        self.df = load_action_frame_table(data_root)
        self._episode_rows = {
            int(ep): group.set_index("frame_index", drop=False)
            for ep, group in self.df.groupby("episode_index", sort=False)
        }

    def target_for_frame(self, episode: int, frame_index: int) -> tuple[np.ndarray, np.ndarray]:
        if episode not in self._episode_rows:
            raise ValueError(f"Episode {episode} not found in {self.data_root}")
        rows = self._episode_rows[episode]
        if frame_index not in rows.index:
            raise ValueError(
                f"Missing action frame episode={episode} frame_index={frame_index}"
            )
        row = rows.loc[frame_index]
        arm = np.asarray(row["action.ee_action"], dtype=np.float32)
        hand = np.asarray(row["action.hand_cmd"], dtype=np.float32)
        if arm.shape != (ARM_DIM,) or hand.shape != (HAND_DIM,):
            raise ValueError(
                f"Bad action shape at episode={episode} frame={frame_index}: "
                f"arm={arm.shape}, hand={hand.shape}"
            )
        return arm, hand


def discover_segment_pairs(
    resolver: ActionResolver,
    segment_root: Path,
    *,
    max_samples: int,
    seed: int,
) -> list[PairSample]:
    if not segment_root.is_dir():
        raise FileNotFoundError(f"Segment root not found: {segment_root}")
    samples: list[PairSample] = []
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
        frame_indices = np.sort(seg_df["frame_index"].to_numpy(dtype=np.int64))
        frame_start = int(frame_indices[0])
        frame_end = int(frame_indices[-1])
        frame_count = frame_end - frame_start + 1
        expected_indices = np.arange(frame_start, frame_end + 1, dtype=np.int64)
        if not np.array_equal(frame_indices, expected_indices):
            raise ValueError(
                f"Segment frame_index must be contiguous for video/action alignment: "
                f"{joints_path}"
            )
        if frame_count < 2:
            continue
        video_path = joints_path.with_name(
            joints_path.name.replace("_joints.parquet", "_video.mp4")
        )
        if not video_path.is_file():
            raise FileNotFoundError(f"Segment video not found: {video_path}")
        segment = joints_path.stem.replace("_joints", "")
        episode_name = joints_path.parent.name
        for rel_t in range(frame_count - 1):
            abs_t = frame_start + rel_t
            arm, hand = resolver.target_for_frame(episode, abs_t)
            samples.append(
                PairSample(
                    video_path=str(video_path),
                    episode=episode,
                    episode_name=episode_name,
                    segment=segment,
                    frame_start=frame_start,
                    rel_frame_t=rel_t,
                    abs_frame_t=abs_t,
                    abs_frame_tp1=abs_t + 1,
                    arm_target=tuple(float(v) for v in arm.tolist()),
                    hand_target=tuple(float(v) for v in hand.tolist()),
                )
            )
    random.Random(seed).shuffle(samples)
    if max_samples > 0:
        samples = samples[:max_samples]
    if not samples:
        raise ValueError(f"No adjacent frame pairs discovered under {segment_root}")
    return samples


def read_video_pair_frames(
    path: str | Path,
    rel_frame_t: int,
    resize: tuple[int, int] | None,
    rel_frame_tp1: int | None = None,
) -> np.ndarray:
    if rel_frame_t < 0:
        raise ValueError(f"rel_frame_t must be non-negative, got {rel_frame_t}")
    if rel_frame_tp1 is None:
        rel_frame_tp1 = rel_frame_t + 1
    if rel_frame_tp1 <= rel_frame_t:
        raise ValueError(
            f"rel_frame_tp1 must be greater than rel_frame_t, got "
            f"{rel_frame_tp1} <= {rel_frame_t}"
        )
    video_path = str(path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    frames = []
    for expected in (rel_frame_t, rel_frame_tp1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, expected)
        ok, frame_bgr = cap.read()
        if not ok:
            cap.release()
            raise RuntimeError(f"Failed to read frame {expected} from {video_path}")
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        if resize is not None:
            frame_rgb = cv2.resize(frame_rgb, resize, interpolation=cv2.INTER_AREA)
        frames.append(frame_rgb)
    cap.release()
    return np.stack(frames, axis=0).astype(np.uint8)


class PairDataset(torch.utils.data.Dataset):
    def __init__(self, samples: list[PairSample], resize: tuple[int, int] | None):
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
            "arm_target": np.asarray(sample.arm_target, dtype=np.float32),
            "hand_target": np.asarray(sample.hand_target, dtype=np.float32),
            "sample_index": idx,
        }


def collate_pair_batch(items: list[dict]) -> dict:
    frames_np = np.stack([item["frames"] for item in items], axis=0)
    frames = torch.from_numpy(frames_np).to(dtype=torch.float32)
    frames = frames.permute(0, 1, 4, 2, 3).contiguous()
    b, t, c, h, w = frames.shape
    if t != 2 or c != 3:
        raise ValueError(f"Expected pair frames [B,2,3,H,W], got {frames.shape}")
    frames = frames.view(b, t * c, h, w).mul_(2.0 / 255.0).sub_(1.0)
    return {
        "frames": frames,
        "arm_target": torch.from_numpy(
            np.stack([item["arm_target"] for item in items], axis=0)
        ),
        "hand_target": torch.from_numpy(
            np.stack([item["hand_target"] for item in items], axis=0)
        ),
        "sample_index": [item["sample_index"] for item in items],
    }


class SmallPairCnn(nn.Module):
    def __init__(
        self,
        output_dim: int,
        *,
        input_channels: int = 6,
        base_channels: int = 32,
        hidden_dim: int = 128,
        dropout: float = 0.0,
    ):
        super().__init__()
        if output_dim <= 0:
            raise ValueError(f"output_dim must be positive, got {output_dim}")
        if input_channels <= 0:
            raise ValueError(f"input_channels must be positive, got {input_channels}")
        if base_channels <= 0:
            raise ValueError(f"base_channels must be positive, got {base_channels}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if dropout < 0.0:
            raise ValueError(f"dropout must be non-negative, got {dropout}")
        self.output_dim = int(output_dim)
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, c1, kernel_size=5, stride=2, padding=2),
            nn.SiLU(),
            nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(c2, c3, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(c3, c3, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.head = nn.Sequential(
            nn.Linear(c3, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.ndim != 4 or frames.shape[1] != 6:
            raise ValueError(f"Expected [B,6,H,W] frame pairs, got {frames.shape}")
        return self.head(self.encoder(frames.float()))


@dataclass
class PairIdmBundle:
    arm_net: SmallPairCnn
    hand_net: SmallPairCnn


def split_samples(
    samples: list[PairSample],
    train_ratio: float,
    split_by: str,
    seed: int,
) -> tuple[list[PairSample], list[PairSample]]:
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


def target_stats(samples: list[PairSample]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    arm_arr = np.asarray([sample.arm_target for sample in samples], dtype=np.float32)
    hand_arr = np.asarray([sample.hand_target for sample in samples], dtype=np.float32)
    arm_mean = arm_arr.mean(axis=0)
    hand_mean = hand_arr.mean(axis=0)
    arm_std = np.maximum(arm_arr.std(axis=0), 1e-4)
    hand_std = np.maximum(hand_arr.std(axis=0), 1e-4)
    return (
        torch.from_numpy(arm_mean),
        torch.from_numpy(arm_std),
        torch.from_numpy(hand_mean),
        torch.from_numpy(hand_std),
    )


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def tensor_float(value: torch.Tensor) -> float:
    return float(value.detach().cpu())


def weighted_loss(
    arm_loss: torch.Tensor,
    hand_loss: torch.Tensor,
    arm_weight: float,
    hand_weight: float,
) -> torch.Tensor:
    if arm_weight <= 0.0 or hand_weight <= 0.0:
        raise ValueError(
            f"loss weights must be positive, got arm={arm_weight}, hand={hand_weight}"
        )
    return (arm_loss * arm_weight + hand_loss * hand_weight) / (arm_weight + hand_weight)


def make_models(args: argparse.Namespace, device: str) -> PairIdmBundle:
    arm_net = SmallPairCnn(
        ARM_DIM,
        base_channels=args.base_channels,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)
    hand_net = SmallPairCnn(
        HAND_DIM,
        base_channels=args.base_channels,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)
    return PairIdmBundle(arm_net=arm_net, hand_net=hand_net)


def write_rows(rows: list[dict], path: Path) -> None:
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


def write_json(path: Path, data: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2))


def mse_np(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean((pred - target) ** 2))


def _valid_mean(values: np.ndarray, valid: np.ndarray) -> float:
    if values.shape != valid.shape:
        raise ValueError(f"valid mask shape {valid.shape} does not match values {values.shape}")
    if int(valid.sum()) == 0:
        return 0.0
    return float(values[valid].mean())


def action_regression_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    *,
    prefix: str,
    dim_prefix: str | None = None,
) -> dict[str, float]:
    if pred.shape != target.shape:
        raise ValueError(f"prediction shape {pred.shape} does not match target {target.shape}")
    if pred.ndim != 2:
        raise ValueError(f"Expected [N,D] predictions, got {pred.shape}")
    if mean.shape != (pred.shape[1],) or std.shape != (pred.shape[1],):
        raise ValueError(
            f"Bad normalization shape for {prefix}: mean={mean.shape} std={std.shape} "
            f"expected {(pred.shape[1],)}"
        )
    if np.any(std <= 0.0):
        raise ValueError(f"{prefix} normalization std must be positive")

    err = pred - target
    norm_err = err / std[None, :]
    dim_mse = np.mean(np.square(err), axis=0)
    dim_norm_mse = np.mean(np.square(norm_err), axis=0)
    pred_std = pred.std(axis=0)
    target_std = target.std(axis=0)

    target_centered = target - target.mean(axis=0, keepdims=True)
    pred_centered = pred - pred.mean(axis=0, keepdims=True)
    target_var_sse = np.square(target_centered).sum(axis=0)
    pred_var_sse = np.square(pred_centered).sum(axis=0)
    model_sse = np.square(err).sum(axis=0)

    eps = 1e-12
    r2_valid = target_var_sse > eps
    r2 = np.zeros_like(dim_mse, dtype=np.float64)
    r2[r2_valid] = 1.0 - model_sse[r2_valid] / target_var_sse[r2_valid]

    corr_denom = np.sqrt(pred_var_sse * target_var_sse)
    corr_valid = corr_denom > eps
    corr = np.zeros_like(dim_mse, dtype=np.float64)
    corr[corr_valid] = (pred_centered * target_centered).sum(axis=0)[corr_valid] / corr_denom[corr_valid]

    std_ratio_valid = target_std > eps
    std_ratio = np.zeros_like(dim_mse, dtype=np.float64)
    std_ratio[std_ratio_valid] = pred_std[std_ratio_valid] / target_std[std_ratio_valid]

    metrics: dict[str, float] = {
        f"{prefix}_norm_mse": float(np.mean(np.square(norm_err))),
        f"{prefix}_mean_dim_norm_mse": float(dim_norm_mse.mean()),
        f"{prefix}_mean_dim_r2": _valid_mean(r2, r2_valid),
        f"{prefix}_r2_valid_dims": float(r2_valid.sum()),
        f"{prefix}_mean_dim_corr": _valid_mean(corr, corr_valid),
        f"{prefix}_corr_valid_dims": float(corr_valid.sum()),
        f"{prefix}_pred_std_ratio_mean": _valid_mean(std_ratio, std_ratio_valid),
        f"{prefix}_pred_std_ratio_valid_dims": float(std_ratio_valid.sum()),
    }
    if dim_prefix is not None:
        for dim in range(pred.shape[1]):
            key = f"{dim_prefix}_{dim:02d}"
            metrics[f"{key}_mse"] = float(dim_mse[dim])
            metrics[f"{key}_norm_mse"] = float(dim_norm_mse[dim])
            metrics[f"{key}_pred_std"] = float(pred_std[dim])
            metrics[f"{key}_target_std"] = float(target_std[dim])
            if std_ratio_valid[dim]:
                metrics[f"{key}_pred_std_ratio"] = float(std_ratio[dim])
            if r2_valid[dim]:
                metrics[f"{key}_r2"] = float(r2[dim])
            if corr_valid[dim]:
                metrics[f"{key}_corr"] = float(corr[dim])
    return metrics


@torch.no_grad()
def validate_samples(
    bundle: PairIdmBundle,
    samples: list[PairSample],
    resize: tuple[int, int] | None,
    arm_mean: torch.Tensor,
    arm_std: torch.Tensor,
    hand_mean: torch.Tensor,
    hand_std: torch.Tensor,
    device: str,
    args: argparse.Namespace,
    *,
    prediction_path: Path | None,
) -> dict[str, float]:
    if not samples:
        raise ValueError("validate_samples received no samples")
    bundle.arm_net.eval()
    bundle.hand_net.eval()
    subset = samples if args.val_max_samples <= 0 else samples[: args.val_max_samples]
    loader = torch.utils.data.DataLoader(
        PairDataset(subset, resize),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate_pair_batch,
        drop_last=False,
    )
    arm_pred_chunks = []
    hand_pred_chunks = []
    arm_target_chunks = []
    hand_target_chunks = []
    pred_rows = []
    for batch in loader:
        frames = batch["frames"].to(device)
        arm_target = batch["arm_target"].to(device)
        hand_target = batch["hand_target"].to(device)
        arm_pred = bundle.arm_net(frames) * arm_std + arm_mean
        hand_pred = bundle.hand_net(frames) * hand_std + hand_mean
        arm_pred_cpu = arm_pred.detach().cpu().numpy()
        hand_pred_cpu = hand_pred.detach().cpu().numpy()
        arm_target_cpu = arm_target.detach().cpu().numpy()
        hand_target_cpu = hand_target.detach().cpu().numpy()
        arm_pred_chunks.append(arm_pred_cpu)
        hand_pred_chunks.append(hand_pred_cpu)
        arm_target_chunks.append(arm_target_cpu)
        hand_target_chunks.append(hand_target_cpu)
        for local_idx, sample_idx in enumerate(batch["sample_index"]):
            sample = subset[int(sample_idx)]
            row = {
                "sample_index": int(sample_idx),
                "episode": sample.episode,
                "episode_name": sample.episode_name,
                "segment": sample.segment,
                "rel_frame_t": sample.rel_frame_t,
                "abs_frame_t": sample.abs_frame_t,
                "abs_frame_tp1": sample.abs_frame_tp1,
                "video_path": sample.video_path,
            }
            for dim in range(ARM_DIM):
                row[f"arm_target_{dim:02d}"] = float(arm_target_cpu[local_idx, dim])
                row[f"arm_pred_{dim:02d}"] = float(arm_pred_cpu[local_idx, dim])
                row[f"arm_err_{dim:02d}"] = float(
                    arm_pred_cpu[local_idx, dim] - arm_target_cpu[local_idx, dim]
                )
            for dim in range(HAND_DIM):
                row[f"hand_target_{dim:02d}"] = float(hand_target_cpu[local_idx, dim])
                row[f"hand_pred_{dim:02d}"] = float(hand_pred_cpu[local_idx, dim])
                row[f"hand_err_{dim:02d}"] = float(
                    hand_pred_cpu[local_idx, dim] - hand_target_cpu[local_idx, dim]
                )
            pred_rows.append(row)
    bundle.arm_net.train()
    bundle.hand_net.train()
    arm_pred_all = np.concatenate(arm_pred_chunks, axis=0)
    hand_pred_all = np.concatenate(hand_pred_chunks, axis=0)
    arm_target_all = np.concatenate(arm_target_chunks, axis=0)
    hand_target_all = np.concatenate(hand_target_chunks, axis=0)
    arm_mean_np = arm_mean.detach().cpu().numpy()
    hand_mean_np = hand_mean.detach().cpu().numpy()
    arm_std_np = arm_std.detach().cpu().numpy()
    hand_std_np = hand_std.detach().cpu().numpy()
    arm_baseline = arm_mean_np[None, :]
    hand_baseline = hand_mean_np[None, :]
    arm_mse = mse_np(arm_pred_all, arm_target_all)
    hand_mse = mse_np(hand_pred_all, hand_target_all)
    baseline_arm_mse = mse_np(np.broadcast_to(arm_baseline, arm_target_all.shape), arm_target_all)
    baseline_hand_mse = mse_np(
        np.broadcast_to(hand_baseline, hand_target_all.shape),
        hand_target_all,
    )
    total_mse = float((arm_mse + hand_mse) / 2.0)
    baseline_total_mse = float((baseline_arm_mse + baseline_hand_mse) / 2.0)
    total_sse = float(
        np.square(arm_pred_all - arm_target_all).sum()
        + np.square(hand_pred_all - hand_target_all).sum()
    )
    total_target_sse = float(
        np.square(arm_target_all).sum() + np.square(hand_target_all).sum()
    )
    metrics = {
        "n_samples": len(subset),
        "total_mse": total_mse,
        "arm_mse": arm_mse,
        "hand_mse": hand_mse,
        "mean_baseline_total_mse": baseline_total_mse,
        "mean_baseline_arm_mse": baseline_arm_mse,
        "mean_baseline_hand_mse": baseline_hand_mse,
        "total_relative_l2_error": float(math.sqrt(total_sse / max(total_target_sse, 1e-12))),
        "arm_pred_std_mean": float(arm_pred_all.std(axis=0).mean()),
        "arm_target_std_mean": float(arm_target_all.std(axis=0).mean()),
        "hand_pred_std_mean": float(hand_pred_all.std(axis=0).mean()),
        "hand_target_std_mean": float(hand_target_all.std(axis=0).mean()),
    }
    metrics.update(
        action_regression_metrics(
            arm_pred_all,
            arm_target_all,
            arm_mean_np,
            arm_std_np,
            prefix="arm",
            dim_prefix="arm_dim",
        )
    )
    metrics.update(
        action_regression_metrics(
            hand_pred_all,
            hand_target_all,
            hand_mean_np,
            hand_std_np,
            prefix="hand",
            dim_prefix="hand_dim",
        )
    )
    metrics.update(
        action_regression_metrics(
            np.concatenate([arm_pred_all, hand_pred_all], axis=1),
            np.concatenate([arm_target_all, hand_target_all], axis=1),
            np.concatenate([arm_mean_np, hand_mean_np], axis=0),
            np.concatenate([arm_std_np, hand_std_np], axis=0),
            prefix="total",
        )
    )
    if prediction_path is not None:
        write_rows(pred_rows, prediction_path)
    return metrics


def save_checkpoint(
    bundle: PairIdmBundle,
    arm_mean: torch.Tensor,
    arm_std: torch.Tensor,
    hand_mean: torch.Tensor,
    hand_std: torch.Tensor,
    args: argparse.Namespace,
    out_dir: Path,
    val_metrics: dict[str, float],
    *,
    filename: str = "checkpoint.pt",
) -> None:
    payload = {
        "arm_model_state": bundle.arm_net.state_dict(),
        "hand_model_state": bundle.hand_net.state_dict(),
        "arm_mean": arm_mean,
        "arm_std": arm_std,
        "hand_mean": hand_mean,
        "hand_std": hand_std,
        "model": {
            "input_channels": 6,
            "base_channels": args.base_channels,
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "arm_dim": ARM_DIM,
            "hand_dim": HAND_DIM,
            "alignment": "frame_pair_t_to_t_plus_1_predict_action_t",
        },
        "config": {
            "task": args.task_full,
            "task_short": args.task_short,
            "data_root": args.data_root,
            "segment_root": args.segment_root,
            "resize": args.resize,
            "max_samples": args.max_samples,
            "seed": args.seed,
            "split_by": args.split_by,
            "train_ratio": args.train_ratio,
            "arm_loss_weight": args.arm_loss_weight,
            "hand_loss_weight": args.hand_loss_weight,
            "lr_scheduler": args.lr_scheduler,
            "min_lr_ratio": args.min_lr_ratio,
        },
        "val_metrics": val_metrics,
    }
    torch.save(payload, out_dir / filename)


def load_pair_idm(checkpoint: Path, device: str) -> tuple[PairIdmBundle, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model_cfg = ckpt["model"]
    bundle = PairIdmBundle(
        arm_net=SmallPairCnn(
            ARM_DIM,
            base_channels=int(model_cfg["base_channels"]),
            hidden_dim=int(model_cfg["hidden_dim"]),
            dropout=float(model_cfg["dropout"]),
        ).to(device),
        hand_net=SmallPairCnn(
            HAND_DIM,
            base_channels=int(model_cfg["base_channels"]),
            hidden_dim=int(model_cfg["hidden_dim"]),
            dropout=float(model_cfg["dropout"]),
        ).to(device),
    )
    bundle.arm_net.load_state_dict(ckpt["arm_model_state"], strict=True)
    bundle.hand_net.load_state_dict(ckpt["hand_model_state"], strict=True)
    bundle.arm_net.eval()
    bundle.hand_net.eval()
    return (
        bundle,
        ckpt["arm_mean"].to(device),
        ckpt["arm_std"].to(device),
        ckpt["hand_mean"].to(device),
        ckpt["hand_std"].to(device),
        ckpt,
    )


def require_checkpoint_config(
    ckpt: dict,
    required: dict[str, object | None],
    *,
    allow_cli_split: bool,
) -> dict[str, object]:
    config = ckpt.get("config")
    if not isinstance(config, dict):
        raise ValueError("Checkpoint does not contain a config dict for validation split replay")
    missing = [key for key, value in required.items() if value is None]
    if missing:
        if allow_cli_split:
            return {key: value for key, value in required.items() if value is not None}
        raise ValueError(
            "Checkpoint is missing validation split config fields "
            f"{missing}. Re-train with the updated pair IDM code, or pass "
            "--allow-cli-split to explicitly use CLI split arguments for this legacy checkpoint."
        )
    return {key: value for key, value in required.items() if value is not None}


def apply_pair_checkpoint_split_config(args: argparse.Namespace, ckpt: dict) -> None:
    config = ckpt.get("config")
    if not isinstance(config, dict):
        raise ValueError("Checkpoint does not contain a config dict")
    task_full = config.get("task_full", config.get("task"))
    values = require_checkpoint_config(
        ckpt,
        {
            "task_short": config.get("task_short"),
            "task_full": task_full,
            "data_root": config.get("data_root"),
            "segment_root": config.get("segment_root"),
            "resize": config.get("resize"),
            "max_samples": config.get("max_samples"),
            "seed": config.get("seed"),
            "split_by": config.get("split_by"),
            "train_ratio": config.get("train_ratio"),
        },
        allow_cli_split=bool(getattr(args, "allow_cli_split", False)),
    )
    for key, value in values.items():
        setattr(args, key, value)


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
        {key: float(value) for key, value in row.items()}
        for row in csv.DictReader(eval_path.open())
    ]
    if not train_rows or not eval_rows:
        raise ValueError(f"Empty loss CSV under {out_dir}")
    train_steps = np.asarray([row["step"] for row in train_rows], dtype=np.float64)
    eval_steps = np.asarray([row["step"] for row in eval_rows], dtype=np.float64)

    def smooth(values: np.ndarray, window: int = 50) -> np.ndarray:
        if len(values) < window:
            return values
        kernel = np.ones(window, dtype=np.float64) / window
        smoothed = np.convolve(values, kernel, mode="valid")
        return np.concatenate([np.full(window - 1, np.nan), smoothed])

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), dpi=160, sharex=False)
    for key, color in (("loss", "#1f77b4"), ("arm_loss", "#ff7f0e"), ("hand_loss", "#2ca02c")):
        values = np.asarray([row[key] for row in train_rows], dtype=np.float64)
        axes[0].plot(train_steps, values, color=color, alpha=0.12, linewidth=1)
        axes[0].plot(train_steps, smooth(values), color=color, linewidth=1.8, label=f"train {key}")
    axes[0].set_ylabel("normalized train MSE")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(frameon=False, ncol=3)

    for key, color in (("total_mse", "#1f77b4"), ("arm_mse", "#ff7f0e"), ("hand_mse", "#2ca02c")):
        values = np.asarray([row[key] for row in eval_rows], dtype=np.float64)
        axes[1].plot(eval_steps, values, marker="o", color=color, label=f"eval {key}")
    baseline = np.asarray([row["mean_baseline_total_mse"] for row in eval_rows], dtype=np.float64)
    axes[1].plot(eval_steps, baseline, color="#555555", linestyle="--", label="mean baseline")
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("action MSE")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(frameon=False, ncol=4)
    fig.suptitle("Two-frame pair IDM training and eval loss")
    fig.tight_layout()
    fig.savefig(out_dir / "loss_curve.png")
    plt.close(fig)


def prepare_samples(args: argparse.Namespace) -> list[PairSample]:
    resolver = ActionResolver(Path(args.data_root))
    return discover_segment_pairs(
        resolver,
        Path(args.segment_root),
        max_samples=args.max_samples,
        seed=int(args.seed),
    )


def train(args: argparse.Namespace) -> None:
    resolve_task_args(args)
    seed = int(args.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    resize = parse_resize(args.resize)
    samples = prepare_samples(args)
    train_samples, val_samples = split_samples(
        samples,
        args.train_ratio,
        args.split_by,
        seed,
    )
    arm_mean, arm_std, hand_mean, hand_std = target_stats(train_samples)
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = str(Path(MAIN_ROOT) / "output" / "wan_pair_idm" / args.task_short)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.write_samples_json:
        write_json(out_dir / "samples.json", [asdict(sample) for sample in samples])
        write_json(out_dir / "train_samples.json", [asdict(sample) for sample in train_samples])
        write_json(out_dir / "val_samples.json", [asdict(sample) for sample in val_samples])
    print(
        f"alignment=(frame_t,frame_t+1)->action_t split_by={args.split_by} "
        f"train_samples={len(train_samples)} val_samples={len(val_samples)} "
        f"train_episodes={len({s.episode for s in train_samples})} "
        f"val_episodes={len({s.episode for s in val_samples})}",
        flush=True,
    )

    device = args.device
    bundle = make_models(args, device)
    print(
        f"trainable_arm_params={count_trainable_parameters(bundle.arm_net)} "
        f"trainable_hand_params={count_trainable_parameters(bundle.hand_net)}",
        flush=True,
    )
    optimizer = torch.optim.AdamW(
        list(bundle.arm_net.parameters()) + list(bundle.hand_net.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
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
        PairDataset(train_samples, resize),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate_pair_batch,
        drop_last=False,
    )
    arm_mean_dev = arm_mean.to(device)
    arm_std_dev = arm_std.to(device)
    hand_mean_dev = hand_mean.to(device)
    hand_std_dev = hand_std.to(device)
    history = []
    eval_history = []
    step = 0
    last_eval_step: int | None = None
    best_eval_total = float("inf")

    def run_eval(eval_step: int, *, write_predictions: bool) -> dict[str, float]:
        nonlocal best_eval_total, last_eval_step
        prediction_path = out_dir / "val_predictions.csv" if write_predictions else None
        metrics = validate_samples(
            bundle,
            val_samples,
            resize,
            arm_mean_dev,
            arm_std_dev,
            hand_mean_dev,
            hand_std_dev,
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
        if metrics["total_mse"] < best_eval_total:
            best_eval_total = metrics["total_mse"]
            save_checkpoint(
                bundle,
                arm_mean,
                arm_std,
                hand_mean,
                hand_std,
                args,
                out_dir,
                metrics,
                filename="best_checkpoint.pt",
            )
        print(
            f"eval_step={eval_step:04d} total_mse={metrics['total_mse']:.6f} "
            f"arm_mse={metrics['arm_mse']:.6f} hand_mse={metrics['hand_mse']:.6f} "
            f"mean_baseline_total_mse={metrics['mean_baseline_total_mse']:.6f} "
            f"best_total_mse={best_eval_total:.6f}",
            flush=True,
        )
        return metrics

    bundle.arm_net.train()
    bundle.hand_net.train()
    if args.eval_every > 0:
        run_eval(0, write_predictions=False)
    while step < args.steps:
        for batch in train_loader:
            if step >= args.steps:
                break
            frames = batch["frames"].to(device)
            arm_target = batch["arm_target"].to(device)
            hand_target = batch["hand_target"].to(device)
            norm_arm = (arm_target - arm_mean_dev) / arm_std_dev
            norm_hand = (hand_target - hand_mean_dev) / hand_std_dev
            pred_arm = bundle.arm_net(frames)
            pred_hand = bundle.hand_net(frames)
            arm_loss = F.mse_loss(pred_arm, norm_arm)
            hand_loss = F.mse_loss(pred_hand, norm_hand)
            loss = weighted_loss(
                arm_loss,
                hand_loss,
                args.arm_loss_weight,
                args.hand_loss_weight,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(
                    list(bundle.arm_net.parameters()) + list(bundle.hand_net.parameters()),
                    args.grad_clip_norm,
                )
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            step += 1
            row = {
                "step": step,
                "loss": tensor_float(loss),
                "arm_loss": tensor_float(arm_loss),
                "hand_loss": tensor_float(hand_loss),
                "lr": float(optimizer.param_groups[0]["lr"]),
            }
            history.append(row)
            if step == 1 or step % args.log_every == 0 or step == args.steps:
                print(
                    f"step={step:04d} loss={row['loss']:.6f} "
                    f"arm={row['arm_loss']:.6f} hand={row['hand_loss']:.6f}",
                    flush=True,
                )
            if args.eval_every > 0 and (step % args.eval_every == 0 or step == args.steps):
                run_eval(step, write_predictions=False)

    val_metrics = run_eval(step, write_predictions=True)
    save_checkpoint(
        bundle,
        arm_mean,
        arm_std,
        hand_mean,
        hand_std,
        args,
        out_dir,
        val_metrics,
    )
    best_ckpt_path = out_dir / "best_checkpoint.pt"
    if best_ckpt_path.is_file():
        best_bundle, best_arm_mean, best_arm_std, best_hand_mean, best_hand_std, _ = load_pair_idm(
            best_ckpt_path,
            device,
        )
        best_metrics = validate_samples(
            best_bundle,
            val_samples,
            resize,
            best_arm_mean,
            best_arm_std,
            best_hand_mean,
            best_hand_std,
            device,
            args,
            prediction_path=out_dir / "best_val_predictions.csv",
        )
        write_json(out_dir / "best_val_metrics.json", best_metrics)
    write_rows(history, out_dir / "train_loss.csv")
    plot_loss_curves(out_dir)
    print(json.dumps({"val": val_metrics, "out_dir": str(out_dir)}, indent=2), flush=True)


@torch.no_grad()
def validate_checkpoint(args: argparse.Namespace) -> None:
    device = args.device
    bundle, arm_mean, arm_std, hand_mean, hand_std, ckpt = load_pair_idm(
        Path(args.checkpoint),
        device,
    )
    apply_pair_checkpoint_split_config(args, ckpt)
    resolve_task_args(args)
    seed = int(args.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    resize = parse_resize(args.resize)
    samples = prepare_samples(args)
    _, val_samples = split_samples(samples, args.train_ratio, args.split_by, seed)
    _ = ckpt
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = validate_samples(
        bundle,
        val_samples,
        resize,
        arm_mean,
        arm_std,
        hand_mean,
        hand_std,
        device,
        args,
        prediction_path=out_dir / "val_predictions.csv",
    )
    write_json(out_dir / "val_metrics.json", metrics)
    print(json.dumps({"val": metrics, "out_dir": str(out_dir)}, indent=2), flush=True)


@torch.no_grad()
def eval_all_pairs(args: argparse.Namespace) -> None:
    device = args.device
    bundle, arm_mean, arm_std, hand_mean, hand_std, ckpt = load_pair_idm(
        Path(args.checkpoint),
        device,
    )
    apply_pair_checkpoint_split_config(args, ckpt)
    resolve_task_args(args)
    seed = int(args.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    resize = parse_resize(args.resize)
    samples = prepare_samples(args)
    _ = ckpt
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = validate_samples(
        bundle,
        samples,
        resize,
        arm_mean,
        arm_std,
        hand_mean,
        hand_std,
        device,
        args,
        prediction_path=out_dir / "predictions.csv",
    )
    write_json(out_dir / "metrics.json", metrics)
    print(json.dumps({"eval": metrics, "out_dir": str(out_dir)}, indent=2), flush=True)


def add_common_data_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--task-short", default=DEFAULT_TASK_SHORT)
    parser.add_argument("--task-full", default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--segment-root", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resize", default="256x256")
    parser.add_argument("--max-samples", type=int, default=0,
                        help="maximum discovered adjacent frame pairs; 0 keeps all")
    parser.add_argument("--seed", type=int, default=42)


def add_eval_split_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--split-by", choices=["episode", "sample"], default="episode")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--val-max-samples", type=int, default=0,
                        help="maximum eval samples; <=0 evaluates all")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Two-frame RGB pair IDM")
    sub = parser.add_subparsers(dest="cmd", required=True)

    train_p = sub.add_parser("train", help="train pair IDM on one task")
    add_common_data_args(train_p)
    add_eval_split_args(train_p)
    train_p.add_argument("--output-dir", default=None)
    train_p.add_argument("--steps", type=int, default=4000)
    train_p.add_argument("--base-channels", type=int, default=32)
    train_p.add_argument("--hidden-dim", type=int, default=128)
    train_p.add_argument("--dropout", type=float, default=0.0)
    train_p.add_argument("--lr", type=float, default=1e-4)
    train_p.add_argument("--weight-decay", type=float, default=1e-4)
    train_p.add_argument("--lr-scheduler", choices=["none", "cosine"], default="cosine")
    train_p.add_argument("--min-lr-ratio", type=float, default=0.05)
    train_p.add_argument("--arm-loss-weight", type=float, default=1.0)
    train_p.add_argument("--hand-loss-weight", type=float, default=1.0)
    train_p.add_argument("--grad-clip-norm", type=float, default=1.0)
    train_p.add_argument("--log-every", type=int, default=50)
    train_p.add_argument("--eval-every", type=int, default=250)
    train_p.add_argument("--write-samples-json", action="store_true")
    train_p.set_defaults(func=train)

    val_p = sub.add_parser("validate", help="validate a checkpoint on held-out pairs")
    add_common_data_args(val_p)
    add_eval_split_args(val_p)
    val_p.add_argument("--checkpoint", required=True)
    val_p.add_argument("--output-dir", required=True)
    val_p.add_argument("--allow-cli-split", action="store_true",
                       help="explicitly use CLI split args for legacy checkpoints missing split config")
    val_p.set_defaults(func=validate_checkpoint)

    eval_p = sub.add_parser("eval", help="evaluate a checkpoint on discovered pairs")
    add_common_data_args(eval_p)
    eval_p.add_argument("--checkpoint", required=True)
    eval_p.add_argument("--output-dir", required=True)
    eval_p.add_argument("--allow-cli-split", action="store_true",
                        help="explicitly use CLI data args for legacy checkpoints missing split config")
    eval_p.add_argument("--batch-size", type=int, default=16)
    eval_p.add_argument("--workers", type=int, default=2)
    eval_p.add_argument("--val-max-samples", type=int, default=0,
                        help="maximum eval samples; <=0 evaluates all")
    eval_p.set_defaults(func=eval_all_pairs)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
