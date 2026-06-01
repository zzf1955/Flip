"""AdaWorld latent-action decoder IDM for Humanoid Everyday H1.

This module trains only the downstream action decoder:

  (frame_t, frame_{t+1}) -> AdaWorld LAM z_t [32] -> H1 action_t [26]

The AdaWorld action encoder is run beforehand by
``src.pipeline.adaworld_action_encoder``.  This decoder consumes its
``latent_actions.npz`` artifact and joins each latent row to the H1 LeRobot
``action`` label at ``rel_frame_t``.
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
from src.pipeline.adaworld_action_encoder import DEFAULT_H1_DATASET, LATENT_DIM
from src.pipeline.wan_pair_idm import (
    action_regression_metrics,
    count_trainable_parameters,
    mse_np,
    tensor_float,
    write_json,
    write_rows,
)

HUMANOID_H1_ACTION_DIM = 26


@dataclass(frozen=True)
class LatentActionSample:
    sample_index: int
    episode: int
    chunk: int
    rel_frame_t: int
    rel_frame_tp1: int
    latent: tuple[float, ...]
    action_target: tuple[float, ...]
    parquet_path: str


class ResidualLatentBlock(nn.Module):
    def __init__(self, dim: int, dropout: float, *, gated: bool, layer_norm: bool) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if dropout < 0.0:
            raise ValueError(f"dropout must be non-negative, got {dropout}")
        self.gated = bool(gated)
        self.pre_norm = nn.LayerNorm(dim) if layer_norm else nn.Identity()
        self.fc = nn.Linear(dim, dim * 2 if gated else dim)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.pre_norm(x)
        if self.gated:
            value, gate = self.fc(h).chunk(2, dim=-1)
            h = F.silu(value) * torch.sigmoid(gate)
        else:
            h = F.silu(self.fc(h))
        h = self.dropout(self.proj(h))
        return x + h


class LatentActionDecoder(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        depth: int,
        dropout: float,
        layer_norm: bool,
        architecture: str,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        if output_dim <= 0:
            raise ValueError(f"output_dim must be positive, got {output_dim}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if depth < 0:
            raise ValueError(f"depth must be non-negative, got {depth}")
        if dropout < 0.0:
            raise ValueError(f"dropout must be non-negative, got {dropout}")
        if architecture not in {"mlp", "residual_mlp", "gated_mlp"}:
            raise ValueError(f"Unsupported architecture={architecture!r}")

        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.hidden_dim = int(hidden_dim)
        self.depth = int(depth)
        self.dropout = float(dropout)
        self.layer_norm = bool(layer_norm)
        self.architecture = str(architecture)

        if self.architecture == "mlp":
            layers: list[nn.Module] = []
            current_dim = input_dim
            for _ in range(depth):
                layers.append(nn.Linear(current_dim, hidden_dim))
                if layer_norm:
                    layers.append(nn.LayerNorm(hidden_dim))
                layers.append(nn.SiLU())
                if dropout > 0.0:
                    layers.append(nn.Dropout(dropout))
                current_dim = hidden_dim
            layers.append(nn.Linear(current_dim, output_dim))
            self.net = nn.Sequential(*layers)
        else:
            gated = self.architecture == "gated_mlp"
            self.input_proj = nn.Linear(input_dim, hidden_dim)
            self.input_norm = nn.LayerNorm(hidden_dim) if layer_norm else nn.Identity()
            self.blocks = nn.ModuleList(
                [
                    ResidualLatentBlock(
                        hidden_dim,
                        dropout,
                        gated=gated,
                        layer_norm=layer_norm,
                    )
                    for _ in range(depth)
                ]
            )
            self.output_norm = nn.LayerNorm(hidden_dim) if layer_norm else nn.Identity()
            self.head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim),
            )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        if latent.ndim != 2 or latent.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected latent [B,{self.input_dim}], got {tuple(latent.shape)}"
            )
        if self.architecture == "mlp":
            return self.net(latent.float())
        x = self.input_proj(latent.float())
        x = self.input_norm(x)
        for block in self.blocks:
            x = block(x)
        x = self.output_norm(x)
        return self.head(x)


class LatentActionDataset(torch.utils.data.Dataset):
    def __init__(self, samples: list[LatentActionSample]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        return {
            "latent": np.asarray(sample.latent, dtype=np.float32),
            "action_target": np.asarray(sample.action_target, dtype=np.float32),
            "sample_index": idx,
        }


def collate_latent_action_batch(items: list[dict]) -> dict:
    return {
        "latent": torch.from_numpy(np.stack([item["latent"] for item in items], axis=0)),
        "action_target": torch.from_numpy(
            np.stack([item["action_target"] for item in items], axis=0)
        ),
        "sample_index": [int(item["sample_index"]) for item in items],
    }


def resolve_latent_path(value: str) -> Path:
    path = Path(value).expanduser().resolve()
    if path.is_dir():
        path = path / "latent_actions.npz"
    if not path.is_file():
        raise FileNotFoundError(f"AdaWorld latent action artifact not found: {path}")
    return path


def resolve_data_root(value: str | None) -> Path:
    path = Path(value).expanduser().resolve() if value else Path(MAIN_ROOT) / DEFAULT_H1_DATASET
    if not path.is_dir():
        raise FileNotFoundError(f"Humanoid Everyday H1 data root not found: {path}")
    return path


def load_latent_artifact(path: Path) -> dict[str, np.ndarray]:
    payload = np.load(path)
    required = {"latent_actions", "episode", "chunk", "rel_frame_t", "rel_frame_tp1"}
    missing = sorted(required - set(payload.files))
    if missing:
        raise ValueError(f"Latent artifact missing arrays {missing}: {path}")
    arrays = {key: payload[key] for key in required}
    latent = arrays["latent_actions"]
    if latent.ndim != 2 or latent.shape[1] != LATENT_DIM:
        raise ValueError(f"Expected latent_actions [N,{LATENT_DIM}], got {latent.shape}")
    n = latent.shape[0]
    for key in ("episode", "chunk", "rel_frame_t", "rel_frame_tp1"):
        if arrays[key].shape != (n,):
            raise ValueError(f"Latent artifact array {key} has shape {arrays[key].shape}, expected {(n,)}")
    if not np.isfinite(latent).all():
        raise ValueError(f"Latent artifact contains non-finite values: {path}")
    return arrays


def parquet_path_for_sample(data_root: Path, chunk: int, episode: int) -> Path:
    path = data_root / "data" / f"chunk-{chunk:03d}" / f"episode_{episode:06d}.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"H1 action parquet not found for latent row: {path}")
    return path


def load_episode_action_table(path: Path, action_dim: int) -> pd.DataFrame:
    df = pd.read_parquet(path, columns=["action", "frame_index", "next.done"])
    if df.empty:
        raise ValueError(f"H1 action parquet is empty: {path}")
    frame_index = df["frame_index"].to_numpy(dtype=np.int64)
    if not np.array_equal(frame_index, np.arange(len(df), dtype=np.int64)):
        raise ValueError(f"H1 frame_index must be contiguous 0..N-1 for latent join: {path}")
    first_action = np.asarray(df.iloc[0]["action"], dtype=np.float32)
    if first_action.shape != (action_dim,):
        raise ValueError(
            f"Bad H1 action shape in {path}: {first_action.shape}, expected {(action_dim,)}"
        )
    return df


def build_samples_from_latents(
    *,
    data_root: Path,
    latent_path: Path,
    action_dim: int,
    max_samples: int,
    seed: int,
) -> list[LatentActionSample]:
    arrays = load_latent_artifact(latent_path)
    latent_actions = arrays["latent_actions"].astype(np.float32, copy=False)
    cache: dict[tuple[int, int], pd.DataFrame] = {}
    samples: list[LatentActionSample] = []
    for idx in range(latent_actions.shape[0]):
        episode = int(arrays["episode"][idx])
        chunk = int(arrays["chunk"][idx])
        rel_frame_t = int(arrays["rel_frame_t"][idx])
        rel_frame_tp1 = int(arrays["rel_frame_tp1"][idx])
        if rel_frame_tp1 <= rel_frame_t:
            raise ValueError(
                f"Latent row {idx} has invalid frame pair {rel_frame_t}->{rel_frame_tp1}"
            )
        key = (chunk, episode)
        if key not in cache:
            cache[key] = load_episode_action_table(
                parquet_path_for_sample(data_root, chunk, episode),
                action_dim,
            )
        df = cache[key]
        if rel_frame_t >= len(df):
            raise ValueError(
                f"Latent row {idx} frame {rel_frame_t} exceeds action parquet length {len(df)}"
            )
        if bool(df.iloc[rel_frame_t]["next.done"]):
            raise ValueError(f"Latent row {idx} points at terminal frame with next.done=true")
        action = np.asarray(df.iloc[rel_frame_t]["action"], dtype=np.float32)
        if action.shape != (action_dim,):
            raise ValueError(
                f"Bad H1 action shape at row {idx}: {action.shape}, expected {(action_dim,)}"
            )
        if not np.isfinite(action).all():
            raise ValueError(f"H1 action contains non-finite values at latent row {idx}")
        samples.append(
            LatentActionSample(
                sample_index=idx,
                episode=episode,
                chunk=chunk,
                rel_frame_t=rel_frame_t,
                rel_frame_tp1=rel_frame_tp1,
                latent=tuple(float(v) for v in latent_actions[idx].tolist()),
                action_target=tuple(float(v) for v in action.tolist()),
                parquet_path=str(parquet_path_for_sample(data_root, chunk, episode)),
            )
        )
    random.Random(seed).shuffle(samples)
    if max_samples > 0:
        samples = samples[:max_samples]
    if not samples:
        raise ValueError(f"No latent action samples loaded from {latent_path}")
    return samples


def split_latent_samples(
    samples: list[LatentActionSample],
    train_ratio: float,
    split_by: str,
    seed: int,
    *,
    train_samples_count: int = 0,
    val_samples_count: int = 0,
) -> tuple[list[LatentActionSample], list[LatentActionSample]]:
    if split_by not in {"sample", "episode"}:
        raise ValueError(f"Unsupported split_by={split_by!r}")
    if train_samples_count > 0 or val_samples_count > 0:
        if train_samples_count <= 0 or val_samples_count <= 0:
            raise ValueError(
                "train_samples_count and val_samples_count must both be positive "
                "when explicit sample counts are used"
            )
        total = train_samples_count + val_samples_count
        if split_by == "sample":
            if total > len(samples):
                raise ValueError(f"Requested train/eval samples exceed available samples: {total} > {len(samples)}")
            return samples[:train_samples_count], samples[train_samples_count:total]

        episode_groups: dict[int, list[LatentActionSample]] = {}
        for sample in samples:
            episode_groups.setdefault(sample.episode, []).append(sample)
        ordered_episodes = sorted(episode_groups)
        random.Random(seed).shuffle(ordered_episodes)
        train_samples: list[LatentActionSample] = []
        val_samples: list[LatentActionSample] = []
        train_boundary = len(ordered_episodes)
        for episode_idx, episode in enumerate(ordered_episodes):
            episode_samples = episode_groups[episode]
            if len(train_samples) >= train_samples_count:
                train_boundary = episode_idx
                break
            remaining = train_samples_count - len(train_samples)
            take = min(remaining, len(episode_samples))
            train_samples.extend(episode_samples[:take])
            train_boundary = episode_idx + 1
            if take < len(episode_samples):
                break
        for episode in ordered_episodes[train_boundary:]:
            if len(val_samples) >= val_samples_count:
                break
            episode_samples = episode_groups[episode]
            remaining = val_samples_count - len(val_samples)
            val_samples.extend(episode_samples[:remaining])
        if len(train_samples) != train_samples_count or len(val_samples) != val_samples_count:
            raise ValueError(
                f"Could not satisfy explicit episode split: train={len(train_samples)}/"
                f"{train_samples_count} val={len(val_samples)}/{val_samples_count}"
            )
        return train_samples, val_samples

    if not 0.0 < train_ratio < 1.0:
        raise ValueError(f"train_ratio must be in (0,1), got {train_ratio}")
    if split_by == "sample":
        n_train = max(1, min(len(samples) - 1, int(round(len(samples) * train_ratio))))
        return samples[:n_train], samples[n_train:]
    episodes = sorted({sample.episode for sample in samples})
    if len(episodes) < 2:
        raise ValueError("Episode split requires at least two episodes")
    random.Random(seed).shuffle(episodes)
    n_train_eps = max(1, min(len(episodes) - 1, int(round(len(episodes) * train_ratio))))
    train_eps = set(episodes[:n_train_eps])
    train_samples = [sample for sample in samples if sample.episode in train_eps]
    val_samples = [sample for sample in samples if sample.episode not in train_eps]
    if not train_samples or not val_samples:
        raise ValueError(f"Invalid episode split: train={len(train_samples)} val={len(val_samples)}")
    return train_samples, val_samples


def sample_stats(samples: list[LatentActionSample]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    latent = np.asarray([sample.latent for sample in samples], dtype=np.float32)
    action = np.asarray([sample.action_target for sample in samples], dtype=np.float32)
    latent_mean = latent.mean(axis=0)
    latent_std = np.maximum(latent.std(axis=0), 1e-4)
    action_mean = action.mean(axis=0)
    action_std = np.maximum(action.std(axis=0), 1e-4)
    return (
        torch.from_numpy(latent_mean),
        torch.from_numpy(latent_std),
        torch.from_numpy(action_mean),
        torch.from_numpy(action_std),
    )


def prepare_samples(args: argparse.Namespace) -> list[LatentActionSample]:
    if args.latent_path is None:
        raise ValueError("--latent-path is required unless replayed from checkpoint config")
    data_root = resolve_data_root(args.data_root)
    latent_path = resolve_latent_path(args.latent_path)
    args.data_root = str(data_root)
    args.latent_path = str(latent_path)
    return build_samples_from_latents(
        data_root=data_root,
        latent_path=latent_path,
        action_dim=int(args.action_dim),
        max_samples=int(args.max_samples),
        seed=int(args.seed),
    )


def apply_checkpoint_config(args: argparse.Namespace, ckpt: dict) -> None:
    config = ckpt.get("config")
    if not isinstance(config, dict):
        raise ValueError("Checkpoint does not contain a config dict")
    required = {
        "data_root": config.get("data_root"),
        "latent_path": config.get("latent_path"),
        "max_samples": config.get("max_samples"),
        "seed": config.get("seed"),
        "action_dim": config.get("action_dim"),
        "split_by": config.get("split_by"),
        "train_ratio": config.get("train_ratio"),
        "train_samples": config.get("train_samples"),
        "eval_samples": config.get("eval_samples"),
    }
    missing = [key for key, value in required.items() if value is None]
    if missing and not bool(getattr(args, "allow_cli_split", False)):
        raise ValueError(
            f"Checkpoint is missing decoder validation config fields {missing}; "
            "pass --allow-cli-split to explicitly use CLI data/split arguments"
        )
    for key, value in required.items():
        if value is not None:
            setattr(args, key, value)


def make_decoder(args: argparse.Namespace) -> LatentActionDecoder:
    return LatentActionDecoder(
        input_dim=LATENT_DIM,
        output_dim=int(args.action_dim),
        hidden_dim=int(args.hidden_dim),
        depth=int(args.depth),
        dropout=float(args.dropout),
        layer_norm=bool(args.layer_norm),
        architecture=str(args.decoder_arch),
    )


def save_checkpoint(
    model: LatentActionDecoder,
    latent_mean: torch.Tensor,
    latent_std: torch.Tensor,
    action_mean: torch.Tensor,
    action_std: torch.Tensor,
    args: argparse.Namespace,
    out_dir: Path,
    val_metrics: dict[str, float],
    *,
    filename: str,
) -> None:
    payload = {
        "model_state": model.state_dict(),
        "latent_mean": latent_mean,
        "latent_std": latent_std,
        "action_mean": action_mean,
        "action_std": action_std,
        "model": {
            "input_dim": LATENT_DIM,
            "output_dim": int(args.action_dim),
            "hidden_dim": int(args.hidden_dim),
            "depth": int(args.depth),
            "dropout": float(args.dropout),
            "layer_norm": bool(args.layer_norm),
            "architecture": f"{args.decoder_arch}_latent_action_decoder",
            "alignment": "(frame_t,frame_t+1)->adaworld_z_t->action_t",
        },
        "config": {
            "data_root": args.data_root,
            "latent_path": args.latent_path,
            "max_samples": int(args.max_samples),
            "seed": int(args.seed),
            "action_dim": int(args.action_dim),
            "split_by": args.split_by,
            "train_ratio": float(args.train_ratio),
            "train_samples": int(args.train_samples),
            "eval_samples": int(args.eval_samples),
            "decoder_arch": str(args.decoder_arch),
            "hidden_dim": int(args.hidden_dim),
            "depth": int(args.depth),
            "dropout": float(args.dropout),
            "layer_norm": bool(args.layer_norm),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "adam_beta1": float(args.adam_beta1),
            "adam_beta2": float(args.adam_beta2),
            "lr_scheduler": args.lr_scheduler,
            "min_lr_ratio": float(args.min_lr_ratio),
            "lr_warmup_steps": int(args.lr_warmup_steps),
            "lr_warmup_steps_effective": int(
                getattr(args, "effective_lr_warmup_steps", args.lr_warmup_steps)
            ),
            "lr_warmup_ratio": float(args.lr_warmup_ratio),
            "dataset": "humanoid_everyday_h1",
            "target_semantics": "action[rel_frame_t]",
            "latent_semantics": "AdaWorld LAM z_mu for (frame_t,frame_t+1)",
        },
        "val_metrics": val_metrics,
    }
    torch.save(payload, out_dir / filename)


def load_decoder_checkpoint(
    checkpoint: Path,
    device: str,
) -> tuple[LatentActionDecoder, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model_cfg = ckpt["model"]
    architecture = str(model_cfg["architecture"])
    if architecture == "mlp_latent_action_decoder":
        decoder_arch = "mlp"
    elif architecture == "residual_mlp_latent_action_decoder":
        decoder_arch = "residual_mlp"
    elif architecture == "gated_mlp_latent_action_decoder":
        decoder_arch = "gated_mlp"
    else:
        raise ValueError(f"Unsupported decoder checkpoint architecture={architecture!r}: {checkpoint}")
    model = LatentActionDecoder(
        input_dim=int(model_cfg["input_dim"]),
        output_dim=int(model_cfg["output_dim"]),
        hidden_dim=int(model_cfg["hidden_dim"]),
        depth=int(model_cfg["depth"]),
        dropout=float(model_cfg["dropout"]),
        layer_norm=bool(model_cfg["layer_norm"]),
        architecture=decoder_arch,
    ).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()
    return (
        model,
        ckpt["latent_mean"].to(device),
        ckpt["latent_std"].to(device),
        ckpt["action_mean"].to(device),
        ckpt["action_std"].to(device),
        ckpt,
    )


def effective_lr_warmup_steps(args: argparse.Namespace) -> int:
    if int(args.steps) <= 0:
        raise ValueError(f"steps must be positive, got {args.steps}")
    if int(args.lr_warmup_steps) < 0:
        raise ValueError(f"lr_warmup_steps must be non-negative, got {args.lr_warmup_steps}")
    if float(args.lr_warmup_ratio) < 0.0:
        raise ValueError(f"lr_warmup_ratio must be non-negative, got {args.lr_warmup_ratio}")
    warmup_steps = int(args.lr_warmup_steps)
    if warmup_steps == 0 and float(args.lr_warmup_ratio) > 0.0:
        warmup_steps = int(round(int(args.steps) * float(args.lr_warmup_ratio)))
    if warmup_steps > int(args.steps):
        raise ValueError(
            f"lr warmup steps must not exceed total steps, got {warmup_steps} > {args.steps}"
        )
    return warmup_steps


def build_decoder_optimizer(
    model: nn.Module,
    args: argparse.Namespace,
) -> torch.optim.Optimizer:
    beta1 = float(args.adam_beta1)
    beta2 = float(args.adam_beta2)
    if not 0.0 <= beta1 < 1.0:
        raise ValueError(f"adam_beta1 must be in [0,1), got {beta1}")
    if not 0.0 <= beta2 < 1.0:
        raise ValueError(f"adam_beta2 must be in [0,1), got {beta2}")
    if beta1 >= beta2:
        raise ValueError(f"Adam betas should satisfy beta1 < beta2, got {beta1} >= {beta2}")
    if float(args.weight_decay) < 0.0:
        raise ValueError(f"weight_decay must be non-negative, got {args.weight_decay}")
    return torch.optim.AdamW(
        model.parameters(),
        lr=float(args.lr),
        betas=(beta1, beta2),
        weight_decay=float(args.weight_decay),
    )


def build_decoder_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
) -> torch.optim.lr_scheduler.LambdaLR | None:
    warmup_steps = effective_lr_warmup_steps(args)
    args.effective_lr_warmup_steps = warmup_steps
    if float(args.min_lr_ratio) < 0.0:
        raise ValueError(f"min_lr_ratio must be non-negative, got {args.min_lr_ratio}")
    if args.lr_scheduler == "none" and warmup_steps == 0:
        return None
    if args.lr_scheduler not in {"none", "cosine"}:
        raise ValueError(f"Unsupported lr_scheduler={args.lr_scheduler!r}")

    def lr_lambda(step_index: int) -> float:
        if warmup_steps > 0 and step_index < warmup_steps:
            return float(step_index + 1) / float(warmup_steps)
        if args.lr_scheduler == "none":
            return 1.0
        decay_steps = max(1, int(args.steps) - warmup_steps)
        progress = min(
            1.0,
            max(0.0, float(step_index - warmup_steps) / float(decay_steps)),
        )
        min_ratio = float(args.min_lr_ratio)
        return min_ratio + 0.5 * (1.0 - min_ratio) * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


@torch.no_grad()
def validate_samples(
    model: LatentActionDecoder,
    samples: list[LatentActionSample],
    latent_mean: torch.Tensor,
    latent_std: torch.Tensor,
    action_mean: torch.Tensor,
    action_std: torch.Tensor,
    device: str,
    args: argparse.Namespace,
    *,
    prediction_path: Path | None,
) -> dict[str, float]:
    if not samples:
        raise ValueError("validate_samples received no samples")
    model.eval()
    subset = samples if args.val_max_samples <= 0 else samples[: args.val_max_samples]
    loader = torch.utils.data.DataLoader(
        LatentActionDataset(subset),
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.workers),
        collate_fn=collate_latent_action_batch,
        drop_last=False,
    )
    pred_chunks = []
    target_chunks = []
    pred_rows = []
    for batch in loader:
        latent = batch["latent"].to(device)
        target = batch["action_target"].to(device)
        norm_latent = (latent - latent_mean) / latent_std
        pred = model(norm_latent) * action_std + action_mean
        pred_cpu = pred.detach().cpu().numpy()
        target_cpu = target.detach().cpu().numpy()
        pred_chunks.append(pred_cpu)
        target_chunks.append(target_cpu)
        for local_idx, sample_idx in enumerate(batch["sample_index"]):
            sample = subset[int(sample_idx)]
            row = {
                "sample_index": sample.sample_index,
                "episode": sample.episode,
                "chunk": sample.chunk,
                "rel_frame_t": sample.rel_frame_t,
                "rel_frame_tp1": sample.rel_frame_tp1,
                "parquet_path": sample.parquet_path,
            }
            for dim in range(int(args.action_dim)):
                row[f"action_target_{dim:02d}"] = float(target_cpu[local_idx, dim])
                row[f"action_pred_{dim:02d}"] = float(pred_cpu[local_idx, dim])
                row[f"action_err_{dim:02d}"] = float(pred_cpu[local_idx, dim] - target_cpu[local_idx, dim])
            pred_rows.append(row)

    pred_all = np.concatenate(pred_chunks, axis=0)
    target_all = np.concatenate(target_chunks, axis=0)
    action_mean_np = action_mean.detach().cpu().numpy()
    action_std_np = action_std.detach().cpu().numpy()
    baseline = action_mean_np[None, :]
    total_sse = float(np.square(pred_all - target_all).sum())
    target_sse = float(np.square(target_all).sum())
    metrics = {
        "n_samples": len(subset),
        "action_mse": mse_np(pred_all, target_all),
        "mean_baseline_action_mse": mse_np(np.broadcast_to(baseline, target_all.shape), target_all),
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
    model.train()
    return metrics


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
        raise ValueError(f"Empty decoder loss CSV under {out_dir}")
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

    axes[1].plot(
        eval_steps,
        np.asarray([row["action_mse"] for row in eval_rows], dtype=np.float64),
        marker="o",
        color="#1f77b4",
        label="eval action MSE",
    )
    axes[1].plot(
        eval_steps,
        np.asarray([row["mean_baseline_action_mse"] for row in eval_rows], dtype=np.float64),
        color="#555555",
        linestyle="--",
        label="mean baseline",
    )
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("action MSE")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(frameon=False)
    fig.suptitle("AdaWorld latent action decoder IDM")
    fig.tight_layout()
    fig.savefig(out_dir / "loss_curve.png")
    plt.close(fig)


def train_decoder(args: argparse.Namespace) -> None:
    seed = int(args.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    samples = prepare_samples(args)
    train_samples, val_samples = split_latent_samples(
        samples,
        float(args.train_ratio),
        args.split_by,
        seed,
        train_samples_count=int(args.train_samples),
        val_samples_count=int(args.eval_samples),
    )
    latent_mean, latent_std, action_mean, action_std = sample_stats(train_samples)
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = str(Path(MAIN_ROOT) / "output" / "adaworld_action_decoder" / "humanoid_everyday_h1")
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.write_samples_json:
        write_json(out_dir / "samples.json", [asdict(sample) for sample in samples])
        write_json(out_dir / "train_samples.json", [asdict(sample) for sample in train_samples])
        write_json(out_dir / "val_samples.json", [asdict(sample) for sample in val_samples])

    print(
        "alignment=(frame_t,frame_t+1)->adaworld_z_t->action_t "
        f"split_by={args.split_by} train_samples={len(train_samples)} "
        f"val_samples={len(val_samples)} train_episodes={len({s.episode for s in train_samples})} "
        f"val_episodes={len({s.episode for s in val_samples})}",
        flush=True,
    )

    device = args.device
    model = make_decoder(args).to(device)
    print(
        f"decoder={args.decoder_arch} depth={args.depth} hidden_dim={args.hidden_dim} "
        f"dropout={args.dropout} layer_norm={args.layer_norm} "
        f"trainable_params={count_trainable_parameters(model)}",
        flush=True,
    )
    optimizer = build_decoder_optimizer(model, args)
    scheduler = build_decoder_lr_scheduler(optimizer, args)
    print(
        f"optimizer=AdamW lr={args.lr:g} betas=({args.adam_beta1:g},{args.adam_beta2:g}) "
        f"weight_decay={args.weight_decay:g} lr_scheduler={args.lr_scheduler} "
        f"warmup_steps={args.effective_lr_warmup_steps} min_lr_ratio={args.min_lr_ratio:g}",
        flush=True,
    )

    train_loader = torch.utils.data.DataLoader(
        LatentActionDataset(train_samples),
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.workers),
        collate_fn=collate_latent_action_batch,
        drop_last=False,
    )
    latent_mean_dev = latent_mean.to(device)
    latent_std_dev = latent_std.to(device)
    action_mean_dev = action_mean.to(device)
    action_std_dev = action_std.to(device)
    history = []
    eval_history = []
    step = 0
    last_eval_step: int | None = None
    best_eval_action = float("inf")

    def run_eval(eval_step: int, *, write_predictions: bool) -> dict[str, float]:
        nonlocal best_eval_action, last_eval_step
        metrics = validate_samples(
            model,
            val_samples,
            latent_mean_dev,
            latent_std_dev,
            action_mean_dev,
            action_std_dev,
            device,
            args,
            prediction_path=out_dir / "val_predictions.csv" if write_predictions else None,
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
            save_checkpoint(
                model,
                latent_mean,
                latent_std,
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
    if int(args.eval_every) > 0:
        run_eval(0, write_predictions=False)
    while step < int(args.steps):
        for batch in train_loader:
            if step >= int(args.steps):
                break
            latent = batch["latent"].to(device)
            target = batch["action_target"].to(device)
            norm_latent = (latent - latent_mean_dev) / latent_std_dev
            norm_target = (target - action_mean_dev) / action_std_dev
            pred = model(norm_latent)
            loss = F.mse_loss(pred, norm_target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if float(args.grad_clip_norm) > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip_norm))
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
            if step == 1 or step % int(args.log_every) == 0 or step == int(args.steps):
                print(f"step={step:04d} loss={row['loss']:.6f}", flush=True)
            if int(args.eval_every) > 0 and (step % int(args.eval_every) == 0 or step == int(args.steps)):
                run_eval(step, write_predictions=False)

    val_metrics = run_eval(step, write_predictions=True)
    save_checkpoint(model, latent_mean, latent_std, action_mean, action_std, args, out_dir, val_metrics, filename="checkpoint.pt")
    best_ckpt_path = out_dir / "best_checkpoint.pt"
    if best_ckpt_path.is_file():
        best_model, best_latent_mean, best_latent_std, best_action_mean, best_action_std, _ = load_decoder_checkpoint(
            best_ckpt_path,
            device,
        )
        best_metrics = validate_samples(
            best_model,
            val_samples,
            best_latent_mean,
            best_latent_std,
            best_action_mean,
            best_action_std,
            device,
            args,
            prediction_path=out_dir / "best_val_predictions.csv",
        )
        write_json(out_dir / "best_val_metrics.json", best_metrics)
    write_rows(history, out_dir / "train_loss.csv")
    plot_loss_curves(out_dir)
    print(json.dumps({"val": val_metrics, "out_dir": str(out_dir)}, indent=2), flush=True)


@torch.no_grad()
def validate_decoder_checkpoint(args: argparse.Namespace) -> None:
    device = args.device
    model, latent_mean, latent_std, action_mean, action_std, ckpt = load_decoder_checkpoint(
        Path(args.checkpoint),
        device,
    )
    apply_checkpoint_config(args, ckpt)
    seed = int(args.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    samples = prepare_samples(args)
    _, val_samples = split_latent_samples(
        samples,
        float(args.train_ratio),
        args.split_by,
        seed,
        train_samples_count=int(args.train_samples),
        val_samples_count=int(args.eval_samples),
    )
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = validate_samples(
        model,
        val_samples,
        latent_mean,
        latent_std,
        action_mean,
        action_std,
        device,
        args,
        prediction_path=out_dir / "val_predictions.csv",
    )
    write_json(out_dir / "val_metrics.json", metrics)
    print(json.dumps({"val": metrics, "out_dir": str(out_dir)}, indent=2), flush=True)


@torch.no_grad()
def eval_decoder(args: argparse.Namespace) -> None:
    device = args.device
    model, latent_mean, latent_std, action_mean, action_std, ckpt = load_decoder_checkpoint(
        Path(args.checkpoint),
        device,
    )
    apply_checkpoint_config(args, ckpt)
    seed = int(args.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    samples = prepare_samples(args)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = validate_samples(
        model,
        samples,
        latent_mean,
        latent_std,
        action_mean,
        action_std,
        device,
        args,
        prediction_path=out_dir / "predictions.csv",
    )
    write_json(out_dir / "metrics.json", metrics)
    print(json.dumps({"eval": metrics, "out_dir": str(out_dir)}, indent=2), flush=True)


def add_data_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--latent-path", default=None, help="path to latent_actions.npz or its directory")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--action-dim", type=int, default=HUMANOID_H1_ACTION_DIM)


def add_split_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--train-samples", type=int, default=0)
    parser.add_argument("--eval-samples", type=int, default=0)
    parser.add_argument("--train-ratio", type=float, default=0.875)
    parser.add_argument("--split-by", choices=["sample", "episode"], default="episode")


def add_runtime_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--val-max-samples", type=int, default=0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    train_p = sub.add_parser("train", help="train AdaWorld latent action decoder")
    add_data_args(train_p)
    add_split_args(train_p)
    add_runtime_args(train_p)
    train_p.add_argument("--output-dir", default=None)
    train_p.add_argument("--steps", type=int, default=1000)
    train_p.add_argument(
        "--decoder-arch",
        choices=["mlp", "residual_mlp", "gated_mlp"],
        default="residual_mlp",
    )
    train_p.add_argument("--hidden-dim", type=int, default=256)
    train_p.add_argument("--depth", type=int, default=4)
    train_p.add_argument("--dropout", type=float, default=0.02)
    train_p.add_argument("--layer-norm", action=argparse.BooleanOptionalAction, default=True)
    train_p.add_argument("--lr", type=float, default=5e-4)
    train_p.add_argument("--weight-decay", type=float, default=1e-4)
    train_p.add_argument("--adam-beta1", type=float, default=0.9)
    train_p.add_argument("--adam-beta2", type=float, default=0.95)
    train_p.add_argument("--lr-scheduler", choices=["none", "cosine"], default="cosine")
    train_p.add_argument("--min-lr-ratio", type=float, default=0.02)
    train_p.add_argument(
        "--lr-warmup-steps",
        type=int,
        default=0,
        help="explicit linear warmup steps; 0 derives from lr-warmup-ratio",
    )
    train_p.add_argument(
        "--lr-warmup-ratio",
        type=float,
        default=0.05,
        help="fraction of total steps used for linear warmup when lr-warmup-steps is 0",
    )
    train_p.add_argument("--grad-clip-norm", type=float, default=1.0)
    train_p.add_argument("--log-every", type=int, default=50)
    train_p.add_argument("--eval-every", type=int, default=100)
    train_p.add_argument("--write-samples-json", action="store_true")
    train_p.set_defaults(func=train_decoder)

    val_p = sub.add_parser("validate", help="validate a decoder checkpoint on held-out samples")
    add_data_args(val_p)
    add_split_args(val_p)
    add_runtime_args(val_p)
    val_p.add_argument("--checkpoint", required=True)
    val_p.add_argument("--output-dir", required=True)
    val_p.add_argument("--allow-cli-split", action="store_true")
    val_p.set_defaults(func=validate_decoder_checkpoint)

    eval_p = sub.add_parser("eval", help="evaluate a decoder checkpoint on all loaded samples")
    add_data_args(eval_p)
    add_runtime_args(eval_p)
    eval_p.add_argument("--checkpoint", required=True)
    eval_p.add_argument("--output-dir", required=True)
    eval_p.add_argument("--allow-cli-split", action="store_true")
    eval_p.set_defaults(func=eval_decoder)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
