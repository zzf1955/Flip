"""H2R HDF5 Diffusion Policy behavior cloning.

The downloaded HumanAndRobot / H2R data is not in LeRobot parquet layout.  This
entry reads the HDF5 layout directly:

- ``data/<task>/episode_<id>.hdf5`` contains ``cam_data/robot_camera``,
  ``qpos``, ``qvel``, ``end_position``, ``gripper_state`` and ``action``.
- ``video/<task>/episode_<id>/robot_camera.mp4`` is checked for alignment, but
  training reads frames from HDF5 to avoid a separate conversion step.

The policy is action-only diffusion BC: clean video/state history conditions a
denoiser over noisy future action chunks.  It does not predict future video.
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
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

DEFAULT_DATA_ROOT = Path("/disk_n/zzf/flip/data/h2r/v1")
DEFAULT_OUTPUT_DIR = Path("tmp/h2r_diffusion_policy_t068")
DEFAULT_TASKS = "grab_cup_v1,grab_cube2_v1,push_box_random_v1"
DEFAULT_STATE_KEYS = "qpos,qvel,end_position,gripper_state"


@dataclass(frozen=True)
class EpisodeInfo:
    task: str
    episode: int
    hdf5_path: str
    video_path: str
    length: int


@dataclass(frozen=True)
class SampleInfo:
    task: str
    episode: int
    hdf5_path: str
    obs_start: int
    action_start: int


@dataclass(frozen=True)
class NormStats:
    action_mean: list[float]
    action_std: list[float]
    state_mean: list[float]
    state_std: list[float]


def parse_csv(value: str) -> list[str]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise ValueError(f"expected comma-separated values, got {value!r}")
    return items


def parse_resize(value: str) -> tuple[int, int]:
    parts = value.lower().replace(" ", "").split("x")
    if len(parts) != 2:
        raise ValueError(f"--resize must be formatted as HxW, got {value!r}")
    height, width = int(parts[0]), int(parts[1])
    if height <= 0 or width <= 0:
        raise ValueError(f"--resize dimensions must be positive, got {value!r}")
    return height, width


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"cannot write empty csv: {path}")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def episode_id(path: Path) -> int:
    if not path.stem.startswith("episode_"):
        raise ValueError(f"unexpected H2R episode filename: {path}")
    return int(path.stem.removeprefix("episode_"))


def read_state_matrix(f: h5py.File, state_keys: list[str]) -> np.ndarray:
    parts: list[np.ndarray] = []
    for key in state_keys:
        if key not in f:
            raise ValueError(f"H2R HDF5 missing state key {key!r}: {f.filename}")
        value = np.asarray(f[key], dtype=np.float32)
        if value.ndim == 1:
            value = value[:, None]
        if value.ndim != 2:
            raise ValueError(f"H2R state key {key!r} must be 1D or 2D, got {value.shape}")
        parts.append(value)
    return np.concatenate(parts, axis=1)


def discover_episodes(args: argparse.Namespace) -> list[EpisodeInfo]:
    data_dir = args.data_root / "data"
    video_dir = args.data_root / "video"
    if not data_dir.is_dir():
        raise FileNotFoundError(f"H2R data dir not found: {data_dir}")
    if not video_dir.is_dir():
        raise FileNotFoundError(f"H2R video dir not found: {video_dir}")

    tasks = sorted(path.name for path in data_dir.iterdir() if path.is_dir()) if args.tasks == "all" else parse_csv(args.tasks)
    state_keys = parse_csv(args.state_keys)
    episodes: list[EpisodeInfo] = []
    for task in tasks:
        task_data = data_dir / task
        task_video = video_dir / task
        if not task_data.is_dir():
            raise FileNotFoundError(f"H2R task data dir not found: {task_data}")
        if not task_video.is_dir():
            raise FileNotFoundError(f"H2R task video dir not found: {task_video}")
        hdf5_paths = sorted(task_data.glob("episode_*.hdf5"), key=episode_id)
        if args.max_episodes_per_task > 0:
            hdf5_paths = hdf5_paths[: args.max_episodes_per_task]
        for hdf5_path in hdf5_paths:
            ep = episode_id(hdf5_path)
            video_path = task_video / f"episode_{ep}" / f"{args.camera_key}.mp4"
            if not video_path.is_file():
                raise FileNotFoundError(f"H2R camera video not found: {video_path}")
            with h5py.File(hdf5_path, "r") as f:
                required = [f"cam_data/{args.camera_key}", args.action_key, *state_keys]
                missing = [key for key in required if key not in f]
                if missing:
                    raise ValueError(f"H2R episode missing keys {missing}: {hdf5_path}")
                length = int(f[args.action_key].shape[0])
                for key in required:
                    if int(f[key].shape[0]) != length:
                        raise ValueError(
                            f"H2R key length mismatch for {key}: {f[key].shape[0]} vs "
                            f"{length} in {hdf5_path}"
                        )
            episodes.append(EpisodeInfo(task, ep, str(hdf5_path), str(video_path), length))
    if not episodes:
        raise ValueError(f"no H2R episodes found under {args.data_root}")
    return episodes


def split_episodes(episodes: list[EpisodeInfo], train_ratio: float, seed: int) -> tuple[list[EpisodeInfo], list[EpisodeInfo]]:
    if not 0.0 < train_ratio < 1.0:
        raise ValueError(f"train_ratio must be in (0,1), got {train_ratio}")
    rng = random.Random(seed)
    by_task: dict[str, list[EpisodeInfo]] = {}
    for episode in episodes:
        by_task.setdefault(episode.task, []).append(episode)
    train: list[EpisodeInfo] = []
    val: list[EpisodeInfo] = []
    for task, group in sorted(by_task.items()):
        ordered = list(group)
        rng.shuffle(ordered)
        if len(ordered) < 2:
            raise ValueError(f"episode split needs at least two episodes for task={task}")
        n_train = max(1, min(len(ordered) - 1, int(round(len(ordered) * train_ratio))))
        train.extend(ordered[:n_train])
        val.extend(ordered[n_train:])
    return train, val


def build_samples(
    episodes: list[EpisodeInfo],
    obs_horizon: int,
    pred_horizon: int,
    frame_stride: int,
    max_samples: int,
    seed: int,
) -> list[SampleInfo]:
    if obs_horizon <= 0 or pred_horizon <= 0:
        raise ValueError("obs_horizon and pred_horizon must be positive")
    if frame_stride <= 0:
        raise ValueError(f"frame_stride must be positive, got {frame_stride}")
    samples: list[SampleInfo] = []
    for episode in episodes:
        if episode.length < obs_horizon + pred_horizon:
            continue
        for action_start in range(obs_horizon - 1, episode.length - pred_horizon + 1, frame_stride):
            samples.append(
                SampleInfo(
                    task=episode.task,
                    episode=episode.episode,
                    hdf5_path=episode.hdf5_path,
                    obs_start=action_start - obs_horizon + 1,
                    action_start=action_start,
                )
            )
    random.Random(seed).shuffle(samples)
    if max_samples > 0:
        samples = samples[:max_samples]
    if not samples:
        raise ValueError("no H2R samples built; check horizons, stride and max sample limits")
    return samples


def compute_norm(episodes: list[EpisodeInfo], args: argparse.Namespace) -> NormStats:
    state_keys = parse_csv(args.state_keys)
    actions: list[np.ndarray] = []
    states: list[np.ndarray] = []
    remaining = args.norm_max_frames
    ordered = list(episodes)
    random.Random(args.seed).shuffle(ordered)
    for episode in ordered:
        with h5py.File(episode.hdf5_path, "r") as f:
            action = np.asarray(f[args.action_key], dtype=np.float32)
            state = read_state_matrix(f, state_keys)
        if remaining > 0 and action.shape[0] > remaining:
            idx = np.linspace(0, action.shape[0] - 1, remaining, dtype=np.int64)
            action = action[idx]
            state = state[idx]
        actions.append(action)
        states.append(state)
        if remaining > 0:
            remaining -= int(action.shape[0])
            if remaining <= 0:
                break
    action_arr = np.concatenate(actions, axis=0)
    state_arr = np.concatenate(states, axis=0)
    return NormStats(
        action_mean=[float(v) for v in action_arr.mean(axis=0).tolist()],
        action_std=[float(v) for v in np.maximum(action_arr.std(axis=0), 1e-6).tolist()],
        state_mean=[float(v) for v in state_arr.mean(axis=0).tolist()],
        state_std=[float(v) for v in np.maximum(state_arr.std(axis=0), 1e-6).tolist()],
    )


class H2RDataset(Dataset):
    def __init__(self, samples: list[SampleInfo], stats: NormStats, args: argparse.Namespace) -> None:
        self.samples = list(samples)
        self.stats = stats
        self.camera_key = args.camera_key
        self.action_key = args.action_key
        self.state_keys = parse_csv(args.state_keys)
        self.obs_horizon = int(args.obs_horizon)
        self.pred_horizon = int(args.pred_horizon)
        self.resize = parse_resize(args.resize)
        self.action_mean = np.asarray(stats.action_mean, dtype=np.float32)
        self.action_std = np.asarray(stats.action_std, dtype=np.float32)
        self.state_mean = np.asarray(stats.state_mean, dtype=np.float32)
        self.state_std = np.asarray(stats.state_std, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        with h5py.File(sample.hdf5_path, "r") as f:
            obs_slice = slice(sample.obs_start, sample.obs_start + self.obs_horizon)
            act_slice = slice(sample.action_start, sample.action_start + self.pred_horizon)
            frames = np.asarray(f[f"cam_data/{self.camera_key}"][obs_slice], dtype=np.float32)
            state = read_state_matrix(f, self.state_keys)[obs_slice]
            action = np.asarray(f[self.action_key][act_slice], dtype=np.float32)
        if frames.shape[0] != self.obs_horizon:
            raise ValueError(f"bad frame horizon for {sample}: {frames.shape}")
        if action.shape[0] != self.pred_horizon:
            raise ValueError(f"bad action horizon for {sample}: {action.shape}")

        height, width = self.resize
        resized = np.empty((self.obs_horizon, height, width, 3), dtype=np.float32)
        for i, frame in enumerate(frames):
            resized[i] = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        video = np.transpose(resized / 127.5 - 1.0, (0, 3, 1, 2)).astype(np.float32)
        state = ((state - self.state_mean) / self.state_std).astype(np.float32)
        action = ((action - self.action_mean) / self.action_std).astype(np.float32)
        return {
            "video": torch.from_numpy(video),
            "state": torch.from_numpy(state),
            "action": torch.from_numpy(action),
            "task": sample.task,
            "episode": sample.episode,
            "action_start": sample.action_start,
        }


def collate(batch: list[dict]) -> dict:
    return {
        "video": torch.stack([item["video"] for item in batch]),
        "state": torch.stack([item["state"] for item in batch]),
        "action": torch.stack([item["action"] for item in batch]),
        "task": [item["task"] for item in batch],
        "episode": torch.tensor([item["episode"] for item in batch], dtype=torch.long),
        "action_start": torch.tensor([item["action_start"] for item in batch], dtype=torch.long),
    }


def build_loaders(args: argparse.Namespace, stats: NormStats | None = None) -> tuple[DataLoader, DataLoader, NormStats, dict]:
    episodes = discover_episodes(args)
    train_eps, val_eps = split_episodes(episodes, args.train_ratio, args.seed)
    train_samples = build_samples(train_eps, args.obs_horizon, args.pred_horizon, args.frame_stride, args.max_train_samples, args.seed)
    val_samples = build_samples(val_eps, args.obs_horizon, args.pred_horizon, args.frame_stride, args.max_val_samples, args.seed + 1)
    if stats is None:
        stats = compute_norm(train_eps, args)
    train_ds = H2RDataset(train_samples, stats, args)
    val_ds = H2RDataset(val_samples, stats, args)
    pin = str(args.device).startswith("cuda")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=pin, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=pin, collate_fn=collate)
    summary = {
        "format": "h2r_hdf5_direct",
        "data_root": str(args.data_root),
        "tasks": sorted({episode.task for episode in episodes}),
        "episodes_total": len(episodes),
        "episodes_train": len(train_eps),
        "episodes_val": len(val_eps),
        "samples_train": len(train_samples),
        "samples_val": len(val_samples),
        "obs_horizon": args.obs_horizon,
        "pred_horizon": args.pred_horizon,
        "state_dim": len(stats.state_mean),
        "action_dim": len(stats.action_mean),
        "state_keys": parse_csv(args.state_keys),
        "action_key": args.action_key,
        "camera_key": args.camera_key,
        "resize": args.resize,
    }
    return train_loader, val_loader, stats, summary


class TimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = int(dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(torch.arange(half, device=t.device, dtype=torch.float32) * (-math.log(10000.0) / max(half - 1, 1)))
        emb = torch.cat([torch.sin(t.float()[:, None] * freqs[None]), torch.cos(t.float()[:, None] * freqs[None])], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class DiffusionPolicy(nn.Module):
    def __init__(
        self,
        obs_horizon: int,
        pred_horizon: int,
        action_dim: int,
        state_dim: int,
        hidden_dim: int,
        time_dim: int,
        depth: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.obs_horizon = int(obs_horizon)
        self.pred_horizon = int(pred_horizon)
        self.action_dim = int(action_dim)
        self.state_dim = int(state_dim)
        self.hidden_dim = int(hidden_dim)
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 16, 5, stride=2, padding=2),
            nn.SiLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, hidden_dim),
            nn.SiLU(),
        )
        self.cond = nn.Sequential(
            nn.Linear(obs_horizon * (hidden_dim + state_dim), hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.time = nn.Sequential(TimeEmbedding(time_dim), nn.Linear(time_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        in_dim = pred_horizon * action_dim + hidden_dim * 2
        layers: list[nn.Module] = []
        cur = in_dim
        for _ in range(depth):
            layers.extend([nn.Linear(cur, hidden_dim), nn.SiLU(), nn.Dropout(dropout)])
            cur = hidden_dim
        layers.append(nn.Linear(cur, pred_horizon * action_dim))
        self.denoiser = nn.Sequential(*layers)

    def encode_condition(self, video: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        bsz, obs, channels, height, width = video.shape
        if obs != self.obs_horizon or channels != 3:
            raise ValueError(f"expected video [B,{self.obs_horizon},3,H,W], got {tuple(video.shape)}")
        if state.shape[1:] != (self.obs_horizon, self.state_dim):
            raise ValueError(f"expected state [B,{self.obs_horizon},{self.state_dim}], got {tuple(state.shape)}")
        image = self.image_encoder(video.reshape(bsz * obs, channels, height, width)).reshape(bsz, obs, self.hidden_dim)
        return self.cond(torch.cat([image, state], dim=-1).reshape(bsz, -1))

    def forward(self, noisy_action: torch.Tensor, t: torch.Tensor, video: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        bsz = noisy_action.shape[0]
        cond = self.encode_condition(video, state)
        time = self.time(t)
        x = torch.cat([noisy_action.reshape(bsz, -1), cond, time], dim=-1)
        return self.denoiser(x).reshape(bsz, self.pred_horizon, self.action_dim)


class DDPMSchedule:
    def __init__(self, steps: int, beta_start: float, beta_end: float, device: torch.device) -> None:
        self.steps = int(steps)
        self.betas = torch.linspace(beta_start, beta_end, steps, device=device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def add_noise(self, clean: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        alpha_bar = self.alpha_bars[t].view(-1, 1, 1)
        return alpha_bar.sqrt() * clean + (1.0 - alpha_bar).sqrt() * noise

    @torch.no_grad()
    def sample(self, model: DiffusionPolicy, video: torch.Tensor, state: torch.Tensor, shape: tuple[int, int, int]) -> torch.Tensor:
        x = torch.randn(shape, device=video.device)
        for step in reversed(range(self.steps)):
            t = torch.full((shape[0],), step, device=video.device, dtype=torch.long)
            eps = model(x, t, video, state)
            beta = self.betas[step]
            alpha = self.alphas[step]
            alpha_bar = self.alpha_bars[step]
            mean = (x - beta / torch.sqrt(1.0 - alpha_bar) * eps) / torch.sqrt(alpha)
            x = mean if step == 0 else mean + torch.sqrt(beta) * torch.randn_like(x)
        return x


def to_device(batch: dict, device: torch.device) -> dict:
    out = dict(batch)
    out["video"] = batch["video"].to(device)
    out["state"] = batch["state"].to(device)
    out["action"] = batch["action"].to(device)
    return out


def make_model(args: argparse.Namespace | dict, state_dim: int, action_dim: int) -> DiffusionPolicy:
    get = args.get if isinstance(args, dict) else lambda key: getattr(args, key)
    return DiffusionPolicy(
        obs_horizon=int(get("obs_horizon")),
        pred_horizon=int(get("pred_horizon")),
        action_dim=action_dim,
        state_dim=state_dim,
        hidden_dim=int(get("hidden_dim")),
        time_dim=int(get("time_dim")),
        depth=int(get("depth")),
        dropout=float(get("dropout")),
    )


def denoise_loss(model: DiffusionPolicy, schedule: DDPMSchedule, batch: dict) -> torch.Tensor:
    action = batch["action"]
    t = torch.randint(0, schedule.steps, (action.shape[0],), device=action.device)
    noise = torch.randn_like(action)
    noisy = schedule.add_noise(action, noise, t)
    pred = model(noisy, t, batch["video"], batch["state"])
    return F.mse_loss(pred, noise)


@torch.no_grad()
def evaluate(model: DiffusionPolicy, schedule: DDPMSchedule, loader: DataLoader, device: torch.device, max_batches: int, sample_actions: bool) -> dict:
    model.eval()
    losses: list[float] = []
    action_mse: list[float] = []
    per_horizon_sum: torch.Tensor | None = None
    per_horizon_count = 0
    for batch_idx, batch in enumerate(loader):
        if max_batches > 0 and batch_idx >= max_batches:
            break
        batch = to_device(batch, device)
        losses.append(float(denoise_loss(model, schedule, batch).item()))
        if sample_actions:
            pred = schedule.sample(model, batch["video"], batch["state"], tuple(batch["action"].shape))
            err = (pred - batch["action"]).pow(2)
            action_mse.append(float(err.mean().item()))
            horizon = err.mean(dim=2).sum(dim=0).cpu()
            per_horizon_sum = horizon if per_horizon_sum is None else per_horizon_sum + horizon
            per_horizon_count += int(err.shape[0])
    if not losses:
        raise ValueError("eval produced no batches")
    metrics: dict[str, object] = {"denoise_loss": float(np.mean(losses))}
    if action_mse:
        metrics["sampled_action_mse_norm"] = float(np.mean(action_mse))
        metrics["sampled_action_mse_norm_per_horizon"] = [float(v) for v in (per_horizon_sum / per_horizon_count).tolist()]
    return metrics


def config_from_args(args: argparse.Namespace) -> dict:
    keys = [
        "data_root", "tasks", "camera_key", "action_key", "state_keys", "max_episodes_per_task",
        "train_ratio", "obs_horizon", "pred_horizon", "frame_stride", "max_train_samples",
        "max_val_samples", "norm_max_frames", "resize", "batch_size", "workers", "seed",
        "hidden_dim", "time_dim", "depth", "dropout", "diffusion_steps", "beta_start", "beta_end",
    ]
    out = {key: getattr(args, key) for key in keys}
    out["data_root"] = str(out["data_root"])
    return out


def command_inspect(args: argparse.Namespace) -> None:
    train_loader, val_loader, stats, summary = build_loaders(args)
    batch = next(iter(train_loader))
    payload = {
        **summary,
        "normalization": asdict(stats),
        "train_batch_shapes": {
            "video": list(batch["video"].shape),
            "state": list(batch["state"].shape),
            "action": list(batch["action"].shape),
        },
        "val_batches": len(val_loader),
    }
    if args.output_json is not None:
        write_json(args.output_json, payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def command_train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device)
    train_loader, val_loader, stats, summary = build_loaders(args)
    model = make_model(args, summary["state_dim"], summary["action_dim"]).to(device)
    schedule = DDPMSchedule(args.diffusion_steps, args.beta_start, args.beta_end, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "train_log.jsonl"
    if log_path.exists() and not args.resume_log:
        log_path.unlink()
    write_json(args.output_dir / "dataset_summary.json", {**summary, "normalization": asdict(stats)})
    print(json.dumps({**summary, "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)}, ensure_ascii=False))

    train_iter = iter(train_loader)
    recent: list[float] = []
    best = math.inf
    best_metrics: dict = {}
    for step in range(1, args.steps + 1):
        model.train()
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        batch = to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        loss = denoise_loss(model, schedule, batch)
        loss.backward()
        if args.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        recent.append(float(loss.item()))
        if step == 1 or step % args.log_every == 0:
            row = {"step": step, "train_denoise_loss": float(np.mean(recent)), "last_train_denoise_loss": float(loss.item())}
            append_jsonl(log_path, row)
            print(json.dumps(row, ensure_ascii=False))
            recent.clear()
        if step == args.steps or step % args.eval_every == 0:
            metrics = evaluate(model, schedule, val_loader, device, args.eval_max_batches, args.eval_sample_actions)
            row = {"step": step, **metrics}
            append_jsonl(log_path, row)
            print(json.dumps(row, ensure_ascii=False))
            if float(metrics["denoise_loss"]) < best:
                best = float(metrics["denoise_loss"])
                best_metrics = row
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "config": config_from_args(args),
                        "normalization": asdict(stats),
                        "summary": summary,
                        "step": step,
                        "metrics": row,
                    },
                    args.output_dir / "best_checkpoint.pt",
                )

    final_metrics = evaluate(model, schedule, val_loader, device, args.eval_max_batches, True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": config_from_args(args),
            "normalization": asdict(stats),
            "summary": summary,
            "step": args.steps,
            "metrics": {"step": args.steps, **final_metrics},
        },
        args.output_dir / "last_checkpoint.pt",
    )
    write_json(args.output_dir / "train_summary.json", {"best_metrics": best_metrics, "final_metrics": final_metrics, "dataset": summary})


def command_eval(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    config = ckpt["config"]
    if not args.allow_cli_data:
        for key, value in config.items():
            if hasattr(args, key):
                setattr(args, key, Path(value) if key == "data_root" else value)
    norm = ckpt["normalization"]
    stats = NormStats(
        action_mean=[float(v) for v in norm["action_mean"]],
        action_std=[float(v) for v in norm["action_std"]],
        state_mean=[float(v) for v in norm["state_mean"]],
        state_std=[float(v) for v in norm["state_std"]],
    )
    _, val_loader, _, summary = build_loaders(args, stats)
    model = make_model(config, summary["state_dim"], summary["action_dim"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    schedule = DDPMSchedule(config["diffusion_steps"], config["beta_start"], config["beta_end"], device)
    metrics = evaluate(model, schedule, val_loader, device, args.eval_max_batches, True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "eval_summary.json", {"metrics": metrics, "dataset": summary})

    rows: list[dict] = []
    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if args.prediction_batches > 0 and batch_idx >= args.prediction_batches:
                break
            batch = to_device(batch, device)
            pred = schedule.sample(model, batch["video"], batch["state"], tuple(batch["action"].shape))
            err = (pred - batch["action"]).pow(2).mean(dim=(1, 2)).cpu().numpy()
            for i, mse in enumerate(err):
                rows.append({"task": batch["task"][i], "episode": int(batch["episode"][i]), "action_start": int(batch["action_start"][i]), "mse_norm": float(mse)})
    if rows:
        write_rows(args.output_dir / "predictions.csv", rows)
    print(json.dumps({"metrics": metrics, "output_dir": str(args.output_dir)}, ensure_ascii=False))


def add_data_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--tasks", default=DEFAULT_TASKS)
    parser.add_argument("--camera-key", default="robot_camera")
    parser.add_argument("--action-key", default="action")
    parser.add_argument("--state-keys", default=DEFAULT_STATE_KEYS)
    parser.add_argument("--max-episodes-per-task", type=int, default=0)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--obs-horizon", type=int, default=2)
    parser.add_argument("--pred-horizon", type=int, default=8)
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--norm-max-frames", type=int, default=20000)
    parser.add_argument("--resize", default="96x96")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda:0")


def add_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--time-dim", type=int, default=64)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--diffusion-steps", type=int, default=32)
    parser.add_argument("--beta-start", type=float, default=1e-4)
    parser.add_argument("--beta-end", type=float, default=0.02)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="H2R HDF5 Diffusion Policy BC")
    sub = parser.add_subparsers(dest="command", required=True)
    inspect_p = sub.add_parser("inspect")
    add_data_args(inspect_p)
    inspect_p.add_argument("--output-json", type=Path, default=None)
    inspect_p.set_defaults(func=command_inspect)

    train_p = sub.add_parser("train")
    add_data_args(train_p)
    add_model_args(train_p)
    train_p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    train_p.add_argument("--steps", type=int, default=1000)
    train_p.add_argument("--lr", type=float, default=3e-4)
    train_p.add_argument("--weight-decay", type=float, default=1e-4)
    train_p.add_argument("--grad-clip", type=float, default=1.0)
    train_p.add_argument("--log-every", type=int, default=20)
    train_p.add_argument("--eval-every", type=int, default=100)
    train_p.add_argument("--eval-max-batches", type=int, default=0)
    train_p.add_argument("--eval-sample-actions", action="store_true")
    train_p.add_argument("--resume-log", action="store_true")
    train_p.set_defaults(func=command_train)

    eval_p = sub.add_parser("eval")
    add_data_args(eval_p)
    add_model_args(eval_p)
    eval_p.add_argument("--checkpoint", type=Path, required=True)
    eval_p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "eval")
    eval_p.add_argument("--eval-max-batches", type=int, default=0)
    eval_p.add_argument("--prediction-batches", type=int, default=4)
    eval_p.add_argument("--allow-cli-data", action="store_true")
    eval_p.set_defaults(func=command_eval)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
