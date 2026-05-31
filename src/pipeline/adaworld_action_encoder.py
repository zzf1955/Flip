"""AdaWorld action encoder extraction for Humanoid Everyday H1 videos.

This module uses only the AdaWorld latent action model (LAM) encoder:

  (frame_t, frame_{t+1}) -> 32-d continuous latent action z_t

It does not load or run the AdaWorld world model. Inputs are RGB frames from the
Humanoid Everyday H1 LeRobot layout:

  - data/chunk-*/episode_*.parquet provides frame indices and episode ends.
  - videos/chunk-*/egocentric/episode_*.mp4 provides RGB frames.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch

from src.core.config import MAIN_ROOT
from src.pipeline.wan_pair_idm import write_json

LATENT_DIM = 32
DEFAULT_H1_DATASET = "data/humanoid-everyday-h1-chunks0-6-8-200"


@dataclass(frozen=True)
class H1FramePairSample:
    video_path: str
    parquet_path: str
    episode: int
    chunk: int
    rel_frame_t: int
    rel_frame_tp1: int


def worktree_parent_root() -> Path | None:
    root = Path(MAIN_ROOT).resolve()
    if root.parent.name == ".worktrees":
        return root.parent.parent
    return None


def resolve_existing_path(value: str | None, candidates: list[Path], label: str) -> Path:
    if value:
        path = Path(value).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")
        return path
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    candidate_text = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"{label} not found; checked: {candidate_text}")


def default_project_candidates(relative: str) -> list[Path]:
    candidates = [Path(MAIN_ROOT) / relative]
    parent = worktree_parent_root()
    if parent is not None:
        candidates.append(parent / relative)
    return candidates


def resolve_data_root(value: str | None) -> Path:
    return resolve_existing_path(
        value,
        default_project_candidates(DEFAULT_H1_DATASET),
        "Humanoid Everyday H1 data root",
    )


def resolve_adaworld_root(value: str | None) -> Path:
    return resolve_existing_path(
        value,
        default_project_candidates("ref-AdaWorld"),
        "AdaWorld code repository",
    )


def resolve_checkpoint(value: str | None) -> Path:
    return resolve_existing_path(
        value,
        default_project_candidates("ref-AdaWorld-hf/lam.ckpt"),
        "AdaWorld LAM checkpoint",
    )


def git_revision(repo_root: Path) -> str:
    if not (repo_root / ".git").exists():
        raise FileNotFoundError(f"Git metadata not found for AdaWorld repo: {repo_root}")
    return subprocess.check_output(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        text=True,
    ).strip()


def discover_h1_frame_pairs(
    data_root: Path,
    *,
    max_samples: int,
    seed: int,
    frame_stride: int,
    max_pairs_per_episode: int,
) -> list[H1FramePairSample]:
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

    samples: list[H1FramePairSample] = []
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

        df = pd.read_parquet(parquet_path, columns=["frame_index", "next.done"])
        if df.empty:
            raise ValueError(f"Humanoid episode parquet is empty: {parquet_path}")
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
            samples.append(
                H1FramePairSample(
                    video_path=str(video_path),
                    parquet_path=str(parquet_path),
                    episode=episode,
                    chunk=chunk,
                    rel_frame_t=row_idx,
                    rel_frame_tp1=row_idx + 1,
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


def center_crop_square(frame_rgb: np.ndarray) -> np.ndarray:
    height, width = frame_rgb.shape[:2]
    square = min(height, width)
    top = (height - square) // 2
    left = (width - square) // 2
    return frame_rgb[top : top + square, left : left + square]


def read_h1_pair_for_adaworld(
    video_path: str | Path,
    rel_frame_t: int,
    *,
    resolution: int,
) -> np.ndarray:
    if rel_frame_t < 0:
        raise ValueError(f"rel_frame_t must be non-negative, got {rel_frame_t}")
    if resolution <= 0:
        raise ValueError(f"resolution must be positive, got {resolution}")
    video_path = str(video_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, rel_frame_t)
    frames: list[np.ndarray] = []
    for expected in (rel_frame_t, rel_frame_t + 1):
        ok, frame_bgr = cap.read()
        if not ok:
            cap.release()
            raise RuntimeError(f"Failed to read frame {expected} from {video_path}")
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb = center_crop_square(frame_rgb)
        if frame_rgb.shape[0] != resolution or frame_rgb.shape[1] != resolution:
            frame_rgb = cv2.resize(
                frame_rgb,
                (resolution, resolution),
                interpolation=cv2.INTER_CUBIC,
            )
        frames.append(frame_rgb.astype(np.float32) / 255.0)
    cap.release()
    return np.stack(frames, axis=0)


class H1AdaWorldPairDataset(torch.utils.data.Dataset):
    def __init__(self, samples: list[H1FramePairSample], resolution: int) -> None:
        self.samples = samples
        self.resolution = resolution

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.samples[idx]
        videos = read_h1_pair_for_adaworld(
            sample.video_path,
            sample.rel_frame_t,
            resolution=self.resolution,
        )
        return {
            "videos": videos,
            "sample_index": idx,
        }


def collate_h1_adaworld_batch(items: list[dict[str, Any]]) -> dict[str, Any]:
    videos = torch.from_numpy(np.stack([item["videos"] for item in items], axis=0))
    if videos.ndim != 5 or videos.shape[1] != 2 or videos.shape[-1] != 3:
        raise ValueError(f"Expected videos [B,2,H,W,3], got {tuple(videos.shape)}")
    return {
        "videos": videos.to(dtype=torch.float32),
        "sample_index": [int(item["sample_index"]) for item in items],
    }


def patch_adaworld_positional_encoding(adaworld_root: Path) -> None:
    lam_source = adaworld_root / "lam"
    if not (lam_source / "lam").is_dir():
        raise FileNotFoundError(f"AdaWorld lam package not found under {lam_source}")
    source = str(lam_source)
    if source not in sys.path:
        sys.path.insert(0, source)

    from lam.modules.blocks import PositionalEncoding

    def forward(self: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
        pos_enc = self.pos_enc[: x.shape[2]].to(device=x.device, dtype=x.dtype)
        return x + pos_enc

    PositionalEncoding.forward = forward


def build_adaworld_lam(adaworld_root: Path) -> torch.nn.Module:
    patch_adaworld_positional_encoding(adaworld_root)
    from lam.modules import LatentActionModel

    return LatentActionModel(
        in_dim=3,
        model_dim=1024,
        latent_dim=LATENT_DIM,
        patch_size=16,
        enc_blocks=16,
        dec_blocks=16,
        num_heads=16,
        dropout=0.0,
    )


def load_adaworld_action_encoder(
    adaworld_root: Path,
    checkpoint: Path,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.nn.Module:
    model = build_adaworld_lam(adaworld_root)
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    if not isinstance(state, dict):
        raise ValueError(f"Unsupported AdaWorld checkpoint payload type: {type(state)!r}")
    lam_state = {
        key.removeprefix("lam."): value
        for key, value in state.items()
        if key.startswith("lam.")
    }
    if not lam_state:
        lam_state = state
    model.load_state_dict(lam_state, strict=True)
    model.eval()
    model.to(device=device)
    if dtype != torch.float32:
        model.to(dtype=dtype)
    return model


def parse_dtype(value: str, device: torch.device) -> torch.dtype:
    if value == "fp32":
        return torch.float32
    if value == "fp16":
        if device.type != "cuda":
            raise ValueError("fp16 inference requires a CUDA device")
        return torch.float16
    if value == "bf16":
        if device.type != "cuda":
            raise ValueError("bf16 inference requires a CUDA device")
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype={value!r}")


@torch.inference_mode()
def encode_batch(
    model: torch.nn.Module,
    videos: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> np.ndarray:
    videos = videos.to(device=device, dtype=dtype)
    outputs = model.encode(videos)
    z_mu = outputs["z_mu"]
    if z_mu.ndim != 2 or z_mu.shape[1] != LATENT_DIM:
        raise ValueError(f"Expected latent shape [N,{LATENT_DIM}], got {tuple(z_mu.shape)}")
    return z_mu.detach().float().cpu().numpy()


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def extract_latents(args: argparse.Namespace) -> None:
    seed = int(args.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    data_root = resolve_data_root(args.data_root)
    adaworld_root = resolve_adaworld_root(args.adaworld_root)
    checkpoint = resolve_checkpoint(args.checkpoint)
    revision = git_revision(adaworld_root)
    device = torch.device(args.device)
    dtype = parse_dtype(args.dtype, device)
    if int(args.resolution) % 16 != 0:
        raise ValueError(f"resolution must be divisible by AdaWorld patch size 16: {args.resolution}")

    samples = discover_h1_frame_pairs(
        data_root,
        max_samples=int(args.max_samples),
        seed=seed,
        frame_stride=int(args.frame_stride),
        max_pairs_per_episode=int(args.max_pairs_per_episode),
    )
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = load_adaworld_action_encoder(
        adaworld_root,
        checkpoint,
        device=device,
        dtype=dtype,
    )
    loader = torch.utils.data.DataLoader(
        H1AdaWorldPairDataset(samples, int(args.resolution)),
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.workers),
        collate_fn=collate_h1_adaworld_batch,
        drop_last=False,
        pin_memory=device.type == "cuda",
    )

    latent_chunks: list[np.ndarray] = []
    manifest_rows: list[dict[str, Any]] = []
    for batch in loader:
        latents = encode_batch(model, batch["videos"], device=device, dtype=dtype)
        latent_chunks.append(latents)
        for local_idx, sample_idx in enumerate(batch["sample_index"]):
            sample = samples[int(sample_idx)]
            manifest_rows.append(
                {
                    **asdict(sample),
                    "sample_index": int(sample_idx),
                    "latent": [float(v) for v in latents[local_idx].tolist()],
                }
            )

    latent_actions = np.concatenate(latent_chunks, axis=0).astype(np.float32)
    if latent_actions.shape != (len(samples), LATENT_DIM):
        raise ValueError(
            f"Expected latent array {(len(samples), LATENT_DIM)}, got {latent_actions.shape}"
        )
    if not np.isfinite(latent_actions).all():
        raise ValueError("AdaWorld latent action output contains non-finite values")

    np.savez_compressed(
        out_dir / "latent_actions.npz",
        latent_actions=latent_actions,
        episode=np.asarray([sample.episode for sample in samples], dtype=np.int64),
        chunk=np.asarray([sample.chunk for sample in samples], dtype=np.int64),
        rel_frame_t=np.asarray([sample.rel_frame_t for sample in samples], dtype=np.int64),
        rel_frame_tp1=np.asarray([sample.rel_frame_tp1 for sample in samples], dtype=np.int64),
    )
    write_jsonl(manifest_rows, out_dir / "manifest.jsonl")
    summary = {
        "alignment": "(frame_t, frame_t+1) -> 32d_continuous_latent_action_z_t",
        "n_samples": int(latent_actions.shape[0]),
        "latent_dim": int(latent_actions.shape[1]),
        "latent_mean": float(latent_actions.mean()),
        "latent_std": float(latent_actions.std()),
        "latent_min": float(latent_actions.min()),
        "latent_max": float(latent_actions.max()),
        "data_root": str(data_root),
        "adaworld_root": str(adaworld_root),
        "adaworld_revision": revision,
        "checkpoint": str(checkpoint),
        "resolution": int(args.resolution),
        "dtype": args.dtype,
        "device": str(device),
    }
    write_json(out_dir / "summary.json", summary)
    print(json.dumps({"summary": summary, "out_dir": str(out_dir)}, indent=2), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract AdaWorld 32-d continuous latent actions from H1 two-frame RGB inputs"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    extract_p = sub.add_parser("extract", help="extract latent actions for H1 adjacent frame pairs")
    extract_p.add_argument("--data-root", default=None)
    extract_p.add_argument("--adaworld-root", default=None)
    extract_p.add_argument("--checkpoint", default=None)
    extract_p.add_argument("--output-dir", required=True)
    extract_p.add_argument("--device", default="cuda:0")
    extract_p.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="fp16")
    extract_p.add_argument("--resolution", type=int, default=256)
    extract_p.add_argument("--max-samples", type=int, default=8)
    extract_p.add_argument("--frame-stride", type=int, default=1)
    extract_p.add_argument("--max-pairs-per-episode", type=int, default=0)
    extract_p.add_argument("--batch-size", type=int, default=1)
    extract_p.add_argument("--workers", type=int, default=2)
    extract_p.add_argument("--seed", type=int, default=42)
    extract_p.set_defaults(func=extract_latents)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
