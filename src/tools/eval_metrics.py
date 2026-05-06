"""Compute reconstruction metrics between generated and ground-truth eval videos.

Scans a training run's eval/ directory, pairs gen_XX.mp4 with gt_XX.mp4,
and computes per-frame and per-video metrics.

Metrics:
  - MSE    (pixel-level, lower is better)
  - PSNR   (pixel-level, higher is better)
  - SSIM   (structural similarity, higher is better)
  - LPIPS  (perceptual distance via VGG, lower is better)
  - FID    (Frechet Inception Distance across all frames, lower is better)
  - FVD    (Frechet Video Distance via S3D video features, lower is better)
  - foreground/background MSE/PSNR/SSIM, foreground Local FID, and
    black-background FVD when SAM2 masks are available and selected

Usage:
  python -m src.tools.eval_metrics --run 2026-04-18_163933
  python -m src.tools.eval_metrics --run 2026-04-18_163933 --split ood --device cuda:2
  python -m src.tools.eval_metrics --run 2026-04-18_163933 --steps 0200 0400 0800
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

MAIN_ROOT = Path(os.environ.get("FLIP_MAIN_ROOT", "/disk_n/zzf/flip"))
TRAINING_LOG_ROOT = MAIN_ROOT / "training_data" / "log"
FFMPEG = os.environ.get(
    "FFMPEG_BIN",
    "/home/leadtek/miniconda3/envs/flip/bin/ffmpeg",
)
FFPROBE = FFMPEG.replace("ffmpeg", "ffprobe")
SEGMENT_FPS = 30.0
TARGET_FPS = 16.0
REGION_NAMES = ("foreground", "background")
LOCAL_FID_MARGIN = 24
LOCAL_FID_SIZE = 299
DEFAULT_METRIC_WORKERS = min(8, os.cpu_count() or 1)
DEFAULT_LPIPS_BATCH_SIZE = 16
DEFAULT_FEATURE_BATCH_SIZE = 32
DEFAULT_FVD_BATCH_SIZE = 4
ProgressCallback = Callable[[str, int, int], None]


@dataclass(frozen=True)
class PairMetricData:
    idx: int
    gen_frames: np.ndarray
    gt_frames: np.ndarray
    masks: np.ndarray | None
    metrics: dict


# ── Video IO ──────��──────────────────────────────���────────────────────


def read_video_frames(path: str) -> np.ndarray:
    """Read MP4 -> uint8 numpy array (T, H, W, 3) via ffmpeg pipe."""
    info_cmd = [
        FFPROBE, "-v", "error",
        "-select_streams", "v",
        "-show_entries", "stream=width,height",
        "-of", "csv=p=0", path,
    ]
    info = subprocess.run(info_cmd, capture_output=True, text=True)
    w, h = map(int, info.stdout.strip().split(","))

    cmd = [
        FFMPEG, "-i", path,
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-v", "error", "pipe:1",
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed on {path}: {proc.stderr.decode()}")
    raw = np.frombuffer(proc.stdout, dtype=np.uint8).copy()
    return raw.reshape(-1, h, w, 3)


# ── SAM2 mask region helpers ─────────────────────────────────────────


def clip_mask_indices(clip_start: float, clip_dur: float,
                      num_frames: int, mask_count: int) -> list[int]:
    """Map output video frames to original 30fps segment mask frame indices."""
    if num_frames <= 0:
        raise ValueError("num_frames must be positive")
    if mask_count <= 0:
        raise ValueError("mask_count must be positive")
    base = int(round(clip_start * SEGMENT_FPS))
    clip_frames = max(1, int(round(clip_dur * SEGMENT_FPS)))
    indices = []
    for i in range(num_frames):
        offset = min(round(i * SEGMENT_FPS / TARGET_FPS), clip_frames - 1)
        indices.append(min(max(base + offset, 0), mask_count - 1))
    return indices


def resolve_sam2_mask_path(record: dict, mask_root: str | Path) -> Path:
    task = record.get("robot_task") or record.get("task")
    episode = record.get("episode")
    seg = record.get("seg")
    missing = [
        name for name, value in (
            ("robot_task/task", task),
            ("episode", episode),
            ("seg", seg),
        )
        if not value
    ]
    if missing:
        raise ValueError(f"Selected record missing SAM2 mask fields {missing}: {record}")
    return Path(mask_root) / str(task) / str(episode) / f"{seg}.npz"


def load_clip_mask_stack(
    record: dict,
    mask_root: str | Path,
    num_frames: int,
    frame_shape: tuple[int, int],
) -> np.ndarray:
    mask_path = resolve_sam2_mask_path(record, mask_root)
    if not mask_path.is_file():
        raise FileNotFoundError(f"SAM2 mask not found for region metrics: {mask_path}")
    for field in ("clip_start", "clip_dur"):
        if field not in record:
            raise ValueError(f"Selected record missing {field} for SAM2 mask alignment: {record}")
    with np.load(mask_path) as mask_npz:
        masks = mask_npz["masks"]
    if masks.ndim != 3:
        raise ValueError(f"Invalid SAM2 mask shape in {mask_path}: {masks.shape}")
    indices = clip_mask_indices(
        float(record["clip_start"]), float(record["clip_dur"]), num_frames, len(masks),
    )
    clip_masks = masks[indices]
    if record.get("augment") == "hflip":
        clip_masks = clip_masks[:, :, ::-1]
    expected_h, expected_w = frame_shape
    if clip_masks.shape[1:] != (expected_h, expected_w):
        raise ValueError(
            f"Mask/frame shape mismatch for {mask_path}: "
            f"mask={clip_masks.shape[1:]}, frame={(expected_h, expected_w)}"
        )
    return (clip_masks > 128)


def split_video_by_mask(frames: np.ndarray, masks: np.ndarray) -> dict[str, np.ndarray]:
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"Expected video frames shape (T,H,W,3), got {frames.shape}")
    if masks.shape != frames.shape[:3]:
        raise ValueError(f"Mask shape {masks.shape} does not match frames {frames.shape[:3]}")
    keep_foreground = masks[..., None]
    return {
        "foreground": np.where(keep_foreground, frames, 0).astype(np.uint8),
        "background": np.where(~keep_foreground, frames, 0).astype(np.uint8),
    }


def mask_bbox(mask: np.ndarray, margin: int = LOCAL_FID_MARGIN) -> tuple[int, int, int, int]:
    """Return expanded xyxy bbox for a non-empty 2D mask."""
    if mask.ndim != 2:
        raise ValueError(f"Expected 2D mask, got {mask.shape}")
    if margin < 0:
        raise ValueError(f"margin must be non-negative, got {margin}")
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        raise ValueError("Local FID mask has no foreground pixels")
    h, w = mask.shape
    x1 = max(int(xs.min()) - margin, 0)
    y1 = max(int(ys.min()) - margin, 0)
    x2 = min(int(xs.max()) + margin + 1, w)
    y2 = min(int(ys.max()) + margin + 1, h)
    if x1 >= x2 or y1 >= y2:
        raise ValueError(f"Invalid Local FID bbox {(x1, y1, x2, y2)} for mask {mask.shape}")
    return x1, y1, x2, y2


def crop_video_by_mask_bbox(
    frames: np.ndarray,
    masks: np.ndarray,
    margin: int = LOCAL_FID_MARGIN,
    output_size: int = LOCAL_FID_SIZE,
) -> np.ndarray:
    """Crop each frame to the mask bbox and resize crops for Local FID.

    This follows the Local FID protocol used by object inpainting work: the
    same mask-derived bbox is applied to generated and GT frames, making FID
    focus on the edited foreground instead of unchanged background.
    """
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"Expected video frames shape (T,H,W,3), got {frames.shape}")
    if masks.shape != frames.shape[:3]:
        raise ValueError(f"Mask shape {masks.shape} does not match frames {frames.shape[:3]}")
    if output_size <= 0:
        raise ValueError(f"output_size must be positive, got {output_size}")

    crops = []
    for frame, mask in zip(frames, masks):
        x1, y1, x2, y2 = mask_bbox(mask, margin)
        crop = frame[y1:y2, x1:x2]
        resized = Image.fromarray(crop).resize(
            (output_size, output_size),
            Image.Resampling.BILINEAR,
        )
        crops.append(np.asarray(resized, dtype=np.uint8))
    return np.stack(crops, axis=0)


def region_masks(masks: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "foreground": masks,
        "background": ~masks,
    }


def mse_to_psnr(mse: float, data_range: float = 255.0) -> float:
    if mse < 0.0:
        raise ValueError(f"MSE must be non-negative, got {mse}")
    if mse == 0.0:
        return float("inf")
    return float(20.0 * np.log10(data_range / np.sqrt(mse)))


def masked_mse(gen_frames: np.ndarray, gt_frames: np.ndarray, mask: np.ndarray) -> float:
    if mask.shape != gen_frames.shape[:3]:
        raise ValueError(f"Mask shape {mask.shape} does not match frames {gen_frames.shape[:3]}")
    if not mask.any():
        raise ValueError("Masked MSE region has no pixels")
    diff = (
        gen_frames.astype(np.float64) - gt_frames.astype(np.float64)
    ) ** 2
    return float(diff[mask].mean())


def masked_ssim(
    gen_frames: np.ndarray,
    gt_frames: np.ndarray,
    mask: np.ndarray,
) -> float:
    if mask.shape != gen_frames.shape[:3]:
        raise ValueError(f"Mask shape {mask.shape} does not match frames {gen_frames.shape[:3]}")
    values = []
    for t in range(len(gen_frames)):
        frame_mask = mask[t]
        if not frame_mask.any():
            continue
        _, ssim_map = structural_similarity(
            gt_frames[t],
            gen_frames[t],
            channel_axis=2,
            data_range=255,
            full=True,
        )
        if ssim_map.ndim == 3:
            weighted = ssim_map[frame_mask].mean()
        else:
            weighted = ssim_map[frame_mask].mean()
        values.append(float(weighted))
    if not values:
        raise ValueError("Masked SSIM region has no pixels")
    return float(np.mean(values))


# ── LPIPS (self-contained VGG) ───────────────��────────────────────────


class LPIPS(nn.Module):
    """Learned Perceptual Image Patch Similarity using VGG16 features.

    Simplified: extracts features from 5 VGG16 layers, normalizes, and
    computes mean L2 distance. No learned linear weights (equal weighting),
    close to the official LPIPS "vgg" variant.
    """

    def __init__(self):
        super().__init__()
        from torchvision.models import vgg16, VGG16_Weights
        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features.eval()
        slices = [4, 9, 16, 23, 30]
        self.blocks = nn.ModuleList()
        prev = 0
        for s in slices:
            self.blocks.append(nn.Sequential(*list(vgg.children())[prev:s]))
            prev = s
        for p in self.parameters():
            p.requires_grad = False
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    @torch.no_grad()
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """x, y: (B, 3, H, W) in [0, 1]. Returns (B,) distances."""
        x = (x - self.mean) / self.std
        y = (y - self.mean) / self.std
        dists = []
        for block in self.blocks:
            x = block(x)
            y = block(y)
            xn = F.normalize(x, dim=1)
            yn = F.normalize(y, dim=1)
            dists.append((xn - yn).pow(2).mean(dim=(1, 2, 3)))
        return torch.stack(dists).mean(dim=0)


# ── InceptionV3 feature extractor (for FID) ──────────────────────────


class InceptionFeatureExtractor(nn.Module):
    """Extract pool3 (2048-d) features from InceptionV3."""

    def __init__(self):
        super().__init__()
        from torchvision.models import inception_v3, Inception_V3_Weights
        net = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1).eval()
        self.blocks = nn.Sequential(
            net.Conv2d_1a_3x3, net.Conv2d_2a_3x3, net.Conv2d_2b_3x3,
            nn.MaxPool2d(3, stride=2),
            net.Conv2d_3b_1x1, net.Conv2d_4a_3x3,
            nn.MaxPool2d(3, stride=2),
            net.Mixed_5b, net.Mixed_5c, net.Mixed_5d,
            net.Mixed_6a, net.Mixed_6b, net.Mixed_6c, net.Mixed_6d, net.Mixed_6e,
            net.Mixed_7a, net.Mixed_7b, net.Mixed_7c,
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        for p in self.parameters():
            p.requires_grad = False
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, H, W) in [0, 1] -> (B, 2048)."""
        x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
        x = (x - self.mean) / self.std
        return self.blocks(x).flatten(1)


def frechet_distance(mu1, sigma1, mu2, sigma2):
    """Compute Frechet Distance between two multivariate Gaussians."""
    from scipy.linalg import sqrtm
    diff = mu1 - mu2
    covmean, _ = sqrtm(sigma1 @ sigma2, disp=False)
    if not np.isfinite(covmean).all():
        eps = 1e-6
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = sqrtm((sigma1 + offset) @ (sigma2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean))


# ── Per-video pair metrics ─��──────────────────────────────────────────


def compute_pairwise_metrics(
    gen_frames: np.ndarray,
    gt_frames: np.ndarray,
    lpips_model: LPIPS | None,
    device: torch.device,
    masks: np.ndarray | None = None,
    lpips_batch_size: int = 8,
) -> dict:
    """Compute PSNR, SSIM, LPIPS between paired frame arrays (T,H,W,3)."""
    if len(gen_frames) != len(gt_frames):
        raise ValueError(
            f"Frame count mismatch: gen has {len(gen_frames)} frames, "
            f"gt has {len(gt_frames)} frames")
    T = len(gen_frames)
    psnrs, ssims, lpipss = [], [], []
    global_mse = float(np.mean(
        (gen_frames.astype(np.float64) - gt_frames.astype(np.float64)) ** 2
    ))

    for t in range(T):
        psnrs.append(peak_signal_noise_ratio(gt_frames[t], gen_frames[t], data_range=255))
        ssims.append(structural_similarity(gt_frames[t], gen_frames[t], channel_axis=2, data_range=255))

    if lpips_model is not None:
        if lpips_batch_size <= 0:
            raise ValueError(f"lpips_batch_size must be positive, got {lpips_batch_size}")
        gen_t = torch.from_numpy(gen_frames[:T]).permute(0, 3, 1, 2).float() / 255.0
        gt_t = torch.from_numpy(gt_frames[:T]).permute(0, 3, 1, 2).float() / 255.0
        for i in range(0, T, lpips_batch_size):
            batch_gen = gen_t[i:i + lpips_batch_size].to(device)
            batch_gt = gt_t[i:i + lpips_batch_size].to(device)
            d = lpips_model(batch_gen, batch_gt)
            lpipss.extend(d.cpu().tolist())

    result = {
        "mse": global_mse,
        "psnr": float(np.mean(psnrs)),
        "ssim": float(np.mean(ssims)),
    }
    if masks is not None:
        for name, region_mask in region_masks(masks).items():
            region_mse = masked_mse(gen_frames, gt_frames, region_mask)
            result[f"{name}_mse"] = region_mse
            result[f"{name}_psnr"] = mse_to_psnr(region_mse)
            result[f"{name}_ssim"] = masked_ssim(gen_frames, gt_frames, region_mask)
    if lpipss:
        result["lpips"] = float(np.mean(lpipss))
    return result


def compute_lpips_per_video(
    video_pairs: list[tuple[np.ndarray, np.ndarray]],
    lpips_model: LPIPS,
    device: torch.device,
    batch_size: int = DEFAULT_LPIPS_BATCH_SIZE,
    progress_callback: ProgressCallback | None = None,
) -> list[float]:
    """Compute LPIPS with frame batches shared across all videos."""
    if batch_size <= 0:
        raise ValueError(f"LPIPS batch size must be positive, got {batch_size}")
    totals = [len(gen) for gen, _ in video_pairs]
    total_frames = sum(totals)
    if total_frames == 0:
        raise ValueError("No frames available for LPIPS")

    per_video_scores = [[] for _ in video_pairs]
    batch_gen: list[np.ndarray] = []
    batch_gt: list[np.ndarray] = []
    batch_owners: list[int] = []
    done = 0

    def flush_batch() -> None:
        nonlocal done
        if not batch_gen:
            return
        gen_t = (
            torch.from_numpy(np.stack(batch_gen, axis=0))
            .permute(0, 3, 1, 2)
            .float()
            / 255.0
        )
        gt_t = (
            torch.from_numpy(np.stack(batch_gt, axis=0))
            .permute(0, 3, 1, 2)
            .float()
            / 255.0
        )
        scores = lpips_model(gen_t.to(device), gt_t.to(device)).cpu().tolist()
        for owner, score in zip(batch_owners, scores):
            per_video_scores[owner].append(float(score))
        done += len(scores)
        if progress_callback is not None:
            progress_callback("lpips", done, total_frames)
        batch_gen.clear()
        batch_gt.clear()
        batch_owners.clear()

    for video_idx, (gen_frames, gt_frames) in enumerate(video_pairs):
        if len(gen_frames) != len(gt_frames):
            raise ValueError(
                f"Frame count mismatch in LPIPS video {video_idx}: "
                f"gen={len(gen_frames)} gt={len(gt_frames)}"
            )
        for gen_frame, gt_frame in zip(gen_frames, gt_frames):
            batch_gen.append(gen_frame)
            batch_gt.append(gt_frame)
            batch_owners.append(video_idx)
            if len(batch_gen) == batch_size:
                flush_batch()
    flush_batch()

    return [float(np.mean(scores)) for scores in per_video_scores]


# ── Inception features ���───────────────────────────────────────────────


def collect_inception_features(
    video_arrays: list[np.ndarray],
    extractor: InceptionFeatureExtractor,
    device: torch.device,
    batch_size: int = DEFAULT_FEATURE_BATCH_SIZE,
    progress_callback: ProgressCallback | None = None,
    progress_phase: str = "fid",
) -> np.ndarray:
    """Extract Inception features from all frames -> (N_total_frames, 2048)."""
    if batch_size <= 0:
        raise ValueError(f"Inception feature batch size must be positive, got {batch_size}")
    all_frames = np.concatenate(video_arrays, axis=0)
    feats = []
    for i in range(0, len(all_frames), batch_size):
        chunk = all_frames[i:i + batch_size]
        batch = torch.from_numpy(chunk).permute(0, 3, 1, 2).float() / 255.0
        feats.append(extractor(batch.to(device)).cpu().numpy())
        if progress_callback is not None:
            progress_callback(progress_phase, min(i + len(chunk), len(all_frames)), len(all_frames))
    return np.concatenate(feats, axis=0)


def collect_local_inception_features(
    video_arrays: list[np.ndarray],
    mask_arrays: list[np.ndarray],
    extractor: InceptionFeatureExtractor,
    device: torch.device,
    margin: int = LOCAL_FID_MARGIN,
    batch_size: int = DEFAULT_FEATURE_BATCH_SIZE,
    progress_callback: ProgressCallback | None = None,
    progress_phase: str = "local_fid",
) -> np.ndarray:
    """Extract Inception features from mask-bbox frame crops for Local FID."""
    if len(video_arrays) != len(mask_arrays):
        raise ValueError(
            f"Video/mask count mismatch: {len(video_arrays)} vs {len(mask_arrays)}"
        )
    cropped_videos = [
        crop_video_by_mask_bbox(video, masks, margin=margin)
        for video, masks in zip(video_arrays, mask_arrays)
    ]
    return collect_inception_features(
        cropped_videos,
        extractor,
        device,
        batch_size,
        progress_callback=progress_callback,
        progress_phase=progress_phase,
    )


def compute_fid(feats_gen: np.ndarray, feats_gt: np.ndarray) -> float:
    mu_gen, sigma_gen = feats_gen.mean(0), np.cov(feats_gen, rowvar=False)
    mu_gt, sigma_gt = feats_gt.mean(0), np.cov(feats_gt, rowvar=False)
    return frechet_distance(mu_gen, sigma_gen, mu_gt, sigma_gt)


# ── FVD ───────────────────────────────────────────────────────────────


class VideoFeatureExtractor(nn.Module):
    """Extract S3D Kinetics video features for FVD-style evaluation."""

    NUM_FRAMES = 16
    SIZE = 224

    def __init__(self):
        super().__init__()
        from torchvision.models.video import S3D_Weights, s3d
        net = s3d(weights=S3D_Weights.KINETICS400_V1).eval()
        self.features = net.features
        self.avgpool = net.avgpool
        for p in self.parameters():
            p.requires_grad = False
        self.register_buffer(
            "mean", torch.tensor([0.43216, 0.394666, 0.37645]).view(1, 3, 1, 1, 1))
        self.register_buffer(
            "std", torch.tensor([0.22803, 0.22145, 0.216989]).view(1, 3, 1, 1, 1))

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, T, H, W) in [0, 1] -> (B, 1024)."""
        x = F.interpolate(
            x,
            size=(self.NUM_FRAMES, self.SIZE, self.SIZE),
            mode="trilinear",
            align_corners=False,
        )
        x = (x - self.mean) / self.std
        x = self.features(x)
        return self.avgpool(x).flatten(1)


def collect_video_features(
    video_arrays: list[np.ndarray],
    extractor: VideoFeatureExtractor,
    device: torch.device,
    batch_size: int = DEFAULT_FVD_BATCH_SIZE,
    progress_callback: ProgressCallback | None = None,
    progress_phase: str = "fvd",
) -> np.ndarray:
    """Extract one S3D spatiotemporal feature vector per video -> (V, 1024)."""
    if batch_size <= 0:
        raise ValueError(f"FVD batch size must be positive, got {batch_size}")
    feats = []
    for i in range(0, len(video_arrays), batch_size):
        chunk = video_arrays[i:i + batch_size]
        tensors = [
            torch.from_numpy(vid).permute(3, 0, 1, 2).float() / 255.0
            for vid in chunk
        ]
        batch = torch.stack(tensors, dim=0)
        feats.append(extractor(batch.to(device)).cpu().numpy())
        if progress_callback is not None:
            progress_callback(progress_phase, min(i + len(chunk), len(video_arrays)), len(video_arrays))
    return np.concatenate(feats, axis=0)


def compute_fvd(
    gen_videos: list[np.ndarray],
    gt_videos: list[np.ndarray],
    extractor: VideoFeatureExtractor,
    device: torch.device,
    batch_size: int = DEFAULT_FVD_BATCH_SIZE,
    progress_callback: ProgressCallback | None = None,
    progress_prefix: str = "fvd",
) -> float | None:
    """Compute FVD with S3D video features.

    Returns None if too few videos are available for covariance estimation.
    """
    if len(gen_videos) < 2:
        return None
    feats_gen = collect_video_features(
        gen_videos,
        extractor,
        device,
        batch_size=batch_size,
        progress_callback=progress_callback,
        progress_phase=f"{progress_prefix}/gen",
    )
    feats_gt = collect_video_features(
        gt_videos,
        extractor,
        device,
        batch_size=batch_size,
        progress_callback=progress_callback,
        progress_phase=f"{progress_prefix}/gt",
    )
    mu_gen, sigma_gen = feats_gen.mean(0), np.cov(feats_gen, rowvar=False)
    mu_gt, sigma_gt = feats_gt.mean(0), np.cov(feats_gt, rowvar=False)
    if sigma_gen.ndim < 2:
        return None
    return frechet_distance(mu_gen, sigma_gen, mu_gt, sigma_gt)


# ── Main ──────────���───────────────────────────��───────────────────────


def find_pairs(step_dir: str) -> list[tuple[str, str, int]]:
    """Find (gen_path, gt_path, index) pairs in a step directory."""
    pairs = []
    for f in sorted(os.listdir(step_dir)):
        if f.startswith("gen_") and f.endswith(".mp4"):
            sample_id = f[len("gen_"):-len(".mp4")]
            idx = int(sample_id)
            gt = os.path.join(step_dir, f"gt_{sample_id}.mp4")
            if os.path.exists(gt):
                pairs.append((os.path.join(step_dir, f), gt, idx))
    return pairs


def _load_pair_metric_data(
    pair: tuple[str, str, int],
    records_by_index: dict[int, dict],
    sam2_mask_root: str | Path | None,
) -> PairMetricData:
    gen_path, gt_path, idx = pair
    gen_frames = read_video_frames(gen_path)
    gt_frames = read_video_frames(gt_path)
    if len(gen_frames) != len(gt_frames):
        raise ValueError(
            f"Frame count mismatch: {gen_path} has {len(gen_frames)} frames, "
            f"{gt_path} has {len(gt_frames)} frames"
        )
    masks = None
    if sam2_mask_root is not None:
        if idx not in records_by_index:
            raise ValueError(f"No selected record for sample index {idx}")
        masks = load_clip_mask_stack(
            records_by_index[idx],
            sam2_mask_root,
            len(gen_frames),
            gen_frames.shape[1:3],
        )
    metrics = compute_pairwise_metrics(
        gen_frames,
        gt_frames,
        lpips_model=None,
        device=torch.device("cpu"),
        masks=masks,
    )
    metrics["sample"] = idx
    return PairMetricData(
        idx=idx,
        gen_frames=gen_frames,
        gt_frames=gt_frames,
        masks=masks,
        metrics=metrics,
    )


def process_step(
    step_dir: str,
    lpips_model: LPIPS | None,
    inception: InceptionFeatureExtractor | None,
    video_extractor: VideoFeatureExtractor | None,
    device: torch.device,
    selected_records: list[dict] | None = None,
    sam2_mask_root: str | Path | None = None,
    metric_workers: int = DEFAULT_METRIC_WORKERS,
    lpips_batch_size: int = DEFAULT_LPIPS_BATCH_SIZE,
    feature_batch_size: int = DEFAULT_FEATURE_BATCH_SIZE,
    fvd_batch_size: int = DEFAULT_FVD_BATCH_SIZE,
    progress_callback: ProgressCallback | None = None,
) -> dict:
    """Compute all metrics for one eval step directory."""
    pairs = find_pairs(step_dir)
    if not pairs:
        return {}
    if metric_workers <= 0:
        raise ValueError(f"metric_workers must be positive, got {metric_workers}")
    if selected_records is not None and len(selected_records) != len(pairs):
        raise ValueError(
            f"Selected record count {len(selected_records)} does not match "
            f"video pair count {len(pairs)} in {step_dir}"
        )
    if (selected_records is None) != (sam2_mask_root is None):
        raise ValueError("selected_records and sam2_mask_root must be provided together")
    records_by_index = (
        {idx: record for idx, record in enumerate(selected_records)}
        if selected_records is not None else {}
    )

    metric_values: dict[str, list[float]] = {
        "mse": [],
        "psnr": [],
        "ssim": [],
        "lpips": [],
        "foreground_mse": [],
        "foreground_psnr": [],
        "foreground_ssim": [],
        "background_mse": [],
        "background_psnr": [],
        "background_ssim": [],
    }
    gen_videos, gt_videos = [], []
    region_gen_videos = {name: [] for name in REGION_NAMES}
    region_gt_videos = {name: [] for name in REGION_NAMES}
    local_fid_masks = []
    pair_data: list[PairMetricData] = []

    with ThreadPoolExecutor(max_workers=metric_workers) as executor:
        futures = [
            executor.submit(_load_pair_metric_data, pair, records_by_index, sam2_mask_root)
            for pair in pairs
        ]
        for done, future in enumerate(as_completed(futures), 1):
            pair_data.append(future.result())
            if progress_callback is not None:
                progress_callback("pairwise", done, len(pairs))

    pair_data.sort(key=lambda item: item.idx)
    per_sample = [item.metrics for item in pair_data]

    if lpips_model is not None:
        lpips_scores = compute_lpips_per_video(
            [(item.gen_frames, item.gt_frames) for item in pair_data],
            lpips_model,
            device,
            batch_size=lpips_batch_size,
            progress_callback=progress_callback,
        )
        for item, score in zip(pair_data, lpips_scores):
            item.metrics["lpips"] = score

    for item in pair_data:
        gen_frames = item.gen_frames
        gt_frames = item.gt_frames
        masks = item.masks
        gen_videos.append(gen_frames)
        gt_videos.append(gt_frames)

        if masks is not None:
            split_gen = split_video_by_mask(gen_frames, masks)
            split_gt = split_video_by_mask(gt_frames, masks)
            for name in REGION_NAMES:
                region_gen_videos[name].append(split_gen[name])
                region_gt_videos[name].append(split_gt[name])
            local_fid_masks.append(masks)

        m = item.metrics
        for key in metric_values:
            if key in m:
                metric_values[key].append(m[key])

    result = {
        "n_samples": len(pairs),
        "per_sample": per_sample,
    }
    for key, values in metric_values.items():
        if values:
            result[key] = float(np.mean(values))

    if inception is not None:
        total_gen_frames = sum(len(v) for v in gen_videos)
        if total_gen_frames >= 2:
            feats_gen = collect_inception_features(
                gen_videos,
                inception,
                device,
                batch_size=feature_batch_size,
                progress_callback=progress_callback,
                progress_phase="fid/gen",
            )
            feats_gt = collect_inception_features(
                gt_videos,
                inception,
                device,
                batch_size=feature_batch_size,
                progress_callback=progress_callback,
                progress_phase="fid/gt",
            )
            result["fid"] = compute_fid(feats_gen, feats_gt)
        if selected_records is not None:
            total_local_frames = sum(len(v) for v in gen_videos)
            if total_local_frames >= 2:
                feats_gen = collect_local_inception_features(
                    gen_videos,
                    local_fid_masks,
                    inception,
                    device,
                    batch_size=feature_batch_size,
                    progress_callback=progress_callback,
                    progress_phase="local_fid/gen",
                )
                feats_gt = collect_local_inception_features(
                    gt_videos,
                    local_fid_masks,
                    inception,
                    device,
                    batch_size=feature_batch_size,
                    progress_callback=progress_callback,
                    progress_phase="local_fid/gt",
                )
                result["foreground_local_fid"] = compute_fid(feats_gen, feats_gt)

    if video_extractor is not None:
        fvd = compute_fvd(
            gen_videos,
            gt_videos,
            video_extractor,
            device,
            batch_size=fvd_batch_size,
            progress_callback=progress_callback,
            progress_prefix="fvd",
        )
        if fvd is not None:
            result["fvd"] = fvd
        if selected_records is not None:
            for name in REGION_NAMES:
                fvd = compute_fvd(
                    region_gen_videos[name],
                    region_gt_videos[name],
                    video_extractor,
                    device,
                    batch_size=fvd_batch_size,
                    progress_callback=progress_callback,
                    progress_prefix=f"{name}_black_fvd",
                )
                if fvd is not None:
                    result[f"{name}_black_fvd"] = fvd

    return result


def make_print_progress(prefix: str) -> ProgressCallback:
    last: dict[str, tuple[int, int]] = {}

    def _progress(phase: str, done: int, total: int) -> None:
        state = (done, total)
        if last.get(phase) == state:
            return
        last[phase] = state
        print(f"  {prefix} {phase}: {done}/{total}", flush=True)

    return _progress


def main():
    parser = argparse.ArgumentParser(description="Compute reconstruction metrics on eval videos")
    parser.add_argument("--run", required=True, help="Training run timestamp (e.g. 2026-04-18_163933)")
    parser.add_argument("--split", default="all", choices=["in_task", "ood", "all"],
                        help="Eval split to evaluate (default: all)")
    parser.add_argument("--steps", nargs="*", help="Specific steps to evaluate (e.g. 0200 0400)")
    parser.add_argument("--device", default="cuda:2", help="Torch device")
    parser.add_argument("--no-lpips", action="store_true", help="Skip LPIPS (saves VRAM)")
    parser.add_argument("--no-fid", action="store_true", help="Skip FID/FVD (saves VRAM)")
    parser.add_argument("--csv", default=None, help="Output CSV path (default: <run>/eval_metrics.csv)")
    parser.add_argument("--metric-workers", type=int, default=DEFAULT_METRIC_WORKERS,
                        help="parallel workers for video decode and CPU pairwise metrics")
    parser.add_argument("--lpips-batch-size", type=int, default=DEFAULT_LPIPS_BATCH_SIZE,
                        help="GPU batch size for LPIPS frame batches")
    parser.add_argument("--feature-batch-size", type=int, default=DEFAULT_FEATURE_BATCH_SIZE,
                        help="GPU batch size for Inception/FID frame features")
    parser.add_argument("--fvd-batch-size", type=int, default=DEFAULT_FVD_BATCH_SIZE,
                        help="GPU batch size for S3D/FVD video features")
    parser.add_argument("--no-progress", action="store_true",
                        help="disable metric progress printing")
    args = parser.parse_args()

    run_dir = TRAINING_LOG_ROOT / args.run
    eval_dir = run_dir / "eval"
    if not eval_dir.exists():
        # flat layout fallback: eval/step-NNNN/ without split subdirs
        if not run_dir.exists():
            print(f"ERROR: {run_dir} not found")
            sys.exit(1)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(f"Loading models on {device} ...")
    lpips_model = None
    if not args.no_lpips:
        lpips_model = LPIPS().to(device)
        print("  LPIPS (VGG16) loaded")

    inception = None
    video_extractor = None
    if not args.no_fid:
        inception = InceptionFeatureExtractor().to(device)
        print("  InceptionV3 loaded")
        video_extractor = VideoFeatureExtractor().to(device)
        print("  S3D video feature extractor loaded")

    # Detect layout: split-based (in_task/ood) or flat (step-NNNN directly)
    splits = []
    if eval_dir.exists():
        subdirs = [d.name for d in eval_dir.iterdir() if d.is_dir()]
        if any(s in subdirs for s in ("in_task", "ood")):
            if args.split == "all":
                splits = [s for s in ("in_task", "ood") if s in subdirs]
            else:
                splits = [args.split] if args.split in subdirs else []
        else:
            splits = [""]  # flat layout, no split subdirectory

    all_results = []

    for split in splits:
        split_dir = eval_dir / split if split else eval_dir
        if not split_dir.exists():
            print(f"  Skip {split} (not found)")
            continue

        step_dirs = sorted(d for d in split_dir.iterdir() if d.is_dir() and d.name.startswith("step-"))
        if args.steps:
            targets = {f"step-{s}" for s in args.steps}
            step_dirs = [d for d in step_dirs if d.name in targets]

        for step_path in step_dirs:
            step_name = step_path.name
            label = f"{split}/{step_name}" if split else step_name
            print(f"\n[{label}]")

            result = process_step(
                str(step_path),
                lpips_model,
                inception,
                video_extractor,
                device,
                metric_workers=args.metric_workers,
                lpips_batch_size=args.lpips_batch_size,
                feature_batch_size=args.feature_batch_size,
                fvd_batch_size=args.fvd_batch_size,
                progress_callback=None if args.no_progress else make_print_progress("metrics"),
            )
            if not result:
                print("  No gen/gt pairs found")
                continue

            result["split"] = split
            result["step"] = step_name
            all_results.append(result)

            line = f"  MSE={result['mse']:.2f}  PSNR={result['psnr']:.2f}  SSIM={result['ssim']:.4f}"
            if "lpips" in result:
                line += f"  LPIPS={result['lpips']:.4f}"
            if "fid" in result:
                line += f"  FID={result['fid']:.1f}"
            if "fvd" in result:
                line += f"  FVD={result['fvd']:.1f}"
            line += f"  (n={result['n_samples']})"
            print(line)

            for s in result.get("per_sample", []):
                det = (
                    f"    sample {s['sample']:02d}: "
                    f"MSE={s['mse']:.2f} PSNR={s['psnr']:.2f} SSIM={s['ssim']:.4f}"
                )
                if "lpips" in s:
                    det += f" LPIPS={s['lpips']:.4f}"
                print(det)

    if not all_results:
        print("\nNo results to write.")
        return

    csv_path = args.csv or str(run_dir / "eval_metrics.csv")
    headers = [
        "split", "step", "n_samples",
        "mse", "psnr", "ssim", "lpips", "fid", "fvd",
        "foreground_mse", "foreground_psnr", "foreground_ssim",
        "background_mse", "background_psnr", "background_ssim",
        "foreground_local_fid",
        "foreground_black_fvd", "background_black_fvd",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        for r in all_results:
            row = {h: r.get(h, "") for h in headers}
            for k in [
                "mse", "psnr", "ssim", "lpips", "fid", "fvd",
                "foreground_mse", "foreground_psnr", "foreground_ssim",
                "background_mse", "background_psnr", "background_ssim",
                "foreground_local_fid",
                "foreground_black_fvd", "background_black_fvd",
            ]:
                if k in r:
                    row[k] = (
                        f"{r[k]:.4f}"
                        if k.endswith("ssim") or k == "lpips"
                        else f"{r[k]:.2f}"
                    )
            writer.writerow(row)
    print(f"\nCSV saved: {csv_path}")

    print("\n=== Summary ===")
    for split in splits:
        rows = [r for r in all_results if r["split"] == split]
        if not rows:
            continue
        title = split if split else "(flat)"
        print(f"\n  {title}:")
        print(f"  {'step':<12} {'MSE':>8} {'PSNR':>7} {'SSIM':>7} {'LPIPS':>7} {'FID':>8} {'FVD':>8}")
        print(f"  {'-' * 12} {'-' * 8} {'-' * 7} {'-' * 7} {'-' * 7} {'-' * 8} {'-' * 8}")
        for r in rows:
            line = f"  {r['step']:<12} {r['mse']:>8.2f} {r['psnr']:>7.2f} {r['ssim']:>7.4f}"
            line += f" {r['lpips']:>7.4f}" if "lpips" in r else f" {'':>7}"
            line += f" {r['fid']:>8.1f}" if "fid" in r else f" {'':>8}"
            line += f" {r['fvd']:>8.1f}" if "fvd" in r else f" {'':>8}"
            print(line)


if __name__ == "__main__":
    main()
