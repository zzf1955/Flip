"""Compute reconstruction metrics between generated and ground-truth eval videos.

Scans a training run's eval/ directory, pairs gen_XX.mp4 with gt_XX.mp4,
and computes per-frame and per-video metrics.

Metrics:
  - PSNR   (pixel-level, higher is better)
  - SSIM   (structural similarity, higher is better)
  - LPIPS  (perceptual distance via VGG, lower is better)
  - FID    (Frechet Inception Distance across all frames, lower is better)
  - FVD    (Frechet Video Distance via Inception features, lower is better)

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
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

MAIN_ROOT = Path(os.environ.get("FLIP_MAIN_ROOT", "/disk_n/zzf/flip"))
TRAINING_LOG_ROOT = MAIN_ROOT / "training_data" / "log"
FFMPEG = os.environ.get(
    "FFMPEG_BIN",
    "/home/leadtek/miniconda3/envs/flip/bin/ffmpeg",
)
FFPROBE = FFMPEG.replace("ffmpeg", "ffprobe")


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


# ── InceptionV3 feature extractor (for FID / FVD) ────────────────────


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
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean))


# ── Per-video pair metrics ─��──────────────────────────────────────────


def compute_pairwise_metrics(
    gen_frames: np.ndarray,
    gt_frames: np.ndarray,
    lpips_model: LPIPS | None,
    device: torch.device,
) -> dict:
    """Compute PSNR, SSIM, LPIPS between paired frame arrays (T,H,W,3)."""
    T = min(len(gen_frames), len(gt_frames))
    psnrs, ssims, lpipss = [], [], []

    for t in range(T):
        psnrs.append(peak_signal_noise_ratio(gt_frames[t], gen_frames[t], data_range=255))
        ssims.append(structural_similarity(gt_frames[t], gen_frames[t], channel_axis=2, data_range=255))

    if lpips_model is not None:
        gen_t = torch.from_numpy(gen_frames[:T]).permute(0, 3, 1, 2).float() / 255.0
        gt_t = torch.from_numpy(gt_frames[:T]).permute(0, 3, 1, 2).float() / 255.0
        for i in range(0, T, 8):
            batch_gen = gen_t[i:i + 8].to(device)
            batch_gt = gt_t[i:i + 8].to(device)
            d = lpips_model(batch_gen, batch_gt)
            lpipss.extend(d.cpu().tolist())

    result = {
        "psnr": float(np.mean(psnrs)),
        "ssim": float(np.mean(ssims)),
    }
    if lpipss:
        result["lpips"] = float(np.mean(lpipss))
    return result


# ── Inception features ���───────────────────────────────────────────────


def collect_inception_features(
    video_arrays: list[np.ndarray],
    extractor: InceptionFeatureExtractor,
    device: torch.device,
    batch_size: int = 16,
) -> np.ndarray:
    """Extract Inception features from all frames -> (N_total_frames, 2048)."""
    all_frames = np.concatenate(video_arrays, axis=0)
    feats = []
    for i in range(0, len(all_frames), batch_size):
        chunk = all_frames[i:i + batch_size]
        batch = torch.from_numpy(chunk).permute(0, 3, 1, 2).float() / 255.0
        feats.append(extractor(batch.to(device)).cpu().numpy())
    return np.concatenate(feats, axis=0)


def compute_fid(feats_gen: np.ndarray, feats_gt: np.ndarray) -> float:
    mu_gen, sigma_gen = feats_gen.mean(0), np.cov(feats_gen, rowvar=False)
    mu_gt, sigma_gt = feats_gt.mean(0), np.cov(feats_gt, rowvar=False)
    return frechet_distance(mu_gen, sigma_gen, mu_gt, sigma_gt)


# ── FVD ──────────────────────────────────────��────────────────────────


def video_inception_features(
    video_arrays: list[np.ndarray],
    extractor: InceptionFeatureExtractor,
    device: torch.device,
) -> np.ndarray:
    """Per-video feature: mean-pool Inception features across frames -> (V, 2048)."""
    feats = []
    for vid in video_arrays:
        t = torch.from_numpy(vid).permute(0, 3, 1, 2).float() / 255.0
        f = extractor(t.to(device))
        feats.append(f.mean(0).cpu().numpy())
    return np.stack(feats)


def compute_fvd(
    gen_videos: list[np.ndarray],
    gt_videos: list[np.ndarray],
    extractor: InceptionFeatureExtractor,
    device: torch.device,
) -> float | None:
    """Compute FVD. Returns None if too few videos for stable covariance."""
    if len(gen_videos) < 3:
        return None
    feats_gen = video_inception_features(gen_videos, extractor, device)
    feats_gt = video_inception_features(gt_videos, extractor, device)
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
            idx = int(f[4:6])
            gt = os.path.join(step_dir, f"gt_{idx:02d}.mp4")
            if os.path.exists(gt):
                pairs.append((os.path.join(step_dir, f), gt, idx))
    return pairs


def process_step(
    step_dir: str,
    lpips_model: LPIPS | None,
    inception: InceptionFeatureExtractor | None,
    device: torch.device,
) -> dict:
    """Compute all metrics for one eval step directory."""
    pairs = find_pairs(step_dir)
    if not pairs:
        return {}

    all_psnr, all_ssim, all_lpips = [], [], []
    gen_videos, gt_videos = [], []
    per_sample = []

    for gen_path, gt_path, idx in pairs:
        gen_frames = read_video_frames(gen_path)
        gt_frames = read_video_frames(gt_path)
        gen_videos.append(gen_frames)
        gt_videos.append(gt_frames)

        m = compute_pairwise_metrics(gen_frames, gt_frames, lpips_model, device)
        m["sample"] = idx
        per_sample.append(m)
        all_psnr.append(m["psnr"])
        all_ssim.append(m["ssim"])
        if "lpips" in m:
            all_lpips.append(m["lpips"])

    result = {
        "psnr": float(np.mean(all_psnr)),
        "ssim": float(np.mean(all_ssim)),
        "n_samples": len(pairs),
        "per_sample": per_sample,
    }
    if all_lpips:
        result["lpips"] = float(np.mean(all_lpips))

    if inception is not None:
        total_gen_frames = sum(len(v) for v in gen_videos)
        if total_gen_frames >= 2:
            feats_gen = collect_inception_features(gen_videos, inception, device)
            feats_gt = collect_inception_features(gt_videos, inception, device)
            result["fid"] = compute_fid(feats_gen, feats_gt)

        fvd = compute_fvd(gen_videos, gt_videos, inception, device)
        if fvd is not None:
            result["fvd"] = fvd

    return result


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
    if not args.no_fid:
        inception = InceptionFeatureExtractor().to(device)
        print("  InceptionV3 loaded")

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

            result = process_step(str(step_path), lpips_model, inception, device)
            if not result:
                print("  No gen/gt pairs found")
                continue

            result["split"] = split
            result["step"] = step_name
            all_results.append(result)

            line = f"  PSNR={result['psnr']:.2f}  SSIM={result['ssim']:.4f}"
            if "lpips" in result:
                line += f"  LPIPS={result['lpips']:.4f}"
            if "fid" in result:
                line += f"  FID={result['fid']:.1f}"
            if "fvd" in result:
                line += f"  FVD={result['fvd']:.1f}"
            line += f"  (n={result['n_samples']})"
            print(line)

            for s in result.get("per_sample", []):
                det = f"    sample {s['sample']:02d}: PSNR={s['psnr']:.2f} SSIM={s['ssim']:.4f}"
                if "lpips" in s:
                    det += f" LPIPS={s['lpips']:.4f}"
                print(det)

    if not all_results:
        print("\nNo results to write.")
        return

    csv_path = args.csv or str(run_dir / "eval_metrics.csv")
    headers = ["split", "step", "n_samples", "psnr", "ssim", "lpips", "fid", "fvd"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        for r in all_results:
            row = {h: r.get(h, "") for h in headers}
            for k in ["psnr", "ssim", "lpips", "fid", "fvd"]:
                if k in r:
                    row[k] = f"{r[k]:.4f}" if k in ("ssim", "lpips") else f"{r[k]:.2f}"
            writer.writerow(row)
    print(f"\nCSV saved: {csv_path}")

    print("\n=== Summary ===")
    for split in splits:
        rows = [r for r in all_results if r["split"] == split]
        if not rows:
            continue
        title = split if split else "(flat)"
        print(f"\n  {title}:")
        print(f"  {'step':<12} {'PSNR':>7} {'SSIM':>7} {'LPIPS':>7} {'FID':>8} {'FVD':>8}")
        print(f"  {'─' * 12} {'─' * 7} {'─' * 7} {'─' * 7} {'─' * 8} {'─' * 8}")
        for r in rows:
            line = f"  {r['step']:<12} {r['psnr']:>7.2f} {r['ssim']:>7.4f}"
            line += f" {r['lpips']:>7.4f}" if "lpips" in r else f" {'':>7}"
            line += f" {r['fid']:>8.1f}" if "fid" in r else f" {'':>8}"
            line += f" {r['fvd']:>8.1f}" if "fvd" in r else f" {'':>8}"
            print(line)


if __name__ == "__main__":
    main()
