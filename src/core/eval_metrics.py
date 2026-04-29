"""Online eval metrics for training: PSNR / SSIM / LPIPS / CLIP / FID / FVD.

Designed to run on rank 0 after eval video generation. Models are lazily
loaded on first use and kept on device for subsequent calls.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

FFMPEG = os.environ.get(
    "FFMPEG_BIN",
    "/home/leadtek/miniconda3/envs/flip/bin/ffmpeg",
)
FFPROBE = FFMPEG.replace("ffmpeg", "ffprobe")


# ── Video IO ──────────────────────────────────────────────────────────


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
    assert proc.returncode == 0, (
        f"ffmpeg failed on {path}: {proc.stderr.decode()}")
    raw = np.frombuffer(proc.stdout, dtype=np.uint8).copy()
    return raw.reshape(-1, h, w, 3)


# ── LPIPS (VGG16) ────────────────────────────────────────────────────


class _LPIPS(nn.Module):
    """VGG16-based perceptual distance (simplified LPIPS "vgg" variant)."""

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


# ── CLIP Score ────────────────────────────────────────────────────────


class _CLIPScorer(nn.Module):
    """Cosine similarity between CLIP image embeddings of gen vs GT."""

    CLIP_MODEL_ID = "openai/clip-vit-large-patch14"

    def __init__(self):
        super().__init__()
        from transformers import CLIPModel, CLIPImageProcessor
        self.model = CLIPModel.from_pretrained(self.CLIP_MODEL_ID).vision_model
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.processor = CLIPImageProcessor.from_pretrained(self.CLIP_MODEL_ID)

    @torch.no_grad()
    def forward(
        self, x: torch.Tensor, y: torch.Tensor,
    ) -> torch.Tensor:
        """x, y: (B, 3, H, W) in [0, 1]. Returns (B,) cosine similarities."""
        x_emb = self._embed(x)
        y_emb = self._embed(y)
        return F.cosine_similarity(x_emb, y_emb, dim=-1)

    def _embed(self, imgs: torch.Tensor) -> torch.Tensor:
        """(B, 3, H, W) [0,1] -> (B, D) normalized embeddings."""
        mean = torch.tensor(
            self.processor.image_mean, device=imgs.device,
        ).view(1, 3, 1, 1)
        std = torch.tensor(
            self.processor.image_std, device=imgs.device,
        ).view(1, 3, 1, 1)
        size = self.processor.size["shortest_edge"]
        imgs = F.interpolate(imgs, size=(size, size),
                             mode="bicubic", align_corners=False)
        imgs = (imgs - mean) / std
        out = self.model(pixel_values=imgs)
        emb = out.pooler_output
        return F.normalize(emb, dim=-1)


# ── Frechet metric feature extractors ─────────────────────────────────


class _InceptionFeatureExtractor(nn.Module):
    """Extract pool3 (2048-d) features from InceptionV3."""

    def __init__(self):
        super().__init__()
        from torchvision.models import Inception_V3_Weights, inception_v3
        net = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1).eval()
        self.blocks = nn.Sequential(
            net.Conv2d_1a_3x3, net.Conv2d_2a_3x3, net.Conv2d_2b_3x3,
            nn.MaxPool2d(3, stride=2),
            net.Conv2d_3b_1x1, net.Conv2d_4a_3x3,
            nn.MaxPool2d(3, stride=2),
            net.Mixed_5b, net.Mixed_5c, net.Mixed_5d,
            net.Mixed_6a, net.Mixed_6b, net.Mixed_6c, net.Mixed_6d,
            net.Mixed_6e, net.Mixed_7a, net.Mixed_7b, net.Mixed_7c,
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
        x = F.interpolate(x, size=(299, 299), mode="bilinear",
                          align_corners=False)
        x = (x - self.mean) / self.std
        return self.blocks(x).flatten(1)


def _frechet_distance(mu1, sigma1, mu2, sigma2) -> float:
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


def _compute_frechet_metric(gen_feats: np.ndarray,
                            gt_feats: np.ndarray) -> float | None:
    """Compute Frechet metric; return None for too few feature vectors."""
    if len(gen_feats) < 2 or len(gt_feats) < 2:
        return None
    mu_gen, sigma_gen = gen_feats.mean(0), np.cov(gen_feats, rowvar=False)
    mu_gt, sigma_gt = gt_feats.mean(0), np.cov(gt_feats, rowvar=False)
    if sigma_gen.ndim < 2 or sigma_gt.ndim < 2:
        return None
    return _frechet_distance(mu_gen, sigma_gen, mu_gt, sigma_gt)


class _VideoFeatureExtractor(nn.Module):
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


# ── Online Metrics ────────────────────────────────────────────────────


class OnlineMetrics:
    """Lazily-loaded metric models for training-time eval.

    Usage::

        om = OnlineMetrics("cuda:0")
        metrics = om.compute_step("/path/to/eval/in_task/step-0200")
        # {"psnr": 18.5, "ssim": 0.72, "lpips": 0.004, "clip_score": 0.92}
    """

    def __init__(self, device: str | torch.device, *, include_frechet: bool = False):
        self.device = torch.device(device)
        self.include_frechet = include_frechet
        self._lpips: _LPIPS | None = None
        self._clip: _CLIPScorer | None = None
        self._inception: _InceptionFeatureExtractor | None = None
        self._video: _VideoFeatureExtractor | None = None

    def _ensure_models(self):
        if self._lpips is None:
            self._lpips = _LPIPS().to(self.device)
        if self._clip is None:
            self._clip = _CLIPScorer().to(self.device)
        if self.include_frechet and self._inception is None:
            self._inception = _InceptionFeatureExtractor().to(self.device)
        if self.include_frechet and self._video is None:
            self._video = _VideoFeatureExtractor().to(self.device)

    def compute_step(self, step_dir: str) -> dict[str, float]:
        """Compute metrics for all gen/gt pairs in a step directory.

        Returns dict with keys: psnr, ssim, lpips, clip_score.
        If include_frechet is enabled, may also include fid and fvd.
        Returns empty dict if no pairs found.
        """
        pairs = self._find_pairs(step_dir)
        if not pairs:
            return {}

        self._ensure_models()

        all_psnr, all_ssim, all_lpips, all_clip = [], [], [], []
        frame_feats_gen, frame_feats_gt = [], []
        video_feats_gen, video_feats_gt = [], []

        for gen_path, gt_path in pairs:
            gen_frames = read_video_frames(gen_path)
            gt_frames = read_video_frames(gt_path)
            if len(gen_frames) != len(gt_frames):
                raise ValueError(
                    f"Frame count mismatch: {gen_path} has {len(gen_frames)} frames, "
                    f"{gt_path} has {len(gt_frames)} frames")
            T = len(gen_frames)

            for t in range(T):
                all_psnr.append(peak_signal_noise_ratio(
                    gt_frames[t], gen_frames[t], data_range=255))
                all_ssim.append(structural_similarity(
                    gt_frames[t], gen_frames[t],
                    channel_axis=2, data_range=255))

            gen_t = (torch.from_numpy(gen_frames[:T])
                     .permute(0, 3, 1, 2).float() / 255.0)
            gt_t = (torch.from_numpy(gt_frames[:T])
                    .permute(0, 3, 1, 2).float() / 255.0)

            for i in range(T):
                b_gen = gen_t[i:i + 1].to(self.device)
                b_gt = gt_t[i:i + 1].to(self.device)
                all_lpips.extend(
                    self._lpips(b_gen, b_gt).cpu().tolist())
                all_clip.extend(
                    self._clip(b_gen, b_gt).cpu().tolist())

                if self._inception is not None:
                    gen_feats = self._inception(b_gen).cpu().numpy()
                    gt_feats = self._inception(b_gt).cpu().numpy()
                    frame_feats_gen.append(gen_feats)
                    frame_feats_gt.append(gt_feats)
            if self._video is not None:
                video_gen = gen_t.permute(1, 0, 2, 3).unsqueeze(0).to(self.device)
                video_gt = gt_t.permute(1, 0, 2, 3).unsqueeze(0).to(self.device)
                video_feats_gen.append(self._video(video_gen).cpu().numpy())
                video_feats_gt.append(self._video(video_gt).cpu().numpy())

        result = {
            "psnr": float(np.mean(all_psnr)),
            "ssim": float(np.mean(all_ssim)),
            "lpips": float(np.mean(all_lpips)),
            "clip_score": float(np.mean(all_clip)),
        }
        if self._inception is not None and frame_feats_gen:
            fid = _compute_frechet_metric(
                np.concatenate(frame_feats_gen, axis=0),
                np.concatenate(frame_feats_gt, axis=0),
            )
            fvd = _compute_frechet_metric(
                np.concatenate(video_feats_gen, axis=0),
                np.concatenate(video_feats_gt, axis=0),
            )
            if fid is not None:
                result["fid"] = fid
            if fvd is not None:
                result["fvd"] = fvd
        return result

    @staticmethod
    def _find_pairs(step_dir: str) -> list[tuple[str, str]]:
        """Find (gen_path, gt_path) pairs in a step directory."""
        pairs = []
        d = Path(step_dir)
        if not d.is_dir():
            return pairs
        for f in sorted(d.iterdir()):
            if f.name.startswith("gen_") and f.suffix == ".mp4":
                idx = f.name[4:6]
                gt = d / f"gt_{idx}.mp4"
                assert gt.exists(), f"Missing GT for {f}: expected {gt}"
                pairs.append((str(f), str(gt)))
        return pairs
