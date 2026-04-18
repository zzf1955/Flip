"""
Camera calibration v3: Gradient-based optimization (Adam) using SAM2 masks.

Reimplements the fisheye projection in PyTorch for differentiable optimization.
Uses vertex-sampling + distance-field loss instead of rasterized mask comparison.

Records per-step: loss, parameter values, gradient values/directions.
Generates analysis plots: loss curve, parameter trajectory, gradient analysis.

Usage:
  python scripts/auto_calibrate_grad.py
  python scripts/auto_calibrate_grad.py --steps 3000 --device cuda:1
  python scripts/auto_calibrate_grad.py --manifest data/calib_frames.json --steps 2000
"""

import sys
import os
import json
import time
import math
import argparse
import numpy as np
import cv2
import pinocchio as pin
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.stdout.reconfigure(line_buffering=True)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "scripts"))

from config import (G1_URDF, MESH_DIR, BEST_PARAMS, SKIP_MESHES,
                     DATASET_ROOT, OUTPUT_DIR, get_hand_type, CAMERA_MODEL)
from camera_models import get_model, build_K, build_D, model_is_fisheye, project_points_cv
from video_inpaint import (build_q, do_fk, parse_urdf_meshes, preload_meshes,
                            load_episode_info)
from auto_calibrate_v2 import (load_calib_frames, project_and_mask, compute_iou,
                                save_comparison, BOUNDS, PARAM_NAMES)

GRAD_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "calib_grad")

# Bounds as torch tensor (for clamping)
BOUNDS_T = torch.tensor(BOUNDS, dtype=torch.float64)


# ============================================================
# CameraParams: wraps N parameters as nn.Parameter (model-aware)
# ============================================================

class CameraParams(nn.Module):
    """Differentiable camera parameters, adapts to active camera model."""

    def __init__(self, init_dict, device='cpu', model_name=None):
        super().__init__()
        if model_name is None:
            model_name = CAMERA_MODEL
        self._model_cfg = get_model(model_name)
        self._param_names = self._model_cfg["param_names"]
        self._bounds = self._model_cfg["bounds"]
        for name in self._param_names:
            val = torch.tensor(float(init_dict[name]), dtype=torch.float64, device=device)
            setattr(self, name, nn.Parameter(val))

    @property
    def param_names(self):
        return self._param_names

    @property
    def model_cfg(self):
        return self._model_cfg

    def get_param_groups(self, base_lr=1.0):
        """Per-group learning rates from model config."""
        groups = []
        for gname, gcfg in self._model_cfg["lr_groups"].items():
            params = [getattr(self, n) for n in gcfg["params"]]
            groups.append({"params": params, "lr": base_lr * gcfg["lr_scale"], "name": gname})
        return groups

    def to_array(self):
        return np.array([getattr(self, n).detach().cpu().item() for n in self._param_names])

    def to_dict(self):
        return {n: getattr(self, n).detach().cpu().item() for n in self._param_names}

    def clamp_to_bounds(self):
        with torch.no_grad():
            for i, name in enumerate(self._param_names):
                p = getattr(self, name)
                lo, hi = self._bounds[i]
                p.clamp_(lo, hi)

    def override_bounds(self, param_name, lo, hi):
        """Override bounds for a specific parameter."""
        idx = self._param_names.index(param_name)
        self._bounds = list(self._bounds)
        self._bounds[idx] = [lo, hi]


# ============================================================
# Differentiable fisheye projection (PyTorch)
# ============================================================

def _build_rotation(pitch_deg, yaw_deg, roll_deg, device):
    """Euler angles (degrees, torch scalars) -> R_cam (3x3 tensor with grad)."""
    pitch = pitch_deg * (math.pi / 180.0)
    yaw = yaw_deg * (math.pi / 180.0)
    roll = roll_deg * (math.pi / 180.0)

    zero = torch.zeros(1, dtype=torch.float64, device=device).squeeze()
    one = torch.ones(1, dtype=torch.float64, device=device).squeeze()

    cp, sp = torch.cos(pitch), torch.sin(pitch)
    R_pitch = torch.stack([
        torch.stack([cp, zero, sp]),
        torch.stack([zero, one, zero]),
        torch.stack([-sp, zero, cp]),
    ])

    cy, sy = torch.cos(yaw), torch.sin(yaw)
    R_yaw = torch.stack([
        torch.stack([cy, -sy, zero]),
        torch.stack([sy, cy, zero]),
        torch.stack([zero, zero, one]),
    ])

    cr, sr = torch.cos(roll), torch.sin(roll)
    R_roll = torch.stack([
        torch.stack([one, zero, zero]),
        torch.stack([zero, cr, -sr]),
        torch.stack([zero, sr, cr]),
    ])

    R_body_to_cam = torch.tensor(
        [[0., -1., 0.], [0., 0., -1.], [1., 0., 0.]],
        dtype=torch.float64, device=device)

    return R_body_to_cam @ R_roll @ R_yaw @ R_pitch


def differentiable_project(verts_world, cam, ref_t, ref_R, device):
    """
    Differentiable projection: world 3D -> pixel 2D.
    Supports pinhole_f, pinhole, and fisheye models via cam.model_cfg.

    Returns:
        pts2d: (M, 2) pixel coordinates for visible vertices
        visible: (N,) bool mask
    """
    R_cam = _build_rotation(cam.pitch, cam.yaw, cam.roll, device)

    offset = torch.stack([cam.dx, cam.dy, cam.dz])
    cam_pos = ref_t + ref_R @ offset
    R_w2c = (ref_R @ R_cam.T).T
    t_w2c = R_w2c @ (-cam_pos)

    p_cam = (R_w2c @ verts_world.T).T + t_w2c
    z = p_cam[:, 2]
    visible = z > 0.01
    p_cam_vis = p_cam[visible]

    if p_cam_vis.shape[0] == 0:
        return torch.zeros(0, 2, dtype=torch.float64, device=device), visible

    x = p_cam_vis[:, 0]
    y = p_cam_vis[:, 1]
    z_vis = p_cam_vis[:, 2]

    if model_is_fisheye(cam.model_cfg):
        # Fisheye equidistant projection
        a = x / (z_vis + 1e-12)
        b = y / (z_vis + 1e-12)
        r = torch.sqrt(a * a + b * b + 1e-12)
        theta = torch.atan(r)
        theta2 = theta * theta
        theta_d = theta * (1.0 + cam.k1 * theta2 + cam.k2 * theta2**2
                           + cam.k3 * theta2**3 + cam.k4 * theta2**4)
        scale = theta_d / (r + 1e-12)
        u = cam.fx * scale * a + cam.cx
        v = cam.fy * scale * b + cam.cy
    elif hasattr(cam, 'f'):
        # Pinhole with unified focal length
        u = cam.f * (x / (z_vis + 1e-12)) + cam.cx
        v = cam.f * (y / (z_vis + 1e-12)) + cam.cy
    else:
        # Pinhole with separate fx, fy
        u = cam.fx * (x / (z_vis + 1e-12)) + cam.cx
        v = cam.fy * (y / (z_vis + 1e-12)) + cam.cy

    pts2d = torch.stack([u, v], dim=1)
    return pts2d, visible

# Backward-compatible alias
differentiable_fisheye_project = differentiable_project


def verify_projection(cam, frames_data, device):
    """Compare PyTorch projection vs OpenCV to verify correctness."""
    lv, ref_t_np, ref_R_np, gt_mask, h, w = frames_data[0]
    all_verts_np = np.concatenate(list(lv.values()), axis=0)

    verts_t = torch.tensor(all_verts_np, dtype=torch.float64, device=device)
    ref_t_t = torch.tensor(ref_t_np, dtype=torch.float64, device=device)
    ref_R_t = torch.tensor(ref_R_np, dtype=torch.float64, device=device)

    with torch.no_grad():
        pts2d_torch, visible = differentiable_fisheye_project(verts_t, cam, ref_t_t, ref_R_t, device)
    pts2d_torch_np = pts2d_torch.cpu().numpy()

    # OpenCV reference
    p = cam.to_dict()
    pitch_deg, yaw_deg, roll_deg = p["pitch"], p["yaw"], p["roll"]
    pitch, yaw, roll = np.radians(pitch_deg), np.radians(yaw_deg), np.radians(roll_deg)

    R_pitch = np.array([[np.cos(pitch),0,np.sin(pitch)],[0,1,0],[-np.sin(pitch),0,np.cos(pitch)]])
    R_yaw = np.array([[np.cos(yaw),-np.sin(yaw),0],[np.sin(yaw),np.cos(yaw),0],[0,0,1]])
    R_roll = np.array([[1,0,0],[0,np.cos(roll),-np.sin(roll)],[0,np.sin(roll),np.cos(roll)]])
    R_b2c = np.array([[0.,-1.,0.],[0.,0.,-1.],[1.,0.,0.]])
    R_cam_np = R_b2c @ R_roll @ R_yaw @ R_pitch

    cam_pos_np = ref_t_np + ref_R_np @ np.array([p["dx"], p["dy"], p["dz"]])
    R_w2c_np = (ref_R_np @ R_cam_np.T).T
    t_w2c_np = R_w2c_np @ (-cam_pos_np)

    mcfg = cam.model_cfg
    K = build_K(p, mcfg)
    D = build_D(p, mcfg)
    _fisheye = model_is_fisheye(mcfg)
    rvec, _ = cv2.Rodrigues(R_w2c_np)
    tvec = t_w2c_np.reshape(3,1)

    # Filter same visible set
    p_cam_np = (R_w2c_np @ all_verts_np.T).T + t_w2c_np
    vis_np = p_cam_np[:, 2] > 0.01
    verts_vis = all_verts_np[vis_np]

    pts2d_cv = project_points_cv(
        verts_vis.reshape(-1, 1, 3).astype(np.float64), rvec, tvec, K, D, _fisheye)
    pts2d_cv = pts2d_cv.reshape(-1, 2)

    # Compare
    vis_torch_np = visible.cpu().numpy()
    assert np.array_equal(vis_np, vis_torch_np), "Visibility mismatch!"

    diff = np.abs(pts2d_torch_np - pts2d_cv)
    finite = np.all(np.isfinite(diff), axis=1)
    if np.sum(finite) == 0:
        print("  WARNING: no finite projections to compare")
        return True

    max_err = diff[finite].max()
    mean_err = diff[finite].mean()
    print(f"  Projection verify: max_err={max_err:.6f}px, mean_err={mean_err:.6f}px "
          f"({np.sum(finite)}/{len(diff)} finite points)")

    if max_err > 0.01:
        print("  ERROR: projection mismatch exceeds 0.01px threshold!")
        return False
    return True


# ============================================================
# Loss functions
# ============================================================

def _bilinear_splat(pts2d, h, w, scale=8):
    """Differentiable bilinear splatting: project vertices onto a downsampled grid.

    Each projected vertex distributes its contribution to the 4 neighboring
    cells using bilinear weights. The result is a soft density map.

    Args:
        pts2d: (M, 2) pixel coordinates [u, v] with gradients
        h, w: original image dimensions
        scale: downsample factor (grid is h//scale x w//scale)

    Returns:
        density: (1, 1, gh, gw) float64 tensor — vertex count per cell
    """
    gh, gw = h // scale, w // scale
    device = pts2d.device

    # Scale coordinates to grid space
    gx = pts2d[:, 0] / scale  # (M,)
    gy = pts2d[:, 1] / scale  # (M,)

    # Floor coordinates (integer grid cell)
    gx0 = gx.floor().long()
    gy0 = gy.floor().long()

    # Bilinear weights (differentiable through gx, gy)
    fx = gx - gx0.double()  # fractional part x
    fy = gy - gy0.double()  # fractional part y

    # 4 corner weights
    w00 = (1 - fx) * (1 - fy)  # top-left
    w10 = fx * (1 - fy)        # top-right
    w01 = (1 - fx) * fy        # bottom-left
    w11 = fx * fy              # bottom-right

    # Accumulate into grid using scatter_add for each corner
    density = torch.zeros(gh * gw, dtype=torch.float64, device=device)

    for dx_off, dy_off, wt in [(0, 0, w00), (1, 0, w10), (0, 1, w01), (1, 1, w11)]:
        cx = gx0 + dx_off
        cy = gy0 + dy_off
        # Bounds check
        valid = (cx >= 0) & (cx < gw) & (cy >= 0) & (cy < gh)
        idx = cy[valid] * gw + cx[valid]
        density.scatter_add_(0, idx, wt[valid])

    return density.view(1, 1, gh, gw)


def compute_loss(pts2d, gt_mask_t, dt_field, h, w, alpha_sample, alpha_dt, splat_scale=8):
    """Soft Dice loss via bilinear splatting + SDF regularization.

    1. Splat projected vertices onto a downsampled grid -> soft predicted mask
    2. Compare with downsampled GT mask using Dice loss
    3. Add SDF term for long-range gradient signal
    """
    device = pts2d.device

    # Filter to in-frame vertices
    valid = (torch.isfinite(pts2d).all(dim=1) &
             (pts2d[:, 0] >= 0) & (pts2d[:, 0] < w) &
             (pts2d[:, 1] >= 0) & (pts2d[:, 1] < h))
    pts = pts2d[valid]

    if pts.shape[0] < 10:
        return torch.tensor(1.0, dtype=torch.float64, device=device)

    # --- Soft Dice via splatting ---
    density = _bilinear_splat(pts, h, w, scale=splat_scale)  # (1,1,gh,gw)

    # Soft mask: sigmoid of density (0 = no vertices, 1 = many vertices)
    # Scale so that ~3 vertices per cell gives ~0.95 probability
    pred_soft = torch.sigmoid((density - 0.5) * 4.0)  # (1,1,gh,gw)

    # Downsample GT mask to same grid
    gh, gw = h // splat_scale, w // splat_scale
    gt_down = F.adaptive_avg_pool2d(gt_mask_t.float(), (gh, gw)).double()  # (1,1,gh,gw)

    # Dice loss: 1 - 2*|A∩B| / (|A| + |B|)
    inter = (pred_soft * gt_down).sum()
    sum_ab = pred_soft.sum() + gt_down.sum()
    dice = 2.0 * inter / (sum_ab + 1e-6)
    loss_dice = 1.0 - dice

    # --- SDF regularization: push ALL projected vertices toward mask ---
    # (including those slightly outside frame, for long-range gradients)
    valid_wide = (torch.isfinite(pts2d).all(dim=1) &
                  (pts2d[:, 0] > -w * 0.3) & (pts2d[:, 0] < w * 1.3) &
                  (pts2d[:, 1] > -h * 0.3) & (pts2d[:, 1] < h * 1.3))
    pts_wide = pts2d[valid_wide]
    if pts_wide.shape[0] > 10:
        grid_x = (pts_wide[:, 0] / (w - 1)) * 2.0 - 1.0
        grid_y = (pts_wide[:, 1] / (h - 1)) * 2.0 - 1.0
        grid = torch.stack([grid_x, grid_y], dim=1).view(1, 1, -1, 2).float()
        sdf_vals = F.grid_sample(dt_field.float(), grid, mode='bilinear',
                                 padding_mode='border', align_corners=True)
        sdf_vals = sdf_vals.squeeze().double()
        # Only penalize vertices outside mask (SDF > 0)
        loss_sdf = F.relu(sdf_vals).mean()
    else:
        loss_sdf = torch.tensor(0.0, dtype=torch.float64, device=device)

    return alpha_sample * loss_dice + alpha_dt * loss_sdf


# ============================================================
# Optimization logger
# ============================================================

class OptimizationLogger:
    """Records per-step optimization state."""

    def __init__(self):
        self.steps = []

    def log_step(self, step, loss, params_dict, grads_dict, true_iou=None):
        entry = {
            "step": step,
            "loss": float(loss),
            "params": {k: float(v) for k, v in params_dict.items()},
            "gradients": {k: float(v) for k, v in grads_dict.items()},
        }
        if true_iou is not None:
            entry["true_iou"] = float(true_iou)
        self.steps.append(entry)

    def save_json(self, path):
        with open(path, 'w') as f:
            json.dump(self.steps, f, indent=2)

    def plot_all(self, output_dir):
        """Generate all analysis plots."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        steps = [s["step"] for s in self.steps]

        # 1. Loss curve + IoU
        fig, ax1 = plt.subplots(figsize=(12, 5))
        ax1.plot(steps, [s["loss"] for s in self.steps], 'b-', alpha=0.7, label='Loss')
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Loss', color='b')
        ax1.tick_params(axis='y', labelcolor='b')

        iou_data = [(s["step"], s["true_iou"]) for s in self.steps if "true_iou" in s]
        if iou_data:
            ax2 = ax1.twinx()
            ax2.plot([x[0] for x in iou_data], [x[1] for x in iou_data],
                     'r-o', markersize=3, label='True IoU (Dice)')
            ax2.set_ylabel('IoU (Dice)', color='r')
            ax2.tick_params(axis='y', labelcolor='r')

        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "loss_curve.png"), dpi=150)
        plt.close(fig)

        # 2. Parameter trajectory (14 subplots)
        fig, axes = plt.subplots(4, 4, figsize=(20, 16))
        axes_flat = axes.flatten()
        for i, name in enumerate(PARAM_NAMES):
            vals = [s["params"][name] for s in self.steps]
            axes_flat[i].plot(steps, vals, linewidth=0.8)
            axes_flat[i].set_title(name, fontsize=10)
            axes_flat[i].grid(True, alpha=0.3)
            # Mark bounds
            axes_flat[i].axhline(y=BOUNDS[i][0], color='r', linestyle=':', alpha=0.3)
            axes_flat[i].axhline(y=BOUNDS[i][1], color='r', linestyle=':', alpha=0.3)
        for i in range(len(PARAM_NAMES), len(axes_flat)):
            axes_flat[i].set_visible(False)
        fig.suptitle('Parameter Trajectory', fontsize=14)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "param_trajectory.png"), dpi=150)
        plt.close(fig)

        # 3. Gradient values over time (14 subplots)
        fig, axes = plt.subplots(4, 4, figsize=(20, 16))
        axes_flat = axes.flatten()
        for i, name in enumerate(PARAM_NAMES):
            grads = [s["gradients"][name] for s in self.steps]
            axes_flat[i].plot(steps, grads, linewidth=0.5, alpha=0.7)
            axes_flat[i].axhline(y=0, color='r', linestyle='--', alpha=0.3)
            axes_flat[i].set_title(f'{name} gradient', fontsize=10)
            axes_flat[i].grid(True, alpha=0.3)
        for i in range(len(PARAM_NAMES), len(axes_flat)):
            axes_flat[i].set_visible(False)
        fig.suptitle('Gradient Values Over Time', fontsize=14)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "gradient_analysis.png"), dpi=150)
        plt.close(fig)

        # 4. Gradient magnitude summary (bar chart)
        mean_abs_grads = {}
        for name in PARAM_NAMES:
            vals = [abs(s["gradients"][name]) for s in self.steps]
            mean_abs_grads[name] = np.mean(vals) if vals else 0.0
        fig, ax = plt.subplots(figsize=(12, 5))
        names = list(mean_abs_grads.keys())
        magnitudes = [mean_abs_grads[n] for n in names]
        bars = ax.bar(names, magnitudes)
        ax.set_ylabel('Mean |gradient|')
        ax.set_title('Average Gradient Magnitude per Parameter')
        ax.set_yscale('log')
        for bar, val in zip(bars, magnitudes):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f'{val:.2e}', ha='center', va='bottom', fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "gradient_summary.png"), dpi=150)
        plt.close(fig)

        print(f"  Plots saved to {output_dir}/")


# ============================================================
# Main optimization loop
# ============================================================

def run_optimization(frames_data, frames_vis, seed_params, args):
    device = torch.device(args.device)
    os.makedirs(GRAD_OUTPUT_DIR, exist_ok=True)

    # Initialize camera parameters
    cam = CameraParams(seed_params, device=device)

    # Verify projection correctness
    print("\nVerifying differentiable projection vs OpenCV...")
    if not verify_projection(cam, frames_data, device):
        print("Aborting: projection verification failed.")
        sys.exit(1)
    print("  Projection verified OK.\n")

    # Prepare per-frame data as torch tensors (on CPU to save GPU memory)
    print("Preparing frame tensors...")
    frame_tensors = []  # (verts, ref_t, ref_R, gt_mask_t, dt_field, h, w)
    from scipy.ndimage import distance_transform_edt
    for link_verts, ref_t, ref_R, gt_mask, h, w in frames_data:
        all_verts = np.concatenate(list(link_verts.values()), axis=0)
        verts_t = torch.tensor(all_verts, dtype=torch.float64)
        ref_t_t = torch.tensor(ref_t, dtype=torch.float64)
        ref_R_t = torch.tensor(ref_R, dtype=torch.float64)
        gt_t = torch.tensor(gt_mask.astype(np.float64), dtype=torch.float64).unsqueeze(0).unsqueeze(0)
        # Distance field
        dt_out = distance_transform_edt(~gt_mask).astype(np.float64)
        dt_in = distance_transform_edt(gt_mask).astype(np.float64)
        diag = math.sqrt(h * h + w * w)
        sdf = (dt_out - dt_in) / diag
        dt_t = torch.tensor(sdf, dtype=torch.float64).unsqueeze(0).unsqueeze(0)
        frame_tensors.append((verts_t, ref_t_t, ref_R_t, gt_t, dt_t, h, w))
    total_verts = sum(ft[0].shape[0] for ft in frame_tensors)
    print(f"  {len(frame_tensors)} frames, {total_verts} total vertices (CPU)")

    # Optimizer
    param_groups = cam.get_param_groups(base_lr=args.lr_scale)
    optimizer = torch.optim.Adam(param_groups)
    # CosineAnnealingLR with relative eta_min (1% of each group's initial lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.steps, eta_min=0.0)

    logger = OptimizationLogger()
    best_iou = 0.0
    best_params = cam.to_dict()
    t0 = time.time()

    print(f"\n{'='*60}")
    print(f"Adam optimization: {len(frames_data)} frames, {args.steps} steps, device={args.device}")
    print(f"Loss weights: alpha_sample={args.alpha_sample}, alpha_dt={args.alpha_dt}")
    print(f"{'='*60}\n")

    rng = np.random.default_rng(42)
    n_frames = len(frame_tensors)
    batch_size = min(args.batch_size, n_frames)

    for step in range(args.steps):
        optimizer.zero_grad()

        # Mini-batch: random subset of frames per step
        if batch_size < n_frames:
            batch_idx = rng.choice(n_frames, batch_size, replace=False)
        else:
            batch_idx = np.arange(n_frames)

        # Forward: project batch frames, accumulate loss
        total_loss = torch.tensor(0.0, dtype=torch.float64, device=device)
        n_valid = 0
        for i in batch_idx:
            verts_t, ref_t_t, ref_R_t, gt_t, dt_t, h, w = frame_tensors[i]
            # Move to GPU for this frame
            verts_g = verts_t.to(device)
            ref_t_g = ref_t_t.to(device)
            ref_R_g = ref_R_t.to(device)
            gt_g = gt_t.to(device)
            dt_g = dt_t.to(device)

            pts2d, visible = differentiable_fisheye_project(verts_g, cam, ref_t_g, ref_R_g, device)
            if pts2d.shape[0] < 10:
                continue
            frame_loss = compute_loss(pts2d, gt_g, dt_g, h, w,
                                      args.alpha_sample, args.alpha_dt)
            total_loss = total_loss + frame_loss
            n_valid += 1

        if n_valid > 0:
            total_loss = total_loss / n_valid

        # Backward
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(cam.parameters(), max_norm=10.0)

        # Collect gradients BEFORE optimizer step
        grad_dict = {}
        for name in PARAM_NAMES:
            p = getattr(cam, name)
            grad_dict[name] = p.grad.item() if p.grad is not None else 0.0

        # Optimizer step
        optimizer.step()
        scheduler.step()
        cam.clamp_to_bounds()

        # Evaluate true IoU at intervals
        true_iou = None
        if step % args.eval_interval == 0 or step == args.steps - 1:
            params_arr = cam.to_array()
            ious = []
            for lv, rt, rR, gm, fh, fw in frames_data:
                pred = project_and_mask(lv, tuple(params_arr), rt, rR, fh, fw)
                ious.append(compute_iou(gm, pred))
            true_iou = float(np.mean(ious))
            if true_iou > best_iou:
                best_iou = true_iou
                best_params = cam.to_dict()

        # Log
        logger.log_step(step, total_loss.item(), cam.to_dict(), grad_dict, true_iou)

        # Console output
        if step % args.print_interval == 0 or step == args.steps - 1:
            elapsed = time.time() - t0
            iou_str = f"  IoU={true_iou:.4f}" if true_iou is not None else ""
            top3 = sorted(grad_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
            grad_str = " ".join(f"{k}:{v:+.4f}" for k, v in top3)
            p = cam.to_dict()
            print(f"  step {step:4d}/{args.steps}  loss={total_loss.item():.6f}{iou_str}  "
                  f"top_grad=[{grad_str}]  "
                  f"({elapsed:.0f}s)")

        # Save comparison images at intervals
        if frames_vis and step % args.save_interval == 0:
            params_arr = cam.to_array()
            n_total = len(frames_data)
            sample_ids = sorted(set([0, n_total // 3, 2 * n_total // 3, min(n_total - 1, n_total)]))
            for si in sample_ids:
                if si >= n_total:
                    continue
                lv, rt, rR, gm, fh, fw = frames_data[si]
                pred = project_and_mask(lv, tuple(params_arr), rt, rR, fh, fw)
                frame_iou = compute_iou(gm, pred)
                img, gt, label = frames_vis[si]
                save_comparison(img, gt, pred, frame_iou, tuple(params_arr),
                                os.path.join(GRAD_OUTPUT_DIR, f"step{step:04d}_{label}.png"))

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"DONE  best_iou={best_iou:.4f}  elapsed={elapsed:.0f}s")
    print(f"Best params: {best_params}")
    print(f"{'='*60}")

    # Save results
    with open(os.path.join(GRAD_OUTPUT_DIR, "best_params.json"), 'w') as f:
        json.dump({"iou": best_iou, "params": best_params}, f, indent=2)

    logger.save_json(os.path.join(GRAD_OUTPUT_DIR, "optimization_log.json"))

    # Per-frame IoU breakdown with best params
    print(f"\nPER-FRAME IoU BREAKDOWN (best params, IoU={best_iou:.4f}):")
    params_arr = np.array([best_params[n] for n in PARAM_NAMES])
    for i, (lv, rt, rR, gm, fh, fw) in enumerate(frames_data):
        pred = project_and_mask(lv, tuple(params_arr), rt, rR, fh, fw)
        iou = compute_iou(gm, pred)
        label = frames_vis[i][2] if frames_vis else f"frame_{i}"
        print(f"  {label}: IoU={iou:.4f}")
        if frames_vis:
            img, gt, lbl = frames_vis[i]
            save_comparison(img, gt, pred, iou, tuple(params_arr),
                            os.path.join(GRAD_OUTPUT_DIR, f"final_{lbl}.png"))

    # Generate plots
    print("\nGenerating analysis plots...")
    logger.plot_all(GRAD_OUTPUT_DIR)

    return best_params, best_iou


# ============================================================
# Main
# ============================================================

def main():
    import warnings
    warnings.filterwarnings('ignore', category=RuntimeWarning)

    parser = argparse.ArgumentParser(description="Camera calibration v3 (Adam gradient)")
    parser.add_argument("--manifest", type=str,
                        default=os.path.join(BASE_DIR, "data", "calib_frames.json"))
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--lr-scale", type=float, default=1.0,
                        help="Multiply all per-group learning rates by this factor")
    parser.add_argument("--alpha-sample", type=float, default=1.0,
                        help="Weight for vertex sampling loss")
    parser.add_argument("--alpha-dt", type=float, default=0.3,
                        help="Weight for distance field loss")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Frames per mini-batch (saves GPU memory)")
    parser.add_argument("--mesh-subsample", type=int, default=4)
    parser.add_argument("--eval-interval", type=int, default=50)
    parser.add_argument("--save-interval", type=int, default=200)
    parser.add_argument("--print-interval", type=int, default=10)
    args = parser.parse_args()

    os.makedirs(GRAD_OUTPUT_DIR, exist_ok=True)

    # Load URDF + meshes
    print("Loading URDF...")
    model = pin.buildModelFromUrdf(G1_URDF, pin.JointModelFreeFlyer())
    data_pin = model.createData()
    link_meshes = parse_urdf_meshes(G1_URDF)
    mesh_cache = preload_meshes(link_meshes, MESH_DIR,
                                 skip_set=SKIP_MESHES,
                                 subsample=args.mesh_subsample)
    print(f"Meshes: {len(mesh_cache)} links (subsample={args.mesh_subsample})")

    # Load calibration frames (reusing v2's loader)
    print(f"\nLoading calibration frames from {args.manifest}")
    frames_data, frames_vis = load_calib_frames(
        args.manifest, model, data_pin, mesh_cache)

    if not frames_data:
        print("ERROR: No calibration frames loaded.")
        sys.exit(1)
    print(f"\nTotal: {len(frames_data)} calibration frames")

    # Run optimization
    seed_params = BEST_PARAMS.copy()
    best_params, best_iou = run_optimization(frames_data, frames_vis, seed_params, args)


if __name__ == "__main__":
    main()
