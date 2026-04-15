"""
Keypoint-based camera calibration via Adam or PSO optimization.

Reads user-annotated PNGs (alpha != 255 marks keypoints),
optimizes camera parameters to minimize reprojection error.
Periodically outputs diagnostic overlay images.

Usage:
  python scripts/optimize_keypoints.py --manifest data/5point/manifest.json
  python scripts/optimize_keypoints.py --optimizer pso --particles 100 --pso-iters 500
  python scripts/optimize_keypoints.py --optimizer adam --steps 2000
  python scripts/optimize_keypoints.py --keypoints L_thumb,L_toe,R_toe,R_thumb
"""

import sys
import os
import json
import time
import math
import argparse
import multiprocessing as mp
import numpy as np
import cv2
import pandas as pd
import pinocchio as pin
import torch
import torch.nn as nn

sys.stdout.reconfigure(line_buffering=True)

from src.core.config import (G1_URDF, MESH_DIR, BEST_PARAMS, SKIP_MESHES,
                     DATASET_ROOT, OUTPUT_DIR, MAIN_ROOT,
                     get_hand_type, get_skip_meshes,
                     CAMERA_MODEL)
from src.core.camera import get_model, model_is_fisheye, project_points_cv
from src.core.fk import (build_q, do_fk, parse_urdf_meshes, preload_meshes)
from src.core.camera import make_camera
from src.tools.render_3view import BODY_GROUPS, GROUP_COLORS, get_color as _get_color_hex


# ── Inlined from deleted auto_calibrate_grad.py ──

class CameraParams(nn.Module):
    """Differentiable camera parameters as nn.Parameters."""

    def __init__(self, init_dict, device='cpu', param_names=None):
        super().__init__()
        self._param_names = param_names or list(init_dict.keys())
        for name in self._param_names:
            val = torch.tensor(float(init_dict[name]), dtype=torch.float64, device=device)
            setattr(self, name, nn.Parameter(val))

    def to_dict(self):
        return {n: getattr(self, n).detach().cpu().item() for n in self._param_names}

    def clamp_to_bounds(self, bounds_dict=None):
        if bounds_dict is None:
            return
        with torch.no_grad():
            for name, (lo, hi) in bounds_dict.items():
                if hasattr(self, name):
                    getattr(self, name).clamp_(lo, hi)

    def override_bounds(self, name, lo, hi):
        """Placeholder for compatibility — actual clamping done in clamp_to_bounds."""
        pass


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

def _hex_to_bgr(h):
    r, g, b = int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16)
    return (b, g, r)

def _get_color_bgr(link_name):
    return _hex_to_bgr(_get_color_hex(link_name))

_MODEL_CFG = get_model(CAMERA_MODEL)
PARAM_NAMES = _MODEL_CFG["param_names"]

OUT_DIR = os.path.join(OUTPUT_DIR, "calibration", "kp_optim")

# ── All keypoints (left-to-right in image) ──
ALL_KEYPOINTS = [
    ("L_wrist", "left_wrist_yaw_link",   np.array([ 0.0046,  0.0000,  0.0300])),
    ("L_thumb", "left_thumb_4",          np.array([-0.0314,  0.0150, -0.0101])),
    ("L_toe",   "left_ankle_roll_link",  np.array([ 0.1424,  0.0000, -0.0210])),
    ("R_toe",   "right_ankle_roll_link", np.array([ 0.1424, -0.0000, -0.0215])),
    ("R_thumb", "right_thumb_4",         np.array([ 0.0314,  0.0150, -0.0101])),
    ("R_wrist", "right_wrist_yaw_link",  np.array([ 0.0046,  0.0000,  0.0300])),
]
ALL_KP_NAMES = [k[0] for k in ALL_KEYPOINTS]

# Active keypoints (set by main() from manifest / --keypoints flag)
KEYPOINTS = list(ALL_KEYPOINTS)
KP_NAMES = list(ALL_KP_NAMES)


def set_active_keypoints(names):
    """Filter KEYPOINTS to only include the given names (in order)."""
    name_set = set(names)
    filtered = [kp for kp in ALL_KEYPOINTS if kp[0] in name_set]
    KEYPOINTS.clear()
    KEYPOINTS.extend(filtered)
    KP_NAMES.clear()
    KP_NAMES.extend(kp[0] for kp in filtered)

# dz bounds override for keypoint optimization
_DZ_BOUNDS = [0.15, 1.00]

# Mesh scale bounds (uniform world-space scaling around pelvis)
_SCALE_BOUNDS = [1.0, 1.0]

# ── Per-frame joint offset DOF (feasibility testing) ──
# URDF q indices for 29 body joints (excludes freeflyer + inspire hands).
# Order matches video_inpaint.py:build_q() docstring.
JOINT_Q_INDICES = list(range(7, 29)) + list(range(41, 48))
JOINT_NAMES = [
    "L_hip_pitch", "L_hip_roll", "L_hip_yaw",
    "L_knee", "L_ankle_pitch", "L_ankle_roll",
    "R_hip_pitch", "R_hip_roll", "R_hip_yaw",
    "R_knee", "R_ankle_pitch", "R_ankle_roll",
    "waist_yaw", "waist_roll", "waist_pitch",
    "L_sh_pitch", "L_sh_roll", "L_sh_yaw",
    "L_elbow", "L_wrist_yaw", "L_wrist_roll", "L_wrist_pitch",
    "R_sh_pitch", "R_sh_roll", "R_sh_yaw",
    "R_elbow", "R_wrist_yaw", "R_wrist_roll", "R_wrist_pitch",
]
assert len(JOINT_Q_INDICES) == 29 and len(JOINT_NAMES) == 29

# Module-level pinocchio model for PSO eval with joint-dof mode.
# Populated in run_pso() when args.joint_dof > 0.
_PIN_MODEL = None
_PIN_DATA = None


# ============================================================
# Alpha-channel keypoint detection
# ============================================================

def detect_keypoints_from_alpha(png_path, expected_count=None):
    """Detect keypoints from semi-transparent alpha markers.

    Args:
        png_path: path to PNG with alpha-channel annotations.
        expected_count: if set, validate cluster count matches.

    Returns (N, 2) array of pixel (x, y) sorted left-to-right.
    """
    from scipy import ndimage

    img = cv2.imread(png_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot read {png_path}")
    if img.shape[2] != 4:
        raise ValueError(f"{png_path} has no alpha channel")

    alpha = img[:, :, 3]
    mask = (alpha != 255).astype(np.uint8)

    labeled, n_clusters = ndimage.label(mask)

    centers = []
    for i in range(1, n_clusters + 1):
        ys, xs = np.where(labeled == i)
        centers.append([xs.mean(), ys.mean()])

    # Merge clusters closer than 15px (split annotations)
    merged = True
    while merged:
        merged = False
        for i in range(len(centers)):
            for j in range(i + 1, len(centers)):
                dist = np.sqrt((centers[i][0] - centers[j][0]) ** 2 +
                               (centers[i][1] - centers[j][1]) ** 2)
                if dist < 15:
                    centers[i] = [(centers[i][0] + centers[j][0]) / 2,
                                  (centers[i][1] + centers[j][1]) / 2]
                    centers.pop(j)
                    merged = True
                    break
            if merged:
                break

    if expected_count is not None and len(centers) != expected_count:
        raise ValueError(
            f"Expected {expected_count} keypoint clusters after merging, "
            f"found {len(centers)} in {png_path}")

    # Sort left-to-right by x
    centers.sort(key=lambda c: c[0])
    return np.array(centers, dtype=np.float64)


# ============================================================
# Flexible episode loader (handles cam_0 and head_stereo_left)
# ============================================================

def load_frame_data(task, episode, frame_idx, model, data_pin, user_img=None):
    """Load FK transforms for a specific frame.

    Args:
        user_img: if provided, use this as the frame image (skip video extraction).

    Returns (img_bgr, transforms, ref_t, ref_R, q_base).
    q_base is the full 60-dim URDF q vector, exposed so PSO joint-dof mode
    can rebuild FK with per-particle joint offsets.
    """
    data_dir = os.path.join(DATASET_ROOT, task)

    # Load joint state from parquet
    meta = pd.read_parquet(
        os.path.join(data_dir, "meta", "episodes", "chunk-000", "file-000.parquet"))
    ep_meta = meta[meta["episode_index"] == episode].iloc[0]
    data_fi = int(ep_meta.get("data/file_index", 0))
    df = pd.read_parquet(
        os.path.join(data_dir, "data", "chunk-000", f"file-{data_fi:03d}.parquet"))
    ep_df = df[df["episode_index"] == episode].sort_values("frame_index")
    row = ep_df[ep_df["frame_index"] == frame_idx].iloc[0]

    rq = np.array(row["observation.state.robot_q_current"], dtype=np.float64)
    hs = np.array(row["observation.state.hand_state"], dtype=np.float64)

    q = build_q(model, rq, hs, hand_type=get_hand_type(task))
    transforms = do_fk(model, data_pin, q)
    ref_t, ref_R = transforms["torso_link"]

    # Use user-provided image directly (avoids video extraction mismatch)
    if user_img is not None:
        return user_img, transforms, ref_t, ref_R, q

    raise RuntimeError("No user_img provided and video extraction removed")


# ============================================================
# World positions from FK
# ============================================================

def get_kp_world_positions(transforms, keypoints=None):
    """Compute keypoint world positions from FK transforms.

    Args:
        transforms: FK transforms dict.
        keypoints: list of (name, link_name, offset) tuples.
                   Defaults to module-level KEYPOINTS.

    Returns (N, 3) numpy array.
    """
    kps = keypoints if keypoints is not None else KEYPOINTS
    pts = []
    for name, link_name, local_offset in kps:
        t_link, R_link = transforms[link_name]
        pts.append(R_link @ local_offset + t_link)
    return np.array(pts, dtype=np.float64)


# ============================================================
# Differentiable keypoint projection (PyTorch, for Adam)
# ============================================================

def project_keypoints_diff(world_pts, cam, ref_t, ref_R, device):
    """Differentiable projection of sparse 3D keypoints.

    Supports pinhole_fixed, pinhole_f, pinhole, and fisheye models via cam.model_cfg.

    Returns:
        (N, 2) tensor of pixel coordinates with gradients.
    """
    R_cam = _build_rotation(cam.pitch, cam.yaw, cam.roll, device)
    offset = torch.stack([cam.dx, cam.dy, cam.dz])
    cam_pos = ref_t + ref_R @ offset
    R_w2c = (ref_R @ R_cam.T).T
    t_w2c = R_w2c @ (-cam_pos)

    p_cam = (R_w2c @ world_pts.T).T + t_w2c
    x, y, z = p_cam[:, 0], p_cam[:, 1], p_cam[:, 2]

    fixed = cam.model_cfg.get("fixed_intrinsics", {})

    if model_is_fisheye(cam.model_cfg):
        a = x / (z + 1e-12)
        b = y / (z + 1e-12)
        r = torch.sqrt(a * a + b * b + 1e-12)
        theta = torch.atan(r)
        theta2 = theta * theta
        theta_d = theta * (1.0 + cam.k1 * theta2 + cam.k2 * theta2**2
                           + cam.k3 * theta2**3 + cam.k4 * theta2**4)
        scale = theta_d / (r + 1e-12)
        u = cam.fx * scale * a + cam.cx
        v = cam.fy * scale * b + cam.cy
    elif hasattr(cam, 'f'):
        u = cam.f * (x / (z + 1e-12)) + cam.cx
        v = cam.f * (y / (z + 1e-12)) + cam.cy
    else:
        fx = fixed.get("fx", getattr(cam, "fx"))
        fy = fixed.get("fy", getattr(cam, "fy"))
        cx = fixed.get("cx", getattr(cam, "cx"))
        u = fx * (x / (z + 1e-12)) + cx
        v = fy * (y / (z + 1e-12)) + cam.cy

    return torch.stack([u, v], dim=1)


# ============================================================
# Non-differentiable keypoint projection (OpenCV, for PSO)
# ============================================================

def project_kp_cv(world_pts, params_dict, transforms):
    """Project 3D keypoints via OpenCV. Returns (N, 2) array."""
    K, D, rvec, tvec, _, _, _fisheye = make_camera(params_dict, transforms)
    pts3d = world_pts.reshape(-1, 1, 3).astype(np.float64)
    pts2d = project_points_cv(pts3d, rvec, tvec, K, D, _fisheye)
    return pts2d.reshape(-1, 2)


# ============================================================
# Mesh scale helper
# ============================================================

def _scale_world_pts(world_pts, params_dict, transforms):
    """Scale world keypoint positions around pelvis center."""
    scale = params_dict.get("scale", 1.0)
    if scale == 1.0:
        return world_pts
    center = transforms["pelvis"][0]
    return center + scale * (world_pts - center)


# ============================================================
# Lit mesh overlay (per-part color + Lambertian shading)
# ============================================================

def render_lit_mesh(mesh_cache, transforms, params_dict, h, w,
                    ambient=0.3, diffuse=0.7,
                    scale=None, scale_center=None):
    """Render per-part colored mesh with lighting onto a canvas.

    Returns (h, w, 3) BGR uint8 image and (h, w) uint8 alpha mask.
    """
    K, D, rvec, tvec, R_w2c, t_w2c, _fisheye = make_camera(params_dict, transforms)
    t_flat = t_w2c.flatten()

    # Light direction in world frame: from camera position toward scene
    cam_pos_world = -R_w2c.T @ t_w2c
    light_dir = np.array([0.0, 0.0, 1.0])  # top-down light in world frame

    # Collect all triangles with colors and normals
    all_world_tris = []
    all_colors_bgr = []
    all_normals = []
    all_tri_counts = []

    for link_name, (tris, _unique_verts) in mesh_cache.items():
        if link_name not in transforms or len(tris) == 0:
            continue
        t_link, R_link = transforms[link_name]
        flat = tris.reshape(-1, 3)
        world = (R_link @ flat.T).T + t_link
        if scale is not None and scale != 1.0 and scale_center is not None:
            world = scale_center + scale * (world - scale_center)
        world_tris = world.reshape(-1, 3, 3)

        # Face normals in world frame
        v0, v1, v2 = world_tris[:, 0], world_tris[:, 1], world_tris[:, 2]
        normals = np.cross(v1 - v0, v2 - v0)
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        normals = normals / norms

        color_bgr = _get_color_bgr(link_name)

        all_world_tris.append(world_tris)
        all_normals.append(normals)
        all_colors_bgr.extend([color_bgr] * len(world_tris))
        all_tri_counts.append(len(world_tris))

    if not all_world_tris:
        return np.zeros((h, w, 3), dtype=np.uint8), np.zeros((h, w), dtype=np.uint8)

    all_world_tris = np.concatenate(all_world_tris, axis=0)  # (T, 3, 3)
    all_normals = np.concatenate(all_normals, axis=0)  # (T, 3)
    n_tris = len(all_world_tris)

    # Project to camera
    flat_all = all_world_tris.reshape(-1, 3)
    cam_pts = (R_w2c @ flat_all.T).T + t_flat
    z = cam_pts[:, 2]

    pts3d_cv = flat_all.reshape(-1, 1, 3).astype(np.float64)
    pts2d_cv = project_points_cv(pts3d_cv, rvec, tvec, K, D, _fisheye)
    pts2d = pts2d_cv.reshape(-1, 2)

    # Per-triangle validity
    z_tri = z.reshape(n_tris, 3)
    pts_tri = pts2d.reshape(n_tris, 3, 2)

    valid = ((z_tri > 0.05).all(axis=1) &
             np.all(np.isfinite(pts_tri), axis=(1, 2)) &
             np.all(np.abs(pts_tri) < 5000, axis=(1, 2)))

    depths = z_tri[valid].mean(axis=1)
    pts_valid = pts_tri[valid].astype(np.int32)
    normals_valid = all_normals[valid]
    colors_valid = [all_colors_bgr[i] for i in np.where(valid)[0]]

    # Lambertian shading
    lambert = np.clip(normals_valid @ light_dir, 0, 1)
    shade = ambient + diffuse * lambert  # (N_valid,)

    # Painter's algorithm: back-to-front
    order = np.argsort(-depths)

    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    alpha = np.zeros((h, w), dtype=np.uint8)

    for idx in order:
        base_bgr = colors_valid[idx]
        s = shade[idx]
        lit_bgr = (int(base_bgr[0] * s), int(base_bgr[1] * s), int(base_bgr[2] * s))
        cv2.fillPoly(canvas, [pts_valid[idx]], lit_bgr)
        cv2.fillPoly(alpha, [pts_valid[idx]], 255)

    return canvas, alpha


# ============================================================
# Diagnostic overlay rendering (OpenCV, non-differentiable)
# ============================================================

def render_overlay(img, mesh_cache, transforms, params_dict,
                   gt_pts, proj_pts, step, rmse, errors):
    """Render diagnostic overlay image.

    Args:
        img: (H, W, 3) BGR video frame
        mesh_cache: preloaded mesh dict
        transforms: FK transforms
        params_dict: current camera params as dict
        gt_pts: (4, 2) GT pixel positions (red dots)
        proj_pts: (4, 2) FK projected positions (blue dots)
        step: iteration number
        rmse: total RMSE in pixels
        errors: (4,) per-point errors in pixels
    """
    h, w = img.shape[:2]
    result = img.copy()

    # ── Lit mesh overlay (per-part coloring + shading) ──
    scale = params_dict.get("scale", 1.0)
    sc_center = transforms["pelvis"][0] if scale != 1.0 else None
    mesh_canvas, mesh_alpha = render_lit_mesh(
        mesh_cache, transforms, params_dict, h, w,
        scale=scale, scale_center=sc_center)
    mask_bool = mesh_alpha > 0
    result[mask_bool] = (result[mask_bool] * 0.4 +
                         mesh_canvas[mask_bool] * 0.6).astype(np.uint8)

    # ── Keypoints ──
    n_kp = len(gt_pts)
    for i in range(n_kp):
        gx, gy = int(gt_pts[i, 0]), int(gt_pts[i, 1])
        px, py = int(proj_pts[i, 0]), int(proj_pts[i, 1])

        # Red dot: GT
        cv2.circle(result, (gx, gy), 6, (0, 0, 255), -1)
        cv2.circle(result, (gx, gy), 8, (255, 255, 255), 1)

        # Blue dot: projected
        cv2.circle(result, (px, py), 6, (255, 0, 0), -1)
        cv2.circle(result, (px, py), 8, (255, 255, 255), 1)

        # Line
        cv2.line(result, (gx, gy), (px, py), (0, 255, 255), 1, cv2.LINE_AA)

        # Label
        cv2.putText(result, f"{KP_NAMES[i]} {errors[i]:.1f}px",
                    (max(gx, px) + 10, (gy + py) // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

    # ── Info text ──
    err_parts = "  ".join(f"{KP_NAMES[i]}={errors[i]:.1f}" for i in range(n_kp))
    lines = [
        f"Step {step}  RMSE={rmse:.2f}px",
        f"  {err_parts}",
    ]
    for li, text in enumerate(lines):
        y = 20 + li * 18
        cv2.rectangle(result, (5, y - 14), (5 + len(text) * 8, y + 4), (0, 0, 0), -1)
        cv2.putText(result, text, (8, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

    # ── Legend ──
    ly = h - 30
    cv2.rectangle(result, (5, ly - 2), (280, ly + 18), (0, 0, 0), -1)
    cv2.circle(result, (15, ly + 8), 5, (0, 0, 255), -1)
    cv2.putText(result, "GT (user)", (25, ly + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
    cv2.circle(result, (110, ly + 8), 5, (255, 0, 0), -1)
    cv2.putText(result, "FK proj", (120, ly + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 0), 1)
    cv2.rectangle(result, (195, ly + 3), (210, ly + 13), (0, 200, 0), -1)
    cv2.putText(result, "Mesh", (215, ly + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 200, 0), 1)

    return result


# ============================================================
# Adam optimizer
# ============================================================

def run_adam(optim_frames, mesh_cache, args, out_dir):
    """Run Adam optimization. Returns (best_rmse, best_params, log_entries)."""
    device = torch.device(args.device)

    cam = CameraParams(BEST_PARAMS, device=device)
    cam.override_bounds("dz", *_DZ_BOUNDS)

    lr = args.lr_scale
    _kp_lr_mult = {"position": 50, "angles": 50, "focal": 20,
                    "principal": 20, "distortion": 20}
    param_groups = []
    for g in cam.get_param_groups(base_lr=1.0):
        mult = _kp_lr_mult.get(g["name"], 10)
        param_groups.append({**g, "lr": lr * g["lr"] * mult})
    optimizer = torch.optim.Adam(param_groups)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.steps, eta_min=lr * 0.001)

    frame_tensors = []
    for f in optim_frames:
        frame_tensors.append({
            "world_pts": torch.tensor(f["world_pts"], dtype=torch.float64, device=device),
            "gt_pts": torch.tensor(f["gt_pts"], dtype=torch.float64, device=device),
            "ref_t": torch.tensor(f["ref_t"], dtype=torch.float64, device=device),
            "ref_R": torch.tensor(f["ref_R"], dtype=torch.float64, device=device),
        })

    init_vals = {}
    for name in PARAM_NAMES:
        init_vals[name] = torch.tensor(
            float(BEST_PARAMS[name]), dtype=torch.float64, device=device)
    _rw = _MODEL_CFG["reg_weights"]
    reg_lambda = args.reg_lambda
    print(f"Regularization lambda={reg_lambda}")

    total_pts = sum(len(f["gt_pts"]) for f in optim_frames)
    print(f"\n{'='*60}")
    print(f"Adam optimization: {len(optim_frames)} frames, "
          f"{total_pts} keypoints, {args.steps} steps")
    print(f"{'='*60}\n")

    best_rmse = float('inf')
    best_params = cam.to_dict()
    log_entries = []
    t0 = time.time()

    for step in range(args.steps + 1):
        optimizer.zero_grad()

        total_loss = torch.tensor(0.0, dtype=torch.float64, device=device)
        n_pts = 0

        for ft in frame_tensors:
            proj2d = project_keypoints_diff(
                ft["world_pts"], cam, ft["ref_t"], ft["ref_R"], device)
            diff = proj2d - ft["gt_pts"]
            total_loss = total_loss + (diff * diff).sum()
            n_pts += len(ft["world_pts"])

        mse = total_loss / n_pts

        # Regularization loss
        reg_loss = torch.tensor(0.0, dtype=torch.float64, device=device)
        for name in PARAM_NAMES:
            p = getattr(cam, name)
            reg_loss = reg_loss + _rw[name] * (p - init_vals[name]) ** 2
        total_opt_loss = mse + reg_lambda * reg_loss

        rmse = torch.sqrt(mse)

        if step < args.steps:
            total_opt_loss.backward()
            torch.nn.utils.clip_grad_norm_(cam.parameters(), max_norm=50.0)
            optimizer.step()
            scheduler.step()
            cam.clamp_to_bounds()

        rmse_val = rmse.item()
        if rmse_val < best_rmse:
            best_rmse = rmse_val
            best_params = cam.to_dict()

        grad_dict = {}
        for name in PARAM_NAMES:
            p = getattr(cam, name)
            grad_dict[name] = p.grad.item() if p.grad is not None else 0.0

        log_entries.append({
            "step": step, "rmse_px": rmse_val,
            "params": cam.to_dict(), "gradients": grad_dict,
        })

        # ── Print ──
        if step % args.print_interval == 0 or step == args.steps:
            elapsed = time.time() - t0
            top3 = sorted(grad_dict.items(), key=lambda x: abs(x[1]),
                          reverse=True)[:3]
            grad_str = " ".join(f"{k}:{v:+.4f}" for k, v in top3)
            print(f"  step {step:4d}/{args.steps}  RMSE={rmse_val:.2f}px  "
                  f"grad=[{grad_str}]  ({elapsed:.0f}s)")

        # ── Save overlay ──
        if step % args.save_interval == 0 or step == args.steps:
            cur_params = cam.to_dict()
            with torch.no_grad():
                for i, ft in enumerate(frame_tensors):
                    proj2d = project_keypoints_diff(
                        ft["world_pts"], cam, ft["ref_t"], ft["ref_R"], device)
                    proj_np = proj2d.cpu().numpy()
                    gt_np = ft["gt_pts"].cpu().numpy()
                    errs = np.sqrt(((proj_np - gt_np) ** 2).sum(axis=1))

                    overlay = render_overlay(
                        optim_frames[i]["img"], mesh_cache,
                        optim_frames[i]["transforms"], cur_params,
                        gt_np, proj_np, step, rmse_val, errs)
                    cv2.imwrite(os.path.join(
                        out_dir,
                        f"step_{step:04d}_{optim_frames[i]['label']}.png"),
                        overlay)

            if step % args.save_interval == 0:
                print(f"    -> overlays saved to {out_dir}/step_{step:04d}_*.png")

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"DONE  best_RMSE={best_rmse:.2f}px  elapsed={elapsed:.0f}s")
    print(f"{'='*60}")

    return best_rmse, best_params, log_entries


# ============================================================
# PSO optimizer (Clerc-Kennedy constriction)
# ============================================================

# ── Multiprocessing worker for PSO ──
_PSO_W = {}


def _eval_one_particle(params_arr, frames, pso_names,
                        joint_dof_on, pin_model, pin_data, joint_reg):
    """Pure module-level eval. Used by both worker and single-process paths.

    Returns fitness = RMSE (px) + joint_reg * sum(offsets^2).
    """
    pd = dict(zip(pso_names, params_arr))
    total_sq = 0.0
    n_pts = 0
    reg_sum_sq = 0.0
    for fi, f in enumerate(frames):
        if joint_dof_on:
            offsets = np.array(
                [pd[f"off_f{fi}_{jn}"] for jn in JOINT_NAMES],
                dtype=np.float64)
            q = f["q_base"].copy()
            q[JOINT_Q_INDICES] += offsets
            transforms = do_fk(pin_model, pin_data, q)
            world_pts = get_kp_world_positions(transforms)
            if joint_reg > 0:
                reg_sum_sq += float(np.sum(offsets * offsets))
        else:
            world_pts = f["world_pts"]
            transforms = f["transforms"]
        scaled_pts = _scale_world_pts(world_pts, pd, transforms)
        proj = project_kp_cv(scaled_pts, pd, transforms)
        diff = proj - f["gt_pts"]
        total_sq += np.sum(diff * diff)
        n_pts += len(world_pts)
    return np.sqrt(total_sq / n_pts) + joint_reg * reg_sum_sq


def _pso_worker_init(eval_frames, pso_names, joint_dof_on, joint_reg, urdf_path):
    """Initialize worker with lightweight frame data + optional pinocchio model."""
    _PSO_W['frames'] = eval_frames
    _PSO_W['pso_names'] = pso_names
    _PSO_W['joint_dof_on'] = joint_dof_on
    _PSO_W['joint_reg'] = joint_reg
    if joint_dof_on:
        _PSO_W['pin_model'] = pin.buildModelFromUrdf(
            urdf_path, pin.JointModelFreeFlyer())
        _PSO_W['pin_data'] = _PSO_W['pin_model'].createData()
    else:
        _PSO_W['pin_model'] = None
        _PSO_W['pin_data'] = None


def _pso_eval_particle(params_tuple):
    """Evaluate one particle in worker process. Returns fitness."""
    params_arr = np.array(params_tuple)
    return _eval_one_particle(
        params_arr,
        _PSO_W['frames'],
        _PSO_W['pso_names'],
        _PSO_W['joint_dof_on'],
        _PSO_W['pin_model'],
        _PSO_W['pin_data'],
        _PSO_W['joint_reg'],
    )


def run_pso(optim_frames, mesh_cache, args, out_dir):
    """PSO optimization for keypoint calibration.

    Returns (best_rmse, best_params_dict, log_entries).

    When args.joint_dof > 0, the parameter vector is extended with 29
    per-frame joint offsets each (total = 29 * n_frames). Offsets are added
    to the base URDF q vector and FK is rebuilt at every particle eval.
    """
    global _PIN_MODEL, _PIN_DATA
    n_frames = len(optim_frames)
    joint_dof_on = args.joint_dof > 0
    joint_dof_rad = math.radians(args.joint_dof) if joint_dof_on else 0.0

    # Extend parameter space with mesh scale
    pso_names = list(PARAM_NAMES) + ["scale"]

    bounds = np.array(_MODEL_CFG["bounds"], dtype=np.float64)
    dz_idx = PARAM_NAMES.index("dz")
    bounds[dz_idx] = _DZ_BOUNDS
    bounds = np.vstack([bounds, _SCALE_BOUNDS])

    # Number of "camera" (non-joint) parameters
    n_cam_params = len(pso_names)

    # Joint offset extension
    if joint_dof_on:
        print(f"  Joint DOF mode: +/-{args.joint_dof:.1f} deg per joint, "
              f"{n_frames} frame(s) * 29 joints = {n_frames * 29} extra params")
        if args.joint_reg > 0:
            print(f"  Joint L2 reg weight: {args.joint_reg}")
        for fi in range(n_frames):
            for jn in JOINT_NAMES:
                pso_names.append(f"off_f{fi}_{jn}")
        joint_bounds = np.tile(
            np.array([[-joint_dof_rad, joint_dof_rad]], dtype=np.float64),
            (n_frames * 29, 1))
        bounds = np.vstack([bounds, joint_bounds])

        # Lazy-init module-level pinocchio model/data for FK rebuilds
        if _PIN_MODEL is None:
            print("  Loading pinocchio model for eval FK...")
            _PIN_MODEL = pin.buildModelFromUrdf(G1_URDF,
                                                pin.JointModelFreeFlyer())
            _PIN_DATA = _PIN_MODEL.createData()

    ndim = len(bounds)
    lo, hi = bounds[:, 0], bounds[:, 1]
    span = hi - lo

    n_particles = args.particles
    n_iters = args.pso_iters
    rng = np.random.default_rng(42)

    # ── Initialize positions ──
    positions = rng.uniform(lo, hi, (n_particles, ndim))

    if not args.random_init:
        # Seed 30% around BEST_PARAMS + zero joint offsets
        seed = np.zeros(ndim, dtype=np.float64)
        for i in range(n_cam_params):
            seed[i] = BEST_PARAMS.get(pso_names[i], 1.0)
        # joint offset entries remain 0.0 (trust parquet state as starting point)
        n_seed = max(n_particles * 3 // 10, 1)
        positions[0] = np.clip(seed, lo, hi)
        for i in range(1, n_seed):
            # Perturb only the camera params; keep joints at 0 for seeded particles
            perturbed = seed.copy()
            perturbed[:n_cam_params] += rng.normal(
                0, span[:n_cam_params] * 0.10, n_cam_params)
            positions[i] = np.clip(perturbed, lo, hi)
        print(f"  Seeded {n_seed}/{n_particles} particles from BEST_PARAMS")
    else:
        print(f"  All {n_particles} particles randomly initialized")

    # Clerc-Kennedy constriction: phi=4.1, chi~0.7298
    phi = 4.1
    chi = 2.0 / abs(2.0 - phi - np.sqrt(phi ** 2 - 4.0 * phi))
    c1 = chi * 2.05
    c2 = chi * 2.05

    v_max = span * 0.5
    velocities = rng.uniform(-v_max * 0.1, v_max * 0.1, (n_particles, ndim))

    pbest_pos = positions.copy()
    pbest_val = np.full(n_particles, np.inf)
    gbest_pos = positions[0].copy()
    gbest_val = np.inf

    # ── Prepare evaluation (single-process or multiprocessing) ──
    # Lightweight frame data for workers (no images)
    # For joint-dof mode, also carry q_base so FK can be rebuilt per-particle
    eval_frames = [{"world_pts": f["world_pts"], "gt_pts": f["gt_pts"],
                     "transforms": f["transforms"],
                     "q_base": f["q_base"]} for f in optim_frames]

    n_workers = args.workers
    pool = None
    if n_workers > 0:
        pool = mp.Pool(n_workers, initializer=_pso_worker_init,
                       initargs=(eval_frames, pso_names,
                                 joint_dof_on, args.joint_reg, G1_URDF))
        print(f"  Multiprocessing: {n_workers} workers")

    def eval_particle_local(params_arr):
        return _eval_one_particle(
            params_arr, eval_frames, pso_names,
            joint_dof_on, _PIN_MODEL, _PIN_DATA, args.joint_reg)

    def eval_all(positions):
        if pool is not None:
            param_list = [tuple(positions[i]) for i in range(len(positions))]
            return np.array(pool.map(_pso_eval_particle, param_list))
        return np.array([eval_particle_local(positions[i])
                         for i in range(len(positions))])

    total_pts = sum(len(f["gt_pts"]) for f in optim_frames)
    print(f"\n{'='*60}")
    print(f"PSO: {n_particles} particles, {n_iters} iters, "
          f"{len(optim_frames)} frames, {total_pts} keypoints")
    print(f"{'='*60}\n")

    log_entries = []
    t0 = time.time()
    stall_count = 0
    prev_best = np.inf

    for it in range(n_iters):
        # Evaluate all particles
        scores = eval_all(positions)

        # Update personal bests
        improved = scores < pbest_val
        pbest_val[improved] = scores[improved]
        pbest_pos[improved] = positions[improved]

        # Update global best
        it_best = np.argmin(scores)
        if scores[it_best] < gbest_val:
            gbest_val = scores[it_best]
            gbest_pos = positions[it_best].copy()

        # Stall detection + reset
        if gbest_val < prev_best - 1e-6:
            prev_best = gbest_val
            stall_count = 0
        else:
            stall_count += 1

        if stall_count >= 50:
            stall_count = 0
            n_reset = n_particles // 5
            worst = np.argsort(pbest_val)[-n_reset:]
            positions[worst] = rng.uniform(lo, hi, (n_reset, ndim))
            velocities[worst] = rng.uniform(
                -v_max * 0.1, v_max * 0.1, (n_reset, ndim))
            pbest_val[worst] = np.inf
            pbest_pos[worst] = positions[worst]
            print(f"  [stall reset] randomized {n_reset} particles")

        best_dict = dict(zip(pso_names, gbest_pos))
        # Keep log entries compact: strip per-joint offsets (they blow up JSON)
        log_params = {k: v for k, v in best_dict.items()
                      if not k.startswith("off_")}
        if joint_dof_on:
            # Summary: max abs offset in degrees
            off_abs = [abs(v) for k, v in best_dict.items()
                       if k.startswith("off_")]
            log_params["joint_off_max_deg"] = math.degrees(max(off_abs)) if off_abs else 0.0
        log_entries.append({
            "step": it, "rmse_px": gbest_val, "params": log_params,
        })

        # ── Print ──
        if (it + 1) % args.print_interval == 0 or it == 0:
            elapsed = time.time() - t0
            p = gbest_pos
            info = " ".join(
                f"{pso_names[j]}={p[j]:.2f}"
                for j in range(min(6, n_cam_params)))
            sc = best_dict.get("scale", 1.0)
            joint_tag = ""
            if joint_dof_on:
                off_abs = [abs(v) for k, v in best_dict.items()
                           if k.startswith("off_")]
                joint_tag = f" joff_max={math.degrees(max(off_abs)):.1f}deg" if off_abs else ""
            print(f"  iter {it+1:4d}/{n_iters}  RMSE={gbest_val:.2f}px  "
                  f"{info} scale={sc:.3f}{joint_tag}  ({elapsed:.0f}s)")

        # ── Save overlay ──
        if (it + 1) % args.save_interval == 0 or it == n_iters - 1:
            for i, f in enumerate(optim_frames):
                # When joint-dof is on, rebuild FK with current offsets so
                # the overlay mesh reflects the optimized joint pose.
                if joint_dof_on:
                    offsets = np.array(
                        [best_dict[f"off_f{i}_{jn}"] for jn in JOINT_NAMES],
                        dtype=np.float64)
                    q = f["q_base"].copy()
                    q[JOINT_Q_INDICES] += offsets
                    transforms_cur = do_fk(_PIN_MODEL, _PIN_DATA, q)
                    world_pts_cur = get_kp_world_positions(transforms_cur)
                else:
                    world_pts_cur = f["world_pts"]
                    transforms_cur = f["transforms"]
                scaled_pts = _scale_world_pts(
                    world_pts_cur, best_dict, transforms_cur)
                proj = project_kp_cv(
                    scaled_pts, best_dict, transforms_cur)
                errs = np.sqrt(np.sum(
                    (proj - f["gt_pts"]) ** 2, axis=1))
                overlay = render_overlay(
                    f["img"], mesh_cache, transforms_cur,
                    best_dict, f["gt_pts"], proj,
                    it + 1, gbest_val, errs)
                cv2.imwrite(os.path.join(
                    out_dir,
                    f"step_{it+1:04d}_{f['label']}.png"),
                    overlay)
            print(f"    -> overlays saved")

        # ── Velocity update ──
        r1 = rng.random((n_particles, ndim))
        r2 = rng.random((n_particles, ndim))
        velocities = chi * (velocities
                            + c1 * r1 * (pbest_pos - positions)
                            + c2 * r2 * (gbest_pos - positions))
        velocities = np.clip(velocities, -v_max, v_max)
        positions = np.clip(positions + velocities, lo, hi)

    if pool is not None:
        pool.close()
        pool.join()

    elapsed = time.time() - t0
    best_params = dict(zip(pso_names, gbest_pos))
    print(f"\n{'='*60}")
    print(f"DONE  best_RMSE={gbest_val:.2f}px  scale={best_params.get('scale', 1.0):.3f}  elapsed={elapsed:.0f}s")
    print(f"{'='*60}")

    return gbest_val, best_params, log_entries


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Keypoint-based camera calibration (Adam / PSO)")
    parser.add_argument("--optimizer", choices=["adam", "pso"], default="adam",
                        help="Optimization method")
    parser.add_argument("--manifest", type=str,
                        default="data/4points/manifest.json")
    parser.add_argument("--device", type=str, default="cuda:2")
    # Adam-specific
    parser.add_argument("--steps", type=int, default=2000,
                        help="Adam optimization steps")
    parser.add_argument("--lr-scale", type=float, default=1.0)
    parser.add_argument("--reg-lambda", type=float, default=0.1,
                        help="Regularization strength (Adam)")
    # PSO-specific
    parser.add_argument("--particles", type=int, default=100,
                        help="Number of PSO particles")
    parser.add_argument("--pso-iters", type=int, default=500,
                        help="Number of PSO iterations")
    parser.add_argument("--random-init", action="store_true",
                        help="All particles random (no BEST_PARAMS seeding)")
    parser.add_argument("--workers", type=int, default=0,
                        help="Multiprocessing workers for PSO (0=single-process)")
    # Common
    parser.add_argument("--save-interval", type=int, default=100)
    parser.add_argument("--print-interval", type=int, default=20)
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override output directory")
    parser.add_argument("--keypoints", type=str, default=None,
                        help="Comma-separated keypoint subset, "
                             "e.g. 'L_thumb,L_toe,R_toe,R_thumb'. "
                             "Default: use manifest's keypoints field or all.")
    # Joint DOF (feasibility test): per-frame body joint offsets
    parser.add_argument("--joint-dof", type=float, default=0.0,
                        help="Per-frame joint offset bound in DEGREES. "
                             "0=off (default). When >0, PSO adds 29 offset "
                             "params per frame (body joints, hands excluded), "
                             "each bounded to +/- this value. PSO only.")
    parser.add_argument("--joint-reg", type=float, default=0.0,
                        help="L2 regularization weight on joint offsets "
                             "(rad^2). 0=no reg (pure feasibility).")
    args = parser.parse_args()

    # Guards for joint-dof mode
    if args.joint_dof > 0:
        if args.optimizer != "pso":
            parser.error("--joint-dof requires --optimizer pso "
                         "(Adam needs differentiable FK, not supported)")

    out_dir = args.output_dir or OUT_DIR
    os.makedirs(out_dir, exist_ok=True)

    # ── Load manifest ──
    manifest_path = os.path.join(MAIN_ROOT, args.manifest)
    with open(manifest_path) as f:
        manifest = json.load(f)

    # ── Determine active keypoints ──
    # manifest_keypoints: what's annotated in the images
    manifest_kp_names = manifest.get("keypoints",
                                      ["L_thumb", "L_toe", "R_toe", "R_thumb"])

    if args.keypoints:
        # CLI selects a subset from the manifest's annotated keypoints
        optim_kp_names = [s.strip() for s in args.keypoints.split(",")]
        # Validate all requested names exist in manifest
        for name in optim_kp_names:
            if name not in manifest_kp_names:
                raise ValueError(
                    f"Keypoint '{name}' not in manifest keypoints "
                    f"{manifest_kp_names}")
    else:
        optim_kp_names = list(manifest_kp_names)

    # Compute index mapping: which manifest keypoints to keep
    _kp_indices = [manifest_kp_names.index(n) for n in optim_kp_names]

    set_active_keypoints(optim_kp_names)
    n_kp = len(KEYPOINTS)
    n_kp_manifest = len(manifest_kp_names)
    print(f"Manifest: {len(manifest['frames'])} frames")
    print(f"Annotated keypoints ({n_kp_manifest}): {manifest_kp_names}")
    print(f"Active keypoints ({n_kp}): {KP_NAMES}")

    # Resolve image directory from manifest path
    manifest_dir = os.path.dirname(manifest_path)

    # ── Load URDF model ──
    print("Loading URDF...")
    model = pin.buildModelFromUrdf(G1_URDF, pin.JointModelFreeFlyer())
    data_pin = model.createData()

    # ── Load meshes for overlay ──
    link_meshes = parse_urdf_meshes(G1_URDF)
    mesh_cache = preload_meshes(link_meshes, MESH_DIR,
                                skip_set=SKIP_MESHES, subsample=4)
    print(f"Mesh cache: {len(mesh_cache)} links")

    # ── Parse each annotated frame ──
    optim_frames = []
    for entry in manifest["frames"]:
        png_path = os.path.join(manifest_dir, entry["image"])
        label = os.path.splitext(entry["image"])[0]

        # Detect all annotated keypoints, then select the active subset
        all_gt_pts = detect_keypoints_from_alpha(png_path,
                                                  expected_count=n_kp_manifest)
        gt_pts = all_gt_pts[_kp_indices]

        user_img_bgra = cv2.imread(png_path, cv2.IMREAD_UNCHANGED)
        user_img_bgr = user_img_bgra[:, :, :3].copy()
        print(f"  {label}: {n_kp} active / {n_kp_manifest} detected")
        for i in range(n_kp):
            print(f"    {KP_NAMES[i]}: ({gt_pts[i, 0]:.1f}, {gt_pts[i, 1]:.1f})")

        img, transforms, ref_t, ref_R, q_base = load_frame_data(
            entry["task"], entry["episode"], entry["frame"],
            model, data_pin, user_img=user_img_bgr)

        world_pts = get_kp_world_positions(transforms)

        optim_frames.append({
            "world_pts": world_pts,
            "gt_pts": gt_pts,
            "ref_t": ref_t,
            "ref_R": ref_R,
            "img": img,
            "transforms": transforms,
            "label": label,
            "q_base": q_base,
            "task": entry["task"],
        })

    total_pts = sum(len(f["gt_pts"]) for f in optim_frames)
    print(f"\nTotal: {len(optim_frames)} frames, {total_pts} keypoints")
    print(f"Optimizer: {args.optimizer}, Camera model: {CAMERA_MODEL}")

    # ── Run optimizer ──
    if args.optimizer == "adam":
        best_rmse, best_params, log_entries = run_adam(
            optim_frames, mesh_cache, args, out_dir)
    else:
        best_rmse, best_params, log_entries = run_pso(
            optim_frames, mesh_cache, args, out_dir)

    # ── Per-keypoint breakdown + per-task accumulation ──
    # When joint-dof is on, rebuild FK per frame using best offsets
    from collections import defaultdict
    joint_dof_on = args.joint_dof > 0
    per_task_acc = defaultdict(
        lambda: {"frames": [], "n_pts": 0, "sq_sum": 0.0})
    print("\nPER-KEYPOINT FINAL ERROR:")
    for i, f in enumerate(optim_frames):
        if joint_dof_on:
            offsets = np.array(
                [best_params[f"off_f{i}_{jn}"] for jn in JOINT_NAMES],
                dtype=np.float64)
            q = f["q_base"].copy()
            q[JOINT_Q_INDICES] += offsets
            transforms_cur = do_fk(model, data_pin, q)
            world_pts_cur = get_kp_world_positions(transforms_cur)
        else:
            transforms_cur = f["transforms"]
            world_pts_cur = f["world_pts"]
        scaled_pts = _scale_world_pts(world_pts_cur, best_params, transforms_cur)
        proj = project_kp_cv(scaled_pts, best_params, transforms_cur)
        errs = np.sqrt(np.sum((proj - f["gt_pts"]) ** 2, axis=1))
        print(f"\n  {f['label']}:")
        for j in range(len(KP_NAMES)):
            print(f"    {KP_NAMES[j]:10s}  err={errs[j]:.1f}px  "
                  f"proj=({proj[j,0]:.0f},{proj[j,1]:.0f})  "
                  f"gt=({f['gt_pts'][j,0]:.0f},{f['gt_pts'][j,1]:.0f})")

        # Accumulate for per-task summary
        frame_record = {
            "label": f["label"],
            "rmse_px": float(np.sqrt(np.mean(errs ** 2))),
            "per_kp_px": {KP_NAMES[j]: float(errs[j])
                          for j in range(len(KP_NAMES))},
        }
        if joint_dof_on:
            frame_record["joint_offsets_deg"] = {
                jn: math.degrees(best_params[f"off_f{i}_{jn}"])
                for jn in JOINT_NAMES
            }
        t = f["task"]
        per_task_acc[t]["frames"].append(frame_record)
        per_task_acc[t]["sq_sum"] += float(np.sum(errs ** 2))
        per_task_acc[t]["n_pts"] += len(errs)

    # ── Per-task summary ──
    per_task_summary = {}
    for t, d in per_task_acc.items():
        per_task_summary[t] = {
            "n_frames": len(d["frames"]),
            "rmse_px": float(np.sqrt(d["sq_sum"] / d["n_pts"])),
            "frames": d["frames"],
        }
    print("\nPER-TASK SUMMARY:")
    for t in sorted(per_task_summary):
        s = per_task_summary[t]
        print(f"  {t}: {s['n_frames']} frame(s), RMSE = {s['rmse_px']:.2f}px")
        for fr in s["frames"]:
            print(f"    {fr['label']:60s} {fr['rmse_px']:6.2f}px")
    with open(os.path.join(out_dir, "per_task_summary.json"), 'w') as f:
        json.dump(per_task_summary, f, indent=2)
    print(f"Per-task summary saved to {out_dir}/per_task_summary.json")

    # ── Joint offset report (top 5 per frame) ──
    joint_offsets_struct = {}  # for best_params.json
    if joint_dof_on:
        print("\nJOINT OFFSETS (top 5 by |deg| per frame):")
        for i, f in enumerate(optim_frames):
            offsets_deg = {
                jn: math.degrees(best_params[f"off_f{i}_{jn}"])
                for jn in JOINT_NAMES
            }
            joint_offsets_struct[f["label"]] = offsets_deg
            top = sorted(offsets_deg.items(), key=lambda x: -abs(x[1]))[:5]
            print(f"\n  {f['label']}:")
            for name, deg in top:
                print(f"    {name:14s} = {deg:+7.2f} deg")

    # ── Save results ──
    out_params = {k: v for k, v in best_params.items()
                  if not k.startswith("off_")}
    save_payload = {
        "rmse_px": best_rmse,
        "params": out_params,
        "optimizer": args.optimizer,
        "camera_model": CAMERA_MODEL,
    }
    if joint_dof_on:
        save_payload["joint_dof_deg"] = args.joint_dof
        save_payload["joint_reg"] = args.joint_reg
        save_payload["joint_offsets_deg"] = joint_offsets_struct
    with open(os.path.join(out_dir, "best_params.json"), 'w') as f:
        json.dump(save_payload, f, indent=2)
    with open(os.path.join(out_dir, "optimization_log.json"), 'w') as f:
        json.dump(log_entries, f)
    print(f"\nBest params saved to {out_dir}/best_params.json")

    # ── Bound check (camera params only; joint offsets summarized) ──
    all_names = list(PARAM_NAMES)
    bounds = list(_MODEL_CFG["bounds"])
    bounds[PARAM_NAMES.index("dz")] = _DZ_BOUNDS
    if "scale" in best_params:
        all_names = all_names + ["scale"]
        bounds = bounds + [_SCALE_BOUNDS]
    print("\nBound check:")
    for i, name in enumerate(all_names):
        b_lo, b_hi = bounds[i]
        v = best_params[name]
        flag = ""
        if abs(v - b_lo) < 1e-4:
            flag = " *** HIT LOWER BOUND ***"
        elif abs(v - b_hi) < 1e-4:
            flag = " *** HIT UPPER BOUND ***"
        print(f"  {name:>6s} = {v:10.4f}  [{b_lo:8.2f}, {b_hi:8.2f}]{flag}")

    if joint_dof_on:
        all_off_rad = [best_params[k] for k in best_params
                       if k.startswith("off_")]
        max_abs_deg = math.degrees(max(abs(v) for v in all_off_rad))
        mean_abs_deg = math.degrees(
            sum(abs(v) for v in all_off_rad) / len(all_off_rad))
        n_at_bound = sum(1 for v in all_off_rad
                         if abs(abs(math.degrees(v)) - args.joint_dof) < 0.05)
        print(f"  joint_offsets: max=|{max_abs_deg:.2f} deg|  "
              f"mean=|{mean_abs_deg:.2f} deg|  "
              f"n_at_bound={n_at_bound}/{len(all_off_rad)}")

    # ── Config snippet ──
    print("\n# For config.py BEST_PARAMS:")
    print("BEST_PARAMS = {")
    for name in all_names:
        print(f'    "{name}": {best_params[name]:.4f},')
    print("}")

    # ── Loss curve ──
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        steps = [e["step"] for e in log_entries]
        rmses = [e["rmse_px"] for e in log_entries]

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(steps, rmses, 'b-', linewidth=0.8)
        ax.set_xlabel('Step' if args.optimizer == 'adam' else 'Iteration')
        ax.set_ylabel('RMSE (pixels)')
        ax.set_title(f'Keypoint Reprojection Error '
                     f'[{args.optimizer.upper()}] (best={best_rmse:.2f}px)')
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "loss_curve.png"), dpi=150)
        plt.close(fig)
        print(f"Loss curve saved to {out_dir}/loss_curve.png")
    except Exception as e:
        print(f"Warning: plot failed: {e}")


if __name__ == "__main__":
    main()
