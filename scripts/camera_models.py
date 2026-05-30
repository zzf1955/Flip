"""Camera model definitions for calibration pipeline.

Three models supported:
  - pinhole_f:  9 params (unified focal length f)
  - pinhole:   10 params (separate fx, fy)
  - fisheye:   14 params (separate fx, fy + k1-k4 distortion)

Switch active model in config.py via CAMERA_MODEL.
"""

import numpy as np
import cv2

CAMERA_MODELS = {
    "pinhole_f": {
        "param_names": [
            "dx", "dy", "dz", "pitch", "yaw", "roll",
            "f", "cx", "cy",
        ],
        "bounds": [
            [-0.05, 0.20],   # dx (m)
            [-0.10, 0.15],   # dy (m)
            [0.15, 0.70],    # dz (m)
            [-80, -10],      # pitch (deg)
            [-15, 15],       # yaw (deg)
            [-10, 10],       # roll (deg)
            [100, 800],      # f (px)
            [300, 340],      # cx (px)
            [220, 260],      # cy (px)
        ],
        "lr_groups": {
            "position":  {"params": ["dx", "dy", "dz"],       "lr_scale": 0.0001},
            "angles":    {"params": ["pitch", "yaw", "roll"],  "lr_scale": 0.01},
            "focal":     {"params": ["f"],                     "lr_scale": 0.1},
            "principal": {"params": ["cx", "cy"],              "lr_scale": 0.05},
        },
        "reg_weights": {
            "dx": 500, "dy": 500, "dz": 500,
            "pitch": 0.5, "yaw": 0.5, "roll": 0.5,
            "f": 0.1, "cx": 0.5, "cy": 0.5,
        },
    },
    "pinhole": {
        "param_names": [
            "dx", "dy", "dz", "pitch", "yaw", "roll",
            "fx", "fy", "cx", "cy",
        ],
        "bounds": [
            [-0.50, 0.50],   # dx (m)      — center ~0, ±0.50
            [-0.50, 0.50],   # dy (m)      — center ~0, ±0.50
            [0.00, 1.50],    # dz (m)      — center ~0.75, ±0.75
            [-89, 25],       # pitch (deg) — center ~-32, ±57
            [-45, 45],       # yaw (deg)   — center 0, ±45
            [-45, 45],       # roll (deg)  — center 0, ±45
            [30, 800],       # fx (px)     — center ~415, ±385
            [30, 800],       # fy (px)     — center ~415, ±385
            [60, 580],       # cx (px)     — center 320, ±260
            [0, 480],        # cy (px)     — center 240, ±240
        ],
        "lr_groups": {
            "position":  {"params": ["dx", "dy", "dz"],       "lr_scale": 0.0001},
            "angles":    {"params": ["pitch", "yaw", "roll"],  "lr_scale": 0.01},
            "focal":     {"params": ["fx", "fy"],              "lr_scale": 0.1},
            "principal": {"params": ["cx", "cy"],              "lr_scale": 0.05},
        },
        "reg_weights": {
            "dx": 500, "dy": 500, "dz": 500,
            "pitch": 0.5, "yaw": 0.5, "roll": 0.5,
            "fx": 0.1, "fy": 0.1, "cx": 0.5, "cy": 0.5,
        },
    },
    "pinhole_fixed": {
        "param_names": [
            "dx", "dy", "dz", "pitch", "yaw", "roll", "cy",
        ],
        "fixed_intrinsics": {"fx": 290.78, "fy": 287.35, "cx": 320.0},
        "bounds": [
            [-0.05, 0.20],   # dx (m)
            [-0.10, 0.15],   # dy (m)
            [0.15, 0.70],    # dz (m)
            [-80, -10],      # pitch (deg)
            [-15, 15],       # yaw (deg)
            [-10, 10],       # roll (deg)
            [0, 480],        # cy (px)
        ],
        "lr_groups": {
            "position":  {"params": ["dx", "dy", "dz"],       "lr_scale": 0.0001},
            "angles":    {"params": ["pitch", "yaw", "roll"],  "lr_scale": 0.01},
            "principal": {"params": ["cy"],                    "lr_scale": 0.05},
        },
        "reg_weights": {
            "dx": 500, "dy": 500, "dz": 500,
            "pitch": 0.5, "yaw": 0.5, "roll": 0.5,
            "cy": 0.5,
        },
    },
    "fisheye": {
        "param_names": [
            "dx", "dy", "dz", "pitch", "yaw", "roll",
            "fx", "fy", "cx", "cy", "k1", "k2", "k3", "k4",
        ],
        "bounds": [
            [-0.05, 0.20],   # dx
            [-0.10, 0.15],   # dy
            [0.15, 0.70],    # dz
            [-80, -10],      # pitch
            [-15, 15],       # yaw
            [-10, 10],       # roll
            [100, 800],      # fx
            [100, 800],      # fy
            [300, 340],      # cx
            [220, 260],      # cy
            [-2.0, 2.0],     # k1
            [-5.0, 5.0],     # k2
            [-5.0, 5.0],     # k3
            [-5.0, 5.0],     # k4
        ],
        "lr_groups": {
            "position":   {"params": ["dx", "dy", "dz"],           "lr_scale": 0.0001},
            "angles":     {"params": ["pitch", "yaw", "roll"],     "lr_scale": 0.01},
            "focal":      {"params": ["fx", "fy"],                 "lr_scale": 0.1},
            "principal":  {"params": ["cx", "cy"],                 "lr_scale": 0.05},
            "distortion": {"params": ["k1", "k2", "k3", "k4"],    "lr_scale": 0.001},
        },
        "reg_weights": {
            "dx": 500, "dy": 500, "dz": 500,
            "pitch": 0.5, "yaw": 0.5, "roll": 0.5,
            "fx": 0.1, "fy": 0.1, "cx": 0.5, "cy": 0.5,
            "k1": 100, "k2": 100, "k3": 100, "k4": 100,
        },
    },
}


def get_model(name):
    """Return model config dict."""
    return CAMERA_MODELS[name]


def build_K(params_dict, model_cfg):
    """Build 3x3 intrinsic matrix from params dict.

    Supports fixed_intrinsics: values not in param_names are read from
    model_cfg["fixed_intrinsics"] instead of params_dict.
    """
    fixed = model_cfg.get("fixed_intrinsics", {})
    if "f" in model_cfg["param_names"]:
        f = params_dict["f"]
        fx = fy = f
    else:
        fx = fixed.get("fx", params_dict.get("fx"))
        fy = fixed.get("fy", params_dict.get("fy"))
    cx = fixed.get("cx", params_dict.get("cx"))
    cy = params_dict.get("cy", fixed.get("cy"))
    return np.array([[fx, 0, cx],
                     [0, fy, cy],
                     [0, 0, 1]], dtype=np.float64)


def build_D(params_dict, model_cfg):
    """Build distortion vector. Returns (4,1) for fisheye, (5,1) zeros otherwise."""
    dist_names = [n for n in model_cfg["param_names"] if n.startswith("k")]
    if dist_names:
        return np.array([[params_dict[n]] for n in dist_names], dtype=np.float64)
    return np.zeros((5, 1), dtype=np.float64)


def model_is_fisheye(model_cfg):
    """Whether this model uses fisheye projection."""
    return any(n.startswith("k") for n in model_cfg["param_names"])


def project_points_cv(pts, rvec, tvec, K, D, fisheye=False):
    """Unified OpenCV projection wrapper.

    Args:
        pts: (N, 1, 3) float64
        fisheye: if True, use cv2.fisheye.projectPoints (D is 4x1)

    Returns:
        pts2d: (N, 1, 2) float64
    """
    if fisheye:
        pts2d, _ = cv2.fisheye.projectPoints(pts, rvec, tvec, K, D)
    else:
        pts2d, _ = cv2.projectPoints(pts, rvec, tvec, K, D)
    return pts2d
