"""Camera model definitions, projection, and per-frame camera matrix construction.

Merges camera_models.py (model definitions + projection) with
make_camera/make_camera_const (extrinsic matrix building) into one module.

Supported models:
  - pinhole_fixed: 7 params (fixed intrinsics, only cy free)
  - pinhole_f:     9 params (unified focal length f)
  - pinhole:      10 params (separate fx, fy)
  - fisheye:      14 params (separate fx, fy + k1-k4 distortion)
"""

import numpy as np
import cv2

from .config import CAMERA_MODEL

# ── Camera model definitions ──

CAMERA_MODELS = {
    "pinhole_f": {
        "param_names": [
            "dx", "dy", "dz", "pitch", "yaw", "roll",
            "f", "cx", "cy",
        ],
        "bounds": [
            [-0.05, 0.20], [-0.10, 0.15], [0.15, 0.70],
            [-80, -10], [-15, 15], [-10, 10],
            [100, 800], [300, 340], [220, 260],
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
            [-0.50, 0.50], [-0.50, 0.50], [0.00, 1.50],
            [-89, 25], [-45, 45], [-45, 45],
            [30, 800], [30, 800], [60, 580], [0, 480],
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
            [-0.05, 0.20], [-0.10, 0.15], [0.15, 0.70],
            [-80, -10], [-15, 15], [-10, 10],
            [0, 480],
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
            [-0.05, 0.20], [-0.10, 0.15], [0.15, 0.70],
            [-80, -10], [-15, 15], [-10, 10],
            [100, 800], [100, 800], [300, 340], [220, 260],
            [-2.0, 2.0], [-5.0, 5.0], [-5.0, 5.0], [-5.0, 5.0],
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


# ── Per-frame camera matrix construction ──

def make_camera_const(params, camera_model=None):
    """Precompute the camera rotation/intrinsics that are constant across frames.

    Args:
        params: dict with dx, dy, dz, pitch, yaw, roll, and intrinsic params
        camera_model: model name string (defaults to config.CAMERA_MODEL)

    Returns:
        dict with keys: R_cam, K, D, offset, fisheye
    """
    if camera_model is None:
        camera_model = CAMERA_MODEL
    p = params
    pitch = np.radians(p["pitch"])
    yaw = np.radians(p["yaw"])
    roll = np.radians(p["roll"])

    R_pitch = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    R_yaw = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw),  np.cos(yaw), 0],
        [0, 0, 1]
    ])
    R_roll = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll),  np.cos(roll)]
    ])
    R_body_to_cam = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]], dtype=np.float64)
    R_cam = R_body_to_cam @ R_roll @ R_yaw @ R_pitch

    model_cfg = get_model(camera_model)
    K = build_K(p, model_cfg)
    D = build_D(p, model_cfg)
    fisheye = model_is_fisheye(model_cfg)
    offset = np.array([p["dx"], p["dy"], p["dz"]], dtype=np.float64)
    return {"R_cam": R_cam, "K": K, "D": D, "offset": offset, "fisheye": fisheye}


def make_camera(params, transforms, _const=None, camera_model=None):
    """Build camera from params + per-frame torso transform.

    Args:
        params: camera parameter dict
        transforms: FK transforms dict (must contain 'torso_link')
        _const: precomputed constants from make_camera_const() (optional)
        camera_model: model name string (optional, for make_camera_const)

    Returns:
        (K, D, rvec, tvec, R_w2c, t_w2c, fisheye)
    """
    if _const is None:
        _const = make_camera_const(params, camera_model)

    R_cam = _const["R_cam"]
    K = _const["K"]
    D = _const["D"]
    offset = _const["offset"]

    ref_t, ref_R = transforms["torso_link"]
    cam_pos = ref_t + ref_R @ offset
    R_w2c = (ref_R @ R_cam.T).T
    t_w2c = R_w2c @ (-cam_pos)

    fisheye = _const.get("fisheye", False)
    rvec, _ = cv2.Rodrigues(R_w2c)
    tvec = t_w2c.reshape(3, 1)
    return K, D, rvec, tvec, R_w2c, t_w2c, fisheye
