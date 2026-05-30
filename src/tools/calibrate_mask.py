"""PSO camera + joint calibration using hand-drawn mask with F1 (Dice) score.

Optimizes camera parameters AND per-joint angle offsets (+-N degrees) by
maximizing F1 overlap between FK-rendered mask and a GT RGBA mask.

Uses convex-hull per-link rendering for fast evaluation.

Usage:
  python scripts/auto_calibrate_mask.py --camera-model pinhole --joint-dof 10
  python scripts/auto_calibrate_mask.py --camera-model pinhole --joint-dof 10 --joint-reg 0.001
  python scripts/auto_calibrate_mask.py --joint-dof 0  # camera-only mode
"""

import sys
import os
import math
import time
import json
import argparse
import numpy as np
import cv2
import pinocchio as pin

sys.stdout.reconfigure(line_buffering=True)

from src.core.config import (
    G1_URDF, MESH_DIR, BEST_PARAMS_BY_MODEL, CAMERA_MODEL,
    DATASET_ROOT, OUTPUT_DIR, CALIB_MASK_DIR,
    get_hand_type, get_skip_meshes,
)
from src.core.camera import get_model, build_K, build_D, model_is_fisheye
from src.core.fk import (
    build_q, do_fk, parse_urdf_meshes, preload_meshes,
)
from src.core.render import render_mask
from src.core.camera import project_points_cv
import src.core.camera as video_inpaint_camera

# ── Constants ──
TASK_NAME = "G1_WBT_Inspire_Put_Clothes_into_Washing_Machine_MainCamOnly"
CAMERA_NAME = "observation.images.cam_0"
GT_MASK_PATH = os.path.join(
    CALIB_MASK_DIR,
    "G1_WBT_Inspire_Put_Clothes_into_Washing_Machine_f001.png",
)
EPISODE = 0
FRAME_IDX = 0
H, W = 480, 640

# ── Joint definitions (from optimize_keypoints.py) ──
JOINT_Q_INDICES = list(range(7, 29)) + list(range(41, 48))  # 29 body joints
JOINT_NAMES = [
    "L_hip_pitch", "L_hip_roll", "L_hip_yaw",
    "L_knee", "L_ankle_pitch", "L_ankle_roll",
    "R_hip_pitch", "R_hip_roll", "R_hip_yaw",
    "R_knee", "R_ankle_pitch", "R_ankle_roll",
    "waist_yaw", "waist_roll", "waist_pitch",
    "L_shoulder_pitch", "L_shoulder_roll", "L_shoulder_yaw",
    "L_elbow", "L_elbow_roll", "L_wrist_yaw", "L_wrist_pitch",
    "R_shoulder_pitch", "R_shoulder_roll", "R_shoulder_yaw",
    "R_elbow", "R_elbow_roll", "R_wrist_yaw", "R_wrist_pitch",
]
N_JOINTS = len(JOINT_NAMES)  # 29


def load_gt_mask(path):
    """Load GT mask from RGBA PNG. alpha < 128 = robot (True)."""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot read mask: {path}")
    if img.ndim != 3 or img.shape[2] != 4:
        raise ValueError(f"Expected RGBA image, got shape {img.shape}")
    return img[:, :, 3] < 128


def compute_f1(pred, gt):
    """Dice coefficient (F1): 2*|A&B| / (|A|+|B|). Both boolean."""
    inter = np.count_nonzero(pred & gt)
    total = np.count_nonzero(pred) + np.count_nonzero(gt)
    return 2.0 * inter / total if total > 0 else 0.0


def precompute_world_verts(mesh_cache, transforms):
    """Precompute world-space unique vertices per link."""
    link_verts = {}
    for link_name, (_, unique_verts) in mesh_cache.items():
        if link_name not in transforms or len(unique_verts) == 0:
            continue
        t_link, R_link = transforms[link_name]
        verts_w = (R_link @ unique_verts.T).T + t_link
        link_verts[link_name] = verts_w.astype(np.float64)
    return link_verts


def project_and_mask(link_verts, params_dict, ref_t, ref_R, h, w, cam_model):
    """Fast convex-hull mask rendering from precomputed world vertices."""
    p = params_dict
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

    model_cfg = get_model(cam_model)
    K = build_K(p, model_cfg)
    D = build_D(p, model_cfg)
    fisheye = model_is_fisheye(model_cfg)

    offset = np.array([p["dx"], p["dy"], p["dz"]], dtype=np.float64)
    cam_pos = ref_t + ref_R @ offset
    R_w2c = (ref_R @ R_cam.T).T
    t_w2c = R_w2c @ (-cam_pos)
    t_w2c_flat = t_w2c.flatten()
    rvec, _ = cv2.Rodrigues(R_w2c)
    tvec = t_w2c.reshape(3, 1)

    mask = np.zeros((h, w), dtype=np.uint8)
    for verts_w in link_verts.values():
        depths = (R_w2c @ verts_w.T).T + t_w2c_flat
        in_front = depths[:, 2] > 0.01
        if np.count_nonzero(in_front) < 3:
            continue
        pts2d = project_points_cv(
            verts_w[in_front].reshape(-1, 1, 3), rvec, tvec, K, D, fisheye
        )
        pts2d = pts2d.reshape(-1, 2)
        finite = np.all(np.isfinite(pts2d), axis=1)
        pts2d = pts2d[finite]
        if len(pts2d) < 3:
            continue
        hull = cv2.convexHull(pts2d.astype(np.float32))
        cv2.fillConvexPoly(mask, hull.astype(np.int32), 255)
    return mask


# ── Global state for evaluate (avoid passing many args) ──
_G = {}


def _worker_init(g_dict):
    """Initialize worker process with its own pinocchio model."""
    _G.update(g_dict)
    # Each worker needs its own pinocchio model+data (not picklable)
    if g_dict["joint_dof_on"]:
        _G["pin_model"] = pin.buildModelFromUrdf(G1_URDF, pin.JointModelFreeFlyer())
        _G["pin_data"] = _G["pin_model"].createData()


def evaluate(params_array):
    """Evaluate a single particle. Returns F1 score (minus regularization)."""
    g = _G
    n_cam = g["n_cam"]
    cam_names = g["cam_names"]
    cam_dict = dict(zip(cam_names, params_array[:n_cam].astype(float)))

    # Rebuild FK if joint offsets are active
    if g["joint_dof_on"]:
        offsets = params_array[n_cam:]
        q = g["q_base"].copy()
        q[JOINT_Q_INDICES] += offsets
        transforms = do_fk(g["pin_model"], g["pin_data"], q)
        link_verts = precompute_world_verts(g["mesh_cache"], transforms)
        ref_t, ref_R = transforms["torso_link"]
    else:
        link_verts = g["link_verts"]
        ref_t = g["ref_t"]
        ref_R = g["ref_R"]

    pred = project_and_mask(link_verts, cam_dict, ref_t, ref_R,
                            g["h"], g["w"], g["cam_model"])
    f1 = compute_f1(pred > 0, g["gt_mask"])

    # L2 regularization on joint offsets
    if g["joint_dof_on"] and g["joint_reg"] > 0:
        f1 -= g["joint_reg"] * float(np.sum(offsets ** 2))

    return f1


def _eval_wrapper(params_tuple):
    """Wrapper for multiprocessing Pool.map (tuple -> evaluate)."""
    return evaluate(np.array(params_tuple))


def render_from_params(params_array):
    """Render mask from params (for visualization)."""
    g = _G
    n_cam = g["n_cam"]
    cam_names = g["cam_names"]
    cam_dict = dict(zip(cam_names, params_array[:n_cam].astype(float)))

    if g["joint_dof_on"]:
        offsets = params_array[n_cam:]
        q = g["q_base"].copy()
        q[JOINT_Q_INDICES] += offsets
        transforms = do_fk(g["pin_model"], g["pin_data"], q)
        link_verts = precompute_world_verts(g["mesh_cache"], transforms)
        ref_t, ref_R = transforms["torso_link"]
    else:
        link_verts = g["link_verts"]
        ref_t = g["ref_t"]
        ref_R = g["ref_R"]

    return project_and_mask(link_verts, cam_dict, ref_t, ref_R,
                            g["h"], g["w"], g["cam_model"])


def extract_video_frame(video_path, from_ts, frame_idx):
    """Extract a single video frame by episode-relative frame index."""
    import av
    container = av.open(video_path)
    stream = container.streams.video[0]
    fps = float(stream.average_rate)
    seek_sec = from_ts + frame_idx / fps
    if seek_sec > 1.0:
        container.seek(int((seek_sec - 1.0) / stream.time_base), stream=stream)
    for f in container.decode(stream):
        pts_sec = float(f.pts * stream.time_base)
        ep_fi = int(round((pts_sec - from_ts) * fps))
        if ep_fi >= frame_idx:
            img = f.to_ndarray(format="bgr24")
            container.close()
            return img
    container.close()
    return None


def pso(n_particles, n_iters, bounds, seed_params, all_param_names,
        rng_seed=42, save_fn=None, pool=None):
    """PSO with Clerc-Kennedy constriction factor."""
    rng = np.random.default_rng(rng_seed)
    ndim = len(all_param_names)
    lo = bounds[:, 0]
    hi = bounds[:, 1]
    span = hi - lo

    # Constriction factor
    phi = 4.1
    chi = 2.0 / abs(2.0 - phi - np.sqrt(phi ** 2 - 4 * phi))
    c1 = chi * 2.05
    c2 = chi * 2.05
    v_max = span * 0.5

    # Initialize positions
    positions = np.zeros((n_particles, ndim))
    n_seed = max(1, int(n_particles * 0.3))

    # First particle = exact seed
    positions[0] = np.clip(seed_params, lo, hi)

    # Seeded particles: gaussian around seed
    for i in range(1, n_seed):
        noise = rng.normal(0, span * 0.10)
        positions[i] = np.clip(seed_params + noise, lo, hi)

    # Random particles
    for i in range(n_seed, n_particles):
        positions[i] = rng.uniform(lo, hi)

    # Initialize velocities
    velocities = rng.uniform(-v_max * 0.1, v_max * 0.1, (n_particles, ndim))

    # Personal bests
    pbest_pos = positions.copy()
    pbest_val = np.full(n_particles, -1.0)

    # Global best
    gbest_pos = positions[0].copy()
    gbest_val = -1.0

    stall_count = 0
    n_cam = _G["n_cam"]
    log_entries = []
    t_start = time.time()

    for it in range(n_iters):
        # Evaluate all particles
        if pool is not None:
            scores = np.array(pool.map(
                _eval_wrapper, [tuple(positions[i]) for i in range(n_particles)]))
        else:
            scores = np.zeros(n_particles)
            for i in range(n_particles):
                scores[i] = evaluate(positions[i])

        # Update personal bests
        improved = scores > pbest_val
        pbest_val[improved] = scores[improved]
        pbest_pos[improved] = positions[improved]

        # Update global best
        best_idx = np.argmax(pbest_val)
        if pbest_val[best_idx] > gbest_val + 1e-6:
            gbest_val = pbest_val[best_idx]
            gbest_pos = pbest_pos[best_idx].copy()
            stall_count = 0
        else:
            stall_count += 1

        # Log entry
        log_entries.append({
            "iter": it,
            "best_f1": round(float(gbest_val), 6),
            "mean_f1": round(float(scores.mean()), 6),
            "elapsed_sec": round(time.time() - t_start, 2),
        })

        # Progress
        if it % 10 == 0 or it == n_iters - 1:
            elapsed = time.time() - t_start
            eta = elapsed / (it + 1) * (n_iters - it - 1)
            cam_str = ", ".join(
                f"{all_param_names[j]}={gbest_pos[j]:.4f}"
                for j in range(n_cam)
            )
            print(
                f"[{it:4d}/{n_iters}] best_F1={gbest_val:.4f}  "
                f"mean={scores.mean():.4f}  stall={stall_count}  "
                f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s"
            )
            if it % 50 == 0:
                print(f"  cam: {cam_str}")
                if _G["joint_dof_on"]:
                    offsets_deg = np.degrees(gbest_pos[n_cam:])
                    top5 = np.argsort(np.abs(offsets_deg))[-5:][::-1]
                    top_str = ", ".join(
                        f"{JOINT_NAMES[j]}={offsets_deg[j]:+.1f}\u00b0"
                        for j in top5
                    )
                    print(f"  top5 joints: {top_str}")

        # Save intermediate result every 10 iters
        if save_fn and (it % 10 == 0 or it == n_iters - 1):
            save_fn(it, gbest_pos, gbest_val)

        # Stall detection: reset 20% worst after 50 iters
        if stall_count >= 50:
            n_reset = max(1, int(n_particles * 0.2))
            worst = np.argsort(pbest_val)[:n_reset]
            for idx in worst:
                positions[idx] = rng.uniform(lo, hi)
                velocities[idx] = rng.uniform(-v_max * 0.1, v_max * 0.1)
                pbest_val[idx] = -1.0
                pbest_pos[idx] = positions[idx].copy()
            stall_count = 0
            print(f"  [stall reset] reinit {n_reset} particles")

        # Velocity & position update
        r1 = rng.random((n_particles, ndim))
        r2 = rng.random((n_particles, ndim))
        velocities = chi * (
            velocities
            + c1 * r1 * (pbest_pos - positions)
            + c2 * r2 * (gbest_pos - positions)
        )
        velocities = np.clip(velocities, -v_max, v_max)
        positions = np.clip(positions + velocities, lo, hi)

    total_time = time.time() - t_start
    print(f"\nPSO done: {n_iters} iters in {total_time:.1f}s")
    return gbest_pos, gbest_val, log_entries


def save_comparison(img, gt_mask, pred_mask_uint8, f1, params_dict,
                    display_names, out_path):
    """Save 2x2 comparison image."""
    h, w = img.shape[:2]
    pred_bool = pred_mask_uint8 > 0
    gt_bool = gt_mask

    tl = img.copy()
    tl[gt_bool, 2] = np.clip(tl[gt_bool, 2].astype(int) + 120, 0, 255).astype(np.uint8)

    tr = img.copy()
    tr[pred_bool, 0] = np.clip(tr[pred_bool, 0].astype(int) + 120, 0, 255).astype(np.uint8)

    bl = img.copy()

    br = img.copy()
    both = gt_bool & pred_bool
    gt_only = gt_bool & ~pred_bool
    pred_only = pred_bool & ~gt_bool
    br[both, 1] = np.clip(br[both, 1].astype(int) + 120, 0, 255).astype(np.uint8)
    br[gt_only, 2] = np.clip(br[gt_only, 2].astype(int) + 120, 0, 255).astype(np.uint8)
    br[pred_only, 0] = np.clip(br[pred_only, 0].astype(int) + 120, 0, 255).astype(np.uint8)

    cv2.putText(tl, "GT mask", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0, 0, 255), 2)
    cv2.putText(tr, "Predicted mask", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (255, 0, 0), 2)
    cv2.putText(bl, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (255, 255, 255), 2)

    cv2.putText(br, f"F1={f1:.4f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0, 255, 0), 2)
    y = 55
    for n in display_names:
        v = params_dict.get(n, 0)
        cv2.putText(br, f"{n}={v:.4f}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (255, 255, 255), 1)
        y += 18

    cv2.putText(br, "green=both  red=GT-only  blue=pred-only",
                (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                (200, 200, 200), 1)

    top = np.hstack([tl, tr])
    bot = np.hstack([bl, br])
    grid = np.vstack([top, bot])

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, grid)
    print(f"Saved comparison: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="PSO camera+joint calibration via mask F1")
    parser.add_argument("--particles", type=int, default=200)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--mesh-subsample", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--camera-model", type=str, default=None,
                        help="Override CAMERA_MODEL (default: from config.py)")
    parser.add_argument("--joint-dof", type=float, default=10.0,
                        help="Per-joint offset freedom in degrees (0=disabled)")
    parser.add_argument("--joint-reg", type=float, default=0.0,
                        help="L2 regularization weight on joint offsets")
    parser.add_argument("--workers", type=int, default=128,
                        help="Number of parallel workers (0=auto, 1=single-process)")
    args = parser.parse_args()

    # Resolve camera model
    cam_model = args.camera_model or CAMERA_MODEL
    if args.camera_model and args.camera_model != CAMERA_MODEL:
        video_inpaint_camera.CAMERA_MODEL = args.camera_model
        print(f"Camera model override: {CAMERA_MODEL} -> {args.camera_model}")

    model_cfg = get_model(cam_model)
    cam_names = model_cfg["param_names"]
    cam_bounds = np.array(model_cfg["bounds"])
    n_cam = len(cam_names)
    seed_params_dict = BEST_PARAMS_BY_MODEL[cam_model]
    cam_seed = np.array([seed_params_dict[n] for n in cam_names])

    # Joint DOF
    joint_dof_on = args.joint_dof > 0
    joint_dof_rad = math.radians(args.joint_dof) if joint_dof_on else 0.0
    joint_offset_names = [f"off_{jn}" for jn in JOINT_NAMES]

    if joint_dof_on:
        joint_bounds = np.tile(
            [[-joint_dof_rad, joint_dof_rad]], (N_JOINTS, 1))
        all_bounds = np.vstack([cam_bounds, joint_bounds])
        all_names = cam_names + joint_offset_names
        all_seed = np.concatenate([cam_seed, np.zeros(N_JOINTS)])
    else:
        all_bounds = cam_bounds
        all_names = cam_names
        all_seed = cam_seed

    ndim = len(all_names)
    print(f"Camera model: {cam_model} ({n_cam} cam params)")
    if joint_dof_on:
        print(f"Joint DOF: \u00b1{args.joint_dof}\u00b0 ({N_JOINTS} joints), reg={args.joint_reg}")
    print(f"Total params: {ndim}")

    # Load URDF + meshes
    hand_type = get_hand_type(TASK_NAME)
    skip_set = get_skip_meshes(hand_type)
    print(f"Loading URDF + meshes (hand_type={hand_type}, subsample={args.mesh_subsample})...")
    pin_model = pin.buildModelFromUrdf(G1_URDF, pin.JointModelFreeFlyer())
    pin_data = pin_model.createData()
    link_meshes = parse_urdf_meshes(G1_URDF)
    mesh_cache = preload_meshes(link_meshes, MESH_DIR, skip_set=skip_set,
                                subsample=args.mesh_subsample)
    print(f"Loaded {len(mesh_cache)} link meshes")

    # Load frame data
    import pandas as pd
    data_dir = os.path.join(DATASET_ROOT, TASK_NAME)
    meta = pd.read_parquet(os.path.join(data_dir, "meta", "episodes",
                                         "chunk-000", "file-000.parquet"))
    ep_meta = meta[meta["episode_index"] == EPISODE].iloc[0]

    vid_col_prefix = f"videos/{CAMERA_NAME}"
    file_idx = int(ep_meta[f"{vid_col_prefix}/file_index"])
    from_ts = float(ep_meta[f"{vid_col_prefix}/from_timestamp"])
    video_path = os.path.join(data_dir, "videos", CAMERA_NAME,
                               "chunk-000", f"file-{file_idx:03d}.mp4")

    data_fi = int(ep_meta.get("data/file_index", 0))
    parquet_path = os.path.join(data_dir, "data", "chunk-000",
                                 f"file-{data_fi:03d}.parquet")
    df = pd.read_parquet(parquet_path)
    ep_df = df[df["episode_index"] == EPISODE].sort_values("frame_index")

    row = ep_df[ep_df["frame_index"] == FRAME_IDX]
    if len(row) == 0:
        raise ValueError(f"frame_index={FRAME_IDX} not found in episode {EPISODE}")
    row = row.iloc[0]
    rq = np.array(row["observation.state.robot_q_current"], dtype=np.float64)
    hs = np.array(row["observation.state.hand_state"], dtype=np.float64)
    print(f"Task: {TASK_NAME}, ep={EPISODE}, frame={FRAME_IDX}")
    print(f"Camera: {CAMERA_NAME}, video: {os.path.basename(video_path)}")

    # FK (baseline)
    q_base = build_q(pin_model, rq, hs, hand_type=hand_type)
    transforms = do_fk(pin_model, pin_data, q_base)

    # Extract video frame
    img = extract_video_frame(video_path, from_ts, FRAME_IDX)
    if img is None:
        print("Warning: could not extract video frame, visualization will be blank")
        img = np.zeros((H, W, 3), dtype=np.uint8)

    # Load GT mask
    gt_mask = load_gt_mask(GT_MASK_PATH)
    gt_pixels = np.count_nonzero(gt_mask)
    print(f"GT mask: {gt_pixels} robot pixels ({gt_pixels/(H*W)*100:.1f}%)")

    # Precompute world verts (used when joint_dof=0)
    link_verts = precompute_world_verts(mesh_cache, transforms)
    ref_t, ref_R = transforms["torso_link"]
    print(f"Precomputed world verts for {len(link_verts)} links")

    # Set up global state for evaluate()
    _G.update({
        "n_cam": n_cam,
        "cam_names": cam_names,
        "joint_dof_on": joint_dof_on,
        "joint_reg": args.joint_reg,
        "q_base": q_base,
        "pin_model": pin_model,
        "pin_data": pin_data,
        "mesh_cache": mesh_cache,
        "link_verts": link_verts,
        "ref_t": ref_t,
        "ref_R": ref_R,
        "gt_mask": gt_mask,
        "h": H,
        "w": W,
        "cam_model": cam_model,
    })

    # Baseline evaluation
    baseline_f1 = evaluate(all_seed)
    print(f"Baseline F1 (BEST_PARAMS, zero offsets): {baseline_f1:.4f}")

    # Save baseline comparison
    out_dir = os.path.join(OUTPUT_DIR, "calibration", "mask_dice")
    os.makedirs(out_dir, exist_ok=True)

    baseline_pred = render_from_params(all_seed)
    save_comparison(img, gt_mask, baseline_pred, baseline_f1,
                    seed_params_dict, cam_names,
                    os.path.join(out_dir, "iter_baseline.png"))

    # Intermediate save callback
    def save_intermediate(it, gbest_pos, gbest_val):
        pred = render_from_params(gbest_pos)
        cam_dict = dict(zip(cam_names, gbest_pos[:n_cam].astype(float)))
        path = os.path.join(out_dir, f"iter_{it:04d}.png")
        save_comparison(img, gt_mask, pred, gbest_val, cam_dict, cam_names, path)
        # Dump full params for external use
        np.save(os.path.join(out_dir, "gbest_params.npy"), gbest_pos)

    # Set up multiprocessing pool
    import multiprocessing as mp
    n_workers = args.workers
    if n_workers == 0:
        n_workers = min(mp.cpu_count(), 128)
    pool = None
    if n_workers > 1:
        # Prepare picklable state for workers (exclude pin_model/pin_data)
        g_picklable = {k: v for k, v in _G.items()
                       if k not in ("pin_model", "pin_data")}
        pool = mp.Pool(n_workers, initializer=_worker_init,
                        initargs=(g_picklable,))
        print(f"Multiprocessing: {n_workers} workers")

    # Run PSO
    print(f"\nStarting PSO: {args.particles} particles, {args.iters} iters, {ndim} dims")
    print(f"Intermediate results: {out_dir}/iter_*.png")
    print("=" * 70)
    best_params, best_f1, log_entries = pso(
        n_particles=args.particles,
        n_iters=args.iters,
        bounds=all_bounds,
        seed_params=all_seed,
        all_param_names=all_names,
        rng_seed=args.seed,
        save_fn=save_intermediate,
        pool=pool,
    )
    if pool is not None:
        pool.close()
        pool.join()
    print("=" * 70)

    # Results -- camera params
    best_cam = dict(zip(cam_names, best_params[:n_cam].astype(float)))
    print(f"\nBaseline F1: {baseline_f1:.4f}")
    print(f"Best F1:     {best_f1:.4f}  (delta={best_f1 - baseline_f1:+.4f})")
    print(f"\nCamera params (copy to config.py):")
    print("  {")
    for n in cam_names:
        print(f'      "{n}": {best_cam[n]:.4f},')
    print("  }")

    # Bound-hit warnings for camera
    for i, n in enumerate(cam_names):
        v = best_params[i]
        if abs(v - all_bounds[i, 0]) < 1e-6 or abs(v - all_bounds[i, 1]) < 1e-6:
            print(f"  WARNING: {n}={v:.4f} at bound [{all_bounds[i,0]}, {all_bounds[i,1]}]")

    # Results -- joint offsets
    if joint_dof_on:
        offsets_rad = best_params[n_cam:]
        offsets_deg = np.degrees(offsets_rad)
        print(f"\nJoint offsets (degrees):")
        sorted_idx = np.argsort(np.abs(offsets_deg))[::-1]
        for j in sorted_idx:
            deg = offsets_deg[j]
            marker = " *" if abs(deg) > args.joint_dof - 0.1 else ""
            print(f"  {JOINT_NAMES[j]:25s} {deg:+7.2f}\u00b0{marker}")
        print(f"\n  max |offset|: {np.max(np.abs(offsets_deg)):.2f}\u00b0")
        print(f"  mean |offset|: {np.mean(np.abs(offsets_deg)):.2f}\u00b0")
        n_at_bound = np.sum(np.abs(offsets_deg) > args.joint_dof - 0.1)
        if n_at_bound > 0:
            print(f"  WARNING: {n_at_bound} joints at \u00b1{args.joint_dof}\u00b0 bound")

    # Save structured results
    results = {
        "camera_model": cam_model,
        "baseline_f1": round(float(baseline_f1), 6),
        "best_f1": round(float(best_f1), 6),
        "delta_f1": round(float(best_f1 - baseline_f1), 6),
        "camera_params": {n: round(float(best_cam[n]), 4) for n in cam_names},
        "optimization": {
            "particles": args.particles,
            "iterations": args.iters,
            "workers": n_workers,
            "seed": args.seed,
            "joint_dof_deg": args.joint_dof,
            "joint_reg": args.joint_reg,
            "mesh_subsample": args.mesh_subsample,
            "total_time_sec": round(log_entries[-1]["elapsed_sec"], 1),
        },
        "bounds_hit": [cam_names[i] for i in range(len(cam_names))
                       if abs(best_params[i] - all_bounds[i, 0]) < 1e-6
                       or abs(best_params[i] - all_bounds[i, 1]) < 1e-6],
    }
    if joint_dof_on:
        offsets_deg_arr = np.degrees(best_params[n_cam:])
        results["joint_offsets_deg"] = {
            JOINT_NAMES[j]: round(float(offsets_deg_arr[j]), 4)
            for j in range(N_JOINTS)
        }

    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results.json")

    with open(os.path.join(out_dir, "optimization_log.json"), "w") as f:
        json.dump(log_entries, f)
    print(f"Saved optimization_log.json ({len(log_entries)} entries)")

    # Save final visualization
    pred_final = render_from_params(best_params)
    save_comparison(img, gt_mask, pred_final, best_f1, best_cam, cam_names,
                    os.path.join(out_dir, "final.png"))
    print(f"\nAll results in: {out_dir}/")


if __name__ == "__main__":
    main()
