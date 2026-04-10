"""
Camera calibration v2: PSO optimization using SAM2 masks as ground truth.

Uses the new URDF with hands (60 DOF) and shared utilities from video_inpaint.py.
Supports multi-task, multi-episode calibration frames defined in a JSON manifest.

Usage:
  python scripts/auto_calibrate_v2.py --manifest data/calib_frames.json
  python scripts/auto_calibrate_v2.py --manifest data/calib_frames.json --particles 200 --iters 400
"""

import sys
import os
import json
import time
import argparse
import numpy as np
import cv2
import multiprocessing as mp
import pinocchio as pin

sys.stdout.reconfigure(line_buffering=True)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "scripts"))

from config import (G1_URDF, MESH_DIR, BEST_PARAMS, SKIP_MESHES,
                     DATASET_ROOT, OUTPUT_DIR, get_hand_type)
from video_inpaint import (build_q, do_fk, parse_urdf_meshes, preload_meshes,
                            load_episode_info)

CALIB_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "calib_v2")
INPAINT_V2_DIR = os.path.join(BASE_DIR, "data", "output", "inpaint_v2")


# ============================================================
# Mesh & projection helpers
# ============================================================

def transform_link_verts(mesh_cache, transforms, skip_set):
    """Transform preloaded unique vertices to world frame per link."""
    link_verts = {}
    for link_name, (tris, unique_verts) in mesh_cache.items():
        if link_name in skip_set or link_name not in transforms:
            continue
        if len(unique_verts) == 0:
            continue
        t, R = transforms[link_name]
        world = (R @ unique_verts.T).T + t
        link_verts[link_name] = world.astype(np.float64)
    return link_verts


def project_and_mask(link_verts, params, ref_t, ref_R, h, w):
    """Project per-link convex hulls to 2D using fisheye model."""
    dx, dy, dz, pitch_deg, yaw_deg, roll_deg, fx, fy, cx, cy, k1, k2, k3, k4 = params
    pitch = np.radians(pitch_deg)
    yaw = np.radians(yaw_deg)
    roll = np.radians(roll_deg)

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

    cam_pos = ref_t + ref_R @ np.array([dx, dy, dz])
    R_w2c = (ref_R @ R_cam.T).T
    t_w2c = R_w2c @ (-cam_pos)

    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    D = np.array([k1, k2, k3, k4], dtype=np.float64).reshape(4, 1)
    rvec, _ = cv2.Rodrigues(R_w2c)
    tvec = t_w2c.reshape(3, 1)

    mask = np.zeros((h, w), dtype=np.uint8)

    for verts3d in link_verts.values():
        depths = (R_w2c @ verts3d.T).T + t_w2c.flatten()
        z_cam = depths[:, 2]
        in_front = z_cam > 0.01
        if np.count_nonzero(in_front) < 3:
            continue

        pts3d_valid = verts3d[in_front].reshape(-1, 1, 3).astype(np.float64)
        pts2d, _ = cv2.fisheye.projectPoints(pts3d_valid, rvec, tvec, K, D)
        pts2d = pts2d.reshape(-1, 2)

        finite = np.all(np.isfinite(pts2d), axis=1)
        pts2d = pts2d[finite]
        if len(pts2d) < 3:
            continue

        hull = cv2.convexHull(pts2d.astype(np.float32))
        cv2.fillConvexPoly(mask, hull.astype(np.int32), 255)

    return mask > 0


def compute_iou(mask_a, mask_b):
    """Dice coefficient (F1) between two binary masks."""
    inter = np.count_nonzero(mask_a & mask_b)
    sum_ab = np.count_nonzero(mask_a) + np.count_nonzero(mask_b)
    return 2 * inter / sum_ab if sum_ab > 0 else 0.0


# ============================================================
# Visualization
# ============================================================

def save_comparison(img, gt_mask, pred_mask, iou, params, out_path):
    """Save 2x2 comparison: GT, pred, original, overlap."""
    h, w = img.shape[:2]

    gt_vis = img.copy()
    gt_vis[gt_mask] = (gt_vis[gt_mask] * 0.4 + np.array([0, 0, 200]) * 0.6).astype(np.uint8)

    pred_vis = img.copy()
    pred_vis[pred_mask] = (pred_vis[pred_mask] * 0.4 + np.array([200, 0, 0]) * 0.6).astype(np.uint8)

    overlap_vis = img.copy()
    both = gt_mask & pred_mask
    gt_only = gt_mask & ~pred_mask
    pred_only = pred_mask & ~gt_mask
    overlap_vis[both] = (overlap_vis[both] * 0.3 + np.array([0, 200, 0]) * 0.7).astype(np.uint8)
    overlap_vis[gt_only] = (overlap_vis[gt_only] * 0.3 + np.array([0, 0, 200]) * 0.7).astype(np.uint8)
    overlap_vis[pred_only] = (overlap_vis[pred_only] * 0.3 + np.array([200, 0, 0]) * 0.7).astype(np.uint8)

    cv2.putText(gt_vis, "GT (red=SAM2)", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(pred_vis, "Pred (blue=FK)", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    dx, dy, dz, pitch, yaw, roll, fx, fy, cx, cy, k1, k2, k3, k4 = params
    params_text = (f"p={pitch:.0f} y={yaw:.1f} r={roll:.1f} "
                   f"fx={fx:.0f} fy={fy:.0f} cx={cx:.0f} cy={cy:.0f} "
                   f"k=[{k1:.2f},{k2:.2f},{k3:.2f},{k4:.2f}]")
    cv2.putText(overlap_vis, f"IoU={iou:.4f}  green=both red=GT blue=pred", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
    cv2.putText(overlap_vis, params_text, (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 255, 255), 1)

    top = np.hstack([gt_vis, pred_vis])
    bottom = np.hstack([img, overlap_vis])
    result = np.vstack([top, bottom])
    cv2.imwrite(out_path, result)


# ============================================================
# PSO with multiprocessing
# ============================================================

_W = {}


def _worker_init(frames_data):
    _W['frames'] = frames_data


def _eval_particle(params):
    """Evaluate one particle on ALL frames. Returns -mean_IoU."""
    total_iou = 0.0
    with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
        for link_verts, ref_t, ref_R, gt_mask, h, w in _W['frames']:
            pred = project_and_mask(link_verts, params, ref_t, ref_R, h, w)
            total_iou += compute_iou(gt_mask, pred)
    return -total_iou / len(_W['frames'])


# 14 parameters: dx, dy, dz, pitch, yaw, roll, fx, fy, cx, cy, k1-k4
BOUNDS = np.array([
    [-0.05, 0.20],    # dx (m)
    [-0.10, 0.15],    # dy (m)
    [0.15, 0.70],     # dz (m)
    [-80, -10],       # pitch (deg)
    [-15, 15],        # yaw (deg)
    [-10, 10],        # roll (deg)
    [100, 800],       # fx (px)
    [100, 800],       # fy (px)
    [280, 360],       # cx (px)
    [200, 280],       # cy (px)
    [-2.0, 2.0],      # k1
    [-5.0, 5.0],      # k2
    [-5.0, 5.0],      # k3
    [-5.0, 5.0],      # k4
], dtype=np.float64)

PARAM_NAMES = ["dx", "dy", "dz", "pitch", "yaw", "roll",
               "fx", "fy", "cx", "cy", "k1", "k2", "k3", "k4"]


def params_dict_to_array(d):
    """Convert BEST_PARAMS dict to 14-element array."""
    return np.array([d[n] for n in PARAM_NAMES], dtype=np.float64)


def params_array_to_dict(arr):
    """Convert 14-element array to dict."""
    return {n: float(arr[i]) for i, n in enumerate(PARAM_NAMES)}


def pso_run(n_particles, n_iters, rng_seed, seed_params, pool,
            frames_data, frames_vis):
    """PSO with Clerc-Kennedy constriction factor."""
    ndim = len(BOUNDS)
    lo, hi = BOUNDS[:, 0], BOUNDS[:, 1]
    span = hi - lo

    rng = np.random.default_rng(rng_seed)

    positions = rng.uniform(lo, hi, (n_particles, ndim))

    # Seed 30% around known good solution
    n_seed = max(n_particles * 3 // 10, 1)
    positions[0] = np.clip(seed_params, lo, hi)
    for i in range(1, n_seed):
        perturbation = rng.normal(0, span * 0.10, ndim)
        positions[i] = np.clip(seed_params + perturbation, lo, hi)

    # Clerc-Kennedy: phi=4.1, chi~0.7298
    phi = 4.1
    chi = 2.0 / abs(2.0 - phi - np.sqrt(phi * phi - 4.0 * phi))
    c1 = chi * 2.05
    c2 = chi * 2.05

    v_max = span * 0.5
    velocities = rng.uniform(-v_max * 0.1, v_max * 0.1, (n_particles, ndim))

    pbest_pos = positions.copy()
    pbest_val = np.full(n_particles, np.inf)
    gbest_pos = positions[0].copy()
    gbest_val = np.inf

    print(f"\nPSO: {n_particles} particles, {n_iters} iters, "
          f"{len(frames_data)} frames, seed=42")

    t0 = time.time()
    stall_count = 0
    prev_best = np.inf

    for it in range(n_iters):
        param_list = [tuple(positions[i]) for i in range(n_particles)]
        scores = np.array(pool.map(_eval_particle, param_list))

        improved = scores < pbest_val
        pbest_val[improved] = scores[improved]
        pbest_pos[improved] = positions[improved]

        it_best_idx = np.argmin(scores)
        if scores[it_best_idx] < gbest_val:
            gbest_val = scores[it_best_idx]
            gbest_pos = positions[it_best_idx].copy()

        # Stall detection
        if gbest_val < prev_best - 1e-6:
            prev_best = gbest_val
            stall_count = 0
        else:
            stall_count += 1

        if stall_count >= 50:
            stall_count = 0
            n_reset = n_particles // 5
            worst_idx = np.argsort(pbest_val)[-n_reset:]
            positions[worst_idx] = rng.uniform(lo, hi, (n_reset, ndim))
            velocities[worst_idx] = rng.uniform(-v_max * 0.1, v_max * 0.1,
                                                 (n_reset, ndim))
            pbest_val[worst_idx] = np.inf
            pbest_pos[worst_idx] = positions[worst_idx]
            print(f"  [stall reset] randomized {n_reset} particles")

        # Velocity update
        r1 = rng.random((n_particles, ndim))
        r2 = rng.random((n_particles, ndim))
        velocities = chi * (velocities
                            + c1 * r1 * (pbest_pos - positions)
                            + c2 * r2 * (gbest_pos - positions))
        velocities = np.clip(velocities, -v_max, v_max)
        positions = np.clip(positions + velocities, lo, hi)

        # Progress report
        if (it + 1) % 20 == 0 or it == 0:
            elapsed = time.time() - t0
            iou = -gbest_val
            p = gbest_pos
            print(f"  iter {it+1:3d}/{n_iters}  IoU={iou:.4f}  "
                  f"dx={p[0]:.3f} dy={p[1]:.3f} dz={p[2]:.3f} "
                  f"p={p[3]:.1f} y={p[4]:.1f} r={p[5]:.1f} "
                  f"fx={p[6]:.0f} fy={p[7]:.0f} cx={p[8]:.0f} cy={p[9]:.0f} "
                  f"k=[{p[10]:.2f},{p[11]:.2f},{p[12]:.2f},{p[13]:.2f}]  "
                  f"({elapsed:.0f}s)")

            # Save sample comparison images every 50 iters
            # Pick 4 evenly spaced frames as samples
            if frames_vis is not None and ((it + 1) % 20 == 0 or it == 0):
                n_total = len(frames_data)
                sample_ids = [0, n_total // 3, 2 * n_total // 3, n_total - 1]
                sample_ids = sorted(set(min(s, n_total - 1) for s in sample_ids))
                for si in sample_ids:
                    lv, rt, rR, gm, fh, fw = frames_data[si]
                    pred = project_and_mask(lv, tuple(gbest_pos), rt, rR, fh, fw)
                    frame_iou = compute_iou(gm, pred)
                    img_vis, gt_vis, label = frames_vis[si]
                    save_comparison(img_vis, gt_vis, pred, frame_iou,
                                    tuple(gbest_pos),
                                    os.path.join(CALIB_OUTPUT_DIR,
                                                 f"iter{it+1:03d}_{label}.png"))
                print(f"         saved {len(sample_ids)} sample images to {CALIB_OUTPUT_DIR}/")

    elapsed = time.time() - t0
    iou = -gbest_val
    print(f"\nDONE  mean IoU={iou:.4f} in {elapsed:.0f}s")
    return gbest_pos, iou


# ============================================================
# Data loading
# ============================================================

def extract_video_frame(video_path, from_ts, frame_idx):
    """Extract a single frame from episode video."""
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
            img = f.to_ndarray(format='bgr24')
            container.close()
            return img

    container.close()
    return None


def load_calib_frames(manifest_path, model, data_pin, mesh_cache):
    """Load all calibration frames from manifest JSON."""
    with open(manifest_path) as f:
        manifest = json.load(f)

    frames_data = []  # (link_verts, ref_t, ref_R, gt_mask, h, w)
    frames_vis = []   # (img, gt_mask, label)

    for entry in manifest["frames"]:
        source_dir = entry["source_dir"]
        task = entry["task"]
        episode = entry["episode"]
        frame_indices = entry["frame_indices"]

        if not frame_indices:
            continue

        data_dir = os.path.join(DATASET_ROOT, task)
        mask_dir = os.path.join(INPAINT_V2_DIR, source_dir, "extracted", "mask")

        print(f"\nLoading {source_dir}: {len(frame_indices)} frames")
        print(f"  task={task}, ep={episode}")

        video_path, from_ts, to_ts, ep_df = load_episode_info(episode, data_dir=data_dir)

        # Build frame_index -> row lookup
        fi_to_row = {}
        for _, row in ep_df.iterrows():
            fi_to_row[int(row["frame_index"])] = row

        for fi in frame_indices:
            if fi not in fi_to_row:
                print(f"  WARNING: frame {fi} not in parquet, skipping")
                continue

            row = fi_to_row[fi]
            rq = np.array(row["observation.state.robot_q_current"], dtype=np.float64)
            hs = np.array(row["observation.state.hand_state"], dtype=np.float64)

            q = build_q(model, rq, hs, hand_type=get_hand_type())
            transforms = do_fk(model, data_pin, q)
            ref_t, ref_R = transforms["torso_link"]

            link_verts = transform_link_verts(mesh_cache, transforms, SKIP_MESHES)

            # Load GT mask
            mask_path = os.path.join(mask_dir, f"{fi:05d}.png")
            if not os.path.exists(mask_path):
                print(f"  WARNING: mask not found: {mask_path}, skipping")
                continue
            gt_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            gt_mask = gt_img > 128
            h, w = gt_mask.shape

            # Load video frame for visualization
            img = extract_video_frame(video_path, from_ts, fi)
            if img is None:
                print(f"  WARNING: cannot extract video frame {fi}, skipping")
                continue

            label = f"{source_dir}_f{fi:04d}"
            frames_data.append((link_verts, ref_t, ref_R, gt_mask, h, w))
            frames_vis.append((img, gt_mask, label))
            total_verts = sum(len(v) for v in link_verts.values())
            print(f"  Loaded: {label} ({len(link_verts)} links, {total_verts} verts, "
                  f"mask {np.count_nonzero(gt_mask)/(h*w)*100:.1f}%)")

    return frames_data, frames_vis


# ============================================================
# Main
# ============================================================

def main():
    import warnings
    warnings.filterwarnings('ignore', category=RuntimeWarning)

    parser = argparse.ArgumentParser(description="Camera calibration v2 (SAM2 + PSO)")
    parser.add_argument("--manifest", type=str,
                        default=os.path.join(BASE_DIR, "data", "calib_frames.json"))
    parser.add_argument("--particles", type=int, default=150)
    parser.add_argument("--iters", type=int, default=300)
    parser.add_argument("--mesh-subsample", type=int, default=2)
    parser.add_argument("--workers", type=int, default=0,
                        help="Number of workers (0=auto)")
    args = parser.parse_args()

    os.makedirs(CALIB_OUTPUT_DIR, exist_ok=True)

    # Load URDF + meshes
    print("Loading URDF...")
    model = pin.buildModelFromUrdf(G1_URDF, pin.JointModelFreeFlyer())
    data_pin = model.createData()
    link_meshes = parse_urdf_meshes(G1_URDF)
    mesh_cache = preload_meshes(link_meshes, MESH_DIR,
                                 skip_set=SKIP_MESHES,
                                 subsample=args.mesh_subsample)
    print(f"Meshes: {len(mesh_cache)} links loaded (subsample={args.mesh_subsample})")

    # Load calibration frames
    print(f"\nLoading calibration frames from {args.manifest}")
    frames_data, frames_vis = load_calib_frames(
        args.manifest, model, data_pin, mesh_cache)

    if not frames_data:
        print("ERROR: No calibration frames loaded. Check manifest and extracted masks.")
        sys.exit(1)

    print(f"\nTotal: {len(frames_data)} calibration frames")

    # Seed from current BEST_PARAMS
    seed_params = params_dict_to_array(BEST_PARAMS)

    # Start PSO
    n_workers = args.workers if args.workers > 0 else min(mp.cpu_count(), 16)
    pool = mp.Pool(n_workers, initializer=_worker_init, initargs=(frames_data,))

    print(f"\n{'='*60}")
    print(f"PSO v2: {len(frames_data)} frames, "
          f"{args.particles} particles x {args.iters} iters, "
          f"{n_workers} workers")
    print(f"{'='*60}")

    best_params, best_iou = pso_run(
        n_particles=args.particles,
        n_iters=args.iters,
        rng_seed=42,
        seed_params=seed_params,
        pool=pool,
        frames_data=frames_data,
        frames_vis=frames_vis,
    )

    pool.close()
    pool.join()

    # Per-frame breakdown + comparison images
    print(f"\n{'='*60}")
    print("PER-FRAME IoU BREAKDOWN:")
    for i, (link_verts, ref_t, ref_R, gt_mask, fh, fw) in enumerate(frames_data):
        pred = project_and_mask(link_verts, tuple(best_params), ref_t, ref_R, fh, fw)
        iou = compute_iou(gt_mask, pred)
        label = frames_vis[i][2]
        print(f"  {label}: IoU={iou:.4f}")

        img, gt, _ = frames_vis[i]
        save_comparison(img, gt, pred, iou, tuple(best_params),
                        os.path.join(CALIB_OUTPUT_DIR, f"calib_{label}.png"))

    # Print optimal parameters
    best_dict = params_array_to_dict(best_params)
    print(f"\nOPTIMAL CAMERA PARAMETERS (mean IoU={best_iou:.4f}):")
    for name in PARAM_NAMES:
        v = best_dict[name]
        unit = "m" if name.startswith("d") else ("deg" if name in ("pitch", "yaw", "roll") else "px")
        if name.startswith("k"):
            unit = ""
        print(f"  {name:6s} = {v:.4f} {unit}")

    # Save as JSON
    json_path = os.path.join(CALIB_OUTPUT_DIR, "best_params.json")
    with open(json_path, 'w') as f:
        json.dump({"mean_iou": best_iou, "params": best_dict}, f, indent=2)
    print(f"\nSaved: {json_path}")

    # Print config.py snippet
    print(f"\n# Copy to config.py BEST_PARAMS:")
    print("BEST_PARAMS = {")
    for name in PARAM_NAMES:
        v = best_dict[name]
        if name in ("fx", "fy", "cx", "cy"):
            print(f'    "{name}": {v:.0f},')
        elif name.startswith("d"):
            print(f'    "{name}": {v:.3f},')
        elif name in ("pitch", "yaw", "roll"):
            print(f'    "{name}": {v:.1f},')
        else:
            print(f'    "{name}": {v:.2f},')
    print("}")

    # Bound warnings
    lo, hi = BOUNDS[:, 0], BOUNDS[:, 1]
    for i, name in enumerate(PARAM_NAMES):
        if abs(best_params[i] - lo[i]) < 1e-6:
            print(f"  WARNING: {name} hit LOWER bound ({lo[i]})")
        if abs(best_params[i] - hi[i]) < 1e-6:
            print(f"  WARNING: {name} hit UPPER bound ({hi[i]})")

    print(f"\n{'='*60}")
    print(f"Comparison images saved to: {CALIB_OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
