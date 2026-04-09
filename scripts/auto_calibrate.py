"""
Auto camera calibration via mask IoU optimization.
PSO (Particle Swarm Optimization) with multiprocessing.
"""

import sys
import numpy as np
import pandas as pd
import cv2
import os
import time
import xml.etree.ElementTree as ET
import multiprocessing as mp
import pinocchio as pin
from stl import mesh as stl_mesh

# Force unbuffered stdout for multiprocessing
sys.stdout.reconfigure(line_buffering=True)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "g1_wbt_task3")
URDF_PATH = os.path.join(BASE_DIR, "data", "g1_urdf", "g1_29dof_rev_1_0.urdf")
MESH_DIR = os.path.join(BASE_DIR, "data", "unitree_ros", "robots", "g1_description", "meshes")
VIDEO_PATH = os.path.join(DATA_DIR, "videos", "observation.images.head_stereo_left",
                          "chunk-000", "file-000.mp4")
PARQUET_PATH = os.path.join(DATA_DIR, "data", "chunk-000", "file-000.parquet")
OUTPUT_DIR = os.path.join(BASE_DIR, "test_results")
GT_MASK_PATH = os.path.join(OUTPUT_DIR, "mask.png")

TARGET_EP = 0
TARGET_FRAME = 276

SKIP_MESHES = {
    "head_link", "logo_link", "d435_link",
    "left_rubber_hand", "right_rubber_hand",
}
MESH_SUBSAMPLE = 1  # full resolution for accuracy


def extract_frame(video_path, frame_idx):
    import av
    container = av.open(video_path)
    stream = container.streams.video[0]
    fps = float(stream.average_rate)
    target_ts = int(frame_idx / fps / stream.time_base)
    container.seek(max(target_ts - int(2 / stream.time_base), 0), stream=stream)
    for f in container.decode(stream):
        pts_sec = float(f.pts * stream.time_base)
        fn = int(round(pts_sec * fps))
        if fn >= frame_idx:
            img = f.to_ndarray(format='bgr24')
            container.close()
            return img
    container.close()
    return None


def build_q(model, rq):
    q = pin.neutral(model)
    q[0:3] = rq[0:3]
    q[3], q[4], q[5], q[6] = rq[4], rq[5], rq[6], rq[3]
    q[7:36] = rq[7:36]
    return q


def do_fk(model, data, q):
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    transforms = {}
    for i in range(model.nframes):
        name = model.frames[i].name
        T = data.oMf[i]
        transforms[name] = (T.translation.copy(), T.rotation.copy())
    return transforms


def parse_urdf_meshes(urdf_path):
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    link_meshes = {}
    for link_elem in root.findall("link"):
        name = link_elem.get("name")
        visual = link_elem.find("visual")
        if visual is None:
            continue
        geom = visual.find("geometry")
        if geom is None:
            continue
        mesh_elem = geom.find("mesh")
        if mesh_elem is None:
            continue
        filename = mesh_elem.get("filename")
        if filename:
            link_meshes[name] = os.path.basename(filename)
    return link_meshes


def load_link_verts(link_meshes, mesh_dir, transforms):
    """Load STL meshes, transform to world frame, return per-link 3D vertices."""
    link_verts = {}  # link_name -> (N, 3) world-frame vertices
    total_verts = 0

    for link_name, filename in link_meshes.items():
        if link_name in SKIP_MESHES:
            continue
        if link_name not in transforms:
            continue
        path = os.path.join(mesh_dir, filename)
        if not os.path.exists(path):
            continue

        m = stl_mesh.Mesh.from_file(path)
        flat = m.vectors.reshape(-1, 3)
        valid = np.all(np.isfinite(flat), axis=1)
        flat = flat[valid]
        if len(flat) == 0:
            continue
        # Deduplicate vertices for speed
        flat = np.unique(flat, axis=0)

        R = transforms[link_name][1]
        t = transforms[link_name][0]
        world = (R @ flat.T).T + t
        link_verts[link_name] = world.astype(np.float64)
        total_verts += len(world)

    print(f"Mesh: {len(link_verts)} links, {total_verts} unique vertices")
    return link_verts


def project_and_mask(link_verts, _unused, params, ref_t, ref_R, h, w):
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
        # Depth filter
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
    """Compute F1 score (Dice coefficient) between two binary masks."""
    inter = np.count_nonzero(mask_a & mask_b)
    sum_ab = np.count_nonzero(mask_a) + np.count_nonzero(mask_b)
    return 2 * inter / sum_ab if sum_ab > 0 else 0.0


def load_gt_mask(mask_path, h, w):
    """Load user mask. Transparent (alpha=0) = robot body."""
    img = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot load {mask_path}")

    print(f"Mask image shape: {img.shape}, dtype: {img.dtype}")

    if img.ndim == 3 and img.shape[2] == 4:
        alpha = img[:, :, 3]
        mask = alpha < 128
    elif img.ndim == 3 and img.shape[2] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask = gray > 200
    else:
        mask = img > 128

    if mask.shape[0] != h or mask.shape[1] != w:
        mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0

    print(f"GT mask: {np.count_nonzero(mask)} pixels ({100*np.count_nonzero(mask)/(h*w):.1f}%)")
    return mask


def save_comparison(img, gt_mask, pred_mask, iou, params, out_path):
    """Save 2x2 comparison: GT mask, pred mask, original, overlap."""
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

    cv2.putText(gt_vis, "GT Mask (red)", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(pred_vis, "Pred Mask (blue)", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(overlap_vis, f"IoU={iou:.4f}  green=overlap red=GTonly blue=predOnly", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

    dx, dy, dz, pitch, yaw, roll, fx, fy, cx, cy, k1, k2, k3, k4 = params
    params_text = (f"p={pitch:.0f} y={yaw:.1f} r={roll:.1f} "
                   f"fx={fx:.0f} fy={fy:.0f} cx={cx:.0f} cy={cy:.0f} "
                   f"k=[{k1:.2f},{k2:.2f},{k3:.2f},{k4:.2f}]")
    cv2.putText(overlap_vis, params_text, (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 255, 255), 1)

    top = np.hstack([gt_vis, pred_vis])
    bottom = np.hstack([img, overlap_vis])
    result = np.vstack([top, bottom])
    cv2.imwrite(out_path, result)
    print(f"Saved: {out_path}")


# ============================================================
# PSO with multiprocessing
# ============================================================

# Worker process globals (set by initializer, avoid pickling large arrays)
_W = {}


def _worker_init(frames_data):
    """frames_data: list of (link_verts, ref_t, ref_R, gt_mask, h, w)"""
    _W['frames'] = frames_data


def _eval_particle(params):
    """Evaluate one particle on ALL frames. Returns -mean_IoU."""
    total_iou = 0.0
    with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
        for link_verts, ref_t, ref_R, gt_mask, h, w in _W['frames']:
            pred = project_and_mask(link_verts, None, params, ref_t, ref_R, h, w)
            total_iou += compute_iou(gt_mask, pred)
    return -total_iou / len(_W['frames'])


# Parameter bounds — fisheye model, 14 dims
# (dx, dy, dz, pitch, yaw, roll, fx, fy, cx, cy, k1, k2, k3, k4)
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
    [-2.0, 2.0],      # k1 (fisheye radial)
    [-5.0, 5.0],      # k2 (fisheye radial)
    [-5.0, 5.0],      # k3 (fisheye radial)
    [-5.0, 5.0],      # k4 (fisheye radial)
], dtype=np.float64)

PARAM_NAMES = ["dx", "dy", "dz", "pitch", "yaw", "roll",
               "fx", "fy", "cx", "cy", "k1", "k2", "k3", "k4"]

# Previous best from single-frame Run1 (IoU=0.8970)
PREV_BEST = np.array([0.039, 0.052, 0.536, -53.6, 4.7, 3.0,
                       315, 302, 334, 230,
                       0.63, 0.17, 1.19, 0.25])


def pso_run(n_particles, n_iters, rng_seed, seed_params=None,
            pool=None, run_id=0, frames_data=None, frames_vis=None):
    """Single PSO run with Clerc-Kennedy constriction factor."""
    ndim = len(BOUNDS)
    lo, hi = BOUNDS[:, 0], BOUNDS[:, 1]
    span = hi - lo

    rng = np.random.default_rng(rng_seed)

    # Initialize positions
    positions = rng.uniform(lo, hi, (n_particles, ndim))

    if seed_params is not None:
        # 30% particles seeded around known good solution with large perturbation
        n_seed = max(n_particles * 3 // 10, 1)
        positions[0] = np.clip(seed_params, lo, hi)
        for i in range(1, n_seed):
            perturbation = rng.normal(0, span * 0.10, ndim)
            positions[i] = np.clip(seed_params + perturbation, lo, hi)

    # Clerc-Kennedy constriction: phi=4.1, chi≈0.7298
    phi = 4.1
    chi = 2.0 / abs(2.0 - phi - np.sqrt(phi * phi - 4.0 * phi))
    c1 = chi * 2.05   # ≈ 1.4962
    c2 = chi * 2.05   # ≈ 1.4962

    # Velocity clamping
    v_max = span * 0.5

    velocities = rng.uniform(-v_max * 0.1, v_max * 0.1, (n_particles, ndim))

    pbest_pos = positions.copy()
    pbest_val = np.full(n_particles, np.inf)
    gbest_pos = positions[0].copy()
    gbest_val = np.inf

    seeded_str = "seeded" if seed_params is not None else "random"
    print(f"\n[Run {run_id}] PSO ({seeded_str}): {n_particles} particles, "
          f"{n_iters} iters, seed={rng_seed}")

    t0 = time.time()
    stall_count = 0
    prev_best = np.inf

    for it in range(n_iters):
        param_list = [tuple(positions[i]) for i in range(n_particles)]
        scores = pool.map(_eval_particle, param_list)
        scores = np.array(scores)

        # Update personal and global bests
        improved = scores < pbest_val
        pbest_val[improved] = scores[improved]
        pbest_pos[improved] = positions[improved]

        it_best_idx = np.argmin(scores)
        if scores[it_best_idx] < gbest_val:
            gbest_val = scores[it_best_idx]
            gbest_pos = positions[it_best_idx].copy()

        # Stall detection: randomize worst 20% if no improvement for 50 iters
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

        # Constriction factor velocity update
        r1 = rng.random((n_particles, ndim))
        r2 = rng.random((n_particles, ndim))
        velocities = chi * (velocities
                            + c1 * r1 * (pbest_pos - positions)
                            + c2 * r2 * (gbest_pos - positions))

        # Clamp velocity
        velocities = np.clip(velocities, -v_max, v_max)
        positions = np.clip(positions + velocities, lo, hi)

        # Progress every 25 iters
        if (it + 1) % 25 == 0 or it == 0:
            elapsed = time.time() - t0
            iou = -gbest_val
            p = gbest_pos
            print(f"  [{run_id}] iter {it+1:3d}/{n_iters}  IoU={iou:.4f}  "
                  f"dx={p[0]:.3f} dy={p[1]:.3f} dz={p[2]:.3f} "
                  f"p={p[3]:.1f} y={p[4]:.1f} r={p[5]:.1f} "
                  f"fx={p[6]:.0f} fy={p[7]:.0f} cx={p[8]:.0f} cy={p[9]:.0f} "
                  f"k=[{p[10]:.2f},{p[11]:.2f},{p[12]:.2f},{p[13]:.2f}]  "
                  f"({elapsed:.0f}s)")

            # Per-frame breakdown + save comparison images
            if frames_data is not None and frames_vis is not None:
                per_frame = []
                for fi, (lv, rt, rR, gm, fh, fw) in enumerate(frames_data):
                    pred = project_and_mask(lv, None, tuple(gbest_pos), rt, rR, fh, fw)
                    frame_iou = compute_iou(gm, pred)
                    per_frame.append(frame_iou)
                    img_vis, gt_vis, name = frames_vis[fi]
                    save_comparison(img_vis, gt_vis, pred, frame_iou, tuple(gbest_pos),
                                    os.path.join(OUTPUT_DIR, f"iter{it+1:03d}_{name}.png"))
                pf_str = " | ".join(f"{frames_vis[i][2].split('_ep')[0]}={v:.3f}"
                                     for i, v in enumerate(per_frame))
                print(f"         per-frame: {pf_str}")

    elapsed = time.time() - t0
    iou = -gbest_val
    print(f"  [{run_id}] DONE  IoU={iou:.4f} in {elapsed:.0f}s")
    return gbest_pos, iou


# Multi-frame calibration targets
# Each: (task_dir, episode, frame, mask_file)
CALIB_FRAMES = [
    ("g1_wbt_task3", 0,   276, "g1_wbt_task3_ep000_f0276.png"),
    ("g1_wbt",       0,   287, "g1_wbt_ep005_f0287.png"),  # mask drawn on ep0 f287 (old bug)
]
MASK_DIR = os.path.join(BASE_DIR, "test_results", "mask")


def load_calib_frames(model, data, link_meshes):
    """Load all calibration frames: video frame + joint state + GT mask."""
    frames_data = []  # for worker: (link_verts, ref_t, ref_R, gt_mask, h, w)
    frames_vis = []   # for visualization: (img, gt_mask, name)

    for task_dir, ep, fi, mask_file in CALIB_FRAMES:
        data_dir = os.path.join(BASE_DIR, "data", task_dir)
        video_path = os.path.join(data_dir, "videos",
                                   "observation.images.head_stereo_left",
                                   "chunk-000", "file-000.mp4")
        parquet_path = os.path.join(data_dir, "data", "chunk-000", "file-000.parquet")
        mask_path = os.path.join(MASK_DIR, mask_file)

        df = pd.read_parquet(parquet_path)
        row = df[(df["episode_index"] == ep) & (df["frame_index"] == fi)].iloc[0]
        rq = np.array(row["observation.state.robot_q_current"], dtype=np.float64)

        q = build_q(model, rq)
        transforms = do_fk(model, data, q)
        ref_t, ref_R = transforms["torso_link"]

        video_frame_idx = int(row["index"])  # global video frame, not per-episode
        img = extract_frame(video_path, video_frame_idx)
        h, w = img.shape[:2]

        link_verts = load_link_verts(link_meshes, MESH_DIR, transforms)
        gt_mask = load_gt_mask(mask_path, h, w)

        name = f"{task_dir}_ep{ep:03d}_f{fi:04d}"
        frames_data.append((link_verts, ref_t, ref_R, gt_mask, h, w))
        frames_vis.append((img, gt_mask, name))
        print(f"  Loaded: {name}")

    return frames_data, frames_vis


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    import warnings
    warnings.filterwarnings('ignore', category=RuntimeWarning)

    print("Loading URDF...")
    model = pin.buildModelFromUrdf(URDF_PATH, pin.JointModelFreeFlyer())
    data = model.createData()
    link_meshes = parse_urdf_meshes(URDF_PATH)

    print(f"\nLoading {len(CALIB_FRAMES)} calibration frames...")
    frames_data, frames_vis = load_calib_frames(model, data, link_meshes)

    h, w = 480, 640

    # Shared worker pool — pass all frames to each worker
    n_workers = min(mp.cpu_count(), 10)
    pool = mp.Pool(n_workers, initializer=_worker_init,
                   initargs=(frames_data,))

    # Single random PSO run
    N_PARTICLES = 150
    N_ITERS = 300

    print(f"\n{'='*60}")
    print(f"Multi-frame PSO: {len(CALIB_FRAMES)} frames, "
          f"{N_PARTICLES} particles x {N_ITERS} iters")
    print(f"{'='*60}")

    best_params, best_iou = pso_run(
        n_particles=N_PARTICLES, n_iters=N_ITERS,
        rng_seed=42, seed_params=PREV_BEST, pool=pool, run_id=0,
        frames_data=frames_data, frames_vis=frames_vis
    )

    pool.close()
    pool.join()

    # Per-frame IoU breakdown
    print(f"\n{'='*60}")
    print(f"PER-FRAME IoU BREAKDOWN:")
    for i, (link_verts, ref_t, ref_R, gt_mask, fh, fw) in enumerate(frames_data):
        pred = project_and_mask(link_verts, None, tuple(best_params),
                                ref_t, ref_R, fh, fw)
        iou = compute_iou(gt_mask, pred)
        name = frames_vis[i][2]
        print(f"  {name}: IoU={iou:.4f}")

        # Save comparison
        img, gt, _ = frames_vis[i]
        save_comparison(img, gt, pred, iou, tuple(best_params),
                        os.path.join(OUTPUT_DIR, f"calib_{name}.png"))

    dx, dy, dz, pitch, yaw, roll, fx, fy, cx, cy, k1, k2, k3, k4 = best_params
    print(f"\nOPTIMAL CAMERA PARAMETERS (fisheye, mean IoU={best_iou:.4f}):")
    print(f"  dx    = {dx:.4f} m")
    print(f"  dy    = {dy:.4f} m")
    print(f"  dz    = {dz:.4f} m")
    print(f"  pitch = {pitch:.2f} deg")
    print(f"  yaw   = {yaw:.2f} deg")
    print(f"  roll  = {roll:.2f} deg")
    print(f"  fx    = {fx:.1f} px")
    print(f"  fy    = {fy:.1f} px")
    print(f"  cx    = {cx:.1f} px")
    print(f"  cy    = {cy:.1f} px")
    print(f"  k1    = {k1:.4f}")
    print(f"  k2    = {k2:.4f}")
    print(f"  k3    = {k3:.4f}")
    print(f"  k4    = {k4:.4f}")

    # Check bound hits
    lo, hi = BOUNDS[:, 0], BOUNDS[:, 1]
    for i, name in enumerate(PARAM_NAMES):
        if abs(best_params[i] - lo[i]) < 1e-6:
            print(f"  WARNING: {name} hit LOWER bound ({lo[i]})")
        if abs(best_params[i] - hi[i]) < 1e-6:
            print(f"  WARNING: {name} hit UPPER bound ({hi[i]})")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
