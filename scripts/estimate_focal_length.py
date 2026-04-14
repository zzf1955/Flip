"""
Focal length cross-validation from annotated keypoints.

Method: Given known 3D FK positions and user-annotated 2D positions,
estimate fx/fy analytically using the pinhole projection equation:
    u = fx * X_cam / Z_cam + cx
    v = fy * Y_cam / Z_cam + cy

With known extrinsics (from URDF d435 position + calibrated offset),
we can compute X_cam, Y_cam, Z_cam for each keypoint, then solve for fx/fy.

Also provides a multi-model comparison of all PSO calibration results.

Usage:
  python scripts/estimate_focal_length.py
"""

import sys
import os
import json
import numpy as np
import cv2
import pandas as pd
import pinocchio as pin
from scipy import ndimage

sys.stdout.reconfigure(line_buffering=True)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "scripts"))

from config import (G1_URDF, MESH_DIR, BEST_PARAMS, SKIP_MESHES,
                     DATASET_ROOT, OUTPUT_DIR, get_hand_type, get_skip_meshes,
                     CAMERA_MODEL)
from camera_models import get_model, build_K, build_D, model_is_fisheye
from video_inpaint import (build_q, do_fk, parse_urdf_meshes, make_camera_const)

# Keypoints definition (same as optimize_keypoints.py)
ALL_KEYPOINTS = [
    ("L_wrist", "left_wrist_yaw_link",   np.array([ 0.0046,  0.0000,  0.0300])),
    ("L_thumb", "left_thumb_4",          np.array([-0.0314,  0.0150, -0.0101])),
    ("L_toe",   "left_ankle_roll_link",  np.array([ 0.1424,  0.0000, -0.0210])),
    ("R_toe",   "right_ankle_roll_link", np.array([ 0.1424, -0.0000, -0.0215])),
    ("R_thumb", "right_thumb_4",         np.array([ 0.0314,  0.0150, -0.0101])),
    ("R_wrist", "right_wrist_yaw_link",  np.array([ 0.0046,  0.0000,  0.0300])),
]
ALL_KP_NAMES = [k[0] for k in ALL_KEYPOINTS]


def detect_keypoints_from_alpha(png_path, expected_count=None):
    """Detect keypoints from semi-transparent alpha markers."""
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

    # Merge clusters closer than 15px
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
            f"Expected {expected_count} keypoint clusters, "
            f"found {len(centers)} in {png_path}")

    centers.sort(key=lambda c: c[0])
    return np.array(centers, dtype=np.float64)


def load_frame_data(task, episode, frame_idx, model, data_pin):
    """Load FK transforms for a specific frame."""
    data_dir = os.path.join(DATASET_ROOT, task)

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
    return transforms, ref_t, ref_R


def get_kp_world_positions(transforms, keypoints):
    """Compute keypoint world positions from FK transforms."""
    pts = []
    for name, link_name, local_offset in keypoints:
        t_link, R_link = transforms[link_name]
        pts.append(R_link @ local_offset + t_link)
    return np.array(pts, dtype=np.float64)


def compute_camera_transform(params, ref_t, ref_R):
    """Compute R_w2c and t_w2c from params + torso transform.
    Returns cam_pos, R_w2c, t_w2c."""
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

    offset = np.array([p["dx"], p["dy"], p["dz"]], dtype=np.float64)
    cam_pos = ref_t + ref_R @ offset
    R_w2c = (ref_R @ R_cam.T).T
    t_w2c = R_w2c @ (-cam_pos)
    return cam_pos, R_w2c, t_w2c


def estimate_focal_from_point(u_gt, v_gt, X_cam, Y_cam, Z_cam, cx, cy):
    """Estimate fx and fy from a single point.
    u_gt = fx * X_cam / Z_cam + cx  =>  fx = (u_gt - cx) * Z_cam / X_cam
    v_gt = fy * Y_cam / Z_cam + cy  =>  fy = (v_gt - cy) * Z_cam / Y_cam
    """
    fx_est = None
    fy_est = None
    if abs(X_cam) > 1e-6:
        fx_est = (u_gt - cx) * Z_cam / X_cam
    if abs(Y_cam) > 1e-6:
        fy_est = (v_gt - cy) * Z_cam / Y_cam
    return fx_est, fy_est


def main():
    print("=" * 70)
    print("FOCAL LENGTH CROSS-VALIDATION")
    print("=" * 70)

    # ── Load URDF ──
    print("\nLoading URDF model...")
    model = pin.buildModelFromUrdf(G1_URDF, pin.JointModelFreeFlyer())
    data_pin = model.createData()

    # ── Load all manifests ──
    manifests = {
        "all_6frames": "data/5point/manifest.json",
        "pillow_only": "data/5point/manifest_pillow.json",
        "basket_only": "data/5point/manifest_basket.json",
        "washing_only": "data/5point/manifest_washing.json",
    }

    # ── Per-task PSO results ──
    pso_results = {
        "combined": {
            "path": "test_results/kp_optim/best_params.json",
            "params": None
        },
        "pillow": {
            "path": "test_results/kp_optim/pso_pillow/best_params.json",
            "params": None
        },
        "basket": {
            "path": "test_results/kp_optim/pso_basket/best_params.json",
            "params": None
        },
        "washing": {
            "path": "test_results/kp_optim/pso_washing/best_params.json",
            "params": None
        },
    }

    for name, info in pso_results.items():
        fpath = os.path.join(BASE_DIR, info["path"])
        if os.path.exists(fpath):
            with open(fpath) as f:
                d = json.load(f)
            info["params"] = d["params"]
            info["rmse"] = d["rmse_px"]
        else:
            print(f"  WARNING: {fpath} not found")

    # ══════════════════════════════════════════════════════════════
    # PART 1: Compare all PSO calibration results
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("PART 1: PSO Calibration Results Summary")
    print("=" * 70)

    header = f"{'Model':12s} {'RMSE(px)':>10s} {'fx':>10s} {'fy':>10s} {'cx':>10s} {'cy':>10s} {'pitch':>10s} {'dz':>10s} {'scale':>8s}"
    print(f"\n{header}")
    print("-" * len(header))

    for name, info in pso_results.items():
        if info["params"] is None:
            continue
        p = info["params"]
        fx = p.get("fx", p.get("f", "?"))
        fy = p.get("fy", p.get("f", "?"))
        sc = p.get("scale", 1.0)
        print(f"{name:12s} {info['rmse']:10.2f} {fx:10.2f} {fy:10.2f} "
              f"{p['cx']:10.2f} {p['cy']:10.2f} {p['pitch']:10.2f} {p['dz']:10.2f} {sc:8.3f}")

    # ══════════════════════════════════════════════════════════════
    # PART 2: Analytical focal length estimation
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("PART 2: Analytical Focal Length Estimation")
    print("  Method: Fix extrinsics, compute camera-frame 3D points,")
    print("  then solve f from pinhole equation per annotated keypoint.")
    print("=" * 70)

    # Use the combined best extrinsics as the reference
    ref_extrinsics = {
        "dx": BEST_PARAMS["dx"],
        "dy": BEST_PARAMS["dy"],
        "dz": BEST_PARAMS["dz"],
        "pitch": BEST_PARAMS["pitch"],
        "yaw": BEST_PARAMS["yaw"],
        "roll": BEST_PARAMS["roll"],
    }

    # Try multiple cx, cy candidates
    cx_cy_candidates = [
        ("combined", BEST_PARAMS.get("cx", 320), BEST_PARAMS.get("cy", 240)),
        ("image_center", 320.0, 240.0),
    ]

    # Load full manifest with all 6 frames
    manifest_path = os.path.join(BASE_DIR, "data/5point/manifest.json")
    with open(manifest_path) as f:
        manifest = json.load(f)

    manifest_kp_names = manifest["keypoints"]
    # Use the 5 keypoints from manifest
    active_kps = [kp for kp in ALL_KEYPOINTS if kp[0] in manifest_kp_names]
    n_kp = len(active_kps)

    print(f"\nUsing {n_kp} keypoints: {[k[0] for k in active_kps]}")
    print(f"Reference extrinsics: dx={ref_extrinsics['dx']:.4f} dy={ref_extrinsics['dy']:.4f} "
          f"dz={ref_extrinsics['dz']:.4f}")
    print(f"  pitch={ref_extrinsics['pitch']:.2f} yaw={ref_extrinsics['yaw']:.2f} "
          f"roll={ref_extrinsics['roll']:.2f}")

    manifest_dir = os.path.dirname(manifest_path)

    all_fx_estimates = []
    all_fy_estimates = []
    all_details = []

    for entry in manifest["frames"]:
        png_path = os.path.join(manifest_dir, entry["image"])
        label = os.path.splitext(entry["image"])[0]

        # Get 2D annotated positions
        gt_pts_all = detect_keypoints_from_alpha(png_path, expected_count=n_kp)
        # Map to active keypoints
        kp_indices = [manifest_kp_names.index(k[0]) for k in active_kps]
        gt_pts = gt_pts_all[kp_indices]

        # Get 3D FK positions
        transforms, ref_t, ref_R = load_frame_data(
            entry["task"], entry["episode"], entry["frame"],
            model, data_pin)
        world_pts = get_kp_world_positions(transforms, active_kps)

        # Compute camera transform with reference extrinsics
        cam_pos, R_w2c, t_w2c = compute_camera_transform(ref_extrinsics, ref_t, ref_R)

        # Transform world points to camera frame
        cam_pts = (R_w2c @ world_pts.T).T + t_w2c.flatten()

        print(f"\n--- {label} (task={entry['task']}, ep={entry['episode']}, frame={entry['frame']}) ---")
        print(f"  {'KP':10s} {'u_gt':>8s} {'v_gt':>8s} {'X_cam':>10s} {'Y_cam':>10s} {'Z_cam':>10s} {'fx_est':>10s} {'fy_est':>10s}")

        # Use combined cx, cy
        cx = BEST_PARAMS.get("cx", 320.0)
        cy = BEST_PARAMS.get("cy", 240.0)

        for i, (kp_name, _, _) in enumerate(active_kps):
            u_gt, v_gt = gt_pts[i]
            X_cam, Y_cam, Z_cam = cam_pts[i]

            fx_est, fy_est = estimate_focal_from_point(
                u_gt, v_gt, X_cam, Y_cam, Z_cam, cx, cy)

            fx_str = f"{fx_est:.2f}" if fx_est is not None else "N/A"
            fy_str = f"{fy_est:.2f}" if fy_est is not None else "N/A"

            print(f"  {kp_name:10s} {u_gt:8.1f} {v_gt:8.1f} "
                  f"{X_cam:10.4f} {Y_cam:10.4f} {Z_cam:10.4f} "
                  f"{fx_str:>10s} {fy_str:>10s}")

            if fx_est is not None and 50 < fx_est < 1000:
                all_fx_estimates.append(fx_est)
                all_details.append({
                    "frame": label, "kp": kp_name, "axis": "fx",
                    "value": fx_est, "Z_cam": Z_cam,
                    "u_gt": u_gt, "v_gt": v_gt,
                    "X_cam": X_cam, "Y_cam": Y_cam
                })
            if fy_est is not None and 50 < fy_est < 1000:
                all_fy_estimates.append(fy_est)
                all_details.append({
                    "frame": label, "kp": kp_name, "axis": "fy",
                    "value": fy_est, "Z_cam": Z_cam,
                    "u_gt": u_gt, "v_gt": v_gt,
                    "X_cam": X_cam, "Y_cam": Y_cam
                })

    # ══════════════════════════════════════════════════════════════
    # PART 3: Statistical analysis
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("PART 3: Statistical Analysis of Focal Length Estimates")
    print(f"  (using cx={cx:.2f}, cy={cy:.2f})")
    print("=" * 70)

    fx_arr = np.array(all_fx_estimates)
    fy_arr = np.array(all_fy_estimates)

    print(f"\n  fx estimates ({len(fx_arr)} points):")
    if len(fx_arr) > 0:
        print(f"    Mean:   {fx_arr.mean():.2f}")
        print(f"    Median: {np.median(fx_arr):.2f}")
        print(f"    Std:    {fx_arr.std():.2f}")
        print(f"    Min:    {fx_arr.min():.2f}")
        print(f"    Max:    {fx_arr.max():.2f}")
        # IQR-based robust estimate
        q1, q3 = np.percentile(fx_arr, [25, 75])
        iqr = q3 - q1
        mask = (fx_arr >= q1 - 1.5 * iqr) & (fx_arr <= q3 + 1.5 * iqr)
        if mask.sum() > 0:
            print(f"    Robust mean (IQR-filtered): {fx_arr[mask].mean():.2f} ({mask.sum()}/{len(fx_arr)} points)")

    print(f"\n  fy estimates ({len(fy_arr)} points):")
    if len(fy_arr) > 0:
        print(f"    Mean:   {fy_arr.mean():.2f}")
        print(f"    Median: {np.median(fy_arr):.2f}")
        print(f"    Std:    {fy_arr.std():.2f}")
        print(f"    Min:    {fy_arr.min():.2f}")
        print(f"    Max:    {fy_arr.max():.2f}")
        q1, q3 = np.percentile(fy_arr, [25, 75])
        iqr = q3 - q1
        mask = (fy_arr >= q1 - 1.5 * iqr) & (fy_arr <= q3 + 1.5 * iqr)
        if mask.sum() > 0:
            print(f"    Robust mean (IQR-filtered): {fy_arr[mask].mean():.2f} ({mask.sum()}/{len(fy_arr)} points)")

    # ══════════════════════════════════════════════════════════════
    # PART 4: Per-keypoint breakdown
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("PART 4: Per-Keypoint Focal Length Estimates")
    print("=" * 70)

    for kp_name in [k[0] for k in active_kps]:
        fx_kp = [d["value"] for d in all_details
                 if d["kp"] == kp_name and d["axis"] == "fx"]
        fy_kp = [d["value"] for d in all_details
                 if d["kp"] == kp_name and d["axis"] == "fy"]
        print(f"\n  {kp_name}:")
        if fx_kp:
            print(f"    fx: {', '.join(f'{v:.1f}' for v in fx_kp)}  "
                  f"-> mean={np.mean(fx_kp):.1f} std={np.std(fx_kp):.1f}")
        if fy_kp:
            print(f"    fy: {', '.join(f'{v:.1f}' for v in fy_kp)}  "
                  f"-> mean={np.mean(fy_kp):.1f} std={np.std(fy_kp):.1f}")

    # ══════════════════════════════════════════════════════════════
    # PART 5: Per-frame breakdown
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("PART 5: Per-Frame Focal Length Estimates")
    print("=" * 70)

    frame_labels = list(dict.fromkeys(d["frame"] for d in all_details))
    for fl in frame_labels:
        fx_frame = [d["value"] for d in all_details
                    if d["frame"] == fl and d["axis"] == "fx"]
        fy_frame = [d["value"] for d in all_details
                    if d["frame"] == fl and d["axis"] == "fy"]
        print(f"\n  {fl}:")
        if fx_frame:
            print(f"    fx: mean={np.mean(fx_frame):.1f} std={np.std(fx_frame):.1f} "
                  f"range=[{min(fx_frame):.1f}, {max(fx_frame):.1f}]")
        if fy_frame:
            print(f"    fy: mean={np.mean(fy_frame):.1f} std={np.std(fy_frame):.1f} "
                  f"range=[{min(fy_frame):.1f}, {max(fy_frame):.1f}]")

    # ══════════════════════════════════════════════════════════════
    # PART 6: Sensitivity to cx, cy
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("PART 6: Sensitivity to Principal Point (cx, cy)")
    print("=" * 70)

    # Test with different cx, cy values
    cx_cy_tests = [
        ("combined_optim", BEST_PARAMS.get("cx", 320), BEST_PARAMS.get("cy", 240)),
        ("image_center",   320.0, 240.0),
        ("pillow_optim",   342.45, 197.00),
        ("basket_optim",   402.80, 458.12),
        ("washing_optim",  291.04, 224.96),
    ]

    # Reload all data for cx/cy sensitivity test
    all_cam_pts = []
    all_gt_pts_list = []
    for entry in manifest["frames"]:
        png_path = os.path.join(manifest_dir, entry["image"])
        gt_pts_all = detect_keypoints_from_alpha(png_path, expected_count=n_kp)
        kp_indices = [manifest_kp_names.index(k[0]) for k in active_kps]
        gt_pts = gt_pts_all[kp_indices]

        transforms, ref_t, ref_R = load_frame_data(
            entry["task"], entry["episode"], entry["frame"],
            model, data_pin)
        world_pts = get_kp_world_positions(transforms, active_kps)
        cam_pos, R_w2c, t_w2c = compute_camera_transform(ref_extrinsics, ref_t, ref_R)
        cam_pts = (R_w2c @ world_pts.T).T + t_w2c.flatten()
        all_cam_pts.append(cam_pts)
        all_gt_pts_list.append(gt_pts)

    print(f"\n  {'cx,cy source':20s} {'cx':>8s} {'cy':>8s} {'fx_mean':>10s} {'fx_med':>10s} {'fx_std':>8s} {'fy_mean':>10s} {'fy_med':>10s} {'fy_std':>8s}")
    print("  " + "-" * 100)

    for label, cx_test, cy_test in cx_cy_tests:
        fxs, fys = [], []
        for cam_pts, gt_pts in zip(all_cam_pts, all_gt_pts_list):
            for i in range(len(active_kps)):
                u_gt, v_gt = gt_pts[i]
                X_cam, Y_cam, Z_cam = cam_pts[i]
                fx_e, fy_e = estimate_focal_from_point(
                    u_gt, v_gt, X_cam, Y_cam, Z_cam, cx_test, cy_test)
                if fx_e is not None and 50 < fx_e < 1000:
                    fxs.append(fx_e)
                if fy_e is not None and 50 < fy_e < 1000:
                    fys.append(fy_e)
        fxs = np.array(fxs)
        fys = np.array(fys)
        if len(fxs) > 0 and len(fys) > 0:
            print(f"  {label:20s} {cx_test:8.1f} {cy_test:8.1f} "
                  f"{fxs.mean():10.1f} {np.median(fxs):10.1f} {fxs.std():8.1f} "
                  f"{fys.mean():10.1f} {np.median(fys):10.1f} {fys.std():8.1f}")

    # ══════════════════════════════════════════════════════════════
    # PART 7: Alternative — fix fx/fy at each PSO result, compute
    #         reprojection error on ALL 6 frames jointly
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("PART 7: Cross-Validation — Each PSO Model on ALL 6 Frames")
    print("  (using each model's full params, evaluating on all frames)")
    print("=" * 70)

    for model_name, info in pso_results.items():
        if info["params"] is None:
            continue
        p = info["params"]
        total_sq = 0.0
        n_pts = 0
        per_frame_errs = []

        for idx, entry in enumerate(manifest["frames"]):
            png_path = os.path.join(manifest_dir, entry["image"])
            gt_pts_all = detect_keypoints_from_alpha(png_path, expected_count=n_kp)
            kp_indices = [manifest_kp_names.index(k[0]) for k in active_kps]
            gt_pts = gt_pts_all[kp_indices]

            transforms, ref_t, ref_R = load_frame_data(
                entry["task"], entry["episode"], entry["frame"],
                model, data_pin)
            world_pts = get_kp_world_positions(transforms, active_kps)

            # Scale if present
            scale = p.get("scale", 1.0)
            if scale != 1.0:
                center = transforms["pelvis"][0]
                world_pts = center + scale * (world_pts - center)

            cam_pos, R_w2c, t_w2c = compute_camera_transform(p, ref_t, ref_R)
            cam_pts = (R_w2c @ world_pts.T).T + t_w2c.flatten()

            fx = p.get("fx", p.get("f", 300))
            fy = p.get("fy", p.get("f", 300))
            cx_p = p["cx"]
            cy_p = p["cy"]

            proj = np.zeros_like(gt_pts)
            for i in range(len(active_kps)):
                X, Y, Z = cam_pts[i]
                if Z > 0.01:
                    proj[i, 0] = fx * X / Z + cx_p
                    proj[i, 1] = fy * Y / Z + cy_p

            diff = proj - gt_pts
            sq = np.sum(diff * diff)
            total_sq += sq
            n_pts += len(gt_pts)
            frame_rmse = np.sqrt(np.mean(np.sum(diff * diff, axis=1)))
            per_frame_errs.append((os.path.splitext(entry["image"])[0], frame_rmse))

        overall_rmse = np.sqrt(total_sq / n_pts)
        print(f"\n  {model_name} (original RMSE={info['rmse']:.2f}px):")
        print(f"    Cross-val RMSE on all 6 frames: {overall_rmse:.2f}px")
        for fl, fe in per_frame_errs:
            print(f"      {fl}: {fe:.2f}px")

    # ══════════════════════════════════════════════════════════════
    # PART 8: Least-squares focal estimation (all frames jointly)
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("PART 8: Least-Squares Focal Estimation (all points jointly)")
    print("  Minimizes sum of squared reprojection errors w.r.t. fx, fy")
    print("  with fixed extrinsics from combined model")
    print("=" * 70)

    cx_ls = BEST_PARAMS.get("cx", 320.0)
    cy_ls = BEST_PARAMS.get("cy", 240.0)

    # Build linear system: for each point,
    #   u_gt - cx = fx * (X_cam / Z_cam)   =>  a_i * fx = b_i
    #   v_gt - cy = fy * (Y_cam / Z_cam)   =>  c_i * fy = d_i
    A_fx, b_fx = [], []
    A_fy, b_fy = [], []

    for cam_pts, gt_pts in zip(all_cam_pts, all_gt_pts_list):
        for i in range(len(active_kps)):
            X, Y, Z = cam_pts[i]
            u_gt, v_gt = gt_pts[i]
            if abs(Z) > 0.01:
                if abs(X) > 1e-6:
                    A_fx.append(X / Z)
                    b_fx.append(u_gt - cx_ls)
                if abs(Y) > 1e-6:
                    A_fy.append(Y / Z)
                    b_fy.append(v_gt - cy_ls)

    A_fx = np.array(A_fx)
    b_fx = np.array(b_fx)
    A_fy = np.array(A_fy)
    b_fy = np.array(b_fy)

    # Least squares: fx = (A^T A)^{-1} A^T b
    fx_ls = np.dot(A_fx, b_fx) / np.dot(A_fx, A_fx)
    fy_ls = np.dot(A_fy, b_fy) / np.dot(A_fy, A_fy)

    # Residuals
    res_fx = b_fx - fx_ls * A_fx
    res_fy = b_fy - fy_ls * A_fy

    # Standard error of the estimate
    if len(A_fx) > 1:
        se_fx = np.sqrt(np.sum(res_fx ** 2) / (len(A_fx) - 1)) / np.sqrt(np.sum(A_fx ** 2))
    else:
        se_fx = float('inf')
    if len(A_fy) > 1:
        se_fy = np.sqrt(np.sum(res_fy ** 2) / (len(A_fy) - 1)) / np.sqrt(np.sum(A_fy ** 2))
    else:
        se_fy = float('inf')

    print(f"\n  Using cx={cx_ls:.2f}, cy={cy_ls:.2f} (combined model)")
    print(f"  fx_LS = {fx_ls:.2f}  (SE = {se_fx:.2f}, 95% CI: [{fx_ls - 1.96*se_fx:.2f}, {fx_ls + 1.96*se_fx:.2f}])")
    print(f"  fy_LS = {fy_ls:.2f}  (SE = {se_fy:.2f}, 95% CI: [{fy_ls - 1.96*se_fy:.2f}, {fy_ls + 1.96*se_fy:.2f}])")
    print(f"  fx residual RMS: {np.sqrt(np.mean(res_fx**2)):.2f}px")
    print(f"  fy residual RMS: {np.sqrt(np.mean(res_fy**2)):.2f}px")

    # Also try joint estimation (single f for both axes)
    A_f_all = np.concatenate([A_fx, A_fy])
    b_f_all = np.concatenate([b_fx, b_fy])
    f_ls = np.dot(A_f_all, b_f_all) / np.dot(A_f_all, A_f_all)
    res_f = b_f_all - f_ls * A_f_all
    if len(A_f_all) > 1:
        se_f = np.sqrt(np.sum(res_f ** 2) / (len(A_f_all) - 1)) / np.sqrt(np.sum(A_f_all ** 2))
    else:
        se_f = float('inf')

    print(f"\n  Joint f_LS = {f_ls:.2f}  (SE = {se_f:.2f}, 95% CI: [{f_ls - 1.96*se_f:.2f}, {f_ls + 1.96*se_f:.2f}])")

    # ══════════════════════════════════════════════════════════════
    # PART 9: Also try LS with image-center principal point
    # ══════════════════════════════════════════════════════════════
    print("\n  --- LS with image-center (cx=320, cy=240) ---")
    cx_ic, cy_ic = 320.0, 240.0
    A_fx2, b_fx2 = [], []
    A_fy2, b_fy2 = [], []
    for cam_pts, gt_pts in zip(all_cam_pts, all_gt_pts_list):
        for i in range(len(active_kps)):
            X, Y, Z = cam_pts[i]
            u_gt, v_gt = gt_pts[i]
            if abs(Z) > 0.01:
                if abs(X) > 1e-6:
                    A_fx2.append(X / Z)
                    b_fx2.append(u_gt - cx_ic)
                if abs(Y) > 1e-6:
                    A_fy2.append(Y / Z)
                    b_fy2.append(v_gt - cy_ic)
    A_fx2, b_fx2 = np.array(A_fx2), np.array(b_fx2)
    A_fy2, b_fy2 = np.array(A_fy2), np.array(b_fy2)
    fx_ls2 = np.dot(A_fx2, b_fx2) / np.dot(A_fx2, A_fx2)
    fy_ls2 = np.dot(A_fy2, b_fy2) / np.dot(A_fy2, A_fy2)
    res_fx2 = b_fx2 - fx_ls2 * A_fx2
    res_fy2 = b_fy2 - fy_ls2 * A_fy2
    print(f"  fx_LS = {fx_ls2:.2f}  residual RMS: {np.sqrt(np.mean(res_fx2**2)):.2f}px")
    print(f"  fy_LS = {fy_ls2:.2f}  residual RMS: {np.sqrt(np.mean(res_fy2**2)):.2f}px")

    # ══════════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    print(f"""
  Focal length estimates from different methods:

  {'Method':35s} {'fx':>10s} {'fy':>10s} {'Notes':>20s}
  {'-'*80}
  PSO combined (6 frames)           {pso_results['combined']['params']['fx']:10.1f} {pso_results['combined']['params']['fy']:10.1f} {'RMSE='+f"{pso_results['combined']['rmse']:.1f}px":>20s}
  PSO pillow (2 frames)             {pso_results['pillow']['params']['fx']:10.1f} {pso_results['pillow']['params']['fy']:10.1f} {'RMSE='+f"{pso_results['pillow']['rmse']:.1f}px":>20s}
  PSO basket (2 frames)             {pso_results['basket']['params']['fx']:10.1f} {pso_results['basket']['params']['fy']:10.1f} {'RMSE='+f"{pso_results['basket']['rmse']:.1f}px":>20s}
  PSO washing (2 frames)            {pso_results['washing']['params']['fx']:10.1f} {pso_results['washing']['params']['fy']:10.1f} {'RMSE='+f"{pso_results['washing']['rmse']:.1f}px":>20s}
  LS (cx={cx_ls:.0f},cy={cy_ls:.0f}, combined ext)  {fx_ls:10.1f} {fy_ls:10.1f} {'analytical':>20s}
  LS (cx=320,cy=240, combined ext)  {fx_ls2:10.1f} {fy_ls2:10.1f} {'image center':>20s}
  Analytical mean (all points)      {fx_arr.mean():10.1f} {fy_arr.mean():10.1f} {'per-point avg':>20s}
  Analytical median (all points)    {np.median(fx_arr):10.1f} {np.median(fy_arr):10.1f} {'per-point med':>20s}
""")

    print("  Key observations:")
    all_fx_vals = [
        pso_results['combined']['params']['fx'],
        pso_results['pillow']['params']['fx'],
        pso_results['basket']['params']['fx'],
        pso_results['washing']['params']['fx'],
        fx_ls, fx_ls2,
    ]
    fx_range = max(all_fx_vals) - min(all_fx_vals)
    print(f"  - fx range across methods: [{min(all_fx_vals):.0f}, {max(all_fx_vals):.0f}] (spread={fx_range:.0f}px)")
    print(f"  - The large spread indicates strong coupling between fx and extrinsics (especially dz, pitch)")
    print(f"  - Per-task PSO results overfit to 2 frames with compensating parameter shifts")
    print(f"  - Combined 6-frame result (fx={pso_results['combined']['params']['fx']:.1f}) is most reliable")


if __name__ == "__main__":
    main()
