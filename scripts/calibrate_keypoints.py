"""
Camera calibration via sparse keypoints (PnP-style).

Two modes:
  --annotate : Interactive GUI to mark keypoints on video frames
  --optimize : Adam optimization using saved annotations

Workflow:
  1. python scripts/calibrate_keypoints.py --annotate --episode 0 --frames 30 80 130
  2. python scripts/calibrate_keypoints.py --optimize --steps 2000

Usage:
  python scripts/calibrate_keypoints.py --annotate --episode 0 --frames 50 100
  python scripts/calibrate_keypoints.py --optimize --device cuda:0 --steps 2000
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

sys.stdout.reconfigure(line_buffering=True)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "scripts"))

from config import (G1_URDF, MESH_DIR, BEST_PARAMS, SKIP_MESHES,
                     DATASET_ROOT, OUTPUT_DIR, get_hand_type)
from video_inpaint import (build_q, do_fk, parse_urdf_meshes, preload_meshes,
                            make_camera, load_episode_info)

KP_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "calib_keypoints")
ANNOTATIONS_PATH = os.path.join(KP_OUTPUT_DIR, "annotations.json")

# ============================================================
# Keypoint definitions: (display_name, link_name, local_offset)
# local_offset = specific vertex in the link's local frame
# ============================================================

# Logo triangle: 3 outer vertices (from convex hull analysis of logo_link.STL)
# Corner A: left-bottom,  Corner B: right (tip),  Corner C: left-top
KEYPOINTS = [
    ("logo_A", "logo_link", np.array([-0.0683, -0.0713, 0.2799])),
    ("logo_B", "logo_link", np.array([ 0.0727, -0.0083, 0.2764])),
    ("logo_C", "logo_link", np.array([-0.0684,  0.0711, 0.2799])),
]

# Colors for each keypoint (BGR)
KP_COLORS = [
    (0, 0, 255),     # A - red
    (0, 255, 0),     # B - green
    (255, 0, 0),     # C - blue
]


def get_keypoint_3d(kp_name, link_name, local_offset, transforms):
    """Get 3D world position of a keypoint from FK transforms."""
    if link_name not in transforms:
        return None
    t_link, R_link = transforms[link_name]
    return R_link @ local_offset + t_link


def project_keypoint_2d(world_pos, params, transforms):
    """Project a 3D keypoint to 2D using current camera params."""
    K, D, rvec, tvec, R_w2c, t_w2c = make_camera(params, transforms)
    pts3d = world_pos.reshape(1, 1, 3).astype(np.float64)
    pts2d, _ = cv2.fisheye.projectPoints(pts3d, rvec, tvec, K, D)
    return pts2d.reshape(2)


def extract_frame(video_path, from_ts, frame_idx, fps=30):
    """Extract a single frame using PyAV."""
    import av
    target_ts = from_ts + frame_idx / fps
    container = av.open(video_path)
    stream = container.streams.video[0]
    tb = float(stream.time_base)
    seek_pts = int(max(0, target_ts - 0.5) / tb)
    container.seek(seek_pts, stream=stream)
    for frame in container.decode(video=0):
        pts_sec = frame.pts * tb
        fi = int(round((pts_sec - from_ts) * fps))
        if fi == frame_idx:
            img = frame.to_ndarray(format='bgr24')
            container.close()
            return img
    container.close()
    return None


# ============================================================
# Annotation mode (interactive GUI)
# ============================================================

def render_reference_overlay(img, transforms, params):
    """Draw FK-projected keypoint positions as reference circles."""
    result = img.copy()
    for i, (kp_name, link_name, local_offset) in enumerate(KEYPOINTS):
        pos3d = get_keypoint_3d(kp_name, link_name, local_offset, transforms)
        if pos3d is None:
            continue
        pos2d = project_keypoint_2d(pos3d, params, transforms)
        if not np.all(np.isfinite(pos2d)):
            continue
        x, y = int(pos2d[0]), int(pos2d[1])
        color = KP_COLORS[i]
        # Draw reference circle (hollow, dashed effect)
        cv2.circle(result, (x, y), 12, color, 1)
        cv2.circle(result, (x, y), 2, color, -1)
        # Label
        cv2.putText(result, kp_name, (x + 14, y + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
    return result


def annotate_frames(args):
    """Interactive annotation GUI."""
    os.makedirs(KP_OUTPUT_DIR, exist_ok=True)

    # Load existing annotations
    annotations = {}
    if os.path.exists(ANNOTATIONS_PATH):
        with open(ANNOTATIONS_PATH) as f:
            annotations = json.load(f)
        print(f"Loaded {len(annotations)} existing annotated frames")

    # Load model
    print("Loading URDF...")
    model = pin.buildModelFromUrdf(G1_URDF, pin.JointModelFreeFlyer())
    data = model.createData()

    # Load episode data
    data_dir = os.path.join(DATASET_ROOT, args.task) if args.task else None
    video_path, from_ts, to_ts, ep_df = load_episode_info(
        args.episode, data_dir=data_dir)
    print(f"Episode {args.episode}: {len(ep_df)} frames")

    click_pos = [None]  # mutable container for callback

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            click_pos[0] = (x, y)

    cv2.namedWindow("Annotate", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Annotate", mouse_callback)

    for fi in args.frames:
        frame_key = f"ep{args.episode}_f{fi}"
        print(f"\n--- Frame {fi} (key={frame_key}) ---")

        # Get joint state
        frame_row = ep_df[ep_df["frame_index"] == fi]
        if len(frame_row) == 0:
            print(f"  Frame {fi} not in dataset, skipping")
            continue
        row = frame_row.iloc[0]
        rq = np.array(row["observation.state.robot_q_current"], dtype=np.float64)
        hs = np.array(row["observation.state.hand_state"], dtype=np.float64)

        # FK
        q = build_q(model, rq, hs, hand_type=get_hand_type(args.task))
        transforms = do_fk(model, data, q)

        # Extract video frame
        img = extract_frame(video_path, from_ts, fi)
        if img is None:
            print(f"  Cannot extract frame {fi}")
            continue
        h, w = img.shape[:2]

        # Initialize frame annotations
        if frame_key not in annotations:
            annotations[frame_key] = {
                "episode": args.episode,
                "frame_index": fi,
                "task": args.task or "",
                "keypoints": {}
            }

        frame_ann = annotations[frame_key]["keypoints"]

        # Annotate each keypoint
        for ki, (kp_name, link_name, local_offset) in enumerate(KEYPOINTS):
            # Draw reference overlay
            display = render_reference_overlay(img, transforms, BEST_PARAMS)

            # Draw already-annotated points for this frame
            for prev_name, prev_pos in frame_ann.items():
                px, py = int(prev_pos[0]), int(prev_pos[1])
                # Find color
                cidx = next((j for j, (n, _, _) in enumerate(KEYPOINTS) if n == prev_name), 0)
                cv2.circle(display, (px, py), 6, KP_COLORS[cidx], -1)
                cv2.circle(display, (px, py), 8, (255, 255, 255), 2)
                cv2.putText(display, prev_name, (px + 10, py - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)

            # Highlight current keypoint's FK reference
            pos3d = get_keypoint_3d(kp_name, link_name, local_offset, transforms)
            if pos3d is not None:
                pos2d = project_keypoint_2d(pos3d, BEST_PARAMS, transforms)
                if np.all(np.isfinite(pos2d)):
                    rx, ry = int(pos2d[0]), int(pos2d[1])
                    cv2.circle(display, (rx, ry), 16, KP_COLORS[ki], 2)
                    cv2.drawMarker(display, (rx, ry), KP_COLORS[ki],
                                   cv2.MARKER_CROSS, 20, 2)

            # Status bar
            status = f"[{ki+1}/{len(KEYPOINTS)}] Click: {kp_name}  |  s=skip  z=undo  q=quit"
            cv2.rectangle(display, (0, h - 25), (w, h), (0, 0, 0), -1)
            cv2.putText(display, status, (5, h - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

            # If already annotated, show existing
            if kp_name in frame_ann:
                ex, ey = int(frame_ann[kp_name][0]), int(frame_ann[kp_name][1])
                cv2.circle(display, (ex, ey), 6, (0, 255, 0), -1)
                status2 = f"  (existing: {ex},{ey} - click to replace, Enter to keep)"
                cv2.putText(display, status2, (5, h - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 200, 0), 1)

            cv2.imshow("Annotate", display)
            click_pos[0] = None

            while True:
                key = cv2.waitKey(50) & 0xFF
                if click_pos[0] is not None:
                    # User clicked
                    cx, cy = click_pos[0]
                    frame_ann[kp_name] = [cx, cy]
                    print(f"  {kp_name}: ({cx}, {cy})")
                    click_pos[0] = None
                    break
                elif key == ord('s'):
                    # Skip this keypoint
                    print(f"  {kp_name}: skipped")
                    break
                elif key == ord('z'):
                    # Undo last annotation in this frame
                    if frame_ann:
                        last_key = list(frame_ann.keys())[-1]
                        del frame_ann[last_key]
                        print(f"  Undid: {last_key}")
                    break
                elif key == 13:  # Enter - keep existing
                    if kp_name in frame_ann:
                        print(f"  {kp_name}: kept ({frame_ann[kp_name]})")
                    break
                elif key == ord('q'):
                    print("Quit requested, saving...")
                    with open(ANNOTATIONS_PATH, 'w') as f:
                        json.dump(annotations, f, indent=2)
                    print(f"Saved to {ANNOTATIONS_PATH}")
                    cv2.destroyAllWindows()
                    return

        # Save after each frame
        with open(ANNOTATIONS_PATH, 'w') as f:
            json.dump(annotations, f, indent=2)
        n_annotated = len(frame_ann)
        print(f"  Frame {fi}: {n_annotated} keypoints annotated, saved.")

    cv2.destroyAllWindows()
    print(f"\nAll done. Annotations saved to {ANNOTATIONS_PATH}")
    print(f"Total: {len(annotations)} frames")


# ============================================================
# Optimization mode (Adam on reprojection error)
# ============================================================

def optimize_from_annotations(args):
    """Optimize camera params using annotated keypoints."""
    import torch
    import torch.nn as nn

    if not os.path.exists(ANNOTATIONS_PATH):
        print(f"ERROR: No annotations found at {ANNOTATIONS_PATH}")
        print("Run with --annotate first.")
        sys.exit(1)

    with open(ANNOTATIONS_PATH) as f:
        annotations = json.load(f)

    print(f"Loaded {len(annotations)} annotated frames")

    # Load model
    model = pin.buildModelFromUrdf(G1_URDF, pin.JointModelFreeFlyer())
    data_pin = model.createData()

    device = torch.device(args.device)

    # Import from grad calibration
    from auto_calibrate_grad import (CameraParams, differentiable_fisheye_project,
                                      PARAM_NAMES, BOUNDS)
    from auto_calibrate_v2 import (project_and_mask, compute_iou,
                                    save_comparison, BOUNDS as BOUNDS_NP)

    # Prepare optimization data: for each annotated frame, compute FK and
    # extract 3D keypoint positions + their 2D annotations
    optim_data = []  # list of (kp_world_3d, kp_pixel_2d, ref_t, ref_R, label)

    for frame_key, frame_ann in annotations.items():
        episode = frame_ann["episode"]
        fi = frame_ann["frame_index"]
        task = frame_ann.get("task", "")
        kps = frame_ann["keypoints"]

        if not kps:
            continue

        data_dir = os.path.join(DATASET_ROOT, task) if task else None
        video_path, from_ts, to_ts, ep_df = load_episode_info(episode, data_dir=data_dir)

        frame_row = ep_df[ep_df["frame_index"] == fi]
        if len(frame_row) == 0:
            print(f"  WARNING: frame {fi} not in dataset")
            continue

        row = frame_row.iloc[0]
        rq = np.array(row["observation.state.robot_q_current"], dtype=np.float64)
        hs = np.array(row["observation.state.hand_state"], dtype=np.float64)
        q = build_q(model, rq, hs, hand_type=get_hand_type(task if task else None))
        transforms = do_fk(model, data_pin, q)
        ref_t, ref_R = transforms["torso_link"]

        # Extract 3D positions for annotated keypoints
        world_pts = []
        pixel_pts = []
        kp_names = []
        for kp_name, link_name, local_offset in KEYPOINTS:
            if kp_name not in kps:
                continue
            pos3d = get_keypoint_3d(kp_name, link_name, local_offset, transforms)
            if pos3d is None:
                continue
            world_pts.append(pos3d)
            pixel_pts.append(kps[kp_name])
            kp_names.append(kp_name)

        if not world_pts:
            continue

        world_pts = np.array(world_pts, dtype=np.float64)
        pixel_pts = np.array(pixel_pts, dtype=np.float64)

        optim_data.append({
            "world_pts": world_pts,
            "pixel_pts": pixel_pts,
            "ref_t": ref_t,
            "ref_R": ref_R,
            "kp_names": kp_names,
            "label": frame_key,
        })
        print(f"  {frame_key}: {len(world_pts)} keypoints")

    if not optim_data:
        print("ERROR: No valid keypoint data")
        sys.exit(1)

    total_kps = sum(len(d["world_pts"]) for d in optim_data)
    print(f"\nTotal: {len(optim_data)} frames, {total_kps} keypoint observations")

    # Initialize camera parameters
    cam = CameraParams(BEST_PARAMS, device=device)

    # Per-group learning rates (adjusted for keypoint optimization)
    param_groups = cam.get_param_groups(base_lr=args.lr_scale)
    optimizer = torch.optim.Adam(param_groups)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.steps, eta_min=0.0)

    # Convert data to torch
    frame_tensors = []
    for d in optim_data:
        frame_tensors.append({
            "world_pts": torch.tensor(d["world_pts"], dtype=torch.float64, device=device),
            "pixel_pts": torch.tensor(d["pixel_pts"], dtype=torch.float64, device=device),
            "ref_t": torch.tensor(d["ref_t"], dtype=torch.float64, device=device),
            "ref_R": torch.tensor(d["ref_R"], dtype=torch.float64, device=device),
            "label": d["label"],
            "kp_names": d["kp_names"],
        })

    # Import the differentiable projection building blocks
    from auto_calibrate_grad import _build_rotation

    def project_keypoints_diff(world_pts, cam, ref_t, ref_R, device):
        """Differentiable projection of sparse 3D keypoints to 2D."""
        R_cam = _build_rotation(cam.pitch, cam.yaw, cam.roll, device)
        offset = torch.stack([cam.dx, cam.dy, cam.dz])
        cam_pos = ref_t + ref_R @ offset
        R_w2c = (ref_R @ R_cam.T).T
        t_w2c = R_w2c @ (-cam_pos)

        # World -> camera
        p_cam = (R_w2c @ world_pts.T).T + t_w2c  # (N, 3)
        x, y, z = p_cam[:, 0], p_cam[:, 1], p_cam[:, 2]

        # Fisheye projection
        a = x / (z + 1e-12)
        b = y / (z + 1e-12)
        r = torch.sqrt(a * a + b * b + 1e-12)
        theta = torch.atan(r)
        theta2 = theta * theta
        theta4 = theta2 * theta2
        theta6 = theta4 * theta2
        theta8 = theta4 * theta4
        theta_d = theta * (1.0 + cam.k1 * theta2 + cam.k2 * theta4
                           + cam.k3 * theta6 + cam.k4 * theta8)
        scale = theta_d / (r + 1e-12)
        u = cam.fx * scale * a + cam.cx
        v = cam.fy * scale * b + cam.cy
        return torch.stack([u, v], dim=1)  # (N, 2)

    # Optimization loop
    print(f"\n{'='*60}")
    print(f"Keypoint optimization: {len(optim_data)} frames, "
          f"{total_kps} points, {args.steps} steps")
    print(f"{'='*60}\n")

    best_err = float('inf')
    best_params = cam.to_dict()
    log_entries = []
    t0 = time.time()

    for step in range(args.steps):
        optimizer.zero_grad()

        total_loss = torch.tensor(0.0, dtype=torch.float64, device=device)
        total_pts = 0

        for ft in frame_tensors:
            proj2d = project_keypoints_diff(
                ft["world_pts"], cam, ft["ref_t"], ft["ref_R"], device)
            # L2 reprojection error (in pixels)
            diff = proj2d - ft["pixel_pts"]
            loss = (diff * diff).sum()
            total_loss = total_loss + loss
            total_pts += len(ft["world_pts"])

        # Mean squared reprojection error
        mse = total_loss / total_pts
        rmse = torch.sqrt(mse)

        mse.backward()
        torch.nn.utils.clip_grad_norm_(cam.parameters(), max_norm=10.0)

        # Collect gradients
        grad_dict = {}
        for name in PARAM_NAMES:
            p = getattr(cam, name)
            grad_dict[name] = p.grad.item() if p.grad is not None else 0.0

        optimizer.step()
        scheduler.step()
        cam.clamp_to_bounds()

        rmse_val = rmse.item()
        if rmse_val < best_err:
            best_err = rmse_val
            best_params = cam.to_dict()

        log_entries.append({
            "step": step,
            "rmse_px": rmse_val,
            "params": cam.to_dict(),
            "gradients": grad_dict,
        })

        if step % args.print_interval == 0 or step == args.steps - 1:
            elapsed = time.time() - t0
            top3 = sorted(grad_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
            grad_str = " ".join(f"{k}:{v:+.4f}" for k, v in top3)
            print(f"  step {step:4d}/{args.steps}  RMSE={rmse_val:.2f}px  "
                  f"top_grad=[{grad_str}]  ({elapsed:.0f}s)")

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"DONE  best_RMSE={best_err:.2f}px  elapsed={elapsed:.0f}s")
    print(f"Best params: {best_params}")
    print(f"{'='*60}")

    # Per-frame, per-keypoint breakdown
    print("\nPER-KEYPOINT REPROJECTION ERROR:")
    cam_best = CameraParams(best_params, device=device)
    with torch.no_grad():
        for ft in frame_tensors:
            proj2d = project_keypoints_diff(
                ft["world_pts"], cam_best, ft["ref_t"], ft["ref_R"], device)
            diff = (proj2d - ft["pixel_pts"]).cpu().numpy()
            errs = np.sqrt((diff ** 2).sum(axis=1))
            print(f"\n  {ft['label']}:")
            for j, kp_name in enumerate(ft["kp_names"]):
                print(f"    {kp_name:20s}  err={errs[j]:.1f}px  "
                      f"proj=({proj2d[j,0]:.0f},{proj2d[j,1]:.0f})  "
                      f"gt=({ft['pixel_pts'][j,0]:.0f},{ft['pixel_pts'][j,1]:.0f})")

    # Save results
    with open(os.path.join(KP_OUTPUT_DIR, "best_params.json"), 'w') as f:
        json.dump({"rmse_px": best_err, "params": best_params}, f, indent=2)

    with open(os.path.join(KP_OUTPUT_DIR, "optimization_log.json"), 'w') as f:
        json.dump(log_entries, f, indent=2)

    # Plot loss curve
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        steps = [e["step"] for e in log_entries]
        rmses = [e["rmse_px"] for e in log_entries]

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(steps, rmses, 'b-')
        ax.set_xlabel('Step')
        ax.set_ylabel('RMSE (pixels)')
        ax.set_title(f'Keypoint Reprojection Error (best={best_err:.2f}px)')
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(KP_OUTPUT_DIR, "loss_curve.png"), dpi=150)
        plt.close(fig)

        # Gradient summary
        mean_abs_grads = {}
        for name in PARAM_NAMES:
            vals = [abs(e["gradients"][name]) for e in log_entries]
            mean_abs_grads[name] = np.mean(vals)
        fig, ax = plt.subplots(figsize=(12, 5))
        names = list(mean_abs_grads.keys())
        mags = [mean_abs_grads[n] for n in names]
        bars = ax.bar(names, mags)
        ax.set_ylabel('Mean |gradient|')
        ax.set_title('Average Gradient Magnitude per Parameter')
        ax.set_yscale('log')
        for bar, val in zip(bars, mags):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                        f'{val:.2e}', ha='center', va='bottom', fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(KP_OUTPUT_DIR, "gradient_summary.png"), dpi=150)
        plt.close(fig)

        # Param trajectory
        fig, axes = plt.subplots(4, 4, figsize=(20, 16))
        axes_flat = axes.flatten()
        for i, name in enumerate(PARAM_NAMES):
            vals = [e["params"][name] for e in log_entries]
            axes_flat[i].plot(steps, vals, linewidth=0.8)
            axes_flat[i].set_title(name)
            axes_flat[i].grid(True, alpha=0.3)
        for i in range(len(PARAM_NAMES), len(axes_flat)):
            axes_flat[i].set_visible(False)
        fig.suptitle('Parameter Trajectory', fontsize=14)
        fig.tight_layout()
        fig.savefig(os.path.join(KP_OUTPUT_DIR, "param_trajectory.png"), dpi=150)
        plt.close(fig)

        print(f"\nPlots saved to {KP_OUTPUT_DIR}/")
    except Exception as e:
        print(f"Warning: plotting failed: {e}")

    return best_params, best_err


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Keypoint-based camera calibration")
    sub = parser.add_subparsers(dest="mode")

    # Annotate mode
    p_ann = sub.add_parser("--annotate", help="Interactive keypoint annotation")
    p_ann.add_argument("--episode", type=int, required=True)
    p_ann.add_argument("--frames", type=int, nargs="+", required=True,
                       help="Frame indices to annotate")
    p_ann.add_argument("--task", type=str, default=None,
                       help="Task name (default: ACTIVE_TASK from config)")

    # Optimize mode
    p_opt = sub.add_parser("--optimize", help="Optimize camera from annotations")
    p_opt.add_argument("--steps", type=int, default=2000)
    p_opt.add_argument("--device", type=str, default="cuda:0")
    p_opt.add_argument("--lr-scale", type=float, default=1.0)
    p_opt.add_argument("--print-interval", type=int, default=20)

    # Also support --annotate/--optimize as flags (simpler CLI)
    parser.add_argument("--annotate", action="store_true")
    parser.add_argument("--optimize", action="store_true")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--frames", type=int, nargs="+", default=[50, 100])
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--lr-scale", type=float, default=1.0)
    parser.add_argument("--print-interval", type=int, default=20)

    args = parser.parse_args()

    os.makedirs(KP_OUTPUT_DIR, exist_ok=True)

    if args.annotate:
        annotate_frames(args)
    elif args.optimize:
        optimize_from_annotations(args)
    else:
        parser.print_help()
        print("\nUse --annotate or --optimize")


if __name__ == "__main__":
    main()
