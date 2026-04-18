"""
Render human arm overlay on G1 ego-centric video frames.

Uses the human_arm_overlay URDF with the same kinematic chain and build_q
mapping as the G1 Inspire hand URDF. Joint angles pass through directly
since G1 arm DOF structure matches human arm DOF exactly.

Usage:
  python scripts/render_human_overlay.py --episode 0
  python scripts/render_human_overlay.py --episode 0 --frame 50
  python scripts/render_human_overlay.py --episode 0 --frame 50 --side-by-side
"""

import sys
import os
import argparse
import numpy as np
import cv2
import pinocchio as pin

sys.stdout.reconfigure(line_buffering=True)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "scripts"))

from config import G1_URDF, HUMAN_URDF, MESH_DIR, BEST_PARAMS, SKIP_MESHES, OUTPUT_DIR, get_hand_type
from video_inpaint import (
    build_q, do_fk, parse_urdf_meshes, preload_meshes,
    make_camera, load_episode_info,
)

HUMAN_OVERLAY_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "human_overlay")

HUMAN_SKIP = {"head_link"}
SKIN_COLOR = (135, 165, 215)  # BGR: warm skin tone


# --- Human-specific rendering ---

def render_human_triangles(img, mesh_cache, transforms, params, color=SKIN_COLOR):
    """Render human arm mesh with per-triangle fill and simple shading."""
    K, D, rvec, tvec, R_w2c, t_w2c = make_camera(params, transforms)
    h, w = img.shape[:2]
    result = img.copy()

    # Collect all triangles with depth for painter's algorithm
    all_tris_2d = []
    all_depths = []
    all_normals_dot = []

    for link_name, (tris, _) in mesh_cache.items():
        if link_name not in transforms:
            continue

        t_link, R_link = transforms[link_name]
        flat = tris.reshape(-1, 3)
        world = (R_link @ flat.T).T + t_link

        cam_pts = (R_w2c @ world.T).T + t_w2c.flatten()
        z_cam = cam_pts[:, 2]

        pts2d, _ = cv2.projectPoints(
            world.reshape(-1, 1, 3), rvec, tvec, K, D)
        pts2d = pts2d.reshape(-1, 2)

        n_tri = len(tris)
        z_tri = z_cam.reshape(n_tri, 3)
        pts_tri = pts2d.reshape(n_tri, 3, 2)

        valid = (z_tri > 0.01).all(axis=1)
        pts_tri = pts_tri[valid]
        z_tri = z_tri[valid]
        if len(pts_tri) == 0:
            continue

        finite = np.all(np.isfinite(pts_tri), axis=(1, 2))
        pts_tri = pts_tri[finite]
        z_tri = z_tri[finite]

        # Compute face normals in camera space for shading
        world_tris = world.reshape(n_tri, 3, 3)[valid][finite]
        v0, v1, v2 = world_tris[:, 0], world_tris[:, 1], world_tris[:, 2]
        normals = np.cross(v1 - v0, v2 - v0)
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        normals = normals / norms

        # View direction: from triangle center toward camera
        centers = world_tris.mean(axis=1)
        view_dirs = cam_pts.reshape(n_tri, 3, 3)[valid][finite].mean(axis=1)
        view_dirs = -view_dirs  # toward camera
        view_norms = np.linalg.norm(view_dirs, axis=1, keepdims=True)
        view_norms = np.maximum(view_norms, 1e-8)
        view_dirs = view_dirs / view_norms

        # Lambertian: abs dot product (abs to handle both face orientations)
        dots = np.abs(np.sum(normals * view_dirs, axis=1))

        avg_depths = z_tri.mean(axis=1)
        all_tris_2d.append(pts_tri)
        all_depths.append(avg_depths)
        all_normals_dot.append(dots)

    if not all_tris_2d:
        return result

    all_tris_2d = np.concatenate(all_tris_2d)
    all_depths = np.concatenate(all_depths)
    all_normals_dot = np.concatenate(all_normals_dot)

    # Painter's algorithm: sort far to near
    order = np.argsort(-all_depths)

    for idx in order:
        tri = all_tris_2d[idx].astype(np.int32)
        shade = 0.4 + 0.6 * all_normals_dot[idx]  # ambient 0.4 + diffuse 0.6
        shaded_color = tuple(int(c * shade) for c in color)
        cv2.fillPoly(result, [tri], shaded_color)

    return result


def render_g1_overlay(img, mesh_cache, transforms, params):
    """Semi-transparent G1 robot overlay (for comparison)."""
    K, D, rvec, tvec, R_w2c, t_w2c = make_camera(params, transforms)
    overlay = np.zeros_like(img)
    result = img.copy()

    for link_name, (_, unique_verts) in mesh_cache.items():
        if link_name not in transforms or len(unique_verts) == 0:
            continue
        t_link, R_link = transforms[link_name]
        verts3d = ((R_link @ unique_verts.T).T + t_link).astype(np.float64)
        depths = (R_w2c @ verts3d.T).T + t_w2c.flatten()
        in_front = depths[:, 2] > 0.01
        if np.count_nonzero(in_front) < 3:
            continue
        pts2d, _ = cv2.projectPoints(
            verts3d[in_front].reshape(-1, 1, 3), rvec, tvec, K, D)
        pts2d = pts2d.reshape(-1, 2)
        finite = np.all(np.isfinite(pts2d), axis=1)
        pts2d = pts2d[finite]
        if len(pts2d) < 3:
            continue
        hull = cv2.convexHull(pts2d.astype(np.float32))
        color = (255, 180, 0) if "left" in link_name else (0, 180, 255)
        cv2.fillConvexPoly(overlay, hull.astype(np.int32), color)

    cv2.addWeighted(overlay, 0.35, result, 1.0, 0, result)
    return result


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


def main():
    parser = argparse.ArgumentParser(description="Human arm overlay test")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--frame", type=int, default=30,
                        help="Frame index within episode (default 30)")
    parser.add_argument("--side-by-side", action="store_true",
                        help="Show G1 overlay + human overlay side by side")
    args = parser.parse_args()

    os.makedirs(HUMAN_OVERLAY_OUTPUT_DIR, exist_ok=True)

    # Load episode
    print(f"Loading episode {args.episode}...")
    video_path, from_ts, to_ts, ep_df = load_episode_info(args.episode)
    print(f"  {len(ep_df)} frames, video: {os.path.basename(video_path)}")

    # Get frame data
    frame_row = ep_df[ep_df["frame_index"] == args.frame]
    if len(frame_row) == 0:
        print(f"Frame {args.frame} not found, using first frame")
        frame_row = ep_df.iloc[[0]]
        args.frame = int(frame_row.iloc[0]["frame_index"])

    rq = np.array(frame_row.iloc[0]["observation.state.robot_q_current"], dtype=np.float64)
    hs = np.array(frame_row.iloc[0]["observation.state.hand_state"], dtype=np.float64)

    # Extract video frame
    print(f"Extracting frame {args.frame}...")
    img = extract_frame(video_path, from_ts, args.frame)
    if img is None:
        print("Failed to extract frame")
        return

    h, w = img.shape[:2]
    print(f"  Frame size: {w}x{h}")

    # Load human URDF + meshes
    print("Loading human URDF...")
    model_h = pin.buildModelFromUrdf(HUMAN_URDF, pin.JointModelFreeFlyer())
    data_h = model_h.createData()
    link_meshes_h = parse_urdf_meshes(HUMAN_URDF)
    mesh_cache_h = preload_meshes(link_meshes_h, MESH_DIR, skip_set=HUMAN_SKIP)
    print(f"  Human: {len(mesh_cache_h)} link meshes")

    # FK for human
    hand_type = get_hand_type()
    q = build_q(model_h, rq, hs, hand_type=hand_type)
    transforms_h = do_fk(model_h, data_h, q)

    # Render human overlay
    print("Rendering human overlay...")
    human_result = render_human_triangles(img, mesh_cache_h, transforms_h, BEST_PARAMS)

    if args.side_by_side:
        # Also load G1 for comparison
        print("Loading G1 URDF for comparison...")
        model_g = pin.buildModelFromUrdf(G1_URDF, pin.JointModelFreeFlyer())
        data_g = model_g.createData()
        link_meshes_g = parse_urdf_meshes(G1_URDF)
        mesh_cache_g = preload_meshes(link_meshes_g, MESH_DIR, skip_set=SKIP_MESHES)
        print(f"  G1: {len(mesh_cache_g)} link meshes")

        q_g = build_q(model_g, rq, hs, hand_type=hand_type)
        transforms_g = do_fk(model_g, data_g, q_g)
        g1_result = render_g1_overlay(img, mesh_cache_g, transforms_g, BEST_PARAMS)

        # Side by side: original | G1 overlay | human overlay
        combined = np.hstack([img, g1_result, human_result])
        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(combined, "Original", (10, 30), font, 0.8, (255,255,255), 2)
        cv2.putText(combined, "G1 Robot", (w + 10, 30), font, 0.8, (255,255,255), 2)
        cv2.putText(combined, "Human Overlay", (2*w + 10, 30), font, 0.8, (255,255,255), 2)

        out_path = os.path.join(HUMAN_OVERLAY_OUTPUT_DIR,
                                 f"ep{args.episode}_f{args.frame}_comparison.png")
        cv2.imwrite(out_path, combined)
        print(f"Saved: {out_path}")
    else:
        out_path = os.path.join(HUMAN_OVERLAY_OUTPUT_DIR,
                                 f"ep{args.episode}_f{args.frame}_human.png")
        cv2.imwrite(out_path, human_result)
        print(f"Saved: {out_path}")

        # Also save just the original for reference
        out_orig = os.path.join(HUMAN_OVERLAY_OUTPUT_DIR,
                                 f"ep{args.episode}_f{args.frame}_original.png")
        cv2.imwrite(out_orig, img)


if __name__ == "__main__":
    main()
