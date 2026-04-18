"""
Render hand-only debug visualization with per-link coloring and zoom.

Outputs a composite image with zoomed-in left/right hand crops and a full
frame overlay, plus a console diagnostic of hand_state -> joint angles.

Usage:
  python scripts/render_hand_debug.py --episode 0 --frame 30
  python scripts/render_hand_debug.py --episode 0 --frame 30 --human
  python scripts/render_hand_debug.py --episode 0 --frames 30,60,90,120
  python scripts/render_hand_debug.py --episode 0 --frame 30 --task G1_WBT_Inspire_Put_Clothes_Into_Basket
"""

import sys
import os
import argparse
import fnmatch
import numpy as np
import cv2
import pinocchio as pin

sys.stdout.reconfigure(line_buffering=True)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "scripts"))

from config import (
    G1_URDF, HUMAN_URDF, MESH_DIR, BEST_PARAMS, SKIP_MESHES,
    OUTPUT_DIR, ACTIVE_TASK, get_hand_type,
)
from video_inpaint import (
    build_q, do_fk, parse_urdf_meshes, preload_meshes,
    make_camera, load_episode_info,
)

HAND_DEBUG_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "hand_debug")

# Link patterns for hand + arm filtering
HAND_ARM_PATTERNS = {
    "left_arm":  ["left_elbow_*", "left_wrist_*"],
    "left_hand": ["left_base_link", "left_palm_force_sensor",
                  "left_thumb_*", "left_index_*", "left_middle_*",
                  "left_ring_*", "left_little_*",
                  "left_rubber_hand"],
    "right_arm": ["right_elbow_*", "right_wrist_*"],
    "right_hand":["right_base_link", "right_palm_force_sensor",
                  "right_thumb_*", "right_index_*", "right_middle_*",
                  "right_ring_*", "right_little_*",
                  "right_rubber_hand"],
}

# Per-link colors (BGR) grouped by finger
LINK_COLORS = {
    # Left hand
    "left_base_link":           (200, 200, 200),
    "left_palm_force_sensor":   (180, 180, 180),
    "left_rubber_hand":         (160, 160, 160),
    "left_thumb_1":             (60, 60, 255),    # red family
    "left_thumb_2":             (40, 40, 220),
    "left_thumb_3":             (30, 30, 180),
    "left_thumb_4":             (20, 20, 140),
    "left_index_1":             (60, 255, 60),    # green family
    "left_index_2":             (40, 200, 40),
    "left_middle_1":            (255, 60, 60),    # blue family
    "left_middle_2":            (200, 40, 40),
    "left_ring_1":              (0, 255, 255),    # yellow family
    "left_ring_2":              (0, 200, 200),
    "left_little_1":            (255, 60, 255),   # magenta family
    "left_little_2":            (200, 40, 200),
    # Left arm
    "left_elbow_link":          (100, 160, 255),
    "left_wrist_roll_link":     (80, 140, 230),
    "left_wrist_pitch_link":    (60, 120, 210),
    "left_wrist_yaw_link":      (40, 100, 190),
    # Right hand (mirror colors)
    "right_base_link":          (200, 200, 200),
    "right_palm_force_sensor":  (180, 180, 180),
    "right_rubber_hand":        (160, 160, 160),
    "right_thumb_1":            (60, 60, 255),
    "right_thumb_2":            (40, 40, 220),
    "right_thumb_3":            (30, 30, 180),
    "right_thumb_4":            (20, 20, 140),
    "right_index_1":            (60, 255, 60),
    "right_index_2":            (40, 200, 40),
    "right_middle_1":           (255, 60, 60),
    "right_middle_2":           (200, 40, 40),
    "right_ring_1":             (0, 255, 255),
    "right_ring_2":             (0, 200, 200),
    "right_little_1":           (255, 60, 255),
    "right_little_2":           (200, 40, 200),
    "right_elbow_link":         (255, 160, 100),
    "right_wrist_roll_link":    (230, 140, 80),
    "right_wrist_pitch_link":   (210, 120, 60),
    "right_wrist_yaw_link":     (190, 100, 40),
}

# Legend entries (finger name -> representative color)
LEGEND_ENTRIES = [
    ("thumb",  (60, 60, 255)),
    ("index",  (60, 255, 60)),
    ("middle", (255, 60, 60)),
    ("ring",   (0, 255, 255)),
    ("little", (255, 60, 255)),
    ("palm",   (200, 200, 200)),
    ("arm",    (100, 160, 255)),
]


def match_links(mesh_cache, patterns):
    """Filter mesh_cache to links matching glob patterns."""
    matched = {}
    for link_name, data in mesh_cache.items():
        for pat in patterns:
            if fnmatch.fnmatch(link_name, pat):
                matched[link_name] = data
                break
    return matched


def get_link_color(link_name):
    if link_name in LINK_COLORS:
        return LINK_COLORS[link_name]
    # Fallback: hash-based color
    h = hash(link_name) % 360
    # HSV -> BGR via OpenCV
    hsv = np.array([[[h // 2, 200, 200]]], dtype=np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return tuple(int(c) for c in bgr)


def render_hand_triangles(img, mesh_cache, transforms, params, subsample=1):
    """Render hand mesh with per-link coloring and Lambertian shading.

    Returns (rendered_image, dict of side -> list of 2D points).
    """
    K, D, rvec, tvec, R_w2c, t_w2c = make_camera(params, transforms)
    result = img.copy()

    all_tris_2d = []
    all_depths = []
    all_normals_dot = []
    all_colors = []
    side_points = {"left": [], "right": []}

    for link_name, (tris, _) in mesh_cache.items():
        if link_name not in transforms:
            continue

        if subsample > 1:
            tris = tris[::subsample]

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
        if len(pts_tri) == 0:
            continue

        # Face normals for shading
        world_tris = world.reshape(n_tri, 3, 3)[valid][finite]
        v0, v1, v2 = world_tris[:, 0], world_tris[:, 1], world_tris[:, 2]
        normals = np.cross(v1 - v0, v2 - v0)
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        normals = normals / norms

        cam_tris = cam_pts.reshape(n_tri, 3, 3)[valid][finite]
        view_dirs = -cam_tris.mean(axis=1)
        view_norms = np.linalg.norm(view_dirs, axis=1, keepdims=True)
        view_norms = np.maximum(view_norms, 1e-8)
        view_dirs = view_dirs / view_norms
        dots = np.abs(np.sum(normals * view_dirs, axis=1))

        avg_depths = z_tri.mean(axis=1)
        color = get_link_color(link_name)

        all_tris_2d.append(pts_tri)
        all_depths.append(avg_depths)
        all_normals_dot.append(dots)
        all_colors.extend([color] * len(pts_tri))

        # Collect 2D points for bounding box
        flat_pts = pts_tri.reshape(-1, 2)
        if "left" in link_name:
            side_points["left"].append(flat_pts)
        elif "right" in link_name:
            side_points["right"].append(flat_pts)

    if not all_tris_2d:
        return result, side_points

    all_tris_2d = np.concatenate(all_tris_2d)
    all_depths = np.concatenate(all_depths)
    all_normals_dot = np.concatenate(all_normals_dot)

    # Painter's algorithm
    order = np.argsort(-all_depths)
    for idx in order:
        tri = all_tris_2d[idx].astype(np.int32)
        shade = 0.4 + 0.6 * all_normals_dot[idx]
        color = all_colors[idx]
        shaded = tuple(int(c * shade) for c in color)
        cv2.fillPoly(result, [tri], shaded)

    return result, side_points


def compute_bbox(pts_list, h, w, padding=0.25, min_size=80):
    """Compute bounding box from list of 2D point arrays."""
    if not pts_list:
        return None
    all_pts = np.concatenate(pts_list)
    if len(all_pts) == 0:
        return None
    x1, y1 = all_pts.min(axis=0)
    x2, y2 = all_pts.max(axis=0)
    pad_x = max((x2 - x1) * padding, min_size * 0.5)
    pad_y = max((y2 - y1) * padding, min_size * 0.5)
    return (
        int(max(0, x1 - pad_x)),
        int(max(0, y1 - pad_y)),
        int(min(w, x2 + pad_x)),
        int(min(h, y2 + pad_y)),
    )


def crop_and_resize(img, bbox, target_size=320):
    """Crop image region and resize to square."""
    if bbox is None:
        placeholder = np.zeros((target_size, target_size, 3), dtype=np.uint8)
        cv2.putText(placeholder, "Not visible", (10, target_size // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 1)
        return placeholder
    x1, y1, x2, y2 = bbox
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return np.zeros((target_size, target_size, 3), dtype=np.uint8)
    # Resize keeping aspect ratio, pad to square
    ch, cw = crop.shape[:2]
    scale = target_size / max(ch, cw)
    new_w, new_h = int(cw * scale), int(ch * scale)
    resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    y_off = (target_size - new_h) // 2
    x_off = (target_size - new_w) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    return canvas


def draw_legend(img, x0, y0):
    """Draw color legend on image."""
    for i, (name, color) in enumerate(LEGEND_ENTRIES):
        y = y0 + i * 20
        cv2.rectangle(img, (x0, y), (x0 + 14, y + 14), color, -1)
        cv2.rectangle(img, (x0, y), (x0 + 14, y + 14), (255, 255, 255), 1)
        cv2.putText(img, name, (x0 + 20, y + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)


def draw_bbox_on_image(img, bbox, label, color=(0, 255, 0)):
    """Draw bounding box rectangle with label."""
    if bbox is None:
        return
    x1, y1, x2, y2 = bbox
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    cv2.putText(img, label, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def print_hand_state(hs, q, hand_type):
    """Print hand_state diagnostic table to console."""
    print(f"\n{'='*60}")
    print(f"Hand State Diagnostic (type: {hand_type})")
    print(f"{'='*60}")

    if hand_type == "brainco":
        labels_raw = ["thumb_close", "thumb_tilt", "index", "middle", "ring", "little"]
    else:
        labels_raw = ["index", "middle", "ring", "little", "thumb_close", "thumb_tilt"]

    for side_name, offset, q_base in [("LEFT", 0, 29), ("RIGHT", 6, 48)]:
        print(f"\n  {side_name} hand (raw hand_state):")
        for i, label in enumerate(labels_raw):
            print(f"    hs[{offset+i:2d}] {label:>12s} = {hs[offset+i]:.4f}")

        print(f"  {side_name} hand (URDF q values):")
        finger_q = [
            ("index_1",   q_base + 0),
            ("index_2",   q_base + 1),
            ("little_1",  q_base + 2),
            ("little_2",  q_base + 3),
            ("middle_1",  q_base + 4),
            ("middle_2",  q_base + 5),
            ("ring_1",    q_base + 6),
            ("ring_2",    q_base + 7),
            ("thumb_1",   q_base + 8),
            ("thumb_2",   q_base + 9),
            ("thumb_3",   q_base + 10),
            ("thumb_4",   q_base + 11),
        ]
        for name, qi in finger_q:
            print(f"    q[{qi:2d}] {name:>10s} = {q[qi]:+.4f} rad ({np.degrees(q[qi]):+.1f} deg)")

    print(f"{'='*60}\n")


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


def render_one_frame(img, rq, hs, model, data_pin, mesh_cache, hand_type, params,
                     zoom_size=320):
    """Render one frame and return (full_render, left_crop, right_crop, q)."""
    q = build_q(model, rq, hs, hand_type=hand_type)
    transforms = do_fk(model, data_pin, q)

    rendered, side_points = render_hand_triangles(
        img, mesh_cache, transforms, params)

    h, w = img.shape[:2]
    left_bbox = compute_bbox(side_points["left"], h, w)
    right_bbox = compute_bbox(side_points["right"], h, w)

    # Draw bounding boxes on full render
    draw_bbox_on_image(rendered, left_bbox, "LEFT", (255, 180, 0))
    draw_bbox_on_image(rendered, right_bbox, "RIGHT", (0, 180, 255))

    left_crop = crop_and_resize(rendered, left_bbox, zoom_size)
    right_crop = crop_and_resize(rendered, right_bbox, zoom_size)

    return rendered, left_crop, right_crop, q


def compose_output(full_render, left_crop, right_crop, frame_idx,
                   human_full=None, human_left=None, human_right=None):
    """Compose final output image layout."""
    h, w = full_render.shape[:2]
    zoom_size = left_crop.shape[0]
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Label the crops
    cv2.putText(left_crop, f"LEFT f{frame_idx}", (5, 20), font, 0.5, (255, 255, 255), 1)
    cv2.putText(right_crop, f"RIGHT f{frame_idx}", (5, 20), font, 0.5, (255, 255, 255), 1)

    # Top row: left zoom + right zoom
    top_row = np.hstack([left_crop, right_crop])
    # Scale full render to match top row width
    top_w = top_row.shape[1]
    scale = top_w / w
    full_resized = cv2.resize(full_render, (top_w, int(h * scale)))
    cv2.putText(full_resized, f"Frame {frame_idx}", (5, 20), font, 0.5, (255, 255, 255), 1)

    # Draw legend on full render
    draw_legend(full_resized, full_resized.shape[1] - 90, 10)

    panels = [top_row, full_resized]

    if human_full is not None:
        human_resized = cv2.resize(human_full, (top_w, int(h * scale)))
        cv2.putText(human_resized, f"Human overlay f{frame_idx}", (5, 20), font, 0.5, (255, 255, 255), 1)

        # Human crops
        cv2.putText(human_left, f"Human LEFT f{frame_idx}", (5, 20), font, 0.5, (255, 255, 255), 1)
        cv2.putText(human_right, f"Human RIGHT f{frame_idx}", (5, 20), font, 0.5, (255, 255, 255), 1)
        human_top = np.hstack([human_left, human_right])
        panels.append(human_top)
        panels.append(human_resized)

    return np.vstack(panels)


def main():
    parser = argparse.ArgumentParser(description="Hand mesh debug rendering")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--frame", type=int, default=30,
                        help="Single frame index (default 30)")
    parser.add_argument("--frames", type=str, default=None,
                        help="Comma-separated frame indices for animation strip")
    parser.add_argument("--task", type=str, default=None,
                        help="Task name (default: ACTIVE_TASK)")
    parser.add_argument("--human", action="store_true",
                        help="Also render human overlay hand")
    parser.add_argument("--zoom-size", type=int, default=320,
                        help="Size of zoomed crop (default 320)")
    args = parser.parse_args()

    os.makedirs(HAND_DEBUG_OUTPUT_DIR, exist_ok=True)

    # Resolve task and hand type
    task_name = args.task or ACTIVE_TASK
    hand_type = get_hand_type(task_name)
    print(f"Task: {task_name}")
    print(f"Hand type: {hand_type}")

    # Resolve data directory
    if args.task:
        from config import DATASET_ROOT
        data_dir = os.path.join(DATASET_ROOT, args.task)
    else:
        data_dir = None

    # Load episode
    print(f"Loading episode {args.episode}...")
    video_path, from_ts, to_ts, ep_df = load_episode_info(args.episode, data_dir=data_dir)
    print(f"  {len(ep_df)} frames, video: {os.path.basename(video_path)}")

    # Load G1 URDF + meshes (filtered to hand+arm only)
    print("Loading G1 URDF...")
    model_g1 = pin.buildModelFromUrdf(G1_URDF, pin.JointModelFreeFlyer())
    data_g1 = model_g1.createData()
    link_meshes_g1 = parse_urdf_meshes(G1_URDF)
    mesh_cache_full = preload_meshes(link_meshes_g1, MESH_DIR, skip_set=SKIP_MESHES,
                                     subsample=1)

    # Filter to hand+arm links only
    all_patterns = []
    for pats in HAND_ARM_PATTERNS.values():
        all_patterns.extend(pats)
    mesh_cache_g1 = match_links(mesh_cache_full, all_patterns)
    print(f"  G1 hand+arm links: {len(mesh_cache_g1)} "
          f"({', '.join(sorted(mesh_cache_g1.keys()))})")

    # Optionally load human URDF
    model_human = data_human = mesh_cache_human = None
    if args.human:
        print("Loading human URDF...")
        model_human = pin.buildModelFromUrdf(HUMAN_URDF, pin.JointModelFreeFlyer())
        data_human = model_human.createData()
        link_meshes_human = parse_urdf_meshes(HUMAN_URDF)
        mesh_cache_human_full = preload_meshes(
            link_meshes_human, MESH_DIR, skip_set={"head_link"}, subsample=1)
        mesh_cache_human = match_links(mesh_cache_human_full, all_patterns)
        print(f"  Human hand+arm links: {len(mesh_cache_human)}")

    # Determine frames to render
    if args.frames:
        frame_indices = [int(f.strip()) for f in args.frames.split(",")]
    else:
        frame_indices = [args.frame]

    outputs = []
    for frame_idx in frame_indices:
        frame_row = ep_df[ep_df["frame_index"] == frame_idx]
        if len(frame_row) == 0:
            print(f"Frame {frame_idx} not found, skipping")
            continue

        rq = np.array(frame_row.iloc[0]["observation.state.robot_q_current"],
                       dtype=np.float64)
        hs = np.array(frame_row.iloc[0]["observation.state.hand_state"],
                       dtype=np.float64)

        print(f"\nExtracting frame {frame_idx}...")
        img = extract_frame(video_path, from_ts, frame_idx)
        if img is None:
            print(f"  Failed to extract frame {frame_idx}")
            continue

        # Render G1 hand
        full_g1, left_g1, right_g1, q = render_one_frame(
            img, rq, hs, model_g1, data_g1, mesh_cache_g1,
            hand_type, BEST_PARAMS, args.zoom_size)

        # Print diagnostic
        print_hand_state(hs, q, hand_type)

        # Render human overlay if requested
        human_full = human_left = human_right = None
        if args.human and model_human is not None:
            from render_human_overlay import render_human_triangles, SKIN_COLOR
            q_h = build_q(model_human, rq, hs, hand_type=hand_type)
            transforms_h = do_fk(model_human, data_human, q_h)
            human_rendered = render_human_triangles(
                img, mesh_cache_human, transforms_h, BEST_PARAMS, color=SKIN_COLOR)

            # Compute human hand bounding boxes (reuse G1 projection for bbox)
            _, human_side_pts = render_hand_triangles(
                img, mesh_cache_human, transforms_h, BEST_PARAMS)
            h, w = img.shape[:2]
            h_left_bbox = compute_bbox(human_side_pts["left"], h, w)
            h_right_bbox = compute_bbox(human_side_pts["right"], h, w)
            draw_bbox_on_image(human_rendered, h_left_bbox, "LEFT", (255, 180, 0))
            draw_bbox_on_image(human_rendered, h_right_bbox, "RIGHT", (0, 180, 255))
            human_full = human_rendered
            human_left = crop_and_resize(human_rendered, h_left_bbox, args.zoom_size)
            human_right = crop_and_resize(human_rendered, h_right_bbox, args.zoom_size)

        output = compose_output(full_g1, left_g1, right_g1, frame_idx,
                                human_full, human_left, human_right)
        outputs.append((frame_idx, output))

    if not outputs:
        print("No frames rendered")
        return

    # Save output
    if len(outputs) == 1:
        frame_idx, output = outputs[0]
        tag = f"ep{args.episode}_f{frame_idx}"
        if args.human:
            tag += "_human"
        out_path = os.path.join(HAND_DEBUG_OUTPUT_DIR, f"{tag}.png")
        cv2.imwrite(out_path, output)
        print(f"\nSaved: {out_path}")
    else:
        # Multi-frame: stack horizontally (just the zoom crops)
        strips_left = []
        strips_right = []
        for frame_idx, _ in outputs:
            frame_row = ep_df[ep_df["frame_index"] == frame_idx]
            rq = np.array(frame_row.iloc[0]["observation.state.robot_q_current"],
                           dtype=np.float64)
            hs = np.array(frame_row.iloc[0]["observation.state.hand_state"],
                           dtype=np.float64)
            img = extract_frame(video_path, from_ts, frame_idx)
            if img is None:
                continue
            _, left_crop, right_crop, _ = render_one_frame(
                img, rq, hs, model_g1, data_g1, mesh_cache_g1,
                hand_type, BEST_PARAMS, args.zoom_size)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(left_crop, f"f{frame_idx}", (5, 20), font, 0.5, (255, 255, 255), 1)
            cv2.putText(right_crop, f"f{frame_idx}", (5, 20), font, 0.5, (255, 255, 255), 1)
            strips_left.append(left_crop)
            strips_right.append(right_crop)

        strip = np.vstack([np.hstack(strips_left), np.hstack(strips_right)])
        font = cv2.FONT_HERSHEY_SIMPLEX
        # Add row labels
        label_w = 60
        label_col = np.zeros((strip.shape[0], label_w, 3), dtype=np.uint8)
        cv2.putText(label_col, "LEFT", (5, args.zoom_size // 2),
                    font, 0.5, (255, 255, 255), 1)
        cv2.putText(label_col, "RIGHT", (5, args.zoom_size + args.zoom_size // 2),
                    font, 0.5, (255, 255, 255), 1)
        strip = np.hstack([label_col, strip])

        frames_str = "_".join(str(fi) for fi, _ in outputs)
        tag = f"ep{args.episode}_frames_{frames_str}"
        out_path = os.path.join(HAND_DEBUG_OUTPUT_DIR, f"{tag}.png")
        cv2.imwrite(out_path, strip)
        print(f"\nSaved: {out_path}")

        # Also save individual composites
        for frame_idx, output in outputs:
            ind_path = os.path.join(HAND_DEBUG_OUTPUT_DIR,
                                     f"ep{args.episode}_f{frame_idx}.png")
            cv2.imwrite(ind_path, output)
            print(f"  Saved: {ind_path}")


if __name__ == "__main__":
    main()
