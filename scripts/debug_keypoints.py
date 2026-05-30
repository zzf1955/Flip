"""
Debug visualization for keypoint calibration data.

For each annotated frame, generates a 6-panel row:
  1. Annotated image (red dots on alpha!=255 keypoints)
  2. Video frame extracted from dataset
  3. Camera perspective (pinhole projection with mesh + keypoints)
  4. Front mesh view (orthographic)
  5. Side mesh view (orthographic)
  6. Top mesh view (orthographic, from above)

Usage:
  python scripts/debug_keypoints.py --manifest data/4points/manifest.json
"""

import sys, os, json, argparse, time
import numpy as np
import cv2
import pandas as pd
import pinocchio as pin
import av
from stl import mesh as stl_mesh
from scipy import ndimage

sys.stdout.reconfigure(line_buffering=True)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "scripts"))

from config import (G1_URDF, MESH_DIR, SKIP_MESHES, BEST_PARAMS, DATASET_ROOT,
                     OUTPUT_DIR, get_hand_type, get_skip_meshes)
from video_inpaint import (build_q, do_fk, parse_urdf_meshes, make_camera)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.collections import PolyCollection

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

# Active keypoints (set by main() from manifest)
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

# ── Body part coloring ──
BODY_GROUPS = {
    'left_leg':  {'left_hip_pitch_link', 'left_hip_roll_link', 'left_hip_yaw_link',
                  'left_knee_link', 'left_ankle_pitch_link', 'left_ankle_roll_link'},
    'right_leg': {'right_hip_pitch_link', 'right_hip_roll_link', 'right_hip_yaw_link',
                  'right_knee_link', 'right_ankle_pitch_link', 'right_ankle_roll_link'},
    'torso':     {'pelvis', 'waist_yaw_link', 'waist_roll_link', 'torso_link'},
    'left_arm':  {'left_shoulder_pitch_link', 'left_shoulder_roll_link',
                  'left_shoulder_yaw_link', 'left_elbow_link',
                  'left_wrist_yaw_link', 'left_wrist_roll_link', 'left_wrist_pitch_link'},
    'right_arm': {'right_shoulder_pitch_link', 'right_shoulder_roll_link',
                  'right_shoulder_yaw_link', 'right_elbow_link',
                  'right_wrist_yaw_link', 'right_wrist_roll_link', 'right_wrist_pitch_link'},
}
GROUP_COLORS_HEX = {
    'left_leg': '#5599DD', 'right_leg': '#3377BB',
    'torso': '#999999', 'head': '#DDAA44',
    'left_arm': '#55CC55', 'right_arm': '#339933',
    'left_hand': '#FF7755', 'right_hand': '#DD5533',
    'other': '#BBBBBB',
}


def _hex_to_bgr(h):
    """'#RRGGBB' -> (B, G, R) tuple."""
    r, g, b = int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16)
    return (b, g, r)


# Pre-compute BGR lookup
GROUP_COLORS_BGR = {k: _hex_to_bgr(v) for k, v in GROUP_COLORS_HEX.items()}


def get_color_hex(link_name):
    if link_name == 'head_link':
        return GROUP_COLORS_HEX['head']
    for group, links in BODY_GROUPS.items():
        if link_name in links:
            return GROUP_COLORS_HEX[group]
    if 'left' in link_name and any(p in link_name for p in
            ['thumb', 'index', 'middle', 'ring', 'little', 'base_link', 'palm']):
        return GROUP_COLORS_HEX['left_hand']
    if 'right' in link_name and any(p in link_name for p in
            ['thumb', 'index', 'middle', 'ring', 'little', 'base_link', 'palm']):
        return GROUP_COLORS_HEX['right_hand']
    return GROUP_COLORS_HEX['other']


def get_color_bgr(link_name):
    return _hex_to_bgr(get_color_hex(link_name))


# ============================================================
# Vectorized oblique projection
# ============================================================

def _build_rotation_matrix(azim_deg, elev_deg):
    """Build combined rotation matrix for oblique projection."""
    az = np.radians(azim_deg)
    el = np.radians(elev_deg)
    Rz = np.array([[ np.cos(az), -np.sin(az), 0],
                    [ np.sin(az),  np.cos(az), 0],
                    [ 0,           0,          1]])
    Rx = np.array([[1, 0,           0],
                    [0, np.cos(el), -np.sin(el)],
                    [0, np.sin(el),  np.cos(el)]])
    return Rx @ Rz


def oblique_project_batch(pts, R):
    """Batch oblique projection. pts: (N, 3), R: (3, 3).

    Returns screen (N, 2) and depth (N,).
    Screen X is negated so robot-left appears on viewer-left (facing the robot).
    """
    rotated = pts @ R.T  # (N, 3)
    screen = rotated[:, [0, 2]].copy()
    screen[:, 0] *= -1  # fix left-right mirror
    depth = rotated[:, 1]
    return screen, depth


# ============================================================
# Keypoint detection
# ============================================================

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
    # Merge nearby clusters (<15px)
    merged = True
    while merged:
        merged = False
        for i in range(len(centers)):
            for j in range(i + 1, len(centers)):
                dist = np.sqrt((centers[i][0] - centers[j][0])**2 +
                               (centers[i][1] - centers[j][1])**2)
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
            f"Expected {expected_count} clusters, found {len(centers)} in {png_path}")
    centers.sort(key=lambda c: c[0])
    return np.array(centers, dtype=np.float64)


# ============================================================
# Video frame extraction
# ============================================================

def find_video_path(task, episode):
    """Find the video path, handling both cam_0 and head_stereo_left."""
    data_dir = os.path.join(DATASET_ROOT, task)
    meta = pd.read_parquet(
        os.path.join(data_dir, "meta", "episodes", "chunk-000", "file-000.parquet"))
    ep_meta = meta[meta["episode_index"] == episode].iloc[0]
    for cam_name in ["cam_0", "head_stereo_left"]:
        col = f"videos/observation.images.{cam_name}/file_index"
        if col in ep_meta.index:
            file_idx = int(ep_meta[col])
            video_path = os.path.join(data_dir, "videos",
                                       f"observation.images.{cam_name}",
                                       "chunk-000", f"file-{file_idx:03d}.mp4")
            if os.path.exists(video_path):
                return video_path, cam_name
    raise FileNotFoundError(f"No video found for {task} ep{episode}")


def extract_video_frame(video_path, frame_idx):
    """Extract a single frame from video by sequential count."""
    container = av.open(video_path)
    count = 0
    for frame in container.decode(video=0):
        if count == frame_idx:
            img = frame.to_ndarray(format='bgr24')
            container.close()
            return img
        count += 1
    container.close()
    raise RuntimeError(f"Frame {frame_idx} not found (video has {count} frames)")


# ============================================================
# FK loading
# ============================================================

def load_fk_transforms(task, episode, frame_idx, model, data_pin):
    """Load joint state from parquet and compute FK transforms."""
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
    return transforms


# ============================================================
# Mesh loading
# ============================================================

def load_mesh_world(transforms, link_meshes, skip_set):
    """Load all mesh triangles transformed to world frame.

    Always includes head_link for debug visualization.
    """
    debug_skip = skip_set - {"head_link"}
    all_tris = {}
    for link_name, filename in link_meshes.items():
        if link_name in debug_skip:
            continue
        if link_name not in transforms:
            continue
        path = os.path.join(MESH_DIR, filename)
        if not os.path.exists(path):
            continue
        m = stl_mesh.Mesh.from_file(path)
        tris = m.vectors.copy()
        t_link, R_link = transforms[link_name]
        flat = tris.reshape(-1, 3)
        world = (R_link @ flat.T).T + t_link
        all_tris[link_name] = world.reshape(-1, 3, 3)
    return all_tris


def get_camera_world_pos(transforms):
    """Compute camera world position from BEST_PARAMS + torso_link transform."""
    ref_t, ref_R = transforms["torso_link"]
    offset = np.array([BEST_PARAMS["dx"], BEST_PARAMS["dy"], BEST_PARAMS["dz"]])
    return ref_t + ref_R @ offset


def get_kp_world(transforms):
    """Compute 4 keypoint world positions."""
    pts = []
    for name, link_name, local_offset in KEYPOINTS:
        if link_name in transforms:
            t_link, R_link = transforms[link_name]
            pts.append(R_link @ local_offset + t_link)
        else:
            pts.append(np.array([np.nan, np.nan, np.nan]))
    return np.array(pts, dtype=np.float64)


# ============================================================
# Vectorized 3-view mesh rendering
# ============================================================

def render_mesh_view(ax, all_tris, kp_world, cam_pos, view_name, azim, elev):
    """Render one orthographic mesh view with keypoints + camera (vectorized)."""
    R = _build_rotation_matrix(azim, elev)

    # Collect all triangles + per-triangle color
    tri_arrays = []
    color_list = []
    for link_name, tris in all_tris.items():
        tri_arrays.append(tris)
        c = get_color_hex(link_name)
        color_list.extend([c] * len(tris))

    if not tri_arrays:
        ax.set_title(view_name, fontsize=8, fontweight='bold')
        return

    all_tri = np.concatenate(tri_arrays, axis=0)  # (T, 3, 3)
    T = len(all_tri)

    # Batch projection: (T*3, 3) -> (T*3, 2)
    flat = all_tri.reshape(-1, 3)
    screen, depth = oblique_project_batch(flat, R)

    screen_tri = screen.reshape(T, 3, 2)
    depth_mean = depth.reshape(T, 3).mean(axis=1)

    # Sort by depth (painter's algorithm)
    order = np.argsort(depth_mean)
    sorted_polys = screen_tri[order]
    sorted_colors = [color_list[i] for i in order]

    pc = PolyCollection(sorted_polys, facecolors=sorted_colors,
                        edgecolors='none', linewidths=0, alpha=0.95)
    ax.add_collection(pc)

    # Camera position
    if cam_pos is not None:
        cam_s, _ = oblique_project_batch(cam_pos.reshape(1, 3), R)
        ax.plot(cam_s[0, 0], cam_s[0, 1], 's', color='blue', markersize=8,
                markeredgecolor='white', markeredgewidth=1.5, zorder=102)
        ax.annotate('CAM', (cam_s[0, 0], cam_s[0, 1]), fontsize=6, color='blue',
                    fontweight='bold', xytext=(4, -10), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='blue', alpha=0.85),
                    zorder=103)

    # Keypoints
    kp_s, _ = oblique_project_batch(kp_world, R)
    for i in range(len(kp_world)):
        if np.isnan(kp_s[i, 0]):
            continue
        ax.plot(kp_s[i, 0], kp_s[i, 1], 'o', color='red', markersize=7,
                markeredgecolor='white', markeredgewidth=1.5, zorder=100)
        ax.annotate(KP_NAMES[i], (kp_s[i, 0], kp_s[i, 1]), fontsize=6,
                    color='red', fontweight='bold',
                    xytext=(4, 4), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='red', alpha=0.85),
                    zorder=101)

    ax.set_aspect('equal')
    ax.set_title(view_name, fontsize=8, fontweight='bold')
    ax.tick_params(labelsize=4)
    ax.set_facecolor('#f0f0f0')
    ax.autoscale()
    xl, yl = ax.get_xlim(), ax.get_ylim()
    ax.set_xlim(xl[0] - 0.01, xl[1] + 0.01)
    ax.set_ylim(yl[0] - 0.01, yl[1] + 0.01)


# ============================================================
# Camera perspective rendering (pinhole projection)
# ============================================================

_CAM_VIEW_SKIP = {"head_link", "logo_link", "d435_link"}


def render_camera_view(all_tris, transforms, kp_world, h=480, w=640):
    """Render mesh from camera perspective using pinhole projection.

    Returns (h, w, 3) BGR image.
    """
    K, D, rvec, tvec, R_w2c, t_w2c, _fisheye = make_camera(BEST_PARAMS, transforms)
    t_flat = t_w2c.flatten()

    canvas = np.full((h, w, 3), 40, dtype=np.uint8)  # dark gray background

    # Collect triangles per-link (vectorized projection), skip head area
    all_depths = []
    all_pts2d = []
    all_colors = []

    for link_name, tris in all_tris.items():
        if link_name in _CAM_VIEW_SKIP:
            continue
        color_bgr = get_color_bgr(link_name)
        flat = tris.reshape(-1, 3)

        cam_pts = (R_w2c @ flat.T).T + t_flat
        z = cam_pts[:, 2]

        from camera_models import project_points_cv
        pts2d = project_points_cv(
            flat.reshape(-1, 1, 3).astype(np.float64), rvec, tvec, K, D, _fisheye)
        pts2d = pts2d.reshape(-1, 2)

        n_tris = len(tris)
        z_tri = z.reshape(n_tris, 3)
        pts_tri = pts2d.reshape(n_tris, 3, 2)

        z_valid = (z_tri > 0.05).all(axis=1)
        finite = np.all(np.isfinite(pts_tri.reshape(n_tris, -1)), axis=1)
        in_range = np.all(np.abs(pts_tri.reshape(n_tris, -1)) < 5000, axis=1)
        valid = z_valid & finite & in_range

        if valid.any():
            all_depths.append(z_tri[valid].mean(axis=1))
            all_pts2d.append(pts_tri[valid])
            all_colors.extend([color_bgr] * int(valid.sum()))

    if not all_depths:
        return canvas

    # Sort back-to-front
    depths = np.concatenate(all_depths)
    pts_all = np.concatenate(all_pts2d)  # (T, 3, 2)
    order = np.argsort(-depths)

    pts_sorted = pts_all[order].astype(np.int32)
    colors_sorted = [all_colors[i] for i in order]

    # Batch draw
    for i in range(len(pts_sorted)):
        cv2.fillPoly(canvas, [pts_sorted[i]], colors_sorted[i])

    # Draw keypoints
    kp_3d = kp_world.reshape(-1, 1, 3).astype(np.float64)
    kp_2d, _ = cv2.projectPoints(kp_3d, rvec, tvec, K, D)
    kp_2d = kp_2d.reshape(-1, 2)
    for i in range(len(kp_world)):
        if not np.all(np.isfinite(kp_2d[i])):
            continue
        x, y = int(kp_2d[i, 0]), int(kp_2d[i, 1])
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(canvas, (x, y), 6, (0, 0, 255), -1)
            cv2.circle(canvas, (x, y), 8, (255, 255, 255), 1)
            cv2.putText(canvas, KP_NAMES[i], (x + 8, y - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    return canvas


# ============================================================
# Main
# ============================================================

# ============================================================
# Hand debug rendering
# ============================================================

_FINGER_KEYWORDS = ['thumb', 'index', 'middle', 'ring', 'little']
_FINGER_COLORS = {
    'thumb':  '#DD3333',
    'index':  '#3366CC',
    'middle': '#33AA33',
    'ring':   '#DD8800',
    'little': '#8833AA',
    'base':   '#888888',
}


def _hand_link_color(link_name):
    """Per-finger color for hand links."""
    for finger, color in _FINGER_COLORS.items():
        if finger in link_name:
            return color
    return _FINGER_COLORS['base']


def _is_hand_link(link_name, side):
    """Check if link_name belongs to the given hand side."""
    if not link_name.startswith(side + '_'):
        return False
    rest = link_name[len(side) + 1:]
    return any(kw in rest for kw in _FINGER_KEYWORDS + ['base_link', 'palm'])


def _short_name(link_name, side):
    """Shorten link name for annotation: 'left_thumb_2' -> 'thumb_2'."""
    return link_name.replace(side + '_', '')


def render_hand_debug(task, episode, frame_idx, model, data_pin, link_meshes, out_dir):
    """Render left and right hands in multiple views with per-finger coloring."""
    import pandas as pd

    # Load FK
    transforms = load_fk_transforms(task, episode, frame_idx, model, data_pin)

    # Load hand_state from parquet for info display
    data_dir = os.path.join(DATASET_ROOT, task)
    meta = pd.read_parquet(
        os.path.join(data_dir, "meta", "episodes", "chunk-000", "file-000.parquet"))
    ep_meta = meta[meta["episode_index"] == episode].iloc[0]
    data_fi = int(ep_meta.get("data/file_index", 0))
    df = pd.read_parquet(
        os.path.join(data_dir, "data", "chunk-000", f"file-{data_fi:03d}.parquet"))
    ep_df = df[df["episode_index"] == episode].sort_values("frame_index")
    row = ep_df[ep_df["frame_index"] == frame_idx].iloc[0]
    hand_state = np.array(row["observation.state.hand_state"], dtype=np.float64)

    print(f"hand_state = {hand_state}")
    print(f"  Left:  idx={hand_state[0]:.2f} mid={hand_state[1]:.2f} "
          f"ring={hand_state[2]:.2f} lit={hand_state[3]:.2f} "
          f"thm_c={hand_state[4]:.2f} thm_t={hand_state[5]:.2f}")
    print(f"  Right: idx={hand_state[6]:.2f} mid={hand_state[7]:.2f} "
          f"ring={hand_state[8]:.2f} lit={hand_state[9]:.2f} "
          f"thm_c={hand_state[10]:.2f} thm_t={hand_state[11]:.2f}")

    # Load world-frame mesh for hand links only
    hand_tris = {'left': {}, 'right': {}}
    for link_name, filename in link_meshes.items():
        if link_name not in transforms:
            continue
        for side in ('left', 'right'):
            if _is_hand_link(link_name, side):
                path = os.path.join(MESH_DIR, filename)
                if not os.path.exists(path):
                    continue
                m = stl_mesh.Mesh.from_file(path)
                tris = m.vectors.copy()
                t_link, R_link = transforms[link_name]
                flat = tris.reshape(-1, 3)
                world = (R_link @ flat.T).T + t_link
                hand_tris[side][link_name] = world.reshape(-1, 3, 3)

    print(f"  Left hand: {len(hand_tris['left'])} links, "
          f"{sum(len(t) for t in hand_tris['left'].values())} tris")
    print(f"  Right hand: {len(hand_tris['right'])} links, "
          f"{sum(len(t) for t in hand_tris['right'].values())} tris")

    # Views for hands
    HAND_VIEWS = [
        ("Front", 0, 0),
        ("Side", 90, 0),
        ("Top", 0, -89),
    ]

    # Layout: 2 rows (left, right) × 3 views + 1 info column
    fig = plt.figure(figsize=(24, 12))
    gs = GridSpec(2, 4, figure=fig, width_ratios=[3, 3, 3, 2], wspace=0.1, hspace=0.15)

    for row, side in enumerate(['left', 'right']):
        tris = hand_tris[side]
        if not tris:
            continue

        for vi, (view_name, azim, elev) in enumerate(HAND_VIEWS):
            ax = fig.add_subplot(gs[row, vi])
            R = _build_rotation_matrix(azim, elev)

            # Collect + project
            tri_arrays = []
            color_list = []
            label_data = []  # (link_name, center_world)

            for link_name, link_tris in tris.items():
                tri_arrays.append(link_tris)
                c = _hand_link_color(link_name)
                color_list.extend([c] * len(link_tris))
                center = link_tris.reshape(-1, 3).mean(axis=0)
                label_data.append((link_name, center))

            all_tri = np.concatenate(tri_arrays, axis=0)
            T = len(all_tri)
            flat = all_tri.reshape(-1, 3)
            screen, depth = oblique_project_batch(flat, R)
            screen_tri = screen.reshape(T, 3, 2)
            depth_mean = depth.reshape(T, 3).mean(axis=1)
            order = np.argsort(depth_mean)

            pc = PolyCollection(screen_tri[order],
                                facecolors=[color_list[i] for i in order],
                                edgecolors='#333333', linewidths=0.05, alpha=0.95)
            ax.add_collection(pc)

            # Label key links (skip force_sensor to reduce clutter)
            for link_name, center in label_data:
                if 'force_sensor' in link_name:
                    continue
                cs, _ = oblique_project_batch(center.reshape(1, 3), R)
                short = _short_name(link_name, side)
                ax.annotate(short, (cs[0, 0], cs[0, 1]), fontsize=5,
                            color='#333333', ha='center',
                            bbox=dict(boxstyle='round,pad=0.1', fc='white',
                                      ec='gray', alpha=0.7),
                            zorder=50)

            ax.set_aspect('equal')
            ax.set_title(f"{side.capitalize()} — {view_name}", fontsize=9, fontweight='bold')
            ax.tick_params(labelsize=4)
            ax.set_facecolor('#f0f0f0')
            ax.autoscale()
            xl, yl = ax.get_xlim(), ax.get_ylim()
            ax.set_xlim(xl[0] - 0.005, xl[1] + 0.005)
            ax.set_ylim(yl[0] - 0.005, yl[1] + 0.005)

    # Info panel (right column, spanning both rows)
    ax_info = fig.add_subplot(gs[:, 3])
    ax_info.axis('off')

    hs = hand_state
    info_lines = [
        f"Task: {task}",
        f"Episode: {episode}  Frame: {frame_idx}",
        "",
        f"hand_state ({'Inspire: 0=closed 1=open' if 'Inspire' in task else 'BrainCo: 0=open 1=closed'}):",
        "",
        "LEFT HAND:",
        f"  [0] little:      {hs[0]:.3f}",
        f"  [1] ring:        {hs[1]:.3f}",
        f"  [2] middle:      {hs[2]:.3f}",
        f"  [3] index:       {hs[3]:.3f}",
        f"  [4] thumb_close: {hs[4]:.3f}",
        f"  [5] thumb_tilt:  {hs[5]:.3f}",
        "",
        "RIGHT HAND:",
        f"  [6] little:      {hs[6]:.3f}",
        f"  [7] ring:        {hs[7]:.3f}",
        f"  [8] middle:      {hs[8]:.3f}",
        f"  [9] index:       {hs[9]:.3f}",
        f"  [10] thumb_close: {hs[10]:.3f}",
        f"  [11] thumb_tilt:  {hs[11]:.3f}",
        "",
        "Finger colors:",
        "  thumb  = red",
        "  index  = blue",
        "  middle = green",
        "  ring   = orange",
        "  little = purple",
        "  base   = gray",
    ]
    ax_info.text(0.05, 0.95, '\n'.join(info_lines), transform=ax_info.transAxes,
                 fontsize=8, family='monospace', verticalalignment='top',
                 bbox=dict(boxstyle='round', fc='lightyellow', ec='gray', alpha=0.9))

    fig.suptitle(f"Hand Debug — {task} ep{episode} f{frame_idx}",
                 fontsize=11, fontweight='bold')

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"hand_ep{episode}_f{frame_idx}.png")
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Debug keypoint visualization")
    parser.add_argument("--manifest", type=str, default="data/4points/manifest.json")

    # Hand debug mode
    parser.add_argument("--hand-debug", action="store_true",
                        help="Render hand mesh debug views")
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--frame", type=int, default=0)

    args = parser.parse_args()

    print("Loading URDF...")
    model = pin.buildModelFromUrdf(G1_URDF, pin.JointModelFreeFlyer())
    data_pin = model.createData()
    link_meshes = parse_urdf_meshes(G1_URDF)

    if args.hand_debug:
        if args.task is None:
            parser.error("--hand-debug requires --task")
        out_dir = os.path.join(OUTPUT_DIR, "hand_debug")
        render_hand_debug(args.task, args.episode, args.frame,
                          model, data_pin, link_meshes, out_dir)
        return

    manifest_path = os.path.join(BASE_DIR, args.manifest)
    with open(manifest_path) as f:
        manifest = json.load(f)
    frames = manifest["frames"]
    n_frames = len(frames)

    # Determine active keypoints from manifest
    if "keypoints" in manifest:
        set_active_keypoints(manifest["keypoints"])
    else:
        set_active_keypoints(["L_thumb", "L_toe", "R_toe", "R_thumb"])

    n_kp = len(KEYPOINTS)
    manifest_dir = os.path.dirname(manifest_path)
    print(f"Manifest: {n_frames} frames, {n_kp} keypoints: {KP_NAMES}")

    out_dir = os.path.join(OUTPUT_DIR, "kp_debug")
    os.makedirs(out_dir, exist_ok=True)

    # 3 mesh view angles: front (facing robot), side (from right), top (from above)
    VIEWS = [
        ("Front",  0,   0),
        ("Side",  90,   0),
        ("Top",    0, -89),  # negative elev = looking from above
    ]

    t_total = time.time()

    for row_idx, entry in enumerate(frames):
        t_frame = time.time()
        png_path = os.path.join(manifest_dir, entry["image"])
        task = entry["task"]
        episode = entry["episode"]
        frame_idx = entry["frame"]
        label = os.path.splitext(entry["image"])[0]

        print(f"\n{'='*60}")
        print(f"[{row_idx+1}/{n_frames}] {label}")
        print(f"  task={task}  ep={episode}  frame={frame_idx}")

        # ── 6-column layout ──
        fig = plt.figure(figsize=(32, 6))
        gs = GridSpec(1, 6, figure=fig,
                      width_ratios=[3, 3, 3, 2, 2, 2], wspace=0.08)
        axes = [fig.add_subplot(gs[0, i]) for i in range(6)]

        # ── Panel 1: Annotated image with red dots ──
        try:
            img_bgra = cv2.imread(png_path, cv2.IMREAD_UNCHANGED)
            img_rgb = cv2.cvtColor(img_bgra[:, :, :3], cv2.COLOR_BGR2RGB)
            gt_pts = detect_keypoints_from_alpha(png_path, expected_count=n_kp)
            axes[0].imshow(img_rgb)
            for i, (x, y) in enumerate(gt_pts):
                axes[0].plot(x, y, 'o', color='red', markersize=10,
                             markeredgecolor='white', markeredgewidth=2, zorder=10)
                axes[0].annotate(
                    f"{KP_NAMES[i]} ({x:.0f},{y:.0f})", (x, y),
                    fontsize=7, color='red', fontweight='bold',
                    xytext=(8, -12), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='red', alpha=0.9),
                    zorder=11)
            axes[0].set_title(f"Annotated: {label}", fontsize=8, fontweight='bold')
            axes[0].axis('off')
        except Exception as e:
            axes[0].text(0.5, 0.5, str(e), ha='center', va='center',
                         fontsize=8, color='red', transform=axes[0].transAxes, wrap=True)
            axes[0].set_title("Annotated: ERROR", fontsize=8)
            axes[0].axis('off')

        # ── Panel 2: Video frame ──
        try:
            video_path, cam_name = find_video_path(task, episode)
            video_frame = extract_video_frame(video_path, frame_idx)
            video_rgb = cv2.cvtColor(video_frame, cv2.COLOR_BGR2RGB)
            axes[1].imshow(video_rgb)
            axes[1].set_title(f"Video: {cam_name} f={frame_idx}", fontsize=8, fontweight='bold')
            axes[1].axis('off')
        except Exception as e:
            axes[1].text(0.5, 0.5, str(e), ha='center', va='center',
                         fontsize=8, color='red', transform=axes[1].transAxes, wrap=True)
            axes[1].set_title("Video: ERROR", fontsize=8)
            axes[1].axis('off')

        # ── Load FK + mesh ──
        try:
            transforms = load_fk_transforms(task, episode, frame_idx, model, data_pin)
            skip_set = get_skip_meshes(get_hand_type(task))
            all_tris = load_mesh_world(transforms, link_meshes, skip_set)
            kp_world = get_kp_world(transforms)
            cam_pos = get_camera_world_pos(transforms)

            # ── Panel 3: Camera perspective ──
            cam_img = render_camera_view(all_tris, transforms, kp_world)
            cam_rgb = cv2.cvtColor(cam_img, cv2.COLOR_BGR2RGB)
            axes[2].imshow(cam_rgb)
            axes[2].set_title("Camera View", fontsize=8, fontweight='bold')
            axes[2].axis('off')

            # ── Panels 4-6: 3-view ──
            for vi, (view_name, azim, elev) in enumerate(VIEWS):
                render_mesh_view(axes[3 + vi], all_tris, kp_world, cam_pos,
                                 view_name, azim, elev)

            print(f"  OK: {len(all_tris)} links, {time.time()-t_frame:.1f}s")
        except Exception as e:
            for col in [2, 3, 4, 5]:
                axes[col].text(0.5, 0.5, str(e), ha='center', va='center',
                               fontsize=8, color='red', transform=axes[col].transAxes, wrap=True)
                axes[col].axis('off')
            print(f"  FK failed: {e}")

        fig.suptitle(
            f"{label}  |  {task}  ep={episode}  frame={frame_idx}",
            fontsize=9, fontweight='bold', y=0.98)

        out_path = os.path.join(out_dir, f"debug_{row_idx:02d}_{label}.png")
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {out_path}")

    elapsed = time.time() - t_total
    print(f"\n{'='*60}")
    print(f"All {n_frames} debug images saved to {out_dir}/  ({elapsed:.1f}s total)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
