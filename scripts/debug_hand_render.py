"""
Debug hand rendering: front orthographic view of just hand links.
Side-by-side comparison of G1 Inspire hand vs Human hand meshes.

Prints hand_state, q values for hand joints, and FK transforms for each hand link.

Usage:
  python scripts/debug_hand_render.py
  python scripts/debug_hand_render.py --episode 0 --frame 100
"""

import sys
import os
import argparse
import numpy as np
import pandas as pd
import cv2
import xml.etree.ElementTree as ET
import pinocchio as pin
from stl import mesh as stl_mesh

sys.stdout.reconfigure(line_buffering=True)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

G1_URDF = os.path.join(BASE_DIR, "data", "mesh",
                       "g1_29dof_rev_1_0_with_inspire_hand_FTP.urdf")
HUMAN_URDF = os.path.join(BASE_DIR, "data", "mesh",
                          "human_arm_overlay.urdf")
MESH_DIR = os.path.join(BASE_DIR, "data", "mesh", "meshes")
DATA_DIR = os.path.join(BASE_DIR, "data", "video", "G1_WBT_Brainco_Make_The_Bed")
OUTPUT_DIR = os.path.join(BASE_DIR, "test_results", "human_overlay")

# Hand links: wrist_yaw_link and everything downstream
HAND_LINK_PREFIXES = [
    "wrist_yaw_link",
    "base_link",
    "palm_force_sensor",
    "thumb_1", "thumb_2", "thumb_3", "thumb_4",
    "thumb_force_sensor_1", "thumb_force_sensor_2",
    "thumb_force_sensor_3", "thumb_force_sensor_4",
    "index_1", "index_2",
    "index_force_sensor_1", "index_force_sensor_2", "index_force_sensor_3",
    "middle_1", "middle_2",
    "middle_force_sensor_1", "middle_force_sensor_2", "middle_force_sensor_3",
    "ring_1", "ring_2",
    "ring_force_sensor_1", "ring_force_sensor_2", "ring_force_sensor_3",
    "little_1", "little_2",
    "little_force_sensor_1", "little_force_sensor_2", "little_force_sensor_3",
]


def is_hand_link(name):
    """Check if a link name is a hand link (left_ or right_ prefix + hand suffix)."""
    for side in ["left_", "right_"]:
        for suffix in HAND_LINK_PREFIXES:
            if name == side + suffix:
                return True
    return False


# Per-link colors for G1 Inspire hand (BGR)
G1_HAND_COLORS = {
    # Left hand - warm blues/purples
    "left_wrist_yaw_link":  (255, 100, 50),
    "left_base_link":       (255, 150, 50),
    "left_palm_force_sensor": (255, 170, 80),
    "left_thumb_1":         (180, 80, 255),
    "left_thumb_2":         (160, 60, 240),
    "left_thumb_3":         (140, 40, 220),
    "left_thumb_4":         (120, 20, 200),
    "left_index_1":         (50, 200, 255),
    "left_index_2":         (30, 180, 240),
    "left_middle_1":        (50, 255, 200),
    "left_middle_2":        (30, 240, 180),
    "left_ring_1":          (100, 255, 100),
    "left_ring_2":          (80, 240, 80),
    "left_little_1":        (200, 255, 50),
    "left_little_2":        (180, 240, 30),
    # Right hand - warm oranges/reds
    "right_wrist_yaw_link": (50, 100, 255),
    "right_base_link":      (50, 150, 255),
    "right_palm_force_sensor": (80, 170, 255),
    "right_thumb_1":        (255, 80, 180),
    "right_thumb_2":        (240, 60, 160),
    "right_thumb_3":        (220, 40, 140),
    "right_thumb_4":        (200, 20, 120),
    "right_index_1":        (255, 200, 50),
    "right_index_2":        (240, 180, 30),
    "right_middle_1":       (200, 255, 50),
    "right_middle_2":       (180, 240, 30),
    "right_ring_1":         (100, 255, 100),
    "right_ring_2":         (80, 240, 80),
    "right_little_1":       (50, 255, 200),
    "right_little_2":       (30, 240, 180),
}

# Human hand uses same link names for fingers (no force sensors)
HUMAN_HAND_COLORS = dict(G1_HAND_COLORS)  # reuse the same color scheme


def build_q(model, rq, hand_state=None):
    """Map dataset rq (36) + hand_state (12) to URDF q (60).
    Works for both G1 Inspire and human URDF (same joint layout).
    """
    q = pin.neutral(model)
    q[0:3] = rq[0:3]
    q[3], q[4], q[5], q[6] = rq[4], rq[5], rq[6], rq[3]
    q[7:29] = rq[7:29]
    q[41:48] = rq[29:36]

    if hand_state is not None:
        hs = hand_state
        # Left hand q[29:41]
        q[29] = hs[0] * 1.4381
        q[30] = hs[0] * 1.4381 * 1.0843
        q[31] = hs[3] * 1.4381
        q[32] = hs[3] * 1.4381 * 1.0843
        q[33] = hs[1] * 1.4381
        q[34] = hs[1] * 1.4381 * 1.0843
        q[35] = hs[2] * 1.4381
        q[36] = hs[2] * 1.4381 * 1.0843
        q[37] = hs[5] * 1.1641
        q[38] = hs[4] * 0.5864
        q[39] = hs[4] * 0.5864 * 0.8024
        q[40] = hs[4] * 0.5864 * 0.8024 * 0.9487
        # Right hand q[48:60]
        q[48] = hs[6] * 1.4381
        q[49] = hs[6] * 1.4381 * 1.0843
        q[50] = hs[9] * 1.4381
        q[51] = hs[9] * 1.4381 * 1.0843
        q[52] = hs[7] * 1.4381
        q[53] = hs[7] * 1.4381 * 1.0843
        q[54] = hs[8] * 1.4381
        q[55] = hs[8] * 1.4381 * 1.0843
        q[56] = hs[11] * 1.1641
        q[57] = hs[10] * 0.5864
        q[58] = hs[10] * 0.5864 * 0.8024
        q[59] = hs[10] * 0.5864 * 0.8024 * 0.9487
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


def preload_meshes(link_meshes, mesh_dir, hand_only=True):
    cache = {}
    for link_name, filename in link_meshes.items():
        if hand_only and not is_hand_link(link_name):
            continue
        path = os.path.join(mesh_dir, filename)
        if not os.path.exists(path):
            continue
        m = stl_mesh.Mesh.from_file(path)
        verts = m.vectors
        flat = verts.reshape(-1, 3)
        valid_per_vert = np.all(np.isfinite(flat), axis=1)
        valid_per_tri = valid_per_vert.reshape(-1, 3).all(axis=1)
        tris = verts[valid_per_tri]
        flat_all = m.vectors.reshape(-1, 3)
        valid_all = np.all(np.isfinite(flat_all), axis=1)
        unique_verts = np.unique(flat_all[valid_all], axis=0)
        if len(tris) > 0:
            cache[link_name] = (tris, unique_verts)
    return cache


def render_hand_front(mesh_cache, transforms, color_map, title,
                      img_h=800, img_w=600, side="both"):
    """Render front orthographic view of hand links with per-link coloring.

    Args:
        side: "left", "right", or "both"
    """
    canvas = np.zeros((img_h, img_w, 3), dtype=np.uint8)
    canvas[:] = 30  # dark background

    # Filter to requested side(s)
    sides = []
    if side in ("left", "both"):
        sides.append("left_")
    if side in ("right", "both"):
        sides.append("right_")

    # Gather all world points for bounding box
    all_pts = []
    link_data = []  # (depth, link_name, world_tris)

    for link_name, (tris, _) in mesh_cache.items():
        if not any(link_name.startswith(s) for s in sides):
            continue
        if link_name not in transforms:
            continue

        t_link, R_link = transforms[link_name]
        flat = tris.reshape(-1, 3)
        world = (R_link @ flat.T).T + t_link
        all_pts.append(world)
        depth = t_link[1]  # Y = depth for front view
        link_data.append((depth, link_name, world.reshape(-1, 3, 3)))

    if not all_pts:
        cv2.putText(canvas, title, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, "No hand meshes found!", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        return canvas

    all_pts = np.vstack(all_pts)

    # Front view: X->right, Z->up (looking from +Y toward -Y)
    x_min, x_max = all_pts[:, 0].min(), all_pts[:, 0].max()
    z_min, z_max = all_pts[:, 2].min(), all_pts[:, 2].max()

    # Auto-zoom: scale to fit with margin
    margin = 80
    x_range = x_max - x_min
    z_range = z_max - z_min
    scale = min((img_w - 2 * margin) / max(x_range, 0.001),
                (img_h - 2 * margin) / max(z_range, 0.001))

    cx = img_w / 2
    cz = img_h / 2
    x_center = (x_min + x_max) / 2
    z_center = (z_min + z_max) / 2

    def project(pts3d):
        sx = (pts3d[:, 0] - x_center) * scale + cx
        sy = -(pts3d[:, 2] - z_center) * scale + cz
        return np.stack([sx, sy], axis=1)

    # Sort by depth (far first = painter's algorithm)
    link_data.sort(key=lambda x: x[0])

    for _, link_name, world_tris in link_data:
        n_tri = world_tris.shape[0]
        pts2d = project(world_tris.reshape(-1, 3)).astype(np.int32)
        tris_2d = pts2d.reshape(n_tri, 3, 2)

        color = color_map.get(link_name, (128, 128, 128))
        fill_color = tuple(max(0, c - 30) for c in color)

        for tri in tris_2d:
            cv2.fillConvexPoly(canvas, tri, fill_color)
        # Draw edges every few triangles for visibility
        cv2.polylines(canvas, tris_2d[::3], True, color, 1, cv2.LINE_AA)

    # Label each link at its origin
    for link_name in mesh_cache:
        if not any(link_name.startswith(s) for s in sides):
            continue
        if link_name not in transforms:
            continue
        t_link = transforms[link_name][0]
        pt2d = project(t_link.reshape(1, 3))[0].astype(int)
        color = color_map.get(link_name, (200, 200, 200))
        short = link_name.replace("left_", "L_").replace("right_", "R_")
        short = short.replace("_force_sensor", "_fs")
        cv2.circle(canvas, (pt2d[0], pt2d[1]), 3, color, -1)
        cv2.putText(canvas, short, (pt2d[0] + 5, pt2d[1] + 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, (255, 255, 255), 1, cv2.LINE_AA)

    # Bounding box info
    info = f"X:[{x_min:.4f},{x_max:.4f}] Z:[{z_min:.4f},{z_max:.4f}] scale={scale:.0f}"
    cv2.putText(canvas, info, (10, img_h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1, cv2.LINE_AA)

    cv2.putText(canvas, title, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
    return canvas


def load_episode_info(ep):
    meta = pd.read_parquet(os.path.join(DATA_DIR, "meta", "episodes",
                                         "chunk-000", "file-000.parquet"))
    ep_meta = meta[meta["episode_index"] == ep]
    if len(ep_meta) == 0:
        raise ValueError(f"Episode {ep} not found")
    ep_meta = ep_meta.iloc[0]

    file_idx = int(ep_meta["videos/observation.images.head_stereo_left/file_index"])
    from_ts = float(ep_meta["videos/observation.images.head_stereo_left/from_timestamp"])
    to_ts = float(ep_meta["videos/observation.images.head_stereo_left/to_timestamp"])

    data_fi = int(ep_meta.get("data/file_index", 0))
    parquet_path = os.path.join(DATA_DIR, "data", "chunk-000",
                                 f"file-{data_fi:03d}.parquet")
    df = pd.read_parquet(parquet_path)
    ep_df = df[df["episode_index"] == ep].sort_values("frame_index")

    return ep_df


def print_hand_diagnostics(q, hand_state, transforms):
    """Print detailed hand joint values and FK transforms."""
    print("\n" + "=" * 70)
    print("HAND STATE (raw from dataset, 12 values)")
    print("=" * 70)
    labels_hs = [
        "L_index", "L_middle", "L_ring", "L_little", "L_thumb_bend", "L_thumb_rot",
        "R_index", "R_middle", "R_ring", "R_little", "R_thumb_bend", "R_thumb_rot",
    ]
    for i, (label, val) in enumerate(zip(labels_hs, hand_state)):
        print(f"  hs[{i:2d}] {label:18s} = {val:+.6f}")

    print("\n" + "=" * 70)
    print("LEFT HAND q[29:41] (mapped joint angles)")
    print("=" * 70)
    left_labels = [
        "L_index_prox", "L_index_dist",
        "L_little_prox", "L_little_dist",
        "L_middle_prox", "L_middle_dist",
        "L_ring_prox", "L_ring_dist",
        "L_thumb_rot", "L_thumb_1", "L_thumb_2", "L_thumb_3",
    ]
    for i in range(12):
        print(f"  q[{29+i:2d}] {left_labels[i]:18s} = {q[29+i]:+.6f}")

    print("\n" + "=" * 70)
    print("RIGHT HAND q[48:60] (mapped joint angles)")
    print("=" * 70)
    right_labels = [
        "R_index_prox", "R_index_dist",
        "R_little_prox", "R_little_dist",
        "R_middle_prox", "R_middle_dist",
        "R_ring_prox", "R_ring_dist",
        "R_thumb_rot", "R_thumb_1", "R_thumb_2", "R_thumb_3",
    ]
    for i in range(12):
        print(f"  q[{48+i:2d}] {right_labels[i]:18s} = {q[48+i]:+.6f}")

    print("\n" + "=" * 70)
    print("FK TRANSFORMS FOR HAND LINKS")
    print("=" * 70)
    for link_name in sorted(transforms.keys()):
        if not is_hand_link(link_name):
            continue
        t, R = transforms[link_name]
        print(f"\n  {link_name}:")
        print(f"    translation = [{t[0]:+.6f}, {t[1]:+.6f}, {t[2]:+.6f}]")
        print(f"    rotation =")
        for row in R:
            print(f"      [{row[0]:+.8f}, {row[1]:+.8f}, {row[2]:+.8f}]")


def main():
    parser = argparse.ArgumentParser(description="Debug hand mesh rendering")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--frame", type=int, default=100)
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- Load episode data ---
    print(f"Loading episode {args.episode}, frame {args.frame}...")
    ep_df = load_episode_info(args.episode)
    print(f"  {len(ep_df)} frames in episode")

    frame_row = ep_df[ep_df["frame_index"] == args.frame]
    if len(frame_row) == 0:
        print(f"Frame {args.frame} not found, using first frame")
        frame_row = ep_df.iloc[[0]]
        args.frame = int(frame_row.iloc[0]["frame_index"])

    rq = np.array(frame_row.iloc[0]["observation.state.robot_q_current"], dtype=np.float64)
    hs = np.array(frame_row.iloc[0]["observation.state.hand_state"], dtype=np.float64)

    print(f"  rq shape: {rq.shape}, hand_state shape: {hs.shape}")

    # --- Load G1 Inspire URDF ---
    print("\nLoading G1 Inspire URDF...")
    model_g1 = pin.buildModelFromUrdf(G1_URDF, pin.JointModelFreeFlyer())
    data_g1 = model_g1.createData()
    link_meshes_g1 = parse_urdf_meshes(G1_URDF)
    mesh_cache_g1 = preload_meshes(link_meshes_g1, MESH_DIR, hand_only=True)
    print(f"  G1 hand links with mesh: {len(mesh_cache_g1)}")
    for name in sorted(mesh_cache_g1.keys()):
        n_tri = mesh_cache_g1[name][0].shape[0]
        print(f"    {name}: {n_tri} triangles")

    # --- Load Human URDF ---
    print("\nLoading Human URDF...")
    model_hu = pin.buildModelFromUrdf(HUMAN_URDF, pin.JointModelFreeFlyer())
    data_hu = model_hu.createData()
    link_meshes_hu = parse_urdf_meshes(HUMAN_URDF)
    mesh_cache_hu = preload_meshes(link_meshes_hu, MESH_DIR, hand_only=True)
    print(f"  Human hand links with mesh: {len(mesh_cache_hu)}")
    for name in sorted(mesh_cache_hu.keys()):
        n_tri = mesh_cache_hu[name][0].shape[0]
        print(f"    {name}: {n_tri} triangles")

    # --- FK for both models ---
    q_g1 = build_q(model_g1, rq, hs)
    transforms_g1 = do_fk(model_g1, data_g1, q_g1)

    q_hu = build_q(model_hu, rq, hs)
    transforms_hu = do_fk(model_hu, data_hu, q_hu)

    # --- Print diagnostics ---
    print("\n" + "#" * 70)
    print("# G1 INSPIRE HAND DIAGNOSTICS")
    print("#" * 70)
    print_hand_diagnostics(q_g1, hs, transforms_g1)

    print("\n\n" + "#" * 70)
    print("# HUMAN HAND DIAGNOSTICS")
    print("#" * 70)
    print_hand_diagnostics(q_hu, hs, transforms_hu)

    # --- Render front views ---
    IMG_H, IMG_W = 800, 600

    # Left hand: G1 vs Human
    print("\n\nRendering left hand views...")
    g1_left = render_hand_front(mesh_cache_g1, transforms_g1, G1_HAND_COLORS,
                                "G1 Inspire - Left Hand",
                                img_h=IMG_H, img_w=IMG_W, side="left")
    hu_left = render_hand_front(mesh_cache_hu, transforms_hu, HUMAN_HAND_COLORS,
                                "Human - Left Hand",
                                img_h=IMG_H, img_w=IMG_W, side="left")

    # Right hand: G1 vs Human
    print("Rendering right hand views...")
    g1_right = render_hand_front(mesh_cache_g1, transforms_g1, G1_HAND_COLORS,
                                 "G1 Inspire - Right Hand",
                                 img_h=IMG_H, img_w=IMG_W, side="right")
    hu_right = render_hand_front(mesh_cache_hu, transforms_hu, HUMAN_HAND_COLORS,
                                 "Human - Right Hand",
                                 img_h=IMG_H, img_w=IMG_W, side="right")

    # Compose 2x2 grid: [G1_Left | Human_Left] / [G1_Right | Human_Right]
    top = np.hstack([g1_left, hu_left])
    bot = np.hstack([g1_right, hu_right])
    combined = np.vstack([top, bot])

    # Add episode/frame info
    info_text = f"Episode {args.episode}, Frame {args.frame}"
    cv2.putText(combined, info_text, (10, combined.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

    out_path = os.path.join(OUTPUT_DIR,
                             f"debug_hand_ep{args.episode}_f{args.frame}.png")
    cv2.imwrite(out_path, combined)
    print(f"\nSaved 2x2 grid: {out_path}")

    # Also save individual panels
    for name, img in [("g1_left", g1_left), ("human_left", hu_left),
                       ("g1_right", g1_right), ("human_right", hu_right)]:
        p = os.path.join(OUTPUT_DIR,
                          f"debug_hand_{name}_ep{args.episode}_f{args.frame}.png")
        cv2.imwrite(p, img)
        print(f"  Saved: {p}")

    print("\nDone!")


if __name__ == "__main__":
    main()
