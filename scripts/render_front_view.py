"""
Render G1 mesh from front view: with and without hands.
Each link colored differently for identification.
"""

import numpy as np
import cv2
import os
import xml.etree.ElementTree as ET
import pinocchio as pin
from stl import mesh as stl_mesh

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
URDF_PATH = os.path.join(BASE_DIR, "data", "g1_urdf", "g1_29dof_rev_1_0.urdf")
MESH_DIR = os.path.join(BASE_DIR, "data", "unitree_ros", "robots", "g1_description", "meshes")
OUTPUT_DIR = os.path.join(BASE_DIR, "test_results")

# Color map per link group
LINK_COLORS = {
    # Torso / waist
    "pelvis": (180, 180, 180),
    "waist_yaw_link": (160, 160, 160),
    "waist_roll_link": (140, 140, 140),
    "torso_link": (200, 200, 200),
    "head_link": (220, 200, 200),
    "logo_link": (100, 100, 100),
    "d435_link": (100, 100, 100),
    "mid360_link": (100, 100, 100),
    # Left arm - blue shades
    "left_shoulder_pitch_link": (255, 180, 80),
    "left_shoulder_roll_link": (255, 160, 60),
    "left_shoulder_yaw_link": (255, 140, 40),
    "left_elbow_link": (255, 120, 20),
    "left_wrist_roll_link": (255, 100, 0),
    "left_wrist_pitch_link": (255, 80, 0),
    "left_wrist_yaw_link": (255, 60, 0),
    "left_rubber_hand": (0, 0, 255),  # bright blue = hand
    # Right arm - orange shades
    "right_shoulder_pitch_link": (80, 180, 255),
    "right_shoulder_roll_link": (60, 160, 255),
    "right_shoulder_yaw_link": (40, 140, 255),
    "right_elbow_link": (20, 120, 255),
    "right_wrist_roll_link": (0, 100, 255),
    "right_wrist_pitch_link": (0, 80, 255),
    "right_wrist_yaw_link": (0, 60, 255),
    "right_rubber_hand": (255, 0, 0),  # bright red = hand
    # Left leg - green shades
    "left_hip_pitch_link": (80, 255, 80),
    "left_hip_roll_link": (60, 230, 60),
    "left_hip_yaw_link": (40, 210, 40),
    "left_knee_link": (20, 190, 20),
    "left_ankle_pitch_link": (0, 170, 0),
    "left_ankle_roll_link": (0, 150, 0),
    # Right leg - cyan shades
    "right_hip_pitch_link": (255, 255, 80),
    "right_hip_roll_link": (230, 230, 60),
    "right_hip_yaw_link": (210, 210, 40),
    "right_knee_link": (190, 190, 20),
    "right_ankle_pitch_link": (170, 170, 0),
    "right_ankle_roll_link": (150, 150, 0),
}


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


def render_front(link_meshes, mesh_dir, transforms, skip_set, title, img_h=1200, img_w=800):
    """Render front orthographic view with per-link colors."""
    canvas = np.zeros((img_h, img_w, 3), dtype=np.uint8)
    canvas[:] = 30  # dark background

    # Find bounding box of all mesh vertices to set up projection
    all_pts = []
    for link_name, filename in link_meshes.items():
        if link_name in skip_set or link_name not in transforms:
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
        R, t = transforms[link_name][1], transforms[link_name][0]
        world = (R @ flat.T).T + t
        all_pts.append(world)

    if not all_pts:
        return canvas

    all_pts = np.vstack(all_pts)
    # Front view: X=right, Z=up, Y=depth (looking from +Y toward -Y)
    y_min, y_max = all_pts[:, 1].min(), all_pts[:, 1].max()
    x_min, x_max = all_pts[:, 0].min(), all_pts[:, 0].max()
    z_min, z_max = all_pts[:, 2].min(), all_pts[:, 2].max()

    # Scale to fit image with margin
    margin = 60
    x_range = x_max - x_min
    z_range = z_max - z_min
    scale = min((img_w - 2 * margin) / max(x_range, 0.01),
                (img_h - 2 * margin) / max(z_range, 0.01))
    cx = img_w / 2
    cz = img_h / 2

    x_center = (x_min + x_max) / 2
    z_center = (z_min + z_max) / 2

    def project(pts3d):
        """Orthographic front projection: X->screen_x, Z->screen_y (inverted)"""
        sx = (pts3d[:, 0] - x_center) * scale + cx
        sy = -(pts3d[:, 2] - z_center) * scale + cz
        return np.stack([sx, sy], axis=1)

    # Sort links by average Y depth (far first = painter's algorithm)
    link_depths = []
    for link_name, filename in link_meshes.items():
        if link_name in skip_set or link_name not in transforms:
            continue
        path = os.path.join(mesh_dir, filename)
        if not os.path.exists(path):
            continue
        t = transforms[link_name][0]
        link_depths.append((t[1], link_name, filename))  # Y depth

    link_depths.sort(key=lambda x: x[0])  # far (negative Y) first

    for _, link_name, filename in link_depths:
        path = os.path.join(mesh_dir, filename)
        m = stl_mesh.Mesh.from_file(path)
        verts = m.vectors  # (n_tri, 3, 3)
        n_tri = verts.shape[0]

        R, t = transforms[link_name][1], transforms[link_name][0]
        flat = verts.reshape(-1, 3)
        valid = np.all(np.isfinite(flat), axis=1)
        flat = np.where(np.isfinite(flat), flat, 0.0)
        world = (R @ flat.T).T + t

        pts2d = project(world).astype(np.int32)
        tris = pts2d.reshape(n_tri, 3, 2)

        color = LINK_COLORS.get(link_name, (128, 128, 128))
        # Slightly darker fill, brighter edge
        fill_color = tuple(max(0, c - 40) for c in color)

        # Fill triangles
        for tri in tris:
            cv2.fillConvexPoly(canvas, tri, fill_color)
        # Draw edges for visibility
        cv2.polylines(canvas, tris[::3], True, color, 1, cv2.LINE_AA)

    # Add link name labels at joint positions
    for link_name in transforms:
        if link_name in skip_set or link_name not in LINK_COLORS:
            continue
        t = transforms[link_name][0]
        pt2d = project(t.reshape(1, 3))[0].astype(int)
        color = LINK_COLORS.get(link_name, (200, 200, 200))
        # Short label
        label = link_name.replace("_link", "").replace("left_", "L_").replace("right_", "R_")
        cv2.circle(canvas, (pt2d[0], pt2d[1]), 4, color, -1)
        cv2.putText(canvas, label, (pt2d[0] + 6, pt2d[1] + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.putText(canvas, title, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    return canvas


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading URDF...")
    model = pin.buildModelFromUrdf(URDF_PATH, pin.JointModelFreeFlyer())
    data = model.createData()

    # Use neutral (T-pose like) configuration
    q = pin.neutral(model)
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)

    transforms = {}
    for i in range(model.nframes):
        name = model.frames[i].name
        T = data.oMf[i]
        transforms[name] = (T.translation.copy(), T.rotation.copy())

    link_meshes = parse_urdf_meshes(URDF_PATH)
    print(f"Links with mesh: {list(link_meshes.keys())}")

    # Render WITHOUT hands
    skip_no_hand = {"head_link", "logo_link", "d435_link", "mid360_link",
                    "left_rubber_hand", "right_rubber_hand"}
    img_no_hand = render_front(link_meshes, MESH_DIR, transforms, skip_no_hand,
                               "WITHOUT hands (current SKIP_MESHES)")

    # Render WITH hands
    skip_with_hand = {"head_link", "logo_link", "d435_link", "mid360_link"}
    img_with_hand = render_front(link_meshes, MESH_DIR, transforms, skip_with_hand,
                                 "WITH rubber_hand (blue=L, red=R)")

    # Side by side
    combined = np.hstack([img_no_hand, img_with_hand])
    out_path = os.path.join(OUTPUT_DIR, "mesh_with_without_hands.png")
    cv2.imwrite(out_path, combined)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
