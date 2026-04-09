"""
Render G1 robot mesh + skeleton in bird's eye views (top + side).
Uses FK to transform STL meshes to world frame, then orthographic projection.
"""

import numpy as np
import pandas as pd
import cv2
import os
import xml.etree.ElementTree as ET
import pinocchio as pin
from stl import mesh as stl_mesh

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "g1_wbt")
URDF_PATH = os.path.join(BASE_DIR, "data", "g1_urdf", "g1_29dof_rev_1_0.urdf")
MESH_DIR = os.path.join(BASE_DIR, "data", "unitree_ros", "robots", "g1_description", "meshes")
VIDEO_PATH = os.path.join(DATA_DIR, "videos", "observation.images.head_stereo_left",
                          "chunk-000", "file-000.mp4")
PARQUET_PATH = os.path.join(DATA_DIR, "data", "chunk-000", "file-000.parquet")
OUTPUT_DIR = os.path.join(BASE_DIR, "test_results")

TARGET_EP = 0
TARGET_FRAME = 813

SKELETON_BONES = [
    ("pelvis", "waist_yaw_link"), ("waist_yaw_link", "waist_roll_link"),
    ("waist_roll_link", "torso_link"),
    ("pelvis", "left_hip_pitch_link"), ("left_hip_pitch_link", "left_hip_roll_link"),
    ("left_hip_roll_link", "left_hip_yaw_link"), ("left_hip_yaw_link", "left_knee_link"),
    ("left_knee_link", "left_ankle_pitch_link"),
    ("left_ankle_pitch_link", "left_ankle_roll_link"),
    ("pelvis", "right_hip_pitch_link"), ("right_hip_pitch_link", "right_hip_roll_link"),
    ("right_hip_roll_link", "right_hip_yaw_link"), ("right_hip_yaw_link", "right_knee_link"),
    ("right_knee_link", "right_ankle_pitch_link"),
    ("right_ankle_pitch_link", "right_ankle_roll_link"),
    ("torso_link", "left_shoulder_pitch_link"),
    ("left_shoulder_pitch_link", "left_shoulder_roll_link"),
    ("left_shoulder_roll_link", "left_shoulder_yaw_link"),
    ("left_shoulder_yaw_link", "left_elbow_link"),
    ("left_elbow_link", "left_wrist_roll_link"),
    ("left_wrist_roll_link", "left_wrist_pitch_link"),
    ("left_wrist_pitch_link", "left_wrist_yaw_link"),
    ("torso_link", "right_shoulder_pitch_link"),
    ("right_shoulder_pitch_link", "right_shoulder_roll_link"),
    ("right_shoulder_roll_link", "right_shoulder_yaw_link"),
    ("right_shoulder_yaw_link", "right_elbow_link"),
    ("right_elbow_link", "right_wrist_roll_link"),
    ("right_wrist_roll_link", "right_wrist_pitch_link"),
    ("right_wrist_pitch_link", "right_wrist_yaw_link"),
]

ENDPOINT_LABELS = {
    "left_wrist_yaw_link": "L-HAND",
    "right_wrist_yaw_link": "R-HAND",
    "left_ankle_roll_link": "L-FOOT",
    "right_ankle_roll_link": "R-FOOT",
}


def link_color(name):
    """BGR color for a link."""
    if "left" in name: return (255, 128, 0)    # orange
    if "right" in name: return (0, 128, 255)   # blue
    return (0, 255, 128)                        # green


def link_color_dark(name):
    """Darker shade for filled mesh faces."""
    if "left" in name: return (180, 90, 0)
    if "right" in name: return (0, 90, 180)
    return (0, 180, 90)


def short_name(n):
    return n.replace("_link", "").replace("_pitch", "P").replace("_roll", "R").replace("_yaw", "Y")


def extract_frame(video_path, frame_idx):
    cap = cv2.VideoCapture(video_path)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        if ret and frame is not None: return frame
    import av
    container = av.open(video_path)
    for i, f in enumerate(container.decode(container.streams.video[0])):
        if i == frame_idx:
            img = f.to_ndarray(format='bgr24')
            container.close()
            return img
    container.close()
    return None


def build_q(model, rq):
    q = pin.neutral(model)
    q[0:3] = rq[0:3]
    # Dataset quaternion (w,x,y,z) -> pinocchio (x,y,z,w)
    q[3], q[4], q[5], q[6] = rq[4], rq[5], rq[6], rq[3]
    q[7:36] = rq[7:36]
    return q


def do_fk(model, data, q):
    """Run FK, return dict of link_name -> (translation, rotation_matrix)."""
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    transforms = {}
    for i in range(model.nframes):
        name = model.frames[i].name
        T = data.oMf[i]
        transforms[name] = (T.translation.copy(), T.rotation.copy())
    return transforms


def parse_urdf_meshes(urdf_path):
    """Parse URDF to get link_name -> mesh_filename mapping."""
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
            # Strip "meshes/" prefix if present
            basename = os.path.basename(filename)
            link_meshes[name] = basename
    return link_meshes


def load_meshes(link_meshes, mesh_dir):
    """Load STL meshes, return dict of link_name -> vertices array (N_tri, 3, 3)."""
    meshes = {}
    for link_name, filename in link_meshes.items():
        path = os.path.join(mesh_dir, filename)
        if not os.path.exists(path):
            continue
        m = stl_mesh.Mesh.from_file(path)
        meshes[link_name] = m.vectors  # (n_triangles, 3_vertices, 3_coords)
    return meshes


def transform_mesh(vertices, R, t):
    """Transform mesh vertices (N,3,3) by rotation R and translation t."""
    # vertices shape: (n_tri, 3, 3)
    n = vertices.shape[0]
    flat = vertices.reshape(-1, 3)  # (n_tri*3, 3)
    transformed = (R @ flat.T).T + t  # (n_tri*3, 3)
    return transformed.reshape(n, 3, 3)


def render_mesh_birdseye(transforms, meshes, cam_pos):
    """Render mesh + skeleton in top view and side view."""
    S = 800
    scale = 300
    # Center on pelvis position
    pelvis_t = transforms.get("pelvis", (np.zeros(3), np.eye(3)))[0]
    cx_world, cy_world = pelvis_t[0], pelvis_t[1]

    def proj_top(p):
        """Top view: looking down from +Z. Up=Forward(X+), Left=Left(Y+)."""
        sx = S // 2 - int((p[1] - cy_world) * scale)
        sy = S // 2 - int((p[0] - cx_world) * scale)
        return (sx, sy)

    def proj_side(p):
        """Side view: from -Y (right). Right=Forward(X+), Up=Up(Z+)."""
        sx = S // 4 + int((p[0] - cx_world) * scale)
        sy = S - 100 - int(p[2] * scale)
        return (sx, sy)

    def render_view(proj, depth_axis, title, ax_label):
        canvas = np.zeros((S, S, 3), dtype=np.uint8) + 30

        # Grid
        for i in range(-5, 6):
            u = S // 2 + int(i * 0.5 * scale)
            cv2.line(canvas, (u, 0), (u, S), (50, 50, 50), 1)
            cv2.line(canvas, (0, u), (S, u), (50, 50, 50), 1)

        # Collect all triangles with depth for painter's algorithm
        all_tris = []  # (depth, projected_pts, fill_color, edge_color)

        for link_name, verts in meshes.items():
            if link_name not in transforms:
                continue
            R, t = transforms[link_name][1], transforms[link_name][0]
            world_verts = transform_mesh(verts, R, t)

            fill_c = link_color_dark(link_name)
            edge_c = link_color(link_name)

            for tri in world_verts:
                # tri shape: (3, 3) — three vertices
                centroid = tri.mean(axis=0)
                depth = centroid[depth_axis]
                pts_2d = np.array([proj(v) for v in tri], dtype=np.int32)
                all_tris.append((depth, pts_2d, fill_c, edge_c))

        # Sort by depth (far to near for painter's algorithm)
        # For top view (depth_axis=2, z): lower z drawn first
        # For side view (depth_axis=1, y): more positive y (left) drawn first
        all_tris.sort(key=lambda x: x[0])

        # Draw filled triangles
        for _, pts, fill_c, edge_c in all_tris:
            cv2.fillPoly(canvas, [pts], fill_c)
            cv2.polylines(canvas, [pts], True, edge_c, 1, cv2.LINE_AA)

        # Draw skeleton bones on top
        positions = {n: t for n, (t, _) in transforms.items()}
        for a, b in SKELETON_BONES:
            if a in positions and b in positions:
                pa, pb = proj(positions[a]), proj(positions[b])
                cv2.line(canvas, pa, pb, (255, 255, 255), 1, cv2.LINE_AA)

        # Draw endpoint labels
        for link_name, label in ENDPOINT_LABELS.items():
            if link_name not in positions:
                continue
            pt = proj(positions[link_name])
            color = link_color(link_name)
            cv2.circle(canvas, pt, 6, color, 2, cv2.LINE_AA)
            cv2.putText(canvas, label, (pt[0] + 10, pt[1] + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

        # Camera
        cp = proj(cam_pos)
        cv2.circle(canvas, cp, 8, (0, 255, 255), 2)
        cv2.putText(canvas, "CAM (d435)", (cp[0] + 10, cp[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

        # Labels
        cv2.putText(canvas, title, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(canvas, ax_label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        cv2.putText(canvas, "Orange=LEFT  Blue=RIGHT  Green=CENTER", (10, S - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

        return canvas

    # depth_axis: which world axis points "into" the screen
    # Top view looks down Z → depth = z (axis 2)
    # Side view looks from -Y → depth = y (axis 1), more negative y = closer
    top = render_view(proj_top, 2, "TOP VIEW (XY) - looking down",
                      "Up=Forward(X+)  Left=Left(Y+)")
    side = render_view(proj_side, 1, "SIDE VIEW (XZ) - from right",
                       "Right=Forward(X+)  Up=Up(Z+)")

    return np.hstack([top, side])


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load URDF model
    model = pin.buildModelFromUrdf(URDF_PATH, pin.JointModelFreeFlyer())
    data = model.createData()

    # Load state data
    df = pd.read_parquet(PARQUET_PATH)
    row = df[(df["episode_index"] == TARGET_EP) & (df["frame_index"] == TARGET_FRAME)].iloc[0]
    rq = np.array(row["observation.state.robot_q_current"], dtype=np.float64)

    # FK — full transforms
    q = build_q(model, rq)
    transforms = do_fk(model, data, q)

    # Parse URDF for mesh references
    link_meshes = parse_urdf_meshes(URDF_PATH)
    print(f"URDF defines {len(link_meshes)} visual meshes")

    # Load STL meshes
    meshes = load_meshes(link_meshes, MESH_DIR)
    print(f"Loaded {len(meshes)} meshes from {MESH_DIR}")
    for name in sorted(meshes.keys()):
        n_tri = meshes[name].shape[0]
        print(f"  {name:35s}: {n_tri:6d} triangles")

    # Camera position from FK
    d435_pos = transforms.get("d435_link", (np.zeros(3), np.eye(3)))[0]

    # Print key positions
    print("\n=== Key joint positions (world frame) ===")
    for name in ["pelvis", "torso_link", "d435_link", "head_link",
                 "left_wrist_yaw_link", "right_wrist_yaw_link",
                 "left_ankle_roll_link", "right_ankle_roll_link"]:
        if name in transforms:
            p = transforms[name][0]
            print(f"  {name:30s}: ({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f})")

    # === Output 0: Raw video frame ===
    img = extract_frame(VIDEO_PATH, TARGET_FRAME)
    if img is not None:
        cv2.imwrite(os.path.join(OUTPUT_DIR, "0_frame.png"), img)
        print("\nSaved 0_frame.png")

    # === Output 1: Mesh bird's eye view ===
    be = render_mesh_birdseye(transforms, meshes, d435_pos)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "1_birdseye_mesh.png"), be)
    print("Saved 1_birdseye_mesh.png")

    print("\nDone!")


if __name__ == "__main__":
    main()
