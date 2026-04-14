"""
Render G1 robot 3-view orthographic + oblique perspective with keypoints.
Outputs: test_results/g1_3view.png, g1_closeup.png, g1_keypoints_oblique.png
"""
import sys, os
import numpy as np
import pinocchio as pin
from stl import mesh as stl_mesh

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from config import G1_URDF, MESH_DIR, SKIP_MESHES, OUTPUT_DIR
from video_inpaint import parse_urdf_meshes

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

# ── Body part grouping for coloring ──
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

GROUP_COLORS = {
    'left_leg':  '#5599DD',
    'right_leg': '#3377BB',
    'torso':     '#999999',
    'left_arm':  '#55CC55',
    'right_arm': '#339933',
    'left_hand': '#FF7755',
    'right_hand':'#DD5533',
    'other':     '#BBBBBB',
}

LABEL_LINKS = {
    'left_thumb_4':  'L thumb tip',
    'right_thumb_4': 'R thumb tip',
    'left_ankle_roll_link':  'L foot',
    'right_ankle_roll_link': 'R foot',
    'torso_link': 'torso',
    'pelvis': 'pelvis',
    'left_elbow_link': 'L elbow',
    'right_elbow_link': 'R elbow',
    'left_knee_link': 'L knee',
    'right_knee_link': 'R knee',
    'left_wrist_pitch_link': 'L wrist',
    'right_wrist_pitch_link': 'R wrist',
}

# ── Keypoints (link_name, local_offset) ──
KEYPOINTS = [
    ('left_wrist_yaw_link',   np.array([ 0.0046,  0.0000,  0.0300]), 'L wrist'),
    ('left_thumb_4',          np.array([-0.0314,  0.0150, -0.0101]), 'L thumb tip'),
    ('left_ankle_roll_link',  np.array([ 0.1424,  0.0000, -0.0210]), 'L toe tip'),
    ('right_ankle_roll_link', np.array([ 0.1424, -0.0000, -0.0215]), 'R toe tip'),
    ('right_thumb_4',         np.array([ 0.0314,  0.0150, -0.0101]), 'R thumb tip'),
    ('right_wrist_yaw_link',  np.array([ 0.0046,  0.0000,  0.0300]), 'R wrist'),
]


def get_color(link_name):
    for group, links in BODY_GROUPS.items():
        if link_name in links:
            return GROUP_COLORS[group]
    if 'left' in link_name and any(p in link_name for p in
            ['thumb', 'index', 'middle', 'ring', 'little', 'base_link', 'palm']):
        return GROUP_COLORS['left_hand']
    if 'right' in link_name and any(p in link_name for p in
            ['thumb', 'index', 'middle', 'ring', 'little', 'base_link', 'palm']):
        return GROUP_COLORS['right_hand']
    return GROUP_COLORS['other']


def load_robot(transforms):
    """Load all mesh triangles in world frame."""
    link_meshes = parse_urdf_meshes(G1_URDF)
    skip = SKIP_MESHES
    all_tris = {}
    link_centers = {}

    for link_name, filename in link_meshes.items():
        if link_name in skip:
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
        link_centers[link_name] = world.mean(axis=0)

    return all_tris, link_centers


def oblique_project(pts, azim_deg=35, elev_deg=25, scale=1.0):
    """Simple oblique projection: rotate then take X,Z as screen coords."""
    az = np.radians(azim_deg)
    el = np.radians(elev_deg)

    # Rotation around Z (azimuth)
    Rz = np.array([[ np.cos(az), -np.sin(az), 0],
                    [ np.sin(az),  np.cos(az), 0],
                    [ 0,           0,          1]])
    # Rotation around X (elevation)
    Rx = np.array([[1, 0,           0],
                    [0, np.cos(el), -np.sin(el)],
                    [0, np.sin(el),  np.cos(el)]])

    R = Rx @ Rz
    rotated = (R @ pts.T).T
    # Screen: X -> horizontal, Z -> vertical
    screen = rotated[:, [0, 2]] * scale
    depth = rotated[:, 1]
    return screen, depth


def main():
    # ── Load model, neutral pose FK ──
    model = pin.buildModelFromUrdf(G1_URDF, pin.JointModelFreeFlyer())
    data = model.createData()
    q = pin.neutral(model)
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)

    transforms = {}
    for i in range(model.nframes):
        name = model.frames[i].name
        T = data.oMf[i]
        transforms[name] = (T.translation.copy(), T.rotation.copy())

    all_tris, link_centers = load_robot(transforms)
    print(f"Loaded {len(all_tris)} links, "
          f"{sum(len(t) for t in all_tris.values())} triangles total")

    # ── Compute keypoint world positions ──
    kp_world = []
    kp_labels = []
    for link_name, local_offset, label in KEYPOINTS:
        t_link, R_link = transforms[link_name]
        world_pos = R_link @ local_offset + t_link
        kp_world.append(world_pos)
        kp_labels.append(label)
        print(f"  {label}: world={world_pos}")
    kp_world = np.array(kp_world)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Oblique view ──
    azim, elev = 35, 20

    fig, ax = plt.subplots(figsize=(14, 18))

    # Collect and sort all triangles by depth
    poly_data = []
    for link_name, tris in all_tris.items():
        color = get_color(link_name)
        for tri in tris:
            screen, depth = oblique_project(tri, azim, elev)
            mean_depth = depth.mean()
            poly_data.append((mean_depth, screen, color))

    poly_data.sort(key=lambda x: x[0])  # painter's algorithm

    polys = [p[1] for p in poly_data]
    colors = [p[2] for p in poly_data]

    pc = PolyCollection(polys, facecolors=colors,
                       edgecolors='#444444', linewidths=0.03, alpha=0.85)
    ax.add_collection(pc)

    # ── Draw keypoints ──
    kp_screen, _ = oblique_project(kp_world, azim, elev)
    for i, (sx, sy) in enumerate(kp_screen):
        # Red dot with white outline
        ax.plot(sx, sy, 'o', color='red', markersize=10, markeredgecolor='white',
                markeredgewidth=2, zorder=100)
        # Label with offset
        offset_x = 0.015 if 'R ' in kp_labels[i] else -0.015
        ha = 'left' if 'R ' in kp_labels[i] else 'right'
        ax.annotate(kp_labels[i], (sx, sy),
                   xytext=(sx + offset_x, sy + 0.01),
                   fontsize=11, fontweight='bold', color='red',
                   bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='red', alpha=0.9),
                   arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                   zorder=101)

    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)
    ax.set_title(f'G1 Oblique View (azim={azim}°, elev={elev}°) — Calibration Keypoints',
                fontsize=14, fontweight='bold')
    ax.tick_params(labelsize=7)
    ax.autoscale()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.set_xlim(xlim[0] - 0.02, xlim[1] + 0.02)
    ax.set_ylim(ylim[0] - 0.02, ylim[1] + 0.02)

    # Keypoint coordinate table
    table_text = "Keypoint Coordinates (local frame offset):\n"
    for link_name, local_offset, label in KEYPOINTS:
        table_text += f"  {label:14s}  link={link_name:25s}  offset=[{local_offset[0]:+.4f}, {local_offset[1]:+.4f}, {local_offset[2]:+.4f}]\n"
    ax.text(0.02, 0.02, table_text, transform=ax.transAxes, fontsize=8,
            family='monospace', verticalalignment='bottom',
            bbox=dict(boxstyle='round', fc='lightyellow', ec='gray', alpha=0.9))

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, 'g1_keypoints_oblique.png')
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
