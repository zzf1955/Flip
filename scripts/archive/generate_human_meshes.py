"""
Generate human arm/hand capsule STL meshes for overlay rendering.

Each mesh is a capsule (cylinder + hemisphere caps) in the link-local frame,
extending from origin toward the next joint position.

Usage:
  python scripts/generate_human_meshes.py
"""

import sys
import numpy as np
from stl import mesh as stl_mesh
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "scripts"))

from config import MESH_DIR

OUT_DIR = MESH_DIR


def create_capsule(start, end, radius, n_circ=16, n_cap=6):
    """Create a capsule mesh (cylinder + hemisphere caps) as an STL Mesh.

    Parameters
    ----------
    start, end : array-like, shape (3,)
        Capsule axis endpoints in link-local frame.
    radius : float
        Capsule radius in meters.
    n_circ : int
        Number of circumferential segments.
    n_cap : int
        Number of latitude rings per hemisphere cap.

    Returns
    -------
    stl.mesh.Mesh
    """
    start = np.asarray(start, dtype=np.float64)
    end = np.asarray(end, dtype=np.float64)
    axis = end - start
    length = np.linalg.norm(axis)

    # Build local coordinate frame
    if length < 1e-6:
        z_ax = np.array([0.0, 0.0, 1.0])
    else:
        z_ax = axis / length
    # Pick a non-parallel reference vector
    ref = np.array([0.0, 0.0, 1.0]) if abs(z_ax[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x_ax = np.cross(z_ax, ref)
    x_ax /= np.linalg.norm(x_ax)
    y_ax = np.cross(z_ax, x_ax)

    theta = np.linspace(0, 2 * np.pi, n_circ, endpoint=False)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    # Generate rings: bottom cap -> cylinder -> top cap
    rings = []

    # Bottom hemisphere (from pole to equator at start)
    for i in range(n_cap, 0, -1):
        phi = (np.pi / 2) * i / n_cap
        r = radius * np.cos(phi)
        offset = -radius * np.sin(phi)
        center = start + z_ax * offset
        ring = np.array([center + r * (cos_t[j] * x_ax + sin_t[j] * y_ax)
                         for j in range(n_circ)])
        rings.append(ring)

    # Cylinder: equator at start
    rings.append(np.array([start + radius * (cos_t[j] * x_ax + sin_t[j] * y_ax)
                           for j in range(n_circ)]))

    # Cylinder: equator at end
    rings.append(np.array([end + radius * (cos_t[j] * x_ax + sin_t[j] * y_ax)
                           for j in range(n_circ)]))

    # Top hemisphere (from equator to pole at end)
    for i in range(1, n_cap + 1):
        phi = (np.pi / 2) * i / n_cap
        r = radius * np.cos(phi)
        offset = radius * np.sin(phi)
        center = end + z_ax * offset
        ring = np.array([center + r * (cos_t[j] * x_ax + sin_t[j] * y_ax)
                         for j in range(n_circ)])
        rings.append(ring)

    # Triangulate adjacent rings
    triangles = []
    for i in range(len(rings) - 1):
        for j in range(n_circ):
            j_next = (j + 1) % n_circ
            a, b = rings[i][j], rings[i][j_next]
            c, d = rings[i + 1][j], rings[i + 1][j_next]
            triangles.append([a, b, d])
            triangles.append([a, d, c])

    # Close poles with triangle fans
    bottom_pole = start - z_ax * radius
    top_pole = end + z_ax * radius
    for j in range(n_circ):
        j_next = (j + 1) % n_circ
        triangles.append([bottom_pole, rings[0][j_next], rings[0][j]])
        triangles.append([top_pole, rings[-1][j], rings[-1][j_next]])

    tri_array = np.array(triangles)
    m = stl_mesh.Mesh(np.zeros(len(tri_array), dtype=stl_mesh.Mesh.dtype))
    m.vectors = tri_array
    return m


def create_flat_box(center, size, n_seg=4):
    """Create a flat rectangular box mesh.

    Parameters
    ----------
    center : array-like (3,)
    size : array-like (3,)  (width_x, width_y, thickness_z)

    Returns
    -------
    stl.mesh.Mesh
    """
    cx, cy, cz = center
    sx, sy, sz = [s / 2 for s in size]
    # 8 corners
    corners = np.array([
        [cx - sx, cy - sy, cz - sz],
        [cx + sx, cy - sy, cz - sz],
        [cx + sx, cy + sy, cz - sz],
        [cx - sx, cy + sy, cz - sz],
        [cx - sx, cy - sy, cz + sz],
        [cx + sx, cy - sy, cz + sz],
        [cx + sx, cy + sy, cz + sz],
        [cx - sx, cy + sy, cz + sz],
    ])
    # 12 triangles (2 per face)
    faces = [
        [0,1,2], [0,2,3],  # bottom
        [4,6,5], [4,7,6],  # top
        [0,4,5], [0,5,1],  # front
        [2,6,7], [2,7,3],  # back
        [0,3,7], [0,7,4],  # left
        [1,5,6], [1,6,2],  # right
    ]
    triangles = np.array([[corners[f[0]], corners[f[1]], corners[f[2]]] for f in faces])
    m = stl_mesh.Mesh(np.zeros(len(triangles), dtype=stl_mesh.Mesh.dtype))
    m.vectors = triangles
    return m


# ---------------------------------------------------------------------------
# Mesh definitions: (start, end, radius) in link-local frame
# Directions derived from G1 URDF joint origins
# ---------------------------------------------------------------------------

ARM_MESHES = {
    # shoulder_pitch_link: short connector from torso to shoulder roll
    "human_shoulder_pitch_link": {
        "start": [0, 0, 0.01],
        "end": [0, 0, -0.035],
        "radius": 0.05,
    },
    # shoulder_roll_link: upper arm proximal segment
    "human_shoulder_roll_link": {
        "start": [0, 0, 0.01],
        "end": [0, 0, -0.103],
        "radius": 0.05,
    },
    # shoulder_yaw_link: upper arm distal segment
    "human_shoulder_yaw_link": {
        "start": [0, 0, 0.01],
        "end": [0.016, 0, -0.081],
        "radius": 0.045,
    },
    # elbow_link: forearm
    "human_elbow_link": {
        "start": [-0.005, 0, 0.005],
        "end": [0.100, 0, -0.010],
        "radius": 0.04,
    },
    # wrist segments
    "human_wrist_roll_link": {
        "start": [-0.005, 0, 0],
        "end": [0.038, 0, 0],
        "radius": 0.032,
    },
    "human_wrist_pitch_link": {
        "start": [-0.005, 0, 0],
        "end": [0.046, 0, 0],
        "radius": 0.030,
    },
    "human_wrist_yaw_link": {
        "start": [-0.005, 0, 0],
        "end": [0.042, 0, 0],
        "radius": 0.028,
    },
}

# Palm: flat box in base_link local frame
# base_link connects to wrist_yaw via rpy="π -π/2 0", so Z is roughly "forward"
PALM_MESH = {
    "human_base_link": {
        "center": [0, 0, 0.08],
        "size": [0.07, 0.02, 0.10],  # width, thickness, length
    },
}

# Finger capsules in finger-local frame
# Regular fingers: _1 extends ~3.3cm in +Y, _2 extends ~5cm in +Y
# Thumb: follows Inspire joint chain directions
FINGER_MESHES = {
    # Proximal phalanx (index, middle, ring, little _1)
    "human_finger_proximal": {
        "start": [0, 0, 0],
        "end": [-0.003, 0.033, 0.001],
        "radius": 0.010,
    },
    # Distal phalanx (index, middle, ring, little _2)
    "human_finger_distal": {
        "start": [0, 0, 0],
        "end": [-0.005, 0.045, 0.001],
        "radius": 0.009,
    },
    # Thumb segments
    "human_thumb_1": {
        "start": [0, 0, 0],
        "end": [0.007, 0.011, 0.005],
        "radius": 0.013,
    },
    "human_thumb_2": {
        "start": [0, 0, 0],
        "end": [-0.031, 0.021, -0.001],
        "radius": 0.012,
    },
    "human_thumb_3": {
        "start": [0, 0, 0],
        "end": [-0.022, 0.013, 0],
        "radius": 0.011,
    },
    "human_thumb_4": {
        "start": [0, 0, 0],
        "end": [-0.015, 0.015, -0.007],
        "radius": 0.010,
    },
}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    count = 0

    # Arm capsules
    for name, spec in ARM_MESHES.items():
        m = create_capsule(spec["start"], spec["end"], spec["radius"])
        path = os.path.join(OUT_DIR, f"{name}.STL")
        m.save(path)
        print(f"  {name}.STL  ({len(m.vectors)} triangles)")
        count += 1

    # Palm box
    for name, spec in PALM_MESH.items():
        m = create_flat_box(spec["center"], spec["size"])
        path = os.path.join(OUT_DIR, f"{name}.STL")
        m.save(path)
        print(f"  {name}.STL  ({len(m.vectors)} triangles)")
        count += 1

    # Finger capsules
    for name, spec in FINGER_MESHES.items():
        m = create_capsule(spec["start"], spec["end"], spec["radius"])
        path = os.path.join(OUT_DIR, f"{name}.STL")
        m.save(path)
        print(f"  {name}.STL  ({len(m.vectors)} triangles)")
        count += 1

    print(f"\nGenerated {count} STL files in {OUT_DIR}")


if __name__ == "__main__":
    main()
