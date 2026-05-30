"""URDF/mesh loading and forward kinematics.

- URDF XML parsing for mesh file references
- STL mesh batch loading and caching
- Joint angle mapping (dataset -> URDF)
- Pinocchio FK wrapper
"""

import os
import numpy as np
import xml.etree.ElementTree as ET
import pinocchio as pin
from stl import mesh as stl_mesh

from .config import G1_URDF, MESH_DIR, SKIP_MESHES


def parse_urdf_meshes(urdf_path):
    """Extract mesh filenames from URDF visual geometry.

    Returns:
        dict[str, str]: link_name -> STL filename
    """
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


def preload_meshes(link_meshes, mesh_dir=None, skip_set=None, subsample=4):
    """Load all STL meshes once.

    Args:
        link_meshes: dict from parse_urdf_meshes()
        mesh_dir: directory containing STL files (defaults to config.MESH_DIR)
        skip_set: set of link names to skip (defaults to config.SKIP_MESHES)
        subsample: keep every Nth triangle

    Returns:
        dict[str, tuple]: link_name -> (triangles (N,3,3), unique_verts (M,3))
    """
    if mesh_dir is None:
        mesh_dir = MESH_DIR
    if skip_set is None:
        skip_set = SKIP_MESHES
    cache = {}
    for link_name, filename in link_meshes.items():
        if link_name in skip_set:
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
        if subsample > 1:
            tris = tris[::subsample]
        flat_all = m.vectors.reshape(-1, 3)
        valid_all = np.all(np.isfinite(flat_all), axis=1)
        unique_verts = np.unique(flat_all[valid_all], axis=0)
        if subsample > 1:
            unique_verts = unique_verts[::subsample]
        if len(tris) > 0:
            cache[link_name] = (tris, unique_verts)
    return cache


def load_robot(urdf_path=None, mesh_dir=None, skip_set=None, subsample=4):
    """Convenience: load URDF model + mesh cache in one call.

    Returns:
        (model, data, mesh_cache)
    """
    if urdf_path is None:
        urdf_path = G1_URDF
    model = pin.buildModelFromUrdf(urdf_path, pin.JointModelFreeFlyer())
    data = model.createData()
    link_meshes = parse_urdf_meshes(urdf_path)
    mesh_cache = preload_meshes(link_meshes, mesh_dir, skip_set, subsample)
    return model, data, mesh_cache


def build_q(model, rq, hand_state=None, hand_type="inspire"):
    """Map dataset rq (36) + hand_state (12) to Inspire FTP URDF q (60).

    Dataset rq layout:
      rq[0:3]   position
      rq[3:7]   quaternion (w,x,y,z)
      rq[7:29]  left leg(6) + right leg(6) + waist(3) + left arm(7)
      rq[29:36] right arm(7)

    hand_state layout (Inspire RH56DFTP, 0=closed 1=open):
      [0] left little  [1] left ring  [2] left middle  [3] left index
      [4] left thumb close  [5] left thumb tilt
      [6] right little  [7] right ring  [8] right middle  [9] right index
      [10] right thumb close  [11] right thumb tilt

    hand_state layout (BrainCo Revo2, 0=open 1=closed):
      [0] left thumb close  [1] left thumb tilt
      [2] left index  [3] left middle  [4] left ring  [5] left little
      [6] right thumb close  [7] right thumb tilt
      [8] right index  [9] right middle  [10] right ring  [11] right little

    URDF q layout (nq=60):
      q[0:7]   freeflyer (pos + quat x,y,z,w)
      q[7:29]  left leg(6) + right leg(6) + waist(3) + left arm(7)
      q[29:41] left hand(12): index(2), little(2), middle(2), ring(2), thumb(4)
      q[41:48] right arm(7)
      q[48:60] right hand(12): index(2), little(2), middle(2), ring(2), thumb(4)
    """
    q = pin.neutral(model)
    q[0:3] = rq[0:3]
    q[3], q[4], q[5], q[6] = rq[4], rq[5], rq[6], rq[3]
    q[7:29] = rq[7:29]
    q[41:48] = rq[29:36]  # right arm

    if hand_state is not None:
        hs = np.array(hand_state, dtype=np.float64)
        if hand_type == "inspire":
            hs = 1.0 - hs
            hs = np.concatenate([
                hs[[3, 2, 1, 0, 4, 5]],
                hs[[9, 8, 7, 6, 10, 11]],
            ])
        elif hand_type == "brainco":
            hs = np.concatenate([
                hs[2:6], hs[0:2],
                hs[8:12], hs[6:8],
            ])

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
    """Forward kinematics: compute frame transforms from joint config.

    Returns:
        dict[str, tuple]: frame_name -> (translation (3,), rotation (3,3))
    """
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    transforms = {}
    for i in range(model.nframes):
        name = model.frames[i].name
        T = data.oMf[i]
        transforms[name] = (T.translation.copy(), T.rotation.copy())
    return transforms
