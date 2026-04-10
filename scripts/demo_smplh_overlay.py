"""
Demo: SMPLH human mesh overlay on G1 ego-centric video.

Shows side-by-side comparison:
  Original | G1 Robot Overlay | SMPLH Human Overlay

The SMPLH model provides realistic human arm/hand mesh with:
  - Shape variation via beta parameters (body proportions)
  - Smooth skinning at joints (linear blend skinning)
  - Per-joint bone correction for T-pose → G1-zero-pose alignment
  - Proper triangle mesh with Lambertian shading

Usage:
  python scripts/demo_smplh_overlay.py --episode 4 --frame 153
  python scripts/demo_smplh_overlay.py --episode 4 --frame 153 --scale 0.45
  python scripts/demo_smplh_overlay.py --episode 4 --frame 153 --debug-mesh
  python scripts/demo_smplh_overlay.py --episode 4 --frame 153 --beta 2.0 -1.0
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

from config import G1_URDF, MESH_DIR, BEST_PARAMS, SKIP_MESHES, OUTPUT_DIR, get_hand_type
from video_inpaint import (
    build_q, do_fk, parse_urdf_meshes, preload_meshes,
    make_camera, load_episode_info,
)

# ── SMPLH model path ──
SMPLH_PATH = (
    "/disk_n/zzf/video-gen/MIMO/video_decomp/"
    "models--menyifang--MIMO_VidDecomp/snapshots/"
    "41a6023cc405f73d888cabe5cb7506da99bbbec6/assets/smplh/SMPLH_NEUTRAL.npz"
)

# ── Frame convention: SMPLH (X=left, Y=up, Z=fwd) → G1 (X=fwd, Y=left, Z=up) ──
R_SMPLH_TO_G1 = np.array([
    [0, 0, 1],   # G1_X = SMPLH_Z (forward)
    [1, 0, 0],   # G1_Y = SMPLH_X (left)
    [0, 1, 0],   # G1_Z = SMPLH_Y (up)
], dtype=np.float64)

# ── Default mesh scale ──
DEFAULT_SCALE = 0.55

# ── SMPLH joint index → G1 FK link name ──
SMPLH_TO_G1 = {
    # Body (for weight blending near shoulder)
    9: 'torso_link',        # spine3
    12: 'torso_link',       # neck
    # Left arm
    13: 'torso_link',                   # left_collar
    16: 'left_shoulder_yaw_link',       # left_shoulder
    18: 'left_elbow_link',              # left_elbow
    20: 'left_wrist_yaw_link',          # left_wrist
    # Left hand
    22: 'left_index_1',   23: 'left_index_2',   24: 'left_index_2',
    25: 'left_middle_1',  26: 'left_middle_2',  27: 'left_middle_2',
    28: 'left_little_1',  29: 'left_little_2',  30: 'left_little_2',
    31: 'left_ring_1',    32: 'left_ring_2',    33: 'left_ring_2',
    34: 'left_thumb_1',   35: 'left_thumb_3',   36: 'left_thumb_4',
    # Right arm
    14: 'torso_link',                   # right_collar
    17: 'right_shoulder_yaw_link',      # right_shoulder
    19: 'right_elbow_link',             # right_elbow
    21: 'right_wrist_yaw_link',         # right_wrist
    # Right hand
    37: 'right_index_1',  38: 'right_index_2',  39: 'right_index_2',
    40: 'right_middle_1', 41: 'right_middle_2', 42: 'right_middle_2',
    43: 'right_little_1', 44: 'right_little_2', 45: 'right_little_2',
    46: 'right_ring_1',   47: 'right_ring_2',   48: 'right_ring_2',
    49: 'right_thumb_1',  50: 'right_thumb_3',  51: 'right_thumb_4',
}

# ── Bone direction pairs for per-joint correction ──
# SMPLH_joint: (SMPLH_child, G1_link, G1_child_link)
BONE_CHILD_MAP = {
    13: (16, 'torso_link', 'left_shoulder_yaw_link'),
    16: (18, 'left_shoulder_yaw_link', 'left_elbow_link'),
    18: (20, 'left_elbow_link', 'left_wrist_yaw_link'),
    20: (25, 'left_wrist_yaw_link', 'left_middle_1'),
    14: (17, 'torso_link', 'right_shoulder_yaw_link'),
    17: (19, 'right_shoulder_yaw_link', 'right_elbow_link'),
    19: (21, 'right_elbow_link', 'right_wrist_yaw_link'),
    21: (40, 'right_wrist_yaw_link', 'right_middle_1'),
}

# Joints defining arm+hand region (for vertex filtering)
ARM_HAND_JOINTS = {13, 14, 16, 17, 18, 19, 20, 21} | set(range(22, 52))

SKIN_COLOR = (135, 165, 215)  # BGR: warm skin tone


def rotation_between_vectors(a, b):
    """Compute minimal rotation matrix that rotates unit vector a to unit vector b."""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    dot = np.clip(np.dot(a, b), -1.0, 1.0)
    if dot > 0.9999:
        return np.eye(3)
    if dot < -0.9999:
        # 180° rotation: pick any perpendicular axis
        perp = np.array([1, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 1, 0])
        axis = np.cross(a, perp)
        axis /= np.linalg.norm(axis)
        # Rodrigues for 180°: R = 2 * outer(axis, axis) - I
        return 2.0 * np.outer(axis, axis) - np.eye(3)
    axis = np.cross(a, b)
    axis /= np.linalg.norm(axis)
    angle = np.arccos(dot)
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


def _compute_g1_zero_transforms():
    """Run G1 FK at zero pose to get rest-pose link positions."""
    model = pin.buildModelFromUrdf(G1_URDF, pin.JointModelFreeFlyer())
    data = model.createData()
    q0 = pin.neutral(model)
    q0[0:3] = [0, 0, 0]
    q0[3:7] = [0, 0, 0, 1]  # identity quaternion (x,y,z,w)
    pin.forwardKinematics(model, data, q0)
    pin.updateFramePlacements(model, data)
    transforms = {}
    for i in range(model.nframes):
        name = model.frames[i].name
        T = data.oMf[i]
        transforms[name] = (T.translation.copy(), T.rotation.copy())
    return transforms


class SMPLHModel:
    """Lightweight SMPLH model for arm+hand rendering via G1 FK."""

    def __init__(self, model_path=SMPLH_PATH):
        print(f"  Loading {os.path.basename(model_path)}...")
        data = np.load(model_path, allow_pickle=True)
        self.v_template = data['v_template']       # (6890, 3)
        self.shapedirs = data['shapedirs']          # (6890, 3, 16)
        self.weights = data['weights']              # (6890, 52)
        self.J_regressor = data['J_regressor']      # (52, 6890)
        self.faces_all = data['f']                  # (13776, 3)

        self.J_rest = self.J_regressor @ self.v_template  # (52, 3)
        self._extract_arm_hand()
        self._compute_bone_corrections()

    def _extract_arm_hand(self):
        """Extract arm+hand subset of the mesh."""
        max_joint = np.argmax(self.weights, axis=1)
        self.arm_mask = np.isin(max_joint, list(ARM_HAND_JOINTS))

        # Remap vertex indices for sub-mesh
        old_to_new = np.full(len(self.v_template), -1, dtype=np.int32)
        arm_indices = np.where(self.arm_mask)[0]
        old_to_new[arm_indices] = np.arange(len(arm_indices))

        # Keep faces where all 3 verts are arm+hand
        face_mask = self.arm_mask[self.faces_all].all(axis=1)
        self.faces = old_to_new[self.faces_all[face_mask]]

        # Subset vertex data
        self.arm_v = self.v_template[self.arm_mask]       # (N, 3)
        self.arm_shapedirs = self.shapedirs[self.arm_mask] # (N, 3, 16)
        self.arm_weights = self.weights[self.arm_mask]     # (N, 52)

        # Renormalize weights to only include mapped joints
        mapped = list(SMPLH_TO_G1.keys())
        w = self.arm_weights[:, mapped].copy()
        w_sum = np.maximum(w.sum(axis=1, keepdims=True), 1e-8)
        self.arm_weights_norm = w / w_sum                  # (N, len(mapped))
        self.mapped_joints = mapped

        print(f"  Arm+hand mesh: {len(self.arm_v)} verts, {len(self.faces)} faces")

    def _compute_bone_corrections(self):
        """Compute per-joint rotation corrections for T-pose → G1-zero-pose.

        The SMPLH mesh is in T-pose (arms horizontal), but G1 at zero angles
        has arms pointing down/forward. We compute a rotation per joint that
        maps the SMPLH bone direction to the G1 zero-pose bone direction.
        """
        print("  Computing bone corrections...")
        g1_zero = _compute_g1_zero_transforms()
        J = self.J_rest

        # Compute per-joint correction: R_bone[j] rotates
        # SMPLH bone direction (in G1 frame) to G1 zero-pose bone direction
        self.R_combined = {}  # j_smplh → 3x3 (R_bone @ R_SMPLH_TO_G1)

        # Default: no bone correction (just frame mapping)
        for j in SMPLH_TO_G1:
            self.R_combined[j] = R_SMPLH_TO_G1.copy()

        # Compute corrections for joints with known bone pairs
        for j_smplh, (j_child, g1_link, g1_child) in BONE_CHILD_MAP.items():
            if g1_link not in g1_zero or g1_child not in g1_zero:
                continue
            # SMPLH bone direction in SMPLH frame
            d_smplh = J[j_child] - J[j_smplh]
            if np.linalg.norm(d_smplh) < 1e-6:
                continue
            # Map to G1 frame
            d_smplh_g1 = R_SMPLH_TO_G1 @ d_smplh

            # G1 zero-pose bone direction
            t_parent = g1_zero[g1_link][0]
            t_child = g1_zero[g1_child][0]
            d_g1 = t_child - t_parent
            if np.linalg.norm(d_g1) < 1e-6:
                continue

            R_bone = rotation_between_vectors(d_smplh_g1, d_g1)
            self.R_combined[j_smplh] = R_bone @ R_SMPLH_TO_G1

        # Finger joints inherit wrist correction
        left_wrist_R = self.R_combined.get(20, R_SMPLH_TO_G1)
        right_wrist_R = self.R_combined.get(21, R_SMPLH_TO_G1)
        for j in range(22, 37):
            self.R_combined[j] = left_wrist_R
        for j in range(37, 52):
            self.R_combined[j] = right_wrist_R

        # Body joints (9, 12) keep default (no bone correction)
        n_corrected = sum(1 for j in BONE_CHILD_MAP
                          if not np.allclose(self.R_combined[j], R_SMPLH_TO_G1))
        print(f"  Bone corrections: {n_corrected} joints corrected")

    def compute_world_verts(self, g1_transforms, betas=None, scale=DEFAULT_SCALE):
        """Compute arm+hand world-space vertices using G1 FK + bone corrections.

        For each vertex v with skinning weight w_j for joint j:
          v_world = Σ_j  w_j * (R_g1_j @ R_combined_j @ (v - J_j) * scale + t_g1_j)

        R_combined_j = R_bone_j @ R_SMPLH_TO_G1, which maps SMPLH T-pose offsets
        to G1 zero-pose orientation before applying the FK rotation.
        """
        # Apply shape blend shapes
        if betas is not None:
            betas = np.asarray(betas, dtype=np.float64)
            n = min(len(betas), self.arm_shapedirs.shape[2])
            arm_v = self.arm_v + np.einsum('vcd,d->vc',
                                           self.arm_shapedirs[:, :, :n], betas[:n])
            J = self.J_regressor @ (
                self.v_template + np.einsum('vcd,d->vc',
                                            self.shapedirs[:, :, :n], betas[:n]))
        else:
            arm_v = self.arm_v
            J = self.J_rest

        N = len(arm_v)
        v_world = np.zeros((N, 3))

        for idx, j_smplh in enumerate(self.mapped_joints):
            g1_link = SMPLH_TO_G1[j_smplh]
            if g1_link not in g1_transforms:
                continue

            w = self.arm_weights_norm[:, idx]  # (N,)
            if w.max() < 1e-6:
                continue

            t_g1, R_g1 = g1_transforms[g1_link]
            R_comb = self.R_combined[j_smplh]

            # Vertex offset in SMPLH frame → bone-corrected G1 frame → scale
            v_local = arm_v - J[j_smplh]                     # (N, 3)
            v_g1_local = (R_comb @ v_local.T).T * scale       # (N, 3)

            # Apply G1 FK transform
            v_posed = (R_g1 @ v_g1_local.T).T + t_g1          # (N, 3)

            # Weighted accumulation
            v_world += w[:, None] * v_posed

        return v_world


def _render_mesh_internal(v_world, faces, params, g1_transforms,
                          img_shape, color=SKIN_COLOR, wireframe=False):
    """Render mesh triangles onto an image-shaped canvas. Returns the canvas."""
    K, D, rvec, tvec, R_w2c, t_w2c = make_camera(params, g1_transforms)
    h, w = img_shape[:2]
    result = np.zeros((h, w, 3), dtype=np.uint8)

    tri_verts = v_world[faces]  # (F, 3, 3)
    flat = tri_verts.reshape(-1, 3).astype(np.float64)

    cam_pts = (R_w2c @ flat.T).T + t_w2c.flatten()
    z_cam = cam_pts[:, 2]

    pts2d, _ = cv2.fisheye.projectPoints(
        flat.reshape(-1, 1, 3), rvec, tvec, K, D)
    pts2d = pts2d.reshape(-1, 2)

    n_tri = len(faces)
    z_tri = z_cam.reshape(n_tri, 3)
    pts_tri = pts2d.reshape(n_tri, 3, 2)

    valid = (z_tri > 0.01).all(axis=1)
    finite = np.all(np.isfinite(pts_tri), axis=(1, 2))
    mask = valid & finite
    if mask.sum() == 0:
        return result

    pts_tri = pts_tri[mask]
    z_tri = z_tri[mask]
    tri_world = tri_verts[mask]

    # Shading
    v0, v1, v2 = tri_world[:, 0], tri_world[:, 1], tri_world[:, 2]
    normals = np.cross(v1 - v0, v2 - v0)
    norms = np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-8)
    normals /= norms
    cam_tri_pts = (R_w2c @ tri_world.mean(axis=1).T).T + t_w2c.flatten()
    view_dirs = -cam_tri_pts
    view_norms = np.maximum(np.linalg.norm(view_dirs, axis=1, keepdims=True), 1e-8)
    view_dirs /= view_norms
    dots = np.abs(np.sum(normals * view_dirs, axis=1))

    order = np.argsort(-z_tri.mean(axis=1))

    for idx in order:
        tri = pts_tri[idx].astype(np.int32)
        shade = 0.3 + 0.7 * dots[idx]
        shaded = tuple(int(c * shade) for c in color)
        cv2.fillPoly(result, [tri], shaded)
        if wireframe:
            cv2.polylines(result, [tri], True, (255, 255, 255), 1, cv2.LINE_AA)

    return result


def render_smplh_triangles(img, smplh, g1_transforms, params,
                           betas=None, color=SKIN_COLOR, scale=DEFAULT_SCALE):
    """Render SMPLH arm+hand mesh with filled triangles on top of img."""
    v_world = smplh.compute_world_verts(g1_transforms, betas=betas, scale=scale)
    mesh_canvas = _render_mesh_internal(
        v_world, smplh.faces, params, g1_transforms, img.shape, color)
    # Composite: mesh pixels override where non-black
    result = img.copy()
    mesh_mask = mesh_canvas.any(axis=2)
    result[mesh_mask] = mesh_canvas[mesh_mask]
    return result


def render_debug_mesh(smplh, g1_transforms, params, img_shape,
                      betas=None, color=SKIN_COLOR, scale=DEFAULT_SCALE):
    """Render SMPLH mesh alone on black background with wireframe."""
    v_world = smplh.compute_world_verts(g1_transforms, betas=betas, scale=scale)
    return _render_mesh_internal(
        v_world, smplh.faces, params, g1_transforms, img_shape,
        color, wireframe=True)


def render_g1_overlay(img, mesh_cache, transforms, params):
    """Semi-transparent G1 robot convex-hull overlay (for comparison)."""
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
        pts2d, _ = cv2.fisheye.projectPoints(
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
    parser = argparse.ArgumentParser(description="SMPLH human overlay demo")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--frame", type=int, default=30)
    parser.add_argument("--scale", type=float, default=DEFAULT_SCALE,
                        help="Mesh thickness scale (default 0.55)")
    parser.add_argument("--beta", type=float, nargs='*', default=None,
                        help="SMPLH shape params (e.g. --beta 2.0 -1.0)")
    parser.add_argument("--debug-mesh", action="store_true",
                        help="Add 4th panel: mesh wireframe on black background")
    args = parser.parse_args()

    out_dir = os.path.join(OUTPUT_DIR, "smplh_demo")
    os.makedirs(out_dir, exist_ok=True)

    # ── 1. Load episode ──
    print(f"Loading episode {args.episode}...")
    video_path, from_ts, to_ts, ep_df = load_episode_info(args.episode)

    frame_row = ep_df[ep_df["frame_index"] == args.frame]
    if len(frame_row) == 0:
        print(f"Frame {args.frame} not found, using first frame")
        frame_row = ep_df.iloc[[0]]
        args.frame = int(frame_row.iloc[0]["frame_index"])

    rq = np.array(frame_row.iloc[0]["observation.state.robot_q_current"],
                  dtype=np.float64)
    hs = np.array(frame_row.iloc[0]["observation.state.hand_state"],
                  dtype=np.float64)

    # ── 2. Extract video frame ──
    print(f"Extracting frame {args.frame}...")
    img = extract_frame(video_path, from_ts, args.frame)
    if img is None:
        print("ERROR: failed to extract frame")
        return
    h, w = img.shape[:2]
    print(f"  Frame size: {w}x{h}")

    # ── 3. G1 FK ──
    print("Loading G1 model and computing FK...")
    model_g = pin.buildModelFromUrdf(G1_URDF, pin.JointModelFreeFlyer())
    data_g = model_g.createData()
    link_meshes_g = parse_urdf_meshes(G1_URDF)
    mesh_cache_g = preload_meshes(link_meshes_g, MESH_DIR, skip_set=SKIP_MESHES)

    hand_type = get_hand_type()
    q = build_q(model_g, rq, hs, hand_type=hand_type)
    transforms = do_fk(model_g, data_g, q)

    # ── 4. Render G1 overlay ──
    print("Rendering G1 overlay...")
    g1_result = render_g1_overlay(img, mesh_cache_g, transforms, BEST_PARAMS)

    # ── 5. Load SMPLH and render human overlay ──
    print("Loading SMPLH model...")
    smplh = SMPLHModel()

    betas = None
    if args.beta:
        betas = np.zeros(16)
        for i, b in enumerate(args.beta[:16]):
            betas[i] = b
        print(f"  Shape betas: {betas[:len(args.beta)]}")

    print("Rendering SMPLH overlay...")
    human_result = render_smplh_triangles(
        img, smplh, transforms, BEST_PARAMS,
        betas=betas, scale=args.scale)

    # ── 6. Compose output ──
    panels = [img, g1_result, human_result]
    labels = ["Original", "G1 Robot", f"SMPLH (s={args.scale})"]

    if args.debug_mesh:
        print("Rendering debug mesh...")
        debug = render_debug_mesh(
            smplh, transforms, BEST_PARAMS, img.shape,
            betas=betas, scale=args.scale)
        panels.append(debug)
        labels.append("Mesh Debug")

    combined = np.hstack(panels)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i, label in enumerate(labels):
        cv2.putText(combined, label, (i * w + 10, 30),
                    font, 0.6, (255, 255, 255), 2)

    beta_tag = ""
    if args.beta:
        beta_tag = "_beta" + "_".join(f"{b:.1f}" for b in args.beta)
    dm_tag = "_debug" if args.debug_mesh else ""
    out_path = os.path.join(
        out_dir,
        f"ep{args.episode}_f{args.frame:04d}_s{args.scale}{beta_tag}{dm_tag}.png")
    cv2.imwrite(out_path, combined)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
