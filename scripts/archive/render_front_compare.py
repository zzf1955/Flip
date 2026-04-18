"""
Render three-view comparison: G1 robot (center) with 4 SMPLH humans around it.

G1 frame: X=forward, Y=left, Z=up.
SMPLH humans placed at front(+X), back(-X), left(+Y), right(-Y) of G1.
Three views: front, side, top.

Usage:
  python scripts/render_front_compare.py
  python scripts/render_front_compare.py --beta 3.0
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

from config import G1_URDF, MESH_DIR, OUTPUT_DIR
from video_inpaint import parse_urdf_meshes, preload_meshes

SMPLH_PATH = (
    "/disk_n/zzf/video-gen/MIMO/video_decomp/"
    "models--menyifang--MIMO_VidDecomp/snapshots/"
    "41a6023cc405f73d888cabe5cb7506da99bbbec6/assets/smplh/SMPLH_NEUTRAL.npz"
)

VIEW_W, VIEW_H = 800, 800


def get_g1_world_triangles(tpose=False):
    """Load G1 mesh, return world-space triangles."""
    model = pin.buildModelFromUrdf(G1_URDF, pin.JointModelFreeFlyer())
    data = model.createData()
    q0 = pin.neutral(model)
    q0[0:3] = [0, 0, 0]
    q0[3:7] = [0, 0, 0, 1]

    if tpose:
        q0[23] = 1.57     # left shoulder roll
        q0[25] = 1.50     # left elbow: nearly straight (10° bend)
        q0[42] = -1.57    # right shoulder roll
        q0[44] = 1.50     # right elbow: nearly straight

    pin.forwardKinematics(model, data, q0)
    pin.updateFramePlacements(model, data)

    transforms = {}
    for i in range(model.nframes):
        name = model.frames[i].name
        T = data.oMf[i]
        transforms[name] = (T.translation.copy(), T.rotation.copy())

    link_meshes = parse_urdf_meshes(G1_URDF)
    mesh_cache = preload_meshes(link_meshes, MESH_DIR, skip_set=set(), subsample=2)

    all_tris = []
    for link_name, (tris, _) in mesh_cache.items():
        if link_name not in transforms:
            continue
        t, R = transforms[link_name]
        flat = tris.reshape(-1, 3)
        world = (R @ flat.T).T + t
        all_tris.append(world.reshape(-1, 3, 3))

    return np.concatenate(all_tris)


def get_smplh_triangles(betas=None):
    """Load SMPLH mesh (full body) in T-pose, converted to G1 frame."""
    data = np.load(SMPLH_PATH, allow_pickle=True)
    v = data['v_template'].copy()
    faces = data['f']

    if betas is not None:
        shapedirs = data['shapedirs']
        n = min(len(betas), shapedirs.shape[2])
        v += np.einsum('vcd,d->vc', shapedirs[:, :, :n], betas[:n])

    # SMPLH (X=left, Y=up, Z=fwd) → G1 (X=fwd, Y=left, Z=up)
    R = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
    v = (R @ v.T).T

    return v[faces], v


def project_perspective(pts_3d, cam_pos, cam_target, cam_up, fov_deg, img_w, img_h):
    """Perspective projection. Returns (N,2) pixel coords and (N,) depths."""
    forward = cam_target - cam_pos
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, cam_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)

    rel = pts_3d - cam_pos
    x_cam = rel @ right
    y_cam = rel @ up
    z_cam = rel @ forward

    f = img_h / (2.0 * np.tan(np.radians(fov_deg) / 2.0))
    u = f * x_cam / np.maximum(z_cam, 1e-6) + img_w / 2.0
    v = -f * y_cam / np.maximum(z_cam, 1e-6) + img_h / 2.0

    return np.stack([u, v], axis=-1), z_cam


def render_triangles(canvas, tris_3d, cam_pos, cam_target, cam_up, fov,
                     color=(180, 180, 180), light_dir=None):
    """Render triangles with Lambertian shading via painter's algorithm."""
    h, w = canvas.shape[:2]
    flat = tris_3d.reshape(-1, 3)
    pts2d, depths = project_perspective(flat, cam_pos, cam_target, cam_up, fov, w, h)

    n_tri = len(tris_3d)
    z_tri = depths.reshape(n_tri, 3)
    pts_tri = pts2d.reshape(n_tri, 3, 2)

    valid = (z_tri > 0.01).all(axis=1)
    finite = np.all(np.isfinite(pts_tri), axis=(1, 2))
    in_frame = ((pts_tri[:, :, 0] > -200) & (pts_tri[:, :, 0] < w + 200)).all(axis=1)
    in_frame &= ((pts_tri[:, :, 1] > -200) & (pts_tri[:, :, 1] < h + 200)).all(axis=1)
    mask = valid & finite & in_frame

    pts_tri = pts_tri[mask]
    z_tri = z_tri[mask]
    tris_w = tris_3d[mask]

    if len(pts_tri) == 0:
        return

    v0, v1, v2 = tris_w[:, 0], tris_w[:, 1], tris_w[:, 2]
    normals = np.cross(v1 - v0, v2 - v0)
    norms = np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-8)
    normals /= norms

    if light_dir is None:
        light_dir = np.array([0.3, -0.2, 1.0])
    light_dir = light_dir / np.linalg.norm(light_dir)
    dots = np.abs(normals @ light_dir)

    order = np.argsort(-z_tri.mean(axis=1))
    for idx in order:
        tri = pts_tri[idx].astype(np.int32)
        shade = 0.25 + 0.75 * dots[idx]
        shaded = tuple(int(c * shade) for c in color)
        cv2.fillPoly(canvas, [tri], shaded)


def shift_tris(tris, dx=0, dy=0, dz=0):
    """Translate triangles."""
    out = tris.copy()
    out[:, :, 0] += dx
    out[:, :, 1] += dy
    out[:, :, 2] += dz
    return out


def rotate_tris_z(tris, angle_deg, center_xy=(0, 0)):
    """Rotate triangles around Z axis (yaw)."""
    a = np.radians(angle_deg)
    c, s = np.cos(a), np.sin(a)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    out = tris.copy()
    cx, cy = center_xy
    out[:, :, 0] -= cx
    out[:, :, 1] -= cy
    flat = out.reshape(-1, 3)
    flat[:] = (Rz @ flat.T).T
    out = flat.reshape(-1, 3, 3)
    out[:, :, 0] += cx
    out[:, :, 1] += cy
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--beta", type=float, nargs='*', default=None)
    parser.add_argument("--spacing", type=float, default=0.7,
                        help="Distance from G1 center to each human (meters)")
    args = parser.parse_args()

    out_dir = os.path.join(OUTPUT_DIR, "smplh_demo")
    os.makedirs(out_dir, exist_ok=True)

    # ── Load models ──
    print("Loading G1 mesh (T-pose)...")
    g1_tris = get_g1_world_triangles(tpose=True)
    g1_all = g1_tris.reshape(-1, 3)
    g1_z_min = g1_all[:, 2].min()
    g1_z_max = g1_all[:, 2].max()
    g1_height = g1_z_max - g1_z_min
    g1_center_z = (g1_z_min + g1_z_max) / 2
    print(f"  G1 height: {g1_height:.4f} m")

    print("Loading SMPLH mesh...")
    betas = None
    if args.beta:
        betas = np.zeros(16)
        for i, b in enumerate(args.beta[:16]):
            betas[i] = b

    smplh_tris_raw, smplh_v = get_smplh_triangles(betas)
    smplh_z_min = smplh_v[:, 2].min()
    smplh_z_max = smplh_v[:, 2].max()
    smplh_height = smplh_z_max - smplh_z_min

    # Scale to match G1 height
    scale = g1_height / smplh_height
    print(f"  SMPLH: {smplh_height:.2f}m → scaled {scale:.4f}x to {g1_height:.2f}m")

    smplh_center = np.array([
        (smplh_v[:, 0].min() + smplh_v[:, 0].max()) / 2,
        (smplh_v[:, 1].min() + smplh_v[:, 1].max()) / 2,
        (smplh_z_min + smplh_z_max) / 2,
    ])
    smplh_tris_s = (smplh_tris_raw - smplh_center) * scale + smplh_center

    # Align bottoms
    smplh_scaled_z_min = smplh_tris_s.reshape(-1, 3)[:, 2].min()
    smplh_tris_s[:, :, 2] += (g1_z_min - smplh_scaled_z_min)

    # Center SMPLH X and Y at origin
    smplh_all = smplh_tris_s.reshape(-1, 3)
    smplh_tris_s[:, :, 0] -= (smplh_all[:, 0].min() + smplh_all[:, 0].max()) / 2
    smplh_tris_s[:, :, 1] -= (smplh_all[:, 1].min() + smplh_all[:, 1].max()) / 2

    # ── Place 4 humans around G1 ──
    # G1 frame: X=forward, Y=left, Z=up
    d = args.spacing
    placements = {
        "front":  (d,  0),    # +X: in front of G1
        "back":   (-d, 0),    # -X: behind G1
        "left":   (0,  d),    # +Y: to G1's left
        "right":  (0, -d),    # -Y: to G1's right
    }

    human_meshes = []
    for label, (dx, dy) in placements.items():
        h_tris = shift_tris(smplh_tris_s, dx=dx, dy=dy)
        human_meshes.append((label, h_tris))

    # ── Three views ──
    # G1 colors
    g1_color = (200, 190, 170)
    skin_color = (145, 175, 220)
    fov = 30
    cam_dist = 3.5
    light = np.array([1.0, -0.3, 0.5])

    views = {
        "Front (from +X)": {
            "pos": np.array([cam_dist, 0, g1_center_z]),
            "target": np.array([0, 0, g1_center_z]),
            "up": np.array([0, 0, 1.0]),
        },
        "Side (from -Y)": {
            "pos": np.array([0, -cam_dist, g1_center_z]),
            "target": np.array([0, 0, g1_center_z]),
            "up": np.array([0, 0, 1.0]),
        },
        "Top (from +Z)": {
            "pos": np.array([0, 0, g1_center_z + cam_dist]),
            "target": np.array([0, 0, g1_center_z]),
            "up": np.array([1, 0, 0]),  # X is "up" in top view
        },
    }

    panels = []
    for view_name, cam in views.items():
        print(f"  Rendering {view_name}...")
        canvas = np.full((VIEW_H, VIEW_W, 3), 40, dtype=np.uint8)

        # Render humans first (behind G1 in front view)
        for label, h_tris in human_meshes:
            render_triangles(canvas, h_tris, cam["pos"], cam["target"],
                             cam["up"], fov, color=skin_color, light_dir=light)

        # Render G1 on top
        render_triangles(canvas, g1_tris, cam["pos"], cam["target"],
                         cam["up"], fov, color=g1_color, light_dir=light)

        # Label
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(canvas, view_name, (10, 30), font, 0.7, (255, 255, 255), 2)
        panels.append(canvas)

    # ── Compose 3-panel output ──
    combined = np.hstack(panels)

    # Title bar
    title = f"G1 ({g1_height:.2f}m) + 4x SMPLH (orig {smplh_height:.2f}m, scale {scale:.2f}x)"
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(combined, title,
                (combined.shape[1] // 2 - 350, combined.shape[0] - 15),
                font, 0.6, (180, 180, 180), 1)

    beta_tag = ""
    if args.beta:
        beta_tag = "_beta" + "_".join(f"{b:.1f}" for b in args.beta)
    out_path = os.path.join(out_dir, f"three_views{beta_tag}.png")
    cv2.imwrite(out_path, combined)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
