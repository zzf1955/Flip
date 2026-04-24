#!/usr/bin/env python3
"""Demo: render G1 mesh at different scale factors for visual comparison.

Renders f090 (Pillow task) with mesh scaled 1.0x / 1.1x / 1.2x around pelvis,
outputs a side-by-side comparison image.
"""

import os, json
import numpy as np
import cv2

from src.core.config import G1_URDF, MESH_DIR, SKIP_MESHES, DATASET_ROOT, TMP_DIR, CALIB_5POINT_DIR, get_hand_type
from src.core.camera import project_points_cv
from src.core.fk import build_q, do_fk, parse_urdf_meshes, preload_meshes
from src.core.camera import make_camera
from src.tools.render_3view import get_color as _get_color_hex
import pandas as pd
import pinocchio as pin


def _hex_to_bgr(h):
    r, g, b = int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16)
    return (b, g, r)


def _get_color_bgr(link_name):
    return _hex_to_bgr(_get_color_hex(link_name))


def render_scaled_lit_mesh(mesh_cache, transforms, params_dict, h, w,
                           scale=1.0, center=None,
                           ambient=0.3, diffuse=0.7):
    """Render mesh with uniform world-space scaling around center point.

    scale: uniform scale factor (1.0 = original)
    center: (3,) world point to scale around. If None, no scaling.
    """
    K, D, rvec, tvec, R_w2c, t_w2c, _fisheye = make_camera(params_dict, transforms)
    t_flat = t_w2c.flatten()
    light_dir = np.array([0.0, 0.0, 1.0])

    all_world_tris = []
    all_colors_bgr = []
    all_normals = []

    for link_name, (tris, _) in mesh_cache.items():
        if link_name not in transforms or len(tris) == 0:
            continue
        t_link, R_link = transforms[link_name]
        flat = tris.reshape(-1, 3)
        world = (R_link @ flat.T).T + t_link

        # Scale around center
        if center is not None and scale != 1.0:
            world = center + scale * (world - center)

        world_tris = world.reshape(-1, 3, 3)

        v0, v1, v2 = world_tris[:, 0], world_tris[:, 1], world_tris[:, 2]
        normals = np.cross(v1 - v0, v2 - v0)
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = normals / np.maximum(norms, 1e-12)

        all_world_tris.append(world_tris)
        all_normals.append(normals)
        all_colors_bgr.extend([_get_color_bgr(link_name)] * len(world_tris))

    if not all_world_tris:
        return np.zeros((h, w, 3), dtype=np.uint8), np.zeros((h, w), dtype=np.uint8)

    all_world_tris = np.concatenate(all_world_tris, axis=0)
    all_normals = np.concatenate(all_normals, axis=0)
    n_tris = len(all_world_tris)

    flat_all = all_world_tris.reshape(-1, 3)
    cam_pts = (R_w2c @ flat_all.T).T + t_flat
    z = cam_pts[:, 2]

    pts3d_cv = flat_all.reshape(-1, 1, 3).astype(np.float64)
    pts2d = project_points_cv(pts3d_cv, rvec, tvec, K, D, _fisheye).reshape(-1, 2)

    z_tri = z.reshape(n_tris, 3)
    pts_tri = pts2d.reshape(n_tris, 3, 2)

    valid = ((z_tri > 0.05).all(axis=1) &
             np.all(np.isfinite(pts_tri), axis=(1, 2)) &
             np.all(np.abs(pts_tri) < 5000, axis=(1, 2)))

    depths = z_tri[valid].mean(axis=1)
    pts_valid = pts_tri[valid].astype(np.int32)
    normals_valid = all_normals[valid]
    colors_valid = [all_colors_bgr[i] for i in np.where(valid)[0]]

    lambert = np.clip(normals_valid @ light_dir, 0, 1)
    shade = ambient + diffuse * lambert

    order = np.argsort(-depths)

    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    alpha = np.zeros((h, w), dtype=np.uint8)

    for idx in order:
        bgr = colors_valid[idx]
        s = shade[idx]
        lit = (int(bgr[0] * s), int(bgr[1] * s), int(bgr[2] * s))
        cv2.fillPoly(canvas, [pts_valid[idx]], lit)
        cv2.fillPoly(alpha, [pts_valid[idx]], 255)

    return canvas, alpha


def main():
    scales = [1.0, 1.1, 1.2]

    # Load params
    params_path = os.path.join(OUTPUT_DIR,
        "calibration/kp_4points/best_params.json")
    with open(params_path) as f:
        params_dict = json.load(f)["params"]
    print(f"Loaded params from {params_path}")

    # Load manifest for frame info
    manifest_path = os.path.join(CALIB_5POINT_DIR, "manifest_pillow.json")
    with open(manifest_path) as f:
        manifest = json.load(f)
    # Use f090 (second frame: frame 164)
    finfo = manifest["frames"][1]
    task, episode, frame_idx = finfo["task"], finfo["episode"], finfo["frame"]
    img_name = finfo["image"]
    print(f"Frame: task={task}, ep={episode}, frame={frame_idx}, img={img_name}")

    # Load image
    img_path = os.path.join(CALIB_5POINT_DIR, img_name)
    img_bgr = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if img_bgr.shape[2] == 4:
        img_bgr = img_bgr[:, :, :3]
    h, w = img_bgr.shape[:2]

    # Load URDF + meshes (subsample=1 for full quality)
    model = pin.buildModelFromUrdf(G1_URDF, pin.JointModelFreeFlyer())
    data_pin = model.createData()
    link_meshes = parse_urdf_meshes(G1_URDF)
    mesh_cache = preload_meshes(link_meshes, MESH_DIR,
                                skip_set=SKIP_MESHES, subsample=1)
    print(f"Loaded {len(mesh_cache)} mesh links (subsample=1)")

    # Load joint state and compute FK
    data_dir = os.path.join(DATASET_ROOT, task)
    meta = pd.read_parquet(
        os.path.join(data_dir, "meta/episodes/chunk-000/file-000.parquet"))
    ep_meta = meta[meta["episode_index"] == episode].iloc[0]
    data_fi = int(ep_meta.get("data/file_index", 0))
    df = pd.read_parquet(
        os.path.join(data_dir, "data/chunk-000", f"file-{data_fi:03d}.parquet"))
    ep_df = df[df["episode_index"] == episode].sort_values("frame_index")
    row = ep_df[ep_df["frame_index"] == frame_idx].iloc[0]

    rq = np.array(row["observation.state.robot_q_current"], dtype=np.float64)
    hs = np.array(row["observation.state.hand_state"], dtype=np.float64)
    q = build_q(model, rq, hs, hand_type=get_hand_type(task))
    transforms = do_fk(model, data_pin, q)

    # Scale center = pelvis position
    center = transforms["pelvis"][0].copy()
    print(f"Scale center (pelvis): {center}")

    # Render each scale
    panels = []
    for sc in scales:
        print(f"Rendering scale={sc:.1f} ...")
        canvas, alpha = render_scaled_lit_mesh(
            mesh_cache, transforms, params_dict, h, w,
            scale=sc, center=center)
        # Alpha blend
        result = img_bgr.copy()
        mask = alpha > 0
        result[mask] = (result[mask] * 0.4 + canvas[mask] * 0.6).astype(np.uint8)
        # Label
        cv2.putText(result, f"scale={sc:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        panels.append(result)

    # Horizontal concatenation
    combined = np.concatenate(panels, axis=1)

    out_path = os.path.join(TMP_DIR, "mesh_scale",
        "f090_scale_compare.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, combined)
    print(f"Saved: {out_path}  ({combined.shape[1]}x{combined.shape[0]})")


if __name__ == "__main__":
    main()
