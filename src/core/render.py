"""Mesh rendering utilities.

- Triangle mask rasterization (FK mesh -> binary mask)
- Convex hull overlay (semi-transparent per-link)
- Lambertian-shaded mesh rendering (face-based)
- Combined mask+overlay single-pass rendering
"""

import numpy as np
import cv2

from .camera import make_camera, make_camera_const, project_points_cv


# ── Link-based rendering (G1 robot mesh cache) ──

def render_mask(mesh_cache, transforms, params, h, w, cam_const=None):
    """Render per-triangle binary mask using preloaded meshes.

    Args:
        mesh_cache: dict from fk.preload_meshes()
        transforms: dict from fk.do_fk()
        params: camera parameter dict
        h, w: output image dimensions
        cam_const: precomputed camera constants (optional)

    Returns:
        mask: (h, w) uint8, 255=robot
    """
    K, D, rvec, tvec, R_w2c, t_w2c, _fisheye = make_camera(
        params, transforms, cam_const)
    t_w2c_flat = t_w2c.flatten()

    all_world_tris = []
    all_tri_counts = []
    for link_name, (tris, _) in mesh_cache.items():
        if link_name not in transforms:
            continue
        t_link, R_link = transforms[link_name]
        flat = tris.reshape(-1, 3)
        world = (R_link @ flat.T).T + t_link
        all_world_tris.append(world)
        all_tri_counts.append(len(tris))

    mask = np.zeros((h, w), dtype=np.uint8)
    if not all_world_tris:
        return mask

    all_world = np.concatenate(all_world_tris, axis=0).astype(np.float64)
    cam_pts = (R_w2c @ all_world.T).T + t_w2c_flat
    z_all = cam_pts[:, 2]

    pts2d = project_points_cv(
        all_world.reshape(-1, 1, 3), rvec, tvec, K, D, _fisheye)
    pts2d = pts2d.reshape(-1, 2)

    offset = 0
    for n_tris in all_tri_counts:
        n_pts = n_tris * 3
        z_seg = z_all[offset:offset + n_pts].reshape(n_tris, 3)
        pts_seg = pts2d[offset:offset + n_pts].reshape(n_tris, 3, 2)

        valid = (z_seg > 0.01).all(axis=1)
        pts_valid = pts_seg[valid]
        if len(pts_valid) > 0:
            finite = np.all(np.isfinite(pts_valid), axis=(1, 2))
            pts_valid = pts_valid[finite]
            if len(pts_valid) > 0:
                cv2.fillPoly(mask, pts_valid.astype(np.int32), 255)

        offset += n_pts

    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
    return mask


def render_overlay(img, mesh_cache, transforms, params, cam_const=None, alpha=0.35):
    """Semi-transparent convex hull overlay, colored by left/right/other.

    Returns:
        overlay image (h, w, 3) uint8
    """
    K, D, rvec, tvec, R_w2c, t_w2c, _fisheye = make_camera(
        params, transforms, cam_const)
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

        pts2d = project_points_cv(
            verts3d[in_front].reshape(-1, 1, 3), rvec, tvec, K, D, _fisheye)
        pts2d = pts2d.reshape(-1, 2)
        finite = np.all(np.isfinite(pts2d), axis=1)
        pts2d = pts2d[finite]
        if len(pts2d) < 3:
            continue

        hull = cv2.convexHull(pts2d.astype(np.float32))
        if "left" in link_name:
            color = (255, 180, 0)
        elif "right" in link_name:
            color = (0, 180, 255)
        else:
            color = (0, 255, 180)
        cv2.fillConvexPoly(overlay, hull.astype(np.int32), color)
        cv2.polylines(result, [hull.astype(np.int32)], True, color, 1, cv2.LINE_AA)

    cv2.addWeighted(overlay, alpha, result, 1.0, 0, result)
    return result


def render_mask_and_overlay(img, mesh_cache, transforms, params, h, w,
                            cam_const=None, alpha=0.35):
    """Render both mask and overlay in a single pass (shared projection).

    Returns:
        (mask, overlay_image)
    """
    K, D, rvec, tvec, R_w2c, t_w2c, _fisheye = make_camera(
        params, transforms, cam_const)
    t_w2c_flat = t_w2c.flatten()

    all_world_tris = []
    all_tri_counts = []
    overlay_links = []

    for link_name, (tris, unique_verts) in mesh_cache.items():
        if link_name not in transforms:
            continue
        t_link, R_link = transforms[link_name]

        flat = tris.reshape(-1, 3)
        world = (R_link @ flat.T).T + t_link
        all_world_tris.append(world)
        all_tri_counts.append(len(tris))

        if len(unique_verts) > 0:
            verts_w = (R_link @ unique_verts.T).T + t_link
            overlay_links.append((link_name, verts_w))

    # Mask
    mask = np.zeros((h, w), dtype=np.uint8)
    if all_world_tris:
        all_world = np.concatenate(all_world_tris, axis=0).astype(np.float64)
        cam_pts = (R_w2c @ all_world.T).T + t_w2c_flat
        z_all = cam_pts[:, 2]

        pts2d = project_points_cv(
            all_world.reshape(-1, 1, 3), rvec, tvec, K, D, _fisheye)
        pts2d = pts2d.reshape(-1, 2)

        offset = 0
        for n_tris in all_tri_counts:
            n_pts = n_tris * 3
            z_seg = z_all[offset:offset + n_pts].reshape(n_tris, 3)
            pts_seg = pts2d[offset:offset + n_pts].reshape(n_tris, 3, 2)

            valid = (z_seg > 0.01).all(axis=1)
            pts_valid = pts_seg[valid]
            if len(pts_valid) > 0:
                finite = np.all(np.isfinite(pts_valid), axis=(1, 2))
                pts_valid = pts_valid[finite]
                if len(pts_valid) > 0:
                    cv2.fillPoly(mask, pts_valid.astype(np.int32), 255)

            offset += n_pts

        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

    # Overlay
    overlay = np.zeros_like(img)
    result = img.copy()
    for link_name, verts_w in overlay_links:
        verts_f64 = verts_w.astype(np.float64)
        depths = (R_w2c @ verts_f64.T).T + t_w2c_flat
        in_front = depths[:, 2] > 0.01
        if np.count_nonzero(in_front) < 3:
            continue

        pts2d_ov = project_points_cv(
            verts_f64[in_front].reshape(-1, 1, 3), rvec, tvec, K, D, _fisheye)
        pts2d_ov = pts2d_ov.reshape(-1, 2)
        finite = np.all(np.isfinite(pts2d_ov), axis=1)
        pts2d_ov = pts2d_ov[finite]
        if len(pts2d_ov) < 3:
            continue

        hull = cv2.convexHull(pts2d_ov.astype(np.float32))
        if "left" in link_name:
            color = (255, 180, 0)
        elif "right" in link_name:
            color = (0, 180, 255)
        else:
            color = (0, 255, 180)
        cv2.fillConvexPoly(overlay, hull.astype(np.int32), color)
        cv2.polylines(result, [hull.astype(np.int32)], True, color, 1, cv2.LINE_AA)

    cv2.addWeighted(overlay, alpha, result, 1.0, 0, result)
    return mask, result


# ── Face-based mesh rendering (SMPLH human mesh) ──

SKIN_COLOR = (135, 165, 215)  # BGR: warm skin tone


def render_mesh_on_image(img, v_world, faces, g1_transforms, params,
                         color=SKIN_COLOR, cam_const=None):
    """Render triangle mesh onto image with Lambertian shading.

    Used for rendering SMPLH human mesh overlays.

    Args:
        img: background image (BGR, uint8)
        v_world: (V, 3) vertex positions in world frame
        faces: (F, 3) triangle face indices
        g1_transforms: FK transforms dict (for camera)
        params: camera parameter dict
        color: base color (BGR tuple)
        cam_const: precomputed camera constants

    Returns:
        composited image (BGR, uint8)
    """
    K, D, rvec, tvec, R_w2c, t_w2c, fisheye = make_camera(
        params, g1_transforms, cam_const)
    h, w = img.shape[:2]
    canvas = np.zeros((h, w, 3), dtype=np.uint8)

    tri_verts = v_world[faces]  # (F, 3, 3)
    flat = tri_verts.reshape(-1, 3).astype(np.float64)

    cam_pts = (R_w2c @ flat.T).T + t_w2c.flatten()
    z_cam = cam_pts[:, 2]

    pts2d = project_points_cv(
        flat.reshape(-1, 1, 3), rvec, tvec, K, D, fisheye)
    pts2d = pts2d.reshape(-1, 2)

    n_tri = len(faces)
    z_tri = z_cam.reshape(n_tri, 3)
    pts_tri = pts2d.reshape(n_tri, 3, 2)

    valid = (z_tri > 0.01).all(axis=1)
    finite = np.all(np.isfinite(pts_tri), axis=(1, 2))
    vis_mask = valid & finite
    if vis_mask.sum() == 0:
        return img.copy()

    pts_tri = pts_tri[vis_mask]
    z_tri = z_tri[vis_mask]
    tri_world = tri_verts[vis_mask]

    # Lambertian shading
    v0, v1, v2 = tri_world[:, 0], tri_world[:, 1], tri_world[:, 2]
    normals = np.cross(v1 - v0, v2 - v0)
    norms = np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-8)
    normals /= norms
    cam_tri_pts = (R_w2c @ tri_world.mean(axis=1).T).T + t_w2c.flatten()
    view_dirs = -cam_tri_pts
    view_norms = np.maximum(np.linalg.norm(view_dirs, axis=1, keepdims=True), 1e-8)
    view_dirs /= view_norms
    dots = np.abs(np.sum(normals * view_dirs, axis=1))

    # Depth-sort (painter's algorithm)
    order = np.argsort(-z_tri.mean(axis=1))
    for idx in order:
        tri = pts_tri[idx].astype(np.int32)
        shade = 0.3 + 0.7 * dots[idx]
        shaded = tuple(int(c * shade) for c in color)
        cv2.fillPoly(canvas, [tri], shaded)

    # Composite
    result = img.copy()
    mesh_mask = canvas.any(axis=2)
    result[mesh_mask] = canvas[mesh_mask]
    return result
