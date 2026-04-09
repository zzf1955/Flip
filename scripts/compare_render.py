"""
Compare two rendering approaches side-by-side:
  Left:  convex hull per link (current)
  Right: triangle fillPoly + morphological close (proposed)
"""

import numpy as np
import pandas as pd
import cv2
import os
import xml.etree.ElementTree as ET
import pinocchio as pin
from stl import mesh as stl_mesh

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "g1_wbt_task3")
URDF_PATH = os.path.join(BASE_DIR, "data", "g1_urdf", "g1_29dof_rev_1_0.urdf")
MESH_DIR = os.path.join(BASE_DIR, "data", "unitree_ros", "robots", "g1_description", "meshes")
VIDEO_PATH = os.path.join(DATA_DIR, "videos", "observation.images.head_stereo_left",
                          "chunk-000", "file-000.mp4")
PARQUET_PATH = os.path.join(DATA_DIR, "data", "chunk-000", "file-000.parquet")
OUTPUT_DIR = os.path.join(BASE_DIR, "test_results")

TARGET_EP = 0
TARGET_FRAME = 276

SKIP_MESHES = {"head_link", "logo_link", "d435_link", "left_rubber_hand", "right_rubber_hand"}

# Best params from PSO
BEST_PARAMS = (0.0371, 0.0127, 0.5968, -59.33, 423.6, 400.6,
               -0.4539, 0.4154, 0.01857, -0.02024, 1.1278)


def extract_frame(video_path, frame_idx):
    cap = cv2.VideoCapture(video_path)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        if ret and frame is not None:
            return frame
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
    q[3], q[4], q[5], q[6] = rq[4], rq[5], rq[6], rq[3]
    q[7:36] = rq[7:36]
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
        if visual is None: continue
        geom = visual.find("geometry")
        if geom is None: continue
        mesh_elem = geom.find("mesh")
        if mesh_elem is None: continue
        filename = mesh_elem.get("filename")
        if filename:
            link_meshes[name] = os.path.basename(filename)
    return link_meshes


def make_camera(params, ref_t, ref_R):
    dx, dy, dz, pitch_deg, fx, fy, k1, k2, p1, p2, k3 = params
    pitch = np.radians(pitch_deg)
    R_pitch = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                        [0, 1, 0],
                        [-np.sin(pitch), 0, np.cos(pitch)]])
    R_body_to_cam = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]], dtype=np.float64)
    R_cam = R_body_to_cam @ R_pitch
    cam_pos = ref_t + ref_R @ np.array([dx, dy, dz])
    R_w2c = (ref_R @ R_cam.T).T
    t_w2c = R_w2c @ (-cam_pos)
    K = np.array([[fx, 0, 320], [0, fy, 240], [0, 0, 1]], dtype=np.float64)
    dist = np.array([k1, k2, p1, p2, k3], dtype=np.float64)
    rvec, _ = cv2.Rodrigues(R_w2c)
    tvec = t_w2c.reshape(3, 1)
    return K, dist, rvec, tvec, R_w2c, t_w2c


def render_convex_hull(link_meshes, mesh_dir, transforms, params, ref_t, ref_R, h, w):
    """Method A: convex hull per link."""
    K, dist, rvec, tvec, R_w2c, t_w2c = make_camera(params, ref_t, ref_R)
    mask = np.zeros((h, w), dtype=np.uint8)

    for link_name, filename in link_meshes.items():
        if link_name in SKIP_MESHES: continue
        if link_name not in transforms: continue
        path = os.path.join(mesh_dir, filename)
        if not os.path.exists(path): continue

        m = stl_mesh.Mesh.from_file(path)
        flat = m.vectors.reshape(-1, 3)
        valid = np.all(np.isfinite(flat), axis=1)
        flat = np.unique(flat[valid], axis=0)
        if len(flat) < 3: continue

        t_link, R_link = transforms[link_name]
        world = (R_link @ flat.T).T + t_link

        depths = (R_w2c @ world.T).T + t_w2c.flatten()
        in_front = depths[:, 2] > 0.01
        if np.count_nonzero(in_front) < 3: continue

        pts2d, _ = cv2.projectPoints(world[in_front], rvec, tvec, K, dist)
        pts2d = pts2d.reshape(-1, 2)
        finite = np.all(np.isfinite(pts2d), axis=1)
        pts2d = pts2d[finite]
        if len(pts2d) < 3: continue

        hull = cv2.convexHull(pts2d.astype(np.float32))
        cv2.fillConvexPoly(mask, hull.astype(np.int32), 255)

    return mask > 0


def render_triangles(link_meshes, mesh_dir, transforms, params, ref_t, ref_R, h, w,
                     morph_close=True, kernel_size=5):
    """Method B: triangle fillPoly + optional morphological close."""
    K, dist, rvec, tvec, R_w2c, t_w2c = make_camera(params, ref_t, ref_R)
    mask = np.zeros((h, w), dtype=np.uint8)

    for link_name, filename in link_meshes.items():
        if link_name in SKIP_MESHES: continue
        if link_name not in transforms: continue
        path = os.path.join(mesh_dir, filename)
        if not os.path.exists(path): continue

        m = stl_mesh.Mesh.from_file(path)
        verts = m.vectors  # (n_tri, 3, 3)
        flat = verts.reshape(-1, 3)
        valid_per_vert = np.all(np.isfinite(flat), axis=1)
        valid_per_tri = valid_per_vert.reshape(-1, 3).all(axis=1)
        verts = verts[valid_per_tri]
        if len(verts) == 0: continue

        t_link, R_link = transforms[link_name]
        flat = verts.reshape(-1, 3)
        world = (R_link @ flat.T).T + t_link

        depths = (R_w2c @ world.T).T + t_w2c.flatten()
        z_cam = depths[:, 2]

        pts2d, _ = cv2.projectPoints(world, rvec, tvec, K, dist)
        pts2d = pts2d.reshape(-1, 2)

        # Reshape back to triangles
        n_tri = len(verts)
        z_tri = z_cam.reshape(n_tri, 3)
        pts_tri = pts2d.reshape(n_tri, 3, 2)

        # Filter: all 3 verts in front of camera
        valid = (z_tri > 0.01).all(axis=1)
        pts_tri = pts_tri[valid]
        if len(pts_tri) == 0: continue

        finite = np.all(np.isfinite(pts_tri), axis=(1, 2))
        pts_tri = pts_tri[finite]
        tris = pts_tri.astype(np.int32)

        if len(tris) > 0:
            cv2.fillPoly(mask, tris, 255)

    if morph_close:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask > 0


def overlay_mask(img, mask, color, alpha=0.5):
    vis = img.copy()
    vis[mask] = (vis[mask] * (1 - alpha) + np.array(color) * alpha).astype(np.uint8)
    return vis


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading...")
    model = pin.buildModelFromUrdf(URDF_PATH, pin.JointModelFreeFlyer())
    data = model.createData()
    df = pd.read_parquet(PARQUET_PATH)
    row = df[(df["episode_index"] == TARGET_EP) & (df["frame_index"] == TARGET_FRAME)].iloc[0]
    rq = np.array(row["observation.state.robot_q_current"], dtype=np.float64)
    q = build_q(model, rq)
    transforms = do_fk(model, data, q)
    ref_t, ref_R = transforms["torso_link"]

    img = extract_frame(VIDEO_PATH, TARGET_FRAME)
    if img is None:
        print("ERROR: Could not extract frame"); return
    h, w = img.shape[:2]

    link_meshes = parse_urdf_meshes(URDF_PATH)

    # Load GT mask
    gt_img = cv2.imread(os.path.join(OUTPUT_DIR, "mask.png"), cv2.IMREAD_UNCHANGED)
    if gt_img is not None and gt_img.ndim == 3 and gt_img.shape[2] == 4:
        gt_mask = gt_img[:, :, 3] < 128
    else:
        gt_mask = None

    print("Rendering convex hull...")
    import time
    t0 = time.time()
    mask_hull = render_convex_hull(link_meshes, MESH_DIR, transforms, BEST_PARAMS, ref_t, ref_R, h, w)
    t_hull = time.time() - t0

    print("Rendering triangles (no morph)...")
    t0 = time.time()
    mask_tri_raw = render_triangles(link_meshes, MESH_DIR, transforms, BEST_PARAMS, ref_t, ref_R, h, w,
                                    morph_close=False)
    t_tri_raw = time.time() - t0

    print("Rendering triangles + morph close (k=5)...")
    t0 = time.time()
    mask_tri_close = render_triangles(link_meshes, MESH_DIR, transforms, BEST_PARAMS, ref_t, ref_R, h, w,
                                      morph_close=True, kernel_size=5)
    t_tri_close = time.time() - t0

    print(f"\nTiming: hull={t_hull:.3f}s  tri_raw={t_tri_raw:.3f}s  tri_close={t_tri_close:.3f}s")

    # Compute IoU if GT available
    if gt_mask is not None:
        for name, mask in [("convex_hull", mask_hull), ("tri_raw", mask_tri_raw), ("tri_close", mask_tri_close)]:
            inter = np.count_nonzero(mask & gt_mask)
            union = np.count_nonzero(mask | gt_mask)
            iou = inter / union if union > 0 else 0
            recall = inter / np.count_nonzero(gt_mask) if np.count_nonzero(gt_mask) > 0 else 0
            precision = inter / np.count_nonzero(mask) if np.count_nonzero(mask) > 0 else 0
            print(f"  {name:12s}: IoU={iou:.4f}  recall={recall:.4f}  precision={precision:.4f}  "
                  f"pixels={np.count_nonzero(mask)}")

    # Side-by-side: mask on original frame, convex hull vs triangles+morph
    # Draw mask contour on original image for clear boundary comparison
    def draw_mask_on_img(img, mask, label):
        vis = img.copy()
        # Semi-transparent fill
        vis[mask] = (vis[mask] * 0.5 + np.array([0, 200, 255]) * 0.5).astype(np.uint8)
        # Draw contour
        contours, _ = cv2.findContours(mask.astype(np.uint8) * 255,
                                        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours, -1, (0, 0, 255), 2)
        cv2.putText(vis, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        return vis

    vis_hull = draw_mask_on_img(img, mask_hull, "Convex Hull")
    vis_tri = draw_mask_on_img(img, mask_tri_close, "Triangles + Morph Close")
    result = np.hstack([vis_hull, vis_tri])

    out_path = os.path.join(OUTPUT_DIR, "render_compare.png")
    cv2.imwrite(out_path, result)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
