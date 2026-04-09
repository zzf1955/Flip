"""
Interactive camera calibration for G1 head_stereo_left.
6 trackbars: dx, dz, pitch, fx, k1, k2.
White semi-transparent mesh + skeleton overlay on video frame.
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
TARGET_FRAME = 325

# Only limb bones for skeleton overlay
SKELETON_BONES = [
    ("left_hip_yaw_link", "left_knee_link"),
    ("left_knee_link", "left_ankle_pitch_link"),
    ("left_ankle_pitch_link", "left_ankle_roll_link"),
    ("right_hip_yaw_link", "right_knee_link"),
    ("right_knee_link", "right_ankle_pitch_link"),
    ("right_ankle_pitch_link", "right_ankle_roll_link"),
    ("left_elbow_link", "left_wrist_roll_link"),
    ("left_wrist_roll_link", "left_wrist_pitch_link"),
    ("left_wrist_pitch_link", "left_wrist_yaw_link"),
    ("right_elbow_link", "right_wrist_roll_link"),
    ("right_wrist_roll_link", "right_wrist_pitch_link"),
    ("right_wrist_pitch_link", "right_wrist_yaw_link"),
]

# Render limb meshes (skip torso/waist/pelvis/head to avoid camera clipping, skip rubber_hand too heavy)
SKIP_MESHES = {
    "head_link", "logo_link", "d435_link",
    "left_rubber_hand", "right_rubber_hand",
}

MESH_SUBSAMPLE = 3  # render every Nth triangle for speed


def bone_color(a, b):
    if "left" in a or "left" in b: return (255, 128, 0)
    if "right" in a or "right" in b: return (0, 128, 255)
    return (0, 255, 128)


def joint_color(n):
    if "left" in n: return (255, 128, 0)
    if "right" in n: return (0, 128, 255)
    return (0, 255, 128)


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


def skeleton_links():
    s = set()
    for a, b in SKELETON_BONES:
        s.add(a); s.add(b)
    return sorted(s)


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


def load_link_verts(link_meshes, mesh_dir, transforms):
    """Load STL meshes, transform to world frame, return per-link unique vertices."""
    link_verts = {}
    total_verts = 0

    for link_name, filename in link_meshes.items():
        if link_name in SKIP_MESHES:
            continue
        if link_name not in transforms:
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
        flat = np.unique(flat, axis=0)

        R = transforms[link_name][1]
        t = transforms[link_name][0]
        world = (R @ flat.T).T + t
        link_verts[link_name] = world.astype(np.float64)
        total_verts += len(world)

    print(f"Mesh: {len(link_verts)} links, {total_verts} unique vertices")
    return link_verts


def project_points(pts3d, dx, dy, dz, pitch_deg, fx, fy, k1, k2, transforms):
    """Project 3D points to 2D with camera params and distortion."""
    cx, cy = 320.0, 240.0
    pitch = np.radians(pitch_deg)
    ref_t, ref_R = transforms["torso_link"]

    R_pitch = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    R_body_to_cam = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]], dtype=np.float64)
    R_cam = R_body_to_cam @ R_pitch

    cam_pos = ref_t + ref_R @ np.array([dx, dy, dz])
    R_w2c = (ref_R @ R_cam.T).T
    t_w2c = R_w2c @ (-cam_pos)

    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    dist = np.array([k1, k2, 0, 0, 0], dtype=np.float64)
    rvec, _ = cv2.Rodrigues(R_w2c)
    tvec = t_w2c.reshape(3, 1)

    pts2d, _ = cv2.projectPoints(pts3d, rvec, tvec, K, dist)
    pts2d = pts2d.reshape(-1, 2)

    # Depth for each point (z in camera frame)
    depths = (R_w2c @ pts3d.T).T + t_w2c.flatten()
    z_cam = depths[:, 2]

    return pts2d, z_cam


def render_mesh(img, link_verts, dx, dy, dz, pitch_deg, fx, fy, k1, k2, transforms, h, w):
    """Render filled convex hull per link overlay."""
    cx, cy = 320.0, 240.0
    pitch = np.radians(pitch_deg)
    ref_t, ref_R = transforms["torso_link"]

    R_pitch = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    R_body_to_cam = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]], dtype=np.float64)
    R_cam = R_body_to_cam @ R_pitch

    cam_pos = ref_t + ref_R @ np.array([dx, dy, dz])
    R_w2c = (ref_R @ R_cam.T).T
    t_w2c = R_w2c @ (-cam_pos)

    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    dist = np.array([k1, k2, 0, 0, 0], dtype=np.float64)
    rvec, _ = cv2.Rodrigues(R_w2c)
    tvec = t_w2c.reshape(3, 1)

    result = img.copy()
    overlay = np.zeros_like(img)
    any_visible = False

    for verts3d in link_verts.values():
        depths = (R_w2c @ verts3d.T).T + t_w2c.flatten()
        z_cam = depths[:, 2]
        in_front = z_cam > 0.01
        if np.count_nonzero(in_front) < 3:
            continue

        pts2d, _ = cv2.projectPoints(verts3d[in_front], rvec, tvec, K, dist)
        pts2d = pts2d.reshape(-1, 2)
        finite = np.all(np.isfinite(pts2d), axis=1)
        pts2d = pts2d[finite]
        if len(pts2d) < 3:
            continue

        hull = cv2.convexHull(pts2d.astype(np.float32))
        cv2.fillConvexPoly(overlay, hull.astype(np.int32), (0, 255, 255))
        cv2.polylines(result, [hull.astype(np.int32)], True, (0, 255, 255), 1, cv2.LINE_AA)
        any_visible = True

    if not any_visible:
        cv2.putText(result, "NO MESH VISIBLE", (10, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        return result

    # Semi-transparent fill
    cv2.addWeighted(overlay, 0.25, result, 1.0, 0, result)
    return result


def draw_skeleton(canvas, name2pt):
    """Draw skeleton bones and joints."""
    for a, b in SKELETON_BONES:
        if a not in name2pt or b not in name2pt:
            continue
        pa, pb = name2pt[a], name2pt[b]
        cv2.line(canvas, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])),
                 bone_color(a, b), 2, cv2.LINE_AA)
    for name, pt in name2pt.items():
        p = (int(pt[0]), int(pt[1]))
        cv2.circle(canvas, p, 4, joint_color(name), -1, cv2.LINE_AA)
        cv2.putText(canvas, short_name(name), (p[0] + 5, p[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


WINDOW = "G1 Camera Calibration"


def nothing(x):
    pass


def main():
    print("Loading URDF...")
    model = pin.buildModelFromUrdf(URDF_PATH, pin.JointModelFreeFlyer())
    data = model.createData()

    print("Loading parquet...")
    df = pd.read_parquet(PARQUET_PATH)
    row = df[(df["episode_index"] == TARGET_EP) & (df["frame_index"] == TARGET_FRAME)].iloc[0]
    rq = np.array(row["observation.state.robot_q_current"], dtype=np.float64)

    q = build_q(model, rq)
    transforms = do_fk(model, data, q)

    print("Extracting video frame...")
    img = extract_frame(VIDEO_PATH, TARGET_FRAME)
    if img is None:
        print("ERROR: Could not extract frame"); return
    h, w = img.shape[:2]
    print(f"Frame size: {w}x{h}")

    # Load meshes per link
    print("Loading meshes...")
    link_meshes = parse_urdf_meshes(URDF_PATH)
    all_verts = load_link_verts(link_meshes, MESH_DIR, transforms)

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, 960, 720)

    # Defaults: consensus extrinsics + estimated intrinsics (square pixels, barrel distortion)
    cv2.createTrackbar("dx+300", WINDOW, 360, 600, nothing)        # 60mm
    cv2.createTrackbar("dy+100", WINDOW, 127, 200, nothing)        # 27mm (left offset)
    cv2.createTrackbar("dz+500", WINDOW, 950, 1500, nothing)       # 450mm
    cv2.createTrackbar("pitch+90", WINDOW, 44, 210, nothing)       # -46 deg
    cv2.createTrackbar("fx", WINDOW, 350, 1500, nothing)           # 350 (start point)
    cv2.createTrackbar("fy", WINDOW, 350, 1500, nothing)           # = fx (square pixels)
    cv2.createTrackbar("k1+500", WINDOW, 200, 500, nothing)        # k1=-0.300
    cv2.createTrackbar("k2+200", WINDOW, 310, 400, nothing)        # k2=+0.110

    print("\nControls (relative to torso_link):")
    print("  dx+300     : forward offset (-300~+300mm)")
    print("  dy+100     : left offset (-100~+100mm)")
    print("  dz+500     : upward offset (-500~+1000mm)")
    print("  pitch+90   : tilt angle (-90~+120, negative=look down)")
    print("  fx         : horizontal focal length (50~1500 px)")
    print("  fy         : vertical focal length (50~1500 px)")
    print("  k1+500     : barrel distortion k1 (-0.500~0.000)")
    print("  k2+200     : distortion k2 (-0.200~+0.200)")
    print("\n  's' save  'q'/ESC quit")

    prev_params = None

    while True:
        dx_mm = cv2.getTrackbarPos("dx+300", WINDOW) - 300
        dy_mm = cv2.getTrackbarPos("dy+100", WINDOW) - 100
        dz_mm = cv2.getTrackbarPos("dz+500", WINDOW) - 500
        pitch_deg = cv2.getTrackbarPos("pitch+90", WINDOW) - 90
        fx = cv2.getTrackbarPos("fx", WINDOW)
        fy = cv2.getTrackbarPos("fy", WINDOW)
        k1_raw = cv2.getTrackbarPos("k1+500", WINDOW) - 500
        k2_raw = cv2.getTrackbarPos("k2+200", WINDOW) - 200

        dx = dx_mm / 1000.0
        dy = dy_mm / 1000.0
        dz = dz_mm / 1000.0
        k1 = k1_raw / 1000.0
        k2 = k2_raw / 1000.0
        if fx < 50: fx = 50
        if fy < 50: fy = 50

        params = (dx, dy, dz, pitch_deg, fx, fy, k1, k2)

        if params != prev_params:
            prev_params = params

            # Project skeleton
            skel_pts3d = np.array([transforms[n][0] for n in skeleton_links() if n in transforms], dtype=np.float64)
            skel_2d, skel_z = project_points(skel_pts3d, dx, dy, dz, pitch_deg, fx, fy, k1, k2, transforms)
            links = [n for n in skeleton_links() if n in transforms]
            name2pt = {}
            for i, n in enumerate(links):
                if skel_z[i] > 0.01:
                    name2pt[n] = (skel_2d[i][0], skel_2d[i][1])

            # Render mesh with convex hull per link
            overlay = render_mesh(img, all_verts, dx, dy, dz, pitch_deg, fx, fy, k1, k2, transforms, h, w)
            draw_skeleton(overlay, name2pt)

            params_text = f"dx={dx:.3f} dy={dy:.3f} dz={dz:.3f} p={pitch_deg} fx={fx} fy={fy} k1={k1:.3f} k2={k2:.3f}"
            cv2.putText(overlay, params_text, (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 255), 1, cv2.LINE_AA)

            # Stack: top=original, bottom=overlay
            canvas = np.vstack([img, overlay])
            cv2.imshow(WINDOW, canvas)

        key = cv2.waitKey(30) & 0xFF
        if key == ord('q') or key == 27:
            print(f"\nFinal parameters:")
            print(f"  dx    = {dx:.3f} m")
            print(f"  dy    = {dy:.3f} m")
            print(f"  dz    = {dz:.3f} m")
            print(f"  pitch = {pitch_deg} deg")
            print(f"  fx    = {fx} px")
            print(f"  fy    = {fy} px")
            print(f"  k1    = {k1:.4f}")
            print(f"  k2    = {k2:.4f}")
            break
        elif key == ord('s'):
            out_path = os.path.join(OUTPUT_DIR, "calibrated_overlay.png")
            cv2.imwrite(out_path, canvas)
            print(f"Saved {out_path}")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
