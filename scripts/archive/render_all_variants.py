#!/usr/bin/env python3
"""Render front-view comparison of all G1 body variants and hand variants.

Usage:
  python scripts/render_all_variants.py
"""
import os, sys, traceback
import numpy as np
import cv2
import xml.etree.ElementTree as ET
import pinocchio as pin
from stl import mesh as stl_mesh

sys.stdout.reconfigure(line_buffering=True)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCAL_MESH = os.path.join(BASE, "data", "unitree_G1_WBT", "mesh", "meshes")
REPO_MESH = "/tmp/unitree_ros/robots/g1_description/meshes"
OUTPUT_DIR = os.path.join(BASE, "test_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# (label, urdf_path, [mesh_dirs], use_freeflyer)
BODY_MODELS = [
    ("G1 23DOF (Basic)",
     "/tmp/unitree_ros/robots/g1_description/g1_23dof_rev_1_0.urdf",
     [REPO_MESH, LOCAL_MESH], True),
    ("G1 29DOF (EDU Plus)",
     os.path.join(BASE, "data/unitree_G1_WBT/mesh/g1_29dof_rev_1_0.urdf"),
     [LOCAL_MESH, REPO_MESH], True),
    ("G1 29DOF + Unitree Hand",
     os.path.join(BASE, "data/unitree_G1_WBT/mesh/g1_29dof_with_hand_rev_1_0.urdf"),
     [LOCAL_MESH, REPO_MESH], True),
    ("G1 29DOF + Inspire FTP",
     os.path.join(BASE, "data/unitree_G1_WBT/mesh/g1_29dof_rev_1_0_with_inspire_hand_FTP.urdf"),
     [LOCAL_MESH, REPO_MESH], True),
]

HAND_MODELS = [
    ("Unitree Dex3",
     "/tmp/xr_teleoperate/assets/unitree_hand/unitree_dex3_left.urdf",
     ["/tmp/xr_teleoperate/assets/unitree_hand/meshes"], False),
    ("BrainCo Revo2",
     "/tmp/xr_teleoperate/assets/brainco_hand/brainco_left.urdf",
     ["/tmp/xr_teleoperate/assets/brainco_hand/meshes"], False),
    ("Inspire RH56",
     "/tmp/xr_teleoperate/assets/inspire_hand/inspire_hand_left.urdf",
     ["/tmp/xr_teleoperate/assets/inspire_hand/meshes"], False),
]

SKIP_BODY = {"head_link", "logo_link", "d435_link", "mid360_link"}


def parse_urdf_meshes(urdf_path):
    """Extract link_name -> mesh_filename, handling multiple <visual> per link."""
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    link_meshes = {}
    for link_elem in root.findall("link"):
        name = link_elem.get("name")
        for visual in link_elem.findall("visual"):
            geom = visual.find("geometry")
            if geom is None:
                continue
            mesh_elem = geom.find("mesh")
            if mesh_elem is not None:
                fn = mesh_elem.get("filename")
                if fn:
                    link_meshes[name] = os.path.basename(fn)
                break
    return link_meshes


def load_meshes(link_meshes, mesh_dirs, subsample=1):
    cache = {}
    for link_name, filename in link_meshes.items():
        path = None
        for d in mesh_dirs:
            p = os.path.join(d, filename)
            if os.path.exists(p):
                path = p
                break
        if path is None:
            print(f"    [WARN] mesh not found: {filename}")
            continue
        m = stl_mesh.Mesh.from_file(path)
        verts = m.vectors
        flat = verts.reshape(-1, 3)
        valid = np.all(np.isfinite(flat), axis=1).reshape(-1, 3).all(axis=1)
        tris = verts[valid]
        if subsample > 1:
            tris = tris[::subsample]
        if len(tris) > 0:
            cache[link_name] = tris
    return cache


def render_front_view(urdf_path, mesh_dirs, img_size=(800, 500),
                      skip_links=None, use_freeflyer=True, wireframe=False):
    """Render orthographic front view at neutral pose. Returns BGR image."""
    if skip_links is None:
        skip_links = SKIP_BODY
    h, w = img_size

    if use_freeflyer:
        model = pin.buildModelFromUrdf(urdf_path, pin.JointModelFreeFlyer())
    else:
        model = pin.buildModelFromUrdf(urdf_path)
    data = model.createData()
    q = pin.neutral(model)
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)

    transforms = {}
    for i in range(model.nframes):
        name = model.frames[i].name
        T = data.oMf[i]
        transforms[name] = (T.translation.copy(), T.rotation.copy())

    link_meshes = parse_urdf_meshes(urdf_path)
    mesh_cache = load_meshes(link_meshes, mesh_dirs)
    print(f"    Loaded {len(mesh_cache)} meshes, nq={model.nq}")

    all_tris_world = []
    all_depths = []
    all_link_names = []
    for link_name, tris in mesh_cache.items():
        if link_name in skip_links or link_name not in transforms:
            continue
        t, R = transforms[link_name]
        flat = tris.reshape(-1, 3)
        world = (R @ flat.T).T + t
        wt = world.reshape(-1, 3, 3)
        depths = wt[:, :, 0].mean(axis=1)  # X = forward = depth
        all_tris_world.append(wt)
        all_depths.append(depths)
        all_link_names.extend([link_name] * len(wt))

    if not all_tris_world:
        img = np.ones((h, w, 3), dtype=np.uint8) * 240
        cv2.putText(img, "No meshes", (w // 4, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 200), 2)
        return img

    all_tris = np.concatenate(all_tris_world, axis=0)
    all_depths_arr = np.concatenate(all_depths, axis=0)

    # Auto-fit bounding box in YZ plane
    pts = all_tris.reshape(-1, 3)
    y_min, y_max = pts[:, 1].min(), pts[:, 1].max()
    z_min, z_max = pts[:, 2].min(), pts[:, 2].max()
    pad = max(y_max - y_min, z_max - z_min) * 0.08
    y_range = y_max - y_min + 2 * pad
    z_range = z_max - z_min + 2 * pad
    scale = min(w / y_range, h / z_range) * 0.92
    y_center = (y_min + y_max) / 2
    z_center = (z_min + z_max) / 2
    cx, cy = w / 2, h / 2

    # Painter's algorithm: draw far triangles first
    order = np.argsort(all_depths_arr)
    bg = 255 if wireframe else 240
    img = np.ones((h, w, 3), dtype=np.uint8) * bg

    for idx in order:
        tri = all_tris[idx]
        ln = all_link_names[idx]

        # Orthographic front view: X=depth, Y=horizontal, Z=vertical
        pts2d = np.zeros((3, 2), dtype=np.float32)
        pts2d[:, 0] = scale * (-(tri[:, 1] - y_center)) + cx
        pts2d[:, 1] = scale * (-(tri[:, 2] - z_center)) + cy

        ln_l = ln.lower()

        if wireframe:
            if "left" in ln_l:
                color = (180, 80, 40)
            elif "right" in ln_l:
                color = (40, 80, 180)
            else:
                color = (80, 80, 80)
            cv2.polylines(img, [pts2d.astype(np.int32)], True, color, 1, cv2.LINE_AA)
        else:
            if "left" in ln_l:
                base = np.array([200, 120, 80])
            elif "right" in ln_l:
                base = np.array([80, 120, 200])
            else:
                base = np.array([140, 145, 145])

            v0, v1, v2 = tri
            normal = np.cross(v1 - v0, v2 - v0)
            nlen = np.linalg.norm(normal)
            shade = (abs(normal[0] / nlen) * 0.4 + 0.6) if nlen > 1e-10 else 0.7
            color = tuple(int(c) for c in (base * shade).clip(0, 255).astype(int))
            cv2.fillConvexPoly(img, pts2d.astype(np.int32), color)

    return img


def add_label(canvas, text, x, y, w, font_scale=0.6):
    ts = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)[0]
    tx = x + (w - ts[0]) // 2
    cv2.putText(canvas, text, (tx, y), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (30, 30, 30), 2, cv2.LINE_AA)


def main():
    # ── Body variants ──
    print("=== Body variants ===")
    body_imgs, body_names = [], []
    for name, urdf, mdirs, ff in BODY_MODELS:
        print(f"  {name}")
        try:
            img = render_front_view(urdf, mdirs, img_size=(800, 500),
                                    use_freeflyer=ff)
        except Exception:
            traceback.print_exc()
            img = np.ones((800, 500, 3), dtype=np.uint8) * 240
        body_imgs.append(img)
        body_names.append(name)

    gap = 12
    pw, ph = body_imgs[0].shape[1], body_imgs[0].shape[0]
    cw = pw * len(body_imgs) + gap * (len(body_imgs) - 1)
    ch = ph + 50
    canvas = np.ones((ch, cw, 3), dtype=np.uint8) * 255
    for i, (img, name) in enumerate(zip(body_imgs, body_names)):
        x0 = i * (pw + gap)
        canvas[50:50 + ph, x0:x0 + pw] = img
        add_label(canvas, name, x0, 35, pw)
    out = os.path.join(OUTPUT_DIR, "g1_body_variants.png")
    cv2.imwrite(out, canvas)
    print(f"  -> {out}")

    # ── Hand variants ──
    print("\n=== Hand variants ===")
    hand_imgs, hand_names = [], []
    for name, urdf, mdirs, ff in HAND_MODELS:
        print(f"  {name}")
        try:
            img = render_front_view(urdf, mdirs, img_size=(500, 450),
                                    skip_links=set(), use_freeflyer=ff)
        except Exception:
            traceback.print_exc()
            img = np.ones((500, 450, 3), dtype=np.uint8) * 240
        hand_imgs.append(img)
        hand_names.append(name)

    pw2, ph2 = hand_imgs[0].shape[1], hand_imgs[0].shape[0]
    cw2 = pw2 * len(hand_imgs) + gap * (len(hand_imgs) - 1)
    ch2 = ph2 + 50
    canvas2 = np.ones((ch2, cw2, 3), dtype=np.uint8) * 255
    for i, (img, name) in enumerate(zip(hand_imgs, hand_names)):
        x0 = i * (pw2 + gap)
        canvas2[50:50 + ph2, x0:x0 + pw2] = img
        add_label(canvas2, name, x0, 35, pw2, font_scale=0.7)
    out2 = os.path.join(OUTPUT_DIR, "g1_hand_variants.png")
    cv2.imwrite(out2, canvas2)
    print(f"  -> {out2}")

    # ── EDU Plus wireframe (torso only, 5x resolution) ──
    print("\n=== EDU Plus wireframe (torso only) ===")
    _, urdf_29, mdirs_29, _ = BODY_MODELS[1]
    torso_only = {
        "pelvis", "pelvis_contour_link",
        "waist_yaw_link", "waist_roll_link", "torso_link",
    }
    all_links = set(parse_urdf_meshes(urdf_29).keys())
    skip_non_torso = (all_links - torso_only) | SKIP_BODY
    wf_img = render_front_view(urdf_29, mdirs_29, img_size=(5000, 3500),
                               skip_links=skip_non_torso,
                               use_freeflyer=True, wireframe=True)
    wf_h, wf_w = wf_img.shape[:2]
    wf_canvas = np.ones((wf_h + 120, wf_w, 3), dtype=np.uint8) * 255
    wf_canvas[120:] = wf_img
    add_label(wf_canvas, "G1 29DOF (EDU Plus) - Torso Wireframe", 0, 80, wf_w, 2.5)
    out3 = os.path.join(OUTPUT_DIR, "g1_29dof_wireframe.png")
    cv2.imwrite(out3, wf_canvas)
    print(f"  -> {out3}")


if __name__ == "__main__":
    main()
