"""
Motion retargeting diagnostic visualization: pure joint-rotation copy from G1 -> SMPLH.

No IK, no optimization. Algorithm:
  1. G1 FK at neutral pose -> rest-pose link rotations.
  2. G1 FK at target frame -> current link rotations.
  3. World-frame rotation delta: dR[link] = R_current @ R_rest^T.
  4. Convert delta from G1 frame to SMPLH frame via R_S2G / R_G2S.
  5. Map SMPLH body joints (0-21) to corresponding G1 links; unmapped joints
     inherit parent's world rotation (-> local rotation = I).
  6. Convert world rotations to local rotations via parent traversal.
  7. body_pose[i-1] = axis-angle(R_local[i]).

Then render 3-view debug (G1 mesh + SMPLH mesh + keypoints).

Usage:
  python scripts/retarget_copy.py --episode 0 --frame 30
"""

import sys
import os
import argparse
import numpy as np
import pinocchio as pin
import cv2
import torch
from stl import mesh as stl_mesh

sys.stdout.reconfigure(line_buffering=True)

from src.core.config import G1_URDF, MESH_DIR, OUTPUT_DIR, DATASET_ROOT, get_hand_type, get_skip_meshes
from src.core.fk import build_q, do_fk, parse_urdf_meshes
from src.core.data import load_episode_info
from src.core.smplh import (SMPLHForIK, extract_g1_targets, G1_KEYPOINTS,
                      R_SMPLH_TO_G1_NP)
from src.core.retarget import (retarget_frame, refine_arms, compute_g1_rest_transforms,
                               scale_hands, build_default_hand_pose,
                               build_thumb_base_pose, build_finger_curl_pose,
                               apply_finger_curl_from_g1,
                               BONE_MAP, SPINE_JOINTS, DIRECT_ROT_MAP,
                               SHOULDER_TWIST_MAP, WRIST_LOCAL_CORRECTION_DEG,
                               FINGER_CURL_AXIS, FINGER_CURL_MAX_DEG,
                               FINGER_CURL_SIGN_L, FINGER_CURL_SIGN_R,
                               THUMB_DEFAULT_L_AXIS, THUMB_DEFAULT_R_AXIS,
                               THUMB_DEFAULT_ANGLE_DEG, FINGER_SLOTS,
                               rot_between, extract_twist_angle, rot_to_axis_angle)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection


# ── Rendering helpers (same style as render_ik_debug.py) ──

def oblique_project(pts, azim_deg=35, elev_deg=25):
    az, el = np.radians(azim_deg), np.radians(elev_deg)
    Rz = np.array([[np.cos(az), -np.sin(az), 0],
                   [np.sin(az),  np.cos(az), 0],
                   [0, 0, 1]])
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(el), -np.sin(el)],
                   [0, np.sin(el),  np.cos(el)]])
    R = Rx @ Rz
    rot = (R @ pts.T).T
    return rot[:, [0, 2]], rot[:, 1]


def load_g1_tris(transforms):
    link_meshes = parse_urdf_meshes(G1_URDF)
    # Render head + Inspire hand (force, regardless of task type).
    # Only skip decorative / sensor links.
    skip = {"logo_link", "d435_link"}
    out = []
    for link_name, filename in link_meshes.items():
        if link_name in skip or link_name not in transforms:
            continue
        path = os.path.join(MESH_DIR, filename)
        if not os.path.exists(path):
            continue
        m = stl_mesh.Mesh.from_file(path)
        tris = m.vectors.copy()
        t_link, R_link = transforms[link_name]
        flat = tris.reshape(-1, 3)
        world = (R_link @ flat.T).T + t_link
        out.append(world.reshape(-1, 3, 3))
    return np.concatenate(out, axis=0) if out else np.zeros((0, 3, 3))


def render_tris(ax, tris, azim, elev, color, alpha=1.0, edge_lw=0.0,
                reverse_depth=False, shade=True):
    """Vectorized triangle render with Lambertian shading.

    tris: (N, 3, 3) numpy, world coords
    color: matplotlib color (name, hex, or RGB tuple)
    """
    if len(tris) == 0:
        return
    import matplotlib.colors as mcolors

    # ── Build view rotation R once ──
    az, el = np.radians(azim), np.radians(elev)
    Rz = np.array([[np.cos(az), -np.sin(az), 0],
                   [np.sin(az),  np.cos(az), 0],
                   [0, 0, 1]])
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(el), -np.sin(el)],
                   [0, np.sin(el),  np.cos(el)]])
    R = Rx @ Rz

    # ── Vectorized projection ──
    N = len(tris)
    flat = tris.reshape(-1, 3)                    # (3N, 3)
    rot = flat @ R.T                              # (3N, 3)
    rot = rot.reshape(N, 3, 3)
    screen = rot[:, :, [0, 2]]                    # (N, 3, 2)
    depth = rot[:, :, 1].mean(axis=1)             # (N,)

    # ── Lambertian shading ──
    if shade:
        v0, v1, v2 = tris[:, 0], tris[:, 1], tris[:, 2]
        normals = np.cross(v1 - v0, v2 - v0)
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = normals / np.maximum(norms, 1e-9)
        # View direction in world = R.T @ [0, 1, 0] (screen "into page" axis)
        view_world = R.T @ np.array([0.0, 1.0, 0.0])
        dots = np.abs(normals @ view_world)       # (N,)
        shade_factor = 0.35 + 0.65 * dots
    else:
        shade_factor = np.ones(N)

    # ── Depth sort ──
    if reverse_depth:
        order = np.argsort(-depth)
    else:
        order = np.argsort(depth)
    screen = screen[order]
    shade_factor = shade_factor[order]

    # ── Per-triangle RGBA ──
    base_rgb = np.array(mcolors.to_rgb(color))
    face_rgb = np.clip(base_rgb[None, :] * shade_factor[:, None], 0.0, 1.0)
    face_rgba = np.concatenate(
        [face_rgb, np.full((N, 1), alpha)], axis=1)

    pc = PolyCollection(screen, facecolors=face_rgba,
                        edgecolors='none', linewidths=edge_lw)
    ax.add_collection(pc)


def render_keypoints(ax, points, labels, azim, elev, color='red', ms=8,
                     show_labels=True):
    pts = np.array(points)
    screen, _ = oblique_project(pts, azim, elev)
    for i, (sx, sy) in enumerate(screen):
        ax.plot(sx, sy, 'o', color=color, markersize=ms,
                markeredgecolor='white', markeredgewidth=1.5, zorder=100)
        if not show_labels:
            continue
        ox = 0.02 if i % 2 == 1 else -0.02
        ha = 'left' if i % 2 == 1 else 'right'
        ax.annotate(labels[i], (sx, sy),
                    xytext=(sx + ox, sy + 0.01),
                    fontsize=8, fontweight='bold', color=color,
                    bbox=dict(boxstyle='round,pad=0.15', fc='white',
                              ec=color, alpha=0.85),
                    arrowprops=dict(arrowstyle='->', color=color, lw=1),
                    ha=ha, zorder=101)


def main():
    parser = argparse.ArgumentParser(description="Pure rotation-copy retarget")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--frame", type=int, default=30)
    parser.add_argument("--beta", type=float, nargs='*', default=None)
    parser.add_argument("--scale", type=float, default=0.75,
                        help="Body scale (default 0.75)")
    parser.add_argument("--hand-scale", type=float, default=1.3,
                        help="Hand-only scale multiplier applied AFTER body "
                             "scale, around each wrist. SMPLH @ body-scale "
                             "0.85 has hands smaller than G1 Inspire; 1.3 "
                             "is a good default.")
    parser.add_argument("--wrist-rot", type=float, nargs=3,
                        default=[0.0, 0.0, 0.0], metavar=("RX", "RY", "RZ"),
                        help="ADDITIONAL wrist local-frame rotation on "
                             "top of the constant WRIST_LOCAL_CORRECTION_DEG. "
                             "Diagnostic only; default (0, 0, 0) means no "
                             "extra rotation. Mirrored L/R.")
    parser.add_argument("--wrist-batch", type=str, default=None,
                        help="Render multiple wrist rotations in one "
                             "process (shares model loading). Format: "
                             "'RX,RY,RZ;RX,RY,RZ;...'. Example: "
                             "'0,0,0;30,0,0;-30,0,0;0,-30,-30'")
    parser.add_argument("--finger-test", type=float, default=0.0,
                        metavar="DEG",
                        help="Diagnostic: render 3 variants with all 4 "
                             "non-thumb fingers curled by DEG per segment "
                             "around local X / Y / Z axes. 0 = off.")
    parser.add_argument("--finger-rot", type=str, default=None,
                        metavar="AXIS,L_DEG,R_DEG",
                        help="Single finger curl variant, with separate "
                             "L/R angles. Example: --finger-rot z,-40,40")
    parser.add_argument("--thumb-test", type=float, default=0.0,
                        metavar="DEG",
                        help="Diagnostic: render 3 variants with ONLY the "
                             "thumb base joint (thumb1) rotated by DEG "
                             "around local X / Y / Z. Mirrored L=-DEG R=+DEG.")
    parser.add_argument("--thumb-rot", type=str, default=None,
                        metavar="AXIS,L_DEG,R_DEG",
                        help="Single thumb-base rotation variant with "
                             "separate L/R angles. Example: "
                             "--thumb-rot y,30,-30")
    parser.add_argument("--base-offset", type=float, default=-0.10,
                        dest="base_offset",
                        help="Base-point offset in G1 +X (meters). Positive "
                             "= base point forward → mesh visually moves back. "
                             "Default -0.10 (base back → mesh forward 10cm).")
    parser.add_argument("--live-hand", dest="live_hand", action="store_true",
                        default=False,
                        help="Use G1 hand_state from episode data to drive "
                             "SMPLH finger pose (Inspire only)")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--refine", action="store_true",
                        help="Enable arm IK refinement (default: off)")
    parser.add_argument("--w-drift", type=float, default=10.0, dest="w_drift",
                        help="Arm IK: joint drift weight (default 10)")
    parser.add_argument("--task", type=str, default=None,
                        help="Override ACTIVE_TASK (e.g. "
                             "G1_WBT_Inspire_Pickup_Pillow_MainCamOnly)")
    args = parser.parse_args()

    out_dir = os.path.join(OUTPUT_DIR, "human", "retarget_diag")
    os.makedirs(out_dir, exist_ok=True)

    # ── Load G1 model + episode data ──
    print("Loading G1 + episode...")
    model_g = pin.buildModelFromUrdf(G1_URDF, pin.JointModelFreeFlyer())
    data_g = model_g.createData()

    if args.task is not None:
        data_dir = os.path.join(DATASET_ROOT, args.task)
        video_path, from_ts, to_ts, ep_df = load_episode_info(
            args.episode, data_dir=data_dir)
        hand_type = get_hand_type(args.task)
    else:
        video_path, from_ts, to_ts, ep_df = load_episode_info(args.episode)
        hand_type = get_hand_type()
    frame_row = ep_df[ep_df["frame_index"] == args.frame]
    if len(frame_row) == 0:
        frame_row = ep_df.iloc[[0]]
    row = frame_row.iloc[0]
    rq = np.array(row["observation.state.robot_q_current"], dtype=np.float64)
    hs = np.array(row["observation.state.hand_state"], dtype=np.float64)

    q = build_q(model_g, rq, hs, hand_type=hand_type)
    transforms = do_fk(model_g, data_g, q)

    # ── Compute rest transforms (for rotation delta) ──
    print("Computing G1 rest transforms...")
    rest_transforms = compute_g1_rest_transforms()

    # ── Load SMPLH model ──
    print("Loading SMPLH...")
    smplh = SMPLHForIK(device=args.device)

    betas = None
    if args.beta:
        betas = np.zeros(16)
        for i, b in enumerate(args.beta[:16]):
            betas[i] = b

    # Default scale: blend of leg and torso ratios.
    # - Leg ratio ≈ 0.88 (pelvis → ankle)
    # - Torso ratio ≈ 0.42 (pelvis → shoulder)   G1 is short-torso
    # Using blend 0.5/0.5 so the human overall size is between the two.
    # root_trans is later shifted to plant feet, so scale independence works.
    if args.scale is None:
        J_rest_np = smplh._J_rest.cpu().numpy()
        smplh_leg = np.linalg.norm(J_rest_np[0] - J_rest_np[7])
        smplh_torso = np.linalg.norm(J_rest_np[0] - J_rest_np[16])  # pelvis → L shoulder

        tmp_model = pin.buildModelFromUrdf(G1_URDF, pin.JointModelFreeFlyer())
        tmp_data = tmp_model.createData()
        q0 = pin.neutral(tmp_model)
        q0[0:3] = [0, 0, 0.75]
        q0[3:7] = [0, 0, 0, 1]
        pin.forwardKinematics(tmp_model, tmp_data, q0)
        pin.updateFramePlacements(tmp_model, tmp_data)
        g1_p = {tmp_model.frames[i].name: tmp_data.oMf[i].translation.copy()
                for i in range(tmp_model.nframes)}
        g1_leg = np.linalg.norm(g1_p['pelvis'] - g1_p['left_ankle_roll_link'])
        g1_torso = np.linalg.norm(g1_p['pelvis'] - g1_p['left_shoulder_pitch_link'])

        leg_r = g1_leg / smplh_leg
        torso_r = g1_torso / smplh_torso
        chosen_scale = 0.5 * leg_r + 0.5 * torso_r
        print(f"  Scale: {chosen_scale:.4f} "
              f"(leg_r={leg_r:.3f}, torso_r={torso_r:.3f})")
    else:
        chosen_scale = args.scale

    # ── Step 0: Scale ──
    print(f"Step 0: body_scale = {chosen_scale:.4f}, hand_scale = {args.hand_scale}")
    J_shaped, v_shaped = smplh.shape_blend(betas, body_scale=chosen_scale)
    J_shaped, v_shaped = scale_hands(smplh, J_shaped, v_shaped, args.hand_scale)

    # ── Load G1 mesh triangles ONCE (shared across batch variants) ──
    # Subsample G1 mesh: 700k → ~175k triangles. Visual fidelity at 2x3
    # panel size is unaffected; render time drops ~4x.
    print("Loading G1 mesh...")
    g1_tris = load_g1_tris(transforms)[::4]
    print(f"  G1: {len(g1_tris)} triangles (subsampled)")

    targets = extract_g1_targets(transforms)
    zero_hand = torch.zeros(45, dtype=torch.float64, device=args.device)

    # ── Parse batch list (or single variant) ──
    if args.wrist_batch:
        variants = []
        for tup in args.wrist_batch.split(';'):
            parts = [float(x) for x in tup.split(',')]
            if len(parts) != 3:
                raise ValueError(f"Bad wrist-batch entry '{tup}'; need RX,RY,RZ")
            variants.append(parts)
    else:
        variants = [list(args.wrist_rot)]

    def render_one(wrist_rot, hand_L=None, hand_R=None, finger_tag=""):
        """Run retarget + render + save for a single variant.

        Args:
            wrist_rot: (rx, ry, rz) degrees for extra wrist local rotation
            hand_L, hand_R: optional (45,) numpy hand poses. None = zeros.
            finger_tag: optional filename suffix for finger-test mode
        """
        import time as _time
        t0 = _time.time()
        print(f"\n── Variant wrist_rot={wrist_rot}{' ' + finger_tag if finger_tag else ''} ──")

        # Default hand pose: thumb opposition + straight fingers.
        # If caller passes hand_L/hand_R, they override entirely.
        if hand_L is None:
            hand_L_np, _ = build_default_hand_pose()
        else:
            hand_L_np = np.asarray(hand_L, dtype=np.float64)
        if hand_R is None:
            _, hand_R_np = build_default_hand_pose()
        else:
            hand_R_np = np.asarray(hand_R, dtype=np.float64)
        hand_L_t = torch.tensor(hand_L_np, dtype=torch.float64, device=args.device)
        hand_R_t = torch.tensor(hand_R_np, dtype=torch.float64, device=args.device)

        # ── Step 1: Retarget ──
        root_trans_np, root_orient_np, body_pose_np = retarget_frame(
            transforms, rest_transforms, smplh, J_shaped,
            wrist_rot_deg=tuple(wrist_rot))

        # Base-point offset (mesh visually moves opposite to base_offset)
        if args.base_offset != 0.0:
            mesh_shift_g1 = np.array([-args.base_offset, 0.0, 0.0])
            mesh_shift_smplh = R_SMPLH_TO_G1_NP.T @ mesh_shift_g1
            root_trans_np = root_trans_np + mesh_shift_smplh

        root_t = torch.tensor(root_trans_np, dtype=torch.float64, device=args.device)
        root_o = torch.tensor(root_orient_np, dtype=torch.float64, device=args.device)
        body_p = torch.tensor(body_pose_np, dtype=torch.float64, device=args.device)

        # ── Step 2: Foot-plant Z ──
        with torch.no_grad():
            positions, rotations = smplh.forward_kinematics(
                root_t, root_o, body_p, J_shaped,
                left_hand_pose=hand_L_t, right_hand_pose=hand_R_t)
            ee_tmp = smplh.end_effector_positions(positions, rotations)
        g1_toe_mid = 0.5 * (targets['L_toe_pos'] + targets['R_toe_pos'])
        smplh_toe_mid = 0.5 * (ee_tmp['L_toe_pos'].cpu().numpy()
                               + ee_tmp['R_toe_pos'].cpu().numpy())
        shift_g1 = np.array([0.0, 0.0, g1_toe_mid[2] - smplh_toe_mid[2]])
        shift_smplh = R_SMPLH_TO_G1_NP.T @ shift_g1
        root_trans_np = root_trans_np + shift_smplh
        root_t = torch.tensor(root_trans_np, dtype=torch.float64, device=args.device)

        # ── Pre-refine errors ──
        with torch.no_grad():
            positions, rotations = smplh.forward_kinematics(
                root_t, root_o, body_p, J_shaped,
                left_hand_pose=hand_L_t, right_hand_pose=hand_R_t)
            ee_pre = smplh.end_effector_positions(positions, rotations)
        err_pre = {name: np.linalg.norm(
            ee_pre[name + "_pos"].cpu().numpy() - targets[name + "_pos"])
                   for name in ["pelvis", "L_toe", "R_toe", "L_thumb", "R_thumb"]}

        # ── Step 3: IK refinement (optional) ──
        if args.refine:
            body_pose_before = body_pose_np.copy()
            body_pose_np = refine_arms(
                smplh, J_shaped, targets,
                root_trans_np, root_orient_np, body_pose_np,
                device=args.device, w_drift=args.w_drift,
                hand_L=hand_L_np, hand_R=hand_R_np)
            body_p = torch.tensor(body_pose_np, dtype=torch.float64, device=args.device)

            # Per-joint drift breakdown
            drift_per_axis = body_pose_np - body_pose_before
            drift_per_joint = np.linalg.norm(
                drift_per_axis.reshape(21, 3), axis=1)
            smplh_joint_names = {
                13: "L_collar", 14: "R_collar",
                16: "L_shoulder", 17: "R_shoulder",
                18: "L_elbow", 19: "R_elbow",
                20: "L_wrist", 21: "R_wrist",
            }
            nonzero = [(i + 1, d) for i, d in enumerate(drift_per_joint)
                       if d > 1e-5]
            nonzero.sort(key=lambda x: -x[1])
            print(f"  IK drift ({len(nonzero)} joints):")
            for j, d in nonzero:
                name = smplh_joint_names.get(j, f"j{j}")
                print(f"    {name:11s} {np.degrees(d):6.2f}°")

        # ── Final FK + LBS ──
        with torch.no_grad():
            positions, rotations = smplh.forward_kinematics(
                root_t, root_o, body_p, J_shaped,
                left_hand_pose=hand_L_t, right_hand_pose=hand_R_t)
            v_g1 = smplh.lbs_to_g1(root_t, root_o, body_p, J_shaped, v_shaped,
                                    left_hand_pose=hand_L_t,
                                    right_hand_pose=hand_R_t)
            ee = smplh.end_effector_positions(positions, rotations)

        # SMPLH triangles (full res — mesh is small)
        human_tris = v_g1[smplh.faces]

        # Error summary
        errs = {}
        for name in ["pelvis", "L_toe", "R_toe", "L_thumb", "R_thumb"]:
            key = name + "_pos"
            errs[name] = np.linalg.norm(ee[key].cpu().numpy() - targets[key])
        err_str = "   ".join(f"{n}={e*1000:.0f}mm" for n, e in errs.items())
        print(f"  errors: {err_str}")

        # Keypoints
        kp_positions = [targets[k] for k in
                        ["pelvis_pos", "L_toe_pos", "R_toe_pos",
                         "L_thumb_pos", "R_thumb_pos"]]
        kp_labels = ["pelvis", "L_toe", "R_toe", "L_thumb", "R_thumb"]
        smplh_kp = [ee[k].cpu().numpy() for k in
                    ["pelvis_pos", "L_toe_pos", "R_toe_pos",
                     "L_thumb_pos", "R_thumb_pos"]]

        # ── 3×3 render: rows = views, cols = (overlay, G1, SMPLH) ──
        # Front  = camera at G1 +X (in front),  screen = (-Y, Z),  reveals L/R
        # Side   = camera at G1 +Y (left side), screen = ( X, Z),  reveals fwd/back
        # Top    = camera at G1 +Z (above),     screen = ( X, Y),  reveals plan view
        views = [
            ("Front", 90, 0),
            ("Side",   0, 0),
            ("Top",    0, 90),
        ]
        contents = ["Overlay", "G1 only", "SMPLH only"]
        fig, axes = plt.subplots(3, 3, figsize=(18, 18))
        G1_COLOR = '#7799BB'
        SMPLH_COLOR = '#D4A574'

        for row, (view_name, azim, elev) in enumerate(views):
            reverse_depth = (elev == 90)
            # Labels are cluttery in the Side view; show only in Top view
            show_labels = (view_name == "Top")
            for col, content in enumerate(contents):
                ax = axes[row, col]
                if content == "Overlay":
                    render_tris(ax, g1_tris, azim, elev, color=G1_COLOR,
                                alpha=0.7, reverse_depth=reverse_depth)
                    render_tris(ax, human_tris, azim, elev, color=SMPLH_COLOR,
                                alpha=0.6, reverse_depth=reverse_depth)
                    render_keypoints(ax, kp_positions, kp_labels,
                                     azim, elev, color='red', ms=8,
                                     show_labels=show_labels)
                    render_keypoints(ax, smplh_kp,
                                     [f"H_{l}" for l in kp_labels],
                                     azim, elev, color='blue', ms=6,
                                     show_labels=show_labels)
                elif content == "G1 only":
                    render_tris(ax, g1_tris, azim, elev, color=G1_COLOR,
                                alpha=1.0, reverse_depth=reverse_depth)
                    render_keypoints(ax, kp_positions, kp_labels,
                                     azim, elev, color='red', ms=8,
                                     show_labels=show_labels)
                else:
                    render_tris(ax, human_tris, azim, elev, color=SMPLH_COLOR,
                                alpha=1.0, reverse_depth=reverse_depth)
                    render_keypoints(ax, smplh_kp,
                                     [f"H_{l}" for l in kp_labels],
                                     azim, elev, color='blue', ms=6,
                                     show_labels=show_labels)
                ax.set_aspect('equal')
                ax.autoscale()
                xl, yl = ax.get_xlim(), ax.get_ylim()
                ax.set_xlim(xl[0] - 0.05, xl[1] + 0.05)
                ax.set_ylim(yl[0] - 0.05, yl[1] + 0.05)
                ax.grid(True, alpha=0.15)
                ax.set_title(f"{view_name} — {content}",
                             fontsize=12, fontweight='bold')
                ax.tick_params(labelsize=7)

        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=G1_COLOR, label='G1 Robot'),
            Patch(facecolor=SMPLH_COLOR, label='SMPLH (retarget)'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
                       markersize=8, label='G1 Keypoints'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue',
                       markersize=8, label='SMPLH End-effectors'),
        ]
        fig.legend(handles=legend_elements, loc='lower center', ncol=4,
                   fontsize=11, frameon=True)

        wr_tag = ""
        rx_, ry_, rz_ = wrist_rot
        if abs(rx_) + abs(ry_) + abs(rz_) > 1e-9:
            def _fmt(v):
                return f"{'p' if v >= 0 else 'm'}{abs(int(round(v)))}"
            wr_tag = f"  wrist=({rx_:+.0f},{ry_:+.0f},{rz_:+.0f})"
        info = (f"Episode {args.episode} Frame {args.frame}   "
                f"retarget{' + IK' if args.refine else ''}{wr_tag}\n"
                + err_str)
        fig.suptitle(info, fontsize=12, y=0.98)
        plt.tight_layout(rect=[0, 0.04, 1, 0.95])

        beta_tag = ""
        if args.beta:
            beta_tag = "_beta" + "_".join(f"{b:.1f}" for b in args.beta)
        if abs(rx_) + abs(ry_) + abs(rz_) > 1e-9:
            def _fmt(v):
                return f"{'p' if v >= 0 else 'm'}{abs(int(round(v)))}"
            beta_tag += f"_wrist{_fmt(rx_)}_{_fmt(ry_)}_{_fmt(rz_)}"
        if finger_tag:
            beta_tag += f"_{finger_tag}"
        out_path = os.path.join(
            out_dir,
            f"retarget_ep{args.episode}_f{args.frame:04d}{beta_tag}.png")
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        elapsed = _time.time() - t0
        print(f"  saved: {out_path}  ({elapsed:.1f}s)")

    # ── Run all variants ──
    if args.finger_test > 0:
        # Diagnostic: render 3 variants with fingers curled around X/Y/Z
        print(f"\n### Finger-test mode: curl = {args.finger_test:.0f}° "
              f"per segment, axes X/Y/Z ###")
        for wrist_rot in variants:
            for axis in ['x', 'y', 'z']:
                hL, hR = build_finger_curl_pose(
                    axis=axis, angle_deg=args.finger_test)
                render_one(wrist_rot, hand_L=hL, hand_R=hR,
                           finger_tag=f"finger{axis}{int(args.finger_test)}")
    elif args.finger_rot:
        parts = args.finger_rot.split(',')
        if len(parts) != 3:
            raise ValueError(f"--finger-rot needs AXIS,L_DEG,R_DEG; got {args.finger_rot}")
        axis = parts[0].strip().lower()
        L_deg = float(parts[1])
        R_deg = float(parts[2])
        hL, _ = build_finger_curl_pose(axis=axis, angle_deg=L_deg)
        _, hR = build_finger_curl_pose(axis=axis, angle_deg=R_deg)
        tag = f"finger{axis}L{int(L_deg):+d}R{int(R_deg):+d}".replace('+', 'p').replace('-', 'm')
        for wrist_rot in variants:
            render_one(wrist_rot, hand_L=hL, hand_R=hR, finger_tag=tag)
    elif args.thumb_test > 0:
        print(f"\n### Thumb-test mode: thumb1 rotation = ±{args.thumb_test:.0f}° "
              f"(L neg, R pos), axes X/Y/Z ###")
        for wrist_rot in variants:
            for axis in ['x', 'y', 'z']:
                hL, hR = build_thumb_base_pose(
                    axis=axis,
                    L_deg=-args.thumb_test,
                    R_deg=+args.thumb_test)
                render_one(wrist_rot, hand_L=hL, hand_R=hR,
                           finger_tag=f"thumb{axis}{int(args.thumb_test)}")
    elif args.thumb_rot:
        parts = args.thumb_rot.split(',')
        if len(parts) != 3:
            raise ValueError(f"--thumb-rot needs AXIS,L_DEG,R_DEG; got {args.thumb_rot}")
        axis = parts[0].strip().lower()
        L_deg = float(parts[1])
        R_deg = float(parts[2])
        hL, hR = build_thumb_base_pose(axis=axis, L_deg=L_deg, R_deg=R_deg)
        tag = f"thumb{axis}L{int(L_deg):+d}R{int(R_deg):+d}".replace('+', 'p').replace('-', 'm')
        for wrist_rot in variants:
            render_one(wrist_rot, hand_L=hL, hand_R=hR, finger_tag=tag)
    elif args.live_hand and hand_type == 'inspire':
        hL, hR = apply_finger_curl_from_g1(hs, hand_type=hand_type)
        for wrist_rot in variants:
            render_one(wrist_rot, hand_L=hL, hand_R=hR, finger_tag="live_hand")
    else:
        for wrist_rot in variants:
            render_one(wrist_rot)


if __name__ == "__main__":
    main()
