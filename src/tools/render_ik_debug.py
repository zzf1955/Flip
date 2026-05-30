"""
Third-person debug view: G1 robot mesh + SMPLH human mesh side-by-side.
Shows alignment of contact points (toe tips, thumb tips, pelvis).

Usage:
  python scripts/render_ik_debug.py --episode 0 --frame 30
  python scripts/render_ik_debug.py --episode 0 --frame 30 --beta 3.0
"""

import sys
import os
import argparse
import numpy as np
import pinocchio as pin
from stl import mesh as stl_mesh
import torch

sys.stdout.reconfigure(line_buffering=True)

from src.core.config import G1_URDF, MESH_DIR, SKIP_MESHES, OUTPUT_DIR, get_hand_type, get_skip_meshes
from src.core.fk import build_q, do_fk, parse_urdf_meshes
from src.core.data import load_episode_info
from src.core.smplh import SMPLHForIK, IKSolver, extract_g1_targets, G1_KEYPOINTS

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection


def oblique_project(pts, azim_deg=35, elev_deg=25):
    """Rotate then take X,Z as screen coords. Returns (screen, depth)."""
    az, el = np.radians(azim_deg), np.radians(elev_deg)
    Rz = np.array([[np.cos(az), -np.sin(az), 0],
                    [np.sin(az),  np.cos(az), 0],
                    [0, 0, 1]])
    Rx = np.array([[1, 0, 0],
                    [0, np.cos(el), -np.sin(el)],
                    [0, np.sin(el),  np.cos(el)]])
    R = Rx @ Rz
    rotated = (R @ pts.T).T
    return rotated[:, [0, 2]], rotated[:, 1]


def load_g1_tris(transforms):
    """Load all G1 mesh triangles in world frame."""
    link_meshes = parse_urdf_meshes(G1_URDF)
    skip = get_skip_meshes(get_hand_type())
    all_tris = []
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
        all_tris.append(world.reshape(-1, 3, 3))
    return np.concatenate(all_tris, axis=0) if all_tris else np.zeros((0, 3, 3))


def smplh_tris(v_world, faces, subsample=4):
    """Get triangle vertices from SMPLH mesh."""
    tris = v_world[faces]  # (F, 3, 3)
    if subsample > 1:
        tris = tris[::subsample]
    return tris


def render_tris(ax, tris, azim, elev, color, alpha=0.6, edge_lw=0.02):
    """Render triangles with painter's algorithm."""
    poly_data = []
    for tri in tris:
        screen, depth = oblique_project(tri, azim, elev)
        poly_data.append((depth.mean(), screen))

    poly_data.sort(key=lambda x: x[0])
    polys = [p[1] for p in poly_data]
    pc = PolyCollection(polys, facecolors=color, edgecolors='#444444',
                        linewidths=edge_lw, alpha=alpha)
    ax.add_collection(pc)


def render_keypoints(ax, points, labels, azim, elev, color='red', ms=8):
    """Render keypoints with labels."""
    pts = np.array(points)
    screen, _ = oblique_project(pts, azim, elev)
    for i, (sx, sy) in enumerate(screen):
        ax.plot(sx, sy, 'o', color=color, markersize=ms,
                markeredgecolor='white', markeredgewidth=1.5, zorder=100)
        offset_x = 0.02 if i % 2 == 1 else -0.02
        ha = 'left' if i % 2 == 1 else 'right'
        ax.annotate(labels[i], (sx, sy),
                    xytext=(sx + offset_x, sy + 0.01),
                    fontsize=8, fontweight='bold', color=color,
                    bbox=dict(boxstyle='round,pad=0.15', fc='white',
                              ec=color, alpha=0.85),
                    arrowprops=dict(arrowstyle='->', color=color, lw=1),
                    ha=ha, zorder=101)


def main():
    parser = argparse.ArgumentParser(description="IK debug: 3rd person view")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--frame", type=int, default=30)
    parser.add_argument("--beta", type=float, nargs='*', default=None)
    parser.add_argument("--scale", type=float, default=None,
                        help="Body scale (default: auto-match G1, 1.0 = original SMPLH)")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    out_dir = os.path.join(OUTPUT_DIR, "human", "ik_debug")
    os.makedirs(out_dir, exist_ok=True)

    # ── 1. Load G1 FK ──
    print("Loading G1 model and episode data...")
    model_g = pin.buildModelFromUrdf(G1_URDF, pin.JointModelFreeFlyer())
    data_g = model_g.createData()

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

    # ── 2. Load G1 mesh triangles ──
    print("Loading G1 mesh...")
    g1_tris = load_g1_tris(transforms)
    print(f"  G1: {len(g1_tris)} triangles")

    # ── 3. G1 keypoint world positions ──
    targets = extract_g1_targets(transforms)
    kp_positions = [targets[k] for k in
                    ["pelvis_pos", "L_toe_pos", "R_toe_pos",
                     "L_thumb_pos", "R_thumb_pos"]]
    kp_labels = ["pelvis", "L_toe", "R_toe", "L_thumb", "R_thumb"]

    # ── 4. Solve IK ──
    print("Solving IK...")
    smplh = SMPLHForIK(device=args.device)
    solver = IKSolver(smplh)

    betas = None
    if args.beta:
        betas = np.zeros(16)
        for i, b in enumerate(args.beta[:16]):
            betas[i] = b

    result = solver.solve_frame(targets, betas=betas, body_scale=args.scale)
    print(f"  Loss: {result.loss:.4f}")
    for name, err in result.pos_errors.items():
        print(f"  {name}: {err*1000:.1f}mm")

    # ── 5. Get SMPLH mesh ──
    print("Computing SMPLH mesh...")
    J_shaped, v_shaped = smplh.shape_blend(betas, body_scale=args.scale)
    root_t = torch.tensor(result.root_trans, dtype=torch.float64, device=args.device)
    root_o = torch.tensor(result.root_orient, dtype=torch.float64, device=args.device)
    body_p = torch.tensor(result.body_pose, dtype=torch.float64, device=args.device)
    v_g1 = smplh.lbs_to_g1(root_t, root_o, body_p, J_shaped, v_shaped)

    # SMPLH end-effector positions (for verification)
    with torch.no_grad():
        pos, rot = smplh.forward_kinematics(root_t, root_o, body_p, J_shaped)
        ee = smplh.end_effector_positions(pos, rot)
    smplh_kp = [ee[k].cpu().numpy() for k in
                ["pelvis_pos", "L_toe_pos", "R_toe_pos",
                 "L_thumb_pos", "R_thumb_pos"]]

    human_tris = smplh_tris(v_g1, smplh.faces, subsample=2)
    print(f"  SMPLH: {len(human_tris)} triangles (subsampled)")

    # ── 6. Render 3 views ──
    views = [
        ("Front (Y-)", 0, 0),
        ("Oblique", 35, 20),
        ("Side (X-)", 90, 10),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(24, 12))

    for ax_idx, (title, azim, elev) in enumerate(views):
        ax = axes[ax_idx]

        # G1 mesh (grey-blue, semi-transparent)
        render_tris(ax, g1_tris, azim, elev,
                    color='#7799BB', alpha=0.4, edge_lw=0.01)

        # SMPLH mesh (skin color, semi-transparent)
        render_tris(ax, human_tris, azim, elev,
                    color='#D4A574', alpha=0.5, edge_lw=0.01)

        # G1 keypoints (red)
        render_keypoints(ax, kp_positions, kp_labels, azim, elev,
                         color='red', ms=8)

        # SMPLH end-effector points (blue)
        render_keypoints(ax, smplh_kp,
                         [f"H_{l}" for l in kp_labels],
                         azim, elev, color='blue', ms=6)

        ax.set_aspect('equal')
        ax.autoscale()
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        ax.set_xlim(xlim[0] - 0.05, xlim[1] + 0.05)
        ax.set_ylim(ylim[0] - 0.05, ylim[1] + 0.05)
        ax.grid(True, alpha=0.15)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.tick_params(labelsize=7)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#7799BB', alpha=0.5, label='G1 Robot'),
        Patch(facecolor='#D4A574', alpha=0.6, label='SMPLH Human'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
                    markersize=8, label='G1 Keypoints'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue',
                    markersize=8, label='SMPLH End-effectors'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4,
               fontsize=11, frameon=True)

    # Info text
    info_lines = [
        f"Episode {args.episode}, Frame {args.frame}, "
        f"Beta={args.beta if args.beta else 'default'}",
        "Position errors: " + ", ".join(
            f"{k.replace('_pos','')}={v*1000:.1f}mm"
            for k, v in result.pos_errors.items()),
    ]
    fig.suptitle("\n".join(info_lines), fontsize=12, y=0.98)

    plt.tight_layout(rect=[0, 0.04, 1, 0.95])

    beta_tag = ""
    if args.beta:
        beta_tag = "_beta" + "_".join(f"{b:.1f}" for b in args.beta)
    out_path = os.path.join(out_dir,
                             f"debug_ep{args.episode}_f{args.frame:04d}{beta_tag}.png")
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
