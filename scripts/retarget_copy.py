"""
Motion retargeting: pure joint-rotation copy from G1 → SMPLH.

No IK, no optimization. Algorithm:
  1. G1 FK at neutral pose → rest-pose link rotations.
  2. G1 FK at target frame → current link rotations.
  3. World-frame rotation delta: ΔR[link] = R_current @ R_rest^T.
  4. Convert delta from G1 frame to SMPLH frame via R_S2G / R_G2S.
  5. Map SMPLH body joints (0-21) to corresponding G1 links; unmapped joints
     inherit parent's world rotation (→ local rotation = I).
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
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import G1_URDF, MESH_DIR, OUTPUT_DIR, get_hand_type, get_skip_meshes
from video_inpaint import build_q, do_fk, parse_urdf_meshes, load_episode_info
from smplh_ik import (SMPLHForIK, extract_g1_targets, G1_KEYPOINTS,
                      R_SMPLH_TO_G1_NP)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection


# ── SMPLH body joint → G1 (parent_link, child_link) pair defining the bone direction ──
#
# The bone from the SMPLH joint to its child is aligned with the G1 vector
# (g1_child_pos - g1_parent_pos). This decouples retarget from rest pose.
#
# Format: smplh_joint_idx: (smplh_child_joint, g1_parent_link, g1_child_link)
BONE_MAP = {
    # Root: use pelvis orientation directly (not bone direction)
    # Spine: handled separately via SPINE_SPLIT below (3-way share of
    # G1 torso rotation, not bone-direction alignment, because bone-align
    # only catches the swing of the spine direction and discards twist
    # + distributes the bend to only one segment)
    # Left leg chain: pelvis → hip_pitch → knee → ankle
    1:  (4,  "left_hip_pitch_link",  "left_knee_link"),     # left hip: thigh direction
    4:  (7,  "left_knee_link",       "left_ankle_roll_link"),  # left knee: shin direction
    # Right leg
    2:  (5,  "right_hip_pitch_link", "right_knee_link"),
    5:  (8,  "right_knee_link",      "right_ankle_roll_link"),
    # Left arm chain
    16: (18, "left_shoulder_pitch_link", "left_elbow_link"),  # shoulder: upper arm
    18: (20, "left_elbow_link",          "left_wrist_yaw_link"),  # elbow: forearm
    # Right arm
    17: (19, "right_shoulder_pitch_link", "right_elbow_link"),
    19: (21, "right_elbow_link",          "right_wrist_yaw_link"),
}

# ── Spine 3-way split ──────────────────────────────────────────────────
#
# G1 torso has 3 hinges (waist_yaw → waist_roll → waist_pitch) producing
# a full 3-DOF rotation between pelvis and torso_link. SMPLH spine has
# 3 joints (spine1/2/3) that should COLLECTIVELY reproduce this rotation
# while distributing the bend across the back instead of jamming it all
# into a single joint.
#
# Algorithm:
#   1. R_torso_rel = R_pelvis^T @ R_torso   (G1, in pelvis-local frame)
#      Subtract rest:  Δ = R_torso_rel_curr @ R_torso_rel_rest^T
#   2. Convert to SMPLH frame: R_G2S @ Δ @ R_S2G
#   3. Cube root via axis-angle: split angle by 3 (axis unchanged)
#   4. Apply identical R_seg as local rotation on each spine joint.
#      Because R_seg has the same axis at every step, composition gives
#      total = R_seg^3 = full Δ, exact for arbitrary angle.
SPINE_JOINTS = (3, 6, 9)  # spine1, spine2, spine3

# Joints whose rotation is copied directly from a G1 link's world orientation
# (using rest-pose offset). Used for end-of-chain joints where bone direction
# doesn't fully determine rotation (ankle sole, wrist hand-plane).
DIRECT_ROT_MAP = {
    7:  "left_ankle_roll_link",     # left ankle: sole orientation
    8:  "right_ankle_roll_link",    # right ankle
    20: "left_wrist_yaw_link",      # left wrist: hand plane
    21: "right_wrist_yaw_link",     # right wrist
}

# ── Wrist local-frame convention correction ────────────────────────────
#
# After DIRECT_ROT_MAP, the SMPLH wrist's WORLD rotation (in G1 frame)
# matches G1's wrist_yaw_link world rotation **exactly** (verified by
# diagnostic script — matrices agree to 3 decimals). But when the hand
# mesh is rendered via LBS, the palm faces the wrong world direction.
#
# Root cause: the SMPLH hand mesh is defined in SMPLH's wrist-local
# frame (T-pose convention: palm-down, fingers along arm), while G1's
# hand meshes live in wrist_yaw_link's local frame (different axis
# convention). Same world rotation applied to different local "palm
# normal" vectors → different rendered palm orientations.
#
# The fix is a CONSTANT local-frame rotation (pose-independent): rotate
# the SMPLH wrist by (-90° around local Y, then -90° around local Z)
# after the DIRECT_ROT_MAP assignment. Calibrated visually on ep0 f30
# by a grid search over Euler (x, y, z) — see commit history / plan
# file for the search trace. The value is a link-frame convention
# constant, not a pose-tunable parameter, so it should not be exposed
# as a CLI flag.
#
# Important trade-off: applying this correction moves the SMPLH thumb
# TIP away from G1's thumb target (thumb is offset from the wrist axis,
# so rotating the palm drags the thumb tip with it). The position
# residual (~180mm at ep0 f30) is expected to be absorbed by the Step 3
# IK refinement, which adjusts shoulder/elbow to hit the thumb target
# while preserving the now-correct palm orientation.
#
# Order: (ry, rz) applied as Rz @ Ry in local frame, mirrored L/R.
WRIST_LOCAL_CORRECTION_DEG = (0.0, -90.0, -90.0)

# ── Shoulder twist extraction map ────────────────────────────────────────
#
# bone-alignment (rot_between) gives 2-DOF: it points the SMPLH upper arm
# in the same direction as G1's upper arm, but drops the twist DOF
# (rotation around the upper arm's own axis). This means the "elbow bend
# plane" can end up rotated wrong, which propagates to forearm / wrist /
# palm orientation.
#
# Fix: extract the twist component of G1's shoulder world rotation (the
# world delta of shoulder_yaw_link, which is the end of the G1 shoulder
# pitch→roll→yaw hinge chain) around the current G1 upper-arm axis, and
# apply the same twist in SMPLH parent-local frame after the swing.
#
# Format: smplh_joint_idx -> (g1_yaw_link, g1_shoulder_base, g1_elbow)
SHOULDER_TWIST_MAP = {
    16: ("left_shoulder_yaw_link",
         "left_shoulder_pitch_link", "left_elbow_link"),
    17: ("right_shoulder_yaw_link",
         "right_shoulder_pitch_link", "right_elbow_link"),
}


def rot_to_axis_angle(R):
    """Rotation matrix (3,3) → axis-angle (3,)."""
    rvec, _ = cv2.Rodrigues(np.ascontiguousarray(R, dtype=np.float64))
    return rvec.flatten()


def extract_twist_angle(R, axis):
    """Extract the signed twist angle (radians) of rotation R around unit `axis`.

    Swing-twist decomposition using quaternion projection: the "twist" part
    of R is the component that rotates around `axis`; the rest is "swing".
    Formula: take R's axis-angle → quaternion (w, xyz), project xyz onto axis,
    twist_angle = 2 * atan2(proj, w). Axis is taken in the same frame as R.
    """
    rvec, _ = cv2.Rodrigues(np.ascontiguousarray(R, dtype=np.float64))
    rvec = rvec.flatten()
    theta = float(np.linalg.norm(rvec))
    if theta < 1e-9:
        return 0.0
    qw = np.cos(theta / 2.0)
    qxyz = np.sin(theta / 2.0) * (rvec / theta)
    proj = float(np.dot(qxyz, axis))
    return 2.0 * np.arctan2(proj, qw)


def rot_between(a, b):
    """Shortest-path rotation matrix that rotates unit vector a onto unit b."""
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    s = float(np.linalg.norm(v))
    if s < 1e-8:
        if c > 0:
            return np.eye(3)
        # 180°: pick any axis perpendicular to a
        ax = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis = np.cross(a, ax)
        axis /= np.linalg.norm(axis)
        K = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
        return np.eye(3) + 2.0 * K @ K
    K = np.array([[0, -v[2], v[1]],
                  [v[2], 0, -v[0]],
                  [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K * ((1.0 - c) / (s * s))


def compute_g1_rest_transforms():
    """G1 FK at q=neutral (zero joint angles). Returns {link: (t, R)}."""
    model = pin.buildModelFromUrdf(G1_URDF, pin.JointModelFreeFlyer())
    data = model.createData()
    q = pin.neutral(model)
    q[0:3] = [0.0, 0.0, 0.0]
    q[3:7] = [0.0, 0.0, 0.0, 1.0]
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    out = {}
    for i in range(model.nframes):
        name = model.frames[i].name
        T = data.oMf[i]
        out[name] = (T.translation.copy(), T.rotation.copy())
    return out


def retarget_frame(g1_transforms, g1_rest_transforms, smplh, J_shaped,
                   wrist_rot_deg=(0.0, 0.0, 0.0)):
    """Retarget one G1 FK frame to SMPLH pose via bone direction alignment.

    Algorithm (topologically ordered):
      - Root orient: copy G1 pelvis world rotation (with rest offset).
      - Chain joints (shoulder/elbow/hip/knee/spine): align SMPLH bone
        (to child joint) with G1 corresponding bone direction. 2-DOF
        vector-to-vector rotation; twist stays 0 (parent's frame).
      - End joints (wrist/ankle): copy G1 link world orientation
        (with rest offset). Provides full 3-DOF incl. twist.

    Returns: root_trans(3), root_orient(3), body_pose(63)
    """
    R_S2G = R_SMPLH_TO_G1_NP
    R_G2S = R_S2G.T

    n = smplh.n_body_joints  # 22
    parents = smplh.parents

    J_np = J_shaped.detach().cpu().numpy() if isinstance(J_shaped, torch.Tensor) else J_shaped

    R_world = [np.eye(3) for _ in range(n)]
    R_local = [np.eye(3) for _ in range(n)]

    # ── Root (joint 0): copy G1 pelvis world orientation ──
    R_g1_pelvis = g1_transforms["pelvis"][1]
    R_g1_pelvis_rest = g1_rest_transforms["pelvis"][1]
    delta_pelvis = R_g1_pelvis @ R_g1_pelvis_rest.T
    R_world[0] = R_G2S @ delta_pelvis @ R_S2G
    R_local[0] = R_world[0]

    # ── Spine: G1 torso rotation (relative to pelvis) split 3 ways ──
    # G1 torso_rel = R_pelvis^T @ R_torso (in pelvis-local G1 frame).
    # Subtract the rest baseline so a still G1 → identity SMPLH spine.
    R_g1_tor_curr = g1_transforms["torso_link"][1]
    R_g1_tor_rest = g1_rest_transforms["torso_link"][1]
    R_torso_rel_curr = R_g1_pelvis.T @ R_g1_tor_curr
    R_torso_rel_rest = R_g1_pelvis_rest.T @ R_g1_tor_rest
    R_torso_delta_g1 = R_torso_rel_curr @ R_torso_rel_rest.T
    # Convert to SMPLH pelvis-local frame
    R_torso_delta_smplh = R_G2S @ R_torso_delta_g1 @ R_S2G
    # Cube root via axis-angle (axis preserved, angle / 3)
    rvec_torso, _ = cv2.Rodrigues(np.ascontiguousarray(R_torso_delta_smplh))
    R_spine_seg = cv2.Rodrigues(rvec_torso.flatten() / 3.0)[0]

    # Traversal order: topological (parents before children).
    # SMPLH parents are monotonic so range(1, n) works.
    for i in range(1, n):
        p = int(parents[i])

        if i in SPINE_JOINTS:
            # Each spine segment gets the same R_seg in its parent-local
            # frame; total composition = R_torso_delta_smplh exactly,
            # because the same rotation around the same axis composes
            # additively (R(a,θ)^3 = R(a,3θ)).
            R_local[i] = R_spine_seg
            R_world[i] = R_world[p] @ R_local[i]

        elif i in BONE_MAP:
            child_smplh, g1_parent, g1_child = BONE_MAP[i]

            # SMPLH bone vector in rest (parent local frame = world at rest, since R_world_rest = I)
            bone_rest = J_np[child_smplh] - J_np[i]
            bone_rest_dir = bone_rest / (np.linalg.norm(bone_rest) + 1e-12)

            # Target direction in SMPLH world frame (from G1)
            g1_vec = g1_transforms[g1_child][0] - g1_transforms[g1_parent][0]
            target_world_smplh = R_G2S @ g1_vec
            target_world_smplh /= (np.linalg.norm(target_world_smplh) + 1e-12)

            # In parent's local frame, the rest bone is just bone_rest_dir
            # (SMPLH rest has all parent rotations identity locally). We want
            # R_world[i] @ bone_rest_dir = target_world_smplh.
            # R_world[i] = R_world[p] @ R_local[i], so
            # R_local[i] @ bone_rest_dir = R_world[p].T @ target_world_smplh
            target_in_parent = R_world[p].T @ target_world_smplh
            R_local[i] = rot_between(bone_rest_dir, target_in_parent)

            # Inject upper-arm twist at the shoulder. bone-alignment gives
            # only swing (2 DOF); the twist DOF (rotation around upper-arm
            # axis) is extracted from G1's shoulder_yaw world-delta and
            # applied post-swing in SMPLH parent-local frame.
            if i in SHOULDER_TWIST_MAP:
                g1_yaw, g1_shoulder, g1_elbow = SHOULDER_TWIST_MAP[i]
                R_g1_curr = g1_transforms[g1_yaw][1]
                R_g1_rest = g1_rest_transforms[g1_yaw][1]
                R_g1_delta = R_g1_curr @ R_g1_rest.T
                arm_vec_g1 = (g1_transforms[g1_elbow][0]
                              - g1_transforms[g1_shoulder][0])
                arm_axis_g1 = arm_vec_g1 / (np.linalg.norm(arm_vec_g1) + 1e-12)
                twist_rad = extract_twist_angle(R_g1_delta, arm_axis_g1)
                # target_in_parent is already unit; Rodrigues(axis * angle)
                R_twist = cv2.Rodrigues(
                    (target_in_parent * twist_rad).astype(np.float64))[0]
                R_local[i] = R_twist @ R_local[i]

            R_world[i] = R_world[p] @ R_local[i]

        elif i in DIRECT_ROT_MAP:
            link = DIRECT_ROT_MAP[i]
            R_g1_curr = g1_transforms[link][1]
            R_g1_rest = g1_rest_transforms[link][1]
            delta_g1 = R_g1_curr @ R_g1_rest.T
            R_world[i] = R_G2S @ delta_g1 @ R_S2G
            R_local[i] = R_world[p].T @ R_world[i]

        else:
            # Inherit parent (local = I)
            R_local[i] = np.eye(3)
            R_world[i] = R_world[p]

    # ── Wrist local-frame rotation ─────────────────────────────────────
    # Two parts:
    #  (a) WRIST_LOCAL_CORRECTION_DEG  — constant link-frame convention
    #      fix (see comment at top of file). Always applied.
    #  (b) wrist_rot_deg (from CLI)    — optional additional rotation,
    #      used only for diagnostic sweeps. Defaults to (0, 0, 0).
    # Both are mirrored L/R and applied as Rz @ Ry @ Rx.
    def _euler_xyz(a, b, c):
        R = np.eye(3)
        for axis, ang in [(np.array([1.0, 0.0, 0.0]), np.radians(a)),
                          (np.array([0.0, 1.0, 0.0]), np.radians(b)),
                          (np.array([0.0, 0.0, 1.0]), np.radians(c))]:
            if abs(ang) > 1e-12:
                R = cv2.Rodrigues((axis * ang).astype(np.float64))[0] @ R
        return R

    # (a) Constant correction — always applied
    cx, cy, cz = WRIST_LOCAL_CORRECTION_DEG
    R_corr_L = _euler_xyz(cx, cy, cz)
    R_corr_R = _euler_xyz(-cx, -cy, -cz)
    R_local[20] = R_local[20] @ R_corr_L
    R_local[21] = R_local[21] @ R_corr_R

    # (b) Optional diagnostic rotation
    rx, ry, rz = wrist_rot_deg
    if abs(rx) + abs(ry) + abs(rz) > 1e-9:
        R_extra_L = _euler_xyz(rx, ry, rz)
        R_extra_R = _euler_xyz(-rx, -ry, -rz)
        R_local[20] = R_local[20] @ R_extra_L
        R_local[21] = R_local[21] @ R_extra_R

    R_world[20] = R_world[int(parents[20])] @ R_local[20]
    R_world[21] = R_world[int(parents[21])] @ R_local[21]

    # World → body_pose
    root_orient = rot_to_axis_angle(R_local[0])
    body_pose = np.zeros(63)
    for i in range(1, n):
        body_pose[(i - 1) * 3:(i - 1) * 3 + 3] = rot_to_axis_angle(R_local[i])

    # Root translation: G1 pelvis position in SMPLH frame.
    p_pelvis_g1 = g1_transforms["pelvis"][0]
    root_trans = R_G2S @ p_pelvis_g1

    return root_trans, root_orient, body_pose


# ── Hand pose construction ─────────────────────────────────────────────
#
# SMPLH hand: 15 joints per side × 3 DOF = 45-dim axis-angle per hand.
# Joint layout within one hand:
#   0-2   = finger1 of index/middle/pinky/ring/thumb
#   3-5   = finger2
#   6-8   = finger3
# Offsets within hand_pose (45-dim) per finger:
#   index:  [0:9]   (3 joints × 3 axes)
#   middle: [9:18]
#   pinky:  [18:27]
#   ring:   [27:36]
#   thumb:  [36:45]
FINGER_SLOTS = {
    "index":  (0, 9),
    "middle": (9, 18),
    "pinky":  (18, 27),
    "ring":   (27, 36),
    "thumb":  (36, 45),
}

# ── Finger curl convention ─────────────────────────────────────────────
#
# **Validated on ep0 f30 via --finger-rot z,-40,40 diagnostic:**
# For 4 non-thumb fingers (index / middle / pinky / ring), curling into
# the palm (matching Inspire hand's closed pose) is:
#   - axis: SMPLH wrist-local **Z** axis
#   - sign: **L = negative, R = positive** (mirrored)
#   - per-segment angle: scales linearly with G1 hand_state value
#
# Mapping at runtime: given G1 value v ∈ [0, 1] where 0=closed, 1=open,
# the SMPLH per-segment curl angle is:
#     angle_deg = (1 - v) * FINGER_CURL_MAX_DEG
# applied to all 3 segments of the finger, around Z axis, with L sign
# flipped (see apply_finger_curl_from_g1() below).
FINGER_CURL_AXIS = 'z'
FINGER_CURL_MAX_DEG = 40.0   # per segment, at fully-closed (v=0)
FINGER_CURL_SIGN_L = -1.0    # left hand sign
FINGER_CURL_SIGN_R = +1.0    # right hand sign

# ── Thumb base opposition (default pose) ──────────────────────────────
#
# SMPLH T-pose thumb is splayed outward — G1 Inspire's default thumb is
# already opposed across the palm. To match, we apply a constant local
# rotation on thumb1 (the base joint). Values found via:
#   1. 6D free optimization minimizing |L_thumb_tip - R_thumb_tip| on
#      ep0 f30 → found axes below at 102°/91° (= maximally inward)
#   2. User visual pick: 50° along the same axes looks closest to G1's
#      default thumb opposition.
#
# Per-hand axes (not exactly mirrored because ep0 f30 is slightly
# asymmetric — but the two values are treated as constants regardless
# of the calibration frame).
THUMB_DEFAULT_L_AXIS = np.array([0.495, 0.681, -0.539])
THUMB_DEFAULT_R_AXIS = np.array([0.521, -0.437,  0.733])
THUMB_DEFAULT_ANGLE_DEG = 50.0


def scale_hands(smplh, J_shaped, v_shaped, hand_scale):
    """Scale both hands (joints + vertices) relative to their respective wrists.

    Body scale (shape_blend) already shrinks the SMPLH mesh ~0.85× to
    match G1 overall body size. But G1's Inspire hand is physically
    larger than 0.85× SMPLH hand, so we enlarge the hand region only.

    Vertices are scaled proportionally to their total LBS weight on
    hand joints (joints 22–36 for L, 37–51 for R), so the transition
    at the wrist is smooth. L and R are scaled around their respective
    wrist joints (20, 21) so the arm chain isn't disturbed.

    Args:
        smplh: SMPLHForIK instance (for weights access)
        J_shaped: (52, 3) torch tensor of joint positions
        v_shaped: (V, 3) torch tensor of vertex positions
        hand_scale: float, 1.0 = no change, 1.3 = 30% larger

    Returns: (J_new, v_new) as torch tensors (same dtype / device).
    """
    if abs(hand_scale - 1.0) < 1e-9:
        return J_shaped, v_shaped

    dev, dtype = J_shaped.device, J_shaped.dtype
    L_hand_joints = list(range(22, 37))
    R_hand_joints = list(range(37, 52))

    L_wrist = J_shaped[20]
    R_wrist = J_shaped[21]

    J_new = J_shaped.clone()
    for j in L_hand_joints:
        J_new[j] = L_wrist + (J_shaped[j] - L_wrist) * hand_scale
    for j in R_hand_joints:
        J_new[j] = R_wrist + (J_shaped[j] - R_wrist) * hand_scale

    # Vertex scaling weighted by hand LBS weight
    W = smplh.weights  # (V, 52)
    L_w = W[:, L_hand_joints].sum(dim=1)  # (V,)
    R_w = W[:, R_hand_joints].sum(dim=1)
    scale_L = 1.0 + L_w * (hand_scale - 1.0)  # (V,), per-vertex
    scale_R = 1.0 + R_w * (hand_scale - 1.0)

    # Apply L scaling first (around L_wrist), then R (around R_wrist).
    # For vertices bound to only one hand the other factor is 1.0 so
    # the second pass is a no-op there.
    v_new = L_wrist[None, :] + (v_shaped - L_wrist[None, :]) * scale_L[:, None]
    v_new = R_wrist[None, :] + (v_new    - R_wrist[None, :]) * scale_R[:, None]
    return J_new, v_new


def build_default_hand_pose():
    """Build (L, R) default hand_pose (45,): thumb at opposition, fingers straight.

    Uses THUMB_DEFAULT_{L,R}_AXIS × THUMB_DEFAULT_ANGLE_DEG on thumb1.
    This is the frame-independent baseline; runtime G1 hand_state data
    should be applied ON TOP via finger curl + additional thumb rotation.
    """
    L_pose = np.zeros(45, dtype=np.float64)
    R_pose = np.zeros(45, dtype=np.float64)
    L_pose[36:39] = THUMB_DEFAULT_L_AXIS * np.radians(THUMB_DEFAULT_ANGLE_DEG)
    R_pose[36:39] = THUMB_DEFAULT_R_AXIS * np.radians(THUMB_DEFAULT_ANGLE_DEG)
    return L_pose, R_pose


def build_thumb_base_pose(axis='z', L_deg=40.0, R_deg=40.0):
    """Rotate ONLY the thumb base joint (thumb1, offset 36:39 in hand_pose).

    Used to test the "thumb opposition" axis — the constant rotation that
    brings SMPLH's splayed T-pose thumb into G1 Inspire hand's default
    "thumb bent across palm" position.

    Returns (L_pose, R_pose) each numpy (45,).
    """
    axis_vec = {
        'x': np.array([1.0, 0.0, 0.0]),
        'y': np.array([0.0, 1.0, 0.0]),
        'z': np.array([0.0, 0.0, 1.0]),
    }[axis]
    L_pose = np.zeros(45, dtype=np.float64)
    R_pose = np.zeros(45, dtype=np.float64)
    L_pose[36:39] = axis_vec * np.radians(L_deg)
    R_pose[36:39] = axis_vec * np.radians(R_deg)
    return L_pose, R_pose


def build_finger_curl_pose(axis='z', angle_deg=40.0, fingers=None):
    """Build (L, R) hand_pose (45,) with the given fingers curled around `axis`.

    Each of the 3 segments per finger gets the same axis-angle rotation,
    i.e. `hand_pose[3i:3i+3] = axis_vec * angle_rad` for every joint in
    the selected fingers.

    Args:
        axis:     'x' | 'y' | 'z' (SMPLH wrist-local frame axes)
        angle_deg: curl angle per segment, degrees
        fingers:  list of finger names to curl; default = all 4
                  non-thumb fingers
    Returns:
        (left_pose, right_pose) each numpy (45,). Currently L and R are
        set identically; caller can mirror if needed.
    """
    if fingers is None:
        fingers = ["index", "middle", "pinky", "ring"]
    axis_vec = {
        'x': np.array([1.0, 0.0, 0.0]),
        'y': np.array([0.0, 1.0, 0.0]),
        'z': np.array([0.0, 0.0, 1.0]),
    }[axis]
    rot = axis_vec * np.radians(angle_deg)  # (3,) axis-angle

    pose = np.zeros(45, dtype=np.float64)
    for f in fingers:
        start, end = FINGER_SLOTS[f]
        # 3 joints × 3 entries; write rot into each joint's slot
        for joint_offset in range(0, end - start, 3):
            pose[start + joint_offset:start + joint_offset + 3] = rot
    return pose.copy(), pose.copy()


# ── IK refinement ──

def refine_arms(smplh, J_shaped, targets,
                root_trans, root_orient, body_pose_init,
                device='cpu',
                w_pos=1000.0, w_drift=10.0,
                free_joints=(13, 14, 16, 17, 18, 19),
                n_outer=2, max_iter=20,
                hand_L=None, hand_R=None):
    """IK refinement — optimize ONLY collar→elbow joints to match thumb targets.

    Called after deterministic root placement and the constant wrist
    correction. Root transform, wrist orientation, and non-arm body_pose
    are all held fixed.

    Free joints (default): L/R_collar (13/14), L/R_shoulder (16/17),
    L/R_elbow (18/19) = 6 joints × 3 = 18 DOF. **Wrist (20/21) is NOT
    optimized** — its orientation is fixed by the retarget step's
    constant WRIST_LOCAL_CORRECTION_DEG, and IK should only adjust the
    upstream chain to bring the thumb tip to the target without
    disturbing the (already correct) palm orientation.

    Loss targets: L_thumb, R_thumb position only (6 DOF target).

    Args:
        smplh: SMPLHForIK instance
        J_shaped: (52,3) shaped joint positions
        targets: dict from extract_g1_targets() — L_thumb_pos, R_thumb_pos used
        root_trans, root_orient, body_pose_init: numpy arrays, all held fixed
            except free-joint entries of body_pose
        free_joints: iterable of SMPLH joint indices to optimize
        w_pos, w_drift: loss weights
        n_outer, max_iter: L-BFGS iterations

    Returns:
        body_pose — numpy (63,) with refined free joints, rest = init
    """
    dev = torch.device(device)
    if hand_L is None:
        hand_L = np.zeros(45, dtype=np.float64)
    if hand_R is None:
        hand_R = np.zeros(45, dtype=np.float64)
    hand_L_t = torch.tensor(hand_L, dtype=torch.float64, device=dev)
    hand_R_t = torch.tensor(hand_R, dtype=torch.float64, device=dev)

    # Fixed tensors
    root_trans_t = torch.tensor(root_trans, dtype=torch.float64, device=dev)
    root_orient_t = torch.tensor(root_orient, dtype=torch.float64, device=dev)
    body_pose_init_t = torch.tensor(body_pose_init, dtype=torch.float64, device=dev)

    # Free-joint mask (63,)
    free_mask_np = np.zeros(63, dtype=np.float64)
    for j in free_joints:
        base = (j - 1) * 3
        free_mask_np[base:base + 3] = 1.0
    free_mask = torch.tensor(free_mask_np, dtype=torch.float64, device=dev)

    # Uniform drift weight (no special wrist handling — wrist is frozen)
    joint_weights_t = torch.full((63,), w_drift, dtype=torch.float64, device=dev)

    # Optimization variable: delta for pose params (masked to free joints)
    delta = torch.zeros(63, dtype=torch.float64, device=dev, requires_grad=True)

    # Thumb targets only
    t_targets = {}
    for name in ["L_thumb", "R_thumb"]:
        key = name + "_pos"
        t_targets[key] = torch.tensor(targets[key], dtype=torch.float64, device=dev)

    optimizer = torch.optim.LBFGS(
        [delta], max_iter=max_iter, lr=1.0,
        line_search_fn='strong_wolfe',
        tolerance_grad=1e-10, tolerance_change=1e-14)

    def closure():
        optimizer.zero_grad()
        body_pose = body_pose_init_t + free_mask * delta
        positions, rotations = smplh.forward_kinematics(
            root_trans_t, root_orient_t, body_pose, J_shaped,
            left_hand_pose=hand_L_t, right_hand_pose=hand_R_t)
        ee = smplh.end_effector_positions(positions, rotations)

        loss = torch.tensor(0.0, dtype=torch.float64, device=dev)
        for name in ["L_thumb", "R_thumb"]:
            key = name + "_pos"
            diff = ee[key] - t_targets[key]
            loss = loss + w_pos * torch.sum(diff * diff)

        drift = body_pose - body_pose_init_t
        loss = loss + torch.sum(joint_weights_t * drift * drift)

        loss.backward()
        return loss

    for _ in range(n_outer):
        optimizer.step(closure)

    body_pose_final = (body_pose_init_t + free_mask * delta).detach().cpu().numpy()
    return body_pose_final


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
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--refine", action="store_true",
                        help="Enable arm IK refinement (default: off)")
    parser.add_argument("--w-drift", type=float, default=10.0, dest="w_drift",
                        help="Arm IK: joint drift weight (default 10)")
    parser.add_argument("--task", type=str, default=None,
                        help="Override ACTIVE_TASK (e.g. "
                             "G1_WBT_Inspire_Pickup_Pillow_MainCamOnly)")
    args = parser.parse_args()

    out_dir = os.path.join(OUTPUT_DIR, "retarget_copy")
    os.makedirs(out_dir, exist_ok=True)

    # ── Load G1 model + episode data ──
    print("Loading G1 + episode...")
    model_g = pin.buildModelFromUrdf(G1_URDF, pin.JointModelFreeFlyer())
    data_g = model_g.createData()

    if args.task is not None:
        from render_overlay_check import load_episode_info as _load_ep_task
        from config import DATASET_ROOT
        data_dir = os.path.join(DATASET_ROOT, args.task)
        video_path, from_ts, to_ts, ep_df = _load_ep_task(
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
    else:
        for wrist_rot in variants:
            render_one(wrist_rot)


if __name__ == "__main__":
    main()
