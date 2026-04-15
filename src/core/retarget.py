"""
Motion retargeting: pure joint-rotation copy from G1 -> SMPLH.

No IK, no optimization. Algorithm:
  1. G1 FK at neutral pose -> rest-pose link rotations.
  2. G1 FK at target frame -> current link rotations.
  3. World-frame rotation delta: dR[link] = R_current @ R_rest^T.
  4. Convert delta from G1 frame to SMPLH frame via R_S2G / R_G2S.
  5. Map SMPLH body joints (0-21) to corresponding G1 links; unmapped joints
     inherit parent's world rotation (-> local rotation = I).
  6. Convert world rotations to local rotations via parent traversal.
  7. body_pose[i-1] = axis-angle(R_local[i]).

Then optionally refine arms via IK to match thumb targets.
"""

import numpy as np
import cv2
import torch
import pinocchio as pin

from .config import G1_URDF
from .smplh import R_SMPLH_TO_G1_NP


# ── SMPLH body joint -> G1 (parent_link, child_link) pair defining the bone direction ──
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
    # Left leg chain: pelvis -> hip_pitch -> knee -> ankle
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
# G1 torso has 3 hinges (waist_yaw -> waist_roll -> waist_pitch) producing
# a full 3-DOF rotation between pelvis and torso_link. SMPLH spine has
# 3 joints (spine1/2/3) that should COLLECTIVELY reproduce this rotation
# while distributing the bend across the back instead of jamming it all
# into a single joint.
#
# Algorithm:
#   1. R_torso_rel = R_pelvis^T @ R_torso   (G1, in pelvis-local frame)
#      Subtract rest:  delta = R_torso_rel_curr @ R_torso_rel_rest^T
#   2. Convert to SMPLH frame: R_G2S @ delta @ R_S2G
#   3. Cube root via axis-angle: split angle by 3 (axis unchanged)
#   4. Apply identical R_seg as local rotation on each spine joint.
#      Because R_seg has the same axis at every step, composition gives
#      total = R_seg^3 = full delta, exact for arbitrary angle.
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
# pitch->roll->yaw hinge chain) around the current G1 upper-arm axis, and
# apply the same twist in SMPLH parent-local frame after the swing.
#
# Format: smplh_joint_idx -> (g1_yaw_link, g1_shoulder_base, g1_elbow)
SHOULDER_TWIST_MAP = {
    16: ("left_shoulder_yaw_link",
         "left_shoulder_pitch_link", "left_elbow_link"),
    17: ("right_shoulder_yaw_link",
         "right_shoulder_pitch_link", "right_elbow_link"),
}

# ── Wrist local-frame convention correction ────────────────────────────
#
# After DIRECT_ROT_MAP, the SMPLH wrist's WORLD rotation (in G1 frame)
# matches G1's wrist_yaw_link world rotation **exactly** (verified by
# diagnostic script -- matrices agree to 3 decimals). But when the hand
# mesh is rendered via LBS, the palm faces the wrong world direction.
#
# Root cause: the SMPLH hand mesh is defined in SMPLH's wrist-local
# frame (T-pose convention: palm-down, fingers along arm), while G1's
# hand meshes live in wrist_yaw_link's local frame (different axis
# convention). Same world rotation applied to different local "palm
# normal" vectors -> different rendered palm orientations.
#
# The fix is a CONSTANT local-frame rotation (pose-independent): rotate
# the SMPLH wrist by (-90 deg around local Y, then -90 deg around local Z)
# after the DIRECT_ROT_MAP assignment. Calibrated visually on ep0 f30
# by a grid search over Euler (x, y, z).
#
# Order: (ry, rz) applied as Rz @ Ry in local frame, mirrored L/R.
WRIST_LOCAL_CORRECTION_DEG = (0.0, -90.0, -90.0)

# ── Finger curl convention ─────────────────────────────────────────────
#
# For 4 non-thumb fingers (index / middle / pinky / ring), curling into
# the palm (matching Inspire hand's closed pose) is:
#   - axis: SMPLH wrist-local **Z** axis
#   - sign: **L = negative, R = positive** (mirrored)
#   - per-segment angle: scales linearly with G1 hand_state value
#
# Mapping at runtime: given G1 value v in [0, 1] where 0=closed, 1=open,
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
# SMPLH T-pose thumb is splayed outward -- G1 Inspire's default thumb is
# already opposed across the palm. To match, we apply a constant local
# rotation on thumb1 (the base joint). Values found via:
#   1. 6D free optimization minimizing |L_thumb_tip - R_thumb_tip| on
#      ep0 f30 -> found axes below at 102 deg/91 deg (= maximally inward)
#   2. User visual pick: 50 deg along the same axes looks closest to G1's
#      default thumb opposition.
THUMB_DEFAULT_L_AXIS = np.array([0.495, 0.681, -0.539])
THUMB_DEFAULT_R_AXIS = np.array([0.521, -0.437,  0.733])
THUMB_DEFAULT_ANGLE_DEG = 50.0

# ── SMPLH hand joint layout ──────────────────────────────────────────
#
# SMPLH hand: 15 joints per side x 3 DOF = 45-dim axis-angle per hand.
# Offsets within hand_pose (45-dim) per finger:
FINGER_SLOTS = {
    "index":  (0, 9),
    "middle": (9, 18),
    "pinky":  (18, 27),
    "ring":   (27, 36),
    "thumb":  (36, 45),
}


# ── Helper functions ──────────────────────────────────────────────────

def rot_to_axis_angle(R):
    """Rotation matrix (3,3) -> axis-angle (3,)."""
    rvec, _ = cv2.Rodrigues(np.ascontiguousarray(R, dtype=np.float64))
    return rvec.flatten()


def extract_twist_angle(R, axis):
    """Extract the signed twist angle (radians) of rotation R around unit `axis`.

    Swing-twist decomposition using quaternion projection: the "twist" part
    of R is the component that rotates around `axis`; the rest is "swing".
    Formula: take R's axis-angle -> quaternion (w, xyz), project xyz onto axis,
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
        # 180 deg: pick any axis perpendicular to a
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


# ── G1 rest transforms ────────────────────────────────────────────────

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


# ── Core retarget algorithm ───────────────────────────────────────────

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

    Args:
        g1_transforms: dict {link_name: (translation, rotation)} from FK
        g1_rest_transforms: dict {link_name: (translation, rotation)} from rest pose
        smplh: SMPLHForIK instance (needs .n_body_joints, .parents)
        J_shaped: (52, 3) shaped joint positions (torch or numpy)
        wrist_rot_deg: (rx, ry, rz) optional additional wrist rotation in degrees

    Returns: root_trans(3), root_orient(3), body_pose(63) as numpy arrays
    """
    # R_SMPLH_TO_G1_NP imported at module level

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
    # Subtract the rest baseline so a still G1 -> identity SMPLH spine.
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
            # additively (R(a,theta)^3 = R(a,3*theta)).
            R_local[i] = R_spine_seg
            R_world[i] = R_world[p] @ R_local[i]

        elif i in BONE_MAP:
            child_smplh, g1_parent, g1_child = BONE_MAP[i]

            # SMPLH bone vector in rest (parent local frame = world at rest,
            # since R_world_rest = I)
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
    #  (a) WRIST_LOCAL_CORRECTION_DEG  -- constant link-frame convention
    #      fix (see comment at top of file). Always applied.
    #  (b) wrist_rot_deg (from caller)  -- optional additional rotation,
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

    # (a) Constant correction -- always applied
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

    # World -> body_pose
    root_orient = rot_to_axis_angle(R_local[0])
    body_pose = np.zeros(63)
    for i in range(1, n):
        body_pose[(i - 1) * 3:(i - 1) * 3 + 3] = rot_to_axis_angle(R_local[i])

    # Root translation: G1 pelvis position in SMPLH frame.
    p_pelvis_g1 = g1_transforms["pelvis"][0]
    root_trans = R_G2S @ p_pelvis_g1

    return root_trans, root_orient, body_pose


# ── Hand pose construction ────────────────────────────────────────────

def scale_hands(smplh, J_shaped, v_shaped, hand_scale):
    """Scale both hands (joints + vertices) relative to their respective wrists.

    Body scale (shape_blend) already shrinks the SMPLH mesh to match G1
    overall body size. But G1's Inspire hand is physically larger than the
    scaled SMPLH hand, so we enlarge the hand region only.

    Vertices are scaled proportionally to their total LBS weight on
    hand joints (joints 22-36 for L, 37-51 for R), so the transition
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

    Uses THUMB_DEFAULT_{L,R}_AXIS x THUMB_DEFAULT_ANGLE_DEG on thumb1.
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

    Used to test the "thumb opposition" axis -- the constant rotation that
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
    i.e. hand_pose[3i:3i+3] = axis_vec * angle_rad for every joint in
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
        # 3 joints x 3 entries; write rot into each joint's slot
        for joint_offset in range(0, end - start, 3):
            pose[start + joint_offset:start + joint_offset + 3] = rot
    return pose.copy(), pose.copy()


# ── IK refinement ────────────────────────────────────────────────────

def refine_arms(smplh, J_shaped, targets,
                root_trans, root_orient, body_pose_init,
                device='cpu',
                w_pos=1000.0, w_drift=10.0,
                free_joints=(13, 14, 16, 17, 18, 19),
                n_outer=2, max_iter=20,
                hand_L=None, hand_R=None):
    """IK refinement -- optimize ONLY collar->elbow joints to match thumb targets.

    Called after deterministic root placement and the constant wrist
    correction. Root transform, wrist orientation, and non-arm body_pose
    are all held fixed.

    Free joints (default): L/R_collar (13/14), L/R_shoulder (16/17),
    L/R_elbow (18/19) = 6 joints x 3 = 18 DOF. **Wrist (20/21) is NOT
    optimized** -- its orientation is fixed by the retarget step's
    constant WRIST_LOCAL_CORRECTION_DEG, and IK should only adjust the
    upstream chain to bring the thumb tip to the target without
    disturbing the (already correct) palm orientation.

    Loss targets: L_thumb, R_thumb position only (6 DOF target).

    Args:
        smplh: SMPLHForIK instance
        J_shaped: (52,3) shaped joint positions
        targets: dict from extract_g1_targets() -- L_thumb_pos, R_thumb_pos used
        root_trans, root_orient, body_pose_init: numpy arrays, all held fixed
            except free-joint entries of body_pose
        free_joints: iterable of SMPLH joint indices to optimize
        w_pos, w_drift: loss weights
        n_outer, max_iter: L-BFGS iterations
        device: torch device string
        hand_L, hand_R: optional (45,) numpy hand poses

    Returns:
        body_pose -- numpy (63,) with refined free joints, rest = init
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

    # Uniform drift weight (no special wrist handling -- wrist is frozen)
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
