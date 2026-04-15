"""
SMPLH IK solver: solve for human body pose from G1 robot contact points.

Given 5 position targets (pelvis, 2 toes, 2 thumbs) + 3 orientation targets
(pelvis, 2 wrists) extracted from G1 FK, find SMPLH body pose parameters
that place the human end-effectors at the target positions.

Usage:
    from smplh_ik import SMPLHForIK, IKSolver, extract_g1_targets
    smplh = SMPLHForIK(device='cuda:2')
    solver = IKSolver(smplh)
    targets = extract_g1_targets(g1_transforms)
    result = solver.solve_frame(targets)
"""

import sys
import os
import numpy as np
import torch
import torch.nn.functional as F
import pinocchio as pin

from .config import G1_URDF, SMPLH_PATH

# -- Frame convention: SMPLH (X=left, Y=up, Z=fwd) -> G1 (X=fwd, Y=left, Z=up) --
R_SMPLH_TO_G1_NP = np.array([
    [0, 0, 1],   # G1_X = SMPLH_Z
    [1, 0, 0],   # G1_Y = SMPLH_X
    [0, 1, 0],   # G1_Z = SMPLH_Y
], dtype=np.float64)

# -- G1 calibrated keypoints (link_name, local_offset) --
G1_KEYPOINTS = {
    "L_toe":   ("left_ankle_roll_link",  np.array([+0.1424,  0.0000, -0.0210])),
    "R_toe":   ("right_ankle_roll_link", np.array([+0.1424,  0.0000, -0.0215])),
    "L_thumb": ("left_thumb_4",          np.array([-0.0314, +0.0150, -0.0101])),
    "R_thumb": ("right_thumb_4",         np.array([+0.0314, +0.0150, -0.0101])),
}

# -- SMPLH end-effector definitions --
# (joint_index, local_offset in SMPLH rest-pose local frame)
# Offsets computed from mesh vertex analysis
SMPLH_END_EFFECTORS = {
    "pelvis":  (0,  np.array([0.0, 0.0, 0.0])),
    "L_toe":   (10, np.array([-0.0046, -0.0119, +0.0696])),
    "R_toe":   (11, np.array([+0.0037, -0.0147, +0.0682])),
    "L_thumb": (36, np.array([+0.0239, +0.0002, +0.0224])),
    "R_thumb": (51, np.array([-0.0242, -0.0012, +0.0225])),
}

# -- IK constraint weights --
DEFAULT_WEIGHTS = {
    "pelvis_pos": 10.0,
    "L_toe_pos": 1000.0,
    "R_toe_pos": 1000.0,
    "L_thumb_pos": 1000.0,
    "R_thumb_pos": 1000.0,
    "pelvis_rot": 100.0,
    "L_wrist_rot": 100.0,
    "R_wrist_rot": 100.0,
    "L_shoulder_rot": 1.0,
    "R_shoulder_rot": 1.0,
    "L_elbow_rot": 1.0,
    "R_elbow_rot": 1.0,
    "L_ankle_rot": 500.0,
    "R_ankle_rot": 500.0,
    "upright": 500.0,
    "pose_reg": 0.01,
    "temporal": 1.0,
}

# -- G1 link -> SMPLH joint for orientation constraints --
ORIENTATION_MAP = {
    "pelvis_rot":  ("pelvis", 0),
    "L_wrist_rot": ("left_wrist_yaw_link", 20),
    "R_wrist_rot": ("right_wrist_yaw_link", 21),
    "L_shoulder_rot": ("left_shoulder_yaw_link", 16),
    "R_shoulder_rot": ("right_shoulder_yaw_link", 17),
    "L_elbow_rot":    ("left_elbow_link", 18),
    "R_elbow_rot":    ("right_elbow_link", 19),
    "L_ankle_rot": ("left_ankle_roll_link", 7),
    "R_ankle_rot": ("right_ankle_roll_link", 8),
}


def _rodrigues_batch(rvec):
    """Batch axis-angle to rotation matrix. (N, 3) -> (N, 3, 3)."""
    theta = torch.norm(rvec, dim=1, keepdim=True).unsqueeze(-1)  # (N, 1, 1)
    # Normalize axis (handle near-zero)
    axis = rvec / (theta.squeeze(-1) + 1e-8)  # (N, 3)

    # Skew-symmetric matrix
    zero = torch.zeros_like(axis[:, 0])
    K = torch.stack([
        zero,      -axis[:, 2],  axis[:, 1],
        axis[:, 2], zero,       -axis[:, 0],
        -axis[:, 1], axis[:, 0], zero,
    ], dim=1).reshape(-1, 3, 3)

    # Rodrigues formula: R = I + sin(theta)K + (1-cos(theta))K^2
    eye = torch.eye(3, dtype=rvec.dtype, device=rvec.device).unsqueeze(0)
    sin_t = torch.sin(theta)
    cos_t = torch.cos(theta)
    R = eye + sin_t * K + (1.0 - cos_t) * (K @ K)

    # For very small angles, use first-order approximation: R ~ I + K
    small = (theta.squeeze(-1).squeeze(-1) < 1e-6)
    if small.any():
        R[small] = eye.expand(small.sum(), -1, -1) + K[small]

    return R


def _rodrigues(rvec):
    """Single axis-angle to rotation matrix. (3,) -> (3, 3)."""
    return _rodrigues_batch(rvec.unsqueeze(0))[0]


def _rotation_distance(R1, R2):
    """Geodesic distance^2 between two rotation matrices."""
    R_diff = R1.T @ R2
    cos_angle = (torch.trace(R_diff) - 1.0) / 2.0
    cos_angle = torch.clamp(cos_angle, -1.0 + 1e-7, 1.0 - 1e-7)
    angle = torch.acos(cos_angle)
    return angle * angle


def extract_g1_targets(transforms):
    """Extract IK target positions and orientations from G1 FK transforms.

    Args:
        transforms: dict from do_fk(), mapping link_name -> (t, R)

    Returns:
        dict with keys:
            pelvis_pos, L_toe_pos, R_toe_pos, L_thumb_pos, R_thumb_pos: (3,) np arrays
            pelvis_rot, L_wrist_rot, R_wrist_rot: (3,3) np arrays
    """
    targets = {}

    # Pelvis position
    targets["pelvis_pos"] = transforms["pelvis"][0].copy()

    # Keypoint positions (toe tips, thumb tips)
    for name, (link, offset) in G1_KEYPOINTS.items():
        t, R = transforms[link]
        targets[name + "_pos"] = R @ offset + t

    # Orientations
    targets["pelvis_rot"] = transforms["pelvis"][1].copy()
    targets["L_wrist_rot"] = transforms["left_wrist_yaw_link"][1].copy()
    targets["R_wrist_rot"] = transforms["right_wrist_yaw_link"][1].copy()
    targets["L_shoulder_rot"] = transforms["left_shoulder_yaw_link"][1].copy()
    targets["R_shoulder_rot"] = transforms["right_shoulder_yaw_link"][1].copy()
    targets["L_elbow_rot"] = transforms["left_elbow_link"][1].copy()
    targets["R_elbow_rot"] = transforms["right_elbow_link"][1].copy()
    targets["L_ankle_rot"] = transforms["left_ankle_roll_link"][1].copy()
    targets["R_ankle_rot"] = transforms["right_ankle_roll_link"][1].copy()

    return targets


class SMPLHForIK:
    """SMPLH model for IK: differentiable FK + LBS."""

    def __init__(self, model_path=SMPLH_PATH, device='cpu'):
        self.device = torch.device(device)

        # Load SMPLH data
        data = np.load(model_path, allow_pickle=True)
        self.v_template = torch.tensor(data['v_template'], dtype=torch.float64,
                                       device=self.device)  # (6890, 3)
        self.shapedirs = torch.tensor(data['shapedirs'], dtype=torch.float64,
                                      device=self.device)   # (6890, 3, 16)
        self.J_regressor = torch.tensor(data['J_regressor'].toarray()
                                        if hasattr(data['J_regressor'], 'toarray')
                                        else data['J_regressor'],
                                        dtype=torch.float64,
                                        device=self.device)  # (52, 6890)
        self.weights = torch.tensor(data['weights'], dtype=torch.float64,
                                    device=self.device)      # (6890, 52)
        self.faces = data['f'].astype(np.int64)              # (13776, 3) keep on CPU

        # Hand PCA
        self.hands_mean_l = torch.tensor(data['hands_meanl'], dtype=torch.float64,
                                         device=self.device)  # (45,)
        self.hands_mean_r = torch.tensor(data['hands_meanr'], dtype=torch.float64,
                                         device=self.device)
        self.hands_components_l = torch.tensor(data['hands_componentsl'],
                                               dtype=torch.float64,
                                               device=self.device)  # (45, 45)
        self.hands_components_r = torch.tensor(data['hands_componentsr'],
                                               dtype=torch.float64,
                                               device=self.device)

        # Kinematic tree
        kt = data['kintree_table'].astype(np.int64)
        self.parents = kt[0].copy()
        self.parents[0] = -1
        self.n_joints = 52
        self.n_body_joints = 22  # 0-21

        # Frame conversion
        self.R_S2G = torch.tensor(R_SMPLH_TO_G1_NP, dtype=torch.float64,
                                  device=self.device)
        self.R_G2S = self.R_S2G.T

        # End-effector offsets (in SMPLH local frame)
        self.ee_offsets = {}
        for name, (joint_id, offset) in SMPLH_END_EFFECTORS.items():
            self.ee_offsets[name] = (
                joint_id,
                torch.tensor(offset, dtype=torch.float64, device=self.device),
            )

        # Precompute default shape
        self._J_rest = (self.J_regressor @ self.v_template)  # (52, 3)

        # Compute G1-matching body scale
        self._g1_body_scale = self._compute_g1_scale()

        # Compute orientation offsets (G1<->SMPLH zero-pose alignment)
        self.R_offsets = self._compute_orientation_offsets()

    def _compute_g1_scale(self):
        """Compute scale factor to match SMPLH body size to G1.

        Uses a weighted blend of leg and arm ratios. Pure leg ratio (0.88)
        makes arms too long; pure arm ratio (0.56) makes legs too short.
        A blend of 0.80 gives reasonable proportions for both.
        """
        model = pin.buildModelFromUrdf(G1_URDF, pin.JointModelFreeFlyer())
        data_pin = model.createData()
        q0 = pin.neutral(model)
        q0[0:3] = [0, 0, 0.75]
        q0[3:7] = [0, 0, 0, 1]
        pin.forwardKinematics(model, data_pin, q0)
        pin.updateFramePlacements(model, data_pin)

        g1_pos = {}
        for i in range(model.nframes):
            name = model.frames[i].name
            g1_pos[name] = data_pin.oMf[i].translation.copy()

        J = self._J_rest.cpu().numpy()

        g1_leg = np.linalg.norm(g1_pos['pelvis'] - g1_pos['left_ankle_roll_link'])
        smplh_leg = np.linalg.norm(J[0] - J[7])
        leg_ratio = g1_leg / smplh_leg

        g1_arm = np.linalg.norm(
            g1_pos['left_shoulder_yaw_link'] - g1_pos['left_wrist_yaw_link'])
        smplh_arm = np.linalg.norm(J[16] - J[20])
        arm_ratio = g1_arm / smplh_arm

        # Weighted blend: favor legs (ground contact) but don't ignore arms
        scale = 0.6 * leg_ratio + 0.4 * arm_ratio
        print(f"  Body scale: {scale:.4f} "
              f"(leg_ratio={leg_ratio:.3f}, arm_ratio={arm_ratio:.3f})")
        return scale

    def shape_blend(self, betas=None, body_scale=None):
        """Apply shape blend shapes + optional body scale.

        Args:
            betas: (16,) shape params or None
            body_scale: float scale factor, None = use G1-matching scale,
                        1.0 = no scaling

        Returns (J_shaped, v_shaped) -- scaled relative to pelvis.
        """
        if betas is None:
            J_shaped = self._J_rest.clone()
            v_shaped = self.v_template.clone()
        else:
            if not isinstance(betas, torch.Tensor):
                betas = torch.tensor(betas, dtype=torch.float64, device=self.device)
            n = min(len(betas), self.shapedirs.shape[2])
            v_shaped = self.v_template + torch.einsum(
                'vcd,d->vc', self.shapedirs[:, :, :n], betas[:n])
            J_shaped = self.J_regressor @ v_shaped

        # Apply body scale (relative to pelvis)
        if body_scale is None:
            body_scale = self._g1_body_scale
        if body_scale != 1.0:
            pelvis = J_shaped[0].clone()
            J_shaped = pelvis + (J_shaped - pelvis) * body_scale
            v_shaped = pelvis + (v_shaped - pelvis) * body_scale

        return J_shaped, v_shaped

    def forward_kinematics(self, root_trans, root_orient, body_pose,
                           J_shaped, left_hand_pose=None, right_hand_pose=None):
        """Differentiable FK (no in-place ops for autograd compatibility).

        Args:
            root_trans: (3,) root translation in SMPLH frame
            root_orient: (3,) root orientation (axis-angle)
            body_pose: (63,) body joints 1-21, axis-angle
            J_shaped: (52, 3) shaped joint positions
            left_hand_pose: (45,) or None (uses hands_mean)
            right_hand_pose: (45,) or None

        Returns:
            positions: (52, 3) joint positions in SMPLH frame
            rotations: (52, 3, 3) global rotation matrices
        """
        if left_hand_pose is None:
            left_hand_pose = self.hands_mean_l
        if right_hand_pose is None:
            right_hand_pose = self.hands_mean_r

        # Batch Rodrigues for all joints
        all_pose = torch.cat([root_orient.unsqueeze(0),
                              body_pose.reshape(-1, 3),
                              left_hand_pose.reshape(-1, 3),
                              right_hand_pose.reshape(-1, 3)], dim=0)  # (52, 3)
        R_local = _rodrigues_batch(all_pose)  # (52, 3, 3)

        # Chain FK using lists (avoid in-place tensor ops)
        pos_list = [None] * self.n_joints
        rot_list = [None] * self.n_joints

        rot_list[0] = R_local[0]
        pos_list[0] = root_trans

        for i in range(1, self.n_joints):
            p = int(self.parents[i])
            bone = J_shaped[i] - J_shaped[p]
            rot_list[i] = rot_list[p] @ R_local[i]
            pos_list[i] = rot_list[p] @ bone + pos_list[p]

        positions = torch.stack(pos_list, dim=0)   # (52, 3)
        rotations = torch.stack(rot_list, dim=0)   # (52, 3, 3)
        return positions, rotations

    def end_effector_positions(self, positions, rotations):
        """Compute end-effector positions in G1 frame.

        Returns dict: name -> (3,) tensor in G1 frame.
        """
        result = {}
        for name, (joint_id, offset) in self.ee_offsets.items():
            # ee = joint_pos + R_global @ local_offset
            ee_smplh = positions[joint_id] + rotations[joint_id] @ offset
            # Convert to G1 frame
            result[name + "_pos"] = self.R_S2G @ ee_smplh
        return result

    def end_effector_orientations(self, rotations):
        """Compute end-effector orientations in G1 frame with R_offset correction.

        Returns dict: constraint_name -> (3,3) tensor in G1 frame.
        """
        result = {}
        for constraint_name, (_, smplh_joint) in ORIENTATION_MAP.items():
            R_smplh = rotations[smplh_joint]
            # Convert SMPLH rotation to G1 frame
            R_g1 = self.R_S2G @ R_smplh @ self.R_G2S
            result[constraint_name] = R_g1
        return result

    def _compute_orientation_offsets(self):
        """Compute R_offset for each orientation constraint.

        At zero pose: R_offset = R_smplh_zero_g1 @ R_g1_zero.T
        This is applied to G1 targets to make them comparable to SMPLH predictions.
        """
        # SMPLH zero pose: all rotations are identity
        # In G1 frame, SMPLH root at zero has rotation R_S2G @ I @ R_G2S = I
        # (since R_S2G is orthogonal so R_S2G @ R_S2G.T = I)
        # Actually: the SMPLH global rotation of joint j at zero pose = identity (all R_local = I)
        # Converted to G1 frame: R_S2G @ I @ R_G2S = R_S2G @ R_S2G.T = I
        # So R_smplh_zero_g1 = I for all joints at zero pose.

        # G1 zero pose FK
        model = pin.buildModelFromUrdf(G1_URDF, pin.JointModelFreeFlyer())
        data_pin = model.createData()
        q0 = pin.neutral(model)
        q0[0:3] = [0, 0, 0]
        q0[3:7] = [0, 0, 0, 1]
        pin.forwardKinematics(model, data_pin, q0)
        pin.updateFramePlacements(model, data_pin)

        g1_zero_transforms = {}
        for i in range(model.nframes):
            name = model.frames[i].name
            T = data_pin.oMf[i]
            g1_zero_transforms[name] = (T.translation.copy(), T.rotation.copy())

        R_offsets = {}
        for constraint_name, (g1_link, smplh_joint) in ORIENTATION_MAP.items():
            R_g1_zero = g1_zero_transforms[g1_link][1]
            # R_smplh_zero in G1 frame = identity (see above)
            # R_offset = R_smplh_zero_g1 @ R_g1_zero.T = I @ R_g1_zero.T = R_g1_zero.T
            R_offset = R_g1_zero.T
            R_offsets[constraint_name] = torch.tensor(
                R_offset, dtype=torch.float64, device=self.device)

        return R_offsets

    def lbs(self, root_trans, root_orient, body_pose, J_shaped, v_shaped,
            left_hand_pose=None, right_hand_pose=None):
        """Linear Blend Skinning: compute posed mesh vertices.

        Returns: (6890, 3) vertices in SMPLH frame.
        """
        positions, rotations = self.forward_kinematics(
            root_trans, root_orient, body_pose, J_shaped,
            left_hand_pose, right_hand_pose)

        # Build per-joint transforms: T[j] = G[j] @ G_rest_inv[j]
        # G_rest_inv[j] = [[I, -J_shaped[j]], [0, 1]]
        # T[j] = [[R_global[j], positions[j] - R_global[j] @ J_shaped[j]], [0, 1]]
        T = torch.zeros(self.n_joints, 4, 4, dtype=torch.float64,
                        device=self.device)
        T[:, :3, :3] = rotations
        T[:, :3, 3] = positions - torch.bmm(
            rotations, J_shaped.unsqueeze(-1)).squeeze(-1)
        T[:, 3, 3] = 1.0

        # Blend: v_posed = sum_j w[v,j] * T[j] @ [v_shaped; 1]
        # Efficient: compute weighted sum of transforms, then apply
        W = self.weights  # (V, J)
        T_blend = torch.einsum('vj,jab->vab', W, T)  # (V, 4, 4)

        v_homo = torch.cat([v_shaped,
                            torch.ones(len(v_shaped), 1, dtype=torch.float64,
                                       device=self.device)], dim=1)  # (V, 4)
        v_posed = torch.bmm(T_blend[:, :3, :],
                            v_homo.unsqueeze(-1)).squeeze(-1)  # (V, 3)
        return v_posed

    def lbs_to_g1(self, root_trans, root_orient, body_pose, J_shaped, v_shaped,
                  left_hand_pose=None, right_hand_pose=None):
        """LBS + convert to G1 frame. Returns (6890, 3) numpy array."""
        v_smplh = self.lbs(root_trans, root_orient, body_pose, J_shaped,
                           v_shaped, left_hand_pose, right_hand_pose)
        v_g1 = (self.R_S2G @ v_smplh.T).T
        return v_g1.detach().cpu().numpy()


class IKResult:
    """Result of IK solve."""
    __slots__ = ('root_trans', 'root_orient', 'body_pose',
                 'loss', 'pos_errors', 'rot_errors', 'n_iters')

    def __init__(self, root_trans, root_orient, body_pose,
                 loss, pos_errors, rot_errors, n_iters):
        self.root_trans = root_trans    # (3,) numpy
        self.root_orient = root_orient  # (3,) numpy
        self.body_pose = body_pose      # (63,) numpy
        self.loss = loss                # scalar
        self.pos_errors = pos_errors    # dict: name -> error in meters
        self.rot_errors = rot_errors    # dict: name -> error in degrees
        self.n_iters = n_iters


class IKSolver:
    """Solve for SMPLH body pose given G1 target points."""

    def __init__(self, smplh_model, weights=None, max_iter=100):
        self.smplh = smplh_model
        self.device = smplh_model.device
        self.weights = dict(DEFAULT_WEIGHTS)
        if weights:
            self.weights.update(weights)
        self.max_iter = max_iter

    def solve_frame(self, targets, betas=None, body_scale=None,
                    init_pose=None, prev_pose=None):
        """Solve IK for a single frame.

        Args:
            targets: dict from extract_g1_targets()
            betas: (16,) shape params or None
            body_scale: float or None (None = auto-match G1 size)
            init_pose: dict with root_trans, root_orient, body_pose for warm start
            prev_pose: body_pose from previous frame (for temporal smoothness)

        Returns: IKResult
        """
        smplh = self.smplh
        w = self.weights

        # Shape blending with scale (constant during optimization)
        J_shaped, _ = smplh.shape_blend(betas, body_scale=body_scale)

        # Convert targets to tensors
        t_targets = {}
        for key, val in targets.items():
            t_targets[key] = torch.tensor(val, dtype=torch.float64,
                                          device=self.device)

        # Initialize parameters
        if init_pose is not None:
            root_trans = torch.tensor(init_pose['root_trans'],
                                      dtype=torch.float64, device=self.device,
                                      requires_grad=True)
            root_orient = torch.tensor(init_pose['root_orient'],
                                       dtype=torch.float64, device=self.device,
                                       requires_grad=True)
            body_pose = torch.tensor(init_pose['body_pose'],
                                     dtype=torch.float64, device=self.device,
                                     requires_grad=True)
        else:
            # Initialize root_trans from G1 pelvis position (converted to SMPLH frame)
            pelvis_g1 = targets["pelvis_pos"]
            root_trans_init = R_SMPLH_TO_G1_NP.T @ pelvis_g1
            root_trans = torch.tensor(root_trans_init, dtype=torch.float64,
                                      device=self.device, requires_grad=True)

            # Initialize root_orient from G1 pelvis rotation
            pelvis_rot_g1 = targets["pelvis_rot"]
            # Convert to SMPLH frame and then to axis-angle
            R_root_smplh = R_SMPLH_TO_G1_NP.T @ pelvis_rot_g1 @ R_SMPLH_TO_G1_NP
            rvec, _ = __import__('cv2').Rodrigues(R_root_smplh)
            root_orient = torch.tensor(rvec.flatten(), dtype=torch.float64,
                                       device=self.device, requires_grad=True)

            body_pose = torch.zeros(63, dtype=torch.float64,
                                    device=self.device, requires_grad=True)

        # Previous pose for temporal smoothness
        t_prev = None
        if prev_pose is not None:
            t_prev = torch.tensor(prev_pose, dtype=torch.float64,
                                  device=self.device)

        params = [root_trans, root_orient, body_pose]

        # L-BFGS optimizer
        optimizer = torch.optim.LBFGS(params, max_iter=50, lr=1.0,
                                       line_search_fn='strong_wolfe',
                                       tolerance_grad=1e-10,
                                       tolerance_change=1e-14)

        best_loss = float('inf')
        n_iters = 0

        def closure():
            nonlocal best_loss, n_iters
            optimizer.zero_grad()

            positions, rotations = smplh.forward_kinematics(
                root_trans, root_orient, body_pose, J_shaped)

            # Position costs
            ee_pos = smplh.end_effector_positions(positions, rotations)
            loss = torch.tensor(0.0, dtype=torch.float64, device=self.device)

            for name in ["pelvis", "L_toe", "R_toe", "L_thumb", "R_thumb"]:
                key = name + "_pos"
                diff = ee_pos[key] - t_targets[key]
                loss = loss + w[key] * torch.sum(diff * diff)

            # Orientation costs (all entries in ORIENTATION_MAP)
            ee_rot = smplh.end_effector_orientations(rotations)
            for constraint_name in ORIENTATION_MAP:
                R_pred = ee_rot[constraint_name]
                R_target = smplh.R_offsets[constraint_name] @ t_targets[constraint_name]
                loss = loss + w[constraint_name] * _rotation_distance(R_pred, R_target)

            # Upright constraint: spine3 (joint 9) Z-axis in G1 frame should point up
            R_spine3_g1 = smplh.R_S2G @ rotations[9] @ smplh.R_G2S
            spine_up = R_spine3_g1[:, 2]  # Z column = local up in G1 frame
            world_up = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64,
                                     device=self.device)
            # Penalize deviation from vertical: 1 - cos(angle) = 1 - dot
            upright_err = 1.0 - torch.dot(spine_up, world_up)
            loss = loss + w["upright"] * upright_err * upright_err

            # Pose regularization
            loss = loss + w["pose_reg"] * torch.sum(body_pose * body_pose)

            # Temporal smoothness
            if t_prev is not None:
                diff = body_pose - t_prev
                loss = loss + w["temporal"] * torch.sum(diff * diff)

            loss.backward()
            n_iters += 1

            if loss.item() < best_loss:
                best_loss = loss.item()

            return loss

        # Run L-BFGS (outer loop for multiple L-BFGS calls)
        for _ in range(10):
            optimizer.step(closure)

        # Compute final errors for reporting
        with torch.no_grad():
            positions, rotations = smplh.forward_kinematics(
                root_trans, root_orient, body_pose, J_shaped)
            ee_pos = smplh.end_effector_positions(positions, rotations)
            ee_rot = smplh.end_effector_orientations(rotations)

            pos_errors = {}
            for name in ["pelvis", "L_toe", "R_toe", "L_thumb", "R_thumb"]:
                key = name + "_pos"
                err = torch.norm(ee_pos[key] - t_targets[key]).item()
                pos_errors[key] = err

            rot_errors = {}
            for constraint_name in ORIENTATION_MAP:
                R_pred = ee_rot[constraint_name]
                R_target = smplh.R_offsets[constraint_name] @ t_targets[constraint_name]
                angle = torch.sqrt(_rotation_distance(R_pred, R_target) + 1e-12).item()
                rot_errors[constraint_name] = np.degrees(angle)

            # Upright error
            R_spine3_g1 = smplh.R_S2G @ rotations[9] @ smplh.R_G2S
            spine_up = R_spine3_g1[:, 2]
            upright_angle = torch.acos(torch.clamp(spine_up[2], -1.0, 1.0))
            rot_errors["upright"] = np.degrees(upright_angle.item())

        return IKResult(
            root_trans=root_trans.detach().cpu().numpy(),
            root_orient=root_orient.detach().cpu().numpy(),
            body_pose=body_pose.detach().cpu().numpy(),
            loss=best_loss,
            pos_errors=pos_errors,
            rot_errors=rot_errors,
            n_iters=n_iters,
        )

    def solve_sequence(self, targets_list, betas=None, body_scale=None):
        """Solve IK for a sequence of frames with warm-starting.

        Args:
            targets_list: list of dicts from extract_g1_targets()
            betas: shape params (shared across frames)
            body_scale: float or None (None = auto-match G1 size)

        Returns: list of IKResult
        """
        results = []
        prev_pose = None
        init_pose = None

        for i, targets in enumerate(targets_list):
            result = self.solve_frame(targets, betas=betas,
                                      body_scale=body_scale,
                                      init_pose=init_pose,
                                      prev_pose=prev_pose)
            results.append(result)

            # Warm start next frame
            prev_pose = result.body_pose.copy()
            init_pose = {
                'root_trans': result.root_trans,
                'root_orient': result.root_orient,
                'body_pose': result.body_pose,
            }

            if (i + 1) % 30 == 0 or i == 0:
                max_pos = max(result.pos_errors.values())
                print(f"  Frame {i+1}/{len(targets_list)}: "
                      f"loss={result.loss:.4f}, "
                      f"max_pos_err={max_pos*1000:.1f}mm, "
                      f"iters={result.n_iters}")

        return results
