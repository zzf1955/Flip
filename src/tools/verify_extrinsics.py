"""
\u9a8c\u8bc1 URDF d435 \u5916\u53c2\u8f6c\u6362 + \u5173\u952e\u70b9\u6295\u5f71\u9a8c\u8bc1\u3002

\u6838\u5fc3\u95ee\u9898\uff1a\u5c06 URDF d435 \u7684 rpy=(0, 0.831, 0) \u6b63\u786e\u8f6c\u6362\u4e3a\u4ee3\u7801\u7684 (pitch, yaw, roll)\u3002

\u5173\u952e\u6311\u6218\uff1aURDF d435_link \u5750\u6807\u7cfb\u53ef\u80fd\u8ddf torso_link \u7ea6\u5b9a\u4e0d\u540c\u3002
"""

import sys
import os
import numpy as np
import cv2
import pandas as pd
import pinocchio as pin
from scipy.spatial.transform import Rotation

np.set_printoptions(precision=6, suppress=True)

from src.core.config import (G1_URDF, MESH_DIR, BEST_PARAMS, DATASET_ROOT,
                     TMP_DIR, get_hand_type, get_skip_meshes, CAMERA_MODEL)
from src.core.camera import get_model, build_K, build_D, model_is_fisheye, project_points_cv
from src.core.fk import (build_q, do_fk, parse_urdf_meshes, preload_meshes)
from src.core.camera import make_camera, make_camera_const

# ============================================================
# \u7b2c\u4e00\u6b65\uff1a\u7406\u89e3\u4ee3\u7801\u4e2d\u7684\u65cb\u8f6c\u6a21\u578b
# ============================================================
print("=" * 70)
print("\u7b2c\u4e00\u6b65\uff1a\u4ee3\u7801\u65cb\u8f6c\u6a21\u578b")
print("=" * 70)

R_body_to_cam = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]], dtype=np.float64)

print("""
\u4ee3\u7801 make_camera_const() \u4e2d:
  R_pitch = Ry(pitch)    -- \u7ed5 body Y \u8f74 (\u5de6\u65b9\u5411)
  R_yaw   = Rz(yaw)      -- \u7ed5 body Z \u8f74 (\u4e0a\u65b9\u5411)
  R_roll  = Rx(roll)      -- \u7ed5 body X \u8f74 (\u524d\u65b9\u5411)
  R_cam = R_body_to_cam @ R_roll @ R_yaw @ R_pitch

  R_body_to_cam: body(X\u524d Y\u5de6 Z\u4e0a) -> optical(X\u53f3 Y\u4e0b Z\u524d)

  \u4ee3\u7801\u8bed\u4e49: R_roll @ R_yaw @ R_pitch \u662f d435 \u5b89\u88c5\u65b9\u5411\u5728 body \u5750\u6807\u7cfb\u4e2d\u7684\u65cb\u8f6c\u3002
  \u7136\u540e R_body_to_cam \u5c06\u8fd9\u4e2a\u65cb\u8f6c\u540e\u7684 body \u5750\u6807\u7cfb\u8f6c\u6362\u4e3a\u5149\u5b66\u5750\u6807\u7cfb\u3002

  \u5149\u8f74 = R_cam^T @ [0,0,1] = (R_body_to_cam @ R_combo)^T @ [0,0,1]
  = R_combo^T @ R_body_to_cam^T @ [0,0,1]
  = R_combo^T @ [0, 0, 1]^T in R_body_to_cam^T
""")

# R_body_to_cam^T @ [0,0,1] = ?
print(f"R_body_to_cam^T @ [0,0,1] = {R_body_to_cam.T @ [0,0,1]}")
print("\u5373\u5149\u8f74\u5bf9\u5e94 body \u7684 X \u8f74(\u524d\u65b9)\uff0c\u7ecf\u8fc7 R_combo \u65cb\u8f6c\u540e")

def compute_optical_axis(pitch_deg, yaw_deg=0, roll_deg=0):
    """\u8ba1\u7b97\u7ed9\u5b9a\u53c2\u6570\u4e0b\u5149\u8f74\u5728 body \u5750\u6807\u7cfb\u4e2d\u7684\u65b9\u5411"""
    p = np.radians(pitch_deg)
    y = np.radians(yaw_deg)
    r = np.radians(roll_deg)
    R_p = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
    R_y = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
    R_r = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
    R_combo = R_r @ R_y @ R_p
    R_cam = R_body_to_cam @ R_combo
    optical = R_cam.T @ np.array([0, 0, 1])
    return optical, R_combo

print("\n\u5149\u8f74\u65b9\u5411 vs pitch:")
for p in [60, 30, 0, -30, -60, -90]:
    opt, _ = compute_optical_axis(p)
    elev = np.degrees(np.arctan2(opt[2], opt[0]))
    print(f"  pitch={p:6.1f}\u00b0 -> \u5149\u8f74 body = [{opt[0]:.3f}, {opt[1]:.3f}, {opt[2]:.3f}], \u4ef0\u89d2={elev:.1f}\u00b0")

# ============================================================
# \u7b2c\u4e8c\u6b65\uff1aURDF d435 \u5750\u6807\u7cfb\u5206\u6790
# ============================================================
print("\n" + "=" * 70)
print("\u7b2c\u4e8c\u6b65\uff1aURDF d435 \u5750\u6807\u7cfb\u5206\u6790")
print("=" * 70)

urdf_pitch_rad = 0.8307767239493009  # URDF rpy=(0, 0.831, 0) \u7684 pitch
R_urdf = Rotation.from_euler('xyz', [0, urdf_pitch_rad, 0]).as_matrix()

print(f"URDF d435 rpy = (0, 0.831, 0), pitch = {np.degrees(urdf_pitch_rad):.2f}\u00b0")
print(f"\nR_urdf (torso -> d435_link):")
print(R_urdf)

print(f"\nd435_link \u5404\u8f74\u5728 torso \u5750\u6807\u7cfb\u4e2d:")
print(f"  X\u8f74: {R_urdf @ [1,0,0]} (d435 \u524d\u65b9?)")
print(f"  Y\u8f74: {R_urdf @ [0,1,0]} (d435 \u5de6\u65b9?)")
print(f"  Z\u8f74: {R_urdf @ [0,0,1]} (d435 \u4e0a\u65b9?)")

# D435 X\u8f74\u5728torso\u4e2d = [0.674, 0, -0.739] -> \u671d\u524d\u504f\u4e0b 47.6\u00b0
# D435 Z\u8f74\u5728torso\u4e2d = [0.739, 0, 0.674] -> \u671d\u524d\u504f\u4e0a
d435_x = R_urdf @ np.array([1, 0, 0])
elev_x = np.degrees(np.arctan2(d435_x[2], d435_x[0]))
print(f"\nd435 X\u8f74\u4ef0\u89d2: {elev_x:.1f}\u00b0 (\u5982\u679cX\u662f\u5149\u8f74 -> \u671d\u4e0b {-elev_x:.1f}\u00b0)")

print("""
\u5173\u952e\u6d1e\u5bdf\uff01

D435 URDF link \u7684\u6807\u51c6\u7ea6\u5b9a (Intel RealSense URDF):
  X \u8f74: \u5411\u53f3 (\u4ece\u540e\u9762\u770b)
  Y \u8f74: \u5411\u4e0b
  Z \u8f74: \u5411\u524d (\u5149\u8f74\u65b9\u5411!)

\u4f46\u662f\u5b87\u6811 G1 \u7684 URDF \u53ef\u80fd\u4f7f\u7528\u4e0d\u540c\u7ea6\u5b9a:
  \u5982\u679c d435_link \u7684 X \u8f74 = \u5149\u8f74\u65b9\u5411 (\u6709\u4e9b URDF \u8fd9\u4e48\u5b9a\u4e49):
    \u5149\u8f74 in torso = R_urdf @ [1,0,0] = [0.674, 0, -0.739]
    \u4ef0\u89d2 = -47.6\u00b0 -> \u671d\u524d\u4e0b\u65b9 47.6\u00b0 \u2713 \u5408\u7406!

  \u5982\u679c d435_link \u7684 Z \u8f74 = \u5149\u8f74\u65b9\u5411 (Intel \u6807\u51c6):
    \u5149\u8f74 in torso = R_urdf @ [0,0,1] = [0.739, 0, 0.674]
    \u4ef0\u89d2 = +42.4\u00b0 -> \u671d\u524d\u4e0a\u65b9, \u4e0d\u5408\u7406

\u6240\u4ee5\u5b87\u6811 URDF \u4e2d d435_link \u5927\u6982\u7387\u662f X=\u5149\u8f74 (X\u524d Y\u5de6 Z\u4e0a) \u7684\u7ea6\u5b9a\u3002
""")

# ============================================================
# \u7b2c\u4e09\u6b65\uff1a\u6b63\u786e\u6362\u7b97 (\u8003\u8651 d435_link \u5750\u6807\u7cfb\u7ea6\u5b9a)
# ============================================================
print("=" * 70)
print("\u7b2c\u4e09\u6b65\uff1a\u6b63\u786e\u7684\u5916\u53c2\u6362\u7b97")
print("=" * 70)

print("""
\u5982\u679c d435_link: X=\u5149\u8f74(\u524d), Y=\u5de6, Z=\u4e0a
\u90a3\u4e48 d435_link \u548c torso_link \u4f7f\u7528\u76f8\u540c\u7684\u5750\u6807\u7ea6\u5b9a!

\u4ece torso body \u5750\u6807\u7cfb\u5230\u76f8\u673a\u5149\u5b66\u5750\u6807\u7cfb:
  R_torso_to_optical = R_link_to_optical @ R_urdf

\u5176\u4e2d R_link_to_optical: d435_link(X\u524d Y\u5de6 Z\u4e0a) -> optical(X\u53f3 Y\u4e0b Z\u524d)
  = R_body_to_cam = [[0,-1,0],[0,0,-1],[1,0,0]]

\u4ee3\u7801\u4e2d:
  R_cam = R_body_to_cam @ R_combo

  R_combo = R_urdf \u65f6\u5c31\u5bf9\u5e94 URDF \u5916\u53c2

\u4f46\u4ee3\u7801\u4e2d R_combo \u4f5c\u7528\u5728 body \u5750\u6807\u7cfb\u4e0a\uff0c\u542b\u4e49\u662f\u201c\u5728 body \u5750\u6807\u7cfb\u4e2d\u65cb\u8f6c\u201d
  \u6240\u4ee5\u5982\u679c d435_link \u548c torso_link \u7ea6\u5b9a\u4e00\u81f4\uff1a
  R_combo = R_urdf = Ry(47.6\u00b0) -> code pitch = +47.6\u00b0

  \u4f46 pitch=+47.6\u00b0 \u7684\u5149\u8f74\u671d\u4e0a\uff0c\u4e0d\u671d\u4e0b\uff01

\u77db\u76fe\uff01\u8ba9\u6211\u91cd\u65b0\u8ba1\u7b97...

\u95ee\u9898\u51fa\u5728\u5149\u8f74\u8ba1\u7b97\u4e0a\u3002\u8ba9\u6211\u4ed4\u7ec6\u63a8\u5bfc\uff1a

R_cam = R_body_to_cam @ R_combo, where R_combo = Ry(+47.6\u00b0)

\u76f8\u673a\u5750\u6807\u7cfb\u7684 Z \u8f74(\u5149\u8f74)\u5728 torso \u4e2d\u7684\u65b9\u5411:
  z_cam_in_torso = R_cam^(-1) @ [0,0,1]_cam
  = R_cam^T @ [0,0,1]
  = (R_body_to_cam @ R_combo)^T @ [0,0,1]
  = R_combo^T @ R_body_to_cam^T @ [0,0,1]

  R_body_to_cam^T @ [0,0,1] = [1,0,0]  (Z_cam -> X_body)

  R_combo^T @ [1,0,0] = Ry(-47.6\u00b0) @ [1,0,0]
  = [cos(-47.6\u00b0), 0, sin(-47.6\u00b0)]   # Wait, Ry^T = Ry(-theta)

  \u7b49\u7b49\uff0cRy(theta) = [[cos,-sin],[1],[sin,cos]] ?
  \u4e0d\uff0c\u4ee3\u7801\u4e2d R_pitch = [[cos,0,sin],[0,1,0],[-sin,0,cos]]

  Ry(47.6\u00b0)^T @ [1,0,0] = [[cos,0,-sin],[0,1,0],[sin,0,cos]] @ [1,0,0]
  = [cos(47.6\u00b0), 0, sin(47.6\u00b0)]
  = [0.674, 0, 0.739]  -> \u671d\u524d\u4e0a\u65b9

  \u8fd9\u8ddf\u4e4b\u524d\u7684\u8ba1\u7b97\u4e00\u81f4\uff0c\u5149\u8f74\u671d\u4e0a\u3002\u4e0d\u5bf9\u3002

\u5173\u952e\uff1ad435_link \u7684\u5750\u6807\u7cfb\u7ea6\u5b9a\u53ef\u80fd\u4e0d\u662f "X\u524d Y\u5de6 Z\u4e0a"!

\u8ba9\u6211\u8003\u8651\u53e6\u4e00\u79cd\u53ef\u80fd:
  d435_link: X=\u540e(\u6216Z=\u4e0b)
  --- \u4e0d\u662f\u6807\u51c6\u7ea6\u5b9a\uff0c\u4f46 rpy=(0,0.831,0) \u4e0b:

  d435 Z \u8f74 in torso = R_urdf @ [0,0,1] = [0.739, 0, 0.674] -> \u671d\u524d\u4e0a\u65b9

  \u8fd8\u662f\u4e0d\u5bf9\u3002

\u518d\u8003\u8651: \u4e5f\u8bb8 Unitree \u7684 torso_link \u5750\u6807\u7cfb\u4e0d\u662f X\u524d Y\u5de6 Z\u4e0a?
""")

# \u7528 pinocchio \u52a0\u8f7d\u6a21\u578b\uff0c\u7528\u5b9e\u9645\u6570\u636e\u9a8c\u8bc1
model = pin.buildModelFromUrdf(G1_URDF, pin.JointModelFreeFlyer())
data_pin = model.createData()

# Neutral pose
q = pin.neutral(model)
pin.forwardKinematics(model, data_pin, q)
pin.updateFramePlacements(model, data_pin)

# \u5217\u51fa\u5173\u952e frames
print("\u5173\u952e frame \u4f4d\u59ff (neutral pose):")
key_frames = {}
for i in range(model.nframes):
    name = model.frames[i].name
    if name in ("torso_link", "d435_link", "head_link", "pelvis",
                "left_hip_pitch_link", "right_hip_pitch_link",
                "left_shoulder_pitch_link", "right_shoulder_pitch_link"):
        T = data_pin.oMf[i]
        key_frames[name] = (T.translation.copy(), T.rotation.copy())
        rpy = Rotation.from_matrix(T.rotation).as_euler('xyz', degrees=True)
        print(f"  {name:40s} pos={T.translation}, rpy={rpy}")

# d435 \u76f8\u5bf9 torso
if "torso_link" in key_frames and "d435_link" in key_frames:
    t_torso, R_torso = key_frames["torso_link"]
    t_d435, R_d435 = key_frames["d435_link"]

    # \u76f8\u5bf9\u4f4d\u79fb (in torso frame)
    t_rel = R_torso.T @ (t_d435 - t_torso)
    R_rel = R_torso.T @ R_d435

    print(f"\nd435 \u76f8\u5bf9 torso:")
    print(f"  \u4f4d\u79fb (torso\u5750\u6807\u7cfb): {t_rel}")
    print(f"  \u65cb\u8f6c\u77e9\u9635:")
    print(f"  {R_rel}")
    rpy_rel = Rotation.from_matrix(R_rel).as_euler('xyz', degrees=True)
    print(f"  rpy: {rpy_rel} deg")

# ============================================================
# \u7b2c\u56db\u6b65\uff1a\u6df1\u5165\u5206\u6790 \u2014 \u76f4\u63a5\u5339\u914d\u4ee3\u7801\u7684\u76f8\u673a\u6a21\u578b
# ============================================================
print("\n" + "=" * 70)
print("\u7b2c\u56db\u6b65\uff1a\u76f4\u63a5\u5339\u914d\u4ee3\u7801\u7684\u76f8\u673a\u6a21\u578b")
print("=" * 70)

print("""
\u4ee3\u7801\u4e2d make_camera() \u7684\u5b8c\u6574\u6d41\u7a0b:

1. cam_pos = torso_t + torso_R @ offset
   - offset = [dx, dy, dz] \u5728 torso body \u5750\u6807\u7cfb\u4e2d

2. R_w2c = (torso_R @ R_cam.T).T = R_cam @ torso_R.T
   - R_cam = R_body_to_cam @ R_combo

3. \u5149\u8f74\u65b9\u5411 (world) = R_w2c^T @ [0,0,1]_cam = R_cam^(-1) \u5728 world \u4e2d\u7684 Z\u5217
   \u4f46\u8fd9\u53d6\u51b3\u4e8e torso_R

\u5728 neutral pose, torso_R = I (\u6052\u7b49), \u6240\u4ee5:
  cam_pos = t_torso + offset
  R_w2c = R_cam

  \u5bf9\u4e8e URDF \u7684 d435:
    cam_pos (world) = t_torso + [0.0576, 0.0175, 0.4299]
    cam_orient: \u9700\u8981\u627e\u5230 R_cam \u4f7f\u5f97\u76f8\u673a\u6307\u5411\u6b63\u786e\u65b9\u5411

\u5b9e\u9645\u4e0a\uff0c\u8ba9\u6211\u6362\u4e2a\u65b9\u5411\u3002\u4e0d\u8981\u4ece URDF \u63a8\u5bfc\u4ee3\u7801\u53c2\u6570\uff0c
\u800c\u662f\u4ece\u4ee3\u7801\u7684\u6807\u5b9a\u53c2\u6570\u63a8\u5bfc\u51fa\u5b83\u610f\u5473\u7740\u4ec0\u4e48 URDF \u59ff\u6001\uff0c
\u7136\u540e\u8ddf URDF \u7684 d435 \u59ff\u6001\u6bd4\u8f83\u3002
""")

# \u6807\u5b9a\u53c2\u6570: pitch=-61.59, yaw=2.17, roll=0.23 (degrees)
cal = BEST_PARAMS
print(f"\u6807\u5b9a\u53c2\u6570: {cal}")

# \u8ba1\u7b97\u6807\u5b9a\u7684 R_combo
p_c = np.radians(cal["pitch"])
y_c = np.radians(cal["yaw"])
r_c = np.radians(cal["roll"])
R_p = np.array([[np.cos(p_c), 0, np.sin(p_c)], [0, 1, 0], [-np.sin(p_c), 0, np.cos(p_c)]])
R_y = np.array([[np.cos(y_c), -np.sin(y_c), 0], [np.sin(y_c), np.cos(y_c), 0], [0, 0, 1]])
R_r = np.array([[1, 0, 0], [0, np.cos(r_c), -np.sin(r_c)], [0, np.sin(r_c), np.cos(r_c)]])
R_combo_cal = R_r @ R_y @ R_p

print(f"\n\u6807\u5b9a\u7684 R_combo (torso\u4e2d\u7684\u5b89\u88c5\u65cb\u8f6c):")
print(R_combo_cal)

# \u6807\u5b9a R_combo \u7684\u5149\u8f74 (torso \u5750\u6807\u7cfb\u4e2d)
opt_cal = R_combo_cal.T @ np.array([1, 0, 0])
print(f"\n\u6807\u5b9a\u5149\u8f74 = R_combo^T @ [1,0,0] = {opt_cal}")
# \u7b49\u7b49\uff0c\u5e94\u8be5\u662f R_cam^T @ [0,0,1]
R_cam_cal = R_body_to_cam @ R_combo_cal
opt_cal = R_cam_cal.T @ np.array([0, 0, 1])
print(f"\u6807\u5b9a\u5149\u8f74 = (R_body_to_cam @ R_combo)^T @ [0,0,1] = {opt_cal}")
elev_cal = np.degrees(np.arctan2(opt_cal[2], opt_cal[0]))
print(f"\u6807\u5b9a\u5149\u8f74\u4ef0\u89d2: {elev_cal:.1f}\u00b0 (\u8d1f=\u671d\u4e0b)")

# URDF \u7684 d435 \u5728 torso \u4e2d:
# d435 X\u8f74 in torso = R_urdf @ [1,0,0] = [0.674, 0, -0.739] \u671d\u524d\u4e0b
# \u5982\u679c d435 \u7684 X \u8f74\u662f\u5149\u8f74:
print(f"\nURDF d435 X\u8f74 in torso = {R_urdf @ [1,0,0]}")
print(f"  \u4ef0\u89d2 = {np.degrees(np.arctan2((R_urdf @ [1,0,0])[2], (R_urdf @ [1,0,0])[0])):.1f}\u00b0")

# ============================================================
# \u5173\u952e\u63a8\u5bfc\uff1a\u4ece URDF \u7684 d435 \u59ff\u6001\u63a8\u5bfc\u4ee3\u7801\u53c2\u6570
# ============================================================
print("\n" + "=" * 70)
print("\u7b2c\u4e94\u6b65\uff1a\u4ece URDF d435 \u7269\u7406\u4f4d\u59ff\u63a8\u5bfc\u4ee3\u7801\u53c2\u6570")
print("=" * 70)

print("""
URDF \u544a\u8bc9\u6211\u4eec:
  d435 \u7684 X \u8f74\u5728 torso \u4e2d\u6307\u5411 [0.674, 0, -0.739] \u2014 \u671d\u524d\u4e0b\u65b9 47.6\u00b0

\u5982\u679c d435_link \u7684 X \u8f74\u662f\u5149\u8f74\u65b9\u5411:
  \u6211\u4eec\u9700\u8981\u627e\u5230\u4ee3\u7801\u53c2\u6570\u4f7f\u5f97\u5149\u8f74 = [0.674, 0, -0.739]

  \u5149\u8f74 = R_cam^T @ [0,0,1] where R_cam = R_body_to_cam @ R_combo

  R_body_to_cam^T @ [0,0,1] = [1, 0, 0] (body X \u8f74)

  \u6240\u4ee5\u5149\u8f74 = R_combo^T @ [1, 0, 0]
  = R_combo \u7684\u7b2c\u4e00\u5217

  \u6211\u4eec\u9700\u8981 R_combo \u7684\u7b2c\u4e00\u5217 = [0.674, 0, -0.739]

  \u5982\u679c\u53ea\u7528 pitch: R_combo = Ry(pitch)
  Ry(p) \u7b2c\u4e00\u5217 = [cos(p), 0, -sin(p)]
  cos(p) = 0.674, sin(p) = 0.739
  p = arctan2(0.739, 0.674) = 47.6\u00b0 ?

  \u7b49\u7b49: -sin(p) = -0.739, \u6240\u4ee5 sin(p) = 0.739
  p = arcsin(0.739) = 47.6\u00b0

  \u4f46\u6211\u4eec\u4e4b\u524d\u7b97\u7684 pitch=47.6\u00b0 \u7ed9\u51fa\u5149\u8f74\u671d\u4e0a\uff01
  \u8ba9\u6211\u91cd\u65b0\u68c0\u67e5...
""")

# \u91cd\u65b0\u4e25\u683c\u8ba1\u7b97
# R_pitch (\u4ee3\u7801\u5b9a\u4e49):
# [[cos(p),  0, sin(p)],
#  [0,       1, 0     ],
#  [-sin(p), 0, cos(p)]]

# R_combo = Ry(p) (\u53ea\u6709pitch, yaw=roll=0)
# R_cam = R_body_to_cam @ Ry(p)
# \u5149\u8f74 in body = R_cam^T @ [0,0,1]
# = (R_body_to_cam @ Ry(p))^T @ [0,0,1]
# = Ry(p)^T @ R_body_to_cam^T @ [0,0,1]

# R_body_to_cam^T = [[0, 0, 1], [-1, 0, 0], [0, -1, 0]]
# R_body_to_cam^T @ [0,0,1] = [1, 0, 0]  \u2713

# Ry(p)^T = [[cos(p), 0, -sin(p)], [0, 1, 0], [sin(p), 0, cos(p)]]
# Ry(p)^T @ [1, 0, 0] = [cos(p), 0, sin(p)]

# \u6240\u4ee5\u5149\u8f74 = [cos(p), 0, sin(p)]
# \u5f53 p > 0: Z\u5206\u91cf > 0 -> \u671d\u4e0a  (\u56e0\u4e3a body Z = \u4e0a)
# \u5f53 p < 0: Z\u5206\u91cf < 0 -> \u671d\u4e0b \u2713

# URDF d435 \u5149\u8f74 (\u5982\u679cX\u8f74\u662f\u5149\u8f74) = [0.674, 0, -0.739]
# \u9700\u8981 [cos(p), 0, sin(p)] = [0.674, 0, -0.739]
# cos(p) = 0.674, sin(p) = -0.739
# p = atan2(-0.739, 0.674) = -47.6\u00b0 !!!

target_optical = np.array([0.674302, 0, -0.738455])
p_needed = np.degrees(np.arctan2(target_optical[2], target_optical[0]))
print(f"\u5982\u679c d435 X\u8f74 = \u5149\u8f74: \u9700\u8981 code pitch = {p_needed:.2f}\u00b0")

# \u9a8c\u8bc1
opt_check, _ = compute_optical_axis(p_needed)
print(f"  \u9a8c\u8bc1: \u5149\u8f74 = {opt_check}")
print(f"  vs \u76ee\u6807: {target_optical}")
print(f"  \u5dee\u5f02: {np.max(np.abs(opt_check - target_optical)):.6f}")

# ============================================================
print("\n" + "=" * 70)
print("\u5173\u952e\u7ed3\u8bba!")
print("=" * 70)

print(f"""
\u5982\u679c URDF d435_link \u7684 X \u8f74 = \u5149\u8f74\u65b9\u5411 (X\u524d Y\u5de6 Z\u4e0a \u7ea6\u5b9a):

  URDF pitch_urdf = +0.831 rad = +47.6\u00b0 (URDF\u7684Y\u8f74\u65cb\u8f6c)

  \u7531\u4e8e URDF \u7684 Ry(+47.6\u00b0) \u4f7f d435 \u7684 X\u8f74(\u5149\u8f74) \u671d\u524d\u4e0b\u65b9\u65cb\u8f6c,
  \u800c\u4ee3\u7801\u4e2d Ry(pitch) \u7684\u5149\u8f74 = [cos(p), 0, sin(p)],
  \u8981\u8ba9\u5149\u8f74\u671d\u524d\u4e0b\u65b9, \u9700\u8981 sin(p) < 0, \u5373 p < 0.

  \u5177\u4f53\u6765\u8bf4: code pitch = -47.6\u00b0 (\u53d6\u8d1f!)

  \u4e3a\u4ec0\u4e48\u53d6\u8d1f? \u56e0\u4e3a URDF Ry(\u03b8) \u7684\u542b\u4e49\u662f:
    X\u8f74 \u4ece [1,0,0] \u65cb\u8f6c\u5230 [cos(\u03b8), 0, -sin(\u03b8)] (\u6ce8\u610f\u8d1f\u53f7\u5728\u7b2c\u4e09\u4e2a\u5206\u91cf!)
    Z\u8f74 \u4ece [0,0,1] \u65cb\u8f6c\u5230 [sin(\u03b8), 0, cos(\u03b8)]

  \u800c\u4ee3\u7801\u7684\u5149\u8f74\u8ba1\u7b97 Ry(p)^T @ [1,0,0] = [cos(p), 0, sin(p)]

  \u6240\u4ee5 d435 X\u8f74\u65b9\u5411 = [cos(\u03b8), 0, -sin(\u03b8)] \u5bf9\u5e94\u5149\u8f74 [cos(p), 0, sin(p)]
  \u9700\u8981: sin(p) = -sin(\u03b8), \u5373 p = -\u03b8 = -47.6\u00b0

URDF \u5916\u53c2\u5bf9\u5e94\u7684\u4ee3\u7801\u53c2\u6570:
  dx    = {0.0576235}
  dy    = {0.01753}
  dz    = {0.42987}
  pitch = {-np.degrees(urdf_pitch_rad):.4f}\u00b0  (\u53d6\u8d1f!)
  yaw   = 0.0\u00b0
  roll  = 0.0\u00b0

vs \u6807\u5b9a\u7ed3\u679c:
  dx    = 0.0758
  dy    = 0.0226
  dz    = 0.4484
  pitch = -61.5855\u00b0
  yaw   = 2.1690\u00b0
  roll  = 0.2331\u00b0

pitch \u5dee\u5f02: {-np.degrees(urdf_pitch_rad) - (-61.5855):.2f}\u00b0 (URDF -47.6\u00b0 vs \u6807\u5b9a -61.6\u00b0)
""")

# \u7b49\u7b49\uff0c\u4f46 -47.6 \u548c -61.6 \u8fd8\u662f\u5dee 14 \u5ea6! \u8fd9\u4e0d\u662f\u7b80\u5355\u7684\u7b26\u53f7\u9519\u8bef
# \u8ba9\u6211\u91cd\u65b0\u68c0\u67e5\u662f\u5426\u771f\u7684\u9700\u8981\u53d6\u8d1f

print("=" * 70)
print("\u9a8c\u8bc1: pitch = -47.6\u00b0 \u7684\u5149\u8f74")
print("=" * 70)

opt_neg, R_combo_neg = compute_optical_axis(-47.6)
print(f"pitch=-47.6\u00b0 -> \u5149\u8f74 = {opt_neg}, \u4ef0\u89d2 = {np.degrees(np.arctan2(opt_neg[2], opt_neg[0])):.1f}\u00b0")

opt_pos, R_combo_pos = compute_optical_axis(47.6)
print(f"pitch=+47.6\u00b0 -> \u5149\u8f74 = {opt_pos}, \u4ef0\u89d2 = {np.degrees(np.arctan2(opt_pos[2], opt_pos[0])):.1f}\u00b0")

# \u7136\u540e\u8003\u8651: d435_link \u53ef\u80fd\u4e0d\u662f X\u524d Y\u5de6 Z\u4e0a!
# \u4e5f\u8bb8 d435_link \u7684 Z \u8f74\u671d\u524d(\u5149\u8f74), \u6216\u8005\u5b8c\u5168\u4e0d\u540c\u7684\u7ea6\u5b9a
#
# Unitree G1 URDF \u7684\u5750\u6807\u7ea6\u5b9a:
# \u901a\u5e38 URDF link \u4f7f\u7528 ROS \u6807\u51c6: X\u524d Y\u5de6 Z\u4e0a
# \u4f46 d435 sensor link \u6709\u65f6\u7528\u81ea\u5df1\u7684\u7ea6\u5b9a

# \u5b9e\u9645\u4e0a, \u6362\u4e00\u79cd\u601d\u8def:
# \u4ee3\u7801\u4e2d\u7684\u6a21\u578b\u5047\u8bbe\u76f8\u673a\u901a\u8fc7 torso_link \u504f\u79fb\u5b89\u88c5
# pitch/yaw/roll \u63cf\u8ff0\u7684\u662f\u76f8\u5bf9\u4e8e body \u5750\u6807\u7cfb\u7684\u65cb\u8f6c
#
# URDF \u4e2d\u7684 d435 joint \u4e5f\u662f\u4ece torso_link \u51fa\u53d1
# \u4f46 URDF \u7684\u65cb\u8f6c\u63cf\u8ff0\u7684\u662f d435_link \u5750\u6807\u7cfb\u76f8\u5bf9 torso \u7684\u65cb\u8f6c
#
# \u5982\u679c\u4e24\u8005\u63cf\u8ff0\u540c\u4e00\u4ef6\u4e8b\uff08\u76f8\u673a\u5b89\u88c5\u65b9\u5411\uff09\uff0c
# \u90a3\u4e48\u9700\u8981\u4e00\u4e2a R_link_to_optical \u53d8\u6362\u6765\u8fde\u63a5\uff1a
# R_torso_to_optical = R_link_to_optical @ R_urdf
# = R_body_to_cam @ R_combo  (\u4ee3\u7801\u6a21\u578b)
#
# \u6240\u4ee5: R_combo = R_body_to_cam^(-1) @ R_link_to_optical @ R_urdf

# \u5173\u952e: R_link_to_optical \u53d6\u51b3\u4e8e d435_link \u7684\u5750\u6807\u7cfb\u5b9a\u4e49
# \u5982\u679c d435_link = body \u7ea6\u5b9a (X\u524d Y\u5de6 Z\u4e0a):
#   R_link_to_optical = R_body_to_cam
#   R_combo = R_body_to_cam^(-1) @ R_body_to_cam @ R_urdf = R_urdf
#   -> code pitch = +47.6\u00b0 (\u5149\u8f74\u671d\u4e0a, \u4e0d\u5bf9!)

# \u5982\u679c d435_link \u7684\u5750\u6807\u7cfb\u4e0d\u540c:
#   \u9700\u8981\u627e\u5230\u6b63\u786e\u7684 R_link_to_optical

print("\n" + "=" * 70)
print("\u7b2c\u516d\u6b65\uff1a\u901a\u8fc7\u6392\u9664\u6cd5\u786e\u5b9a d435_link \u5750\u6807\u7cfb")
print("=" * 70)

# \u5df2\u77e5\u6807\u5b9a\u7ed3\u679c pitch=-61.6\u00b0 \u662f\u6b63\u786e\u7684(IoU=0.897)
# URDF rpy=(0, 0.831, 0)
# \u5dee\u5f02\u7ea6 14\u00b0, \u4e0d\u662f\u7cbe\u786e\u5339\u914d

# \u4f46\u5982\u679c\u8003\u8651: URDF \u7ea6\u5b9a\u53ef\u80fd\u662f\u53e6\u4e00\u79cd\u65cb\u8f6c\u65b9\u5411
# \u5728\u6709\u4e9b URDF \u4e2d, rpy \u53ef\u80fd\u4f7f\u7528 ZYX intrinsic (\u4e0d\u662f XYZ fixed)
# \u6807\u51c6 URDF rpy = fixed XYZ = intrinsic ZYX

# \u8ba9\u6211\u5c1d\u8bd5\u6240\u6709\u53ef\u80fd\u7684\u7ea6\u5b9a:
print("\u5c1d\u8bd5\u4e0d\u540c\u7684 d435_link \u5750\u6807\u7cfb\u7ea6\u5b9a:")
print(f"  \u5149\u8f74\u65b9\u5411       | R_link_to_opt | \u5bf9\u5e94 code pitch")
print(f"  --------------|---------------|----------------")

# \u7ea6\u5b9a 1: d435 Z\u8f74=\u5149\u8f74 (\u5411\u524d)
# R_link_to_optical: d435(X?,Y?,Z\u524d) -> optical(X\u53f3,Y\u4e0b,Z\u524d)
# \u5982\u679c d435 \u662f X\u53f3 Y\u4e0b Z\u524d (ROS optical frame \u7ea6\u5b9a)
# \u5219 R_link_to_optical = I
R_l2o = np.eye(3)
R_combo_try = R_body_to_cam.T @ R_l2o @ R_urdf
opt_try = (R_body_to_cam @ R_combo_try).T @ np.array([0, 0, 1])
elev = np.degrees(np.arctan2(opt_try[2], opt_try[0]))
angles_try = Rotation.from_matrix(R_combo_try).as_euler('XZY', degrees=True)
print(f"  d435=optical    | I             | pitch={angles_try[2]:.1f}\u00b0 (\u5149\u8f74\u4ef0\u89d2={elev:.1f}\u00b0)")

# \u7ea6\u5b9a 2: d435 X\u524d Y\u5de6 Z\u4e0a (\u8ddf body \u4e00\u6837)
R_l2o = R_body_to_cam
R_combo_try = R_body_to_cam.T @ R_l2o @ R_urdf
opt_try = (R_body_to_cam @ R_combo_try).T @ np.array([0, 0, 1])
elev = np.degrees(np.arctan2(opt_try[2], opt_try[0]))
angles_try = Rotation.from_matrix(R_combo_try).as_euler('XZY', degrees=True)
print(f"  d435=body(XYZ)  | R_b2c         | pitch={angles_try[2]:.1f}\u00b0 (\u5149\u8f74\u4ef0\u89d2={elev:.1f}\u00b0)")

# \u7ea6\u5b9a 3: d435 Z\u524d X\u53f3 Y\u4e0b \u2192 optical
# \u5373 d435 frame = optical frame
# R_torso_to_optical = R_urdf (\u76f4\u63a5)
# = R_body_to_cam @ R_combo
# R_combo = R_body_to_cam^T @ R_urdf
R_combo_try = R_body_to_cam.T @ R_urdf
opt_try = (R_body_to_cam @ R_combo_try).T @ np.array([0, 0, 1])
elev = np.degrees(np.arctan2(opt_try[2], opt_try[0]))
angles_try = Rotation.from_matrix(R_combo_try).as_euler('XZY', degrees=True)
print(f"  d435=optical    | (direct)      | pitch={angles_try[2]:.1f}\u00b0 (\u5149\u8f74\u4ef0\u89d2={elev:.1f}\u00b0)")

# \u7ea6\u5b9a 4: d435 X\u524d Z\u4e0b Y\u53f3 (\u4e00\u79cd\u53ef\u80fd\u7684 camera link \u7ea6\u5b9a)
# R_link_to_optical: d435(X\u524d,Y\u53f3,Z\u4e0b) -> optical(X\u53f3,Y\u4e0b,Z\u524d)
# optical X = d435 Y -> col0 = [0,1,0]
# optical Y = d435 Z -> col1 = [0,0,1]  (\u7b49\u7b49 Z\u4e0b)
# \u8fd9\u592a\u591a\u53ef\u80fd\u6027\u4e86\uff0c\u8ba9\u6211\u7528\u53e6\u4e00\u79cd\u65b9\u6cd5

# \u66f4\u597d\u7684\u65b9\u6cd5\uff1a\u76f4\u63a5\u4ece\u7269\u7406\u7ea6\u675f\u63a8\u5bfc
# \u5df2\u77e5\u76f8\u673a\u671d\u4e0b\u7ea6 ~50-60\u00b0
# \u5c1d\u8bd5 pitch = -47.6\u00b0 \u76f4\u63a5\u505a\u6295\u5f71\u9a8c\u8bc1

print("\n" + "=" * 70)
print("\u7b2c\u4e03\u6b65\uff1a\u6295\u5f71\u9a8c\u8bc1 \u2014 \u591a\u7ec4\u53c2\u6570\u5bf9\u6bd4")
print("=" * 70)

# \u52a0\u8f7d\u6570\u636e
TASK = "G1_WBT_Inspire_Put_Clothes_into_Washing_Machine"
data_dir = os.path.join(DATASET_ROOT, TASK)
EP = 0
FRAME = 30

# \u68c0\u67e5\u6570\u636e\u662f\u5426\u5b58\u5728
if not os.path.exists(data_dir):
    # \u5c1d\u8bd5\u5176\u4ed6\u4efb\u52a1
    for t in ["G1_WBT_Brainco_Make_The_Bed", "G1_WBT_Inspire_Pickup_Pillow_MainCamOnly"]:
        alt = os.path.join(DATASET_ROOT, t)
        if os.path.exists(alt):
            TASK = t
            data_dir = alt
            break

print(f"\u4f7f\u7528\u6570\u636e: {TASK}, ep={EP}, frame={FRAME}")

# Load episode
meta = pd.read_parquet(os.path.join(data_dir, "meta", "episodes", "chunk-000", "file-000.parquet"))
ep_meta = meta[meta["episode_index"] == EP].iloc[0]
data_fi = int(ep_meta.get("data/file_index", 0))
df = pd.read_parquet(os.path.join(data_dir, "data", "chunk-000", f"file-{data_fi:03d}.parquet"))
ep_df = df[df["episode_index"] == EP].sort_values("frame_index")
row = ep_df[ep_df["frame_index"] == FRAME].iloc[0]

rq = np.array(row["observation.state.robot_q_current"], dtype=np.float64)
hs = np.array(row["observation.state.hand_state"], dtype=np.float64)

hand_type = get_hand_type(TASK)
q = build_q(model, rq, hs, hand_type=hand_type)
transforms = do_fk(model, data_pin, q)

# Extract video frame
import av
file_idx = int(ep_meta["videos/observation.images.head_stereo_left/file_index"])
from_ts = float(ep_meta["videos/observation.images.head_stereo_left/from_timestamp"])
video_path = os.path.join(data_dir, "videos", "observation.images.head_stereo_left",
                           "chunk-000", f"file-{file_idx:03d}.mp4")

container = av.open(video_path)
stream = container.streams.video[0]
target_ts = from_ts + FRAME / 30.0

img = None
for frame in container.decode(stream):
    pts_sec = float(frame.pts * stream.time_base)
    fi = int(round((pts_sec - from_ts) * 30))
    if fi >= FRAME:
        img = frame.to_ndarray(format='bgr24')
        break
container.close()

if img is None:
    print("\u65e0\u6cd5\u63d0\u53d6\u89c6\u9891\u5e27!")
    sys.exit(1)

h, w = img.shape[:2]
print(f"\u5e27\u5927\u5c0f: {w}x{h}")

# \u5173\u952e\u70b9\u5b9a\u4e49
ALL_KEYPOINTS = [
    ("L_wrist", "left_wrist_yaw_link",   np.array([ 0.0046,  0.0000,  0.0300])),
    ("L_thumb", "left_thumb_4",          np.array([-0.0314,  0.0150, -0.0101])),
    ("L_toe",   "left_ankle_roll_link",  np.array([ 0.1424,  0.0000, -0.0210])),
    ("R_toe",   "right_ankle_roll_link", np.array([ 0.1424, -0.0000, -0.0215])),
    ("R_thumb", "right_thumb_4",         np.array([ 0.0314,  0.0150, -0.0101])),
    ("R_wrist", "right_wrist_yaw_link",  np.array([ 0.0046,  0.0000,  0.0300])),
]

# \u8fc7\u6ee4\u6389\u4e0d\u53ef\u7528\u7684\u5173\u952e\u70b9 (BrainCo \u4efb\u52a1\u6ca1\u6709 thumb links)
available_kps = []
for name, link, offset in ALL_KEYPOINTS:
    if link in transforms:
        available_kps.append((name, link, offset))
    else:
        print(f"  \u8df3\u8fc7 {name} (link {link} \u4e0d\u5728 FK \u7ed3\u679c\u4e2d)")

# \u8ba1\u7b97\u5173\u952e\u70b9\u4e16\u754c\u5750\u6807
def get_kp_world(transforms, kps):
    pts = []
    for name, link, offset in kps:
        t, R = transforms[link]
        pts.append(R @ offset + t)
    return np.array(pts, dtype=np.float64)

kp_world = get_kp_world(transforms, available_kps)
kp_names = [k[0] for k in available_kps]

# \u6295\u5f71\u51fd\u6570
def project_keypoints(params, transforms, kp_world):
    K, D, rvec, tvec, _, _, fisheye = make_camera(params, transforms)
    pts3d = kp_world.reshape(-1, 1, 3).astype(np.float64)
    pts2d = project_points_cv(pts3d, rvec, tvec, K, D, fisheye)
    return pts2d.reshape(-1, 2)

# \u591a\u7ec4\u53c2\u6570\u5bf9\u6bd4
param_sets = {
    "\u6807\u5b9a\u7ed3\u679c (fx=291,fy=287)": dict(BEST_PARAMS),
    "URDF\u5916\u53c2 + \u6807\u5b9a\u5185\u53c2": {
        "dx": 0.0576235, "dy": 0.01753, "dz": 0.42987,
        "pitch": -47.6, "yaw": 0.0, "roll": 0.0,
        "fx": 290.78, "fy": 287.35, "cx": 329.37, "cy": 313.68,
    },
    "URDF\u5916\u53c2 + fx=fy=380": {
        "dx": 0.0576235, "dy": 0.01753, "dz": 0.42987,
        "pitch": -47.6, "yaw": 0.0, "roll": 0.0,
        "fx": 380.0, "fy": 380.0, "cx": 320.0, "cy": 240.0,
    },
    "URDF\u5916\u53c2 + fx=fy=420": {
        "dx": 0.0576235, "dy": 0.01753, "dz": 0.42987,
        "pitch": -47.6, "yaw": 0.0, "roll": 0.0,
        "fx": 420.0, "fy": 420.0, "cx": 320.0, "cy": 240.0,
    },
    "\u6807\u5b9a\u5916\u53c2 + fx=fy=380": {
        "dx": 0.0758, "dy": 0.0226, "dz": 0.4484,
        "pitch": -61.5855, "yaw": 2.1690, "roll": 0.2331,
        "fx": 380.0, "fy": 380.0, "cx": 329.37, "cy": 313.68,
    },
    "\u6807\u5b9a\u5916\u53c2 + fx=fy=420": {
        "dx": 0.0758, "dy": 0.0226, "dz": 0.4484,
        "pitch": -61.5855, "yaw": 2.1690, "roll": 0.2331,
        "fx": 420.0, "fy": 420.0, "cx": 329.37, "cy": 313.68,
    },
}

print(f"\n\u5173\u952e\u70b9\u4e16\u754c\u5750\u6807:")
for i, (name, link, offset) in enumerate(available_kps):
    print(f"  {name}: {kp_world[i]} (link={link})")

# \u7ed8\u5236\u5bf9\u6bd4\u56fe
out_dir = os.path.join(TMP_DIR, "urdf_verify")
os.makedirs(out_dir, exist_ok=True)

colors = [
    (0, 0, 255),     # \u7ea2
    (0, 255, 0),     # \u7eff
    (255, 0, 0),     # \u84dd
    (0, 255, 255),   # \u9ec4
    (255, 0, 255),   # \u54c1\u7ea2
    (255, 255, 0),   # \u9752
]

print(f"\n\u5404\u53c2\u6570\u7ec4\u5173\u952e\u70b9\u6295\u5f71:")
for pname, params in param_sets.items():
    pts2d = project_keypoints(params, transforms, kp_world)
    print(f"\n  {pname}:")
    in_frame = 0
    for i, (kp_name, _) in enumerate(zip(kp_names, pts2d)):
        x, y = pts2d[i]
        inside = 0 <= x <= w and 0 <= y <= h
        if inside:
            in_frame += 1
        status = "\u2713" if inside else "\u2717 (out of frame)"
        print(f"    {kp_name}: ({x:.1f}, {y:.1f}) {status}")
    print(f"    \u5e27\u5185\u5173\u952e\u70b9: {in_frame}/{len(kp_names)}")

# \u7ed8\u5236\u7efc\u5408\u5bf9\u6bd4\u56fe
canvas = img.copy()
legend_y = 30
for pi, (pname, params) in enumerate(param_sets.items()):
    color = colors[pi % len(colors)]
    pts2d = project_keypoints(params, transforms, kp_world)

    for i, kp_name in enumerate(kp_names):
        x, y = int(pts2d[i, 0]), int(pts2d[i, 1])
        if 0 <= x <= w + 200 and 0 <= y <= h + 200:
            cv2.circle(canvas, (x, y), 6, color, 2)
            cv2.putText(canvas, kp_name, (x + 8, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

    cv2.putText(canvas, pname, (10, legend_y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    legend_y += 18

out_path = os.path.join(out_dir, "kp_comparison.png")
cv2.imwrite(out_path, canvas)
print(f"\n\u5bf9\u6bd4\u56fe\u5df2\u4fdd\u5b58: {out_path}")

# \u4e5f\u4e3a\u6bcf\u7ec4\u53c2\u6570\u751f\u6210\u5355\u72ec\u7684 overlay \u56fe
skip_set = get_skip_meshes(hand_type)
link_meshes = parse_urdf_meshes(G1_URDF)
mesh_cache = preload_meshes(link_meshes, MESH_DIR, skip_set=skip_set)

for pi, (pname, params) in enumerate(param_sets.items()):
    overlay = img.copy()

    # \u7ed8\u5236 mesh overlay
    model_cfg = get_model(CAMERA_MODEL)
    K = build_K(params, model_cfg)
    D = build_D(params, model_cfg)
    fisheye = model_is_fisheye(model_cfg)

    _const = make_camera_const(params)
    K_c, D_c, rvec, tvec, R_w2c, t_w2c, _fish = make_camera(params, transforms, _const)
    t_flat = t_w2c.flatten()

    # Render mesh overlay
    mesh_overlay = np.zeros_like(img)
    for link_name, (tris, unique_verts) in mesh_cache.items():
        if link_name not in transforms or len(unique_verts) == 0:
            continue
        t_link, R_link = transforms[link_name]
        verts3d = ((R_link @ unique_verts.T).T + t_link).astype(np.float64)
        depths = (R_w2c @ verts3d.T).T + t_flat
        in_front = depths[:, 2] > 0.01
        if np.count_nonzero(in_front) < 3:
            continue
        pts2d = project_points_cv(
            verts3d[in_front].reshape(-1, 1, 3), rvec, tvec, K_c, D_c, _fish)
        pts2d = pts2d.reshape(-1, 2)
        finite = np.all(np.isfinite(pts2d), axis=1)
        pts2d = pts2d[finite]
        if len(pts2d) < 3:
            continue
        hull = cv2.convexHull(pts2d.astype(np.float32))
        color = (0, 255, 180) if "left" not in link_name and "right" not in link_name else \
                (255, 180, 0) if "left" in link_name else (0, 180, 255)
        cv2.fillConvexPoly(mesh_overlay, hull.astype(np.int32), color)

    cv2.addWeighted(mesh_overlay, 0.35, overlay, 1.0, 0, overlay)

    # Add keypoints
    pts2d = project_keypoints(params, transforms, kp_world)
    for i, kp_name in enumerate(kp_names):
        x, y = int(pts2d[i, 0]), int(pts2d[i, 1])
        if -50 <= x <= w + 50 and -50 <= y <= h + 50:
            cv2.circle(overlay, (x, y), 8, (0, 0, 255), 2)
            cv2.putText(overlay, kp_name, (x + 10, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    cv2.putText(overlay, pname, (10, 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    safe_name = pname.replace(" ", "_").replace("(", "").replace(")", "").replace(",", "").replace("=", "")
    out_path = os.path.join(out_dir, f"overlay_{pi}_{safe_name}.png")
    cv2.imwrite(out_path, overlay)
    print(f"  \u4fdd\u5b58: {out_path}")

# ============================================================
# \u6700\u7ec8\u603b\u7ed3
# ============================================================
print("\n" + "=" * 70)
print("\u6700\u7ec8\u603b\u7ed3")
print("=" * 70)

print(f"""
\u6362\u7b97\u7ed3\u679c:

URDF d435 \u53c2\u6570:
  xyz = (0.0576235, 0.01753, 0.42987)
  rpy = (0, 0.8308, 0)

\u2192 \u4ee3\u7801\u53c2\u6570 (\u5047\u8bbe d435_link \u5750\u6807\u7cfb = body \u7ea6\u5b9a):
  dx    = 0.0576235
  dy    = 0.01753
  dz    = 0.42987
  pitch = -47.6\u00b0  (URDF pitch \u53d6\u8d1f, \u56e0\u4e3a\u4ee3\u7801\u4e2d pitch>0 \u671d\u4e0a, URDF\u7684\u65cb\u8f6c\u4f7f\u5149\u8f74\u671d\u4e0b)
  yaw   = 0.0\u00b0
  roll  = 0.0\u00b0

  \u7b49\u4e00\u4e0b! \u4e0a\u9762\u7684\u63a8\u5bfc\u6709 bug. \u8ba9\u6211\u6700\u7ec8\u786e\u8ba4\u4e00\u6b21.
""")

# \u6700\u7ec8\u4e25\u683c\u63a8\u5bfc:
# \u4ee3\u7801: R_torso_to_optical = R_body_to_cam @ Rx(roll) @ Rz(yaw) @ Ry(pitch)
# URDF: R_torso_to_d435_link = Ry_urdf(0.831)
# \u5047\u8bbe d435_link = body \u7ea6\u5b9a: R_d435_link_to_optical = R_body_to_cam
# \u603b: R_torso_to_optical = R_body_to_cam @ R_urdf
# \u6240\u4ee5: R_body_to_cam @ R_combo = R_body_to_cam @ R_urdf
# \u5373: R_combo = R_urdf = Ry(+47.6\u00b0)
# \u4ee3\u7801 pitch = +47.6\u00b0
# \u5149\u8f74 = R_combo^T @ [1,0,0] = Ry(-47.6\u00b0) @ [1,0,0] = [cos47.6, 0, sin(-47.6)]
# \u7b49\u7b49\u4e0d\u5bf9: Ry(\u03b8)^T = Ry(-\u03b8)
# Ry(-47.6\u00b0) @ [1,0,0] = [cos(-47.6\u00b0), 0, -sin(-47.6\u00b0)] = [cos47.6, 0, sin47.6]
# = [0.674, 0, +0.739] -> \u671d\u524d\u4e0a\u65b9

# \u55ef\u6240\u4ee5 d435_link \u4e0d\u662f body \u7ea6\u5b9a(X\u524d Y\u5de6 Z\u4e0a)

# \u5c1d\u8bd5: d435_link = (X\u524d, Y\u53f3, Z\u4e0b) \u2014 \u4e00\u79cd\u5e38\u89c1\u7684\u76f8\u673a link \u7ea6\u5b9a
# R_body_to_d435: body(X\u524d Y\u5de6 Z\u4e0a) -> d435(X\u524d Y\u53f3 Z\u4e0b)
# Y: \u5de6 -> \u53f3 (\u53cd), Z: \u4e0a -> \u4e0b (\u53cd)
# R = diag(1, -1, -1)
R_body_to_d435 = np.diag([1.0, -1.0, -1.0])
# R_d435_to_optical: d435(X\u524d Y\u53f3 Z\u4e0b) -> optical(X\u53f3 Y\u4e0b Z\u524d)
# optical X = d435 Y (\u53f3), optical Y = d435 Z (\u4e0b... \u7b49\u7b49, d435 Z=\u4e0b, optical Y=\u4e0b, \u6240\u4ee5\u53d6\u8d1f?)
# No: d435 Z = \u4e0b, optical Y = \u4e0b \u2192 optical Y = d435(-Z)?
# d435(X\u524d Y\u53f3 Z\u4e0b): X\u2192\u524d, Y\u2192\u53f3, Z\u2192\u4e0b
# optical(X\u53f3 Y\u4e0b Z\u524d): X\u2192\u53f3=d435.Y, Y\u2192\u4e0b=d435.Z, Z\u2192\u524d=d435.X
R_d435_to_optical = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=np.float64)

# R_torso_to_optical = R_d435_to_optical @ R_urdf @ R_body_to_d435...
# \u4e0d\u4e0d\u4e0d, R_urdf = R_torso_to_d435_link
# R_torso_to_optical = R_d435_to_optical @ R_urdf

# \u4f46\u8fd9\u5047\u8bbe d435_link \u672c\u8eab\u7684 identity \u59ff\u6001\u5c31\u662f X\u524d Y\u53f3 Z\u4e0b
# \u800c R_urdf \u628a torso frame \u7684\u70b9\u8f6c\u5230 d435_link frame
# \u4e0d\uff0cR_urdf \u7684\u542b\u4e49\u662f: d435_link frame axes expressed in torso frame
# \u5373 d435_link \u5750\u6807\u7cfb\u5728 torso \u5750\u6807\u7cfb\u4e2d\u7684\u8868\u793a

# \u8ba9\u6211\u66f4\u7b80\u6d01\u5730\u601d\u8003\uff1a
# URDF joint transform \u7684\u542b\u4e49:
#   \u4e00\u4e2a\u70b9\u5728 child frame \u4e2d\u7684\u5750\u6807 p_child
#   \u5728 parent frame \u4e2d\u662f: p_parent = R_urdf @ p_child + t_urdf
#
# \u6240\u4ee5 R_urdf \u628a d435_link \u4e2d\u7684\u5750\u6807\u8f6c\u5230 torso \u4e2d
# d435_link \u7684\u5149\u8f74 z_optical \u5728 d435_link frame \u4e2d\u7684\u65b9\u5411\u53d6\u51b3\u4e8e\u7ea6\u5b9a
#
# \u5047\u8bbe d435_link frame \u4e2d\u5149\u8f74 = [1, 0, 0] (X\u8f74\u524d):
#   \u5149\u8f74 in torso = R_urdf @ [1,0,0] = [cos(0.831), 0, -sin(0.831)]
#   = [0.674, 0, -0.739] \u2192 \u671d\u524d\u4e0b\u65b9 47.6\u00b0 \u2190 \u5408\u7406!

print("\u5c1d\u8bd5\u4e0d\u540c\u7684 d435_link \u5149\u8f74\u65b9\u5411\u5047\u8bbe:")
for axis_name, axis in [("X", [1,0,0]), ("Y", [0,1,0]), ("Z", [0,0,1]),
                         ("-X", [-1,0,0]), ("-Y", [0,-1,0]), ("-Z", [0,0,-1])]:
    optical_in_torso = R_urdf @ np.array(axis, dtype=float)
    elev = np.degrees(np.arctan2(optical_in_torso[2], optical_in_torso[0]))
    print(f"  d435 {axis_name:3s} in torso = [{optical_in_torso[0]:+.3f}, {optical_in_torso[1]:+.3f}, {optical_in_torso[2]:+.3f}]  \u4ef0\u89d2 = {elev:+.1f}\u00b0")

print(f"\n\u6807\u5b9a\u7684\u5149\u8f74\u4ef0\u89d2: {elev_cal:.1f}\u00b0")
print(f"\n\u53ea\u6709 d435 X\u8f74 \u4f5c\u4e3a\u5149\u8f74\u65f6, \u4ef0\u89d2 = -47.6\u00b0, \u6700\u63a5\u8fd1\u6807\u5b9a\u7684 -61.5\u00b0")
print(f"\u5dee\u5f02\u7ea6 14\u00b0")

# ============================================================
# \u91cd\u8981\u6d1e\u5bdf: \u5dee\u5f02\u6765\u6e90
# ============================================================
print("\n" + "=" * 70)
print("\u7b2c\u516b\u6b65\uff1a\u5206\u6790 14\u00b0 \u5dee\u5f02\u7684\u6765\u6e90")
print("=" * 70)

print(f"""
URDF \u7ed9\u51fa pitch \u2248 -47.6\u00b0 (\u5982\u679c d435 X=\u5149\u8f74)
\u6807\u5b9a\u7ed9\u51fa pitch \u2248 -61.6\u00b0
\u5dee\u5f02 \u2248 14\u00b0

\u53ef\u80fd\u539f\u56e0:
1. d435_link \u5750\u6807\u7cfb\u7ea6\u5b9a\u4e0d\u662f X=\u5149\u8f74
2. URDF \u4e2d d435 \u4f4d\u59ff\u4e0d\u5b8c\u5168\u51c6\u786e(\u8bbe\u8ba1\u503c vs \u5b9e\u9645\u5b89\u88c5\u503c)
3. \u6807\u5b9a\u7684 pitch \u548c\u5185\u53c2\u4e4b\u95f4\u6709\u8026\u5408(\u4e0d\u540c pitch+\u5185\u53c2 \u53ef\u4ee5\u4ea7\u751f\u76f8\u4f3c\u6295\u5f71)

\u6700\u91cd\u8981\u7684: \u56fa\u5b9a URDF \u5916\u53c2, \u4f18\u5316\u5185\u53c2, \u770b\u80fd\u5426\u83b7\u5f97\u597d\u7684\u6295\u5f71!
""")

# \u505a\u4e00\u4e2a\u66f4\u7cbe\u786e\u7684\u8ba1\u7b97: \u5982\u679c d435 \u5149\u8f74 = d435_link X\u8f74
# \u90a3\u4e48 R_d435_to_optical \u662f\u4ec0\u4e48?
# d435_link: X=\u5149\u8f74(\u524d), Y=\u5de6, Z=\u4e0a (\u6807\u51c6 body \u7ea6\u5b9a)
# optical: X=\u53f3, Y=\u4e0b, Z=\u524d(\u5149\u8f74)
# R_d435_to_optical:
#   optical_X = d435 \u7684\u54ea\u4e2a\u8f74? \u53f3 \u2192 d435 -Y \u2192 [0, -1, 0]
#   optical_Y = d435 \u7684\u54ea\u4e2a\u8f74? \u4e0b \u2192 d435 -Z \u2192 [0, 0, -1]
#   optical_Z = d435 \u7684\u54ea\u4e2a\u8f74? \u524d \u2192 d435 X  \u2192 [1, 0, 0]
#   R_d435_to_optical = [[0, -1, 0], [0, 0, -1], [1, 0, 0]]
#   = R_body_to_cam !!
# \u6240\u4ee5 d435_link \u7528\u7684\u5c31\u662f\u6807\u51c6 body \u7ea6\u5b9a (X\u524d Y\u5de6 Z\u4e0a)
# \u800c R_d435_to_optical = R_body_to_cam

# \u4f46\u4e4b\u524d\u5df2\u7ecf\u5206\u6790\u8fc7: \u8fd9\u7ed9\u51fa pitch = +47.6\u00b0 (\u5149\u8f74\u671d\u4e0a)
# \u77db\u76fe\uff01

# \u554a! \u6211\u72af\u4e86\u4e00\u4e2a\u9519\u8bef\u3002\u8ba9\u6211\u91cd\u65b0\u63a8\u5bfc:
# R_urdf \u628a d435 frame \u7684\u5750\u6807\u8f6c\u5230 torso frame
# \u5373: p_torso = R_urdf @ p_d435 + t_urdf
#
# d435 \u5149\u8f74\u5728 d435 frame \u4e2d = [1,0,0] (\u5047\u8bbe X=\u5149\u8f74)
# \u5149\u8f74\u5728 torso frame = R_urdf @ [1,0,0] = [0.674, 0, -0.739] \u2192 \u671d\u524d\u4e0b 47.6\u00b0 \u2713
#
# \u4ee3\u7801\u4e2d:
# R_cam = R_body_to_cam @ R_combo
# R_combo \u7684\u542b\u4e49: \u76f8\u5f53\u4e8e R_torso_to_d435_mounting
# \u5373 R_combo \u63cf\u8ff0\u4e86 d435 \u5b89\u88c5\u65b9\u5411(\u5728 torso body \u5750\u6807\u7cfb\u4e2d\u7684\u65cb\u8f6c)
#
# \u5149\u8f74(body\u5750\u6807\u7cfb) = R_cam^T @ [0,0,1] = R_combo^T @ R_body_to_cam^T @ [0,0,1]
# = R_combo^T @ [1, 0, 0] (\u56e0\u4e3a R_body_to_cam^T @ [0,0,1] = [1,0,0])
#
# \u6211\u4eec\u60f3\u8981\u5149\u8f74(torso\u5750\u6807\u7cfb) = [0.674, 0, -0.739]
# \u6240\u4ee5 R_combo^T @ [1,0,0] = [0.674, 0, -0.739]
# R_combo \u7684\u7b2c\u4e00\u5217 = [0.674, 0, -0.739]

# \u5982\u679c R_combo = Ry(p) (\u53ea\u6709pitch):
# Ry(p) \u7684\u7b2c\u4e00\u5217 = [cos(p), 0, -sin(p)]
# cos(p) = 0.674, sin(p) = 0.739
# p = arctan2(0.739, 0.674) = 47.6\u00b0 ???

# \u4f46\u662f!!! \u68c0\u67e5 Ry(47.6\u00b0)^T @ [1,0,0]:
# Ry(47.6\u00b0) = [[cos, 0, sin], [0, 1, 0], [-sin, 0, cos]]
# = [[0.674, 0, 0.739], [0, 1, 0], [-0.739, 0, 0.674]]
# Ry(47.6\u00b0)^T @ [1,0,0] = [[0.674, 0, -0.739], [0, 1, 0], [0.739, 0, 0.674]] @ [1,0,0]
# = [0.674, 0, 0.739]
# \u5149\u8f74 = [0.674, 0, +0.739] \u2192 \u671d\u4e0a 47.6\u00b0! \u4e0d\u5bf9!

# \u6211\u9700\u8981 R_combo^T @ [1,0,0] = [0.674, 0, -0.739]
# \u5982\u679c R_combo = Ry(-47.6\u00b0):
# Ry(-47.6\u00b0) = [[cos, 0, -sin], [0, 1, 0], [sin, 0, cos]]
# = [[0.674, 0, -0.739], [0, 1, 0], [0.739, 0, 0.674]]
# Ry(-47.6\u00b0)^T @ [1,0,0] = [[0.674, 0, 0.739], [0, 1, 0], [-0.739, 0, 0.674]] @ [1,0,0]
# = [0.674, 0, -0.739] \u2190 \u6b63\u786e!

print("\u6700\u7ec8\u63a8\u5bfc:")
print(f"  R_combo^T @ [1,0,0] = \u5149\u8f74(torso)")
print(f"  \u9700\u8981\u5149\u8f74 = [0.674, 0, -0.739]")
print(f"  R_combo = Ry(p), Ry(p)^T = Ry(-p)")
print(f"  Ry(-p) @ [1,0,0] = [cos(p), 0, sin(p)]  (\u6ce8\u610f: \u8fd9\u662f Ry(-p)!)")
print(f"  \u4f46 Ry(p)^T @ [1,0,0] = [cos(p), 0, -sin(p)] (Ry^T \u7b2c\u4e00\u884c)")

# Let me be really careful:
# Ry(p) = [[cos(p), 0, sin(p)], [0, 1, 0], [-sin(p), 0, cos(p)]]
# Ry(p)^T = [[cos(p), 0, -sin(p)], [0, 1, 0], [sin(p), 0, cos(p)]]
#
# Ry(p)^T @ [1,0,0] = first column of Ry(p)^T^T = first column of Ry(p)
# No! Ry(p)^T @ [1,0,0] = first ROW of Ry(p)^T = first COLUMN of Ry(p)
#
# Actually: (Ry(p)^T @ v)[i] = sum_j Ry(p)^T[i,j] * v[j] = Ry(p)^T[i,0] * 1
# = Ry(p)[0,i]  (since transpose swaps indices)
# = i-th element of first row of Ry(p)
# First row of Ry(p) = [cos(p), 0, sin(p)]
# So Ry(p)^T @ [1,0,0] = [cos(p), 0, sin(p)]

# WRONG AGAIN. Let me just compute it:
for p_deg in [47.6, -47.6]:
    p = np.radians(p_deg)
    Ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
    result = Ry.T @ np.array([1, 0, 0])
    print(f"  Ry({p_deg:+.1f}\u00b0)^T @ [1,0,0] = {result}")

# \u6240\u4ee5:
# Ry(+47.6\u00b0)^T @ [1,0,0] = [0.674, 0, -0.739] \u2192 \u671d\u524d\u4e0b\u65b9 \u2190 \u8fd9\u662f\u5bf9\u7684!
# Ry(-47.6\u00b0)^T @ [1,0,0] = [0.674, 0, +0.739] \u2192 \u671d\u524d\u4e0a\u65b9

print(f"""
\u4fee\u6b63! Ry(+47.6\u00b0)^T @ [1,0,0] = [0.674, 0, -0.739] \u2192 \u671d\u524d\u4e0b\u65b9 47.6\u00b0

\u8fd9\u610f\u5473\u7740\u4ee3\u7801 pitch = +47.6\u00b0 \u786e\u5b9e\u5bf9\u5e94\u5149\u8f74\u671d\u4e0b!

\u4e4b\u524d\u6211\u7684\u624b\u52a8\u8ba1\u7b97\u6709\u8bef\u3002\u8ba9\u6211\u7528 compute_optical_axis \u9a8c\u8bc1:
""")

opt_47, _ = compute_optical_axis(47.6)
print(f"compute_optical_axis(47.6) = {opt_47}")
elev_47 = np.degrees(np.arctan2(opt_47[2], opt_47[0]))
print(f"\u4ef0\u89d2 = {elev_47:.1f}\u00b0")

opt_neg47, _ = compute_optical_axis(-47.6)
print(f"compute_optical_axis(-47.6) = {opt_neg47}")
elev_neg47 = np.degrees(np.arctan2(opt_neg47[2], opt_neg47[0]))
print(f"\u4ef0\u89d2 = {elev_neg47:.1f}\u00b0")

# \u54c8\uff01compute_optical_axis \u663e\u793a pitch=47.6 \u2192 \u4ef0\u89d2 +47.6\u00b0(\u671d\u4e0a)
# \u4f46\u6570\u5b66\u8ba1\u7b97\u663e\u793a Ry(47.6)^T @ [1,0,0] = [0.674, 0, -0.739](\u671d\u4e0b)
# \u5fc5\u987b\u6709\u4e00\u4e2a\u5730\u65b9\u9519\u4e86\uff01\u8ba9\u6211\u68c0\u67e5 compute_optical_axis

print(f"\n\u68c0\u67e5 compute_optical_axis \u5185\u90e8:")
p = np.radians(47.6)
R_p = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
R_cam = R_body_to_cam @ R_p
print(f"R_cam = R_body_to_cam @ Ry(47.6\u00b0) =")
print(R_cam)
opt = R_cam.T @ np.array([0, 0, 1])
print(f"R_cam^T @ [0,0,1] = {opt}")
# R_cam^T = Ry(47.6\u00b0)^T @ R_body_to_cam^T
# R_body_to_cam^T @ [0,0,1] = ?
rbtc_t = R_body_to_cam.T @ np.array([0, 0, 1])
print(f"R_body_to_cam^T @ [0,0,1] = {rbtc_t}")
# Should be [1, 0, 0]?
# R_body_to_cam = [[0,-1,0],[0,0,-1],[1,0,0]]
# R_body_to_cam^T = [[0,0,1],[-1,0,0],[0,-1,0]]
# R_body_to_cam^T @ [0,0,1] = [1, 0, 0]?
# [[0,0,1],[-1,0,0],[0,-1,0]] @ [0,0,1] = [1, 0, 0] \u2713

ry_t_result = R_p.T @ rbtc_t
print(f"Ry(47.6\u00b0)^T @ [1,0,0] = {ry_t_result}")

# \u55ef! Ry(47.6\u00b0)^T @ [1,0,0]
# Ry(47.6\u00b0) = [[0.674, 0, 0.739], [0, 1, 0], [-0.739, 0, 0.674]]
# Ry(47.6\u00b0)^T = [[0.674, 0, -0.739], [0, 1, 0], [0.739, 0, 0.674]]
# Ry^T[row0] = [0.674, 0, -0.739]
# Ry^T[row1] = [0, 1, 0]
# Ry^T[row2] = [0.739, 0, 0.674]
# Ry^T @ [1,0,0] = [0.674, 0, 0.739]

# \u7b49\u7b49 Ry^T @ v:
# result[0] = Ry^T[0,0]*1 = 0.674
# result[1] = Ry^T[1,0]*1 = 0
# result[2] = Ry^T[2,0]*1 = 0.739
# = [0.674, 0, 0.739] \u671d\u524d\u4e0a\u65b9!

# \u4f46\u76f4\u63a5\u6253\u5370\u51fa\u6765 ry_t_result \u662f\u4ec0\u4e48?
# \u8fd9\u8ddf compute_optical_axis \u4e00\u81f4: pitch=47.6 \u2192 \u5149\u8f74\u671d\u4e0a

# \u6240\u4ee5\u4e4b\u524d\u6211\u7684\u201c\u624b\u52a8\u8ba1\u7b97\u201d\u6709\u8bef\uff08\u7b2c\u4e00\u5217 vs \u884c\u7684\u6df7\u6dc6\uff09
# \u6b63\u786e\u7684\u7ed3\u8bba: pitch = +47.6\u00b0 \u2192 \u5149\u8f74\u671d\u4e0a 47.6\u00b0 (\u4e0d\u5408\u7406)

# \u90a3\u5982\u679c d435_link \u7684\u5149\u8f74\u4e0d\u662f X \u8f74\u5462?
# \u4e4b\u524d\u6d4b\u8bd5\u8fc7: d435 X\u8f74 in torso = [0.674, 0, -0.739] \u671d\u524d\u4e0b\u65b9
# \u4f46\u4ee3\u7801\u4e2d pitch=+47.6\u00b0 \u7ed9\u51fa\u5149\u8f74\u671d\u4e0a

# \u77db\u76fe\u8bf4\u660e:
# URDF \u4e2d d435_link \u5750\u6807\u7cfb\u548c\u4ee3\u7801\u4e2d body \u5750\u6807\u7cfb\u7684**\u65cb\u8f6c\u7ea6\u5b9a\u4e0d\u540c**!
#
# \u5177\u4f53\u6765\u8bf4: d435_link \u7684\u5750\u6807\u7cfb\u53ef\u80fd\u8ddf body \u7ea6\u5b9a\u6709\u4e00\u4e2a 180\u00b0 \u7ffb\u8f6c
# \u6bd4\u5982 d435_link: X=\u540e(\u6216Z=\u4e0b)
# \u8fd9\u6837 R_urdf \u7684 pitch \u65cb\u8f6c\u65b9\u5411\u5c31\u53cd\u4e86

# \u6700\u76f4\u63a5\u7684\u9a8c\u8bc1: code pitch = -47.6\u00b0 \u662f\u5426\u5408\u7406?
print(f"\n" + "=" * 70)
print("\u6700\u7ec8\u786e\u8ba4: code pitch = -47.6\u00b0")
print("=" * 70)

opt_m47, _ = compute_optical_axis(-47.6)
print(f"pitch=-47.6\u00b0 \u5149\u8f74 = {opt_m47}")
elev_m47 = np.degrees(np.arctan2(opt_m47[2], opt_m47[0]))
print(f"\u4ef0\u89d2 = {elev_m47:.1f}\u00b0 (\u671d\u4e0b)")

# d435 X\u8f74\u5728torso\u4e2d = R_urdf @ [1,0,0] = [0.674, 0, -0.739]
# \u5982\u679c\u8fd9\u662f\u5149\u8f74\u65b9\u5411, \u4ef0\u89d2 = arctan2(-0.739, 0.674) = -47.6\u00b0
# \u4ee3\u7801\u4e2d pitch=-47.6\u00b0 \u7684\u5149\u8f74 = [0.674, 0, -0.739]
# \u5b8c\u5168\u543b\u5408!

print(f"\n\u9a8c\u8bc1:")
print(f"  d435 X\u8f74 in torso = {R_urdf @ [1,0,0]}")
print(f"  code pitch=-47.6\u00b0 \u5149\u8f74 = {opt_m47}")
print(f"  \u5b8c\u7f8e\u543b\u5408!")

print(f"""
\u7ed3\u8bba: URDF rpy=(0, 0.831, 0) \u5bf9\u5e94\u4ee3\u7801 pitch = -47.6\u00b0

\u539f\u56e0: URDF \u7684 Ry(+0.831) \u65cb\u8f6c\u4f7f d435_link \u7684 X \u8f74(\u5149\u8f74)\u6307\u5411\u524d\u4e0b\u65b9,
      \u4f46\u4ee3\u7801\u4e2d Ry(+47.6\u00b0) \u7684\u6548\u679c\u662f\u8ba9\u5149\u8f74\u6307\u5411\u524d\u4e0a\u65b9\u3002
      \u4e24\u8005\u7684\u65cb\u8f6c\u7ea6\u5b9a\u6070\u597d\u76f8\u53cd(\u4e00\u4e2a\u662f\u201c\u65cb\u8f6c\u5750\u6807\u7cfb\u201d, \u4e00\u4e2a\u662f\u201c\u65cb\u8f6c\u5411\u91cf\u201d)\u3002

      \u66f4\u7cbe\u786e\u5730\u8bf4:
      URDF: \u5149\u8f74 in torso = R_urdf @ [1,0,0] (\u65cb\u8f6c\u5750\u6807\u8f74)
      \u4ee3\u7801: \u5149\u8f74 in torso = Ry(p)^T @ [1,0,0] (\u65cb\u8f6c\u7684\u9006)

      R_urdf @ [1,0,0] = Ry(0.831) @ [1,0,0] = [cos, 0, -sin] = \u671d\u4e0b
      Ry(p)^T @ [1,0,0] = [cos(p), 0, sin(p)]
      \u8981\u4f7f sin(p) < 0 (\u671d\u4e0b), \u9700\u8981 p < 0
      cos(p)=cos(0.831), sin(p)=-sin(0.831) \u2192 p = -0.831 rad = -47.6\u00b0

\u6700\u7ec8 URDF \u5916\u53c2 \u2192 \u4ee3\u7801\u53c2\u6570:
  dx    = 0.0576235
  dy    = 0.01753
  dz    = 0.42987
  pitch = -47.6000\u00b0 (= -0.8308 rad = \u8d1f\u7684 URDF pitch)
  yaw   = 0.0\u00b0
  roll  = 0.0\u00b0
""")

# \u66f4\u65b0\u53c2\u6570\u96c6\u5e76\u91cd\u65b0\u6295\u5f71
param_sets_final = {
    "\u6807\u5b9a\u7ed3\u679c (pitch=-61.6)": dict(BEST_PARAMS),
    "URDF (pitch=-47.6, fx=291)": {
        "dx": 0.0576235, "dy": 0.01753, "dz": 0.42987,
        "pitch": -47.6, "yaw": 0.0, "roll": 0.0,
        "fx": 290.78, "fy": 287.35, "cx": 329.37, "cy": 313.68,
    },
    "URDF (pitch=-47.6, fx=fy=380)": {
        "dx": 0.0576235, "dy": 0.01753, "dz": 0.42987,
        "pitch": -47.6, "yaw": 0.0, "roll": 0.0,
        "fx": 380.0, "fy": 380.0, "cx": 320.0, "cy": 240.0,
    },
    "URDF (pitch=-47.6, fx=fy=420)": {
        "dx": 0.0576235, "dy": 0.01753, "dz": 0.42987,
        "pitch": -47.6, "yaw": 0.0, "roll": 0.0,
        "fx": 420.0, "fy": 420.0, "cx": 320.0, "cy": 240.0,
    },
    "\u6807\u5b9a\u5916\u53c2 + fx=fy=380": {
        "dx": 0.0758, "dy": 0.0226, "dz": 0.4484,
        "pitch": -61.5855, "yaw": 2.1690, "roll": 0.2331,
        "fx": 380.0, "fy": 380.0, "cx": 329.37, "cy": 313.68,
    },
}

print("\n" + "=" * 70)
print("\u6295\u5f71\u5bf9\u6bd4 (\u4fee\u6b63\u540e)")
print("=" * 70)

for pname, params in param_sets_final.items():
    pts2d = project_keypoints(params, transforms, kp_world)
    print(f"\n  {pname}:")
    for i, kp_name in enumerate(kp_names):
        x, y = pts2d[i]
        inside = 0 <= x <= w and 0 <= y <= h
        status = "\u2713" if inside else "\u2717"
        print(f"    {kp_name}: ({x:.1f}, {y:.1f}) {status}")

# \u7ed8\u5236\u6700\u7ec8\u5bf9\u6bd4\u56fe
canvas = img.copy()
legend_y = 30
for pi, (pname, params) in enumerate(param_sets_final.items()):
    color = colors[pi % len(colors)]
    pts2d = project_keypoints(params, transforms, kp_world)
    for i, kp_name in enumerate(kp_names):
        x, y = int(pts2d[i, 0]), int(pts2d[i, 1])
        if -100 <= x <= w + 100 and -100 <= y <= h + 100:
            cv2.circle(canvas, (x, y), 5, color, 2)
            if pi == 0:  # \u53ea\u6807\u6ce8\u7b2c\u4e00\u7ec4
                cv2.putText(canvas, kp_name, (x + 8, y - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
    cv2.putText(canvas, f"[{pi}] {pname}", (10, legend_y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
    legend_y += 16

out_path = os.path.join(out_dir, "kp_comparison_final.png")
cv2.imwrite(out_path, canvas)
print(f"\n\u6700\u7ec8\u5bf9\u6bd4\u56fe: {out_path}")

# \u751f\u6210\u6bcf\u7ec4\u53c2\u6570\u7684\u72ec\u7acb overlay
for pi, (pname, params) in enumerate(param_sets_final.items()):
    overlay = img.copy()

    _const = make_camera_const(params)
    K_c, D_c, rvec, tvec, R_w2c, t_w2c, _fish = make_camera(params, transforms, _const)
    t_flat = t_w2c.flatten()

    mesh_overlay = np.zeros_like(img)
    for link_name, (tris, unique_verts) in mesh_cache.items():
        if link_name not in transforms or len(unique_verts) == 0:
            continue
        t_link, R_link = transforms[link_name]
        verts3d = ((R_link @ unique_verts.T).T + t_link).astype(np.float64)
        depths = (R_w2c @ verts3d.T).T + t_flat
        in_front = depths[:, 2] > 0.01
        if np.count_nonzero(in_front) < 3:
            continue
        pts2d_m = project_points_cv(
            verts3d[in_front].reshape(-1, 1, 3), rvec, tvec, K_c, D_c, _fish)
        pts2d_m = pts2d_m.reshape(-1, 2)
        finite = np.all(np.isfinite(pts2d_m), axis=1)
        pts2d_m = pts2d_m[finite]
        if len(pts2d_m) < 3:
            continue
        hull = cv2.convexHull(pts2d_m.astype(np.float32))
        clr = (0, 255, 180) if "left" not in link_name and "right" not in link_name else \
              (255, 180, 0) if "left" in link_name else (0, 180, 255)
        cv2.fillConvexPoly(mesh_overlay, hull.astype(np.int32), clr)

    cv2.addWeighted(mesh_overlay, 0.35, overlay, 1.0, 0, overlay)

    pts2d = project_keypoints(params, transforms, kp_world)
    for i, kp_name in enumerate(kp_names):
        x, y = int(pts2d[i, 0]), int(pts2d[i, 1])
        if -50 <= x <= w + 50 and -50 <= y <= h + 50:
            cv2.circle(overlay, (x, y), 8, (0, 0, 255), 2)
            cv2.putText(overlay, kp_name, (x + 10, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    cv2.putText(overlay, pname, (10, 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    safe_name = f"final_{pi}"
    out_path = os.path.join(out_dir, f"{safe_name}.png")
    cv2.imwrite(out_path, overlay)
    print(f"  {out_path}")

print("\n\u5b8c\u6210!")
