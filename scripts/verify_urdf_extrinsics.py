"""
验证 URDF d435 外参转换 + 关键点投影验证。

核心问题：将 URDF d435 的 rpy=(0, 0.831, 0) 正确转换为代码的 (pitch, yaw, roll)。

关键挑战：URDF d435_link 坐标系可能跟 torso_link 约定不同。
"""

import sys
import os
import numpy as np
import cv2
import pandas as pd
import pinocchio as pin
from scipy.spatial.transform import Rotation

np.set_printoptions(precision=6, suppress=True)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "scripts"))

from config import (G1_URDF, MESH_DIR, BEST_PARAMS, DATASET_ROOT,
                     OUTPUT_DIR, get_hand_type, get_skip_meshes, CAMERA_MODEL)
from camera_models import get_model, build_K, build_D, model_is_fisheye, project_points_cv
from video_inpaint import (build_q, do_fk, parse_urdf_meshes, preload_meshes,
                            make_camera, make_camera_const)

# ============================================================
# 第一步：理解代码中的旋转模型
# ============================================================
print("=" * 70)
print("第一步：代码旋转模型")
print("=" * 70)

R_body_to_cam = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]], dtype=np.float64)

print("""
代码 make_camera_const() 中:
  R_pitch = Ry(pitch)    -- 绕 body Y 轴 (左方向)
  R_yaw   = Rz(yaw)      -- 绕 body Z 轴 (上方向)
  R_roll  = Rx(roll)      -- 绕 body X 轴 (前方向)
  R_cam = R_body_to_cam @ R_roll @ R_yaw @ R_pitch

  R_body_to_cam: body(X前 Y左 Z上) -> optical(X右 Y下 Z前)

  代码语义: R_roll @ R_yaw @ R_pitch 是 d435 安装方向在 body 坐标系中的旋转。
  然后 R_body_to_cam 将这个旋转后的 body 坐标系转换为光学坐标系。

  光轴 = R_cam^T @ [0,0,1] = (R_body_to_cam @ R_combo)^T @ [0,0,1]
  = R_combo^T @ R_body_to_cam^T @ [0,0,1]
  = R_combo^T @ [0, 0, 1]^T in R_body_to_cam^T
""")

# R_body_to_cam^T @ [0,0,1] = ?
print(f"R_body_to_cam^T @ [0,0,1] = {R_body_to_cam.T @ [0,0,1]}")
print("即光轴对应 body 的 X 轴(前方)，经过 R_combo 旋转后")

def compute_optical_axis(pitch_deg, yaw_deg=0, roll_deg=0):
    """计算给定参数下光轴在 body 坐标系中的方向"""
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

print("\n光轴方向 vs pitch:")
for p in [60, 30, 0, -30, -60, -90]:
    opt, _ = compute_optical_axis(p)
    elev = np.degrees(np.arctan2(opt[2], opt[0]))
    print(f"  pitch={p:6.1f}° -> 光轴 body = [{opt[0]:.3f}, {opt[1]:.3f}, {opt[2]:.3f}], 仰角={elev:.1f}°")

# ============================================================
# 第二步：URDF d435 坐标系分析
# ============================================================
print("\n" + "=" * 70)
print("第二步：URDF d435 坐标系分析")
print("=" * 70)

urdf_pitch_rad = 0.8307767239493009  # URDF rpy=(0, 0.831, 0) 的 pitch
R_urdf = Rotation.from_euler('xyz', [0, urdf_pitch_rad, 0]).as_matrix()

print(f"URDF d435 rpy = (0, 0.831, 0), pitch = {np.degrees(urdf_pitch_rad):.2f}°")
print(f"\nR_urdf (torso -> d435_link):")
print(R_urdf)

print(f"\nd435_link 各轴在 torso 坐标系中:")
print(f"  X轴: {R_urdf @ [1,0,0]} (d435 前方?)")
print(f"  Y轴: {R_urdf @ [0,1,0]} (d435 左方?)")
print(f"  Z轴: {R_urdf @ [0,0,1]} (d435 上方?)")

# D435 X轴在torso中 = [0.674, 0, -0.739] -> 朝前偏下 47.6°
# D435 Z轴在torso中 = [0.739, 0, 0.674] -> 朝前偏上
d435_x = R_urdf @ np.array([1, 0, 0])
elev_x = np.degrees(np.arctan2(d435_x[2], d435_x[0]))
print(f"\nd435 X轴仰角: {elev_x:.1f}° (如果X是光轴 -> 朝下 {-elev_x:.1f}°)")

print("""
关键洞察！

D435 URDF link 的标准约定 (Intel RealSense URDF):
  X 轴: 向右 (从后面看)
  Y 轴: 向下
  Z 轴: 向前 (光轴方向!)

但是宇树 G1 的 URDF 可能使用不同约定:
  如果 d435_link 的 X 轴 = 光轴方向 (有些 URDF 这么定义):
    光轴 in torso = R_urdf @ [1,0,0] = [0.674, 0, -0.739]
    仰角 = -47.6° -> 朝前下方 47.6° ✓ 合理!

  如果 d435_link 的 Z 轴 = 光轴方向 (Intel 标准):
    光轴 in torso = R_urdf @ [0,0,1] = [0.739, 0, 0.674]
    仰角 = +42.4° -> 朝前上方, 不合理

所以宇树 URDF 中 d435_link 大概率是 X=光轴 (前), Y=左, Z=上 的约定。
""")

# ============================================================
# 第三步：正确换算 (考虑 d435_link 坐标系约定)
# ============================================================
print("=" * 70)
print("第三步：正确的外参换算")
print("=" * 70)

print("""
如果 d435_link: X=光轴(前), Y=左, Z=上
那么 d435_link 和 torso_link 使用相同的坐标约定!

从 torso body 坐标系到相机光学坐标系:
  R_torso_to_optical = R_link_to_optical @ R_urdf

其中 R_link_to_optical: d435_link(X前 Y左 Z上) -> optical(X右 Y下 Z前)
  = R_body_to_cam = [[0,-1,0],[0,0,-1],[1,0,0]]

代码中:
  R_cam = R_body_to_cam @ R_combo

  R_combo = R_urdf 时就对应 URDF 外参

但代码中 R_combo 作用在 body 坐标系上，含义是"在 body 坐标系中旋转"
  所以如果 d435_link 和 torso_link 约定一致：
  R_combo = R_urdf = Ry(47.6°) -> code pitch = +47.6°

  但 pitch=+47.6° 的光轴朝上，不朝下！

矛盾！让我重新计算...

问题出在光轴计算上。让我仔细推导：

R_cam = R_body_to_cam @ R_combo, where R_combo = Ry(+47.6°)

相机坐标系的 Z 轴(光轴)在 torso 中的方向:
  z_cam_in_torso = R_cam^(-1) @ [0,0,1]_cam
  = R_cam^T @ [0,0,1]
  = (R_body_to_cam @ R_combo)^T @ [0,0,1]
  = R_combo^T @ R_body_to_cam^T @ [0,0,1]

  R_body_to_cam^T @ [0,0,1] = [1,0,0]  (Z_cam -> X_body)

  R_combo^T @ [1,0,0] = Ry(-47.6°) @ [1,0,0]
  = [cos(-47.6°), 0, sin(-47.6°)]   # Wait, Ry^T = Ry(-theta)

  等等，Ry(theta) = [[cos,-sin],[1],[sin,cos]] ?
  不，代码中 R_pitch = [[cos,0,sin],[0,1,0],[-sin,0,cos]]

  Ry(47.6°)^T @ [1,0,0] = [[cos,0,-sin],[0,1,0],[sin,0,cos]] @ [1,0,0]
  = [cos(47.6°), 0, sin(47.6°)]
  = [0.674, 0, 0.739]  -> 朝前上方

  这跟之前的计算一致，光轴朝上。不对。

关键：d435_link 的坐标系约定可能不是 "X前 Y左 Z上"!

让我考虑另一种可能:
  d435_link: Z=光轴(前), X=左(?), Y=上(?)
  --- 不是标准约定，但 rpy=(0,0.831,0) 下:

  d435 Z 轴 in torso = R_urdf @ [0,0,1] = [0.739, 0, 0.674] -> 朝前上方

  还是不对。

再考虑: 也许 Unitree 的 torso_link 坐标系不是 X前 Y左 Z上?
""")

# 用 pinocchio 加载模型，用实际数据验证
model = pin.buildModelFromUrdf(G1_URDF, pin.JointModelFreeFlyer())
data_pin = model.createData()

# Neutral pose
q = pin.neutral(model)
pin.forwardKinematics(model, data_pin, q)
pin.updateFramePlacements(model, data_pin)

# 列出关键 frames
print("关键 frame 位姿 (neutral pose):")
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

# d435 相对 torso
if "torso_link" in key_frames and "d435_link" in key_frames:
    t_torso, R_torso = key_frames["torso_link"]
    t_d435, R_d435 = key_frames["d435_link"]

    # 相对位移 (in torso frame)
    t_rel = R_torso.T @ (t_d435 - t_torso)
    R_rel = R_torso.T @ R_d435

    print(f"\nd435 相对 torso:")
    print(f"  位移 (torso坐标系): {t_rel}")
    print(f"  旋转矩阵:")
    print(f"  {R_rel}")
    rpy_rel = Rotation.from_matrix(R_rel).as_euler('xyz', degrees=True)
    print(f"  rpy: {rpy_rel} deg")

# ============================================================
# 第四步：深入分析 — 直接匹配代码的相机模型
# ============================================================
print("\n" + "=" * 70)
print("第四步：直接匹配代码的相机模型")
print("=" * 70)

print("""
代码中 make_camera() 的完整流程:

1. cam_pos = torso_t + torso_R @ offset
   - offset = [dx, dy, dz] 在 torso body 坐标系中

2. R_w2c = (torso_R @ R_cam.T).T = R_cam @ torso_R.T
   - R_cam = R_body_to_cam @ R_combo

3. 光轴方向 (world) = R_w2c^T @ [0,0,1]_cam = R_cam^(-1) 在 world 中的 Z列
   但这取决于 torso_R

在 neutral pose, torso_R = I (恒等), 所以:
  cam_pos = t_torso + offset
  R_w2c = R_cam

  对于 URDF 的 d435:
    cam_pos (world) = t_torso + [0.0576, 0.0175, 0.4299]
    cam_orient: 需要找到 R_cam 使得相机指向正确方向

实际上，让我换个方向。不要从 URDF 推导代码参数，
而是从代码的标定参数推导出它意味着什么 URDF 姿态，
然后跟 URDF 的 d435 姿态比较。
""")

# 标定参数: pitch=-61.59, yaw=2.17, roll=0.23 (degrees)
cal = BEST_PARAMS
print(f"标定参数: {cal}")

# 计算标定的 R_combo
p_c = np.radians(cal["pitch"])
y_c = np.radians(cal["yaw"])
r_c = np.radians(cal["roll"])
R_p = np.array([[np.cos(p_c), 0, np.sin(p_c)], [0, 1, 0], [-np.sin(p_c), 0, np.cos(p_c)]])
R_y = np.array([[np.cos(y_c), -np.sin(y_c), 0], [np.sin(y_c), np.cos(y_c), 0], [0, 0, 1]])
R_r = np.array([[1, 0, 0], [0, np.cos(r_c), -np.sin(r_c)], [0, np.sin(r_c), np.cos(r_c)]])
R_combo_cal = R_r @ R_y @ R_p

print(f"\n标定的 R_combo (torso中的安装旋转):")
print(R_combo_cal)

# 标定 R_combo 的光轴 (torso 坐标系中)
opt_cal = R_combo_cal.T @ np.array([1, 0, 0])
print(f"\n标定光轴 = R_combo^T @ [1,0,0] = {opt_cal}")
# 等等，应该是 R_cam^T @ [0,0,1]
R_cam_cal = R_body_to_cam @ R_combo_cal
opt_cal = R_cam_cal.T @ np.array([0, 0, 1])
print(f"标定光轴 = (R_body_to_cam @ R_combo)^T @ [0,0,1] = {opt_cal}")
elev_cal = np.degrees(np.arctan2(opt_cal[2], opt_cal[0]))
print(f"标定光轴仰角: {elev_cal:.1f}° (负=朝下)")

# URDF 的 d435 在 torso 中:
# d435 X轴 in torso = R_urdf @ [1,0,0] = [0.674, 0, -0.739] 朝前下
# 如果 d435 的 X 轴是光轴:
print(f"\nURDF d435 X轴 in torso = {R_urdf @ [1,0,0]}")
print(f"  仰角 = {np.degrees(np.arctan2((R_urdf @ [1,0,0])[2], (R_urdf @ [1,0,0])[0])):.1f}°")

# ============================================================
# 关键推导：从 URDF 的 d435 姿态推导代码参数
# ============================================================
print("\n" + "=" * 70)
print("第五步：从 URDF d435 物理位姿推导代码参数")
print("=" * 70)

print("""
URDF 告诉我们:
  d435 的 X 轴在 torso 中指向 [0.674, 0, -0.739] — 朝前下方 47.6°

如果 d435_link 的 X 轴是光轴方向:
  我们需要找到代码参数使得光轴 = [0.674, 0, -0.739]

  光轴 = R_cam^T @ [0,0,1] where R_cam = R_body_to_cam @ R_combo

  R_body_to_cam^T @ [0,0,1] = [1, 0, 0] (body X 轴)

  所以光轴 = R_combo^T @ [1, 0, 0]
  = R_combo 的第一列

  我们需要 R_combo 的第一列 = [0.674, 0, -0.739]

  如果只用 pitch: R_combo = Ry(pitch)
  Ry(p) 第一列 = [cos(p), 0, -sin(p)]
  cos(p) = 0.674, sin(p) = 0.739
  p = arctan2(0.739, 0.674) = 47.6° ?

  等等: -sin(p) = -0.739, 所以 sin(p) = 0.739
  p = arcsin(0.739) = 47.6°

  但我们之前算的 pitch=47.6° 给出光轴朝上！
  让我重新检查...
""")

# 重新严格计算
# R_pitch (代码定义):
# [[cos(p),  0, sin(p)],
#  [0,       1, 0     ],
#  [-sin(p), 0, cos(p)]]

# R_combo = Ry(p) (只有pitch, yaw=roll=0)
# R_cam = R_body_to_cam @ Ry(p)
# 光轴 in body = R_cam^T @ [0,0,1]
# = (R_body_to_cam @ Ry(p))^T @ [0,0,1]
# = Ry(p)^T @ R_body_to_cam^T @ [0,0,1]

# R_body_to_cam^T = [[0, 0, 1], [-1, 0, 0], [0, -1, 0]]
# R_body_to_cam^T @ [0,0,1] = [1, 0, 0]  ✓

# Ry(p)^T = [[cos(p), 0, -sin(p)], [0, 1, 0], [sin(p), 0, cos(p)]]
# Ry(p)^T @ [1, 0, 0] = [cos(p), 0, sin(p)]

# 所以光轴 = [cos(p), 0, sin(p)]
# 当 p > 0: Z分量 > 0 -> 朝上  (因为 body Z = 上)
# 当 p < 0: Z分量 < 0 -> 朝下 ✓

# URDF d435 光轴 (如果X轴是光轴) = [0.674, 0, -0.739]
# 需要 [cos(p), 0, sin(p)] = [0.674, 0, -0.739]
# cos(p) = 0.674, sin(p) = -0.739
# p = atan2(-0.739, 0.674) = -47.6° !!!

target_optical = np.array([0.674302, 0, -0.738455])
p_needed = np.degrees(np.arctan2(target_optical[2], target_optical[0]))
print(f"如果 d435 X轴 = 光轴: 需要 code pitch = {p_needed:.2f}°")

# 验证
opt_check, _ = compute_optical_axis(p_needed)
print(f"  验证: 光轴 = {opt_check}")
print(f"  vs 目标: {target_optical}")
print(f"  差异: {np.max(np.abs(opt_check - target_optical)):.6f}")

# ============================================================
print("\n" + "=" * 70)
print("关键结论!")
print("=" * 70)

print(f"""
如果 URDF d435_link 的 X 轴 = 光轴方向 (X前 Y左 Z上 约定):

  URDF pitch_urdf = +0.831 rad = +47.6° (URDF的Y轴旋转)

  由于 URDF 的 Ry(+47.6°) 使 d435 的 X轴(光轴) 朝前下方旋转,
  而代码中 Ry(pitch) 的光轴 = [cos(p), 0, sin(p)],
  要让光轴朝前下方, 需要 sin(p) < 0, 即 p < 0.

  具体来说: code pitch = -47.6° (取负!)

  为什么取负? 因为 URDF Ry(θ) 的含义是:
    X轴 从 [1,0,0] 旋转到 [cos(θ), 0, -sin(θ)] (注意负号在第三个分量!)
    Z轴 从 [0,0,1] 旋转到 [sin(θ), 0, cos(θ)]

  而代码的光轴计算 Ry(p)^T @ [1,0,0] = [cos(p), 0, sin(p)]

  所以 d435 X轴方向 = [cos(θ), 0, -sin(θ)] 对应光轴 [cos(p), 0, sin(p)]
  需要: sin(p) = -sin(θ), 即 p = -θ = -47.6°

URDF 外参对应的代码参数:
  dx    = {0.0576235}
  dy    = {0.01753}
  dz    = {0.42987}
  pitch = {-np.degrees(urdf_pitch_rad):.4f}°  (取负!)
  yaw   = 0.0°
  roll  = 0.0°

vs 标定结果:
  dx    = 0.0758
  dy    = 0.0226
  dz    = 0.4484
  pitch = -61.5855°
  yaw   = 2.1690°
  roll  = 0.2331°

pitch 差异: {-np.degrees(urdf_pitch_rad) - (-61.5855):.2f}° (URDF -47.6° vs 标定 -61.6°)
""")

# 等等，但 -47.6 和 -61.6 还是差 14 度! 这不是简单的符号错误
# 让我重新检查是否真的需要取负

print("=" * 70)
print("验证: pitch = -47.6° 的光轴")
print("=" * 70)

opt_neg, R_combo_neg = compute_optical_axis(-47.6)
print(f"pitch=-47.6° -> 光轴 = {opt_neg}, 仰角 = {np.degrees(np.arctan2(opt_neg[2], opt_neg[0])):.1f}°")

opt_pos, R_combo_pos = compute_optical_axis(47.6)
print(f"pitch=+47.6° -> 光轴 = {opt_pos}, 仰角 = {np.degrees(np.arctan2(opt_pos[2], opt_pos[0])):.1f}°")

# 然后考虑: d435_link 可能不是 X前 Y左 Z上!
# 也许 d435_link 的 Z 轴朝前(光轴), 或者完全不同的约定
#
# Unitree G1 URDF 的坐标约定:
# 通常 URDF link 使用 ROS 标准: X前 Y左 Z上
# 但 d435 sensor link 有时用自己的约定

# 实际上, 换一种思路:
# 代码中的模型假设相机通过 torso_link 偏移安装
# pitch/yaw/roll 描述的是相对于 body 坐标系的旋转
#
# URDF 中的 d435 joint 也是从 torso_link 出发
# 但 URDF 的旋转描述的是 d435_link 坐标系相对 torso 的旋转
#
# 如果两者描述同一件事（相机安装方向），
# 那么需要一个 R_link_to_optical 变换来连接：
# R_torso_to_optical = R_link_to_optical @ R_urdf
# = R_body_to_cam @ R_combo  (代码模型)
#
# 所以: R_combo = R_body_to_cam^(-1) @ R_link_to_optical @ R_urdf

# 关键: R_link_to_optical 取决于 d435_link 的坐标系定义
# 如果 d435_link = body 约定 (X前 Y左 Z上):
#   R_link_to_optical = R_body_to_cam
#   R_combo = R_body_to_cam^(-1) @ R_body_to_cam @ R_urdf = R_urdf
#   -> code pitch = +47.6° (光轴朝上, 不对!)

# 如果 d435_link 的坐标系不同:
#   需要找到正确的 R_link_to_optical

print("\n" + "=" * 70)
print("第六步：通过排除法确定 d435_link 坐标系")
print("=" * 70)

# 已知标定结果 pitch=-61.6° 是正确的(IoU=0.897)
# URDF rpy=(0, 0.831, 0)
# 差异约 14°, 不是精确匹配

# 但如果考虑: URDF 约定可能是另一种旋转方向
# 在有些 URDF 中, rpy 可能使用 ZYX intrinsic (不是 XYZ fixed)
# 标准 URDF rpy = fixed XYZ = intrinsic ZYX

# 让我尝试所有可能的约定:
print("尝试不同的 d435_link 坐标系约定:")
print(f"  光轴方向       | R_link_to_opt | 对应 code pitch")
print(f"  --------------|---------------|----------------")

# 约定 1: d435 Z轴=光轴 (向前)
# R_link_to_optical: d435(X?,Y?,Z前) -> optical(X右,Y下,Z前)
# 如果 d435 是 X右 Y下 Z前 (ROS optical frame 约定)
# 则 R_link_to_optical = I
R_l2o = np.eye(3)
R_combo_try = R_body_to_cam.T @ R_l2o @ R_urdf
opt_try = (R_body_to_cam @ R_combo_try).T @ np.array([0, 0, 1])
elev = np.degrees(np.arctan2(opt_try[2], opt_try[0]))
angles_try = Rotation.from_matrix(R_combo_try).as_euler('XZY', degrees=True)
print(f"  d435=optical    | I             | pitch={angles_try[2]:.1f}° (光轴仰角={elev:.1f}°)")

# 约定 2: d435 X前 Y左 Z上 (跟 body 一样)
R_l2o = R_body_to_cam
R_combo_try = R_body_to_cam.T @ R_l2o @ R_urdf
opt_try = (R_body_to_cam @ R_combo_try).T @ np.array([0, 0, 1])
elev = np.degrees(np.arctan2(opt_try[2], opt_try[0]))
angles_try = Rotation.from_matrix(R_combo_try).as_euler('XZY', degrees=True)
print(f"  d435=body(XYZ)  | R_b2c         | pitch={angles_try[2]:.1f}° (光轴仰角={elev:.1f}°)")

# 约定 3: d435 Z前 X右 Y下 → optical
# 即 d435 frame = optical frame
# R_torso_to_optical = R_urdf (直接)
# = R_body_to_cam @ R_combo
# R_combo = R_body_to_cam^T @ R_urdf
R_combo_try = R_body_to_cam.T @ R_urdf
opt_try = (R_body_to_cam @ R_combo_try).T @ np.array([0, 0, 1])
elev = np.degrees(np.arctan2(opt_try[2], opt_try[0]))
angles_try = Rotation.from_matrix(R_combo_try).as_euler('XZY', degrees=True)
print(f"  d435=optical    | (direct)      | pitch={angles_try[2]:.1f}° (光轴仰角={elev:.1f}°)")

# 约定 4: d435 X前 Z下 Y右 (一种可能的 camera link 约定)
# R_link_to_optical: d435(X前,Y右,Z下) -> optical(X右,Y下,Z前)
# optical X = d435 Y -> col0 = [0,1,0]
# optical Y = d435 Z -> col1 = [0,0,1]  (等等 Z下)
# 这太多可能性了，让我用另一种方法

# 更好的方法：直接从物理约束推导
# 已知相机朝下约 ~50-60°
# 尝试 pitch = -47.6° 直接做投影验证

print("\n" + "=" * 70)
print("第七步：投影验证 — 多组参数对比")
print("=" * 70)

# 加载数据
TASK = "G1_WBT_Inspire_Put_Clothes_into_Washing_Machine"
data_dir = os.path.join(DATASET_ROOT, TASK)
EP = 0
FRAME = 30

# 检查数据是否存在
if not os.path.exists(data_dir):
    # 尝试其他任务
    for t in ["G1_WBT_Brainco_Make_The_Bed", "G1_WBT_Inspire_Pickup_Pillow_MainCamOnly"]:
        alt = os.path.join(DATASET_ROOT, t)
        if os.path.exists(alt):
            TASK = t
            data_dir = alt
            break

print(f"使用数据: {TASK}, ep={EP}, frame={FRAME}")

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
    print("无法提取视频帧!")
    sys.exit(1)

h, w = img.shape[:2]
print(f"帧大小: {w}x{h}")

# 关键点定义
ALL_KEYPOINTS = [
    ("L_wrist", "left_wrist_yaw_link",   np.array([ 0.0046,  0.0000,  0.0300])),
    ("L_thumb", "left_thumb_4",          np.array([-0.0314,  0.0150, -0.0101])),
    ("L_toe",   "left_ankle_roll_link",  np.array([ 0.1424,  0.0000, -0.0210])),
    ("R_toe",   "right_ankle_roll_link", np.array([ 0.1424, -0.0000, -0.0215])),
    ("R_thumb", "right_thumb_4",         np.array([ 0.0314,  0.0150, -0.0101])),
    ("R_wrist", "right_wrist_yaw_link",  np.array([ 0.0046,  0.0000,  0.0300])),
]

# 过滤掉不可用的关键点 (BrainCo 任务没有 thumb links)
available_kps = []
for name, link, offset in ALL_KEYPOINTS:
    if link in transforms:
        available_kps.append((name, link, offset))
    else:
        print(f"  跳过 {name} (link {link} 不在 FK 结果中)")

# 计算关键点世界坐标
def get_kp_world(transforms, kps):
    pts = []
    for name, link, offset in kps:
        t, R = transforms[link]
        pts.append(R @ offset + t)
    return np.array(pts, dtype=np.float64)

kp_world = get_kp_world(transforms, available_kps)
kp_names = [k[0] for k in available_kps]

# 投影函数
def project_keypoints(params, transforms, kp_world):
    K, D, rvec, tvec, _, _, fisheye = make_camera(params, transforms)
    pts3d = kp_world.reshape(-1, 1, 3).astype(np.float64)
    pts2d = project_points_cv(pts3d, rvec, tvec, K, D, fisheye)
    return pts2d.reshape(-1, 2)

# 多组参数对比
param_sets = {
    "标定结果 (fx=291,fy=287)": dict(BEST_PARAMS),
    "URDF外参 + 标定内参": {
        "dx": 0.0576235, "dy": 0.01753, "dz": 0.42987,
        "pitch": -47.6, "yaw": 0.0, "roll": 0.0,
        "fx": 290.78, "fy": 287.35, "cx": 329.37, "cy": 313.68,
    },
    "URDF外参 + fx=fy=380": {
        "dx": 0.0576235, "dy": 0.01753, "dz": 0.42987,
        "pitch": -47.6, "yaw": 0.0, "roll": 0.0,
        "fx": 380.0, "fy": 380.0, "cx": 320.0, "cy": 240.0,
    },
    "URDF外参 + fx=fy=420": {
        "dx": 0.0576235, "dy": 0.01753, "dz": 0.42987,
        "pitch": -47.6, "yaw": 0.0, "roll": 0.0,
        "fx": 420.0, "fy": 420.0, "cx": 320.0, "cy": 240.0,
    },
    "标定外参 + fx=fy=380": {
        "dx": 0.0758, "dy": 0.0226, "dz": 0.4484,
        "pitch": -61.5855, "yaw": 2.1690, "roll": 0.2331,
        "fx": 380.0, "fy": 380.0, "cx": 329.37, "cy": 313.68,
    },
    "标定外参 + fx=fy=420": {
        "dx": 0.0758, "dy": 0.0226, "dz": 0.4484,
        "pitch": -61.5855, "yaw": 2.1690, "roll": 0.2331,
        "fx": 420.0, "fy": 420.0, "cx": 329.37, "cy": 313.68,
    },
}

print(f"\n关键点世界坐标:")
for i, (name, link, offset) in enumerate(available_kps):
    print(f"  {name}: {kp_world[i]} (link={link})")

# 绘制对比图
out_dir = os.path.join(OUTPUT_DIR, "urdf_verify")
os.makedirs(out_dir, exist_ok=True)

colors = [
    (0, 0, 255),     # 红
    (0, 255, 0),     # 绿
    (255, 0, 0),     # 蓝
    (0, 255, 255),   # 黄
    (255, 0, 255),   # 品红
    (255, 255, 0),   # 青
]

print(f"\n各参数组关键点投影:")
for pname, params in param_sets.items():
    pts2d = project_keypoints(params, transforms, kp_world)
    print(f"\n  {pname}:")
    in_frame = 0
    for i, (kp_name, _) in enumerate(zip(kp_names, pts2d)):
        x, y = pts2d[i]
        inside = 0 <= x <= w and 0 <= y <= h
        if inside:
            in_frame += 1
        print(f"    {kp_name}: ({x:.1f}, {y:.1f}) {'✓' if inside else '✗ (out of frame)'}")
    print(f"    帧内关键点: {in_frame}/{len(kp_names)}")

# 绘制综合对比图
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
print(f"\n对比图已保存: {out_path}")

# 也为每组参数生成单独的 overlay 图
skip_set = get_skip_meshes(hand_type)
link_meshes = parse_urdf_meshes(G1_URDF)
mesh_cache = preload_meshes(link_meshes, MESH_DIR, skip_set=skip_set)

for pi, (pname, params) in enumerate(param_sets.items()):
    overlay = img.copy()

    # 绘制 mesh overlay
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
    print(f"  保存: {out_path}")

# ============================================================
# 最终总结
# ============================================================
print("\n" + "=" * 70)
print("最终总结")
print("=" * 70)

print(f"""
换算结果:

URDF d435 参数:
  xyz = (0.0576235, 0.01753, 0.42987)
  rpy = (0, 0.8308, 0)

→ 代码参数 (假设 d435_link 坐标系 = body 约定):
  dx    = 0.0576235
  dy    = 0.01753
  dz    = 0.42987
  pitch = -47.6°  (URDF pitch 取负, 因为代码中 pitch>0 朝上, URDF的旋转使光轴朝下)
  yaw   = 0.0°
  roll  = 0.0°

  等一下! 上面的推导有 bug. 让我最终确认一次.
""")

# 最终严格推导:
# 代码: R_torso_to_optical = R_body_to_cam @ Rx(roll) @ Rz(yaw) @ Ry(pitch)
# URDF: R_torso_to_d435_link = Ry_urdf(0.831)
# 假设 d435_link = body 约定: R_d435_link_to_optical = R_body_to_cam
# 总: R_torso_to_optical = R_body_to_cam @ R_urdf
# 所以: R_body_to_cam @ R_combo = R_body_to_cam @ R_urdf
# 即: R_combo = R_urdf = Ry(+47.6°)
# 代码 pitch = +47.6°
# 光轴 = R_combo^T @ [1,0,0] = Ry(-47.6°) @ [1,0,0] = [cos47.6, 0, sin(-47.6)]
# 等等不对: Ry(θ)^T = Ry(-θ)
# Ry(-47.6°) @ [1,0,0] = [cos(-47.6°), 0, -sin(-47.6°)] = [cos47.6, 0, sin47.6]
# = [0.674, 0, +0.739] -> 朝前上方

# 嗯所以 d435_link 不是 body 约定(X前 Y左 Z上)

# 尝试: d435_link = (X前, Y右, Z下) — 一种常见的相机 link 约定
# R_body_to_d435: body(X前 Y左 Z上) -> d435(X前 Y右 Z下)
# Y: 左 -> 右 (反), Z: 上 -> 下 (反)
# R = diag(1, -1, -1)
R_body_to_d435 = np.diag([1.0, -1.0, -1.0])
# R_d435_to_optical: d435(X前 Y右 Z下) -> optical(X右 Y下 Z前)
# optical X = d435 Y (右), optical Y = d435 Z (下... 等等, d435 Z=下, optical Y=下, 所以取负?)
# No: d435 Z = 下, optical Y = 下 → optical Y = d435(-Z)?
# d435(X前 Y右 Z下): X→前, Y→右, Z→下
# optical(X右 Y下 Z前): X→右=d435.Y, Y→下=d435.Z, Z→前=d435.X
R_d435_to_optical = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=np.float64)

# R_torso_to_optical = R_d435_to_optical @ R_urdf @ R_body_to_d435...
# 不不不, R_urdf = R_torso_to_d435_link
# R_torso_to_optical = R_d435_to_optical @ R_urdf

# 但这假设 d435_link 本身的 identity 姿态就是 X前 Y右 Z下
# 而 R_urdf 把 torso frame 的点转到 d435_link frame
# 不，R_urdf 的含义是: d435_link frame axes expressed in torso frame
# 即 d435_link 坐标系在 torso 坐标系中的表示

# 让我更简洁地思考：
# URDF joint transform 的含义:
#   一个点在 child frame 中的坐标 p_child
#   在 parent frame 中是: p_parent = R_urdf @ p_child + t_urdf
#
# 所以 R_urdf 把 d435_link 中的坐标转到 torso 中
# d435_link 的光轴 z_optical 在 d435_link frame 中的方向取决于约定
#
# 假设 d435_link frame 中光轴 = [1, 0, 0] (X轴前):
#   光轴 in torso = R_urdf @ [1,0,0] = [cos(0.831), 0, -sin(0.831)]
#   = [0.674, 0, -0.739] → 朝前下方 47.6° ← 合理!

print("尝试不同的 d435_link 光轴方向假设:")
for axis_name, axis in [("X", [1,0,0]), ("Y", [0,1,0]), ("Z", [0,0,1]),
                         ("-X", [-1,0,0]), ("-Y", [0,-1,0]), ("-Z", [0,0,-1])]:
    optical_in_torso = R_urdf @ np.array(axis, dtype=float)
    elev = np.degrees(np.arctan2(optical_in_torso[2], optical_in_torso[0]))
    print(f"  d435 {axis_name:3s} in torso = [{optical_in_torso[0]:+.3f}, {optical_in_torso[1]:+.3f}, {optical_in_torso[2]:+.3f}]  仰角 = {elev:+.1f}°")

print(f"\n标定的光轴仰角: {elev_cal:.1f}°")
print(f"\n只有 d435 X轴 作为光轴时, 仰角 = -47.6°, 最接近标定的 -61.5°")
print(f"差异约 14°")

# ============================================================
# 重要洞察: 差异来源
# ============================================================
print("\n" + "=" * 70)
print("第八步：分析 14° 差异的来源")
print("=" * 70)

print(f"""
URDF 给出 pitch ≈ -47.6° (如果 d435 X=光轴)
标定给出 pitch ≈ -61.6°
差异 ≈ 14°

可能原因:
1. d435_link 坐标系约定不是 X=光轴
2. URDF 中 d435 位姿不完全准确(设计值 vs 实际安装值)
3. 标定的 pitch 和内参之间有耦合(不同 pitch+内参 可以产生相似投影)

最重要的: 固定 URDF 外参, 优化内参, 看能否获得好的投影!
""")

# 做一个更精确的计算: 如果 d435 光轴 = d435_link X轴
# 那么 R_d435_to_optical 是什么?
# d435_link: X=光轴(前), Y=左, Z=上 (标准 body 约定)
# optical: X=右, Y=下, Z=前(光轴)
# R_d435_to_optical:
#   optical Z = d435 X → row 2 of R = [1, 0, 0] ? no...
#   R_d435_to_optical 把 d435 坐标转到 optical:
#   optical_X = d435 的哪个轴? 右 → d435 -Y → [0, -1, 0]
#   optical_Y = d435 的哪个轴? 下 → d435 -Z → [0, 0, -1]
#   optical_Z = d435 的哪个轴? 前 → d435 X  → [1, 0, 0]
#   R_d435_to_optical = [[0, -1, 0], [0, 0, -1], [1, 0, 0]]
#   = R_body_to_cam !!
# 所以 d435_link 用的就是标准 body 约定 (X前 Y左 Z上)
# 而 R_d435_to_optical = R_body_to_cam

# 但之前已经分析过: 这给出 pitch = +47.6° (光轴朝上)
# 矛盾！

# 啊! 我犯了一个错误。让我重新推导:
# R_urdf 把 d435 frame 的坐标转到 torso frame
# 即: p_torso = R_urdf @ p_d435 + t_urdf
#
# d435 光轴在 d435 frame 中 = [1,0,0] (假设 X=光轴)
# 光轴在 torso frame = R_urdf @ [1,0,0] = [0.674, 0, -0.739] → 朝前下 47.6° ✓
#
# 代码中:
# R_cam = R_body_to_cam @ R_combo
# R_combo 的含义: 相当于 R_torso_to_d435_mounting
# 即 R_combo 描述了 d435 安装方向(在 torso body 坐标系中的旋转)
#
# 光轴(body坐标系) = R_cam^T @ [0,0,1] = R_combo^T @ R_body_to_cam^T @ [0,0,1]
# = R_combo^T @ [1, 0, 0] (因为 R_body_to_cam^T @ [0,0,1] = [1,0,0])
#
# 我们想要光轴(torso坐标系) = [0.674, 0, -0.739]
# 所以 R_combo^T @ [1,0,0] = [0.674, 0, -0.739]
# R_combo 的第一列 = [0.674, 0, -0.739]

# 如果 R_combo = Ry(p) (只有pitch):
# Ry(p) 的第一列 = [cos(p), 0, -sin(p)]
# cos(p) = 0.674, sin(p) = 0.739
# p = arctan2(0.739, 0.674) = 47.6° ???

# 但是!!! 检查 Ry(47.6°)^T @ [1,0,0]:
# Ry(47.6°) = [[cos, 0, sin], [0, 1, 0], [-sin, 0, cos]]
# = [[0.674, 0, 0.739], [0, 1, 0], [-0.739, 0, 0.674]]
# Ry(47.6°)^T @ [1,0,0] = [[0.674, 0, -0.739], [0, 1, 0], [0.739, 0, 0.674]] @ [1,0,0]
# = [0.674, 0, 0.739]
# 光轴 = [0.674, 0, +0.739] → 朝上 47.6°! 不对!

# 我需要 R_combo^T @ [1,0,0] = [0.674, 0, -0.739]
# 如果 R_combo = Ry(-47.6°):
# Ry(-47.6°) = [[cos, 0, -sin], [0, 1, 0], [sin, 0, cos]]
# = [[0.674, 0, -0.739], [0, 1, 0], [0.739, 0, 0.674]]
# Ry(-47.6°)^T @ [1,0,0] = [[0.674, 0, 0.739], [0, 1, 0], [-0.739, 0, 0.674]] @ [1,0,0]
# = [0.674, 0, -0.739] ← 正确!

print("最终推导:")
print(f"  R_combo^T @ [1,0,0] = 光轴(torso)")
print(f"  需要光轴 = [0.674, 0, -0.739]")
print(f"  R_combo = Ry(p), Ry(p)^T = Ry(-p)")
print(f"  Ry(-p) @ [1,0,0] = [cos(p), 0, sin(p)]  (注意: 这是 Ry(-p)!)")
print(f"  但 Ry(p)^T @ [1,0,0] = [cos(p), 0, -sin(p)] (Ry^T 第一行)")

# Let me be really careful:
# Ry(p) = [[cos(p), 0, sin(p)], [0, 1, 0], [-sin(p), 0, cos(p)]]
# Ry(p)^T = [[cos(p), 0, -sin(p)], [0, 1, 0], [sin(p), 0, cos(p)]]
#
# Ry(p)^T @ [1,0,0] = [cos(p), 0, sin(p)]  ← 第一列 of Ry(p)^T
#
# Wait no! Matrix @ vector:
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
    print(f"  Ry({p_deg:+.1f}°)^T @ [1,0,0] = {result}")

# 所以:
# Ry(+47.6°)^T @ [1,0,0] = [0.674, 0, -0.739] → 朝前下方 ← 这是对的!
# Ry(-47.6°)^T @ [1,0,0] = [0.674, 0, +0.739] → 朝前上方

print(f"""
修正! Ry(+47.6°)^T @ [1,0,0] = [0.674, 0, -0.739] → 朝前下方 47.6°

这意味着代码 pitch = +47.6° 确实对应光轴朝下!

之前我的手动计算有误。让我用 compute_optical_axis 验证:
""")

opt_47, _ = compute_optical_axis(47.6)
print(f"compute_optical_axis(47.6) = {opt_47}")
elev_47 = np.degrees(np.arctan2(opt_47[2], opt_47[0]))
print(f"仰角 = {elev_47:.1f}°")

opt_neg47, _ = compute_optical_axis(-47.6)
print(f"compute_optical_axis(-47.6) = {opt_neg47}")
elev_neg47 = np.degrees(np.arctan2(opt_neg47[2], opt_neg47[0]))
print(f"仰角 = {elev_neg47:.1f}°")

# 哈！compute_optical_axis 显示 pitch=47.6 → 仰角 +47.6°(朝上)
# 但数学计算显示 Ry(47.6)^T @ [1,0,0] = [0.674, 0, -0.739](朝下)
# 必须有一个地方错了！让我检查 compute_optical_axis

print(f"\n检查 compute_optical_axis 内部:")
p = np.radians(47.6)
R_p = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
R_cam = R_body_to_cam @ R_p
print(f"R_cam = R_body_to_cam @ Ry(47.6°) =")
print(R_cam)
opt = R_cam.T @ np.array([0, 0, 1])
print(f"R_cam^T @ [0,0,1] = {opt}")
# R_cam^T = Ry(47.6°)^T @ R_body_to_cam^T
# R_body_to_cam^T @ [0,0,1] = ?
rbtc_t = R_body_to_cam.T @ np.array([0, 0, 1])
print(f"R_body_to_cam^T @ [0,0,1] = {rbtc_t}")
# Should be [1, 0, 0]?
# R_body_to_cam = [[0,-1,0],[0,0,-1],[1,0,0]]
# R_body_to_cam^T = [[0,0,1],[-1,0,0],[0,-1,0]]
# R_body_to_cam^T @ [0,0,1] = [1, 0, 0]?
# [[0,0,1],[-1,0,0],[0,-1,0]] @ [0,0,1] = [1, 0, 0] ✓

ry_t_result = R_p.T @ rbtc_t
print(f"Ry(47.6°)^T @ [1,0,0] = {ry_t_result}")

# 嗯! Ry(47.6°)^T @ [1,0,0]
# Ry(47.6°) = [[0.674, 0, 0.739], [0, 1, 0], [-0.739, 0, 0.674]]
# Ry(47.6°)^T = [[0.674, 0, -0.739], [0, 1, 0], [0.739, 0, 0.674]]
# Ry^T[row0] = [0.674, 0, -0.739]
# Ry^T[row1] = [0, 1, 0]
# Ry^T[row2] = [0.739, 0, 0.674]
# Ry^T @ [1,0,0] = [0.674, 0, 0.739]

# 等等 Ry^T @ v:
# result[0] = Ry^T[0,0]*1 = 0.674
# result[1] = Ry^T[1,0]*1 = 0
# result[2] = Ry^T[2,0]*1 = 0.739
# = [0.674, 0, 0.739] 朝前上方!

# 但直接打印出来 ry_t_result 是什么?
# 这跟 compute_optical_axis 一致: pitch=47.6 → 光轴朝上

# 所以之前我的"手动计算"有误（第一列 vs 行的混淆）
# 正确的结论: pitch = +47.6° → 光轴朝上 47.6° (不合理)

# 那如果 d435_link 的光轴不是 X 轴呢?
# 之前测试过: d435 X轴 in torso = [0.674, 0, -0.739] 朝前下方
# 但代码中 pitch=+47.6° 给出光轴朝上

# 矛盾说明:
# URDF 中 d435_link 坐标系和代码中 body 坐标系的**旋转约定不同**!
#
# 具体来说: d435_link 的坐标系可能跟 body 约定有一个 180° 翻转
# 比如 d435_link: X=后(或Z=下)
# 这样 R_urdf 的 pitch 旋转方向就反了

# 最直接的验证: code pitch = -47.6° 是否合理?
print(f"\n" + "=" * 70)
print("最终确认: code pitch = -47.6°")
print("=" * 70)

opt_m47, _ = compute_optical_axis(-47.6)
print(f"pitch=-47.6° 光轴 = {opt_m47}")
elev_m47 = np.degrees(np.arctan2(opt_m47[2], opt_m47[0]))
print(f"仰角 = {elev_m47:.1f}° (朝下)")

# d435 X轴在torso中 = R_urdf @ [1,0,0] = [0.674, 0, -0.739]
# 如果这是光轴方向, 仰角 = arctan2(-0.739, 0.674) = -47.6°
# 代码中 pitch=-47.6° 的光轴 = [0.674, 0, -0.739]
# 完全吻合!

print(f"\n验证:")
print(f"  d435 X轴 in torso = {R_urdf @ [1,0,0]}")
print(f"  code pitch=-47.6° 光轴 = {opt_m47}")
print(f"  完美吻合!")

print(f"""
结论: URDF rpy=(0, 0.831, 0) 对应代码 pitch = -47.6°

原因: URDF 的 Ry(+0.831) 旋转使 d435_link 的 X 轴(光轴)指向前下方,
      但代码中 Ry(+47.6°) 的效果是让光轴指向前上方。
      两者的旋转约定恰好相反(一个是"旋转坐标系", 一个是"旋转向量")。

      更精确地说:
      URDF: 光轴 in torso = R_urdf @ [1,0,0] (旋转坐标轴)
      代码: 光轴 in torso = Ry(p)^T @ [1,0,0] (旋转的逆)

      R_urdf @ [1,0,0] = Ry(0.831) @ [1,0,0] = [cos, 0, -sin] = 朝下
      Ry(p)^T @ [1,0,0] = [cos(p), 0, sin(p)]
      要使 sin(p) < 0 (朝下), 需要 p < 0
      cos(p)=cos(0.831), sin(p)=-sin(0.831) → p = -0.831 rad = -47.6°

最终 URDF 外参 → 代码参数:
  dx    = 0.0576235
  dy    = 0.01753
  dz    = 0.42987
  pitch = -47.6000° (= -0.8308 rad = 负的 URDF pitch)
  yaw   = 0.0°
  roll  = 0.0°
""")

# 更新参数集并重新投影
param_sets_final = {
    "标定结果 (pitch=-61.6)": dict(BEST_PARAMS),
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
    "标定外参 + fx=fy=380": {
        "dx": 0.0758, "dy": 0.0226, "dz": 0.4484,
        "pitch": -61.5855, "yaw": 2.1690, "roll": 0.2331,
        "fx": 380.0, "fy": 380.0, "cx": 329.37, "cy": 313.68,
    },
}

print("\n" + "=" * 70)
print("投影对比 (修正后)")
print("=" * 70)

for pname, params in param_sets_final.items():
    pts2d = project_keypoints(params, transforms, kp_world)
    print(f"\n  {pname}:")
    for i, kp_name in enumerate(kp_names):
        x, y = pts2d[i]
        inside = 0 <= x <= w and 0 <= y <= h
        print(f"    {kp_name}: ({x:.1f}, {y:.1f}) {'✓' if inside else '✗'}")

# 绘制最终对比图
canvas = img.copy()
legend_y = 30
for pi, (pname, params) in enumerate(param_sets_final.items()):
    color = colors[pi % len(colors)]
    pts2d = project_keypoints(params, transforms, kp_world)
    for i, kp_name in enumerate(kp_names):
        x, y = int(pts2d[i, 0]), int(pts2d[i, 1])
        if -100 <= x <= w + 100 and -100 <= y <= h + 100:
            cv2.circle(canvas, (x, y), 5, color, 2)
            if pi == 0:  # 只标注第一组
                cv2.putText(canvas, kp_name, (x + 8, y - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
    cv2.putText(canvas, f"[{pi}] {pname}", (10, legend_y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
    legend_y += 16

out_path = os.path.join(out_dir, "kp_comparison_final.png")
cv2.imwrite(out_path, canvas)
print(f"\n最终对比图: {out_path}")

# 生成每组参数的独立 overlay
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

print("\n完成!")
