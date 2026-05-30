"""
验证 G1 机器人 STL mesh 尺寸和 URDF FK 关节距离是否与真实 G1 匹配。

真实 G1 规格:
  - 身高约 127cm (不含头), 总高约 132cm
  - 上臂长约 28cm, 前臂长约 24cm
  - 大腿长约 30cm, 小腿长约 30cm
  - 躯干宽约 35cm
"""

import sys
import os
import numpy as np
from stl import mesh as stl_mesh
import pinocchio as pin

sys.stdout.reconfigure(line_buffering=True)

MESH_DIR = "/disk_n/zzf/flip/data/unitree_G1_WBT/mesh/meshes"
URDF_PATH = "/disk_n/zzf/flip/data/unitree_G1_WBT/mesh/g1_29dof_rev_1_0_with_inspire_hand_FTP.urdf"
URDF_DIR = "/disk_n/zzf/flip/data/unitree_G1_WBT/mesh"

# ============================================================
# Part 1: STL Bounding Box 分析
# ============================================================
print("=" * 70)
print("Part 1: STL Mesh Bounding Box 分析")
print("=" * 70)

stl_files = [
    ("pelvis.STL", "骨盆"),
    ("left_hip_pitch_link.STL", "左髋 pitch"),
    ("left_hip_roll_link.STL", "左髋 roll"),
    ("left_hip_yaw_link.STL", "左髋 yaw"),
    ("left_knee_link.STL", "左膝"),
    ("left_ankle_pitch_link.STL", "左踝 pitch"),
    ("left_ankle_roll_link.STL", "左踝 roll (脚)"),
    ("torso_link_rev_1_0.STL", "躯干"),
    ("left_shoulder_pitch_link.STL", "左肩 pitch"),
    ("left_shoulder_roll_link.STL", "左肩 roll"),
    ("left_shoulder_yaw_link.STL", "左肩 yaw"),
    ("left_elbow_link.STL", "左肘"),
    ("left_wrist_roll_link.STL", "左腕 roll"),
    ("left_wrist_pitch_link.STL", "左腕 pitch"),
    ("left_wrist_yaw_link.STL", "左腕 yaw"),
    ("head_link.STL", "头部"),
    ("waist_yaw_link_rev_1_0.STL", "腰 yaw"),
    ("waist_roll_link_rev_1_0.STL", "腰 roll"),
]

print(f"\n{'部件':<20} {'X(mm)':<12} {'Y(mm)':<12} {'Z(mm)':<12} {'X范围':<20} {'Y范围':<20} {'Z范围':<20}")
print("-" * 116)

for fname, label in stl_files:
    fpath = os.path.join(MESH_DIR, fname)
    if not os.path.exists(fpath):
        print(f"{label:<20} FILE NOT FOUND: {fname}")
        continue
    m = stl_mesh.Mesh.from_file(fpath)
    verts = m.vectors.reshape(-1, 3)
    mn = verts.min(axis=0)
    mx = verts.max(axis=0)
    sz = mx - mn
    print(f"{label:<20} {sz[0]*1000:>8.1f}    {sz[1]*1000:>8.1f}    {sz[2]*1000:>8.1f}    "
          f"[{mn[0]*1000:>7.1f}, {mx[0]*1000:>7.1f}]   "
          f"[{mn[1]*1000:>7.1f}, {mx[1]*1000:>7.1f}]   "
          f"[{mn[2]*1000:>7.1f}, {mx[2]*1000:>7.1f}]")

# ============================================================
# Part 2: URDF 关节偏移链分析
# ============================================================
print("\n" + "=" * 70)
print("Part 2: URDF 关节偏移链分析 (从 joint origin xyz)")
print("=" * 70)

# 从 URDF 直接读取的关节偏移 (xyz)
joint_offsets = {
    # 左腿链: pelvis -> left_hip_pitch -> left_hip_roll -> left_hip_yaw -> left_knee -> left_ankle_pitch -> left_ankle_roll
    "left_hip_pitch_joint": np.array([0, 0.064452, -0.1027]),
    "left_hip_roll_joint": np.array([0, 0.052, -0.030465]),
    "left_hip_yaw_joint": np.array([0.025001, 0, -0.12412]),
    "left_knee_joint": np.array([-0.078273, 0.0021489, -0.17734]),
    "left_ankle_pitch_joint": np.array([0, -9.4445e-05, -0.30001]),
    "left_ankle_roll_joint": np.array([0, 0, -0.017558]),

    # 上半身: pelvis -> waist_yaw -> waist_roll -> torso
    "waist_yaw_joint": np.array([0, 0, 0]),
    "waist_roll_joint": np.array([-0.0039635, 0, 0.044]),
    "waist_pitch_joint": np.array([0, 0, 0]),

    # 左臂: torso -> left_shoulder_pitch -> left_shoulder_roll -> left_shoulder_yaw -> left_elbow -> left_wrist_roll -> left_wrist_pitch -> left_wrist_yaw
    "left_shoulder_pitch_joint": np.array([0.0039563, 0.10022, 0.24778]),
    "left_shoulder_roll_joint": np.array([0, 0.038, -0.013831]),
    "left_shoulder_yaw_joint": np.array([0, 0.00624, -0.1032]),
    "left_elbow_joint": np.array([0.015783, 0, -0.080518]),
    "left_wrist_roll_joint": np.array([0.100, 0.00188791, -0.010]),
    "left_wrist_pitch_joint": np.array([0.038, 0, 0]),
    "left_wrist_yaw_joint": np.array([0.046, 0, 0]),

    # 头部
    "head_joint": np.array([0.0039635, 0, -0.044]),
}

# 计算关键段长度
print("\n--- 左腿链 ---")
# 髋关节到膝关节: hip_pitch + hip_roll + hip_yaw + knee 的偏移
hip_to_knee_offsets = [
    joint_offsets["left_hip_roll_joint"],
    joint_offsets["left_hip_yaw_joint"],
    joint_offsets["left_knee_joint"],
]
hip_to_knee_total = sum(hip_to_knee_offsets)
hip_to_knee_dist = np.linalg.norm(hip_to_knee_total)
print(f"  髋 pitch -> 膝 (总偏移): {hip_to_knee_total*1000} mm")
print(f"  髋 pitch -> 膝 (直线距离): {hip_to_knee_dist*1000:.1f} mm")

# 各段
for name, offset in [("hip_roll", "left_hip_roll_joint"),
                     ("hip_yaw", "left_hip_yaw_joint"),
                     ("knee", "left_knee_joint")]:
    d = np.linalg.norm(joint_offsets[offset])
    print(f"    {name} 偏移: {joint_offsets[offset]*1000} -> {d*1000:.1f} mm")

# 膝关节到踝关节
knee_to_ankle = joint_offsets["left_ankle_pitch_joint"]
knee_to_ankle_dist = np.linalg.norm(knee_to_ankle)
print(f"\n  膝 -> 踝 pitch (偏移): {knee_to_ankle*1000} mm")
print(f"  膝 -> 踝 pitch (直线距离): {knee_to_ankle_dist*1000:.1f} mm")

# 踝到脚底
ankle_to_foot = joint_offsets["left_ankle_roll_joint"]
ankle_to_foot_dist = np.linalg.norm(ankle_to_foot)
print(f"  踝 pitch -> 踝 roll (偏移): {ankle_to_foot*1000} mm")

print("\n--- 躯干链 ---")
# pelvis -> waist_yaw -> waist_roll -> torso
waist_total = (joint_offsets["waist_yaw_joint"] +
               joint_offsets["waist_roll_joint"] +
               joint_offsets["waist_pitch_joint"])
print(f"  骨盆 -> 躯干 (总偏移): {waist_total*1000} mm, dist={np.linalg.norm(waist_total)*1000:.1f} mm")

# torso -> shoulder
print(f"  躯干 -> 左肩 pitch: {joint_offsets['left_shoulder_pitch_joint']*1000} mm")
print(f"    -> 竖直方向 (Z): {joint_offsets['left_shoulder_pitch_joint'][2]*1000:.1f} mm")
print(f"    -> 横向 (Y): {joint_offsets['left_shoulder_pitch_joint'][1]*1000:.1f} mm")

# 肩宽: 2 * Y偏移
shoulder_width = 2 * abs(joint_offsets['left_shoulder_pitch_joint'][1])
print(f"  肩宽 (2*Y): {shoulder_width*1000:.1f} mm")

# 头
head_offset = joint_offsets["head_joint"]
print(f"  躯干 -> 头: {head_offset*1000} mm")

print("\n--- 左臂链 ---")
# 肩到肘
shoulder_to_elbow_offsets = [
    joint_offsets["left_shoulder_roll_joint"],
    joint_offsets["left_shoulder_yaw_joint"],
    joint_offsets["left_elbow_joint"],
]
shoulder_to_elbow_total = sum(shoulder_to_elbow_offsets)
shoulder_to_elbow_dist = np.linalg.norm(shoulder_to_elbow_total)
print(f"  肩 pitch -> 肘 (总偏移): {shoulder_to_elbow_total*1000} mm")
print(f"  肩 pitch -> 肘 (直线距离): {shoulder_to_elbow_dist*1000:.1f} mm")

for name, offset in [("shoulder_roll", "left_shoulder_roll_joint"),
                     ("shoulder_yaw", "left_shoulder_yaw_joint"),
                     ("elbow", "left_elbow_joint")]:
    d = np.linalg.norm(joint_offsets[offset])
    print(f"    {name} 偏移: {joint_offsets[offset]*1000} -> {d*1000:.1f} mm")

# 肘到腕
elbow_to_wrist_offsets = [
    joint_offsets["left_wrist_roll_joint"],
    joint_offsets["left_wrist_pitch_joint"],
    joint_offsets["left_wrist_yaw_joint"],
]
elbow_to_wrist_total = sum(elbow_to_wrist_offsets)
elbow_to_wrist_dist = np.linalg.norm(elbow_to_wrist_total)
print(f"\n  肘 -> 腕末端 (总偏移): {elbow_to_wrist_total*1000} mm")
print(f"  肘 -> 腕末端 (直线距离): {elbow_to_wrist_dist*1000:.1f} mm")

for name, offset in [("wrist_roll", "left_wrist_roll_joint"),
                     ("wrist_pitch", "left_wrist_pitch_joint"),
                     ("wrist_yaw", "left_wrist_yaw_joint")]:
    d = np.linalg.norm(joint_offsets[offset])
    print(f"    {name} 偏移: {joint_offsets[offset]*1000} -> {d*1000:.1f} mm")


# ============================================================
# Part 3: Pinocchio FK 验证
# ============================================================
print("\n" + "=" * 70)
print("Part 3: Pinocchio FK 零位姿态关节位置")
print("=" * 70)

model = pin.buildModelFromUrdf(URDF_PATH, pin.JointModelFreeFlyer())
data = model.createData()

# 零位 (neutral)
q0 = pin.neutral(model)
# 把 pelvis 放到 (0,0,1) 高度，方便观察
q0[2] = 1.0

pin.forwardKinematics(model, data, q0)
pin.updateFramePlacements(model, data)

# 列出所有 frame
print("\n所有 frame (type=BODY 或 JOINT):")
key_frames = {}
for i in range(model.nframes):
    frame = model.frames[i]
    if frame.type in (pin.FrameType.BODY, pin.FrameType.JOINT):
        pos = data.oMf[i].translation
        key_frames[frame.name] = pos.copy()
        # 只打印关键的
        if any(kw in frame.name for kw in ['pelvis', 'hip_pitch', 'knee', 'ankle_roll', 'torso',
                                             'shoulder_pitch', 'elbow', 'wrist_yaw', 'head',
                                             'left_hand_palm']):
            print(f"  {frame.name:<40} pos=({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f})")

# 计算关键距离
print("\n--- FK 关键距离 (零位) ---")

def frame_dist(name1, name2):
    if name1 in key_frames and name2 in key_frames:
        d = np.linalg.norm(key_frames[name1] - key_frames[name2])
        diff = key_frames[name2] - key_frames[name1]
        return d, diff
    return None, None

pairs = [
    ("pelvis", "left_hip_pitch_link", "骨盆 -> 左髋 pitch"),
    ("left_hip_pitch_link", "left_knee_link", "左髋 pitch -> 左膝 (大腿)"),
    ("left_knee_link", "left_ankle_pitch_link", "左膝 -> 左踝 pitch (小腿)"),
    ("left_ankle_pitch_link", "left_ankle_roll_link", "左踝 pitch -> 左踝 roll"),
    ("pelvis", "torso_link", "骨盆 -> 躯干"),
    ("torso_link", "left_shoulder_pitch_link", "躯干 -> 左肩 pitch"),
    ("left_shoulder_pitch_link", "left_elbow_link", "左肩 pitch -> 左肘 (上臂)"),
    ("left_elbow_link", "left_wrist_yaw_link", "左肘 -> 左腕 yaw (前臂)"),
    ("torso_link", "head_link", "躯干 -> 头部"),
    ("pelvis", "left_ankle_roll_link", "骨盆 -> 左脚踝 (全腿)"),
    ("pelvis", "head_link", "骨盆 -> 头 (躯干高度)"),
    ("left_shoulder_pitch_link", "right_shoulder_pitch_link", "左肩 -> 右肩 (肩宽)"),
]

print(f"\n{'描述':<35} {'距离(mm)':<12} {'差值 (dx,dy,dz) mm'}")
print("-" * 80)
for f1, f2, desc in pairs:
    d, diff = frame_dist(f1, f2)
    if d is not None:
        print(f"  {desc:<35} {d*1000:>8.1f}    ({diff[0]*1000:>7.1f}, {diff[1]*1000:>7.1f}, {diff[2]*1000:>7.1f})")
    else:
        print(f"  {desc:<35} FRAME NOT FOUND")


# ============================================================
# Part 4: 与真实 G1 规格对比
# ============================================================
print("\n" + "=" * 70)
print("Part 4: 与真实 G1 规格对比")
print("=" * 70)

specs = {
    "大腿 (髋pitch->膝)": (None, 300),  # 30cm
    "小腿 (膝->踝)": (None, 300),        # 30cm
    "上臂 (肩pitch->肘)": (None, 280),   # 28cm
    "前臂 (肘->腕)": (None, 240),        # 24cm
    "肩宽": (None, 350),                  # 35cm
    "身高(不含头)": (None, 1270),          # 127cm
}

# 计算 URDF 值
d_thigh, _ = frame_dist("left_hip_pitch_link", "left_knee_link")
d_shin, _ = frame_dist("left_knee_link", "left_ankle_pitch_link")
d_upper_arm, _ = frame_dist("left_shoulder_pitch_link", "left_elbow_link")
d_forearm, _ = frame_dist("left_elbow_link", "left_wrist_yaw_link")
d_shoulder_w, _ = frame_dist("left_shoulder_pitch_link", "right_shoulder_pitch_link")
d_pelvis_head, _ = frame_dist("pelvis", "head_link")
d_full_leg, _ = frame_dist("pelvis", "left_ankle_roll_link")

comparisons = [
    ("大腿 (髋pitch->膝)", d_thigh, 0.300),
    ("小腿 (膝->踝)", d_shin, 0.300),
    ("上臂 (肩pitch->肘)", d_upper_arm, 0.280),
    ("前臂 (肘->腕yaw)", d_forearm, 0.240),
    ("肩宽 (左肩-右肩)", d_shoulder_w, 0.350),
    ("骨盆->头 (躯干高)", d_pelvis_head, 0.500),  # 大约
    ("全腿 (骨盆->踝)", d_full_leg, 0.650),  # 大约
]

print(f"\n{'部位':<25} {'URDF(mm)':<12} {'真实约(mm)':<12} {'差异(%)':<12}")
print("-" * 65)
for desc, urdf_val, real_val in comparisons:
    if urdf_val is not None:
        urdf_mm = urdf_val * 1000
        real_mm = real_val * 1000
        pct = (urdf_mm - real_mm) / real_mm * 100
        print(f"  {desc:<25} {urdf_mm:>8.1f}    {real_mm:>8.1f}    {pct:>+7.1f}%")


# ============================================================
# Part 5: 检查 STL 文件的 mesh 单位
# ============================================================
print("\n" + "=" * 70)
print("Part 5: STL 文件单位检查 (是否为米/毫米)")
print("=" * 70)

# 选几个关键文件检查顶点坐标范围
check_files = [
    ("pelvis.STL", "骨盆"),
    ("torso_link_rev_1_0.STL", "躯干"),
    ("left_knee_link.STL", "小腿"),
    ("left_elbow_link.STL", "前臂"),
]

for fname, label in check_files:
    fpath = os.path.join(MESH_DIR, fname)
    m = stl_mesh.Mesh.from_file(fpath)
    verts = m.vectors.reshape(-1, 3)
    mn = verts.min(axis=0)
    mx = verts.max(axis=0)
    sz = mx - mn
    max_coord = np.abs(verts).max()
    print(f"\n  {label} ({fname}):")
    print(f"    顶点坐标范围: X[{mn[0]:.6f}, {mx[0]:.6f}]  Y[{mn[1]:.6f}, {mx[1]:.6f}]  Z[{mn[2]:.6f}, {mx[2]:.6f}]")
    print(f"    bbox 尺寸: {sz[0]:.6f} x {sz[1]:.6f} x {sz[2]:.6f}")
    print(f"    最大绝对坐标值: {max_coord:.6f}")
    if max_coord < 1.0:
        print(f"    -> 单位推断: 米 (m)")
    elif max_coord < 1000:
        print(f"    -> 单位推断: 可能是毫米 (mm) 或厘米 (cm)")
    else:
        print(f"    -> 单位推断: 毫米 (mm)")

# ============================================================
# Part 6: URDF 中是否有 mesh scale 属性
# ============================================================
print("\n" + "=" * 70)
print("Part 6: 检查 URDF 中的 mesh scale")
print("=" * 70)

import xml.etree.ElementTree as ET
tree = ET.parse(URDF_PATH)
root = tree.getroot()

has_scale = False
for visual in root.iter('visual'):
    geom = visual.find('geometry')
    if geom is not None:
        mesh_elem = geom.find('mesh')
        if mesh_elem is not None:
            scale = mesh_elem.get('scale')
            if scale is not None:
                print(f"  Found mesh scale: {mesh_elem.get('filename')} -> scale={scale}")
                has_scale = True

if not has_scale:
    print("  URDF 中没有任何 mesh 的 scale 属性 (默认 scale=1)")
    print("  -> STL 文件中的坐标直接被当作米(m)使用")


print("\n" + "=" * 70)
print("总结")
print("=" * 70)
print("""
如果 STL 坐标是米(m)，且 URDF 无 scale，则 mesh 尺寸应当与关节偏移一致。
如果发现 mesh 偏小（如 STL 尺寸是真实的 1/1000），则 STL 可能用的毫米单位。
检查上面 Part 5 的输出即可判断。
""")
