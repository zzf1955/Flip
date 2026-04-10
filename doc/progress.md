# FLIP 项目进展总结

## 项目目标

FLIP (Flipped-Direction Learning via Inpainting Pipeline) — 在真实 G1 机器人第一人称视频上合成人体，得到（合成 human, 真实 robot）配对数据，用于微调 video-to-video 模型。

核心流程：**FK Mesh 渲染 → 机器人分割 → 背景修复 → 人体叠加**

---

## 一、数据准备

### 1.1 数据集下载

从 HuggingFace (`unitreerobotics`) 下载了 **8 个** G1 WBT (Whole Body Teleoperation) 数据集，按灵巧手类型分为两组：

| 手类型 | 数据集 | 数量 |
|--------|--------|------|
| **Inspire Hand** (因时) | Collect_Clothes, Pickup_Pillow, Put_Clothes_Into_Basket, Put_Clothes_into_Washing_Machine (×2) | 5 |
| **BrainCo Hand** (强脑) | Collect_Plates_Into_Dishwasher, Make_The_Bed, Pickup_Pillow | 3 |

两种手的控制接口均为 **每手 6 DOF**（hand_state 共 12 维，0.0–1.0 表示张开→闭合），但物理外形不同。

### 1.2 数据格式

- LeRobot 格式：parquet (关节状态 30Hz) + MP4 视频 (640×480@30fps)
- 每个数据集包含多个 episode，每 episode 对应一段操作视频
- 关节状态：29 DOF body + 12 DOF hands = 41 DOF

### 1.3 URDF 与 Mesh

准备了 4 个 URDF 文件：

| URDF | 用途 |
|------|------|
| `g1_29dof_rev_1_0.urdf` | 基础 29DOF（标定用） |
| `g1_29dof_rev_1_0_with_inspire_hand_FTP.urdf` | 含 Inspire 手指的完整 URDF（主用） |
| `g1_29dof_with_hand_rev_1_0.urdf` | 含手的另一版本 |
| `human_arm_overlay.urdf` | 人手臂叠加用 |

STL mesh 共 91 个，包括 G1 各关节 link 和人体 capsule mesh。

---

## 二、相机标定

### 2.1 PSO 自动标定 (`auto_calibrate.py`)

使用粒子群优化 (PSO) 对鱼眼相机参数进行自动标定，优化目标为 FK 渲染 mask 与 GT mask 的 IoU。

**最终结果：IoU = 0.8970**

标定参数（已固化到 `config.py`）：
- 外参偏移 (相对 torso_link)：dx=0.039, dy=0.052, dz=0.536, pitch=-53.6°, yaw=4.7°, roll=3.0°
- 内参：fx=315, fy=302, cx=334, cy=230
- 畸变：k1=0.63, k2=0.17, k3=1.19, k4=0.25

### 2.2 其他标定尝试

- `auto_calibrate_v2.py` — 改进版自动标定
- `auto_calibrate_grad.py` — 基于梯度的标定方法
- `calibrate_keypoints.py` — 基于关键点的标定
- `interactive_calibrate.py` — GUI 交互标定（需显示器）

---

## 三、核心 Pipeline

### 3.1 正向运动学 (FK) + Mesh 渲染

**模块**：`video_inpaint.py`

流程：
1. `pinocchio.buildModelFromUrdf` 加载 URDF + STL mesh
2. 从 parquet 读取关节角度，`build_q()` 映射到 URDF 配置空间
3. `pin.forwardKinematics` + `pin.updateFramePlacements` 计算各 link 位姿
4. 将 mesh 顶点变换到世界坐标系
5. 通过 OpenCV 鱼眼模型（等距投影）投影到图像平面
6. 渲染 mask 或 overlay

### 3.2 SAM2 分割 (`sam2_inpaint_pipeline.py`)

主 pipeline：FK → SAM2 box prompt → LaMa inpaint

输出 7 个中间结果视频（便于调试）：
1. `original.mp4` — 原始视频
2. `fk_overlay.mp4` — FK mesh 叠加
3. `fk_mask.mp4` — FK 三角面 mask
4. `sam2_mask.mp4` — SAM2 分割 mask（按部位着色）
5. `sam2_overlay.mp4` — SAM2 mask 叠加
6. `final_mask.mp4` — 后处理二值 mask（平滑+膨胀+边缘模糊）
7. `inpaint.mp4` — LaMa 修复结果

### 3.3 视频修复 (`video_inpaint.py`)

逐帧处理：解码视频 → FK mask → 平滑 mask → LaMa 修复 → 编码输出

### 3.4 SAM2 分割实验 (`sam2_segment.py`)

支持 box 和 point 两种 prompt 模式的分割实验。

---

## 四、人体叠加

### 4.1 人手 Mesh 生成 (`generate_human_meshes.py`)

使用 capsule（圆柱+半球端盖）几何体生成人手臂各 link 的 STL mesh，包括：
- 上臂、前臂、手腕各关节
- 手指近端/远端、拇指各段

### 4.2 人手叠加渲染 (`render_human_overlay.py`)

使用 `human_arm_overlay.urdf`，将 G1 手臂关节角度直接映射到人手臂 mesh，在视频帧上渲染带光照的皮肤色手臂叠加。

### 4.3 SMPL-H 叠加实验 (`demo_smplh_overlay.py`)

SMPL-H 人体模型叠加的 demo 实验。

---

## 五、辅助工具

| 脚本 | 功能 |
|------|------|
| `config.py` | 集中配置：路径、相机参数、活跃任务/episodes、手类型检测 |
| `download_g1_wbt.sh` | 数据集批量下载（hf-mirror） |
| `inspect_g1_datasets.py` | 数据集统计检查 |
| `extract_sam2_frames.py` | 提取 SAM2 分割用的帧 |
| `downsample_videos.py` | 视频降采样 |
| `render_all_variants.py` | 渲染多种变体对比 |
| `render_front_compare.py` | 正面视角渲染对比 |
| `render_hand_debug.py` | 手部渲染调试 |

---

## 六、早期探索（已完成/归档）

Git 历史中记录的早期工作：

1. **Task 005**: Linux 环境搭建 (conda flip, ComfyUI, 依赖)
2. **Task 006**: LEVERB 数据集下载
3. **Task 007**: 模型文件下载
4. **Task 008**: 克隆 Cosmos Transfer 仓库
5. **Task 009**: Flux 数据合成 Pipeline（ComfyUI workflow）
6. **Task 012**: DepthAnything V2 替代 Pose 作为 ControlNet 条件
7. **Task 013**: DWPose 替代 SDPose 作为 ControlNet 条件
8. **规模化实验**: 52 个视频的 DWPose pipeline batch 处理

---

## 七、当前状态

- **活跃任务**: `G1_WBT_Brainco_Make_The_Bed`
- **活跃 episodes**: [0, 4, 50]
- **主 URDF**: `g1_29dof_rev_1_0_with_inspire_hand_FTP.urdf`（Inspire 手）
- **相机标定**: 已完成，IoU=0.8970
- **Pipeline 状态**: FK → SAM2 → LaMa 主流程已跑通
- **人体叠加**: capsule mesh + overlay 渲染已实现，SMPL-H 实验中
- **待解决**: Brainco 数据集使用 Inspire 手 URDF 的兼容性问题
