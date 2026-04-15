# FLIP 项目进展总结

## 项目目标

FLIP (Flipped-Direction Learning via Inpainting Pipeline) — 在真实 G1 机器人第一人称视频上合成人体，得到（合成 human, 真实 robot）配对数据，用于微调 video-to-video 模型。

核心流程：**FK Mesh 渲染 → 机器人分割 → 背景修复 → 人体叠加**

---

## 一、数据准备

### 1.1 数据集

从 HuggingFace (`unitreerobotics`) 下载了 **8 个** G1 WBT 数据集：

| 手类型 | 数据集 | 数量 |
|--------|--------|------|
| **Inspire Hand** | Collect_Clothes, Pickup_Pillow, Put_Clothes_Into_Basket, Put_Clothes_into_Washing_Machine (×2) | 5 |
| **BrainCo Hand** | Collect_Plates_Into_Dishwasher, Make_The_Bed, Pickup_Pillow | 3 |

- LeRobot 格式：parquet (30Hz, 29 DOF body + 12 DOF hands) + MP4 (640×480@30fps)
- **注意**：`videos_16fps/` 降采样版本与 state 帧对应错误，所有代码仅使用原始 30fps

### 1.2 URDF 与 Mesh

- 主 URDF：`g1_29dof_rev_1_0_with_inspire_hand_FTP.urdf`（含 Inspire 手指，91 个 STL mesh）
- BrainCo 任务跳过手部 mesh 渲染（`config.py:get_skip_meshes()`）

---

## 二、相机标定

### 2.1 相机型号确认

头部相机为**外置 UVC 双目 RGB 广角相机**（非 D435i），125° FOV，DECXIN Cherry Dual Camera。

### 2.2 畸变分析

Canny + HoughLinesP 分析确认 edge/center 弯曲度比仅 1.12x，**pinhole 模型合适，k1-k4=0**。

### 2.3 内参确定

| 参数 | 值 | 验证 |
|------|-----|------|
| fx | **290.78** | PSO + 独立 LS 95% CI [284, 297]，HFOV≈95.5° |
| fy | **287.35** | 与 VFOV=80° 理论值吻合（误差 0.5%） |
| cx | **≈320** | 接近图像中心 |
| k1-k4 | **= 0** | 畸变可忽略 |

### 2.4 标定实验

当前使用模型：`pinhole_fixed`（7参数，固定 fx/fy/cx）

| 实验 | 方法 | 指标 | 输出 |
|------|------|------|------|
| mask Dice | PSO 200粒子×100迭代 | F1=0.948 | `output/calibration/mask_dice/` |
| 4关键点 | PSO (L_thumb, L_toe, R_toe, R_thumb) | RMSE=16.1px | `output/calibration/kp_4points/` |
| 5关键点 | PSO (+L_wrist) | RMSE=16.99px | `output/calibration/kp_5points/` |

### 2.5 参数退化问题

| 耦合对 | 相关系数 | 影响 |
|--------|---------|------|
| pitch ↔ cy | r = -0.95 | 几乎完全退化 |
| yaw ↔ cx | r = +0.91 | 退化 |
| dz ↔ fx | r = +0.77 | 退化 |

当前最优参数（`config.py`）：
```
dx=0.0758, dy=0.0226, dz=0.4484, pitch=-61.59°, yaw=2.17°, roll=0.23°
```

---

## 三、Inpaint Pipeline

### 3.1 主 pipeline（`pipeline/sam2_inpaint.py`）

FK → SAM2 box prompt → LaMa/ProPainter，7 部位独立跟踪（左/右 arm/hand/leg + torso）。

输出 7 种中间结果：original, fk_overlay, fk_mask, sam2_mask, sam2_overlay, final_mask, inpaint。

```bash
python -m src.pipeline.sam2_inpaint --episode 0 --start 5 --duration 5 \
  --task G1_WBT_Inspire_Pickup_Pillow_MainCamOnly --device cuda:1
```

E2E 验证通过 → `output/inpaint/sam2_propainter/`

### 3.2 逐帧 pipeline（`pipeline/video_inpaint.py`）

FK → GrabCut → postprocess → LaMa，输出 4 种视频。

### 3.3 SAM2 分割实验（`pipeline/sam2_segment.py`）

支持 box/point 两种 prompt 模式。E2E 验证通过 → `output/inpaint/sam2_segment/`

### 3.4 多 GPU 批量（`pipeline/batch_inpaint.py`）

队列调度，自动分配 GPU，per-task 日志。

---

## 四、人体 Retarget

### 4.1 SMPLH 模型

6,890 顶点，52 关节，LBS 蒙皮。坐标系：SMPLH (X=left, Y=up, Z=fwd) ↔ G1 (X=fwd, Y=left, Z=up)。

### 4.2 Retarget 算法（`core/retarget.py`）

纯关节旋转拷贝 G1→SMPLH，无全局 IK：
1. 中性姿态 FK → rest-pose 旋转
2. 目标帧 FK → 当前旋转
3. 世界坐标系旋转 delta
4. 链式关节映射：bone direction alignment + spine 3等分 + shoulder twist + 直接旋转拷贝
5. 手指弯曲：hand_state → SMPLH 45-dim hand_pose
6. 可选 IK arm refinement（L-BFGS，优化 collar/shoulder/elbow 匹配 thumb target）

E2E 验证：
- 9宫格诊断 → `output/human/retarget_diag/`
- 3-panel 视频 → `output/human/retarget_video/`
- SMPLH IK overlay → `output/human/smplh_ik/`
- 误差可视化：mean d(G1,SMPLH) = 17.07px, 3D = 71.40mm

### 4.3 尺寸不匹配

| 部位 | G1 | 人体 (SMPLH) | 比值 |
|------|-----|-------------|------|
| 身高 | 1.32m | 1.72m | 0.77x |
| 臂长 | 0.27m | 0.51m | 0.53x |
| 手长 | 107mm | 164mm | 0.65x |

通过 `scale=0.75` + `hand_scale=1.3` 部分补偿。

---

## 五、代码重构

### 5.1 架构变更

从 `scripts/`（25 个文件平铺）重构为 `src/`（3 层结构）：

| 层级 | 文件数 | 职责 |
|------|--------|------|
| `src/core/` | 8 | 基础库（config, data, camera, fk, render, mask, smplh, retarget） |
| `src/pipeline/` | 5 | 可执行 pipeline 入口 |
| `src/tools/` | 17 | 实验/调试/可视化 |

### 5.2 关键改进

- **消除 god module**：`video_inpaint.py` (710行) 拆分为 5 个模块
- **消除代码重复**：`load_episode_info` 等 4 个函数从 3 处重复减为 1 处定义
- **统一渲染**：4 处独立渲染实现合并到 `core/render.py`
- **统一输出路径**：`test_results/` (无组织) → `output/<stage>/<exp_name>/`

### 5.3 E2E 验证

全部 30 个脚本 import 测试通过。按 4 个阶段重跑验证：

| Stage | 脚本数 | 状态 |
|-------|--------|------|
| tmp/ (一次性验证) | 6 | ✓ (4 通过，2 依赖标定推后) |
| calibration/ (标定) | 3 | ✓ |
| inpaint/ (修复) | 4 | ✓ |
| human/ (retarget) | 5 | ✓ |

---

## 六、当前状态

- **活跃任务**: `G1_WBT_Inspire_Pickup_Pillow_MainCamOnly`（Inspire 手，单摄像头）
- **活跃 episodes**: [0, 4, 50]
- **相机模型**: pinhole_fixed（7参数）
- **Pipeline**: FK → SAM2 → LaMa/ProPainter ✓
- **Retarget**: 关节拷贝 + IK arm refinement ✓
- **代码结构**: `src/` 三层架构 ✓
- **输出规范**: `output/<stage>/<exp_name>/` ✓

---

## 七、待解决

1. **BrainCo 手部 mesh**：集成 BrainCo Revo2 URDF + STL
2. **Human 渲染质量**：SMPLH mesh → ControlNet/深度图重绘
3. **视频连贯性**：时序一致性 + 遮挡处理
4. **下游训练**：Wan 2.1 + LoRA 微调
