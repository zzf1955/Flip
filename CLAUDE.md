# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

第一人称人形机器人视频生成研究项目。核心思路是**反向数据构造**：在真实机器人视频上合成人手，得到（合成human, 真实robot）配对数据，用于微调 video-to-video 模型（Wan 2.2 + LoRA）。目标机器人为宇树 G1。

当前阶段聚焦于 pipeline 的**相机标定与 Mesh 渲染**：利用本体感知（关节编码器数据）和正向运动学，将 G1 的 3D Mesh 投影到自我视角相机帧上，生成精确的手臂分割 Mask。

## 环境

```bash
conda activate videoedit
```

## 关键依赖

- **pinocchio** — 从 URDF + 关节角度做正向运动学
- **numpy-stl** (`stl.mesh`) — 加载 STL 网格文件
- **OpenCV** (`cv2`) — 图像处理、鱼眼投影、渲染
- **pandas** — 读取 parquet 文件（LeRobot 数据集格式）
- **PyAV** (`av`) — 视频帧提取（精确 seek，优于 cv2.VideoCapture）

## 运行脚本

所有脚本独立运行，从项目根目录执行：
```bash
python scripts/render_g1_skeleton.py      # 俯视图骨骼 + Mesh 渲染
python scripts/render_front_view.py       # 正视图彩色 Mesh（区分各 link）
python scripts/interactive_calibrate.py   # GUI 滑动条交互标定（需要显示器）
python scripts/auto_calibrate.py          # PSO 自动标定（多进程）
python scripts/compare_render.py          # 凸包 vs 三角面片渲染对比
python scripts/sample_render.py           # 多任务/多 episode 批量 overlay 验证
```

输出到 `test_results/`。

## 数据布局

- `data/g1_urdf/` — G1 URDF（`g1_29dof_rev_1_0.urdf`）
- `data/unitree_ros/robots/g1_description/meshes/` — 各 link 的 STL 网格
- `data/unitree_model/` — 宇树官方模型仓库
- `data/g1_wbt/`, `data/g1_wbt_task2/`, `data/g1_wbt_task3/` — LeRobot 格式数据集
  - `videos/observation.images.head_stereo_left/chunk-*/file-*.mp4` — 自我视角视频
  - `data/chunk-*/file-*.parquet` — 关节状态（29 自由度）

## 代码架构

各脚本共享同一模式：
1. `pinocchio.buildModelFromUrdf` 加载 URDF + 加载各 link 的 STL Mesh
2. 从 parquet 读取目标帧的关节角度（29 DOF）
3. 正向运动学（`pin.forwardKinematics` + `pin.updateFramePlacements`）得到各 link 位姿
4. 将 Mesh 顶点变换到世界坐标系，通过鱼眼相机模型投影到图像
5. 在视频帧上渲染 Mask / Overlay

相机模型使用 OpenCV 鱼眼（等距投影），参数包括：相对于 `head_link` 的外参偏移（dx, dy, dz, pitch, yaw, roll）和内参（fx, fy, cx, cy, k1-k4）。标定通过优化 Mask IoU 完成。
