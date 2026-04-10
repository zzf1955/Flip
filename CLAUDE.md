# CLAUDE.md

用中文回答

## 项目概述

FLIP: Flipped-Direction Learning via Inpainting Pipeline for Cross-Embodiment Video Editing

第一人称人形机器人视频生成研究项目。核心思路是**反向数据构造**：在真实机器人视频上合成人体，得到（合成human, 真实robot）配对数据，用于微调 video-to-video 模型（Wan 2.1 + LoRA）。目标机器人为宇树 G1。

当前阶段聚焦于 **FK Mesh 渲染 → SAM2 分割 → LaMa 修复**，以及**人体叠加**。

## 环境

### 基础
- OS: Ubuntu 24.04.2 LTS
- GPU: 4x NVIDIA RTX 4090 D (24 GB each)
- CUDA: 12.8 (V12.8.61), Driver 570.133.07
- Conda env: `flip` (`conda activate flip`)
- Python: 3.10, PyTorch 2.11.0+cu128

### 运行脚本
```bash
LD_PRELOAD=/home/leadtek/miniconda3/envs/flip/lib/libjpeg.so.8 \
  no_proxy=localhost,127.0.0.1 \
  python scripts/<script>.py
```

### 网络 / 代理
- Clash 代理端口: 20171
- `export http_proxy=http://127.0.0.1:20171`
- `export https_proxy=http://127.0.0.1:20171`

### GPU 分配
- GPU 0: ComfyUI (端口 8001, `/disk_n/zzf/ComfyUI`)
- GPU 1-3: 脚本使用 (`--device cuda:2` 等)

### 缓存路径 (根分区仅 ~14GB, 所有缓存必须放 /disk_n/zzf/)
- HuggingFace: `/disk_n/zzf/.cache/huggingface` (HF_HOME)
- pip: `/disk_n/zzf/.pip_cache`
- uv (Cosmos): `/disk_n/zzf/.cache/uv`

### Git
- committer: zzf621

## 关键依赖

- **pinocchio** -- URDF + 正向运动学
- **numpy-stl** (`stl.mesh`) -- STL mesh 加载
- **OpenCV** (`cv2`) -- 图像处理、鱼眼投影
- **pandas** -- parquet (LeRobot 数据集格式)
- **PyAV** (`av`) -- 视频帧提取
- **sam2** -- SAM2 视频分割
- **simple_lama_inpainting** -- LaMa 修复

## 项目结构

### 核心脚本 (scripts/)

| 脚本 | 功能 |
|------|------|
| `config.py` | 集中配置: 路径、相机参数、活跃任务/episodes |
| `video_inpaint.py` | 核心工具模块: FK、渲染、mask、LaMa、视频 IO |
| `sam2_inpaint_pipeline.py` | **主 pipeline**: FK -> SAM2 box prompt -> LaMa inpaint |
| `sam2_segment.py` | SAM2 分割实验 (box/point 模式) |
| `render_human_overlay.py` | 人手 mesh 叠加渲染 |
| `generate_human_meshes.py` | 人手 STL capsule 生成 |
| `auto_calibrate.py` | PSO 相机自动标定 (已完成, IoU=0.8970) |
| `interactive_calibrate.py` | GUI 交互标定 (需要显示器) |
| `download_g1_wbt.sh` | 数据集批量下载 (hf-mirror) |
| `inspect_g1_datasets.py` | 数据集统计检查 |

### 配置切换

编辑 `scripts/config.py` 中的 `ACTIVE_TASK` 和 `ACTIVE_EPISODES`。

### 数据布局

```
data/
├── mesh/                              # URDF + STL
│   ├── g1_29dof_rev_1_0_with_inspire_hand_FTP.urdf
│   ├── human_arm_overlay.urdf
│   └── meshes/                        # STL files (51+)
└── unitree_G1_WBT/                    # 8 个 LeRobot 格式任务数据集
    └── <task_name>/
        ├── meta/episodes/chunk-000/   # episode -> 视频文件映射
        ├── data/chunk-000/            # parquet (关节状态 30Hz)
        └── videos/observation.images.head_stereo_left/
            └── chunk-000/             # MP4 视频 640x480@30fps
```

## 代码架构

各脚本共享同一模式:
1. `pinocchio.buildModelFromUrdf` 加载 URDF + 加载各 link 的 STL Mesh
2. 从 parquet 读取目标帧的关节角度 (29 DOF body + 12 DOF hands)
3. 正向运动学 (`pin.forwardKinematics` + `pin.updateFramePlacements`) 得到各 link 位姿
4. 将 Mesh 顶点变换到世界坐标系，通过鱼眼相机模型投影到图像
5. 在视频帧上渲染 Mask / Overlay

相机模型使用 OpenCV 鱼眼 (等距投影), 参数包括: 相对于 `torso_link` 的外参偏移 (dx, dy, dz, pitch, yaw, roll) 和内参 (fx, fy, cx, cy, k1-k4)。

## 运行示例

```bash
# 主 pipeline (FK -> SAM2 -> LaMa)
python scripts/sam2_inpaint_pipeline.py --episode 4 --start 5 --duration 5

# SAM2 分割实验
python scripts/sam2_segment.py --episode 4 --start 5 --duration 5 --mode box

# 人手叠加
python scripts/render_human_overlay.py --episode 0 --frame 30 --side-by-side
```

输出到 `test_results/`。
