# CLAUDE.md

用中文回答

除非用户要求，否则不要直接看看视频和图片。简单的视觉任务请使用 python 代码操作，而不是直接看图片和视频。

Notion MCP (`mcp__notion__API-*`) 是 Notion REST API 的透传，ToolSearch 返回的 schema 不必严格遵守；按 Notion 官方 API 的参数结构调用即可。

## 项目概述

FLIP: Flipped-Direction Learning via Inpainting Pipeline for Cross-Embodiment Video Editing

第一人称人形机器人视频生成研究项目。核心思路是**反向数据构造**：在真实机器人视频上合成人体，得到（合成human, 真实robot）配对数据，用于微调 video-to-video 模型（Wan 2.1 + LoRA）。目标机器人为宇树 G1。

### Pipeline 概览

```
G1 第一人称视频 + 关节编码器数据
  │
  ├─ Step 1  Pose 获取：精确本体感知 ✓
  ├─ Step 2  Robot 分割+去除：FK → Mesh → SAM2 mask → 背景 inpaint ✓
  ├─ Step 3  运动学映射：Robot → Human pose (关节拷贝 + IK 微调) ✓
  ├─ Step 4  Human 渲染：SMPLH mesh → ControlNet 重绘 (进行中)
  └─ Step 5  (合成 human, 真实 robot) → Wan 2.1 + LoRA
```

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
  python -m src.pipeline.<script>
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
- **OpenCV** (`cv2`) -- 图像处理、投影
- **pandas** -- parquet (LeRobot 数据集格式)
- **PyAV** (`av`) -- 视频帧提取
- **sam2** -- SAM2 视频分割
- **simple_lama_inpainting** -- LaMa 修复
- **torch** -- SMPLH IK 求解、可微标定

## 代码结构

### 目录布局

```
src/
├── core/                    # 基础库模块（不可直接运行）
│   ├── config.py            # 集中配置：路径、相机参数、任务选择
│   ├── data.py              # 数据加载：episode/parquet + 视频 IO
│   ├── camera.py            # 相机模型 + OpenCV 投影 + make_camera
│   ├── fk.py                # URDF/mesh 加载 + build_q + FK
│   ├── render.py            # mask/overlay/Lambertian 渲染
│   ├── mask.py              # mask 后处理 + LaMa/GrabCut
│   ├── smplh.py             # SMPLH 模型 + IK 求解器
│   └── retarget.py          # G1→SMPLH retarget 算法
│
├── pipeline/                # 可执行主 pipeline
│   ├── sam2_inpaint.py      # FK → SAM2 → LaMa/ProPainter（主入口）
│   ├── sam2_segment.py      # SAM2 多部位分割实验
│   ├── batch_inpaint.py     # 多 GPU 批量调度
│   ├── video_inpaint.py     # 逐帧 FK + GrabCut + LaMa
│   └── retarget_video.py    # retarget 3-panel 视频渲染
│
└── tools/                   # 实验/调试/可视化工具
    ├── calibrate_mask.py    # PSO mask Dice 标定
    ├── calibrate_keypoints.py  # PSO/Adam 关键点标定
    ├── estimate_focal.py    # 焦距解析估计
    ├── distortion_analysis.py  # 畸变分析
    ├── verify_extrinsics.py # URDF 外参验证
    ├── verify_mesh.py       # STL/URDF 尺寸验证
    ├── render_3view.py      # G1 三视图渲染
    ├── render_overlay_check.py  # 多视频 overlay 泛化验证
    ├── render_lit_overlay.py   # Lambertian overlay
    ├── render_smplh_ik.py   # SMPLH IK overlay
    ├── render_ik_debug.py   # 第三人称 IK 调试
    ├── debug_keypoints.py   # 关键点可视化
    ├── debug_retarget.py    # retarget 误差可视化
    ├── retarget_diag.py     # retarget 9宫格诊断
    ├── demo_mesh_scale.py   # mesh 缩放对比
    └── svg2gif.py           # SVG→GIF 转换
```

旧代码保留在 `scripts/`（已归档，不再使用）。

### 依赖关系

```
core/config.py
  ├── core/data.py          (episode 加载, 视频 IO)
  ├── core/camera.py        (相机模型, 投影, make_camera)
  ├── core/fk.py            (URDF, mesh, build_q, do_fk)
  ├── core/render.py        (mask, overlay, Lambertian)
  ├── core/mask.py          (后处理, LaMa)
  ├── core/smplh.py         (SMPLH + IK)
  └── core/retarget.py      (G1→SMPLH 映射)
```

### 配置切换

编辑 `src/core/config.py` 中的 `ACTIVE_TASK` 和 `ACTIVE_EPISODES`。

## 输出目录规范

所有实验输出到 `output/<stage>/<exp_name>/`：

```
output/
├── calibration/             # 相机标定
│   ├── kp_4points/          # 4关键点 PSO
│   ├── kp_5points/          # 5关键点 PSO
│   └── mask_dice/           # mask Dice PSO
├── inpaint/                 # 修复 pipeline
│   ├── sam2_propainter/     # SAM2 + ProPainter
│   └── sam2_segment/        # SAM2 分割
├── human/                   # 人体 retarget
│   ├── retarget_video/      # 3-panel 视频
│   ├── retarget_diag/       # 9宫格诊断
│   ├── smplh_ik/            # IK overlay
│   ├── ik_debug/            # 第三人称调试
│   └── debug_retarget/      # 误差可视化
└── tmp/                     # 一次性验证
    ├── 3view/               # G1 三视图
    ├── distortion/          # 畸变分析
    ├── mesh_scale/          # mesh 缩放
    └── urdf_verify/         # URDF 外参
```

## 数据布局

```
data/
├── mesh/                              # URDF + STL
│   ├── g1_29dof_rev_1_0_with_inspire_hand_FTP.urdf
│   ├── human_arm_overlay.urdf
│   └── meshes/                        # STL files (91+)
└── unitree_G1_WBT/                    # 8 个 LeRobot 格式任务数据集
    └── <task_name>/
        ├── meta/episodes/chunk-000/   # episode → 视频文件映射
        ├── data/chunk-000/            # parquet (关节状态 30Hz)
        └── videos/observation.images.head_stereo_left/
            └── chunk-000/             # MP4 视频 640x480@30fps（原始，勿用 16fps）
```

**注意**：`videos_16fps/` 目录的降采样视频与 parquet state 的帧对应关系**全是错的**，所有脚本仅使用原始 30fps 视频。

## 相机模型

使用 pinhole 模型（畸变已确认可忽略，k1-k4=0）。头部相机为外置 UVC 双目 RGB（非 D435i），125° FOV。

当前最优参数（`config.py:BEST_PARAMS_BY_MODEL`）：
- 外参：dx=0.0758, dy=0.0226, dz=0.4484, pitch=-61.59°, yaw=2.17°, roll=0.23°
- 内参：fx=290.78, fy=287.35, cx≈320, cy≈314
- 模型选择：`CAMERA_MODEL = "pinhole_fixed"`（7参数，固定 fx/fy/cx）

## hand_state 编码规则（重要）

两种灵巧手的 hand_state（12维，每手6维）**编码方向相反**：

| 手型 | 0.0 | 1.0 | 原因 |
|------|-----|-----|------|
| **Inspire RH56DFTP** | 闭合 (closed) | 张开 (open) | 硬件 ANGLE: 0=弯曲, 1000=张开 |
| **BrainCo Revo2** | 张开 (open) | 闭合 (closed) | normalize 时多了 `1.0 -` 反转 |

`core/fk.py:build_q()` 内部统一处理。详见 `doc/hand_data_mapping.md`。

## 已知问题

### 手部 Mesh 与数据集不匹配

仅有 Inspire 手部 URDF，BrainCo 任务跳过手部渲染（`config.py:get_skip_meshes()`）。
**待解决**：集成 BrainCo Revo2 URDF + STL。

## 运行示例

```bash
# 主 pipeline (FK → SAM2 → LaMa，5秒)
python -m src.pipeline.sam2_inpaint --episode 0 --start 5 --duration 5 \
  --task G1_WBT_Inspire_Pickup_Pillow_MainCamOnly --device cuda:1

# SAM2 分割实验
python -m src.pipeline.sam2_segment --episode 0 --start 5 --duration 5 --mode box

# 多 GPU 批量修复
python -m src.pipeline.batch_inpaint --tasks G1_WBT_Inspire_Pickup_Pillow_MainCamOnly \
  G1_WBT_Inspire_Put_Clothes_Into_Basket --gpus 1 2 3 --output-root output/inpaint/v1

# Retarget 视频（3-panel：原始 | G1 | SMPLH）
python -m src.pipeline.retarget_video --task G1_WBT_Inspire_Pickup_Pillow_MainCamOnly \
  --episode 0 --duration 5 --device cuda:2

# 相机标定（4关键点 PSO）
python -m src.tools.calibrate_keypoints --optimizer pso --keypoints L_thumb,L_toe,R_toe,R_thumb \
  --output-dir output/calibration/kp_4points

# retarget 诊断（单帧 9宫格）
python -m src.tools.retarget_diag --episode 0 --frame 30 \
  --task G1_WBT_Inspire_Pickup_Pillow_MainCamOnly --device cuda:2
```

输出到 `output/<stage>/<exp_name>/`。
