# FLIP 代码架构（src/）

## 目录结构

```
src/
├── core/          9 个基础库模块（不可直接运行）
├── pipeline/     18 个可执行 pipeline 入口
└── tools/        17 个实验/调试/可视化工具
```

旧代码保留在 `scripts/`（已归档）。

---

## 一、core/ — 基础库模块

### 依赖关系

```
config.py          ← 无依赖
data.py            ← config
camera.py          ← config
fk.py              ← config, pinocchio, stl
render.py          ← camera
mask.py            ← (独立)
smplh.py           ← config, torch, pinocchio
retarget.py        ← config, smplh, fk
eval_metrics.py    ← torch, skimage, transformers (CLIP)
```

### 模块说明

| 模块 | 功能 | 主要导出 |
|------|------|----------|
| `config.py` | 集中配置：路径、相机参数、任务选择 | `MAIN_ROOT`, `DATA_ROOT`, `TRAINING_DATA_ROOT`, `PAIR_DIR`, `BEST_PARAMS` |
| `data.py` | 数据加载 + 视频 IO | `load_episode_info()`, `open_video_writer()`, `write_frame()` |
| `camera.py` | 相机模型 + 投影 | `make_camera()`, `project_points_cv()` |
| `fk.py` | URDF/mesh + FK | `load_robot()`, `build_q()`, `do_fk()` |
| `render.py` | mask/overlay/Lambertian 渲染 | `render_mask()`, `render_overlay()` |
| `mask.py` | mask 后处理 + LaMa | `postprocess_mask()`, `init_lama()`, `run_lama()` |
| `smplh.py` | SMPLH 模型 + IK 求解器 | `SMPLHForIK`, `IKSolver` |
| `retarget.py` | G1→SMPLH retarget | `retarget_frame()`, `refine_arms()` |
| `eval_metrics.py` | 训练在线评估指标 (PSNR/SSIM/LPIPS/CLIP Score) | `OnlineMetrics` |

---

## 二、pipeline/ — 可执行 pipeline

### 数据生成与预处理

| 脚本 | 功能 | 输入 | 输出 |
|------|------|------|------|
| `segment_episodes.py` | 原始 episode 切分为 4s segment | LeRobot 数据集 | `training_data/segment/` |
| `seedance_gen.py` | Volcengine Seedance 2.0 API 生成人体视频 | robot 视频 | `training_data/seedance_direct/4s/` |
| `seedance_batch.py` | seedance_gen 的批量包装 | 多个 robot 视频 | 同上 |
| `seedance_clip.py` | 4s Seedance 视频切成 1s/2s 片段 | `seedance_direct/4s/` | `seedance_direct/{1s,2s}/` |

### 分割与修复

| 脚本 | 功能 | 输入 | 输出 |
|------|------|------|------|
| `sam2_inpaint.py` | FK → SAM2 分割 → LaMa/ProPainter 修复 | episode 视频 | `output/inpaint/` |
| `sam2_segment.py` | SAM2 多部位分割实验 | episode 视频 | `output/inpaint/sam2_segment/` |
| `sam2_precompute.py` | SAM2 mask 预计算（FK bbox prompt → SAM2 propagation → npz） | segment 视频 | `training_data/sam2_mask/` |
| `batch_sam2_precompute.py` | sam2_precompute 多 GPU 调度（多 worker/GPU） | 多 task | `training_data/sam2_mask/` |
| `batch_inpaint.py` | 多 GPU 批量修复调度 | 多 task/episode | 自定义 |
| `video_inpaint.py` | 逐帧 FK + GrabCut + LaMa | episode 视频 | `output/inpaint/per_frame_lama/` |

### 渲染与 Overlay

| 脚本 | 功能 | 输入 | 输出 |
|------|------|------|------|
| `segment_pipeline.py` | **主 pipeline**：FK→SAM2→inpaint→SMPLH retarget→overlay | segment 视频 | `training_data/overlay/4s/` |
| `human_overlay.py` | SMPLH mesh 叠加到修复背景 | 修复视频 + retarget 数据 | overlay MP4 |
| `retarget_video.py` | 3-panel 对比视频 [原始\|G1\|SMPLH] | episode 视频 | `output/human/retarget_video/` |

### 训练数据配对

| 脚本 | 功能 | 输入 | 输出 |
|------|------|------|------|
| `make_pair.py` | 匹配 robot+human 视频，重采样 16fps，4k+1 帧 | segment + seedance/overlay | `training_data/pair/{1s,2s,4s}/` |
| `robot_patch.py` | 全身降质数据（FK mesh 或 SAM2 mask → blur/noise/mean） | segment + parquet/sam2_mask | `training_data/pair/1s_patch/` |

### LoRA 训练

| 脚本 | 功能 | 输入 | 输出 |
|------|------|------|------|
| `train_lora.py` | 自写 Wan 2.1 LoRA 训练（train/eval split + loss 日志 + eval 视频） | `output/data_cache_*/*.pth` | `training_data/log/<date>/` |

### 人体重绘（Step 4 本地方案）

| 脚本 | 功能 | 输入 | 输出 |
|------|------|------|------|
| `cosmos_prepare.py` | 前置：composite + depth + mask + spec.json（两种 regen 共用） | overlay 视频 | `output/human/cosmos_prepare/` |
| `wan_regen.py` | **主方案**：ComfyUI + Wan 2.1 VACE 1.3B depth+mask 重绘 | cosmos_prepare 输出 | `output/human/wan_regen/` |
| `cosmos_regen.py` | Cosmos Transfer 2.5 推理（**已弃用**，保留作对照存档） | cosmos_prepare 输出 | `output/human/cosmos_regen/` |

详见 `doc/step_4_wan_vace_regen.md`。

### 运行方式

```bash
LD_PRELOAD=/home/leadtek/miniconda3/envs/flip/lib/libjpeg.so.8 \
  python -m src.pipeline.<script_name> [args]
```

GPU / CUDA 命令优先走统一入口，便于 Codex 按子命令保存越权批准规则：

```bash
scripts/flip_run.sh mitty_cache --cuda 0 -- <mitty_cache args>
scripts/flip_run.sh sam2_precompute --cuda 0 -- <sam2_precompute args>
scripts/flip_run.sh train_mitty --cuda 2,3 --nproc 2 -- <train_mitty args>
scripts/flip_run.sh nvidia-smi
```

当前 Codex 可使用 `danger-full-access` 直接访问 GPU；`scripts/flip_run.sh` 仍作为统一环境与白名单入口保留。Bash 高危命令由 `scripts/codex_pre_tool_use_guard.py` 通过 Codex `PreToolUse` hook 做最佳努力拦截。

---

## 三、tools/ — 实验/调试工具

### 相机标定

| 脚本 | 功能 | 输出路径 |
|------|------|----------|
| `calibrate_mask.py` | PSO mask Dice 标定 | `output/calibration/mask_dice/` |
| `calibrate_keypoints.py` | PSO/Adam 关键点标定 | `output/calibration/kp_optim/` |
| `estimate_focal.py` | 焦距解析估计 | stdout |
| `distortion_analysis.py` | 畸变分析 | `output/tmp/distortion/` |
| `verify_extrinsics.py` | URDF 外参验证 | `output/tmp/urdf_verify/` |
| `verify_mesh.py` | STL/URDF 尺寸验证 | stdout |

### 人体 retarget

| 脚本 | 功能 | 输出路径 |
|------|------|----------|
| `retarget_diag.py` | retarget 9宫格诊断 | `output/human/retarget_diag/` |
| `render_smplh_ik.py` | SMPLH IK overlay | `output/human/smplh_ik/` |
| `render_ik_debug.py` | 第三人称 IK 调试 | `output/human/ik_debug/` |
| `debug_retarget.py` | retarget 误差可视化 | `output/human/debug_retarget/` |

### 渲染验证

| 脚本 | 功能 | 输出路径 |
|------|------|----------|
| `render_3view.py` | G1 三视图渲染 | `output/tmp/3view/` |
| `render_overlay_check.py` | 多视频 overlay 泛化 | `output/tmp/overlay_check/` |
| `render_lit_overlay.py` | Lambertian overlay | `output/tmp/lit_overlay/` |
| `demo_mesh_scale.py` | mesh 缩放对比 | `output/tmp/mesh_scale/` |
| `debug_keypoints.py` | 关键点可视化 | `output/tmp/kp_debug/` |

### 工具

| 脚本 | 功能 |
|------|------|
| `svg2gif.py` | SVG→GIF 转换（独立） |

### 运行方式

```bash
python -m src.tools.<script_name> [args]
```

---

## 四、数据流

### 完整 Pipeline：原始视频 → 训练好的 LoRA

```
G1 第一人称视频 (LeRobot dataset, 30fps)
│
├─ [segment_episodes.py]
│   → training_data/segment/<task>/ep*/seg*_video.mp4  (4s@30fps, 28K 文件, 19GB)
│
├─ [segment_pipeline.py]  (FK → SAM2 → inpaint → SMPLH retarget → overlay)
│   → training_data/overlay/4s/<task>/ep*/seg*_human.mp4
│
├─ [seedance_gen.py / seedance_batch.py]  (Volcengine API: robot → human)
│   → training_data/seedance_direct/4s/<task>/ep*/seg*_human.mp4
│   │
│   └─ [seedance_clip.py]  (4s → 1s/2s clips)
│       → training_data/seedance_direct/{1s,2s}/<task>/ep*/seg*_clip*.mp4
│
├─ [make_pair.py]  (匹配 robot+human, 重采样 16fps, 4k+1 帧, 统一编号)
│   → training_data/pair/1s/
│       ├── video/pair_NNNN.mp4        (robot, 17帧@16fps)
│       ├── control_video/pair_NNNN.mp4 (human, 17帧@16fps)
│       └── metadata.csv
│
├─ [sam2_precompute.py / batch_sam2_precompute.py]  (FK bbox → SAM2 pixel mask)
│   → training_data/sam2_mask/<task>/ep*/seg*.npz  (masks: uint8 (120,480,640))
│
├─ [robot_patch.py]  (全身降质: FK/SAM2 mask → blur/noise/mean)
│   → training_data/pair/1s_patch/
│       ├── video/pair_NNNN.mp4        (clean robot, 17帧@16fps)
│       ├── control_video/pair_NNNN.mp4 (degraded robot, 17帧@16fps)
│       ├── patch/pair_NNNN.pth        (latent mask + weights)
│       └── metadata.csv
│
├─ [DiffSynth data_process]  (阶段 1: T5+CLIP+VAE embedding 缓存)
│   → output/data_cache_80/0/*.pth  (80 样本, ~50MB/个, 共 3.9GB)
│
└─ [train_lora.py]  (阶段 2: 自写训练循环)
    → training_data/log/YYYY-MM-DD_HHMMSS/
        ├── ckpt/step-NNNN.safetensors  (LoRA 权重, ~147MB)
        ├── eval/step-NNNN/             (gt + ctrl + gen 视频)
        └── train.log                   (每步 loss + eval)
```

### 训练数据格式要求

| 参数 | 要求 |
|------|------|
| 分辨率 | 640×480 |
| 帧率 | 16 fps |
| 帧数 | **4k+1**（1s=17帧, 2s=33帧, 4s=65帧） |
| 编码 | H.264, yuv420p |
| Prompt | `A first-person view robot arm performing household tasks flip_v2v` |

### 当前数据规模

| 阶段 | 数量 | 大小 |
|------|------|------|
| Segment (4s robot) | 28,548 | 19 GB |
| Overlay (4s human) | 53 | 64 MB |
| Seedance (1s human) | 80 | 15 MB |
| Training pairs (1s) | 120 对 | 37 MB |
| Cached embeddings | 80 | 3.9 GB |

---

## 五、输出目录规范

```
output/                          # per-worktree 实验产物
├── calibration/                 # 相机标定
├── inpaint/                     # SAM2 + 修复
├── human/                       # retarget + overlay
│   ├── retarget_video/
│   ├── cosmos_prepare/
│   └── cosmos_regen/
├── segment_pipeline/            # 主 pipeline 中间产物
├── data_cache_80/               # DiffSynth embedding 缓存
├── lora_funcontrol_*/           # DiffSynth baseline 训练输出
└── tmp/                         # 一次性验证

training_data/                   # per-worktree 训练数据
├── segment/                     # 4s robot segments
├── sam2_mask/                   # SAM2 预计算 mask (sam2_precompute 输出)
│   └── <task>/ep*/seg*.npz      # masks: uint8 (120, 480, 640), 0/255
├── seedance_direct/             # Seedance human videos
├── overlay/                     # SMPLH overlay human videos
├── pair/                        # 配对训练数据
│   ├── 1s/{video/, control_video/, metadata.csv}  (make_pair 输出)
│   └── 1s_patch/{video/, control_video/, patch/, metadata.csv}  (robot_patch 输出)
├── compare/                     # 对比视频
└── log/                         # train_lora.py 训练输出
    └── YYYY-MM-DD_HHMMSS/{ckpt/, eval/, train.log}
```
