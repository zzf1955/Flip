# FLIP 代码架构（src/）

## 目录结构

```
src/
├── core/          8 个基础库模块（不可直接运行）
├── pipeline/      5 个可执行 pipeline 入口
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
```

### 模块说明

| 模块 | 行数 | 功能 | 主要导出 |
|------|------|------|----------|
| `config.py` | ~100 | 集中配置 | `BASE_DIR`, `G1_URDF`, `BEST_PARAMS`, `get_hand_type()`, `get_skip_meshes()` |
| `data.py` | ~120 | 数据加载 + 视频 IO | `load_episode_info()`, `open_video_writer()`, `write_frame()`, `close_video()`, `detect_keypoints_from_alpha()` |
| `camera.py` | ~200 | 相机模型 + 投影 | `CAMERA_MODELS`, `get_model()`, `build_K()`, `project_points_cv()`, `make_camera_const()`, `make_camera()` |
| `fk.py` | ~180 | URDF/mesh + FK | `parse_urdf_meshes()`, `preload_meshes()`, `load_robot()`, `build_q()`, `do_fk()` |
| `render.py` | ~250 | 渲染 | `render_mask()`, `render_overlay()`, `render_mask_and_overlay()`, `render_mesh_on_image()` |
| `mask.py` | ~80 | mask 后处理 + LaMa | `postprocess_mask()`, `grabcut_refine()`, `init_lama()`, `run_lama()` |
| `smplh.py` | ~660 | SMPLH + IK | `SMPLHForIK`, `IKSolver`, `extract_g1_targets()`, `R_SMPLH_TO_G1_NP` |
| `retarget.py` | ~430 | G1→SMPLH retarget | `retarget_frame()`, `refine_arms()`, `compute_g1_rest_transforms()`, `build_default_hand_pose()`, `scale_hands()` |

---

## 二、pipeline/ — 可执行 pipeline

| 脚本 | 功能 | 关键参数 | 输出路径 |
|------|------|----------|----------|
| `sam2_inpaint.py` | FK → SAM2 → LaMa/ProPainter | `--task`, `--episode`, `--duration`, `--device`, `--inpaint-method` | `output/inpaint/sam2_propainter/` |
| `sam2_segment.py` | SAM2 多部位分割 | `--episode`, `--duration`, `--mode box/point` | `output/inpaint/sam2_segment/` |
| `batch_inpaint.py` | 多 GPU 批量调度 | `--tasks`, `--gpus`, `--output-root` | 自定义 |
| `video_inpaint.py` | 逐帧 FK + GrabCut + LaMa | `--episode`, `--duration` | `output/inpaint/per_frame_lama/` |
| `retarget_video.py` | retarget 3-panel 视频 | `--task`, `--episode`, `--duration`, `--device` | `output/human/retarget_video/` |

### 运行方式

```bash
python -m src.pipeline.<script_name> [args]
```

---

## 三、tools/ — 实验/调试工具

### 相机标定 (calibration/)

| 脚本 | 功能 | 输出路径 |
|------|------|----------|
| `calibrate_mask.py` | PSO mask Dice 标定 | `output/calibration/mask_dice/` |
| `calibrate_keypoints.py` | PSO/Adam 关键点标定 | `output/calibration/kp_optim/` (可 `--output-dir` 覆盖) |
| `estimate_focal.py` | 焦距解析估计 | stdout |
| `distortion_analysis.py` | 畸变分析 | `output/tmp/distortion/` |
| `verify_extrinsics.py` | URDF 外参验证 | `output/tmp/urdf_verify/` |
| `verify_mesh.py` | STL/URDF 尺寸验证 | stdout |

### 人体 retarget (human/)

| 脚本 | 功能 | 输出路径 |
|------|------|----------|
| `retarget_diag.py` | retarget 9宫格诊断 | `output/human/retarget_diag/` |
| `render_smplh_ik.py` | SMPLH IK overlay | `output/human/smplh_ik/` |
| `render_ik_debug.py` | 第三人称 IK 调试 | `output/human/ik_debug/` |
| `debug_retarget.py` | retarget 误差可视化 | `output/human/debug_retarget/` |

### 渲染验证 (tmp/)

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

## 四、输出目录规范

所有输出统一到 `output/<stage>/<exp_name>/`：

```
output/
├── calibration/          # 相机标定实验
├── inpaint/              # 修复 pipeline
├── human/                # 人体 retarget
└── tmp/                  # 一次性验证
```

---

## 五、数据流

```
parquet (30Hz)     video (30fps)     URDF + STL
     │                  │                │
     ▼                  ▼                ▼
  build_q()      PyAV decode     parse_urdf_meshes()
     │                  │         preload_meshes()
     ▼                  │                │
  do_fk()              │                │
     │                  │                │
     ▼                  ▼                ▼
 transforms ──→ render_mask/overlay ──→ SAM2 prompt
                        │                │
                        ▼                ▼
                   postprocess      propagate
                        │                │
                        ▼                ▼
                    LaMa/ProPainter  ──→ inpaint.mp4
```
