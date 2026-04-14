# FLIP 脚本清单与分类

## 依赖关系

```
config.py  ←── 所有现代脚本的配置中心
  │
  └─→ video_inpaint.py  ←── 核心工具库 (FK, 渲染, mask, LaMa, 视频 IO)
        │
        ├─→ sam2_inpaint_pipeline.py   (主 pipeline)
        ├─→ sam2_segment.py            (SAM2 分割实验)
        ├─→ auto_calibrate_v2.py       (PSO 标定 v2)
        │     ├─→ auto_calibrate_grad.py   (梯度标定)
        │     └─→ calibrate_keypoints.py   (关键点标定)
        ├─→ render_human_overlay.py    (人手叠加)
        ├─→ demo_smplh_overlay.py      (SMPLH 叠加)
        ├─→ render_hand_debug.py       (手部调试)
        └─→ render_front_compare.py    (三视图对比)

独立脚本 (不依赖 config/video_inpaint):
  auto_calibrate.py        ← 废弃，硬编码旧路径
  interactive_calibrate.py ← 废弃，硬编码旧路径
  render_all_variants.py   ← 独立，依赖 /tmp/unitree_ros
  inspect_g1_datasets.py   ← 独立
  extract_sam2_frames.py   ← 独立
```

---

## 一、核心 Pipeline

| 脚本 | 功能 | 状态 | 运行示例 |
|------|------|------|----------|
| `config.py` | 集中配置：路径、相机参数、任务、手型检测 | **活跃** | (被其他脚本 import) |
| `video_inpaint.py` | 核心工具库：FK、mesh 渲染、mask、LaMa 修复、视频 IO；也可独立运行 FK+LaMa pipeline | **活跃** | `python scripts/video_inpaint.py --episode 4` |
| `sam2_inpaint_pipeline.py` | **主 pipeline**：FK → SAM2 box prompt → LaMa inpaint，输出 7 种中间结果视频 | **活跃 (主入口)** | `python scripts/sam2_inpaint_pipeline.py --episode 4 --start 5 --duration 5` |
| `sam2_segment.py` | SAM2 分割实验（box/point 两种 prompt 模式） | **活跃** | `python scripts/sam2_segment.py --episode 4 --mode box` |

---

## 二、相机标定

### 活跃

| 脚本 | 功能 | 状态 | 说明 |
|------|------|------|------|
| `auto_calibrate_v2.py` | PSO 标定 v2：新 URDF、JSON manifest、共享工具函数 | **活跃** | 提供 `load_calib_frames()`, `project_and_mask()`, `compute_iou()` 给其他标定脚本复用 |
| `auto_calibrate_grad.py` | 梯度标定 (Adam)：PyTorch 可微鱼眼投影，逐参数学习率 | **活跃** | 提供 `CameraParams`, `differentiable_fisheye_project()` |
| `calibrate_keypoints.py` | 关键点标定：GUI 标注 + Adam 优化，PnP 风格 | **活跃 (下阶段)** | `--annotate` 标注 → `--optimize` 优化 |

### 废弃

| 脚本 | 功能 | 状态 | 废弃原因 |
|------|------|------|----------|
| `auto_calibrate.py` | PSO 标定 v1 | **废弃** | 使用旧 URDF (`g1_29dof_rev_1_0.urdf`)、硬编码旧数据路径 (`data/g1_wbt_task3`)，已被 `auto_calibrate_v2.py` 完全取代 |
| `interactive_calibrate.py` | GUI 交互标定 (trackbar) | **废弃** | 使用旧 URDF、旧路径 (`data/g1_wbt`)，需要显示器，已被 `calibrate_keypoints.py` 取代 |

---

## 三、人体叠加

| 脚本 | 功能 | 状态 | 说明 |
|------|------|------|------|
| `render_human_overlay.py` | 人手臂 capsule mesh 叠加渲染，Lambertian 光照 | **活跃 (下阶段)** | 使用 `human_arm_overlay.urdf` |
| `demo_smplh_overlay.py` | SMPLH 参数化人体叠加 demo：LBS 蒙皮 + per-joint bone correction | **活跃 (下阶段)** | 依赖外部 SMPLH 模型文件 |
| `generate_human_meshes.py` | 生成 capsule STL mesh（圆柱+半球） | **活跃** | 一次性运行，输出到 `MESH_DIR` |
| `render_front_compare.py` | G1 vs SMPLH 三视图正交对比 (front/side/top) | **活跃 (参考)** | 输出到 `test_results/` |

---

## 四、可视化调试

| 脚本 | 功能 | 状态 | 说明 |
|------|------|------|------|
| `render_hand_debug.py` | 手部逐 link 着色渲染 + 放大裁切，诊断手指映射 | **活跃 (调试)** | 支持 `--human` 切换人手 mesh |
| `render_all_variants.py` | 多变体正面渲染：4 种身体 + 3 种灵巧手 | **活跃 (参考)** | 独立脚本，需 `/tmp/unitree_ros` 克隆仓库 |

---

## 五、数据工具

| 脚本 | 功能 | 状态 | 说明 |
|------|------|------|------|
| `download_g1_wbt.sh` | 批量下载 8 个数据集 (hf-mirror, 重试逻辑) | **活跃** | ~77GB 总计 |
| `inspect_g1_datasets.py` | 数据集统计表 + 首帧提取 | **活跃** | 输出到 `_inspect/` |
| `downsample_videos.py` | 批量 30fps → 16fps 转换 | **活跃** | FFmpeg + 并行处理 |
| `extract_sam2_frames.py` | 从 SAM2 结果视频提取帧 + 二值 mask + 缩略图 | **活跃** | 生成 `calib_frames.json` 模板 |

---

## 六、根目录文件

| 文件 | 功能 | 状态 | 说明 |
|------|------|------|------|
| `cmd.sh` | 环境变量设置 (proxy, HF cache, conda) | **待修复** | conda env 名写的是 `dfar`，应改为 `flip` |
| `idea.md` | 早期方案探索笔记 (DisMo, MIMO, Cosmos Transfer) | 归档 | 记录了方案选型历史 |
| `progress.md` | 旧版 pipeline 进度文档（含流程图） | **已取代** | `doc/progress.md` 为当前权威版本 |
| `report_outline.md` | 汇报大纲（方法、动机、进度） | 活跃 (参考) | 面向导师汇报用 |
| `CLAUDE.md` | Claude Code 项目上下文 | 活跃 | 保持更新 |

---

## 七、清理建议

### 可安全删除

- **`scripts/auto_calibrate.py`** — 完全被 `auto_calibrate_v2.py` 取代。使用已不存在的旧路径 (`data/g1_wbt_task3`, `data/g1_urdf`)，旧 URDF (`g1_29dof_rev_1_0.urdf`)。标定结果已固化到 `config.py`。
- **`scripts/interactive_calibrate.py`** — 完全被 `calibrate_keypoints.py` 取代。同样使用旧路径和旧 URDF。需要显示器 GUI。

### 待修复

- **`cmd.sh`** — `conda activate dfar` 应改为 `conda activate flip`

### 可归档

- **根目录 `progress.md`** — 内容已被 `doc/progress.md` 涵盖。根目录版本包含的流程图如有需要可迁移到 `doc/progress.md` 后删除根目录副本。

### 统计

| 分类 | 数量 | 脚本 |
|------|------|------|
| 核心 Pipeline | 4 | config, video_inpaint, sam2_inpaint_pipeline, sam2_segment |
| 相机标定 (活跃) | 3 | auto_calibrate_v2, auto_calibrate_grad, calibrate_keypoints |
| 相机标定 (废弃) | 2 | auto_calibrate, interactive_calibrate |
| 人体叠加 | 4 | render_human_overlay, demo_smplh_overlay, generate_human_meshes, render_front_compare |
| 可视化调试 | 2 | render_hand_debug, render_all_variants |
| 数据工具 | 4 (+1 sh) | download_g1_wbt.sh, inspect_g1_datasets, downsample_videos, extract_sam2_frames |
| **合计** | **19 + 1 sh** | |
