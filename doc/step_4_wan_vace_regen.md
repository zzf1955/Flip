# Step 4 — 本地 Wan 2.1 VACE depth+mask 人体重绘

## 概述

FLIP pipeline 第 4 阶段（Human 渲染 → ControlNet 重绘）的**本地开源方案**。给定 Step 3 的 SMPLH retarget overlay + 机器人背景已修复的视频，通过 depth + mask 条件引导，重生成一张"自然人体、背景保持原貌"的第一人称视频。

Step 4 目前有两条路径，本文档描述后者：

| 路径 | 方式 | 文档 |
|------|------|------|
| 商业 API | Volcengine Seedance 2.0 直接 V2V | `step_4_seedance_api.md` |
| **本地开源** | **ComfyUI + Wan 2.1 VACE 1.3B（depth+mask conditioning）** | **本文档** |

## 流程

```
Step 3 输出 (修复 bg + SMPLH overlay)
  │
  ├── cosmos_prepare.py   → composite.mp4 + depth_blurred.mp4 + guided_mask.mp4
  │                        + depth_raw.mp4 + smplh_mask.mp4 + spec.json
  │
  └── wan_regen.py (当前方案)    → output.webm
      (或 cosmos_regen.py，已弃用)
```

前置 `cosmos_prepare.py` 名字保留 `cosmos_` 前缀，是因为它最初为 Cosmos Transfer 2.5 设计。但其输出（depth / mask / composite）对两种 regen 方案都适用，所以 Wan 方案直接复用，不重复造轮子。

## 方案对比：为什么从 Cosmos 换到 Wan

| 维度 | Cosmos Transfer 2.5 (`cosmos_regen.py`) | Wan 2.1 VACE 1.3B (`wan_regen.py`) |
|------|------------------------------------------|-----------------------------------------|
| 显存 | 需要 2–3 张 4090（单卡装不下） | 单卡 4090 (<16GB) |
| CPU RAM | 峰值 >60GB；共享机被别人占 58GB 就 OOM kill | 常驻 ComfyUI ~5GB |
| Mask 严格度 | `guided_generation_mask` 仅软约束，mask 外背景也会被改 | VACE `control_masks` 硬约束，背景像素几乎不变 |
| Depth 控制 | 支持（`control_weight=1.0`） | 原生支持（VACE conditioning 里 control_video 就是 depth） |
| 启动成本 | `torchrun --nproc-per-node=N` 每次重拉模型（冷启动 ~3 分钟） | ComfyUI 常驻，一次加载反复调用 |
| 结论 | 跑通但**放弃**：OOM + mask 泄漏 | **采用** |

Cosmos 的四次多卡重试记录都保留在 `output/human/cosmos_regen/.../run*.log` 和 `torchrun_logs*/` 里，作为选型对照存档。

## 环境

### ComfyUI

- 位置：`/disk_n/zzf/ComfyUI/`
- 端口：8001
- GPU：0（独占，与其他 pipeline 脚本分开）
- 启动：

```bash
cd /disk_n/zzf/ComfyUI && python main.py --port 8001 --cuda-device 0
```

### 模型权重（`/disk_n/zzf/ComfyUI/models/`）

| 文件 | 大小 | 目录 |
|------|------|------|
| `wan2.1_vace_1.3B_fp16.safetensors` | ~2.7 GB | `diffusion_models/` |
| `umt5_xxl_fp8_e4m3fn_scaled.safetensors` | ~6.5 GB | `text_encoders/` |
| `wan_2.1_vae.safetensors` | ~485 MB | `vae/` |

## 运行

### 生成 prepare 输出

```bash
LD_PRELOAD=/home/leadtek/miniconda3/envs/flip/lib/libjpeg.so.8 \
  python -m src.pipeline.cosmos_prepare \
    --task G1_WBT_Inspire_Pickup_Pillow_MainCamOnly \
    --episode 0 --start 5 --duration 1 --device cuda:1
```

输出 `output/human/cosmos_prepare/<tag>/`：

| 文件 | 说明 |
|------|------|
| `composite.mp4` | 修复背景 + SMPLH overlay，作为视觉参考 |
| `depth_raw.mp4` | VideoDepthAnything 原始深度 |
| `depth_blurred.mp4` | 人体区域深度模糊（用作 control_video） |
| `smplh_mask.mp4` | SMPLH 二值人体 mask |
| `guided_mask.mp4` | 反转 mask（白=背景，用作 control_masks） |
| `spec.json` | Cosmos 推理配置（Wan 路径不读） |

关键参数：

- `--scale 0.75` SMPLH 整体缩放
- `--hand-scale 1.3` 手部单独缩放
- `--depth-blur-ratio 0.15` 人体区域深度模糊强度
- `--output-fps 16` 输出 16 fps（匹配 Wan VACE）

### Wan VACE 重绘

```bash
LD_PRELOAD=/home/leadtek/miniconda3/envs/flip/lib/libjpeg.so.8 \
  python -m src.pipeline.wan_regen \
    --prepare-dir output/human/cosmos_prepare/Pickup_Pillow_MainCamOnly_ep0_s5_d1 \
    --server http://localhost:8001 \
    --steps 30 --cfg 6.0 --strength 1.0 --seed 2025
```

主要参数：

| 参数 | 默认 | 说明 |
|------|------|------|
| `--prepare-dir` | — | cosmos_prepare 输出目录 |
| `--server` | `http://localhost:8001` | ComfyUI HTTP endpoint |
| `--steps` | 30 | KSampler 迭代步数 |
| `--cfg` | 6.0 | classifier-free guidance 强度 |
| `--strength` | 1.0 | VACE conditioning 强度 |
| `--seed` | 2025 | 采样种子 |
| `--prompt` | 内置英文 prompt | 正面提示词 |
| `--no-composite` | off | 关闭后处理 bg 合成 |

VACE 约束：length 必须 `4k+1`。`wan_regen.py` 自动按 `((n_frames-1)//4)*4 + 1` 对齐，输入 16 帧 → 实际 17 帧。

## ComfyUI 工作流结构

`wan_regen.py:build_vace_workflow()` 以 JSON 形式构造节点图（非拖 UI），14 个节点：

```
UNETLoader (wan2.1_vace_1.3B)           ─┐
CLIPLoader (umt5_xxl_fp8, type=wan)     ─┤
VAELoader (wan_2.1_vae)                 ─┤
CLIPTextEncode (pos_prompt)             ─┤
CLIPTextEncode (neg_prompt)             ─┤
LoadImageSequence (depth_blurred.mp4)   ─┤
LoadImageSequence (guided_mask.mp4)     ─┤
                                         ▼
WanVaceToVideo (positive/negative/vae/control_video/control_masks/width/height/length/strength)
                                         │
                                         ▼
KSampler (euler/normal/steps=30/cfg=6.0)
                                         │
                                         ▼
TrimVideoLatent (去参考帧 padding)
                                         │
                                         ▼
VAEDecode
                                         │
                                         ▼
SaveAnimatedWEBP (fps=16, lossless=false, quality=90)
```

`LoadImageSequence` 读目录 / `VHS_LoadVideo` 读 mp4 由 `control_frames_dir` 是否是目录自动切换。

## 产物

位置：`output/human/wan_regen/`

| 文件 | 分辨率 | 帧数 | 说明 |
|------|--------|------|------|
| `Pickup_Pillow_MainCamOnly_ep0_s5_d1.webm` | 640×480 | 17 | v1 初版，用 depth_blurred 作 control |
| `Pickup_Pillow_v2_composite_control.webm` | 640×480 | 17 | v2，用 composite 作 control（对比实验） |
| `test_workflow.json` | — | — | ComfyUI 导出的节点图快照 |

## 已知限制 / 待办

1. **1.06 秒长度**：17 帧 @ 16 fps 仅 ~1s，Wan VACE 1.3B 长视频能力有限，后续评估 14B 版本。
2. **未批量化**：当前 `wan_regen.py` 一次只跑一个 prepare-dir，批量需另写调度器（类似 `batch_inpaint.py`）。
3. **Prompt 统一**：用的是通用"第一人称室内家务"描述，未按 task 差异化（Pickup_Pillow vs Put_Clothes 的视觉内容差很大）。
4. **评估缺失**：只做了视觉合格检查，未量化背景像素保真度（可对比 mask 外区域的 PSNR/SSIM）和人体区域合理性（IK alignment 已有但生成质量无指标）。
5. **ComfyUI 进程管理**：目前手工拉起；可考虑把启动/健康检查封装进 `wan_regen.py`。

## 相关文件

| 文件 | 作用 |
|------|------|
| `src/pipeline/cosmos_prepare.py` | 前置：composite + depth + mask + spec |
| `src/pipeline/cosmos_regen.py` | Cosmos Transfer 2.5 包装（对照存档，不再使用） |
| `src/pipeline/wan_regen.py` | **主入口**：ComfyUI API + VACE workflow |
| `src/core/config.py` | `COSMOS_PREPARE_DIR / COSMOS_REGEN_DIR / COMFYUI_ROOT / WAN_REGEN_DIR` |
| `src/core/render.py:render_smplh_mask` | 二值人体 mask 渲染 |
| `doc/tasks/done/002.md` | 本阶段任务规格与交付记录 |
| `doc/step_4_seedance_api.md` | Step 4 商业 API 路径（Seedance） |
