# FLIP 项目迁移文档

## Context

项目 FLIP（**F**rom Robot to Human — **L**earning Video Editing via **I**nverse **P**ipeline）在 Mac 平台完成了阶段一（关键帧编辑）的核心实验，现需迁移到 Linux + GPU 平台。本文档记录 Mac 上的全部工作，供迁移后快速恢复。

---

## 1. 项目概述

**核心思路**：反向生成式数据合成
- 传统：人类视频(真实) → 合成pipeline → 机器人视频(合成，有domain gap)
- FLIP：G1机器人视频(真实) → 合成pipeline → 人类视频(合成) → 训练 video editing 模型
- 推理时：人类视频(真实) → 训练好的模型 → G1机器人视频(真实)

**核心优势**：输出端零 domain gap、一对多数据增强、无需 3D 资产

**详见**：`doc/handwritten/idea_humanoid.md`

---

## 2. 当前进度总结

### 已完成（阶段一：关键帧编辑）

| 任务 | 内容 | 状态 |
|------|------|------|
| Task 001 | LEVERB 关键帧提取（每 chunk 1 视频 × 3 帧） | ✅ review |
| Task 002 | Flux Fill 无 mask 全图 inpaint 实验 | ✅ review |
| Task 003 | Qwen-Image-Edit 文本指令编辑实验 | ✅ review |
| Task 004 | **自动化 pipeline**：GroundingDINO 检测 → SAM2 像素级 mask → SDPose 骨骼提取 → ControlNet openpose 控制 → Flux Fill inpaint | ✅ review |

### 实验结论

- **Flux Fill + mask + ControlNet pose** 效果最佳，人物姿态保持好，背景几乎无损
- **SAM2 像素级 mask** 远优于 bbox 方框 mask（方框容易误伤背景）
- **GrowMask expand=5** 是平衡点（太小人消失，太大背景受损）
- **Qwen-Image-Edit** 速度较慢，效果不如 Flux Fill + mask
- **480P（856×480）** 是 Mac MPS 上的合理分辨率，单帧约 5 分钟

### 未开始（阶段二：视频合成）

- 控制信号提取（DepthAnything V2 + Canny + SAM2 分割图）
- Cosmos Transfer 2.5 关键帧锚定视频生成
- 一对多外观增强

---

## 3. 数据

### LEVERB 数据集
- **来源**：`ember-lab-berkeley/LeVERB-Bench-Dataset`（HuggingFace）
- **规模**：3697 episodes，4 chunks（chunk-000 ~ chunk-003）
- **格式**：mp4, AV1 codec, 1080×1920 竖屏
- **视角**：仅下载 tpv_cam（第三人称视角），跳过 fpv_cam
- **本地路径**：`data/leverb/videos/chunk-{000..003}/observation.images.tpv_cam/`
- **下载脚本**：`scripts/data/download_leverb_full.py`

### 提取的关键帧
- **路径**：`data/leverb_frames/`
- **内容**：每 chunk 取 1 个视频，每视频 3 帧（first/mid/last），共 12 帧（manifest.json 记录）
- **实际可用**：目前 2 帧有 PNG 文件（episode_001030、episode_002027）
- **提取脚本**：`scripts/extract_keyframes.py`

### 编辑结果
- **路径**：`data/leverb_edited/`
  - `flux/` — Task 002 Flux Fill 实验输出
  - `batch/` — Task 004 bbox mask 早期实验
  - `batch_sam2_1/` — Task 004 SAM2 mask 最终输出（每帧 5 张图）

---

## 4. 模型清单

### 图像编辑（ComfyUI 内）

| 模型 | 文件名 | 用途 | 大小 | 来源 |
|------|--------|------|------|------|
| Flux Fill | `flux1-fill-dev-Q8_0.gguf` | inpainting 主模型 | ~12GB | black-forest-labs/FLUX.1-Fill-dev (GGUF) |
| CLIP-L | `clip_l.safetensors` | Flux 文本编码（短） | ~250MB | comfyanonymous |
| T5-XXL fp8 | `t5xxl_fp8_e4m3fn.safetensors` | Flux 文本编码（长） | ~5GB | comfyanonymous |
| VAE | `ae.safetensors` | Flux 图像编解码 | ~300MB | black-forest-labs |
| SDPose | `sdpose_wholebody_fp16.safetensors` | 全身骨骼关键点提取 | ~2GB | ??? |
| ControlNet Union Pro | `flux-controlnet-union-pro/diffusion_pytorch_model.safetensors` | 多控制类型 ControlNet | ~6.2GB | Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro |
| Qwen-Image-Edit | `Qwen_Image_Edit-Q4_K_S.gguf` | 文本指令编辑（备选） | ~4GB | Qwen |
| Qwen VL Encoder | `qwen_2.5_vl_7b_fp8_scaled.safetensors` | Qwen 图像编码器 | ~8GB | Qwen |
| Qwen VAE | `qwen_image_vae.safetensors` | Qwen 图像空间 | ~100MB | Qwen |

### 检测 & 分割（Python 端，HuggingFace 自动下载）

| 模型 | HF Repo | 用途 |
|------|---------|------|
| GroundingDINO | `IDEA-Research/grounding-dino-tiny` | 零样本目标检测（"robot . humanoid"） |
| SAM2 | `facebook/sam2-hiera-small` | 像素级分割（bbox prompt → mask） |

### 视频生成（阶段二，尚未使用）

| 模型 | 用途 | 说明 |
|------|------|------|
| Cosmos Transfer 2.5 | 条件视频生成 | 关键帧锚定 + 控制信号（深度/边缘/分割）→ 完整视频 |
| WAN 2.2 | DiT 视频生成 | 后续 video editing 模型训练基座 |
| DepthAnything V2 | 深度图提取 | 控制信号之一 |

**WAN 2.2 下载脚本**：`scripts/download_wan2.2.py`（支持 hf-mirror 和 proxy）

---

## 5. 核心脚本

### 5.1 `scripts/batch_robot2human.py`（Task 004 主脚本）

**完整 pipeline**：
```
输入帧 → resize 856×480 → GroundingDINO 检测 bbox → SAM2 像素级 mask(dilate 1px)
    → 上传 ComfyUI → SDPose 骨骼提取 → ControlNet openpose(strength=0.8)
    → GrowMask(expand=5) → Flux Fill inpaint(ddim, 16步, denoise=0.85)
    → 下载 result.png + pose.png
```

**输出目录结构**：
```
output_dir/<frame_name>/
├── original.png   # 480P 缩放原图
├── bbox.png       # GroundingDINO 检测框可视化
├── mask.png       # SAM2 像素级 mask
├── pose.png       # SDPose 骨骼姿态图
└── result.png     # 机器人→人 替换结果
```

**关键参数**：
```bash
python scripts/batch_robot2human.py \
  --input-dir data/leverb_frames/ \
  --output-dir data/leverb_edited/batch_sam2/ \
  --width 856 --height 480 \
  --prompt "a human" --prompt-t5 "a human is walking" \
  --detect-text "robot . humanoid" --detect-threshold 0.25 \
  --bbox-padding 1 \      # SAM2 python端 dilation
  --steps 16 --denoise 0.85 --cn-strength 0.8 \
  --port 8000
```

### 5.2 其他脚本

| 脚本 | 功能 |
|------|------|
| `scripts/extract_keyframes.py` | LEVERB 视频关键帧提取 |
| `scripts/comfyui_client.py` | ComfyUI REST API 客户端（upload/queue/wait/download） |
| `scripts/comfyui_flux_inpaint.py` | Flux Fill 单独 inpaint workflow builder |
| `scripts/comfyui_qwen_edit.py` | Qwen-Image-Edit workflow builder |
| `scripts/download_wan2.2.py` | WAN 2.2 模型下载（支持 mirror/proxy） |
| `scripts/data/download_leverb_full.py` | LEVERB 数据集下载 |

---

## 6. ComfyUI Workflows

**项目内导出**（`workflows/`）：

| 文件 | 用途 | 状态 |
|------|------|------|
| `flux_controlnet_pose_robot2human.json` | ControlNet pose + Flux Fill（Task 004 主 workflow） | ✅ 验证通过 |
| `flux_inpaint_robot2human.json` | Flux Fill 纯 mask inpaint | ✅ 验证通过 |
| `qwen_edit_robot2human.json` | Qwen 文本编辑 | ✅ 可用 |
| `qwen_mask_robot2human.json` | Qwen mask 编辑 | ✅ 可用 |

**主 workflow 节点链**（`flux_controlnet_pose_robot2human`）：
```
LoadImage(original) ─┬─ SDPoseKeypointExtractor(SDPose MODEL+VAE) → SDPoseDrawKeypoints → SaveImage(pose)
                     │
                     ├─ ControlNetApplyAdvanced(Union Pro openpose, strength=0.8, vae=ae)
                     │
LoadImage(mask) ──── GrowMask(expand=5) → InpaintModelConditioning
                     │
DualCLIPLoader ───── CLIPTextEncodeFlux(clip_l="a human", t5xxl="a human is walking")
                     │
UnetLoaderGGUF ───── DifferentialDiffusion → KSampler(ddim, ddim_uniform, 16步, cfg=1, denoise=0.85)
                     │
                     └─ VAEDecode(ae) → SaveImage(result)
```

---

## 7. 参考论文

| 文件 | 论文 | 与本项目关系 |
|------|------|-------------|
| `2502.Human2Robot.pdf` | Human2Robot | 人→机器人方向的先驱工作 |
| `2503.Cosmos-Transfer1.pdf` | Cosmos Transfer v1 | 条件视频生成基础架构 |
| `2508.Masquerade.pdf` | Masquerade | 多主体视频编辑 |
| `2511.Cosmos-Predict2.5.pdf` | Cosmos Predict 2.5 | Cosmos 系列世界模型 |
| `2512.Mitty.pdf` | Mitty | 角色替换方法 |
| `2512.X-Humanoid.pdf` | X-Humanoid | **主要竞品**：UE5 渲染合成（有 sim2real gap） |
| `2603.Dream2Act.pdf` | Dream2Act | 视频→动作生成 |
| `2603.Kiwi-Edit.pdf` | Kiwi-Edit | 视觉知识编辑 |

---

## 8. 环境配置

### Mac 环境（当前）
- **Conda**：`videoedit`，路径 `/opt/homebrew/Caskroom/miniconda/base/envs/videoedit/`
- **代理**：Clash 端口 7897
- **ComfyUI**：`/Users/zzf/Documents/ComfyUI/`，端口 8000（注意有时被 recall.py 占用为 8001）
- **ComfyUI 输出**：`/Users/zzf/Documents/ComfyUI/output/`
- **设备**：MPS（Apple Silicon），SAM2/GroundingDINO 跑 CPU

### 踩坑记录（doc/notice.md）
1. ComfyUI 端口可能在 8000/8001 之间变化，脚本用 `--port` 参数控制
2. `conda run` 有权限问题，用绝对路径 python 代替
3. LEVERB 为 AV1 编码，cv2 可正常读取
4. GroundingDINO transformers v5.4 API: 用 `threshold=` 而非 `box_threshold=`
5. SAM2 在 Mac 上不支持 CUDA/MPS，必须 `device='cpu'`
6. ControlNetApplyAdvanced 需要 VAE 输入（ae.safetensors）
7. SDPoseKeypointExtractor 需要自己的 CheckpointLoaderSimple，不能接 Flux 的 MODEL

### 迁移到 Linux 需要

1. **安装 ComfyUI** + 以下自定义节点：
   - ComfyUI-GGUF（加载 GGUF 模型）
   - ComfyUI-SDPose（骨骼提取）
   - ComfyUI-ControlNet-Union（ControlNet Union Pro）
2. **下载模型**：见第 4 节模型清单
3. **安装 Python 依赖**：torch(CUDA), transformers, sam2, opencv-python, pillow, numpy
4. **迁移数据**：LEVERB 数据集重新下载（`scripts/data/download_leverb_full.py`）
5. **导入 workflows**：`workflows/*.json` 放入 ComfyUI workflows 目录
6. **分辨率可调高**：GPU 上可尝试 720P (1280×720) 或原始 1080P

---

## 9. 项目文件结构

```
VideoEdit/
├── scripts/
│   ├── batch_robot2human.py       # ★ 核心 pipeline（Task 004）
│   ├── extract_keyframes.py       # 关键帧提取
│   ├── comfyui_client.py          # ComfyUI REST API 客户端
│   ├── comfyui_flux_inpaint.py    # Flux Fill workflow builder
│   ├── comfyui_qwen_edit.py       # Qwen Edit workflow builder
│   ├── download_wan2.2.py         # WAN 2.2 下载器
│   └── data/download_leverb_full.py
├── workflows/                     # ComfyUI workflow JSON 导出
├── doc/
│   ├── handwritten/
│   │   ├── idea_humanoid.md       # 核心思路
│   │   └── experiments.md         # 三阶段实验计划
│   ├── framework.md               # 架构概览
│   ├── notice.md                  # 踩坑记录
│   └── tasks/{pending,active,review,done}/
├── paper/                         # 8 篇参考论文
├── ref-cosmos-transfer2.5/        # Cosmos Transfer 2.5 参考实现
├── data/
│   ├── leverb/                    # LEVERB 原始数据
│   ├── leverb_frames/             # 提取的关键帧
│   └── leverb_edited/             # 编辑输出
└── archive/                       # 早期参考实现
```

---

## 10. 下一步工作（阶段二）

1. **控制信号提取**：DepthAnything V2 深度图 + Canny 边缘 + SAM2 分割图
2. **Cosmos Transfer 2.5 视频合成**：关键帧锚定 + 控制信号 → 完整人类视频
3. **一对多增强**：同一 G1 视频 → N 种人类外观
4. **端到端 pipeline**：串联所有步骤，批量处理

详见 `doc/handwritten/experiments.md`
