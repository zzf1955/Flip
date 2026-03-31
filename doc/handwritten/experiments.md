# 人形机器人实验计划

> 项目思路见 [idea_humanoid.md](idea_humanoid.md)。

## 概述

**目标:** 从 G1 第三人称视频出发，通过关键帧编辑 + 条件视频生成，合成配对的人类第三人称视频。

**Pipeline:**
1. **关键帧编辑（图像级）**: 将 G1 关键帧中的机器人全身替换为人类
2. **视频合成（视频级）**: Cosmos Transfer 2.5 + 控制信号 + 关键帧锚定，生成完整人类视频

**工具链**: ComfyUI + Flux Fill / Z Image / Flux Dev + Cosmos Transfer 2.5

---

## 阶段一: 关键帧编辑

**公共数据:** G1 第三人称视频关键帧，实验图片存放 `data/experiment/g1/`

### 实验 1.1: Flux Fill — 整体 mask

将 G1 全身统一 mask 后一次性 inpaint 为人类。

- mask 方案: 手动标注 / GroundingDINO + SAM2 自动检测
- Prompt: `"a human performing the same action, realistic, third person view, photorealistic"`
- 参考参数: steps=28, cfg=1.0, euler, denoise=0.85, guidance=30.0

**风险:** 大面积全身 mask 可能导致生成质量下降；背景保留应较好。

### 实验 1.2: Z Image — 无 mask 指令式编辑

通过文本指令直接将 G1 替换为人类，无需 mask。

- Prompt: `"Replace the robot with a realistic human performing the same action. Keep background unchanged."`

**风险:** 语义编辑对 G1 全身的识别准确性未知。

### 实验 1.3: Flux Dev (img2img) — 无 mask 全局编辑

Flux Dev 原模型 img2img，以 G1 关键帧为初始 latent。

- Prompt: `"a human performing a task, realistic, third person view, no robot, no mechanical parts"`
- Denoise 扫参: 0.35, 0.45, 0.55, 0.65

**风险:** 无法平衡背景保留和全身替换，作为 baseline 对照。

### 实验 1.4: Flux Fill — 分段 mask

按部位分段（头部/躯干、手臂、腿部），使用针对性 prompt 分步 inpaint。

- 按 躯干→手臂→腿部 顺序分步处理
- 每步使用针对性 prompt

**风险:** 多步 inpaint 累积误差，步间风格一致性需控制。与 1.1 对照。

### 评估指标

- 人体自然度（姿态、比例、手指）
- 背景保留完整度
- 人与物体的交互合理性

---

## 阶段二: 视频合成

### 实验 2.1: 控制信号提取

从 G1 视频中提取：
- 深度图 (DepthAnything V2)
- 边缘图 (Canny)
- 分割图 (SAM2)

SalientObject 策略：前景（G1 主体）强约束，背景弱约束。

### 实验 2.2: Cosmos Transfer 关键帧锚定视频生成

以阶段一关键帧为锚，配合控制信号生成完整人类视频。

**锚定策略:** 首帧 / 首尾帧 / 均匀多帧 (K=8/16/32)

**控制信号组合:**
| 组合 | 信号 |
|------|------|
| A | depth only |
| B | depth + edge |
| C | depth + edge + seg |
| D | SalientObject (前景强/背景弱) |

**评估:** 时序一致性、运动对齐度、外观一致性、场景保真度

### 实验 2.3: 不同关键帧方案对比

用实验 1.1~1.4 的最佳结果分别作为锚点生成视频，横向对比。

---

## 阶段三: 规模化

### 实验 3.1: 端到端 Pipeline

```
G1视频 → 抽帧 → 自动mask → Flux Fill编辑 → 控制信号提取 → Cosmos Transfer → 合成人类视频
```

评估: 端到端成功率、单视频处理时间、失败模式。

### 实验 3.2: 一对多增强

对同一 G1 视频生成 N=1,3,5,10 种外观的人类视频，对比下游编辑模型效果。

---

## 优先级

| 级别 | 实验 | 目的 |
|------|------|------|
| P0 | 1.1~1.4 + 2.1 | 核心可行性验证 |
| P1 | 2.2 + 2.3 | Pipeline 核心步骤 |
| P3 | 3.1 + 3.2 | 规模化与增强 |

---

## 工具

| 组件 | 用途 |
|------|------|
| Flux Fill (GGUF Q8) | 关键帧 mask inpainting |
| Flux Dev (GGUF Q8) | img2img 无 mask 编辑 |
| Z Image | 无 mask 指令式编辑 |
| Cosmos Transfer 2.5 | 条件视频生成 |
| DepthAnything V2 | 深度图 |
| GroundingDINO + SAM2 | 自动 mask |

## 待开发脚本

| 脚本 | 功能 | 优先级 |
|------|------|--------|
| `scripts/full/extract_keyframes.py` | 视频关键帧提取 | P0 |
| `scripts/full/auto_mask.py` | 自动检测 G1 并生成 mask | P0 |
| `scripts/full/comfyui_inpaint.py` | Flux Fill 关键帧编辑 | P0 |
| `scripts/full/extract_control_signals.py` | 控制信号提取 | P0 |
| `scripts/full/cosmos_transfer_generate.py` | Cosmos Transfer 视频合成 | P1 |
| `scripts/full/pipeline_e2e.py` | 端到端 Pipeline | P3 |
