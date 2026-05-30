# Wan 2.1 FunControl 14B LoRA 微调 Baseline

> **归档说明（2026-04-24 / T032）**：FunControl baseline 已退出当前维护主线，`src.pipeline.train_lora` 已删除。本文仅作为历史记录保留，不要按本文命令启动新实验。当前训练入口见 [`step_5_training_infra.md`](step_5_training_infra.md)。

## 概述

使用 DiffSynth-Studio 框架对 Wan 2.1 FunControl 14B 进行 LoRA 微调，实现 human→robot V2V 视频转换。

训练分**两阶段**进行——阶段 1 预处理 embedding 缓存，阶段 2 只训 DiT LoRA，解决单卡 24GB 装不下完整模型的问题。

## 硬件要求

- **GPU**: 至少 2 张 RTX 4090 (24GB)，必须干净（其他进程占用 >500MB 就可能 OOM）
- **RAM**: ~60GB 可用（模型加载时的 CPU 峰值）
- **磁盘**: ~50GB（模型权重）+ 训练数据

## 环境

```bash
conda activate flip
# DiffSynth-Studio 已安装: /disk_n/zzf/DiffSynth-Studio
```

## 模型权重

位置: `/disk_n/zzf/.cache/huggingface/hub/models--alibaba-pai--Wan2.1-Fun-V1.1-14B-Control/manual/`

| 文件 | 大小 | 用途 |
|------|------|------|
| `diffusion_pytorch_model.safetensors` | 32.8 GB | DiT 主模型 (16.4B 参数) |
| `models_t5_umt5-xxl-enc-bf16.pth` | 11 GB | T5 文本编码器 |
| `models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth` | 4.5 GB | CLIP 图像编码器 |
| `Wan2.1_VAE.pth` | 485 MB | VAE |

来源: `alibaba-pai/Wan2.1-Fun-V1.1-14B-Control` (HuggingFace)

## 训练数据格式

```
training_data/
├── video/              ← target 视频 (robot)
│   ├── pair_0000.mp4
│   ├── pair_0001.mp4
│   └── ...
├── control_video/      ← condition 视频 (human)
│   ├── pair_0000.mp4
│   ├── pair_0001.mp4
│   └── ...
└── metadata.csv
```

### metadata.csv

```csv
video,prompt,control_video
video/pair_0000.mp4,A first-person view robot arm performing household tasks flip_v2v,control_video/pair_0000.mp4
video/pair_0001.mp4,A first-person view robot arm performing household tasks flip_v2v,control_video/pair_0001.mp4
```

### 视频规格

| 参数 | 要求 |
|------|------|
| 分辨率 | 640×480 |
| 帧率 | 16 fps |
| 帧数 | **17 帧 (1s)** — 必须满足 4k+1 格式 |
| 编码 | H.264, yuv420p |
| video/ 和 control_video/ 的文件名 | 必须一一对应 |

### 视频预处理示例

```bash
# 从 4s 原始视频切成 1s 片段
ffmpeg -y -i human.mp4 -vf "scale=640:480,fps=16" -frames:v 17 \
  -c:v libx264 -pix_fmt yuv420p control_video/pair_0000.mp4

ffmpeg -y -i robot.mp4 -vf "fps=16" -frames:v 17 \
  -c:v libx264 -pix_fmt yuv420p video/pair_0000.mp4
```

## 训练流程

### 变量定义

```bash
DEST="/disk_n/zzf/.cache/huggingface/hub/models--alibaba-pai--Wan2.1-Fun-V1.1-14B-Control/manual"
DATA="/disk_n/zzf/flip/training_data"
TRAIN_SCRIPT="/disk_n/zzf/DiffSynth-Studio/examples/wanvideo/model_training/train.py"
MODEL_PATHS="[\"$DEST/diffusion_pytorch_model.safetensors\",\"$DEST/models_t5_umt5-xxl-enc-bf16.pth\",\"$DEST/Wan2.1_VAE.pth\",\"$DEST/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth\"]"
```

### 阶段 1: 数据预处理 (单卡)

在一张 GPU 上加载 T5 + CLIP + VAE，预处理所有样本的 embedding 并缓存。DiT offload 到磁盘不占 GPU。

```bash
CUDA_VISIBLE_DEVICES=3 accelerate launch --num_processes=1 \
  $TRAIN_SCRIPT \
  --task "sft:data_process" \
  --dataset_base_path "$DATA" \
  --dataset_metadata_path "$DATA/metadata.csv" \
  --data_file_keys "video,control_video" \
  --model_paths "$MODEL_PATHS" \
  --extra_inputs "control_video" \
  --offload_models "$DEST/diffusion_pytorch_model.safetensors" \
  --lora_base_model "dit" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank 16 \
  --height 480 --width 640 --num_frames 17 \
  --output_path "output/data_cache"
```

输出: `output/data_cache/0/*.pth`（每样本 ~50MB）

### 阶段 2: LoRA 训练 (双卡 DDP)

从缓存读 embedding，只加载 DiT FP8 + LoRA。

```bash
CUDA_VISIBLE_DEVICES=0,3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  accelerate launch --num_processes=2 \
  $TRAIN_SCRIPT \
  --task "sft:train" \
  --dataset_base_path "output/data_cache" \
  --model_paths "[\"$DEST/diffusion_pytorch_model.safetensors\"]" \
  --fp8_models "$DEST/diffusion_pytorch_model.safetensors" \
  --lora_base_model "dit" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank 16 \
  --use_gradient_checkpointing \
  --learning_rate 1e-4 \
  --num_epochs 1 \
  --dataset_repeat 5 \
  --save_steps 5 \
  --output_path "output/lora_funcontrol_test"
```

输出: `output/lora_funcontrol_test/step-N.safetensors`（~147MB/checkpoint）

## 当前训练参数

| 参数 | 值 | 说明 |
|------|-----|------|
| 基底模型 | Wan2.1-Fun-V1.1-14B-Control | FunControl 变体, 16.4B 参数 |
| 精度 | DiT FP8, LoRA BF16 | 冻结权重 FP8, 可训参数 BF16 |
| LoRA rank | 16 | 50.5M 可训参数 |
| LoRA targets | q,k,v,o,ffn.0,ffn.2 | attention 全部 + FFN |
| 视频 | 640×480, 17 帧, 16fps | |
| 学习率 | 1e-4 | |
| 优化器 | AdamW (weight_decay=1e-2) | |
| Gradient checkpoint | 开 | 必须, 否则激活值 OOM |
| 训练 GPU | 2 卡 DDP | |
| 每步耗时 | ~6 秒 | |

## VRAM 预算 (每卡)

```
DiT FP8 权重:              16.48 GB  (16.4B 参数, 含 80M BF16 norms)
LoRA + AdamW:               0.61 GB
激活值 (grad checkpoint):   2.68 GB  (40 blocks × 57MB checkpoints)
FP8↔BF16 转换 buffer:     ~2.5 GB   (forward 时临时类型转换)
其他 (DDP, NCCL, 碎片):   ~1.4 GB
─────────────────────────────────────
总计:                      ~23.7 GB / 23.53 GB 可用
```

## 注意事项

### 帧数限制

- **17 帧 (1s) 是上限**。33 帧 (2s) 会 OOM（差 180MB）
- 帧数必须满足 `4k+1`: 5, 9, 13, 17, 21, 25, 29, 33...
- 改帧数需要重跑阶段 1（重新生成缓存）

### GPU 必须干净

- 24GB 余量仅 ~1GB，任何其他进程占用 GPU 内存都可能导致 OOM
- 训练前先检查: `nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits`
- 确保目标 GPU 的 memory.used < 100 MB

### 两阶段不能合并

- T5 (11GB) + DiT FP8 (16.4GB) = 27.4GB > 24GB，无法同卡
- 阶段 1 用 `--offload_models` 把 DiT offload 到磁盘
- 阶段 2 只加载 DiT，从缓存读 T5/CLIP 的 embedding

### "14B" 模型实际是 16.4B

- FunControl 变体在 base 14B 上增加了额外参数（ref_conv 等）
- FP8 实际占 16.48GB（不是预期的 14GB）
- 这是 VRAM 紧张的根本原因

### data_process 会下载 tokenizer

- 首次运行阶段 1 会从 ModelScope 下载 `Wan-AI/Wan2.1-T2V-1.3B` 的 tokenizer（~几百 MB）
- 缓存在 `models/Wan-AI/Wan2.1-T2V-1.3B/`

### PYTORCH_CUDA_ALLOC_CONF

- 阶段 2 必须设置 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
- 减少 CUDA 内存碎片，否则 reserved-but-unused 内存会导致假 OOM

### DiffSynth-Studio 的局限

- 框架抽象层多（21 个 PipelineUnit 类），改网络结构困难
- FP8 forward 每次都做 FP8→BF16 临时转换，增加 ~2.5GB 隐藏开销
- 不输出 train loss（计算后被 ModelLogger 丢弃）
- 无 train/eval 数据集分割
- 无任何日志系统（无 TensorBoard / WandB / print）

---

## 自写训练脚本 (train_lora.py)

> T032 后该脚本已删除；以下内容仅解释历史实现。

DiffSynth baseline 的训练可观测性为零，因此自写了 `src/pipeline/train_lora.py`。

### 与 DiffSynth baseline 的关系

- **复用**：WanTrainingModule（模型加载 + FP8 + LoRA 注入 + forward pass）
- **替换**：训练循环、日志系统、eval 机制、checkpoint 管理
- **不改**：DiffSynth-Studio 源码

### 架构说明

`task="sft:train"` 时 DiffSynth 的 24 个 PipelineUnit 全部被 `split_pipeline_units` 移除，forward 调用链实际是：

```
model({}, inputs=cached_data)
  → transfer_data_to_device()     # CPU→GPU 拷贝 (~5ms)
  → FlowMatchSFTLoss()            # 采样 t → 加噪 → DiT forward → MSE
```

框架开销可忽略（~2μs Python 分支检查 vs 秒级 DiT 计算）。

### 功能

| 功能 | DiffSynth baseline | train_lora.py |
|------|-------------------|---------------|
| Train loss 输出 | 无 | 每步输出 |
| Train/Eval 分割 | 无 | 固定 seed 随机分割 |
| Eval loss | 无 | 定期计算 |
| Eval 视频 | 无 | GT + control + generated (CFG + VAE decode) |
| 日志 | tqdm 进度条 | 文本日志 (stdout + train.log) |
| Checkpoint 格式 | safetensors | safetensors（兼容） |
| 多卡 | accelerate DDP | 单卡（不用 accelerate） |

### 用法

> 以下命令已废弃，不能在当前代码中运行。

```bash
# 阶段 1 仍然使用 DiffSynth（见上方「阶段 1」）
# 阶段 2 使用自写脚本：

# 基本训练（单卡）
python -m src.pipeline.train_lora \
  --cache-dir output/data_cache_80 \
  --device cuda:0 \
  --epochs 1 --repeat 10 \
  --save-steps 50 --eval-steps 50

# 带 eval 视频（需 VAE，多占 ~500MB 显存）
python -m src.pipeline.train_lora \
  --cache-dir output/data_cache_80 \
  --device cuda:0 \
  --eval-video-steps 50 --eval-video-samples 2 \
  --num-inference-steps 30
```

### 输出目录

```
training_data/log/YYYY-MM-DD_HHMMSS/
├── ckpt/
│   ├── step-0050.safetensors   (~147MB, 800 LoRA tensors)
│   └── step-0100.safetensors
├── eval/
│   └── step-0050/
│       ├── gt_00.mp4           (ground truth robot 视频)
│       ├── ctrl_00.mp4         (human condition 视频)
│       └── gen_00.mp4          (模型生成视频, CFG=5.0)
└── train.log
```

### CLI 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--cache-dir` | (必填) | DiffSynth data_process 输出的 .pth 缓存目录 |
| `--device` | cuda:3 | GPU 设备 |
| `--lr` | 1e-4 | 学习率 |
| `--epochs` | 1 | 训练轮数 |
| `--repeat` | 10 | 每 epoch 重复数据次数 |
| `--lora-rank` | 16 | LoRA rank |
| `--save-steps` | 50 | checkpoint 保存间隔 |
| `--eval-steps` | 50 | eval loss 计算间隔 |
| `--eval-ratio` | 0.1 | eval 集比例 |
| `--eval-video-steps` | 0 | eval 视频生成间隔（0=关闭） |
| `--eval-video-samples` | 2 | 每次生成的 eval 视频数 |
| `--num-inference-steps` | 30 | eval 推理 denoising 步数 |

### 性能

| 指标 | 值 |
|------|-----|
| 单步训练耗时 | ~6.1s |
| Eval loss (8 样本) | ~15s |
| Eval 视频 (1 个, 30 步, 含 CFG) | ~100s |
| 显存 (DiT FP8 + LoRA) | ~22.3 GB |
| 显存 (+ VAE for eval 视频) | ~22.8 GB |
