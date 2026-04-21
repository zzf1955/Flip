# 训练基础设施

训练相关的 DDP 配置、模型加载机制、微调命令、目录结构。
算法原理见 [`step_5_wan22_ti2v5b.md`](step_5_wan22_ti2v5b.md)，两阶段策略见 [`step_5_two_stage_training.md`](step_5_two_stage_training.md)。

## DDP 训练注意事项

### 模型加载

`wan_loader.py` 所有加载函数直接读到目标 GPU，不经过 CPU 中转：
- `load_dit()`: safetensors 直接到 GPU (~10GB bf16)
- `load_vae()`: `torch.load(map_location=device)` / safetensors (~0.67GB)
- `load_text_encoder()`: `torch.load(map_location=device)` (~5.5GB，仅 mitty_cache 用)
- 训练样本 cache: `load_sample(path, device=device)` 直接到 GPU

**DiT bf16 预转换**：上游 safetensors 是 FP32 (20GB)，一次性转 bf16 (10GB):
```bash
python -m src.tools.convert_dit_bf16
```
`build_dit_shard_list()` 自动优先使用 bf16 单文件。

**DDP broadcast 加载**：rank 0 独占读盘，其他 rank 通过 `dist.broadcast` 接收权重。
`load_dit(skip_load=True)` 分配空壳，`train.py` 中 broadcast `model.pipe.dit` 全部参数。

## Cache 管理

训练前需要预计算 embedding 缓存，分为 **T5 文本缓存** 和 **VAE 视频缓存** 两类。

### 目录结构

```
training_data/cache/
├── t5/                              # T5 text embedding（prompt 级别，全局共享）
│   ├── prompt_<hash>.pth            # {embedding: (1,512,4096), prompt: str}
│   └── negative.pth                 # 负向 prompt embedding
└── vae/                             # VAE latent（per-sample）
    ├── pair_1s/                     # 对应 training_data/pair/1s
    │   ├── train/pair_NNNN.pth
    │   ├── eval/pair_NNNN.pth
    │   └── ood_eval/pair_NNNN.pth
    ├── pair_1s_identity/            # 对应 pair/1s_identity
    └── robot_1s/                    # 对应 robot_pair/1s
```

**VAE cache .pth 内容**（新格式）：
- `human_latent`: (1, 48, 5, 30, 40) — 人类视频 VAE 编码
- `robot_latent`: (1, 48, 5, 30, 40) — 机器人视频 VAE 编码
- `prompt`: str — 原始文本（用于查找 T5 cache）
- `source_id`: str — 溯源 key

T5 embedding 不再嵌入每个样本文件，而是从共享 `t5/` 目录按 prompt 文本匹配加载。训练脚本通过 `--t5-cache-dir` 指定，默认 `training_data/cache/t5/`。

**向后兼容**：旧格式（含 `context_posi`/`context_nega`）仍可直接使用，`load_sample()` 检测到已有 T5 字段时不注入。

### 生成 cache

```bash
# 新格式（推荐）：VAE-only .pth + 共享 T5 cache
python -m src.pipeline.mitty_cache \
  --pair-dir training_data/pair/1s/train \
  --output   training_data/cache/vae/pair_1s/train \
  --device cuda:0

# 旧格式（嵌入 T5 + 可选 frames）
python -m src.pipeline.mitty_cache \
  --pair-dir training_data/pair/1s/train \
  --output   output/mitty_cache_1s/train \
  --legacy --device cuda:0
```

T5 cache 在首次运行时自动编码并保存到 `--t5-cache-dir`（默认 `training_data/cache/t5/`），后续运行会跳过已存在的文件。

### config.py 路径常量

```python
from src.core.config import (
    CACHE_ROOT,      # training_data/cache
    T5_CACHE_DIR,    # training_data/cache/t5
    VAE_CACHE_DIR,   # training_data/cache/vae
)
```

## LoRA 微调流程

训练产物统一放到 `training_data/log/<timestamp>/` 下。

### 阶段 1: data_process（embedding 缓存，单卡）

```bash
DEST="/disk_n/zzf/.cache/huggingface/hub/models--alibaba-pai--Wan2.1-Fun-V1.1-14B-Control/manual"
DATA="training_data/pair/1s"
TRAIN_SCRIPT="/disk_n/zzf/DiffSynth-Studio/examples/wanvideo/model_training/train.py"
MODEL_PATHS="[\"$DEST/diffusion_pytorch_model.safetensors\",\"$DEST/models_t5_umt5-xxl-enc-bf16.pth\",\"$DEST/Wan2.1_VAE.pth\",\"$DEST/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth\"]"

RUN_DIR="training_data/log/$(date +%Y-%m-%d_%H%M%S)"
mkdir -p "$RUN_DIR"

CUDA_VISIBLE_DEVICES=3 \
LD_PRELOAD=/home/leadtek/miniconda3/envs/flip/lib/libjpeg.so.8 \
  accelerate launch --num_processes=1 \
  "$TRAIN_SCRIPT" \
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
  --output_path "$RUN_DIR/data_cache"
```

输出：`$RUN_DIR/data_cache/0/*.pth`（每样本 ~50MB）

### 阶段 2: LoRA 训练（单卡）

```bash
python -m src.pipeline.train_lora \
  --cache-dir "$RUN_DIR/data_cache" \
  --device cuda:0 \
  --max-steps 400 \
  --save-steps 50 --eval-steps 50 \
  --eval-video-steps 50 --eval-video-samples 2
```

输出：`training_data/log/<auto-timestamp>/ckpt/`、`eval/`、`train.log`

### Wan 2.2 TI2V-5B 两阶段训练策略

训练分两阶段进行（详见 `doc/step_5_two_stage_training.md`）：

**Phase 1: 恒等重建**（robot→robot，学习重建能力）
```bash
# 生成 cache（新格式）
python -m src.pipeline.mitty_cache \
  --pair-dir training_data/pair/1s_identity/train \
  --output training_data/cache/vae/pair_1s_identity/train --device cuda:2

# 训练
python -m src.pipeline.train_mitty \
  --cache-train training_data/cache/vae/pair_1s_identity/train \
  --cache-eval  training_data/cache/vae/pair_1s_identity/eval \
  --cache-ood   training_data/cache/vae/pair_1s_identity/ood_eval \
  --max-steps 400 --save-steps 50 --eval-steps 50
```

**Phase 2: 外观替换**（human→robot，用 Phase 1 ckpt 初始化）
```bash
python -m src.pipeline.train_mitty \
  --init-lora training_data/log/<phase1-run>/ckpt/step-NNNN.safetensors \
  --cache-train training_data/cache/vae/pair_1s/train \
  --cache-eval  training_data/cache/vae/pair_1s/eval \
  --cache-ood   training_data/cache/vae/pair_1s/ood_eval \
  --max-steps 400 --save-steps 50 --eval-steps 50
```

### Wan 2.2 TI2V-5B 统一训练入口（推荐）

两个消融维度：主干 `--backbone {mitty,rectflow}`、loss `--loss {uniform,hand_patch}`。
`--loss hand_patch` 必须配 `--patch-dir`，`--loss uniform` 不允许传 `--patch-dir`（argparse 硬校验）。

```bash
# Mitty + uniform（baseline）
torchrun --nproc_per_node=4 -m src.pipeline.train \
  --backbone mitty --loss uniform \
  --cache-train training_data/cache/vae/pair_1s/train \
  --cache-eval  training_data/cache/vae/pair_1s/eval \
  --cache-ood   training_data/cache/vae/pair_1s/ood_eval \
  --epochs 3 --repeat 5 --save-steps 50 --eval-steps 50

# Mitty + hand_patch 加权
torchrun --nproc_per_node=4 -m src.pipeline.train \
  --backbone mitty --loss hand_patch \
  --patch-dir training_data/pair/1s/train/hand_patch \
  --cache-train training_data/cache/vae/pair_1s/train ...

# RectFlow（Route A：source 代替 Gaussian noise）
torchrun --nproc_per_node=4 -m src.pipeline.train \
  --backbone rectflow --loss uniform \
  --cache-train training_data/cache/vae/pair_1s/train ...
```

W&B tags 自动 `[backbone, loss, ...wandb-tags]`，消融表按 tag 分面。
训练用 `torch.manual_seed(args.seed + rank)` 固定随机，结果可复现。

**Legacy 等效命令**（旧脚本保留，单独维护；新实验建议统一走 `train.py`）：

```bash
# 等效于 --backbone mitty --loss uniform
python -m src.pipeline.train_mitty --cache-train ...
# 等效于 --backbone mitty --loss hand_patch --patch-dir ...
python -m src.pipeline.train_mitty --patch-dir ... --cache-train ...
# 等效于 --backbone rectflow --loss uniform
python -m src.pipeline.train_rf --cache-train ...
```

### 训练目录结构

```
training_data/log/
└── YYYY-MM-DD_HHMMSS/
    ├── data_cache/0/*.pth        # 阶段 1 embedding 缓存 (legacy DiffSynth)
    ├── ckpt/step-NNN.safetensors # 阶段 2 LoRA checkpoint
    ├── eval/step-NNN/            # eval 视频 (gt + ctrl + gen)
    └── train.log                 # 每步 loss + eval 日志
```
