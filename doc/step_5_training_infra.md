# 训练基础设施

训练相关的 DDP 配置、模型加载机制、微调命令、目录结构。
算法原理见 [`step_5_wan22_ti2v5b.md`](step_5_wan22_ti2v5b.md)，两阶段策略见 [`step_5_two_stage_training.md`](step_5_two_stage_training.md)。

## DDP 训练注意事项

### 模型加载

`wan_loader.py` 所有加载函数直接读到目标 GPU，不经过 CPU 中转：
- `load_dit()`: safetensors `safe_open(device="cuda:X")` (~10GB)
- `load_vae()`: `torch.load(map_location=device)` / safetensors (~0.67GB)
- `load_text_encoder()`: `torch.load(map_location=device)` (~5.5GB，仅 mitty_cache 用)
- 训练样本 cache: `load_sample(path, device=device)` 直接到 GPU
- DDP 各 rank 并行加载互不干扰，无需错峰

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
# 数据：training_data/pair/1s_identity/ (control_video = video，同一个 robot 视频)
# Cache：output/mitty_cache_1s_identity/

python -m src.pipeline.mitty_cache \
  --pair-dir training_data/pair/1s_identity/train \
  --output output/mitty_cache_1s_identity/train --device cuda:2

python -m src.pipeline.train_mitty \
  --cache-train output/mitty_cache_1s_identity/train \
  --cache-eval  output/mitty_cache_1s_identity/eval \
  --cache-ood   output/mitty_cache_1s_identity/ood_eval \
  --max-steps 400 --save-steps 50 --eval-steps 50
```

**Phase 2: 外观替换**（human→robot，用 Phase 1 ckpt 初始化）
```bash
# 数据：training_data/pair/1s/ (human≠robot，真实 pair)
# Cache：output/mitty_cache_1s/

python -m src.pipeline.train_mitty \
  --init-lora training_data/log/<phase1-run>/ckpt/step-NNNN.safetensors \
  --cache-train output/mitty_cache_1s/train \
  --cache-eval  output/mitty_cache_1s/eval \
  --cache-ood   output/mitty_cache_1s/ood_eval \
  --max-steps 400 --save-steps 50 --eval-steps 50
```

### Wan 2.2 TI2V-5B 统一训练入口（推荐）

两个消融维度：主干 `--backbone {mitty,rectflow}`、loss `--loss {uniform,hand_patch}`。
`--loss hand_patch` 必须配 `--patch-dir`，`--loss uniform` 不允许传 `--patch-dir`（argparse 硬校验）。

```bash
# Mitty + uniform（baseline）
torchrun --nproc_per_node=4 -m src.pipeline.train \
  --backbone mitty --loss uniform \
  --cache-train output/mitty_cache_1s/train \
  --cache-eval  output/mitty_cache_1s/eval \
  --cache-ood   output/mitty_cache_1s/ood_eval \
  --epochs 3 --repeat 5 --save-steps 50 --eval-steps 50

# Mitty + hand_patch 加权
torchrun --nproc_per_node=4 -m src.pipeline.train \
  --backbone mitty --loss hand_patch \
  --patch-dir training_data/pair/1s/train/hand_patch \
  --cache-train output/mitty_cache_1s/train ...

# RectFlow（Route A：source 代替 Gaussian noise）
torchrun --nproc_per_node=4 -m src.pipeline.train \
  --backbone rectflow --loss uniform \
  --cache-train output/mitty_cache_1s/train ...
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
    ├── data_cache/0/*.pth        # 阶段 1 embedding 缓存
    ├── ckpt/step-NNN.safetensors # 阶段 2 LoRA checkpoint
    ├── eval/step-NNN/            # eval 视频 (gt + ctrl + gen)
    └── train.log                 # 每步 loss + eval 日志
```
