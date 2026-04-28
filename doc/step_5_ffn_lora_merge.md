# FFN LoRA — 合并 identity LoRA + FFN 层外观重建

## 动机

identity LoRA（Phase 1）在 q/k/v/o 注意力层上学会了恒等重建。但 LoRA 容量有限，可能无法同时记住机器人的外观细节。

本阶段在 identity LoRA 基础上，额外在 FFN 层（`ffn.0`, `ffn.2`）加 LoRA，训练从降质视频重建完整视频。目标是让 FFN 层学习并记忆机器人的视觉外观。

## 架构

```
Base DiT (frozen)
  + identity LoRA (q,k,v,o) → 合并到 base weights (frozen)
  + FFN LoRA (ffn.0, ffn.2) → 新增，可训练
```

### LoRA 合并原理

`merge_lora_into_weights()` 将 identity LoRA 永久合并到基础权重：

```
base_weight += (lora_alpha / lora_rank) * lora_B @ lora_A
```

训练入口会从 checkpoint 的 LoRA A/B 权重形状自动检测 rank，并默认
`lora_alpha = detected_rank`，scaling = 1.0。合并后模型行为与冻结 LoRA
adapter 完全等价，但实现更简洁——无需在同一模型上叠加两套 PEFT adapter。

### FFN 层结构

每个 DiTBlock（共 30 层）的 FFN：

```
ffn = Sequential(
    [0]: Linear(3072, 14336)   ← ffn.0, LoRA 注入点
    [1]: GELU(tanh)
    [2]: Linear(14336, 3072)   ← ffn.2, LoRA 注入点
)
```

FFN LoRA 可训练参数：30 blocks × 2 layers × 2 × rank × dim ≈ **100M**（rank=96 时）。

## 数据

使用 `robot_patch` pipeline 生成的降质数据（详见 [`step_5_robot_patch.md`](step_5_robot_patch.md)）：

```
training_data/pair/1s_patch/
├── train/
│   ├── video/           # 原始 robot 视频（目标）
│   ├── control_video/   # 降质 robot 视频（输入条件）
│   └── patch/           # latent-space 权重 {1.0, 3.0}
└── eval/
    └── (同上)
```

`metadata.csv` 如不存在，`mitty_cache.py` 会自动从 `video/` 目录枚举生成。

## 执行流程

### 1. 编码 cache

```bash
# train split
LD_PRELOAD=/home/leadtek/miniconda3/envs/flip/lib/libjpeg.so.8 \
  python -m src.pipeline.mitty_cache \
  --pair-dir training_data/pair/1s_patch/train \
  --output   training_data/cache/1s_patch/train \
  --device cuda:0 --no-frames

# eval split
LD_PRELOAD=/home/leadtek/miniconda3/envs/flip/lib/libjpeg.so.8 \
  python -m src.pipeline.mitty_cache \
  --pair-dir training_data/pair/1s_patch/eval \
  --output   training_data/cache/1s_patch/eval \
  --device cuda:0 --no-frames
```

### 2. 训练

```bash
scripts/flip_run.sh train --cuda 2,3 --nproc 2 -- \
  --task-name attn_ffn_selected \
  --merge-lora training_data/log/identity-mitty-bs8-2k/ckpt/step-4850.safetensors \
  --lora-target-modules "ffn.0,ffn.2" \
  --lora-rank 96 \
  --max-steps 2000 --save-steps 100 --eval-steps 100 \
  --eval-video-steps 100 --eval-video-samples-in-task 4 \
  --lr 1e-4 --warmup-steps 50 \
  --wandb-run-name ffn-patch-mitty-bs2-2k \
  --wandb-tags ffn_lora patch_recon
```

推荐通过统一入口 `scripts/flip_run.sh train`：

```bash
scripts/flip_run.sh train --cuda 2,3 --nproc 2 -- \
  --task-name attn_ffn_selected \
  --merge-lora training_data/log/identity-mitty-bs8-2k/ckpt/step-4850.safetensors \
  --lora-target-modules "ffn.0,ffn.2" \
  --lora-rank 96 \
  --max-steps 2000 --save-steps 100 --eval-steps 100 \
  --eval-video-steps 100 \
  --lr 1e-4 --warmup-steps 50 \
  --wandb-run-name ffn-patch-mitty-bs2-2k \
  --wandb-tags ffn_lora patch_recon
```

## 新增 CLI 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--merge-lora` | `""` | 预训练 LoRA checkpoint 路径，可重复传入，按顺序合并到 base weights |

被合并 LoRA 的 rank 会从 checkpoint 自动检测；无法唯一检测时训练直接失败。

## 训练预期

- **Patch loss**：降质区域权重 3x，模型重点学习被破坏的机器人外观
- **Train loss**：初始高于 identity（降质输入 ≠ 目标），应稳步下降
- **Eval video**：对比 control（降质）和 gen（重建），降质区域应清晰恢复

## 推理时的权重组合

训练产出的是 FFN LoRA checkpoint。推理时需要两步加载：

1. Base DiT + merge identity LoRA（合并到权重）
2. 加载 FFN LoRA（作为 adapter）

```python
# 伪代码
model = load_dit(...)
merge_lora_into_weights(model, identity_ckpt)
model = inject_lora(model, target_modules=["ffn.0", "ffn.2"], rank=96)
model.load_state_dict(load_file(ffn_ckpt), strict=False)
```

## 相关文件

| 文件 | 作用 |
|------|------|
| `src/pipeline/train_mitty.py` | `merge_lora_into_weights()` + `--merge-lora` 参数 |
| `src/pipeline/train.py` | 统一入口，同步支持 `--merge-lora` |
| `src/pipeline/mitty_cache.py` | metadata.csv 自动生成 fallback |
| `src/pipeline/robot_patch.py` | 降质数据生成 |
| `doc/step_5_robot_patch.md` | 降质数据生成文档 |
| `doc/step_5_two_stage_training.md` | 两阶段训练策略 |
