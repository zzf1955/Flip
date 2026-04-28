# Wan 2.2 TI2V-5B 两阶段训练策略

## 动机

直接训练 human→robot 外观替换任务时，模型需要同时学习两件事：
1. **重建能力**：机器人外观、背景纹理、运动连贯性
2. **映射能力**：从 human 条件视频推断对应 robot 姿态和外观

将这两个学习目标解耦为两个阶段，可以降低任务难度、加速收敛。

## 两阶段设计

### Phase 1: 恒等重建（Identity Reconstruction）

**输入**：robot 视频 → **输出**：重建同一个 robot 视频

训练数据中 `control_video`（条件）和 `video`（目标）使用**同一个 robot 视频**。模型在 in-context 结构下学习：
- 将 clean 段（前 f_H 帧）的信息忠实地传递到 noisy 段（后 f_R 帧）
- 机器人手臂的外观细节、背景环境、光照条件
- 运动的时间连贯性

这一阶段的 loss 理论最优是 0（完美重建），模型有明确的学习信号。

### Phase 2: 外观替换（Appearance Transfer）

**输入**：human 视频 → **输出**：对应 robot 视频

用 Phase 1 的 LoRA checkpoint 作为 `--init-lora`，继续在真实的 (human, robot) pair 数据上训练。模型已经掌握了重建能力，只需额外学习跨外观的映射。

## 数据准备

当前维护的数据 Task 固定为三个机器人 Task：

- `Inspire_Pickup_Pillow_MainCamOnly`
- `Inspire_Put_Clothes_Into_Basket`
- `Inspire_Put_Clothes_into_Washing_Machine`

数据目录不再预切 `train/eval/ood_eval`；磁盘按 `data_type / duration / robot_task`
组织，训练运行时用 `--train-tasks` 和 `--ood-tasks` 决定 in-task 与 OOD。默认 preset
使用 Basket + Washing 作为 in-task，Pillow 作为 OOD。

### Phase 1 数据结构

```text
training_data/pair/identity_r2r/1s/<robot_task>/
├── video/pair_NNNN.mp4
├── control_video/pair_NNNN.mp4
├── metadata.csv
└── manifest.jsonl

training_data/cache/vae/identity_r2r/1s/<robot_task>/
├── pair_NNNN.pth
└── manifest.jsonl
```

identity 数据中 `control_video` 和 `video` 内容一致，cache 内会记录
`data_type=identity_r2r`、`robot_task`、`source_id`、`source_segment_id` 等运行时 split 字段。

### Phase 2 数据结构

```text
training_data/pair/h2r/1s/<robot_task>/
training_data/cache/vae/h2r/1s/<robot_task>/
training_data/cache/t5/h2r/1s/
```

Phase 2 使用 `h2r`：输入 human/control 视频，输出 robot/target 视频。

## 执行流程

### Phase 1: 生成 identity cache + 训练

```bash
# 0. 生成 split-free identity pair；--task all 只展开三个 canonical Task
python -m src.pipeline.make_robot_pair \
  --task all \
  --max-segments 500 \
  --clean

# 1. 为每个 robot task 生成 identity cache
python -m src.pipeline.mitty_cache \
  --pair-dir training_data/pair/identity_r2r/1s/Inspire_Put_Clothes_Into_Basket \
  --output   training_data/cache/vae/identity_r2r/1s/Inspire_Put_Clothes_Into_Basket \
  --t5-cache-dir training_data/cache/t5/identity_r2r/1s \
  --device   cuda:2

# 2. Phase 1 训练（恒等重建）；in-task/OOD 运行时决定
scripts/flip_run.sh train --cuda 2 -- \
  --task-name identity_r2r_1s \
  --train-tasks Inspire_Put_Clothes_Into_Basket,Inspire_Put_Clothes_into_Washing_Machine \
  --ood-tasks Inspire_Pickup_Pillow_MainCamOnly \
  --max-steps 400 --save-steps 100 --eval-steps 100 \
  --eval-video-steps 100
```

### Phase 2: 外观替换训练

```bash
# 生成 h2r pair/cache 后，用 Phase 1 最佳 ckpt 初始化
scripts/flip_run.sh train --cuda 2 -- \
  --task-name h2r_1s \
  --init-lora training_data/log/<phase1-run>/ckpt/step-0400.safetensors \
  --train-size 0 \
  --in-task-eval-size 32 \
  --in-task-video-size 4 \
  --ood-eval-size 32 \
  --ood-video-size 2 \
  --data-seed 42 \
  --max-steps 400 --save-steps 100 --eval-steps 100
```

### DDP 多卡版本

```bash
scripts/flip_run.sh train --cuda 0,1,2,3 --nproc 4 -- \
  --task-name identity_r2r_1s \
  --max-steps 400 --save-steps 100 --eval-steps 100

scripts/flip_run.sh train --cuda 0,1,2,3 --nproc 4 -- \
  --task-name h2r_1s \
  --init-lora training_data/log/<phase1-run>/ckpt/step-0400.safetensors \
  --max-steps 400 --save-steps 100 --eval-steps 100
```

## 训练预期

### Phase 1 指标

- **Train loss**：应快速收敛到接近 0（完美重建是可实现的）
- **Eval video**：生成视频应与输入 robot 视频几乎一致
- 如果 loss 不收敛，说明模型容量或学习率有问题

### Phase 2 指标

- **Train loss**：比从头训更低的起点（因为重建能力已学会）
- **Eval video**：robot 外观正确、背景合理、动作与 human 条件对齐

## 技术细节

### mitty_cache.py 在 identity 模式下的行为

`encode_video(vae, frames)` 对同一视频调用两次，生成两个相同的 latent：
- `human_latent`：作为 clean 条件段
- `robot_latent`：作为 noisy 目标段

训练时 `MittyFlowMatchLoss` 将 human_latent（clean）和加噪后的 robot_latent 沿 temporal 拼接，DiT 预测 robot 段的噪声。因为两段内容相同，模型学习的是"把 clean 段的信息透传到 noisy 段"。

### 为什么不直接修改代码

可以给 `mitty_cache.py` 加 `--identity` flag，但创建新 pair 目录的方式：
- 无需改代码，整个 pipeline 原样复用
- 数据清晰分离，不容易混淆
- Phase 1 和 Phase 2 的 cache 独立存放，可以随时切换

### --no-frames 选项

Identity cache 中 `human_frames` 和 `robot_frames` 完全相同。可以使用 `--no-frames` 减小 .pth 体积（~55MB → ~9MB/样本），eval video 时不会保存 gt/ctrl 对比帧：

```bash
python -m src.pipeline.mitty_cache \
  --pair-dir training_data/robot_pair/1s/train \
  --output training_data/cache/vae/robot_1s/train \
  --device cuda:2 --no-frames
```

## 相关文件

| 文件 | 作用 |
|------|------|
| `src/pipeline/mitty_cache.py` | 编码 pair 视频为 .pth cache |
| `src/pipeline/train_mitty.py` | Mitty LoRA 训练主脚本 |
| `src/pipeline/mitty_model_fn.py` | Mitty forward: temporal concat + partial noise |
| `src/core/wan_loader.py` | DiT/VAE/T5 直接 GPU 加载器 |
| `src/core/train_utils.py` | 训练工具函数（日志、DDP、数据加载） |
