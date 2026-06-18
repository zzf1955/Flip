## DreamZero 风格 Wan2.2-5B video-action 原型

task067 记录下一阶段探索路线：在没有实机控制闭环的前提下，先做离线
closed-loop video-action world-action model。主线采用 `Wan-AI/Wan2.2-TI2V-5B`
作为视频基座，参考 DreamZero 的 action/state register 和 joint video-action diffusion
训练方式。

当前结论：

- 未看到官方发布的 Wan2.2-TI2V-5B DreamZero 训练好 checkpoint；官方 DreamZero
  公开 checkpoint 主要是 Wan2.1-I2V-14B 系。
- DreamZero 仓库已有 Wan2.2-TI2V-5B backbone 配置和训练脚本，但语义是从 Wan2.2-5B
  视频基座训练 DreamZero-style 模型，不是加载已有 DreamZero-5B policy。
- `train_architecture=lora` 时，LoRA 只作用在 Wan DiT 主干；`state_encoder`、
  `action_encoder`、`action_decoder` 是新加模块，需要从头全量训练。
- text encoder、image encoder、VAE 冻结；Wan DiT base weight 冻结，只训练 LoRA
  adapter 和 action/state register 相关模块。
- 离线 rollout 时，上一轮生成的视频可以作为下一轮 visual context；上一轮 action
  不应直接作为下一轮 action condition，state 需要固定、由 action 简化推进，或由后续
  数据/模型显式维护。

建议的最小训练配置：

```text
backbone: Wan2.2-TI2V-5B
resolution: 160x320
num_frames: 33
num_frame_per_block: 2
action_horizon: 24
num_action_per_block: 24
num_state_per_block: 1
train_architecture: lora
lora_target_modules: q,k,v,o,ffn.0,ffn.2
lora_rank: 16 -> 32/64
lora_alpha: same as rank
```

实现时必须验证 checkpoint 保存和恢复：`save_lora_only` 需要覆盖 LoRA adapter 以及
`state_encoder/action_encoder/action_decoder`，恢复后必须打印 missing / unexpected keys
并跑同一 batch inference。若 key rewrite 或 PEFT wrapper 导致 register 参数没有恢复，
应直接修复保存/加载逻辑，不允许静默随机初始化。

### Task067 单卡 smoke 记录

当前已在 DreamZero 独立 checkout 中实现低显存 adapter 和 trainable-only save wrapper；
FLIP 侧只记录运行结果和路径。task067 的 Wan2.2-5B + DreamZero-style LoRA 原型只按单卡
运行，不使用多卡 DDP / DeepSpeed。

已验证的单卡 smoke（2026-06-05，GPU 2，RTX 4090D 24GB）：

- 训练配置：`global_step=1`、`per_device_train_batch_size=1`、`LORA_RANK=4`、
  `LORA_ALPHA=4`、`NUM_GPUS=1`、`training_args.deepspeed=null`。
- human2robot 数据已转成 DreamZero / Gear 可读的 LeRobot-style 目录：
  `training_data/dreamzero_h2r_v1`（历史目录名），包含 190 个有效 episode、73,885 个训练 step，
  state/action 维度均为 7，video keys 为 `robot_camera` / `human_camera`。
- DiT 使用 `diffusion_pytorch_model-bf16.safetensors`，通过
  `safe_open(..., device="cuda")` 直接读入 GPU；T5 / CLIP / VAE `.pth` 权重使用
  `torch.load(..., mmap=True)`，并在 encode 后 offload。
- 本地 Wan2.2 5B DiT safetensors 只有 825 个 base key，不包含 DreamZero TI2V wrapper
  期望的 `cross_attn.*_img` / `img_emb` key；adapter 在 direct-GPU load 后对这条
  image branch 做确定性零初始化，避免冻结随机图像分支，并让 pretrained load 的
  missing keys 只剩 `state_encoder`、`action_encoder` 和 `action_decoder`。
- 训练前 GPU memory 日志为 `10.462 GB`；DiT direct-GPU 加载瞬时显存观测约
  `20.7 GB`；训练 step 低于 24GB 单卡容量。
- loss：`dynamics_loss_avg=0.5703880786895752`、
  `action_loss_avg=0.22939424216747284`、`train_loss=0.7997823357582092`。
- `save_lora_only` 输出
  `training_data/log/dreamzero_h2r_wam_wan22_lora_smoke_actionstate_only_v3/model.safetensors`，
  约 89.9MB，614 个 tensor，44,890,144 个 trainable 参数；没有生成
  `model-0000*.safetensors` 全量分片。
- checkpoint 恢复 smoke 通过：614 个 trainable tensor 与恢复后
  `model.state_dict()` 对应 tensor exact compare 通过，`unexpected_keys_count=0`。
- 离线 rollout smoke 通过：`seed=42`，滚动 2 个 chunk，`action_chunks` shape 为
  `(2,1,24,32)`，`final_video_latent` shape 为 `(1,48,2,10,20)`；导出的
  `output/task067_rollout_smoke_actionstate_only_v3/rollout_smoke.mp4` 可被 OpenCV 打开，
  分辨率为 `320x160`，共 14 帧。

当前 smoke 为了压低单卡风险使用 `LORA_RANK=4` / `LORA_ALPHA=4`；正式小步训练建议先用
`rank=16` 复跑，确认 24GB 单卡峰值稳定后再尝试 `32/64`。

### Robot WAM train-wan 调参汇总

task072 基于 task071 的 human2robot top-level 1157 episode robot-only cache，复用以下产物：

- T5 cache：`training_data/cache/t5/robot_wam_h2r_top1157_s8`
- VAE cache：`training_data/cache/vae/robot_wam/h2r_top1157_s8_160x320`
- baseline run：`training_data/log/robot_wam/h2r_top1157_s8_wan_lora`

调参 run 统一输出到：

```text
training_data/log/robot_wam/h2r_top1157_s8_tune/<run_name>/
```

每个有效 run 至少应包含：

- `config.json`
- `train_log.jsonl`
- `train_summary.json`
- `best_summary.json`
- `best_checkpoint.safetensors`
- 至少一个 `step_*.safetensors`

汇总工具：

```bash
/home/leadtek/miniconda3/envs/flip/bin/python -m src.tools.summarize_robot_wam_tune \
  --baseline-dir /disk_n/zzf/flip/training_data/log/robot_wam/h2r_top1157_s8_wan_lora \
  --baseline-name baseline \
  --baseline-lr 1e-4 \
  --tune-dir /disk_n/zzf/flip/training_data/log/robot_wam/h2r_top1157_s8_tune \
  --default-state-tokens 4 \
  --out-csv /disk_n/zzf/flip/training_data/log/robot_wam/h2r_top1157_s8_tune/summary.csv \
  --out-md /disk_n/zzf/flip/training_data/log/robot_wam/h2r_top1157_s8_tune/summary.md
```

`summarize_robot_wam_tune` 会读取 baseline/tune run 的配置、日志和 best summary，
输出统一 CSV/Markdown，并用 `safetensors.safe_open` 对 best checkpoint 做 hard audit：
禁止 `human` / `control` key，只允许 LoRA、`state_encoder`、`action_decoder` trainable
权重。审计失败时默认直接退出失败。

task072 已完成 4 个 1000-step tune run：

| run | lr | action_loss_weight | best_eval_loss | best_eval_video_loss | best_eval_action_loss |
| --- | --- | --- | --- | --- | --- |
| `r16_lr5e-5_aw1_s1k` | 5e-5 | 1.0 | 1368.6613 | 0.1210 | 1368.5404 |
| `r16_lr2e-4_aw1_s1k` | 2e-4 | 1.0 | 1076.8829 | 0.0893 | 1076.7936 |
| `r16_lr1e-4_aw0p1_s1k` | 1e-4 | 0.1 | 112.7114 | 0.0813 | 1126.3013 |
| `r16_lr1e-4_aw0p01_s1k` | 1e-4 | 0.01 | 11.5169 | 0.0795 | 1143.7440 |

同一 `action_loss_weight=1.0` 口径下，`r16_lr2e-4_aw1_s1k` 优于 task071 baseline
`1145.6176`，下一轮推荐先把该配置延长到 `max_steps=3000`。`aw=0.1/0.01` 的
weighted total loss 不应直接与 `aw=1.0` 排名混用；如需要更低 video loss，可把
`lr=1e-4, action_loss_weight=0.1` 保留作平衡对照。

后续 `train-wan` 的 `config.json` 应记录 optimizer、训练步数、eval/save/log 间隔、
seed 和 state/action 模型参数。早期 run 如果缺少这些字段，汇总工具可以从稳定 run name
和 `train_log.jsonl` 中补齐 `lr/max_steps`。

### Robot WAM 固定 split 完整训练

task073 将 human2robot top1157 robot-only WAM 从 task072 的短调参升级为固定 split
完整训练。固定 split 输出在：

```text
training_data/robot_wam/splits/h2r_top1157_s8_fixed_v1/
├── train.jsonl
├── eval_in_task.jsonl
├── eval_ood.jsonl
└── summary.json
```

Split 规则：

- OOD task 完整留出：`grab_cube2_v1`、`push_box_random_v1`、
  `push_box_two_v1`、`roll`。
- 其余 15 个 task 作为训练 task。
- 训练 task 内按 episode 稳定排序，尾部 10% episode 进入 `eval_in_task`，
  每个 task 至少保留 1 个 held-out episode。
- train / eval_in_task / eval_ood 的 `sample_id` 互斥，OOD task 不进入 train。

Split 统计：

| split | samples | tasks | episodes |
| --- | ---: | ---: | ---: |
| train | 39024 | 15 | 861 |
| eval_in_task | 4693 | 15 | 105 |
| eval_ood | 4931 | 4 | 189 |

生成命令：

```bash
scripts/flip_run.sh robot_wam -- build-split \
  --cache-dir /disk_n/zzf/flip/training_data/cache/vae/robot_wam/h2r_top1157_s8_160x320 \
  --output-dir /disk_n/zzf/flip/training_data/robot_wam/splits/h2r_top1157_s8_fixed_v1 \
  --ood-tasks roll,push_box_random_v1,push_box_two_v1,grab_cube2_v1 \
  --in-task-eval-episode-fraction 0.1 \
  --min-eval-episodes-per-task 1
```

`train-wan` 支持以下固定 split 参数：

- `--train-manifest`
- `--eval-manifest`
- `--eval-ood-manifest`
- `--best-metric`，默认 `eval_mean_loss`

每次 eval 会记录：

- `eval_in_task_loss` / `eval_in_task_video_loss` / `eval_in_task_action_loss`
- `eval_ood_loss` / `eval_ood_video_loss` / `eval_ood_action_loss`
- `eval_mean_loss`

完整训练输出：

```text
training_data/log/robot_wam/h2r_top1157_s8_fixed_v1_full/
├── r16_lr1e-4_aw1_s39024_eval512/
├── r16_lr2e-4_aw1_s39024_eval512/
├── r32_lr2e-4_aw1_s39024_eval512/
├── summary.csv
└── summary.md
```

训练内 eval 为了控制耗时，每个 eval split 最多抽样 512 条。三组完整
39,024-step run 的 best 结果：

| run | rank | lr | best_step | best_eval_mean_loss | best_in_task | best_ood |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `r16_lr1e-4_aw1_s39024_eval512` | 16 | 1e-4 | 39024 | 326.157 | 372.536 | 279.778 |
| `r16_lr2e-4_aw1_s39024_eval512` | 16 | 2e-4 | 30000 | 349.804 | 396.927 | 302.680 |
| `r32_lr2e-4_aw1_s39024_eval512` | 32 | 2e-4 | 10000 | 408.997 | 474.481 | 343.513 |

当前训练内推荐 checkpoint：

```text
training_data/log/robot_wam/h2r_top1157_s8_fixed_v1_full/
  r16_lr1e-4_aw1_s39024_eval512/best_checkpoint.safetensors
```

该 checkpoint audit 通过：494 个 trainable tensors，44,719,272 参数，无
`human` / `control` key。

`eval-wan` 用于 eval-only 恢复 trainable-only checkpoint。恢复时会同时加载
DiT LoRA、`state_encoder` 和 `action_decoder`；如果 checkpoint 中缺少预期
trainable 权重会直接暴露为指标/加载问题，不做旧行为 fallback。

推荐 checkpoint 的完整 fixed eval 命令使用 `--eval-batches 0 --max-eval-samples 0`，
覆盖 `eval_in_task=4693` 和 `eval_ood=4931` 全部样本：

```bash
scripts/flip_run.sh robot_wam --cuda 1 -- eval-wan \
  --cache-train /disk_n/zzf/flip/training_data/cache/vae/robot_wam/h2r_top1157_s8_160x320 \
  --cache-eval /disk_n/zzf/flip/training_data/cache/vae/robot_wam/h2r_top1157_s8_160x320 \
  --eval-manifest /disk_n/zzf/flip/training_data/robot_wam/splits/h2r_top1157_s8_fixed_v1/eval_in_task.jsonl \
  --eval-ood-manifest /disk_n/zzf/flip/training_data/robot_wam/splits/h2r_top1157_s8_fixed_v1/eval_ood.jsonl \
  --t5-cache-dir /disk_n/zzf/flip/training_data/cache/t5/robot_wam_h2r_top1157_s8 \
  --batch-size 1 --workers 4 --prefetch-factor 2 --persistent-workers \
  --lora-rank 16 \
  --lora-attn-types self,cross \
  --lora-attn-projections q,k,v,o \
  --state-tokens 4 --state-hidden-dim 1024 --action-hidden-dim 512 --action-depth 3 \
  --eval-batches 0 \
  --max-eval-samples 0 \
  --init-lora /disk_n/zzf/flip/training_data/log/robot_wam/h2r_top1157_s8_fixed_v1_full/r16_lr1e-4_aw1_s39024_eval512/best_checkpoint.safetensors \
  --output-json /disk_n/zzf/flip/training_data/log/robot_wam/h2r_top1157_s8_fixed_v1_full/r16_lr1e-4_aw1_s39024_eval512/full_eval_best.json \
  --device cuda:0 --no-skip-dit-load
```

完整 eval 结果：

| metric | value |
| --- | ---: |
| `eval_in_task_loss` | 163.882 |
| `eval_in_task_video_loss` | 0.094833 |
| `eval_in_task_action_loss` | 163.787 |
| `eval_ood_loss` | 2108.090 |
| `eval_ood_video_loss` | 0.104568 |
| `eval_ood_action_loss` | 2107.985 |
| `eval_mean_loss` | 1135.986 |

因此后续报告 robot_wam 结果时必须区分两种口径：

- 训练内抽样 eval：用于同一 run 内选择 checkpoint，本轮最佳为 `326.157`。
- 完整 fixed eval：用于最终外推判断，本轮 OOD action loss 很高，`eval_mean_loss=1135.986`。

当前结论是 `r16_lr1e-4` 比 `r16_lr2e-4` 更适合 1 epoch 训练，rank32 在本轮没有优势；
下一轮应优先提高 OOD 覆盖或降低 eval 抽样噪声，而不是继续单纯增大 rank。