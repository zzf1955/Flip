
### AdaWorld Action Encoder

> 归档说明：AdaWorld action encoder/decoder 现在属于 `src.pipeline.archive.idm.*`
> 历史实验线，不再作为当前主线入口维护。

`src.pipeline.archive.idm.adaworld_action_encoder` 只接入 AdaWorld 的 LAM action encoder，不运行
AdaWorld 后续 world model。该入口从 Humanoid Everyday H1 LeRobot 数据读取相邻两帧
egocentric RGB，按 AdaWorld LAM 训练口径中心裁方、resize 到 256、归一化到 `[0,1]`，
并在提取过程中实时落盘 latent：

- 输入：`(frame_t, frame_{t+1})`，来自
  `videos/chunk-*/egocentric/episode_*.mp4`。
- 输出：`z_t`，shape 为 `[N, 32]` 的 continuous latent action。
- 该入口不读取真实 action label，也不训练下游 action head。
- 参考代码仓库为 `ref-AdaWorld`；LAM checkpoint 只使用
  `ref-AdaWorld-hf/lam.ckpt`，不下载或加载 `adaworld.safetensors` world model。
- world model checkpoint 的 HF LFS 指针大小约 `11.46 GB`，对应一个明显重于 LAM 的
  video diffusion 模型；当前 FLIP 目标只保留 action encoder，不建议把 world model 作为
  小规模复现目标。

输出目录包含：

- `latent_actions.npy`：提取过程中实时写入的 memmap latent 矩阵。
- `latent_actions.npz`：最终封装的 `latent_actions [N,32]`，以及 episode / chunk /
  `rel_frame_t` / `rel_frame_tp1` 数组。
- `manifest.jsonl`：逐样本视频路径、parquet 路径、帧号和 latent list。
- `summary.json`：AdaWorld revision、checkpoint、预处理配置、latent mean/std/min/max，
  以及 `latent_actions.npy` 路径。

H1 action encoder smoke 示例：

```bash
CUDA_VISIBLE_DEVICES=2 python -m src.pipeline.archive.idm.adaworld_action_encoder extract \
  --device cuda:0 \
  --data-root /disk_n/zzf/flip/data/humanoid-everyday-h1-chunks0-6-8-200 \
  --output-dir tmp/adaworld_action_encoder_h1_smoke \
  --max-samples 8 \
  --max-pairs-per-episode 1 \
  --batch-size 1 \
  --dtype fp16
```

如果使用原始 `data/humanoid-everyday-h1-chunks0-6-8-200`，当前复核没有发现不可读 parquet；
入口仍然对不可读 parquet 直接失败，不做静默跳过。

### AdaWorld Latent Action Decoder

`src.pipeline.archive.idm.adaworld_action_decoder` 只训练 AdaWorld latent action 的下游解码器，不再
回到图像端。它读取 task054 产出的 `latent_actions.npz`，其中每一行对应
`(frame_t, frame_{t+1}) -> z_t`，并通过 `episode/chunk/rel_frame_t` 回查 H1 LeRobot
parquet 中同一帧的 `action` 标签，形成监督对：

- 输入：`z_t`，shape 为 `[N, 32]` 的 AdaWorld continuous latent action。
- 输出：H1 `action_t`，当前实现按 26 维 `action` 向量训练。
- 模型：仍保留 `mlp` baseline；当前推荐默认是 `residual_mlp`，用
  `hidden_dim=384`、`depth=4`、`dropout=0.02`、LayerNorm、shared output head、
  `lr=8e-4`、`weight_decay=1e-4`、`betas=(0.9,0.95)`、cosine scheduler + 5% warmup、
  `min_lr_ratio=0.02`。`gated_mlp`、`per_dim` head 和 grouped head 保留作消融对照。
  训练和验证都使用 latent / action 双边标准化。
- 训练目标：默认是标准化后的 action MSE；也可显式切到 weighted MSE、SmoothL1、
  weighted SmoothL1，并用轻量 variance calibration penalty 约束预测方差。验证时回到
  原始 action 空间，输出 MSE、R2、correlation、方差比等指标。checkpoint 保存完整
  decoder 架构、head、loss、优化器和 warmup 配置，`validate` / `eval` 可复算 replay。

输出目录包含：

- `checkpoint.pt` / `best_checkpoint.pt`：decoder 权重、latent 统计量、action 统计量和
  训练配置。
- `train_loss.csv` / `eval_loss.csv`：训练与验证损失。
- `val_predictions.csv` / `best_val_predictions.csv`：逐样本预测表。
- `metrics.json` / `val_metrics.json` / `best_val_metrics.json`：评估摘要。
- `loss_curve.png`：训练 / 验证损失曲线。

训练示例：

```bash
CUDA_VISIBLE_DEVICES=2 python -m src.pipeline.archive.idm.adaworld_action_decoder train \
  --device cuda:0 \
  --data-root /disk_n/zzf/flip/data/humanoid-everyday-h1-chunks0-6-8-200 \
  --latent-path tmp/adaworld_action_encoder_h1_smoke/latent_actions.npz \
  --output-dir tmp/adaworld_action_decoder_h1_smoke \
  --max-samples 8 \
  --steps 100 \
  --batch-size 8 \
  --eval-every 50 \
  --val-max-samples 4
```

H1 全量 1600 episode 训练记录：

```bash
CUDA_VISIBLE_DEVICES=1 python -m src.pipeline.archive.idm.adaworld_action_decoder train \
  --device cuda:0 \
  --data-root /disk_n/zzf/flip/data/humanoid-everyday-h1-chunks0-6-8-200 \
  --latent-path /disk_n/zzf/flip/tmp/adaworld_action_encoder_h1_full_t057/latent_actions.npz \
  --output-dir tmp/adaworld_action_decoder_h1_full_t057 \
  --split-by episode \
  --steps 2000 \
  --batch-size 1024 \
  --eval-every 500 \
  --log-every 100 \
  --workers 0
```

本次全量 artifact 来自 `tmp/adaworld_action_encoder_h1_full_t057/latent_actions.npz`：

- H1 episode：`1600`
- 相邻帧 latent/action 样本：`560422`
- train split：`488936` samples / `1400` episodes
- val split：`71486` samples / `200` episodes
- split 口径：episode-level，train/val episode 不重叠

Held-out best checkpoint 结果：

- `action_mse = 0.07853357493877411`
- `mean_baseline_action_mse = 0.15635326504707336`
- `action_mean_dim_r2 = 0.504274274294193`
- `action_mean_dim_corr = 0.7063991037698892`
- `action_pred_std_ratio_mean = 0.7120954027542701`

全量 eval 结果：

- checkpoint：`tmp/adaworld_action_decoder_h1_full_t057/best_checkpoint.pt`
- 输出：`tmp/adaworld_action_decoder_h1_full_t057_eval_best/metrics.json`
- 样本数：`560422`
- `action_mse = 0.07298979163169861`
- `mean_baseline_action_mse = 0.15620023012161255`
- `action_mean_dim_r2 = 0.5340813008638529`
- `action_mean_dim_corr = 0.7275451834385211`
- `action_pred_std_ratio_mean = 0.7190383053742923`

`validate` 复算 `best_checkpoint.pt` 与训练保存的 `best_val_metrics.json` 一致。
与同 split mean baseline 相比，held-out action MSE 约降低 `49.8%`；与全量 eval mean
baseline 相比，action MSE 约降低 `53.3%`。历史两帧 RGB H1 IDM 结果使用不同样本量、
split 或 target 语义，只作为参考，不作为严格同 split 对照。

### AdaWorld Latent Action Decoder 优化版

task061 在 task057 的完整 latent artifact 和 episode-level split 上继续优化 decoder，
不再改 latent 提取口径，只调学习率、warmup、网络宽度/深度、残差结构和少量正则。
当前推荐配置是 `residual_mlp`：

```bash
CUDA_VISIBLE_DEVICES=1 python -m src.pipeline.archive.idm.adaworld_action_decoder train \
  --device cuda:0 \
  --data-root /disk_n/zzf/flip/data/humanoid-everyday-h1-chunks0-6-8-200 \
  --latent-path /disk_n/zzf/flip/tmp/adaworld_action_encoder_h1_full_t057/latent_actions.npz \
  --output-dir tmp/adaworld_action_decoder_h1_full_t061 \
  --split-by episode \
  --steps 3000 \
  --batch-size 1024 \
  --eval-every 500 \
  --log-every 100 \
  --workers 0 \
  --decoder-arch residual_mlp \
  --hidden-dim 256 \
  --depth 4 \
  --dropout 0.02 \
  --layer-norm \
  --lr 5e-4 \
  --weight-decay 1e-4 \
  --adam-beta1 0.9 \
  --adam-beta2 0.95 \
  --lr-scheduler cosine \
  --min-lr-ratio 0.02 \
  --lr-warmup-ratio 0.05
```

task061 的 held-out best checkpoint 指标：

- `action_mse = 0.054645732045173645`
- `mean_baseline_action_mse = 0.15635326504707336`
- `action_mean_dim_r2 = 0.6565681374990023`
- `action_mean_dim_corr = 0.809087702861199`
- `action_pred_std_ratio_mean = 0.8399375424935267`

task061 的全量 eval 指标：

- `action_mse = 0.04235832020640373`
- `mean_baseline_action_mse = 0.15620023012161255`
- `action_mean_dim_r2 = 0.72525387773147`
- `action_mean_dim_corr = 0.850245631658114`
- `action_pred_std_ratio_mean = 0.8563885574157422`

对比 task057 baseline，这个 residual MLP decoder 在 held-out 上把 `action_mse`
从 `0.07853357493877411` 降到 `0.054645732045173645`，约降低 `30.4%`；在全量
eval 上把 `action_mse` 从 `0.07298979163169861` 降到 `0.04235832020640373`，
约降低 `42.0%`。高误差维度仍主要集中在 `action_dim_06/07/08/09/10/22/23`，
但这些维度的预测方差已经明显回升，整体更接近 target 分布而不是均值回归。

复算命令：

```bash
CUDA_VISIBLE_DEVICES=1 python -m src.pipeline.archive.idm.adaworld_action_decoder validate \
  --device cuda:0 \
  --checkpoint tmp/adaworld_action_decoder_h1_full_t061/best_checkpoint.pt \
  --output-dir tmp/adaworld_action_decoder_h1_full_t061_validate_best \
  --workers 0
```

```bash
CUDA_VISIBLE_DEVICES=1 python -m src.pipeline.archive.idm.adaworld_action_decoder eval \
  --device cuda:0 \
  --checkpoint tmp/adaworld_action_decoder_h1_full_t061/best_checkpoint.pt \
  --output-dir tmp/adaworld_action_decoder_h1_full_t061_eval_best \
  --workers 0
```

### AdaWorld Latent Action Decoder 二阶段优化版

task063 在 task061 的最佳 checkpoint 预测表上先生成逐维诊断，再做容量、学习率、
正则、gated block、loss 和 head 消融。诊断输入和输出：

- 输入：`tmp/adaworld_action_decoder_h1_full_t061/best_val_predictions.csv`
- 输入：`tmp/adaworld_action_decoder_h1_full_t061_eval_best/predictions.csv`
- 输出：`tmp/adaworld_action_decoder_t063_analysis/per_dim_summary.csv`
- 输出：`tmp/adaworld_action_decoder_t063_analysis/loss_weights.json`

task061 剩余绝对 MSE 最高的 full eval 维度仍是
`action_dim_06/07/22/08/09/10/23`。held-out normalized MSE 权重更关注
`action_dim_05/11/25/24/09` 等相对难维度，但 weighted loss 在本轮没有超过直接优化总
MSE 的配置。

1500-step sweep 结论：

| 配置 | held-out action MSE | 方差比 |
|------|---------------------|--------|
| `hidden=384, lr=8e-4` | `0.052365291863679886` | `0.848025924884356` |
| `hidden=384, per_dim head` | `0.05515256151556969` | `0.8425686244781201` |
| `hidden=384, weighted_mse` | `0.05591168254613876` | `0.8340834585519937` |
| `hidden=384` | `0.05593988299369812` | `0.8334829944830674` |
| `hidden=384, variance_loss_weight=0.03` | `0.05599823221564293` | `0.8380882464922391` |
| `lr=8e-4` | `0.056573834270238876` | `0.8274800479412079` |
| `depth=6` | `0.0578899160027504` | `0.8200816328708942` |
| `gated_mlp` | `0.05870381370186806` | `0.815233615728525` |
| `dropout=0.0` | `0.06048283725976944` | `0.805783198429988` |
| `weight_decay=1e-5` | `0.06090730428695679` | `0.8092035972155057` |
| `hidden=192` | `0.06515232473611832` | `0.7895869520994333` |
| `lr=3e-4` | `0.06696344912052155` | `0.7816409056003277` |

当前推荐配置升级为：

```bash
CUDA_VISIBLE_DEVICES=2 python -m src.pipeline.archive.idm.adaworld_action_decoder train \
  --device cuda:0 \
  --data-root /disk_n/zzf/flip/data/humanoid-everyday-h1-chunks0-6-8-200 \
  --latent-path /disk_n/zzf/flip/tmp/adaworld_action_encoder_h1_full_t057/latent_actions.npz \
  --output-dir tmp/adaworld_action_decoder_t063_full_c09_h384_lr8e4 \
  --split-by episode \
  --steps 3000 \
  --batch-size 1024 \
  --eval-every 500 \
  --log-every 100 \
  --workers 0 \
  --decoder-arch residual_mlp \
  --hidden-dim 384 \
  --depth 4 \
  --dropout 0.02 \
  --layer-norm \
  --head-arch shared \
  --loss-type mse \
  --lr 8e-4 \
  --weight-decay 1e-4 \
  --adam-beta1 0.9 \
  --adam-beta2 0.95 \
  --lr-scheduler cosine \
  --min-lr-ratio 0.02 \
  --lr-warmup-ratio 0.05
```

task063 最佳 held-out checkpoint：

- 输出目录：`tmp/adaworld_action_decoder_t063_full_c09_h384_lr8e4`
- `action_mse = 0.05023810639977455`
- `mean_baseline_action_mse = 0.15635326504707336`
- `action_mean_dim_r2 = 0.6858815573729001`
- `action_mean_dim_corr = 0.8282286180899694`
- `action_pred_std_ratio_mean = 0.8748558117793157`

task063 最佳全量 eval：

- 输出目录：`tmp/adaworld_action_decoder_t063_full_c09_eval_best`
- `action_mse = 0.029545826837420464`
- `mean_baseline_action_mse = 0.15620023012161255`
- `action_mean_dim_r2 = 0.8002844131909884`
- `action_mean_dim_corr = 0.893933926637356`
- `action_pred_std_ratio_mean = 0.9002112241891714`

相比 task061，task063 最佳配置在 held-out 上把 `action_mse` 从
`0.054645732045173645` 降到 `0.05023810639977455`，约降低 `8.1%`；全量 eval 从
`0.04235832020640373` 降到 `0.029545826837420464`，约降低 `30.3%`。这轮结果说明
task061 的 `610k` 参数量不是容量上限，`1.36M` 参数的 `hidden=384` shared-head
decoder 更合适；继续堆到 per-dim head 的约 `5.05M` 参数没有带来同等收益。

复算命令：

```bash
CUDA_VISIBLE_DEVICES=2 python -m src.pipeline.archive.idm.adaworld_action_decoder validate \
  --device cuda:0 \
  --checkpoint tmp/adaworld_action_decoder_t063_full_c09_h384_lr8e4/best_checkpoint.pt \
  --output-dir tmp/adaworld_action_decoder_t063_full_c09_validate_best \
  --workers 0
```

```bash
CUDA_VISIBLE_DEVICES=2 python -m src.pipeline.archive.idm.adaworld_action_decoder eval \
  --device cuda:0 \
  --checkpoint tmp/adaworld_action_decoder_t063_full_c09_h384_lr8e4/best_checkpoint.pt \
  --output-dir tmp/adaworld_action_decoder_t063_full_c09_eval_best \
  --workers 0
```

小规模 smoke 示例：

```bash
CUDA_VISIBLE_DEVICES=2 python -m src.pipeline.archive.idm.wan_pair_idm train \
  --device cuda:0 \
  --task-short Inspire_Pickup_Pillow_MainCamOnly \
  --output-dir tmp/wan_pair_idm_pillow_smoke \
  --max-samples 128 \
  --steps 10 \
  --batch-size 8 \
  --eval-every 5 \
  --val-max-samples 32 \
  --resize 256x256
```

Pick up Pillow 正式训练示例：

```bash
CUDA_VISIBLE_DEVICES=2 python -m src.pipeline.archive.idm.wan_pair_idm train \
  --device cuda:0 \
  --task-short Inspire_Pickup_Pillow_MainCamOnly \
  --output-dir tmp/wan_pair_idm_pillow_s4000 \
  --max-samples 0 \
  --steps 4000 \
  --batch-size 16 \
  --lr 1e-4 \
  --lr-scheduler cosine \
  --min-lr-ratio 0.05 \
  --eval-every 250 \
  --val-max-samples 2048 \
  --resize 256x256
```

完整 held-out 复算：

```bash
CUDA_VISIBLE_DEVICES=2 python -m src.pipeline.archive.idm.wan_pair_idm validate \
  --device cuda:0 \
  --task-short Inspire_Pickup_Pillow_MainCamOnly \
  --checkpoint tmp/wan_pair_idm_pillow_s4000/best_checkpoint.pt \
  --output-dir tmp/wan_pair_idm_pillow_s4000/validate_all \
  --max-samples 0 \
  --val-max-samples 0 \
  --batch-size 16 \
  --resize 256x256
```

三任务 H2R Baseline/Ours 生成视频 action 复算使用 `eval-h2r` 子命令。该入口按
`robot_task` 分派到 Collect Clothes、Washing Machine、Pickup Pillow 三个 IDM
checkpoint，只接受这三个任务；records 中出现其它任务会直接失败。当前只统计
`augment=normal` 的 records，跳过 hflip 增强样本，避免翻转视频和未翻转 action label
不一致污染 action 误差。Washing Machine
历史 eval records 可能使用无 `MainCamOnly` 后缀的
`Inspire_Put_Clothes_into_Washing_Machine`，脚本会显式把它映射到
Washing Machine checkpoint，同时仍从对应无后缀原始数据和 segment 中解析 GT action。

示例命令：

```bash
CUDA_VISIBLE_DEVICES=2 python -m src.pipeline.archive.idm.wan_vae_idm eval-h2r \
  --device cuda:0 \
  --run Baseline=training_data/log/final_mitty_0507_004901-h2r_1s-400d_r96_self_qkv_1000s_0507_004922 \
  --run Ours=training_data/log/final_ours_step3_0507_004839-h2r_1s-400d_r96_self_qkvo_ffn_1000s_0507_051842 \
  --collect-checkpoint .worktrees/t047/tmp/wan_vae_idm_collect_stride05_s4000_arm2_cosine/best_checkpoint.pt \
  --wash-checkpoint .worktrees/t048/tmp/wan_vae_idm_wash_stride05_s4000_arm2_cosine/best_checkpoint.pt \
  --pillow-checkpoint .worktrees/t048/tmp/wan_vae_idm_pillow_stride05_s4000_arm2_cosine/best_checkpoint.pt \
  --output-dir output/idm_h2r_action_eval \
  --resize 256x256
```

输出写到 `output/idm_h2r_action_eval/`，每个 run 子目录包含
`per_sample_actions.jsonl`、`per_sample_metrics.csv`、`summary_by_task.csv` 和
`config.json`。根目录额外包含跨 run 的 `per_sample_metrics.csv`、
`summary_by_task.csv`、`summary_compare_baseline_ours.csv` 和 `config.json`。
逐样本指标同时包含真实视频预测 action vs GT action、生成视频预测 action vs GT
action、生成视频预测 action vs 真实视频预测 action 的 arm / hand / arm_hand MSE。
`summary_compare_baseline_ours.csv` 中的 `delta_*` 字段定义为 Ours 减 Baseline；
对 MSE / gap / ratio 来说，负值表示 Ours 更接近对应目标。

如需复现实验中已有 `full_eval/` 视频的前景 patch 与背景 patch 分布差异，可使用
归档的独立脚本：

```bash
python scripts/archive/eval_analysis/eval_background_patch_fid.py \
  final_mitty_r128_0507-h2r_1s-400d_r128_self_qkv_1000s_0507_203146 \
  --device cuda:0
```

脚本默认读取 `training_data/log/<log>/full_eval/{in_task_eval,ood_eval}` 和
同目录 `data_split/*.jsonl`，从 `data_split/config.json` 继承 SAM2 mask root
与 patch 参数，并把结果写到 `output/background_fid/<log>/summary.csv` 与
`summary.json`。输出字段包括 `foreground_patch_fid`、`background_patch_fid`、
`foreground_patch_count` 和 `background_patch_count`。背景 patch 定义为同一
固定网格下未被前景 Patch FID 规则选中的 patch，因此和前景 patch 集合互补；
可用 `--max-samples` 做 GPU smoke，或用 `--patch-max-per-frame` 控制背景
patch 数量。