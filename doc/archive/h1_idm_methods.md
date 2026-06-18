# Humanoid H1 IDM 方法与实验统一说明

本文只整理 Humanoid Everyday H1 数据上的 inverse dynamics model, 不覆盖 WBT、
H2R、visible mask、Masquerade 等其他数据线。

当前 H1 IDM 的核心问题是：

```text
给定第一人称相邻两帧 (frame_t, frame_{t+1})，预测 H1 机器人 action_t。
```

在 `src.pipeline.archive.idm.humanoid_pair_idm` 中，这个监督口径也写作
`mean(action[t:t+frame_delta])`。当 `frame_delta=1` 时，它等价于 `action_t`。
AdaWorld latent decoder 路线同样使用相邻两帧和 `action_t`，因此可以在完整 H1
episode split 上做严格对比。

## 当前结论

从 task057 到 task064，H1 上实际在优化的是两条可公平比较的 IDM 路线：

- **AdaWorld latent decoder**：先用 AdaWorld LAM 把相邻两帧编码成 `z_t[32]`，再训练
  H1 action decoder。
- **RGB motion Transformer**：直接从相邻两帧 RGB 预测 `action_t[26]`。

两条路线当前都使用同一个完整 Humanoid Everyday H1 数据根、同一个 episode-level split、
同一个 `frame_delta=1 / action_t` 监督口径。完整 held-out 集固定为 `71486` samples /
`200` episodes。

当前最强结果：

| 路线 | 当前最佳 task | checkpoint | norm MSE | relative L2 | pred norm var | action_mse |
|------|------|------|------:|------:|------:|------:|
| RGB motion Transformer | task064 `motion_transformer_v2` | `tmp/humanoid_pair_idm_t064_v2_p16_s8000/best_checkpoint.pt` | `0.196960` | `0.203904` | `0.853166` | `0.028906` |
| AdaWorld latent decoder | task063 wider residual MLP | `tmp/adaworld_action_decoder_t063_full_c09_h384_lr8e4/best_checkpoint.pt` | `0.320246` | `0.268812` | `0.765571` | `0.050238` |

对应完整 held-out metrics：

```text
tmp/humanoid_pair_idm_t064_v2_p16_s8000_validate/val_metrics.json
tmp/adaworld_action_decoder_t063_full_c09_validate_best/val_metrics.json
```

结论很直接：**t064 RGB `motion_transformer_v2` 是当前最好的 H1 IDM**。它相对当前最强
AdaWorld decoder，held-out `action_mse` 从 `0.050238` 降到 `0.028906`，约低 `42.5%`；
`action_norm_mse` 从 `0.320246` 降到 `0.196960`，约低 `38.5%`；预测方差也更接近
target 方差。

AdaWorld 路线的价值仍然明确：它只训练低维 latent decoder，训练成本低，且 t063 已经略优于
t060 的上一代 RGB motion Transformer。但 t064 的 raw RGB motion stem、patch16、residual
readout、variance loss、AMP 和更长训练把 RGB 路线重新拉开了差距。

## task057 到 task064 脉络

| task | 范围 | 结果定位 |
|------|------|------|
| task057 | AdaWorld LAM latent 全量提取 + 基础 MLP decoder | H1 AdaWorld baseline，held-out `action_mse=0.078534` |
| task058 / task059 | Masquerade direct-render baseline | 视觉 baseline，不参与 H1 IDM 排名 |
| task060 | H1 RGB `motion_transformer` 架构与训练超参优化 | 上一代强 RGB baseline，held-out `action_mse=0.051443` |
| task061 | AdaWorld decoder 基础架构/超参优化 | residual MLP 把 AdaWorld held-out 降到 `0.054646` |
| task062 | G1 Pick-Up-Cloth 独立 IDM 对比 | G1 数据迁移实验，不混入 H1 held-out 排名 |
| task063 | AdaWorld decoder 二阶段容量/loss/head 消融 | 当前最强 AdaWorld decoder，held-out `action_mse=0.050238` |
| task064 | H1 RGB `motion_transformer_v2` 准确率优化 | 当前最强 H1 IDM，held-out `action_mse=0.028906` |

注意：本文后续的统一表只比较 H1 完整 held-out split。task062 的 G1 结果有独立数据、
独立 action schema 和独立 eval 集，不能与 H1 的 `71486` held-out 指标直接排序。

## 数据口径

统一数据根：

```text
data/humanoid-everyday-h1-chunks0-6-8-200
```

该目录是 Humanoid Everyday H1 LeRobot layout：

- `data/chunk-*/episode_*.parquet`：包含 `action`、`frame_index`、`next.done`。
- `videos/chunk-*/egocentric/episode_*.mp4`：对应第一人称 RGB 视频。

完整数据统计：

| item | value |
|------|------:|
| chunks | `chunk-000` 到 `chunk-006`，以及 `chunk-008` |
| episodes / parquet files | `1600` |
| total frames / parquet rows | `562022` |
| adjacent frame pairs | `560422` |
| action dim | `26` |

完整训练 / 验证 split 对齐 AdaWorld task057：

| split | episodes | samples |
|------|---------:|--------:|
| train | `1400` | `488936` |
| val held-out | `200` | `71486` |

split 配置：

```text
split_by=episode
train_ratio=0.875
seed=42
frame_stride=1
max_samples=0
```

这里的 `max_samples=0` 表示使用完整可发现样本。不要用 `train_samples=700` /
`eval_samples=100` 这类小口径判断最终效果；它只适合 smoke 或快速调试。

## 指标口径

所有可训练方法都用 train split 的 action mean / std 做 target normalization：

```text
normalized_target = (action - train_action_mean) / train_action_std
```

后续报告 H1 IDM 指标时，默认优先报告三项：

1. `action_norm_mse`：归一化空间 MSE。它是主要误差指标，避免不同 action 维度量纲差异主导结论。
2. `relative_l2_error`：原始 action 空间的整体相对 L2 error，即
   `||pred - target||_2 / ||target||_2`。
3. `pred_norm_var_mean`：归一化空间的预测方差，定义为先做
   `(pred - train_action_mean) / train_action_std`，再对 26 个 action 维度的预测方差取均值。

辅助指标可以继续报告：

- `action_mse`：原始 action 空间 MSE。
- `mean_baseline_action_mse`：均值 baseline 在同一 eval 集上的原始 MSE。
- `target_norm_var_mean`：归一化空间 target 方差，完整 H1 held-out split 上约为 `1.0037`。
- `pred_norm_var_ratio`：`pred_norm_var_mean / target_norm_var_mean`，用于判断输出方差是否偏低或偏高。
- `action_mean_dim_r2`：26 个维度的平均 R2。
- `action_mean_dim_corr`：26 个维度的平均 correlation。
- `action_pred_std_ratio_mean`：预测标准差 / 目标标准差的逐维均值。历史日志里常见，但后续表格优先使用
  `pred_norm_var_mean`。

## 方法 1：Mean Baseline

Mean baseline 不是神经网络，也不读取图像。它的预测是 train split 上 26 维 action 的均值：

```text
pred = train_action_mean
```

作用：

- 给出当前 split 的最低有效参考线。
- 如果一个 IDM 的 `action_mse` 不能低于 `mean_baseline_action_mse`，说明它没有可靠学到动作。
- 如果 `action_mse` 略低于 baseline 但 `action_pred_std_ratio_mean` 很低，说明模型可能仍接近均值回归。

在完整 H1 held-out split 上：

```text
mean_baseline_action_mse = 0.15635335445404053
action_norm_mse = 1.0070025849587083
relative_l2_error = 0.4742259937243753
pred_norm_var_mean = 0.0
target_norm_var_mean = 1.0037333875918515
```

## 方法 2：AdaWorld Latent Decoder

AdaWorld 路线分两步：

```text
(frame_t, frame_{t+1}) -> AdaWorld LAM z_t[32] -> H1 action_t[26]
```

### Encoder

入口：

```text
src.pipeline.archive.idm.adaworld_action_encoder
```

它只使用 AdaWorld 的 LAM action encoder：

- 不加载 AdaWorld world model。
- 不训练 AdaWorld。
- 输入是 H1 相邻两帧 egocentric RGB。
- 图像预处理按 AdaWorld 口径：中心裁方、resize 到 `256x256`、归一化到 `[0,1]`。
- 输出 `32` 维 continuous latent action。

全量 latent artifact：

```text
tmp/adaworld_action_encoder_h1_full_t057/latent_actions.npz
```

artifact 内容包括：

- `latent_actions`: `[560422, 32]`
- `episode`
- `chunk`
- `rel_frame_t`
- `rel_frame_tp1`

### Decoder

入口：

```text
src.pipeline.archive.idm.adaworld_action_decoder
```

decoder 从 latent artifact 回查原始 H1 parquet 中的 `action_t`。task057 是基础 MLP
baseline；task061 在同一 latent artifact 和同一 split 上优化 decoder 架构与训练超参。

task057 基础 MLP：

```text
32 -> 128 -> 128 -> 26
```

task057 全量训练配置：

```text
steps=2000
batch_size=1024
optimizer=AdamW
lr=1e-3
weight_decay=1e-4
lr_scheduler=cosine
min_lr_ratio=0.05
split_by=episode
train_ratio=0.875
```

训练输出：

```text
tmp/adaworld_action_decoder_h1_full_t057
```

held-out validate 输出：

```text
tmp/adaworld_action_decoder_h1_full_t057_validate_best/val_metrics.json
```

完整 H1 held-out 结果：

| metric | value |
|------|------:|
| `n_samples` | `71486` |
| `action_mse` | `0.07853357493877411` |
| `mean_baseline_action_mse` | `0.15635326504707336` |
| `action_norm_mse` | `0.5034212470054626` |
| `relative_l2_error` | `0.3360931247763039` |
| `pred_norm_var_mean` | `0.5087748970905759` |
| `target_norm_var_mean` | `1.0037333875918515` |
| `action_mean_dim_r2` | `0.504274274294193` |
| `action_mean_dim_corr` | `0.7063991037698892` |
| `action_pred_std_ratio_mean` | `0.7120954027542701` |

相对 held-out mean baseline，AdaWorld latent decoder 的 `action_mse` 降低约 `49.8%`。

全量 eval 结果覆盖 train + val 的全部 `560422` samples，主要用于检查整体数据分布：

| metric | value |
|------|------:|
| `n_samples` | `560422` |
| `action_mse` | `0.07298979163169861` |
| `mean_baseline_action_mse` | `0.15620023012161255` |
| `action_norm_mse` | `0.46723851561546326` |
| `action_mean_dim_r2` | `0.5340813008638529` |
| `action_mean_dim_corr` | `0.7275451834385211` |
| `action_pred_std_ratio_mean` | `0.7190383053742923` |

### task061 优化版 Decoder

task061 仍然只优化 AdaWorld latent action decoder，不重新训练 AdaWorld encoder，也不加载
AdaWorld world model。推荐配置：

```text
model_arch=residual_mlp
hidden_dim=256
depth=4
dropout=0.02
steps=3000
batch_size=1024
optimizer=AdamW
lr=5e-4
weight_decay=1e-4
betas=(0.9, 0.95)
lr_scheduler=cosine
min_lr_ratio=0.02
lr_warmup_ratio=0.05
```

输出：

```text
tmp/adaworld_action_decoder_h1_full_t061
tmp/adaworld_action_decoder_h1_full_t061_validate_best/val_metrics.json
```

完整 H1 held-out 结果：

| metric | value |
|------|------:|
| `n_samples` | `71486` |
| `action_mse` | `0.054645732045173645` |
| `mean_baseline_action_mse` | `0.15635326504707336` |
| `action_norm_mse` | `0.3499008119106293` |
| `relative_l2_error` | `0.2803561375375912` |
| `pred_norm_var_mean` | `0.7060706437165513` |
| `target_norm_var_mean` | `1.0037333875918515` |
| `action_mean_dim_r2` | `0.656568128329057` |
| `action_mean_dim_corr` | `0.8090876913987674` |
| `action_pred_std_ratio_mean` | `0.839937531031095` |

task061 的 full eval 覆盖 train + val 全部 `560422` samples，`action_mse=0.04235832020640373`；
它包含训练集，不能和 held-out validate 指标直接比较。

### task063 二阶段优化版 Decoder

task063 继续固定 AdaWorld latent artifact、H1 数据根和 episode-level split，只探索 decoder
容量、学习率、loss 和 output head。最佳配置：

```text
model_arch=residual_mlp
hidden_dim=384
depth=4
dropout=0.02
head_arch=shared
loss_type=mse
steps=3000
batch_size=1024
optimizer=AdamW
lr=8e-4
weight_decay=1e-4
betas=(0.9, 0.95)
lr_scheduler=cosine
min_lr_ratio=0.02
lr_warmup_ratio=0.05
```

输出：

```text
tmp/adaworld_action_decoder_t063_full_c09_h384_lr8e4
tmp/adaworld_action_decoder_t063_full_c09_validate_best/val_metrics.json
tmp/adaworld_action_decoder_t063_full_c09_eval_best/metrics.json
```

完整 H1 held-out 结果：

| metric | value |
|------|------:|
| `n_samples` | `71486` |
| `action_mse` | `0.05023810639977455` |
| `mean_baseline_action_mse` | `0.15635326504707336` |
| `action_norm_mse` | `0.3202458918094635` |
| `relative_l2_error` | `0.26881195165744` |
| `pred_norm_var_mean` | `0.7655707894142848` |
| `target_norm_var_mean` | `1.0037333875918515` |
| `action_mean_dim_r2` | `0.6858815573729001` |
| `action_mean_dim_corr` | `0.8282286180899694` |
| `action_pred_std_ratio_mean` | `0.8748558117793157` |

task063 的 full eval 覆盖 train + val 全部 `560422` samples，`action_mse=0.029545826837420464`，
`action_mean_dim_r2=0.8002844131909884`，`action_mean_dim_corr=0.893933926637356`，
`action_pred_std_ratio_mean=0.9002112241891714`。它包含训练集，不能和 held-out validate
指标直接比较。

## 方法 3：RGB Motion Transformer

RGB Transformer 路线不经过 AdaWorld latent bottleneck，直接从两帧 RGB 预测 H1 action：

```text
(frame_t, frame_{t+1}) -> motion_transformer -> action_t[26]
```

入口：

```text
src.pipeline.archive.idm.humanoid_pair_idm
```

当前推荐模型：

```text
--model-arch motion_transformer
```

task064 开始新增优化实验模型：

```text
--model-arch motion_transformer_v2
```

`motion_transformer_v2` 仍沿用两帧 patch motion token 的主干，但额外加入 RGB
差分 motion stem 和 residual readout head，目标是减少当前 `pred_norm_var_mean`
低于 target 的均值回归倾向。正式可部署 checkpoint 仍以完整 held-out validate 指标为准；
在 task064 长训完成前，task060 checkpoint 仍是已完成全量验证的基线。

### 输入与预处理

- 读取同一个 H1 LeRobot 数据根。
- 输入帧为 `frame_t` 和 `frame_{t+1}`。
- resize 到 `256x256`。
- 两帧 RGB 拼成 6 通道输入。
- 像素归一化到 `[-1,1]`。

### 架构

`motion_transformer` 的核心是显式建模两帧 patch 差异：

```text
x_t      = patch_embed(frame_t)
x_tp1    = patch_embed(frame_{t+1})
motion   = MLP([x_t, x_tp1, x_tp1 - x_t, |x_tp1 - x_t|])
```

task064 的 `motion_transformer_v2` 进一步把原始 RGB 空间的运动差异注入 motion token：

```text
raw_motion = ConvPatch([frame_tp1 - frame_t, |frame_tp1 - frame_t|])
motion_v2  = motion + LayerNorm(raw_motion)
```

token 组成：

- `cls_token`
- `motion_cls_token`
- `frame_t` patch tokens
- `frame_{t+1}` patch tokens
- motion patch tokens
- 2D sin/cos spatial position embedding
- frame embedding / motion embedding

readout：

```text
concat(cls, motion_cls, frame0_mean, frame1_mean, motion_mean) -> MLP head -> action[26]
```

task064 新增 readout head 选项：

- `legacy_mlp`：保持 task060 checkpoint 的原始 head 结构，保证旧权重可严格 replay。
- `mlp`：可配置深度的普通 MLP head。
- `residual_mlp`：pre-norm residual MLP blocks，`motion_transformer_v2` 默认使用。
- `gated_mlp`：residual block 内使用 SiLU value 和 sigmoid gate 的 gated 变体。

训练侧新增：

- `--variance-loss-weight` / `--variance-loss-warmup-ratio`：在 normalized action 空间匹配
  batch-level 预测方差，抑制输出方差偏低。
- `--grad-accum-steps`：支持 patch16 或更大 batch 的有效 batch size。
- `--amp`：CUDA bfloat16 autocast，用于降低 patch16 / 长训显存压力。
- `--init-checkpoint`：从已有 checkpoint 严格初始化 train，要求模型配置和当前 train
  split action mean / std 完全一致；用于显式二阶段 fine-tune，不改变默认训练行为。

### 全量训练配置

输出目录：

```text
tmp/humanoid_pair_idm_t060_full_adaworld_protocol
```

命令：

```bash
CUDA_VISIBLE_DEVICES=2 python -m src.pipeline.archive.idm.humanoid_pair_idm train \
  --device cuda:0 \
  --data-root /disk_n/zzf/flip/data/humanoid-everyday-h1-chunks0-6-8-200 \
  --output-dir tmp/humanoid_pair_idm_t060_full_adaworld_protocol \
  --max-samples 0 \
  --frame-stride 1 \
  --split-by episode \
  --train-ratio 0.875 \
  --steps 2000 \
  --batch-size 32 \
  --workers 8 \
  --eval-every 500 \
  --val-max-samples 4096 \
  --resize 256x256 \
  --model-arch motion_transformer \
  --transformer-embed-dim 192 \
  --transformer-depth 4 \
  --transformer-num-heads 6 \
  --hidden-dim 256
```

默认优化配置：

```text
optimizer=AdamW
lr=3e-4
weight_decay=1e-2
betas=(0.9, 0.95)
lr_scheduler=cosine
lr_warmup_ratio=0.05
min_lr_ratio=0.02
grad_clip_norm=1.0
```

训练中为了控制耗时，`eval_every` 只抽 `4096` 个 held-out samples。训练完成后需要单独运行完整 held-out validate。

完整 held-out validate 命令：

```bash
CUDA_VISIBLE_DEVICES=2 python -m src.pipeline.archive.idm.humanoid_pair_idm validate \
  --device cuda:0 \
  --checkpoint tmp/humanoid_pair_idm_t060_full_adaworld_protocol/best_checkpoint.pt \
  --output-dir tmp/humanoid_pair_idm_t060_full_adaworld_protocol_validate \
  --batch-size 32 \
  --workers 8
```

### task064 优化实验配置

第一优先级是把 task060 的 under-train 变量和新 v2 架构拆开比较。所有正式实验仍使用完整
H1 数据、episode split、`frame_delta=1`，训练中途可以用 `--val-max-samples 4096`
监控，但最终必须跑完整 `71486` held-out validate。

推荐的第一组长训：

```bash
CUDA_VISIBLE_DEVICES=2 python -m src.pipeline.archive.idm.humanoid_pair_idm train \
  --device cuda:0 \
  --data-root /disk_n/zzf/flip/data/humanoid-everyday-h1-chunks0-6-8-200 \
  --output-dir tmp/humanoid_pair_idm_t064_v2_p16_s8000 \
  --max-samples 0 \
  --frame-stride 1 \
  --frame-delta 1 \
  --split-by episode \
  --train-ratio 0.875 \
  --steps 8000 \
  --batch-size 16 \
  --workers 8 \
  --eval-every 1000 \
  --val-max-samples 4096 \
  --resize 256x256 \
  --model-arch motion_transformer_v2 \
  --transformer-patch-size 16 \
  --transformer-embed-dim 128 \
  --transformer-depth 4 \
  --transformer-num-heads 4 \
  --transformer-dropout 0.02 \
  --hidden-dim 256 \
  --transformer-head-depth 2 \
  --lr 3e-4 \
  --weight-decay 3e-3 \
  --lr-warmup-ratio 0.05 \
  --min-lr-ratio 0.02 \
  --variance-loss-weight 0.03 \
  --variance-loss-warmup-ratio 0.05 \
  --grad-accum-steps 2 \
  --amp
```

对照组建议：

- `T-base-8k`：task060 架构不变，`patch=32/embed=192/depth=4`，只把训练步数拉到
  `8000`，确认是否主要是 under-train。
- `T-wd-drop`：task060 架构，`weight_decay=3e-3`、`dropout=0.02`，看预测方差是否上升。
- `V2-p32`：`motion_transformer_v2`，但保持 `patch=32/embed=192/depth=4`，单独看 raw
  RGB diff stem + residual head 的收益。
- `V2-p16`：上面的推荐命令，测试更细 patch 对第一人称手臂 / 接触动作的收益。

完整验证：

```bash
CUDA_VISIBLE_DEVICES=2 python -m src.pipeline.archive.idm.humanoid_pair_idm validate \
  --device cuda:0 \
  --checkpoint tmp/humanoid_pair_idm_t064_v2_p16_s8000/best_checkpoint.pt \
  --output-dir tmp/humanoid_pair_idm_t064_v2_p16_s8000_validate \
  --batch-size 16 \
  --workers 8
```

判断标准仍以 task060 完整 held-out 指标为基线：

```text
action_norm_mse=0.320904
action_mse=0.051443
relative_l2_error=0.272016
pred_norm_var_mean=0.689259
```

task064 `motion_transformer_v2` patch16 `8000` step 已完成完整 held-out validate：

```text
tmp/humanoid_pair_idm_t064_v2_p16_s8000
tmp/humanoid_pair_idm_t064_v2_p16_s8000_validate/val_metrics.json
```

完整 `71486` held-out samples：

| metric | value |
|------|------:|
| `action_mse` | `0.028906064108014107` |
| `mean_baseline_action_mse` | `0.15635335445404053` |
| `action_norm_mse` | `0.1969597190618515` |
| `relative_l2_error` | `0.20390427137523634` |
| `pred_norm_var_mean` | `0.8531658785508718` |
| `target_norm_var_mean` | `1.0037351498588492` |
| `action_mean_dim_r2` | `0.8071291836408468` |
| `action_mean_dim_corr` | `0.8984220096698174` |
| `action_pred_std_ratio_mean` | `0.9228853445786697` |

这里的 `pred_norm_var_mean` / `target_norm_var_mean` 由
`val_predictions.csv` 和 checkpoint 中保存的 train action mean / std 复算得到；
`val_metrics.json` 直接保存的是 `action_pred_std_ratio_mean`。

相对 task060 RGB motion Transformer 完整 held-out，task064 v2 的 `action_mse`
从 `0.05144283175468445` 降到 `0.028906064108014107`，约降低 `43.8%`；
`action_norm_mse` 从 `0.32090431451797485` 降到 `0.1969597190618515`，约降低 `38.6%`；
预测方差也更接近 target，`pred_norm_var_mean` 从 `0.6892590912375324`
提高到 `0.8531658785508718`。

二阶段 fine-tune 暂未带来收益，当前推荐保持 `s8000` checkpoint：

| fine-tune | init | best subset action_mse | 结论 |
|------|------|------:|------|
| `tmp/humanoid_pair_idm_t064_v2_p16_ft_s3000_lr1e4` | `s8000/best_checkpoint.pt` | `0.029392` at step 0 | `lr=1e-4` 后 step500/1000 退化到 `0.032071` / `0.032695` |
| `tmp/humanoid_pair_idm_t064_v2_p16_ft_s1000_lr3e5` | `s8000/best_checkpoint.pt` | `0.029392` at step 0 | `lr=3e-5` 后 step250/500 为 `0.030179` / `0.029506` |

这两条 fine-tune 都在确认无刷新后中止，没有运行完整 held-out validate；完整验证仍以
`tmp/humanoid_pair_idm_t064_v2_p16_s8000_validate/val_metrics.json` 为准。

### 训练曲线

中途 eval 为 held-out `4096` sample 子集。这个历史 `eval_loss.csv` 没有保存每个中途点的
prediction，因此不能复算 `pred_norm_var_mean`；后续如果报告中途 checkpoint，也应同时保存
prediction 或在 evaluator 中直接写出 `pred_norm_var_mean`。

| step | norm MSE | relative L2 | action_mse | mean baseline |
|------|---------:|------------:|-----------:|--------------:|
| 0 | `1.015960` | `0.475358` | `0.157638` | `0.157396` |
| 500 | `0.604957` | `0.377881` | `0.099574` | `0.157396` |
| 1000 | `0.410739` | `0.313524` | `0.068527` | `0.157396` |
| 1500 | `0.357181` | `0.288629` | `0.058076` | `0.157396` |
| 2000 | `0.328745` | `0.274981` | `0.052878` | `0.157396` |

### 完整 Held-Out 结果

输出：

```text
tmp/humanoid_pair_idm_t060_full_adaworld_protocol_validate/val_metrics.json
tmp/humanoid_pair_idm_t060_full_adaworld_protocol_validate/val_predictions.csv
```

完整 `71486` held-out samples：

| metric | value |
|------|------:|
| `n_samples` | `71486` |
| `action_mse` | `0.05144283175468445` |
| `mean_baseline_action_mse` | `0.15635335445404053` |
| `action_norm_mse` | `0.32090431451797485` |
| `relative_l2_error` | `0.27201589200626863` |
| `pred_norm_var_mean` | `0.6892590912375324` |
| `target_norm_var_mean` | `1.0037351498588492` |
| `action_mean_dim_r2` | `0.6850515902042389` |
| `action_mean_dim_corr` | `0.826011096055691` |
| `action_pred_std_ratio_mean` | `0.8289548089871039` |

相对 held-out mean baseline，RGB motion Transformer 的 `action_mse` 降低约 `67.1%`。

相对同 split task057 AdaWorld baseline decoder held-out `action_mse=0.07853357493877411`，
RGB motion Transformer 的 `action_mse` 低约 `34.5%`。相对 task061 optimized AdaWorld
decoder held-out `action_mse=0.054645732045173645`，RGB motion Transformer 低约 `5.9%`。
task063 再把 AdaWorld latent decoder held-out `action_mse` 降到 `0.05023810639977455`，
略低于 task060 RGB motion Transformer 的 `0.051443`。

## 统一结果表

以下表格只比较完整 H1 held-out split：`71486` samples / `200` episodes。

| method | input | trainable part | norm MSE | relative L2 | pred norm var | action_mse |
|------|------|------|------:|------:|------:|------:|
| Mean baseline | none | none | `1.007003` | `0.474226` | `0.000000` | `0.156353` |
| AdaWorld task057 baseline decoder | AdaWorld LAM `z_t[32]` | MLP decoder | `0.503421` | `0.336093` | `0.508775` | `0.078534` |
| AdaWorld task061 optimized decoder | AdaWorld LAM `z_t[32]` | residual MLP decoder | `0.349901` | `0.280356` | `0.706071` | `0.054646` |
| AdaWorld task063 optimized decoder | AdaWorld LAM `z_t[32]` | wider residual MLP decoder | `0.320246` | `0.268812` | `0.765571` | `0.050238` |
| RGB motion Transformer task060 | two RGB frames | full RGB IDM | `0.320904` | `0.272016` | `0.689259` | `0.051443` |
| RGB motion Transformer v2 task064 | two RGB frames | full RGB IDM | `0.196960` | `0.203904` | `0.853166` | `0.028906` |

结论：

- Mean baseline 是必要参考线，不是可部署模型。
- AdaWorld latent decoder 明显优于 mean baseline，说明 AdaWorld LAM latent 包含 H1 action 相关信息。
- task061/task063 优化版 AdaWorld decoder 与 task060/task064 RGB motion Transformer 的数据 split / target /
  held-out eval 集一致，可以公平比较 held-out 指标。
- task063 的 wider residual MLP 是当前最强 AdaWorld latent-action decoder；它在完整 H1
  同 split 上略优于 task060 RGB motion Transformer：`action_mse` 从 `0.051443` 降到
  `0.050238`，`norm MSE` 两者几乎持平。
- task064 `motion_transformer_v2` 是当前最强 H1 IDM：相对 task060，`action_mse` 约降低
  `43.8%`，`norm MSE` 约降低 `38.6%`，同时预测方差更接近 held-out target。
- 小样本 `700/100` 或 `800` 级别实验会严重受 episode 覆盖影响，不能代表完整 H1 效果。

## 当前推荐

如果目标是在 H1 上训练一个用于 action consistency / IDM 评估的模型，当前推荐顺序：

1. 首选 task064 `src.pipeline.archive.idm.humanoid_pair_idm --model-arch motion_transformer_v2`，
   使用完整 H1 数据和 AdaWorld task057/task061 同口径 episode split。
2. task060 `motion_transformer` 现在作为上一代强基线保留。
3. 保留 task063 AdaWorld optimized decoder 作为最强 latent-action baseline；它复用
   AdaWorld LAM latent，训练成本低于 RGB motion Transformer。
4. task057/task061 作为历史 AdaWorld decoder baseline；task057 只代表基础 MLP。
5. Mean baseline 只作为 sanity check，不应作为最终 IDM。

推荐的正式 H1 RGB IDM checkpoint：

```text
tmp/humanoid_pair_idm_t064_v2_p16_s8000/best_checkpoint.pt
```

推荐的完整 held-out 指标：

```text
tmp/humanoid_pair_idm_t064_v2_p16_s8000_validate/val_metrics.json
```

## 注意事项

- 本文只讨论 H1 数据，不把其他 task / WBT / H2R 的结果混进来。
- RGB motion Transformer 当前使用 `frame_delta=1`。如果改成 interval mean action，需要重新与
  AdaWorld decoder 区分，因为 AdaWorld 当前是相邻帧 `action_t` 口径。
- Transformer 训练中的 `--val-max-samples 4096` 只是中途监控；正式结果必须看完整 held-out
  validate。
- 旧的 `small_cnn` 和 legacy `transformer` 结果使用过不同样本量、split 或 target 语义，只能作历史参考。
