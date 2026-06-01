# Humanoid H1 IDM 方法与实验统一说明

本文只整理 Humanoid Everyday H1 数据上的 inverse dynamics model, 不覆盖 WBT、
H2R、visible mask、Masquerade 等其他数据线。

当前 H1 IDM 的核心问题是：

```text
给定第一人称相邻两帧 (frame_t, frame_{t+1})，预测 H1 机器人 action_t。
```

在 `src.pipeline.humanoid_pair_idm` 中，这个监督口径也写作
`mean(action[t:t+frame_delta])`。当 `frame_delta=1` 时，它等价于 `action_t`。
AdaWorld latent decoder 路线同样使用相邻两帧和 `action_t`，因此可以在完整 H1
episode split 上做严格对比。

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
src.pipeline.adaworld_action_encoder
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
src.pipeline.adaworld_action_decoder
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
| `pred_norm_var_mean` | `0.765373` |
| `target_norm_var_mean` | `1.003733` |
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
src.pipeline.humanoid_pair_idm
```

当前推荐模型：

```text
--model-arch motion_transformer
```

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

### 全量训练配置

输出目录：

```text
tmp/humanoid_pair_idm_t060_full_adaworld_protocol
```

命令：

```bash
scripts/flip_run.sh humanoid_pair_idm --cuda 2 -- train \
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
scripts/flip_run.sh humanoid_pair_idm --cuda 2 -- validate \
  --device cuda:0 \
  --checkpoint tmp/humanoid_pair_idm_t060_full_adaworld_protocol/best_checkpoint.pt \
  --output-dir tmp/humanoid_pair_idm_t060_full_adaworld_protocol_validate \
  --batch-size 32 \
  --workers 8
```

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
| AdaWorld task063 optimized decoder | AdaWorld LAM `z_t[32]` | wider residual MLP decoder | `0.320246` | `0.268812` | `0.765373` | `0.050238` |
| RGB motion Transformer task060 | two RGB frames | full RGB IDM | `0.320904` | `0.272016` | `0.689259` | `0.051443` |

结论：

- Mean baseline 是必要参考线，不是可部署模型。
- AdaWorld latent decoder 明显优于 mean baseline，说明 AdaWorld LAM latent 包含 H1 action 相关信息。
- task061/task063 优化版 AdaWorld decoder 与 task060 RGB motion Transformer 的数据 split / target /
  held-out eval 集一致，可以公平比较 held-out 指标。
- task063 的 wider residual MLP 在完整 H1 同 split 上略优于 task060 RGB motion Transformer：
  `action_mse` 从 RGB Transformer 的 `0.051443` 降到 `0.050238`，`norm MSE` 两者几乎持平。
- 小样本 `700/100` 或 `800` 级别实验会严重受 episode 覆盖影响，不能代表完整 H1 效果。

## 当前推荐

如果目标是在 H1 上训练一个用于 action consistency / IDM 评估的模型，当前推荐顺序：

1. 首选 task063 AdaWorld optimized decoder 作为当前最强 H1 held-out IDM baseline；它复用
   AdaWorld LAM latent，训练成本低于 RGB motion Transformer。
2. 保留 `src.pipeline.humanoid_pair_idm --model-arch motion_transformer` 作为不经过
   AdaWorld latent bottleneck 的 RGB 端对照；它仍适合验证 latent bottleneck 是否限制信息。
3. task057/task061 作为历史 AdaWorld decoder baseline；task057 只代表基础 MLP。
4. Mean baseline 只作为 sanity check，不应作为最终 IDM。

推荐的正式 H1 RGB IDM checkpoint：

```text
tmp/humanoid_pair_idm_t060_full_adaworld_protocol/best_checkpoint.pt
```

推荐的完整 held-out 指标：

```text
tmp/humanoid_pair_idm_t060_full_adaworld_protocol_validate/val_metrics.json
```

## 注意事项

- 本文只讨论 H1 数据，不把其他 task / WBT / H2R 的结果混进来。
- RGB motion Transformer 当前使用 `frame_delta=1`。如果改成 interval mean action，需要重新与
  AdaWorld decoder 区分，因为 AdaWorld 当前是相邻帧 `action_t` 口径。
- Transformer 训练中的 `--val-max-samples 4096` 只是中途监控；正式结果必须看完整 held-out
  validate。
- 旧的 `small_cnn` 和 legacy `transformer` 结果使用过不同样本量、split 或 target 语义，只能作历史参考。
