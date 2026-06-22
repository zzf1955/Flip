# 三阶段训练

当前 Wan2.2 + Mitty LoRA 训练主入口是：

```bash
scripts/flip_run.sh train --cuda 2,3 --nproc 2 -- <train args>
```

底层执行：

```bash
python -m torch.distributed.run --standalone --nproc_per_node=<N> -m src.pipeline.train
```

## 阶段

1. `identity_r2r`：清晰 robot -> 清晰 robot，学习基本重建。
2. `blur_r2r`：模糊 robot -> 清晰 robot，学习机器人外观恢复。
3. `h2r` / `r2h`：跨域编辑训练。

## 机器人外观域

G1 WBT 训练必须按机器人外观域分开执行。当前 WBT 数据包含三条训练外观域：

- `brainco`：Brainco 手外观域。
- `inspire_app1`：Inspire 手外观 1，任务分组来自
  `data/unitree_G1_WBT/Appearance.md` 中标记为 `1` 的条目。
- `inspire_app2`：Inspire 手外观 2，任务分组来自
  `data/unitree_G1_WBT/Appearance.md` 中标记为 `2` 的条目。

这三种外观不能混到同一个三阶段训练链路里。`identity_r2r`、`blur_r2r`、`h2r` /
`r2h` 的 pair、cache、runtime split、LoRA checkpoint 和 eval 输出都应按外观域分开。
推荐做法是并行跑三套训练：

```text
Brainco raw data
  -> brainco identity_r2r / blur_r2r / h2r-or-r2h pair
  -> brainco VAE/T5 cache
  -> brainco 三阶段 LoRA

Inspire appearance 1 raw data
  -> inspire_app1 identity_r2r / blur_r2r / h2r-or-r2h pair
  -> inspire_app1 VAE/T5 cache
  -> inspire_app1 三阶段 LoRA

Inspire appearance 2 raw data
  -> inspire_app2 identity_r2r / blur_r2r / h2r-or-r2h pair
  -> inspire_app2 VAE/T5 cache
  -> inspire_app2 三阶段 LoRA
```

三条线可以共用 base model、代码入口、训练超参和 GPU 调度策略，但不能共用同一个
appearance-learning LoRA。若需要比较不同外观，应分别训练后在 eval 汇总层面对比；不要
通过 `--task all` 或手工拼接 pair/cache 把 Brainco、Inspire 外观 1、Inspire 外观 2
目标混入同一个 run。

## 通用 wrapper

顶层仍保留两个通用训练 wrapper：

- `scripts/train_lora_grid.py`
- `scripts/train_three_stage_single_lora.py`

旧 H2R / final / SAM3 stage shell 启动器已归档到 `scripts/archive/training_launchers/`。

## 配置来源

- 训练 preset：`src/pipeline/train_config.py`
- runtime split：`src/pipeline/runtime_data.py`
- Wan/Mitty 实现：`src/pipeline/train.py`、`src/pipeline/train_mitty.py`、
  `src/pipeline/mitty_model_fn.py`

详细训练命令、eval、指标和历史 run 见 [训练基础设施完整记录](training_infra.md)。
