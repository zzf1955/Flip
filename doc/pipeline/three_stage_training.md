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

