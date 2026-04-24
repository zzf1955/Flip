# 训练基础设施

本页记录当前维护的训练入口、cache 结构、路径约定和验证方式。历史 FunControl、RectFlow/Dxxx Flow、直接替换噪声等实验只保留在 `doc/tasks/done/` 的历史记录中，不再作为新实验入口维护。

## 维护边界

### 保留入口

| 入口 | 用途 | 状态 |
|------|------|------|
| `python -m src.pipeline.mitty_cache` | 生成 Wan2.2 / Mitty 训练 cache | 维护 |
| `python -m src.pipeline.train` | Mitty LoRA 正式训练入口 | 维护 |
| `src.pipeline.train_mitty` | Mitty 训练实现模块与兼容入口 | 维护中，后续逐步收敛到 `train.py` |

### 移除入口

| 旧入口 | 原用途 | 处理 |
|--------|--------|------|
| `src.pipeline.train_lora` | Wan2.1 FunControl legacy LoRA | 移除，不再跑新 baseline |
| `src.pipeline.train_rf` | Rectified Flow / Dxxx Flow 对照实验 | 移除 |
| `src.pipeline.rf_model_fn` | RectFlow forward/loss | 移除 |
| `src.pipeline.backbones.rectflow` | RectFlow backbone 注册 | 移除 |

## DiffSynth 边界

- 新重构不再把外部 DiffSynth 训练脚本作为主流程入口。
- 主线训练优先使用本仓库 `src/core/wan_loader.py`、`src/core/train_utils.py`、`src/pipeline/mitty_model_fn.py`。
- 当前少量底层类/模型定义仍来自已安装依赖包；这些属于短期底层兼容边界，不允许新增 pipeline 级业务依赖或反向 import legacy 入口。
- 新增代码不得依赖 `/disk_n/zzf/DiffSynth-Studio/examples/...` 这类外部脚本路径。

## GPU 分配

- Codex 验证只使用卡 2：`CUDA_VISIBLE_DEVICES=2 ...`。
- 卡 3 留给用户实验，Codex 不使用。
- 训练命令优先通过 `scripts/flip_run.sh` 包装；没有包装的轻量命令必须显式设置 `CUDA_VISIBLE_DEVICES=2`。

## 目录规范

| 路径 | 职责 | 是否可删除 |
|------|------|------------|
| `data/` | 原始数据、标定、机器人资产，只读共享 | 否 |
| `training_data/` | 可复现实验数据、pair、cache、训练日志 | 谨慎 |
| `output/` | pipeline 中间产物、人工检查结果 | 视实验而定 |
| `tmp/` | smoke、测试、一次性调试产物 | 是 |

`src/core/config.py` 提供统一常量：

```python
from src.core.config import (
    DATA_ROOT,
    TRAINING_DATA_ROOT,
    OUTPUT_DIR,
    TMP_DIR,
    CACHE_ROOT,
    T5_CACHE_DIR,
    VAE_CACHE_DIR,
)
```

所有测试命令默认写入 `./tmp/<task>/...`，不写入 `output/tmp` 或训练日志目录。

## Cache 管理

训练前需要预计算 embedding 缓存，分为 **T5 文本缓存** 和 **VAE 视频缓存**。

```text
training_data/cache/
├── t5/
│   ├── prompt_<hash>.pth
│   └── negative.pth
└── vae/
    ├── pair_1s/
    │   ├── train/pair_NNNN.pth
    │   ├── eval/pair_NNNN.pth
    │   └── ood_eval/pair_NNNN.pth
    ├── pair_1s_identity/
    └── robot_1s/
```

VAE cache 样本字段：

- `human_latent`: human/control 视频 latent。
- `robot_latent`: robot/target 视频 latent。
- `prompt`: prompt 文本，用于匹配共享 T5 cache。
- `source_id`: 数据溯源 key。

T5 embedding 不再重复嵌入每个样本文件。训练脚本通过 `--t5-cache-dir` 指定，默认 `training_data/cache/t5/`。

## 生成 Cache

```bash
CUDA_VISIBLE_DEVICES=2 python -m src.pipeline.mitty_cache \
  --pair-dir training_data/pair/1s/train \
  --output training_data/cache/vae/pair_1s/train \
  --t5-cache-dir training_data/cache/t5 \
  --device cuda:0 \
  --batch-size 4 \
  --prefetch-workers 8 \
  --prefetch-batches 2 \
  --save-workers 1
```

## 训练命令

### 轻量 smoke

```bash
CUDA_VISIBLE_DEVICES=2 python -m src.pipeline.train \
  --task-name smoke \
  --loss uniform \
  --cache-train training_data/cache/vae/pair_1s/train \
  --cache-eval training_data/cache/vae/pair_1s/eval \
  --t5-cache-dir training_data/cache/t5 \
  --output-dir tmp/t032/train_smoke \
  --max-steps 10 \
  --save-steps 10 \
  --eval-steps 10 \
  --eval-video-steps 0 \
  --wandb-project ""
```

### 正式 Mitty 训练

```bash
scripts/flip_run.sh train_mitty --cuda 2 -- \
  --task-name appearance \
  --loss uniform \
  --cache-train training_data/cache/vae/pair_1s/train \
  --cache-eval training_data/cache/vae/pair_1s/eval \
  --cache-ood training_data/cache/vae/pair_1s/ood_eval \
  --t5-cache-dir training_data/cache/t5 \
  --output-dir training_data/log \
  --max-steps 400 \
  --save-steps 50 \
  --eval-steps 50 \
  --eval-video-steps 50
```

### Hand-patch 加权

```bash
scripts/flip_run.sh train_mitty --cuda 2 -- \
  --task-name hand_patch \
  --loss hand_patch \
  --patch-dir training_data/pair/1s/train/hand_patch \
  --cache-train training_data/cache/vae/pair_1s/train \
  --cache-eval training_data/cache/vae/pair_1s/eval \
  --t5-cache-dir training_data/cache/t5 \
  --output-dir training_data/log \
  --max-steps 400
```

## 验证

```bash
/home/leadtek/miniconda3/envs/flip/bin/python scripts/smoke_t032_refactor.py
CUDA_VISIBLE_DEVICES=2 /home/leadtek/miniconda3/envs/flip/bin/python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

`smoke_t032_refactor.py` 覆盖：`compileall`、保留 pipeline/tool 入口的 `--help`、废弃训练模块不可 import。验证日志统一写入 `tmp/t032/smoke/`。

真实 GPU 端到端 smoke：

```bash
CUDA_VISIBLE_DEVICES=2 /home/leadtek/miniconda3/envs/flip/bin/python scripts/smoke_t032_gpu.py
```

该脚本复制 1 条 pair 到 `tmp/t032/gpu_smoke/`，执行 `mitty_cache` 生成 VAE cache，再跑 `train.py` 1 step + 1 sample eval。卡 3 留给用户实验。
