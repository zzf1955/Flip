# 训练基础设施

本页记录当前维护的训练入口、cache 结构、路径约定和验证方式。历史 FunControl、RectFlow/Dxxx Flow、直接替换噪声等实验只保留在 `doc/tasks/done/` 的历史记录中，不再作为新实验入口维护。

## 维护边界

### 保留入口

| 入口 | 用途 | 状态 |
|------|------|------|
| `python -m src.pipeline.mitty_cache` | 生成 Wan2.2 / Mitty 训练 cache | 维护 |
| `python -m src.pipeline.train` | Mitty LoRA 正式训练入口 | 维护 |
| `python -m src.pipeline.evaluate_mitty_models` | 离线生成评估视频并计算 PSNR/SSIM/LPIPS/FID/FVD | 维护 |
| `src.pipeline.train_mitty` | 旧 Mitty 实现模块 | 仅供 `train.py` 复用 helper；不要直接启动 |

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

Seedance direct 的 1s 训练数据由 `src.pipeline.seedance_clip` 从
`training_data/seedance_direct/4s/` 重新后处理得到：每个 4s 源视频按
1s 窗口、0.5s 步长生成 7 个普通切片，再生成 7 个水平翻转切片，编号
`clip00`–`clip06` 为普通样本，`clip07`–`clip13` 为翻转样本。脚本会写出
`training_data/seedance_direct/1s/<task>/manifest.jsonl`，`make_pair.py` 依赖
该 manifest 对齐真实 `clip_start` 与增强类型；重建 1s 切片、pair 和 cache
时不要改动 `seedance_direct/4s/` 原始 API 输出。

当前 Seedance direct 1s 划分方案：`Inspire_Pickup_Pillow_MainCamOnly` 作为
OOD，取 32 条进入 `ood_eval`；其余两个训练任务各取 8 条进入 `eval`。
训练任务集合固定为：`Inspire_Collect_Clothes_MainCamOnly`、
`Inspire_Pickup_Pillow_MainCamOnly`、`Inspire_Put_Clothes_into_Washing_Machine`。

弃用数据集：

- `Inspire_Put_Clothes_Into_Basket`：合成手部外观与其他任务不一致，不进入训练分布。
- `Inspire_Put_Clothes_into_Washing_Machine_MainCamOnly`：只有一条有效数据且是重复数据，没有必要纳入训练。

使用 `--max-ood-per-task 32 --per-task-eval-clips 8` 控制该划分。

`src.pipeline.make_pair --task all` 与 `src.pipeline.make_robot_pair --task all`
默认展开为上述三任务训练集合；如需调试历史/非训练任务，显式传入任务短名或
`--task inspire`。

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
    └── robot_1s/
```

VAE cache 样本字段：

- `human_latent`: human/control 视频 latent。
- `robot_latent`: robot/target 视频 latent。
- `prompt`: prompt 文本，用于匹配共享 T5 cache。
- `source_id`: 数据溯源 key。

T5 embedding 不再重复嵌入每个样本文件。正式训练入口不再从命令行接收
cache/T5 路径，而是通过 `src/pipeline/train_config.py` 的 `--task-name`
固定映射选择。

## 生成 Cache

### Mitty 直接训练数据

```bash
# --task all 等价于当前维护的三个 TRAINING_TASKS，不包含 Basket/MainCamOnly 重复集
python -m src.pipeline.make_pair \
  --task all \
  --second 1s \
  --human-source seedance_direct \
  --ood-tasks Inspire_Pickup_Pillow_MainCamOnly \
  --max-ood-per-task 32 \
  --per-task-eval-clips 8 \
  --clean

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

如果只是调试历史任务或检查被弃用任务，需要显式指定任务，避免误混入正式训练数据：

```bash
python -m src.pipeline.make_pair \
  --task Inspire_Put_Clothes_Into_Basket \
  --second 1s \
  --human-source seedance_direct \
  --clean
```

### 三阶段 LoRA identity 数据

```bash
# --task all 同样只展开为 TRAINING_TASKS，供 identity 阶段使用
python -m src.pipeline.make_robot_pair \
  --task all \
  --max-segments 500 \
  --per-task-eval 5 \
  --clean

CUDA_VISIBLE_DEVICES=2 python -m src.pipeline.mitty_cache \
  --pair-dir training_data/robot_pair/1s/train \
  --output training_data/cache/vae/robot_1s/train \
  --t5-cache-dir training_data/cache/t5 \
  --device cuda:0 \
  --batch-size 4
```

## 训练命令

## 离线综合评估

`src.pipeline.evaluate_mitty_models` 用于比较训练完成的 Mitty LoRA run：先从
`training_data/cache/vae/pair_1s/{eval,ood_eval}` 读取 1s pair cache 生成视频，
再使用 `training_data/pair/1s/{split}/video` 中的原始 robot MP4 作为 GT，
计算 PSNR、SSIM、LPIPS、FID 和 FVD。默认评估：

- `Mitty-transfer-124d_r128_2000s_0425_1456/ckpt/step-2000.safetensors`
- `Mitty-transfer2LoRA-124d_r128_2000s_0425_1425/ckpt/step-2000.safetensors`
- `eval` 32 条 + `ood_eval` 32 条，即用户口径的 `32+32`

推荐通过统一入口运行 GPU 评估：

```bash
scripts/flip_run.sh eval_mitty --cuda 2 -- \
  --device cuda:0 \
  --samples-per-split 32
```

输出目录默认是 `training_data/eval/mitty_pair_1s/`：每个 run/checkpoint/split
下保存 `gen_*.mp4`、`gt_*.mp4`、`ctrl_*.mp4`，并在根目录写出
`summary.csv` 与 `summary.json`。如只想复算已有视频的指标，可加
`--no-generate`；如评估全集，可设 `--samples-per-split -1`。

### 轻量 smoke

```bash
CUDA_VISIBLE_DEVICES=2 python -m src.pipeline.train \
  --task-name smoke_t032_e2e \
  --max-steps 1 \
  --save-steps 1 \
  --eval-steps 1 \
  --eval-video-steps 0 \
  --wandb-project ""
```

### 正式 Mitty 训练

`--task-name` 从 `src/pipeline/train_config.py` 选择固定数据配置。当前维护
的 task 包括：

| task name | train/eval/ood cache |
| --- | --- |
| `pair_1s` | `training_data/cache/vae/pair_1s/` |
| `pair_1s_r2h` | `training_data/cache/vae/pair_1s_r2h/` |
| `pair_1s_train3` | `training_data/cache/vae/pair_1s_train3/` |
| `pair_1s_16` | `training_data/cache/vae/pair_1s_16/` |
| `robot_1s` | `training_data/cache/vae/robot_1s/` |
| `attn_ffn_selected` | `output/mitty_cache_1s/` |

新增训练数据集时只更新 `train_config.py`，不要在正式命令中重新暴露
`--cache-train` / `--cache-eval` / `--t5-cache-dir`。

单卡：

```bash
scripts/flip_run.sh train --cuda 2 -- \
  --task-name pair_1s \
  --max-steps 1000 \
  --save-steps 100 \
  --eval-steps 100 \
  --eval-video-steps 100
```

多卡：

```bash
scripts/flip_run.sh train --cuda 2,3 --nproc 2 -- \
  --task-name pair_1s \
  --max-steps 1000 \
  --save-steps 100 \
  --eval-steps 100 \
  --eval-video-steps 100
```

`train` 的 DDP 评估规则：

- `eval loss` 按 cache 文件索引在所有 rank 间切分，每个 rank 计算自己的子集，再 `all_reduce` 成全局均值；随机种子使用全局样本索引，避免 GPU 数量变化改变评估语义。
- `eval video` 按待生成视频的全局样本索引在所有 rank 间切分；所有 rank 写入同一个 `step-XXXX/`，文件名仍为 `gen_00.mp4`、`gt_00.mp4`、`ctrl_00.mp4` 这类全局编号。
- CSV、W&B、eval video 上传和在线指标只在 rank 0 执行；视频生成完成后会用 DDP barrier 等待所有 rank 写完。
- 正式实验默认 `--max-steps 1000 --save-steps 100 --eval-steps 100 --eval-video-steps 100`；smoke/debug 可临时调小。
- `--loss`、`--patch-dir` 已从正式训练入口移除，当前统一使用标准 Mitty loss。

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
