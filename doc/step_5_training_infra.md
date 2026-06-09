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

当前维护的数据 Task 固定为三个机器人 Task：

- `Inspire_Pickup_Pillow_MainCamOnly`
- `Inspire_Collect_Clothes_MainCamOnly`
- `Inspire_Put_Clothes_into_Washing_Machine`

数据本身不再预先切成 `train/eval/ood_eval`。磁盘只按物理属性组织；
训练时通过 CLI 指定 `--train-tasks` 与 `--ood-tasks`。每个 task 的
`training_data/pair/<data_type>/<duration>/<robot_task>/pair_order.jsonl`
保存该 task 的固定 pair 乱序表；第一次训练缺失该表时用 `--data-seed`
生成，之后训练只读取并校验，不会重新洗牌。默认 preset 使用 Collect +
Washing 作为 in-task，Pillow 作为 OOD。

四类数据类型：

- `identity_r2r`：清晰机器人 → 同一清晰机器人。
- `blur_r2r`：模糊机器人 → 清晰机器人。该数据不依赖 human/Seedance，
  直接枚举三个 canonical robot Task 的全部 `training_data/segment/` 数据；
  `video/` 是清晰 robot target，`control_video/` 由同一 robot clip 结合
  `training_data/sam2_mask/` 全身 mask 做局部 Gaussian blur 生成，语义与旧
  `1s_patch` 的 blur 数据一致，但输出为当前 task-organized 布局。
- `h2r`：人 → 机器人。
- `r2h`：机器人 → 人。

`src.pipeline.make_pair --task all` 与 `src.pipeline.make_robot_pair --task all`
默认只展开上述三个 canonical robot Task；如需调试历史/非训练任务，显式传入任务短名或
`--task inspire`。

训练前需要预计算 embedding 缓存，分为 **T5 文本缓存** 和 **VAE 视频缓存**。

```text
training_data/pair/
└── <data_type>/
    └── <duration>/
        ├── <robot_task>/
        │   ├── video/pair_NNNN.mp4
        │   ├── control_video/pair_NNNN.mp4
        │   ├── metadata.csv
        │   ├── manifest.jsonl
        │   └── pair_order.jsonl
        └── index.jsonl

training_data/cache/
├── t5/<data_type>/<duration>/
│   ├── prompt_<hash>.pth
│   └── negative.pth
└── vae/<data_type>/<duration>/<robot_task>/
    ├── pair_NNNN.pth
    └── manifest.jsonl
```

VAE cache 样本字段：

- `human_latent`: control/input 视频 latent。
- `robot_latent`: video/target 视频 latent。
- `prompt`: prompt 文本，用于匹配共享 T5 cache。
- `data_type`、`duration`、`robot_task`、`source_id`、`source_segment_id`: 运行时 split 和溯源字段。

T5 embedding 不再重复嵌入每个样本文件。T5 cache 目录与数据类型和 duration
匹配，例如 `h2r/1s` 使用 `training_data/cache/t5/h2r/1s/`。
正式训练入口通过 `src/pipeline/train_config.py` 的 `--task-name` 选择 preset，
也可用 CLI 覆盖 `--data-type`、`--duration`、`--train-tasks`、`--ood-tasks`
以及各类 size。

运行时 split 规则：

- pair 顺序来自各 task pair 目录下的 `pair_order.jsonl`。如果文件不存在，
  训练入口会按 `--data-seed` 从该 task 的 `manifest.jsonl` 生成一次；如果文件
  已存在，后续运行会复用该顺序并校验它与 manifest/cache 是否一致。
- `--train-size` 是全局训练样本数；训练入口先按各 in-task task 在扣除 eval
  尾部后的可用样本量做比例分配，再从每个 task 顺序表头部取对应数量。
- `--in-task-eval-size` 为正数时表示全局 in-task eval 样本数，按 task 数据量
  比例分配后从各 task 顺序表尾部取；`0/-1` 表示自动使用约 10% 的尾部样本，
  并至少保留一个样本可用于训练。
- `--ood-eval-size` 为正数时按 OOD task 数据量比例分配，并从各 OOD task
  顺序表尾部取；`0/-1` 表示使用全部 OOD 样本。
- eval video 子采样不再随 step 改变，同一个 run 的不同 eval step 使用同一批
  eval video 样本，便于肉眼和指标对比。
- 训练启动日志会打印每个 split 下各 task 的实际样本数，并在
  `data_split/config.json` 记录 `split_counts` 和 `pair_order_paths`。

## 生成 Cache

### Mitty 直接训练数据

```bash
python -m src.pipeline.make_pair \
  --task all \
  --second 1s \
  --data-type h2r \
  --human-source seedance_direct \
  --clean

CUDA_VISIBLE_DEVICES=2 python -m src.pipeline.mitty_cache \
  --pair-dir training_data/pair/h2r/1s/Inspire_Collect_Clothes_MainCamOnly \
  --output training_data/cache/vae/h2r/1s/Inspire_Collect_Clothes_MainCamOnly \
  --t5-cache-dir training_data/cache/t5/h2r/1s \
  --device cuda:0 \
  --batch-size 4 \
  --prefetch-workers 8 \
  --prefetch-batches 2 \
  --save-workers 1
```

需要为每个 robot task 分别运行 `mitty_cache`，输出到对应 task 子目录。
如果已有旧 split 目录，可先迁移为新布局：

```bash
python scripts/migrate_task_layout.py --data-type h2r --duration 1s --clean
python scripts/migrate_task_layout.py --data-type blur_r2r --duration 1s --clean
python scripts/migrate_task_layout.py --data-type identity_r2r --duration 1s --clean
```

### r2h 模型自合成 Human Pair

`src.pipeline.r2h_synthesize` 使用训练好的 r2h Mitty LoRA，从
`training_data/segment/<task>/<episode>/seg*_video.mp4` 直接枚举 robot source，
切成与 h2r 训练一致的 clip，然后生成 synthetic human，并落盘为新的 h2r `_syn`
task。已有 Seedance/original h2r manifest 只用于排除已覆盖 robot source，不作为
自合成的主要输入。

默认互斥规则：

- 默认排除 `ep000`、`ep001`、`ep002`、`ep003`，这些 episode 视作已由
  Seedance 覆盖。
- 如传入 `--seedance-covered-manifest`，脚本会读取其中的 `robot_source_key` 或从
  `episode/seg/clip_start/clip_dur` 等字段派生覆盖 key，并校验 syn 输出与覆盖集
  无交集。
- syn pair 的 `manifest.jsonl` 会记录 `robot_source_key`、`source_segment_path`、
  `source_clip_start`、`source_clip_dur`、`source_clip_index`、r2h run/checkpoint
  和生成参数，供后续 mixed h2r 训练做来源互斥校验。

先只检查可用 source：

```bash
python -m src.pipeline.r2h_synthesize \
  --source-task Inspire_Collect_Clothes_MainCamOnly \
  --duration 1s \
  --list-only
```

生成 `_syn` pair：

```bash
CUDA_VISIBLE_DEVICES=0 python -m src.pipeline.r2h_synthesize \
  --source-task Inspire_Collect_Clothes_MainCamOnly \
  --duration 1s \
  --run <r2h_run_name_or_path> \
  --checkpoint latest \
  --num-samples 200 \
  --resume-existing
```

如果要从多个 task 中生成到固定总上限，并按各 task 的可用 robot clip 数比例分配，
使用 `--source-task ... --allocate-by-task proportional`。例如把三个默认 task 的
`_syn` 数据集生成到总上限 4000 条：

```bash
CUDA_VISIBLE_DEVICES=0 python -m src.pipeline.r2h_synthesize \
  --source-task all \
  --duration 1s \
  --run <r2h_run_name_or_path> \
  --checkpoint latest \
  --num-samples 4000 \
  --allocate-by-task proportional \
  --resume-existing
```

`global_head` 是默认选择模式，会按全局稳定顺序取前 N 条；`proportional` 只支持
配合 `--num-samples` 使用，会按每个 task 过滤后的 eligible clip 数量做比例分配。
`--num-samples` 表示目标数据集总上限，不表示本次新增数量。配合
`--resume-existing` 时，入口会先读取目标 `_syn` task 目录下已有的
`manifest.jsonl`，保留已有 `pair_NNNN`、`metadata.csv` 和 `pair_order.jsonl`
记录，然后从已有数量之后继续生成；已有数量已经达到或超过本次目标上限时直接
保持原数据集不变。继续扩充数据时应把 `--num-samples` 设为期望的最终总量，例如
已有 800 条、希望再追加约 200 条时传 `--num-samples 1000`。
不要通过降低 `--num-samples` 来“重建” `_syn`，该入口的续跑语义是不覆盖旧数据。
续跑同一个 `_syn` 目录时必须沿用同一套 source task、exclude/covered manifest 和
allocation 设置；如果已有 manifest 前缀与当前选择规则不一致，入口会直接报错。

多卡扩充 `_syn` 数据时使用 `scripts/run_r2h_synthesize_queue.py`。该 launcher
先按一个全局 `--num-samples` 上限计算每个 source task 的目标数量，再生成单 task
`r2h_synthesize` 命令队列；运行时每张指定 CUDA 卡一次只跑一个 task，某张卡空闲后
取队列下一项继续跑。子进程仍通过 `scripts/flip_run.sh r2h_synthesize` 设置环境，
因此子进程内部设备固定传 `--device cuda:0`。默认只打印并写出队列；确认后加
`--execute` 启动。

```bash
/home/leadtek/miniconda3/envs/flip/bin/python scripts/run_r2h_synthesize_queue.py \
  --cuda 0,1,2 \
  --source-task in_task \
  --duration 1s \
  --run <r2h_run_name_or_path> \
  --checkpoint latest \
  --num-samples 1000
```

确认队列后执行：

```bash
/home/leadtek/miniconda3/envs/flip/bin/python scripts/run_r2h_synthesize_queue.py \
  --cuda 0,1,2 \
  --source-task in_task \
  --duration 1s \
  --run <r2h_run_name_or_path> \
  --checkpoint latest \
  --num-samples 1000 \
  --execute
```

队列文件、每个 task 的执行日志和配置会写到
`training_data/log/r2h_synthesize_queue/<timestamp>/`。如果只传两个 source task，
即使提供三张卡也只会启动两个并发任务；多于 CUDA 数量的 task 会自动排队。

### ep000-ep003 syn 误差分析生成

如果只想在 Seedance 覆盖的 episode 上检查自合成效果，不要写入正式
`training_data/pair`，使用独立脚本 `scripts/run_syn_error_analysis.py`。默认
source task 为两个 in-task 加一个 OOD，默认 episode 为 `ep000`、`ep001`、
`ep002`、`ep003`，默认从每个 4s segment 切成 1s 非重叠 robot clip（start 为
0s、1s、2s、3s），再用指定 r2h checkpoint 生成 syn human。输出固定在
`output/syn_error_analysis/` 下，包含 `robot/`、`syn/`，可选 `compare/`，以及
根目录 `manifest.jsonl` 和 `metadata.csv`。

先检查将要处理的 clip：

```bash
python scripts/run_syn_error_analysis.py --list-only
```

指定 CUDA 跑完整生成：

```bash
scripts/flip_run.sh syn_error_analysis --cuda 0 -- \
  --run final_data_r2h_all_in_task-r2h_1s-529d_r96_self_qkvo_ffn_1000s_0507_124705 \
  --checkpoint latest \
  --device cuda:0
```

`scripts/flip_run_2.sh` 也支持同名子命令。默认不会覆盖已有完整视频；如需重建
目标目录中的同名 clip，显式加 `--overwrite`。

默认输出：

```text
training_data/pair/h2r/1s/Inspire_Collect_Clothes_MainCamOnly_syn/
├── video/pair_NNNN.mp4
├── control_video/pair_NNNN.mp4
├── metadata.csv
├── manifest.jsonl
└── pair_order.jsonl
```

### Masquerade 风格 h2r 直接渲染 baseline

`src.pipeline.masquerade_baseline` 读取现有 `training_data/pair/h2r/1s/<task>/manifest.jsonl`，
按 task 选择 pair，再从 `training_data/segment/<task>/<episode>/<seg>_joints.parquet`
和对应 `seg*_video.mp4` 重新渲染 robot baseline。human 侧不依赖人工标注，而是直接从
`control_video/pair_XXXX.mp4` 自动估计 foreground mask、左右半边 bbox、trajectory
和逐帧 annotation JSONL，然后用 mask 对 control frame 做 inpaint 背景重绘，再在
该背景上合成不透明机器人 mesh。
这是一版可跑通的复现骨架，human 分割和背景重绘仍是启发式实现，后续需要继续提升
mask 稳定性、inpaint 质量和机器人遮挡边界。

默认输出：

```text
output/masquerade_baseline/h2r/1s/<task>/
├── video/pair_NNNN.mp4            (baseline robot render)
├── control_video/pair_NNNN.mp4    (原 human control clip)
├── background/pair_NNNN.mp4       (human mask inpaint 背景重绘)
├── gt_video/pair_NNNN.mp4         (原 robot GT clip)
├── human_overlay/pair_NNNN.mp4    (human mask/bbox overlay)
├── human_annotations/pair_NNNN.jsonl
├── human_annotations/pair_NNNN.npz
├── compare/pair_NNNN.mp4          (human | baseline | GT)
├── manifest.jsonl
└── summary.json
```

典型运行：

```bash
scripts/flip_run.sh masquerade_baseline -- \
  --task Inspire_Pickup_Pillow_MainCamOnly \
  --head 1 \
  --output-root tmp/masquerade_baseline_smoke
```

对于 `_syn` 数据，随后按普通 h2r pair 运行 VAE/T5 cache：

```bash
CUDA_VISIBLE_DEVICES=2 python -m src.pipeline.mitty_cache \
  --pair-dir training_data/pair/h2r/1s/Inspire_Collect_Clothes_MainCamOnly_syn \
  --output training_data/cache/vae/h2r/1s/Inspire_Collect_Clothes_MainCamOnly_syn \
  --t5-cache-dir training_data/cache/t5/h2r/1s \
  --device cuda:0 \
  --batch-size 4 \
  --prefetch-workers 8 \
  --prefetch-batches 2 \
  --save-workers 1
```

### Blur R2R 数据

`blur_r2r` 通过 `make_pair.py` 生成，但不再从 `seedance_direct/1s`
匹配可用 human clip；它直接使用三个 canonical robot Task 的全部
`training_data/segment/` 数据。1s 数据对每条 4s segment 生成 4 个非重叠
robot clip。生成 control 时要求对应
`training_data/sam2_mask/<task>/<episode>/<seg>.npz` 已存在，缺失会直接报错。

```bash
python -m src.pipeline.make_pair \
  --task all \
  --second 1s \
  --data-type blur_r2r \
  --workers 64 \
  --clean
```

默认 blur 参数对齐旧 patch 数据：`--blur-ksize 51`，
`--blur-pixel-expand 16`。如需调试历史任务，必须先确保该任务已有 SAM2 mask。

### Identity 数据

```bash
python -m src.pipeline.make_robot_pair \
  --task all \
  --max-segments 500 \
  --clean

CUDA_VISIBLE_DEVICES=2 python -m src.pipeline.mitty_cache \
  --pair-dir training_data/pair/identity_r2r/1s/Inspire_Collect_Clothes_MainCamOnly \
  --output training_data/cache/vae/identity_r2r/1s/Inspire_Collect_Clothes_MainCamOnly \
  --t5-cache-dir training_data/cache/t5/identity_r2r/1s \
  --device cuda:0 \
  --batch-size 4
```

## 训练命令

## 离线综合评估

`src.pipeline.evaluate_mitty_models` 用于比较训练完成的 Mitty LoRA run：按
训练入口相同的 task 级 `pair_order.jsonl` 顺序表选择样本。正式评估推荐显式传
`--in-task-eval-size K --ood-eval-size M`：in-task 会按各 in-task task 的数据量
比例分配总数 K，并从每个 task 顺序表尾部读取对应数量；OOD 会从 OOD task
顺序表尾部读取总数 M（多 OOD task 时同样按数据量比例分配）。未传固定数量时，
兼容旧逻辑，从每个 in-task task 和 OOD task 的顺序表尾部读取
`--eval-tail-percent`。选中样本再用对应 VAE cache 生成视频，并复制 pair 目录中
的原始 `video` / `control_video` 作为 GT 和 Control，计算 PSNR、SSIM、LPIPS、
FID 和 FVD。默认评估：

- `Mitty-transfer-124d_r128_2000s_0425_1456/ckpt/step-2000.safetensors`
- `Mitty-transfer2LoRA-124d_r128_2000s_0425_1425/ckpt/step-2000.safetensors`
- `pair_1s` preset 下每个 in-task/OOD task 的尾部 10%

推荐通过统一入口运行 GPU 评估：

```bash
scripts/flip_run.sh eval_mitty --cuda 2 -- \
  --device cuda:0 \
  --in-task-eval-size 32 \
  --ood-eval-size 8
```

输出目录默认写在对应 ckpt 旁边：`training_data/log/<run>/ckpt/<step>_eval/`。
每个 split 下保存 `gen_*.mp4`、`gt_*.mp4`、`ctrl_*.mp4`，并在该目录写出
`summary.csv`、`summary.json` 和 `data_split/` 选择记录。可用 `--splits
in_task_eval ood_eval` 指定 split；`eval` 是 `in_task_eval` 的兼容别名，
`ood` 是 `ood_eval` 的兼容别名。如只想复算已有视频的指标，可加
`--no-generate`；如生成过程被中断，可加 `--resume-existing` 复用已经完整写出的
`gen/gt/ctrl` 三元组，只补齐缺失或不可解码的样本。如需继续写到集中目录，可显式传
`--output-dir`。
`data_split/*.jsonl` 会保留每个样本的 `pair_id`、`order_index`、
`pair_order_path`、`source_id`、`episode`、`seg`、`clip_start`、`clip_dur`
等字段，用于把评估视频对应回 pair 顺序表。

离线指标计算会打印 generation、Local crop 和 metrics 各阶段进度。metrics 阶段
默认用 `--metric-workers 8` 并行读取视频、计算 PSNR/SSIM 和 mask 局部指标；
LPIPS 默认按 `--lpips-batch-size 16` 合批跑 VGG，FID/Local FID 默认按
`--feature-batch-size 32` 合批跑 InceptionV3，FVD 默认按
`--fvd-batch-size 4` 合批跑 S3D。显存不足时可调小这些 batch size；CPU
解码/SSIM 不足时可调大 `--metric-workers`。如需安静日志，可加
`--no-progress`。

当评估数据类型为 `blur_r2r` 时，`evaluate_mitty_models` 默认会启用
mask-region 指标：从
`training_data/sam2_mask/<task>/<episode>/<seg>.npz` 读取与该 clip 对齐的
机器人 mask，按 `clip_start` / `clip_dur` / `augment` 对齐。summary 会额外写出
全局 `mse`，以及局部 `foreground_mse`、`foreground_psnr`、
`foreground_ssim`、`background_mse`、`background_psnr`、`background_ssim`；
这些局部配对指标只统计 mask 内或 mask 外像素，不比较黑底区域。区域 FID 使用
Local FID 口径：按每帧 robot mask 的 bounding box 向外扩展 24 px 后裁剪
gen/GT 局部图像，再把局部 crop resize 到 InceptionV3 输入尺寸计算 Frechet
距离，字段为 `foreground_local_fid`。区域 FVD 使用同一个 bbox crop 局部口径：
先把每帧 gen/GT 裁到 robot mask bbox，再用 S3D video feature 计算视频级
Frechet 距离，字段为 `foreground_local_fvd`。旧的黑底区域 FVD 字段
`foreground_black_fvd`、`background_black_fvd` 不再输出。缺少 mask 或 manifest
缺少对齐字段会直接报错。可用 `--mask-region-metrics off` 关闭，或用
`--sam2-mask-root` 指向其他 SAM2 mask 根目录。`--no-fid` 会关闭全局 FID/FVD、
Local FID 和 Local FVD，局部 MSE/PSNR/SSIM 仍会计算。

如需查看 Local FID 实际关注的局部区域，可加 `--write-local-videos`。脚本会在
每个 split 下写出 `local_fid/`：

- `gen_*.mp4`、`gt_*.mp4`、`ctrl_*.mp4`：按 robot mask bbox 裁剪并 resize 后的
  局部视频，和 Local FID / Local FVD 使用同一组 bbox。
- `compare_*.mp4`：`GT | gen | ctrl` 三列局部对比。
- `gen_overlay_*.mp4`、`gt_overlay_*.mp4`、`ctrl_overlay_*.mp4`：原始画面大小，
  用黄色半透明 robot mask 和黄色 bbox 标出参与 Local 指标计算的区域。
- `compare_overlay_*.mp4`：`GT | gen | ctrl` 三列原始画面 overlay 对比。
- `patch_index.jsonl`：每个 sample 的 `sample_id`、`pair_id`、`order_index`、
  `pair_order_path`、`source_id`、`mask_path`、Local/overlay 视频路径、
  `frame_bboxes_xyxy`、`union_bbox_xyxy` 和 overlay 颜色。

Local 视频默认使用与 Local FID / Local FVD 相同的 per-frame bbox 口径，默认
`--local-video-margin 24 --local-video-size 300 --local-video-bbox-mode frame`。
Local FID 指标内部仍使用 InceptionV3 的 299 输入尺寸；视频默认写成 300 是因为
H.264 `yuv420p` 输出要求宽高为偶数。
如果只为人工观看、希望减少 bbox 抖动，可改用 `--local-video-bbox-mode union`。

如需避免 bbox 引入大面积背景，可启用 mask-selected Patch FID。Patch FID
在每帧按固定网格选择 mask 像素数严格大于阈值的 patch，并把同一组 patch 坐标同时
应用到 gen/GT，提取 InceptionV3 特征后计算 Frechet 距离，字段为
`foreground_patch_fid`。常用只复算已有视频的命令：

```bash
scripts/flip_run.sh eval_mitty --cuda 2 -- \
  --no-generate \
  --patch-fid-only \
  --write-patch-overlays \
  --patch-size 64 \
  --patch-stride 32 \
  --patch-min-mask-pixels 5
```

`--patch-fid-only` 会跳过 LPIPS、全局 FID/FVD 和 bbox Local FID/FVD，只加载
InceptionV3 计算 `foreground_patch_fid`。`--write-patch-overlays` 会在每个
split 下写出 `patch_fid/`，包含 `gen_overlay_*.mp4`、`gt_overlay_*.mp4`、
`ctrl_overlay_*.mp4`、`compare_overlay_*.mp4` 和 `patch_index.jsonl`；overlay
中半透明红色表示 robot mask，青/绿色矩形表示该帧参与 Patch FID 的 patch。
默认 `--patch-min-mask-pixels 5 --patch-coverage-threshold 0.0
--patch-max-per-frame 0`，表示 patch 内 mask 像素数必须严格大于 5，且不限制
每帧 patch 数量；如需更稀疏的人工检查或更快的 smoke，可显式调高像素/coverage
阈值或设置每帧上限。
summary 会额外记录 `foreground_patch_count`、`foreground_patch_size`、
`foreground_patch_stride`、`foreground_patch_coverage_threshold`、
`foreground_patch_min_mask_pixels`、`foreground_patch_max_per_frame` 和
`foreground_patch_max_per_video`。

### Wan VAE IDM 动作一致性

`src.pipeline.wan_vae_idm` 训练一个 Video2Action IDM，用冻结的 Wan 2.2 VAE
latent 作为视频特征，再接纯 3D CNN + MLP action head。256x256、17 帧输入经
Wan VAE 后实际 latent 为 `[B, 48, 5, 16, 16]`；默认 `cnn_mlp` head 使用
`conv_channels=256`、`conv_blocks=4`、`readout_dim=1024`、`hidden_dim=1024`、
`mlp_layers=3`，3D CNN 输出 `[B, 256, 5, 8, 8]`，空间平均池化得到
`[B, 5, 256]`，flatten 后经 1024 维 readout 和 MLP 输出 action。
默认任务仍是 `G1_WBT_Inspire_Collect_Clothes_MainCamOnly`，也可以用
`--task-short` / `--task-full` 显式切到其他 MainCamOnly 任务；脚本不会在任务名
不匹配时隐式 fallback。

训练标签直接从原始 LeRobot parquet 读取；脚本会读取 `data/chunk-*/*.parquet`
下所有 parquet 文件，不再假设每个任务只有 `file-000.parquet`：

- `action.ee_action`：12 维双臂末端执行器 action。
- `action.hand_cmd`：12 维双手命令。
- `action.robot_q_desired`：36 维全身期望关节 / base action，可用于 full-body
  IDM。

训练样本从 `training_data/segment/<task-short>/` 的 30fps segment 视频切 1s
clip，默认采样 17 帧 @ 16fps，并只取 17 帧窗口中间帧对应的 action 作为
target。target 会按训练集
mean/std 标准化后训练，checkpoint 中保存反标准化参数。缺少原始 action 字段、
segment 对齐字段或 eval 记录任务名不匹配时会直接报错。
默认 `--target-mode arm_hand` 保持 task47/task48 的 24 维口径；显式传
`--target-mode full_body` 时，target 改为
`action.robot_q_desired(36) + action.hand_cmd(12)` 共 48 维，模型输出头也随之变为
48 维。

可选的可见 action mask 由 `src.pipeline.action_mask_precompute` 生成，默认写到
`training_data/action_mask/<task>/<episode>/<seg>.npz`，并在根目录写
`index.jsonl`。precompute 复用 G1 FK + mesh 投影。默认 `--target-mode arm_hand`
逐帧渲染 `left_arm`、`right_arm`、`left_hand`、`right_hand` 四个 action 相关
body part；`--target-mode full_body` 会改为
`torso`、`left_leg`、`right_leg`、`left_arm`、`right_arm`、`left_hand`、
`right_hand` 七个 body part。
只有 mesh 像素数达到 `--min-part-pixels`，且至少一个对应 link origin 投影在图像
范围内时，该 part 才算可见。artifact 包含 `part_visibility`、
`part_pixel_counts`、`part_origin_in_frame_counts`、`action_mask`、`part_names`、
`action_dim_names`、`action_dim_parts` 和 `metadata_json`。metadata 记录 task、
episode、segment、fps、图像尺寸、阈值、相机参数和 schema version；task /
episode / segment / version 不匹配时训练和评估会直接报错。
由于当前 IDM 只监督 17 帧 clip 的中间帧，precompute 可用
`--clip-middle-only` 只渲染训练 clip 会访问到的中间帧，并用 `--workers N`
按 segment 并行。训练端读取这类 artifact 时会检查 clip 中间帧是否已渲染；
如果 clip 参数和 precompute 参数不匹配，会直接报错，不会把未渲染帧静默当成
不可见。

当前 IDM action mask 映射显式固定为：

- `action.ee_action[0:6]`：左臂末端 action，对应 `left_arm OR left_hand` 可见。
- `action.ee_action[6:12]`：右臂末端 action，对应 `right_arm OR right_hand` 可见。
- `action.hand_cmd[0:6]`：左手命令，对应 `left_hand` 可见。
- `action.hand_cmd[6:12]`：右手命令，对应 `right_hand` 可见。

full-body 模式的 48 维 mask 映射为：

- `action.robot_q_desired[0:7]`：root / base，对应 `torso` 可见。
- `action.robot_q_desired[7:13]`：左腿，对应 `left_leg` 可见。
- `action.robot_q_desired[13:19]`：右腿，对应 `right_leg` 可见。
- `action.robot_q_desired[19:22]`：腰部，对应 `torso` 可见。
- `action.robot_q_desired[22:29]`：左臂，对应 `left_arm OR left_hand` 可见。
- `action.robot_q_desired[29:36]`：右臂，对应 `right_arm OR right_hand` 可见。
- `action.hand_cmd[0:6]`：左手，对应 `left_hand` 可见。
- `action.hand_cmd[6:12]`：右手，对应 `right_hand` 可见。

只生成当前 H2R 三个任务的 mask 时可逐任务运行，例如：

```bash
/home/leadtek/miniconda3/envs/flip/bin/python -m src.pipeline.action_mask_precompute \
  --task Inspire_Collect_Clothes_MainCamOnly \
  --target-mode full_body \
  --output training_data/action_mask \
  --resume

/home/leadtek/miniconda3/envs/flip/bin/python -m src.pipeline.action_mask_precompute \
  --task Inspire_Put_Clothes_into_Washing_Machine_MainCamOnly \
  --target-mode full_body \
  --output training_data/action_mask \
  --resume

/home/leadtek/miniconda3/envs/flip/bin/python -m src.pipeline.action_mask_precompute \
  --task Inspire_Pickup_Pillow_MainCamOnly \
  --target-mode full_body \
  --output training_data/action_mask \
  --clip-middle-only \
  --clip-stride 0.5 \
  --workers 16 \
  --resume
```

训练、验证和 eval 可通过 `--action-mask-root default` 启用 mask。clip-level mask
与 action label 对齐到同一 clip 的中间采样帧；`--action-mask-min-frame-ratio`
保留为阈值参数，但中间帧 mask 的 ratio 只会是 0 或 1。若某个 clip
没有任何可见 action 维度，默认 `--empty-action-mask-policy error` 直接失败；
需要显式过滤时使用 `--empty-action-mask-policy drop`。

小规模训练 smoke：

```bash
scripts/flip_run.sh wan_vae_idm --cuda 2 -- train \
  --device cuda:0 \
  --task-short Inspire_Collect_Clothes_MainCamOnly \
  --output-dir tmp/wan_vae_idm_collect_smoke \
  --max-samples 32 \
  --steps 40 \
  --batch-size 1 \
  --eval-every 10 \
  --resize 256x256
```

启用 mask 的训练 smoke 示例：

```bash
scripts/flip_run.sh wan_vae_idm --cuda 2 -- train \
  --device cuda:0 \
  --task-short Inspire_Collect_Clothes_MainCamOnly \
  --target-mode full_body \
  --action-mask-root default \
  --empty-action-mask-policy drop \
  --output-dir tmp/wan_vae_idm_collect_masked_smoke \
  --max-samples 32 \
  --steps 40 \
  --batch-size 1 \
  --eval-every 10 \
  --resize 256x256
```

训练过程会写出 `train_loss.csv`、`eval_loss.csv`、`val_predictions.csv`、
`best_val_predictions.csv`、`checkpoint.pt`、`best_checkpoint.pt` 和
`loss_curve.png`。`eval_loss.csv` 默认每 `--eval-every` step 在 held-out
episode split 上评估，`loss_curve.png` 同时画训练 loss 滑动均值和 eval
loss 曲线。`val_predictions.csv` 对应 final step，`best_val_predictions.csv`
对应验证集 total MSE 最低的 checkpoint；启用 mask 时 checkpoint 选择改用
`masked_total_mse`。验证指标同时输出原始 action 单位下的 `*_mse`、
`masked_*_mse`、`*_relative_l2_error` 和 `masked_*_relative_l2_error`。
正式复算 action consistency 时优先使用 `best_checkpoint.pt`。
`--val-max-samples 0` 表示验证时使用全部 held-out samples。
如果训练更长，可用 `--lr-scheduler cosine --min-lr-ratio 0.05` 做学习率衰减。

已训练 checkpoint 可用同一 held-out split 复算验证集预测，不重训：

```bash
scripts/flip_run.sh wan_vae_idm --cuda 2 -- validate \
  --device cuda:0 \
  --task-short Inspire_Collect_Clothes_MainCamOnly \
  --checkpoint tmp/wan_vae_idm_collect_smoke/best_checkpoint.pt \
  --output-dir tmp/wan_vae_idm_collect_smoke/validate_all \
  --action-mask-root default \
  --max-samples 0 \
  --clip-stride 0.5 \
  --val-max-samples 0 \
  --resize 256x256
```

已有 eval 视频复算 action consistency：

```bash
scripts/flip_run.sh wan_vae_idm --cuda 2 -- eval \
  --device cuda:0 \
  --task-short Inspire_Collect_Clothes_MainCamOnly \
  --checkpoint tmp/wan_vae_idm_collect_smoke/best_checkpoint.pt \
  --eval-dir training_data/log/<run>/full_eval/in_task_eval \
  --records-jsonl training_data/log/<run>/full_eval/data_split/in_task_eval.jsonl \
  --output-csv tmp/wan_vae_idm_collect_eval/action_metrics.csv \
  --action-mask-root default \
  --max-samples 8 \
  --resize 256x256
```

输出 CSV/JSON 字段包括 `gt_idm_arm_mse`、`gt_idm_hand_mse`、
`gt_idm_arm_hand_mse`、`gen_idm_arm_mse`、`gen_idm_hand_mse`、
`gen_idm_arm_hand_mse`，以及对应的 `idm_*_gap` / `idm_*_ratio`。启用 mask
后额外输出 `gt_idm_masked_*_mse`、`gen_idm_masked_*_mse`、
`idm_masked_*_gap` / `idm_masked_*_ratio`、`visible_action_count`、
`visible_action_ratio`、`visible_arm_ratio`、`visible_hand_ratio` 和逐维
`mask_XX` / `mask_ratio_XX`。其中 GT 视频上的 IDM error 作为该监督模型在真实
视频上的误差下限，生成视频指标越接近 GT 下限，说明 arm-hand action cue 越一致。

### 两帧 Pair IDM

`src.pipeline.wan_pair_idm` 是独立于 `wan_vae_idm` 的两帧 inverse dynamics
baseline。它不加载 Wan VAE，而是直接读取 segment 视频中的相邻 RGB 帧，将
`frame_t` 和 `frame_{t+1}` resize 后拼成 6 通道输入，并预测同一原始 LeRobot
episode 中 `frame_index=t` 的 action。该脚本固定使用
`(s_t, s_{t+1}) -> a_t` 对齐口径：

- arm 网络预测 `action.ee_action` 12 维。
- hand 网络预测 `action.hand_cmd` 12 维。
- arm / hand 是两套独立小 CNN，在同一训练入口中同时训练和保存，不共享参数。
- `validate` / `eval` 默认复用 checkpoint 中保存的数据 root、resize、split 和
  seed，避免训练与复算使用不同划分；需要旧 checkpoint 回退到 CLI split 时，
  必须显式传 `--allow-cli-split`。
- 指标同时包含原始 MSE、normalized MSE、per-dim R2 / correlation 和预测方差比，
  用于判断模型是否只是回归均值。

训练输出包括 `train_loss.csv`、`eval_loss.csv`、`checkpoint.pt`、
`best_checkpoint.pt`、`val_predictions.csv`、`best_val_predictions.csv` 和
`loss_curve.png`。checkpoint 中分别保存 `arm_model_state` 与
`hand_model_state`，并保存两类 action 各自的 mean/std。

Humanoid Everyday H1 子集使用独立 `src.pipeline.humanoid_pair_idm` 入口。该路径直接读取 LeRobot 目录：
`data/chunk-*/*.parquet` 提供 `action`，`videos/chunk-*/egocentric/*.mp4` 提供
RGB 帧。H1 版本不再拆成旧 WBT 的 `ee_action` / `hand_cmd` 两个 12 维头，而是
训练一个单独模型输出完整 26 维 `action`。当前默认 `--model-arch motion_transformer`，
旧的 `--model-arch transformer` 仍保留为 legacy checkpoint / ablation 对照，也可以显式传
`--model-arch small_cnn` 复现实验对照。task064 新增 `--model-arch motion_transformer_v2`
作为准确率优化实验模型：

H1 上 Mean baseline、AdaWorld latent decoder 和 RGB motion Transformer 的统一方法、
数据 split、训练命令和指标对比见 `doc/h1_idm_methods.md`。

- 输入：`frame_t` 与 `frame_{t+d}` resize 后拼成 6 通道 RGB。
- 输出：半开区间 `action[t:t+d]` 的均值，默认 `d=1`，`frame_delta` 记录这个区间
  长度。
- `frame_stride` 只表示候选起点采样步长；`frame_delta` 表示动作平均窗口和第二帧间隔。
- 默认 split 为 `episode`，避免同一 episode 的 pair 同时落入 train / eval；
  显式 `--train-samples` / `--eval-samples` 也会按 episode 顺序截取，保证
  train/eval 来自不重叠 episode。
- 旧 checkpoint 若缺少 replay 所需的 split/config 字段，需要显式传
  `--allow-cli-split` 才能继续用 CLI 参数复算。
- 指标同时包含原始 MSE、mean baseline、normalized MSE、per-dim R2 / correlation
  和预测方差比，方便判断模型是否只是回归均值。
- `motion_transformer` 在两帧 RGB 上做 patch embedding 后，额外构造 patch 级 motion token
  `[(x_t, x_{t+d}, x_{t+d}-x_t, |x_{t+d}-x_t|)]`，再加入 CLS token、motion CLS、frame
  embedding 和 2D sin/cos 位置编码，用 `TransformerEncoder` 聚合后通过
  `cls + motion_cls + frame0_mean + frame1_mean + motion_mean` 的 5 路读出回归 26 维 action。
- `motion_transformer_v2` 保持同一监督口径和 token 读出，但把原始 RGB
  `[frame_{t+d}-frame_t, |frame_{t+d}-frame_t|]` 通过独立 patch stem 注入 motion token，
  并使用 residual MLP readout head，减少旧模型输出方差偏低的问题。旧 checkpoint 若缺少
  `head_arch` / `raw_motion_stem` 字段，会按 `legacy_mlp` / `false` 严格 replay。
- 训练侧新增 `--variance-loss-weight`、`--variance-loss-warmup-ratio`、`--grad-accum-steps`
  和 `--amp`。variance loss 只在 normalized action 空间匹配 batch-level 预测方差；
  `--amp` 使用 CUDA bfloat16 autocast，主要用于 patch16 / 长训降低显存压力。
- `--init-checkpoint` 可从已有 checkpoint 严格初始化 train，要求模型配置和当前 train split
  action mean / std 一致；用于显式二阶段 fine-tune，不做隐式 fallback。
- 训练默认使用 AdamW `lr=3e-4`、`weight_decay=1e-2`、`betas=(0.9, 0.95)`，并使用
  cosine scheduler + 5% warmup，`min_lr_ratio=0.02`；Transformer 默认 `hidden_dim=256`、
  `transformer_depth=6`、`transformer_dropout=0.05`。
- task 太碎，当前架构对照不再默认按 task 切训练；使用 sample / episode split，并只把
  task 作为数据背景和后续审计维度。
- 和 AdaWorld task057 对齐的完整 H1 数据口径使用 `max_samples=0`、`frame_stride=1`、
  `split_by=episode`、`train_ratio=0.875`，得到 `488936` train samples / `1400`
  train episodes，以及 `71486` val samples / `200` val episodes。不要再用
  `train_samples=700` / `eval_samples=100` 这类 800-sample 小口径判断最终效果。
- 全量口径 `motion_transformer` 对照（`steps=2000`、`batch_size=32`、中途 eval 抽
  `4096` 个 held-out sample）在 `step=2000` 达到子集 normalized MSE `0.328745`、
  action relative L2 `0.274981`、`action_mse=0.052878`。
- 对 `best_checkpoint.pt` 运行完整 held-out validate（`71486` samples）得到
  normalized MSE `0.320904`、action relative L2 `0.272016`、归一化空间预测方差
  `0.689259`，`action_mse=0.051443`。同 split task061 optimized AdaWorld latent decoder
  held-out normalized MSE `0.349901`、action relative L2 `0.280356`、归一化空间预测方差
  `0.706071`、`action_mse=0.054646`；当前 RGB motion Transformer `action_mse` 低约 `5.9%`。
- task064 `motion_transformer_v2` patch16 `8000` step 已成为当前最强 H1 RGB IDM：
  完整 held-out validate（`71486` samples）得到 normalized MSE `0.196960`、
  action relative L2 `0.203904`、归一化空间预测方差 `0.853158`、
  `action_mse=0.028906`、`action_mean_dim_corr=0.898422`。相对 task060，`action_mse`
  约降低 `43.8%`，normalized MSE 约降低 `38.6%`。
- 早期 interval sweep 里，`d=1` 在 `1/2/4/8/16` 中最好，`best action_mse=0.107009`，
  比 mean baseline `0.110856` 略好；`d>1` 没有带来稳定提升，所以默认仍保持
  `frame_delta=1`。
- 当前仓库里的 `data/humanoid-everyday-h1-chunks0-6-8-200` 已重新复核：
  所有 1600 个 parquet 都能按 `action/frame_index/next.done` 口径读取，且对应视频文件都存在。
  task052 当时使用的临时 symlink 根 `tmp/h1_t052_valid_200_v2` 现在只作为历史记录保留，
  不再是当前必需的替代数据根。

### H2R Diffusion Policy BC

`src.pipeline.h2r_diffusion_policy` 是 task068 新增的 H2R / HumanAndRobot 下游
behavior cloning 入口。当前下载的 H2R v1 数据不是 LeRobot parquet，而是：

- `data/h2r/v1/data/<task>/episode_<id>.hdf5`
- `data/h2r/v1/video/<task>/episode_<id>/robot_camera.mp4`

入口会检查对应视频文件存在，但训练直接读取 HDF5 中的 `cam_data/robot_camera`，
避免先落盘转换成 LeRobot。默认状态由 `qpos,qvel,end_position,gripper_state`
拼接得到，当前维度为 21；监督 action 读取 HDF5 的 `action`，当前维度为 7。

Diffusion Policy 口径如下：

- condition：历史 clean robot video frames + 历史 clean state。
- diffusion variable：未来 action chunk。
- loss：对 GT action chunk 加噪后预测 noise 的 denoising MSE。
- inference / eval：从 action noise 采样 clean action chunk，输出 normalized action
  MSE 和 per-horizon MSE。
- 不预测未来视频，不使用 DreamZero / WAM，也不做真机或仿真闭环。

数据检查示例：

```bash
scripts/flip_run.sh h2r_diffusion_policy -- inspect \
  --device cpu \
  --tasks grab_cup_v1,grab_cube2_v1 \
  --max-episodes-per-task 3 \
  --max-train-samples 16 \
  --max-val-samples 8 \
  --resize 64x64 \
  --batch-size 4 \
  --output-json tmp/h2r_dp_t068_inspect.json
```

task068 smoke 训练示例：

```bash
scripts/flip_run.sh h2r_diffusion_policy --cuda 1 -- train \
  --device cuda:0 \
  --tasks grab_cup_v1,grab_cube2_v1 \
  --max-episodes-per-task 3 \
  --max-train-samples 32 \
  --max-val-samples 16 \
  --resize 64x64 \
  --batch-size 16 \
  --steps 220 \
  --log-every 20 \
  --eval-every 110 \
  --eval-max-batches 1 \
  --diffusion-steps 16 \
  --hidden-dim 128 \
  --depth 4 \
  --dropout 0.0 \
  --lr 0.001 \
  --output-dir tmp/h2r_diffusion_policy_t068_overfit
```

该 bounded smoke 使用两个 H2R task、4 个 train episode、2 个 val episode、32 个
train samples 和 16 个 val samples，验证目标只是跑通 HDF5 数据适配和 Diffusion Policy
BC 训练。当前结果显示 train denoising loss 从 `step=1` 的 `1.023897` 降到
`step=220` 的 `0.921348`；val denoising loss 从 `step=110` 的 `0.961218`
降到 `step=220` 的 `0.897937`。由于数据量很小，这个结果只证明训练链路和 BC loss
可下降，不代表绝对控制效果已经充分。

checkpoint 恢复 / eval 示例：

```bash
scripts/flip_run.sh h2r_diffusion_policy --cuda 1 -- eval \
  --device cuda:0 \
  --checkpoint tmp/h2r_diffusion_policy_t068_overfit/last_checkpoint.pt \
  --output-dir tmp/h2r_diffusion_policy_t068_overfit/eval_last \
  --eval-max-batches 1 \
  --prediction-batches 1
```

当前恢复验证输出 `denoise_loss=0.905152`、`sampled_action_mse_norm=1.848881`，
并写出 `eval_summary.json` 和 `predictions.csv`。后续完整下游评估可在该入口基础上
增加 video override，用同一 state/action label 固定 policy，只替换 GT / Ours /
Mitty / Phantom video observation。

### H2R SAM3 blur_r2r 外观训练复现

当前 H2R 三阶段复现只覆盖前两步：

1. step1 不重新训练 identity LoRA，直接复用已经可用的 checkpoint：
   `training_data/log/final_ours_step1_0507_004839-identity_r2r_1s-22592d_r32_self_qkvo_ffn_1000s_0507_004911/ckpt/step-1000.safetensors`。
2. step2 使用 H2R robot-camera 视频和 SAM3/SAM3.1 robot-arm mask 生成
   `blur_r2r` pair：清晰 robot clip 作为 target，SAM3 mask 区域 Gaussian blur
   后作为 control，训练外观恢复 adapter。
3. step3 暂不做。它需要 `(human/synthetic human, real robot)` 配对数据；当前
   H2R 原始数据只提供 robot-camera / state / action，不应在没有配对数据时启动 h2r 阶段。

SAM3 分割本身是显存较重的前置步骤，先按 `doc/sam3_h2r_segmentation.md`
的结论生成 mask artifact。`h2r_sam3_precompute` 使用 `sam3` conda 环境，
按 1s / 17 帧短段逐 clip 调用 SAM3.1，输出 episode 级 `.npz`；只填充训练 clip
会用到的 source frame，其它帧保持 0。训练数据转换入口只消费预计算结果，不会隐式调用 SAM3：

```text
training_data/h2r_sam3_mask/<task>/episode_<id>.npz
training_data/h2r_sam3_mask/<task>/episode_<id>/robot_camera.npz
training_data/h2r_sam3_mask/<task>/episode_<id>/robot_camera_mask.npz
training_data/h2r_sam3_mask/<task>/episode_<id>/robot_camera_mask.mp4
```

`.npz` 需要包含 `masks`，shape 为 `[T,H,W]` 或 `[T,N,H,W]`；多 object 会先取
union。默认 mask stride 为 1，即 SAM3 mask 帧与原视频帧一一对应。若 SAM3 按
抽帧结果落盘，必须显式传 `--mask-stride`，否则 frame/mask 对齐会直接报错。
`h2r_sam3_precompute --resume` 会读取 `.npz` 中的 `covered_frames`；只有现有
mask 覆盖本次请求的全部 source frame 时才跳过，否则会重算该 episode，避免
1-clip smoke 产物污染正式全量准备。

SAM3 mask precompute 的 dry-run：

```bash
scripts/flip_run.sh h2r_sam3_precompute -- \
  --tasks grab_cup_v1,grab_cube2_v1,push_box_random_v1 \
  --max-episodes-per-task 1 \
  --max-clips-per-episode 1 \
  --dry-run
```

单卡 smoke / 正式预计算示例：

```bash
scripts/flip_run.sh h2r_sam3_precompute --cuda 2 -- \
  --tasks grab_cup_v1,grab_cube2_v1,push_box_random_v1,roll \
  --output-root training_data/h2r_sam3_mask \
  --prompt "robot arm" \
  --backup-prompt "robotic arm" \
  --max-num-objects 1 \
  --clip-stride 1.0 \
  --resume
```

生成 H2R SAM3 blur pair 的 smoke / dry-run：

```bash
scripts/flip_run.sh h2r_sam3_blur_pair -- \
  --tasks grab_cup_v1,grab_cube2_v1,push_box_random_v1 \
  --max-episodes-per-task 1 \
  --max-clips-per-episode 2 \
  --dry-run
```

正式生成 pair：

```bash
scripts/flip_run.sh h2r_sam3_blur_pair -- \
  --tasks grab_cup_v1,grab_cube2_v1,push_box_random_v1,roll \
  --mask-root training_data/h2r_sam3_mask \
  --pair-root training_data/pair \
  --resize 224x416 \
  --clip-stride 1.0 \
  --clean
```

`224x416` 是当前 H2R SAM3 blur_r2r 的默认尺寸：Wan VAE 后 latent grid 为
`14x26`，能和 Wan/Mitty 训练前向输出对齐。不要使用 `240x432` 这类会得到
`15x27` 奇数 latent grid 的尺寸；训练 loss 阶段会因预测/目标 latent 尺寸不一致而失败。

输出仍使用维护中的 Mitty layout：

```text
training_data/pair/blur_r2r/1s/<h2r_task>/
├── video/pair_NNNN.mp4          # clear robot target
├── control_video/pair_NNNN.mp4  # SAM3-mask blurred robot control
├── metadata.csv
└── manifest.jsonl
```

随后逐 task 生成 VAE/T5 cache。例如：

```bash
scripts/flip_run.sh mitty_cache --cuda 2 -- \
  --pair-dir training_data/pair/blur_r2r/1s/grab_cup_v1 \
  --output training_data/cache/vae/blur_r2r/1s/grab_cup_v1 \
  --t5-cache-dir training_data/cache/t5/blur_r2r/1s \
  --device cuda:0 \
  --batch-size 4 \
  --prefetch-workers 8 \
  --prefetch-batches 2 \
  --save-workers 1
```

stage2 launcher：

```bash
DRY_RUN=1 scripts/run_final_ours_three_stage.sh

CUDA_ID=2 NPROC=1 \
MAIN_TRAIN_TASKS=grab_cup_v1,grab_cube2_v1,push_box_random_v1 \
OOD_TASKS=roll \
scripts/run_final_ours_three_stage.sh
```

`scripts/run_final_ours_three_stage.sh` 当前默认 `RUN_STAGE3=0`，因此只会复用 step1
checkpoint 并启动 step2。若以后补齐配对数据再启用第三阶段，必须显式传
`RUN_STAGE3=1`，并确认 `h2r_1s` 的 H2R 配对 pair/cache 已经存在。

2026-06-03 正式复现实验记录：

- H2R SAM3 mask 全量准备完成：`grab_cup_v1`、`grab_cube2_v1`、
  `push_box_random_v1`、`roll` 共 40 episodes / 353 clips。
- H2R `blur_r2r/1s` pair/cache 行数分别为：
  `grab_cup_v1=57`、`grab_cube2_v1=90`、`push_box_random_v1=64`、
  `roll=142`；VAE cache 抽样 latent shape 为 `(1,48,5,14,26)`。
- stage2 正式训练 run：
  `training_data/log/final_ours_h2r_sam3_step2_0603_220432-blur_r2r_1s-207d_r256_self_qkvo_ffn_1000s_0603_220442`。
  该 run 复用 step1 checkpoint，merge 180 个 rank-32 LoRA pair；stage2 训练
  rank 256 LoRA，targets 为 `self_attn.q,self_attn.k,self_attn.v,self_attn.o,ffn.0,ffn.2`。
- 为避免正式训练在模型加载期受到外部 SIGTERM / 显存压力影响，本次复现保留
  1000 steps 和 rank/target 设置，但使用较轻评估配置：
  `BATCH_SIZE=1 IN_TASK_EVAL_SIZE=4 OOD_EVAL_SIZE=4 IN_TASK_VIDEO_SIZE=0 OOD_VIDEO_SIZE=0 EVAL_VIDEO_STEPS=0`。
- 训练完成 1000 steps，用时 554s，最终 checkpoint：
  `training_data/log/final_ours_h2r_sam3_step2_0603_220432-blur_r2r_1s-207d_r256_self_qkvo_ffn_1000s_0603_220442/ckpt/step-1000.safetensors`。
  最终 eval：`eval_loss_in_task=0.1478`，`eval_loss_ood=0.1658`
  （各 4 samples x 5 t）。stage3 仍未启动。

H1 smoke 示例：

```bash
scripts/flip_run.sh humanoid_pair_idm --cuda 2 -- train \
  --device cuda:0 \
  --data-root /disk_n/zzf/flip/data/humanoid-everyday-h1-chunks0-6-8-200 \
  --output-dir tmp/humanoid_pair_idm_h1_smoke \
  --max-samples 128 \
  --max-pairs-per-episode 4 \
  --frame-delta 4 \
  --steps 10 \
  --batch-size 8 \
  --eval-every 5 \
  --val-max-samples 32 \
  --resize 256x256 \
  --split-by episode \
  --model-arch motion_transformer \
  --transformer-embed-dim 128 \
  --transformer-depth 2 \
  --transformer-num-heads 4 \
  --lr 3e-4 \
  --lr-warmup-ratio 0.05
```

task064 `motion_transformer_v2` smoke 示例，覆盖 raw motion stem、residual head、
variance loss、AMP 和梯度累积：

```bash
scripts/flip_run.sh humanoid_pair_idm --cuda 2 -- train \
  --device cuda:0 \
  --data-root /disk_n/zzf/flip/data/humanoid-everyday-h1-chunks0-6-8-200 \
  --output-dir tmp/humanoid_pair_idm_t064_v2_smoke \
  --max-samples 64 \
  --max-pairs-per-episode 4 \
  --frame-delta 1 \
  --split-by episode \
  --train-ratio 0.75 \
  --steps 4 \
  --batch-size 2 \
  --workers 0 \
  --eval-every 2 \
  --val-max-samples 16 \
  --resize 256x256 \
  --model-arch motion_transformer_v2 \
  --transformer-patch-size 16 \
  --transformer-embed-dim 128 \
  --transformer-depth 2 \
  --transformer-num-heads 4 \
  --transformer-dropout 0.02 \
  --hidden-dim 128 \
  --transformer-head-depth 2 \
  --lr 3e-4 \
  --weight-decay 3e-3 \
  --lr-warmup-ratio 0.25 \
  --variance-loss-weight 0.03 \
  --variance-loss-warmup-ratio 0.5 \
  --grad-accum-steps 2 \
  --amp
```

task064 正式长训首选 patch16 v2 配置：

```bash
scripts/flip_run.sh humanoid_pair_idm --cuda 2 -- train \
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

完整 held-out validate 仍必须单独跑：

```bash
scripts/flip_run.sh humanoid_pair_idm --cuda 2 -- validate \
  --device cuda:0 \
  --checkpoint tmp/humanoid_pair_idm_t064_v2_p16_s8000/best_checkpoint.pt \
  --output-dir tmp/humanoid_pair_idm_t064_v2_p16_s8000_validate \
  --batch-size 16 \
  --workers 8
```

task064 完整结果输出：

```text
tmp/humanoid_pair_idm_t064_v2_p16_s8000/best_checkpoint.pt
tmp/humanoid_pair_idm_t064_v2_p16_s8000/eval_loss.csv
tmp/humanoid_pair_idm_t064_v2_p16_s8000_validate/val_metrics.json
tmp/humanoid_pair_idm_t064_v2_p16_s8000_validate/val_predictions.csv
```

中途 `4096` held-out subset eval 曲线：

| step | norm MSE | relative L2 | action_mse | pred std ratio |
|------|---------:|------------:|-----------:|---------------:|
| 1000 | `0.462395` | `0.331649` | `0.076917` | `0.807803` |
| 2000 | `0.335283` | `0.283008` | `0.056010` | `0.863389` |
| 3000 | `0.295146` | `0.262496` | `0.048185` | `0.871074` |
| 4000 | `0.253979` | `0.235231` | `0.038695` | `0.898377` |
| 5000 | `0.231605` | `0.223533` | `0.034942` | `0.905373` |
| 6000 | `0.218726` | `0.215771` | `0.032558` | `0.924237` |
| 7000 | `0.203808` | `0.205522` | `0.029538` | `0.917772` |
| 8000 | `0.202892` | `0.205012` | `0.029392` | `0.926239` |

完整 `71486` held-out validate：

| metric | value |
|------|------:|
| `action_mse` | `0.028906064108014107` |
| `mean_baseline_action_mse` | `0.15635335445404053` |
| `action_norm_mse` | `0.1969597190618515` |
| `relative_l2_error` | `0.20390427137523634` |
| `pred_norm_var_mean` | `0.8531579971313477` |
| `target_norm_var_mean` | `1.003715991973877` |
| `action_mean_dim_r2` | `0.8071291836408468` |
| `action_mean_dim_corr` | `0.8984220096698174` |
| `action_pred_std_ratio_mean` | `0.9228853445786697` |

这里的 `pred_norm_var_mean` / `target_norm_var_mean` 由
`val_predictions.csv` 和 checkpoint 中保存的 train action mean / std 复算得到；
`val_metrics.json` 直接保存的是 `action_pred_std_ratio_mean`。

二阶段 fine-tune 结果：

- `tmp/humanoid_pair_idm_t064_v2_p16_ft_s3000_lr1e4` 从 `s8000/best_checkpoint.pt`
  初始化，`lr=1e-4`、variance loss `0.01`；4096-sample subset 在 step500/1000 为
  `action_mse=0.032071` / `0.032695`，没有刷新初始化 step0 的 `0.029392`。
- `tmp/humanoid_pair_idm_t064_v2_p16_ft_s1000_lr3e5` 从同一 checkpoint 初始化，
  `lr=3e-5`、无 variance loss / weight decay；step250/500 为 `0.030179` /
  `0.029506`，也没有刷新初始化。
- 因此当前推荐 checkpoint 仍是
  `tmp/humanoid_pair_idm_t064_v2_p16_s8000/best_checkpoint.pt`。

H1 700 训练 / 100 eval / 1000 step 示例：

```bash
scripts/flip_run.sh humanoid_pair_idm --cuda 2 -- train \
  --device cuda:0 \
  --data-root /disk_n/zzf/flip/data/humanoid-everyday-h1-chunks0-6-8-200 \
  --output-dir tmp/humanoid_pair_idm_h1_700train_100eval_s1000 \
  --max-samples 0 \
  --frame-stride 4 \
  --split-by episode \
  --train-samples 700 \
  --eval-samples 100 \
  --steps 1000 \
  --batch-size 16 \
  --eval-every 100 \
  --resize 256x256 \
  --model-arch motion_transformer \
  --lr 3e-4 \
  --lr-warmup-ratio 0.05
```

H1 interval sweep 示例：

```bash
for d in 1 2 4 8 16; do
  scripts/flip_run.sh humanoid_pair_idm --cuda 2 -- train \
    --device cuda:0 \
    --data-root /disk_n/zzf/flip/data/humanoid-everyday-h1-chunks0-6-8-200 \
    --output-dir tmp/humanoid_pair_idm_h1_sweep_d${d} \
    --max-samples 512 \
    --max-pairs-per-episode 16 \
    --frame-delta "${d}" \
    --split-by episode \
    --steps 200 \
    --batch-size 32 \
    --eval-every 50 \
    --val-max-samples 128 \
    --resize 256x256
done
```

### AdaWorld Action Encoder

`src.pipeline.adaworld_action_encoder` 只接入 AdaWorld 的 LAM action encoder，不运行
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
scripts/flip_run.sh adaworld_action_encoder --cuda 2 -- extract \
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

`src.pipeline.adaworld_action_decoder` 只训练 AdaWorld latent action 的下游解码器，不再
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
scripts/flip_run.sh adaworld_action_decoder --cuda 2 -- train \
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
scripts/flip_run.sh adaworld_action_decoder --cuda 1 -- train \
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
scripts/flip_run.sh adaworld_action_decoder --cuda 1 -- train \
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
scripts/flip_run.sh adaworld_action_decoder --cuda 1 -- validate \
  --device cuda:0 \
  --checkpoint tmp/adaworld_action_decoder_h1_full_t061/best_checkpoint.pt \
  --output-dir tmp/adaworld_action_decoder_h1_full_t061_validate_best \
  --workers 0
```

```bash
scripts/flip_run.sh adaworld_action_decoder --cuda 1 -- eval \
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
scripts/flip_run.sh adaworld_action_decoder --cuda 2 -- train \
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
scripts/flip_run.sh adaworld_action_decoder --cuda 2 -- validate \
  --device cuda:0 \
  --checkpoint tmp/adaworld_action_decoder_t063_full_c09_h384_lr8e4/best_checkpoint.pt \
  --output-dir tmp/adaworld_action_decoder_t063_full_c09_validate_best \
  --workers 0
```

```bash
scripts/flip_run.sh adaworld_action_decoder --cuda 2 -- eval \
  --device cuda:0 \
  --checkpoint tmp/adaworld_action_decoder_t063_full_c09_h384_lr8e4/best_checkpoint.pt \
  --output-dir tmp/adaworld_action_decoder_t063_full_c09_eval_best \
  --workers 0
```

小规模 smoke 示例：

```bash
scripts/flip_run.sh wan_pair_idm --cuda 2 -- train \
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
scripts/flip_run.sh wan_pair_idm --cuda 2 -- train \
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
scripts/flip_run.sh wan_pair_idm --cuda 2 -- validate \
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
scripts/flip_run.sh wan_vae_idm --cuda 2 -- eval-h2r \
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

如需在已有 `full_eval/` 视频上同时查看前景 patch 与前景 patch 之外的背景
patch 分布差异，可使用独立脚本：

```bash
python scripts/eval_background_patch_fid.py \
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

### 批量补跑 final step-1000 离线评估

`scripts/eval_final_step1000_missing.py` 用于补跑已经训练到 1000 step、但还没有
离线 `summary.csv` / `summary.json` 的正式 `final*` run。脚本默认跳过
`data_type=r2h` 和 `final_ours_step1*` / `final_ours_step2*`，并对其余 run 显式开启
`--mask-region-metrics on` 和 `--patch-fid`，使 summary 包含 `foreground_local_fid`、
`foreground_local_fvd`、`foreground_patch_fid` 以及前景/背景 MSE/PSNR/SSIM。
评估样本固定沿用旧正式口径：`--in-task-eval-size 80 --ood-eval-size 42`。
脚本会扫描 `training_data/log/final*`，只选择同时满足以下条件的 run：

- `train.log` 中出现 `step=1000/1000`；
- 存在 `ckpt/step-1000.safetensors`；
- `<run>/full_eval/summary.csv` 或 `summary.json` 缺失，或 summary 中缺少
  当前训练配置应有的 split / 指标字段。

脚本从每个 run 的 `train.log` 中解析原始训练参数，复用当时的 `task_name`、
`train_tasks`、`ood_tasks`、cache/pair/T5 路径和模型路径，再覆盖为固定
`80 in-task + 42 OOD`，通过 `scripts/flip_run.sh eval_mitty` 顺序补跑离线综合评估。
对 mixed h2r 训练日志中的 `h2r_1s_d400_s400`，脚本会按 `data_type=h2r` 映射到
离线评估入口支持的 `h2r_1s`，并使用 `original_train_tasks` /
`ood_eval_tasks` 作为评估 split 来源。

每个 run 的视频、`summary.csv` / `summary.json`、`data_split/` 和执行日志都会写到
该 run 自己的 `full_eval/`，例如：
`training_data/log/final_mitty_r128_0507-h2r_1s-400d_r128_self_qkv_1000s_0507_214548/full_eval/`。
其中 wrapper 日志为 `full_eval/eval.log`。

默认是 dry-run，只打印将要执行的命令：

```bash
/home/leadtek/miniconda3/envs/flip/bin/python scripts/eval_final_step1000_missing.py
```

确认 GPU 后执行完整补跑：

```bash
CUDA_ID=2 /home/leadtek/miniconda3/envs/flip/bin/python \
  scripts/eval_final_step1000_missing.py --execute
```

如需把任务队列分发到多张卡，传逗号分隔的 CUDA id。执行模式下脚本会按
`--poll-interval` 周期轮询这些卡的 `nvidia-smi` compute 进程；某张卡没有
compute 进程且当前脚本没有在该卡运行 eval 时，才会把队列里的下一个 run 投到
该卡。每张卡同时最多运行一个由本脚本启动的 eval，任务结束后继续等待下一次
空闲再取下一个 run：

```bash
/home/leadtek/miniconda3/envs/flip/bin/python \
  scripts/eval_final_step1000_missing.py \
  --runner flip_run_2 \
  --cuda-list 0,2,3 \
  --poll-interval 30 \
  --execute
```

可用 `--limit N` 先跑少量 run，`--cuda ID` 指定单卡评估，`--cuda-list IDS`
指定多卡轮询队列，`--poll-interval SEC` 调整空闲检查间隔，
`--runner flip_run_2` 可切换到 `scripts/flip_run_2.sh`，
`--force` 强制重跑已有 summary 的 run，
`--include-r2h` 可把默认跳过的 r2h run 纳入补跑，
`--include-ours-step1-step2` 可把默认跳过的 `final_ours_step1*` /
`final_ours_step2*` 纳入补跑，
`--no-local-metrics` 可关闭脚本强制添加的 Local 指标，`--no-patch-fid` 可关闭
脚本强制添加的 Patch FID。可用 `--in-task-eval-size`、`--ood-eval-size` 临时覆盖
默认 `80/42`，用 `--output-subdir` 临时覆盖默认 `full_eval`。

### 冒烟训练

```bash
CUDA_VISIBLE_DEVICES=2 python -m src.pipeline.train \
  --task-name smoke_test \
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

默认 run name 同时用于本地训练目录和 W&B run name，格式为：
`{Backbone}-{task_name}-{n_train}d_r{rank}_{lora_targets}_{max_steps}s_{MMDD_HHMMSS}`。
其中 `lora_targets` 会把 `--lora-target-modules` 压成文件名安全的短签名，
例如 `self_attn.q,self_attn.k,self_attn.v,cross_attn.q,cross_attn.k,cross_attn.v,ffn.0,ffn.2`
→ `self_qkv_cross_qkv_ffn`，`ffn.0,ffn.2` → `ffn`。时间戳精确到秒，避免同一分钟内启动多个实验时目录重名。
如果显式传 `--wandb-run-name <name>`，训练入口会直接用 `<name>` 作为
本地 `training_data/log/<name>/` 目录名和 W&B run name；grid/bash launcher
只需要在外层生成一次带时间戳的 run name，避免 W&B 与本地 log 目录各自取
创建时间导致后缀不一致。
`scripts/flip_run.sh train` 通过 flip 环境的 Python 执行
`torch.distributed.run`，避免依赖交互 shell 的 `torchrun` PATH。

需要让本地训练目录与默认 W&B run name 使用实验族前缀时，传
`--run-prefix <prefix>`；该值会原样替换默认 `Mitty` 前缀，例如
`--run-prefix mitty_h2r` 会生成 `training_data/log/mitty_h2r-...`。

LoRA 注入默认走细粒度 Attention 控制：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--lora-attn-types` | `self,cross` | Attention 类型，可选 `self` / `cross`，会展开到 `self_attn.*` / `cross_attn.*` |
| `--lora-attn-projections` | `q,k,v,o` | Attention 投影层，可选 `q`、`k`、`v`、`o` |
| `--lora-target-modules` | 自动展开 | 显式 PEFT target suffix 覆盖入口，例如 `self_attn.q,cross_attn.v` 或 `ffn.0,ffn.2` |

示例：

```bash
# 只在 self attention 的 q/o 上加 LoRA
scripts/flip_run.sh train --cuda 2 -- \
  --task-name pair_1s \
  --lora-attn-types self \
  --lora-attn-projections q,o

# 只在 cross attention 的 k/v 上加 LoRA
scripts/flip_run.sh train --cuda 2 -- \
  --task-name pair_1s \
  --lora-attn-types cross \
  --lora-attn-projections k,v

# Attention + FFN 等混合目标使用显式 target 覆盖
scripts/flip_run.sh train --cuda 2 -- \
  --task-name pair_1s \
  --lora-target-modules self_attn.q,cross_attn.q,ffn.0,ffn.2
```

### Mixed h2r 训练入口

`src.pipeline.train_mitty_mixed_h2r` 是独立实验入口，用于混合原始 h2r
数据和 r2h 模型自合成的 `_syn` h2r 数据。它不改变
`src.pipeline.runtime_data`、`src.pipeline.train` 或 `src.pipeline.train_mitty`
的默认 split 语义；新入口先构建显式 mixed split，再把选中的 cache 文件以
symlink 形式物化到当前 run 目录下的 `mixed_cache/`，最后复用 Mitty 训练循环。

固定 eval 规则：

- in-task eval 只从 `--original-train-tasks` 的原始 h2r `pair_order.jsonl`
  尾部按 `--in-task-eval-size` 选取，默认 80 条。
- OOD eval 只从 `--ood-eval-tasks` 的原始 h2r `pair_order.jsonl` 尾部按
  `--ood-eval-size` 选取，默认 42 条。
- `--syn-train-tasks` 只进入训练，不进入 in-task eval 或 OOD eval。
- original/syn 训练样本不按 `pair_order.jsonl` 头部取，而是按每个 task 内
  `pair_id` 升序取前 k 条；original 训练会先排除 stable eval 样本后再取。
  这样后续继续追加 `pair_XXXX` 时，同一配置下已覆盖的前 N 条训练样本保持稳定。
- 原始训练样本会排除稳定 eval 集；original/syn 训练样本的
  `robot_source_key` 必须无交集。缺少显式 `robot_source_key` 时入口会从
  `source_robot_task`、`episode`、`seg`、`clip_start`、`clip_dur`、
  `source_segment_id`、`source_id` 等溯源字段构造；构造失败直接报错。

示例：

```bash
scripts/flip_run.sh train_mitty_mixed_h2r --cuda 2,3 --nproc 2 -- \
  --task-name mixed_h2r_400_400 \
  --original-train-tasks Inspire_Collect_Clothes_MainCamOnly,Inspire_Put_Clothes_into_Washing_Machine \
  --syn-train-tasks Inspire_Collect_Clothes_MainCamOnly_syn,Inspire_Put_Clothes_into_Washing_Machine_syn \
  --ood-eval-tasks Inspire_Pickup_Pillow_MainCamOnly \
  --original-train-size 400 \
  --syn-train-size 400 \
  --in-task-eval-size 80 \
  --ood-eval-size 42 \
  --max-steps 1000 \
  --save-steps 100 \
  --eval-steps 100
```

每个 run 会写出：

- `data_split/train.jsonl`：original 与 syn 训练记录，字段 `mix_source`
  标记为 `original` 或 `syn`。
- `data_split/in_task_eval.jsonl`：只包含 original in-task eval。
- `data_split/ood_eval.jsonl`：只包含 original OOD eval。
- `data_split/config.json`：记录 `mode: mixed_h2r`、原始/syn/OOD task、
  请求数量、实际数量、`pair_order` 路径、`train_selection_order`、
  `eval_selection_order`、`data_seed` 和生成时间。

### LoRA layout/rank 网格搜索

`scripts/train_lora_grid.py` 是维护中的 LoRA 搜索入口，用于把
`LoRA layout × LoRA rank` 展开成一组单卡训练命令，并在指定 CUDA 列表上轮转顺序执行。
实际训练仍通过 `scripts/flip_run.sh train --nproc 1` 启动，因此会复用统一环境变量、cache 和 GPU 入口。
默认 launcher 会按机器 IP 自动选择：普通机器用 `scripts/flip_run.sh`，`10.20.1.2`
会自动切到 `scripts/flip_run_2.sh`；如需强制指定，可传 `--runner`。

常用示例：

```bash
scripts/train_lora_grid.py \
  --cuda 0,1 \
  --task-name h2r_1s \
  --train-size 490 \
  --merge-lora training_data/log/Mitty-identity_r2r_1s-10000d_r64_ffn_1000s_0429_185108/ckpt/step-0900.safetensors \
  --layouts self_qkv,self_qkvo,self_qkv_cross_qkv,self_qkvo_cross_qkvo,self_qkv_cross_qkv_ffn \
  --ranks 64,128,256
```

关键参数：

| 参数 | 说明 |
| --- | --- |
| `--merge-lora` | 空格或逗号分隔的 checkpoint 列表；训练入口会在加载时自动检测每个 merge LoRA 的 rank 与 target modules |
| `--train-lora` | 可训练 LoRA checkpoint；不显式传 `--ranks` / `--layouts` 时会让训练入口自动检测 rank 与 target modules |
| `--task-name` / `--data-type` / `--duration` | 数据 preset 与可选覆盖；默认 task 分配来自 `train_config.py` |
| `--pair-root` | pair 数据根目录，默认 `MAIN_ROOT/training_data/pair`；其中每个 task 目录保存 `pair_order.jsonl` |
| `--train-size` | 固定搜索用训练数据量；按各 train task 可用样本量比例分配，`0/-1` 使用扣除 eval 尾部后的全部训练样本 |
| `--in-task-eval-size` | in-task eval 总样本数；正数按 task 比例分配并从顺序表尾部取，`0/-1` 自动使用约 10% 尾部样本 |
| `--layouts` | layout 短名列表，支持 `self_qkv`、`self_qkvo`、`cross_qkv`、`cross_qkvo`、`self_qkv_cross_qkv`、`self_qkvo_cross_qkvo`、`ffn`、`self_qkv_ffn`、`self_qkvo_ffn`、`self_qkv_cross_qkv_ffn`、`self_qkvo_cross_qkvo_ffn` |
| `--ranks` | 逗号分隔的 LoRA rank 列表 |
| `--cuda` | 逗号分隔 CUDA id；展开后的实验按顺序轮转分配，并逐个等待完成 |
| `--dry-run` | 只打印命令，不启动训练 |

layout 名称直接写入本地 log 目录和 W&B run name，例如
`h2r_1s_self_qkv_cross_qkv_ffn_r128_20260502_153000`。该名字由 launcher 一次性生成并
通过 `--wandb-run-name` 传给训练入口，因此本地目录和 W&B 面板使用同一个
时间戳。需要临时 layout 时可写
`name=target1,target2`，例如 `self_q=self_attn.q`。
对比 qkv-only 时使用 `self_qkv` / `cross_qkv` / `self_qkv_cross_qkv`，这些 layout 不会在 `o` projection 上加 LoRA。

加载训练好的 LoRA 时，`--train-lora` / `--continue-lora` / `--init-lora`
会从 checkpoint 的 LoRA A/B tensor
自动检测 rank 和 target modules；未显式传 `--lora-rank` /
`--lora-target-modules` 时不需要手动维护这些参数。若显式传入的 rank 与
checkpoint 不一致，训练会直接报错；若注入后的 LoRA tensor 没有从 checkpoint
完整加载，也会直接报错，避免部分随机初始化。

推荐新命令使用更明确的两组参数：

| 参数 | 语义 |
| --- | --- |
| `--merge-lora <ckpt>` | 把 checkpoint 的 LoRA delta 写入 frozen base weight；可重复传多个 |
| `--train-lora <ckpt>` | 选择当前唯一开放训练的 LoRA checkpoint；会继续训练这套 adapter |
| `--train-lora-rank` / `--train-lora-target-modules` | 不传 `--train-lora` 时，按这些参数新建一套可训练 LoRA |

`--init-lora` 和 `--continue-lora` 仍保留为兼容别名。`--merge-lora` 与
`--train-lora` 可以同时使用：前者是冻结背景能力，后者是本次 optimizer
实际更新的 LoRA。若要“全程只有一个 LoRA 被训练”，不要传 `--merge-lora`，
只让后一阶段用上一阶段 checkpoint 作为 `--train-lora`。

### 单 LoRA 三阶段训练

需要让 identity → blur → h2r 三个任务共用一个 LoRA 时，使用
`scripts/train_three_stage_single_lora.py` 串行启动三个 stage。脚本在每个
stage 结束后读取该 run 最新 checkpoint，并作为下一 stage 的 `--train-lora`：

```bash
scripts/train_three_stage_single_lora.py \
  --cuda 0,1 \
  --nproc 2 \
  --lora-target-modules self_attn.q,self_attn.k,self_attn.v,self_attn.o,ffn.0,ffn.2 \
  --stage identity_r2r_1s:1000 \
  --stage blur_r2r_1s:1000 \
  --stage h2r_1s:1000 \
  --train-size 0 \
  --in-task-eval-size 16 \
  --ood-eval-size 16
```

如果第一阶段要接着已有 checkpoint 继续训练，传
`--train-lora <ckpt.safetensors>`；否则第一阶段从随机初始化的单个 LoRA
开始。三个 stage 默认会自动串联上一阶段 checkpoint。

也可以在 stage 里显式选择 merge 哪些 LoRA、训练哪套 LoRA：

```bash
scripts/train_three_stage_single_lora.py \
  --cuda 0,1 \
  --nproc 2 \
  --stage 'task=identity_r2r_1s;steps=1000;train=fresh;rank=64;targets=ffn.0,ffn.2' \
  --stage 'task=blur_r2r_1s;steps=1000;train=previous' \
  --stage 'task=h2r_1s;steps=1000;merge=training_data/log/frozen_style/ckpt/step-1000.safetensors;train=previous'
```

其中 `train=fresh` 表示新建可训练 LoRA，`train=previous` 表示继续上一阶段
输出，`train=<ckpt>` 表示从指定 checkpoint 继续训练。`merge=<ckpt1>,<ckpt2>`
表示这些 LoRA 只合并进 frozen base，不进入 optimizer。

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
- `eval video` 的样本集合由 `--data-seed` 和 eval split 固定，不随 step 改变；生成时按待生成视频的全局样本索引在所有 rank 间切分，所有 rank 写入同一个 `step-XXXX/`，文件名仍为 `gen_00.mp4`、`gt_00.mp4`、`ctrl_00.mp4` 这类全局编号。
- CSV、W&B、eval video 上传和在线指标只在 rank 0 执行；视频生成完成后会用 DDP barrier 等待所有 rank 写完。默认在线指标为 PSNR/SSIM/LPIPS/CLIP/FID/FVD；FID 使用 InceptionV3 pool3 逐帧特征计算帧级 Frechet 距离，FVD 使用 torchvision S3D Kinetics-400 1024-d 时空视频特征计算视频级 Frechet 距离。在线指标会按 frame batch 跑 LPIPS/CLIP/Inception、按 video batch 跑 S3D，并在训练日志中打印 `pairwise`、`lpips`、`clip`、`fid/*`、`fvd/*` 进度；这些指标仍只在 rank 0 训练卡上占用额外显存。如需关闭 FID/FVD，加 `--no-eval-video-frechet-metrics`；正式对比也可训练后用 `src.pipeline.evaluate_mitty_models` 或 `src.tools.eval_metrics` 离线计算。FID/FVD 是分布指标，样本太少时方差很大，正式汇报建议使用几十条以上视频。
- 正式实验默认 `--max-steps 1000 --save-steps 100 --eval-steps 100 --eval-video-steps 100`；smoke/debug 可临时调小。
- `--loss`、`--patch-dir` 已从正式训练入口移除，当前统一使用标准 Mitty loss。

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
- H2R 数据已转成 DreamZero / Gear 可读的 LeRobot-style 目录：
  `training_data/dreamzero_h2r_v1`，包含 190 个有效 episode、73,885 个训练 step，
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

task072 基于 task071 的 H2R top-level 1157 episode robot-only cache，复用以下产物：

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

task073 将 H2R top1157 robot-only WAM 从 task072 的短调参升级为固定 split
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

### Robot WAM 高 rank LoRA 扩容实验

task074 在 task073 的固定 split 和 sampled eval 口径上直接把 LoRA rank 提到
`256/512/768/1024`。固定配置保持 `lr=1e-4`、`action_loss_weight=1`、
`max_steps=39024`、`eval_every=5000`、每个 eval split 抽样 512 条，
`best_metric=eval_mean_loss`。

输出目录：

```text
training_data/log/robot_wam/h2r_top1157_s8_high_rank_v1/
├── r256_lr1e-4_aw1_s39024_eval512/
├── r512_lr1e-4_aw1_s39024_eval512/
├── r768_lr1e-4_aw1_s39024_eval512/
├── summary.csv
└── summary.md
```

容量验证：

| rank | 结果 | trainable params | best checkpoint size |
| ---: | --- | ---: | ---: |
| 256 | 通过 | 398,613,672 | 800.6 MiB |
| 512 | 通过 | 776,101,032 | 1520.6 MiB |
| 768 | 通过 | 1,153,588,392 | 2240.6 MiB |
| 1024 | OOM | 1,531,075,752 | 未产出 |

`rank=1024` 在单卡 24GB 上进入 AdamW optimizer step 时 CUDA OOM；`rank=768`
是本轮已验证可跑满的最高 rank。

训练内 sampled eval 结果：

| run | rank | best_step | best_eval_mean_loss | best_in_task | best_ood |
| --- | ---: | ---: | ---: | ---: | ---: |
| `r16_lr1e-4_aw1_s39024_eval512` | 16 | 39024 | 326.157 | 372.536 | 279.778 |
| `r256_lr1e-4_aw1_s39024_eval512` | 256 | 39024 | 326.424 | 371.793 | 281.054 |
| `r512_lr1e-4_aw1_s39024_eval512` | 512 | 35000 | 333.735 | 378.920 | 288.550 |
| `r768_lr1e-4_aw1_s39024_eval512` | 768 | 39024 | 339.105 | 381.299 | 296.910 |

高 rank 三组 best checkpoint audit 均通过：494 个 trainable tensor，无 `human` /
`control` key。单纯增大 rank 没有刷新 task073 的 `rank=16` sampled best，因此本轮
没有追加完整 fixed eval；最终泛化判断仍以 task073 的完整 fixed eval 作为当前参考。

### Cosmos Predict2B 备选路线

Cosmos Predict2B 只作为备选路线，不与 task067 主线混用。它适合在 Wan2.2-5B LoRA
仍超出显存预算时，先做更小的 video-only offline rollout 或重做一个 Cosmos-native
video-action 模型。

边界如下：

- Cosmos Predict2B 有 `Text+Image` / `Text+Video -> Video` 能力，因此可以作为更小
  video backbone 候选。
- 它不能直接替换 DreamZero 的 Wan backbone；Cosmos 的 VAE、DiT、scheduler、token
  layout、condition injection 和 cache 语义都不同。
- 如果走 Cosmos 2B，建议参考 Cosmos Policy 的 latent-frame / latent-slot injection：
  把 proprio、action、future-state 等编码成 Cosmos 原生 latent sequence 中的条件/预测槽位。
- 第一阶段可先做 video-only loop：`initial image/video + text -> future video`，再把
  生成片段回填继续生成；确认稳定后再加入 action/state。

## 验证

```bash
/home/leadtek/miniconda3/envs/flip/bin/python scripts/smoke_test.py --cuda 2
```

`smoke_test.py` 每次都会先跑轻量冒烟，再由 GPU 冒烟脚本执行
`nvidia-smi` 并把显卡状态写入 `tmp/smoke_test/gpu/nvidia_smi_before.log`，
随后复制 1 条 pair 到 `tmp/smoke_test/gpu/`，执行 `mitty_cache` 生成 VAE
cache，再跑 `train.py` 1 step + 1 sample eval。最终报告会标明本次 GPU
训练是 `single-card`、`dual-card` 还是更多卡测试。

如需明确双卡冒烟：

```bash
/home/leadtek/miniconda3/envs/flip/bin/python scripts/smoke_test.py --cuda 2,3 --nproc 2
```
