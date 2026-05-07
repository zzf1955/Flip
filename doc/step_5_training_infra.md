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

如果要从三个默认 task 中先生成固定总量，并按各 task 的可用 robot clip 数比例分配，
使用 `--source-task all --allocate-by-task proportional`。例如先生成 4000 条：

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

默认输出：

```text
training_data/pair/h2r/1s/Inspire_Collect_Clothes_MainCamOnly_syn/
├── video/pair_NNNN.mp4
├── control_video/pair_NNNN.mp4
├── metadata.csv
├── manifest.jsonl
└── pair_order.jsonl
```

随后按普通 h2r pair 运行 VAE/T5 cache：

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
