# Robot Patch — 全身机器人降质数据生成

为外观记忆学习生成训练数据。思路：对机器人全身区域降质，模型从降质 condition 恢复原始外观，从而学会记忆机器人的视觉细节。

与 `hand_patch` 的区别：`hand_patch` 仅处理手部 bounding box，`robot_patch` 使用 FK mesh 渲染的全身 pixel-level mask。

## 数据结构

```
training_data/pair/1s_patch/
├── train/
│   ├── video/           # pair_0000.mp4 ... (原始 robot 视频, 17帧@16fps)
│   ├── control_video/   # pair_0000.mp4 ... (降质 robot 视频)
│   ├── patch/           # pair_0000.pth ... (latent-space binary mask)
│   └── metadata.csv     # video, prompt, control_video 三列
├── eval/
│   ├── video/
│   ├── control_video/
│   ├── patch/
│   └── metadata.csv
└── source_map.json      # pair → (task, episode, seg, clip) 溯源
```

### patch.pth 格式

```python
{
    "mask":    Tensor(5, 30, 40),   # 二值 {0.0, 1.0}
    "weights": Tensor(5, 30, 40),   # 1.0 + 2.0 * mask → {1.0, 3.0}
}
```

- 空间：480×640 → 30×40（VAE 16× 下采样，每 16×16 block max-pool）
- 时间：17帧 → 5组 (4+4+4+4+1)，每组 mask 取 union
- `--patch-expand N`：latent 空间膨胀 N 个 cell（椭圆结构元素），默认 1
- `weights` key 与 `train.py` 的 `_load_patch_weights` 直接兼容

## 处理流程

```
training_data/segment/{task}/ → 4s segments (30fps)
  ↓ 每 segment 切 4 个 1s clip (30帧)
  ↓ 30fps→16fps 重采样选 17 帧
  ↓ 逐帧 FK → render_mask() → 全身 pixel mask
  ↓ soften_mask() → alpha 混合降质
  ↓ pixel mask → latent mask (5, 30, 40)
  → video/ + control_video/ + patch/
```

### Segment 级批处理

同一 segment 的 4 个 clip 共享同一次视频读取和 parquet 加载，避免重复 I/O。

### 降质模式

三种模式均使用 `soften_mask()` 做 alpha 混合，避免硬边界伪影：

| 模式 | 参数 | 效果 |
|------|------|------|
| `blur` | `--blur-ksize 51` | 机器人区域高斯模糊 |
| `noise` | `--noise-std 50.0` | 机器人区域叠加高斯噪声 |
| `mean` | — | 机器人区域填充该区域的 RGB 均值 |

### soften_mask 参数

```python
7×7 blur → threshold 128 → 15×15 ellipse dilate → 21×21 blur
```

比 inpaint 用的 `postprocess_mask`（41×41 膨胀）更紧凑，因为目的是降质而非去除。

### FK 模型切换

按 task 名自动检测 hand_type（inspire/brainco），切换时重新加载 FK model + mesh_cache。BrainCo task 跳过 Inspire 手部 mesh。

## 命令

```bash
# 全量生成（默认 blur）
python -m src.pipeline.robot_patch --task all --degrade blur

# noise 模式
python -m src.pipeline.robot_patch --task all --degrade noise --noise-std 60

# mean 模式（Inspire 任务只取 50 segments）
python -m src.pipeline.robot_patch --task inspire --degrade mean --max-segments 50

# 断点续跑
python -m src.pipeline.robot_patch --task all --degrade blur --resume

# 清空重来
python -m src.pipeline.robot_patch --task all --degrade blur --clean
```

### CLI 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--task` | required | 任务短名 / 逗号分隔 / `all` / `inspire` / `brainco` |
| `--degrade` | `blur` | `blur` / `noise` / `mean` |
| `--blur-ksize` | 51 | blur 高斯核大小（奇数） |
| `--noise-std` | 50.0 | noise 模式的标准差（0-255 scale） |
| `--patch-expand` | 1 | latent mask 向外膨胀的 cell 数（0=不扩展） |
| `--max-segments` | 0 | 每 task 最大 segment 数（0=不限） |
| `--per-task-eval` | 5 | 每 task 预留 eval 的 segment 数 |
| `--seed` | 42 | train/eval 划分随机种子 |
| `--workers` | 4 | 视频写入线程数 |
| `--resume` | false | 跳过已存在的输出 |
| `--clean` | false | 清空 `pair/1s_patch/` 后重新生成 |

## 下游训练

生成的数据直接兼容现有 cache + 训练 pipeline：

```bash
# 1. 编码为 VAE cache
python -m src.pipeline.mitty_cache \
  --pair-dir training_data/pair/1s_patch/train \
  --output output/cache/robot_patch/train --device cuda:0

python -m src.pipeline.mitty_cache \
  --pair-dir training_data/pair/1s_patch/eval \
  --output output/cache/robot_patch/eval --device cuda:0

# 2. 训练（robot patch 加权 loss）
torchrun --nproc_per_node=4 -m src.pipeline.train \
  --backbone mitty --loss hand_patch \
  --patch-dir training_data/pair/1s_patch/train/patch \
  --cache-train output/cache/robot_patch/train \
  --cache-eval output/cache/robot_patch/eval \
  --epochs 3 --repeat 5 --save-steps 50 --eval-steps 50
```

## 相关文件

| 文件 | 作用 |
|------|------|
| `src/pipeline/robot_patch.py` | 数据生成主脚本 |
| `src/core/render.py` | `render_mask()` — FK mesh → pixel mask |
| `src/core/fk.py` | `build_q()` + `do_fk()` + mesh 预加载 |
| `src/pipeline/mitty_cache.py` | 编码 pair → .pth cache |
| `src/pipeline/train.py` | 训练入口，`_load_patch_weights` 消费 patch.pth |
