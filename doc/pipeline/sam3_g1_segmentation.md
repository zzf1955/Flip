# SAM3.1 G1 分割 Smoke 记录

## 目标

验证 SAM3/SAM3.1 能否直接替换当前 G1 blur_r2r 使用的 SAM2 mask。当前测试只覆盖
G1 segment 级 mask 预计算和 prompt 可行性，不代表已经切换训练数据生成口径。

## 入口

新增 G1 专用入口：

```bash
scripts/flip_run.sh g1_sam3_precompute --cuda <id> -- \
  --task <task-or-all> \
  --frame-count 61 \
  --chunk-size 17 \
  --output-root tmp/g1_sam3_61f_chunk17_clean \
  --prompt "robot arm" \
  --backup-prompt "robotic arm" \
  --max-num-objects 1 \
  --multiplex-count 16 \
  --print-gpu-memory \
  --write-overlay
```

该入口读取：

```text
training_data/segment/<task>/<episode>/seg*_video.mp4
```

并写出：

```text
<output-root>/<task>/<episode>/<seg>.npz
<output-root>/<task>/<episode>/<seg>.json
<output-root>/<task>/<episode>/<seg>_overlay.mp4
<output-root>/index.json
```

`.npz` 包含 `masks`、`covered_frames`、`selected_frames`、source video metadata、
prompt、SAM3 model 和 chunk 参数。`masks` 是 segment 全长 `[T,H,W]`，未选中的帧保持
0。summary 会额外与现有 `training_data/sam2_mask/` 做 IoU / 覆盖率对照。

## 显存测试

测试时间：2026-06-13。运行前显存：

```text
GPU0: 22682 / 24564 MiB used, only 1408 MiB free
GPU1:  3592 / 24564 MiB used, 20499 MiB free
GPU2:  3592 / 24564 MiB used, 20499 MiB free
GPU3:  4482 / 24564 MiB used, 19609 MiB free
```

GPU0 被 VLLM 占用，不适合跑 SAM3。后续 smoke 均使用 GPU1。

### 单 session 61 帧

命令：

```bash
scripts/flip_run.sh g1_sam3_precompute --cuda 1 -- \
  --task Inspire_Collect_Clothes_MainCamOnly \
  --max-segments 1 \
  --frame-count 61 \
  --chunk-size 61 \
  --output-root tmp/g1_sam3_61f_oom \
  --tmp-dir tmp/g1_sam3_61f_oom_frames \
  --prompt "robot arm" \
  --backup-prompt "robotic arm" \
  --max-num-objects 1 \
  --multiplex-count 16 \
  --print-gpu-memory
```

结果：OOM。

- SAM3.1 load 后 GPU1 从 `3592 MiB used` 增至 `7842 MiB used`。
- propagation 到第 16 帧附近时，本进程约占 `19.28 GiB`，已有其它进程占
  `3.49 GiB`，可见 GPU 剩余约 `751.75 MiB`。
- 报错为 `torch.OutOfMemoryError: Tried to allocate 1.27 GiB`。

结论：当前机器负载下，G1 2s 的 61 帧不能作为一个 SAM3.1 video session 直接跑。
G1 2s / 4s 预计算必须按短 chunk 处理。

### 17 帧 chunk

命令：

```bash
scripts/flip_run.sh g1_sam3_precompute --cuda 1 -- \
  --task all \
  --max-segments-per-task 1 \
  --frame-count 61 \
  --chunk-size 17 \
  --output-root tmp/g1_sam3_61f_chunk17_clean \
  --tmp-dir tmp/g1_sam3_61f_chunk17_clean_frames \
  --prompt "robot arm" \
  --backup-prompt "robotic arm" \
  --max-num-objects 1 \
  --multiplex-count 16 \
  --print-gpu-memory \
  --write-overlay
```

结果：三任务各 1 个 segment 均跑通，无 OOM。每个 61 帧 segment 被拆成 4 个 chunk。
每个 17 帧 chunk propagation 时显存会临时接近 `18.5 GiB used`，结束 session 后
SAM3 会释放大约 15GB reserved cache。

## Prompt 结果

指标含义：

- `nonempty`: SAM3 mask 非空帧数 / 61。
- `area`: SAM3 mask 平均面积占整帧比例。
- `IoU`: SAM3 mask 与现有 SAM2 mask 的平均 IoU。SAM2 不是人工 GT，但可作为当前
  blur mask 口径的 baseline。
- `sam2_in_sam3`: 当前 SAM2 mask 中有多少比例被 SAM3 覆盖，越低表示漏掉当前 blur
  口径中的机器人区域。

### `robot arm` / backup `robotic arm`

输出：

```text
tmp/g1_sam3_61f_chunk17_clean/
```

| task | nonempty | area | IoU | sam2_in_sam3 | 结论 |
| --- | ---: | ---: | ---: | ---: | --- |
| Collect Clothes | 61/61 | 0.0318 | 0.0587 | 0.0659 | 稳定但覆盖区域很小，明显不是 SAM2 全身口径 |
| Pickup Pillow | 48/61 | 0.0296 | 0.3094 | 0.3132 | 部分有效，仍漏较多 |
| Washing Machine | 9/61 | 0.0039 | 0.0590 | 0.0728 | 基本不可用 |

聚合：`mean_nonempty_frame_ratio=0.645`，`mean_sam2_iou=0.142`。

### `robot` / backup `humanoid robot`

输出：

```text
tmp/g1_sam3_61f_chunk17_robot/
```

| task | nonempty | area | IoU | sam2_in_sam3 | 结论 |
| --- | ---: | ---: | ---: | ---: | --- |
| Collect Clothes | 61/61 | 0.1399 | 0.3164 | 0.7789 | 覆盖更大，但明显过分割，面积约为 SAM2 的 2.29x |
| Pickup Pillow | 44/61 | 0.0728 | 0.5869 | 0.6633 | 当前三任务里最好，但仍有空帧 |
| Washing Machine | 0/61 | 0.0000 | 0.0000 | 0.0000 | 完全失败 |

聚合：`mean_nonempty_frame_ratio=0.574`，`mean_sam2_iou=0.301`。

### `robotic arm` / backup `robot arm`

只对 Washing Machine 单独测试：

```text
tmp/g1_sam3_61f_chunk17_robotic_arm_washing/
```

结果：`20/61` 帧非空，`mean_area_ratio=0.0051`，`IoU=0.1113`，
`sam2_in_sam3=0.1268`。比 `robot arm` 略好，但仍不能作为 clean G1 robot mask。

### 7 个 robot-only prompt sweep

根据补充要求，后续测试只处理 G1 robot segment，不处理 human / H2R 数据。命令：

```bash
scripts/flip_run.sh g1_sam3_precompute --cuda 1 -- \
  --task all \
  --max-segments-per-task 1 \
  --frame-count 61 \
  --chunk-size 17 \
  --prompt-mode text \
  --prompt-list "robot,robot arm,robot hand,robot body,humanoid robot,Unitree G1 robot,mechanical arm" \
  --backup-prompt "" \
  --output-root tmp/g1_sam3_prompt_sweep_robot_text \
  --tmp-dir tmp/g1_sam3_prompt_sweep_robot_text_frames
```

运行前再次检查全卡显存，GPU1 基本空闲；运行时 GPU1 上有约 4.46GB 的
`mitty_cache` 进程，但 61 帧拆 17 帧 chunk 的 7 prompt sweep 仍全部跑通，无 OOM。

聚合结果：

| prompt | nonempty ratio | area | IoU | 结论 |
| --- | ---: | ---: | ---: | --- |
| `robot` | 0.574 | 0.0709 | 0.301 | 当前 IoU 最高；Collect/Pillow 有效，Washing 全空 |
| `mechanical arm` | 0.432 | 0.0374 | 0.188 | 比 `robot arm` IoU 高，但仍漏很多，Washing 全空 |
| `robot arm` | 0.601 | 0.0204 | 0.123 | 多为局部 mask，和全身 blur 口径不一致 |
| `robot hand` | 0.754 | 0.0052 | 0.069 | 非空率最高但面积极小，只像手部局部 |
| `robot body` | 0.000 | 0.0000 | 0.000 | 全空 |
| `humanoid robot` | 0.000 | 0.0000 | 0.000 | 全空 |
| `Unitree G1 robot` | 0.000 | 0.0000 | 0.000 | 全空 |

`robot` per-task 结果：

| task | nonempty | area | IoU | sam2_in_sam3 | SAM3/SAM2 area ratio | 结论 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Collect Clothes | 61/61 | 0.1399 | 0.3164 | 0.7789 | 2.29x | 覆盖多但明显过分割 |
| Pickup Pillow | 44/61 | 0.0728 | 0.5869 | 0.6633 | 0.76x | 三任务里最好，仍有空帧 |
| Washing Machine | 0/61 | 0.0000 | 0.0000 | 0.0000 | 0.00x | 完全失败 |

### Text -> bbox 二阶段

脚本新增 `--prompt-mode text_bbox`：

1. 第一阶段用 text prompt 跑一次 SAM3.1。
2. 从 text mask 生成归一化 `[xmin, ymin, width, height]` bbox。
3. 第二阶段重新开 SAM3.1 session，用 bbox prompt 再跑一次；`--bbox-include-text`
   会把同一个 text prompt 一起传入第二阶段。

API smoke 结论：

- 纯 bbox prompt 接口可跑，但在 Collect Clothes 17 帧样本上最终 mask 全空。
- `bbox + 同一 text prompt` 可产生非空 mask；Collect Clothes 17 帧样本
  `17/17` 非空，IoU 约 `0.295`。

61 帧二阶段测试命令：

```bash
scripts/flip_run.sh g1_sam3_precompute --cuda 1 -- \
  --task all \
  --max-segments-per-task 1 \
  --frame-count 61 \
  --chunk-size 17 \
  --prompt-mode text_bbox \
  --bbox-include-text \
  --prompt-list "robot,mechanical arm" \
  --backup-prompt "" \
  --output-root tmp/g1_sam3_prompt_sweep_robot_text_bbox \
  --tmp-dir tmp/g1_sam3_prompt_sweep_robot_text_bbox_frames
```

结果：

| prompt | bbox chunks | no-bbox chunks | nonempty ratio | area | IoU | 相比 text-only |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `robot` | 7 | 5 | 0.574 | 0.0726 | 0.302 | 基本持平；未解决 Washing 全空 |
| `mechanical arm` | 7 | 5 | 0.574 | 0.0662 | 0.232 | 比 text-only IoU 0.188 有提升，但仍不足 |

`robot` 二阶段 per-task：

| task | nonempty | area | IoU | sam2_in_sam3 | SAM3/SAM2 area ratio | 结论 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Collect Clothes | 61/61 | 0.1434 | 0.3118 | 0.7837 | 2.35x | 仍过分割 |
| Pickup Pillow | 44/61 | 0.0745 | 0.5948 | 0.6787 | 0.78x | 略好于 text-only |
| Washing Machine | 0/61 | 0.0000 | 0.0000 | 0.0000 | 0.00x | 完全失败 |

`mechanical arm` 二阶段 per-task：

| task | nonempty | area | IoU | sam2_in_sam3 | SAM3/SAM2 area ratio | 结论 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Collect Clothes | 61/61 | 0.1559 | 0.2870 | 0.7868 | 2.53x | 覆盖更大但过分割更明显 |
| Pickup Pillow | 44/61 | 0.0426 | 0.4093 | 0.4510 | 0.51x | 有提升但仍漏 |
| Washing Machine | 0/61 | 0.0000 | 0.0000 | 0.0000 | 0.00x | 完全失败 |

## 当前结论

SAM3.1 在 G1 上可以按短 chunk 跑通，但 text-only prompt 或当前 text->bbox 二阶段
都不能稳定、干净地替代当前 SAM2 blur mask：

- 61 帧单 session 会 OOM；G1 2s/4s 必须按 17 帧左右短 chunk 处理。
- `robot arm` 在某些任务稳定但只覆盖小区域，和当前 SAM2 全身 blur 口径不一致。
- `robot` 在 Collect/Pillow 上覆盖更大，但 Collect 出现明显过分割，Washing Machine
  在 61 帧样本上全空。
- 纯 SAM3.1 在 Washing Machine 上只能算一般，严格说是不稳定：7 个 text prompt 中
  多数全空，`robot` / `mechanical arm` / `text->bbox` 都是 0/61；早先 `robot arm`
  只有 9/61 帧非空，`robotic arm` 也只有 20/61 帧非空且 IoU 0.111，不能形成干净的
  G1 robot blur mask。
- `robot hand` 非空率高但面积极小，只能说明局部手/夹爪可被分割。
- `robot body`、`humanoid robot`、`Unitree G1 robot` 在当前三任务样本上全空。
- text->bbox 的 `bbox + text` 二阶段可以跑，但只是轻微改变 `robot` / `mechanical arm`
  结果，不能解决 Washing Machine 全空，也不能消除 Collect 过分割。

因此，后续不能直接把 G1 blur_r2r 全量切到纯 SAM3 prompt 方案。若继续推进 SAM3 mask，
需要引入更强的空间约束或质量控制，例如 SAM2/FK bbox 约束、点 prompt / instance refine、
任务级 prompt 策略、跨 chunk mask 质量过滤，或保留 SAM2/FK 作为 mask seed 而只用 SAM3
做 refinement；否则 stage2 blur 会产生大量漏 mask 或过分割样本。
