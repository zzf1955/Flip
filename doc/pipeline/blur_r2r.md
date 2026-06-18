# 外观模糊数据构造

`blur_r2r` 用于三阶段训练的第二阶段：模型输入是局部模糊的 robot 视频，目标是清晰 robot
视频。它训练模型恢复机器人外观，而不引入 human 数据。

## G1 blur_r2r

G1 blur 数据主要来自 SAM2 mask：

```text
training_data/pair/blur_r2r/<duration>/<task>/
├── video/          # 清晰 robot target
└── control_video/  # SAM2 mask 区域局部 Gaussian blur 的 robot input
```

`2s61f30` / `2s61f30_slide` 都有对应 blur pair。

## human2robot blur_r2r

human2robot blur 数据使用 SAM3/SAM3.1 robot-arm mask：

```bash
scripts/flip_run.sh h2r_sam3_precompute --cuda 2 -- <args>
scripts/flip_run.sh h2r_sam3_blur_pair -- <args>
```

输出：

```text
training_data/h2r_sam3_mask/<task>/episode_*.npz
training_data/pair/blur_r2r/1s/<h2r_task>/
```

SAM3 细节见 [SAM2/SAM3 mask](sam2_sam3_masks.md) 和
[SAM3 H2R 分割记录](sam3_h2r_segmentation.md)。

