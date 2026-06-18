# SAM2 / SAM3 Mask

当前主线同时保留 SAM2 和 SAM3/SAM3.1：

- SAM2：G1 segment 的稳定 mask 预计算，主要服务 G1 `blur_r2r`。
- SAM3/SAM3.1：human2robot robot-arm / gripper mask，以及 G1 prompt smoke。

## SAM2

入口：

```bash
scripts/flip_run.sh sam2_precompute --cuda 0 -- --task all --device cuda:0 --resume
```

输出：

```text
training_data/sam2_mask/<task>/...
```

## SAM3 / SAM3.1

G1 分割 smoke：

```bash
scripts/flip_run.sh g1_sam3_precompute --cuda 2 -- <args>
```

human2robot 机械臂 mask：

```bash
scripts/flip_run.sh h2r_sam3_precompute --cuda 2 -- <args>
```

SAM3 入口使用 `sam3` conda 环境。详细 prompt、显存和质量结论见：

- [SAM3 G1 分割记录](sam3_g1_segmentation.md)
- [SAM3 H2R 分割记录](sam3_h2r_segmentation.md)

