# 数据切片

数据切片负责把原始 episode / segment 转成训练需要的固定窗口。

## G1 2s/30fps

当前 G1 2s 口径使用 `2s61f30`：

- 61 帧，30fps，语义上约 2 秒。
- 61 帧满足 Wan VAE `4k+1` 输入长度约束。
- 对 4s/120 帧 segment，默认输出首窗口 `0..60` 和 tail-aligned 窗口 `59..119`。

入口：

```bash
python -m src.pipeline.g1_2s_slice_data --task all --workers 8
```

输出：

```text
training_data/slice/g1_2s61f30/
training_data/pair/identity_r2r/2s61f30/
training_data/pair/blur_r2r/2s61f30/
training_data/pair/h2r/2s61f30/
```

## G1 2s Seedance 滑窗

Seedance 2s 滑窗使用独立 duration `2s61f30_slide`：

```bash
python -m src.pipeline.g1_2s_seedance_slide_data --task all --workers 8
```

输出：

```text
training_data/g1_2s61f30_seedance_slide/
training_data/pair/identity_r2r/2s61f30_slide/
training_data/pair/blur_r2r/2s61f30_slide/
training_data/pair/h2r/2s61f30_slide/
```

## human2robot 2s/30fps

human2robot 原始配对数据使用固定首窗口，不做滑动窗口：

- MP4 源：`2s61f30_human2robot_v1`
- HDF5 源：`2s61f30_human2robot_hdf5_v1`

HDF5 源当前是更完整的数据口径。详见 [human2robot 数据集](../datasets/human2robot.md)。

