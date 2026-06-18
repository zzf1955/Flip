# 配对数据构造

训练数据统一发布为 pair layout：

```text
training_data/pair/<data_type>/<duration>/<task>/
├── video/pair_NNNN.mp4
├── control_video/pair_NNNN.mp4
├── metadata.csv
├── manifest.jsonl
└── pair_order.jsonl
```

字段语义：

- `video/`：训练 target。
- `control_video/`：训练 input / condition。
- `manifest.jsonl`：逐样本来源、帧窗口、数据集、task 和角色信息。
- `pair_order.jsonl`：该 task 下固定乱序表，供 runtime split 稳定复用。

## 主要 data_type

- `identity_r2r`：清晰 robot -> 同一清晰 robot。
- `blur_r2r`：模糊 robot -> 清晰 robot。
- `h2r`：human -> robot。
- `r2h`：robot -> human。

## 当前重要 duration

- `1s`：旧 1 秒数据。
- `2s61f30`：G1 2 秒、61 帧、30fps。
- `2s61f30_slide`：G1 Seedance 2 秒滑窗。
- `2s61f30_human2robot_v1`：human2robot MP4 源 robot -> human。
- `2s61f30_human2robot_hdf5_v1`：human2robot HDF5 源 robot -> human。

命名边界详见 [命名约定](../datasets/naming.md)。

