# human2robot 数据集

`human2robot` 是外部原始配对数据集名。当前本地 legacy 路径仍是：

```text
data/h2r/v1/
├── data/<task>/episode_<id>.hdf5
└── video/<task>/episode_<id>/{robot_camera,human_camera}.mp4
```

不要把数据集本身再写成 `H2R` 或 `HumanAndRobot`。`h2r` 在 FLIP 内部表示方向
human -> robot。

## MP4 源

MP4 源使用 duration：

```text
2s61f30_human2robot_v1
```

当前统计：

- 210 个 episode。
- `push_box_two_v1/episode_5` 只有 8 帧，不满足 61 帧。
- 可用 209 条，22 个 task。
- runtime split 校验：`train=189`、`in_task_eval=20`、`ood=0`。

## HDF5 源

HDF5 源使用 duration：

```text
2s61f30_human2robot_hdf5_v1
```

当前统计：

- 1312 个 episode。
- `grab_pencil_v1/episode_50.hdf5` 只有 2 帧。
- `push_box_two_v1/episode_5.hdf5` 只有 8 帧。
- 可用 1310 条，32 个 task。
- runtime split 校验：`train=1179`、`in_task_eval=131`、`ood=0`。

## R2H pair/cache

robot -> human pair layout：

```text
training_data/pair/r2h/<duration>/<task>/
  video/pair_NNNN.mp4          # human target
  control_video/pair_NNNN.mp4  # robot input
  metadata.csv
  manifest.jsonl
  pair_order.jsonl
training_data/cache/vae/r2h/<duration>/<task>/pair_NNNN.pth
training_data/cache/t5/r2h/<duration>/
```

字段语义：

- `dataset_name=human2robot`
- `dataset_legacy_root=data/h2r/v1`
- `data_type=r2h`
- `input_role=robot`
- `target_role=human`

更多命令见 [训练基础设施完整记录](../pipeline/training_infra.md)。

