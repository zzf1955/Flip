# human2robot 数据集

`human2robot` 是外部原始配对数据集名。新的本地规范路径是：

```text
data/human2robot/
├── data/<task>/episode_<id>.hdf5
└── video/<task>/episode_<id>/{robot_camera,human_camera}.mp4
```

旧本地 legacy 路径仍保留为源数据，不应再作为新流程的默认路径：

```text
data/h2r/v1/
├── data/<task>/episode_<id>.hdf5
└── video/<task>/episode_<id>/{robot_camera,human_camera}.mp4
```

不要把数据集本身再写成 `H2R` 或 `HumanAndRobot`。`h2r` 在 FLIP 内部表示方向
human -> robot。

`data/human2robot` 由 `scripts/migrate_human2robot_dataset.py` 从 legacy
`data/h2r/v1` 物化生成。该脚本不修改 legacy 目录；迁移时会把 legacy HDF5
相机数组按 `BGR -> RGB` 写入新的 HDF5，使
`cam_data/{robot_camera,human_camera}` 在新目录中明确为 `uint8 RGB [T,H,W,3]`，
并写入 `color_space=RGB` / `channel_order=RGB` 属性。`video/` 下的 MP4 从新
HDF5 的 RGB 帧重新编码，覆盖全部 HDF5 episode，不再依赖 legacy `video/`
中较小的旁路子集。

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
- `dataset_root=data/human2robot`
- `dataset_legacy_root=data/h2r/v1`
- `data_type=r2h`
- `input_role=robot`
- `target_role=human`

当前 forward sliding-window 版本使用 duration：

```text
2s61f30_human2robot_r2h_forward_v1
```

该版本从 `training_data/slice/human2robot_2s61f30_forward_v1` hardlink 生成
pair。正向 prompt 为 `把机器人视频编辑为人类视频`；T5 负向 prompt 使用简化文本
`低质量，模糊，过曝，变形，错误手部，错误脸部，文字，水印，杂乱背景`。

该版本的训练划分写在 pair 根目录：

```text
training_data/pair/r2h/2s61f30_human2robot_r2h_forward_v1/train.jsonl
training_data/pair/r2h/2s61f30_human2robot_r2h_forward_v1/eval.jsonl
training_data/pair/r2h/2s61f30_human2robot_r2h_forward_v1/test.jsonl
```

训练入口使用 preset `r2h_human2robot_2s61f30_forward`，默认
`split_source=explicit`，只从 `train.jsonl` 抽训练样本，online eval 从
`eval.jsonl` 按 `eval_role=in_task/ood` 拆成 4 个 in-task 与 4 个 OOD。
小数据量 ablation 的 `--train-size` 会在 train split 的 task 间近似均匀分配；
完整测试保留 `test.jsonl` 全量样本。

更多命令见 [训练基础设施完整记录](../pipeline/training_infra.md)。
