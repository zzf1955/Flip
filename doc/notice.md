# 注意事项

## 2026-04-30 — 1s clip 数量不能直接和 segment 总量对齐

`training_data/segment/<task>/` 是完整机器人 segment 集合；`h2r` 和
`r2h` 的 1s pair/cache 只会覆盖已经存在于
`training_data/seedance_direct/1s/<task>/manifest.jsonl` 的人体生成 clip。
当前 `seedance_direct/1s` 来自少量 `seedance_direct/4s` API 输出，不代表
`segment` 全量。`blur_r2r` 不涉及 human，应直接使用三个 canonical Task 的
全部 segment 数据。

`seedance_clip.py` 会把每条 4s human 视频切成 14 条 1s clip：`normal` 与
`hflip` 两种增强，各自按 0.5s stride 取 7 个窗口。因此 1 条 4s source
对应 14 条训练样本；`identity_r2r` 和 `blur_r2r` 则由 robot segment
直接切成 4 条 1s clip，没有 hflip，也不依赖 Seedance human 输出。
