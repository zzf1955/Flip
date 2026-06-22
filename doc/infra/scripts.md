# 脚本入口

当前顶层 `scripts/` 只保留现役入口：

```text
scripts/
├── flip_run.sh
├── flip_run_2.sh
├── codex_pre_tool_use_guard.py
├── download_unitree_wbt.py
├── migrate_human2robot_dataset.py
├── run_r2h_synthesize_queue.py
├── run_syn_error_analysis.py
├── smoke_test.py
├── smoke_test_gpu.py
├── smoke_test_light.py
├── train_lora_grid.py
└── train_three_stage_single_lora.py
```

详细代码结构见 [代码结构](code_structure.md)。

旧 camera、IK、render、segmentation/inpaint、dataset utility、eval helper、临时训练
shell 和 one-off smoke 均已移到 `scripts/archive/`。

旧 `src/pipeline` 实验入口已移到 `src/pipeline/archive/`。

`download_unitree_wbt.py` 是当前 Unitree WBT 数据下载入口；默认下载 Brainco 手和
两组 Inspire 手数据到 `data/unitree_G1_WBT/`，支持并发、断点跳过、后台运行和
`inspire_dex5` 远端目录解包。推荐通过 `scripts/flip_run.sh unitree_wbt_download -- ...`
启动。

`migrate_human2robot_dataset.py` 是 human2robot legacy 数据迁移入口；从
`data/h2r/v1` 读取 HDF5，写入新的 `data/human2robot`，把 legacy 相机数组按
`BGR -> RGB` 规范化，并从新 HDF5 的 RGB 相机帧重新编码完整 `video/` 目录。
该脚本不修改 legacy `data/h2r/v1`。
