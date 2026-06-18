# scripts/archive

这里存放已经不作为主入口维护的旧脚本。新开发和日常运行优先使用 `src/` 下的模块，或
`scripts/` 顶层保留的现役 launcher / 批处理脚本。

| 子目录 | 内容 |
|--------|------|
| `camera_calibration/` | 旧相机模型、mask/keypoint calibration、焦距、畸变、外参验证脚本 |
| `retarget_ik/` | 旧 SMPLH IK、retarget copy、retarget debug / video 脚本 |
| `render_debug/` | 旧 mesh/render overlay/debug/svg2gif 脚本 |
| `segmentation_inpaint/` | 旧 SAM2 / inpaint pipeline 与批处理脚本 |
| `dataset_utils/` | 旧数据检查、抽帧、降采样、一次性 cache/backfill/reverse 数据脚本 |
| `training_launchers/` | 旧 H2R / final / SAM3 stage shell 训练启动器；当前训练优先用 `scripts/flip_run.sh train` |
| `eval_analysis/` | 旧 final eval 补跑和背景 Patch FID 辅助脚本 |
| `migration/` | 一次性旧 task/cache layout 迁移脚本 |
| `smoke/` | one-off smoke / proof-of-concept 脚本 |
