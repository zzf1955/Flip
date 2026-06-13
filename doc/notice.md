# 注意事项

## 2026-06-13 — H2R Seedance 直接 robot2human 暂不可作为训练数据来源

`src.pipeline.h2r_seedance_edit` 已跑通 H2R 三段 robot-camera smoke、Seedance API 调用和
`488x256 @ 30fps` 后处理，但两轮 prompt 结果都不足以进入训练数据主线。主要失败模式是
Seedance 在机械夹爪附近额外生成一只人手，或不能稳定擦除原有黑色两指夹爪/白色机械外壳，
而不是把夹爪在原位置替换成人手。

后续不要直接把 `tmp/h2r_seedance_edit_smoke/` 或 `tmp/h2r_seedance_edit_prompt_v2/`
作为 H2R human/robot 配对数据来源。若继续探索，应优先考虑显式 mask / inpaint / 局部编辑
约束，或回到可控的分割、渲染、训练式 r2h 路线。

同日新增的 `src.pipeline.h2r_seedance_sam3_edit` 已把“显式 mask”路线跑成 smoke：
先用 H2R SAM3/SAM3.1 `robot arm` mask 标出目标区域，再让 Seedance 只替换 marker
指示区域。最早的全臂红色 baseline 产物位于 `tmp/h2r_seedance_sam3_red_edit/`，
仍需人工复核 `_review/` 并排视频后才能判断是否可继续扩大；在人工确认前也不要把该目录
直接当作训练配对数据来源。

同日后续实验表明，全臂红色 marker 仍不能稳定解决“夹爪旁边新增手”的问题。当前较好的
Seedance 引导方式是用 SAM3 `robot arm` mask 和暗像素过滤得到黑色夹爪目标，再用黄色
`bbox` 或紫色 `fill` marker 标注局部区域；量化评估入口为
`src.pipeline.h2r_seedance_sam3_eval`。现有 17 次成功 API 结果中，
`tmp/h2r_seedance_sam3_exp04_dark_yellow_bbox/` 的平均 target coverage / IoU 最高，
但 `hand_on_target_ratio` 仍偏低，说明可能还有额外人手或分割误检。该路线也暂时只能作为
Seedance prompt/mask 调试，不要直接进入 Wan 训练。

后续追加的 `exp06_dark_skin_fill_one` 和 `exp07_dark_yellow_bbox_strict_prompt` 均为负例：
肤色预填没有让 Seedance 稳定生成手，过度强调“只在方框内编辑/输出不要保留方框”的黄框
prompt 反而让 SAM3.1 hand coverage 接近 0。继续探索时应回到 `exp04/exp05` 的
yellow/magenta bbox 风格，少量改 bbox 尺寸或自然语言动作提示，不要大幅改变输入语义。

同日继续沿黄/紫方框方向试了两个单条 `grab_cup_v1` 变体：加粗放大黄框
`exp08_yellow_bbox_big_cup` 能生成清楚人手，但人手位置/轨迹偏离原夹爪目标，
SAM3.1 target coverage 只有 `0.006`；紫色内框 + 黄色外框 `exp09_dual_bbox_cup`
基本没有替换机器人，hand eval 为 0。因此当前不要把“更大框”或“双色叠框”扩展成三条或
训练数据生产；最稳的框类输入仍是单色 `exp04_dark_yellow_bbox` /
`exp05_dark_magenta_bbox`，但仍需人工复核，不可直接进入 Wan 训练。

最终收口时，ROI 放大路线只保留 dry-run 产物用于记录：
`tmp/h2r_seedance_sam3_roi_yellow_bbox_dryrun/` 与
`tmp/h2r_seedance_sam3_roi_input_contact_sheet.jpg`。该输入在部分帧会框到白色机械臂或整块
crop，未调用 Seedance API，未纳入当前最优方案。后续复现或交接时以
`exp04_dark_yellow_bbox` 为当前最优基线。

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
