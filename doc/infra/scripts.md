# 脚本入口

当前顶层 `scripts/` 只保留现役入口：

```text
scripts/
├── flip_run.sh
├── flip_run_2.sh
├── codex_pre_tool_use_guard.py
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

