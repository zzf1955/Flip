# WAN R2H 自合成

WAN R2H 自合成使用训练好的 robot -> human Mitty LoRA 生成 synthetic human，并把结果写成
h2r `_syn` pair，供后续混合训练或误差分析使用。

## 单任务生成

入口：

```bash
scripts/flip_run.sh r2h_synthesize --cuda 0 -- \
  --source-task Inspire_Collect_Clothes_MainCamOnly \
  --duration 1s \
  --run <r2h_run_name_or_path> \
  --checkpoint latest \
  --num-samples 1000 \
  --resume-existing
```

输出：

```text
training_data/pair/h2r/<duration>/<task>_syn/
├── video/
├── control_video/
├── metadata.csv
├── manifest.jsonl
└── pair_order.jsonl
```

## 多卡队列

使用 `scripts/run_r2h_synthesize_queue.py` 对多个 source task 做队列调度。子进程仍通过
`scripts/flip_run.sh r2h_synthesize` 设置环境。

## 误差分析

`scripts/run_syn_error_analysis.py` 只写 `output/syn_error_analysis/`，不进入正式 pair。
它用于检查指定 episode / clip 上的 R2H 生成质量。

完整参数和历史结果见 [训练基础设施完整记录](training_infra.md)。

