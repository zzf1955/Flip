# Cache

训练前需要把文本和视频预先编码成 cache：

- T5 cache：prompt embedding，共享到同一 `data_type/duration`。
- VAE cache：每个 pair 的 input / target latent。

## 目录

```text
training_data/cache/
├── t5/<data_type>/<duration>/
│   ├── prompt_<hash>.pth
│   └── negative.pth
└── vae/<data_type>/<duration>/<task>/
    ├── pair_NNNN.pth
    └── manifest.jsonl
```

## 生成命令

```bash
scripts/flip_run.sh mitty_cache --cuda 0 -- \
  --pair-dir training_data/pair/<data_type>/<duration>/<task> \
  --output training_data/cache/vae/<data_type>/<duration>/<task> \
  --t5-cache-dir training_data/cache/t5/<data_type>/<duration> \
  --device cuda:0
```

## VAE cache 字段

- `human_latent`：input / control latent。
- `robot_latent`：target latent。
- `prompt`：用于匹配 T5 cache。
- `data_type`、`duration`、`robot_task`、`source_id`：runtime split 和溯源字段。

字段名沿用 Mitty 历史命名。即使 `data_type=r2h`，`human_latent` 也表示 condition/input，
`robot_latent` 表示 denoise target；角色语义以 manifest 中的 `input_role` /
`target_role` 为准。

