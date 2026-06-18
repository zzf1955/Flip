# 数据布局

## 原始数据

```text
data/
├── unitree_G1_WBT/
└── h2r/v1/
```

## 训练 pair

```text
training_data/pair/<data_type>/<duration>/<task>/
├── video/
├── control_video/
├── metadata.csv
├── manifest.jsonl
└── pair_order.jsonl
```

## Cache

```text
training_data/cache/
├── t5/<data_type>/<duration>/
└── vae/<data_type>/<duration>/<task>/
```

## Segment / slice / mask

```text
training_data/segment/
training_data/slice/
training_data/sam2_mask/
training_data/h2r_sam3_mask/
training_data/g1_sam3_mask/
```

## Logs

```text
training_data/log/
```

训练 run、eval video、data split、W&B 相关日志默认写到这里。

