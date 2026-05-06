# 实验计划

- LoRA 搜索
  - 维度/位置/三阶段
  - 得到 LoRA 最优位置

- WAN2.2 数据合成 LoRA 搜索

- 加上WAN 2.2 之后测试
  - 测试加了数据之后会不会变好

- identity：FFN
- 
- h2r：无明显差异


现在整理一下训练的搜索脚本。 需要支持以下参数：
1. 合并哪些 LoRA（自动检测位置和 rank）
2. 使用哪些数据（train size，Task 分配按照默认即可，主要是数据量，数据路径按照训练类型来找就行）
3. 在那些层加 LoRA，维度是多少
4. 使用哪些 cuda

然后 LoRA layout x LoRA rank 展开后，在指定的 GPU 上顺序分配。
实验的 Log 名称和 WAN db 的 log 名称要加时间/日期
然后 log 目录的名称中，要写 rank layout，命名为 self_qkvo_cross_qkvo_ffn 这样的格式，不要全部展开写
然后 LoRA layout，支持仅在 qkv 加 LoRA。不在 o 上加 LoRA。





1. 整理训练脚本
2. 低 rank 测试
3. Mitty 去掉 o 的 LoRA
4. 找其他 Baseline
5. 增加数据，测试数据 scale
6. 整理实验计划
  1. main result
  2. LoRA layout
  3. 数据量

- identity
  - ffn + qkvo
  - qkvo

- blur_r2r
  - ffn + qkvo

- h2r
  - ffn + qkvo

- Baseline Mitty
  - LoRA rank 96
  - layout：QKV (self attention)
  - task：h2r
  - training size 400

- ours
  - step1
    - LoRA rank 96
    - layout QKVO
    - task identity_r2r
    - training size 10000
  - step2
    - LoRA rank 96
    - layout QKVO+FFN
    - task blur_r2r
    - training size 10000
    - merge LoRA: flip/training_data/log/archive.5.2 high rank search/Mitty-identity_r2r_1s-10000d_r32_qkvoffn0ffn2_1000s_0428_195227
  - step3
    - LoRA rank 96
    - layout QKVO+FFN
    - task h2r
    - training size 400
    - merge LoRA: 
      - flip/training_data/log/archive.5.2 high rank search/Mitty-identity_r2r_1s-10000d_r32_qkvoffn0ffn2_1000s_0428_195227
      - flip/training_data/log/Mitty-h2r_1s-400d_r96_self_qkv_cross_qkv_1000s_0503_154803

scripts/train_lora_grid.py \
  --cuda 1 \
  --task-name h2r_1s \
  --train-size 400 \
  --layouts self_qkv_cross_qkv \
  --ranks 96 \
  --name-prefix baseline_mitty_h2r


cuda 0
cuda 1
cuda 2
cuda 3
