# 实验计划

- LoRA 搜索
  - 维度/位置/三阶段
  - 得到 LoRA 最优位置
  - 结论：qkvo + ffn 是最优的

- LoRA 策略
  - step 1 + 2 合并 + step 3 单独训练：和三段式差不多
  - step 1、2、3 训练同一个 LoRA：效果更差

- LoRA Rank
  - 总体上 rank 越大效果越好
  - step3 是 rank 越小效果越好，因为 Mitty 仅靠 step3 的 LoRA 来记住信息，rank32 能看出明显效果，虽然 FID/FVD 比较起来提神很大，但是看起来效果都一般

- Mitty 的表现与性能
  - Mitty 在 LoRA Rank = 96 的情况下表现比较好，但是机械臂细节效果很差。
  - 机械臂在画面中的占比很低，导致计算全集的 FID/FVD 无法指示机械臂生成质量

- Local FID 与 Local FVD
  - 选取 bbox 之后进行计算
  - 但是 bbox 本身不准，而且是方形，依然会引入很多背景，所以效果依然不好

- WAN2.2 数据合成 LoRA 搜索
  - todo

- 加上WAN 2.2 之后测试
  - 测试加了数据之后会不会变好
  - todo

## next step

- ours
  - step 1:identity
    - rank 32
    - layout qkvo+ffn
    - prefix final_ours_step1_xxxx
  - step 2:r2r
    - lora merge: step1
    - rank 256
    - layout qkvo+ffn
    - prefix final_ours_step2_xxxx
  - step3:h2r
    - lora merge: step1 + step2
    - rank 96
    - layout qkvo+ffn
    - prefix final_ours_step3_xxxx

- mitty
  - h2r
    - rank 96
    - layout qkv

- wan2.2 数据合成
  - 当前能不能做到，不设置 ood task，全设置为 in task，训练数据用全部，然后按比例做 9:1 的划分，然后训练 WAN2.2？