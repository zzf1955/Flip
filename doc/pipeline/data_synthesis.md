# 数据合成

当前数据合成主要有两条路线：

1. Seedance：直接或 marker-mask 引导地把 robot 视频编辑成 human 视频。
2. WAN R2H：使用训练好的 robot -> human Mitty LoRA 自合成 human-side pair。

## Seedance

Seedance 主要用于生成 human-side 数据或做 human2robot smoke：

- G1 Seedance 4s 输出可进一步切成 1s 或 2s 训练样本。
- human2robot 直接 robot -> human hand 效果当前不稳定，不作为正式训练数据来源。
- human2robot + SAM3 marker-mask 路线用于 prompt / mask / API 调试。

详见 [Seedance](seedance.md)。

## WAN R2H 自合成

`src.pipeline.r2h_synthesize` 使用已训练的 R2H Mitty LoRA，从 G1 robot segment 生成
synthetic human，并发布为新的 h2r `_syn` pair。

详见 [WAN R2H 自合成](wan_r2h.md)。

