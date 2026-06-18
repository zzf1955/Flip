# Pipeline 文档

当前 pipeline 主线分为：

1. 数据切片：从 G1 segment 或 human2robot episode 生成固定时长视频样本。
2. 数据合成：通过 Seedance 或训练好的 R2H 模型生成 human-side 数据。
3. 配对数据构造：发布为统一 pair layout，并按 `data_type` / `duration` 组织。
4. 外观模糊数据构造：生成 `blur_r2r`，用于三阶段训练的第二阶段。
5. Cache：为 Wan/Mitty 训练预计算 VAE latent 和 T5 embedding。
6. 三阶段训练：identity、blur、h2r/r2h 等阶段化 LoRA 训练。

## 推荐阅读顺序

- [主线流程](overview.md)
- [数据切片](data_slicing.md)
- [配对数据构造](pair_data.md)
- [Cache](cache.md)
- [三阶段训练](three_stage_training.md)

## 详细记录

- [训练基础设施完整记录](training_infra.md) 保留了当前路径、命令、实验结果和旧路线说明。
- [Seedance](seedance.md) 保留 API 调用、H2R smoke 和 marker-mask 评估记录。
- [SAM3 G1 分割](sam3_g1_segmentation.md) 与 [SAM3 H2R 分割](sam3_h2r_segmentation.md)
  是 SAM3/SAM3.1 prompt、显存和质量结论的原始记录。

