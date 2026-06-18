# FLIP 文档索引

当前文档按“怎么做事”组织，而不是按早期 `step_2/3/4/5` 阶段组织。

## 当前主线

- [Pipeline 总览](pipeline/README.md)
- [数据集总览](datasets/README.md)
- [模型与训练骨干](models/README.md)
- [工程入口与环境](infra/README.md)

## Pipeline

- [主线流程](pipeline/overview.md)
- [数据切片](pipeline/data_slicing.md)
- [数据合成](pipeline/data_synthesis.md)
- [配对数据构造](pipeline/pair_data.md)
- [Seedance](pipeline/seedance.md)
- [WAN R2H 自合成](pipeline/wan_r2h.md)
- [外观模糊数据构造](pipeline/blur_r2r.md)
- [Cache](pipeline/cache.md)
- [三阶段训练](pipeline/three_stage_training.md)
- [SAM2/SAM3 mask](pipeline/sam2_sam3_masks.md)
- [训练基础设施完整记录](pipeline/training_infra.md)

## 数据集

- [G1 数据集](datasets/g1.md)
- [G1 相机](datasets/g1_camera.md)
- [G1 手部映射](datasets/g1_hand_mapping.md)
- [human2robot 数据集](datasets/human2robot.md)
- [命名约定](datasets/naming.md)
- [数据布局](datasets/layouts.md)

## 模型

- [Wan2.2 + Mitty](models/wan22_mitty.md)
- [Wan2.2 DiT 架构](models/wan22_architecture.md)
- [Diffusion Policy](models/diffusion_policy.md)
- [WAM](models/wam.md)

## 工程

- [代码结构](infra/code_structure.md)
- [脚本入口](infra/scripts.md)
- [运行环境](infra/runtime.md)
- [W&B](infra/wandb.md)
- [注意事项](infra/notices.md)

## 任务与历史

- [任务目录](tasks/)
- [需求日志](requirement-log.md)
- [归档文档](archive/)

