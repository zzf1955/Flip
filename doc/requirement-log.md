# 需求日志

## 2026-04-16

**用户原始需求：**
> 我需要统一定位 data，不希望 worktree copy 工作区的 data，然后我需要在规范中写上所有的 data 读取写绝对路径，从 main 获取，有没有什么好方法？

讨论要点：
- `data/` 95GB 已 gitignore，worktree 不会 copy，但现有 `BASE_DIR/data/...` 在 worktree 下指向空目录
- 扩大到所有大目录：`data/`、`weights/`、`paper/`、`ref-cosmos-transfer1/`、`ref-cosmos-transfer2.5/`（13GB）、`ProPainter/` 共 ~110GB 全部共享指向 main
- `output/` 保持 per-worktree 隔离，避免实验产物互相覆盖
- `data/output/` 566MB 确认是旧 pipeline 残留，代码无引用，任务中一并删除
- main 路径锁定 `/disk_n/zzf/flip/`，保留 `FLIP_MAIN_ROOT` 环境变量逃生口

**创建的任务：**
- [001] 统一 data/权重/参考大目录定位，worktree 共享 main，output 隔离

---

**用户原始需求：**
> 人体渲染基本跑通了，需要做深度图提取 + 人体区域深度模糊 + 重绘 mask + 按深度+mask 重新生成视频。先试 Cosmos Transfer 2.5，不行换 Wan 2.1。

讨论要点：
- 先完成了 cosmos_prepare.py（composite + depth + mask 生成）和 cosmos_regen.py（Cosmos 推理包装）
- Cosmos Transfer 2.5 在共享机器上因 CPU RAM OOM（其他用户占 58GB）反复崩溃
- 双卡勉强跑通但 guided generation 效果不理想，mask 外背景也被影响
- 决定改用 Wan 2.1 VACE，单卡 4090 可跑，原生支持 depth + mask

**创建的任务：**
- [002] Wan 2.1 VACE depth+mask 人体重绘测试

---

## 2026-04-17

**用户原始需求：**
> 在代码中自己写一个同样的 baseline 训练代码，需要支持 eval loss 等东西，数据先只输出 log，训练输出专门写到 training_data/log 中，按照日期标，里面存 ckpt、eval 的视频、日志文件

讨论要点：
- DiffSynth-Studio 训练框架不输出 train loss、无 train/eval 分割、无日志系统
- 复用 DiffSynth 的 WanTrainingModule（模型加载 + LoRA + forward），自写训练循环
- 单卡先跑通，不用 accelerate DDP
- 输出目录 `training_data/log/<date>/` 下按 ckpt、eval、train.log 组织

**创建的任务：**
- [003] 自写 Wan 2.1 FunControl LoRA 训练脚本

---

**用户原始需求：**
> 现在看一下整个视频微调的算法，现在要进行修改了。思路是模仿 Mitty（2512.Mitty.pdf），将序列拼接，加上 LoRA 微调。模型原打算用普通的 WAN 2.1。

讨论要点：
- Mitty 核心：human 视频 latent 保持 clean、robot 视频 latent 加噪，沿 temporal dim 拼接 → full self-attention，loss 只在 robot 段
- Wan 2.1 能做但要改 RoPE + modality embedding + per-frame timestep（工作量中等）
- 调研发现 **Wan 2.2 TI2V-5B 架构原生支持** Mitty 所需的 partial-noising：`seperated_timestep=True` + `fuse_vae_embedding_in_latents=True`，timestep 已经是 per-patch 构造（见 `diffsynth/pipelines/wan_video.py:1376-1380`）
- Mitty 论文主实验也是 TI2V-5B dense（非 MoE），论文效果对齐我们需求
- 5B 参数 FP8 约 5GB，4090 单卡宽松；Wan 2.2 14B 是 MoE 两个 branch，调试成本高，先不碰
- Pipeline 完全重写，不复用 FunControl 的 ref_conv 路径；现有 `train_lora.py` 保留作为 baseline

**创建的任务：**
- [004] Wan 2.2 TI2V-5B 环境准备 + seperated_timestep 机制验证
- [005] Mitty-style in-context 训练 pipeline (Wan 2.2 TI2V-5B)
- [006] Mitty 方案推理 + 与 FunControl baseline 定量对比

---

**用户原始需求：**
> task 内 1 条 eval 数据，剩下的训练；OOD task 为 pick up pillow，不参与训练；每次 eval 输出两种 eval 视频和 GT；bs=4 训练。

讨论要点：
- OOD 只含 Inspire_Pickup_Pillow_MainCamOnly（Brainco pillow 不参与）
- 非 OOD 的 5 个 task 各抽 1 条做 in-task eval，共 5 条；pillow 8 条全 ood_eval；剩余 143 条 train
- bs=4 用 torchrun 4 卡 DDP × bs=1 实现（复用 train_lora.py 的 DDP 基础设施，不做单卡 grad_accum）
- 数据组织：`training_data/pair/1s/{train,eval,ood_eval}/` 三个独立子目录，各自编号 pair_NNNN 和 metadata.csv；`source_map.json` 记录反查
- 改 `src/pipeline/make_pair.py`：加 `--ood-tasks`、`--per-task-eval`、`--split-seed`、`--clean` 参数，按 task 分 split
- T005 同步更新：`mitty_cache.py` 按 split 跑三次；`train_mitty.py` 三个 cache 目录，eval 视频分 in_task/ood 子目录

**完成改动（非新建 task，直接改 make_pair + 更新 T005 文档）：**
- `src/pipeline/make_pair.py`：split-aware 重写
- `training_data/pair/1s/{train,eval,ood_eval}/`：重新生成，共 156 条（train 143 / eval 5 / ood_eval 8）
- `doc/tasks/pending/005.md`：CLI 参数和 eval 视频目录结构更新

---

## 2026-04-18

**用户原始需求：**
> 现在我要加强手部的准确率，做法是：1. 根据手部 Mesh overlay，框出手部的大致位置；2. 调高这部分 patch 的 loss 权重。详细看一下当前的数据 pipeline 改如何实现，patch 估算和训练 pipeline 分开，因为我可能会替换 patch pipeline。然后还中间结果可调试，比如你把视频中的 patch 也做一个 overlay。

讨论要点：
- 利用 FK 投影手部 mesh → 2D bbox → latent 空间 (30×40) 权重图
- patch 生成和训练完全解耦：hand_patch.py 独立产出 .pth 权重文件，train_mitty.py 通过 --patch-dir 可选加载
- MittyFlowMatchLoss 增加 patch_weights 分支，向后兼容
- debug overlay 可视化：手部 mesh + latent grid + 高亮 cell

**创建的任务：**
- [007] 手部 patch 加权 loss：FK mesh 投影 → latent 权重图 → 训练加权

---

**用户原始需求：**
> 先实现映射，这个映射直接 copy inspire 手的角度即可，可能有范围映射，然后手指关节均匀分这个角度。当前只做 inspire 手。

**创建的任务：**
- [008] Inspire hand_state → SMPLH 手指姿态实时映射

---

**用户原始需求：**
> 重建指标是不是主要看 FID? 如何对比两个视频的 FID? 你看一下当前的训练 pipeline，现在训练的 log 中有每次 eval 的视频，其中 Control, gen, ground Truth 都有，给一个计算重建指标的代码，多计算几个指标。

讨论要点：
- FID 需要大量样本才稳定，配对视频编辑任务更适合逐帧指标（LPIPS/SSIM/PSNR）
- 环境无 lpips/torchmetrics 包，用 VGG16/InceptionV3 自实现 LPIPS/FID/FVD
- 独立 CLI 工具，不集成到训练循环

**创建的任务：**
- [009] 训练 eval 视频重建指标计算工具

---

## 2026-04-19

**用户原始需求：**
> 重构数据 pipeline。加一条新线：inpaint + 人体 Mesh overlay → Seedance 合成增强数据（seedance_advance）。加手部 patch 数据生成功能。make_pair 整合 hand patch，切片时不输出中间数据，直接输出 control、gt、hand patch。

讨论要点：
- seedance_advance 复用 seedance_gen API 函数，overlay 作为 Seedance 输入
- 手部 patch 拆为两阶段：4s segment 级 per-frame bbox（parquet）+ make_pair 内联生成 latent weight map
- Seedance prompt 采用 CG→真实增强风格

**创建的任务：**
- [010] 数据 Pipeline 重构: seedance_advance + 手部 patch + make_pair 整合

---

**用户原始需求：**
> 训练 IO 瓶颈严重（.pth 文件 55MB 但训练只用 9MB，90% 是 PIL 帧浪费）。要求：1. 加上 W&B 数据上传 2. eval 集合缩小到 50 3. 优化 PIL 剥离 + prefetch 4. eval 视频和 eval 频率对齐，每次输出 4 条

**创建的任务：**
- [011] 训练 pipeline IO 优化 + eval 对齐 + W&B 完善

---

## 2026-04-20

**用户原始需求：**
> 恒等映射训练效果不好，猜测是算法瓶颈。分析 Wan 2.2 TI2V-5B 的 I2V 机制后，提出方案 A：Rectified Flow——把初始噪声换成原视频（source latent），去掉 Mitty concat，作为 Mitty 的对比实验。双卡 DDP。

**创建的任务：**
- [012] Rectified Flow Route A 训练代码

---

**用户原始需求：**
> 当前 DiffSynth 模型加载代码太慢，替换成自己的代码（先只改训练，T5 不动）。

讨论要点：
- DiffSynth `WanVideoPipeline.from_pretrained` 为通用 plug-and-play 设计：`hash_model_file` 扫 metadata、`DiskMap` 建索引、`AutoWrappedLinear` 包每个 Linear、25 个 PipelineUnit 实例化，对固定模型+固定显存布局的训练全是 overhead
- 绕开 `from_pretrained` 写直给式 loader：固定 TI2V-5B 配置，跳过 hash/DiskMap/VRAM 管理
- 范围限定仅 `train_mitty.py::build_pipe`；T5 路径（`mitty_cache.py`）、LoRA 训练（`train_lora.py`）、推理脚本不动

**创建的任务：**
- [013] Wan2.2 TI2V-5B 直给式 loader 替换 DiffSynth from_pretrained（仅训练路径）
