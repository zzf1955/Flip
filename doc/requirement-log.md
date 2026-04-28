# 需求日志

## 2026-04-26

**用户原始需求：**
> 写一个综合评估训练好的模型的脚本. 我需要在比较大的数据集上跑一些 metrics. 使用 Mitty-transfer-124d_r128_2000s_0425_1456 和 Mitty-transfer2LoRA-124d_r128_2000s_0425_1425 两个模型,在 flip/training_data/pair/1s 中的视频上测试现在的 FID,psnr 之类的指标. 先用模型生成视频(32+32),再计算指标. 写新的脚本进行测试

**完成改动：**
- 新增 `src/pipeline/evaluate_mitty_models.py`：默认评估两个指定 Mitty run 的 `step-2000.safetensors`，在 `eval` 与 `ood_eval` 各取 32 条 1s pair cache，先生成 `gen` 视频，并从 `training_data/pair/1s` 复制原始 `gt/ctrl` 视频，再计算 PSNR、SSIM、LPIPS、FID、FVD。
- 更新 `src/tools/eval_metrics.py`：视频配对解析支持超过 99 个样本的可变长度编号，便于大数据集评估。
- 更新 `scripts/flip_run.sh`：新增 `eval_mitty` 统一 GPU 入口。
- 更新 `doc/step_5_training_infra.md`：记录离线综合评估入口、默认模型、`32+32` 数据口径、输出路径和复算指标方式。

---

## 2026-04-24

**用户原始需求：**
> 当前 Codex sandbox 无法访问 `nvidia-smi` / GPU，需要说明 sandbox 机制，并记录哪些命令应该越权执行。

讨论要点：
- 当前项目切换为 `danger-full-access` + `approval_policy=never`，让 GPU 查询、CUDA 推理、训练、多卡 DDP 直接访问宿主 `/dev/nvidia*`。
- `sandbox_mode` 和 `approval_policy` 必须放在 `~/.codex/config.toml` 顶层；`[projects."/disk_n/zzf/flip"]` 只保留 `trust_level = "trusted"`。
- `writable_roots` 已包含项目目录、`/tmp`、HuggingFace cache 和 pip cache，作为回退到 `workspace-write` 时的写入白名单保留。
- Linux `workspace-write` sandbox 使用 bwrap 风格隔离，`/dev` 是最小设备目录，不透传 `/dev/nvidia*`；如回退到该模式，GPU 命令仍需要越权或改回 full access。
- full access 下依赖 Codex `PreToolUse` hook 做高危 Bash 命令的最佳努力拦截；hook matcher 已放宽为兼容 shell/Bash/exec 工具名。

**完成改动（文档更新）：**
- `doc/codex_migration.md`：更新当前 Codex 配置、Linux sandbox 现状、GPU 限制和越权执行准则。
- `AGENTS.md`：补充 full access + Codex hooks 的 GPU/训练安全边界，以及统一 GPU 命令入口。
- `scripts/flip_run.sh`：新增统一 GPU/训练入口，支持按子命令保存 Codex prefix rule。
- `doc/scripts_inventory.md`：补充 GPU 命令统一入口用法。
- `scripts/codex_pre_tool_use_guard.py`：新增 Codex Bash `PreToolUse` hook 护栏，阻止 `sudo`、危险删除和危险 Git 操作等。
- `~/.codex/config.toml`：切换为 `danger-full-access` + `approval_policy=never` 并启用 `codex_hooks`；备份为 `/home/leadtek/.codex/config.toml.bak-fullaccess-hooks-20260424`。
- `~/.codex/hooks.json`：配置全局 `PreToolUse` hook 指向项目护栏脚本，matcher 使用 `Bash|exec_command|shell|exec_command`；已验证 `sudo -n true` 会被 hook 拦截。

---

## 2026-04-23

**用户原始需求：**
> 训练过程中加上 FID 之类的指标，在 eval 的时候一起计算。推荐算哪些指标，是否应该跑多一些视频。

**创建的任务：**
- [030] 训练在线评估指标集成（PSNR/SSIM/LPIPS/CLIP Score）

**用户原始需求：**
> FFN 之后下一阶段是 attention 弄重新加 LoRA, 然后用 124 样本跑外观替换

**创建的任务：**
- [029] 多 LoRA 合并 + Attention LoRA 外观替换训练

**用户原始需求：**
> 当前 training 的代码中, eval 的时候只有单卡. 能不能拆到所有卡上

**创建的任务：**
- [028] 分布式 Eval：多卡并行评估

## 2026-04-22

**用户原始需求：**
> 看一下当前 Mitty appearance 的数据构造方式。目前是直接使用机器人的 Mesh 来进行模糊，不是很准。能不能复用 inpaint pipeline 中 SAM2 的 Mask，得到高质量的机器人 patch。需要并行加速优化。

**创建的任务：**
- [026] SAM2 mask 预计算 + robot_patch 集成

**用户原始需求：**
> 做一下流水线优化，然后输出 pair 视频的时候连 SAM2 原始的 Mask 视频一起输出，输出到 train/mask eval/mask 中，方便检查中间结果。

**创建的任务：**
- [027] sam2_precompute 流水线优化 + mask 视频输出

**用户原始需求：**
> 在 LoRA 恒等映射后面加一个阶段的训练。在 FFN 层加 LoRA，重建 1s_patch 中模糊的视频。前一阶段的 identity LoRA 保持不变加到模型上，FFN 加新 LoRA。

**创建的任务：**
- [023] FFN LoRA 训练：合并 identity LoRA + FFN 层重建

**用户原始需求：**
> 看一下当前恒等映射/FFN 外观补全的训练代码，其中 eval 没有输出 GT 和 Control，加上

**创建的任务：**
- [025] eval 输出 GT 和 Control 视频（latent fallback）

**用户原始需求：**
> 统一本地 log 目录和 W&B run name，保持一致，带日期、任务名、max step、lora rank、数据规模。规范 W&B tag 记录详细信息。尽可能不需要手动指定命名。

**创建的任务：**
- [024] 统一 run name 与 W&B 命名规范

## 2026-04-21

**用户原始需求：**
> DiT safetensors 是 FP32 存的，每次加载都要重新转 bf16，而且多卡同时加载很慢。改成预转 bf16 存盘 + 单卡读取后 NCCL broadcast 到其他 rank。

**创建的任务：**
- [020] DiT 加载重构：bf16 预存 + DDP broadcast

**用户原始需求：**
> 把手部 patch 换成机器人全身的 patch。patch 阶段产出机器人视频、机器人 patch 视频、patch.pth。用于外观学习。patch 视频支持三种方式：patch 区域模糊、patch 区域加噪声、patch 区域均值填充。

**创建的任务：**
- [022] 全身机器人 Patch 数据生成 Pipeline

**用户原始需求：**
> 看一下当前的 cache 管理。应该分为：数据的 VAE 缓存、T5 缓存，然后每个训练和 eval 的样本分开。需要统一 cache 管理，放到 training data 文件下，并规范这部分数据的命名。

**创建的任务：**
- [021] 统一 cache 管理：分离 VAE/T5，迁移至 training_data/cache/

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

---

## 2026-04-20

**用户原始需求：**
> 看一下当前的 wb 和同级口径, dataloader 之类的, 我需要全部按照 step 来控制训练, 而不是 epoch. 然后 wb 的实验全部放到 Flip 项目中, run 命名前缀也需要加上, 不然全是日期命名。wb 的 tag 和实验命名统一管理, 在其中记录参数之类的, 详细一些。

讨论要点：
- 三个训练脚本 (train_lora/train_mitty/train_rf) 均以 `--epochs × --repeat` 间接控制步数，改为 `--max-steps` 直接控制
- dataloader 从 epoch 循环改为 `infinite_file_batches()` 无限迭代器，数据自动循环洗牌
- W&B project 默认改为 `"Flip"`，run 命名格式 `{prefix}-r{rank}-lr{lr}-{timestamp}`
- W&B tag 自动从 args 提取关键超参（lora_rank, lr, batch_size, warmup, max_steps 等）
- 三个函数统一放 `train_utils.py`：`infinite_file_batches`, `build_run_name`, `build_wandb_tags`

**创建的任务：**
- [014] 训练循环 step 化 + W&B 统一管理

---

**用户原始需求：**
> 看一下目前的训练 pipeline,当前应该有几个不同的组件, 主干有 Mitty 和直接 Rectifie 的做法两种,然后 loss 有 hand patch 增强. 能不能把训练 pipeline 整合一下,方便我做消融实验,这几个可以选择主干,选择 loss 类型

讨论要点：
- 两个 train_*.py 90% 重复，差异仅 5 处：model_fn/Loss、denoise 内循环、logger name、wandb tag、argparse description
- 新增 `src/pipeline/train.py` 统一入口 + `backbones/{mitty,rectflow}.py` BackboneSpec
- 旧脚本 `train_mitty.py`/`train_rf.py` 完全保留不动（用户明确"先留着"），由新入口反向 import 复用
- 显式 `--loss {uniform,hand_patch}`（用户要求多一个参数），`--patch-dir` 冲突走 `ap.error` 硬失败
- W&B tags 自动 = `[backbone, loss]`，消融维度可按 tag 分面

**创建的任务：**
- [015] 统一训练 pipeline 入口（backbone / loss 消融友好）

---

**用户原始需求：**
> 你看一下当前 Mitty 的重建实验,视频输入输出是多少帧? 为什么 eval 数据中 2026-04-20_004842 的 eval 视频和 Control 视频, GT 视频长度不一样? ... 修一下这个 bug, 然后跑一下训练,看看 Control 和 GT 是否一致

讨论要点：
- 排查发现 `make_robot_pair.py:89-103` 把 4s segment 切相邻 1s clip 配对（c0t1, c1t2, c2t3），不是 identity 重建
- cache 里 `human_latent` ≠ `robot_latent`，eval 视频 ctrl/gt 内容不同
- 修法：每 segment 4 个 (c{i}, c{i}) identity pair，shutil.copyfile 让两份 mp4 完全相同
- 全量 cache 重生 ~10000 sample 单卡十几小时，本任务只用小集 (`--max-segments 10`) 跑通验证

**创建的任务：**
- [016] 修复 robot-recon 数据生成 bug（identity 配对）

---

**用户原始需求：**
> 新建一个 Task，做一下 GPU 直接加载。T5 分词器的预处理 GPU 直接加载，DiT 的权重直接加载，然后 VAE 处理 cache 的时候提高 cache。所有加载不要走 CPU。

讨论要点：
- DiT 直接 GPU 已在 t014b 分支完成，合并到 main
- mitty_cache.py 中 T5 + VAE 仍走 DiffSynth `WanVideoPipeline.from_pretrained()`，需彻底脱离
- wan_loader.py 新增 `load_text_encoder()` + `load_tokenizer()` 直接加载器
- 训练脚本的 load_sample / _load_patch_weights / init_lora / VAE 全部传 device 到 GPU
- train_lora.py（Wan 2.1 legacy）不在范围内

**创建的任务：**
- [017] GPU 直接加载 — 消除所有 CPU 中转

## 2026-04-21

**用户原始需求：**
> 文档有点乱，整理一下。CLAUDE.md 写重要内容和子文档引导，注明走 /develop skill 并及时更新文档。doc 按 step_x 分阶段，按模块分类（数据/视频inpaint/Human渲染/Seedance API/微调算法）。训练 infra 单独拿出来。

**创建的任务：**
- [018] 文档整理：CLAUDE.md 精简 + 训练 infra 独立 + 模块化索引

---

**用户原始需求：**
> 看一下当前数据加载的模块，不希望使用 epoch 作为统计口径，难以精准控制。能不能纯用 step 来控制？适配当前 Mitty 的两种训练（恒等和外观替换）。

讨论要点：
- `train.py` 用 `--epochs` + `--repeat` 间接算 total_steps，epoch 边界丢尾部 batch，步数不精确
- legacy `train_mitty.py` / `train_rf.py` 已经是纯 step-based，接口不一致
- 方案：`train.py` 改用 `--max-steps` + `infinite_file_batches()`，删除 `--epochs` / `--repeat`
- `--repeat` 完全砍掉（用户确认），数据量纯靠 `--max-steps` 控制
- 恒等和外观替换两种训练不受影响（区别仅在数据目录和 `--init-lora`）

**创建的任务：**
- [019] train.py: epoch-based → 纯 step-based 训练控制

## 2026-04-24

**用户原始需求：**
> 新建一个 worktree,按照 /develop 的 skill 进行. 你可以在 worktree 中进行大幅度重构. 当前项目中: FunControl 的部分不用留了；直接替换噪声那部分不用管了, 这个实验是废弃的；主要的训练 pipeline 是直接外观替换(Mitty) 和三阶段的 LoRA 训练。

**创建的任务：**
- [031] 训练主线重构：清理废弃实验并聚焦 Mitty 与三阶段 LoRA

## 2026-04-24 — 仓库级重构：收敛训练主线、数据规范与配置命名

### 需求

> 走 develop 流程重新规划重构；不沿用旧 T031。去掉 Fun Control、Dxxx/RectFlow 等废弃方法，重点规范数据管理、命名规范和 config；测试产物统一放 `./tmp`；Codex 只用卡 2 测试，卡 3 留给用户实验；新重构不要依赖外部 DiffSynth 训练脚本。

### 决策

- 新建 T032：`doc/tasks/active/032.md`，分支 `feat/t032-repo-refactor`，worktree `.worktrees/t032`。
- 取消旧未完成任务：T005、T006、T011、T021、T031，后续由 T032 统一收敛。
- 删除正式入口中的 FunControl / RectFlow / Dxxx Flow 暴露，不再维护 `train_lora.py`、`train_rf.py`、`rf_model_fn.py`、`backbones/rectflow.py`。
- 新增 `TMP_DIR = ./tmp` 配置约定；smoke/测试默认写入 `tmp/`。
- 文档更新 `doc/step_5_training_infra.md`，明确保留入口、移除入口、DiffSynth 边界、GPU 分配与验证命令。

### 相关任务

- [032] 仓库级重构：收敛训练主线、数据规范与配置命名

### 验证补充

- 新增 `scripts/smoke_t032_refactor.py`，统一执行 T032 轻量冒烟：`compileall`、保留流程入口 `--help`、废弃训练模块不可 import。
- 验证日志写入 `tmp/t032/smoke/`；使用 `CUDA_VISIBLE_DEVICES=2` 确认 Codex 只看到卡 2，卡 3 未使用。

- 新增 `scripts/smoke_t032_gpu.py`，在 `CUDA_VISIBLE_DEVICES=2` 下复制 1 条 pair 到 `tmp/t032/gpu_smoke/`，完成 `mitty_cache` 真实 VAE cache 生成和 `train.py` 1-step 训练/eval。

## 2026-04-24 — 整体重构：配置、实验、Pipeline 与训练文档

**用户原始需求：**
> 当前项目有点乱，我要整体重构：统一路径管理与实验设置到 config；明确 Seedance direct、Seedance + Human Mesh/inpaint、Mitty transfer、Mitty 三阶段 LoRA pipeline；统一实验命名和参数管理，自动记录数据用量；实验类型通过参数指定；完全与外部 Diff sync 解耦；整理文档并细分 train Step5。给我一个重构计划，按照 /develop 走流程。

讨论要点：
- 该需求属于中大型重构，按 `doc/tasks/` 工作流先创建 pending 任务，不在 main 直接实施。
- 新任务编号为 033，独立于已完成/待审的 T032。
- 后续确认后应移动到 `active/033.md`，创建 `.worktrees/t033` 与 `feat/t033-config-pipeline-experiment-refactor` 分支实施。
- 重点交付包括 config 路径中心化、受控 `experiment_type`、自动实验命名、manifest 数据用量记录、三阶段 LoRA 维度自动检测、外部 DiffSynth sync 解耦、Step5 train 文档重划分。

**创建的任务：**
- [033] 整体重构：配置化路径、实验管理、Pipeline 与训练文档重划分

## 2026-04-25 — Seedance direct 1s 数据充分利用

**用户原始需求：**
> 当前 Seedance_direct 数据较少，需要重新做数据切片：1s 长度、0.5s 步长重叠滑窗；切片前/后加入左右翻转增强，输出到 Seedance_direct post-process，翻转 segment 编号往后排；不能做仿射变换；重新划分 eval 和 OOD，OOD task 换成 Put cloth into basket；可以重做切片和 cache，但不动 4s 切片数据。

讨论要点：
- `seedance_clip.py` 改为只从 `seedance_direct/4s/` 生成 `seedance_direct/1s/`，不修改 4s API 输出。
- 每个 4s 源视频生成 7 个普通 1s 窗口和 7 个 hflip 窗口；普通编号 `clip00`–`clip06`，翻转编号 `clip07`–`clip13`。
- 写出 `manifest.jsonl` 记录 `clip_start`、`duration`、`window_idx`、`augment`，避免后续按编号错误反推时间。
- `make_pair.py` 读取 manifest，对 hflip 样本同步翻转 robot target 和 hand patch 权重图。
- train/eval 改为按原始 4s segment 分组划分，避免重叠切片或翻转样本跨 split 泄漏。
- OOD 默认任务改为 `Inspire_Pickup_Pillow_MainCamOnly`。
---

## 2026-04-25 — 人体 Mesh 全身贴合优化

**用户原始需求：**
> 可以把整个人体 Mesh 的贴合都改一下，不单单是手部的问题；尝试放大手部、调整骨骼，并验证人手渲染和机器人 Mesh 的贴合程度。

**处理要点：**
- 新增 `src/tools/body_fit_search.py`，以 G1 robot FK mesh mask 为目标，渲染 SMPLH mask 后计算 body/hand IoU、recall、precision 与加权 score。
- 搜索参数包含 `body_scale`、`hand_scale`、G1 frame 下的 `root_x/root_z` 偏移。
- `segment_pipeline.py` 主线接入动态手型，并将默认参数更新为：`body_scale=0.75`、`hand_scale=1.8`、`root_offset_g1=[0.02, 0.0, 0.0]`。
- 验证结果：
  - Basket/right-hand 样本 score 从 0.4372 提升到 0.5769，hand IoU 从 0.2021 提升到 0.4637。
  - Collect/left-hand 样本 score 从 0.5791 提升到 0.6336，hand IoU 从 0.4121 提升到 0.5043。

**创建的任务：**
- [035] 人体 Mesh 全身贴合优化


补充决策：
- Pickup Pillow 改为 OOD task：`Inspire_Pickup_Pillow_MainCamOnly`。
- OOD 取 32 条；非 OOD 的 4 个 Inspire 任务各取 8 条 eval，共 64 条评估数据。
- `make_pair.py` 新增 `--max-ood-per-task` 和 `--per-task-eval-clips` 支持该固定数量划分。

### 2026-04-25 Seedance overlay 批量输出目录

- 需求：用 `training_data/overlay/4s/` 下的人体 mesh overlay 视频作为 Seedance 输入，prompt 为“修复视频中人的肢体, 白色上衣,黑色裤子,拖鞋.”，三路并发输出到 `training_data/seedance_overlay/4s/`，保持与 `seedance_direct/4s/` 一致的数据结构。
- 实现：`src/pipeline/seedance_advance.py` 增加 `--output-root` 参数；默认行为仍输出 `training_data/seedance_advance/4s/`，本次实验使用 `--output-root training_data/seedance_overlay/4s`。
- 运行：`python -m src.pipeline.seedance_advance --task all --output-root training_data/seedance_overlay/4s --prompt '修复视频中人的肢体, 白色上衣,黑色裤子,拖鞋.' --resume --workers 3`。

## 2026-04-26

**用户原始需求：**
> 当前的训练代码中加几个功能：eval 视频的生成平分到所有 cuda 运行；eval loss 的计算平分到所有 cuda 运行。随后澄清：正式训练应每 100 step eval 一次；统一使用 `train` 作为入口，不要直接启动旧入口。

**完成改动：**
- `src/pipeline/train_mitty.py`：旧实现模块补齐 DDP eval loss / eval video 分片能力，便于 `train.py` 复用时保持一致行为。
- `src/pipeline/train.py`：正式训练入口默认 `--eval-steps 100`；`--eval-video-steps -1` 继续表示跟随 eval loss 频率。
- `scripts/flip_run.sh`、`AGENTS.md`、`doc/scripts_inventory.md`、`doc/codex_migration.md`：统一训练启动命令为 `scripts/flip_run.sh train ...`。
- `doc/step_5_training_infra.md`、`doc/step_5_ffn_lora_merge.md`、`doc/step_5_wandb_setup.md`、`doc/step_5_two_stage_training.md`：明确正式实验使用 `train` 入口，eval loss / eval video 每 100 step 触发一次。

## 2026-04-26 — 训练任务集合弃用 Basket

**用户原始需求：**
> 当前训练 pipeline 需要弃用 put the cloth into basket 任务；该任务手部外观有问题。目前仅使用三个 Task 集合：Inspire_Collect_Clothes_MainCamOnly、Inspire_Pickup_Pillow_MainCamOnly、Inspire_Put_Clothes_into_Washing_Machine。需要同时修改 Mitty 直接训练和 identity + appearance + transfer 三阶段 LoRA 训练。

**完成改动：**
- `src/core/config.py` 新增 `TRAINING_TASKS`，固定为三个维护中的训练任务，显式排除 `G1_WBT_Inspire_Put_Clothes_Into_Basket`。
- `src/pipeline/make_pair.py --task all` 默认展开为 `TRAINING_TASKS`，用于 Mitty 直接外观 transfer 的 human→robot pair；仍可用显式任务短名或 `--task inspire` 调试历史任务。
- `src/pipeline/make_robot_pair.py --task all` 默认展开为 `TRAINING_TASKS`，用于三阶段 LoRA 的 identity robot→robot pair。
- `doc/step_5_training_infra.md` 与 `doc/step_5_two_stage_training.md` 同步记录三任务集合和 Basket 弃用原因。

**补充说明：**
- 只保留三个训练数据集的原因：`Inspire_Put_Clothes_into_Washing_Machine_MainCamOnly` 只有一条有效数据且是重复数据；`Inspire_Put_Clothes_Into_Basket` 的手部外观与其他任务不一致。
- 文档补充了 Mitty 直接训练的 `make_pair --task all` 示例、三阶段 LoRA identity 的 `make_robot_pair --task all` 示例，以及显式调试历史任务的用法。

## 2026-04-28 — T037 训练入口配置化重构

**用户原始需求：**
> 重新搞一下训练入口。公共的、基本不变的参数写到配置中；不同 task name 的训练数据路径和 T5 cache 固定；去掉 loss 选项；merge LoRA 保留并自动检测；保留可训练 LoRA 参数；max step 默认 1000，eval/save 默认 100；eval 相关默认可覆盖；W&B tag 写入所有参数。

**完成改动：**
- 新增 `src/pipeline/train_config.py`，用 `--task-name` 固定选择 train/eval/OOD cache、T5 cache 和默认输出目录。
- `src/pipeline/train.py` 移除 `--loss`、`--cache-*`、`--t5-cache-dir`、`--patch-dir`、`--merge-lora-rank` 公开参数；保留训练、eval、LoRA 和 W&B 覆盖参数。
- `--max-steps` 默认 1000，`--save-steps` / `--eval-steps` / `--eval-video-steps` 默认 100。
- `src/pipeline/train_mitty.py` 的 LoRA merge 从 checkpoint 自动检测 rank；多个 `--merge-lora` 按顺序合并，检测失败直接报错。
- `src/core/train_utils.py` 的 W&B tags 追加所有最终生效参数的 `p:key=value` 形式。
- 更新训练文档、FFN LoRA merge 文档、W&B 文档、`scripts/flip_run.sh` help 和 GPU smoke 脚本。

**创建的任务：**
- [037] 训练入口配置化重构
