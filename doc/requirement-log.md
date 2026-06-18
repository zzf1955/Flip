# 需求日志

## 2026-06-07

**用户原始需求：**
> 1. 归一化
> 2. eval 修一下,这部分我记得是固定一部分样本作为 eval 集合
> 3. 你参考 DreamZero 的实现, 看一下 Action 这块他是咋处理的. 我咋记得 backbone 也参与计算呢, 因为 backbone 中有专门加噪的 Action token

**创建/推进的任务：**
- [075] robot_wam action 归一化、eval 修复与 DreamZero action-token 对齐

**进展更新：**
- 已在 `.worktrees/t070` 将 `train-wan` / `eval-wan` 从外置 MLP action head
  改为最小 DreamZero-style action-token：normalized action 加噪后进入 Wan
  self-attention backbone，再由 action token slice 预测 action flow target；同时保留
  action normalization、task-stratified fixed eval 抽样和 raw-unit action RMSE 诊断。

**用户原始需求：**
> 加 rank,直接上 256/512/1024, 参数没有爆就往上加
>
> 768
>
> 试一下

**创建/推进的任务：**
- [074] H2R top1157 robot_wam 高 rank LoRA 扩容训练

**实验产物：**
- 输出：
  `training_data/log/robot_wam/h2r_top1157_s8_high_rank_v1/`
- 汇总：
  `training_data/log/robot_wam/h2r_top1157_s8_high_rank_v1/summary.csv`
  和 `summary.md`。

**执行结果：**
- 真实 DiT smoke：
  - `rank=256` 通过，trainable params `398,613,672`，checkpoint 约 `800.6 MiB`。
  - `rank=512` 通过，trainable params `776,101,032`，checkpoint 约 `1520.6 MiB`。
  - `rank=768` 通过，trainable params `1,153,588,392`，checkpoint 约 `2240.6 MiB`。
  - `rank=1024` 在单卡 24GB 上 AdamW optimizer step CUDA OOM。
- 三组可跑 rank 均完成 39,024-step 训练，固定 split 与 task073 相同，每个 eval split
  抽样 512 条，`best_metric=eval_mean_loss`：
  - `r256_lr1e-4_aw1_s39024_eval512`：best step 39,024，
    `eval_mean_loss=326.424`，`eval_in_task_loss=371.793`，
    `eval_ood_loss=281.054`。
  - `r512_lr1e-4_aw1_s39024_eval512`：best step 35,000，
    `eval_mean_loss=333.735`，`eval_in_task_loss=378.920`，
    `eval_ood_loss=288.550`。
  - `r768_lr1e-4_aw1_s39024_eval512`：best step 39,024，
    `eval_mean_loss=339.105`，`eval_in_task_loss=381.299`，
    `eval_ood_loss=296.910`。
- Checkpoint audit 通过：三组 best 均为 494 个 trainable tensor，无 `human` /
  `control` key。

**结论：**
- 当前单卡 24GB + AdamW 配置下，`rank=768` 可跑满，`rank=1024` 不可行。
- 高 rank sampled eval 没有刷新 task073 的 `rank=16` sampled best
  `eval_mean_loss=326.157`；本轮最佳高 rank 是 `rank=256` 的 `326.424`。
- 因没有新 sampled best，本轮未追加完整 fixed eval；完整 fixed eval 的当前参考仍是
  task073 `r16_lr1e-4_aw1_s39024_eval512` 的 `eval_mean_loss=1135.986`。

## 2026-06-06

**用户原始需求：**
> 全量训练?

**创建/推进的任务：**
- [073] H2R top1157 robot_wam 固定 split 完整训练与调参

**直接修改：**
- `.worktrees/t070/src/pipeline/robot_wam.py`：
  - `RobotWAMCacheDataset` 支持固定 split manifest。
  - 新增 `build-split` 和 `eval-wan` 子命令。
  - `train-wan` 支持 `--train-manifest`、`--eval-manifest`、`--eval-ood-manifest`
    和 `--best-metric`。
- `.worktrees/t070/src/pipeline/robot_wam_wan.py`：
  - `train-wan` 同时输出 in-task eval、OOD eval 和 `eval_mean_loss`。
  - best checkpoint 默认按 `eval_mean_loss` 选择。
  - `--init-lora` 恢复时同步加载 LoRA、`state_encoder` 和 `action_decoder`。
- `src/tools/summarize_robot_wam_tune.py`：
  - 汇总工具兼容 split eval 字段，输出 `best_metric`、`best_metric_value`、
    `best_eval_in_task_*`、`best_eval_ood_*`、`best_eval_mean_loss`。

**固定 split：**
- 输出：`training_data/robot_wam/splits/h2r_top1157_s8_fixed_v1/`
- train：39,024 samples，15 tasks，861 episodes。
- eval_in_task：4,693 samples，15 tasks，105 episodes。
- eval_ood：4,931 samples，4 tasks，189 episodes。
- OOD task：`grab_cube2_v1`、`push_box_random_v1`、`push_box_two_v1`、`roll`。
- 已校验三组 split 的 `sample_id` 互斥，OOD task 不进入 train。

**完整训练结果：**
- 输出：`training_data/log/robot_wam/h2r_top1157_s8_fixed_v1_full/`
- 三组完整 39,024-step run 均完成：
  - `r16_lr1e-4_aw1_s39024_eval512`：best step 39,024，
    `eval_mean_loss=326.157`，`eval_in_task_loss=372.536`，
    `eval_ood_loss=279.778`。
  - `r16_lr2e-4_aw1_s39024_eval512`：best step 30,000，
    `eval_mean_loss=349.804`。
  - `r32_lr2e-4_aw1_s39024_eval512`：best step 10,000，
    `eval_mean_loss=408.997`。
- 推荐 checkpoint：
  `training_data/log/robot_wam/h2r_top1157_s8_fixed_v1_full/r16_lr1e-4_aw1_s39024_eval512/best_checkpoint.safetensors`。
- 汇总：`summary.csv` / `summary.md` 已生成，checkpoint audit 通过。

**完整 eval：**
- 使用推荐 checkpoint 对 `eval_in_task=4693` 和 `eval_ood=4931` 全量样本运行
  `eval-wan --eval-batches 0 --max-eval-samples 0`。
- 输出：
  `training_data/log/robot_wam/h2r_top1157_s8_fixed_v1_full/r16_lr1e-4_aw1_s39024_eval512/full_eval_best.json`。
- 全量 fixed eval 结果：
  `eval_in_task_loss=163.882`、`eval_ood_loss=2108.090`、
  `eval_mean_loss=1135.986`。
- 结论：训练内 512-sample eval 能用于 checkpoint selection，但不能代表全量 OOD；
  当前模型 in-task 明显下降，task-level OOD action loss 很高，后续优化应优先处理 OOD 泛化。

**用户原始需求：**
> 好的,发布新任务,训练+调参

**创建的任务：**
- [072] H2R top1157 robot_wam train-wan 调参与对比训练

**进展更新：**
- 已完成 4 个 H2R top1157 robot-only `train-wan` 调参 run，输出到
  `training_data/log/robot_wam/h2r_top1157_s8_tune/`：
  `r16_lr5e-5_aw1_s1k`、`r16_lr2e-4_aw1_s1k`、`r16_lr1e-4_aw0p1_s1k`、
  `r16_lr1e-4_aw0p01_s1k`。
- 汇总表写入 `summary.csv` / `summary.md`，包含 task071 baseline 和 4 个 tune run；
  best checkpoint 均通过 key 审计：无 `human` / `control` key，只包含 LoRA、
  `state_encoder` 和 `action_decoder` trainable 权重。
- 同一 `action_loss_weight=1.0` 口径下，`r16_lr2e-4_aw1_s1k`
  优于 task071 baseline：`best_eval_loss=1076.8829` vs `1145.6176`；
  推荐下一轮优先把该配置延长到 3000 steps。
- 新增 `src.tools.summarize_robot_wam_tune` 汇总工具；`.worktrees/t070` 的
  `train-wan` 配置输出补充 optimizer、训练步数、eval/save 间隔和 state/action 参数，
  便于后续调参 run 直接追溯。

## 2026-06-03 — H2R SAM3 blur_r2r 三阶段复现收口

**用户原始需求：**
> 你看一下现在 H2R 的数据集,现在我要在这上面复现三阶段的训练. 需要做的有
> 1. step1 完全重用之前能用的 ckpt 即可
> 2. 第二阶段用 SAM3 模糊机器人, 在外观训练
> 3. 第三阶段先不做, 因为需要配对数据,这部分之后做

**直接修改：**
- 新增 `src.pipeline.h2r_sam3_precompute`，通过 `sam3` conda 环境对
  `data/h2r/v1/video/<task>/episode_*/robot_camera.mp4` 逐 1s clip 运行 SAM3.1
  text segmentation，默认 prompt 为 `robot arm`、backup prompt 为 `robotic arm`，
  输出 `training_data/h2r_sam3_mask/<task>/episode_*.npz`。
- 新增 `src.pipeline.h2r_sam3_blur_pair`，把 `data/h2r/v1/video/<task>/episode_*/robot_camera.mp4`
  与预计算 SAM3/SAM3.1 mask 转成 `training_data/pair/blur_r2r/1s/<h2r_task>/`。
  清晰 robot clip 作为 target，SAM3 mask 区域 Gaussian blur 后作为 control；缺 mask、
  mask 帧数不足或 frame/mask 尺寸不一致时直接失败。
- `scripts/flip_run.sh` 新增 `h2r_sam3_precompute` 与 `h2r_sam3_blur_pair`
  子命令，分别使用 SAM3 环境和项目 `flip` 环境。
- `scripts/run_final_ours_three_stage.sh` 默认改为：
  - step1 只复用已有 identity checkpoint；
  - step2 运行 H2R SAM3 blur_r2r 外观训练；
  - step3 默认 `RUN_STAGE3=0`，不在缺少 H2R 配对数据时启动。
- 更新 `doc/step_5_training_infra.md` 与 `doc/scripts_inventory.md`，记录 H2R SAM3 mask
  artifact 格式、pair/cache 生成命令和 stage2 launcher 用法。

**当前边界：**
- SAM3 precompute 是显式前置步骤；blur pair 转换入口只消费 `training_data/h2r_sam3_mask`
  下的 mask artifact，不会隐式加载 SAM3。
- 第三阶段 H2R h2r 配对训练暂未实现，等待后续配对数据。

## 2026-06-03 — SAM3/SAM3.1 H2R 机械臂/夹爪分割复现

**用户原始需求：**
> 新建环境复现 SAM3，输入视频和 text condition，在 H2R robot video 中稳定分割机械臂/夹爪，比较 prompt 与模型规模、checkpoint 大小，并把结果放到 tmp/ 下查看。

**直接修改：**
- clone 官方 `facebookresearch/sam3` 到 `ref-sam3/`，并将 `ref-sam3/` 加入 `.gitignore`。
- 新建 conda 环境 `sam3`，完成 SAM3.1 text-conditioned video segmentation 复现。
- 在 `data/h2r/v1/video/*/*/robot_camera.mp4` 上测试多条视频、多种 prompt、`max_num_objects` 和 point refinement 策略。
- 新增 `doc/sam3_h2r_segmentation.md`，记录模型输入输出、参数量/ckpt 大小、显存约束、稳定 prompt、keyframe point-refine 夹爪候选策略，以及 `tmp/` 产物路径。

**结论摘要：**
- 整条机械臂首选 `prompt="robot arm"`，备用 `robotic arm`，推荐 `max_num_objects=1`。
- `robot gripper`、`mechanical gripper`、`end effector` 在当前 H2R 数据上不稳定。
- 夹爪/末端候选可用 `robot arm` 先得到整臂轨迹，再在 keyframes `0,4,8,12` 上做 point refine；单帧 point refine 不稳定。

## 2026-06-03

**用户原始需求：**
> 先不改 draft，列一个新的 task：从 H2R 的数据中训练一个 Diffusion Policy。

**创建的任务：**
- [068] H2R Diffusion Policy BC 下游控制训练

## 2026-06-02

**用户原始需求：**
> 发布一个 task，把已经讨论好的 WAN2.2 5B + DreamZero 的技术细节写进去，然后更新文档，
> 说明 Cosmos Predict2B 路线作为备选。

**创建的任务：**
- [067] Wan2.2-5B + DreamZero-style LoRA 离线 video-action 原型

**直接修改：**
- 新增 `doc/tasks/pending/067.md`，明确主线是从 `Wan-AI/Wan2.2-TI2V-5B` 视频基座开始，
  在 Wan DiT 上训练 LoRA，并从头训练 DreamZero-style `state_encoder`、
  `action_encoder`、`action_decoder`。
- 更新 `doc/step_5_training_infra.md`，记录 Wan2.2-5B + DreamZero LoRA 的训练语义、
  checkpoint 保存/恢复风险、建议最小配置，以及 Cosmos Predict2B 作为备选路线的边界：
  只能复用高层思路，不能直接替换 DreamZero 的 Wan wrapper。

## 2026-06-01

**用户原始需求：**
> 按照当前的思路，优化这个 Transformer IDM 的效果，在当前 H1 的视频上尽可能准确。

**创建的任务：**
- [064] H1 RGB Motion Transformer IDM 准确率优化

**直接修改：**
- `src.pipeline.humanoid_pair_idm` 新增 `motion_transformer_v2`：在 task060 patch-level
  motion token 基础上加入 raw RGB diff/abs-diff patch stem，并使用 residual MLP readout head。
- 旧 `motion_transformer` checkpoint 继续按 `legacy_mlp` / `raw_motion_stem=false`
  严格 replay；新 checkpoint 保存 `head_arch`、`head_depth`、`raw_motion_stem`。
- 训练新增 normalized action 空间的 `--variance-loss-weight` /
  `--variance-loss-warmup-ratio`、`--grad-accum-steps` 和 CUDA bfloat16 `--amp`。
- 已完成 compile、help、新 v2 checkpoint replay、旧 task060 checkpoint replay、真实 H1
  smoke 和 smoke checkpoint validate。
- 已完成 task064 正式长训：`motion_transformer_v2`、patch16、`steps=8000`、
  `batch_size=16`、`grad_accum_steps=2`、AMP、variance loss `0.03`。
- 完整 H1 held-out validate（`71486` samples）结果：
  `action_mse=0.028906064108014107`、`action_norm_mse=0.1969597190618515`、
  `relative_l2_error=0.20390427137523634`、`pred_norm_var_mean=0.8531579971313477`、
  `action_mean_dim_corr=0.8984220096698174`；`pred_norm_var_mean` 由完整
  `val_predictions.csv` 和 checkpoint train mean/std 复算。
- 相比 task060 RGB motion Transformer 完整 held-out，`action_mse` 约降低 `43.8%`，
  normalized MSE 约降低 `38.6%`。
- 新增并验证 `--init-checkpoint` 二阶段初始化；两组 fine-tune（`lr=1e-4` + variance
  loss `0.01`，以及 `lr=3e-5` 无 variance loss）均未刷新 `s8000` 的 4096-sample
  subset best，因此当前推荐仍是 `tmp/humanoid_pair_idm_t064_v2_p16_s8000/best_checkpoint.pt`。
- 更新 `doc/h1_idm_methods.md`、`doc/step_5_training_infra.md`、
  `doc/scripts_inventory.md`，记录 v2 架构、smoke 命令和正式长训矩阵。
- 已将 `feat/t064-h1-motion-transformer-accuracy` 通过 `git merge --no-ff` 合并回 `main`，
  并将 [064] 从 `doc/tasks/active/064.md` 移动到 `doc/tasks/done/064.md`。

**用户原始需求：**
> 先看一下当前不同 task 中, Transformer 的做法/原 Ada World decoder 的效果. 然后跑一下这个改进后的 Ada World decoder.

**上下文复核：**
- task053 的 H1 RGB Transformer 是两帧 patch embedding + CLS / frame embedding +
  `TransformerEncoder`，比 small CNN 更好，但仍是图像端 IDM。
- task060 的 active worktree 已把 RGB IDM 推到 `motion_transformer`：额外构造
  patch 级 motion token，使用 `cls + motion_cls + frame0_mean + frame1_mean +
  motion_mean` 读出，并补上 AdamW betas、warmup + cosine、min lr 等训练细节。
- task057 的 AdaWorld decoder baseline 是 `32 -> 128 -> 128 -> 26` 小 MLP，
  held-out `action_mse=0.07853357493877411`，全量 eval `action_mse=0.07298979163169861`。

**直接修改：**
- `src.pipeline.adaworld_action_decoder` 新增 `--decoder-arch {mlp,residual_mlp,gated_mlp}`；
  原 MLP baseline 保留，当前推荐默认切到 `residual_mlp`。
- `residual_mlp` 使用 hidden projection + pre-norm residual MLP blocks + 2 层输出 head；
  `gated_mlp` 在 residual block 内用 SiLU value 和 sigmoid gate 做 gated 变体。
- 训练超参新增 AdamW `--adam-beta1/--adam-beta2`、`--lr-warmup-steps`、
  `--lr-warmup-ratio`，scheduler 改为可表达 warmup + cosine 或 warmup + flat。
- checkpoint 保存完整 decoder 架构、学习率、weight decay、betas、warmup 和 scheduler
  配置，`validate` / `eval` 严格 replay。
- 更新 `doc/step_5_training_infra.md`、`doc/scripts_inventory.md`，记录推荐配置、命令和
  task061 对照指标。

**task061 全量结果：**
- 推荐训练配置：`residual_mlp`、`hidden_dim=256`、`depth=4`、`dropout=0.02`、
  `lr=5e-4`、`weight_decay=1e-4`、`betas=(0.9,0.95)`、cosine scheduler、
  `min_lr_ratio=0.02`、`lr_warmup_ratio=0.05`、`steps=3000`、`batch_size=1024`。
- held-out best checkpoint：`action_mse=0.054645732045173645`，
  `action_mean_dim_r2=0.6565681374990023`，`action_mean_dim_corr=0.809087702861199`，
  `action_pred_std_ratio_mean=0.8399375424935267`。
- 全量 eval：`action_mse=0.04235832020640373`，
  `action_mean_dim_r2=0.72525387773147`，`action_mean_dim_corr=0.850245631658114`，
  `action_pred_std_ratio_mean=0.8563885574157422`。
- 相比 task057，held-out `action_mse` 约降低 `30.4%`；全量 eval `action_mse` 约降低
  `42.0%`。剩余高 MSE 维度主要是 `action_dim_06/07/08/09/10/22/23`，但整体预测方差比
  已从 task057 的约 `0.71` 提高到约 `0.84-0.86`。

**用户原始需求：**
> merge 当前 task；新建 task。

**直接修改：**
- 已将 `feat/t061-adaworld-idm-opt` 通过 `git merge --no-ff` 合并回 `main`。
- 已将 [061] 从 `doc/tasks/active/061.md` 移动到 `doc/tasks/done/061.md`，并补充交付记录。

**创建的任务：**
- [063] AdaWorld decoder 二阶段消融与 loss/head 优化

**用户原始需求：**
> 继续优化 63，可以并行跑多个实验

**直接修改：**
- `src.pipeline.adaworld_action_decoder` 新增 `--head-arch` / `--head-groups`、
  `--loss-type`、`--loss-weights`、`--smooth-l1-beta`、`--variance-loss-weight`，
  支持 shared / per-dim / grouped head、weighted MSE、SmoothL1 和方差校准项。
- 新增 `src.pipeline.adaworld_decoder_diagnostics`：从 `best_val_predictions.csv` /
  `predictions.csv` 汇总逐维 MSE、normalized MSE、R2、correlation、预测方差比，并导出
  可复用的 loss 权重 JSON/CSV。
- AdaWorld decoder 训练默认从 task061 的 `hidden_dim=256` / `lr=5e-4` 升级为
  `hidden_dim=384` / `lr=8e-4` 的 residual MLP shared-head 配置。
- 更新 `doc/step_5_training_infra.md`、`doc/scripts_inventory.md` 和
  `doc/h1_idm_methods.md`，记录 task063 诊断、消融、最佳配置和 task061/task057 对照。

**task063 实验结果：**
- 逐维诊断表：`tmp/adaworld_action_decoder_t063_analysis/per_dim_summary.csv`
- loss 权重：`tmp/adaworld_action_decoder_t063_analysis/loss_weights.json`
- 1500-step sweep 里最好的候选是 `hidden_dim=384` + `lr=8e-4` 的 residual MLP shared head，
  held-out `action_mse=0.052365291863679886`；`per_dim head`、`weighted_mse`、
  `variance_loss` 都没有超过这个配置。
- 3000-step 完整训练后，最佳 checkpoint 位于
  `tmp/adaworld_action_decoder_t063_full_c09_h384_lr8e4/best_checkpoint.pt`，held-out
  `action_mse=0.05023810639977455`，`action_mean_dim_r2=0.6858815573729001`，
  `action_mean_dim_corr=0.8282286180899694`，`action_pred_std_ratio_mean=0.8748558117793157`。
- 对应全量 eval 位于 `tmp/adaworld_action_decoder_t063_full_c09_eval_best/metrics.json`，
  `action_mse=0.029545826837420464`，`action_mean_dim_r2=0.8002844131909884`，
  `action_mean_dim_corr=0.893933926637356`，`action_pred_std_ratio_mean=0.9002112241891714`。
- 相比 task061，task063 最佳配置在 held-out 上把 `action_mse` 再降约 `8.1%`，在全量
  eval 上再降约 `30.3%`；说明 `610k` 参数量并非容量上限，`1.36M` 参数的
  `hidden_dim=384` shared-head decoder 更合适。

## 2026-05-31

**用户原始需求：**
> 现在几个不同的方法：Baseline、Ada World decoder、Transformer。写一个文档，统一说明这个
> IDM 的思路、实验、数据配置等；先仅写 H1 数据上的相关信息。

**直接修改：**
- 新增 `doc/h1_idm_methods.md`，只整理 Humanoid Everyday H1 数据上的 IDM 方法和实验。
- 文档统一说明完整 H1 数据根、`560422` 相邻帧 pair、`1400/200` episode split、
  mean baseline 定义、AdaWorld latent decoder 数据流和 RGB motion Transformer 数据流。
- 汇总完整 held-out split 上三种方法的同口径指标，并把后续报告主指标固定为：
  归一化空间 MSE、action relative L2、归一化空间预测方差。
- 补充 task061 optimized AdaWorld decoder；完整 held-out split 上：
  mean baseline normalized MSE `1.007003` / relative L2 `0.474226` / pred norm var `0.0`，
  task061 AdaWorld normalized MSE `0.349901` / relative L2 `0.280356` / pred norm var `0.706071`，
  RGB motion Transformer normalized MSE `0.320904` / relative L2 `0.272016` /
  pred norm var `0.689259`。
- 在 `doc/step_5_training_infra.md` 的 H1 IDM 段落加入该统一文档入口。

**用户原始需求：**
> 好的, 现在你想办法优化一下 tf 的这个 IDM, 包括模型架构, 学习率之类的细节

**创建的任务：**
- [060] H1 Transformer IDM 架构与训练超参优化

**直接修改：**
- `src.pipeline.humanoid_pair_idm` 新增默认 `motion_transformer` 骨干，保留旧
  `transformer` 作为 legacy checkpoint / ablation 对照，并把两帧 patch 差异显式编码成
  motion tokens，再用 `cls + motion_cls + frame0_mean + frame1_mean + motion_mean` 读出。
- 训练默认改为 AdamW `lr=3e-4`、`weight_decay=1e-2`、`betas=(0.9,0.95)`，并使用
  cosine scheduler + 5% warmup，`min_lr_ratio=0.02`；Transformer 默认 `hidden_dim=256`、
  `transformer_depth=6`、`transformer_dropout=0.05`。
- smoke、中等规模对照和完整 H1 数据口径训练都已跑通，`validate` 对 `best_checkpoint.pt`
  的复算与训练内指标一致。
- 按 AdaWorld task057 的完整数据口径运行：`data/humanoid-everyday-h1-chunks0-6-8-200`、
  `max_samples=0`、`frame_stride=1`、episode-level split，得到 `488936` train samples /
  `1400` train episodes 和 `71486` val samples / `200` val episodes。
- 完整口径 run（`steps=2000`、`batch_size=32`，中途 eval 抽 `4096` held-out samples）
  在子集上达到 `action_mse=0.052878`，mean baseline `0.157396`，`normalized MSE=0.328745`，
  `pred_std_ratio_mean=0.827459`。
- 对 `best_checkpoint.pt` 跑完整 held-out validate：`71486` val samples 上
  `action_mse=0.051443`，mean baseline `0.156353`，`normalized MSE=0.320904`，
  `action_mean_dim_r2=0.685052`，`action_mean_dim_corr=0.826011`，
  `pred_std_ratio_mean=0.828955`。同 split task061 optimized AdaWorld latent decoder
  held-out `action_mse=0.054646`，当前 RGB motion Transformer 低约 `5.9%`；task057
  AdaWorld baseline decoder `action_mse=0.078534` 仅作为历史基础 MLP baseline。
- 同步更新 `doc/step_5_training_infra.md`、`doc/scripts_inventory.md`、
  `doc/requirement-log.md`，把新的默认架构、学习率和实验结果写回文档。

**用户原始需求：**
> 你看一下当前 Ada world的 IDM 模型和已有的结果, 开一个 task 来优化, 关注学习率,
> 网络架构等基础参数, 试试能不能把预测的误差优化一下

**创建的任务：**
- [061] AdaWorld IDM 基础超参与 decoder 架构优化

**补充结论：**
- 当前 AdaWorld IDM 是 `src.pipeline.adaworld_action_decoder` 的 latent decoder 路线：
  `(frame_t, frame_{t+1}) -> AdaWorld z_t[32] -> action_t[26]`。
- task057 全量 H1 结果使用默认小 MLP `32 -> 128 -> 128 -> 26`；held-out
  `action_mse=0.07853357493877411`，全量 eval `action_mse=0.07298979163169861`。
- 新任务聚焦学习率、scheduler、weight decay、batch size、MLP 宽度/深度、归一化、
  dropout 和轻量残差/gated decoder 变体，不重新训练 AdaWorld action encoder，也不加载
  world model。

**用户原始需求：**
> 先改别的 2. 加指标 3. 解决数据的问题, 现在是不是 eval 划分的不太对, 如果不对的话修正过来

**直接修改：**
- 为 `src.pipeline.wan_pair_idm` 和 `src.pipeline.humanoid_pair_idm` 新增
  normalized MSE、per-dim R2、per-dim correlation、预测方差比等诊断指标；
  validation 现在会输出更多能判断“是否只是在回归均值”的统计量。
- 两个入口的 `validate` / `eval` 现在默认复用 checkpoint 中保存的数据 root、
  resize、split 和 seed，避免训练和复算使用不同的样本划分。
- Humanoid H1 默认 split 改为 `episode`，并让显式 `--train-samples` /
  `--eval-samples` 在 episode 不重叠的前提下截取样本，修正原先 sample split 的
  episode 泄漏问题。
- 更新 `doc/step_5_training_infra.md`、`doc/scripts_inventory.md`，同步记录新的指标
  和 split 语义。

**用户原始需求：**
> 详细调查一下情况，然后发布一个新的 task，在哪已有的 pair 数据上跑 Masquerade

**创建的任务：**
- [058] 现有 h2r pair 的 Masquerade 渲染 baseline

**补充结论：**
- 当前实现已经能跑通现有 h2r pair 的 baseline 生成，但 human 分割、背景 inpaint 和
  机器人遮挡边界仍是第一版启发式复现，后续还需要继续改进复现质量。

**用户原始需求：**
> 发布一个新的 task：提升渲染精度，渲染真实的 Mesh，现在是快速渲染的 proxy；并回答现在
> human 检测 / 分割 / inpaint 都是什么模型做的。

**创建的任务：**
- [059] Masquerade baseline 真实 Mesh 高精度渲染

**补充结论：**
- 当前 Masquerade baseline 的 human 检测 / 分割不是专门模型，而是 OpenCV 启发式
  foreground / skin mask；背景重绘使用 OpenCV Telea inpaint，不是 LaMa、ProPainter
  或扩散式视频 inpaint。

**用户原始需求：**
> 发布一个新的 task,做区间的预测

**创建的任务：**
- [052] 两帧区间平均动作 IDM

**直接修改（task 052 完成）：**
- `src.pipeline.humanoid_pair_idm` 新增 `--frame-delta`，把输入改成
  `(frame_t, frame_{t+d})`，标签改成 `mean(action[t:t+d])`；checkpoint 和验证回放
  都保存并复用 `frame_delta` / `target_semantics` / split 配置。
- `validate` / `eval` / `train --help` 全部确认接到 `--frame-delta`，并继续使用
  episode-level split；旧 checkpoint 需要显式 `--allow-cli-split`。
- 完成 H1 smoke：`frame_delta=4, steps=10, max_samples=128`，训练与验证都无 NaN。
- 完成 H1 sweep：`d=1/2/4/8/16`，结果是 `d=1` 最好，`best action_mse=0.107009`，
  `mean baseline=0.110856`；`d>1` 没有稳定优于默认 baseline，因此默认仍保持
  `frame_delta=1`。
- 发现 `data/humanoid-everyday-h1-chunks0-6-8-200` 含 13 个不可读 parquet，实际 smoke /
  sweep 使用了临时 symlink 根 `tmp/h1_t052_valid_200_v2`，没有修改原始数据。
- 2026-05-31 复核当前数据根时，1600 个 parquet 均可按 `action/frame_index/next.done`
  读取，且对应 `videos/chunk-*/egocentric/episode_*.mp4` 全部存在；当前无需删除坏数据或
  从外网补源。
- 同步更新 `doc/step_5_training_infra.md`、`doc/scripts_inventory.md`，把区间预测语义、
  sweep 结论和数据注意事项写回文档。

**用户原始需求：**
> 新建一个 task,做如下事情
> 1. 在 flip/data/humanoid-everyday-h1-chunks0-6-8-200 上训练,这个数据多
> 2. 分 task 训练
> 3. 探索不同的模型架构,比如下游换 tf

**创建的任务：**
- [053] H1 全量 task 分组训练与 Transformer 架构探索

**追加要求：**
> task 太碎，之后训练不再区分 task；先尝试 Transformer。

**直接修改：**
- `src.pipeline.humanoid_pair_idm` 在主线已有 interval mean action 语义上新增
  `--model-arch {transformer,small_cnn}`，默认使用 Transformer，保留 small CNN 作为
  对照 baseline。
- Transformer 入口使用两帧 RGB patch embedding、CLS token、frame embedding 和
  `TransformerEncoder` 聚合后回归 26 维 action；checkpoint 保存完整架构配置，
  `validate` / `eval` 可复用 Transformer checkpoint。
- 后续架构对照不再默认按 task 切训练，沿用 sample / episode split；task 信息只作为数据审计
  背景，不作为当前训练主轴。
- 已完成同口径 sample split 对照：Transformer `1000 step` 的 normalized RMSE 为
  `0.760`，small CNN 为 `0.980`，mean baseline 为 `1.001`；26 个 action 维度上
  Transformer 均优于 CNN 和 mean baseline。

**用户原始需求：**
> 复现 `https://huggingface.co/Little-Podi/AdaWorld`，参考 task051；当前 IDM
> 效果不好，希望从纯视觉输入中提取 action，先用 latent 空间表示 action，再接一个小
> action head 输出具体 action。
> 澄清：仓库 clone 名称应为 `ref-<repo_name>`，这里是 `ref-AdaWorld`；AdaWorld
> 有 action encoder 和 world model 两部分，本次只做 `(f_t, f_{t+1}) -> 32`
> 维连续 latent action 的 action encoder；world model 先不管。当前需要跑通的是
> H1 两帧图像输入 action encoder，并输出 latent action。

**创建的任务：**
- [054] AdaWorld H1 两帧 action encoder latent 提取

**补充结论：**
- AdaWorld HF 权重仓库中 `lam.ckpt` 约 1.8GB，已用于 action encoder smoke。
- `adaworld.safetensors` 的 LFS 指针大小约 11.46GB，是下游 SVD/Vista 风格 video
  diffusion world model；当前目标只复现 action encoder，不把 world model 纳入本阶段。

**用户原始需求：**
> 现在要基于 AdaWorld 做 IDM；已经复现 AdaWorld 的 action Encoder，需要从 latent
> action 中把机器人具体的 action 解码出来。数据在
> `flip/data/humanoid-everyday-h1-chunks0-6-8-200`，数据流是两帧图像
> `--AdaWorld--> latent action --action decoder--> action`；需要确定 decoder
> 架构并实现整个训练 pipeline。

**创建的任务：**
- [056] AdaWorld latent action decoder IDM

**直接修改：**
- 新增 `src.pipeline.adaworld_action_decoder`：读取 task054 的 `latent_actions.npz`，
  按 `episode/chunk/rel_frame_t` 回查 H1 `action` 标签，训练
  `(frame_t, frame_{t+1}) -> z_t -> action_t` 的下游 action decoder。
- decoder 架构采用小 MLP baseline，默认 `32 -> 128 -> 128 -> 26`；因为 AdaWorld LAM
  已经把两帧图像压成低维 action latent，第一版不再使用 CNN。后续只有需要多步时序
  上下文或 task 条件时才考虑 Transformer / RNN。
- `scripts/flip_run.sh` 新增 `adaworld_action_decoder` 子命令，并更新
  `doc/scripts_inventory.md`、`doc/step_5_training_infra.md`。

**用户原始需求：**
> 好的,你发布一个新的 task,在 1600 条 h1 的数据上,跑这个基于 Ada World 的 IDM.

**用户追加需求：**
> 在 H1 完整 1600 条数据上跑 AdaWorld latent decoder；全量 latent 提取需要进度条，
> 并且数据要实时落盘，避免长时间无反馈和中途失败后完全丢失产物。

**创建的任务：**
- [057] AdaWorld latent action decoder H1 全量 1600 条训练

**目标摘要：**
- 基于 task056 的 AdaWorld latent action decoder，在完整
  `data/humanoid-everyday-h1-chunks0-6-8-200` 上跑全量训练与复算。
- 必要时先用 task054 的 action encoder 对完整 H1 数据提取 latent artifact，再训练
  `(frame_t, frame_{t+1}) -> z_t -> action_t` decoder。
- 重点输出全量训练 checkpoint、验证指标和与 mean baseline / 旧两帧 RGB IDM 的对比。

**直接修改与全量结果（task 057 完成）：**
- `src.pipeline.adaworld_action_encoder` 的全量提取改为 tqdm 进度条；
  `latent_actions.npy` 通过 memmap 按 batch 实时写盘，`manifest.jsonl` 逐 batch flush，
  结束后仍生成 decoder 兼容的 `latent_actions.npz`。
- 更新 `doc/scripts_inventory.md`、`doc/step_5_training_infra.md`，记录流式输出语义和
  H1 全量训练结果。
- 完整 H1 数据根：`data/humanoid-everyday-h1-chunks0-6-8-200`
- 全量 AdaWorld latent artifact：`tmp/adaworld_action_encoder_h1_full_t057/latent_actions.npz`
- latent/action 样本数：`560422`，episode 数：`1600`
- decoder 训练输出：`tmp/adaworld_action_decoder_h1_full_t057`
- split：episode-level，`488936` train samples / `1400` train episodes，
  `71486` val samples / `200` val episodes
- decoder 配置：默认 MLP baseline，`32 -> 128 -> 128 -> 26`，`steps=2000`，
  `batch_size=1024`
- held-out best checkpoint：
  - `action_mse = 0.07853357493877411`
  - `mean_baseline_action_mse = 0.15635326504707336`
  - `action_mean_dim_r2 = 0.504274274294193`
  - `action_mean_dim_corr = 0.7063991037698892`
- 全量 eval：
  - `action_mse = 0.07298979163169861`
  - `mean_baseline_action_mse = 0.15620023012161255`
  - `action_mean_dim_r2 = 0.5340813008638529`
  - `action_mean_dim_corr = 0.7275451834385211`
- `validate` 复算 `best_checkpoint.pt` 与训练保存的 `best_val_metrics.json` 一致。
- 与同 split mean baseline 相比，held-out action MSE 约降低 `49.8%`；与全量 eval mean
  baseline 相比，action MSE 约降低 `53.3%`。历史两帧 RGB H1 IDM 结果使用不同样本量、
  split 或 target 语义，只作为参考，不作为严格同 split 对照。

## 2026-05-30

**用户原始需求：**
> 新写一个 IDM：明确 action 与相邻帧状态的关系；看到两帧预测两帧之间的 action；
> 使用小 CNN；手部和胳膊训练两个小网络，而不是混在一起。

**创建的任务：**
- [051] 两帧小 CNN 双网络 IDM

**完成改动：**
- 新增 `src.pipeline.wan_pair_idm`：使用相邻 RGB 帧 `(s_t, s_{t+1})` 预测
  `a_t`，明确 action 对齐为 `frame_index=t` 的 `action.ee_action` /
  `action.hand_cmd`。
- arm 与 hand 使用两套独立小 CNN，在同一训练入口中同时训练、保存和评估。
- `scripts/flip_run.sh` 新增 `wan_pair_idm` 子命令；文档记录 smoke、正式训练和
  held-out validate 命令。

**追加改动：**
- 新增独立 `src.pipeline.humanoid_pair_idm`：Humanoid Everyday H1 LeRobot 训练
  不再挂在 `wan_pair_idm` 里，而是通过 `train` / `validate` / `eval` 子命令直接读
  `data/chunk-*/*.parquet` 与 `videos/chunk-*/egocentric/*.mp4`。
- H1 路径保持 `(frame_t, frame_{t+1}) -> action_t` 对齐口径，使用单个小 CNN
  输出完整 26 维 `action`；`wan_pair_idm` 只保留旧 WBT 的 `ee_action` /
  `hand_cmd` 双头训练。
- `scripts/flip_run.sh` 新增 `humanoid_pair_idm` 子命令，方便直接启动 Humanoid
  Everyday H1 的 700 训练 / 100 eval / 1000 step 实验。
- 已完成 Humanoid Everyday H1 全量实验：`700 train + 100 eval + 1000 step`
  的训练跑通，best checkpoint 的 held-out `action_mse=0.143645`，
  `mean_baseline_action_mse=0.154795`，并通过独立 `validate` 复算。

---

## 2026-05-07

**用户原始需求：**
> 你看一下计算 metrics 的代码,你写一个代码,将 1000 步跑完了,还没有测试 metrics 的, final 开头的 training log 中的 ckpt,跑完整的测评

**直接修改：**
- 新增 `scripts/eval_final_step1000_missing.py`：扫描 `training_data/log/final*`，
  找出已完成 `step=1000/1000`、存在 `ckpt/step-1000.safetensors`、但缺少离线
  `ckpt/step-1000_eval/summary.*` 的 run，并按训练日志中的数据配置调用
  `scripts/flip_run.sh eval_mitty` 补跑完整离线评估。
- 根据追加要求，脚本默认跳过 `data_type=r2h`，并对其余 run 显式添加
  `--mask-region-metrics on`，保证离线 summary 包含 Local FID / Local FVD 和
  前景/背景局部指标。
- 根据追加要求，脚本同时显式添加 `--patch-fid`，保证离线 summary 包含
  `foreground_patch_fid` 和 patch 选择配置字段。
- 根据追加要求，脚本固定沿用旧评估口径 `80 in-task + 42 OOD`，并把每个 run 的
  视频、summary、`data_split/` 和执行日志写到该 run 的 `full_eval/`。
- 根据追加要求，脚本新增 `--runner {flip_run,flip_run_2}`，可选择通过
  `scripts/flip_run.sh` 或 `scripts/flip_run_2.sh` 启动评估。
- 根据追加要求，脚本新增 `--cuda-list` 队列调度：用户给出可用 CUDA 后，脚本为
  每张卡启动一个 worker，每张卡同时只跑一个 eval，跑完自动取下一个 run。
- 根据追加要求，脚本默认跳过 `final_ours_step1*` 和 `final_ours_step2*`，并提供
  `--include-ours-step1-step2` 作为显式纳入开关；dry-run 会打印跳过原因统计。
- `src.pipeline.evaluate_mitty_models` 新增 `--output-exact-dir`，用于让批处理脚本
  把单个 run 的评估产物精确写到 `<run>/full_eval/`，不经过
  `--output-dir/<run>/<step>` 嵌套。
- `doc/step_5_training_infra.md`：记录批量补跑 final step-1000 离线评估的
  dry-run、执行命令、筛选规则和输出位置。

---

## 2026-05-07

**用户原始需求：**
> 先做 patch FID，评测的时候加一个开关只跑 patch FID，中间结果要输出 patch 位置的 overlay，能看出来视频的每一帧选的是哪些 patch；用 `training_data/log/eval_h2r_80in_42ood_local_0506/Mitty-h2r_1s-400d_r96_self_qkv_1000s_0503_154657/step-1000/in_task_eval` 这个数据做。

**创建的任务：**
- [044] 新增 mask patch FID 与 patch overlay

**需求跟进：**
- Patch 选择默认改为只要 patch 内有任意 mask 像素就选中，即
  `--patch-coverage-threshold 0.0 --patch-max-per-frame 0`。
- Patch 选择继续收紧为 patch 内 mask 像素数严格大于 5 才计入，即
  `--patch-min-mask-pixels 5`。

## 2026-05-06

**用户原始需求：**
> eval mitty 脚本太大需要拆分；当前 FVD 是黑色背景 FVD，应改成 Local FVD；当前输出的 local fid 视频文件夹里视频没有标记哪些区域参与计算，需要改进。

**创建的任务：**
- [043] 重构 Mitty 离线评估与 Local FVD

**完成改动：**
- `src/pipeline/evaluate_mitty_models.py`：保留为离线 Mitty eval 入口，run/spec、CLI、生成、Local 视频输出和 summary 写入拆到 `src/pipeline/eval_mitty/`。
- `src/tools/eval_metrics.py`：区域 FVD 改为 bbox crop Local FVD，输出字段为 `foreground_local_fvd`；旧黑底区域 FVD 字段不再输出。
- `--write-local-videos`：除 Local crop 视频外，新增原始画面 overlay 视频和三列 overlay compare，用 mask 与 bbox 标出参与 Local FID / Local FVD 计算的区域。
- `doc/step_5_training_infra.md`：同步 Local FVD 字段、`--no-fid` 影响范围和 Local 视频输出说明。

---

## 2026-05-06

**用户原始需求：**
> 看一下当前计算 metrics 的代码,加一个功能. 从 blur_r2r 的 mask 中读数据,将 Video 分割成 前景和背景,然后计算分割之后的视频的 fid/fvd in task/ood task

**创建的任务：**
- [040] mask-region FID/FVD eval for blur_r2r

**完成改动：**
- `src/tools/eval_metrics.py`：新增 SAM2 mask 路径解析、clip mask 对齐、前景/背景视频拆分，并在 `process_step` 中支持区域 Frechet 指标。
- `src/pipeline/evaluate_mitty_models.py`：`data_type=blur_r2r` 时默认启用 mask-region 指标，覆盖 `in_task_eval` 和 `ood_eval`；支持 `--mask-region-metrics off/on/auto` 与 `--sam2-mask-root`。
- `tests/test_eval_mask_regions.py`、`doc/step_5_training_infra.md`：补充 mask 对齐/区域拆分测试和离线评估文档。

**追加改动：**
- `src/tools/eval_metrics.py`：新增全局 `mse`，局部 `foreground_mse`、`foreground_psnr`、`foreground_ssim`、`background_mse`、`background_psnr`、`background_ssim`；局部 MSE/PSNR/SSIM 只统计 mask 内/外像素。
- 区域 Frechet 字段改为黑底口径命名：`foreground_black_fid`、`foreground_black_fvd`、`background_black_fid`、`background_black_fvd`。
- `--no-fid` 只关闭全局和黑底区域 FID/FVD，不关闭局部 MSE/PSNR/SSIM。
- `src/pipeline/evaluate_mitty_models.py` 与文档同步输出字段。

---

## 2026-05-06

**用户原始需求：**
> 看一下当前计算 eval metrics 的代码。现在需要改成：1. 按照数据划分表，从表后面读取 k% 来进行 eval；2. eval 完之后，把结果写到对应 ckpt 的位置。

**直接修改：**
- `src/pipeline/runtime_data.py`：新增 `build_tail_eval_split`，按每个 task 的 `pair_order.jsonl` 尾部百分比选择 in-task/OOD eval 样本。
- `src/pipeline/evaluate_mitty_models.py`：离线综合评估改为读取 task 级顺序表尾部 `--eval-tail-percent`，默认结果写到 `training_data/log/<run>/ckpt/<step>_eval/`，并保存 `summary.*` 与 `data_split/`。
- `tests/test_runtime_data.py`、`scripts/flip_run.sh`、`doc/step_5_training_infra.md`：补充尾部百分比选择测试、入口示例和离线评估文档。

---

## 2026-05-06

**用户原始需求：**
> 当前训练每一轮 eval 数据不同，需要统一。每个 task 的 pair 目录下生成一次乱序表，例如 `training_data/pair/blur_r2r/1s/Inspire_Collect_Clothes_MainCamOnly`；训练时从这个表取数据，按 pair 划分，不按 segment；`--train-size` 按 task 比例分配，并在训练前打印每个数据集选了多少 training size。

**创建的任务：**
- [039] 固定 task 级 pair 顺序表的数据划分

**完成改动：**
- `src/pipeline/runtime_data.py`：新增 task 级 `pair_order.jsonl` 生成/读取/校验；训练从顺序表头部按比例取 train，从尾部取 in-task/OOD eval；eval video 子采样不再随 step 变化。
- `src/pipeline/train.py`：新增 `--pair-root`，训练启动日志打印每个 split/task 的实际样本数和顺序表路径。
- `src/pipeline/train_config.py`、`scripts/smoke_test_gpu.py`：为训练 preset 补充 `pair_root`，让 smoke 使用自己的临时结构化 pair 目录，避免临时 cache 与主数据完整 manifest 混用。
- `tests/test_runtime_data.py`：覆盖顺序表首次生成与复用、train size 按 task 比例分配、train/eval 分离、eval video step 固定。
- `doc/step_5_training_infra.md`、`doc/step_5_two_stage_training.md`：记录 `pair_order.jsonl` 位置、划分规则和参数语义。

---

## 2026-04-29

**用户原始需求：**
> 改一下当前的 LoRA 注入,我需要更细化地控制 LoRA 的情况：在 Attention 中支持选择 self Attention 和 Cross Attention 注入 LoRA；支持在 QKVO 中的几个层注入 LoRA；训练好的 LoRA 读取时自动识别大小。

**完成改动：**
- `src/pipeline/train_mitty.py`：新增 LoRA 目标解析，默认通过 `--lora-attn-types self,cross` 与 `--lora-attn-projections q,k,v,o` 展开为 `self_attn.q`、`cross_attn.v` 等精确注入点；`--lora-target-modules` 仍可显式覆盖到任意 PEFT target suffix，例如 `ffn.0,ffn.2`。
- `src/pipeline/train.py`、`src/pipeline/train_mitty.py`：`--lora-rank` 改为可省略；指定 `--init-lora` 时从 checkpoint 的 LoRA A/B 形状自动检测 rank，并从 checkpoint key 自动检测 target modules。
- `src/pipeline/train_mitty.py`：加载 `--init-lora` 后校验所有已注入 LoRA tensor 都来自 checkpoint，避免 target/rank 不匹配时静默随机初始化部分 LoRA。
- `src/pipeline/evaluate_mitty_models.py`：综合评估加载训练好的 LoRA 时默认自动检测 rank 和 target modules，不再需要手动传 `--lora-rank` / `--lora-target-modules`。
- `tests/test_lora_config.py`：新增 LoRA attention 控制、checkpoint rank/target 检测和 CLI args 自动填充单测。

---

**用户原始需求：**
> 修复 FID 和 FVD 的计算，并说明标准 FVD 的实现复杂度和显存占用。

**完成改动：**
- `src/core/eval_metrics.py`：在线 FID 保持 InceptionV3 pool3 帧级 Frechet 距离；FVD 改为 torchvision S3D Kinetics-400 时空视频特征，不再使用 Inception 帧特征时间均值伪 FVD。
- `src/tools/eval_metrics.py`：离线 FVD 同步改为 S3D 视频特征，FID/FVD 分别使用图像/视频特征抽取器，并在 gen/gt 帧数不一致时直接报错。
- `src/pipeline/evaluate_mitty_models.py`：综合评估入口同步加载 S3D 视频特征模型，保持与离线工具签名一致。
- `doc/step_5_training_infra.md`：记录 FID/FVD 当前口径、显存/耗时影响和小样本不稳定注意事项。

---

## 2026-04-28

**用户原始需求：**
> 修几个 bug：W&B 的 log 目录和训练名字后面加具体秒，保证目录名字不会冲突；目录中加上 LoRA 的位置，比如 ffn0ffn2 / qkvo，当前没有 LoRA 看不出来且路径名字会冲突。

**完成改动：**
- `src/core/train_utils.py`：默认 run name 时间戳从 `MMDD_HHMM` 改为 `MMDD_HHMMSS`，同一分钟内启动多个训练时不再共用目录名。
- `src/core/train_utils.py`：默认 run name 加入 LoRA target 短签名，例如 `q,k,v,o` → `qkvo`、`ffn.0,ffn.2` → `ffn0ffn2`。
- `doc/step_5_training_infra.md`、`doc/step_5_wandb_setup.md`：同步记录新的 run name 格式。

---

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
- 环境无 lpips/torchmetrics 包，用 VGG16/InceptionV3 自实现 LPIPS/FID；FVD 后续已升级为 torchvision S3D Kinetics 视频特征
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

## 2026-04-28 Smoke test 与 T5 cache 目录统一

用户要求：
> 1. T5 的 cache 目录,和 pair 目录匹配一下. 现在应该是同一个目录
> 2. smoke t32 的脚本,全部改名成 smoke test,然后数据也改成 smoke test.
> 3. 每次冒烟测试的时候,轻量和 GPU 都跑, 跑 GPU 前先看显卡情况, 再跑. 然后报告测试的时候,报告跑的是单卡还是双卡测试

落实：
- `src.pipeline.train_config` 将正式任务的 T5 cache 映射改为与 VAE/pair 数据集同名目录，例如 `pair_1s` → `training_data/cache/t5/pair_1s/`。
- 冒烟训练任务名从 `smoke_t032_e2e` 改为 `smoke_test`，临时数据迁移到 `tmp/smoke_test/`。
- `scripts/smoke_t032_refactor.py` / `scripts/smoke_t032_gpu.py` 改名为 `scripts/smoke_test_light.py` / `scripts/smoke_test_gpu.py`，新增统一入口 `scripts/smoke_test.py`。
- `scripts/smoke_test.py` 每次串行执行轻量冒烟与 GPU 冒烟；GPU 冒烟先记录 `nvidia-smi` 到 `tmp/smoke_test/gpu/nvidia_smi_before.log`，最终 summary 标明 `single-card` / `dual-card` / `N-card`。
- GPU smoke 的 eval cache 改为使用与 pair 数据匹配的 `training_data/cache/vae/pair_1s/eval/pair_0001.pth`。

## 2026-04-28 数据按 Task 组织与运行时 OOD 划分

用户要求：
> 只用三个 Task；数据按类型、duration、机器人 Task 组织；OOD 和 in-task 不在磁盘预切，训练运行时决定；支持按 seed 指定 train/eval/video 子集大小。

落实：
- `src/core/config.py` 将 canonical 数据 Task 改为 Pillow、Basket、Washing 三个 Inspire Task，并提供默认 runtime train/OOD Task 集合。
- `make_pair.py` / `make_robot_pair.py` 输出 `training_data/pair/<data_type>/<duration>/<robot_task>/`，每个 Task 目录写 `manifest.jsonl`。
- `mitty_cache.py` 输出 `training_data/cache/vae/<data_type>/<duration>/<robot_task>/`，复制 manifest 字段到 cache manifest 和 `.pth`。
- 新增 `src/pipeline/runtime_data.py`，训练时按 `--train-tasks`、`--ood-tasks`、`--data-seed`、size 参数生成 run-local split，并写入 `run_dir/data_split/`。
- `train.py` 新增 `--data-type`、`--duration`、`--train-size`、`--in-task-eval-size`、`--in-task-video-size`、`--ood-eval-size`、`--ood-video-size`、`--data-seed` 等参数。
- 新增 `scripts/migrate_task_layout.py`，用于把旧 split 目录迁移为 split-free task 目录。

## 2026-04-28 修复 blur_r2r control 视频未降质

用户要求：
> 当前这个 blur_r2r 有问题. control Video 应该是模糊过的, Video 是清晰的. 现在这个 clip 输出的都是清晰的. 你看一下 blur 的数据应该怎么做, 之前对的数据在 1s_patch 中

落实：
- `src/pipeline/make_pair.py` 的 `blur_r2r` 不再把 control 当作普通 robot clip 裁剪；现在先生成清晰 robot target，再用 `training_data/sam2_mask/` 对同一 clip 的 robot 全身区域做 Gaussian blur，写入 `control_video/`。
- blur 参数默认对齐旧 `1s_patch` blur 语义：`--blur-ksize 51`，`--blur-pixel-expand 16`；hflip 样本会同步翻转 SAM2 mask 后再降质。
- `manifest.jsonl` 为 `blur_r2r` 记录 `control_degrade=sam2_blur`、`blur_ksize`、`blur_pixel_expand`，便于 cache 与训练追溯。
- `doc/step_5_training_infra.md` 补充 blur_r2r 生成方式、SAM2 mask 依赖和正式命令。

## 2026-04-29 外观替换 LoRA 网格搜索脚本

**用户原始需求：**
> 用 `Mitty-identity_r2r_1s-10000d_r64_ffn0ffn2_1000s_0429_185108` 的 LoRA，在 cuda0/1/2 上跑外观替换搜索；LoRA 位置为 ffn/qk/vo/qkvo/ffn+qkvo，rank 为 64/128/256；使用 `flip_run_2.sh` 生成笛卡尔乘积，一个卡一个任务，跑完再下一个；不要改 Task name。

**直接修改：**
- 新增 `scripts/train_h2r_lora_grid.sh`，固定 `--task-name h2r_1s`，默认 merge `step-0900.safetensors`，按 5 个 LoRA layout × 3 个 rank 调度到 CUDA 0/1/2。

## 2026-04-30 全量 LoRA 搜索布局

**用户原始需求：**
> Cross Attention 和 FFN 都放开，改成 FFN + Cross attention qkvo + Self Attention qkvo 试一试，就是全量 LoRA；LoRA 就一种，然后三种 rank。

**直接修改：**
- 更新 `scripts/train_h2r_lora_grid.sh`，LoRA layout 从多种局部组合改为单一 `full_lora`，target modules 为 self attention q/k/v/o、cross attention q/k/v/o 和 `ffn.0,ffn.2`；rank 仍为 64/128/256。

## 2026-04-30 — 当前训练 Task 改回 Collect

**用户原始需求：**
> 数据有点问题。更新当前的 Task，把 basket 换成 Inspire_Collect_Clothes_MainCamOnly；统计 training_data/segment 下的视频数量；解释当前 clip 出来的样本数量和 segment 下的样本数量为什么对不上。

**直接修改：**
- `src/core/config.py` 将 canonical/default 训练 Task 从 Basket + Pillow + Washing 改为 Collect + Pillow + Washing；默认 in-task 为 Collect + Washing，OOD 仍为 Pillow。
- `scripts/smoke_test_gpu.py` 的 `SMOKE_TASK` 同步改为 `Inspire_Collect_Clothes_MainCamOnly`。
- `doc/step_5_training_infra.md` 与 `doc/step_5_two_stage_training.md` 同步更新当前 Task 集合、默认 in-task 说明和示例路径。
- 新增 `doc/notice.md`，记录 `seedance_direct/1s` clip 与 `training_data/segment` 全量 segment 不同口径：1 条 4s human source 会生成 14 条 1s clip，而 identity_r2r 是每条 robot segment 生成 4 条 1s clip。

**数据现状：**
- `training_data/seedance_direct/1s/Inspire_Collect_Clothes_MainCamOnly/manifest.jsonl` 已存在 112 条 1s clip，来自 8 条 4s human source。
- `training_data/pair/{h2r,blur_r2r,identity_r2r}/1s/` 与 `training_data/cache/vae/*/1s/` 当前仍缺少 Collect 目录，需要按新 Task 集合重建 pair/cache 后训练入口才能直接使用。

## 2026-04-30 — blur_r2r 改为使用全量 segment

**用户原始需求：**
> blur 那一块，使用全部的 segment 数据，不是从 Seedance 中匹配；只有 h2r 和 r2h 是从 Seedance 匹配，这一步不涉及 Human，所以是三个 Task 的全部 segment 数据。改一下代码，然后给我重新构造数据+cache 的指令。

**直接修改：**
- `src/pipeline/make_pair.py` 为 `blur_r2r` 新增 robot-only segment 枚举路径：`--task all` 展开三个 canonical Task，并直接遍历 `training_data/segment/<task>/ep*/seg*_video.mp4`。
- `blur_r2r` 的 1s 数据现在每条 4s segment 生成 4 条非重叠 robot clip；`h2r` / `r2h` 仍使用 human source 与 Seedance/overlay manifest 匹配。
- `blur_r2r` manifest 不再写入 `human_src`，继续记录 `robot_src`、`source_segment_id`、`clip_start`、`clip_dur` 和 SAM2 blur 参数。
- `doc/step_5_training_infra.md` 与 `doc/notice.md` 同步更新 blur_r2r 数据来源说明。

## 2026-05-01 — H2R/blur grid 支持多 LoRA merge 与时间戳 run name

**用户原始需求：**
> 这个应该合并两个 LoRA，一个是当前的，一个是 `scripts/train_h2r_lora_grid_cuda3_serial.sh`。然后注意 wb 的实验名字要加时间。

**直接修改：**
- `scripts/train_h2r_lora_grid_cuda3_serial.sh` 与 `scripts/train_h2r_lora_grid.sh` 支持 `MERGE_LORAS` 环境变量，接受空格或逗号分隔的多个 `.safetensors`，并展开为多个 `--merge-lora`。
- 保留旧 `MERGE_LORA` 单路径兼容；未设置 `MERGE_LORAS` 时沿用原默认 identity LoRA。
- W&B run name 追加 `RUN_TIMESTAMP`，默认格式为 `MMDD_HHMMSS`；同一轮 grid 共用同一个时间戳，并把 `run:<timestamp>` 加入 W&B tags。

## 2026-05-01 — H2R 双 LoRA qkvo stack 三卡并行脚本

**用户原始需求：**
> 改成跑一个 qkvo(self), qkvo(self+cross), qkvo(self+cross)+ffn, qkvo(self)+ffn，三卡并行 cuda012，flip_run_2 新写一个脚本，参考 `scripts/train_h2r_lora_grid_cuda3_serial.sh`，两个 lora+h2r 任务+三卡并行。

**直接修改：**
- 新增 `scripts/train_h2r_lora_qkvo_stack_cuda012.sh`，使用 `scripts/flip_run_2.sh` 在 CUDA 0/1/2 上并行调度 H2R 训练。
- 默认合并两个 LoRA：identity `step-0900.safetensors` 和 blur_r2r r256 `step-0500.safetensors`；可用 `MERGE_LORAS` 覆盖。
- 默认跑四个 layout：`qkvo(self)`、`qkvo(self+cross)`、`qkvo(self+cross)+ffn`、`qkvo(self)+ffn`；默认 rank 为 256，可用 `LORA_RANKS` 覆盖为多 rank。
- 默认 `TRAIN_SIZE=490`，匹配当前 H2R Collect + Washing runtime train pool。

## 2026-05-01 — blur 多卡 LoRA 搜索改为 qkvo 组合

**用户原始需求：**
> 改成跑一个 qkvo(self), qkvo(self+cross), qkvo(self+cross)+ffn, qkvo(self)+ffn，三卡并行 cuda012，flip_run_2。

**直接修改：**
- `scripts/train_h2r_lora_grid.sh` 保持默认 `FLIP_RUNNER=scripts/flip_run_2.sh` 与 `CUDA_DEVICES=0,1,2`。
- LoRA layout 改为 `qkvo_self`、`qkvo_self_cross`、`qkvo_self_cross_ffn`、`qkvo_self_ffn` 四种。
- `qkvo_self` 使用 `--lora-attn-types self --lora-attn-projections q,k,v,o`；`qkvo_self_cross` 使用 self+cross q/k/v/o；带 ffn 的组合显式指定对应 `self_attn.*`、`cross_attn.*` 与 `ffn.0,ffn.2` target modules。

## 2026-05-01 — H2R grid W&B 命名修正

**用户原始需求：**
> 为什么 wb 上实验是 appearance? 不应该是 h2r 吗

**直接修改：**
- `scripts/train_h2r_lora_grid.sh` 默认 `TASK_NAME` 从 `blur_r2r_1s` 改为 `h2r_1s`。
- 默认 `TRAIN_SIZE` 从 `2000` 改为当前 H2R runtime split 可用的 `490`。
- W&B run name 前缀从 `appearance_` 改为 `h2r_`，tags 从 `appearance` 改为 `h2r`。

## 2026-04-30 恢复局部 LoRA 搜索布局

**用户原始需求：**
> 全量 LoRA 效果不太行，改回去。

**直接修改：**
- 将 `scripts/train_h2r_lora_grid.sh` 的 LoRA layout 从单一 `full_lora` 恢复为 `ffn/qk/vo/qkvo/ffn_qkvo` 五种组合，rank 仍为 64/128/256。

## 2026-05-01 H2R CUDA3 串行 LoRA 搜索脚本

**用户原始需求：**
> 写一个新脚本，在 cuda3 上，使用 h2r 任务，搜索 LoRA 布局和 rank，串行跑。

**直接修改：**
- 新增 `scripts/train_h2r_lora_grid_cuda3_serial.sh`，固定默认 `--task-name h2r_1s`、`CUDA_DEVICE=3`，串行运行 `ffn/qk/vo/qkvo/ffn_qkvo × 64/128/256` 的 LoRA 搜索。

## 2026-05-01 — H2R merge identity + blur_r512 后三布局 LoRA

**用户原始需求：**
> 写一个 bash 脚本. 跑 h2r 的任务,lora 合并 Mitty-blur_r2r_1s-2000d_r512_selfattnqselfattnkselfattnvselfattnoffn0ffn2_1000s_0501_005940和这个训练使用的前一个 identity lora. 然后新加一个 LoRA, 256 维度,位置有三种 FFN qkvo(self) ffn+qkvo(self) 给出指令

**直接修改：**
- 新增 `scripts/train_h2r_lora_blur_r512_stack3_cuda012.sh`，默认使用 `scripts/flip_run_2.sh` 在 CUDA 0/1/2 上并行运行 H2R。
- 默认按顺序 merge blur_r2r r512 训练使用的 identity LoRA `Mitty-identity_r2r_1s-10000d_r64_ffn0ffn2_1000s_0429_185108/ckpt/step-0900.safetensors`，再 merge `Mitty-blur_r2r_1s-2000d_r512_selfattnqselfattnkselfattnvselfattnoffn0ffn2_1000s_0501_005940/ckpt/step-1000.safetensors`。
- 新增训练 LoRA 固定默认 rank 256，跑三种 layout：`ffn`、`qkvo(self)`、`ffn+qkvo(self)`；可通过 `MERGE_LORAS`、`BLUR_MERGE_LORA`、`IDENTITY_MERGE_LORA`、`CUDA_DEVICES`、`LORA_RANK` 覆盖。

## 2026-05-01 — mitty_h2r self-attn qkvo LoRA 256/512 双卡脚本

**用户原始需求：**
> 1. 改回去 2. 新写一个脚本 3. log 目录和wb 的目录中,名字前缀是 mitty_h2r

**直接修改：**
- 将 `scripts/train_h2r_lora_qkvo_stack_cuda012.sh` 恢复为原 stack 脚本行为：默认 `scripts/flip_run_2.sh`、CUDA 0/1/2、合并 identity + blur LoRA、四个 qkvo stack layout。
- 新增 `scripts/train_mitty_h2r_lora_self_qkvo_cuda01.sh`，使用 `scripts/flip_run.sh` 跑 H2R 数据，不传任何 `--merge-lora`，仅在 self attention 的 q/k/v/o 上加新 LoRA。
- 新脚本默认在 CUDA 0/1 上并行跑 rank 256/512，并把 W&B run name 设为 `mitty_h2r_qkvo_self_r{rank}_{timestamp}`。
- `src/pipeline/train.py` 新增 `--run-prefix`，`src/core/train_utils.py` 使用该前缀生成本地 log run 目录；新脚本默认传 `--run-prefix mitty_h2r`，使 `training_data/log/` 下目录前缀为 `mitty_h2r-...`。
- `scripts/flip_run.sh train` 改为通过固定 flip Python 执行 `torch.distributed.run`，不再依赖交互 shell PATH 中存在裸 `torchrun`。
- `doc/step_5_training_infra.md` 补充 `--run-prefix` 的命名规则。

## 2026-05-02 — LoRA layout/rank 搜索脚本整理

用户要求：
> 确定三个阶段里 LoRA 最佳位置；找出数据量相同且各 LoRA layout 都有的实验，比较性能。整理训练搜索脚本：支持指定 merge LoRA、数据量、layout/rank、CUDA；layout × rank 展开后在指定 GPU 上顺序分配；log 和 W&B 名称加日期时间；log 目录写 rank 与 layout 短名；支持 qkv-only 不含 o。

落实：
- 新增 `scripts/train_lora_grid.py`，统一展开 `LoRA layout × rank`，并通过 `scripts/flip_run.sh train --nproc 1` 顺序启动训练。
- 支持 `--merge-lora` 多 checkpoint、`--task-name`/数据覆盖、`--train-size`、eval/video size、`--layouts`、`--ranks`、`--cuda`、`--dry-run` 和额外 train 参数透传。
- 内置 `self_qkv`、`cross_qkv`、`self_qkv_cross_qkv` 等 qkv-only layout，以及 qkvo/ffn 组合 layout；本地 run dir 与 W&B run name 使用 `{task}_{layout}_r{rank}_{YYYYMMDD_HHMMSS}`。
- `src/core/train_utils.py` 将显式 LoRA target modules 压缩为 `self_qkv_cross_qkv_ffn` 这类短名，避免目录名展开过长。
- 更新 `doc/step_5_training_infra.md` 与 `doc/scripts_inventory.md` 的使用说明。

## 2026-05-02 — grid launcher 按机器 IP 选择 runner

用户要求：
> DEFAULT_FLIP_RUNNER = PROJECT_ROOT / "scripts" / "flip_run.sh" 这个地方改一下,按照机器的 ip 来. 当前机器的 ip 可能是 10.20.1.4, 如果是 .1.2 的话,就是用 flip_run_2.sh

直接修改：
- `scripts/train_lora_grid.py` 新增本机 IP 探测逻辑；默认 runner 仍为 `scripts/flip_run.sh`，但当本机 IPv4 命中 `10.20.1.2` 时自动切到 `scripts/flip_run_2.sh`。
- 保留 `--runner` 参数作为显式覆盖入口，避免特例机器需要临时指定 launcher。

## 2026-05-04 — 三阶段训练改为单 LoRA 继续训练语义

用户要求：
> 当前训练的 pipeline 想改成全程只用一个 LoRA，三个阶段任务不同，但是开放一个 LoRA 被训练；可能要改现有的 LoRA 加载代码。

直接修改：
- `src/pipeline/train.py` 和 `src/pipeline/train_mitty.py` 新增 `--continue-lora`，作为 `--init-lora` 的语义化别名，明确表示继续训练同一个 adapter；若同时传入不同的 `--init-lora` 和 `--continue-lora` 会直接报错。
- `scripts/train_lora_grid.py` 同步支持 `--continue-lora`，展开训练命令时使用语义更明确的 `--train-lora`。
- 新增 `scripts/train_three_stage_single_lora.py`，默认串行运行 `identity_r2r_1s -> blur_r2r_1s -> h2r_1s`，每个 stage 成功后把最新 checkpoint 传给下一 stage 的 `--train-lora`，不使用 `--merge-lora`。
- 文档明确区分：`--continue-lora`/`--init-lora` 是可训练 LoRA 继续训练；`--merge-lora` 是把旧 LoRA 合并进 frozen base 后再新开一个 LoRA，属于 stack 实验，不是单 LoRA 三阶段。

## 2026-05-04 — 启动时显式选择 merge LoRA 与 train LoRA

用户要求：
> 调整一下. 在启动指令的时候,选择 merge 哪些 LoRA, 训练哪些 LoRA. 训练的 LoRA 如果传入已有的 LoRA 就继续训练,如果传入参数就新建一个 LoRA 训练. 这样更灵活.

直接修改：
- `src/pipeline/train.py`、`src/pipeline/train_mitty.py` 新增 `--train-lora`、`--train-lora-rank`、`--train-lora-target-modules`。`--train-lora <ckpt>` 表示继续训练该 checkpoint；未传 `--train-lora` 时按 rank/target/attn 参数新建可训练 LoRA。
- `--merge-lora` 继续表示冻结合并，可和 `--train-lora` 同时使用；入口会拒绝 `--init-lora`、`--continue-lora`、`--train-lora` 指向不同 checkpoint。
- `scripts/train_lora_grid.py` 支持 `--train-lora`，展开命令时使用新参数名。
- `scripts/train_three_stage_single_lora.py` 支持 stage 级 `task=...;steps=...;merge=...;train=path|fresh|previous;rank=...;targets=...`，可以逐 stage 决定哪些 LoRA merge、哪套 LoRA 训练。

## 2026-05-06 — 统一显式 W&B run name 与本地 log 目录名

用户要求：
> 现在需要改一下文件夹命名,当前 wb 的时间戳和本地 log 不一致,应该是创建时间不同,这个统一一下

直接修改：
- `src/pipeline/train.py` 和 `src/pipeline/train_mitty.py` 在传入 `--wandb-run-name` 时直接复用该值作为本地 run 目录名和 W&B run name。
- grid/bash launcher 只需在外层生成一次带时间戳的 run name；训练入口不再为本地目录另取一次创建时间，避免 W&B 面板和 `training_data/log/` 后缀不一致。
- `doc/step_5_training_infra.md` 同步说明 `--wandb-run-name` 现在同时控制本地 log 目录和 W&B run name。

## 2026-05-06 — Local FID 替换黑底区域 Frechet 指标

**用户原始需求：**
> 你先实现 Local FID, 把当前那个黑色背景的方法替换成这个 Local FID

**创建的任务：**
- [041] Local FID 替换黑底区域 Frechet 指标

## 2026-05-06 — Count-based eval selection 与 Local FID 可视化输出

**用户原始需求：**
> 当前的 eval 要适配新的随机数据表；每次 eval 输入 in task 样本数量和 ood 样本数量，in task 按照比例划分，从表的最后开始读取后 k 个作为 eval，ood task 只有一个；输出 Local 的视频，patch 数据应该可以按照数据索引对应回去。

**创建的任务：**
- [042] Count-based eval selection 与 Local FID 可视化输出

## 2026-05-06 — Mitty eval metrics 并行与进度输出

用户要求：
> 我把代码改了,现在并行程度高了, 会输出中间进度了, 两个模型并行跑一下

落实：
- `src/tools/eval_metrics.py`：`process_step()` 使用可配置 `metric_workers` 并行读取视频和计算 CPU pairwise 指标；LPIPS、FID/Local FID、S3D FVD 改为跨视频合批执行，并提供阶段进度回调。
- `src/pipeline/evaluate_mitty_models.py`：新增 `--metric-workers`、`--lpips-batch-size`、`--feature-batch-size`、`--fvd-batch-size`、`--no-progress`；generation、Local crop 和 metrics 阶段默认打印进度。
- `src/core/eval_metrics.py`、`src/pipeline/train.py`、`src/pipeline/train_mitty.py`：训练时在线 eval metrics 也改为 frame/video batch，并在训练日志中打印指标阶段进度。
- 已并行跑完 qkv 与 qkvo_ffn 的 `step-1000`、`80` 个 in-task 和 `42` 个 OOD 样本评估，summary 与 Local 视频均输出到 `training_data/log/eval_h2r_80in_42ood_local_0506`。

## 2026-05-07 — 独立 mixed h2r 训练入口

用户要求：
> 看一下下一个 task，新写一个训练的入口和数据读取，跑混合数据的训练。

落实：
- 新增 `src/pipeline/runtime_mixed_h2r.py`，独立构建 original h2r + `_syn` h2r split；in-task/OOD eval 固定从 original `pair_order.jsonl` 尾部按数量选取，syn 只进入 train。
- mixed h2r 训练样本按每个 task 内 `pair_id` 升序取前 k 条；original 训练先排除 stable eval 后再取，适配后续继续追加 pair 的数据扩充方式。
- 新增 `src/pipeline/train_mitty_mixed_h2r.py`，先生成显式 split 和 run 目录内 `mixed_cache/` symlink cache，再复用 `train_mitty` 的 Mitty 训练循环。
- `src/pipeline/train_mitty.py` 抽出 parser 与参数归一化 helper，默认 CLI 行为不变，供 mixed 入口复用训练参数。
- `scripts/flip_run.sh` 增加 `train_mitty_mixed_h2r` 子命令，便于混合训练继续走统一 GPU/环境入口。
- 文档更新 `doc/step_5_training_infra.md`、`doc/scripts_inventory.md`，说明 mixed h2r 命令、稳定 eval 规则、`data_split/` 输出和隔离边界。

## 2026-05-07 — Final ours 与 Mitty 训练启动脚本

用户要求：
> 写两个脚本：一个串起来跑 final ours 三阶段训练，一个单独跑 Mitty；分别用 CUDA 0 和 CUDA 2，三阶段训练要自动接力 checkpoint 文件名。

直接修改：
- 新增 `scripts/run_final_ours_three_stage.sh`：默认 CUDA 0 单卡串行运行 identity → blur_r2r → h2r，自动读取每阶段实际生成的最新 checkpoint 并作为下一阶段 `--merge-lora` 输入。
- 新增 `scripts/run_final_mitty.sh`：默认 CUDA 2 单卡运行 Mitty h2r qkv baseline。
- 两个脚本保留 batch size、train size、eval/video size、runner、run id 等环境变量覆盖入口。

## 2026-05-07 — r2h 生成 syn pair 与独立 mixed h2r 训练入口

**用户原始需求：**
> 当前 r2h 模型训练完之后，需要用这个模型合成 Human 视频，替代 Seedance 的视频合成；在 `training_data/pair` 下生成类似 `Inspire_Collect_Clothes_MainCamOnly_syn` 的数据；训练时需要指定原始数据和 syn 数据的比例，统一训练；希望新增一部分代码，避免影响当前稳定流程。进一步明确拆成两个 task：一个做 syn pair，一个做新的数据混合 h2r 训练；混合训练单独写入口，不影响当前代码；eval 集固定沿用当前训练已使用的 80 + 42 配置，从 pair_order 尾部倒着选，syn 数据只进 training 不进 eval。再明确 robot 数据来源：已经被 Seedance 合成过的 robot 来源和自合成来源不能重叠；自合成的主要 robot 输入应从 `training_data/segment` 枚举，默认排除 Seedance 覆盖的 `ep000`-`ep003`。

**创建的任务：**
- [045] r2h 微调模型生成 syn pair 数据集
- [046] 独立混合 h2r 训练入口与稳定 eval 集

## 2026-05-08 — 明确 r2h `_syn` 续跑语义

**用户原始需求：**
> 文档里写明白，生成 h2r 数据或者使用 r2h 模型生成，指的是默认生成 `_syn` 的功能；不能覆盖以前的，每次生成是接着生成，设置上限并用 resume 跳过已有数据。

**直接修改：**
- `src/pipeline/r2h_synthesize.py`：`--resume-existing` 改为读取已有 `_syn` manifest，并把 `--num-samples` 作为目标总上限；已有数量达到上限时不改旧数据，未达到时从下一个 `pair_NNNN` 继续生成。
- `tests/test_r2h_synthesize.py`：增加覆盖已有数量超过目标上限时 manifest 保持不变的测试。
- `doc/step_5_training_infra.md`、`doc/tasks/done/045.md`：明确 `_syn` 续跑/扩充方式，不把降低 `--num-samples` 解释为覆盖重建。

## 2026-05-08 — 修复 `mitty_cache --resume` manifest 覆盖问题

**用户原始需求：**
> mixed h2r 训练请求 syn 800 时只识别到 142 条；既然 syn 数据依赖 pair，能不能不走严格校验？

**直接修改：**
- `src/pipeline/mitty_cache.py`：`--resume` 跳过已有 `pair_*.pth` 时也写入对应 cache manifest 记录，避免 manifest 只包含本次新增样本。
- 直接重建当前两个 in-task `_syn` VAE manifest：Collect 106 条、Washing 694 条；mixed h2r split 已能识别 syn 800 条。

## 2026-05-08 — final eval 多卡空闲轮询调度

**用户原始需求：**
> 把 `scripts/eval_final_step1000_missing.py --runner flip_run_2 --cuda-list 0,1,2 --execute` 换一种写法；轮询给出的卡，如果有空闲，就把下一个 eval 放上去跑。

**直接修改：**
- `scripts/eval_final_step1000_missing.py`：执行模式改为轮询 `--cuda-list` 中的 GPU，只有目标卡没有 `nvidia-smi` compute 进程且本脚本未在该卡运行 eval 时，才启动下一个 eval。
- 新增 `--poll-interval` 控制空闲检查间隔；文档同步更新多卡补跑说明。

## 2026-05-08 — r2h `_syn` 多卡队列 launcher

**用户原始需求：**
> 改一下这个脚本，生成一个指令队列，然后我指定卡，把任务依次放到这三张卡上跑。排队跑，我给出可用 cuda。

**直接修改：**
- `scripts/flip_run.sh`、`scripts/flip_run_2.sh`：新增 `r2h_synthesize` 子命令，统一设置项目环境后运行 `src.pipeline.r2h_synthesize`。
- 新增 `scripts/run_r2h_synthesize_queue.py`：按全局 `_syn` 目标上限计算每个 source task 的目标数量，生成 `queue.jsonl` / `commands.sh`，并可按 `--cuda` 列表并发调度；每张卡一次跑一个 task，结束后取队列下一项。
- `doc/step_5_training_infra.md`、`doc/scripts_inventory.md`：记录多卡队列 launcher 的 dry-run / execute 用法和日志输出位置。

## 2026-05-08 — ep000-ep003 syn 误差分析脚本

**用户原始需求：**
> 写个别的脚本，依旧合成数据，把 in task 和 ood task 的 ep0123 跑一下，我要对比这个 syn 和 Seedance 的效果，先切片，在生成，结果放到 output/syn_error_analysis/ 下；不用和 Seedance 对齐，也不用滑动窗口。

**直接修改：**
- 新增 `scripts/run_syn_error_analysis.py`：默认选择两个 in-task 和一个 OOD task 的 `ep000`-`ep003`，把 4s segment 切成 1s 非重叠 robot clip，再用 r2h checkpoint 生成 syn human 到 `output/syn_error_analysis/`。
- `scripts/flip_run.sh`、`scripts/flip_run_2.sh`：新增 `syn_error_analysis` 子命令，支持通过 `--cuda` 指定可见 GPU。
- `doc/step_5_training_infra.md`、`doc/scripts_inventory.md`：记录脚本用途、输出布局和运行命令。

## 2026-05-09 — 前景/背景 Patch FID 独立评测脚本

**用户原始需求：**
> 你参考自动评测,写一个新功能, 其中有一个 FID 是patch FID,你加一个计算 patch 之外的 FID,就是分为前景和背景两种 FID. 写一个新脚本评测, 结果输出到 output 中即可,单独脚本, 传入 log 的名称, 然后自动读取+评测,在 output/background_fid 下写结果

**直接修改：**
- `src/tools/eval_metrics.py`：新增背景 patch 选择与 `compute_background_patch_fid()`，背景 patch 定义为同一固定网格下未被前景 Patch FID 规则选中的 patch。
- 新增 `scripts/eval_background_patch_fid.py`：输入 log 名称，自动读取 `full_eval` 视频、`data_split/*.jsonl` 和 `config.json`，计算 `foreground_patch_fid` / `background_patch_fid`，结果写到 `output/background_fid/<log>/summary.csv` 和 `summary.json`。
- `doc/step_5_training_infra.md`、`doc/scripts_inventory.md`：补充脚本用法、输出位置和指标口径。

## 2026-05-11 — 忽略本地输出与草稿文件

**用户原始需求：**
> 把 eval 写到 git ignore 中；logs 也写进去；加上 cmd,draft

**直接修改：**
- `.gitignore`：在项目生成物目录列表中加入 `eval/`，避免评估结果和视频索引产物被误提交。
- `.gitignore`：加入 `logs/`，避免本地日志目录被误提交。
- `.gitignore`：加入 `cmd.sh` 与 `draft.md`，并从 Git 索引移除这两个已跟踪的本地草稿文件，保留工作区文件。
- `.gitignore`：加入 `.env` / `.env.*`，避免本地 token 配置被误提交，同时保留 `.env.example` 可被跟踪。

## 2026-05-17 — segment pipeline 展示 blur_r2r 构造

**用户原始需求：**
> `output/segment_pipeline` 里面已有 Mask，需要加上原视频，以及用这个 Mask 模糊过的视频，在 pipeline 中展示 blur r2r 数据构造过程。

**直接修改：**
- `src/pipeline/segment_pipeline.py`：中间产物新增 `00_original.mp4` 和 `08_blur_r2r_control.mp4`；后者使用 postprocess 后的 SAM2 mask，按正式 `blur_r2r` 默认参数 `blur_ksize=51`、`blur_pixel_expand=16` 做局部 Gaussian blur。
- `src/pipeline/segment_pipeline.py`：`--resume` 会检查并补齐新增原视频/模糊视频，旧目录缺少这两个产物时不再只因 human/inpaint 已存在而跳过。
- `doc/scripts_inventory.md`：同步记录 `segment_pipeline` 的新增中间产物。

## 2026-05-17 — 单独补齐 segment_pipeline 原视频与 blur 视频

**用户原始需求：**
> 写一个单独的脚本，只补 blur 和原视频即可，其他不用重跑。

**直接修改：**
- 新增 `scripts/backfill_segment_pipeline_blur.py`：枚举已有 `output/segment_pipeline/<task>/ep*/seg*/05_sam2_postproc.mp4`，读取对应 `training_data/segment/<task>/ep*/seg*_video.mp4`，只生成 `00_original.mp4` 和 `08_blur_r2r_control.mp4`。
- 脚本默认跳过已存在产物，支持 `--dry-run`、`--overwrite`、`--tasks`、`--episodes`、`--limit`，不调用 FK/SAM2/inpaint/human overlay。
- `doc/scripts_inventory.md`：补充 backfill 脚本用途与输出路径。

## 2026-05-29

**用户原始需求：**
> 用 Wan VAE 做 arm-hand IDM：输入视频输出手部+胳膊 action，接入离线 action consistency 指标，并跑小规模训练确认 action loss 是否下降。

**创建的任务：**
- [047] Wan VAE arm-hand IDM 动作一致性指标

**进展更新：**
- `t047` worktree 中新增 `src.pipeline.wan_vae_idm`，冻结 Wan VAE 训练 arm-hand Video2Action head，输出 `action.ee_action` 12 维 + `action.hand_cmd` 12 维。
- 训练输出包含 `eval_loss.csv`、`loss_curve.png`、`best_checkpoint.pt`、`best_val_predictions.csv`，并新增 `validate` 子命令复算 checkpoint 的 held-out 指标。
- 当前最好设置为 `clip_stride=0.5`、small head、`arm_loss_weight=2.0`、cosine LR、4000 step；全量 784 held-out clip 上 `total_mse=0.01109`、`arm_mse=0.02061`、`hand_mse=0.00156`，mean baseline `total_mse=0.07020`。

## 2026-05-30 — IDM 可见关节 mask 与三任务 H2R action 复算

**用户原始需求：**
> task47 做了从视频重建 action 的实验，是为了验证我们生成的视频对下游的 action 相关任务有帮助。后续需要改进：当前视频不一定有胳膊和腿部，画面外 action 预测差；第二阶段机器人 blur 使用 Mesh，可初步判断哪些关节在画面中；新的策略是每帧计算画面中关节，loss 仅计算画面中的关节，并保存 action 部分 mask。还需要用 IDM 模型评估 H2R 外观编辑 Baseline/Ours 在生成视频上是否真的学到 action，比较真实视频提取 action、生成视频提取 action 和 GT 的差距。发布 task 时只在 pick up pillow、wash machine、pick up cloth 三个任务上跑，不做其它任务。

**创建的任务：**
- [049] Wan VAE IDM 可见关节 action mask 训练
- [050] 三任务 H2R 生成视频 IDM action 复算实验

**范围约束：**
- 仅覆盖 `Inspire_Collect_Clothes_MainCamOnly`、`Inspire_Put_Clothes_into_Washing_Machine` / `Inspire_Put_Clothes_into_Washing_Machine_MainCamOnly`、`Inspire_Pickup_Pillow_MainCamOnly`。
- H2R action 复算只比较用户指定的 Baseline 与 Ours 两个 run，不扩大到其它 checkpoint 或任务。

## 2026-05-30 — 三任务 H2R 生成视频 IDM action 复算

**用户原始需求：**
> 用 IDM 模型判断 H2R 外观编辑 Baseline/Ours 是否真的学到了 action；比较真实视频提取 action、生成视频提取 action 和 GT 的差距。只在 pick up pillow、wash machine、pick up cloth 三个任务上跑，不做其它任务。

**直接修改：**
- `src/pipeline/wan_vae_idm.py`：新增 `eval-h2r` 子命令，支持用 Collect Clothes、Washing Machine、Pickup Pillow 三个 task-specific IDM checkpoint 评估两个 H2R `full_eval` run。
- `eval-h2r` 输出逐样本 action 向量、逐样本指标、按任务汇总和 Baseline/Ours delta；指标同时覆盖真实视频预测 action vs GT action、生成视频预测 action vs GT action、生成视频预测 action vs 真实视频预测 action。
- `eval-h2r` 当前只统计 `augment=normal` 的 eval records，跳过 hflip 增强样本，避免翻转视频和未翻转 action label 不一致。
- `src/pipeline/wan_vae_idm.py`：action parquet 读取扩展为 `data/chunk-*/*.parquet`，用于 Washing Machine 等多 parquet 原始数据。
- 文档更新 `doc/step_5_training_infra.md` 和 `doc/scripts_inventory.md`，记录命令、输出目录和三任务限制。
## 2026-05-30 — Wan VAE IDM 可见 action mask 训练

**用户原始需求：**
> task47 / task48 的 Wan VAE arm-hand IDM 直接对完整 24 维 action 计算 loss，但第一人称视频中不一定能看到完整胳膊或手。需要基于 robot mesh / 投影信息逐帧计算 visible joint / action mask，loss 只计算画面内可见 action 维度，并把 mask artifact 稳定保存；当前只覆盖 Collect Clothes、Washing Machine、Pickup Pillow 三个 H2R 任务。

**直接修改：**
- 新增 `src/pipeline/action_mask.py`：定义 24 维 arm-hand 和 48 维 full-body IDM action mask schema、显式 body-part 映射、artifact 读取和 clip-level mask 聚合。
- 新增 `src/pipeline/action_mask_precompute.py`：基于 G1 FK mesh 投影逐帧计算 action 相关 body part 可见性；`--target-mode arm_hand` 覆盖左右臂/手，`--target-mode full_body` 覆盖 torso、左右腿、左右臂、左右手；支持 `--clip-middle-only` 只渲染 IDM 监督会访问的中间帧，支持 `--workers` 按 segment 并行预计算；写出 `training_data/action_mask/<task>/<episode>/<seg>.npz` 和 `index.jsonl`。
- `src/pipeline/wan_vae_idm.py`：`train` / `validate` / `eval` 新增 `--target-mode`、`--action-mask-root`、`--action-mask-min-frame-ratio`、`--empty-action-mask-policy`；`arm_hand` 使用 `action.ee_action + action.hand_cmd` 24 维，`full_body` 使用 `action.robot_q_desired + action.hand_cmd` 48 维；action label 与 mask 都对齐到 17 帧 clip 的中间帧；IDM head 改为纯 3D CNN + MLP，Wan VAE latent `[B,48,5,16,16]` 经 CNN 到 `[B,256,5,8,8]` 后 spatial pool/readout/MLP 输出 action；启用 mask 后训练使用 visible 维度 masked loss，验证/eval 同时输出 unmasked 与 masked MSE、relative L2 error、visible count/ratio 和逐维 mask。
- `doc/step_5_training_infra.md`、`doc/scripts_inventory.md`：补充 action mask precompute、维度映射、masked 训练/验证/eval 参数与输出字段。

## 2026-06-01 — G1 Pick-Up-Cloth 独立 IDM 对比实验

**用户原始需求：**
> 现有 Transformer IDM 和 AdaWorld IDM 目前是在纯 Humanoid Everyday 上训练的，需要在 Unitree G1 数据集上跑一下。Transformer 可以直接训；AdaWorld 需要先跑 action Encoder，再重新训练 decoder。action 只用 arm 和 hand，loss mask 应该已经处理好了。训练没问题后，在两个指定 full_eval run 上只用 pick up cloth 任务做新数据 eval。不同数据集分开做，不要把已有脚本改太大，可以给不同数据加不同入口。

**创建的任务：**
- [062] G1 Pick-Up-Cloth 独立 IDM 对比实验

## 2026-06-01 — H1 两条 IDM 路线最佳结果整理

**用户原始需求：**
> 看一下从 t57 开始一直到最新的 task，一直在优化两个 IDM 模型，对比目前最好的结果，然后整理文档。

**直接修改：**
- `doc/h1_idm_methods.md`：补充 task057 到 task064 的 H1 IDM 任务脉络，明确 task058/task059 为视觉 baseline、task062 为 G1 独立实验，不混入 H1 held-out 排名。
- `doc/h1_idm_methods.md`：新增当前最佳结果摘要，记录 t064 RGB `motion_transformer_v2` 是当前最强 H1 IDM，t063 wider residual MLP 是当前最强 AdaWorld latent decoder，并补齐实际 worktree checkpoint / metrics 路径。
- `doc/h1_idm_methods.md`：按 `val_predictions.csv` 和 checkpoint action mean/std 复算并校正 `pred_norm_var_mean`，统一完整 H1 held-out `71486` samples 的比较口径。

## 2026-06-01 — done task 产物迁回 main 并清理 worktree

**用户原始需求：**
> 整理数据/数据路径/worktree/task 文件；tasks 文件直接提交；已经完成的 tasks，阶段性产物挪到 Main 中并更新文档；确认 worktree 中没有有用的东西且已经 done 后删除 worktree。

**直接修改：**
- 将已完成且 worktree 干净的 task057、task058、task060、task061、task063、task064 正式/阶段性产物从 `.worktrees/tNNN/` 复制回 main 工作区的 `tmp/` / `output/` 同名目录，smoke 目录不迁移。
- 更新 `doc/h1_idm_methods.md`、`doc/step_5_training_infra.md`、`doc/scripts_inventory.md`、相关 done task 文档中的产物路径，避免继续引用 `.worktrees/tNNN/tmp/...`。
- 保留 task059、task062 worktree；它们仍是 active，且 worktree 中有未提交实现改动。
- 进一步将 task051、task053、task054、task056 的正式/阶段性产物迁回 main 的 `tmp/` / `output/`，并同步收口它们在 task 文档中的产物路径。

## 2026-06-01 — G1 三任务 IDM 重训与 action-label 误差表

**用户原始需求：**
> 目前有两种 IDM 方法，在 task063 task064 中，在 G1 三个 task(in task 和 ood) 上分别重新 train，然后比较 Baseline, Transformer, ada world 的效果。最终呈现的表格是：Task, IDM Method, MSE, relevant Error。其中的数值是模型预测和 action label 的对比，其中 IDM Baseline 是 Ours, Mitty, Baseline(均值,理论最差)。

**创建的任务：**
- [065] G1 三任务 IDM 重训与 action-label 误差表

**进展更新：**
- `t065` worktree 中新增 G1 专用 `g1_pair_idm`、`g1_adaworld_action_encoder`、
  `g1_adaworld_action_decoder`、`g1_h2r_pair_idm_eval` 和
  `g1_three_task_idm_report` 入口，并接入 `scripts/flip_run.sh`。
- 在 Collect Clothes、Washing Machine、Pickup Pillow 三个 G1 task 上分别重训
  Transformer IDM 和 AdaWorld decoder；本轮 bounded 口径为 Transformer `s1000`、
  AdaWorld latent `s40k`、AdaWorld decoder `s1000`，统一使用 unmasked 24 维
  `action.ee_action + action.hand_cmd` MSE。
- 最终 H2R action-label 主表写入
  `output/g1_three_task_action_eval_t065/task065_final_idm_action_label_report.csv`，
  共 18 行，schema 为 `Task,IDM Method,MSE,relative Error`；主指标是
  `MSE(IDM(video), action_label)`，`relative Error` 以对应 task/method 的
  train-action mean baseline MSE 为分母。
- 六个 best checkpoint 均已完成 `validate --val-max-samples 2048` replay；新增 /
  修改的 G1 IDM 模块已通过 `compileall`，相关 CLI `--help` 已验证。

## 2026-06-03 — H2R SAM3 blur_r2r 三阶段复现收口

**用户原始需求：**
> 看当前 H2R 数据集能不能复现三阶段训练：step1 直接复用之前跑通的 checkpoint；stage2 用 SAM3 把 robot 模糊掉做外观训练；stage3 暂时不做，因为需要后续配对数据。

**直接修改：**
- 新增 `src/pipeline/h2r_sam3_precompute.py`：用 SAM3/SAM3.1 预计算 H2R robot-camera 的 robot-arm mask，输出 episode 级 `training_data/h2r_sam3_mask/<task>/episode_<id>.npz`。
- 新增 `src/pipeline/h2r_sam3_blur_pair.py`：消费预计算 SAM3 mask，把 H2R robot-camera 转成维护中的 Mitty `blur_r2r/1s/<task>` pair layout；target 是清晰 robot，control 是 SAM3 mask 区域 Gaussian blur 后的 robot。
- `scripts/flip_run.sh` 增加 `h2r_sam3_precompute` 和 `h2r_sam3_blur_pair` 子命令；`scripts/prepare_h2r_sam3_stage2.sh` 串联 precompute、pair、cache 和可选训练。
- `scripts/run_final_ours_three_stage.sh` 默认复用既有 step1 checkpoint，只运行 H2R SAM3 `blur_r2r` stage2，并默认 `RUN_STAGE3=0`。
- H2R SAM3 blur pair 默认尺寸改为 `224x416`，并要求 resize 维度为 32 的倍数。`240x432` 会得到 `15x27` 奇数 latent grid，Wan/Mitty 前向输出为 `14x26`，训练 loss 会形状不一致。

**验证记录：**
- H2R 数据集检查：`data/h2r/v1/video` 下共有 22 个 task、210 个 `robot_camera.mp4`；本轮 stage2 选定 `grab_cup_v1`、`grab_cube2_v1`、`push_box_random_v1` 作为 in-task，`roll` 作为 OOD。四个任务全量 dry-run 规划为 40 episodes / 353 clips。
- SAM3 smoke：GPU2 上对 `grab_cup_v1` 的 1 episode / 1 clip 预计算成功，生成 `masks (242,240,426) uint8`，17 个训练 source frame 中 12 帧有非空 mask。
- 小规模真实准备：四个任务各生成 1 个 mask、1 个 `blur_r2r` pair、1 个 VAE cache；重建后 cache latent shape 均为 `(1,48,5,14,26)`。
- stage2 训练冒烟：`MAX_STEPS=1 SAVE_STEPS=1 EVAL_STEPS=999 EVAL_VIDEO_STEPS=0`、`WANDB_MODE=offline` 跑通，成功 merge step1 LoRA、训练 1 step、保存 `training_data/log/final_ours_h2r_sam3_step2_0603_211422-blur_r2r_1s-2d_r256_self_qkvo_ffn_1s_0603_211429/ckpt/step-0001.safetensors`；step3 保持禁用。
- 全量准备完成：SAM3 mask index 为 40 episodes / 353 clips；`blur_r2r/1s` pair 和 VAE cache 行数为 `grab_cup_v1=57`、`grab_cube2_v1=90`、`push_box_random_v1=64`、`roll=142`，cache 抽样 `human_latent` / `robot_latent` shape 均为 `(1,48,5,14,26)`。
- stage2 正式复现完成：`BATCH_SIZE=1 IN_TASK_EVAL_SIZE=4 OOD_EVAL_SIZE=4 IN_TASK_VIDEO_SIZE=0 OOD_VIDEO_SIZE=0 EVAL_VIDEO_STEPS=0` 跑满 1000 steps，复用 step1 checkpoint 并 merge 180 个 rank-32 LoRA pair，stage2 训练 rank-256 `self_qkvo_ffn` LoRA；最终 checkpoint 为 `training_data/log/final_ours_h2r_sam3_step2_0603_220432-blur_r2r_1s-207d_r256_self_qkvo_ffn_1000s_0603_220442/ckpt/step-1000.safetensors`，最终 eval 为 `eval_loss_in_task=0.1478`、`eval_loss_ood=0.1658`。stage3 未启动，因为仍缺少 H2R 配对数据。

## 2026-06-03 — task067 标题术语修正

**用户原始需求：**
> 当前 tasks 里有一个 WAM 的，应该是 WAN2.2 + LoRA。

**直接修改：**
- `doc/tasks/active/067.md`：将标题中的 `WAM 原型` 改为明确的 `Wan2.2-5B + DreamZero-style LoRA 离线 video-action 原型`，避免把该任务误标为 WAM。

## 2026-06-05 — task067 Wan2.2 + LoRA WAM 单卡低显存 smoke

**用户原始需求：**
> 当前 tasks 中 WAM 应为 WAN2.2 + LoRA；先计算显存是否足够，方案应单卡不爆显存。不要多卡，尝试单卡跑。上次操作导致 Codex 崩溃，不要做危险操作。继续确认是爆内存还是机器重启，并参考 FLIP 代码，尽量不要往内存加载，直接往显卡读取。

**直接修改：**
- DreamZero 独立 checkout 中新增低显存 Wan action head adapter、H2R Hydra config 和
  `flip_experiment.py` wrapper；FLIP 仓库仅记录文档，不合入 DreamZero 运行代码。
- 低显存 adapter：DiT 使用 Wan2.2 bf16 safetensors，通过 `safe_open(..., device="cuda")`
  直接读到 GPU；frozen T5 / CLIP / VAE 不参与 Trainer 全模型 `.to(cuda)`，`.pth` 加载使用
  `mmap=True` 降低 CPU 匿名内存压力，并在 encode 后 offload。
- 本地 Wan2.2 5B DiT safetensors 不包含 DreamZero TI2V wrapper 期望的
  `cross_attn.*_img` / `img_emb` key；adapter 在 direct-GPU load 后对这条 image branch
  做确定性零初始化，避免冻结随机图像分支，并让 pretrained load missing keys 只剩新建
  action/state 模块。

**验证：**
- 机器未重启；journal 显示前一次会话是在 2026-06-04 14:48:48 被 `systemd-oomd`
  杀掉 tmux scope，属于 CPU memory pressure，不是 GPU OOM 或机器重启。
- 单卡 GPU 2 smoke：`MAX_STEPS=1`、`Batch Size=1`、`LORA_RANK=4` 完成 1 个训练 step；
  初始化 missing keys 仅为 `state_encoder`、`action_encoder` 和 `action_decoder`。
- 观测：训练前 GPU memory 日志 `10.462 GB`，DiT direct-GPU 加载瞬时显存约
  `20.7 GB`，低于 RTX 4090D 24GB；CPU RSS 未复现 oomd，训练结束后内存恢复。
- loss：`dynamics_loss_avg=0.5703880786895752`、
  `action_loss_avg=0.22939424216747284`、`train_loss=0.7997823357582092`。
- 保存：`model.safetensors` 约 89.9MB，614 个 tensor，44,890,144 个 trainable 参数；
  未生成 `model-0000*.safetensors` 全量分片。
- checkpoint 恢复 smoke 成功恢复 trainable-only checkpoint；614 个 tensor exact compare
  通过，`unexpected_keys_count=0`，所有 trainable key 均在 checkpoint 内。
- 离线 rollout smoke 成功输出 2 个连续 chunk；`seed=42`，`action_chunks` shape 为
  `(2,1,24,32)`，`final_video_latent` shape 为 `(1,48,2,10,20)`，并导出
  `output/task067_rollout_smoke_actionstate_only_v3/rollout_smoke.mp4`。

## 2026-06-09 — G1 2s30fps 切片与 stage2 blur 数据交付

**用户原始需求：**
> 交付 2s 的原始视频切片数据、Seedance 的 2s 切片数据、SAM2 分割 pipeline 产出的分割+模糊后的 2s 切片视频；第二阶段原始机器人模糊数据集切片可以直接 2s 步长切，因为数据很多。

**创建的任务：**
- [076] G1 2s30fps 切片与 stage2 blur 数据交付（已完成并合入主线）

**直接修改：**
- 新增 `src/pipeline/g1_2s_slice_data.py`：统一生成 G1 `2s61f30`
  original、Seedance direct、SAM2 blur slice，并 hardlink 成
  `identity_r2r`、`blur_r2r`、`h2r` 三类 pair layout。
- 更新 `doc/step_5_training_infra.md` 和 `doc/scripts_inventory.md`，记录
  `2s61f30` 的 61 帧/30fps 口径、tail-aligned 第二窗口、输出路径、
  stage2 blur_r2r 用法，以及后续 action/state 必须按 manifest 的
  `source_frame_indices` 回查源 parquet。
- `feat/t076-g1-2s30fps-slices` 已通过 `--no-ff` merge 合入 `main`；
  实现 commit 为 `f3b225b`，merge commit 为 `dd45bf6`。

**交付数据：**
- `training_data/slice/g1_2s61f30/original/`：10908 个 2s robot original clips。
- `training_data/slice/g1_2s61f30/seedance_direct/`：84 个 2s Seedance clips。
- `training_data/slice/g1_2s61f30/sam2_blur/`：10908 个 2s SAM2 mask blur clips。
- `training_data/pair/identity_r2r/2s61f30/`：10908 个 pair。
- `training_data/pair/blur_r2r/2s61f30/`：10908 个 pair。
- `training_data/pair/h2r/2s61f30/`：84 个 pair。

**验证记录：**
- `python -m compileall -q src/pipeline/g1_2s_slice_data.py` 通过。
- `python -m src.pipeline.g1_2s_slice_data --task all --dry-run` 规划为
  5454 个 robot segments、10908 个 robot clips、84 个 Seedance clips。
- 小样本生成 smoke 通过：每个 task 限 1 个 segment，生成 6 个 robot/SAM2 clips、
  2 个 Seedance clips 和对应 pair layout。
- 正式生成完成：manifest schema 校验确认全部样本 `fps=30`、`num_frames=61`、
  `source_frame_indices` 长度为 61；抽样视频均为 61 帧、30fps、640x480。

**当前未完成项：**
- 尚未生成 `2s61f30` 对应的 T5/VAE cache。
- 尚未启动新的三阶段训练。
- 尚未把 action/state 物化为独立 2s 标签文件；现阶段通过 manifest 中的
  `source_segment_id`、`clip_start_frame` 和 `source_frame_indices` 与源 parquet 对齐。

## 2026-06-09 — Seedance 2s 滑动窗口切片与 step layout 任务发布

**用户原始需求：**
> 写一个新的 task，用滑动窗口切 Seedance 的数据。生成的数据应该有两部分：
> 1. 已有的 origin 数据/blur 用于 Step2 训练，这部分目前没有问题，只要配对；
> 2. Seedance 新切片 + 对应的 origin 切片，因为切片方式不同，不能用原来的 origin 数据。
> 最终在 training data 中呈现 step2/origin、step2/blur、step1/origin、step1/human。
> 仅发布 task，不执行。

**创建的任务：**
- [077] Seedance 2s 滑动窗口切片与 step1/step2 数据布局重建

**状态：**
- 只创建 pending task；未实现脚本、未生成数据、未创建分支、未启动训练。

## 2026-06-13 — G1 2s Seedance 滑窗与 cache 补全任务发布

**用户原始需求：**
> 发布一个新的 task，把 G1 2s 的 Seedance 滑窗切片 / VAE Cache / T5 Cache 全部补全。

**创建的任务：**
- [078] G1 2s Seedance 滑窗切片与训练 cache 补全

**状态：**
- 创建 task078，默认不覆盖 task076 的 `2s61f30` 产物；以新的
  `2s61f30_slide` 口径收口 Seedance 滑窗、三阶段 pair layout、T5 cache 和 VAE cache。

## 2026-06-13 — G1 blur_r2r 切换到 SAM3 mask 任务发布

**用户原始需求：**
> 看一下当前 Unitree-g1 的数据和 sam3 分割的 pipeline，现在的分割 pipeline 非常复杂，我想把 g1 的 blur 也换成 SAM3。

**创建的任务：**
- [079] G1 blur_r2r 切换到 SAM3 mask

**前置结论：**
- 当前 G1 `2s61f30` blur 数据由 `src.pipeline.g1_2s_slice_data` 从
  `training_data/sam2_mask/` 生成，产物命名为 `sam2_blur`，`blur_r2r/2s61f30`
  manifest 记录 `control_degrade=sam2_blur`。
- H2R SAM3 已有清晰的两段式实现：`h2r_sam3_precompute` 显式预计算 mask，
  `h2r_sam3_blur_pair` 只消费 mask，不隐式运行 SAM3。
- G1 canonical 三任务当前合计 5454 个 segment，`2s61f30` blur pair 为 10908 条；
  G1 2s blur 需要覆盖每个 segment 的 0..119 全部 120 帧，因此 SAM3 预计算应按短
  chunk 处理并用 `covered_frames` 校验。

**状态：**
- 只创建 pending task；未实现脚本、未生成 SAM3 mask、未重建 blur 数据。
- task078 当前 active 需求仍以 SAM2 step2 blur 为基础；若本任务先实施，task078 的
  step2 blur/cache 口径需要同步切到 SAM3，避免继续生成旧 SAM2 cache。

**2026-06-13 执行记录：**
- 已认领 task079，创建 worktree `.worktrees/t079` 和分支 `feat/t079-g1-sam3-blur`。
- 新增 G1 SAM3 smoke / 预计算入口 `src.pipeline.g1_sam3_precompute`，并在
  `scripts/flip_run.sh` 中加入 `g1_sam3_precompute` 子命令。
- 按用户要求，真实 SAM3.1 运行前先检查全卡显存：
  - GPU0 约 22682/24564 MiB used，只剩约 1408 MiB free，不适合跑 SAM3。
  - GPU1/GPU2 约 3592/24564 MiB used，约 20499 MiB free。
  - GPU3 约 4482/24564 MiB used，约 19609 MiB free。
- 在 GPU1 测试 G1 2s 61 帧单 session：确认 OOM。传播到约第 16 帧时，
  SAM3 进程约占 19.28 GiB，尝试再分配 1.27 GiB 失败。
- 在 GPU1 测试 61 帧拆成 17 帧 chunk：三任务各 1 个 segment 跑通，无 OOM。
- 分割质量结论：
  - `robot arm` 聚合 `mean_nonempty_frame_ratio=0.645`、`mean_sam2_iou=0.142`。
  - `robot` 聚合 `mean_nonempty_frame_ratio=0.574`、`mean_sam2_iou=0.301`，但
    Washing Machine 任务 0/61 帧非空。
  - `robotic arm` 对 Washing Machine 只有 20/61 帧非空，IoU 0.111。
- 当前判断：SAM3.1 可以按短 chunk 跑 G1 segment，但 text-only prompt 不能干净替代
  当前 SAM2 全身 blur mask；后续不能直接全量切换，需要 point/box refine、SAM2/FK
  bbox 约束、跨 chunk 质量过滤或任务级 prompt 策略。

**2026-06-13 扩展执行记录：**
- 按用户补充要求，本轮只处理 G1 robot segment，不处理 human / H2R 数据。
- `src.pipeline.g1_sam3_precompute` 增加 `--prompt-list`，可在一次 SAM3.1 模型加载下
  sweep 多组 robot prompt；增加 `--prompt-mode text_bbox` 和 `--bbox-include-text`，
  支持 text mask 生成 bbox 后第二阶段重新跑 bbox+text prompt。
- 7 个 robot-only text prompt sweep 在 GPU1 上完成，无 OOM：
  - `robot`：聚合 `mean_nonempty_frame_ratio=0.574`、`mean_sam2_iou=0.301`，当前最好；
    Collect Clothes 过分割，Washing Machine 0/61 全空。
  - `mechanical arm`：IoU 0.188；`robot arm`：IoU 0.123；`robot hand` 非空率 0.754
    但平均面积只有 0.0052；`robot body`、`humanoid robot`、`Unitree G1 robot` 全空。
- text->bbox 二阶段完成，无 OOM：
  - 纯 bbox prompt 接口可跑，但 17 帧样本输出全空；`bbox + 同 text prompt` 才有实际 mask。
  - `robot` 二阶段聚合 IoU 0.302，基本等同 text-only 0.301，Washing Machine 仍全空。
  - `mechanical arm` 二阶段 IoU 从 0.188 提升到 0.232，但仍不能满足全身 robot blur mask 口径。
- 当前结论更新：SAM3.1 61 帧短 chunk 可运行，但纯 SAM3 prompt 与 text->bbox 二阶段都不能
  干净替代 SAM2/FK 全身 blur mask；纯 SAM3.1 在 Washing Machine 上表现一般且不稳定，
  多数 prompt 全空，能出 mask 的 prompt 也只覆盖少量局部帧。暂不应继续生成全量
  `sam3_blur` pair/cache，除非引入 SAM2/FK bbox seed、point/instance refine、跨 chunk
  质量过滤或任务级 prompt 策略。

## 2026-06-13 — H2R Seedance 机械臂转人手 smoke 入口

**用户原始需求：**
> 256 x 488 吧，这块先不用进入 WAN 的训练，目前先只管 Seedance。处理 3 段机器人视频，
> 使用 Seedance 编辑为人手：看 api pipeline 怎么用、三路并发、看 prompt 有没有问题，
> 目前的视频中只有机械臂，原来的 prompt 可能不适用了。

**直接修改：**
- 新增 `src/pipeline/h2r_seedance_edit.py`：H2R HDF5 专用 Seedance smoke 入口，
  默认导出 `grab_both_cubes_v1`、`grab_cup_v1`、`roll` 三段 `robot_camera`
  为 16:9 `864x480`、120 帧、30fps 的 Seedance reference 视频，支持三路并发调用
  API，并保存 raw Seedance 输出与 `256x488`（HxW）、120 帧、30fps 的本地 review
  输出；API key 支持从项目 `.env` 的 `ARK_API_KEY` 读取。
- 更新 `doc/step_4_seedance_api.md`：记录 H2R smoke 的入口、默认样本、尺寸口径和
  新 prompt。新 prompt 只替换可见机械臂/机械手/夹爪为人类前臂和手，不再使用
  G1 full-body prompt。

**状态：**
- 已运行 dry-run，生成 3 段 Seedance 输入视频和运行计划到
  `tmp/h2r_seedance_edit_smoke/`。
- 按用户确认将默认 prompt 收敛为：`把视频中机械臂换成人类手臂，无袖子。`
- 已使用 `.env` 中的 `ARK_API_KEY` 实际三路并发调用 Seedance，3/3 成功；
  汇总写入 `tmp/h2r_seedance_edit_smoke/seedance_summary.json`。
- 本轮 Seedance raw 输出均为 `864x496`、24fps、约 4.04s；脚本已生成
  `final/` 下 `488x256`、30fps、4.0s 的 review 视频。
- 复看原视频和本轮输出后确认失败模式：Seedance 容易在夹爪旁新增一只手，而不是覆盖
  原机械夹爪；默认 prompt 已改为“擦除机器人机械臂/黑色两指夹爪/白色机械外壳，并在
  完全相同位置替换为一只裸露人类手和前臂”，同时显式禁止额外生成手、袖子、完整人物和
  机器人残留。
- 已用新版原位置替换 prompt 重新三路并发运行 3 段 Seedance，输出到
  `tmp/h2r_seedance_edit_prompt_v2/`；3/3 成功，汇总为
  `tmp/h2r_seedance_edit_prompt_v2/seedance_summary.json`。raw 输出仍为 `864x496`、
  24fps、97 帧，final 输出为 `488x256`、30fps、120 帧。
- 用户复核后确认效果仍不理想；本轮先收口保留 smoke 入口和中间产物，但文档明确记录：
  Seedance 直接 H2R robot2human / 机械夹爪转人手当前效果不好，暂不进入正式配对数据生成或
  Wan 训练。

## 2026-06-13 — H2R Seedance SAM3 红色 mask 引导 smoke

**用户原始需求：**
> 换一种方式, 目前有 SAM3 分割 human2robot 中机械臂的 pipeline, 你尝试先 SAM3 分割,
> 然后 Seedance prompt 改成, 将图中红色部分替换为 xxxx

**直接修改：**
- 新增 `src/pipeline/h2r_seedance_sam3_edit.py`：消费 H2R SAM3/SAM3.1 episode 级
  `masks` / `covered_frames` npz，读取 HDF5 `cam_data/robot_camera` 的三段 120 帧
  30fps 视频，把 SAM3 mask 区域染成红色后作为 Seedance reference video。
- 默认 prompt 改为“图中红色半透明区域标出了需要编辑的机器人机械臂、机械夹爪和机械外壳；
  请只将红色部分替换为真实裸露的人类手和前臂，无袖子”，并显式要求非红色区域完全不变、
  红色标记消失、禁止额外生成手或保留机械残留。
- 更新 `doc/step_4_seedance_api.md`、`doc/scripts_inventory.md` 和 `doc/notice.md`，
  记录 SAM3 红色 mask 引导 smoke 的命令、输入输出、prompt、产物路径和使用限制。

**状态：**
- 使用 `scripts/flip_run.sh h2r_sam3_precompute --cuda 1` 对
  `grab_both_cubes_v1`、`grab_cup_v1`、`roll` 的 `episode_0` 前 4 个 1s clip 运行
  SAM3.1 `robot arm` 分割，mask 输出到
  `tmp/h2r_seedance_sam3_red_edit/sam3_mask/`。
- 已 dry-run 生成三段 `864x480`、120 帧、30fps 的原图 reference、红色 mask reference 和
  mask review 视频；`prepared_inputs.jsonl` 记录每段 `mask_nonzero_frames`、mask 面积和
  30fps 到 SAM3 `covered_frames` 的最近邻映射 gap。
- 已使用 `.env` 中 `ARK_API_KEY` 三路并发调用 Seedance，3/3 成功；
  汇总写入 `tmp/h2r_seedance_sam3_red_edit/seedance_summary.json`，`elapsed_sec=314.5`。
- Seedance raw 输出均为 `864x496`、24fps、约 4.04s；本地 `final/` 已统一后处理为
  `488x256`、30fps、120 帧。
- 额外生成 `tmp/h2r_seedance_sam3_red_edit/_review/*_original_red_final_compare.mp4`，
  三列展示 `original | sam3 red input | seedance final`，供人工复核显式红色区域 prompt
  是否改善替换质量。

**2026-06-13 追加：简化 prompt 单条 smoke**
- 按用户要求，将 `src.pipeline.h2r_seedance_sam3_edit` 默认 prompt 简化为：
  `把视频中红色部分替换为人手，保持动作和交互不变。`
- 入口样本数限制从“必须 3 条”放宽为“至少 1 条”，便于用 `--sample` 跑单条 prompt
  smoke，同时不影响默认三条样本。
- 已用简化 prompt 跑通单条 `grab_both_cubes_v1:0:0`，输出到
  `tmp/h2r_seedance_sam3_red_simple_prompt_one/`；`seedance_summary.json` 记录
  `ok=1`、`failed=0`、`elapsed_sec=255.8`。raw 输出为 `864x496`、24fps、约 4.04s，
  final 输出为 `488x256`、30fps、120 帧，并生成 `_review/` 三列并排视频供人工核对。

**2026-06-13 追加：裸露手臂 prompt 单条 smoke**
- 按用户要求，将 `src.pipeline.h2r_seedance_sam3_edit` 默认 prompt 更新为：
  `把视频中红色机械臂更换成人的手臂, 手臂裸露无袖, 与机械臂动作一致, 保持背景不变`
- 单条输出目录使用 `tmp/h2r_seedance_sam3_red_bare_arm_prompt_one/`，避免覆盖上一版
  `simple_prompt_one` 结果。
- 已跑通单条 `grab_both_cubes_v1:0:0`，`seedance_summary.json` 记录 `ok=1`、
  `failed=0`、`elapsed_sec=211.1`。raw 输出为 `864x496`、24fps、约 4.04s，
  final 输出为 `488x256`、30fps、120 帧，并生成 `_review/` 三列并排视频供人工核对。

**2026-06-13 追加：按 H2R task 中文动作名生成三条 prompt**
- 按用户要求，将 `src.pipeline.h2r_seedance_sam3_edit` 改为默认 task-specific prompt：
  模板为 `把视频移动的红色装置替换成裸露的人类胳膊。保持背景不变，人手和红色装置的动作轨迹保持一致。人类手臂{task_name}`。
- 当前三条默认样本的中文动作名为：
  `grab_both_cubes_v1=抓起物块。`、`grab_cup_v1=抓起杯子。`、`roll=滚动物体。`。
- 已三路并发跑通默认三样本，输出到 `tmp/h2r_seedance_sam3_red_task_prompts/`；
  `seedance_summary.json` 记录 `ok=3`、`failed=0`、`elapsed_sec=184.0`。三条 raw
  输出均为 `864x496`、24fps、约 4.04s；final 输出均为 `488x256`、30fps、120 帧；
  `_review/` 下生成三条三列并排视频和 `task_prompts_contact_sheet.jpg`。

## 2026-06-13 — G1 2s Seedance 滑窗与三阶段 cache 补全

**用户原始需求：**
> 发布一个新的 task，把 G1 2s 的 Seedance 滑窗切片 / VAE Cache / T5 Cache 全部补全。
> build

**交付状态：**
- 已完成并合入 `main`：实现 commit `e82a420`，merge commit `f57b377`。
- 任务文档为 `doc/tasks/done/078.md`，开发 worktree 为 `.worktrees/t078`，
  分支为 `feat/t078-g1-2s-seedance-cache`。
- 新增 `src.pipeline.g1_2s_seedance_slide_data`，以新的 duration label
  `2s61f30_slide` 和 `training_data/g1_2s61f30_seedance_slide/` step layout
  生成数据，不覆盖 task076 的 `2s61f30`。
- 注册三阶段训练 preset：
  `identity_r2r_2s61f30_slide`、`blur_r2r_2s61f30_slide`、
  `h2r_2s61f30_slide`。
- 扩展 `src.pipeline.mitty_cache`：identity 同文件输入/目标只做一次 VAE encode；
  `blur_r2r` 可通过 `--target-cache-dir` 复用 identity 的 `robot_latent`。

**最终数量：**
- Step2 origin / blur：各 10908 条，三任务分布为 Collect 1186、Pillow 1928、
  Washing 7794。
- Step1 origin / human：各 210 条，三任务分布为 Collect 40、Pillow 15、Washing 155；
  默认 0.5s stride 起点分布为 `{0: 42, 15: 42, 30: 42, 45: 42, 59: 42}`。
- Pair / VAE cache：
  - `identity_r2r/2s61f30_slide`：10908。
  - `blur_r2r/2s61f30_slide`：10908。
  - `h2r/2s61f30_slide`：210。
- T5 cache：`identity_r2r`、`blur_r2r`、`h2r` 均各有 1 个 positive prompt cache
  和 `negative.pth`。

**验证：**
- `compileall` 覆盖 `g1_2s_seedance_slide_data.py`、`train_config.py`、
  `mitty_cache.py` 通过。
- slide 数据入口 dry-run、小样本 smoke、全量数据 manifest/video 校验通过。
- 最终 cache 校验通过：T5 embedding shape 为 `(1,512,4096)`；VAE latent shape 为
  `(1,48,16,30,40)`；pair/cache manifest 与 `.pth` 文件数量一致；identity 抽样
  `human_latent == robot_latent`；blur/Washing tail 抽样 target latent 与 identity
  cache 一致。

## 2026-06-13 — G1 2s 数据接入 Diffusion Policy

**用户原始需求：**
> 你看一下当前 Diffusion Policy 的训练.
> 现在有新的 2s 机器人数据切片, 你看一下当前 2s 的数据如何和 robot state/robot Action 对齐, 我需要在这些新数据上训练 Diffusion Policy

**结论与实现：**
- 现有 `src.pipeline.h2r_diffusion_policy` 原本只支持 H2R HDF5 数据；
  本次新增 `--dataset-kind g1_2s_pair`，可直接读取 G1 `2s61f30_slide` pair
  manifest 训练同一个 action-only Diffusion Policy。
- 默认 G1 训练数据源设为
  `training_data/pair/identity_r2r/2s61f30_slide`，即真实 robot→真实 robot 的
  10908 条 robot-only clips；`h2r/2s61f30_slide` 只有 210 条 human→robot pair，
  后续更适合作为固定 label 的 video override / eval 数据。
- 对齐口径已修正：manifest 的 `source_frame_indices` 是 segment video 内的局部帧号，
  不是 episode 全局 `frame_index`。训练时先按局部帧号读取
  `training_data/segment/<task>/<episode>/<seg>_joints.parquet`；再用 joints 行中的
  全局 `frame_index` 回查
  `data/unitree_G1_WBT/G1_WBT_<task>/data/chunk-*/*.parquet` action。
- 默认 state 为 `observation.state.robot_q_current(36) + observation.state.hand_state(12)`，
  默认 action 为 `action.robot_q_desired(36) + action.hand_cmd(12)`，二者均为 48 维。
- G1 hand command 存在近常量维度；G1 分支默认 `--g1-norm-std-floor 0.01`，并写入
  checkpoint config / dataset summary，避免极小 std 放大训练输入。

**验证：**
- `python -m compileall -q src/pipeline/h2r_diffusion_policy.py` 通过。
- 单任务 Inspect 通过：
  `tmp/g1_dp_2s_inspect_collect.json`，Collect clips `1186`，batch shapes 为
  video `[4,2,3,64,64]`、state `[4,2,48]`、action `[4,8,48]`。
- 三任务 Inspect 通过：
  `tmp/g1_dp_2s_inspect_all.json`，clips `10908`，episode split 后
  train/val clips 为 `8796/2112`。
- 12-step G1 smoke 训练通过：
  `tmp/g1_dp_2s_smoke_floor1e2_lr3e4`，最终 eval step 输出
  `denoise_loss=1.001034`、`sampled_action_mse_norm=1.891784`。
- Checkpoint 恢复 eval 通过：
  `tmp/g1_dp_2s_smoke_floor1e2_lr3e4/eval_last`，
  `denoise_loss=0.960461`、`sampled_action_mse_norm=1.897220`。

**后续长训准备更新：**
- 明确当前 `h2r_diffusion_policy` 默认仍是 H2R HDF5；Unitree G1 2s 训练必须显式使用
  `--dataset-kind g1_2s_pair`，否则不会读取
  `training_data/pair/identity_r2r/2s61f30_slide`。
- Eval / train-log 的采样指标新增反归一化 action 空间统计：
  `sampled_action_mse_action`、`sampled_action_rmse_action`、
  `sampled_action_relative_l2_action` 和 per-horizon relative L2；
  `predictions.csv` 逐样本写出 `mse_action`、`relative_l2_action`。
  之后参数试验同时看 `sampled_action_mse_norm` 和 action 空间相对误差。
- 训练新增 `--best-metric`；默认仍为 `denoise_loss`，G1 长训若要按真实 action 空间误差
  选 best checkpoint，应设置 `--best-metric sampled_action_relative_l2_action` 并开启
  `--eval-sample-actions`。
- Action sampling eval 新增 `--eval-sample-seed`，默认 `12345`，用于固定 DDPM 采样噪声，
  让同一 checkpoint 的 normalized MSE 与 action-space 相对误差可复现。

**2026-06-13 长训/调参进展：**
- 当前训练已切到 Unitree G1：所有新 run 的 dataset summary 均为
  `format=g1_2s_pair_manifest`，数据源为
  `training_data/pair/identity_r2r/2s61f30_slide`，三任务 `10908` clips，
  state/action 均为 48 维。
- 小试结果：
  - `diffusion_steps=16/dropout=0.05/lr=3e-4` 不占优，160-step relative L2
    约 `0.6987`。
  - `diffusion_steps=8/dropout=0/lr=3e-4` 更好，512 train / 128 val 的 step80
    relative L2 约 `0.6605`。
  - 同配置 `lr=1e-4` 没有明显收益，best step80 relative L2 约 `0.6607`。
- 已按较优参数加到 1000 steps：
  `tmp/g1_dp_2s_long_g1_s1000_h128_d4_diff8_drop0_lr3e4_m4096_norm5k`。
  训练集 4096 samples、验证集 512 samples、frame stride 4、norm 5000 frames；
  train summary final `sampled_action_mse_norm=2.5580`、
  `sampled_action_mse_action=0.12145`、`sampled_action_relative_l2_action=0.66357`。
- 1000-step last checkpoint 恢复 eval 通过；同一 checkpoint / `--eval-sample-seed 12345`
  连跑两次 eval 完全一致：
  `denoise_loss=1.003447`、`sampled_action_mse_norm=2.788211`、
  `sampled_action_relative_l2_action=0.682433`。`predictions.csv` 已含
  `mse_action`、`relative_l2_action` 和 `source_id`。

## 2026-06-13 — H2R Seedance SAM3.1 人手评估与 dark gripper marker 实验

**用户原始需求：**
> 1. 你在分析视频的时候, 可以使用 SAM3.1 进行分割+统计像素, 目前项目中有 SAM3.1, 然 SAM3.1 分割人手,搭配之前标红用的分割, 就能比较出来
> 2. 实验控制在 20 个左右, 可以调用 API,

**实现与实验：**
- 新增 `src.pipeline.h2r_seedance_sam3_eval`：读取 Seedance `seedance_results.jsonl`，
  用 SAM3.1 prompt `human hand,bare human hand,human fingers` 分割 final 视频中的人手，
  与 Seedance 输入侧 `target_mask_video_path` 对齐统计
  `target_covered_by_hand_ratio`、`hand_on_target_ratio`、`mean_iou` 等指标，并写出
  `sam3_hand_green_target_red_contour.mp4` overlay 供人工复核。
- `scripts/flip_run.sh` 新增 `h2r_seedance_sam3_eval` 子命令，使用 `sam3` conda 环境运行
  评估入口。
- `src.pipeline.h2r_seedance_sam3_edit` 从固定红色全臂 marker 扩展为通用 marker-mask
  smoke：支持 `--mask-filter full|dark`、`--marker-color red|magenta|cyan|yellow|green`、
  `--annotation-mode fill|outline|bbox`。默认 prompt 模板改为使用 `{marker_desc}` 和
  `{task_name}`，避免 marker 颜色/形态变化后 prompt 仍写死“红色装置”。
- `src.pipeline.h2r_sam3_precompute` 新增 `--prompt-frame-position first|middle|<int>`；
  可在每个 17 帧 SAM3.1 clip 中选择非首帧做 text prompt。末帧 prompt 不作为稳定路径，
  因为 SAM3.1 从最后一帧反向传播时会出现缓存缺失异常。
- `src.pipeline.seedance_gen` 新增环境变量 `ARK_REQUEST_TIMEOUT`，默认从 30s 提高到
  120s，避免 Seedance 创建任务请求慢时被本地 urllib 超时误判失败。

**运行结果：**
- 先用 SAM3.1 `robot gripper` / `black gripper` prompt 分割夹爪，first-frame 与
  middle-frame 两轮 smoke 的目标 mask 均为空；当前放弃直接 text prompt 夹爪分割。
- 改用已有 `robot arm` mask 与暗像素相交：`--mask-filter dark --dark-threshold 80`。
  dry-run 的 target 平均面积约为 `grab_both_cubes_v1=3774px`、
  `grab_cup_v1=2940px`、`roll=1381px`，三条样本均有 100+ 非空帧。
- API 实验控制在 20 次以内：第一批 15 次 Seedance 成功生成，另有 3 次创建任务请求在
  30s timeout 后失败且没有返回 task id；后续沿黄/紫框方向追加 2 次单样本成功生成，
  总成功数为 17。第一批成功实验目录为：
  `tmp/h2r_seedance_sam3_exp01b_dark_magenta_fill/`、
  `tmp/h2r_seedance_sam3_exp02_dark_cyan_fill/`、
  `tmp/h2r_seedance_sam3_exp03_dark_yellow_outline/`、
  `tmp/h2r_seedance_sam3_exp04_dark_yellow_bbox/`、
  `tmp/h2r_seedance_sam3_exp05_dark_magenta_bbox/`。
- SAM3.1 评估汇总写入 `tmp/h2r_seedance_sam3_experiment_summary.csv`；对应评估目录为
  `tmp/h2r_seedance_sam3_eval_exp01b_dark_magenta_fill/`、
  `tmp/h2r_seedance_sam3_eval_exp02_dark_cyan_fill/`、
  `tmp/h2r_seedance_sam3_eval_exp03_dark_yellow_outline/`、
  `tmp/h2r_seedance_sam3_eval_exp04_dark_yellow_bbox/`、
  `tmp/h2r_seedance_sam3_eval_exp05_dark_magenta_bbox/`。
- 三样本平均指标中，`exp04_dark_yellow_bbox` 最好：
  `target_covered_by_hand_ratio=0.507`、`hand_on_target_ratio=0.207`、
  `mean_iou=0.156`、`mean_hand_area_px=6893.5`。
  `grab_cup_v1` 单条最好的是 `exp01b_dark_magenta_fill`。

**当前结论：**
- dark gripper target + marker 显著优于全臂红色 baseline；全臂 baseline 的平均
  `target_covered_by_hand_ratio` 只有 `0.062`。
- 但当前最佳的 `hand_on_target_ratio` 仍偏低，说明 Seedance 输出里可能还有额外人手、
  手臂过大或 SAM3.1 hand false positive。该路线暂不能作为训练数据生产路径。
- 人工优先复核：
  `tmp/h2r_seedance_sam3_eval_exp04_dark_yellow_bbox/*/sam3_hand_green_target_red_contour.mp4`
  以及
  `tmp/h2r_seedance_sam3_eval_exp01b_dark_magenta_fill/grab_cup_v1_ep000000_f000000/sam3_hand_green_target_red_contour.mp4`。

**2026-06-13 追加：回到黄框/紫框方向**
- 用户反馈此前黄色和紫色方框效果相对更好，要求继续沿这个方向尝试。
- 先验证一个非 bbox 方向：`exp06_dark_skin_fill_one` 把暗色夹爪区域预填成肤色，只跑
  `grab_both_cubes_v1` 单条。Seedance API 成功，但 SAM3.1 hand eval 为 0：
  `target_covered_by_hand_ratio=0.0`、`hand_on_target_ratio=0.0`、`mean_iou=0.0`。
  结论：肤色预填不是有效方向。
- 新增 `src.pipeline.h2r_seedance_sam3_edit` 的实验选项：
  `--mask-filter distal_dark`、`--distal-max-area`、`--distal-max-aspect`、
  `--distal-temporal-weight`，以及 `--marker-color skin`。`distal_dark` 能缩小暗色区域，
  但 dry-run contact sheet 显示个别帧仍会跳到桌面黑色长条或线缆，因此暂不烧 API。
- 沿黄框方向跑 `exp07_dark_yellow_bbox_strict_prompt`：保持 `dark + yellow bbox`，
  但 prompt 明确“黄色方框只是定位标记，输出不要保留方框，只在方框内删除黑色两指夹爪”。
  三条 API 3/3 成功，评估为明显负结果：
  平均 `target_covered_by_hand_ratio=0.0017`、`hand_on_target_ratio=0.0021`、
  `mean_iou=0.0008`。
- 结论更新：继续优先黄/紫 `bbox`，但不要把 prompt 收得太硬；当前最佳仍是
  `exp04_dark_yellow_bbox` 的自然描述 prompt。下一轮若继续烧 API，应小步改
  bbox 尺寸、颜色或任务动作短语，而不是改成肤色预填或强约束“只在方框内”。

**2026-06-13 追加：黄框放大与紫/黄双框验证**
- 新增 `src.pipeline.h2r_seedance_sam3_edit --annotation-mode fill_bbox|dual_bbox`：
  `fill_bbox` 可填充目标并叠加第二颜色方框，`dual_bbox` 只画两层方框，例如
  `--marker-color magenta --bbox-marker-color yellow` 生成紫色内框 + 黄色外框；
  新增 `--secondary-bbox-extra-pixels` 控制第二层框外扩距离。`dual_bbox` 必须显式设置
  `--bbox-marker-color`，否则直接报错。
- 先离线生成 contact sheet：
  `tmp/h2r_seedance_sam3_bbox_candidate_contact_sheet.jpg` 与
  `tmp/h2r_seedance_sam3_dual_bbox_contact_sheet.jpg`。紫色填充 + 黄色框会把白色机械臂也染色，
  风险较高；紫/黄双框不填充，输入看起来更干净。
- API 低成本验证 2 条单样本 `grab_cup_v1`：
  `tmp/h2r_seedance_sam3_exp08_yellow_bbox_big_cup/` 与
  `tmp/h2r_seedance_sam3_exp09_dual_bbox_cup/`，评估目录分别为
  `tmp/h2r_seedance_sam3_eval_exp08_yellow_bbox_big_cup/`、
  `tmp/h2r_seedance_sam3_eval_exp09_dual_bbox_cup/`。
- 自动评估更新版汇总写入
  `tmp/h2r_seedance_sam3_experiment_summary_updated.csv`。`exp08_yellow_bbox_big_cup`
  的 `target_covered_by_hand_ratio=0.006`、`hand_on_target_ratio=0.004`、`mean_iou=0.002`；
  视觉 contact sheet 中能看到清楚人手，但动作位置偏离原夹爪目标。
  `exp09_dual_bbox_cup` 的 hand eval 全为 0，视觉上基本保留机器人。
- 结论：不要继续扩展“更大黄框”或“紫/黄双层框”；当前黄/紫框方向仍以单色
  `exp04_dark_yellow_bbox` / `exp05_dark_magenta_bbox` 为基线。如果继续调用 API，
  应回到单色 bbox 并小幅改自然语言动作短语，而不是增加复杂图形标记。

**2026-06-13 最终收口：回退到当前最优**
- 用户要求停止继续探索并回退到当前最优方案。当前保留的最优 Seedance 方案为
  `exp04_dark_yellow_bbox`：SAM3/SAM3.1 `robot arm` mask + 暗像素过滤夹爪 +
  单色黄色 bbox + task-specific 自然 prompt。
- ROI 放大方向只完成了 dry-run 输入检查，产物为
  `tmp/h2r_seedance_sam3_roi_yellow_bbox_dryrun/` 和
  `tmp/h2r_seedance_sam3_roi_input_contact_sheet.jpg`。检查发现部分帧的 ROI 黄框会框到白色
  机械臂或整块 crop，目标不够干净，因此没有调用 Seedance API，也不纳入最终方案。
- 未验证的 ROI 代码已回退；最终交付只保留已跑通并量化评估过的全画面
  `h2r_seedance_sam3_edit.py`、`h2r_seedance_sam3_eval.py`、SAM3 mask 预计算改动和
  Seedance request timeout 改动。

## 2026-06-13

**用户原始需求：**
> 把其他参数在实验计划中补全，保持 `- 参数:` 的形式；启动第二阶段的训练，打开 eval Video。

**直接修改：**
- 更新 `实验计划.md`：补齐三阶段外观编辑训练的通用参数、W&B 参数、eval video 参数、LoRA target modules、stage 间 `merge-lora` 连接方式和第一阶段已完成 ckpt 路径。
- 第二阶段计划使用空闲单卡、`batch size: 2`、`eval-video step: 50`、W&B 视频上传开启，并显式关闭 Frechet 在线指标以降低 2s/rank256 训练的显存风险。

## 2026-06-14

**用户原始需求：**
> 训练完了；之后训练 eval 时候加上 FID FVD；先测一下刚才训练的 FID FVD。

**直接修改：**
- 更新 `实验计划.md`：将通用 `eval-video-frechet-metrics` 改为 `true`，后续三阶段训练 eval video 默认开启 FID/FVD。
- 标注第二阶段已完成 run 的 FID/FVD 需要用离线方式补算，避免误认为本次训练时已在线记录 Frechet 指标。

**运行结果：**
- 用 CPU 离线补算第二阶段 run 的 `step-2000` 全局 FID/FVD，输出到
  `training_data/log/appearance_edit_2s_s2_blur_bs2-blur_r2r_2s61f30_slide-8972d_r256_self_qkvo_ffn_2000s_0613_223130/eval_metrics_fid_fvd_step2000_cpu.csv`。
- `in_task`: `FID=35.03`、`FVD=16.12`、`n_samples=4`。
- `ood`: `FID=79.30`、`FVD=37.17`、`n_samples=2`。

**用户原始需求：**
> 先只跑前四个任务，dryrun 确定没问题之后，调用 API 跑一下，输出到 tmp/。

**直接修改：**
- `src.pipeline.h2r_sam3_precompute` 新增 `--clip-starts-file`，支持显式指定 1s SAM3
  clip start frames，用于 H2R Seedance 4s 分段和末尾倒数 4s 片段的 mask 覆盖。
- `src.pipeline.h2r_seedance_sam3_edit` 补齐 H2R 19 个 task 的中文动作短语，避免批量
  Seedance 运行时 task-specific prompt 因缺少 `{task_name}` 直接失败。
- 新增 `tmp/h2r_seedance_batch4_clip_starts.json`，记录本轮前四个任务、每任务一个 episode
  的 4s/tail-aligned Seedance 批量输入所需的最小 1s SAM3 clip starts。
- 更新 `doc/scripts_inventory.md` 记录该参数用途。

## 2026-06-14

**用户原始需求：**
> 合并前两阶段的 LoRA，跑外观编辑训练；训练 eval 带 FID/FVD，in-task/OOD eval 样本数量都是 8；先确定训练参数、指令、数据集和 train/eval 划分。

**直接修改：**
- 更新 `实验计划.md`：第三阶段 `h2r_2s61f30_slide` 将 `in task video` 和 `ood task video` 都设为 `8`，保证在线 FID/FVD 使用 8 条 in-task 视频和 8 条 OOD 视频计算。
- 记录本轮第三阶段训练的实际 runtime split：`train=187`、`in_task_eval=8`、`ood_eval=8`，其中 in-task eval 分配为 `Collect=2`、`Washing=6`，OOD eval 为 `Pickup_Pillow=8`。

**运行结果：**
- 已在空闲 GPU 1 启动第三阶段训练，run 为
  `training_data/log/appearance_edit_2s_s3_h2r_bs2-h2r_2s61f30_slide-187d_r96_self_qkvo_ffn_2000s_0614_162834`，W&B run id 为 `5d9p942j`。
- 训练参数确认：`batch-size=2`、`rank=96`、`max-steps=2000`、`save/eval/eval-video=50`、合并 identity `step-0500` 与 blur `step-2000` 两个 LoRA，在线 `eval-video-frechet-metrics=True`。
- step=1 已完整跑过 8 条 in-task 和 8 条 OOD eval video，并写入 FID/FVD：in-task `FID=198.6684`、`FVD=141.1637`；OOD `FID=171.7567`、`FVD=146.1568`。
- Frechet 指标阶段观察到 GPU 1 显存约 `23304 MiB / 24564 MiB`，训练随后继续到后续 step，未在 step=1 OOM。

## 2026-06-15

**用户原始需求：**
> 看一下 Mitty 的训练，直接使用 2s 的数据 + 96 LoRA 训练，不分三阶段；eval 间隔调成 200 step。

**直接修改：**
- 更新 `实验计划.md`：新增 `direct Mitty` baseline，使用 `h2r_2s61f30_slide`、fresh rank96 LoRA、无 `merge-lora`、`batch-size=2`、`max-steps=2000`，并将 `save/eval/eval-video` 间隔统一设为 `200`。
- 该 baseline 沿用当前 2s h2r runtime split：`train=187`、`in_task_eval=8`、`ood_eval=8`，eval video 也是 `8/8`，训练时继续开启 FID/FVD。

**运行结果：**
- 已在空闲 GPU 1 启动单阶段 direct Mitty 训练，run 为
  `training_data/log/appearance_edit_2s_mitty_direct_h2r_bs2-h2r_2s61f30_slide-187d_r96_self_qkvo_ffn_2000s_0615_184419`，W&B run id 为 `e730tnj6`。
- 启动参数确认：`merge_lora=None`、fresh rank96 LoRA、`save/eval/eval-video=200`、`eval_video_frechet_metrics=True`。
- step=1 普通训练已跑通，`train_loss=0.6571`；训练脚本会固定在 step=1 做一次初始 eval/video，之后按 200 step 间隔评估。

## 2026-06-17

**用户原始需求：**
> 看一下 Mitty 的训练，把 LoRA 改成仅 QKV 的位置加 LoRA，再训练一个；其他参数和之前的 `training_data/log/appearance_edit_2s_mitty_direct_h2r_bs2-h2r_2s61f30_slide-187d_r96_self_qkvo_ffn_2000s_0615_184419` 保持一致。

**直接修改：**
- 更新 `实验计划.md`：新增 `direct Mitty QKV LoRA` 对照实验，明确只把 `LoRA target modules` 改为 `self_attn.q,self_attn.k,self_attn.v`，其余数据、rank、batch、step、eval、FID/FVD 与 direct Mitty baseline 保持一致。

**运行结果：**
- 启动前用 `scripts/flip_run.sh nvidia-smi` 确认 GPU 1/2/3 空闲，选择 GPU 1。
- 已在 GPU 1 启动 QKV-only direct Mitty 训练，run 为
  `training_data/log/appearance_edit_2s_mitty_direct_qkv_h2r_bs2-h2r_2s61f30_slide-187d_r96_self_qkv_2000s_0617_233623`，W&B run id 为 `yq5po4dq`。
- 启动参数确认：`merge_lora=None`、fresh rank96 LoRA、`lora_target_modules=self_attn.q,self_attn.k,self_attn.v`、`save/eval/eval-video=200`、`eval_video_frechet_metrics=True`，split 仍为 `train=187`、`in_task_eval=8`、`ood_eval=8`。
- 启动检查已跑通：`step=1` 的 `train_loss=0.1535`、`eval_loss_in_task=0.2268`、`eval_loss_ood=0.2508`；step=1 的 eval video/FID/FVD 后续继续在后台执行。

## 2026-06-18

**用户原始需求：**
> `seedance_workspace/video/` 下的视频 R/B 通道反了，全部换一下；通道反转后再试一次 Seedance，请求和上一轮相比只有原视频不同。

**直接修改：**
- 对 `seedance_workspace/video/` 下普通 mp4 做 R/B channel swap，先在同目录保留
  `*.before_rb_swap.mp4` 备份；本轮候选 26 个 mp4，其中 25 个完成覆盖修正，1 个此前已修正的人手
  `human_camera.mp4` 被检测为已 swap 并跳过。
- 生成修正记录 `seedance_workspace/video/rb_swap_manifest_20260618_003257.json`。
- 复跑 Seedance `exp04`，沿用 `exp03` 的 prompt、参考图片 URL、模型、ratio、duration 和 resolution，
  只把 `reference_video` 换成修正 R/B 后的 original robot 视频。

**运行结果：**
- `exp04` task id：`cgt-20260618003408-djbts`。
- 输出视频：
  `seedance_workspace/output/exp04/grab_both_cubes_v1_ep000001_f000000_seedance_raw.mp4`。
- 输出规格：`864x496`、`24fps`、`4.041667s`；`metadata.json` 中已记录
  `same_as_exp03_except_reference_video_url: true`。

**用户原始需求：**
> 看之前 SAM3 pipeline，把当前视频中的机械臂用黄色框标出来，黑色夹爪部分用红色标出来，输出到
> `seedance_workspace/video/grab_both_cubes_v1_ep000001_f000000/input/yellow_bbox`。

**直接修改：**
- 基于 R/B 修正后的
  `seedance_workspace/video/grab_both_cubes_v1_ep000001_f000000/input/grab_both_cubes_v1_ep000001_f000000_robot_camera_original_ref_864x480.mp4`
  生成 SAM3 标注视频。
- 复用 `tmp/h2r_seedance_exp04_prompt_ep1_sam3_mask/grab_both_cubes_v1/episode_1.npz`：
  整臂 SAM3 mask 映射到 `864x480` 后画黄色 bbox，`SAM3 arm mask ∩ dark_threshold<=80`
  的黑色夹爪区域用红色高亮。

**运行结果：**
- 输出标注视频：
  `seedance_workspace/video/grab_both_cubes_v1_ep000001_f000000/input/yellow_bbox/grab_both_cubes_v1_ep000001_f000000_robot_camera_yellow_arm_bbox_red_gripper_ref_864x480.mp4`。
- 同目录输出红色夹爪 mask 视频和 JSON metadata；标注视频规格为 `864x480`、`30fps`、`4.0s`，
  整臂 mask 与红色夹爪 mask 都覆盖 120 帧。

**用户原始需求：**
> 去掉红色，只保留黄色框。

**直接修改：**
- 在同一目录新增黄色框-only 标注视频：
  `seedance_workspace/video/grab_both_cubes_v1_ep000001_f000000/input/yellow_bbox/grab_both_cubes_v1_ep000001_f000000_robot_camera_yellow_arm_bbox_ref_864x480.mp4`。
- 继续基于 R/B 修正后的 original robot 视频和同一个 SAM3 整臂 mask；不再叠加红色夹爪区域。

**运行结果：**
- 黄色框-only 视频规格为 `864x480`、`30fps`、`4.0s`，SAM3 整臂 mask 覆盖 120 帧。

**用户原始需求：**
> 看一下原视频，把原视频拼成 15s，试一口气转换；现在小片段效果不好。

**直接修改：**
- 使用 `seedance_workspace/video/grab_both_cubes_v1_ep000001_f000000` 的 R/B 修正后
  original robot 4s 片段，拼接 `f000000`、`f000120`、`f000240`、`f000360` 四段后裁到前 15s。
- 输出 15s Seedance 输入视频到
  `seedance_workspace/video/grab_both_cubes_v1_ep000001_f000000/input/long_15s/`。

**运行结果：**
- 输出视频：
  `seedance_workspace/video/grab_both_cubes_v1_ep000001_f000000/input/long_15s/grab_both_cubes_v1_ep000001_f000000_robot_camera_original_ref_864x480_15s.mp4`。
- 规格：`864x480`、`30fps`、`450` 帧、`15.0s`，对应 episode_1 的前 15s。

**用户原始需求：**
> 生成一下反向的 Cache。

**直接修改：**
- 新增 `scripts/build_reverse_r2h_cache.py`，用于从现有 `h2r/2s61f30_slide` pair/cache
  生成 `r2h/2s61f30_slide` 反向视图；视频和 T5 使用 hardlink，VAE `.pth` 重新写入并交换
  `human_latent` / `robot_latent`。
- 更新 `doc/step_5_training_infra.md`，记录反向 cache 的生成命令、字段语义和 `--resume`
  行为。

**运行结果：**
- 已生成 `training_data/pair/r2h/2s61f30_slide`、
  `training_data/cache/vae/r2h/2s61f30_slide` 和
  `training_data/cache/t5/r2h/2s61f30_slide`。
- 样本数量：`Inspire_Collect_Clothes_MainCamOnly=40`、
  `Inspire_Put_Clothes_into_Washing_Machine=155`、
  `Inspire_Pickup_Pillow_MainCamOnly=15`。
- 运行时 split 校验通过：`train=187`、`in_task_eval=8`、`ood_eval=8`；
  抽样校验确认新的 `human_latent` 等于源 H2R `robot_latent`，新的 `robot_latent` 等于源 H2R
  `human_latent`，pair 视频和 T5 cache 均为 hardlink 复用。

**用户原始需求：**
> 从头做 human2robot / H2R 原始配对数据的 robot 视频到 human 视频编辑 cache：2s 数据、
> 30fps、不需要滑动窗口，并行处理。

**创建的任务：**
- [082] H2R 原始配对数据构造 r2h 2s30fps cache

**当前设计结论：**
- 新数据不复用 G1 `2s61f30_slide` 反向 cache；单独写入
  `training_data/pair/r2h/2s61f30_h2r_v1` 和
  `training_data/cache/vae/r2h/2s61f30_h2r_v1`。
- 2s30fps 使用 61 帧 `0..60`，每个 episode 默认只取首个窗口，不做滑窗。
- MP4 源当前 210 个 episode 中 1 个只有 8 帧，首窗口 2s61f30 可用 209 条。
- HDF5 全量当前 1312 个 episode 中 2 个短于 61 帧，首窗口 2s61f30 可用 1310 条。

**直接修改：**
- 在任务 worktree `.worktrees/t082` 新增 `src.pipeline.h2r_r2h_pair`，用于从
  `data/h2r/v1/video/<task>/episode_*/{robot_camera,human_camera}.mp4`
  直接构造 r2h Mitty pair。
- 接入 `scripts/flip_run.sh h2r_r2h_pair`，支持 `--tasks all`、`--workers`、
  `--clean`、`--resume`、`--dry-run`、`--max-episodes-per-task` 等参数。
- 更新 `.worktrees/t082/doc/step_5_training_infra.md`，记录
  `training_data/pair/r2h/2s61f30_h2r_v1` 与
  `training_data/cache/vae/r2h/2s61f30_h2r_v1` 的 layout、字段语义和运行命令。

**运行结果：**
- dry-run：`--tasks all` 得到 `pairs=209`、`short skipped=1`。
- 正式 pair 已生成到 `training_data/pair/r2h/2s61f30_h2r_v1`：
  - `index.jsonl` 209 行；
  - 209 个 human target 视频和 209 个 robot control 视频；
  - 全量校验均为 61 帧、30fps、416x224；
  - `summary.json` 记录跳过 `push_box_two_v1/episode_5`，因为该 episode 只有 8 帧。
- VAE/T5 cache 已生成：
  - `training_data/cache/vae/r2h/2s61f30_h2r_v1`：209 个 `.pth`，22 个 task manifest；
  - `training_data/cache/t5/r2h/2s61f30_h2r_v1`：`prompt_3c4c5fbd.pth` 和 `negative.pth`。
- Runtime split 校验通过：`data_type=r2h`、`duration=2s61f30_h2r_v1` 下
  209 个 cache 默认切分为 `train=189`、`in_task_eval=20`、`ood=0`，并写出 22 个
  `pair_order.jsonl`。
- 代码检查通过：
  - `python -m compileall -q src/pipeline/h2r_r2h_pair.py`；
  - `PYTHONPATH=. python tests/test_runtime_data.py`。
