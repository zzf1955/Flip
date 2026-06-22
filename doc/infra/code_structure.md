# FLIP 代码架构（src/）

## 目录结构

```
src/
├── core/          12 个基础库模块（不可直接运行）
├── pipeline/     28 个当前可执行 pipeline 入口
│   └── archive/  23 个归档 pipeline 入口
└── tools/        22 个实验/调试/可视化工具
```

`src/` 是当前主代码结构。`scripts/` 顶层只保留现役统一入口、通用训练 wrapper、
R2H 队列/分析脚本和 smoke；旧 IK、camera、render/debug、segmentation/inpaint、
dataset utility、一次性 cache/backfill/migration/eval helper 和临时训练 shell
已归入 `scripts/archive/` 的分类子目录。

`src/pipeline/archive/` 保存不再作为当前主线维护的旧 pipeline。它们保留用于复现
历史结果或手动排查，但不再由 `scripts/flip_run.sh` / `scripts/flip_run_2.sh`
分发。当前主线只保留 SAM2/SAM3 mask 预计算、human2robot / R2H 数据构造、
Mitty/Wan 三阶段训练、评估和 Diffusion Policy 相关入口。

`src/pipeline/archive/` 分类如下：

| 子目录 | 内容 |
|--------|------|
| `inpaint_retarget/` | 旧 inpaint pipeline、segment pipeline、human overlay、retarget video |
| `patch/` | 旧 robot patch 和 hand patch 数据工具 |
| `comfyui_wan/` | 旧 ComfyUI Wan / Cosmos 本地重绘实验 |
| `idm/` | 旧 Wan VAE / Pair / Humanoid Everyday / AdaWorld IDM 和 action mask 实验 |
| `training_legacy/` | 旧 mixed h2r 训练入口和 runtime split |
| `baseline/` | 旧 Masquerade-style direct render baseline |

## scripts/ 当前边界

顶层 `scripts/` 保持精简，只放仍会直接调用的入口：

| 类别 | 顶层脚本 |
|------|----------|
| 统一入口 | `flip_run.sh`, `flip_run_2.sh`, `codex_pre_tool_use_guard.py` |
| 训练 wrapper | `train_lora_grid.py`, `train_three_stage_single_lora.py` |
| R2H 队列 / 分析 | `run_r2h_synthesize_queue.py`, `run_syn_error_analysis.py` |
| smoke | `smoke_test.py`, `smoke_test_gpu.py`, `smoke_test_light.py` |

`scripts/archive/` 分类如下：

| 子目录 | 内容 | 当前替代入口 |
|--------|------|--------------|
| `camera_calibration/` | 旧相机模型、mask/keypoint calibration、焦距/畸变/外参验证 | `src.tools.calibrate_mask`, `src.tools.calibrate_keypoints`, `src.tools.estimate_focal`, `src.tools.verify_extrinsics` |
| `retarget_ik/` | 旧 SMPLH IK、retarget copy、retarget debug/video | `src.tools.retarget_diag`, `src.tools.render_ik_debug`, `src.tools.render_smplh_ik` |
| `render_debug/` | 旧 mesh/render overlay/debug/svg2gif 工具 | `src.tools.render_3view`, `src.tools.render_overlay_check`, `src.tools.render_lit_overlay`, `src.tools.svg2gif` |
| `segmentation_inpaint/` | 旧 SAM2/inpaint pipeline 与批处理 launcher | `src.pipeline.sam2_precompute`, `src.pipeline.sam2_segment` |
| `dataset_utils/` | 旧数据检查、抽帧、降采样和一次性 cache/backfill/reverse 数据脚本 | 按需迁移到 `src.tools` 或一次性使用 |
| `training_launchers/` | 旧 H2R / final / SAM3 stage shell 训练启动器 | `scripts/flip_run.sh train`，或通用 wrapper `scripts/train_lora_grid.py` / `scripts/train_three_stage_single_lora.py` |
| `eval_analysis/` | 旧 final eval 补跑和背景 Patch FID 辅助脚本 | `scripts/flip_run.sh eval_mitty` 或 `src.tools.eval_metrics` |
| `migration/` | 一次性旧 split/cache layout 迁移脚本 | 新数据直接写 task layout，不再迁移 |
| `smoke/` | one-off smoke / proof-of-concept 脚本 | 顶层 `scripts/smoke_test*.py` |

---

## 一、core/ — 基础库模块

### 依赖关系

```
config.py          ← 无依赖
data.py            ← config
camera.py          ← config
fk.py              ← config, pinocchio, stl
render.py          ← camera
mask.py            ← (独立)
smplh.py           ← config, torch, pinocchio
retarget.py        ← config, smplh, fk
eval_metrics.py    ← torch, skimage, transformers (CLIP)
```

### 模块说明

| 模块 | 功能 | 主要导出 |
|------|------|----------|
| `config.py` | 集中配置：路径、相机参数、任务选择 | `MAIN_ROOT`, `DATA_ROOT`, `TRAINING_DATA_ROOT`, `PAIR_DIR`, `TRAINING_TASKS`, `BEST_PARAMS` |
| `data.py` | 数据加载 + 视频 IO | `load_episode_info()`, `open_video_writer()`, `write_frame()` |
| `camera.py` | 相机模型 + 投影 | `make_camera()`, `project_points_cv()` |
| `fk.py` | URDF/mesh + FK | `load_robot()`, `build_q()`, `do_fk()` |
| `render.py` | mask/overlay/Lambertian 渲染 | `render_mask()`, `render_overlay()` |
| `mask.py` | mask 后处理 + LaMa | `postprocess_mask()`, `init_lama()`, `run_lama()` |
| `smplh.py` | SMPLH 模型 + IK 求解器 | `SMPLHForIK`, `IKSolver` |
| `retarget.py` | G1→SMPLH retarget | `retarget_frame()`, `refine_arms()` |
| `eval_metrics.py` | 训练在线评估指标 (PSNR/SSIM/LPIPS/CLIP Score) | `OnlineMetrics` |

---

## 二、pipeline/ — 可执行 pipeline

### 数据生成与预处理

| 脚本 | 功能 | 输入 | 输出 |
|------|------|------|------|
| `segment_episodes.py` | 原始 episode 切分为 4s segment | LeRobot 数据集 | `training_data/segment/` |
| `seedance_gen.py` | Volcengine Seedance 2.0 API 生成人体视频；请求超时可用 `ARK_REQUEST_TIMEOUT` 配置，默认 120s | robot 视频 | `training_data/seedance_direct/4s/` |
| `seedance_batch.py` | seedance_gen 的批量包装 | 多个 robot 视频 | 同上 |
| `seedance_raw_batch.py` | WBT Seedance 扩展 CSV 专用 raw-only batch；只处理 `needs_seedance_api=true` 行，临时抽取或放大 4s API 输入，最终只保留 Seedance API 原始 mp4，不写默认 batch log 或 `.raw.mp4` side file | `tmp/...seedance_raw_new_generation.csv` + `training_data/segment/` 或 WBT raw episode mp4 | `training_data/seedance_raw/4s/<task>/<episode>/<clip>_human.mp4` |
| `seedance_advance.py` | overlay 视频经 Seedance 增强；`--output-root` 可切到 `training_data/seedance_overlay/4s/` 做 prompt/输入源对比 | `training_data/overlay/4s/` | 默认 `training_data/seedance_advance/4s/` |
| `h2r_seedance_edit.py` | human2robot HDF5 robot-camera → Seedance 人手编辑 smoke；已验证直接 robot2human 效果不好，仅保留为 API/尺寸/prompt 调试入口，不进入正式训练数据主线 | `data/h2r/v1/data/<task>/episode_*.hdf5` | `tmp/h2r_seedance_edit_*` |
| `h2r_seedance_sam3_edit.py` | human2robot HDF5 robot-camera + SAM3 mask → marker 引导的 Seedance 人手编辑 smoke；支持 `full` 全臂 mask、`dark` 暗色夹爪过滤和实验性 `distal_dark` 远端暗色连通域过滤，支持红/紫/青/黄/绿/肤色 marker 与 fill/outline/bbox/fill_bbox/dual_bbox 标注；当前推荐仍是 `dark + yellow/magenta bbox`，`fill_bbox` 和 `dual_bbox` 已作为负例/探索项记录，仅作为显式 mask prompt 调试，不自动进入训练数据主线 | `data/h2r/v1/data/<task>/episode_*.hdf5` + `tmp/.../sam3_mask/<task>/episode_*.npz` 或同格式 mask root | `tmp/h2r_seedance_sam3_*` |
| `h2r_seedance_sam3_eval.py` | human2robot Seedance 输出量化评估；用 SAM3.1 `human hand` prompt 分割 final 视频的人手区域，并与 Seedance 输入的 target mask 统计 coverage / IoU，写 overlay review 视频 | `h2r_seedance_sam3_edit.py` 的 `seedance_results.jsonl` + `target_mask_video_path` | `tmp/h2r_seedance_sam3_eval_*` |
| `seedance_clip.py` | Seedance direct 4s 视频 post-process：1s、0.5s stride 滑窗 + 水平翻转增强 | `seedance_direct/4s/` | `seedance_direct/1s/` + `manifest.jsonl` |

### 分割与修复

| 脚本 | 功能 | 输入 | 输出 |
|------|------|------|------|
| `archive/inpaint_retarget/sam2_inpaint.py` | FK → SAM2 分割 → LaMa/ProPainter 修复 | episode 视频 | `output/inpaint/` |
| `sam2_segment.py` | SAM2 多部位分割实验 | episode 视频 | `output/inpaint/sam2_segment/` |
| `sam2_precompute.py` | SAM2 mask 预计算（FK bbox prompt → SAM2 propagation → npz） | segment 视频 | `training_data/sam2_mask/` |
| `batch_sam2_precompute.py` | sam2_precompute 多 GPU 调度（多 worker/GPU） | 多 task | `training_data/sam2_mask/` |
| `g1_sam3_precompute.py` | G1 segment SAM3/SAM3.1 mask smoke / 预计算；通过 `sam3` conda 环境按短 chunk 调用 text prompt，支持 `--prompt-list` sweep 和 `--prompt-mode text_bbox` 二阶段 bbox prompt，写出 segment 级 `.npz`、summary 和可选 overlay；当前 smoke 结论是 61 帧单 session 会 OOM，纯 SAM3 prompt 不能干净替代 SAM2 全身 mask | `training_data/segment/<task>/ep*/seg*_video.mp4` + `ref-sam3` / SAM3.1 checkpoint | `training_data/g1_sam3_mask/<task>/ep*/seg*.npz` 或 `tmp/g1_sam3_*` |
| `archive/inpaint_retarget/batch_inpaint.py` | 多 GPU 批量修复调度 | 多 task/episode | 自定义 |
| `archive/inpaint_retarget/video_inpaint.py` | 逐帧 FK + GrabCut + LaMa | episode 视频 | `output/inpaint/per_frame_lama/` |

### 渲染与 Overlay

| 脚本 | 功能 | 输入 | 输出 |
|------|------|------|------|
| `archive/inpaint_retarget/segment_pipeline.py` | **主 pipeline**：FK→SAM2→inpaint→SMPLH retarget→overlay，并在中间目录展示原视频与 SAM2 mask blur_r2r control 视频 | segment 视频 | `output/segment_pipeline/`、`training_data/overlay/4s/` |
| `body_fit_search.py` | 搜索 SMPLH body/hand scale 与 root offset，用 G1 mesh mask 量化贴合 | segment 视频 + joints parquet | `output/body_fit_search*/` |
| `archive/inpaint_retarget/human_overlay.py` | SMPLH mesh 叠加到修复背景 | 修复视频 + retarget 数据 | overlay MP4 |
| `archive/inpaint_retarget/retarget_video.py` | 3-panel 对比视频 [原始\|G1\|SMPLH] | episode 视频 | `output/human/retarget_video/` |

### 训练数据配对

| 脚本 | 功能 | 输入 | 输出 |
|------|------|------|------|
| `make_pair.py` | 匹配 robot+human 视频，重采样 16fps，4k+1 帧；正式训练时 `--task all` 只展开 `TRAINING_TASKS` 三任务集合 | segment + seedance/overlay | `training_data/pair/{1s,2s,4s}/` |
| `g1_2s_slice_data.py` | G1 2s/30fps 切片交付工具；生成 61 帧 `2s61f30` original、Seedance direct、SAM2 blur slice，并 hardlink 成 `identity_r2r`、`blur_r2r`、`h2r` pair layout；stage2 blur 数据按 2s stride + tail 对齐切片 | `training_data/segment/` + `training_data/seedance_direct/4s/` + `training_data/sam2_mask/` | `training_data/slice/g1_2s61f30/`、`training_data/pair/{identity_r2r,blur_r2r,h2r}/2s61f30/` |
| `g1_2s_seedance_slide_data.py` | G1 2s Seedance 密集滑窗交付工具；复用 task076 的 full robot-only slice 生成 `step2/origin`、`step2/blur`，并按 0.5s 默认 stride 从 Seedance 4s source 切 `step1/human` 和对齐的 `step1/origin`；发布 `2s61f30_slide` 的 `identity_r2r`、`blur_r2r`、`h2r` pair layout | task076 `training_data/slice/g1_2s61f30/` + `training_data/segment/` + `training_data/seedance_direct/4s/` | `training_data/g1_2s61f30_seedance_slide/`、`training_data/pair/{identity_r2r,blur_r2r,h2r}/2s61f30_slide/` |
| `h2r_sam3_precompute.py` | human2robot SAM3/SAM3.1 robot-arm mask 预计算；通过 `sam3` conda 环境逐 1s clip 调用 text prompt `robot arm`，支持 `--prompt-frame-position first|middle|<int>`；`--clip-starts-file` 可显式指定 1s clip start frames，用于 Seedance 4s / tail-aligned 批量输入的 mask 覆盖；把 source frame mask 写入 episode 级 `.npz`，供 stage2 blur 或 Seedance marker smoke 复用 | `data/h2r/v1/video/<task>/episode_*/robot_camera.mp4` + `ref-sam3` / SAM3.1 checkpoint | `training_data/h2r_sam3_mask/<h2r_task>/episode_*.npz` 或 `tmp/.../sam3_mask/` |
| `h2r_sam3_blur_pair.py` | human2robot stage2 外观训练数据转换；读取 `data/h2r/v1/video/<task>/episode_*/robot_camera.mp4` 和预计算 SAM3/SAM3.1 mask，将清晰 robot clip 写为 target，将 SAM3 mask 区域模糊后写为 control；不隐式运行 SAM3，缺 mask 或帧对齐不一致时直接失败 | human2robot robot-camera mp4 + `training_data/h2r_sam3_mask/<task>/episode_*.npz` 或 mask mp4 | `training_data/pair/blur_r2r/1s/<h2r_task>/` |
| `make_robot_pair.py` | 生成 robot→robot identity pair；正式三阶段 LoRA identity 时 `--task all` 只展开 `TRAINING_TASKS` 三任务集合 | segment | `training_data/robot_pair/1s/` |
| `r2h_synthesize.py` | 用训练好的 r2h Mitty LoRA 从 `training_data/segment` 枚举 robot clip，排除 Seedance 已覆盖来源后生成 h2r `_syn` pair；支持按 task 可用量比例分配固定生成总量 | segment + r2h checkpoint | `training_data/pair/h2r/<duration>/<task>_syn/` |
| `run_r2h_synthesize_queue.py` | 按全局 `_syn` 目标总量计算每个 source task 的目标数，生成单 task `r2h_synthesize` 命令队列，并按用户提供的 CUDA 列表调度；每张卡一次跑一个任务，结束后取队列下一项 | segment + r2h checkpoint + CUDA 列表 | `training_data/pair/h2r/<duration>/<task>_syn/`、`training_data/log/r2h_synthesize_queue/<timestamp>/` |
| `run_syn_error_analysis.py` | 独立分析脚本：默认取 in-task + OOD 的 `ep000`-`ep003`，从 4s segment 切成 1s 非重叠 robot clip，再用 r2h checkpoint 生成 syn human，结果不写入训练 pair | segment + r2h checkpoint | `output/syn_error_analysis/{robot,syn,compare}/1s/<task>/<episode>/`、`manifest.jsonl` |
| `archive/baseline/masquerade_baseline.py` | Masquerade-style h2r 直接渲染 baseline：从 `training_data/pair/h2r/1s/<task>/manifest.jsonl` 读取 human/robot pair，自动估计 human foreground mask、左右半边 bbox 和 trajectory，用 mask 对 control frame 做 inpaint 背景重绘，再用 `training_data/segment/<task>/<episode>/<seg>_joints.parquet` + G1 URDF/mesh + 标定相机渲染不透明 robot mesh，并写出 baseline / background / human overlay / compare / annotation 产物 | h2r pair manifest + segment joints parquet + robot video | `output/masquerade_baseline/h2r/1s/<task>/` |
| `archive/patch/robot_patch.py` | 全身降质数据（FK mesh 或 SAM2 mask → blur/noise/mean） | segment + parquet/sam2_mask | `training_data/pair/1s_patch/` |
| `scripts/archive/dataset_utils/backfill_segment_pipeline_blur.py` | 归档的一次性脚本：只用已有 `segment_pipeline` postprocess mask 补 `00_original.mp4` 与 `08_blur_r2r_control.mp4`，不重跑 FK/SAM2/inpaint/human | `output/segment_pipeline/` + `training_data/segment/` | `output/segment_pipeline/<task>/ep*/seg*/00_original.mp4`、`08_blur_r2r_control.mp4` |

### LoRA 训练

| 脚本 | 功能 | 输入 | 输出 |
|------|------|------|------|
| `train.py` | Mitty / Wan2.2 LoRA 正式训练入口 | `training_data/cache/vae/...` + `training_data/cache/t5` | `training_data/log/` 或 `tmp/` smoke |
| `archive/training_legacy/train_mitty_mixed_h2r.py` | 独立 mixed h2r Mitty 训练入口；构建 original + `_syn` 显式 split，稳定 eval 固定来自 original pair_order 尾部 | `training_data/pair/h2r/<duration>/{task}` + `training_data/cache/vae/h2r/<duration>/{task}` | `training_data/log/<run>/mixed_cache/`、`data_split/`、ckpt/eval/log |
| `archive/idm/action_mask_precompute.py` | 基于 G1 FK mesh 投影为 Wan VAE IDM 预计算逐帧 visible body-part mask 与 action mask；`arm_hand` 输出 24 维，`full_body` 输出 48 维并覆盖 torso/legs/arms/hands；支持 `--clip-middle-only` 只渲染 IDM 实际监督的中间帧，支持 `--workers` 并行分段预计算；输出 metadata 和 `index.jsonl` | `training_data/segment/<task-short>/ep*/seg*_joints.parquet` + robot mesh/URDF | `training_data/action_mask/<task-short>/ep*/seg*.npz` |
| `archive/idm/wan_vae_idm.py` | 冻结 Wan VAE 的 MainCamOnly Video2Action IDM；默认 Collect Clothes，可用 `--task-short` / `--task-full` 切到其他任务；`--target-mode arm_hand` 输出 `ee_action + hand_cmd` 24 维，`--target-mode full_body` 输出 `robot_q_desired + hand_cmd` 48 维；中间帧 action 监督，纯 3D CNN + MLP head；可通过 `--action-mask-root` 启用 visible action masked loss / metric；训练中写 `eval_loss.csv`、`best_checkpoint.pt`、`best_val_predictions.csv` 和 `loss_curve.png`，支持 `validate` 复算 held-out checkpoint 指标，也可复算 eval gen/gt action consistency；`eval-h2r` 可用三任务 checkpoint 对 Baseline/Ours H2R `full_eval` 生成视频做 action 复算和汇总 | 原始 `data/unitree_G1_WBT/G1_WBT_<task-short>` + `training_data/segment/<task-short>` + 可选 `training_data/action_mask/<task-short>`；H2R 复算读取 `training_data/log/<run>/full_eval` | `tmp/wan_vae_idm_*`、`output/wan_vae_idm/<task-short>/` 或 `output/idm_h2r_action_eval/` |
| `archive/idm/wan_pair_idm.py` | 两帧 RGB inverse dynamics IDM；旧 WBT segment 模式从相邻帧 `(s_t, s_{t+1})` 预测 `action[t]`，arm 使用 `action.ee_action`、hand 使用 `action.hand_cmd`，同一入口训练两套独立小 CNN；输出 train/eval loss、checkpoint、逐样本预测表、normalized MSE / R2 / correlation / 方差比等诊断指标；`validate` 默认复用 checkpoint split/config | 原始 `data/unitree_G1_WBT/G1_WBT_<task-short>` + `training_data/segment/<task-short>` | `tmp/wan_pair_idm_*` 或 `output/wan_pair_idm/<task-short>/` |
| `archive/idm/humanoid_pair_idm.py` | Humanoid Everyday H1 两帧 RGB inverse dynamics IDM；直接读取 LeRobot `data/chunk-*/*.parquet` 与 `videos/chunk-*/egocentric/*.mp4`，输入 `frame_t/frame_{t+d}` 并预测 `mean(action[t:t+d])`；`frame_stride` 只表示候选起点步长，`frame_delta` 表示区间长度；默认 `--model-arch motion_transformer`，legacy `transformer` 保留作 checkpoint / ablation 对照，也可切回 `small_cnn`；task064 新增 `motion_transformer_v2`，在 patch-level motion token 外加入 RGB diff/abs-diff raw motion stem，并默认使用 residual readout head；训练支持 AdamW betas、cosine + warmup、variance calibration loss、梯度累积、CUDA bfloat16 AMP 和严格 `--init-checkpoint` 二阶段初始化；默认按 episode 划分，显式 `--train-samples` / `--eval-samples` 也保持 train/eval episode 不重叠；输出 train/eval loss、checkpoint、逐样本预测表和诊断指标；`validate` / `eval` 复用 checkpoint 中保存的 split/config，旧 checkpoint 按 `legacy_mlp` 严格 replay | `data/humanoid-everyday-h1-chunks0-6-8-200` 这类 Humanoid Everyday H1 LeRobot 子集（本地 sweep 可用剔除坏 parquet 的临时 symlink 根） | `tmp/humanoid_pair_idm_*` 或 `output/humanoid_pair_idm/humanoid_everyday_h1` |
| `h2r_diffusion_policy.py` | Action-only Diffusion Policy BC；默认 `--dataset-kind h2r_hdf5` 读取 human2robot HDF5 中的 `cam_data/robot_camera`、`qpos/qvel/end_position/gripper_state` 和 `action`；显式 `--dataset-kind g1_2s_pair` 读取 G1 `2s61f30_slide` pair manifest，按 segment-local `source_frame_indices` 取 joints state，再用 joints 行里的全局 `frame_index` 回查 LeRobot action；clean video/state history 作为 condition，future action chunk 加噪后训练 denoising loss，不预测未来视频；输出 dataset summary、`train_log.jsonl`、best/last checkpoint、eval summary，并同时记录 normalized action MSE 与反归一化 action 空间 MSE/RMSE/L2 相对误差；`--best-metric` 可指定 best checkpoint 挑选指标 | human2robot: `data/h2r/v1`；G1: `training_data/pair/identity_r2r/2s61f30_slide` + `training_data/segment` + `data/unitree_G1_WBT` | `tmp/h2r_diffusion_policy_t068*`、`tmp/g1_dp_2s_*` |
| `archive/idm/adaworld_action_encoder.py` | AdaWorld LAM action encoder 提取入口；只运行 `(frame_t, frame_{t+1}) -> 32d continuous latent action`，不加载 world model；复用 H1 LeRobot egocentric mp4，两帧按 AdaWorld 口径中心裁方、resize 到 256、归一化到 `[0,1]`；提取时显示 tqdm 进度条，并实时写 `latent_actions.npy` memmap 与 `manifest.jsonl`，结束后生成 decoder 兼容的 `latent_actions.npz` 和 `summary.json` | `ref-AdaWorld` 代码仓库 + `ref-AdaWorld-hf/lam.ckpt` + Humanoid Everyday H1 LeRobot 数据 | `tmp/adaworld_action_encoder_*` 或 `output/adaworld_action_encoder/humanoid_everyday_h1` |
| `archive/idm/adaworld_action_decoder.py` | AdaWorld latent action decoder；读取 task054/task057 的 `latent_actions.npz`，按 `episode/chunk/rel_frame_t` 回查 H1 `action` 标签，训练 `(frame_t, frame_{t+1}) -> z_t -> action_t` 解码器；支持 `mlp` baseline、`residual_mlp` 默认推荐、`gated_mlp` 消融，以及 shared/per-dim/grouped output head；AdamW 可配置 betas、cosine scheduler、warmup、min lr，loss 可切到 weighted MSE / SmoothL1 / variance calibration；保存 checkpoint、预测表、loss curve、per-dim 诊断指标，并可通过 checkpoint replay 复算 `validate` / `eval` | `tmp/adaworld_action_encoder_*` 或 `tmp/adaworld_action_encoder_h1_full_t057/latent_actions.npz` + `data/humanoid-everyday-h1-chunks0-6-8-200` | `tmp/adaworld_action_decoder_*` 或 `output/adaworld_action_decoder/humanoid_everyday_h1` |
| `archive/idm/adaworld_decoder_diagnostics.py` | AdaWorld decoder 预测 CSV 诊断；从 `val_predictions.csv` / `predictions.csv` 汇总逐维 MSE、normalized MSE、R2、correlation、预测方差比，并可导出加权 loss 权重 JSON/CSV 供下一轮训练复用 | AdaWorld decoder 预测 CSV | `tmp/adaworld_action_decoder_t063_analysis/` 或自定义目录 |
| `train_lora_grid.py` | LoRA layout × rank 网格搜索启动器；支持 merge LoRA、数据量、layout/rank、CUDA 轮转与 dry-run | LoRA checkpoint + `train.py` 数据 preset | 多个 `training_data/log/*_{layout}_r{rank}_{timestamp}/` run |
| `train_three_stage_single_lora.py` | staged LoRA launcher；每阶段可显式选择 merge LoRA 和 train LoRA，默认 identity → blur → h2r 串联上一阶段 checkpoint | `train.py` 数据 preset + 可选 merge/train LoRA checkpoint | 多个连续 `training_data/log/single_lora3_s*` run，后一阶段可继承前一阶段 ckpt |

### 评估与分析脚本

| 脚本 | 功能 | 输入 | 输出 |
|------|------|------|------|
| `scripts/archive/eval_analysis/eval_background_patch_fid.py` | 归档辅助脚本：复用已有 `full_eval` 视频计算前景/背景 Patch FID | `training_data/log/<log>/full_eval/` + `data_split/*.jsonl` + SAM2 mask | `output/background_fid/<log>/summary.*` |

### 人体重绘（Step 4 本地方案）

| 脚本 | 功能 | 输入 | 输出 |
|------|------|------|------|
| `archive/comfyui_wan/cosmos_prepare.py` | 前置：composite + depth + mask + spec.json（两种 regen 共用） | overlay 视频 | `output/human/cosmos_prepare/` |
| `archive/comfyui_wan/wan_regen.py` | **主方案**：ComfyUI + Wan 2.1 VACE 1.3B depth+mask 重绘 | cosmos_prepare 输出 | `output/human/wan_regen/` |
| `archive/comfyui_wan/cosmos_regen.py` | Cosmos Transfer 2.5 推理（**已弃用**，保留作对照存档） | cosmos_prepare 输出 | `output/human/cosmos_regen/` |

详见 `doc/archive/step_4_wan_vace_regen.md`。

### 运行方式

```bash
LD_PRELOAD=/home/leadtek/miniconda3/envs/flip/lib/libjpeg.so.8 \
  python -m src.pipeline.<script_name> [args]
```

GPU / CUDA 命令优先走统一入口，便于 Codex 按子命令保存越权批准规则：

```bash
scripts/flip_run.sh mitty_cache --cuda 0 -- <mitty_cache args>
scripts/flip_run.sh sam2_precompute --cuda 0 -- <sam2_precompute args>
scripts/flip_run.sh train --cuda 2,3 --nproc 2 -- <train args>
scripts/flip_run.sh h2r_seedance_sam3_eval --cuda 1 -- <eval args>
scripts/flip_run.sh nvidia-smi
```

当前 Codex 可使用 `danger-full-access` 直接访问 GPU；`scripts/flip_run.sh` 仍作为统一环境与白名单入口保留。Bash 高危命令由 `scripts/codex_pre_tool_use_guard.py` 通过 Codex `PreToolUse` hook 做最佳努力拦截。

---

## 三、tools/ — 实验/调试工具

### 相机标定

| 脚本 | 功能 | 输出路径 |
|------|------|----------|
| `calibrate_mask.py` | PSO mask Dice 标定 | `output/calibration/mask_dice/` |
| `calibrate_keypoints.py` | PSO/Adam 关键点标定 | `output/calibration/kp_optim/` |
| `estimate_focal.py` | 焦距解析估计 | stdout |
| `distortion_analysis.py` | 畸变分析 | `tmp/distortion/` |
| `verify_extrinsics.py` | URDF 外参验证 | `tmp/urdf_verify/` |
| `verify_mesh.py` | STL/URDF 尺寸验证 | stdout |

### 人体 retarget

| 脚本 | 功能 | 输出路径 |
|------|------|----------|
| `retarget_diag.py` | retarget 9宫格诊断 | `output/human/retarget_diag/` |
| `render_smplh_ik.py` | SMPLH IK overlay | `output/human/smplh_ik/` |
| `render_ik_debug.py` | 第三人称 IK 调试 | `output/human/ik_debug/` |
| `debug_retarget.py` | retarget 误差可视化 | `output/human/debug_retarget/` |

### 渲染验证

| 脚本 | 功能 | 输出路径 |
|------|------|----------|
| `render_3view.py` | G1 三视图渲染 | `tmp/3view/` |
| `render_overlay_check.py` | 多视频 overlay 泛化 | `tmp/overlay_check/` |
| `render_lit_overlay.py` | Lambertian overlay | `tmp/lit_overlay/` |
| `demo_mesh_scale.py` | mesh 缩放对比 | `tmp/mesh_scale/` |
| `debug_keypoints.py` | 关键点可视化 | `tmp/kp_debug/` |

### 工具

| 脚本 | 功能 |
|------|------|
| `svg2gif.py` | SVG→GIF 转换（独立） |
| `summarize_robot_wam_tune.py` | 汇总 robot_wam `train-wan` baseline/tune/full run；读取 `config.json`、`train_log.jsonl`、`train_summary.json` 和 `best_summary.json`，输出 `summary.csv` / `summary.md`；兼容旧单 eval 字段和 task073 的 `eval_in_task_*`、`eval_ood_*`、`eval_mean_loss` split eval 字段，并用 `safetensors.safe_open` 审计 best checkpoint 只包含 LoRA、`state_encoder`、`action_decoder`，无 `human` / `control` key |

### 运行方式

```bash
python -m src.tools.<script_name> [args]
```

---

## 四、数据流

### 完整 Pipeline：原始视频 → 训练好的 LoRA

```
G1 第一人称视频 (LeRobot dataset, 30fps)
│
├─ [segment_episodes.py]
│   → training_data/segment/<task>/ep*/seg*_video.mp4  (4s@30fps, 28K 文件, 19GB)
│
├─ [segment_pipeline.py]  (FK → SAM2 → inpaint → SMPLH retarget → overlay)
│   → training_data/overlay/4s/<task>/ep*/seg*_human.mp4
│
├─ [seedance_gen.py / seedance_batch.py]  (Volcengine API: robot → human)
│   → training_data/seedance_direct/4s/<task>/ep*/seg*_human.mp4
│   │
│   └─ [seedance_clip.py]  (4s → 1s clips, 0.5s stride + hflip)
│       → training_data/seedance_direct/1s/<task>/ep*/seg*_clip*.mp4
│       → training_data/seedance_direct/1s/<task>/manifest.jsonl
│
├─ [make_pair.py]  (匹配 robot+human, 重采样 16fps, 4k+1 帧, 统一编号)
│   → training_data/pair/1s/
│       ├── video/pair_NNNN.mp4        (robot, 17帧@16fps)
│       ├── control_video/pair_NNNN.mp4 (human, 17帧@16fps)
│       └── metadata.csv
│
├─ [sam2_precompute.py / batch_sam2_precompute.py]  (FK bbox → SAM2 pixel mask)
│   → training_data/sam2_mask/<task>/ep*/seg*.npz  (masks: uint8 (120,480,640))
│
├─ [robot_patch.py]  (全身降质: FK/SAM2 mask → blur/noise/mean)
│   → training_data/pair/1s_patch/
│       ├── video/pair_NNNN.mp4        (clean robot, 17帧@16fps)
│       ├── control_video/pair_NNNN.mp4 (degraded robot, 17帧@16fps)
│       ├── patch/pair_NNNN.pth        (latent mask + weights)
│       └── metadata.csv
│
├─ [mitty_cache.py]  (T5 + VAE cache)
│   → training_data/cache/{t5,vae}/...
│
└─ [train.py]  (Mitty / Wan2.2 LoRA 训练)
    → training_data/log/<run-name>/
        ├── ckpt/step-NNNN.safetensors
        ├── eval/step-NNNN/
        ├── train.csv
        └── train.log
```

### 训练数据格式要求

| 参数 | 要求 |
|------|------|
| 分辨率 | 640×480 |
| 帧率 | 16 fps |
| 帧数 | **4k+1**（1s=17帧, 2s=33帧, 4s=65帧） |
| 编码 | H.264, yuv420p |
| Prompt | `A first-person view robot arm performing household tasks flip_v2v` |

### 当前数据规模

| 阶段 | 数量 | 大小 |
|------|------|------|
| Segment (4s robot) | 28,548 | 19 GB |
| Overlay (4s human) | 53 | 64 MB |
| Seedance (1s human) | 80 | 15 MB |
| Training pairs (1s) | 120 对 | 37 MB |
| Cached embeddings | 按 split 生成 | `training_data/cache/{t5,vae}/` |

---

## 五、输出目录规范

```
output/                          # per-worktree 实验产物
├── calibration/                 # 相机标定
├── inpaint/                     # SAM2 + 修复
├── human/                       # retarget + overlay
│   ├── retarget_video/
│   ├── cosmos_prepare/
│   └── cosmos_regen/
├── segment_pipeline/            # 主 pipeline 中间产物，含 00_original 与 08_blur_r2r_control
└── human/                       # 人体重绘 / 中间可视化输出

training_data/                   # per-worktree 训练数据
├── segment/                     # 4s robot segments
├── sam2_mask/                   # SAM2 预计算 mask (sam2_precompute 输出)
│   └── <task>/ep*/seg*.npz      # masks: uint8 (120, 480, 640), 0/255
├── seedance_direct/             # Seedance human videos
├── overlay/                     # SMPLH overlay human videos
├── pair/                        # 配对训练数据
│   ├── 1s/{video/, control_video/, metadata.csv}  (make_pair 输出)
│   └── 1s_patch/{video/, control_video/, patch/, metadata.csv}  (robot_patch 输出)
├── compare/                     # 对比视频
├── cache/{t5,vae}/              # Mitty / Wan2.2 训练 cache
└── log/                         # train.py / train_mitty.py 训练输出
    └── <run-name>/{ckpt/, eval/, train.csv, train.log}

tmp/                             # smoke / 一次性验证，可删除
└── <task>/...
```
