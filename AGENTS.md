# Codex 项目规则

## 基本约定

- 默认用中文回答。
- 除非用户明确要求，不要直接查看视频和图片；简单视觉任务优先用 Python/OpenCV 等代码处理。
- 修改代码时应更新对应模块文档；涉及架构、数据流、环境、配置或注意事项时，同步更新 `doc/` 下相关文档。
- 代码中禁止用宽泛的 `try/except` 吞掉异常或 fallback 到旧行为；预期外行为应直接暴露并失败。
- 修改应彻底升级到新行为，不做错误 fallback，也不为了兼容旧逻辑保留隐式分支。

## 安全边界

- 当前项目为了让 GPU/训练命令直接访问 `/dev/nvidia*`，Codex 可使用 `danger-full-access` + `approval_policy=never`；必须同时启用 Codex hooks，并使用 `scripts/codex_pre_tool_use_guard.py` 作为 Bash `PreToolUse` 护栏。
- Hook 是最佳努力的命令前拦截，不是强沙箱；禁止依赖它执行高风险系统操作。
- Hook 必须阻止 `sudo`、`su`、`doas`、`pkexec`、setuid/setgid chmod、chown root、明显的项目外递归删除、危险 git reset/clean/force push 等命令。
- GPU/训练命令优先通过 `scripts/flip_run.sh <subcommand>` 统一入口执行，例如 `scripts/flip_run.sh train`、`scripts/flip_run.sh mitty_cache`、`scripts/flip_run.sh sam2_precompute`。
- 禁止删除项目外文件或目录；如确需清理项目外缓存、模型或临时文件，必须先明确告知用户路径和影响并等待确认。
- 对项目内的 destructive 操作（如 `rm`、`git reset`、批量覆盖生成结果），除非用户明确要求，也应先说明影响。

## 项目背景

revert 是第一人称人形机器人视频生成 / 视频编辑研究项目。当前主线围绕真实 G1
机器人视频和 human2robot 原始配对数据构造可训练的 robot / human 视频 pair，
并微调 Wan 2.2 + Mitty LoRA，使模型学习机器人外观恢复与 human ↔ robot
跨域编辑。下游还包含 action-only Diffusion Policy 与 robot-only WAM 方向。

当前主流程：

1. 数据切片：从 G1 segment 或 human2robot episode 生成 2s/30fps、61 帧样本。
2. 数据合成：使用 Seedance 或训练好的 R2H Wan/Mitty 模型生成 human-side 数据。
3. Mask / blur 数据：用 SAM2 或 SAM3/SAM3.1 预计算 robot mask，构造 `blur_r2r` 外观恢复数据。
4. 配对数据构造：按 `identity_r2r`、`blur_r2r`、`h2r`、`r2h` 发布统一 pair layout。
5. Cache：为 Wan/Mitty 训练预计算 VAE latent 和 T5 embedding。
6. 三阶段训练：
   - `identity_r2r`：清晰 robot → 清晰 robot，学习背景和基础重建。
   - `blur_r2r`：模糊 robot → 清晰 robot，学习目标机器人的外观细节。
   - `h2r` / `r2h`：human ↔ robot 跨域编辑。
7. 下游策略学习：在 human2robot 或 G1 数据上训练 Diffusion Policy / WAM。

## 数据命名

- 外部原始配对数据集统一称为 `human2robot`；历史本地目录仍是 `data/h2r/v1`，文档中只把它作为 legacy path 描述。
- `h2r` / `r2h` 是 revert 内部任务或数据方向名：`h2r` 表示 human → robot，`r2h` 表示 robot → human。
- 已有任务名、preset 和 data_type 中的 `h2r` 保留原命名；新的 human2robot 派生 duration / cache 名不要再用 `*_h2r_v1`，使用例如 `2s61f30_human2robot_v1`。

## 常用环境

- Conda 环境：`flip`。
- Python：3.10。
- CUDA：12.8。
- HuggingFace cache：`/disk_n/zzf/.cache/huggingface`。
- pip cache：`/disk_n/zzf/.pip_cache`。
- 运行脚本常用前缀：

```bash
LD_PRELOAD=/home/leadtek/miniconda3/envs/flip/lib/libjpeg.so.8 \
  no_proxy=localhost,127.0.0.1 \
  python -m src.pipeline.<script>
```

```bash
scripts/flip_run.sh train --cuda 2,3 --nproc 2 -- <train args>
scripts/flip_run.sh mitty_cache --cuda 0 -- <mitty_cache args>
scripts/flip_run.sh sam2_precompute --cuda 0 -- <sam2_precompute args>
scripts/flip_run.sh nvidia-smi
```

## 代码结构

- `src/core/`：基础库模块，不直接作为主入口运行。
- `src/pipeline/`：当前可执行 pipeline 与训练脚本；主线包括 SAM2/SAM3 mask、数据切片、pair/cache、Seedance/R2H 数据合成、Wan/Mitty 训练评估和 Diffusion Policy。
- `src/pipeline/backbones/`、`src/pipeline/eval_mitty/`：Wan/Mitty 训练与离线评估的支撑模块。
- `src/pipeline/archive/`：旧 inpaint/retarget、robot/hand patch、ComfyUI Wan/Cosmos、IDM、mixed h2r、Masquerade baseline 等归档 pipeline；不要作为新主线入口。
- `src/tools/`：标定、调试、可视化、日志转换等工具。
- `scripts/`：现役统一 launcher、R2H 队列/分析脚本、通用训练 wrapper 和 smoke；GPU/训练命令优先通过 `scripts/flip_run.sh <subcommand>` 启动。
- `scripts/archive/`：旧 camera、IK、render/debug、segmentation/inpaint、dataset utility、migration/eval helper、临时训练 shell 和 one-off smoke，按子目录归档。
- `doc/`：当前文档按用途分为 `pipeline/`、`datasets/`、`models/`、`infra/`、`tasks/` 和 `archive/`；入口见 `doc/README.md`。
- `paper/`：相关论文资料。

## 任务工作流

## Codex Skills

- 当前项目的 Claude 旧 skill 已迁移为 Codex 可发现的全局 skill，位于 `/home/leadtek/.codex/skills/flip-*`。
- 用户提到 `/discuss`、`/develop`、`/fix`、`/review`，或提出对应类型需求时，优先使用对应的 `flip-discuss`、`flip-develop`、`flip-fix`、`flip-review` skill。
- Skill 只提供项目工作流提示；本文件和用户当前指令优先级更高。
- 迁移后的 skill 已移除 Claude 专用行为；除非用户明确要求或确认任务工作流步骤，不要自动 commit、建分支、合并或清理 worktree。
- 如果 skill 内容与本文件冲突，以本文件为准，并在执行前向用户说明冲突点。

### 小修改

- 可在 `main` 分支直接修改。
- 修改后运行相关测试或说明未运行原因。
- 如果影响文档、配置或环境，更新 `doc/requirement-log.md` 和对应文档。

### 中大型开发

- 优先使用 `doc/tasks/` 工作流。
- 新任务编号从 `doc/tasks/{pending,active,done,blocked,cancelled}/` 中取最大 id + 1，三位数字。
- 新任务放在 `doc/tasks/pending/NNN.md`，包含：背景、目标、范围、实施计划、验收标准、测试要求。
- 认领任务时从 `pending/` 移到 `active/`，更新 frontmatter，并使用 `.worktrees/tNNN` 开发。
- 禁止在 `main` 上直接做中大型功能开发；使用 `feat/tNNN-<slug>` 或 `fix/tNNN-<slug>` 分支。
- 合并回 `main` 使用 `git merge --no-ff`。
- 完成后将任务移到 `done/`，追加交付记录。

## Review 工作流

- 待审核任务位于 `doc/tasks/done/*.md`。
- frontmatter 中存在 `review` 且不为 `done` 的任务需要审核。
- 审核应查看任务交付记录、相关 commit/diff，并运行相关测试。
- 审核通过后设置 `review: "done"` 和 `review_at`，并追加审核记录。

## Git 约定

- committer：`zzf621`。
- 不要自动 `git commit`，除非用户明确要求或正在执行任务工作流中规定的提交步骤。
- 不要自动创建或删除分支，除非用户明确要求或任务工作流要求。
- 避免修改与当前任务无关的文件；发现已有未提交改动时，不要覆盖。
