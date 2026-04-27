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

FLIP 是第一人称人形机器人视频生成研究项目：在真实 G1 机器人视频上合成人体，构造 `(synthetic human, real robot)` 配对数据，用于微调 video-to-video 模型（Wan 2.2 + LoRA）。

主流程：

1. G1 pose 获取：关节编码器 / 本体感知。
2. Robot 分割与去除：FK → mesh → SAM2 mask → 背景 inpaint。
3. Robot-to-human retarget：G1 pose → SMPLH pose。
4. Human 渲染 / 重绘：SMPLH mesh → ControlNet 或视频生成模型。
5. 训练：`(human, robot)` 配对数据 → Wan 2.2 / Mitty-style / LoRA。

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

- 需要 GPU / CUDA 的命令优先使用统一入口，便于 Codex 按子命令自动批准越权：

```bash
scripts/flip_run.sh train --cuda 2,3 --nproc 2 -- <train args>
scripts/flip_run.sh mitty_cache --cuda 0 -- <mitty_cache args>
scripts/flip_run.sh sam2_precompute --cuda 0 -- <sam2_precompute args>
scripts/flip_run.sh nvidia-smi
```

## 代码结构

- `src/core/`：基础库模块，不直接作为主入口运行。
- `src/pipeline/`：可执行 pipeline 与训练脚本。
- `src/tools/`：标定、调试、可视化、日志转换等工具。
- `scripts/`：旧脚本和归档脚本，优先使用 `src/` 下的新结构。
- `doc/`：设计、实验、进度和任务文档。
- `paper/`：相关论文资料。

## 任务工作流

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
