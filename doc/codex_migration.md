# Claude 到 Codex 迁移说明

创建时间：2026-04-24T06:11:34Z

## 迁移目标

- 保留 Claude 时代的项目规则和任务工作流。
- Codex 中尽量放开项目内读写和命令执行权限。
- GPU/训练场景使用 full access 直接访问宿主设备，并通过 Codex hook 做高危 Bash 命令护栏。
- `.claude/` 暂不删除，作为历史配置和对照来源。

## Codex 权限策略

当前机器的 Codex 全局配置位于 `~/.codex/config.toml`。

已为本项目配置：

```toml
approval_policy = "never"
sandbox_mode = "danger-full-access"

[projects."/disk_n/zzf/flip"]
trust_level = "trusted"

[sandbox_workspace_write]
writable_roots = [
  "/disk_n/zzf/flip",
  "/tmp",
  "/disk_n/zzf/.cache/huggingface",
  "/disk_n/zzf/.pip_cache",
]
network_access = true

[features]
codex_hooks = true
```

含义：

- `trust_level = "trusted"`：Codex 认为该项目可信，减少不必要确认。
- `sandbox_mode = "danger-full-access"`：让 GPU/训练命令直接访问宿主 `/dev/nvidia*`，不再进入 bwrap 文件系统沙箱。
- `approval_policy = "never"`：不再弹出命令确认，依赖 hook 做基础高危命令拦截。
- `codex_hooks = true`：启用 Codex hooks；当前全局 `~/.codex/hooks.json` 指向 `scripts/codex_pre_tool_use_guard.py`。
- `writable_roots`：保留为历史/回退到 `workspace-write` 时使用；full access 模式下不再作为强制写入边界。
- `network_access = true`：允许沙箱内网络访问；若当前 Codex 版本不支持该字段，则以 CLI 实际行为为准。

注意：`sandbox_mode` 和 `approval_policy` 必须放在 `config.toml` 顶层；`[projects."/disk_n/zzf/flip"]` 当前只保留 `trust_level`。如果把 sandbox/approval 写进项目表，Codex 可能仍按默认或会话权限启动。

## Full access + Hook 策略

当前为了减少 GPU 训练命令反复确认，项目使用 `danger-full-access` 配合 Bash `PreToolUse` hook。该模式可直接运行 `nvidia-smi`、CUDA/PyTorch、`torchrun` 和多卡训练，但安全性低于 `workspace-write` 沙箱。

Hook 配置：

- 全局配置文件：`~/.codex/hooks.json`
- 项目 hook 脚本：`scripts/codex_pre_tool_use_guard.py`
- 拦截范围：Codex shell/Bash 工具调用前的命令字符串
- 当前 matcher 使用 `Bash|exec_command|shell|exec_command`，兼容不同 Codex 版本/工具名；旧的 `^Bash$` 在 full access 会话中可能匹配不到。

Hook 会阻止：

- 提权命令：`sudo`、`su`、`doas`、`pkexec`
- setuid/setgid：`chmod +s`、setuid/setgid 数字模式
- root 归属修改：`chown root`、`chown 0`
- 明显危险删除：对 `/`、`~`、`/home/leadtek`、`/disk_n`、`/disk_n/zzf`、`/home/leadtek/.codex` 等路径递归删除
- 项目外递归删除：`rm -r` / `rm -rf` 指向非白名单绝对路径
- 危险 Git 操作：`git reset --hard`、`git clean`、`git push --force`、`git push -f`

Hook 不等同于强沙箱。它不能保证覆盖所有非 Bash 工具、脚本内部动态行为或复杂 shell 语义；执行高风险系统操作前仍必须明确告知用户并等待确认。

当前实测状态：`ps -p 1 -o args=` 显示宿主 init 而非 `codex-linux-sandbox`，`/dev/nvidia0-3` 与 `/dev/nvidiactl` 可见，`scripts/flip_run.sh nvidia-smi` 可正常输出 4 张 4090D；`sudo -n true` 会被 PreToolUse hook 拦截。

## Linux workspace sandbox 现状（历史/回退说明）

当前 Linux 环境中 Codex 使用 bwrap/bubblewrap 风格的沙箱执行 shell 命令。实测特征：

- 命令进入新的 mount、PID 和 user namespace；`NoNewPrivs=1`，不能在沙箱内再提权。
- 宿主根目录 `/` 以只读方式挂载；项目目录和 `writable_roots` 中的目录以可写方式挂载。
- `/disk_n/zzf/flip/.git` 和 `/disk_n/zzf/flip/.codex` 被只读保护，避免误改仓库元数据和 Codex 项目配置标记。
- `/dev` 不是宿主机完整 `/dev`，而是沙箱创建的最小设备目录，只包含 `null`、`zero`、`random`、`urandom`、`tty`、`pts`、`shm` 等基础设备。
- `/proc/driver/nvidia` 和 `/sys/bus/pci/drivers/nvidia` 可能可见，但 `/dev/nvidia*` 不会透传；因此 `nvidia-smi`、CUDA、PyTorch GPU 训练/推理在普通 `workspace-write` 沙箱内不可用。

`writable_roots` 只控制文件写入白名单，不能把 `/dev/nvidia*` 这类设备节点透传进 bwrap 沙箱。当前 `codex-cli 0.124.0` 暴露的配置只包含 `read-only`、`workspace-write`、`danger-full-access` 三档 sandbox mode，没有发现“保持 workspace-write，但额外 dev-bind NVIDIA 设备”的项目级配置项。

## GPU 命令统一入口

GPU 命令应优先走统一入口 `scripts/flip_run.sh`，便于保持命令形态稳定并集中设置环境变量。

示例：

```bash
scripts/flip_run.sh nvidia-smi
scripts/flip_run.sh mitty_cache --cuda 0 -- --pair-dir training_data/pair/1s/train --output training_data/cache/1s/train --device cuda:0 --no-frames
scripts/flip_run.sh sam2_precompute --cuda 0 -- --task all --device cuda:0 --resume
scripts/flip_run.sh train_mitty --cuda 2,3 --nproc 2 -- --cache-train training_data/cache/1s/train --cache-eval training_data/cache/1s/eval
```

如需回退到强文件系统边界，可将 `sandbox_mode` 改回 `workspace-write`，并将 `approval_policy` 改回 `on-request`；此时 GPU 命令需要再次通过越权请求在沙箱外执行。

## Claude allow list 对照

原 `.claude/settings.local.json`：

```json
{
  "permissions": {
    "allow": [
      "Bash(*)",
      "Read(*)",
      "Edit(*)",
      "Write(*)",
      "Fetch(*)",
      "WebFetch(*)",
      "WebSearch(*)",
      "mcp__notion__*"
    ]
  }
}
```

Codex 对照：

| Claude 权限 | Codex 迁移方式 |
| --- | --- |
| `Bash(*)` | `trust_level = "trusted"` + `danger-full-access` + `approval_policy = "never"`；由 Codex hooks 拦截高危 Bash 命令 |
| `Read(*)` | Codex 默认可读；沙箱允许读取文件 |
| `Edit(*)` / `Write(*)` | full access 会话可写宿主用户权限允许的路径；项目规则要求避免无关文件和项目外 destructive 操作 |
| `Fetch(*)` / `WebFetch(*)` / `WebSearch(*)` | 依赖 Codex 网络和搜索配置；项目规则中不硬编码 token |
| `mcp__notion__*` | 需要单独通过 `codex mcp` 配置 Notion MCP |

## Claude skill 对照

| Claude skill | Codex 中的迁移位置 |
| --- | --- |
| `.claude/skills/develop/SKILL.md` | `AGENTS.md` 的中大型开发和任务工作流 |
| `.claude/skills/discuss/SKILL.md` | `AGENTS.md` 的任务创建规范；Codex 默认通过对话和 plan 工具澄清需求 |
| `.claude/skills/fix/SKILL.md` | `AGENTS.md` 的小修改流程 |
| `.claude/skills/review/SKILL.md` | `AGENTS.md` 的 Review 工作流 |

## 项目级规则

Codex 自动读取仓库根目录的 `AGENTS.md`。该文件承担 Claude 时代 `CLAUDE.md` 与 `.claude/skills/*` 的主要迁移职责：

- 中文回答。
- 不直接看视频/图片，除非用户要求。
- 禁止宽泛异常吞噬和 fallback 到旧行为。
- 记录项目背景、环境、目录结构。
- 保留 task/worktree/review 工作流。
- 明确安全边界：不请求 `danger-full-access`，不删除项目外文件。

## MCP 迁移

已从 `notion_mcp_config.json` 迁移 Notion MCP 到 Codex 全局配置：

```bash
codex mcp add notion \
  --env OPENAPI_MCP_HEADERS=<redacted> \
  --env HTTP_PROXY= \
  --env HTTPS_PROXY= \
  --env http_proxy= \
  --env https_proxy= \
  --env NO_PROXY=api.notion.com \
  -- npx -y @notionhq/notion-mcp-server
```

验证命令：

```bash
codex mcp list
codex mcp get notion --json
```

注意：`OPENAPI_MCP_HEADERS` 包含 Notion token，不应提交到 git 或打印到公开日志。

## 尚未迁移

- Claude hook：`.claude/settings.local.json` 中的 `PreToolUse` hook 调用 `.claude/hooks/guard.sh`，但当前仓库没有发现该脚本；Codex 侧用沙箱替代该安全能力。
- Claude slash command：Codex 不直接复用 `/develop`、`/fix`、`/review` 语法，改为通过自然语言触发对应工作流。

## 验证建议

1. 重启 Codex 会话，使新的 `~/.codex/config.toml` 生效。
2. 执行 `ps -p 1 -o args=`，确认不是 `codex-linux-sandbox` / `bwrap`，且未显示 `workspace-write`。
3. 执行 `scripts/flip_run.sh nvidia-smi`，确认可访问 `/dev/nvidia*` 和 GPU driver。
4. 执行 `sudo -n true`，确认被 `Codex PreToolUse guard blocked command` 拦截；不要用真实重要路径做删除测试。
5. 如需检查 Notion MCP，执行 `codex mcp list`，确认 `notion` 状态为 `enabled`。
