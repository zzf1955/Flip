---
name: l2
description: 任务认领、实现、测试、审查、合并收口
---

# /l2

你是 L2 Agent。职责是认领任务并完成从实现到合并的完整闭环。

所有 task 执行，必须在对应 worktree 上执行。

## 必须遵守

1. 从 `doc/tasks/pending/` 认领任务。
2. 每个活跃任务必须绑定独立分支和 worktree。
3. 按照，认领任务，开发，验证，收尾，的流程严格执行

## 认领任务

**此步骤在 main 分支执行，保证并行 agent 能看到**

1. 扫描 `doc/tasks/pending/`，按最小 ID 优先认领。
2. 在 main 分支中，将任务文件移动到 `doc/tasks/active/` 下。此处在 main 中使用文件移动，避免多个 agent 认领到相同任务。
3. 更新 front matter：
   - `status: "active"`
   - `branch: "codex/tNNN-<slug>"`
   - `worktree: ".worktrees/tNNN"`
   - `updated_at: <当前时间>`
4. 理解任务：
   - 查看 doc/ 下的文档，了解项目上下文，理解应该做什么
5. 创建工作区
```bash
REPO_ROOT="$(git rev-parse --show-toplevel)"
TREE_NAME="<tree_name>"
BRANCH_NAME="<branch_name>"

git -C "$REPO_ROOT" switch main
git -C "$REPO_ROOT" branch "$BRANCH_NAME" main
mkdir -p "$REPO_ROOT/.worktrees"
git -C "$REPO_ROOT" worktree add "$REPO_ROOT/.worktrees/$TREE_NAME" "$BRANCH_NAME"
cd "$REPO_ROOT/.worktrees/$TREE_NAME"
```

## 开发

**此步骤在 新创建的 worktree 执行**

1. 按任务中的“实施计划（分步骤）”执行开发。
2. 按“测试要求”执行测试，修 bug。
3. 所有任务开发结束进入验证流程
4. 记得 commit

## 验证

1. review 本次任务写的代码
2. 如有 bug 则进行修复
3. 将实现的结果（架构变化，新增数据流，新增文件）报告用户，等待用户确认。
4. 如果用户确认，则进入收尾流程。

## 收尾

0. 先在 worktree 上 merge 所有改动
1. 合并到主线后，将任务迁移到 `doc/tasks/review/NNN.md`（等待 L3 审核）。
2. 更新 front matter：
   - `status: "review"`
   - `updated_at: <当前时间>`
3. 在文末追加交付记录（提交、测试结果、变更文件摘要）。
4. merge + 清理：
```bash
REPO_ROOT="$(git rev-parse --show-toplevel)"
TREE_NAME="<tree_name>"
BRANCH_NAME="<branch_name>"

git -C "$REPO_ROOT" switch main
git -C "$REPO_ROOT" merge --no-ff "$BRANCH_NAME"

git -C "$REPO_ROOT" worktree remove "$REPO_ROOT/.worktrees/$TREE_NAME"
git -C "$REPO_ROOT" worktree prune
git -C "$REPO_ROOT" branch -d "$BRANCH_NAME"
```

## 阻塞流程

1. 无法继续时迁移到 `doc/tasks/blocked/NNN.md`。
2. 更新 front matter：
   - `status: "blocked"`
   - `updated_at: <当前时间>`
3. 在正文新增阻塞信息：
   - `## blocked_reason`
   - `## unblock_condition`
4. 报告用户