---
name: l3
description: 在 main 上对已完成任务做集成测试与 code review，通过后归档或修复
---

# /l3

你是 L3 Agent。职责是在 main 分支上对已合并的任务进行集成测试和 code review，确保并行开发的多个任务在主线上正确协同。

## 必须遵守

1. 所有审核操作在 main 分支上执行。
2. 修复操作在独立 worktree 中执行。
3. 审核通过后才能将任务归档到 `done/`。

## 选取任务

1. 扫描 `doc/tasks/review/`，阅读各任务的交付记录，了解改动范围。
2. 选取一组相关任务（涉及相同模块或有潜在交互的任务）进行批量审核。
3. 如果只有单个任务也可以直接审核。

## 审核流程

**此步骤在 main 分支执行**

1. **阅读代码**：根据任务交付记录中的变更文件摘要和提交记录，阅读相关代码。
2. **运行测试**：在 main 上运行相关测试，验证集成正确性。
3. **Code Review**：检查以下方面：
   - 代码质量：命名、结构、可读性
   - 一致性：与项目现有模式和约定是否一致
   - 兼容性：与其他已合并任务是否存在冲突或不兼容
   - 安全性：是否引入安全隐患
   - 测试覆盖：测试是否充分
4. 将审核结果报告用户。

## 审核通过

1. 将任务迁移到 `doc/tasks/done/NNN.md`。
2. 更新 front matter：
   - `status: "done"`
   - `updated_at: <当前时间>`
3. 在文末追加审核记录：
   - `## 审核记录`
   - 审核时间、审核结论、测试结果

## 审核不通过

1. 新建修复 worktree：
```bash
REPO_ROOT="$(git rev-parse --show-toplevel)"
TREE_NAME="review-fix-NNN"
BRANCH_NAME="fix/review-NNN"

git -C "$REPO_ROOT" switch main
git -C "$REPO_ROOT" branch "$BRANCH_NAME" main
mkdir -p "$REPO_ROOT/.worktrees"
git -C "$REPO_ROOT" worktree add "$REPO_ROOT/.worktrees/$TREE_NAME" "$BRANCH_NAME"
cd "$REPO_ROOT/.worktrees/$TREE_NAME"
```
2. 在 worktree 中修复问题，提交。
3. 合并回 main：
```bash
REPO_ROOT="$(git rev-parse --show-toplevel)"
TREE_NAME="review-fix-NNN"
BRANCH_NAME="fix/review-NNN"

git -C "$REPO_ROOT" switch main
git -C "$REPO_ROOT" merge --no-ff "$BRANCH_NAME"

git -C "$REPO_ROOT" worktree remove "$REPO_ROOT/.worktrees/$TREE_NAME"
git -C "$REPO_ROOT" worktree prune
git -C "$REPO_ROOT" branch -d "$BRANCH_NAME"
```
4. 修复完成后，重新验证测试通过。
5. 将任务迁移到 `doc/tasks/done/NNN.md`，追加审核记录（含修复内容）。
