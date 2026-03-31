# start.md

## 流程

- 用户提出了需求，首先读取 `doc/agents/skills/l1/SKILL.md` ，然后按照此 skill 执行
- 用户未提出需求，仅发送了此文件，直接读取 `doc/agents/skills/l2/SKILL.md` ，然后按照此 skill 去领取任务执行。不要再次询问用户是否有需求。

## 行为原则

- 如果遇到环境问题，或者反复出错，首先查询 `doc/notice.md` 有没有解决方案。
- 如果一个问题反复调试，或者工具调用出现问题，解决后记录到 `doc/notice.md` 中

## skill/角色

- `l1`：需求澄清、方案设计、任务拆解、与用户确认。
- `l2`：任务认领、开发实现、测试、代码审查、合并与收口。

## 文档目录

- doc/env.md: 开发环境
- doc/framework.md: 项目整体架构
- doc/notice.md: 项目踩坑记录
- doc/tasks/: agent 任务规划
- doc/agents: skill 存放位置

## Git Worktree 规范

- 所有 task 执行，必须在对应 worktree 上执行。
- 提交的时候使用 git merge --no-ff 这样我能清晰的看到提交记录，不要使用快进
- 使用 zzf621 为名字进行提交

- 开发前：创建新的分支
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

- 开发中提交
```bash
git add -A
git commit -m "xxxx"
```

- 开发后：merge+清理
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
