# CLAUDE.md

- 本文件仅提供：
    - 项目背景信息
    - agent 行为规范
    - 环境信息
    - 其他参考信息源
    - 其他文档指路
- 不提供
    - 具体的项目内容说明
    - 具体的文件结构
    - 具体的算法说明

## 项目背景

FLIP: Flipped-Direction Learning via Inpainting Ppeline for Cross-Embodiment Video Editing

FLIP 是一个仿人机器人视频生成合成系统，采用**逆向生成数据合成**方法：
- 传统方向：人类视频 → 机器人视频
- 本项目方向：**G1 机器人视频 → 人类视频**（用于训练 video editing 模型）
- 核心优势：零域差（输出为真实机器人视频）、一对多数据增强、无需配对采集

详见 `doc/idea_humanoid.md`（核心思路）和 `doc/experiments.md`（实验计划）。

## 环境

- `conda activate videoedit`
- 代理：clash 默认端口 7897，一般在虚拟网卡环境下工作
- 提交者名称：`zzf621`

## Agent 行为规范

- 入口文档：`doc/agents/start.md`
- 遇到环境问题或反复出错，先查 `doc/notice.md`
- 解决新问题后记录到 `doc/notice.md`

### 角色分工

| 角色 | 职责 |
|------|------|
| L1 | 需求澄清、方案设计、任务拆解、与用户确认 |
| L2 | 任务认领、开发实现、测试、代码审查、合并与收口 |
| L3 | 集成测试、代码审查、归档 |

### Git Worktree 规范

- 所有 task 在对应 worktree 上执行
- 合并使用 `git merge --no-ff`（保留分支历史）
- 分支命名：`codex/tNNN-<slug>`
- worktree 路径：`.worktrees/tNNN`

## 文档索引

| 路径 | 内容 |
|------|------|
| `doc/env.md` | 开发环境配置 |
| `doc/framework.md` | 项目整体架构 |
| `doc/notice.md` | 踩坑记录 |
| `doc/data.md` | 数据说明 |
| `doc/tasks/` | 任务管理（pending/active/blocked/review/done） |
| `doc/agents/` | Agent skill 定义 |
| `paper/` | 参考论文 |
