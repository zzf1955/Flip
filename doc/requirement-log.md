# 需求日志

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
