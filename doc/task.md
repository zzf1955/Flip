# 阶段总结

## 一、已完成

### 1.1 相机标定

- **相机型号确认**：外置 UVC 双目 RGB（非 D435i），125° FOV
- **内参确定**：fx=290.78, fy=287.35, cx≈320, k1-k4=0（pinhole 模型）
- **标定实验**：mask Dice F1=0.948, 关键点 RMSE=16.1px
- 输出 → `output/calibration/`

### 1.2 Inpaint Pipeline

- FK → SAM2 box prompt → LaMa/ProPainter，7 部位独立跟踪
- 逐帧 FK + GrabCut + LaMa
- 多 GPU 批量调度
- 输出 → `output/inpaint/`

### 1.3 人体 Retarget

- SMPLH 模型集成（6890 顶点，LBS 蒙皮）
- G1→SMPLH 关节拷贝 retarget（bone alignment + spine split + twist injection）
- IK arm refinement（L-BFGS）
- retarget 误差：17.07px / 71.40mm
- 输出 → `output/human/`

### 1.4 代码重构

- `scripts/`（25文件平铺）→ `src/`（core/pipeline/tools 三层）
- 消除 god module + 代码重复 + 统一输出路径
- 30 个脚本 import 测试 + E2E 验证通过

---

## 二、已知问题

### 2.1 手部 Mesh 不适配

- BrainCo 任务复用 Inspire mesh（跳过渲染）
- 需集成 BrainCo Revo2 URDF + STL

### 2.2 人形肢体比例不匹配

G1 手臂仅人类的 0.53x，通过 scale=0.75 + hand_scale=1.3 部分补偿。

### 2.3 相机外参退化

pitch↔cy 耦合 r=-0.95，外参和内参无法独立标定。已固定内参只优化外参+cy。

---

## 三、下一阶段：Human 渲染 + 视频生成

### 3.1 Human 渲染质量提升

- SMPLH mesh 提供轮廓和透视
- 深度图 → ControlNet 重绘细节
- 处理头颈遮挡（当前已裁剪头部三角面）

### 3.2 视频生成

- (合成 human, 真实 robot) 配对 → Wan 2.1 + LoRA 微调
- 可选：World Action Model 联合预测 action + video

---

## 四、参考文档

| 文档 | 内容 |
|------|------|
| `CLAUDE.md` | 项目配置 + 代码结构 + 运行示例 |
| `doc/progress.md` | 全局进展总结 |
| `doc/scripts_inventory.md` | 新架构详细说明 |
| `doc/camera_investigation.md` | 相机型号 + 标定详情 |
| `doc/hand_data_mapping.md` | 手部编码映射 |
| `doc/g1_variants.md` | G1 机器人型号变体 |

## 2026-04-24 — 整体重构：配置化路径、实验管理、Pipeline 与训练文档重划分

### 需求

> 统一路径管理与实验设置到 config；明确 Seedance direct、Seedance + Human Mesh/inpaint、Mitty transfer、Mitty 三阶段 LoRA pipeline；统一实验命名和参数管理并记录数据用量；实验类型通过参数指定；完全与外部 Diff sync 解耦；重写 train/Step5 文档。

### 计划

- 新建 T033：`doc/tasks/pending/033.md`，等待确认后进入 active。
- 计划分支：`feat/t033-config-pipeline-experiment-refactor`。
- 计划 worktree：`.worktrees/t033`。
- 实施重点：config 路径中心化、受控 `experiment_type`、自动实验名、manifest、LoRA 维度自动检测、DiffSynth sync 解耦、Step5 文档重划分、`AGENTS.md` pipeline 索引。

### 相关任务

- [033] 整体重构：配置化路径、实验管理、Pipeline 与训练文档重划分
