# 阶段总结：背景擦除 → 人体叠加

## 一、当前阶段成果

### 1.1 相机标定

PSO 自动标定完成，FK 渲染 mask 与 SAM2 GT mask 的 **IoU = 0.8970**。14 个参数（外参 6 + 内参 4 + 畸变 4）已固化到 `config.py`。另完成了梯度标定 (`auto_calibrate_grad.py`) 和关键点标定框架 (`calibrate_keypoints.py`)。

### 1.2 FK Mesh 渲染 Pipeline

完整的正向运动学 + 鱼眼投影 pipeline：
- 加载 URDF (60 DOF = 29 body + 12×2 hands) + 91 个 STL mesh
- 从 parquet 读取关节角度，`build_q()` 统一映射 Inspire/BrainCo 两种手型
- pinocchio FK → 世界坐标 → OpenCV 鱼眼投影 → mask/overlay 渲染

### 1.3 SAM2 分割 + LaMa 修复

主 pipeline (`sam2_inpaint_pipeline.py`) 已跑通：
- FK convex hull → SAM2 box prompt → 7 部位独立跟踪（左/右臂、左/右手、左/右腿、躯干）
- LaMa 神经修复，输出 7 种中间结果视频便于调试
- 每 30 帧自动重 prompt，新部位出现时追加

### 1.4 人体叠加实验

- Capsule mesh 人手臂生成 + overlay 渲染（`generate_human_meshes.py` + `render_human_overlay.py`）
- SMPLH 参数化人体集成：LBS 蒙皮、per-joint bone correction、三视图对比（`demo_smplh_overlay.py`）

### 1.5 调研与文档

- G1 机器人变体调研：身体构型、手部选配、SKU 对应关系 → `doc/g1_variants.md`
- 人体 mesh 模型调研：SMPLH/MANO/SMPL-X 对比、身体尺寸差异 → `doc/human_mesh_investigation.md`
- 手部数据映射关系：手型↔任务↔关节映射 → `doc/hand_data_mapping.md`

---

## 二、已发现的问题

### 2.1 手部 Mesh 不适配

- **Inspire 手偏大**：Inspire RH56DFTP mesh (132×67×107mm) 渲染后明显大于视频中实际外观
- **两种手型仅一套 Mesh**：BrainCo 任务复用 Inspire mesh，仅重排关节角度，渲染形状不正确
- **不同数据需区分手型**：5 个 Inspire 任务 + 3 个 BrainCo 任务，物理外形不同
- 详见 → `doc/hand_data_mapping.md`

### 2.2 躯干 Mesh 不一致

G1 视频中胸口有一个镂空三角形（logo 区域），但当前 mesh 上没有。可能需要单独修改 torso mesh 或在 mask 后处理中补偿。

### 2.3 人形肢体与 G1 不匹配

直接用 G1 骨骼映射人体存在比例问题：

| 部位 | G1 | 人体 (SMPLH) | 比值 |
|------|-----|-------------|------|
| 身高 | 1.32m | 1.72m | 0.77x |
| 臂长 | 0.27m | 0.51m | 0.53x |
| 手长 | 107mm | 164mm | 0.65x |

G1 的手臂相对身高比例远短于人类（0.53x vs 全身 0.77x），关节位置完全不同。

### 2.4 相机标定精度

IoU=0.8970 整体可用，但手部末端等细节区域精度不足。关键点标定 (`calibrate_keypoints.py`) 尚未使用，可进一步提升对齐精度。

---

## 三、下一阶段计划：人体叠加

### 3.1 精调相机参数

1. 标注关键点（拇指顶端、脚尖等具有明确对应关系的位置）
2. 使用 `calibrate_keypoints.py` 的 Adam 优化，pixel-level 对齐 mesh 关键点与图像关键点
3. 关键点全部对齐后，重新评估 mesh 各部位匹配情况

### 3.2 IK-based 人体姿态还原

- 不再直接用 G1 骨骼近似人体（比例差异过大）
- 选取关键点（五指指尖、root、脚尖、脚跟）作为 IK target
- 使用 IK 求解标准人体骨骼姿态
- 渲染标准人体 mesh，不受 G1 比例限制

### 3.3 人体渲染模型选型

| 模型 | 顶点数 | 手部关节 | 适用性 |
|------|--------|---------|--------|
| SMPLH | 6,890 | 15/手 | 已集成，手部低面数 (580 顶点) |
| MANO | 778/手 | 15/手 | 专用手部模型，精度更高 |
| SMPL-X | 10,475 | 15/手 | 全身统一，手部精度最高 |

需根据渲染效果选择。当前 SMPLH 手部 580 顶点偏粗糙。

### 3.4 逐任务手型适配

集成 BrainCo 手部 URDF + mesh，使 BrainCo 任务渲染正确的手部形状。利用 Pickup_Pillow（两种手型版本）做交叉验证。

---

## 四、参考文档

| 文档 | 内容 |
|------|------|
| `doc/progress.md` | 项目全局进展总结 |
| `doc/g1_variants.md` | G1 机器人型号与变体调研 |
| `doc/human_mesh_investigation.md` | 人体 Mesh 模型调研与集成 |
| `doc/hand_data_mapping.md` | 手部类型与数据集映射关系 |
| `doc/scripts_inventory.md` | 脚本清单与分类 |
