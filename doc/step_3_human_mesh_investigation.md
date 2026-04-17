# 人体 Mesh 模型调研与集成

## 背景

项目需要在 G1 机器人第一人称视频上叠加合成人体，构造（合成 human，真实 robot）配对数据。原有方案使用简单 capsule mesh（圆柱+球），外观过于粗糙，需要升级为参数化人体模型。

## 已完成工作

### 1. SMPLH 参数化人体模型集成

**模型来源**：MIMO 项目中已有的 `SMPLH_NEUTRAL.npz`

| 属性 | 值 |
|------|-----|
| 顶点数 | 6,890 |
| 面数 | 13,776 |
| 关节数 | 52（身体 22 + 左手 15 + 右手 15） |
| 形状参数 (beta) | 16 维 |
| 蒙皮权重 | 每顶点 52 个关节权重 |

**坐标系转换**：SMPLH (X=left, Y=up, Z=forward) → G1 (X=forward, Y=left, Z=up)
```python
R_SMPLH_TO_G1 = [[0,0,1],[1,0,0],[0,1,0]]
```

### 2. demo_smplh_overlay.py — 人体叠加渲染

实现了 SMPLH mesh 在 G1 第一人称视频上的叠加渲染：

- **SMPLHModel 类**：加载 npz，提取手臂+手部顶点（2786 个），LBS 蒙皮
- **关节映射**：SMPLH 52 关节 → G1 FK link（肩/肘/腕/手指）
- **Per-joint bone correction**：解决 SMPLH T-pose 与 G1 zero-pose 骨骼方向不匹配导致的手臂扭曲
- **输出面板**：Original | G1 Overlay | SMPLH Overlay（可选 Debug Mesh 第四面板）
- **支持参数**：`--scale`（mesh 粗细），`--beta`（体型变化），`--debug-mesh`（线框调试）

**核心 LBS 公式**：
```
v_world = Σ_j w_j × (R_g1_j @ R_bone_j @ R_SMPLH_TO_G1 @ (v - J_j) × scale + t_g1_j)
```

### 3. render_front_compare.py — G1 vs SMPLH 三视图对比

正交投影三视图（Front / Side / Top），G1 居中 T-pose，四周放置 4 个等高 SMPLH 人体：

| 参数 | 值 |
|------|-----|
| G1 身高 | ~1.32m |
| SMPLH 原始身高 | ~1.72m |
| 缩放比 | 0.77x |
| 放置方式 | 前(+X) / 后(-X) / 左(+Y) / 右(-Y)，均不旋转 |
| G1 T-pose | shoulder_roll=±1.57, elbow=1.50 |

### 4. 手部模型调研

#### G1 手部 — Inspire RH56 系列灵巧手

当前 URDF（`g1_29dof_rev_1_0_with_inspire_hand_FTP.urdf`）使用 **Inspire Robots RH56DFTP** 灵巧手：

| 参数 | 值 |
|------|-----|
| 型号 | RH56DFTP（带触觉反馈） |
| DOF | 6 |
| 关节数 | 12 |
| 重量 | 540g |
| 最大握力 | 拇指 6N，四指 4N |
| 手指链 | index(2), middle(2), ring(2), little(2), thumb(4) |

官网：https://en.inspire-robots.com/product/rh56dfx

#### rubber_hand — 仿真人手外壳

`left_rubber_hand.STL` / `right_rubber_hand.STL`：

| 属性 | 值 |
|------|-----|
| 三角面数 | 45,748 / 43,852 |
| 尺寸 | 132 × 67 × 107 mm |
| 所属 URDF | `g1_29dof_rev_1_0.urdf`（无灵巧手版本） |
| 连接方式 | fixed joint → `left_wrist_yaw_link` |
| **骨骼** | **无** — 整只手是一个刚体，手指不能活动 |

这是 G1 没装 Inspire 灵巧手时的橡胶手套，外观仿人手，mesh 精度高但无骨骼驱动。

#### SMPLH 手部

| 属性 | 值 |
|------|-----|
| 每只手顶点数 | ~580（占全身 8.4%） |
| 手部关节 | 15/只手 (5 指 × 3 DOF) |
| 原始手长 | 164mm |
| 缩放后 (0.55x) | 90mm |

SMPLH 手部 mesh 精度很低（580 vs G1 的 45,748 顶点），外观像低面数手套。

#### 更精细的手部模型选项

| 模型 | 顶点数 | 关节 | 特点 |
|------|--------|------|------|
| SMPLH 手部 | 580/手 | 15/手 | 精度低，已集成 |
| MANO | 778/手 | 15/手 | 专用手部模型，从 1000+ 3D 扫描学习 |
| SMPL-X | 10,475 全身 | 54 | SMPL + MANO + FLAME，手部精度最高 |

- MANO 官网：https://mano.is.tue.mpg.de/
- SMPL-X 官网：https://smpl-x.is.tue.mpg.de/

## 身体尺寸对比

| 部位 | G1 机器人 | SMPLH 人体 | 缩放比 |
|------|----------|-----------|--------|
| 身高 | 1.32m | 1.72m | 0.77x |
| 臂长 | 0.27m | 0.51m | 0.53x |
| 手长 | 107mm | 164mm | 0.65x |
| 手宽 | 132mm | 167mm | 0.79x |

G1 的手臂相对身高比人类短（0.53x vs 0.77x 身高比），手掌相对更宽更厚（机械结构）。

## 文件清单

| 文件 | 说明 |
|------|------|
| `scripts/demo_smplh_overlay.py` | SMPLH 人体叠加渲染 demo |
| `scripts/render_front_compare.py` | G1 vs SMPLH 三视图对比 |
| `scripts/generate_human_meshes.py` | 旧版 capsule mesh 生成 |
| `scripts/render_human_overlay.py` | 人体 overlay 渲染（使用 human_arm_overlay.urdf） |
| `data/.../meshes/left_rubber_hand.STL` | 仿真人手外壳 mesh（无骨骼） |
| `test_results/smplh_demo/` | 渲染输出 |
