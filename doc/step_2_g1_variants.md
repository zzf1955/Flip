# 宇树 G1 机器人型号与变体调研

## 1. 宇树人形机器人产品线

宇树科技共有 6 个人形机器人系列：

| 型号 | 定位 | 身高 | 自由度 | 起售价 | 发布时间 |
|------|------|------|--------|--------|----------|
| R1 | 消费级 | ~121cm | 20-40 DOF | $4,900 | 2025.7 |
| **G1** | **紧凑型/研究级** | **~132cm** | **23-43 DOF** | **$13,500** | **2024.5** |
| G1-D | 轮式操作平台 | G1上半身+轮底盘 | 19-31 DOF | — | 2025.11 |
| H1 | 旗舰研究级 | ~180cm | 19-27 DOF | $90,000 | 2023 |
| H1-2 | H1 升级版 | ~180cm | 27 DOF | ~$150,000 | 2024 |
| H2 | 中高端/工业级 | ~180cm | 31 DOF | $29,900 | 2025 |

## 2. G1 身体构型

G1 有两种身体构型（23DOF / 29DOF），差异在腰部和手臂：

| 部位 | 23DOF (Basic) | 29DOF (EDU Plus+) |
|------|---------------|-------------------|
| 腿 | 6 x 2 = 12 | 6 x 2 = 12 |
| 腰 | **1** DOF (仅 yaw) | **3** DOF (yaw+roll+pitch) |
| 臂 | **5** x 2 = 10 (无腕) | **7** x 2 = 14 (含腕 roll/pitch) |

## 3. G1 手部选配

| 手型 | 厂商 | 手指 | DOF/手 | 加身体后总 DOF | 特点 |
|------|------|------|--------|---------------|------|
| **Rubber Hand** | 宇树 | 一体式 | 0 (fixed) | 29 | 不可动橡胶手，EDU Plus 默认配置 |
| **Unitree Dex3** | 宇树自研 | 3指 | 7 | 43 | 力控三指，适合工业抓取 |
| **Inspire RH56 FTP** | Inspire Robots | 5指 | 6 (12关节) | 41 | 仿人五指，适合遥操作 |
| **BrainCo Revo2** | 强脑科技 | 5指 | 6 (11关节) | ~41 | 轻量五指，内置 RGB 摄像头 |

### EDU Plus 手部确认

通过 URDF 数据确认，`g1_29dof_rev_1_0.urdf` 中 EDU Plus 自带 **rubber_hand（橡胶手）**：

```
left_hand_palm_joint (fixed) → left_rubber_hand   # meshes/left_rubber_hand.STL
right_hand_palm_joint (fixed) → right_rubber_hand  # meshes/right_rubber_hand.STL
```

不是"没有手"，而是一体式不可动手掌。

## 4. G1 主要 SKU

| 代号 | 名称 | 身体 DOF | 手部 | 总 DOF | 价格 |
|------|------|----------|------|--------|------|
| — | Basic | 23 | 无/固定 | 23 | ~$17,990 |
| U1 | EDU Standard | 23 | 无/固定 | 23 | ~$43,900 |
| U2 | EDU Plus | 29 | Rubber Hand | 29 | ~$53,900 |
| U3 | Ultimate A | 29 | Dex3 三指 | 43 | ~$64,300 |
| U5 | Ultimate C | 29 | Inspire 五指 | 41 | ~$66,300 |
| U6 | Ultimate D | 29 | Inspire 五指+触觉 | 41 | ~$73,900 |

Basic 无 SDK（纯演示），EDU 以上有完整开发能力。

## 5. URDF 初始姿态分析

### 关节 origin rpy 偏移

URDF 中多个关节的 origin 有非零 rpy，这是 STL 导出的参考构型补偿：

| 关节 | rpy 偏移 | 含义 |
|------|----------|------|
| left/right_hip_roll_joint | pitch -0.1749 rad (-10.0°) | 大腿微弯 |
| left/right_knee_joint | pitch +0.1749 rad (+10.0°) | 膝盖反向补偿 |
| left_shoulder_pitch_joint | roll +0.2793 rad (+16°) | 手臂自然外展 |
| right_shoulder_pitch_joint | roll -0.2793 rad (-16°) | 手臂自然外展 |

### 左右对称性

所有关节偏移完全左右对称，URDF 零位**不会**导致机器人向左或向右歪。
如果渲染出歪，原因来自数据集中 freeflyer 四元数 `rq[3:7]` 反映的实际机器人朝向。

### 代码处理

`video_inpaint.py` 中的 `build_q` + `do_fk` 正确处理：
- Pinocchio FK 引擎自动处理 joint origin rpy 偏移
- 四元数转换 `(w,x,y,z)` → Pinocchio `(x,y,z,w)` 正确

## 6. 渲染对比

### 资源来源

| 仓库 | 内容 |
|------|------|
| `unitreerobotics/unitree_ros` | G1 所有身体 URDF + 165 STL mesh |
| `unitreerobotics/xr_teleoperate` | Dex3 / BrainCo / Inspire 独立手 URDF + mesh |

### 渲染脚本

`scripts/render_all_variants.py` — 正交正面渲染，支持 filled / wireframe 模式。

### 输出

| 文件 | 内容 |
|------|------|
| `test_results/g1_body_variants.png` | 4 种身体变体正面填充渲染 (23DOF, 29DOF, 29DOF+Unitree Hand, 29DOF+Inspire FTP) |
| `test_results/g1_hand_variants.png` | 3 种灵巧手正面填充渲染 (Dex3, BrainCo, Inspire) |
| `test_results/g1_29dof_wireframe.png` | 29DOF 躯干线框图 (5000x3620, pelvis+waist+torso) |

### 本项目使用的配置

URDF: `g1_29dof_rev_1_0_with_inspire_hand_FTP.urdf`
- 身体: 29 DOF (3DOF 腰 + 7DOF 臂)
- 手部: Inspire RH56DFTP 五指手 (6DOF x 2 = 12)
- 总: 29 + 12 = **41 DOF** (对应 Ultimate C / U5)
- 数据集中同时有 Inspire 和 BrainCo 两种手型任务，`config.py` 中 `get_hand_type()` 根据任务名自动切换
