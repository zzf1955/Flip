# 手部类型与数据集映射关系

## 一、数据集手型对应表

项目共 8 个 G1 WBT 数据集，按灵巧手类型分两组：

### Inspire Hand（因时 RH56DFTP）— 5 个任务

| 数据集 | 相机 |
|--------|------|
| `G1_WBT_Inspire_Collect_Clothes_MainCamOnly` | head_stereo_left |
| `G1_WBT_Inspire_Pickup_Pillow_MainCamOnly` | head_stereo_left |
| `G1_WBT_Inspire_Put_Clothes_Into_Basket` | head_stereo_left |
| `G1_WBT_Inspire_Put_Clothes_into_Washing_Machine` | head_stereo_left |
| `G1_WBT_Inspire_Put_Clothes_into_Washing_Machine_MainCamOnly` | head_stereo_left |

### BrainCo Hand（强脑 Revo2）— 3 个任务

| 数据集 | 相机 |
|--------|------|
| `G1_WBT_Brainco_Collect_Plates_Into_Dishwasher` | head_stereo_left |
| `G1_WBT_Brainco_Make_The_Bed` | head_stereo_left |
| `G1_WBT_Brainco_Pickup_Pillow` | head_stereo_left |

### 自动检测逻辑

`config.py:get_hand_type()` (L54-58)：按任务名是否包含 `"Brainco"` 判断，返回 `"brainco"` 或 `"inspire"`。

---

## 二、hand_state 数据格式

两种手型的 parquet 中 hand_state 均为 12 维（每手 6 值），但**编码方向和索引顺序都不同**：

**编码方向：**
- **Inspire**: `0.0=闭合(closed)`, `1.0=张开(open)` — 硬件 ANGLE 0=弯曲, 1000=张开, `hs=angle/1000`
- **BrainCo**: `0.0=张开(open)`, `1.0=闭合(closed)` — normalize 时多了 `1.0 -` 反转

### Inspire hand_state 布局（0=closed, 1=open）

| 索引 | 左手 | 索引 | 右手 |
|------|------|------|------|
| 0 | 小指 (little) | 6 | 小指 (little) |
| 1 | 无名指 (ring) | 7 | 无名指 (ring) |
| 2 | 中指 (middle) | 8 | 中指 (middle) |
| 3 | 食指 (index) | 9 | 食指 (index) |
| 4 | 拇指闭合 (thumb_close) | 10 | 拇指闭合 (thumb_close) |
| 5 | 拇指侧摆 (thumb_tilt) | 11 | 拇指侧摆 (thumb_tilt) |

### BrainCo hand_state 布局（0=open, 1=closed）

| 索引 | 左手 | 索引 | 右手 |
|------|------|------|------|
| 0 | 拇指闭合 (thumb_close) | 6 | 拇指闭合 (thumb_close) |
| 1 | 拇指侧摆 (thumb_tilt) | 7 | 拇指侧摆 (thumb_tilt) |
| 2 | 食指 (index) | 8 | 食指 (index) |
| 3 | 中指 (middle) | 9 | 中指 (middle) |
| 4 | 无名指 (ring) | 10 | 无名指 (ring) |
| 5 | 小指 (little) | 11 | 小指 (little) |

### 重排逻辑

`video_inpaint.py:build_q()` 将两种手型统一到 URDF 顺序 `[index, middle, ring, little, thumb_c, thumb_t]`，值域统一为 `0=open, 1=closed`：

```python
# Inspire: 先反转 (1-hs)，再重排 [little,ring,mid,index,...] → [index,mid,ring,little,...]
hs = 1.0 - hs
hs = np.concatenate([hs[[3,2,1,0,4,5]], hs[[9,8,7,6,10,11]]])

# BrainCo: 无需反转，仅重排 [thumb_c,thumb_t,index,mid,ring,little] → [index,mid,ring,little,thumb_c,thumb_t]
hs = np.concatenate([hs[2:6], hs[0:2], hs[8:12], hs[6:8]])
```

---

## 三、hand_state → URDF 关节角度映射

重排+归一化后的 hs（统一为 URDF 顺序 `[index, middle, ring, little, thumb_c, thumb_t]`，0=open, 1=closed）映射到 URDF 60 维 q-vector：

### 手指关节映射表

| 手指 | 关节 | 公式 | 最大角度 (rad) |
|------|------|------|---------------|
| 食指/中指/无名指/小指 | `_1` (近端) | `hs[i] × 1.4381` | 1.4381 |
| 食指/中指/无名指/小指 | `_2` (远端, mimic) | `hs[i] × 1.4381 × 1.0843` | 1.5594 |
| 拇指 | `_1` (侧摆 tilt) | `hs[tilt] × 1.1641` | 1.1641 |
| 拇指 | `_2` (闭合 close) | `hs[close] × 0.5864` | 0.5864 |
| 拇指 | `_3` (mimic of _2) | `hs[close] × 0.5864 × 0.8024` | 0.4704 |
| 拇指 | `_4` (mimic of _3) | `hs[close] × 0.5864 × 0.8024 × 0.9487` | 0.4463 |

### URDF q-vector 手部区间

**注意**：URDF 中手指顺序是 **index → little → middle → ring**，不是顺序排列。

**左手 q[29:41]**：

| q 索引 | 关节 | 来源 hs 索引 |
|--------|------|-------------|
| 29 | left_index_1 | hs[0] |
| 30 | left_index_2 (mimic) | hs[0] |
| 31 | left_little_1 | hs[3] |
| 32 | left_little_2 (mimic) | hs[3] |
| 33 | left_middle_1 | hs[1] |
| 34 | left_middle_2 (mimic) | hs[1] |
| 35 | left_ring_1 | hs[2] |
| 36 | left_ring_2 (mimic) | hs[2] |
| 37 | left_thumb_1 (tilt) | hs[5] |
| 38 | left_thumb_2 (close) | hs[4] |
| 39 | left_thumb_3 (mimic) | hs[4] |
| 40 | left_thumb_4 (mimic) | hs[4] |

**右手 q[48:60]**：

| q 索引 | 关节 | 来源 hs 索引 |
|--------|------|-------------|
| 48 | right_index_1 | hs[6] |
| 49 | right_index_2 (mimic) | hs[6] |
| 50 | right_little_1 | hs[9] |
| 51 | right_little_2 (mimic) | hs[9] |
| 52 | right_middle_1 | hs[7] |
| 53 | right_middle_2 (mimic) | hs[7] |
| 54 | right_ring_1 | hs[8] |
| 55 | right_ring_2 (mimic) | hs[8] |
| 56 | right_thumb_1 (tilt) | hs[11] |
| 57 | right_thumb_2 (close) | hs[10] |
| 58 | right_thumb_3 (mimic) | hs[10] |
| 59 | right_thumb_4 (mimic) | hs[10] |

---

## 四、URDF 与 Mesh 现状

### 当前使用的 URDF

`g1_29dof_rev_1_0_with_inspire_hand_FTP.urdf` — 包含 Inspire RH56DFTP 手部：

- 每只手 12 个关节（6 active + 6 mimic），共 24 关节
- 手指链：index(2), middle(2), ring(2), little(2), thumb(4)
- 含力传感器 link（不参与渲染）

### BrainCo 任务的处理方式

BrainCo 任务**复用 Inspire 手部 mesh**，仅在 `build_q()` 中重排关节角度。渲染出的手部形状是 Inspire 手，不是 BrainCo 手。

### 其他手部 Mesh

| Mesh | 面数 | 所属 URDF | 特点 |
|------|------|----------|------|
| rubber_hand (左/右) | 45,748 / 43,852 | `g1_29dof_rev_1_0.urdf` | 无骨骼，fixed joint，不可动 |
| Inspire 手指 STL | 各数百面 | 主 URDF | 5 指可动，12 关节 |
| human capsule mesh | 低面数 | `human_arm_overlay.urdf` | 圆柱+半球近似 |

---

## 五、已知不兼容问题

### 1. Inspire 手部 Mesh 尺寸偏大

Inspire RH56DFTP mesh 尺寸 132 × 67 × 107 mm，在视频中渲染后明显偏大于实际外观。可能原因：
- 相机标定尚未针对手部精细优化
- Mesh 可能对应不同子型号（RH56DFTP 有触觉反馈版，更大）

### 2. 无 BrainCo 手部 Mesh

项目中不存在 BrainCo Revo2 的 URDF/STL。`unitreerobotics/xr_teleoperate` 仓库有独立的 BrainCo 手部模型但尚未集成。BrainCo 任务渲染的手部形状不正确。

### 3. Mimic Joint 需手动处理

Pinocchio 不自动处理 URDF 中的 mimic joint。`build_q()` 中手动应用了 mimic 系数（1.0843, 0.8024, 0.9487）。如果更换 URDF，这些系数需要同步更新。

### 4. Pickup_Pillow 可交叉验证

`G1_WBT_Inspire_Pickup_Pillow_MainCamOnly` 和 `G1_WBT_Brainco_Pickup_Pillow` 是同一任务的两种手型版本，可用于验证关节映射的一致性。

---

## 六、下一步工作

1. **集成 BrainCo 手部 URDF**：从 `xr_teleoperate` 仓库获取 Revo2 手部 mesh + URDF，创建 `g1_29dof_with_brainco_hand.urdf`
2. **验证手部尺寸**：在多个关键帧上对比渲染手部与视频实际手部，确认是否需要缩放
3. **Pickup_Pillow 交叉验证**：用两种手型的同一任务验证关节重排的正确性
4. **统一手部接口**：考虑将 `build_q()` 中的 if-else 手部映射提取为配置/类
