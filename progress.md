# 项目进度：第一人称机器人视频重绘 Pipeline

## 目标

从宇树 G1 机器人的第一人称视频中，精确分割并去除机器人手臂/身体，生成干净背景视频。用于后续合成人手数据，微调 video-to-video 模型。

## 主流程：Mesh + SAM2 + LaMa

```
原始视频 + 关节数据(parquet)
         │
         ▼
┌─────────────────────────────┐
│  Stage 1: FK Mesh 投影       │
│  URDF + 关节角度 → 正向运动学  │
│  → 鱼眼相机投影 → 各部位 bbox │
│  → SAM2 box prompt          │
└─────────┬───────────────────┘
          │ JPEG 帧 + prompt
          ▼
┌─────────────────────────────┐
│  Stage 2: SAM2 视频分割       │
│  7 个部位独立跟踪传播          │
│  (左/右臂, 左/右手, 左/右腿,  │
│   躯干)                      │
│  每 30 帧 + 新部位出现时重 prompt │
└─────────┬───────────────────┘
          │ per-part mask 序列
          ▼
┌─────────────────────────────┐
│  Stage 3: Mask 后处理 + 修复  │
│  合并部位 mask → 平滑 → 膨胀  │
│  → 边缘模糊 → LaMa inpaint  │
└─────────┬───────────────────┘
          │
          ▼
   7 个中间结果视频 + prompt 可视化
```

### 主脚本

| 脚本 | 功能 | 状态 |
|------|------|------|
| `scripts/sam2_inpaint_pipeline.py` | **完整 pipeline**：FK → SAM2 → LaMa，输出所有中间结果 | **当前主力** |
| `scripts/sam2_segment.py` | SAM2 分割实验（box/point 模式对比） | 已验证，box 效果更好 |

### 输出（`test_results/inpaint_video/{tag}/`）

| 文件 | 内容 |
|------|------|
| `original.mp4` | 原始视频帧 |
| `fk_overlay.mp4` | FK mesh 凸包叠加（验证关节映射） |
| `fk_mask.mp4` | FK 三角面片 mask（粗粒度） |
| `sam2_mask.mp4` | SAM2 分割 mask（彩色，按部位） |
| `sam2_overlay.mp4` | SAM2 mask 叠加原图 |
| `final_mask.mp4` | 后处理 mask（平滑+膨胀+边缘模糊） |
| `inpaint.mp4` | LaMa 修复结果 |
| `prompt_vis/` | prompt 可视化帧（bbox 标注） |

---

## 已完成的工作

### 1. 相机标定

手动标定 + PSO 自动优化，鱼眼相机外参/内参：
- 外参：相对 `head_link` 的偏移 (dx, dy, dz, pitch, yaw, roll)
- 内参：fx, fy, cx, cy, k1-k4 (等距鱼眼模型)
- 最佳 IoU: 0.8970

| 脚本 | 功能 |
|------|------|
| `scripts/interactive_calibrate.py` | GUI 滑动条交互标定 |
| `scripts/auto_calibrate.py` | PSO 粒子群自动标定 |

### 2. Mesh 渲染与验证

| 脚本 | 功能 |
|------|------|
| `scripts/render_g1_skeleton.py` | 俯视图骨骼 + mesh 渲染 |
| `scripts/render_front_view.py` | 正视图彩色 mesh（区分各 link） |
| `scripts/sample_render.py` | 多任务/多 episode 批量 overlay 验证 |
| `scripts/compare_render.py` | 凸包 vs 三角面片渲染对比 |

### 3. URDF 与灵巧手

- **URDF**: `g1_29dof_rev_1_0_with_inspire_hand_FTP.urdf`（Inspire Dex5 五指灵巧手）
- **关节映射**: `robot_q_current`(36) + `hand_state`(12) → URDF q(60)
  - 身体 29 DOF：腿(12) + 腰(3) + 左臂(7) + 右臂(7)
  - 灵巧手 12 DOF/手：index(2) + little(2) + middle(2) + ring(2) + thumb(4)
  - mimic joints 手动设置（pinocchio 不自动处理）

### 4. Mask 生成方案对比

| 方案 | 质量 | 速度 | 时间一致性 |
|------|------|------|-----------|
| FK mesh 投影 | 粗糙，边界不准 | 快 | 好 |
| FK + GrabCut | 中等 | 中 | 中 |
| **FK prompt + SAM2** | **好** | 慢（MPS） | **好** |

### 5. Inpaint 方案对比

| 方案 | 质量 | 速度 | 时间一致性 |
|------|------|------|-----------|
| LaMa 逐帧 | 中等 | 0.3 fps | 差（闪烁） |
| LaMa + EMA 混合 | 中等 | 0.3 fps | 中 |
| ProPainter | 好 | CPU 极慢 | 好 |
| **LaMa + SAM2 mask** | **好** | 0.3 fps | 中 |

---

## 数据布局

```
data/
├── mesh/                          # URDF + STL
│   ├── g1_29dof_rev_1_0_with_inspire_hand_FTP.urdf
│   └── meshes/                    # 51+ STL 文件（含 5 指灵巧手）
└── video/
    └── G1_WBT_Brainco_Make_The_Bed/   # LeRobot 格式数据集
        ├── data/chunk-000/             # parquet（关节状态 30Hz）
        ├── meta/episodes/              # episode → 视频文件映射
        └── videos/observation.images.head_stereo_left/
            └── chunk-000/file-{000-010}.mp4   # 300 episodes, 640x480@30fps
```

---

## 其他辅助脚本

| 脚本 | 功能 |
|------|------|
| `scripts/grabcut_mask.py` | GrabCut mask 优化 + 7 栏面板输出 |
| `scripts/mask_inpaint.py` | 旧版 FK mask + LaMa inpaint |
| `scripts/lama_inpaint.py` | 独立 LaMa inpaint（读预生成 mask） |
| `scripts/inpaint_compare.py` | LaMa vs Flux Fill 对比 |
| `scripts/video_inpaint.py` | 逐帧 FK+GrabCut+LaMa 视频 pipeline |
| `scripts/video_propainter.py` | ProPainter 两阶段 pipeline |
| `scripts/comfyui_client.py` | ComfyUI API 客户端 |
| `scripts/generate_human_meshes.py` | 人手 mesh 生成 |
| `scripts/render_human_overlay.py` | 合成人手叠加 |
| `scripts/make_ppt.py` | 结果 PPT 生成 |
| `scripts/debug_hand_render.py` | 灵巧手 mesh 调试 |

---

## 待做

- [ ] 拆分 SAM2 推理为独立脚本，支持 4090 远程运行（MPS 上 7 obj 太慢）
- [ ] 用 SAM2 大模型 (large) 对比 small 的分割质量
- [ ] 探索 sim-to-real 方案：渲染合成数据训练专用分割模型
- [ ] 时间一致性优化：SAM2 mask + ProPainter inpaint 组合
- [ ] 批量处理多 episode
