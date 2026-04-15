# 汇报框架：第一人称人形机器人视频生成

## 目录

1. 上次讨论回顾与新文献
2. 目前思路的调整
3. 问题背景
4. 方法：反向数据构造
5. 目前进度
6. 待解决的问题

- 还有就是关于调研的事情。一方面是实习总共几个月，时间较短，之前跟姜老师讨论过，说是实习结束好像还有个面试，她说是最好有成果；而且我的成果不需要非常成体系，所以我是倾向于有 idea 就直接做。
- 所以这里把完整的背景、动机和方法重新梳理一遍，我觉得只要效果没问题，写个文章应该是问题不大。

---

## 一、上次讨论回顾与新文献

### 上次提到的两个方向

上次提到像素空间编辑困难，因此考察了两个方向：

因为这块文献都说自己效果好，我感觉具体效果怎么样还是得复现看一看，所以就挑了两篇，一篇是 stability ai 做的，一篇是阿里万象做的。

**方向一：Latent 空间动作表示（DisMo）**
- 用 latent motion 表征绕开像素空间的 motion 表示
- 困难：
  - 表示的是整个视频的动态，只能表示简单动作，精细人体动作效果差
  - 绕不开像素空间编辑：latent motion 序列 + 像素细节 = 视频
- 复现效果：只要单帧编辑没问题基本都能处理，但整体效果差

**方向二：外观替换（MIMO）**
- 特征工程绑定第三人称，必须有完整肢体
- 对机器人识别度不高
- 复现效果：效果好，但很多情况识别不到（肢体遮挡、机器人识别置信度低）

**之前的想法：Cosmos Transfer**
- 缺乏强条件约束，参考帧并不是强制首帧
- 首帧注入会有漂移，效果非常差

---

## 二、目前思路的调整

### 第三人称 → 第一人称

放弃第三人称的原因：
- **数据太少**：MIMO / X-Humanoid / LeVerb（第三人称）数据量对比悬殊
- **遮挡难处理**
- **机器人识别度低**

### 编辑方法的问题

**生成式编辑的局限：**
- Ego 数据多为鱼眼透视，文生图模型对此 OOD，效果差
- ControlNet 在边缘情况下效果差（可能因为卷积）

**混合编辑的可能：**
- 宇树 G1 第一人称数据自带姿态数据
- 可结合 MIMO 的做法做精准渲染

**结论：回到第一人称 + 更传统的编辑方法**

---

## 三、问题背景：Human-to-Robot 视频生成现状

### 维度一：视角 × 机器人类型

|              | 机械臂                | 人形机器人                        |
| ------------ | --------------------- | --------------------------------- |
| **第一人称** | Masquerade, Mitty     | **空白（本工作）**                |
| **第三人称** | Human2Robot           | X-Humanoid, Dream2Act            |

- 人形 Ego 中腿部和躯干经常出现，与机械臂差距比较大
- 人形 Ego 视角的数据合成尚无工作
- 宇树 G1 数据集近期变多，数据驱动成为可能

### 维度二：数据构造方向

现有工作均为 **Human → Robot** 方向：
- Masquerade / Mitty：人类视频 → 去手 → inpaint → 渲染机械臂 → (真 human, 合成 robot)
- X-Humanoid：UE5 渲染配对 human/humanoid 动画 → (合成 human, 合成 robot)

**共同问题**：目标端（robot 视频）是渲染/合成的，存在 sim2real gap

### 各工作详述

**Masquerade (ICLR 2026)**
- 任务：第一人称人类视频 → 机械臂视频（rendering overlay）
- 数据：HaMeR 提手部 pose → Detectron2+SAM2 去手 → E2FGVI inpaint → RobotSuite 渲染机械臂
- 局限：rule-based overlay，累积误差大；仅限特定型号机械臂；目标是渲染的

**Mitty (ICLR 2026 under review)**
- 任务：第一人称 human → robot 视频生成（e2e diffusion）
- 方法：Wan 2.2 + LoRA，video in-context learning，双向注意力
- 数据：复用 Masquerade pipeline 合成 ~6,000 对 + H2R 数据集 1,019 对
- 局限：仅限机械臂；依赖文本描述；不保证机械臂物理结构

**X-Humanoid (2512)**
- 任务：第三人称 human → humanoid（Tesla Optimus）视频翻译
- 方法：Wan 2.2 video-in-video-out + LoRA（rank-96，仅 500 steps）
- 数据：UE5 渲染 17+ 小时配对视频（IK Rig Retargeting 对齐骨骼）
- 局限：仅限第三人称；合成数据两端都是渲染的

**Dream2Act (2603)**
- 任务：零样本人形机器人（Unitree G1）视频"幻想" → 提取关节轨迹执行
- 方法：直接用 Seedance 2.0 从图片+文本生成机器人视频
- 局限：第三人称；零样本精度有限（37.5% 成功率）；粗粒度全身动作

---

## 四、方法：反向数据构造 —— Robot → Human

### 核心观察

**反转数据构造方向**：不是在人类视频上渲染机器人（目标合成），而是在机器人视频上合成人手（输入合成）。

```
现有方向：  真 human 视频 → 渲染 robot → (真 human, 合成 robot) → 训练 human→robot 模型
                                         ^^^^^^^^ sim2real gap 在目标端

本工作：    真 robot 视频 → 合成 human → (合成 human, 真 robot) → 训练 human→robot 模型
                                         ^^^^^^^^ domain gap 在输入端（影响更小）
```

### 为什么 domain gap 在输入端更好？

- **目标端是真实的** → 模型学到的是生成真实机器人视频的分布
- **输入端有 gap** → 但推理时输入是真实人类视频，质量 ≥ 合成人类视频，泛化方向是"从差到好"
- **对比**：Masquerade 已证明即使 imperfect overlay 也能提升 5-6x 下游性能 → 输入端的容忍度很高

### Pipeline 设计

```
G1 第一人称视频 + G1 姿态数据
    │
    ├─ 1. Pose 获取：直接读取 G1 关节
    │      → 精确的关节角度，无需视觉 pose estimation
    │      → 这是相比 Masquerade 的关键优势
    │
    ├─ 2. Robot 手臂分割+去除：姿态数据 + Mesh 直接渲染 Mask
    │      → E2FGVI 背景 inpainting
    │
    ├─ 3. 运动学映射：Robot joint state → Human pose
    │
    ├─ 4. Human 渲染
    │
    └─ 得到 (合成 human 视频, 真实 robot 视频) 配对数据
                │
                ▼
        微调 Wan 2.2 + LoRA (video-in-video-out)
                │
                ▼
        推理：真实人类视频 → 目标机器人视频
```

- 视频合成创新点
        - 结合 world action model 范式，联合预测 action 和画面
        - 提升精准性

### 优势

1. **Pose 获取精度高**：用 proprioception 而非视觉估计，消除了 Masquerade 最大的误差源
2. **目标是真实视频**：无 sim2real gap
3. **仅需机器人数据**：不需要手工构造配对视频，只需更换机器人数据源，pipeline 可复用
4. **WAM**：可结合 world action model 范式，联合预测 action 和 video 帧，提升物理精准性

---

## 五、目前进度

- 姿态数据获取
- 精准渲染 Mask
  - 简述已有做法
  - 展示效果

---

## 六、待解决的问题

1. **Human hand rendering 质量**：渲染质量如何？视频连贯性如何？
2. **运动学映射**：G1 → 人是否自然
3. **Mask 擦除效果**：目前 mask 覆盖率较低，擦除效果未知
