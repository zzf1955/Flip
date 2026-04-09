- 首帧编辑 v
- cosmos transfer v
    - 效果差
    - 没有硬性条件约束，参考帧仅为参考
    - 深度图约束太强
- dismo v
    - latent 动作表示
    - 表示整个视频的 action
    - 可做非人，第一人称
    - action 重建效果较差，性能不行
- mimo v
    - 人物换皮
    - 全身，第三视角，人形 效果好
    - 机器人：人体识别有问题，动作提取有问题
    - 无法做第一人称

- 第一人称的困难
    - mimo 的姿态估计，特征工程，能否用到第一人称视频中？
    - 如果使用无监督的 action 表征，就要提供首帧。首帧细节 + 视频运动

- 第三人称视角的困难
    - 应用场景少
    - 数据量小
    - 环境物体
        - 编辑
        - 视频

- 第一人称视频 robot2human
    - mimo 思路提取完整骨骼，TiV Pose 提取关节位置 + KL 提取 3D 位置
    - 反向合成人类视频
    - 得到 (human video, robot video) 数据
    - 微调下游 video2video 模型，由 human 到 robot

- 贡献
    - 第一人称机器人视频合成
    - 数据合成 pipeline，仅需要目标机器人数据，无需手工构造配对视频。
    - sim2real gap：目标视频全是真的，不是渲染出来的。domain gap 在输入端，影响较小。

- 算法贡献
    - 也许可以联合预测 action，类似 world  action model 的思路，看看能否有提升
