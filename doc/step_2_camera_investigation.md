# G1 头部相机调查结论

## 相机型号

WBT 数据集中 `head_stereo_left` / `head_stereo_right` 来自**外置 UVC 双目 RGB 广角相机**，不是 Intel RealSense D435i。

| 项目 | 值 | 来源 |
|------|-----|------|
| 类型 | USB UVC 双目 RGB 相机 | [teleimager/cam_config_server.yaml](https://github.com/unitreerobotics/teleimager/blob/main/cam_config_server.yaml): `type: uvc, binocular: true` |
| 标称 FOV | 125° | [xr_teleoperate/Device.md](https://github.com/unitreerobotics/xr_teleoperate/blob/main/Device.md): "125°FOV, 60mm baseline" |
| 基线 | 60mm | 同上 |
| 输出分辨率 | 1280×480（左右各 640×480） | cam_config_server.yaml: `image_shape: [480, 1280]` |
| FPS | 30 | 同上 |
| 可能的硬件型号 | Cherry Dual Camera (DECXIN) | [teleimager/README.md](https://github.com/unitreerobotics/teleimager/blob/main/README.md): idVendor=7119, idProduct=11599 |

**不是 D435i 的证据：**
- 视频帧是真彩色（R-G 通道差值=12），D435i IR 传感器输出灰度
- D435i 只有 1 个 RGB 传感器，无法产生双目立体彩色流
- URDF 中 `d435_link` 是 D435i 深度相机的安装位置，与此相机无关

## 内参

unitree 官方仓库中没有公开这款相机的出厂内参。当前值来自 PSO 标定，与硬件规格交叉验证一致。

### 已确定参数

| 参数 | 值 | 验证 |
|------|-----|------|
| fx | **290.78** | PSO 标定 + 独立最小二乘 95% CI [284, 297]，对应 HFOV≈95.5°（标称 125° 裁切到 640×480 后合理） |
| fy | **287.35** | PSO 标定，与 VFOV=80° 理论值 240/tan(40°)=286.0 吻合（误差 0.5%） |
| cx | **≈320** | PSO 得到 329，接近图像中心，各实验间稳定 |
| k1-k4 | **= 0** | 图像直线分析确认 edge/center 弯曲度比仅 1.12x，畸变可忽略 |

### 未确定参数

| 参数 | 问题 |
|------|------|
| cy | 与 pitch 强耦合 (r=-0.95)，各任务散度 197~458，不可靠 |

### 验证方法

1. **硬件规格推算**：Device.md 标称 125° FOV → VFOV≈80° → fy=240/tan(40°)=286，与标定值 287.35 几乎完全一致
2. **独立最小二乘**：固定外参后仅拟合 fx/fy，得到 fx=290.7, fy=287.3
3. **逐点解析估计**：fx 中位数 293.4，落在 LS 置信区间内
4. **畸变分析**：对视频帧做 Canny + HoughLinesP 直线检测，边缘/中心弯曲度比 1.12x，确认 pinhole 模型合适

## 外参

### 无可用参考值

URDF 中 `d435_link` 的位置（相对 torso_link: dx=0.058, dy=0.018, dz=0.430, pitch=-47.6°）是 D435i 的安装位置，**不适用于外置双目相机**。外置相机通过 3D 打印支架安装在头部，位置未在任何文件中定义。

### 当前标定值

```python
dx=0.0758, dy=0.0226, dz=0.4484, pitch=-61.59°, yaw=2.17°, roll=0.23°
```

### 标定退化问题

当前使用 4 个关键点（L_thumb, L_toe, R_toe, R_thumb）标定 10 个参数，存在严重退化：

| 耦合对 | 相关系数 | 含义 |
|--------|---------|------|
| pitch ↔ cy | r = -0.95 | 俯仰角和主点 y 几乎完全退化 |
| yaw ↔ cx | r = +0.91 | 偏航角和主点 x 退化 |
| dz ↔ fx | r = +0.77 | 距离和焦距退化 |

各任务 PSO 结果散度极大（fx: 296~424, pitch: -43°~-89°, cy: 197~458），说明标定约束严重不足。

### 改进方向

1. 固定已确定的内参（fx=291, fy=287, cx=320, k1-k4=0），只优化 6 外参 + cy = 7 个参数
2. 增加关键点数量和空间分布（特别是 Y 方向跨度），打破 pitch-cy 退化
3. 物理测量相机安装位置获取外参先验

## Mesh 渲染偏细分析

### Mesh 本身没有问题

- STL 文件使用米单位，包含完整外壳（截面分析确认是空心 shell）
- URDF 与 unitree 官方仓库完全一致
- FK 关节距离与真实 G1 尺寸一致（小腿 300mm 等）
- 所有 visual origin 均为 (0,0,0)，不存在遗漏偏移
- pinocchio frame 无重名冲突

### 偏细原因

内参已确认正确，问题在于外参标定的退化。dz-fx 耦合（r=0.77）导致相机距离和焦距无法独立确定，不同的 (dz, fx) 组合产生不同的投影尺寸。统一标定的外参是对多个任务的折中，可能导致系统性投影偏差。
