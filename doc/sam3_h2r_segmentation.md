# SAM3.1 H2R 机械臂/夹爪分割复现记录

## 目标

验证 SAM3/SAM3.1 是否能在 H2R robot-camera 视频中，通过 text condition 和交互提示稳定分割画面中的机械臂/夹爪区域，并给出可复用的 prompt 与参数建议。

## 仓库与环境

- 官方仓库：`ref-sam3/`
- 当前复现环境：conda env `sam3`
- Python：3.12
- PyTorch：`2.10.0+cu128`
- TorchVision：`0.25.0+cu128`
- HuggingFace token：从项目 `.env` 的 `HUGGINGFACE_TOKEN` 读取
- HuggingFace cache：`/disk_n/zzf/.cache/huggingface`

SAM3.1 checkpoint 已缓存到：

```text
/disk_n/zzf/.cache/huggingface/hub/models--facebook--sam3.1/snapshots/daa63191845a41281374e725f4c9e51c7a824460/sam3.1_multiplex.pt
```

官方仓库 `build_sam3_predictor(version="sam3.1")` 支持 video session API：

```text
start_session -> add_prompt -> propagate_in_video -> close_session
```

`add_prompt` 支持 text、box 和 point 提示。SAM3.1 multiplex 中：

- text / box prompt 走 SAM3 video grounding 路径。
- point prompt 走 SAM2 instance refinement 路径，需要传 `obj_id`。
- text prompt 与 point prompt 不能在同一次 `add_prompt` 请求里混用；实际流程应先用 text 建立 object，再用 point refine 已有 object。

## 模型规模

基础 SAM3：

```text
官方 README 参数量：848M
facebook/sam3/sam3.pt: 3,450,062,241 bytes
facebook/sam3/model.safetensors: 3,439,938,512 bytes
```

SAM3.1 multiplex：

```text
本地 checkpoint tensor 参数量：874,365,676
facebook/sam3.1/sam3.1_multiplex.pt: 3,502,755,717 bytes
```

## 显存观察

测试在物理 GPU 2 上运行。当前机器上 GPU 2 长期已有约 4.4GiB 其他进程占用。

实测：

- SAM3.1 16 帧 text segmentation 可跑通，峰值约 16.2GiB。
- 32 帧和 80 帧在当前 GPU 占用下 OOM。
- h2r 16 帧批量 sweep 使用 `max_num_objects=1` 或 `4` 均可稳定运行。
- keyframe point-refine 的峰值仍约 16.2GiB，因为流程包含 text grounding 传播；单帧 point-refine 传播阶段峰值较低，但效果不稳定。

推荐先用短 clip 或分段处理长视频。

## H2R 测试数据

测试视频均来自 `data/h2r/v1/video/*/*/robot_camera.mp4`，原始尺寸为 `426x240`，30fps。已测试：

```text
roll/episode_0
push_box_random_v1/episode_0
grab_two_cubes2_v1/episode_2
grab_cup_v1/episode_0
cloth/cloth12/episode_0
```

片段设置：

```text
initial segment: start_frame=0, stride=2, max_frames=16
mid segment:     start_frame=60, stride=3, max_frames=16
fps output:      8
```

## Text Prompt 结论

整条可见机械臂最稳定的 text prompt 是：

```text
robot arm
```

备用 prompt：

```text
robotic arm
```

不推荐作为主 prompt：

```text
robot
humanoid robot
mechanical arm
robot hand
robot gripper
mechanical gripper
end effector
```

原因：

- `robot arm` 在初始片段和中段片段上覆盖最全面，通常每帧只输出 1 个目标。
- `robotic arm` 在中段 sweep 中也很稳定，可作为 text retry。
- `robot gripper`、`mechanical gripper`、`end effector` 对这批 H2R robot-camera 视频基本没有稳定响应。
- `robot` 只在部分视频有效，不如 `robot arm` 稳定。

46 个 text/prompt 组合的聚合统计：

```text
robotic arm        n= 5 mean_score=125.48 median=120.59 nonempty=1.00 mean_iou=0.81 area_cv=0.33
robot arm          n=10 mean_score=120.09 median=120.58 nonempty=0.96 mean_iou=0.84 area_cv=0.28
mechanical arm     n= 5 mean_score= 59.13 median= 40.75 nonempty=0.46 mean_iou=0.90 area_cv=0.07
robot hand         n= 5 mean_score= 51.11 median= 46.25 nonempty=0.41 mean_iou=0.92 area_cv=0.12
robot              n= 5 mean_score= 40.63 median=  0.00 nonempty=0.39 mean_iou=0.93 area_cv=0.16
end effector       n= 5 mean_score=  9.40 median=  0.00 nonempty=0.07 mean_iou=0.98 area_cv=0.02
humanoid robot     n= 3 mean_score=  0.00 median=  0.00 nonempty=0.00 mean_iou=1.00 area_cv=0.00
mechanical gripper n= 3 mean_score=  0.00 median=  0.00 nonempty=0.00 mean_iou=1.00 area_cv=0.00
robot gripper      n= 5 mean_score=  0.00 median=  0.00 nonempty=0.00 mean_iou=1.00 area_cv=0.00
```

## `max_num_objects` 结论

`robot arm` 基本只产生单对象输出，因此 `max_num_objects=1` 与 `max_num_objects=4` 在中段测试上稳定性一致：

```text
video                      maxobj4_score maxobj1_score maxobj4_nonempty maxobj1_nonempty
grab_cup_ep0                     142.77       142.77            1.00            1.00
push_box_random_ep0              120.63       120.63            1.00            1.00
cloth12_ep0                      120.53       120.53            1.00            1.00
grab_two_cubes_ep2               112.08       112.08            1.00            1.00
roll_ep0                          96.42        96.42            0.94            0.94
```

推荐默认使用：

```text
max_num_objects=1
```

这样可以降低后续把桌面物体或局部反光误作为额外 object 的风险。

## 推荐配置

整条机械臂分割：

```text
version: sam3.1
prompt: robot arm
backup prompt: robotic arm
max_num_objects: 1
multiplex_count: 16
output_prob_thresh: 0.5
offload_video_to_cpu: true
async_loading_frames: false
compile: false
use_fa3: false
use_rope_real: false
```

若第 0 帧 `robot arm` 输出为空，可 retry `robotic arm`，或在更清晰的 keyframe 上加 prompt。

## 夹爪 / 末端候选

仅靠 text prompt 分割夹爪不稳定。`robot gripper`、`mechanical gripper`、`end effector` 均未在当前 H2R 数据上稳定检出。

测试了两类 point refinement：

### 单帧 point refine

流程：

1. 第 0 帧用 `robot arm` 得到整臂 mask。
2. 从整臂 mask 自动取末端点作为正点击，并取中心/相对远端作为负点击。
3. 用 point prompt refine 同一个 `obj_id`。
4. 向后传播。

结果：

```text
point_distal / point_lower over 3 videos: nonempty=1/16 each
```

结论：单帧 point refine 可以把第 0 帧 mask 缩小到末端候选，但传播后很快消失，不适合稳定夹爪分割。

### Keyframe point refine

流程：

1. 先用 `robot arm` text prompt 跑完整段，得到整臂轨迹。
2. 在 keyframes `0,4,8,12` 上，从整臂 mask 自动取 point prompt。
3. 对同一个 `obj_id` 多次 refine。
4. 再传播一次。

测试策略：

- `keypoint_distal`：取离整臂 mask 质心最远的端点作为正点击。
- `keypoint_lower`：取整臂 mask 下端点作为正点击。

结果：

```text
push_box_random_ep0      keypoint_distal nonempty=16/16 mean_area_ratio_vs_robot_arm=0.068
grab_two_cubes_ep2       keypoint_distal nonempty=15/16 mean_area_ratio_vs_robot_arm=0.087
grab_cup_ep0             keypoint_distal nonempty=16/16 mean_area_ratio_vs_robot_arm=0.197

push_box_random_ep0      keypoint_lower  nonempty=16/16 mean_area_ratio_vs_robot_arm=0.382
grab_two_cubes_ep2       keypoint_lower  nonempty=16/16 mean_area_ratio_vs_robot_arm=0.062
grab_cup_ep0             keypoint_lower  nonempty=16/16 mean_area_ratio_vs_robot_arm=0.322
```

结论：

- `keypoint_distal` 更像小的夹爪/末端候选，面积通常只有整臂 mask 的 6%-20%。
- `keypoint_lower` 更稳定，但可能包含更多手腕/下段机械臂。
- keyframe point-refine 的 temporal consistency 弱于完整 `robot arm` mask，建议作为夹爪候选，再配合人工检查或后处理。

## 产物路径

正式记录和指标：

```text
tmp/sam3_h2r_stability/report.md
tmp/sam3_h2r_stability/initial_prompt_stability.csv
tmp/sam3_h2r_stability/mid_prompt_stability.csv
tmp/sam3_h2r_stability/maxobj1_mid_stability.csv
tmp/sam3_h2r_stability/point_refine_mid_stability.csv
tmp/sam3_h2r_stability/keypoint_refine_mid_stability.csv
```

总览图：

```text
tmp/sam3_h2r_stability/all_prompt_runs_contact_sheet.jpg
tmp/sam3_h2r_stability/mid_prompt_sweep_contact_sheet.jpg
tmp/sam3_h2r_stability/maxobj1_mid_contact_sheet.jpg
tmp/sam3_h2r_stability/point_refine_mid_contact_sheet.jpg
tmp/sam3_h2r_stability/keypoint_refine_mid_contact_sheet.jpg
```

视频输出：

```text
tmp/sam3_h2r_robot_text/
tmp/sam3_h2r_robot_text_extra/
tmp/sam3_h2r_robot_text_prompt_sweep_mid/
tmp/sam3_h2r_robot_text_maxobj1_mid/
tmp/sam3_h2r_point_refine_mid/
tmp/sam3_h2r_keypoint_refine_mid/
tmp/sam3_h2r_keypoint_refine_lower_mid/
```

辅助脚本：

```text
tmp/sam3_h2r_batch_robot_text.py
tmp/analyze_sam3_h2r_stability.py
tmp/make_sam3_h2r_contact_sheet.py
tmp/sam3_h2r_point_refine.py
tmp/sam3_h2r_keyframe_point_refine.py
```

这些脚本和视频产物位于 `tmp/`，默认不进入 git；正式文档只记录可复现实验路径和结论。

## 后续建议

1. 整臂 mask 可直接作为 robot removal / inpaint 的 SAM3 候选。
2. 若需要夹爪 mask，优先使用 `robot arm` + keyframe point-refine 生成候选，再做人工检查或几何/运动后处理。
3. 长视频应分段处理。当前显存条件下，SAM3.1 16 帧稳定，32 帧在 GPU 2 上会 OOM。
4. 若要接入正式 pipeline，建议把 `robot arm` text prompt 作为第一阶段，把 keyframe point-refine 作为可选第二阶段，而不是直接依赖 `gripper` text prompt。
