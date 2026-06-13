# Seedance 2.0 视频生成 API 调用指南

## 概述

使用火山方舟 Seedance 2.0 API 将机器人第一人称视频转换为真人视频（robot → human）。
属于 FLIP pipeline 的 Step 4 替代方案：直接用商业 video-to-video 模型替换机器人外观。

## API 端点

| 操作 | 方法 | URL |
|------|------|-----|
| 上传文件 | POST | `https://ark.cn-beijing.volces.com/api/v3/files` |
| 创建任务 | POST | `https://ark.cn-beijing.volces.com/api/v3/contents/generations/tasks` |
| 查询任务 | GET  | `https://ark.cn-beijing.volces.com/api/v3/contents/generations/tasks/{task_id}` |

鉴权：`Authorization: Bearer <ARK_API_KEY>`

## 输入限制

- 视频像素数 ≥ 409,600（即至少 640×640）
- 视频时长 2–15 秒
- 单请求最多 3 个 reference_video
- 文件大小 ≤ 50MB
- **reference_video 必须是公网 URL**，不支持 base64 / file_id

## 输出分辨率对照表（固定，不可自定义像素）

| 宽高比 | 480p | 720p | 1080p |
|--------|------|------|-------|
| 4:3    | 736×544 | 1120×832 | 1664×1248 |
| 16:9   | 864×480 | 1248×704 | 1920×1088 |
| 1:1    | 640×640 | 960×960  | 1440×1440 |
| 21:9   | 960×416 | 1504×640 | 2176×928  |
| 9:16   | 480×864 | 704×1248 | 1088×1920 |
| 3:4    | 544×736 | 832×1120 | 1248×1664 |

输出帧率固定 24fps，不可配置。

## 模型选择

| model ID | 特点 |
|----------|------|
| `doubao-seedance-2-0-260128` | 标准版，质量更高 |
| `doubao-seedance-2-0-fast-260128` | 快速版，生成更快 |

## 完整流程

### Step 1: 上传视频获取公网 URL

由于 API 要求公网 URL，而方舟 Files API 返回的 file_id 无法直接用于 reference_video，
需要通过第三方临时文件托管服务上传：

```bash
# 需要代理访问外网
export https_proxy=http://127.0.0.1:20171
curl -s -F "reqtype=fileupload" -F "time=24h" \
  -F "fileToUpload=@input.mp4" \
  https://litterbox.catbox.moe/resources/internals/api.php
# 返回: https://litter.catbox.moe/xxxxx.mp4
```

> **注意**：输入视频像素数不足 409,600 时需先放大（如 640×480 → 800×600）。

### Step 2: 创建视频生成任务

```bash
curl -s -X POST https://ark.cn-beijing.volces.com/api/v3/contents/generations/tasks \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $ARK_API_KEY" \
  -d '{
    "model": "doubao-seedance-2-0-fast-260128",
    "content": [
      {"type": "text", "text": "你的 prompt"},
      {"type": "video_url", "video_url": {"url": "https://公网URL"}, "role": "reference_video"}
    ],
    "resolution": "480p",
    "ratio": "4:3",
    "duration": 4,
    "watermark": false
  }'
# 返回: {"id": "cgt-xxxxxxxxx-xxxxx"}
```

### Step 3: 轮询等待结果

```bash
curl -s https://ark.cn-beijing.volces.com/api/v3/contents/generations/tasks/{task_id} \
  -H "Authorization: Bearer $ARK_API_KEY"
```

状态流转：`queued → running → succeeded / failed`

成功后 `content.video_url` 为 TOS 签名下载链接（**24 小时过期**）。

### Step 4: 下载 + 后处理

```bash
# 下载
curl -s -o output.mp4 "<video_url>"

# 缩放到目标分辨率（如 640×480）
ffmpeg -i output.mp4 -vf "scale=640:480" -c:v libx264 -crf 18 final.mp4
```

## 自动化脚本

```bash
python -m src.pipeline.seedance_gen \
  --input training_data/long/pair_0000/robot.mp4 \
  --output training_data/long/pair_0000/human.mp4 \
  --prompt "将视频中的机器人完全替换为真人..." \
  --resolution 480p --ratio 4:3 --duration 4 \
  --target-size 640x480
```

详见 `src/pipeline/seedance_gen.py`。

### H2R 机械臂 → 人手 smoke

`src.pipeline.h2r_seedance_edit` 是 H2R HDF5 专用 smoke 入口，不复用
`seedance_batch.py` 的 4:3 / 640×480 G1 假设。该入口默认从
`data/h2r/v1/data/<task>/episode_*.hdf5` 读取 `cam_data/robot_camera`，
导出 3 段 16:9 参考视频，并用三路并发调用 Seedance：

**当前结论：Seedance 直接 H2R robot → human hand 效果不好，暂不进入正式数据生成或
Wan 训练。** 两轮 prompt smoke 都能跑通 API 和尺寸后处理，但视觉结果不稳定：
第一版 prompt 容易在机械夹爪旁新增一只人手而不是替换夹爪；第二版加入“原位置覆盖替换”
约束后仍未达到可用质量。因此该入口目前只作为 API/尺寸/prompt 调试工具保留，不作为
H2R 配对数据生产路径。

```bash
python -m src.pipeline.h2r_seedance_edit \
  --output-root tmp/h2r_seedance_edit_smoke \
  --workers 3
```

默认样本为：

- `grab_both_cubes_v1:0:0`
- `grab_cup_v1:0:0`
- `roll:0:0`

默认几何：

- HDF5 source：`426×240`（WxH）。
- Seedance reference：`864×480`（WxH），CLI 参数写作 `--api-size 480x864`
  （HxW），`120` 帧、`30fps`、`ratio=16:9`、`resolution=480p`、`duration=4`。
- 本地 review 输出：`488×256`（WxH），CLI 参数写作 `--final-size 256x488`
  （HxW），重采样为 `120` 帧、`30fps`。Seedance API raw 输出帧率由服务端决定，
  另存为 `seedance_raw/` 下的原始 mp4。
- 2026-06-13 本轮 3 段 H2R smoke 观测到 Seedance raw 输出均为
  `864×496`（WxH）、`24fps`、约 `4.04s`；`final/` 下的 review 视频已后处理为
  `488×256`（WxH）、`30fps`、`4.0s`。

默认 prompt 采用原位置替换约束，避免模型在夹爪旁额外生成一只手：

```text
视频编辑任务：把画面中原有的机器人机械臂、黑色两指夹爪和白色机械外壳完全擦除，
并在完全相同的位置替换成一只真实裸露的人类手和前臂。手掌和手指必须覆盖原夹爪位置，
前臂从原机械臂进入画面的同一边缘伸入，沿用原机械臂的运动轨迹、朝向、抓取点、接触关系和遮挡关系。
保持桌面、背景、方块、杯子、托盘、杆、相机视角、光照、阴影和所有物体位置完全不变。
禁止在夹爪旁边、桌面空白处或画面边缘额外生成手；不要出现第二只手、袖子、手套、
完整人物、头、脸、身体、机器人夹爪、机械零件、金属或塑料残留。
```

该 prompt 不再沿用 G1 full-body prompt，也不再额外要求生成完整人体、头部、脸或衣服。
实际 smoke 结论仍是负结果：prompt 约束可以描述目标，但 Seedance 直接编辑第一人称
小尺寸机械夹爪时没有稳定完成“删除夹爪并在原位置替换成人手”的操作。
如只想准备 3 段输入视频和运行计划，不调用 API：

```bash
python -m src.pipeline.h2r_seedance_edit \
  --output-root tmp/h2r_seedance_edit_smoke \
  --workers 3 \
  --dry-run
```

### H2R SAM3 marker-mask 引导 smoke 与评估

`src.pipeline.h2r_seedance_sam3_edit` 是直接 H2R robot2human smoke 的显式 mask
变体：先用 `src.pipeline.h2r_sam3_precompute` 对 H2R robot-camera 视频做
SAM3/SAM3.1 `robot arm` 分割，再把 Seedance reference video 中的目标区域用可配置
marker 标出来，让 Seedance 只编辑 marker 指示的位置。该入口仍只服务 Seedance 调试，
不进入 Wan 训练或正式配对数据生成。

默认样本仍为 `grab_both_cubes_v1:0:0`、`grab_cup_v1:0:0`、`roll:0:0`，从 HDF5
`cam_data/robot_camera` 读取连续 `120` 帧、`30fps`。参考视频保持 `864x480`，
本地 review 输出保持 `488x256`、`120` 帧、`30fps`。Seedance raw 输出由服务端决定，
本轮仍观测到 `864x496`、`24fps`、约 `4.04s`，因此后处理输出不能省略。

SAM3 mask 预计算命令：

```bash
scripts/flip_run.sh h2r_sam3_precompute --cuda 1 -- \
  --tasks grab_both_cubes_v1,grab_cup_v1,roll \
  --output-root tmp/h2r_seedance_sam3_red_edit/sam3_mask \
  --tmp-dir tmp/h2r_seedance_sam3_red_edit/sam3_frames \
  --max-episodes-per-task 1 \
  --max-clips-per-episode 4 \
  --prompt "robot arm" \
  --backup-prompt "robotic arm" \
  --resume
```

`h2r_sam3_precompute` 新增 `--prompt-frame-position first|middle|<int>`。默认仍为
`first`；`middle` 可用于检查 prompt 在机械臂可见位置更稳定时的效果。末帧 prompt
不作为默认路径，因为 SAM3.1 从最后一帧做反向传播时会出现缓存缺失异常。

`h2r_seedance_sam3_edit` 支持三类目标 mask：

- `--mask-filter full`：直接使用 SAM3 `robot arm` 全 mask，适合全臂红色 baseline。
- `--mask-filter dark`：取 `SAM3 robot arm mask ∩ RGB max-channel <= dark_threshold`，
  用于把目标缩小到黑色夹爪。SAM3.1 直接用 `robot gripper` / `black gripper`
  text prompt 的两轮 smoke 均得到空 mask，因此当前更可靠的是先分出机械臂，再用暗像素过滤夹爪。
- `--mask-filter distal_dark`：实验性地从 `dark` mask 中选择远离整臂质心的小暗色连通域，
  可通过 `--distal-max-area`、`--distal-max-aspect`、`--distal-temporal-weight` 等参数控制。
  该方法能缩小局部区域，但当前仍会在个别帧跳到桌面黑色长条或线缆，暂不作为 Seedance
  推荐输入。

marker 可通过 `--marker-color {red,magenta,cyan,yellow,green,skin}` 和
`--annotation-mode {fill,outline,bbox,fill_bbox,dual_bbox}` 组合控制；实验性 `skin`
marker 只用于肤色预填负例，不推荐继续使用。`fill_bbox` 用一种颜色填充目标、另一种颜色画框，
`dual_bbox` 只画双层框，例如紫色内框 + 黄色外框；这两个模式用于 prompt/marker smoke，
当前不是推荐默认。默认 prompt 模板会使用
`{marker_desc}` 和 `{task_name}`：

```text
把视频中{marker_desc}标出的移动装置替换成裸露的人类胳膊。保持背景不变，人手和该装置的动作轨迹保持一致。人类手臂{task_name}
```

三段默认样本的中文动作名为：

- `grab_both_cubes_v1`：`抓起物块。`
- `grab_cup_v1`：`抓起杯子。`
- `roll`：`滚动物体。`

当前最佳候选 smoke 使用 dark gripper target + 黄色方框：

```bash
ARK_REQUEST_TIMEOUT=120 python -m src.pipeline.h2r_seedance_sam3_edit \
  --output-root tmp/h2r_seedance_sam3_exp04_dark_yellow_bbox \
  --mask-root tmp/h2r_seedance_sam3_red_edit/sam3_mask \
  --mask-filter dark \
  --marker-color yellow \
  --annotation-mode bbox \
  --prompt-template "把视频中{marker_desc}标出的黑色机械夹爪替换成一只裸露的人类手，手掌和手指出现在原夹爪位置，不要在旁边生成额外手。保持背景和物体不变。人手{task_name}" \
  --workers 3
```

每个 sample 的中间产物：

- `input/*_original_ref_864x480.mp4`：原始 16:9 Seedance reference，`120` 帧、
  `30fps`。
- `input/*_sam3_<filter>_<color>_<mode>_ref_864x480.mp4`：实际上传给 Seedance
  的 marker reference。
- `mask/*_sam3_<filter>_target_mask_864x480.mp4`：用于评估和 review 的二值目标 mask。
- `seedance_raw/*_human_hand_raw.mp4`：Seedance API 原始输出。
- `final/*_human_hand_256x488_hxw.mp4`：本地后处理后的 `488x256`、`120` 帧、
  `30fps` review 输出。
- `_review/*_original_red_final_compare.mp4`：三列并排 review 视频。文件名沿用
  `red` 历史命名，但内容跟随当前 marker 配置。

SAM3.1 评估入口为 `src.pipeline.h2r_seedance_sam3_eval`，通过 SAM3.1
`human hand,bare human hand,human fingers` prompt 对 Seedance final 输出做人手分割，
并与 `target_mask_video_path` 的目标区域做 IoU / coverage 统计：

```bash
scripts/flip_run.sh h2r_seedance_sam3_eval --cuda 1 -- \
  --result-jsonl tmp/h2r_seedance_sam3_exp04_dark_yellow_bbox/seedance_results.jsonl \
  --output-root tmp/h2r_seedance_sam3_eval_exp04_dark_yellow_bbox \
  --frame-stride 5 \
  --chunk-size 17 \
  --write-overlay
```

评估输出包括每条样本的 `sam3_hand_eval.npz`、`sam3_hand_eval.json`、
`sam3_hand_green_target_red_contour.mp4`，以及全局 `summary.json` / `summary.csv`。
overlay 约定为绿色人手 mask、红色目标轮廓。

2026-06-13 的 API 实验控制在约 20 次以内：17 次成功生成，另有 3 次创建任务请求在
30s timeout 后失败且没有返回 task id。为避免误判长请求失败，`seedance_gen.py`
新增 `ARK_REQUEST_TIMEOUT`，默认 `120s`。

量化结果汇总在 `tmp/h2r_seedance_sam3_experiment_summary.csv`。下表为三条样本平均值；
baseline 使用全臂红色目标，dark 实验使用夹爪目标，目标区域不同，不能只按一列指标直接
等价比较。

| 实验 | target covered by hand | hand on target | mean IoU | mean hand area px |
|------|------------------------|----------------|----------|-------------------|
| `baseline_full_red_task_prompt` | `0.062` | `0.296` | `0.040` | `3444.1` |
| `exp01b_dark_magenta_fill` | `0.432` | `0.144` | `0.088` | `10916.4` |
| `exp02_dark_cyan_fill` | `0.129` | `0.074` | `0.051` | `7629.4` |
| `exp03_dark_yellow_outline` | `0.133` | `0.037` | `0.021` | `4571.2` |
| `exp04_dark_yellow_bbox` | `0.507` | `0.207` | `0.156` | `6893.5` |
| `exp05_dark_magenta_bbox` | `0.441` | `0.140` | `0.107` | `8905.3` |
| `exp06_dark_skin_fill_one` | `0.000` | `0.000` | `0.000` | `0.0` |
| `exp07_dark_yellow_bbox_strict_prompt` | `0.002` | `0.002` | `0.001` | `5582.7` |

2026-06-13 沿黄框/紫框方向追加两条 `grab_cup_v1` 单条验证，结果已写入
`tmp/h2r_seedance_sam3_experiment_summary_updated.csv`：

| 实验 | target covered by hand | hand on target | mean IoU | mean hand area px | 结论 |
|------|------------------------|----------------|----------|-------------------|------|
| `exp08_yellow_bbox_big_cup` | `0.006` | `0.004` | `0.002` | `5783.3` | 黄框加粗放大能生成清楚人手，但动作/位置偏离原夹爪目标，target overlap 很低。 |
| `exp09_dual_bbox_cup` | `0.000` | `0.000` | `0.000` | `0.0` | 紫框 + 黄框双层框没有生成可检出人手，视觉上基本保留机器人，属于负例。 |

按 `target_covered_by_hand_ratio` 看，`exp04_dark_yellow_bbox` 是当前最佳整体候选，
也是 `grab_both_cubes_v1` 和 `roll` 的最佳；`grab_cup_v1` 单条最好的是
`exp01b_dark_magenta_fill`。但 `hand_on_target_ratio` 仍偏低，说明仍可能存在额外人手、
人手区域过大或 SAM3.1 hand false positive。结论保持谨慎：dark gripper + marker 的确
优于全臂红色 baseline，但还不能作为训练数据生产路径，需要先人工核对以下 overlay：

- `tmp/h2r_seedance_sam3_eval_exp04_dark_yellow_bbox/*/sam3_hand_green_target_red_contour.mp4`
- `tmp/h2r_seedance_sam3_eval_exp01b_dark_magenta_fill/grab_cup_v1_ep000000_f000000/sam3_hand_green_target_red_contour.mp4`

2026-06-13 追加两条负例：

- `exp06_dark_skin_fill_one`：把暗色夹爪区域预填成肤色后只跑
  `grab_both_cubes_v1` 单条，SAM3.1 没有检出人手，说明肤色预填会被 Seedance 当作待擦除
  伪影而不是人手先验，暂不继续。
- `exp07_dark_yellow_bbox_strict_prompt`：保留黄框 bbox，但 prompt 过度强调“只在方框内”
  和“输出不要保留方框”。三条平均 `target_covered_by_hand_ratio` 退化到 `0.002`，
  明显差于 `exp04`。后续 prompt 应保持 `exp04` 这种较自然的黄/紫方框描述，不要过度收窄。
- `exp08_yellow_bbox_big_cup`：只跑 `grab_cup_v1`，加粗并放大黄框。输出里有更明显的人手，
  但人手未稳定贴住原夹爪轨迹，SAM3.1 target coverage 只有 `0.006`，不建议扩到三条。
- `exp09_dual_bbox_cup`：新增 `dual_bbox` 后用紫色内框 + 黄色外框只跑 `grab_cup_v1`。
  Seedance 基本没有替换机器人，SAM3.1 hand eval 为 0。当前不再沿“叠加双色框”继续烧 API。

## 已知问题

- 输出帧率固定 24fps，输入 30fps 会导致时长略有变化
- 480p + 4:3 输出 736×544（或 752×560），非标准 640×480，需后处理 resize
- reference_video 不接受 base64 或 file_id，必须公网 URL
- catbox 临时链接 24h 过期；大规模使用建议搭建自己的文件服务
