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

## 已知问题

- 输出帧率固定 24fps，输入 30fps 会导致时长略有变化
- 480p + 4:3 输出 736×544（或 752×560），非标准 640×480，需后处理 resize
- reference_video 不接受 base64 或 file_id，必须公网 URL
- catbox 临时链接 24h 过期；大规模使用建议搭建自己的文件服务
