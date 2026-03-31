# 项目架构

## 概览

Flip 采用逆向生成数据合成方法，将 G1 机器人视频转换为人类视频，用于训练 video editing 模型。

## 目录结构

```
VideoEdit/
├── archive/          # 归档的参考实现
├── data/             # 数据集（leverb 等）
├── doc/              # 项目文档
│   ├── agents/       # Agent skill 定义
│   └── tasks/        # 任务管理
├── paper/            # 参考论文
└── (待开发的核心模块)
```

## 技术路线

1. **关键帧编辑**：从 G1 视频中提取关键帧，使用图像编辑模型将机器人替换为人类
2. **视频合成**：提取控制信号（深度、边缘、分割），使用 Cosmos Transfer 生成视频
3. **规模化**：端到端 pipeline，一对多增强

## 核心脚本（规划中）

- `extract_keyframes.py` — 关键帧提取
- `auto_mask.py` — 自动遮罩生成
- `comfyui_inpaint.py` — ComfyUI 图像修复
- `extract_control_signals.py` — 控制信号提取
- `cosmos_transfer_generate.py` — Cosmos Transfer 视频生成
- `pipeline_e2e.py` — 端到端 pipeline
