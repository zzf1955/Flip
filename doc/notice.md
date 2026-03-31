# 踩坑记录

> 遇到环境问题或反复出错时先查此文件；解决新问题后记录到此处。

<!-- 格式示例：
## [简短标题]
- 现象：...
- 原因：...
- 解决：...
-->

## ComfyUI 端口为 8001
- 现象：脚本连接 localhost:8000 报 BrokenPipe / Connection refused
- 原因：port 8000 上运行的是 recall.py，ComfyUI 实际在 8001
- 解决：所有 ComfyUI 脚本默认端口改为 8001

## conda 环境需用绝对路径
- 现象：`conda run -n videoedit` 报 permission denied
- 原因：shell 环境未正确初始化 conda
- 解决：直接用 `/opt/homebrew/Caskroom/miniconda/base/envs/videoedit/bin/python` 运行

## LEVERB 视频为 av1 编码
- 现象：1080x1920 竖屏视频，av1 codec
- 原因：LEVERB 数据集默认格式
- 解决：cv2 可正常读取，提取帧约 1-2.5MB/帧（PNG）
