这个项目之前是在 windows 上网基于 comfyui 做的，现在迁移到 Linux 上了。

1. 重新下载 leverb 数据集
2. 重新系在 flux fill、t5、vae、controlnet 等模型
3. clone cosmos transfer 1 和 2.5，作为引用
4. 跑通 flux 数据合成的 pipeline，合成完你自己看一眼图片来 check

---

5. 选做：多样性，一致性实验：通过修改 workflow 中的生成 prompt，来保持单条视频多个关键帧的人物形态是一致的
6. cosmos transfer 复现：编辑过的首帧 + 原视频的 depth 条件，生成视频。测试仅首帧/深度控制的效果

