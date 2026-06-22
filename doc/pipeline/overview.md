# Pipeline 主线流程

当前 FLIP 主线围绕第一人称机器人视频生成训练数据展开：

```text
G1 / human2robot 原始数据
  -> 数据切片
  -> 数据合成 / mask 预计算
  -> pair layout
  -> VAE/T5 cache
  -> 三阶段 Wan2.2 + Mitty LoRA 训练
  -> Diffusion Policy / WAM 下游实验
```

## 数据类型

- `identity_r2r`：清晰 robot -> 同一清晰 robot。
- `blur_r2r`：局部模糊 robot -> 清晰 robot。
- `h2r`：human -> robot。
- `r2h`：robot -> human。

## 外观域边界

G1 WBT 的 Brainco 手和 Inspire 手不能混合训练；Inspire 手当前又按
`data/unitree_G1_WBT/Appearance.md` 拆成 `inspire_app1` 与 `inspire_app2` 两种外观。
外观恢复与跨域编辑训练应按 `brainco`、`inspire_app1`、`inspire_app2` 三条线分开并行
构造 pair/cache 和 LoRA run，不能把不同 robot appearance 混入同一个训练集或同一个
appearance-learning checkpoint。`inspire_dex5` 与 `inspire_flat` 只是源数据 layout
分组，不代表训练外观域。

## 当前维护入口

- `scripts/flip_run.sh sam2_precompute`
- `scripts/flip_run.sh h2r_sam3_precompute`
- `scripts/flip_run.sh g1_sam3_precompute`
- `scripts/flip_run.sh h2r_sam3_blur_pair`
- `scripts/flip_run.sh r2h_synthesize`
- `scripts/flip_run.sh mitty_cache`
- `scripts/flip_run.sh train`
- `scripts/flip_run.sh eval_mitty`
- `scripts/flip_run.sh h2r_diffusion_policy`

旧 inpaint、retarget、ComfyUI Wan、IDM、robot_patch 和 Masquerade baseline 已移到
`src/pipeline/archive/`，不再作为当前主线入口。

## 主要路径

- 原始 G1 数据：`data/unitree_G1_WBT/`
- 原始 human2robot 数据：`data/h2r/v1/`
- 中间与训练数据：`training_data/`
- 检查输出：`output/`
- smoke / 临时输出：`tmp/`
