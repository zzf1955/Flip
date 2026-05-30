# Wan 2.2 TI2V-5B — Mitty 方案的落地骨干

## 为什么选它

Mitty（arXiv 2512.Mitty）的核心是**视频 In-Context Learning**：把 human 演示视频的 VAE latent（clean）和 robot 视频 latent（noisy）沿 temporal 维拼接，过一次 full self-attention，让两段跨模态互相看见；loss 只在 robot 段。

`Wan 2.2 TI2V-5B` 的架构**天然就是 Mitty 结构的特例**（前缀长度 = 1 帧），只需把"第 1 帧 clean"推广到"前 f_H 帧 clean"即可。其他选项对比：

| 骨干 | `seperated_timestep` | `fuse_vae_embedding` | DiffSynth 训练支持 | Mitty 原文 |
|------|----|----|----|----|
| Wan 2.1 T2V-1.3B / 14B | ✗ | ✗ | ✓ | ✗ |
| Wan 2.1 I2V-14B | ✗ (走 y 通道) | ✗ | ✓ | ✗ |
| Wan 2.1 Fun-V1.1-14B-Control | ✗ (走 ref_conv) | ✗ | ✓ | ✗ |
| Wan 2.2 T2V-14B / I2V-14B (MoE) | ✗ | ✗ | ✓ 但双分支 LoRA 复杂 | ✓ 对照实验 |
| **Wan 2.2 TI2V-5B (dense)** | **✓** | **✓** | **✓** | **✓ 主实验** |

## 架构配置

`diffsynth/configs/model_configs.py:290-294`:

```python
{
    "model_class": "diffsynth.models.wan_video_dit.WanModel",
    "extra_kwargs": {
        'has_image_input': False,          # 不走 CLIP
        'patch_size': [1, 2, 2],           # temporal 1, spatial 2×2 patch
        'in_dim': 48, 'out_dim': 48,       # VAE latent 48 channels
        'dim': 3072, 'ffn_dim': 14336,
        'num_heads': 24, 'num_layers': 30,
        'freq_dim': 256, 'text_dim': 4096,
        'eps': 1e-06,
        'seperated_timestep': True,          # 每时空 patch 独立 timestep
        'require_clip_embedding': False,
        'require_vae_embedding': False,      # 不用 y 通道
        'fuse_vae_embedding_in_latents': True,  # latents[:,:,0:1] 替换成 input_image 的 VAE latent
    }
}
```

参数量：5B（dense，非 MoE），bf16 ≈ 10 GB 权重，safetensors (fp32) ≈ 20 GB。

## Wan 2.2 VAE (`WanVideoVAE38`)

`diffsynth/models/wan_video_vae.py:1285-1297`:

- `z_dim=48`（latent 48 channels，和 DiT `in_dim=48` 对齐）
- `dim=160`, `dec_dim=256`
- `temperal_downsample=[False, True, True]` → temporal 压缩 2×2=**4x**
- spatial 压缩 **8x**（dim_mult `[1,2,4,4]`）

对 480×640 / 17 帧输入：
- VAE latent shape: `(1, 48, 5, 60, 80)` — temporal 5 = (17-1)/4+1
- DiT patchify (1,2,2): `5 × 30 × 40 = 6000` tokens

Mitty 拼接（17+17 帧）：
- latent shape: `(1, 48, 10, 60, 80)`（沿 temporal concat）
- DiT tokens: `12000`

## Partial-Noising 机制 —— Mitty 的核心复用点

### model_fn 里的 per-patch timestep 构造

`diffsynth/pipelines/wan_video.py:1376-1380`：

```python
if dit.seperated_timestep and fuse_vae_embedding_in_latents:
    timestep = torch.concat([
        torch.zeros((1, latents.shape[3] * latents.shape[4] // 4)),      # 第 1 帧 t=0 (clean)
        torch.ones((latents.shape[2] - 1, ...)) * timestep               # 其余帧 t=sampled
    ]).flatten()
```

### 训练 loss 里的 first-frame 替换 + 排除

`diffsynth/diffusion/loss.py:FlowMatchSFTLoss`：

```python
# 加噪
inputs["latents"] = pipe.scheduler.add_noise(inputs["input_latents"], noise, timestep)
training_target = pipe.scheduler.training_target(...)

# 若有 first_frame_latents，把 latents[:,:,0:1] 覆盖回 clean
if "first_frame_latents" in inputs:
    inputs["latents"][:, :, 0:1] = inputs["first_frame_latents"]

# forward
noise_pred = pipe.model_fn(**models, **inputs, timestep=timestep)

# loss 排除第 1 帧（那帧是 clean，不需要预测）
if "first_frame_latents" in inputs:
    noise_pred = noise_pred[:, :, 1:]
    training_target = training_target[:, :, 1:]

loss = mse(noise_pred, training_target)
```

### Mitty 改造：把 "1" 推广成 f_H

**改动只有 2 个文件，总共 ~10 行代码**：

**1. `model_fn_wan_video` 的 timestep 构造**（1376-1380）：
```python
f_H = first_frame_latents.shape[2]  # human 段长度
timestep = torch.concat([
    torch.zeros((f_H, latents.shape[3] * latents.shape[4] // 4)),
    torch.ones((latents.shape[2] - f_H, ...)) * timestep
]).flatten()
```

**2. `FlowMatchSFTLoss`**（loss.py:16-24）：
```python
if "first_frame_latents" in inputs:
    f_H = inputs["first_frame_latents"].shape[2]
    inputs["latents"][:, :, 0:f_H] = inputs["first_frame_latents"]
...
if "first_frame_latents" in inputs:
    f_H = inputs["first_frame_latents"].shape[2]
    noise_pred = noise_pred[:, :, f_H:]
    training_target = training_target[:, :, f_H:]
```

其余全部保持 DiffSynth 原样：LoRA 注入、VAE encode/decode、T5 embed、checkpoint、FP8、DDP 路径全能复用。

## 显存预算（4090 24GB, bf16）

### T004 实测数据（480×640, 17 帧, 20 steps, 4090 24GB）

脚本：`scripts/smoke_wan22_ti2v5b.py`, CUDA_VISIBLE_DEVICES=2, GPU 2 (24.5 GB)

#### 方案 A：全模型 disk offload（`--low-vram` 旧配置）

| 阶段 | 值 |
|------|-----|
| 模型加载（meta 读取） | 80-120 s |
| T2V inference, cold | 155 s |
| T2V inference, warm | 73 s |
| TI2V inference, warm | 23 s |
| **Peak 显存** | **10.83 GB** |
| 稳态 allocated | ~0 GB（所有模型不用时都在 disk） |

DiT forward 本身 ~7s (20 步, 3.65 it/s)；剩余时间都在 disk IO 和 onload。

#### 方案 B：T5 disk offload + DiT 常驻 GPU + VAE CPU offload（当前默认，对齐训练）

| 阶段 | 值 |
|------|-----|
| 模型加载（DiT 一次性 load 到 GPU） | **105 s** |
| T2V inference | **12.3 s** |
| TI2V inference | **11.8 s** |
| 稳态 allocated (DiT on GPU) | **9.31 GB** |
| **Peak 显存**（T5 encode 时同时 onload） | **19.88 GB** |

DiT 常驻意味着不再反复 load，推理速度**比方案 A 快 6-8 倍**。

#### 对比和取舍

| 维度 | 方案 A (disk) | 方案 B (T5 offload) |
|-----|--------------|--------------------|
| Peak VRAM | 10.83 GB | 19.88 GB |
| T2V 速度 | 73 s | **12.3 s** |
| 适用场景 | 极小显存 / 多模型共卡 | **训练 + 正常推理** |

**T005 训练采用方案 B 的变体**：T5 不加载（训练前已 cache embedding），DiT 常驻，VAE CPU offload，预计训练峰值 ~15-17 GB（见下文）。

### 必需的 vram 配置

4090 **不能用 DiffSynth 默认加载方式**（全部放 GPU 会 OOM，第一次测试时 T5+DiT 加载到 23GB 后 VAE 装不下）。必须用 low_vram 模式：

```python
vram_kwargs = {
    "offload_dtype": "disk",       # 不用时存磁盘（不占 CPU/GPU）
    "offload_device": "disk",
    "onload_dtype": torch.bfloat16,
    "onload_device": "cpu",        # 先读到 CPU
    "preparing_dtype": torch.bfloat16,
    "preparing_device": "cuda",    # 然后 prepare 到 GPU
    "computation_dtype": torch.bfloat16,
    "computation_device": "cuda",  # 在 GPU 上 compute
}
```

### 训练峰值估算（基于实测）

| 项 | 估算 |
|----|------|
| DiT 权重（bf16, on-GPU） | ~10 GB |
| LoRA rank 96 + AdamW state | ~0.6 GB |
| 激活值（grad ckpt, ~12k tokens, 30 layers） | ~3-5 GB |
| VAE（eval 时加载, offload 也可） | ~0.5 GB |
| 其他（碎片、临时 buffer） | ~1 GB |
| **训练峰值（DiT GPU 常驻 + T5/VAE offload）** | **~15-17 GB** |

训练时的策略：T5 用于 data_process 阶段（缓存 context embedding），train 阶段不加载 T5；VAE 也可在训练主循环中 offload；DiT + LoRA 常驻 GPU。

## 与 Wan 2.1 FunControl 方案的对比

| 维度 | Wan 2.1 Fun-V1.1-14B-Control（T003） | Wan 2.2 TI2V-5B（本方案） |
|------|-------------------------------------|-------------------------|
| 条件注入 | control_video → ref_conv → y channel | human video latent 沿 temporal 拼接到 robot 前 |
| 要求 | human 和 robot 空间上 pixel-aligned | 只要求 temporal 对齐，空间可差异 |
| 架构改动 | 无（DiffSynth 原生支持） | 2 处共 10 行代码 |
| 参数量 | 16.4B（14B + FunControl extra）| 5B |
| 精度需求 | 必须 FP8（bf16 放不下） | bf16 够用 |
| 显存余量（4090） | < 1 GB | 6-9 GB |
| LoRA rank | 16 | 96（对齐 Mitty） |
| LoRA target | q,k,v,o + ffn | q,k,v,o（对齐 Mitty, 不训 ffn） |
| 帧数上限（单视频） | 17 (1s @ 16fps) | 预估 17+17 或更多 |
| DiffSynth 训练支持 | ✓ | ✓ |

## 下一步

- T005：实现 `mitty_cache.py` + `mitty_model_fn.py` + `train_mitty.py`
- T006：推理脚本 + 与 FunControl baseline 定量对比
