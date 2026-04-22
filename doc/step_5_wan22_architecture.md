# Wan 2.2 TI2V-5B DiT 架构

5B 参数，30 层 DiTBlock，bf16 ≈ 10GB。

源码：`DiffSynth-Studio/diffsynth/models/wan_video_dit.py`

## 数据流

```
input latent (b, 48, f, h, w)
  → patch_embedding: Conv3d(48→3072, k=(1,2,2), s=(1,2,2))
  → 30× DiTBlock(x, context, t_mod, freqs)
  → head: LayerNorm → Linear(3072→192) → unpatchify
  → output (b, 48, f, h, w)
```

## DiTBlock（×30）

每个 block 三步：self-attention → cross-attention → FFN，全部带 modulation（shift/scale/gate）。

```python
# 1. Self-Attention + 3D RoPE
x = x + gate_msa * self_attn(modulate(norm1(x), shift_msa, scale_msa), freqs)
# 2. Cross-Attention (text conditioning)
x = x + cross_attn(norm3(x), context)
# 3. FFN
x = x + gate_mlp * ffn(modulate(norm2(x), shift_mlp, scale_mlp))
```

### 每个 DiTBlock 内的可训练模块

| 模块 | 结构 | 参数量 |
|------|------|--------|
| `self_attn.q` | Linear(3072→3072) | 9.4M |
| `self_attn.k` | Linear(3072→3072) | 9.4M |
| `self_attn.v` | Linear(3072→3072) | 9.4M |
| `self_attn.o` | Linear(3072→3072) | 9.4M |
| `self_attn.norm_q` | RMSNorm(3072) | 3K |
| `self_attn.norm_k` | RMSNorm(3072) | 3K |
| `cross_attn.q` | Linear(3072→3072) | 9.4M |
| `cross_attn.k` | Linear(3072→3072) | 9.4M |
| `cross_attn.v` | Linear(3072→3072) | 9.4M |
| `cross_attn.o` | Linear(3072→3072) | 9.4M |
| `cross_attn.norm_q` | RMSNorm(3072) | 3K |
| `cross_attn.norm_k` | RMSNorm(3072) | 3K |
| `norm1` | LayerNorm(3072, no affine) | 0 |
| `norm2` | LayerNorm(3072, no affine) | 0 |
| `norm3` | LayerNorm(3072, affine) | 6K |
| `ffn.0` | Linear(3072→14336) | 44M |
| `ffn.2` | Linear(14336→3072) | 44M |
| `modulation` | Parameter(1, 6, 3072) | 18K |
| **block 合计** | | **~163M** |

30 blocks × 163M ≈ 4.9B，加上 embedding/head ≈ **5B total**。

## Block 外的模块

| 模块 | 结构 |
|------|------|
| `patch_embedding` | Conv3d(48→3072, k=(1,2,2)) |
| `text_embedding` | Linear(4096→3072) → GELU → Linear(3072→3072) |
| `time_embedding` | Linear(256→3072) → SiLU → Linear(3072→3072) |
| `time_projection` | SiLU → Linear(3072→18432) |
| `head.head` | Linear(3072→192) |
| `head.modulation` | Parameter(1, 2, 3072) |

## TI2V-5B 特有配置

- `seperated_timestep=True`：每个时空 patch 独立 timestep（Mitty 的 clean/noisy 分段关键）
- `fuse_vae_embedding_in_latents=True`：第一帧 VAE latent 直接拼进输入，不走额外 y 通道
- `has_image_input=False`：不用 CLIP image encoder
- 3D RoPE：temporal/height/width 三轴独立频率编码

## 梯度分析粒度

30 个 block，每个 block 内按功能分 3 组：

| 组 | 模块 | 每 block 参数 | 职责 |
|----|------|--------------|------|
| self_attn | q/k/v/o | 37.7M | 空间-时间关系建模 |
| cross_attn | q/k/v/o | 37.7M | text prompt 条件注入 |
| ffn | 0/2 | 88M (54%) | 特征变换，容量最大 |

对比恒等映射 vs 替换任务的权重距离，按 `blocks.{i}.{self_attn|cross_attn|ffn}` 共 90 个组分析。
