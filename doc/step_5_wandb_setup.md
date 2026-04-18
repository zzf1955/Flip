# W&B 配置（Wan 2.2 TI2V-5B 训练）

## 问题

wandb 0.26 默认的 Go core（`wandb-core` 二进制）在本机房通过 Clash 代理（端口 20171）初始化 run 时**稳定失败**：

```
wandb.errors.errors.CommError: Run initialization has timed out after 90.0 sec.
```

`wandb/run-*/logs/debug-internal.log` 显示 wandb-core 在 POST `upsertBucket` 到 `https://api.wandb.ai/graphql` 时被服务端以 `context canceled` 500 掉：

```
{"level":"INFO","msg":"api: retrying HTTP error",
 "status":500,"url":"https://api.wandb.ai/graphql",
 "body":"{\"errors\":[{\"message\":\"context canceled\",\"path\":[\"upsertBucket\"]}]}"}
```

## 诊断

1. **curl 通**：通过 Clash `HTTPS_PROXY=http://127.0.0.1:20171` 直接 POST `api.wandb.ai/graphql` 200，50 KB body 也 2 秒内通过。证明代理本身能转发大 POST。
2. **Python wandb SDK 能 login**（已读取 `~/.netrc`），只是 run init 挂。
3. **小写 `http_proxy`、大写 `HTTPS_PROXY` 都不解决**。
4. **唯一可用路径**：禁用 wandb-core（Go binary），走 **0.26 自带的 legacy Python service**：

   ```bash
   WANDB_X_DISABLE_SERVICE=true WANDB_CORE=disabled
   ```

   init 时间 11-12 秒。

## 根因推断

wandb 0.26 的 Go core 通过 HTTP 代理 tunnel 到 `api.wandb.ai` 时，某个环节（可能 HTTP/2、连接复用、或长 body 上传）与 Clash 不兼容，导致 wandb server 端在处理 `upsertBucket` 时上游读不到完整 body，主动 cancel 上下文返回 500。Clash → wandb 的链路具体哪一步失败尚未定位，但 workaround 足够稳定。

## 尝试过的无效方案

| 方案 | 结果 |
|------|------|
| 大写 `HTTPS_PROXY` + `HTTP_PROXY`（对 Go stdlib 要求） | 仍 timeout |
| 加长 `wandb.init(settings=wandb.Settings(init_timeout=300))` | 仍以 server 500 失败 |
| 降级 `wandb==0.18.7` | 0.18.7 也默认走 wandb-core，同样挂 |
| 降级 0.18.7 + `WANDB_X_DISABLE_SERVICE=true WANDB_CORE=disabled` | 挂（0.18 的 legacy 路径和 0.26 实现不同，不通） |
| `wandb==0.26.0` + 上述 env（legacy Python service） | ✓ **成功，11-12s init** |

## 推荐配置

### 必需的环境变量

```bash
# Clash 代理（其余 HF 下载等也需要）
export HTTP_PROXY=http://127.0.0.1:20171
export HTTPS_PROXY=http://127.0.0.1:20171
export http_proxy=http://127.0.0.1:20171
export https_proxy=http://127.0.0.1:20171
export NO_PROXY=localhost,127.0.0.1
export no_proxy=localhost,127.0.0.1

# 禁用 wandb Go core（关键 workaround）
export WANDB_X_DISABLE_SERVICE=true
export WANDB_CORE=disabled
```

### 安装版本

```bash
# 保持 0.26（官方最新，功能完整）；必须配合上面的 env
pip install wandb==0.26.0
```

### 训练启动示例

```bash
LD_PRELOAD=/home/leadtek/miniconda3/envs/flip/lib/libjpeg.so.8 \
  HTTP_PROXY=http://127.0.0.1:20171 HTTPS_PROXY=http://127.0.0.1:20171 \
  NO_PROXY=localhost,127.0.0.1 \
  WANDB_X_DISABLE_SERVICE=true WANDB_CORE=disabled \
  CUDA_VISIBLE_DEVICES=2,3 \
  torchrun --nproc_per_node=2 --master-port=29501 \
    -m src.pipeline.train_mitty \
    --cache-train output/mitty_cache_1s/train \
    --cache-eval  output/mitty_cache_1s/eval \
    --cache-ood   output/mitty_cache_1s/ood_eval \
    --batch-size 4 --epochs 1 --repeat 112 \
    --lora-rank 96 --warmup-steps 50 --lr 1e-4 --lr-min 1e-6 \
    --save-steps 200 --eval-steps 50 --eval-video-steps 200 \
    --eval-video-samples-ood -1 --eval-t-samples 5 \
    --wandb-project flip-mitty
```

## 训练代码约定

`src/core/train_utils.py:WandbLogger` 在 rank 0 调用 `wandb.init(project=...)`，其他 rank `project=None` 即禁用。`train_mitty.py` 和 `train_lora.py` 都经过这条路径。

## 已知副作用

- 安装 wandb 0.26 **会把 `protobuf` 从 7.x 降到 5.29.6**。目前对 torch / diffsynth 的工作流无观察到影响，若后续有 protobuf 版本冲突需单独处理。
- `WANDB_X_DISABLE_SERVICE=true` 是 wandb 内部的 underscore-prefixed flag，未来版本可能改名或移除。升级 wandb 时需重新验证能否 init。

## 磁盘注意事项

pip 安装 wandb 需要 ~16 MB 临时空间。根分区（`/`）仅 ~14 GB，长期会被 `~/.cache/pip` 占满。**安装前设置**：

```bash
export PIP_CACHE_DIR=/disk_n/zzf/.pip_cache
export TMPDIR=/disk_n/zzf/tmp_pip
```

这和 `CLAUDE.md` 的缓存规范一致。

## 后续可探索

- 定位 Clash 哪个规则/协议处理导致 wandb-core 失败
- 如需启用 wandb-core 的 artifact / 新 Panel 功能，需要用不走 Clash 的网络（如 IPv6 直连 api.wandb.ai，或换代理）
- 若团队有 wandb self-hosted / mirror，切换到那边可绕开整个代理链路
