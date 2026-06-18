# 运行环境

## Conda

- 主环境：`flip`
- Python：3.10
- CUDA：12.8
- SAM3/SAM3.1：使用单独 `sam3` conda 环境

## Cache

```text
HF_HOME=/disk_n/zzf/.cache/huggingface
PIP_CACHE_DIR=/disk_n/zzf/.pip_cache
```

## 常用前缀

```bash
LD_PRELOAD=/home/leadtek/miniconda3/envs/flip/lib/libjpeg.so.8 \
  no_proxy=localhost,127.0.0.1 \
  python -m src.pipeline.<module>
```

## 统一入口

GPU / 训练命令优先使用：

```bash
scripts/flip_run.sh <subcommand> --cuda <ids> -- <args>
```

常用：

```bash
scripts/flip_run.sh mitty_cache --cuda 0 -- <args>
scripts/flip_run.sh sam2_precompute --cuda 0 -- <args>
scripts/flip_run.sh h2r_sam3_precompute --cuda 2 -- <args>
scripts/flip_run.sh train --cuda 2,3 --nproc 2 -- <args>
scripts/flip_run.sh eval_mitty --cuda 2 -- <args>
```

