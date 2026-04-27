#!/usr/bin/env bash
set -euo pipefail
cd /disk_n/zzf/flip
export CUDA_VISIBLE_DEVICES=0
export LD_PRELOAD=/home/leadtek/miniconda3/envs/flip/lib/libjpeg.so.8
export no_proxy=localhost,127.0.0.1
export PYTHONFAULTHANDLER=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec /home/leadtek/miniconda3/envs/flip/bin/python -u -m src.pipeline.segment_pipeline \
  --manifest /disk_n/zzf/flip/training_data/segment/manifest.json \
  --tasks Inspire_Collect_Clothes_MainCamOnly Inspire_Pickup_Pillow_MainCamOnly Inspire_Put_Clothes_Into_Basket Inspire_Put_Clothes_into_Washing_Machine Inspire_Put_Clothes_into_Washing_Machine_MainCamOnly \
  --episodes 2 3 \
  --inpaint-method propainter \
  --sam2-model small \
  --prompt-interval 10 \
  --bbox-margin 0 \
  --final-root training_data/overlay/4s \
  --resume \
  --device cuda:0
