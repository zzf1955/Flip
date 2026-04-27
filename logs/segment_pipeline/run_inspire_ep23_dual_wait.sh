#!/usr/bin/env bash
set -euo pipefail
cd /disk_n/zzf/flip
mkdir -p logs/segment_pipeline
COMMON_ARGS=(
  --manifest /disk_n/zzf/flip/training_data/segment/manifest.json
  --tasks Inspire_Collect_Clothes_MainCamOnly Inspire_Pickup_Pillow_MainCamOnly Inspire_Put_Clothes_Into_Basket Inspire_Put_Clothes_into_Washing_Machine Inspire_Put_Clothes_into_Washing_Machine_MainCamOnly
  --episodes 2 3
  --inpaint-method propainter
  --sam2-model small
  --prompt-interval 10
  --bbox-margin 0
  --final-root training_data/overlay/4s
  --resume
)
(
  export CUDA_VISIBLE_DEVICES=0
  export LD_PRELOAD=/home/leadtek/miniconda3/envs/flip/lib/libjpeg.so.8
  export no_proxy=localhost,127.0.0.1
  export PYTHONFAULTHANDLER=1
  exec /home/leadtek/miniconda3/envs/flip/bin/python -u -m src.pipeline.segment_pipeline "${COMMON_ARGS[@]}" --offset 0 --limit 28 --device cuda:0
) > logs/segment_pipeline/inspire_ep23_gpu0.log 2>&1 &
pid0=$!
(
  export CUDA_VISIBLE_DEVICES=1
  export LD_PRELOAD=/home/leadtek/miniconda3/envs/flip/lib/libjpeg.so.8
  export no_proxy=localhost,127.0.0.1
  export PYTHONFAULTHANDLER=1
  exec /home/leadtek/miniconda3/envs/flip/bin/python -u -m src.pipeline.segment_pipeline "${COMMON_ARGS[@]}" --offset 28 --limit 27 --device cuda:0
) > logs/segment_pipeline/inspire_ep23_gpu1.log 2>&1 &
pid1=$!
echo "$pid0" > logs/segment_pipeline/inspire_ep23_gpu0.pid
echo "$pid1" > logs/segment_pipeline/inspire_ep23_gpu1.pid
echo "started gpu0 pid=$pid0 gpu1 pid=$pid1"
status=0
wait "$pid0" || status=$?
wait "$pid1" || status=$?
echo "workers finished status=$status"
exit "$status"
