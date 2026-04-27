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
OFFSETS=(13 38 49 54)
CUDAS=(0 0 1 1)
pids=()
for i in "${!OFFSETS[@]}"; do
  log="logs/segment_pipeline/inspire_ep23_retry4_w${i}.log"
  (
    export CUDA_VISIBLE_DEVICES="${CUDAS[$i]}"
    export LD_PRELOAD=/home/leadtek/miniconda3/envs/flip/lib/libjpeg.so.8
    export no_proxy=localhost,127.0.0.1
    export PYTHONFAULTHANDLER=1
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    exec /home/leadtek/miniconda3/envs/flip/bin/python -u -m src.pipeline.segment_pipeline "${COMMON_ARGS[@]}" --offset "${OFFSETS[$i]}" --limit 1 --device cuda:0
  ) > "$log" 2>&1 &
  pid=$!
  pids+=("$pid")
  echo "$pid" > "logs/segment_pipeline/inspire_ep23_retry4_w${i}.pid"
  echo "started retry4_w${i} cuda=${CUDAS[$i]} offset=${OFFSETS[$i]} pid=$pid log=$log"
done
status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=$?
done
echo "retry4 workers finished status=$status"
exit "$status"
