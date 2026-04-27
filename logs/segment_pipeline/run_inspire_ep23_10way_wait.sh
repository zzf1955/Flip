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
# 55 segments split into 10 non-overlapping shards.
OFFSETS=(0 6 12 18 24 30 36 42 48 54)
LIMITS=(6 6 6 6 6 6 6 6 6 1)
CUDAS=(0 0 0 0 0 1 1 1 1 1)
pids=()
for i in "${!OFFSETS[@]}"; do
  log="logs/segment_pipeline/inspire_ep23_10w${i}.log"
  (
    export CUDA_VISIBLE_DEVICES="${CUDAS[$i]}"
    export LD_PRELOAD=/home/leadtek/miniconda3/envs/flip/lib/libjpeg.so.8
    export no_proxy=localhost,127.0.0.1
    export PYTHONFAULTHANDLER=1
    exec /home/leadtek/miniconda3/envs/flip/bin/python -u -m src.pipeline.segment_pipeline "${COMMON_ARGS[@]}" --offset "${OFFSETS[$i]}" --limit "${LIMITS[$i]}" --device cuda:0
  ) > "$log" 2>&1 &
  pid=$!
  pids+=("$pid")
  echo "$pid" > "logs/segment_pipeline/inspire_ep23_10w${i}.pid"
  echo "started w${i} cuda=${CUDAS[$i]} offset=${OFFSETS[$i]} limit=${LIMITS[$i]} pid=$pid log=$log"
done
status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=$?
done
echo "10way workers finished status=$status"
exit "$status"
