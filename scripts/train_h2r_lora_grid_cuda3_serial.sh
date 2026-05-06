#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

FLIP_RUNNER="${FLIP_RUNNER:-$PROJECT_ROOT/scripts/flip_run.sh}"
if [[ ! -x "$FLIP_RUNNER" ]]; then
  echo "Missing executable runner: $FLIP_RUNNER" >&2
  exit 2
fi

DEFAULT_MERGE_LORA="training_data/log/Mitty-identity_r2r_1s-10000d_r64_ffn0ffn2_1000s_0429_185108/ckpt/step-0900.safetensors"
MERGE_LORAS="${MERGE_LORAS:-${MERGE_LORA:-$DEFAULT_MERGE_LORA}}"
MERGE_LORAS="${MERGE_LORAS//,/ }"
read -r -a MERGE_LORA_ARR <<< "$MERGE_LORAS"
if [[ "${#MERGE_LORA_ARR[@]}" -eq 0 ]]; then
  echo "MERGE_LORAS is empty" >&2
  exit 2
fi
for lora_path in "${MERGE_LORA_ARR[@]}"; do
  if [[ ! -f "$lora_path" ]]; then
    echo "Missing merge LoRA checkpoint: $lora_path" >&2
    exit 2
  fi
done

CUDA_DEVICE="${CUDA_DEVICE:-3}"
TASK_NAME="${TASK_NAME:-h2r_1s}"
BATCH_SIZE="${BATCH_SIZE:-4}"
TRAIN_SIZE="${TRAIN_SIZE:-490}"
IN_TASK_EVAL_SIZE="${IN_TASK_EVAL_SIZE:-16}"
OOD_EVAL_SIZE="${OOD_EVAL_SIZE:-16}"
IN_TASK_VIDEO_SIZE="${IN_TASK_VIDEO_SIZE:-8}"
OOD_VIDEO_SIZE="${OOD_VIDEO_SIZE:-8}"
MAX_STEPS="${MAX_STEPS:-1000}"
SAVE_STEPS="${SAVE_STEPS:-100}"
EVAL_STEPS="${EVAL_STEPS:-100}"
EVAL_VIDEO_STEPS="${EVAL_VIDEO_STEPS:-100}"

LR="${LR:-1e-4}"
LR_MIN="${LR_MIN:-1e-6}"
WARMUP_STEPS="${WARMUP_STEPS:-50}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
WANDB_PROJECT="${WANDB_PROJECT:-Flip}"
RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%m%d_%H%M%S)}"

LORA_LAYOUTS=(ffn qk vo qkvo ffn_qkvo)
LORA_RANKS=(64 128 256)

layout_args() {
  case "$1" in
    ffn)
      printf '%s\0%s\0' --lora-target-modules ffn.0,ffn.2
      ;;
    qk)
      printf '%s\0%s\0%s\0%s\0' --lora-attn-types self --lora-attn-projections q,k
      ;;
    vo)
      printf '%s\0%s\0%s\0%s\0' --lora-attn-types self --lora-attn-projections v,o
      ;;
    qkvo)
      printf '%s\0%s\0%s\0%s\0' --lora-attn-types self --lora-attn-projections q,k,v,o
      ;;
    ffn_qkvo)
      printf '%s\0%s\0' --lora-target-modules self_attn.q,self_attn.k,self_attn.v,self_attn.o,ffn.0,ffn.2
      ;;
    *)
      echo "Unknown LoRA layout: $1" >&2
      exit 2
      ;;
  esac
}

run_one() {
  local layout="$1"
  local rank="$2"
  local layout_tag="${layout//_/+}"
  local run_name="h2r_${layout}_r${rank}_${RUN_TIMESTAMP}"
  local layout_cli=()
  while IFS= read -r -d '' item; do
    layout_cli+=("$item")
  done < <(layout_args "$layout")
  local merge_cli=()
  for lora_path in "${MERGE_LORA_ARR[@]}"; do
    merge_cli+=(--merge-lora "$lora_path")
  done

  local cmd=(
    "$FLIP_RUNNER" train --cuda "$CUDA_DEVICE" --nproc 1 --
    --task-name "$TASK_NAME"
    "${merge_cli[@]}"
    --lora-rank "$rank"
    "${layout_cli[@]}"
    --batch-size "$BATCH_SIZE"
    --train-size "$TRAIN_SIZE"
    --in-task-eval-size "$IN_TASK_EVAL_SIZE"
    --ood-eval-size "$OOD_EVAL_SIZE"
    --in-task-video-size "$IN_TASK_VIDEO_SIZE"
    --ood-video-size "$OOD_VIDEO_SIZE"
    --max-steps "$MAX_STEPS"
    --save-steps "$SAVE_STEPS"
    --eval-steps "$EVAL_STEPS"
    --eval-video-steps "$EVAL_VIDEO_STEPS"
    --lr "$LR"
    --lr-min "$LR_MIN"
    --warmup-steps "$WARMUP_STEPS"
    --weight-decay "$WEIGHT_DECAY"
    --wandb-project "$WANDB_PROJECT"
    --wandb-run-name "$run_name"
    --wandb-tags h2r appearance "layout:${layout_tag}" "rank:${rank}" "run:${RUN_TIMESTAMP}" merge_lora_stack grid_search cuda3_serial
  )

  printf '\n[%(%F %T)T] CUDA %s start layout=%s rank=%s\n' -1 "$CUDA_DEVICE" "$layout" "$rank"
  printf 'Merge LoRAs:'
  printf ' %q' "${MERGE_LORA_ARR[@]}"
  printf '\n'
  printf 'Command:'
  printf ' %q' "${cmd[@]}"
  printf '\n'

  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    return 0
  fi

  "${cmd[@]}"
}

for layout in "${LORA_LAYOUTS[@]}"; do
  for rank in "${LORA_RANKS[@]}"; do
    run_one "$layout" "$rank"
  done
done
