#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"

FLIP_RUNNER="${FLIP_RUNNER:-$PROJECT_ROOT/scripts/flip_run.sh}"
if [[ ! -x "$FLIP_RUNNER" ]]; then
  echo "Missing executable runner: $FLIP_RUNNER" >&2
  exit 2
fi

CUDA_DEVICES_CSV="${CUDA_DEVICES:-0,1}"
IFS=',' read -r -a CUDA_DEVICES_ARR <<< "$CUDA_DEVICES_CSV"
if [[ "${#CUDA_DEVICES_ARR[@]}" -eq 0 ]]; then
  echo "CUDA_DEVICES is empty" >&2
  exit 2
fi

TASK_NAME="${TASK_NAME:-h2r_1s}"
RUN_PREFIX="${RUN_PREFIX:-mitty_h2r}"
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

LORA_RANKS_SPEC="${LORA_RANKS:-16 32 64 96}"
LORA_RANKS_SPEC="${LORA_RANKS_SPEC//,/ }"
read -r -a LORA_RANKS_ARR <<< "$LORA_RANKS_SPEC"
if [[ "${#LORA_RANKS_ARR[@]}" -eq 0 ]]; then
  echo "LORA_RANKS is empty" >&2
  exit 2
fi

pids=()
pid_cuda=()
exit_code=0

reap_finished() {
  local idx
  for idx in "${!pids[@]}"; do
    if ! kill -0 "${pids[$idx]}" 2>/dev/null; then
      if ! wait "${pids[$idx]}"; then
        exit_code=1
      fi
      unset 'pids[idx]'
      unset 'pid_cuda[idx]'
      pids=("${pids[@]}")
      pid_cuda=("${pid_cuda[@]}")
      return 0
    fi
  done
  return 1
}

wait_for_slot() {
  while [[ "${#pids[@]}" -ge "${#CUDA_DEVICES_ARR[@]}" ]]; do
    reap_finished || sleep 30
  done
}

next_cuda() {
  local cuda
  for cuda in "${CUDA_DEVICES_ARR[@]}"; do
    local used=0
    local busy_cuda
    for busy_cuda in "${pid_cuda[@]:-}"; do
      if [[ "$busy_cuda" == "$cuda" ]]; then
        used=1
        break
      fi
    done
    if [[ "$used" -eq 0 ]]; then
      printf '%s\n' "$cuda"
      return
    fi
  done
  echo "No free CUDA device" >&2
  exit 2
}

run_one() {
  local cuda="$1"
  local rank="$2"
  local run_name="${RUN_PREFIX}_qkvo_self_r${rank}_${RUN_TIMESTAMP}"

  local cmd=(
    "$FLIP_RUNNER" train --cuda "$cuda" --nproc 1 --
    --task-name "$TASK_NAME"
    --run-prefix "$RUN_PREFIX"
    --lora-rank "$rank"
    --lora-attn-types self
    --lora-attn-projections q,k,v,o
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
    --wandb-tags h2r self_attn_qkvo no_merge_lora "rank:${rank}" "run:${RUN_TIMESTAMP}" cuda01_parallel
  )

  printf '\n[%(%F %T)T] CUDA %s start layout=qkvo(self) rank=%s\n' -1 "$cuda" "$rank"
  printf 'Run prefix: %q\n' "$RUN_PREFIX"
  printf 'Merge LoRAs: none\n'
  printf 'Command:'
  printf ' %q' "${cmd[@]}"
  printf '\n'

  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    return 0
  fi

  "${cmd[@]}"
}

for rank in "${LORA_RANKS_ARR[@]}"; do
  wait_for_slot
  cuda="$(next_cuda)"
  run_one "$cuda" "$rank" &
  pids+=("$!")
  pid_cuda+=("$cuda")
  sleep 2
done

for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    exit_code=1
  fi
done

exit "$exit_code"
