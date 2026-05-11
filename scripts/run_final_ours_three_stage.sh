#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

FLIP_RUNNER="$PROJECT_ROOT/scripts/flip_run.sh"

CUDA_ID="${CUDA_ID:-0}"
NPROC="${NPROC:-1}"
RUN_ID="${RUN_ID:-$(date +%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-training_data/log}"

BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_STEPS="${MAX_STEPS:-1000}"
SAVE_STEPS="${SAVE_STEPS:-100}"
EVAL_STEPS="${EVAL_STEPS:-100}"
EVAL_VIDEO_STEPS="${EVAL_VIDEO_STEPS:-100}"

LR="${LR:-1e-4}"
LR_MIN="${LR_MIN:-1e-6}"
WARMUP_STEPS="${WARMUP_STEPS:-50}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
WANDB_PROJECT="${WANDB_PROJECT:-Flip}"

IN_TASK_EVAL_SIZE="${IN_TASK_EVAL_SIZE:-16}"
OOD_EVAL_SIZE="${OOD_EVAL_SIZE:-16}"
IN_TASK_VIDEO_SIZE="${IN_TASK_VIDEO_SIZE:-4}"
OOD_VIDEO_SIZE="${OOD_VIDEO_SIZE:-2}"
DATA_SEED="${DATA_SEED:-42}"

STAGE1_TRAIN_SIZE="${STAGE1_TRAIN_SIZE:-0}"
STAGE2_TRAIN_SIZE="${STAGE2_TRAIN_SIZE:-0}"
STAGE3_TRAIN_SIZE="${STAGE3_TRAIN_SIZE:-400}"

IDENTITY_TRAIN_TASKS="${IDENTITY_TRAIN_TASKS:-Inspire_Put_Clothes_into_Washing_Machine,Inspire_Put_Clothes_Into_Basket}"
MAIN_TRAIN_TASKS="${MAIN_TRAIN_TASKS:-Inspire_Collect_Clothes_MainCamOnly,Inspire_Put_Clothes_into_Washing_Machine}"
OOD_TASKS="${OOD_TASKS:-Inspire_Pickup_Pillow_MainCamOnly}"

QKVO_FFN_TARGETS="${QKVO_FFN_TARGETS:-self_attn.q,self_attn.k,self_attn.v,self_attn.o,ffn.0,ffn.2}"

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "Missing required file: $path" >&2
    exit 2
  fi
}

latest_run_dir() {
  local run_prefix="$1"
  local run_dir
  run_dir="$(
    find "$OUTPUT_DIR" -maxdepth 1 -type d -name "${run_prefix}-*" -printf '%T@ %p\n' \
      | sort -nr \
      | awk 'NR == 1 {print $2}'
  )"
  if [[ -z "$run_dir" ]]; then
    echo "No run dir found for prefix: $run_prefix" >&2
    exit 2
  fi
  printf '%s\n' "$run_dir"
}

latest_checkpoint() {
  local run_dir="$1"
  local ckpt
  ckpt="$(
    find "$run_dir/ckpt" -maxdepth 1 -type f -name 'step-*.safetensors' -printf '%f\n' \
      | sort -V \
      | tail -n 1
  )"
  if [[ -z "$ckpt" ]]; then
    echo "No checkpoint found under: $run_dir/ckpt" >&2
    exit 2
  fi
  printf '%s\n' "$run_dir/ckpt/$ckpt"
}

run_train() {
  local stage_name="$1"
  shift
  local cmd=(
    "$FLIP_RUNNER" train --cuda "$CUDA_ID" --nproc "$NPROC" --
    "$@"
  )

  printf '\n[%(%F %T)T] START %s\n' -1 "$stage_name"
  printf 'Command:'
  printf ' %q' "${cmd[@]}"
  printf '\n'

  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    return 0
  fi

  "${cmd[@]}"
}

COMMON_ARGS=(
  --output-dir "$OUTPUT_DIR"
  --in-task-eval-size "$IN_TASK_EVAL_SIZE"
  --ood-eval-size "$OOD_EVAL_SIZE"
  --in-task-video-size "$IN_TASK_VIDEO_SIZE"
  --ood-video-size "$OOD_VIDEO_SIZE"
  --data-seed "$DATA_SEED"
  --max-steps "$MAX_STEPS"
  --save-steps "$SAVE_STEPS"
  --eval-steps "$EVAL_STEPS"
  --eval-video-steps "$EVAL_VIDEO_STEPS"
  --batch-size "$BATCH_SIZE"
  --lr "$LR"
  --lr-min "$LR_MIN"
  --warmup-steps "$WARMUP_STEPS"
  --weight-decay "$WEIGHT_DECAY"
  --wandb-project "$WANDB_PROJECT"
)

STAGE1_PREFIX="final_ours_step1_${RUN_ID}"
STAGE2_PREFIX="final_ours_step2_${RUN_ID}"
STAGE3_PREFIX="final_ours_step3_${RUN_ID}"

run_train "ours step1 identity" \
  --task-name identity_r2r_1s \
  --train-tasks "$IDENTITY_TRAIN_TASKS" \
  --ood-tasks "$OOD_TASKS" \
  --run-prefix "$STAGE1_PREFIX" \
  --lora-rank 32 \
  --lora-target-modules "$QKVO_FFN_TARGETS" \
  --train-size "$STAGE1_TRAIN_SIZE" \
  "${COMMON_ARGS[@]}" \
  --wandb-tags final ours step1 identity_r2r layout:self_qkvo_ffn rank:32

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  STEP1_CKPT="<dry-run-step1-checkpoint>"
else
  STEP1_RUN_DIR="$(latest_run_dir "$STAGE1_PREFIX")"
  STEP1_CKPT="$(latest_checkpoint "$STEP1_RUN_DIR")"
  require_file "$STEP1_CKPT"
fi
printf 'STEP1_CKPT=%s\n' "$STEP1_CKPT"

run_train "ours step2 blur_r2r" \
  --task-name blur_r2r_1s \
  --train-tasks "$MAIN_TRAIN_TASKS" \
  --ood-tasks "$OOD_TASKS" \
  --run-prefix "$STAGE2_PREFIX" \
  --merge-lora "$STEP1_CKPT" \
  --lora-rank 256 \
  --lora-target-modules "$QKVO_FFN_TARGETS" \
  --train-size "$STAGE2_TRAIN_SIZE" \
  "${COMMON_ARGS[@]}" \
  --wandb-tags final ours step2 blur_r2r merge_step1 layout:self_qkvo_ffn rank:256

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  STEP2_CKPT="<dry-run-step2-checkpoint>"
else
  STEP2_RUN_DIR="$(latest_run_dir "$STAGE2_PREFIX")"
  STEP2_CKPT="$(latest_checkpoint "$STEP2_RUN_DIR")"
  require_file "$STEP2_CKPT"
fi
printf 'STEP2_CKPT=%s\n' "$STEP2_CKPT"

run_train "ours step3 h2r" \
  --task-name h2r_1s \
  --train-tasks "$MAIN_TRAIN_TASKS" \
  --ood-tasks "$OOD_TASKS" \
  --run-prefix "$STAGE3_PREFIX" \
  --merge-lora "$STEP1_CKPT" \
  --merge-lora "$STEP2_CKPT" \
  --lora-rank 96 \
  --lora-target-modules "$QKVO_FFN_TARGETS" \
  --train-size "$STAGE3_TRAIN_SIZE" \
  "${COMMON_ARGS[@]}" \
  --wandb-tags final ours step3 h2r merge_step1_step2 layout:self_qkvo_ffn rank:96

if [[ "${DRY_RUN:-0}" != "1" ]]; then
  STEP3_RUN_DIR="$(latest_run_dir "$STAGE3_PREFIX")"
  STEP3_CKPT="$(latest_checkpoint "$STEP3_RUN_DIR")"
  require_file "$STEP3_CKPT"
  printf '\nFinal checkpoints:\n'
  printf '  step1: %s\n' "$STEP1_CKPT"
  printf '  step2: %s\n' "$STEP2_CKPT"
  printf '  step3: %s\n' "$STEP3_CKPT"
fi
