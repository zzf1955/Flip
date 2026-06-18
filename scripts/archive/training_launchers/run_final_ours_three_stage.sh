#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"

FLIP_RUNNER="$PROJECT_ROOT/scripts/flip_run.sh"

CUDA_ID="${CUDA_ID:-0}"
NPROC="${NPROC:-1}"
RUN_ID="${RUN_ID:-$(date +%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-training_data/log}"
RUN_STAGE2="${RUN_STAGE2:-1}"
RUN_STAGE3="${RUN_STAGE3:-0}"

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

STAGE2_TRAIN_SIZE="${STAGE2_TRAIN_SIZE:-0}"
STAGE3_TRAIN_SIZE="${STAGE3_TRAIN_SIZE:-400}"

STEP1_CKPT="${STEP1_CKPT:-training_data/log/final_ours_step1_0507_004839-identity_r2r_1s-22592d_r32_self_qkvo_ffn_1000s_0507_004911/ckpt/step-1000.safetensors}"
MAIN_TRAIN_TASKS="${MAIN_TRAIN_TASKS:-grab_cup_v1,grab_cube2_v1,push_box_random_v1}"
OOD_TASKS="${OOD_TASKS:-roll}"
PAIR_ROOT="${PAIR_ROOT:-training_data/pair}"
CACHE_ROOT="${CACHE_ROOT:-training_data/cache/vae}"
T5_CACHE_DIR="${T5_CACHE_DIR:-training_data/cache/t5/blur_r2r/1s}"

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
  --pair-root "$PAIR_ROOT"
  --cache-root "$CACHE_ROOT"
  --t5-cache-dir "$T5_CACHE_DIR"
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

require_file "$STEP1_CKPT"
STAGE2_PREFIX="final_ours_h2r_sam3_step2_${RUN_ID}"
STAGE3_PREFIX="final_ours_h2r_step3_${RUN_ID}"
printf 'STEP1_CKPT=%s\n' "$STEP1_CKPT"

if [[ "$RUN_STAGE2" == "1" ]]; then
run_train "ours step2 H2R SAM3 blur_r2r appearance" \
  --task-name blur_r2r_1s \
  --train-tasks "$MAIN_TRAIN_TASKS" \
  --ood-tasks "$OOD_TASKS" \
  --run-prefix "$STAGE2_PREFIX" \
  --merge-lora "$STEP1_CKPT" \
  --lora-rank 256 \
  --lora-target-modules "$QKVO_FFN_TARGETS" \
  --train-size "$STAGE2_TRAIN_SIZE" \
  "${COMMON_ARGS[@]}" \
  --wandb-tags final ours h2r_sam3 step2 blur_r2r merge_step1 layout:self_qkvo_ffn rank:256

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  STEP2_CKPT="<dry-run-step2-checkpoint>"
else
  STEP2_RUN_DIR="$(latest_run_dir "$STAGE2_PREFIX")"
  STEP2_CKPT="$(latest_checkpoint "$STEP2_RUN_DIR")"
  require_file "$STEP2_CKPT"
fi
printf 'STEP2_CKPT=%s\n' "$STEP2_CKPT"
else
  STEP2_CKPT="${STEP2_CKPT:-}"
  printf 'RUN_STAGE2=0, skipping step2. STEP2_CKPT=%s\n' "${STEP2_CKPT:-<unset>}"
fi

if [[ "$RUN_STAGE3" != "1" ]]; then
  printf '\nStep3 is disabled by default. Pair data is not part of this reproduction yet.\n'
  printf 'Current checkpoints:\n'
  printf '  step1(reused): %s\n' "$STEP1_CKPT"
  printf '  step2: %s\n' "${STEP2_CKPT:-<not-run>}"
  exit 0
fi

if [[ -z "${STEP2_CKPT:-}" ]]; then
  echo "RUN_STAGE3=1 requires STEP2_CKPT or RUN_STAGE2=1" >&2
  exit 2
fi
if [[ "$STEP2_CKPT" != "<dry-run-step2-checkpoint>" ]]; then
  require_file "$STEP2_CKPT"
fi

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
