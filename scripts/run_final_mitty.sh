#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

FLIP_RUNNER="$PROJECT_ROOT/scripts/flip_run.sh"

CUDA_ID="${CUDA_ID:-2}"
NPROC="${NPROC:-1}"
RUN_ID="${RUN_ID:-$(date +%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-training_data/log}"

BATCH_SIZE="${BATCH_SIZE:-4}"
TRAIN_SIZE="${TRAIN_SIZE:-400}"
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

MAIN_TRAIN_TASKS="${MAIN_TRAIN_TASKS:-Inspire_Collect_Clothes_MainCamOnly,Inspire_Put_Clothes_into_Washing_Machine}"
OOD_TASKS="${OOD_TASKS:-Inspire_Pickup_Pillow_MainCamOnly}"
QKV_TARGETS="${QKV_TARGETS:-self_attn.q,self_attn.k,self_attn.v}"
RUN_PREFIX="${RUN_PREFIX:-final_mitty_${RUN_ID}}"

cmd=(
  "$FLIP_RUNNER" train --cuda "$CUDA_ID" --nproc "$NPROC" --
  --task-name h2r_1s
  --train-tasks "$MAIN_TRAIN_TASKS"
  --ood-tasks "$OOD_TASKS"
  --run-prefix "$RUN_PREFIX"
  --output-dir "$OUTPUT_DIR"
  --lora-rank 96
  --lora-target-modules "$QKV_TARGETS"
  --train-size "$TRAIN_SIZE"
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
  --wandb-tags final mitty h2r layout:self_qkv rank:96
)

printf '\n[%(%F %T)T] START final Mitty h2r\n' -1
printf 'Command:'
printf ' %q' "${cmd[@]}"
printf '\n'

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  exit 0
fi

exec "${cmd[@]}"
