#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

if [[ -z "${FLIP_RUNNER:-}" ]]; then
  if [[ -x "$PROJECT_ROOT/scripts/flip_run_2.sh" ]]; then
    FLIP_RUNNER="$PROJECT_ROOT/scripts/flip_run_2.sh"
  else
    FLIP_RUNNER="$PROJECT_ROOT/scripts/flip_run.sh"
  fi
fi

CUDA_IDS="${CUDA_IDS:-0,2}"
NPROC="${NPROC:-2}"

TASK_NAME="${TASK_NAME:-attn_ffn_selected}"
LOSS="${LOSS:-uniform}"

CACHE_TRAIN="${CACHE_TRAIN:-output/mitty_cache_1s/train}"
CACHE_EVAL="${CACHE_EVAL:-output/mitty_cache_1s/eval}"
CACHE_OOD="${CACHE_OOD:-output/mitty_cache_1s/ood_eval}"
T5_CACHE_DIR="${T5_CACHE_DIR:-training_data/cache/t5}"
OUTPUT_DIR="${OUTPUT_DIR:-training_data/log}"

LORA_RANK="${LORA_RANK:-16}"
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-q,k,v,o,ffn.0,ffn.2}"

BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_STEPS="${MAX_STEPS:-2000}"
SAVE_STEPS="${SAVE_STEPS:-100}"
EVAL_STEPS="${EVAL_STEPS:-100}"
EVAL_T_SAMPLES="${EVAL_T_SAMPLES:-5}"
EVAL_VIDEO_STEPS="${EVAL_VIDEO_STEPS:-100}"
EVAL_VIDEO_SAMPLES_IN_TASK="${EVAL_VIDEO_SAMPLES_IN_TASK:-4}"
EVAL_VIDEO_SAMPLES_OOD="${EVAL_VIDEO_SAMPLES_OOD:-2}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-30}"

LR="${LR:-1e-4}"
LR_MIN="${LR_MIN:-1e-6}"
WARMUP_STEPS="${WARMUP_STEPS:-50}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"

WANDB_PROJECT="${WANDB_PROJECT:-flip}"
WANDB_TAGS="${WANDB_TAGS:-attn_ffn_lora selected_data r${LORA_RANK} bs${BATCH_SIZE}}"

cmd=(
  "$FLIP_RUNNER" train --cuda "$CUDA_IDS" --nproc "$NPROC" --
  --task-name "$TASK_NAME"
  --loss "$LOSS"
  --cache-train "$CACHE_TRAIN"
  --cache-eval "$CACHE_EVAL"
  --cache-ood "$CACHE_OOD"
  --t5-cache-dir "$T5_CACHE_DIR"
  --output-dir "$OUTPUT_DIR"
  --lora-rank "$LORA_RANK"
  --lora-target-modules "$LORA_TARGET_MODULES"
  --batch-size "$BATCH_SIZE"
  --max-steps "$MAX_STEPS"
  --save-steps "$SAVE_STEPS"
  --eval-steps "$EVAL_STEPS"
  --eval-t-samples "$EVAL_T_SAMPLES"
  --eval-video-steps "$EVAL_VIDEO_STEPS"
  --eval-video-samples-in-task "$EVAL_VIDEO_SAMPLES_IN_TASK"
  --eval-video-samples-ood "$EVAL_VIDEO_SAMPLES_OOD"
  --num-inference-steps "$NUM_INFERENCE_STEPS"
  --lr "$LR"
  --lr-min "$LR_MIN"
  --warmup-steps "$WARMUP_STEPS"
  --weight-decay "$WEIGHT_DECAY"
  --wandb-project "$WANDB_PROJECT"
  --wandb-tags
)

read -r -a wandb_tags <<< "$WANDB_TAGS"
cmd+=("${wandb_tags[@]}")

if [[ -n "${WANDB_RUN_NAME:-}" ]]; then
  cmd+=(--wandb-run-name "$WANDB_RUN_NAME")
fi

if [[ -n "${INIT_LORA:-}" ]]; then
  cmd+=(--init-lora "$INIT_LORA")
fi

if [[ -n "${MERGE_LORA:-}" ]]; then
  read -r -a merge_loras <<< "$MERGE_LORA"
  for lora_path in "${merge_loras[@]}"; do
    cmd+=(--merge-lora "$lora_path")
  done
  cmd+=(--merge-lora-rank "${MERGE_LORA_RANK:-96}")
fi

if [[ -n "${PATCH_DIR:-}" ]]; then
  cmd+=(--patch-dir "$PATCH_DIR")
fi

cmd+=("$@")

printf 'Running:'
printf ' %q' "${cmd[@]}"
printf '\n'

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  exit 0
fi

exec "${cmd[@]}"
