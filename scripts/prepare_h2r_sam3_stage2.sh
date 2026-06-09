#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

FLIP_RUNNER="$PROJECT_ROOT/scripts/flip_run.sh"

CUDA_ID="${CUDA_ID:-2}"
TASKS="${TASKS:-grab_cup_v1,grab_cube2_v1,push_box_random_v1,roll}"
MAIN_TRAIN_TASKS="${MAIN_TRAIN_TASKS:-grab_cup_v1,grab_cube2_v1,push_box_random_v1}"
OOD_TASKS="${OOD_TASKS:-roll}"

MASK_ROOT="${MASK_ROOT:-training_data/h2r_sam3_mask}"
PAIR_ROOT="${PAIR_ROOT:-training_data/pair}"
CACHE_ROOT="${CACHE_ROOT:-training_data/cache/vae}"
T5_CACHE_DIR="${T5_CACHE_DIR:-training_data/cache/t5/blur_r2r/1s}"
OUTPUT_DIR="${OUTPUT_DIR:-training_data/log}"

RUN_PRECOMPUTE="${RUN_PRECOMPUTE:-1}"
RUN_PAIR="${RUN_PAIR:-1}"
RUN_CACHE="${RUN_CACHE:-1}"
RUN_TRAIN="${RUN_TRAIN:-0}"
DRY_RUN="${DRY_RUN:-0}"

MAX_EPISODES_PER_TASK="${MAX_EPISODES_PER_TASK:-0}"
MAX_CLIPS_PER_EPISODE="${MAX_CLIPS_PER_EPISODE:-0}"
CLIP_STRIDE="${CLIP_STRIDE:-1.0}"
RESIZE="${RESIZE:-224x416}"
CLEAN_PAIR="${CLEAN_PAIR:-0}"

SAM3_PROMPT="${SAM3_PROMPT:-robot arm}"
SAM3_BACKUP_PROMPT="${SAM3_BACKUP_PROMPT:-robotic arm}"
SAM3_MAX_OBJECTS="${SAM3_MAX_OBJECTS:-1}"

CACHE_BATCH_SIZE="${CACHE_BATCH_SIZE:-4}"
CACHE_PREFETCH_WORKERS="${CACHE_PREFETCH_WORKERS:-8}"
CACHE_PREFETCH_BATCHES="${CACHE_PREFETCH_BATCHES:-2}"
CACHE_SAVE_WORKERS="${CACHE_SAVE_WORKERS:-1}"

run_cmd() {
  printf '\nCommand:'
  printf ' %q' "$@"
  printf '\n'
  if [[ "$DRY_RUN" == "1" ]]; then
    return 0
  fi
  "$@"
}

split_csv() {
  local value="$1"
  local old_ifs="$IFS"
  IFS=','
  read -r -a _split_items <<< "$value"
  IFS="$old_ifs"
  for item in "${_split_items[@]}"; do
    item="${item#"${item%%[![:space:]]*}"}"
    item="${item%"${item##*[![:space:]]}"}"
    [[ -n "$item" ]] && printf '%s\n' "$item"
  done
}

printf 'H2R SAM3 stage2 preparation\n'
printf '  tasks:       %s\n' "$TASKS"
printf '  train tasks: %s\n' "$MAIN_TRAIN_TASKS"
printf '  ood tasks:   %s\n' "$OOD_TASKS"
printf '  cuda:        %s\n' "$CUDA_ID"
printf '  mask root:   %s\n' "$MASK_ROOT"
printf '  pair root:   %s\n' "$PAIR_ROOT"
printf '  cache root:  %s\n' "$CACHE_ROOT"
printf '  resize:      %s\n' "$RESIZE"
printf '  dry run:     %s\n' "$DRY_RUN"

if [[ "$RUN_PRECOMPUTE" == "1" ]]; then
  run_cmd "$FLIP_RUNNER" h2r_sam3_precompute --cuda "$CUDA_ID" -- \
    --tasks "$TASKS" \
    --output-root "$MASK_ROOT" \
    --prompt "$SAM3_PROMPT" \
    --backup-prompt "$SAM3_BACKUP_PROMPT" \
    --max-num-objects "$SAM3_MAX_OBJECTS" \
    --clip-stride "$CLIP_STRIDE" \
    --max-episodes-per-task "$MAX_EPISODES_PER_TASK" \
    --max-clips-per-episode "$MAX_CLIPS_PER_EPISODE" \
    --resume
fi

if [[ "$RUN_PAIR" == "1" ]]; then
  pair_args=(
    "$FLIP_RUNNER" h2r_sam3_blur_pair --
    --tasks "$TASKS"
    --mask-root "$MASK_ROOT"
    --pair-root "$PAIR_ROOT"
    --resize "$RESIZE"
    --clip-stride "$CLIP_STRIDE"
    --max-episodes-per-task "$MAX_EPISODES_PER_TASK"
    --max-clips-per-episode "$MAX_CLIPS_PER_EPISODE"
  )
  if [[ "$CLEAN_PAIR" == "1" ]]; then
    pair_args+=(--clean)
  else
    pair_args+=(--resume)
  fi
  run_cmd "${pair_args[@]}"
fi

if [[ "$RUN_CACHE" == "1" ]]; then
  while IFS= read -r task; do
    run_cmd "$FLIP_RUNNER" mitty_cache --cuda "$CUDA_ID" -- \
      --pair-dir "$PAIR_ROOT/blur_r2r/1s/$task" \
      --output "$CACHE_ROOT/blur_r2r/1s/$task" \
      --t5-cache-dir "$T5_CACHE_DIR" \
      --device cuda:0 \
      --batch-size "$CACHE_BATCH_SIZE" \
      --prefetch-workers "$CACHE_PREFETCH_WORKERS" \
      --prefetch-batches "$CACHE_PREFETCH_BATCHES" \
      --save-workers "$CACHE_SAVE_WORKERS"
  done < <(split_csv "$TASKS")
fi

if [[ "$RUN_TRAIN" == "1" ]]; then
  run_cmd env \
    CUDA_ID="$CUDA_ID" \
    MAIN_TRAIN_TASKS="$MAIN_TRAIN_TASKS" \
    OOD_TASKS="$OOD_TASKS" \
    PAIR_ROOT="$PAIR_ROOT" \
    CACHE_ROOT="$CACHE_ROOT" \
    T5_CACHE_DIR="$T5_CACHE_DIR" \
    OUTPUT_DIR="$OUTPUT_DIR" \
    "$PROJECT_ROOT/scripts/run_final_ours_three_stage.sh"
fi
