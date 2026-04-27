#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="/home/leadtek/miniconda3/envs/flip/bin/python"
LIBJPEG_SO="/home/leadtek/miniconda3/envs/flip/lib/libjpeg.so.8"
HF_HOME_DEFAULT="/disk_n/zzf/.cache/huggingface"
PIP_CACHE_DIR_DEFAULT="/disk_n/zzf/.pip_cache"

usage() {
  cat <<'USAGE'
Usage:
  scripts/flip_run.sh <subcommand> [launcher options] -- [script args...]
  scripts/flip_run.sh nvidia-smi [nvidia-smi args...]

Subcommands:
  nvidia-smi       Run NVIDIA-SMI on the host GPU device.
  mitty_cache      Run python -m src.pipeline.mitty_cache.
  sam2_precompute  Run python -m src.pipeline.sam2_precompute.
  train            Run torchrun -m src.pipeline.train.
  eval_mitty       Run python -m src.pipeline.evaluate_mitty_models.

Launcher options:
  --cuda IDS       Set CUDA_VISIBLE_DEVICES inside this launcher, e.g. 0 or 2,3.
  --nproc N        Set torchrun --nproc_per_node for train.
  -h, --help       Show this help.

Examples:
  scripts/flip_run.sh nvidia-smi
  scripts/flip_run.sh mitty_cache --cuda 0 -- --pair-dir training_data/pair/1s/train --output training_data/cache/1s/train --device cuda:0 --no-frames
  scripts/flip_run.sh sam2_precompute --cuda 0 -- --task all --device cuda:0 --resume
  scripts/flip_run.sh train --cuda 2,3 --nproc 2 -- --task-name appearance --loss uniform --cache-train training_data/cache/1s/train --cache-eval training_data/cache/1s/eval
  scripts/flip_run.sh eval_mitty --cuda 2 -- --device cuda:0 --samples-per-split 32
USAGE
}

fail() {
  echo "flip_run.sh: $*" >&2
  exit 2
}

count_cuda_devices() {
  local ids="$1"
  [[ -n "$ids" ]] || return 1
  local without_commas="${ids//,/}"
  [[ -n "$without_commas" ]] || return 1
  local commas="${ids//[^,]/}"
  echo $((${#commas} + 1))
}

if [[ $# -eq 0 ]]; then
  usage
  exit 2
fi

subcommand="$1"
shift

case "$subcommand" in
  -h|--help|help)
    usage
    exit 0
    ;;
  nvidia-smi)
    exec nvidia-smi "$@"
    ;;
  mitty_cache|sam2_precompute|train|eval_mitty)
    ;;
  *)
    usage >&2
    fail "unsupported subcommand: $subcommand"
    ;;
esac

cuda_devices=""
nproc=""
script_args=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cuda)
      [[ $# -ge 2 ]] || fail "--cuda requires a value"
      cuda_devices="$2"
      shift 2
      ;;
    --nproc)
      [[ $# -ge 2 ]] || fail "--nproc requires a value"
      nproc="$2"
      [[ "$nproc" =~ ^[0-9]+$ ]] || fail "--nproc must be a positive integer"
      [[ "$nproc" -gt 0 ]] || fail "--nproc must be a positive integer"
      shift 2
      ;;
    --)
      shift
      script_args=("$@")
      break
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      fail "launcher options must appear before --; unexpected argument: $1"
      ;;
  esac
done

if [[ ${#script_args[@]} -eq 0 ]]; then
  fail "missing -- [script args...] separator or script arguments"
fi

export LD_PRELOAD="$LIBJPEG_SO${LD_PRELOAD:+:$LD_PRELOAD}"
export HF_HOME="${HF_HOME:-$HF_HOME_DEFAULT}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$PIP_CACHE_DIR_DEFAULT}"
export no_proxy="${no_proxy:-localhost,127.0.0.1}"

if [[ -n "$cuda_devices" ]]; then
  export CUDA_VISIBLE_DEVICES="$cuda_devices"
fi

case "$subcommand" in
  mitty_cache)
    exec "$PYTHON_BIN" -m src.pipeline.mitty_cache "${script_args[@]}"
    ;;
  sam2_precompute)
    exec "$PYTHON_BIN" -m src.pipeline.sam2_precompute "${script_args[@]}"
    ;;
  train)
    if [[ -z "$nproc" ]]; then
      if [[ -n "$cuda_devices" ]]; then
        nproc="$(count_cuda_devices "$cuda_devices")"
      else
        nproc="1"
      fi
    fi
    exec torchrun --nproc_per_node="$nproc" -m src.pipeline.train "${script_args[@]}"
    ;;
  eval_mitty)
    exec "$PYTHON_BIN" -m src.pipeline.evaluate_mitty_models "${script_args[@]}"
    ;;
esac
