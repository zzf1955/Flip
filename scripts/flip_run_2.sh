#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT_DEFAULT="$(cd "$SCRIPT_DIR/.." && pwd)"

CONDA_PREFIX_DEFAULT="/disk_n/zzf_tmp/conda_envs/flip"
CONDA_PREFIX="${FLIP_CONDA_PREFIX:-$CONDA_PREFIX_DEFAULT}"
PYTHON_BIN="${FLIP_PYTHON_BIN:-$CONDA_PREFIX/bin/python}"
LIBJPEG_SO="${FLIP_LIBJPEG_SO:-$CONDA_PREFIX/lib/libjpeg.so.8}"
FFMPEG_BIN_DEFAULT="$CONDA_PREFIX/bin/ffmpeg"
SSL_CERT_DEFAULT="$CONDA_PREFIX/ssl/cert.pem"

PROJECT_ROOT="${FLIP_PROJECT_ROOT:-$PROJECT_ROOT_DEFAULT}"
WORKDIR="${FLIP_WORKDIR:-$PROJECT_ROOT}"

HF_HOME_DEFAULT="/disk_n/zzf/.cache/huggingface"
TORCH_HOME_DEFAULT="/disk_n/zzf/.cache/torch"
PIP_CACHE_DIR_DEFAULT="/disk_n/zzf/.pip_cache"

usage() {
  cat <<'USAGE'
Usage:
  scripts/flip_run_2.sh <subcommand> [launcher options] -- [script args...]
  scripts/flip_run_2.sh nvidia-smi [nvidia-smi args...]

Subcommands:
  nvidia-smi       Run NVIDIA-SMI on the host GPU device.
  mitty_cache      Run python -m src.pipeline.mitty_cache.
  sam2_precompute  Run python -m src.pipeline.sam2_precompute.
  train            Run python -m torch.distributed.run -m src.pipeline.train.
  eval_mitty       Run python -m src.pipeline.evaluate_mitty_models.

Launcher options:
  --cuda IDS       Set CUDA_VISIBLE_DEVICES inside this launcher, e.g. 0 or 0,2.
  --nproc N        Set torch.distributed.run --nproc_per_node for train.
  --dry-run        Print the resolved command and environment without running.
  -h, --help       Show this help.

Environment overrides:
  FLIP_CONDA_PREFIX  Conda env path. Default: /disk_n/zzf_tmp/conda_envs/flip
  FLIP_PROJECT_ROOT  Code root added to PYTHONPATH. Default: this script's parent.
  FLIP_WORKDIR       Working directory for relative paths. Default: FLIP_PROJECT_ROOT.
  FLIP_PYTHON_BIN    Python executable. Default: $FLIP_CONDA_PREFIX/bin/python

Examples:
  scripts/flip_run_2.sh nvidia-smi
  scripts/flip_run_2.sh train --cuda 0,2 --nproc 2 -- --task-name R2H --loss uniform --cache-train training_data/cache/vae/pair_1s_r2h/train --cache-eval training_data/cache/vae/pair_1s_r2h/eval
  FLIP_WORKDIR=/disk_n/zzf/flip scripts/flip_run_2.sh train --cuda 0,2 --nproc 2 -- --task-name transfer --loss uniform --cache-train training_data/cache/vae/pair_1s_train3/train
USAGE
}

fail() {
  echo "flip_run_2.sh: $*" >&2
  exit 2
}

require_path() {
  local path="$1"
  local label="$2"
  [[ -e "$path" ]] || fail "$label not found: $path"
}

count_cuda_devices() {
  local ids="$1"
  [[ -n "$ids" ]] || return 1
  local without_commas="${ids//,/}"
  [[ -n "$without_commas" ]] || return 1
  local commas="${ids//[^,]/}"
  echo $((${#commas} + 1))
}

quote_cmd() {
  printf '%q ' "$@"
  printf '\n'
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
dry_run=0
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
    --dry-run)
      dry_run=1
      shift
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

require_path "$PYTHON_BIN" "python"
require_path "$LIBJPEG_SO" "libjpeg"
require_path "$PROJECT_ROOT/src" "project src"
require_path "$WORKDIR" "workdir"

export LD_PRELOAD="$LIBJPEG_SO${LD_PRELOAD:+:$LD_PRELOAD}"
export HF_HOME="${HF_HOME:-$HF_HOME_DEFAULT}"
export TORCH_HOME="${TORCH_HOME:-$TORCH_HOME_DEFAULT}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$PIP_CACHE_DIR_DEFAULT}"
export FFMPEG_BIN="${FFMPEG_BIN:-$FFMPEG_BIN_DEFAULT}"
export SSL_CERT_FILE="${SSL_CERT_FILE:-$SSL_CERT_DEFAULT}"
export REQUESTS_CA_BUNDLE="${REQUESTS_CA_BUNDLE:-$SSL_CERT_DEFAULT}"
export CURL_CA_BUNDLE="${CURL_CA_BUNDLE:-$SSL_CERT_DEFAULT}"
export WANDB_X_DISABLE_SERVICE="${WANDB_X_DISABLE_SERVICE:-true}"
export WANDB_CORE="${WANDB_CORE:-disabled}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export no_proxy="${no_proxy:-localhost,127.0.0.1,10.20.1.0/24}"
export NO_PROXY="${NO_PROXY:-$no_proxy}"

if [[ -n "$cuda_devices" ]]; then
  export CUDA_VISIBLE_DEVICES="$cuda_devices"
fi

if [[ -n "${PYTHONPATH:-}" ]]; then
  export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
else
  export PYTHONPATH="$PROJECT_ROOT"
fi

cd "$WORKDIR"

cmd=()
case "$subcommand" in
  mitty_cache)
    cmd=("$PYTHON_BIN" -m src.pipeline.mitty_cache "${script_args[@]}")
    ;;
  sam2_precompute)
    cmd=("$PYTHON_BIN" -m src.pipeline.sam2_precompute "${script_args[@]}")
    ;;
  train)
    if [[ -z "$nproc" ]]; then
      if [[ -n "$cuda_devices" ]]; then
        nproc="$(count_cuda_devices "$cuda_devices")"
      else
        nproc="1"
      fi
    fi
    cmd=(
      "$PYTHON_BIN" -m torch.distributed.run
      --standalone
      --nproc_per_node="$nproc"
      -m src.pipeline.train
      "${script_args[@]}"
    )
    ;;
  eval_mitty)
    cmd=("$PYTHON_BIN" -m src.pipeline.evaluate_mitty_models "${script_args[@]}")
    ;;
esac

if [[ "$dry_run" -eq 1 ]]; then
  echo "WORKDIR=$WORKDIR"
  echo "PROJECT_ROOT=$PROJECT_ROOT"
  echo "PYTHON_BIN=$PYTHON_BIN"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
  echo "LD_PRELOAD=$LD_PRELOAD"
  echo "HF_HOME=$HF_HOME"
  echo "TORCH_HOME=$TORCH_HOME"
  echo "PIP_CACHE_DIR=$PIP_CACHE_DIR"
  echo "FFMPEG_BIN=$FFMPEG_BIN"
  echo "SSL_CERT_FILE=$SSL_CERT_FILE"
  echo "REQUESTS_CA_BUNDLE=$REQUESTS_CA_BUNDLE"
  echo "CURL_CA_BUNDLE=$CURL_CA_BUNDLE"
  echo "WANDB_X_DISABLE_SERVICE=$WANDB_X_DISABLE_SERVICE"
  echo "WANDB_CORE=$WANDB_CORE"
  echo -n "CMD="
  quote_cmd "${cmd[@]}"
  exit 0
fi

exec "${cmd[@]}"
