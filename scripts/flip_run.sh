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
  r2h_synthesize   Run python -m src.pipeline.r2h_synthesize.
  syn_error_analysis
                   Run scripts/run_syn_error_analysis.py.
  wan_vae_idm      Run python -m src.pipeline.wan_vae_idm.
  wan_pair_idm     Run python -m src.pipeline.wan_pair_idm.
  train            Run python -m torch.distributed.run -m src.pipeline.train.
  train_mitty_mixed_h2r
                   Run python -m torch.distributed.run -m src.pipeline.train_mitty_mixed_h2r.
  eval_mitty       Run python -m src.pipeline.evaluate_mitty_models.

Launcher options:
  --cuda IDS       Set CUDA_VISIBLE_DEVICES inside this launcher, e.g. 0 or 2,3.
  --nproc N        Set torch.distributed.run --nproc_per_node for train.
  -h, --help       Show this help.

Examples:
  scripts/flip_run.sh nvidia-smi
  scripts/flip_run.sh mitty_cache --cuda 0 -- --pair-dir training_data/pair/1s/train --output training_data/cache/1s/train --device cuda:0 --no-frames
  scripts/flip_run.sh sam2_precompute --cuda 0 -- --task all --device cuda:0 --resume
  scripts/flip_run.sh r2h_synthesize --cuda 0 -- --source-task Inspire_Collect_Clothes_MainCamOnly --duration 1s --run RUN --checkpoint latest --num-samples 1000 --resume-existing
  scripts/flip_run.sh syn_error_analysis --cuda 0 -- --run RUN --checkpoint latest
  scripts/flip_run.sh wan_vae_idm --cuda 2 -- train --device cuda:0 --max-samples 32
  scripts/flip_run.sh wan_pair_idm --cuda 2 -- train --device cuda:0 --max-samples 32
  scripts/flip_run.sh train --cuda 2,3 --nproc 2 -- --task-name pair_1s --max-steps 1000
  scripts/flip_run.sh train_mitty_mixed_h2r --cuda 2,3 --nproc 2 -- --task-name mixed_h2r --original-train-tasks Task_A --syn-train-tasks Task_A_syn --ood-eval-tasks Task_C --original-train-size 400 --syn-train-size 400 --max-steps 1000
  scripts/flip_run.sh eval_mitty --cuda 2 -- --device cuda:0 --eval-tail-percent 10
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
  mitty_cache|sam2_precompute|r2h_synthesize|syn_error_analysis|wan_vae_idm|wan_pair_idm|train|train_mitty_mixed_h2r|eval_mitty)
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
  r2h_synthesize)
    exec "$PYTHON_BIN" -m src.pipeline.r2h_synthesize "${script_args[@]}"
    ;;
  syn_error_analysis)
    exec "$PYTHON_BIN" scripts/run_syn_error_analysis.py "${script_args[@]}"
    ;;
  wan_vae_idm)
    exec "$PYTHON_BIN" -m src.pipeline.wan_vae_idm "${script_args[@]}"
    ;;
  wan_pair_idm)
    exec "$PYTHON_BIN" -m src.pipeline.wan_pair_idm "${script_args[@]}"
    ;;
  train)
    if [[ -z "$nproc" ]]; then
      if [[ -n "$cuda_devices" ]]; then
        nproc="$(count_cuda_devices "$cuda_devices")"
      else
        nproc="1"
      fi
    fi
    exec "$PYTHON_BIN" -m torch.distributed.run \
      --standalone \
      --nproc_per_node="$nproc" \
      -m src.pipeline.train \
      "${script_args[@]}"
    ;;
  train_mitty_mixed_h2r)
    if [[ -z "$nproc" ]]; then
      if [[ -n "$cuda_devices" ]]; then
        nproc="$(count_cuda_devices "$cuda_devices")"
      else
        nproc="1"
      fi
    fi
    exec "$PYTHON_BIN" -m torch.distributed.run \
      --standalone \
      --nproc_per_node="$nproc" \
      -m src.pipeline.train_mitty_mixed_h2r \
      "${script_args[@]}"
    ;;
  eval_mitty)
    exec "$PYTHON_BIN" -m src.pipeline.evaluate_mitty_models "${script_args[@]}"
    ;;
esac
