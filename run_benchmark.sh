#!/usr/bin/env bash

set -euo pipefail

CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
NPROC_PER_NODE="${NPROC_PER_NODE:-}"
MASTER_PORT="${MASTER_PORT:-29600}"
LOG_DIR="${LOG_DIR:-logs}"
RUN_DIR="${RUN_DIR:-runs}"
PYTHON_BIN="${PYTHON_BIN:-python}"

TASKS=(
  "gsm8k:128"
  "math500:128"
  "aime24:30"
  "aime25:30"
  "humaneval:164"
  "mbpp:128"
  "livecodebench:128"
  "swe-bench:128"
  "mt-bench:80"
  "alpaca:128"
)

MODEL_DRAFT_PAIRS=(
  "Qwen/Qwen3-4B|z-lab/Qwen3-4B-DFlash-b16"
  "Qwen/Qwen3-8B|z-lab/Qwen3-8B-DFlash-b16"
  "Qwen/Qwen3-Coder-30B-A3B-Instruct|z-lab/Qwen3-Coder-30B-A3B-DFlash"
)

TEMPERATURES=(
  "0.0"
  "1.0"
)

MODES=(
  "sdpa"
  "flash_attn"
)

MAX_NEW_TOKENS=2048
DRAFT_TYPE="dflash"

usage() {
  cat <<'EOF'
Usage: bash run_benchmark.sh [options]

With no options, runs the complete benchmark sweep. Repeat --task,
--model-draft-pair, --temperature, or --mode to build a custom sweep.

Options:
  --task DATASET:MAX_SAMPLES       Dataset and sample limit (repeatable)
  --model-draft-pair MODEL|DRAFT   Target and draft model pair (repeatable)
  --temperature VALUE              Sampling temperature (repeatable)
  --mode MODE                      sdpa or flash_attn (repeatable)
  --draft-type TYPE                dflash or dflash2 (default: dflash)
  --gpus IDS                       CUDA device IDs, for example 0 or 0,1
  --nproc-per-node COUNT           Worker count (defaults to number of GPUs)
  --max-new-tokens COUNT           Generation limit per sample (default: 2048)
  --master-port PORT               torchrun master port (default: 29600)
  --python PATH                    Python interpreter (default: python)
  --log-dir PATH                   Log output directory (default: logs)
  --run-dir PATH                   Benchmark output directory (default: runs)
  -h, --help                       Show this help
EOF
}

custom_tasks=false
custom_pairs=false
custom_temperatures=false
custom_modes=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --task)
      if [[ "${custom_tasks}" == false ]]; then
        TASKS=()
        custom_tasks=true
      fi
      TASKS+=("$2")
      shift 2
      ;;
    --model-draft-pair)
      if [[ "${custom_pairs}" == false ]]; then
        MODEL_DRAFT_PAIRS=()
        custom_pairs=true
      fi
      MODEL_DRAFT_PAIRS+=("$2")
      shift 2
      ;;
    --temperature)
      if [[ "${custom_temperatures}" == false ]]; then
        TEMPERATURES=()
        custom_temperatures=true
      fi
      TEMPERATURES+=("$2")
      shift 2
      ;;
    --mode)
      if [[ "${custom_modes}" == false ]]; then
        MODES=()
        custom_modes=true
      fi
      MODES+=("$2")
      shift 2
      ;;
    --draft-type)
      DRAFT_TYPE="$2"
      shift 2
      ;;
    --gpus)
      CUDA_DEVICES="$2"
      shift 2
      ;;
    --nproc-per-node)
      NPROC_PER_NODE="$2"
      shift 2
      ;;
    --max-new-tokens)
      MAX_NEW_TOKENS="$2"
      shift 2
      ;;
    --master-port)
      MASTER_PORT="$2"
      shift 2
      ;;
    --python)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --log-dir)
      LOG_DIR="$2"
      shift 2
      ;;
    --run-dir)
      RUN_DIR="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"
if [[ -z "${NPROC_PER_NODE}" ]]; then
  IFS=',' read -r -a visible_devices <<< "${CUDA_VISIBLE_DEVICES}"
  NPROC_PER_NODE="${#visible_devices[@]}"
fi

for mode in "${MODES[@]}"; do
  if [[ "${mode}" != "sdpa" && "${mode}" != "flash_attn" ]]; then
    echo "Invalid mode '${mode}': expected sdpa or flash_attn" >&2
    exit 2
  fi
done

if [[ "${DRAFT_TYPE}" != "dflash" && "${DRAFT_TYPE}" != "dflash2" ]]; then
  echo "Invalid draft type '${DRAFT_TYPE}': expected dflash or dflash2" >&2
  exit 2
fi
if [[ "${DRAFT_TYPE}" == "dflash2" ]]; then
  if [[ "${NPROC_PER_NODE}" != "1" ]]; then
    echo "DFlash2 proof-of-concept benchmarking currently requires one GPU" >&2
    exit 2
  fi
  for mode in "${MODES[@]}"; do
    if [[ "${mode}" != "sdpa" ]]; then
      echo "DFlash2 currently supports only --mode sdpa" >&2
      exit 2
    fi
  done
  for temperature in "${TEMPERATURES[@]}"; do
    if [[ "${temperature}" != "0" && "${temperature}" != "0.0" ]]; then
      echo "DFlash2 currently supports only --temperature 0.0" >&2
      exit 2
    fi
  done
fi

mkdir -p "$LOG_DIR" "$RUN_DIR"

if ! "${PYTHON_BIN}" -c "import torch, loguru" >/dev/null 2>&1; then
  echo "Python environment '${PYTHON_BIN}' is missing benchmark dependencies." >&2
  echo "Activate the project venv or pass --python /path/to/venv/bin/python." >&2
  exit 2
fi

COMMON_BENCHMARK_ARGS=(
  --max-new-tokens "${MAX_NEW_TOKENS}"
)

slugify() {
  local value="$1"
  value="${value//\//_}"
  value="${value//:/_}"
  value="${value// /_}"
  echo "$value"
}

run_benchmark() {
  local dataset_name="$1"
  local max_samples="$2"
  local model_name="$3"
  local draft_name="$4"
  local mode_name="$5"
  local save_path="$6"
  local log_path="$7"
  shift 7

  echo "========================================================"
  echo "Running Benchmark: dataset=${dataset_name} max_samples=${max_samples} model=${model_name} draft=${draft_name} mode=${mode_name}"
  echo "========================================================"

  if [[ -f "${save_path}" ]]; then
    echo "Skipping existing run: ${save_path}"
    return
  fi

  "${PYTHON_BIN}" -m torch.distributed.run \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --master_port="${MASTER_PORT}" \
    benchmark.py \
    --dataset "${dataset_name}" \
    --max-samples "${max_samples}" \
    --model-name-or-path "${model_name}" \
    --draft-name-or-path "${draft_name}" \
    --draft-type "${DRAFT_TYPE}" \
    --save-path "${save_path}" \
    "${COMMON_BENCHMARK_ARGS[@]}" \
    "$@" \
    2>&1 | tee "${log_path}"
}

for task in "${TASKS[@]}"; do
  IFS=':' read -r dataset_name max_samples <<< "${task}"

  for pair in "${MODEL_DRAFT_PAIRS[@]}"; do
    IFS='|' read -r model_name draft_name <<< "${pair}"

    model_slug="$(slugify "${model_name}")"
    draft_slug="$(slugify "${draft_name}")"
    for temperature in "${TEMPERATURES[@]}"; do
      temperature_slug="$(slugify "${temperature}")"
      run_name="${dataset_name}__${model_slug}__${draft_slug}__temp${temperature_slug}"
      if [[ "${DRAFT_TYPE}" != "dflash" ]]; then
        run_name="${run_name}__${DRAFT_TYPE}"
      fi

      for mode in "${MODES[@]}"; do
        mode_args=()
        if [[ "${mode}" == "flash_attn" ]]; then
          mode_args+=(--flash-attn)
        fi

        run_benchmark \
          "${dataset_name}" \
          "${max_samples}" \
          "${model_name}" \
          "${draft_name}" \
          "${mode}" \
          "${RUN_DIR}/${run_name}__${mode}.pt" \
          "${LOG_DIR}/${run_name}__${mode}.log" \
          --temperature "${temperature}" \
          "${mode_args[@]}"
      done
    done
  done
done
