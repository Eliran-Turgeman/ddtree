#!/usr/bin/env bash

set -euo pipefail

TARGET="Qwen/Qwen3.8-27B"
TARGET_REVISION="1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
DRAFTER="incoai/Qwen3.8-27B-DFlash2"
DRAFTER_REVISION="dedf8df68adfb1afeaf7b7480c0a0243108177b4"
FULL_CONFIGS="dflash2_original_ddtree:7,16,32,64;dflash2_pairwise_k16:7,16,32,64;dflash2_unary_k16:32,64"
PROFILE="${1:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-artifacts/step8}"

if [[ -z "${PROFILE}" ]]; then
  echo "usage: $0 {one|matrix|domains|full}" >&2
  exit 2
fi
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "refusing to benchmark a dirty tracked worktree" >&2
  exit 1
fi
if [[ "${CUDA_VISIBLE_DEVICES:-0}" == *,* ]]; then
  echo "Step 8 requires exactly one visible GPU" >&2
  exit 1
fi

COMMIT="$(git rev-parse HEAD)"
OUTPUT_DIR="${OUTPUT_ROOT}/${COMMIT}/${PROFILE}"
mkdir -p "${OUTPUT_DIR}"

run_dataset() {
  local dataset="$1"
  local samples="$2"
  local max_new_tokens="$3"
  local configs="$4"
  local suffix="${5:-controlled}"
  local native_args=()
  if [[ "${suffix}" == "native" ]]; then
    native_args+=(--native-method-trajectories)
  fi

  python benchmark.py \
    --model-name-or-path "${TARGET}" \
    --model-revision "${TARGET_REVISION}" \
    --draft-name-or-path "${DRAFTER}" \
    --draft-revision "${DRAFTER_REVISION}" \
    --draft-type dflash2 \
    --dataset "${dataset}" \
    --max-samples "${samples}" \
    --max-new-tokens "${max_new_tokens}" \
    --temperature 0 \
    --dflash2-tree-configs "${configs}" \
    --collect-allocation-data \
    --save-path "${OUTPUT_DIR}/${dataset}_${suffix}.pt" \
    "${native_args[@]}"
}

case "${PROFILE}" in
  one)
    run_dataset \
      gsm8k 1 32 \
      "dflash2_original_ddtree:7;dflash2_pairwise_k16:7"
    ;;
  matrix)
    run_dataset gsm8k 2 32 "${FULL_CONFIGS}"
    ;;
  domains)
    run_dataset gsm8k 4 64 "${FULL_CONFIGS}"
    run_dataset math500 4 64 "${FULL_CONFIGS}"
    run_dataset humaneval 4 64 "${FULL_CONFIGS}"
    run_dataset mt-bench 4 64 "${FULL_CONFIGS}"
    run_dataset mt-bench 4 64 "${FULL_CONFIGS}" native
    ;;
  full)
    run_dataset gsm8k 128 256 "${FULL_CONFIGS}"
    run_dataset math500 128 256 "${FULL_CONFIGS}"
    run_dataset humaneval 164 256 "${FULL_CONFIGS}"
    run_dataset mt-bench 80 256 "${FULL_CONFIGS}"
    run_dataset mt-bench 80 256 "${FULL_CONFIGS}" native
    ;;
  *)
    echo "unknown profile: ${PROFILE}" >&2
    exit 2
    ;;
esac
