#!/usr/bin/env bash

set -euo pipefail

# Step 9.4b: pinned, dual-drafter, one-H100 throughput protocol.
#
# Research question: is DFlash2+Pairwise faster in tokens/s than (a) raw
# DFlash2, (b) DFlash2+original DDTree, (c) raw original DFlash, and (d)
# original DFlash+DDTree?
#
# This launches exactly two benchmark.py calls per dataset/profile, one per
# drafter family, so both families share the same commit/profile/dataset
# artifact directory and can be paired directly by the analyzer.
#
#   Original-DFlash family (raw dflash + original DDTree B16/B32/B64):
#     benchmark.py --draft-type dflash --tree-budget 16,32,64
#     (the target model defaults to SDPA when --flash-attn is omitted; this
#     single call produces method keys "dflash", "ddtree_tb16",
#     "ddtree_tb32", and "ddtree_tb64". The original drafter itself always
#     runs FlashAttention2 -- benchmark.py hardcodes this for --draft-type
#     dflash regardless of --flash-attn, so flash_attn must be installed.
#     Do not describe this family as "both drafts on SDPA": only the
#     target is SDPA here, the draft is FlashAttention2.)
#
#   DFlash2 family (raw dflash2 + original unary DDTree B7/16/32/64 +
#   Pairwise B7/16/32/64 + fixed Unary-K16 B32/64):
#     benchmark.py --draft-type dflash2 --dflash2-tree-configs "$DFLASH2_FULL_CONFIGS"
#     (method key "dflash2" is always included ahead of the requested tree
#     configs. Both the target and the DFlash2 drafter run SDPA here.)
#
# --collect-allocation-data is deliberately NEVER passed to any timed call
# below. It runs CPU-side lattice/tree bookkeeping copies inside the timed
# generation path, which would unfairly depress DFlash2/Pairwise tokens/s
# relative to the original-DFlash family that has no such instrumentation.
# Step 9 is throughput-only; allocation-trace analysis is already covered
# by Step 8. No stage in this script enables trace collection -- if a trace
# diagnostic is ever needed, run it as a separate, explicitly-labeled,
# non-reported invocation outside these artifacts, never mixed into a
# profile used for a throughput claim.

TARGET="Qwen/Qwen3-4B"
# Repository-known revision, frozen for Step 6.2 and Step 6.3
# (research_notes/dflash2_step62_cross_domain_validation.md,
# research_notes/dflash2_step63_wider_unary.md). Override only if the
# upstream repository moves this ref.
TARGET_REVISION="${TARGET_REVISION:-1cfa9a7208912126459214e8b04321603b3df60c}"

ORIGINAL_DRAFTER="z-lab/Qwen3-4B-DFlash-b16"
# Verified on the Hugging Face Hub for this protocol (no prior pin existed
# anywhere in this repository: README.md, run_benchmark.sh, and every prior
# run/trace artifact reference it unpinned). Override only if the upstream
# repository moves this ref.
ORIGINAL_DRAFTER_REVISION="${ORIGINAL_DRAFTER_REVISION:-b74e3a329c4d963783143b1e970d95b002be72bd}"

DFLASH2_DRAFTER="mgoin/Qwen3-4B-speculator.dflash2"
# Repository-known revision, frozen for Step 6.3
# (research_notes/dflash2_step63_wider_unary.md). Override only if the
# upstream repository moves this ref.
DFLASH2_DRAFTER_REVISION="${DFLASH2_DRAFTER_REVISION:-e3e7a18e4f541fa3841c2fb0666a7759079ab6fd}"

GSM8K_REVISION="740312add88f781978c0658806c59bc2815b9866"
MATH500_REVISION="6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be"
HUMANEVAL_REVISION="7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544"
MT_BENCH_REVISION="e3a795c5e9a82ee40611c416b8a7786c73198991"

ORIGINAL_FAMILY_TREE_BUDGET="16,32,64"
DFLASH2_FULL_CONFIGS="dflash2_original_ddtree:7,16,32,64;dflash2_pairwise_k16:7,16,32,64;dflash2_unary_k16:32,64"

# --timing-repetitions is a benchmark.py flag: it repeats each method's
# timing measurement N times per prompt/turn from the same input context
# (only the first repetition advances history and is compared against the
# baseline; every repetition's timing is preserved in the saved artifact).
# Stage defaults below are intentionally not analysis results: "stability"
# exists specifically to decide whether "full" needs more than one
# repetition per prompt before the frozen run is spent. If "stability"
# shows high per-prompt timing variance, override FULL_TIMING_REPETITIONS
# to 2 or 3 before launching "full"; do not silently keep the default.
FULL_TIMING_REPETITIONS="${FULL_TIMING_REPETITIONS:-1}"

PROFILE="${1:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-artifacts/step9_4b}"

if [[ -z "${PROFILE}" ]]; then
  echo "usage: $0 {one|matrix|domains|stability|full}" >&2
  exit 2
fi

if [[ -z "${ORIGINAL_DRAFTER_REVISION}" ]]; then
  echo "ORIGINAL_DRAFTER_REVISION is required and must not be empty" >&2
  exit 1
fi

if ! [[ "${FULL_TIMING_REPETITIONS}" =~ ^[0-9]+$ ]] || [[ "${FULL_TIMING_REPETITIONS}" -lt 1 ]]; then
  echo "FULL_TIMING_REPETITIONS must be a positive integer" >&2
  exit 1
fi

if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "refusing to benchmark a dirty tracked worktree" >&2
  exit 1
fi

if [[ "${CUDA_VISIBLE_DEVICES:-0}" == *,* ]]; then
  echo "Step 9.4b requires exactly one visible GPU" >&2
  exit 1
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  VISIBLE_GPU_COUNT="$(nvidia-smi -L | wc -l)"
  if [[ "${VISIBLE_GPU_COUNT}" -ne 1 ]]; then
    echo "Step 9.4b requires exactly one visible GPU (nvidia-smi -L reported ${VISIBLE_GPU_COUNT})" >&2
    exit 1
  fi
fi

COMMIT="$(git rev-parse HEAD)"
OUTPUT_DIR="${OUTPUT_ROOT}/${COMMIT}/${PROFILE}"
mkdir -p "${OUTPUT_DIR}"

dataset_revision() {
  case "$1" in
    gsm8k) echo "${GSM8K_REVISION}" ;;
    math500) echo "${MATH500_REVISION}" ;;
    humaneval) echo "${HUMANEVAL_REVISION}" ;;
    mt-bench) echo "${MT_BENCH_REVISION}" ;;
    *)
      echo "no frozen dataset revision for '$1'" >&2
      return 1
      ;;
  esac
}

run_original_family() {
  local dataset="$1"
  local samples="$2"
  local max_new_tokens="$3"
  local suffix="${4:-controlled}"
  local revision
  revision="$(dataset_revision "${dataset}")"
  local native_args=()
  if [[ "${suffix}" == "native" ]]; then
    native_args+=(--native-method-trajectories)
  fi

  python benchmark.py \
    --model-name-or-path "${TARGET}" \
    --model-revision "${TARGET_REVISION}" \
    --draft-name-or-path "${ORIGINAL_DRAFTER}" \
    --draft-revision "${ORIGINAL_DRAFTER_REVISION}" \
    --draft-type dflash \
    --tree-budget "${ORIGINAL_FAMILY_TREE_BUDGET}" \
    --dataset "${dataset}" \
    --dataset-revision "${revision}" \
    --max-samples "${samples}" \
    --max-new-tokens "${max_new_tokens}" \
    --temperature 0 \
    --timing-repetitions "${TIMING_REPETITIONS}" \
    --save-path "${OUTPUT_DIR}/${dataset}_original_${suffix}.pt" \
    "${native_args[@]}"
}

run_dflash2_family() {
  local dataset="$1"
  local samples="$2"
  local max_new_tokens="$3"
  local configs="$4"
  local suffix="${5:-controlled}"
  local revision
  revision="$(dataset_revision "${dataset}")"
  local native_args=()
  if [[ "${suffix}" == "native" ]]; then
    native_args+=(--native-method-trajectories)
  fi

  python benchmark.py \
    --model-name-or-path "${TARGET}" \
    --model-revision "${TARGET_REVISION}" \
    --draft-name-or-path "${DFLASH2_DRAFTER}" \
    --draft-revision "${DFLASH2_DRAFTER_REVISION}" \
    --draft-type dflash2 \
    --dataset "${dataset}" \
    --dataset-revision "${revision}" \
    --max-samples "${samples}" \
    --max-new-tokens "${max_new_tokens}" \
    --temperature 0 \
    --dflash2-tree-configs "${configs}" \
    --timing-repetitions "${TIMING_REPETITIONS}" \
    --save-path "${OUTPUT_DIR}/${dataset}_dflash2_${suffix}.pt" \
    "${native_args[@]}"
}

run_paired_dataset() {
  local dataset="$1"
  local samples="$2"
  local max_new_tokens="$3"
  local dflash2_configs="$4"
  local suffix="${5:-controlled}"
  # Cross-process family order: which benchmark.py process (original-family
  # or DFlash2-family) launches first for this dataset/suffix. Running the
  # same order every time confounds any tokens/s difference with whatever
  # systematic GPU-state drift (clock boost ramp, thermal creep, cache
  # warmth) correlates with launch position -- a real effect for a shared
  # H100 across long sequential runs. See "order" in the case block below
  # for the deterministic policy per profile.
  local order="${6:-original-first}"
  case "${order}" in
    original-first)
      run_original_family "${dataset}" "${samples}" "${max_new_tokens}" "${suffix}"
      run_dflash2_family "${dataset}" "${samples}" "${max_new_tokens}" "${dflash2_configs}" "${suffix}"
      ;;
    dflash2-first)
      run_dflash2_family "${dataset}" "${samples}" "${max_new_tokens}" "${dflash2_configs}" "${suffix}"
      run_original_family "${dataset}" "${samples}" "${max_new_tokens}" "${suffix}"
      ;;
    *)
      echo "run_paired_dataset: unknown order '${order}' (expected original-first|dflash2-first)" >&2
      exit 1
      ;;
  esac
}

case "${PROFILE}" in
  one)
    TIMING_REPETITIONS=1
    # One-prompt model-family smoke: confirm both drafter families load and
    # produce a matched artifact pair for the same single GSM8K prompt.
    # Order stays at the run_paired_dataset default (original-first): this
    # is a load/crash smoke check, not a timing result, so the cross-process
    # order bias this section's later stages guard against does not apply.
    run_paired_dataset \
      gsm8k 1 32 \
      "dflash2_original_ddtree:7;dflash2_pairwise_k16:7"
    ;;
  matrix)
    TIMING_REPETITIONS=1
    # Two-prompt full matrix: every method/budget in both families on GSM8K.
    # Order stays original-first (default): a correctness/coverage check,
    # not a reported timing result.
    run_paired_dataset gsm8k 2 32 "${DFLASH2_FULL_CONFIGS}"
    ;;
  domains)
    TIMING_REPETITIONS=1
    # Four-sample all domains: both families across every dataset, including
    # controlled-history and native-history MT-Bench. Order stays
    # original-first (default): a coverage/sign-stability check ahead of
    # the timing-focused "stability" and "full" stages, not itself a
    # reported timing result.
    run_paired_dataset gsm8k 4 64 "${DFLASH2_FULL_CONFIGS}"
    run_paired_dataset math500 4 64 "${DFLASH2_FULL_CONFIGS}"
    run_paired_dataset humaneval 4 64 "${DFLASH2_FULL_CONFIGS}"
    run_paired_dataset mt-bench 4 64 "${DFLASH2_FULL_CONFIGS}"
    run_paired_dataset mt-bench 4 64 "${DFLASH2_FULL_CONFIGS}" native
    ;;
  stability)
    TIMING_REPETITIONS=3
    # 16-prompt timing stability: repeat the GSM8K pair at a larger sample
    # count, with 3 timing repetitions per prompt, to check whether tokens/s
    # estimates are stable enough that "full" can run with a single
    # repetition. Inspect the per-repetition variance in the saved artifact
    # (not just the designated repetition) before deciding.
    #
    # Two full 16-prompt artifact pairs are launched back-to-back with
    # opposite process order and distinct "ab"/"ba" suffixes: "ab" runs
    # original-family first then DFlash2-family; "ba" runs DFlash2-family
    # first then original-family. analyze_step9_4b_throughput.py merges
    # multiple artifacts passed under the same --pair label, so pairing
    # "gsm8k_original_ab.pt:gsm8k_dflash2_ab.pt" and
    # "gsm8k_original_ba.pt:gsm8k_dflash2_ba.pt" under one label ("GSM8K")
    # combines both orders into a single stability read. If the AB and BA
    # artifacts disagree materially on tokens/s (beyond ordinary
    # per-repetition variance), that is itself evidence of a launch-order
    # confound to resolve before trusting the "full" stage's absolute
    # cross-family comparisons.
    run_paired_dataset gsm8k 16 256 "${DFLASH2_FULL_CONFIGS}" ab original-first
    run_paired_dataset gsm8k 16 256 "${DFLASH2_FULL_CONFIGS}" ba dflash2-first
    ;;
  full)
    TIMING_REPETITIONS="${FULL_TIMING_REPETITIONS}"
    # Deterministic order alternation by dataset position, so no single
    # family systematically launches first (and therefore systematically
    # benefits or suffers from GPU clock-ramp/thermal state) across the
    # whole "full" stage:
    #   1. gsm8k             -> original-first
    #   2. math500           -> dflash2-first
    #   3. humaneval         -> original-first
    #   4. mt-bench controlled -> dflash2-first
    #   5. mt-bench native     -> original-first
    # This also gives the two MT-Bench trajectory variants (controlled,
    # native) opposite orders from each other, per the same rationale.
    run_paired_dataset gsm8k 128 256 "${DFLASH2_FULL_CONFIGS}" controlled original-first
    run_paired_dataset math500 128 256 "${DFLASH2_FULL_CONFIGS}" controlled dflash2-first
    run_paired_dataset humaneval 164 256 "${DFLASH2_FULL_CONFIGS}" controlled original-first
    run_paired_dataset mt-bench 80 256 "${DFLASH2_FULL_CONFIGS}" controlled dflash2-first
    run_paired_dataset mt-bench 80 256 "${DFLASH2_FULL_CONFIGS}" native original-first
    ;;
  *)
    echo "unknown profile: ${PROFILE}" >&2
    exit 2
    ;;
esac
