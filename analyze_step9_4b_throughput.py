#!/usr/bin/env python3
"""Step 9.4b: cross-family Qwen3-4B throughput analyzer.

This analyzer safely combines two ``benchmark.py`` artifacts per dataset,
produced by ``run_step9_4b_throughput.sh``:

* the "original" DFlash family (``--draft-type dflash``), which yields the
  sequential baseline plus raw DFlash and original DDTree at B16/B32/B64,
  and
* the "DFlash2" family (``--draft-type dflash2``), which yields its own
  copy of the sequential baseline plus raw DFlash2, DFlash2 original-DDTree
  at B7/B16/B32/B64, Pairwise-K16 at B7/B16/B32/B64, and the fixed
  Unary-K16 diagnostic at B32/B64.

Both families share the same pinned target commit/revision, the same
target hardware, and the same dataset/prompt/decoding settings, so their
per-prompt outputs can be paired directly. The primary research question is
whether DFlash2 Pairwise-K16 is faster (tokens/s) than DFlash2
original-DDTree at the matched budget B16; every other comparison is
secondary and is explicitly labeled "controlled" (same drafter family) or
"cross_drafter" (different drafter checkpoints) so the two kinds of
evidence are never conflated.

A separate timing-runner effort wired ``end_to_end_timing_fields()`` (see
``dflash.py``) into every ``*_generate`` variant (``dflash.py``,
``ddtree.py``, ``dflash2.py``, ``dflash2_tree.py`` all now call it and
splat ``total_generation_time``/``tokens_per_second`` into their results),
but that integration may still evolve (new fields, renamed fields) and
older/partial artifacts predating it may still show up for smoke analysis.
All access to those fields goes through ``extract_timing_fields`` below:
the foundational fields that have always existed are required
unconditionally, and the end-to-end fields are required in strict mode but
only derived (and clearly flagged as derived) under ``--allow-partial``.
That fallback omits the first draft round, so it underestimates total time
and overestimates throughput; partial-mode values are diagnostic only.
Update that one function, not the call sites, if the schema changes again.

Every ``*_generate`` result also now carries repetition bookkeeping from
``benchmark.py``'s ``--timing-repetitions``/``attach_repetition_bookkeeping``
(a ``.repetitions`` list of every repeated measurement of that turn, plus
top-level ``.prompt_id``/``.prompt_hash``) and a top-level
``target_dtype``/``draft_dtype``/``target_attn_implementation``/
``draft_attn_implementation`` on the artifact. Aggregation therefore
happens in two explicit stages so repetitions and multi-turn conversations
(e.g. MT-Bench) are never treated as independent prompts: repeated
measurements of the *same* turn are averaged (``aggregate_repetitions``),
then turns belonging to the *same* conversation are combined by summing
tokens/time/target-calls and recomputing throughput from those sums
(``aggregate_turns_to_conversation``) -- never by arithmetic-averaging
each turn's own tokens/s.

Correctness is always recomputed by the analyzer directly from
``output_ids`` tensors (``exact_output_match``); the serialized
``.matches_baseline`` flag benchmark.py also writes is never read. A
mismatch between a speculative method and its own family's baseline is
recorded (never dropped) in ``correctness.csv``'s exact-match rate, but
never raises. Cross-family artifact pairing additionally requires
``validate_cross_family`` to confirm that *baseline* output is identical
between the two families for every paired response under controlled
trajectories (and for turn 0 only under native-trajectory MT-Bench, since
later native turns condition on each process's own prior-turn generation);
this is the one correctness check that fails strict pairing, because both
families run the identical target/prompt/settings and a divergent
baseline means the pairing itself is unsound.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Pinned Step 9.4b protocol constants (mirrors run_step9_4b_throughput.sh).
# ---------------------------------------------------------------------------

TARGET_MODEL = "Qwen/Qwen3-4B"
# Repository-known revision, frozen for Step 6.2/6.3 and reused by Step 9.4b.
TARGET_REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"

ORIGINAL_DRAFTER = "z-lab/Qwen3-4B-DFlash-b16"
# No revision for z-lab/Qwen3-4B-DFlash-b16 was ever pinned anywhere in this
# repository before Step 9.4b (README.md, run_benchmark.sh, and every prior
# run/trace artifact reference it unpinned). It was verified on the Hugging
# Face Hub for this protocol and is now pinned in run_step9_4b_throughput.sh
# as ORIGINAL_DRAFTER_REVISION (see research_notes/step9_4b_throughput_protocol.md).
ORIGINAL_DRAFTER_REVISION_DEFAULT: str | None = (
    "b74e3a329c4d963783143b1e970d95b002be72bd"
)

DFLASH2_DRAFTER = "mgoin/Qwen3-4B-speculator.dflash2"
# Repository-known revision, frozen for Step 6.3 and reused by Step 9.4b.
DFLASH2_DRAFTER_REVISION = "e3e7a18e4f541fa3841c2fb0666a7759079ab6fd"

FAMILY_ORIGINAL = "original"
FAMILY_DFLASH2 = "dflash2"

EXPECTED_SAMPLES = {
    "gsm8k": 128,
    "math500": 128,
    "humaneval": 164,
    "mt-bench": 80,
}
EXPECTED_DATASET_REVISIONS = {
    "gsm8k": "740312add88f781978c0658806c59bc2815b9866",
    "math500": "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be",
    "humaneval": "7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544",
    "mt-bench": "e3a795c5e9a82ee40611c416b8a7786c73198991",
}
FULL_TEMPERATURE = 0.0
FULL_MAX_NEW_TOKENS = 256

ORIGINAL_TREE_BUDGET_ARG = "16,32,64"
ORIGINAL_BUDGETS = (16, 32, 64)
DFLASH2_BUDGETS = (7, 16, 32, 64)
DFLASH2_TREE_CONFIGS_FULL = (
    "dflash2_original_ddtree:7,16,32,64;"
    "dflash2_pairwise_k16:7,16,32,64;"
    "dflash2_unary_k16:32,64"
)

ORIGINAL_METHODS = (
    "baseline",
    "dflash",
    "ddtree_tb16",
    "ddtree_tb32",
    "ddtree_tb64",
)
DFLASH2_METHODS = (
    "baseline",
    "dflash2",
    "dflash2_original_ddtree_tb7",
    "dflash2_original_ddtree_tb16",
    "dflash2_original_ddtree_tb32",
    "dflash2_original_ddtree_tb64",
    "dflash2_pairwise_k16_tb7",
    "dflash2_pairwise_k16_tb16",
    "dflash2_pairwise_k16_tb32",
    "dflash2_pairwise_k16_tb64",
    "dflash2_unary_k16_tb32",
    "dflash2_unary_k16_tb64",
)

RUNTIME_CRITICAL_KEYS = (
    "python",
    "pytorch",
    "transformers",
    "cuda",
    "gpu",
    "decode_timing_excludes_first_draft_prefill",
)

# Attention-implementation/dtype invariants (see benchmark.py): the target
# always runs SDPA. The original DFlash draft always runs FlashAttention-2
# by design (dflash.py's draft is always FlashAttention-2 regardless of
# --flash-attn, which only toggles the *target*'s implementation), while
# the DFlash2 draft always runs SDPA (DFlash2 benchmarking only supports
# SDPA mode). Both target and draft are always loaded in torch.bfloat16.
EXPECTED_TARGET_ATTN_IMPLEMENTATION = "sdpa"
EXPECTED_DRAFT_ATTN_IMPLEMENTATION = {
    FAMILY_ORIGINAL: "flash_attention_2",
    FAMILY_DFLASH2: "sdpa",
}
EXPECTED_DTYPE = "torch.bfloat16"

BOOTSTRAP_SEED = 20260906
DEFAULT_BOOTSTRAP_SAMPLES = 10000

PROMPT_ID_RE = re.compile(r":selected:(\d+):turn-(\d+):")

# Fields that every *_generate() variant has always returned. Missing any of
# these is a hard error, in every mode: they are not a "the timing-runner
# has not gotten to this generator yet" gap, they are a broken artifact.
REQUIRED_PRIMARY_TIMING_FIELDS = (
    "time_to_first_token",
    "time_per_output_token",
    "num_output_tokens",
    "decode_rounds",
    "stage_times",
)
# Fields added by dflash.end_to_end_timing_fields(). All four *_generate()
# variants (dflash.py, ddtree.py, dflash2.py, dflash2_tree.py) call it as of
# this writing, but the timing-runner integration may still evolve (new or
# renamed fields) and older/partial artifacts predating it may still need
# analysis. Never silently default these: see extract_timing_fields().
END_TO_END_TIMING_FIELDS = ("total_generation_time", "tokens_per_second")


def method_label(key: str) -> str:
    if key == "baseline":
        return "Sequential-Baseline"
    if key == "dflash":
        return "Raw-DFlash"
    if key == "dflash2":
        return "Raw-DFlash2"
    match = re.match(r"^ddtree_tb(\d+)$", key)
    if match:
        return f"DFlash-Original-DDTree-B{match.group(1)}"
    match = re.match(r"^dflash2_original_ddtree_tb(\d+)$", key)
    if match:
        return f"DFlash2-Original-DDTree-B{match.group(1)}"
    match = re.match(r"^dflash2_pairwise_k16_tb(\d+)$", key)
    if match:
        return f"DFlash2-Pairwise-K16-B{match.group(1)}"
    match = re.match(r"^dflash2_unary_k16_tb(\d+)$", key)
    if match:
        return f"DFlash2-Unary-K16-B{match.group(1)}"
    raise ValueError(f"unknown method key: {key}")


def method_family(key: str) -> str:
    return FAMILY_ORIGINAL if key in ORIGINAL_METHODS else FAMILY_DFLASH2


def method_budget(key: str) -> int | None:
    if key in ("baseline", "dflash", "dflash2"):
        return None
    return int(key.rsplit("_tb", maxsplit=1)[1])


# ---------------------------------------------------------------------------
# Artifact loading and validation.
# ---------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_artifact(path: Path) -> dict[str, Any]:
    run = torch.load(path, map_location="cpu", weights_only=False)
    required_keys = {
        "responses",
        "args",
        "repository",
        "target_revision",
        "draft_revision",
        "completed_dataset_indices",
        "runtime",
        "trajectory_mode",
        "target_attn_implementation",
        "draft_attn_implementation",
        "target_dtype",
        "draft_dtype",
    }
    missing = required_keys - run.keys()
    if missing:
        raise ValueError(f"{path} is missing required keys: {sorted(missing)}")
    if not run["responses"]:
        raise ValueError(f"{path} contains no responses")
    return run


def validate_artifact(
    run: dict[str, Any],
    path: Path,
    *,
    family: str,
    expected_commit: str,
    target_revision: str,
    draft_revision: str,
    allow_partial: bool,
) -> None:
    """Validate a single family artifact against the frozen Step 9.4b protocol.

    Identity checks (commit, model, revisions, draft type) are always
    enforced. Settings/completeness checks (decode settings, sample counts,
    method matrix, attention implementation) are enforced unless
    ``allow_partial`` is set, matching the existing analyze_step8_protocol.py
    convention for smoke/stability artifacts.
    """
    args = run["args"]
    if run["repository"] != {"commit": expected_commit, "dirty": False}:
        raise ValueError(f"{path} was not produced by clean {expected_commit}")
    if run["target_revision"] != target_revision:
        raise ValueError(
            f"{path} has unexpected target revision {run['target_revision']!r}"
        )
    if run["draft_revision"] != draft_revision:
        raise ValueError(
            f"{path} has unexpected draft revision {run['draft_revision']!r}"
        )
    if args["model_name_or_path"] != TARGET_MODEL:
        raise ValueError(f"{path} does not target {TARGET_MODEL}")

    expected_draft_type = "dflash" if family == FAMILY_ORIGINAL else "dflash2"
    if args["draft_type"] != expected_draft_type:
        raise ValueError(
            f"{path} has draft_type {args['draft_type']!r}, expected "
            f"{expected_draft_type!r} for the {family} family"
        )
    expected_draft_name = (
        ORIGINAL_DRAFTER if family == FAMILY_ORIGINAL else DFLASH2_DRAFTER
    )
    if args["draft_name_or_path"] != expected_draft_name:
        raise ValueError(
            f"{path} has draft_name_or_path {args['draft_name_or_path']!r}, "
            f"expected {expected_draft_name!r}"
        )

    dataset = args["dataset"]
    if dataset not in EXPECTED_SAMPLES:
        raise ValueError(f"{path} has unrecognized dataset {dataset!r}")
    expected_dataset_revision = EXPECTED_DATASET_REVISIONS[dataset]
    if args.get("dataset_revision") != expected_dataset_revision:
        raise ValueError(
            f"{path} dataset revision is {args.get('dataset_revision')!r}, "
            f"expected {expected_dataset_revision!r}"
        )

    if run["target_attn_implementation"] != EXPECTED_TARGET_ATTN_IMPLEMENTATION:
        raise ValueError(
            f"{path} target attention implementation is "
            f"{run['target_attn_implementation']!r}, expected "
            f"{EXPECTED_TARGET_ATTN_IMPLEMENTATION!r} (target always runs BF16/SDPA)"
        )
    expected_draft_attn = EXPECTED_DRAFT_ATTN_IMPLEMENTATION[family]
    if run["draft_attn_implementation"] != expected_draft_attn:
        raise ValueError(
            f"{path} draft attention implementation is "
            f"{run['draft_attn_implementation']!r}, expected "
            f"{expected_draft_attn!r} for the {family} family (the original "
            "DFlash draft always runs FlashAttention-2 by design; the "
            "DFlash2 draft always runs SDPA)"
        )
    if run["target_dtype"] != EXPECTED_DTYPE or run["draft_dtype"] != EXPECTED_DTYPE:
        raise ValueError(
            f"{path} does not load both target and draft in {EXPECTED_DTYPE} "
            f"(target={run['target_dtype']!r}, draft={run['draft_dtype']!r})"
        )

    if not allow_partial:
        if args["temperature"] != FULL_TEMPERATURE:
            raise ValueError(f"{path} does not use temperature={FULL_TEMPERATURE}")
        if args["max_new_tokens"] != FULL_MAX_NEW_TOKENS:
            raise ValueError(
                f"{path} does not use max_new_tokens={FULL_MAX_NEW_TOKENS}"
            )
        if (
            family == FAMILY_ORIGINAL
            and args.get("tree_budget") != ORIGINAL_TREE_BUDGET_ARG
        ):
            raise ValueError(f"{path} does not use the frozen original tree budgets")
        if (
            family == FAMILY_DFLASH2
            and args.get("dflash2_tree_configs") != DFLASH2_TREE_CONFIGS_FULL
        ):
            raise ValueError(f"{path} does not use the frozen DFlash2 method matrix")

    completed = run["completed_dataset_indices"]
    if len(completed) != args["max_samples"]:
        raise ValueError(f"{path} is incomplete: fewer samples than requested")
    if not allow_partial and len(completed) != EXPECTED_SAMPLES[dataset]:
        raise ValueError(
            f"{path} has {len(completed)} samples, expected "
            f"{EXPECTED_SAMPLES[dataset]} for {dataset}"
        )

    required_methods = set(
        ORIGINAL_METHODS if family == FAMILY_ORIGINAL else DFLASH2_METHODS
    )
    for position, response in enumerate(run["responses"]):
        missing_methods = required_methods - response.keys()
        if missing_methods and not allow_partial:
            raise ValueError(
                f"{path} response {position} misses methods {sorted(missing_methods)} "
                "(no missing repetitions allowed in strict mode)"
            )

    validate_repetitions(
        run, path, methods=required_methods, allow_partial=allow_partial
    )


def validate_repetitions(
    run: dict[str, Any],
    path: Path,
    *,
    methods: set[str],
    allow_partial: bool,
) -> None:
    """Validate benchmark.py's per-turn repetition bookkeeping.

    Every present method's result carries a ``.repetitions`` list (see
    ``attach_repetition_bookkeeping`` in benchmark.py -- even a single
    repetition still produces a one-element list containing itself). This
    checks that: every method has exactly ``timing_repetitions`` entries
    (repetition counts must match across methods within one artifact), and
    every repetition of the same turn shares the same ``prompt_hash`` (they
    are repeated measurements from the identical input context, so a
    mismatch means the rotation/bookkeeping is broken, not merely
    incomplete). This is a data-integrity check, not a sample-size
    completeness check, so it is not gated behind ``allow_partial`` except
    when the bookkeeping fields themselves are entirely absent (older,
    pre-repetition artifacts).
    """
    if "timing_repetitions" not in run:
        if allow_partial:
            return
        raise ValueError(f"{path} is missing the timing_repetitions field")
    expected_repetitions = int(run["timing_repetitions"])
    for position, response in enumerate(run["responses"]):
        for method in methods:
            result = response.get(method)
            if result is None:
                continue
            repetitions = getattr(result, "repetitions", None)
            if repetitions is None:
                if allow_partial:
                    continue
                raise ValueError(
                    f"{path} response {position} method {method} has no "
                    ".repetitions bookkeeping"
                )
            if len(repetitions) != expected_repetitions:
                raise ValueError(
                    f"{path} response {position} method {method} has "
                    f"{len(repetitions)} repetitions, expected "
                    f"{expected_repetitions} (timing_repetitions)"
                )
            hashes = {getattr(rep, "prompt_hash", None) for rep in repetitions}
            hashes.discard(None)
            if len(hashes) > 1:
                raise ValueError(
                    f"{path} response {position} method {method} has "
                    f"repetitions with mismatched prompt hashes: {sorted(hashes)}"
                )


def validate_cross_family(
    original_run: dict[str, Any],
    dflash2_run: dict[str, Any],
    *,
    dataset_label: str,
    allow_partial: bool,
) -> None:
    """Validate that a paired (original, dflash2) artifact pair is comparable."""
    original_indices = set(original_run["completed_dataset_indices"])
    dflash2_indices = set(dflash2_run["completed_dataset_indices"])
    if original_indices != dflash2_indices and not allow_partial:
        raise ValueError(
            f"{dataset_label}: original and dflash2 artifacts cover different "
            f"dataset indices ({sorted(original_indices)} vs "
            f"{sorted(dflash2_indices)})"
        )

    if original_run["trajectory_mode"] != dflash2_run["trajectory_mode"]:
        raise ValueError(
            f"{dataset_label}: trajectory_mode mismatch between families "
            f"({original_run['trajectory_mode']!r} vs "
            f"{dflash2_run['trajectory_mode']!r})"
        )

    if not allow_partial:
        original_reps = original_run.get("timing_repetitions")
        dflash2_reps = dflash2_run.get("timing_repetitions")
        if original_reps != dflash2_reps:
            raise ValueError(
                f"{dataset_label}: timing_repetitions differ between families "
                f"({original_reps!r} vs {dflash2_reps!r}); repeated timing "
                "measurements must be comparable across families"
            )

    if not allow_partial:
        original_runtime = {
            key: original_run["runtime"].get(key) for key in RUNTIME_CRITICAL_KEYS
        }
        dflash2_runtime = {
            key: dflash2_run["runtime"].get(key) for key in RUNTIME_CRITICAL_KEYS
        }
        if original_runtime != dflash2_runtime:
            raise ValueError(
                f"{dataset_label}: runtime-critical settings differ between "
                f"families ({original_runtime} vs {dflash2_runtime})"
            )

    turn0_original = _turn_zero_prompt_signatures(original_run)
    turn0_dflash2 = _turn_zero_prompt_signatures(dflash2_run)
    shared = sorted(set(turn0_original) & set(turn0_dflash2))
    if not shared:
        raise ValueError(f"{dataset_label}: no shared turn-0 prompts between families")
    mismatched = [idx for idx in shared if turn0_original[idx] != turn0_dflash2[idx]]
    if mismatched and not allow_partial:
        raise ValueError(
            f"{dataset_label}: prompt hashes diverge across families for "
            f"dataset indices {mismatched}; families must share the same "
            "target/hardware/prompts/settings"
        )

    # Both families run the identical target model against the identical
    # prompt/settings, so baseline (sequential, no speculation) must produce
    # byte-identical output_ids -- this is recomputed directly from the
    # tensors, never trusted from a serialized flag. Under controlled
    # trajectories every turn shares a fixed canonical conversation history,
    # so every turn is comparable. Under native trajectories, turn >=1's
    # history is each process's own prior-turn generation, which is not
    # guaranteed to be bit-identical across two independently launched
    # processes even with temp=0 -- only turn 0 (the fixed first user
    # message, no generation history involved) is guaranteed comparable.
    original_baselines = _baseline_by_cluster_turn(original_run)
    dflash2_baselines = _baseline_by_cluster_turn(dflash2_run)
    shared_turn_keys = sorted(set(original_baselines) & set(dflash2_baselines))
    if original_run["trajectory_mode"] == "native":
        shared_turn_keys = [key for key in shared_turn_keys if key[1] == 0]
    baseline_mismatches = [
        key
        for key in shared_turn_keys
        if not torch.equal(
            original_baselines[key].output_ids, dflash2_baselines[key].output_ids
        )
    ]
    if baseline_mismatches and not allow_partial:
        raise ValueError(
            f"{dataset_label}: baseline output diverges across families for "
            f"(dataset index, turn) pairs {baseline_mismatches}; both "
            "families run the identical target/prompt/settings, so baseline "
            "output must be identical for controlled trajectories (and for "
            "turn 0 of native-trajectory MT-Bench)"
        )


def _baseline_by_cluster_turn(run: dict[str, Any]) -> dict[tuple[int, int], Any]:
    """Map (dataset index, turn index within its conversation) -> baseline result.

    Turn index is 0-based and always 0 for single-turn datasets; MT-Bench's
    fixed two-turn conversations get indices 0 and 1. This is the shared
    basis for both turn-0 prompt-hash matching and full cross-family
    baseline-output identity checks.
    """
    cluster_ids = build_cluster_ids(run)
    turns_per_instance = _turns_per_instance(run)
    mapping: dict[tuple[int, int], Any] = {}
    for position, response in enumerate(run["responses"]):
        baseline = response.get("baseline")
        if baseline is None:
            continue
        turn_index = position % turns_per_instance if turns_per_instance else 0
        mapping[(cluster_ids[position], turn_index)] = baseline
    return mapping


def _turn_zero_prompt_signatures(run: dict[str, Any]) -> dict[int, str]:
    """Map dataset index -> sha256 of the turn-0 input token ids, via baseline.

    Baseline is present in both families and is the only method guaranteed
    to carry no method-specific prompt bookkeeping, so it is the most
    portable anchor for cross-family prompt identity. Turn 0 is unambiguous
    for every dataset (it is always the first response for its dataset
    index), so this does not depend on inferring turn boundaries.
    """
    return {
        cluster: prompt_signature(baseline)
        for (cluster, turn_index), baseline in _baseline_by_cluster_turn(run).items()
        if turn_index == 0
    }


def prompt_signature(result: Any) -> str:
    prefix = result.output_ids[0, : result.num_input_tokens].tolist()
    return hashlib.sha256(json.dumps(prefix).encode("utf-8")).hexdigest()


def _turns_per_instance(run: dict[str, Any]) -> int | None:
    responses = run["responses"]
    completed = run["completed_dataset_indices"]
    if not completed or len(responses) % len(completed) != 0:
        return None
    return len(responses) // len(completed)


def build_cluster_ids(run: dict[str, Any]) -> list[int]:
    """Map each response-list position to its dataset index ("prompt" cluster).

    Prefers the authoritative prompt_id embedded in round_metrics (available
    on every DFlash2-family method and on raw DFlash2) and cross-checks it
    against the position-based estimate. Falls back to position-based
    estimation alone when no method in a response carries round_metrics
    (raw DFlash / original DDTree), assuming a uniform number of turns per
    dataset instance -- true for every Step 9.4b dataset (single-turn
    GSM8K/MATH500/HumanEval, and fixed two-turn MT-Bench).
    """
    completed = list(run["completed_dataset_indices"])
    turns_per_instance = _turns_per_instance(run)
    cluster_ids = []
    for position, response in enumerate(run["responses"]):
        estimated = None
        if turns_per_instance is not None:
            estimated = completed[position // turns_per_instance]
        authoritative = _authoritative_cluster_id(response)
        if authoritative is not None:
            if estimated is not None and authoritative != estimated:
                raise ValueError(
                    f"response {position} prompt_id points at dataset index "
                    f"{authoritative}, but position-based estimation expected "
                    f"{estimated}"
                )
            cluster_ids.append(authoritative)
        elif estimated is not None:
            cluster_ids.append(estimated)
        else:
            raise ValueError(
                f"response {position} has no prompt_id and turn alignment "
                "could not be inferred from completed_dataset_indices"
            )
    return cluster_ids


def _authoritative_cluster_id(response: dict[str, Any]) -> int | None:
    for result in response.values():
        round_metrics = getattr(result, "round_metrics", None)
        if not round_metrics:
            continue
        match = PROMPT_ID_RE.search(round_metrics[0]["prompt_id"])
        if match is not None:
            return int(match.group(1))
    return None


# ---------------------------------------------------------------------------
# Field adapter: isolates the analyzer from the in-progress timing-runner
# integration. Update this function (and only this function) when every
# *_generate() variant has been migrated to end_to_end_timing_fields().
# ---------------------------------------------------------------------------


def extract_timing_fields(
    result: Any,
    *,
    path: Path,
    method: str,
    allow_partial: bool,
) -> dict[str, float | str]:
    missing_primary = [
        field for field in REQUIRED_PRIMARY_TIMING_FIELDS if not hasattr(result, field)
    ]
    if missing_primary:
        raise ValueError(
            f"{path} method {method} misses required timing fields {missing_primary}"
        )

    time_to_first_token = float(result.time_to_first_token)
    time_per_output_token = float(result.time_per_output_token)
    num_output_tokens = int(result.num_output_tokens)
    decode_only_time = time_per_output_token * max(num_output_tokens, 1)

    fields: dict[str, float | str] = {
        "time_to_first_token": time_to_first_token,
        "time_per_output_token": time_per_output_token,
        "num_output_tokens": num_output_tokens,
        "decode_rounds": int(result.decode_rounds),
        "decode_only_time": decode_only_time,
        "decode_only_tokens_per_second": (
            1.0 / time_per_output_token if time_per_output_token > 0 else 0.0
        ),
    }

    has_end_to_end = all(hasattr(result, field) for field in END_TO_END_TIMING_FIELDS)
    if has_end_to_end:
        fields["total_generation_time"] = float(result.total_generation_time)
        fields["tokens_per_second_end_to_end"] = float(result.tokens_per_second)
        fields["timing_source"] = "runner"
    elif allow_partial:
        derived_total = time_to_first_token + decode_only_time
        fields["total_generation_time"] = derived_total
        fields["tokens_per_second_end_to_end"] = (
            num_output_tokens / derived_total if derived_total > 0 else 0.0
        )
        fields["timing_source"] = "derived"
    else:
        raise ValueError(
            f"{path} method {method} misses end-to-end timing fields "
            f"{END_TO_END_TIMING_FIELDS} (the timing-runner integration has "
            "not reached this generator yet); rerun with --allow-partial to "
            "use a clearly-labeled derived fallback for smoke/stability "
            "artifacts"
        )
    return fields


def matched_and_committed(result: Any, method: str) -> tuple[float, float]:
    """Return (mean matched speculative tokens, mean committed tokens) per round.

    DFlash2-family tree/raw methods carry explicit per-round arrays. Raw
    DFlash and original DDTree only carry acceptance_lengths, where each
    entry already equals the tokens committed that round (accepted draft
    tokens plus the one bonus token); matched tokens is that value minus the
    bonus token, floored at zero, matching summarize_run.py's convention.
    """
    matched_per_round = getattr(result, "matched_draft_tokens_per_round", None)
    committed_per_round = getattr(result, "committed_tokens_per_round", None)
    if matched_per_round and committed_per_round:
        return (
            float(np.mean(matched_per_round)),
            float(np.mean(committed_per_round)),
        )
    acceptance_lengths = getattr(result, "acceptance_lengths", None)
    if not acceptance_lengths:
        return 0.0, 0.0
    committed = np.asarray(acceptance_lengths, dtype=float)
    matched = np.clip(committed - 1.0, a_min=0.0, a_max=None)
    return float(matched.mean()), float(committed.mean())


def is_truncated(result: Any, max_new_tokens: int) -> bool:
    return int(result.num_output_tokens) >= max_new_tokens


def exact_output_match(result: Any, baseline: Any) -> bool:
    """Recompute exact-output-match directly from token tensors.

    benchmark.py also serializes its own ``.matches_baseline`` flag onto
    results, but the analyzer must never read or trust it: correctness is
    always independently recomputed here via ``torch.equal`` on
    ``output_ids``, so a stale/incorrect serialized flag can never silently
    change what correctness.csv reports.
    """
    return bool(torch.equal(result.output_ids, baseline.output_ids))


def _is_nan(value: object) -> bool:
    return isinstance(value, float) and value != value


def _nanmean(values: list[float]) -> float:
    clean = [value for value in values if not _is_nan(value)]
    return float(np.mean(clean)) if clean else float("nan")


def _nanmax(values: list[float]) -> float:
    clean = [value for value in values if not _is_nan(value)]
    return float(max(clean)) if clean else float("nan")


def _weighted_mean(values: list[float], weights: list[float]) -> float:
    values_arr = np.asarray(values, dtype=float)
    weights_arr = np.asarray(weights, dtype=float)
    mask = ~np.isnan(values_arr)
    values_arr, weights_arr = values_arr[mask], weights_arr[mask]
    if values_arr.size == 0:
        return float("nan")
    if weights_arr.sum() <= 0:
        return float(np.mean(values_arr))
    return float(np.sum(values_arr * weights_arr) / np.sum(weights_arr))


# ---------------------------------------------------------------------------
# Per-turn / per-cluster ("prompt") metric collection.
#
# Aggregation happens in explicit stages so repetitions, MT-Bench turns, and
# repeated --pair artifacts are never treated as independent prompts:
#   1. collect_repetition_metrics(): one metrics dict per repetition of a
#      single turn's result (benchmark.py's --timing-repetitions /
#      .repetitions bookkeeping).
#   2. aggregate_repetitions(): averages those repeated measurements of the
#      *same* turn into one per-turn dict, recomputing throughput from the
#      averaged totals (never by averaging each repetition's own tokens/s).
#   3. aggregate_turns_to_conversation(): combines multiple turns of the
#      *same* conversation (e.g. MT-Bench's two turns) by summing tokens,
#      time, and target/decode calls -- turns are additional work, not
#      repeated measurements -- then recomputes throughput from those sums.
#   4. average_conversation_replicates() (used by merge_clustered): averages
#      whole-conversation measurements coming from repeated --pair artifacts
#      for the same dataset label, the same way step 2 averages
#      sub-turn repetitions.
# ---------------------------------------------------------------------------


def collect_repetition_metrics(
    result: Any,
    *,
    path: Path,
    method: str,
    allow_partial: bool,
    max_new_tokens: int,
    baseline: Any,
) -> list[dict[str, float]]:
    """Return one metrics dict per repetition of a single turn's result.

    Every result carries a ``.repetitions`` list (see
    ``attach_repetition_bookkeeping`` in benchmark.py); even a single
    repetition still produces a one-element list containing itself, so this
    is always safe to use once an artifact has been through that
    bookkeeping. Older artifacts lacking it fall back to treating the
    designated result as its own sole repetition, only under
    ``--allow-partial``.
    """
    repetitions = getattr(result, "repetitions", None)
    if not repetitions:
        if not allow_partial:
            raise ValueError(f"{path} method {method} has no .repetitions bookkeeping")
        repetitions = [result]
    baseline_repetitions = (
        (getattr(baseline, "repetitions", None) or [baseline])
        if baseline is not None
        else []
    )
    if (
        baseline is not None
        and method != "baseline"
        and len(baseline_repetitions) != len(repetitions)
        and not allow_partial
    ):
        raise ValueError(
            f"{path} method {method} has {len(repetitions)} repetitions but "
            f"baseline has {len(baseline_repetitions)}"
        )

    metrics = []
    for repetition_index, repetition in enumerate(repetitions):
        timing = extract_timing_fields(
            repetition, path=path, method=method, allow_partial=allow_partial
        )
        matched, committed = matched_and_committed(repetition, method)
        peak_allocated = getattr(repetition, "peak_allocated_gib", None)
        peak_reserved = getattr(repetition, "peak_reserved_gib", None)
        if (peak_allocated is None or peak_reserved is None) and not allow_partial:
            raise ValueError(
                f"{path} method {method} repetition {repetition_index} is "
                "missing per-method peak memory fields"
            )
        corresponding_baseline = (
            baseline_repetitions[repetition_index]
            if repetition_index < len(baseline_repetitions)
            else baseline
        )
        metrics.append(
            {
                **timing,
                "matched_tokens": matched,
                "committed_tokens": committed,
                "peak_allocated_gib": (
                    float(peak_allocated)
                    if peak_allocated is not None
                    else float("nan")
                ),
                "peak_reserved_gib": (
                    float(peak_reserved) if peak_reserved is not None else float("nan")
                ),
                "truncated": float(is_truncated(repetition, max_new_tokens)),
                "exact_match": (
                    float(exact_output_match(repetition, corresponding_baseline))
                    if corresponding_baseline is not None and method != "baseline"
                    else float("nan")
                ),
            }
        )
    return metrics


def _recompute_rates(totals: dict[str, float]) -> None:
    """Recompute rate fields from aggregated totals in-place.

    Rates must never be arithmetic-averaged across repetitions/turns: they
    are recomputed here from the (already averaged-or-summed, as
    appropriate) total tokens and total time.
    """
    output_tokens = totals["num_output_tokens"]
    decode_only_time = totals["decode_only_time"]
    total_generation_time = totals["total_generation_time"]
    totals["time_per_output_token"] = (
        decode_only_time / output_tokens if output_tokens > 0 else float("nan")
    )
    totals["decode_only_tokens_per_second"] = (
        output_tokens / decode_only_time if decode_only_time > 0 else 0.0
    )
    totals["tokens_per_second_end_to_end"] = (
        output_tokens / total_generation_time if total_generation_time > 0 else 0.0
    )


def _combine_measurements(
    entries: list[dict[str, float]],
    *,
    totals_reducer: Any,
    timing_source_is_raw: bool,
) -> dict[str, float]:
    """Shared combination logic for both the "average repeated measurements
    of the same work" case (repetitions, replicate conversations) and the
    "sum additional work" case (turns within a conversation).

    ``totals_reducer`` is ``np.mean`` for averaging or ``np.sum`` for
    summing, applied to num_output_tokens/decode_rounds/decode_only_time/
    total_generation_time. time_to_first_token (a latency, not a rate) is
    always averaged. matched/committed tokens per round are always combined
    weighted by decode_rounds, so turns/repetitions with more rounds
    contribute proportionally rather than being weighted equally with
    short ones.
    """
    decode_rounds_values = [entry["decode_rounds"] for entry in entries]
    aggregated: dict[str, float] = {
        "time_to_first_token": float(
            np.mean([entry["time_to_first_token"] for entry in entries])
        ),
        "num_output_tokens": float(
            totals_reducer([entry["num_output_tokens"] for entry in entries])
        ),
        "decode_rounds": float(totals_reducer(decode_rounds_values)),
        "decode_only_time": float(
            totals_reducer([entry["decode_only_time"] for entry in entries])
        ),
        "total_generation_time": float(
            totals_reducer([entry["total_generation_time"] for entry in entries])
        ),
        "matched_tokens": _weighted_mean(
            [entry["matched_tokens"] for entry in entries], decode_rounds_values
        ),
        "committed_tokens": _weighted_mean(
            [entry["committed_tokens"] for entry in entries], decode_rounds_values
        ),
        "truncated": float(np.mean([entry["truncated"] for entry in entries])),
        "exact_match": _nanmean([entry["exact_match"] for entry in entries]),
        "peak_allocated_gib": _nanmax(
            [entry.get("peak_allocated_gib", float("nan")) for entry in entries]
        ),
        "peak_reserved_gib": _nanmax(
            [entry.get("peak_reserved_gib", float("nan")) for entry in entries]
        ),
    }
    if timing_source_is_raw:
        aggregated["timing_source_is_runner"] = float(
            np.mean([entry["timing_source"] == "runner" for entry in entries])
        )
    else:
        aggregated["timing_source_is_runner"] = float(
            np.mean([entry["timing_source_is_runner"] for entry in entries])
        )
    _recompute_rates(aggregated)
    return aggregated


def aggregate_repetitions(entries: list[dict[str, float]]) -> dict[str, float]:
    """Average repeated timing measurements of the *same* turn (identical
    input/context -- see benchmark.py's --timing-repetitions), then
    recompute throughput from those averaged totals.
    """
    return _combine_measurements(
        entries, totals_reducer=np.mean, timing_source_is_raw=True
    )


def aggregate_turns_to_conversation(
    entries: list[dict[str, float]],
) -> dict[str, float]:
    """Combine multiple turns of the same conversation (e.g. MT-Bench's two
    turns) by summing tokens/time/target-calls -- turns are additional work,
    not repeated measurements of the same work -- then recompute throughput
    from those summed totals. Never arithmetic-mean each turn's own
    tokens/s.
    """
    return _combine_measurements(
        entries, totals_reducer=np.sum, timing_source_is_raw=False
    )


def average_conversation_replicates(
    entries: list[dict[str, float]],
) -> dict[str, float]:
    """Average repeated whole-conversation measurements coming from repeated
    --pair artifacts for the same dataset label. These are replicate
    measurements of the same conversation, not additional turns, so they
    are combined the same way aggregate_repetitions combines sub-turn
    repetitions.
    """
    return _combine_measurements(
        entries, totals_reducer=np.mean, timing_source_is_raw=False
    )


def collect_response_metrics(
    run: dict[str, Any],
    *,
    path: Path,
    methods: tuple[str, ...],
    allow_partial: bool,
) -> dict[str, dict[int, dict[str, float]]]:
    """Collect one conversation-level metrics dict per method per prompt
    cluster for a single artifact.

    Every repetition of a turn is aggregated (mean) before turns belonging
    to the same conversation are combined (sum) -- see the module-level
    aggregation stages above. Callers merging multiple artifacts for the
    same dataset label (repeated full runs) must further combine with
    ``merge_clustered``/``average_conversation_replicates``, not by feeding
    raw entries back into this function.
    """
    cluster_ids = build_cluster_ids(run)
    turns_per_instance = _turns_per_instance(run)
    max_new_tokens = int(run["args"]["max_new_tokens"])
    turns_by_cluster: dict[str, dict[int, list[dict[str, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for position, response in enumerate(run["responses"]):
        cluster = cluster_ids[position]
        turn_index = (
            position % turns_per_instance if turns_per_instance is not None else 0
        )
        baseline = (
            None
            if run.get("trajectory_mode", "controlled_shared") == "native"
            and turn_index > 0
            else response.get("baseline")
        )
        for method in methods:
            result = response.get(method)
            if result is None:
                continue
            repetition_metrics = collect_repetition_metrics(
                result,
                path=path,
                method=method,
                allow_partial=allow_partial,
                max_new_tokens=max_new_tokens,
                baseline=baseline,
            )
            turns_by_cluster[method][cluster].append(
                aggregate_repetitions(repetition_metrics)
            )
    return {
        method: {
            cluster: aggregate_turns_to_conversation(turns)
            for cluster, turns in clusters.items()
        }
        for method, clusters in turns_by_cluster.items()
    }


def merge_clustered(
    *sources: dict[str, dict[int, dict[str, float]]],
) -> dict[str, dict[int, dict[str, float]]]:
    """Merge conversation-level metrics from possibly-repeated --pair
    artifacts for the same dataset label.

    These are replicate measurements of the same conversation, not
    additional turns, so they are averaged (not summed) via
    ``average_conversation_replicates``.
    """
    merged_raw: dict[str, dict[int, list[dict[str, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for source in sources:
        for method, clusters in source.items():
            for cluster, metrics in clusters.items():
                merged_raw[method][cluster].append(metrics)
    return {
        method: {
            cluster: average_conversation_replicates(entries)
            for cluster, entries in clusters.items()
        }
        for method, clusters in merged_raw.items()
    }


# ---------------------------------------------------------------------------
# Bootstrap.
# ---------------------------------------------------------------------------


def bootstrap_interval(
    values: np.ndarray, samples: int, seed: int
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    estimates = values[indices].mean(axis=1)
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high)


# ---------------------------------------------------------------------------
# Comparison matrix.
# ---------------------------------------------------------------------------


def comparison_pairs() -> list[tuple[str, str, bool, str]]:
    """(left, right, is_primary, comparison_type) tuples.

    comparison_type is "controlled" when both methods share the same
    drafter family (only the tree-construction method/budget differs), and
    "cross_drafter" when the compared methods come from different drafter
    checkpoints -- these must never be conflated.
    """
    pairs: list[tuple[str, str, bool, str]] = [
        # Primary endpoint.
        (
            "dflash2_pairwise_k16_tb16",
            "dflash2_original_ddtree_tb16",
            True,
            "controlled",
        ),
    ]
    # Secondary, controlled (same DFlash2 drafter family).
    for budget in (7, 32, 64):
        pairs.append(
            (
                f"dflash2_pairwise_k16_tb{budget}",
                f"dflash2_original_ddtree_tb{budget}",
                False,
                "controlled",
            )
        )
    pairs.append(("dflash2_pairwise_k16_tb7", "dflash2", False, "controlled"))
    for budget in (32, 64):
        pairs.append(
            (
                f"dflash2_pairwise_k16_tb{budget}",
                f"dflash2_unary_k16_tb{budget}",
                False,
                "controlled",
            )
        )
    # Secondary, cross-drafter (different drafter checkpoint entirely).
    for budget in (16, 32, 64):
        pairs.append(
            (f"dflash2_pairwise_k16_tb{budget}", "dflash", False, "cross_drafter")
        )
        pairs.append(
            (
                f"dflash2_pairwise_k16_tb{budget}",
                f"ddtree_tb{budget}",
                False,
                "cross_drafter",
            )
        )
    return pairs


def compare_methods(
    dataset_label: str,
    clustered: dict[str, dict[int, dict[str, float]]],
    left: str,
    right: str,
    *,
    is_primary: bool,
    comparison_type: str,
    samples: int,
    seed: int,
    left_baseline: dict[int, dict[str, float]] | None = None,
    right_baseline: dict[int, dict[str, float]] | None = None,
) -> dict[str, object] | None:
    if left not in clustered or right not in clustered:
        return None
    shared = sorted(set(clustered[left]) & set(clustered[right]))
    if not shared:
        return None
    left_throughput = np.asarray(
        [clustered[left][cluster]["tokens_per_second_end_to_end"] for cluster in shared]
    )
    right_throughput = np.asarray(
        [
            clustered[right][cluster]["tokens_per_second_end_to_end"]
            for cluster in shared
        ]
    )
    diff = left_throughput - right_throughput
    ci_low, ci_high = bootstrap_interval(diff, samples, seed)
    relative_pct = np.asarray(
        [
            100.0 * (left_v - right_v) / right_v if right_v != 0 else float("nan")
            for left_v, right_v in zip(left_throughput, right_throughput)
        ]
    )
    normalized_diff = np.asarray([], dtype=float)
    normalized_ci_low = float("nan")
    normalized_ci_high = float("nan")
    if left_baseline is not None and right_baseline is not None:
        normalized_shared = [
            cluster
            for cluster in shared
            if cluster in left_baseline
            and cluster in right_baseline
            and left_baseline[cluster]["tokens_per_second_end_to_end"] > 0
            and right_baseline[cluster]["tokens_per_second_end_to_end"] > 0
        ]
        normalized_diff = np.asarray(
            [
                clustered[left][cluster]["tokens_per_second_end_to_end"]
                / left_baseline[cluster]["tokens_per_second_end_to_end"]
                - clustered[right][cluster]["tokens_per_second_end_to_end"]
                / right_baseline[cluster]["tokens_per_second_end_to_end"]
                for cluster in normalized_shared
            ]
        )
        if normalized_diff.size:
            normalized_ci_low, normalized_ci_high = bootstrap_interval(
                normalized_diff, samples, seed + 1
            )
    return {
        "dataset": dataset_label,
        "is_primary": is_primary,
        "comparison_type": comparison_type,
        "left_method": method_label(left),
        "right_method": method_label(right),
        "left_budget": method_budget(left),
        "right_budget": method_budget(right),
        "prompts": len(shared),
        "mean_tokens_per_second_diff": float(diff.mean()),
        "tokens_per_second_diff_ci_low": ci_low,
        "tokens_per_second_diff_ci_high": ci_high,
        "mean_relative_pct_diff": float(np.nanmean(relative_pct)),
        "mean_baseline_normalized_speedup_diff": (
            float(normalized_diff.mean()) if normalized_diff.size else float("nan")
        ),
        "baseline_normalized_speedup_diff_ci_low": normalized_ci_low,
        "baseline_normalized_speedup_diff_ci_high": normalized_ci_high,
        "improve_prompts": int((diff > 0).sum()),
        "tie_prompts": int((diff == 0).sum()),
        "hurt_prompts": int((diff < 0).sum()),
    }


def build_comparison_rows(
    dataset_label: str,
    clustered: dict[str, dict[int, dict[str, float]]],
    *,
    samples: int,
    dataset_offset: int,
    original_baseline: dict[int, dict[str, float]] | None = None,
    dflash2_baseline: dict[int, dict[str, float]] | None = None,
) -> list[dict[str, object]]:
    rows = []
    for index, (left, right, is_primary, comparison_type) in enumerate(
        comparison_pairs()
    ):
        row = compare_methods(
            dataset_label,
            clustered,
            left,
            right,
            is_primary=is_primary,
            comparison_type=comparison_type,
            samples=samples,
            seed=BOOTSTRAP_SEED + dataset_offset * 1000 + index * 2,
            left_baseline=(
                dflash2_baseline if comparison_type == "cross_drafter" else None
            ),
            right_baseline=(
                original_baseline if comparison_type == "cross_drafter" else None
            ),
        )
        if row is not None:
            rows.append(row)
    return rows


def build_family_reference_rows(
    dataset_label: str,
    clustered: dict[str, dict[int, dict[str, float]]],
    methods: tuple[str, ...],
    *,
    samples: int,
    seed_offset: int,
) -> list[dict[str, object]]:
    rows = []
    for index, method in enumerate(
        method for method in methods if method != "baseline"
    ):
        row = compare_methods(
            dataset_label,
            clustered,
            method,
            "baseline",
            is_primary=False,
            comparison_type="within_family_vs_sequential",
            samples=samples,
            seed=BOOTSTRAP_SEED + seed_offset + index * 2,
        )
        if row is not None:
            rows.append(row)
    for index, budget in enumerate((16, 32, 64)):
        row = compare_methods(
            dataset_label,
            clustered,
            f"ddtree_tb{budget}",
            "dflash",
            is_primary=False,
            comparison_type="controlled",
            samples=samples,
            seed=BOOTSTRAP_SEED + seed_offset + 100 + index * 2,
        )
        if row is not None:
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Timing decomposition.
# ---------------------------------------------------------------------------


def stage_ms_per_call(
    entry_stage_times: dict[str, float], decode_rounds: float
) -> dict[str, float]:
    if decode_rounds <= 0:
        return {stage: 0.0 for stage in entry_stage_times}
    return {
        stage: 1000.0 * total / decode_rounds
        for stage, total in entry_stage_times.items()
    }


# Stages that only exist for tree-construction methods (DFlash2's
# candidate_select/tree_build/tree_compile, or original DDTree's
# tree_build/tree_compile -- see dflash2_tree.py / ddtree.py). The "draft"
# stage delta is deliberately excluded and reported separately: it is the
# forward-pass cost of the draft model itself, not tree-construction
# overhead.
TREE_OVERHEAD_STAGES = ("candidate_select", "tree_build", "tree_compile")


def build_timing_decomposition_rows(
    dataset_label: str,
    clustered: dict[str, dict[int, dict[str, float]]],
    raw_stage_times: dict[str, dict[int, list[dict[str, float]]]],
) -> list[dict[str, object]]:
    rows = []
    for left, right, is_primary, comparison_type in comparison_pairs():
        if left not in clustered or right not in clustered:
            continue
        shared = sorted(set(clustered[left]) & set(clustered[right]))
        if not shared:
            continue
        left_metrics = clustered[left]
        right_metrics = clustered[right]
        matched_gain = np.mean(
            [
                left_metrics[c]["matched_tokens"] - right_metrics[c]["matched_tokens"]
                for c in shared
            ]
        )
        committed_gain = np.mean(
            [
                left_metrics[c]["committed_tokens"]
                - right_metrics[c]["committed_tokens"]
                for c in shared
            ]
        )
        calls_left = np.mean([left_metrics[c]["decode_rounds"] for c in shared])
        calls_right = np.mean([right_metrics[c]["decode_rounds"] for c in shared])
        throughput_gain = np.mean(
            [
                left_metrics[c]["tokens_per_second_end_to_end"]
                - right_metrics[c]["tokens_per_second_end_to_end"]
                for c in shared
            ]
        )
        left_stage = _mean_stage_ms_per_call(raw_stage_times.get(left, {}), shared)
        right_stage = _mean_stage_ms_per_call(raw_stage_times.get(right, {}), shared)
        draft_stage_change = left_stage.get("draft", 0.0) - right_stage.get(
            "draft", 0.0
        )
        tree_overhead_ms_per_call = sum(
            left_stage.get(stage, 0.0) - right_stage.get(stage, 0.0)
            for stage in TREE_OVERHEAD_STAGES
        )
        rows.append(
            {
                "dataset": dataset_label,
                "is_primary": is_primary,
                "comparison_type": comparison_type,
                "left_method": method_label(left),
                "right_method": method_label(right),
                "prompts": len(shared),
                "matched_tokens_gain": float(matched_gain),
                "committed_tokens_gain": float(committed_gain),
                "target_calls_avoided": float(calls_right - calls_left),
                "draft_stage_ms_per_call_left": left_stage.get("draft", 0.0),
                "draft_stage_ms_per_call_right": right_stage.get("draft", 0.0),
                "draft_stage_ms_per_call_change": draft_stage_change,
                "tree_overhead_ms_per_call": tree_overhead_ms_per_call,
                "verify_stage_ms_per_call_left": left_stage.get("verify", 0.0),
                "verify_stage_ms_per_call_right": right_stage.get("verify", 0.0),
                "verify_stage_ms_per_call_change": (
                    left_stage.get("verify", 0.0) - right_stage.get("verify", 0.0)
                ),
                "net_tokens_per_second_gain": float(throughput_gain),
            }
        )
    return rows


def _mean_stage_ms_per_call(
    clusters: dict[int, list[dict[str, float]]], shared: list[int]
) -> dict[str, float]:
    per_stage: dict[str, list[float]] = defaultdict(list)
    for cluster in shared:
        for entry in clusters.get(cluster, []):
            for stage, ms_value in entry.items():
                per_stage[stage].append(ms_value)
    return {
        stage: float(np.mean(values)) for stage, values in per_stage.items() if values
    }


def _mean_stage_dict(stage_dicts: list[dict[str, float]]) -> dict[str, float]:
    per_stage: dict[str, list[float]] = defaultdict(list)
    for stage_dict in stage_dicts:
        for stage, value in stage_dict.items():
            per_stage[stage].append(value)
    return {stage: float(np.mean(values)) for stage, values in per_stage.items()}


def collect_stage_times(
    run: dict[str, Any], cluster_ids: list[int], methods: tuple[str, ...]
) -> dict[str, dict[int, list[dict[str, float]]]]:
    """Collect one (repetition-averaged) stage-ms-per-call dict per turn,
    keyed by prompt cluster.

    Every repetition of a turn is included: each repetition's own
    stage_times/decode_rounds are converted to ms-per-call, then averaged
    across repetitions of that turn (the same "average repeated
    measurements of the same work" principle as aggregate_repetitions),
    rather than reading only the designated repetition's stage_times.
    """
    per_method: dict[str, dict[int, list[dict[str, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for position, response in enumerate(run["responses"]):
        cluster = cluster_ids[position]
        for method in methods:
            result = response.get(method)
            if result is None:
                continue
            repetitions = getattr(result, "repetitions", None) or [result]
            per_repetition_stage_ms = [
                stage_ms_per_call(
                    dict(repetition.stage_times),
                    float(getattr(repetition, "decode_rounds", 0)),
                )
                for repetition in repetitions
            ]
            per_method[method][cluster].append(
                _mean_stage_dict(per_repetition_stage_ms)
            )
    return per_method


# ---------------------------------------------------------------------------
# Method metrics / correctness rollups.
# ---------------------------------------------------------------------------


def build_method_metrics_rows(
    dataset_label: str,
    family: str,
    run: dict[str, Any],
    clustered: dict[str, dict[int, dict[str, float]]],
    methods: tuple[str, ...],
) -> list[dict[str, object]]:
    rows = []
    baseline_clusters = clustered.get("baseline", {})
    for method in methods:
        if method not in clustered:
            continue
        clusters = clustered[method]
        values = list(clusters.values())
        shared_with_baseline = sorted(set(clusters) & set(baseline_clusters))
        speedup_end_to_end = float("nan")
        speedup_decode_only = float("nan")
        if shared_with_baseline and method != "baseline":
            speedup_end_to_end = float(
                np.mean(
                    [
                        clusters[c]["tokens_per_second_end_to_end"]
                        / baseline_clusters[c]["tokens_per_second_end_to_end"]
                        for c in shared_with_baseline
                        if baseline_clusters[c]["tokens_per_second_end_to_end"] > 0
                    ]
                )
            )
            speedup_decode_only = float(
                np.mean(
                    [
                        clusters[c]["decode_only_tokens_per_second"]
                        / baseline_clusters[c]["decode_only_tokens_per_second"]
                        for c in shared_with_baseline
                        if baseline_clusters[c]["decode_only_tokens_per_second"] > 0
                    ]
                )
            )
        rows.append(
            {
                "dataset": dataset_label,
                "family": family,
                "method": method_label(method),
                "method_key": method,
                "budget": method_budget(method),
                "prompts": len(values),
                "mean_end_to_end_tokens_per_second": np.mean(
                    [v["tokens_per_second_end_to_end"] for v in values]
                ),
                "mean_decode_only_tokens_per_second": np.mean(
                    [v["decode_only_tokens_per_second"] for v in values]
                ),
                "speedup_vs_baseline_end_to_end": speedup_end_to_end,
                "speedup_vs_baseline_decode_only": speedup_decode_only,
                "mean_matched_tokens_per_round": np.mean(
                    [v["matched_tokens"] for v in values]
                ),
                "mean_committed_tokens_per_round": np.mean(
                    [v["committed_tokens"] for v in values]
                ),
                "mean_target_calls": np.mean([v["decode_rounds"] for v in values]),
                "mean_ttft_ms": 1000.0
                * np.mean([v["time_to_first_token"] for v in values]),
                "mean_total_generation_time_s": np.mean(
                    [v["total_generation_time"] for v in values]
                ),
                "peak_allocated_gib": _nanmax(
                    [v["peak_allocated_gib"] for v in values]
                ),
                "peak_reserved_gib": _nanmax([v["peak_reserved_gib"] for v in values]),
                "timing_source_runner_fraction": np.mean(
                    [v["timing_source_is_runner"] for v in values]
                ),
            }
        )
    return rows


def build_correctness_rows(
    dataset_label: str,
    family: str,
    clustered: dict[str, dict[int, dict[str, float]]],
    methods: tuple[str, ...],
) -> list[dict[str, object]]:
    rows = []
    for method in methods:
        if method not in clustered:
            continue
        values = list(clustered[method].values())
        exact_values = [
            v["exact_match"] for v in values if not _is_nan(v["exact_match"])
        ]
        rows.append(
            {
                "dataset": dataset_label,
                "family": family,
                "method": method_label(method),
                "method_key": method,
                "prompts": len(values),
                "exact_output_rate": (
                    float(np.mean(exact_values)) if exact_values else float("nan")
                ),
                "truncated_rate": float(np.mean([v["truncated"] for v in values])),
                "stopped_rate": float(1.0 - np.mean([v["truncated"] for v in values])),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# CSV / JSON output.
# ---------------------------------------------------------------------------


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# CLI / orchestration.
# ---------------------------------------------------------------------------


def parse_pair(value: str) -> tuple[str, Path, Path]:
    try:
        label, paths = value.split("=", maxsplit=1)
        original_path, dflash2_path = paths.split(":", maxsplit=1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--pair must be DATASET_LABEL=ORIGINAL_PATH:DFLASH2_PATH"
        ) from exc
    return label, Path(original_path), Path(dflash2_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze the frozen Step 9.4b cross-family (raw DFlash / original "
            "DDTree vs raw DFlash2 / DFlash2 tree methods) Qwen3-4B "
            "throughput protocol."
        )
    )
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--target-revision", default=TARGET_REVISION)
    parser.add_argument(
        "--original-draft-revision",
        default=ORIGINAL_DRAFTER_REVISION_DEFAULT,
        required=ORIGINAL_DRAFTER_REVISION_DEFAULT is None,
        help=(
            "Pinned in run_step9_4b_throughput.sh as ORIGINAL_DRAFTER_REVISION; "
            "override only if that ref moves."
        ),
    )
    parser.add_argument("--dflash2-draft-revision", default=DFLASH2_DRAFTER_REVISION)
    parser.add_argument(
        "--pair",
        action="append",
        required=True,
        type=parse_pair,
        metavar="DATASET_LABEL=ORIGINAL_PATH:DFLASH2_PATH",
        help=(
            "May be repeated for the same DATASET_LABEL to merge repeated "
            "runs/artifacts of the same dataset (aggregated within prompt "
            "clusters, never treated as independent prompts)."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--bootstrap-samples", type=int, default=DEFAULT_BOOTSTRAP_SAMPLES
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Allow smoke/stability artifacts with fewer than the frozen sample count.",
    )
    return parser.parse_args()


def run_analysis(args: argparse.Namespace) -> None:
    pairs_by_label: dict[str, list[tuple[Path, Path]]] = defaultdict(list)
    for label, original_path, dflash2_path in args.pair:
        pairs_by_label[label].append((original_path, dflash2_path))

    method_metrics_rows: list[dict[str, object]] = []
    comparison_rows: list[dict[str, object]] = []
    timing_rows: list[dict[str, object]] = []
    correctness_rows: list[dict[str, object]] = []
    provenance_pairs: list[dict[str, object]] = []
    timing_source_counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {"runner": 0, "derived": 0}
    )

    for dataset_offset, (label, artifact_pairs) in enumerate(
        sorted(pairs_by_label.items())
    ):
        original_clustered_parts = []
        dflash2_clustered_parts = []
        original_stage_parts = []
        dflash2_stage_parts = []
        for original_path, dflash2_path in artifact_pairs:
            original_run = load_artifact(original_path)
            dflash2_run = load_artifact(dflash2_path)
            validate_artifact(
                original_run,
                original_path,
                family=FAMILY_ORIGINAL,
                expected_commit=args.expected_commit,
                target_revision=args.target_revision,
                draft_revision=args.original_draft_revision,
                allow_partial=args.allow_partial,
            )
            validate_artifact(
                dflash2_run,
                dflash2_path,
                family=FAMILY_DFLASH2,
                expected_commit=args.expected_commit,
                target_revision=args.target_revision,
                draft_revision=args.dflash2_draft_revision,
                allow_partial=args.allow_partial,
            )
            validate_cross_family(
                original_run,
                dflash2_run,
                dataset_label=label,
                allow_partial=args.allow_partial,
            )

            original_metrics = collect_response_metrics(
                original_run,
                path=original_path,
                methods=ORIGINAL_METHODS,
                allow_partial=args.allow_partial,
            )
            dflash2_metrics = collect_response_metrics(
                dflash2_run,
                path=dflash2_path,
                methods=DFLASH2_METHODS,
                allow_partial=args.allow_partial,
            )
            original_clustered_parts.append(original_metrics)
            dflash2_clustered_parts.append(dflash2_metrics)

            original_cluster_ids = build_cluster_ids(original_run)
            dflash2_cluster_ids = build_cluster_ids(dflash2_run)
            original_stage_parts.append(
                collect_stage_times(
                    original_run, original_cluster_ids, ORIGINAL_METHODS
                )
            )
            dflash2_stage_parts.append(
                collect_stage_times(dflash2_run, dflash2_cluster_ids, DFLASH2_METHODS)
            )

            for family, path, run in (
                (FAMILY_ORIGINAL, original_path, original_run),
                (FAMILY_DFLASH2, dflash2_path, dflash2_run),
            ):
                provenance_pairs.append(
                    {
                        "dataset_label": label,
                        "family": family,
                        "path": str(path),
                        "sha256": sha256_file(path),
                        "dataset": run["args"]["dataset"],
                        "dataset_revision": run["args"]["dataset_revision"],
                        "trajectory_mode": run["trajectory_mode"],
                        "sample_count": len(run["completed_dataset_indices"]),
                    }
                )

        original_clustered = merge_clustered(*original_clustered_parts)
        dflash2_clustered = merge_clustered(*dflash2_clustered_parts)

        for clustered in (original_clustered, dflash2_clustered):
            for method, clusters in clustered.items():
                for metrics in clusters.values():
                    if metrics["timing_source_is_runner"] >= 0.5:
                        timing_source_counts[method]["runner"] += 1
                    else:
                        timing_source_counts[method]["derived"] += 1

        combined_clustered = {
            **{
                method: clusters
                for method, clusters in original_clustered.items()
                if method != "baseline"
            },
            **{
                method: clusters
                for method, clusters in dflash2_clustered.items()
                if method != "baseline"
            },
        }

        method_metrics_rows.extend(
            build_method_metrics_rows(
                label,
                FAMILY_ORIGINAL,
                original_run,
                original_clustered,
                ORIGINAL_METHODS,
            )
        )
        method_metrics_rows.extend(
            build_method_metrics_rows(
                label,
                FAMILY_DFLASH2,
                dflash2_run,
                dflash2_clustered,
                DFLASH2_METHODS,
            )
        )
        correctness_rows.extend(
            build_correctness_rows(
                label, FAMILY_ORIGINAL, original_clustered, ORIGINAL_METHODS
            )
        )
        correctness_rows.extend(
            build_correctness_rows(
                label, FAMILY_DFLASH2, dflash2_clustered, DFLASH2_METHODS
            )
        )
        comparison_rows.extend(
            build_comparison_rows(
                label,
                combined_clustered,
                samples=args.bootstrap_samples,
                dataset_offset=dataset_offset,
                original_baseline=original_clustered.get("baseline"),
                dflash2_baseline=dflash2_clustered.get("baseline"),
            )
        )
        comparison_rows.extend(
            build_family_reference_rows(
                label,
                original_clustered,
                ORIGINAL_METHODS,
                samples=args.bootstrap_samples,
                seed_offset=dataset_offset * 1000 + 300,
            )
        )
        comparison_rows.extend(
            build_family_reference_rows(
                label,
                dflash2_clustered,
                DFLASH2_METHODS,
                samples=args.bootstrap_samples,
                seed_offset=dataset_offset * 1000 + 600,
            )
        )

        combined_stage_times = merge_stage_sources(
            *original_stage_parts, *dflash2_stage_parts
        )
        timing_rows.extend(
            build_timing_decomposition_rows(
                label, combined_clustered, combined_stage_times
            )
        )

    output_dir = args.output_dir
    write_csv(output_dir / "method_metrics.csv", method_metrics_rows)
    write_csv(output_dir / "paired_throughput_comparisons.csv", comparison_rows)
    write_csv(output_dir / "timing_decomposition.csv", timing_rows)
    write_csv(output_dir / "correctness.csv", correctness_rows)

    provenance = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "expected_commit": args.expected_commit,
        "target_model": TARGET_MODEL,
        "target_revision": args.target_revision,
        "original_draft_revision": args.original_draft_revision,
        "dflash2_draft_revision": args.dflash2_draft_revision,
        "allow_partial": args.allow_partial,
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "artifacts": provenance_pairs,
        "timing_field_source_counts": timing_source_counts,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True), encoding="utf-8"
    )


def merge_stage_sources(
    *sources: dict[str, dict[int, list[dict[str, float]]]],
) -> dict[str, dict[int, list[dict[str, float]]]]:
    merged: dict[str, dict[int, list[dict[str, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for source in sources:
        for method, clusters in source.items():
            for cluster, entries in clusters.items():
                merged[method][cluster].extend(entries)
    return merged


def main() -> None:
    args = parse_args()
    run_analysis(args)


if __name__ == "__main__":
    main()
