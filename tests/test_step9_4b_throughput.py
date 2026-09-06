from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from analyze_step9_4b_throughput import (
    DFLASH2_DRAFTER,
    DFLASH2_METHODS,
    DFLASH2_TREE_CONFIGS_FULL,
    EXPECTED_DATASET_REVISIONS,
    ORIGINAL_DRAFTER,
    ORIGINAL_METHODS,
    ORIGINAL_TREE_BUDGET_ARG,
    TARGET_MODEL,
    build_cluster_ids,
    collect_repetition_metrics,
    comparison_pairs,
    compare_methods,
    exact_output_match,
    extract_timing_fields,
    matched_and_committed,
    merge_clustered,
    method_label,
    run_analysis,
    validate_artifact,
    validate_cross_family,
)

COMMIT = "a" * 40
TARGET_REVISION = "t" * 40
ORIGINAL_DRAFT_REVISION = "o" * 40
DFLASH2_DRAFT_REVISION = "d" * 40


def make_result(
    *,
    idx: int = 0,
    output_suffix: tuple[int, ...] = (100, 101, 102),
    time_to_first_token: float = 0.01,
    time_per_output_token: float = 0.02,
    decode_rounds: int = 2,
    stage_times: dict[str, float] | None = None,
    acceptance_lengths: list[int] | None = None,
    matched_per_round: list[float] | None = None,
    committed_per_round: list[float] | None = None,
    round_metrics: list[dict[str, object]] | None = None,
    include_end_to_end: bool = True,
    prompt_hash: str = "0" * 12,
    repetition_overrides: list[dict[str, object]] | None = None,
    peak_allocated_gib: float = 8.0,
    peak_reserved_gib: float = 10.0,
) -> SimpleNamespace:
    prefix = [idx * 10 + 1, idx * 10 + 2, idx * 10 + 3, idx * 10 + 4]
    output_ids = torch.tensor([prefix + list(output_suffix)], dtype=torch.long)
    num_input_tokens = len(prefix)
    num_output_tokens = len(output_suffix)
    fields: dict[str, object] = {
        "output_ids": output_ids,
        "num_input_tokens": num_input_tokens,
        "num_output_tokens": num_output_tokens,
        "time_to_first_token": time_to_first_token,
        "time_per_output_token": time_per_output_token,
        "decode_rounds": decode_rounds,
        "stage_times": stage_times
        or {"draft": 0.001, "verify": 0.004, "commit": 0.0005},
        "acceptance_lengths": acceptance_lengths or [3, 2],
        "peak_allocated_gib": peak_allocated_gib,
        "peak_reserved_gib": peak_reserved_gib,
    }
    if matched_per_round is not None:
        fields["matched_draft_tokens_per_round"] = matched_per_round
    if committed_per_round is not None:
        fields["committed_tokens_per_round"] = committed_per_round
    if round_metrics is not None:
        fields["round_metrics"] = round_metrics
    if include_end_to_end:
        total = time_to_first_token + time_per_output_token * num_output_tokens
        fields["total_generation_time"] = total
        fields["tokens_per_second"] = num_output_tokens / total
    result = SimpleNamespace(**fields)
    result.prompt_hash = prompt_hash

    # Mirror benchmark.py's attach_repetition_bookkeeping: every result
    # carries a .repetitions list, defaulting to a single-element list
    # containing itself when no repeated measurements are requested.
    if repetition_overrides is None:
        result.repetitions = [result]
    else:
        repetitions = []
        for overrides in repetition_overrides:
            rep_fields = dict(fields)
            rep_fields.update(
                {key: value for key, value in overrides.items() if key != "prompt_hash"}
            )
            if include_end_to_end and (
                "total_generation_time" not in overrides
                and {
                    "time_to_first_token",
                    "time_per_output_token",
                    "num_output_tokens",
                }
                & overrides.keys()
            ):
                rep_ttft = rep_fields["time_to_first_token"]
                rep_tpot = rep_fields["time_per_output_token"]
                rep_tokens = rep_fields["num_output_tokens"]
                rep_total = rep_ttft + rep_tpot * rep_tokens
                rep_fields["total_generation_time"] = rep_total
                rep_fields["tokens_per_second"] = (
                    rep_tokens / rep_total if rep_total > 0 else 0.0
                )
            repetition = SimpleNamespace(**rep_fields)
            repetition.prompt_hash = overrides.get("prompt_hash", prompt_hash)
            repetitions.append(repetition)
        result.repetitions = repetitions
    return result


def prompt_id(dataset: str, idx: int, method: str) -> str:
    return f"{dataset}:selected:{idx}:turn-0:{method}:{'0' * 12}"


def _original_turn_methods(
    *, idx: int, suffix: tuple[int, ...], include_end_to_end: bool = True
) -> dict[str, SimpleNamespace]:
    """Build one turn's original-family method dict (shared by the artifact
    builder and by tests that need to hand-construct multi-turn artifacts)."""
    return {
        "baseline": make_result(
            idx=idx, output_suffix=suffix, include_end_to_end=include_end_to_end
        ),
        "dflash": make_result(
            idx=idx, output_suffix=suffix, include_end_to_end=include_end_to_end
        ),
        "ddtree_tb16": make_result(
            idx=idx, output_suffix=suffix, include_end_to_end=include_end_to_end
        ),
        "ddtree_tb32": make_result(
            idx=idx, output_suffix=suffix, include_end_to_end=include_end_to_end
        ),
        "ddtree_tb64": make_result(
            idx=idx, output_suffix=suffix, include_end_to_end=include_end_to_end
        ),
    }


def _dflash2_turn_methods(
    *, dataset: str, idx: int, suffix: tuple[int, ...], include_end_to_end: bool = True
) -> dict[str, SimpleNamespace]:
    """Build one turn's dflash2-family method dict (shared by the artifact
    builder and by tests that need to hand-construct multi-turn artifacts)."""
    methods: dict[str, SimpleNamespace] = {}
    for method in DFLASH2_METHODS:
        kwargs: dict[str, object] = dict(
            idx=idx,
            output_suffix=suffix,
            include_end_to_end=include_end_to_end,
        )
        if method != "baseline":
            kwargs["matched_per_round"] = [6.0, 5.0]
            kwargs["committed_per_round"] = [7.0, 6.0]
            kwargs["round_metrics"] = [{"prompt_id": prompt_id(dataset, idx, method)}]
        methods[method] = make_result(**kwargs)
    return methods


def build_original_artifact(
    *,
    dataset: str = "gsm8k",
    indices: tuple[int, ...] = (0, 1),
    temperature: float = 0.0,
    max_new_tokens: int = 256,
    tree_budget: str = ORIGINAL_TREE_BUDGET_ARG,
    target_attn: str = "sdpa",
    draft_attn: str = "flash_attention_2",
    target_dtype: str = "torch.bfloat16",
    draft_dtype: str = "torch.bfloat16",
    timing_repetitions: int = 1,
    repository: dict[str, object] | None = None,
    target_revision: str = TARGET_REVISION,
    draft_revision: str = ORIGINAL_DRAFT_REVISION,
    gpu: str = "A100",
    cuda: str = "12.1",
    trajectory_mode: str = "controlled_shared",
    include_end_to_end: bool = True,
    output_suffix_by_idx: dict[int, tuple[int, ...]] | None = None,
    drop_method_at_position: int | None = None,
    drop_method_name: str = "ddtree_tb64",
) -> dict[str, object]:
    responses = []
    for position, idx in enumerate(indices):
        suffix = (output_suffix_by_idx or {}).get(
            idx, (100 + idx, 101 + idx, 102 + idx)
        )
        methods = _original_turn_methods(
            idx=idx, suffix=suffix, include_end_to_end=include_end_to_end
        )
        if drop_method_at_position == position:
            del methods[drop_method_name]
        responses.append(methods)
    return {
        "responses": responses,
        "args": {
            "model_name_or_path": TARGET_MODEL,
            "draft_name_or_path": ORIGINAL_DRAFTER,
            "draft_type": "dflash",
            "dataset": dataset,
            "dataset_revision": EXPECTED_DATASET_REVISIONS[dataset.replace("_", "-")],
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
            "max_samples": len(indices),
            "tree_budget": tree_budget,
        },
        "repository": repository or {"commit": COMMIT, "dirty": False},
        "target_revision": target_revision,
        "draft_revision": draft_revision,
        "target_attn_implementation": target_attn,
        "draft_attn_implementation": draft_attn,
        "target_dtype": target_dtype,
        "draft_dtype": draft_dtype,
        "timing_repetitions": timing_repetitions,
        "runtime": {
            "python": "3.12.0",
            "pytorch": "2.4.0",
            "transformers": "4.44.0",
            "cuda": cuda,
            "gpu": gpu,
            "peak_allocated_gib": 10.0,
            "peak_reserved_gib": 12.0,
            "decode_timing_excludes_first_draft_prefill": True,
        },
        "trajectory_mode": trajectory_mode,
        "completed_dataset_indices": list(indices),
    }


def build_dflash2_artifact(
    *,
    dataset: str = "gsm8k",
    indices: tuple[int, ...] = (0, 1),
    temperature: float = 0.0,
    max_new_tokens: int = 256,
    dflash2_tree_configs: str = DFLASH2_TREE_CONFIGS_FULL,
    target_attn: str = "sdpa",
    draft_attn: str = "sdpa",
    target_dtype: str = "torch.bfloat16",
    draft_dtype: str = "torch.bfloat16",
    timing_repetitions: int = 1,
    repository: dict[str, object] | None = None,
    target_revision: str = TARGET_REVISION,
    draft_revision: str = DFLASH2_DRAFT_REVISION,
    gpu: str = "A100",
    cuda: str = "12.1",
    trajectory_mode: str = "controlled_shared",
    include_end_to_end: bool = True,
    output_suffix_by_idx: dict[int, tuple[int, ...]] | None = None,
    drop_method_at_position: int | None = None,
    drop_method_name: str = "dflash2_unary_k16_tb64",
) -> dict[str, object]:
    responses = []
    for position, idx in enumerate(indices):
        suffix = (output_suffix_by_idx or {}).get(
            idx, (100 + idx, 101 + idx, 102 + idx)
        )
        methods = _dflash2_turn_methods(
            dataset=dataset,
            idx=idx,
            suffix=suffix,
            include_end_to_end=include_end_to_end,
        )
        if drop_method_at_position == position:
            del methods[drop_method_name]
        responses.append(methods)
    return {
        "responses": responses,
        "args": {
            "model_name_or_path": TARGET_MODEL,
            "draft_name_or_path": DFLASH2_DRAFTER,
            "draft_type": "dflash2",
            "dataset": dataset,
            "dataset_revision": EXPECTED_DATASET_REVISIONS[dataset.replace("_", "-")],
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
            "max_samples": len(indices),
            "dflash2_tree_configs": dflash2_tree_configs,
        },
        "repository": repository or {"commit": COMMIT, "dirty": False},
        "target_revision": target_revision,
        "draft_revision": draft_revision,
        "target_attn_implementation": target_attn,
        "draft_attn_implementation": draft_attn,
        "target_dtype": target_dtype,
        "draft_dtype": draft_dtype,
        "timing_repetitions": timing_repetitions,
        "runtime": {
            "python": "3.12.0",
            "pytorch": "2.4.0",
            "transformers": "4.44.0",
            "cuda": cuda,
            "gpu": gpu,
            "peak_allocated_gib": 11.0,
            "peak_reserved_gib": 13.0,
            "decode_timing_excludes_first_draft_prefill": True,
        },
        "trajectory_mode": trajectory_mode,
        "completed_dataset_indices": list(indices),
    }


# ---------------------------------------------------------------------------
# validate_artifact
# ---------------------------------------------------------------------------


def test_validate_artifact_accepts_clean_matching_pair() -> None:
    original = build_original_artifact()
    dflash2 = build_dflash2_artifact()

    validate_artifact(
        original,
        Path("original.pt"),
        family="original",
        expected_commit=COMMIT,
        target_revision=TARGET_REVISION,
        draft_revision=ORIGINAL_DRAFT_REVISION,
        allow_partial=True,
    )
    validate_artifact(
        dflash2,
        Path("dflash2.pt"),
        family="dflash2",
        expected_commit=COMMIT,
        target_revision=TARGET_REVISION,
        draft_revision=DFLASH2_DRAFT_REVISION,
        allow_partial=True,
    )
    validate_cross_family(original, dflash2, dataset_label="gsm8k", allow_partial=True)


def test_validate_artifact_rejects_dirty_repository() -> None:
    original = build_original_artifact(repository={"commit": COMMIT, "dirty": True})
    with pytest.raises(ValueError, match="clean"):
        validate_artifact(
            original,
            Path("original.pt"),
            family="original",
            expected_commit=COMMIT,
            target_revision=TARGET_REVISION,
            draft_revision=ORIGINAL_DRAFT_REVISION,
            allow_partial=True,
        )


def test_validate_artifact_rejects_wrong_target_revision() -> None:
    original = build_original_artifact(target_revision="wrong-revision")
    with pytest.raises(ValueError, match="target revision"):
        validate_artifact(
            original,
            Path("original.pt"),
            family="original",
            expected_commit=COMMIT,
            target_revision=TARGET_REVISION,
            draft_revision=ORIGINAL_DRAFT_REVISION,
            allow_partial=True,
        )


def test_validate_artifact_rejects_wrong_draft_revision() -> None:
    dflash2 = build_dflash2_artifact(draft_revision="wrong-revision")
    with pytest.raises(ValueError, match="draft revision"):
        validate_artifact(
            dflash2,
            Path("dflash2.pt"),
            family="dflash2",
            expected_commit=COMMIT,
            target_revision=TARGET_REVISION,
            draft_revision=DFLASH2_DRAFT_REVISION,
            allow_partial=True,
        )


def test_validate_artifact_rejects_target_not_sdpa() -> None:
    original = build_original_artifact(target_attn="flash_attention_2")
    with pytest.raises(ValueError, match="target attention implementation"):
        validate_artifact(
            original,
            Path("original.pt"),
            family="original",
            expected_commit=COMMIT,
            target_revision=TARGET_REVISION,
            draft_revision=ORIGINAL_DRAFT_REVISION,
            allow_partial=True,
        )


def test_validate_artifact_rejects_original_draft_not_flash_attention2() -> None:
    # The original DFlash draft always runs FlashAttention-2 by design
    # (dflash.py forces this regardless of --flash-attn, which only toggles
    # the *target*'s implementation); SDPA on the original draft is invalid.
    original = build_original_artifact(draft_attn="sdpa")
    with pytest.raises(ValueError, match="draft attention implementation"):
        validate_artifact(
            original,
            Path("original.pt"),
            family="original",
            expected_commit=COMMIT,
            target_revision=TARGET_REVISION,
            draft_revision=ORIGINAL_DRAFT_REVISION,
            allow_partial=True,
        )


def test_validate_artifact_rejects_dflash2_draft_not_sdpa() -> None:
    # DFlash2 benchmarking only supports SDPA mode for the draft.
    dflash2 = build_dflash2_artifact(draft_attn="flash_attention_2")
    with pytest.raises(ValueError, match="draft attention implementation"):
        validate_artifact(
            dflash2,
            Path("dflash2.pt"),
            family="dflash2",
            expected_commit=COMMIT,
            target_revision=TARGET_REVISION,
            draft_revision=DFLASH2_DRAFT_REVISION,
            allow_partial=True,
        )


def test_validate_artifact_accepts_original_family_default_flash_attention2_draft() -> (
    None
):
    # build_original_artifact()'s default draft_attn is flash_attention_2,
    # matching benchmark.py's forced original-DFlash draft implementation.
    original = build_original_artifact()
    validate_artifact(
        original,
        Path("original.pt"),
        family="original",
        expected_commit=COMMIT,
        target_revision=TARGET_REVISION,
        draft_revision=ORIGINAL_DRAFT_REVISION,
        allow_partial=True,
    )


@pytest.mark.parametrize("dtype_field", ["target_dtype", "draft_dtype"])
def test_validate_artifact_rejects_non_bfloat16_dtype(dtype_field: str) -> None:
    original = build_original_artifact(**{dtype_field: "torch.float16"})
    with pytest.raises(ValueError, match="bfloat16"):
        validate_artifact(
            original,
            Path("original.pt"),
            family="original",
            expected_commit=COMMIT,
            target_revision=TARGET_REVISION,
            draft_revision=ORIGINAL_DRAFT_REVISION,
            allow_partial=True,
        )


def test_validate_artifact_strict_mode_requires_full_sample_count() -> None:
    original = build_original_artifact(indices=(0, 1))
    with pytest.raises(ValueError, match="128"):
        validate_artifact(
            original,
            Path("original.pt"),
            family="original",
            expected_commit=COMMIT,
            target_revision=TARGET_REVISION,
            draft_revision=ORIGINAL_DRAFT_REVISION,
            allow_partial=False,
        )


def test_validate_artifact_strict_mode_requires_complete_method_matrix() -> None:
    dflash2 = build_dflash2_artifact(
        indices=tuple(range(128)), drop_method_at_position=0
    )

    with pytest.raises(ValueError, match="misses methods"):
        validate_artifact(
            dflash2,
            Path("dflash2.pt"),
            family="dflash2",
            expected_commit=COMMIT,
            target_revision=TARGET_REVISION,
            draft_revision=DFLASH2_DRAFT_REVISION,
            allow_partial=False,
        )

    # The same artifact is accepted for smoke/stability analysis.
    validate_artifact(
        dflash2,
        Path("dflash2.pt"),
        family="dflash2",
        expected_commit=COMMIT,
        target_revision=TARGET_REVISION,
        draft_revision=DFLASH2_DRAFT_REVISION,
        allow_partial=True,
    )


def test_validate_artifact_rejects_repetition_count_mismatch() -> None:
    # The artifact declares timing_repetitions=2 but every result was built
    # with the default single-element .repetitions bookkeeping.
    original = build_original_artifact(indices=tuple(range(128)), timing_repetitions=2)
    with pytest.raises(ValueError, match="repetitions, expected"):
        validate_artifact(
            original,
            Path("original.pt"),
            family="original",
            expected_commit=COMMIT,
            target_revision=TARGET_REVISION,
            draft_revision=ORIGINAL_DRAFT_REVISION,
            allow_partial=False,
        )


def test_validate_artifact_rejects_mismatched_prompt_hash_across_repetitions() -> None:
    original = build_original_artifact(indices=tuple(range(128)), timing_repetitions=2)
    for response in original["responses"]:
        for result in response.values():
            result.repetitions = [
                make_result(idx=0, prompt_hash="cccccccccccc") for _ in range(2)
            ]
    # Break just one method's repetition bookkeeping: the two repetitions
    # of the same turn must never diverge in prompt_hash.
    original["responses"][0]["dflash"].repetitions = [
        make_result(idx=0, prompt_hash="aaaaaaaaaaaa"),
        make_result(idx=0, prompt_hash="bbbbbbbbbbbb"),
    ]
    with pytest.raises(ValueError, match="mismatched prompt hashes"):
        validate_artifact(
            original,
            Path("original.pt"),
            family="original",
            expected_commit=COMMIT,
            target_revision=TARGET_REVISION,
            draft_revision=ORIGINAL_DRAFT_REVISION,
            allow_partial=False,
        )


def test_validate_artifact_accepts_consistent_repetition_bookkeeping() -> None:
    original = build_original_artifact(indices=tuple(range(128)), timing_repetitions=3)
    for response in original["responses"]:
        for result in response.values():
            result.repetitions = [
                make_result(idx=0, prompt_hash="cccccccccccc") for _ in range(3)
            ]
    validate_artifact(
        original,
        Path("original.pt"),
        family="original",
        expected_commit=COMMIT,
        target_revision=TARGET_REVISION,
        draft_revision=ORIGINAL_DRAFT_REVISION,
        allow_partial=False,
    )


# ---------------------------------------------------------------------------
# validate_cross_family
# ---------------------------------------------------------------------------


def test_validate_cross_family_rejects_dataset_index_mismatch() -> None:
    original = build_original_artifact(indices=(0, 1))
    dflash2 = build_dflash2_artifact(indices=(0, 2))

    with pytest.raises(ValueError, match="different dataset indices"):
        validate_cross_family(
            original, dflash2, dataset_label="gsm8k", allow_partial=False
        )


def test_validate_cross_family_rejects_prompt_hash_mismatch() -> None:
    original = build_original_artifact()
    # Swap idx 0/1 baseline outputs in the dflash2 artifact so the turn-0
    # input token ids for a given dataset index diverge across families
    # while output_ids' *shape* stays valid.
    dflash2 = build_dflash2_artifact(
        output_suffix_by_idx={0: (999, 998, 997), 1: (996, 995, 994)}
    )
    # Force divergent *input* tokens by rebuilding with a mismatched idx
    # mapping: reuse idx=1's prefix for the idx=0 response.
    swapped_baseline = make_result(idx=1, output_suffix=(100, 101, 102))
    dflash2["responses"][0]["baseline"] = swapped_baseline

    with pytest.raises(ValueError, match="prompt hashes diverge"):
        validate_cross_family(
            original, dflash2, dataset_label="gsm8k", allow_partial=False
        )


def test_validate_cross_family_rejects_trajectory_mode_mismatch() -> None:
    original = build_original_artifact(trajectory_mode="controlled_shared")
    dflash2 = build_dflash2_artifact(trajectory_mode="native")

    with pytest.raises(ValueError, match="trajectory_mode"):
        validate_cross_family(
            original, dflash2, dataset_label="gsm8k", allow_partial=False
        )


def test_validate_cross_family_rejects_runtime_mismatch() -> None:
    original = build_original_artifact(gpu="A100")
    dflash2 = build_dflash2_artifact(gpu="H100")

    with pytest.raises(ValueError, match="runtime-critical"):
        validate_cross_family(
            original, dflash2, dataset_label="gsm8k", allow_partial=False
        )


def test_validate_cross_family_rejects_timing_repetitions_mismatch() -> None:
    original = build_original_artifact(timing_repetitions=1)
    dflash2 = build_dflash2_artifact(timing_repetitions=3)

    with pytest.raises(ValueError, match="timing_repetitions differ"):
        validate_cross_family(
            original, dflash2, dataset_label="gsm8k", allow_partial=False
        )


def test_validate_cross_family_accepts_matching_baseline_output_strict() -> None:
    # Default builders already produce identical baseline output_ids per
    # dataset index across families (same suffix-generation formula), so
    # strict pairing must succeed without raising.
    original = build_original_artifact()
    dflash2 = build_dflash2_artifact()
    validate_cross_family(original, dflash2, dataset_label="gsm8k", allow_partial=False)


def test_validate_cross_family_rejects_baseline_output_mismatch_under_controlled_trajectories() -> (
    None
):
    original = build_original_artifact(trajectory_mode="controlled_shared")
    dflash2 = build_dflash2_artifact(trajectory_mode="controlled_shared")
    # Both processes claim to run the identical target/prompt, so baseline
    # (no speculation involved) must be byte-identical; corrupt just the
    # dflash2 side's baseline output to prove the analyzer recomputes this
    # from output_ids rather than trusting any pre-declared agreement.
    dflash2["responses"][0]["baseline"] = make_result(
        idx=0, output_suffix=(999, 998, 997)
    )

    with pytest.raises(ValueError, match="baseline output diverges across families"):
        validate_cross_family(
            original, dflash2, dataset_label="gsm8k", allow_partial=False
        )


def test_validate_cross_family_native_trajectories_only_enforce_turn_zero_baseline() -> (
    None
):
    # Build a genuine two-turn ("MT-Bench-style") single-conversation
    # artifact pair: turn 0 must match (fixed first user message), but
    # under native trajectories turn 1's baseline is allowed to diverge
    # since each process built it from its own turn-0 generation history.
    original = build_original_artifact(
        dataset="mt_bench", indices=(0,), trajectory_mode="native"
    )
    dflash2 = build_dflash2_artifact(
        dataset="mt_bench", indices=(0,), trajectory_mode="native"
    )
    turn0_suffix = (100, 101, 102)
    turn1_suffix_original = (200, 201, 202)
    turn1_suffix_dflash2 = (777, 776, 775)  # deliberately diverges

    original["responses"] = [
        _original_turn_methods(idx=0, suffix=turn0_suffix),
        _original_turn_methods(idx=0, suffix=turn1_suffix_original),
    ]
    dflash2["responses"] = [
        _dflash2_turn_methods(dataset="mt_bench", idx=0, suffix=turn0_suffix),
        _dflash2_turn_methods(dataset="mt_bench", idx=0, suffix=turn1_suffix_dflash2),
    ]

    # Native trajectories: turn-1 baseline divergence must NOT fail pairing.
    validate_cross_family(
        original, dflash2, dataset_label="mt_bench", allow_partial=False
    )

    # But a turn-0 baseline divergence must still fail, even under native
    # trajectories, since turn 0 has no generation-history dependency.
    dflash2["responses"][0] = _dflash2_turn_methods(
        dataset="mt_bench", idx=0, suffix=(111, 112, 113)
    )
    with pytest.raises(ValueError, match="baseline output diverges across families"):
        validate_cross_family(
            original, dflash2, dataset_label="mt_bench", allow_partial=False
        )


def test_validate_cross_family_controlled_trajectories_enforce_every_turn_baseline() -> (
    None
):
    # Under controlled_shared trajectories every turn uses a fixed
    # canonical conversation history, so unlike the native case, a
    # turn-1-only baseline mismatch must still fail strict pairing.
    original = build_original_artifact(
        dataset="mt_bench", indices=(0,), trajectory_mode="controlled_shared"
    )
    dflash2 = build_dflash2_artifact(
        dataset="mt_bench", indices=(0,), trajectory_mode="controlled_shared"
    )
    turn0_suffix = (100, 101, 102)

    original["responses"] = [
        _original_turn_methods(idx=0, suffix=turn0_suffix),
        _original_turn_methods(idx=0, suffix=(200, 201, 202)),
    ]
    dflash2["responses"] = [
        _dflash2_turn_methods(dataset="mt_bench", idx=0, suffix=turn0_suffix),
        _dflash2_turn_methods(dataset="mt_bench", idx=0, suffix=(777, 776, 775)),
    ]

    with pytest.raises(ValueError, match="baseline output diverges across families"):
        validate_cross_family(
            original, dflash2, dataset_label="mt_bench", allow_partial=False
        )


# ---------------------------------------------------------------------------
# build_cluster_ids
# ---------------------------------------------------------------------------


def test_build_cluster_ids_position_based_for_original_family() -> None:
    original = build_original_artifact(indices=(3, 7))
    assert build_cluster_ids(original) == [3, 7]


def test_build_cluster_ids_cross_checks_prompt_id_for_dflash2_family() -> None:
    dflash2 = build_dflash2_artifact(indices=(3, 7))
    assert build_cluster_ids(dflash2) == [3, 7]


def test_build_cluster_ids_raises_on_prompt_id_position_conflict() -> None:
    dflash2 = build_dflash2_artifact(indices=(3, 7))
    # Corrupt the prompt_id on one method so it disagrees with the
    # position-based estimate derived from completed_dataset_indices.
    dflash2["responses"][0]["dflash2"].round_metrics[0]["prompt_id"] = prompt_id(
        "gsm8k", 999, "dflash2"
    )
    with pytest.raises(ValueError, match="prompt_id points at dataset index"):
        build_cluster_ids(dflash2)


# ---------------------------------------------------------------------------
# matched_and_committed
# ---------------------------------------------------------------------------


def test_matched_and_committed_uses_explicit_fields_when_present() -> None:
    result = make_result(matched_per_round=[6.0, 4.0], committed_per_round=[7.0, 5.0])
    matched, committed = matched_and_committed(result, "dflash2_pairwise_k16_tb16")
    assert matched == pytest.approx(5.0)
    assert committed == pytest.approx(6.0)


def test_matched_and_committed_falls_back_to_acceptance_lengths() -> None:
    result = make_result(acceptance_lengths=[3, 5])
    matched, committed = matched_and_committed(result, "ddtree_tb16")
    assert committed == pytest.approx(4.0)
    assert matched == pytest.approx(3.0)  # mean(3-1, 5-1) = mean(2, 4) = 3


def test_matched_and_committed_floors_at_zero() -> None:
    result = make_result(acceptance_lengths=[0, 1])
    matched, _ = matched_and_committed(result, "ddtree_tb16")
    assert matched == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# extract_timing_fields adapter
# ---------------------------------------------------------------------------


def test_extract_timing_fields_uses_runner_fields_when_present() -> None:
    result = make_result(
        time_to_first_token=0.1,
        time_per_output_token=0.02,
        include_end_to_end=True,
    )
    fields = extract_timing_fields(
        result, path=Path("run.pt"), method="dflash2", allow_partial=False
    )
    assert fields["timing_source"] == "runner"
    assert fields["tokens_per_second_end_to_end"] == pytest.approx(
        result.tokens_per_second
    )


def test_extract_timing_fields_derives_under_allow_partial() -> None:
    result = make_result(
        time_to_first_token=0.1,
        time_per_output_token=0.02,
        include_end_to_end=False,
    )
    fields = extract_timing_fields(
        result, path=Path("run.pt"), method="dflash2", allow_partial=True
    )
    assert fields["timing_source"] == "derived"
    expected_total = 0.1 + 0.02 * result.num_output_tokens
    assert fields["total_generation_time"] == pytest.approx(expected_total)
    assert fields["tokens_per_second_end_to_end"] == pytest.approx(
        result.num_output_tokens / expected_total
    )


def test_extract_timing_fields_raises_strict_without_runner_fields() -> None:
    result = make_result(include_end_to_end=False)
    with pytest.raises(ValueError, match="end-to-end timing fields"):
        extract_timing_fields(
            result, path=Path("run.pt"), method="dflash2", allow_partial=False
        )


def test_extract_timing_fields_never_silently_defaults_missing_primary_field() -> None:
    result = make_result()
    del result.decode_rounds
    with pytest.raises(ValueError, match="misses required timing fields"):
        extract_timing_fields(
            result, path=Path("run.pt"), method="dflash2", allow_partial=True
        )


# ---------------------------------------------------------------------------
# comparison matrix labeling
# ---------------------------------------------------------------------------


def test_comparison_pairs_include_labeled_primary() -> None:
    pairs = comparison_pairs()
    primary = [pair for pair in pairs if pair[2] is True]
    assert primary == [
        (
            "dflash2_pairwise_k16_tb16",
            "dflash2_original_ddtree_tb16",
            True,
            "controlled",
        )
    ]


def test_comparison_pairs_label_cross_drafter_vs_controlled() -> None:
    pairs = {
        (left, right): comparison_type
        for left, right, _, comparison_type in comparison_pairs()
    }
    assert (
        pairs[("dflash2_pairwise_k16_tb16", "dflash2_original_ddtree_tb16")]
        == "controlled"
    )
    assert pairs[("dflash2_pairwise_k16_tb16", "dflash")] == "cross_drafter"
    assert pairs[("dflash2_pairwise_k16_tb32", "ddtree_tb32")] == "cross_drafter"
    assert pairs[("dflash2_pairwise_k16_tb7", "dflash2")] == "controlled"


def test_original_draft_revision_has_pinned_default() -> None:
    from analyze_step9_4b_throughput import ORIGINAL_DRAFTER_REVISION_DEFAULT

    # run_step9_4b_throughput.sh now pins ORIGINAL_DRAFTER_REVISION to a
    # verified Hugging Face Hub revision; the analyzer's CLI default must
    # track it so --original-draft-revision is optional, matching the
    # shell script's optional environment-variable override.
    assert (
        ORIGINAL_DRAFTER_REVISION_DEFAULT == "b74e3a329c4d963783143b1e970d95b002be72bd"
    )


def test_method_label_covers_every_method_key() -> None:
    for key in (*ORIGINAL_METHODS, *DFLASH2_METHODS):
        if key == "baseline":
            continue
        assert method_label(key)  # must not raise


# ---------------------------------------------------------------------------
# repetition aggregation + bootstrap-compatible comparisons
# ---------------------------------------------------------------------------


def test_merge_clustered_averages_repeated_conversation_measurements_and_recomputes_rate() -> (
    None
):
    # Two "repeated --pair artifact" measurements of the same single
    # conversation cluster (idx=0). merge_clustered must average the
    # underlying totals and recompute the throughput rate from that
    # average, never arithmetic-mean the pre-computed rate field directly
    # (which would silently hide how the totals actually combined).
    source_a = {
        "dflash2": {
            0: {
                "time_to_first_token": 0.1,
                "num_output_tokens": 10.0,
                "decode_rounds": 2.0,
                "decode_only_time": 0.15,
                "total_generation_time": 0.2,
                "tokens_per_second_end_to_end": 50.0,
                "matched_tokens": 5.0,
                "committed_tokens": 6.0,
                "truncated": 0.0,
                "exact_match": 1.0,
                "timing_source_is_runner": 1.0,
            }
        }
    }
    source_b = {
        "dflash2": {
            0: {
                **source_a["dflash2"][0],
                "decode_only_time": 0.25,
                "total_generation_time": 0.3,
                "tokens_per_second_end_to_end": 33.333333333333336,
                "timing_source_is_runner": 0.0,
            }
        }
    }
    merged = merge_clustered(source_a, source_b)
    assert list(merged["dflash2"].keys()) == [0]
    # Averaged totals: total_generation_time = mean(0.2, 0.3) = 0.25,
    # num_output_tokens = mean(10, 10) = 10 -> recomputed rate = 10/0.25 = 40.
    assert merged["dflash2"][0]["total_generation_time"] == pytest.approx(0.25)
    assert merged["dflash2"][0]["tokens_per_second_end_to_end"] == pytest.approx(40.0)
    assert merged["dflash2"][0]["decode_only_tokens_per_second"] == pytest.approx(50.0)
    assert merged["dflash2"][0]["timing_source_is_runner"] == pytest.approx(0.5)


def test_aggregate_repetitions_averages_totals_and_recomputes_rate_not_mean_of_rates() -> (
    None
):
    from analyze_step9_4b_throughput import aggregate_repetitions

    entries = [
        {
            "time_to_first_token": 0.1,
            "time_per_output_token": 0.01,
            "num_output_tokens": 10.0,
            "decode_rounds": 2.0,
            "decode_only_time": 0.1,
            "decode_only_tokens_per_second": 100.0,
            "total_generation_time": 0.2,
            "tokens_per_second_end_to_end": 50.0,
            "matched_tokens": 4.0,
            "committed_tokens": 5.0,
            "truncated": 0.0,
            "exact_match": 1.0,
            "timing_source": "runner",
        },
        {
            "time_to_first_token": 0.1,
            "time_per_output_token": 0.01,
            "num_output_tokens": 10.0,
            "decode_rounds": 2.0,
            "decode_only_time": 0.3,
            "decode_only_tokens_per_second": 33.333333333333336,
            "total_generation_time": 0.4,
            "tokens_per_second_end_to_end": 25.0,
            "matched_tokens": 4.0,
            "committed_tokens": 5.0,
            "truncated": 0.0,
            "exact_match": 1.0,
            "timing_source": "derived",
        },
    ]
    aggregated = aggregate_repetitions(entries)
    # total_generation_time averaged: mean(0.2, 0.4) = 0.3; the recomputed
    # rate must come from that average, NOT the naive mean of the two
    # repetitions' own rates (mean(50.0, 25.0) = 37.5).
    assert aggregated["total_generation_time"] == pytest.approx(0.3)
    assert aggregated["tokens_per_second_end_to_end"] == pytest.approx(10.0 / 0.3)
    assert aggregated["tokens_per_second_end_to_end"] != pytest.approx(37.5)
    assert aggregated["timing_source_is_runner"] == pytest.approx(0.5)


def test_aggregate_turns_to_conversation_sums_totals_and_weights_matched_by_decode_rounds() -> (
    None
):
    from analyze_step9_4b_throughput import aggregate_turns_to_conversation

    turn0 = {
        "time_to_first_token": 0.05,
        "time_per_output_token": 0.01,
        "num_output_tokens": 10.0,
        "decode_rounds": 2.0,
        "decode_only_time": 0.1,
        "decode_only_tokens_per_second": 100.0,
        "total_generation_time": 0.1,
        "tokens_per_second_end_to_end": 100.0,
        "matched_tokens": 3.0,
        "committed_tokens": 4.0,
        "truncated": 0.0,
        "exact_match": 1.0,
        "timing_source_is_runner": 1.0,
    }
    turn1 = {
        "time_to_first_token": 0.05,
        "time_per_output_token": 0.02,
        "num_output_tokens": 30.0,
        "decode_rounds": 3.0,
        "decode_only_time": 0.6,
        "decode_only_tokens_per_second": 50.0,
        "total_generation_time": 0.6,
        "tokens_per_second_end_to_end": 50.0,
        "matched_tokens": 5.0,
        "committed_tokens": 6.0,
        "truncated": 0.0,
        "exact_match": 1.0,
        "timing_source_is_runner": 1.0,
    }
    aggregated = aggregate_turns_to_conversation([turn0, turn1])
    # Target/decode calls sum across turns.
    assert aggregated["decode_rounds"] == pytest.approx(5.0)
    assert aggregated["num_output_tokens"] == pytest.approx(40.0)
    assert aggregated["total_generation_time"] == pytest.approx(0.7)
    # Recomputed from the summed totals: 40 / 0.7, never the naive
    # arithmetic mean of the two turns' own rates (mean(100, 50) = 75).
    assert aggregated["tokens_per_second_end_to_end"] == pytest.approx(40.0 / 0.7)
    assert aggregated["tokens_per_second_end_to_end"] != pytest.approx(75.0)
    # matched/committed tokens per round weighted by each turn's decode_rounds.
    assert aggregated["matched_tokens"] == pytest.approx((3.0 * 2 + 5.0 * 3) / 5)
    assert aggregated["committed_tokens"] == pytest.approx((4.0 * 2 + 6.0 * 3) / 5)


def test_collect_response_metrics_aggregates_repetitions_within_a_turn() -> None:
    from analyze_step9_4b_throughput import collect_response_metrics

    result = make_result(
        idx=0,
        include_end_to_end=True,
        repetition_overrides=[
            {
                "time_to_first_token": 0.1,
                "time_per_output_token": 0.01,
                "num_output_tokens": 10,
            },
            {
                "time_to_first_token": 0.1,
                "time_per_output_token": 0.03,
                "num_output_tokens": 10,
            },
        ],
    )
    baseline = make_result(idx=0)
    run = {
        "args": {"max_new_tokens": 256},
        "completed_dataset_indices": [0],
        "responses": [{"baseline": baseline, "dflash": result}],
    }
    clustered = collect_response_metrics(
        run, path=Path("run.pt"), methods=("baseline", "dflash"), allow_partial=True
    )
    # Repetition 1: total = 0.1 + 0.01*10 = 0.2 -> rate 50.
    # Repetition 2: total = 0.1 + 0.03*10 = 0.4 -> rate 25.
    # Averaged total = 0.3 -> recomputed rate = 10/0.3, NOT mean(50, 25)=37.5.
    entry = clustered["dflash"][0]
    assert entry["total_generation_time"] == pytest.approx(0.3)
    assert entry["tokens_per_second_end_to_end"] == pytest.approx(10.0 / 0.3)
    assert entry["tokens_per_second_end_to_end"] != pytest.approx(37.5)


def test_collect_response_metrics_sums_across_conversation_turns() -> None:
    from analyze_step9_4b_throughput import collect_response_metrics

    # Two turns of the same MT-Bench-style conversation (dataset index 0):
    # turns are additional work, not repeated measurements, so the
    # cluster-level throughput must be sum(tokens)/sum(time), never the
    # arithmetic mean of the two turns' own tokens/s.
    turn0 = make_result(
        idx=0,
        output_suffix=tuple(range(10)),
        time_to_first_token=0.0,
        time_per_output_token=0.01,
        decode_rounds=2,
        matched_per_round=[3.0, 3.0],
        committed_per_round=[4.0, 4.0],
    )
    turn1 = make_result(
        idx=0,
        output_suffix=tuple(range(30)),
        time_to_first_token=0.0,
        time_per_output_token=0.02,
        decode_rounds=3,
        matched_per_round=[5.0, 5.0, 5.0],
        committed_per_round=[6.0, 6.0, 6.0],
    )
    run = {
        "args": {"max_new_tokens": 256},
        "completed_dataset_indices": [0],
        "responses": [
            {
                "baseline": make_result(idx=0, output_suffix=tuple(range(10))),
                "dflash": turn0,
            },
            {
                "baseline": make_result(idx=0, output_suffix=tuple(range(30))),
                "dflash": turn1,
            },
        ],
    }
    clustered = collect_response_metrics(
        run, path=Path("run.pt"), methods=("baseline", "dflash"), allow_partial=True
    )
    entry = clustered["dflash"][0]
    assert entry["num_output_tokens"] == pytest.approx(40.0)
    assert entry["decode_rounds"] == pytest.approx(5.0)
    assert entry["total_generation_time"] == pytest.approx(0.7)
    assert entry["tokens_per_second_end_to_end"] == pytest.approx(40.0 / 0.7)
    assert entry["tokens_per_second_end_to_end"] != pytest.approx(
        75.0
    )  # not mean(100, 50)
    assert entry["matched_tokens"] == pytest.approx((3.0 * 2 + 5.0 * 3) / 5)
    assert entry["committed_tokens"] == pytest.approx((4.0 * 2 + 6.0 * 3) / 5)


def test_native_conversation_correctness_excludes_later_turns() -> None:
    from analyze_step9_4b_throughput import collect_response_metrics

    turn0 = make_result(idx=0, output_suffix=(10, 11))
    turn1 = make_result(idx=0, output_suffix=(20, 21))
    run = {
        "args": {"max_new_tokens": 256},
        "completed_dataset_indices": [0],
        "trajectory_mode": "native",
        "responses": [
            {
                "baseline": make_result(idx=0, output_suffix=(10, 11)),
                "dflash": turn0,
            },
            {
                "baseline": make_result(idx=0, output_suffix=(99, 98)),
                "dflash": turn1,
            },
        ],
    }

    clustered = collect_response_metrics(
        run,
        path=Path("run.pt"),
        methods=("baseline", "dflash"),
        allow_partial=False,
    )

    assert clustered["dflash"][0]["exact_match"] == pytest.approx(1.0)


def test_collect_stage_times_averages_across_repetitions() -> None:
    from analyze_step9_4b_throughput import collect_stage_times

    result = make_result(
        idx=0,
        repetition_overrides=[
            {
                "stage_times": {"draft": 0.002, "verify": 0.004, "commit": 0.001},
                "decode_rounds": 2,
            },
            {
                "stage_times": {"draft": 0.004, "verify": 0.008, "commit": 0.002},
                "decode_rounds": 2,
            },
        ],
    )
    run = {"responses": [{"dflash": result}]}
    stage_times = collect_stage_times(run, cluster_ids=[0], methods=("dflash",))
    entry = stage_times["dflash"][0][0]
    # rep1: 1000*0.002/2=1.0ms draft, 1000*0.004/2=2.0ms verify, 1000*0.001/2=0.5ms commit
    # rep2: 1000*0.004/2=2.0ms draft, 1000*0.008/2=4.0ms verify, 1000*0.002/2=1.0ms commit
    # averaged across the two repetitions (not just reading the designated result):
    assert entry["draft"] == pytest.approx(1.5)
    assert entry["verify"] == pytest.approx(3.0)
    assert entry["commit"] == pytest.approx(0.75)


def test_build_timing_decomposition_rows_tree_overhead_excludes_draft_delta() -> None:
    from analyze_step9_4b_throughput import build_timing_decomposition_rows

    clustered = {
        "dflash2_pairwise_k16_tb16": {
            0: {
                "matched_tokens": 5.0,
                "committed_tokens": 6.0,
                "decode_rounds": 3.0,
                "tokens_per_second_end_to_end": 60.0,
            },
        },
        "dflash2_original_ddtree_tb16": {
            0: {
                "matched_tokens": 4.0,
                "committed_tokens": 5.0,
                "decode_rounds": 4.0,
                "tokens_per_second_end_to_end": 50.0,
            },
        },
    }
    raw_stage_times = {
        "dflash2_pairwise_k16_tb16": {
            0: [
                {
                    "draft": 10.0,
                    "candidate_select": 2.0,
                    "tree_build": 3.0,
                    "tree_compile": 1.0,
                    "verify": 5.0,
                    "commit": 0.5,
                }
            ],
        },
        "dflash2_original_ddtree_tb16": {
            0: [
                {
                    "draft": 8.0,
                    "tree_build": 4.0,
                    "tree_compile": 2.0,
                    "verify": 6.0,
                    "commit": 0.5,
                }
            ],
        },
    }
    rows = build_timing_decomposition_rows("gsm8k", clustered, raw_stage_times)
    assert len(rows) == 1
    row = rows[0]
    # tree_overhead = (candidate_select+tree_build+tree_compile) deltas:
    # left: 2+3+1=6 (candidate_select absent on right -> treated as 0);
    # right: 0+4+2=6 -> 6 - 6 = 0. The draft delta (10 - 8 = 2) must be
    # kept as its own separate field, never folded into tree_overhead.
    assert row["tree_overhead_ms_per_call"] == pytest.approx(0.0)
    assert row["draft_stage_ms_per_call_change"] == pytest.approx(2.0)
    assert row["draft_stage_ms_per_call_left"] == pytest.approx(10.0)
    assert row["draft_stage_ms_per_call_right"] == pytest.approx(8.0)


def test_exact_output_match_ignores_serialized_matches_baseline_flag() -> None:
    # benchmark.py serializes its own .matches_baseline flag; the analyzer
    # must never read it and must instead recompute exact-match directly
    # from output_ids. Set a deliberately-wrong .matches_baseline on both a
    # true match and a true mismatch and prove the recomputed result
    # ignores it entirely.
    baseline = make_result(idx=0, output_suffix=(100, 101, 102))

    truly_matching = make_result(idx=0, output_suffix=(100, 101, 102))
    truly_matching.matches_baseline = False  # deliberately contradicts reality
    assert exact_output_match(truly_matching, baseline) is True

    truly_diverging = make_result(idx=0, output_suffix=(999, 998, 997))
    truly_diverging.matches_baseline = True  # deliberately contradicts reality
    assert exact_output_match(truly_diverging, baseline) is False


def test_collect_repetition_metrics_recomputes_exact_match_per_repetition() -> None:
    # Every repetition of a method's result must have its own exact_match
    # recomputed against output_ids -- not trust any serialized flag on the
    # repetition, and not just check the designated (first/last) result.
    prefix = [1, 2, 3, 4]
    baseline = make_result(idx=0, output_suffix=(100, 101, 102))
    matching_ids = torch.tensor([prefix + [100, 101, 102]], dtype=torch.long)
    diverging_ids = torch.tensor([prefix + [321, 322, 323]], dtype=torch.long)
    result = make_result(
        idx=0,
        output_suffix=(100, 101, 102),
        repetition_overrides=[
            {"output_ids": matching_ids, "matches_baseline": False},  # actually matches
            {
                "output_ids": diverging_ids,
                "matches_baseline": True,
            },  # actually diverges
        ],
    )
    metrics = collect_repetition_metrics(
        result,
        path=Path("run.pt"),
        method="dflash",
        allow_partial=True,
        max_new_tokens=256,
        baseline=baseline,
    )
    assert [entry["exact_match"] for entry in metrics] == [1.0, 0.0]


def test_collect_repetition_metrics_uses_corresponding_baseline_repetition() -> None:
    prefix = [1, 2, 3, 4]
    first_ids = torch.tensor([prefix + [100, 101, 102]], dtype=torch.long)
    second_ids = torch.tensor([prefix + [200, 201, 202]], dtype=torch.long)
    baseline = make_result(
        idx=0,
        repetition_overrides=[
            {"output_ids": first_ids},
            {"output_ids": second_ids},
        ],
    )
    result = make_result(
        idx=0,
        repetition_overrides=[
            {"output_ids": first_ids},
            {"output_ids": second_ids},
        ],
    )

    metrics = collect_repetition_metrics(
        result,
        path=Path("run.pt"),
        method="dflash",
        allow_partial=False,
        max_new_tokens=256,
        baseline=baseline,
    )

    assert [entry["exact_match"] for entry in metrics] == [1.0, 1.0]


def test_aggregate_repetitions_keeps_max_per_method_peak_memory() -> None:
    from analyze_step9_4b_throughput import aggregate_repetitions

    result = make_result(
        repetition_overrides=[
            {"peak_allocated_gib": 8.0, "peak_reserved_gib": 10.0},
            {"peak_allocated_gib": 9.5, "peak_reserved_gib": 11.0},
        ]
    )
    entries = collect_repetition_metrics(
        result,
        path=Path("run.pt"),
        method="baseline",
        allow_partial=False,
        max_new_tokens=256,
        baseline=result,
    )

    aggregated = aggregate_repetitions(entries)

    assert aggregated["peak_allocated_gib"] == pytest.approx(9.5)
    assert aggregated["peak_reserved_gib"] == pytest.approx(11.0)


def test_compare_methods_bootstrap_ci_and_counts() -> None:
    clustered = {
        "dflash2_pairwise_k16_tb16": {
            0: {"tokens_per_second_end_to_end": 60.0},
            1: {"tokens_per_second_end_to_end": 40.0},
            2: {"tokens_per_second_end_to_end": 55.0},
        },
        "dflash2_original_ddtree_tb16": {
            0: {"tokens_per_second_end_to_end": 50.0},
            1: {"tokens_per_second_end_to_end": 45.0},
            2: {"tokens_per_second_end_to_end": 55.0},
        },
    }
    row = compare_methods(
        "gsm8k",
        clustered,
        "dflash2_pairwise_k16_tb16",
        "dflash2_original_ddtree_tb16",
        is_primary=True,
        comparison_type="controlled",
        samples=2000,
        seed=42,
    )
    assert row is not None
    assert row["prompts"] == 3
    assert row["improve_prompts"] == 1
    assert row["hurt_prompts"] == 1
    assert row["tie_prompts"] == 1
    assert row["mean_tokens_per_second_diff"] == pytest.approx((10 - 5 + 0) / 3)
    assert row["tokens_per_second_diff_ci_low"] <= row["mean_tokens_per_second_diff"]
    assert row["tokens_per_second_diff_ci_high"] >= row["mean_tokens_per_second_diff"]


def test_cross_drafter_comparison_reports_baseline_normalized_difference() -> None:
    clustered = {
        "dflash2_pairwise_k16_tb16": {
            0: {"tokens_per_second_end_to_end": 60.0},
        },
        "ddtree_tb16": {
            0: {"tokens_per_second_end_to_end": 45.0},
        },
    }

    row = compare_methods(
        "gsm8k",
        clustered,
        "dflash2_pairwise_k16_tb16",
        "ddtree_tb16",
        is_primary=False,
        comparison_type="cross_drafter",
        samples=100,
        seed=1,
        left_baseline={0: {"tokens_per_second_end_to_end": 30.0}},
        right_baseline={0: {"tokens_per_second_end_to_end": 30.0}},
    )

    assert row is not None
    assert row["mean_baseline_normalized_speedup_diff"] == pytest.approx(0.5)
    assert row["baseline_normalized_speedup_diff_ci_low"] == pytest.approx(0.5)
    assert row["baseline_normalized_speedup_diff_ci_high"] == pytest.approx(0.5)


def test_compare_methods_returns_none_without_shared_prompts() -> None:
    clustered = {
        "dflash2_pairwise_k16_tb16": {0: {"tokens_per_second_end_to_end": 60.0}},
        "dflash2_original_ddtree_tb16": {1: {"tokens_per_second_end_to_end": 50.0}},
    }
    row = compare_methods(
        "gsm8k",
        clustered,
        "dflash2_pairwise_k16_tb16",
        "dflash2_original_ddtree_tb16",
        is_primary=True,
        comparison_type="controlled",
        samples=100,
        seed=1,
    )
    assert row is None


# ---------------------------------------------------------------------------
# End-to-end integration over synthetic .pt artifacts.
# ---------------------------------------------------------------------------


def test_run_analysis_end_to_end_writes_expected_outputs(tmp_path: Path) -> None:
    original = build_original_artifact(indices=(0, 1))
    dflash2 = build_dflash2_artifact(indices=(0, 1))

    original_path = tmp_path / "gsm8k_original_controlled.pt"
    dflash2_path = tmp_path / "gsm8k_dflash2_controlled.pt"
    torch.save(original, original_path)
    torch.save(dflash2, dflash2_path)

    output_dir = tmp_path / "out"
    args = argparse.Namespace(
        expected_commit=COMMIT,
        target_revision=TARGET_REVISION,
        original_draft_revision=ORIGINAL_DRAFT_REVISION,
        dflash2_draft_revision=DFLASH2_DRAFT_REVISION,
        pair=[("gsm8k", original_path, dflash2_path)],
        output_dir=output_dir,
        bootstrap_samples=200,
        allow_partial=True,
    )

    run_analysis(args)

    assert (output_dir / "method_metrics.csv").exists()
    assert (output_dir / "paired_throughput_comparisons.csv").exists()
    assert (output_dir / "timing_decomposition.csv").exists()
    assert (output_dir / "correctness.csv").exists()
    provenance_path = output_dir / "provenance.json"
    assert provenance_path.exists()

    import csv
    import json

    with (output_dir / "paired_throughput_comparisons.csv").open(
        encoding="utf-8"
    ) as handle:
        comparison_rows = list(csv.DictReader(handle))
    primary_rows = [row for row in comparison_rows if row["is_primary"] == "True"]
    assert len(primary_rows) == 1
    assert primary_rows[0]["left_method"] == "DFlash2-Pairwise-K16-B16"
    assert primary_rows[0]["right_method"] == "DFlash2-Original-DDTree-B16"
    assert primary_rows[0]["comparison_type"] == "controlled"

    comparison_types = {row["comparison_type"] for row in comparison_rows}
    assert comparison_types == {
        "controlled",
        "cross_drafter",
        "within_family_vs_sequential",
    }

    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    assert provenance["expected_commit"] == COMMIT
    assert len(provenance["artifacts"]) == 2
    for artifact in provenance["artifacts"]:
        assert len(artifact["sha256"]) == 64


def test_run_analysis_merges_repeated_pairs_for_same_dataset_label(
    tmp_path: Path,
) -> None:
    # Two "repetition" artifact pairs for the same dataset label/indices;
    # the analyzer must aggregate them within-prompt rather than doubling
    # the effective prompt count used for bootstrap comparisons.
    original_a = build_original_artifact(indices=(0,))
    dflash2_a = build_dflash2_artifact(indices=(0,))
    original_b = build_original_artifact(indices=(0,))
    dflash2_b = build_dflash2_artifact(indices=(0,))

    paths = []
    for label, artifact in (
        ("original_a", original_a),
        ("dflash2_a", dflash2_a),
        ("original_b", original_b),
        ("dflash2_b", dflash2_b),
    ):
        path = tmp_path / f"{label}.pt"
        torch.save(artifact, path)
        paths.append(path)

    output_dir = tmp_path / "out"
    args = argparse.Namespace(
        expected_commit=COMMIT,
        target_revision=TARGET_REVISION,
        original_draft_revision=ORIGINAL_DRAFT_REVISION,
        dflash2_draft_revision=DFLASH2_DRAFT_REVISION,
        pair=[
            ("gsm8k", paths[0], paths[1]),
            ("gsm8k", paths[2], paths[3]),
        ],
        output_dir=output_dir,
        bootstrap_samples=200,
        allow_partial=True,
    )

    run_analysis(args)

    import csv

    with (output_dir / "paired_throughput_comparisons.csv").open(
        encoding="utf-8"
    ) as handle:
        comparison_rows = list(csv.DictReader(handle))
    primary_rows = [row for row in comparison_rows if row["is_primary"] == "True"]
    assert len(primary_rows) == 1
    # Still exactly one prompt cluster (idx=0), even though two artifact
    # pairs were supplied for the same dataset label.
    assert primary_rows[0]["prompts"] == "1"
