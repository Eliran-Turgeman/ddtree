"""CPU/synthetic tests for the Step-9 throughput protocol additions in
benchmark.py and the shared end-to-end timing helper in dflash.py.

These exercise pure bookkeeping logic (method-order rotation, repetition
folding, dispatch routing, timing-metric arithmetic, peak-memory bracketing,
and baseline-comparison bookkeeping) without requiring a GPU or real model
weights.
"""

from types import SimpleNamespace

import pytest
import torch

import benchmark
import dflash
from benchmark import (
    apply_baseline_comparison,
    attach_repetition_bookkeeping,
    measure_method_with_peak_memory,
    rotate_method_order,
    run_method,
)
from dflash import end_to_end_timing_fields


# ---------------------------------------------------------------------------
# deterministic method ordering
# ---------------------------------------------------------------------------


def test_rotate_method_order_is_a_permutation() -> None:
    keys = ["baseline", "dflash", "ddtree_tb16"]

    order = rotate_method_order(keys, dataset_index=1, turn_index=0, repetition_index=0)

    assert sorted(order) == sorted(keys)
    assert len(order) == len(keys)


def test_rotate_method_order_is_deterministic_given_same_seed() -> None:
    keys = ["baseline", "dflash", "ddtree_tb16", "ddtree_tb32"]

    first = rotate_method_order(keys, dataset_index=7, turn_index=2, repetition_index=1)
    second = rotate_method_order(
        keys, dataset_index=7, turn_index=2, repetition_index=1
    )

    assert first == second


def test_rotate_method_order_varies_across_dataset_indices() -> None:
    keys = ["baseline", "dflash", "ddtree_tb16", "ddtree_tb32"]

    orders = {
        tuple(
            rotate_method_order(keys, dataset_index=i, turn_index=0, repetition_index=0)
        )
        for i in range(20)
    }

    # Across a representative prompt sweep, no method is pinned first.
    first_positions = {order[0] for order in orders}
    assert first_positions == set(keys)


def test_rotate_method_order_breaks_fixed_adjacency() -> None:
    keys = ["baseline", "dflash", "ddtree_tb16", "ddtree_tb32"]

    predecessors = set()
    for dataset_index in range(20):
        order = rotate_method_order(
            keys,
            dataset_index=dataset_index,
            turn_index=0,
            repetition_index=0,
        )
        index = order.index("ddtree_tb16")
        predecessors.add(order[index - 1] if index > 0 else None)

    assert len(predecessors) > 1


def test_rotate_method_order_varies_with_turn_and_repetition() -> None:
    keys = ["baseline", "dflash", "ddtree_tb16"]

    by_turn = rotate_method_order(
        keys, dataset_index=0, turn_index=1, repetition_index=0
    )
    by_repetition = rotate_method_order(
        keys, dataset_index=0, turn_index=0, repetition_index=1
    )
    baseline_order = rotate_method_order(
        keys, dataset_index=0, turn_index=0, repetition_index=0
    )

    # Rotation is seeded from dataset index, turn, and repetition together,
    # so changing either alone should (for this key count) move the offset.
    assert by_turn != baseline_order or by_repetition != baseline_order


def test_rotate_method_order_handles_empty_input() -> None:
    assert (
        rotate_method_order([], dataset_index=3, turn_index=1, repetition_index=2) == []
    )


# ---------------------------------------------------------------------------
# attach_repetition_bookkeeping
# ---------------------------------------------------------------------------


def test_attach_repetition_bookkeeping_default_repetition_preserves_shape() -> None:
    """With a single repetition (the default), the designated result must
    remain the same object with the same directly-readable attributes as
    before repetitions existed, plus a `.repetitions` list containing only
    itself -- so downstream consumers reading `response[method_key].foo`
    are unaffected.
    """
    single_result = SimpleNamespace(output_ids="tokens", time_per_output_token=0.01)
    method_repetition_results = {"dflash": [single_result]}

    response = attach_repetition_bookkeeping(method_repetition_results)

    assert response["dflash"] is single_result
    assert response["dflash"].output_ids == "tokens"
    assert response["dflash"].time_per_output_token == 0.01
    assert response["dflash"].repetitions == [single_result]


def test_attach_repetition_bookkeeping_preserves_every_repetition() -> None:
    results = [
        SimpleNamespace(total_generation_time=1.0),
        SimpleNamespace(total_generation_time=1.2),
        SimpleNamespace(total_generation_time=0.9),
    ]
    method_repetition_results = {"dflash2": results}

    response = attach_repetition_bookkeeping(method_repetition_results)

    assert response["dflash2"] is results[0]
    assert response["dflash2"].repetitions is results
    assert [r.total_generation_time for r in response["dflash2"].repetitions] == [
        1.0,
        1.2,
        0.9,
    ]


def test_attach_repetition_bookkeeping_designated_index_selects_history_result() -> (
    None
):
    results = [SimpleNamespace(tag="rep0"), SimpleNamespace(tag="rep1")]
    method_repetition_results = {"dflash": results}

    response = attach_repetition_bookkeeping(
        method_repetition_results, designated_index=1
    )

    assert response["dflash"].tag == "rep1"
    assert response["dflash"].repetitions == results


def test_attach_repetition_bookkeeping_handles_multiple_methods_independently() -> None:
    method_repetition_results = {
        "baseline": [SimpleNamespace(tag="baseline-rep0")],
        "dflash": [
            SimpleNamespace(tag="dflash-rep0"),
            SimpleNamespace(tag="dflash-rep1"),
        ],
    }

    response = attach_repetition_bookkeeping(method_repetition_results)

    assert response["baseline"].tag == "baseline-rep0"
    assert len(response["baseline"].repetitions) == 1
    assert response["dflash"].tag == "dflash-rep0"
    assert len(response["dflash"].repetitions) == 2


# ---------------------------------------------------------------------------
# run_method dispatch
# ---------------------------------------------------------------------------


@pytest.fixture
def stub_generators(monkeypatch: pytest.MonkeyPatch) -> dict[str, list]:
    """Replace every underlying *_generate function benchmark.py dispatches
    to with a recording stub, so dispatch routing can be verified without a
    GPU or real model weights.
    """
    calls: dict[str, list] = {
        "dflash_generate": [],
        "ddtree_generate": [],
        "dflash2_generate": [],
        "dflash2_tree_generate": [],
    }

    def make_stub(name: str):
        def stub(**kwargs):
            calls[name].append(kwargs)
            return SimpleNamespace(name=name, kwargs=kwargs)

        return stub

    for name in calls:
        monkeypatch.setattr(benchmark, name, make_stub(name))
    return calls


class _FakeTokenizer:
    eos_token_id = 99


class _FakeDraftModel:
    mask_token_id = 7


def test_run_method_dispatches_baseline_with_block_size_one(stub_generators) -> None:
    result = run_method(
        "baseline",
        draft_model=_FakeDraftModel(),
        target=object(),
        tokenizer=_FakeTokenizer(),
        input_ids="ids",
        max_new_tokens=16,
        temperature=0.0,
        block_size=4,
        method_key_to_tree_budget={},
        method_key_to_tree_method={},
        collect_allocation_data=False,
    )

    assert result.name == "dflash_generate"
    assert result.kwargs["block_size"] == 1
    assert not stub_generators["ddtree_generate"]
    assert not stub_generators["dflash2_generate"]
    assert not stub_generators["dflash2_tree_generate"]


def test_run_method_dispatches_dflash_with_configured_block_size(
    stub_generators,
) -> None:
    result = run_method(
        "dflash",
        draft_model=_FakeDraftModel(),
        target=object(),
        tokenizer=_FakeTokenizer(),
        input_ids="ids",
        max_new_tokens=16,
        temperature=0.0,
        block_size=4,
        method_key_to_tree_budget={},
        method_key_to_tree_method={},
        collect_allocation_data=False,
    )

    assert result.name == "dflash_generate"
    assert result.kwargs["block_size"] == 4


def test_run_method_dispatches_ddtree_with_tree_budget(stub_generators) -> None:
    result = run_method(
        "ddtree_tb16",
        draft_model=_FakeDraftModel(),
        target=object(),
        tokenizer=_FakeTokenizer(),
        input_ids="ids",
        max_new_tokens=16,
        temperature=0.0,
        block_size=4,
        method_key_to_tree_budget={"ddtree_tb16": 16},
        method_key_to_tree_method={},
        collect_allocation_data=False,
    )

    assert result.name == "ddtree_generate"
    assert result.kwargs["tree_budget"] == 16


def test_run_method_dispatches_dflash2_with_prompt_id(stub_generators) -> None:
    result = run_method(
        "dflash2",
        draft_model=_FakeDraftModel(),
        target=object(),
        tokenizer=_FakeTokenizer(),
        input_ids="ids",
        max_new_tokens=16,
        temperature=0.0,
        block_size=4,
        method_key_to_tree_budget={},
        method_key_to_tree_method={},
        collect_allocation_data=True,
        prompt_id="prompt-abc",
    )

    assert result.name == "dflash2_generate"
    assert result.kwargs["prompt_id"] == "prompt-abc"
    assert result.kwargs["collect_traces"] is True


def test_run_method_dispatches_dflash2_tree_variants(stub_generators) -> None:
    result = run_method(
        "dflash2_pairwise_k16_tb32",
        draft_model=_FakeDraftModel(),
        target=object(),
        tokenizer=_FakeTokenizer(),
        input_ids="ids",
        max_new_tokens=16,
        temperature=0.0,
        block_size=4,
        method_key_to_tree_budget={"dflash2_pairwise_k16_tb32": 32},
        method_key_to_tree_method={"dflash2_pairwise_k16_tb32": "dflash2_pairwise_k16"},
        collect_allocation_data=False,
        prompt_id="prompt-xyz",
    )

    assert result.name == "dflash2_tree_generate"
    assert result.kwargs["tree_budget"] == 32
    assert result.kwargs["tree_method"] == "dflash2_pairwise_k16"
    assert result.kwargs["prompt_id"] == "prompt-xyz"


# ---------------------------------------------------------------------------
# end_to_end_timing_fields (shared by every *_generate implementation)
# ---------------------------------------------------------------------------


def test_end_to_end_timing_fields_computes_expected_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Simulate a single cuda_time() call at t=10.0 marking the end of
    # generation, with prefill starting at t=1.0 and (post-first-round)
    # decode timing starting at t=3.0.
    monkeypatch.setattr(dflash, "cuda_time", lambda: 10.0)

    fields = end_to_end_timing_fields(
        prefill_start=1.0,
        decode_start=3.0,
        num_output_tokens=14,
    )

    assert fields["decode_time"] == pytest.approx(7.0)
    assert fields["total_generation_time"] == pytest.approx(9.0)
    assert fields["tokens_per_second"] == pytest.approx(14 / 9.0)


def test_end_to_end_timing_fields_guards_against_zero_duration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(dflash, "cuda_time", lambda: 5.0)

    fields = end_to_end_timing_fields(
        prefill_start=5.0,
        decode_start=5.0,
        num_output_tokens=3,
    )

    assert fields["total_generation_time"] == 0.0
    assert fields["tokens_per_second"] == 0.0


def test_end_to_end_timing_fields_single_sync_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The helper must call cuda_time() exactly once so it adds no extra
    CUDA synchronization beyond what generators already performed."""
    call_count = 0

    def counting_cuda_time() -> float:
        nonlocal call_count
        call_count += 1
        return 42.0

    monkeypatch.setattr(dflash, "cuda_time", counting_cuda_time)

    end_to_end_timing_fields(prefill_start=40.0, decode_start=41.0, num_output_tokens=1)

    assert call_count == 1


# ---------------------------------------------------------------------------
# measure_method_with_peak_memory
# ---------------------------------------------------------------------------


class _FakeCudaPeakMemory:
    """Records reset calls and replays scripted max_memory_* readings so
    measure_method_with_peak_memory can be exercised without a real GPU."""

    def __init__(
        self, allocated_readings: list[int], reserved_readings: list[int]
    ) -> None:
        self.allocated_readings = iter(allocated_readings)
        self.reserved_readings = iter(reserved_readings)
        self.reset_calls: list[object] = []

    def reset_peak_memory_stats(self, device: object) -> None:
        self.reset_calls.append(device)

    def max_memory_allocated(self, device: object) -> int:
        return next(self.allocated_readings)

    def max_memory_reserved(self, device: object) -> int:
        return next(self.reserved_readings)


def test_measure_method_with_peak_memory_resets_before_and_attaches_after(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # First reading (before reset) simulates whatever peak accumulated from
    # earlier calls/warmup; second reading (after the call) is this call's
    # own isolated peak.
    fake_cuda = _FakeCudaPeakMemory(
        allocated_readings=[2_000_000_000, 5_000_000_000],
        reserved_readings=[3_000_000_000, 6_000_000_000],
    )
    monkeypatch.setattr(
        benchmark.torch.cuda,
        "reset_peak_memory_stats",
        fake_cuda.reset_peak_memory_stats,
    )
    monkeypatch.setattr(
        benchmark.torch.cuda, "max_memory_allocated", fake_cuda.max_memory_allocated
    )
    monkeypatch.setattr(
        benchmark.torch.cuda, "max_memory_reserved", fake_cuda.max_memory_reserved
    )
    monkeypatch.setattr(
        benchmark,
        "run_method",
        lambda method_key, **kwargs: SimpleNamespace(name=method_key),
    )

    tracker = {"allocated_bytes": 0.0, "reserved_bytes": 0.0}
    result = measure_method_with_peak_memory(
        "dflash", device="cuda:0", peak_memory_tracker=tracker
    )

    assert fake_cuda.reset_calls == ["cuda:0"]
    assert result.name == "dflash"
    assert result.peak_allocated_gib == pytest.approx(5_000_000_000 / 1024**3)
    assert result.peak_reserved_gib == pytest.approx(6_000_000_000 / 1024**3)
    assert tracker["allocated_bytes"] == 5_000_000_000
    assert tracker["reserved_bytes"] == 6_000_000_000


def test_measure_method_with_peak_memory_preserves_running_max_across_resets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The pre-reset reading (9 GB-ish) is larger than this call's own peak
    # (1 GB-ish); the tracker must keep the larger value even though the
    # per-call result reflects only the isolated post-reset peak.
    fake_cuda = _FakeCudaPeakMemory(
        allocated_readings=[9_000_000_000, 1_000_000_000],
        reserved_readings=[9_500_000_000, 1_500_000_000],
    )
    monkeypatch.setattr(
        benchmark.torch.cuda,
        "reset_peak_memory_stats",
        fake_cuda.reset_peak_memory_stats,
    )
    monkeypatch.setattr(
        benchmark.torch.cuda, "max_memory_allocated", fake_cuda.max_memory_allocated
    )
    monkeypatch.setattr(
        benchmark.torch.cuda, "max_memory_reserved", fake_cuda.max_memory_reserved
    )
    monkeypatch.setattr(
        benchmark, "run_method", lambda method_key, **kwargs: SimpleNamespace()
    )

    tracker = {"allocated_bytes": 0.0, "reserved_bytes": 0.0}
    result = measure_method_with_peak_memory(
        "ddtree_tb16", device="cuda:0", peak_memory_tracker=tracker
    )

    assert result.peak_allocated_gib == pytest.approx(1_000_000_000 / 1024**3)
    assert result.peak_reserved_gib == pytest.approx(1_500_000_000 / 1024**3)
    assert tracker["allocated_bytes"] == 9_000_000_000
    assert tracker["reserved_bytes"] == 9_500_000_000


def test_measure_method_with_peak_memory_forwards_kwargs_to_run_method(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_cuda = _FakeCudaPeakMemory(
        allocated_readings=[0, 0],
        reserved_readings=[0, 0],
    )
    monkeypatch.setattr(
        benchmark.torch.cuda,
        "reset_peak_memory_stats",
        fake_cuda.reset_peak_memory_stats,
    )
    monkeypatch.setattr(
        benchmark.torch.cuda, "max_memory_allocated", fake_cuda.max_memory_allocated
    )
    monkeypatch.setattr(
        benchmark.torch.cuda, "max_memory_reserved", fake_cuda.max_memory_reserved
    )

    captured_kwargs: dict[str, object] = {}

    def fake_run_method(method_key: str, **kwargs: object) -> SimpleNamespace:
        captured_kwargs["method_key"] = method_key
        captured_kwargs.update(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr(benchmark, "run_method", fake_run_method)

    measure_method_with_peak_memory(
        "dflash2",
        device="cuda:0",
        peak_memory_tracker={"allocated_bytes": 0.0, "reserved_bytes": 0.0},
        prompt_id="prompt-abc",
        max_new_tokens=32,
    )

    assert captured_kwargs["method_key"] == "dflash2"
    assert captured_kwargs["prompt_id"] == "prompt-abc"
    assert captured_kwargs["max_new_tokens"] == 32


# ---------------------------------------------------------------------------
# apply_baseline_comparison
# ---------------------------------------------------------------------------


def _result_with_output(token_ids: list[int]) -> SimpleNamespace:
    result = SimpleNamespace(output_ids=torch.tensor([token_ids]))
    result.repetitions = [result]
    return result


def test_apply_baseline_comparison_covers_original_family_methods() -> None:
    """Raw DFlash and original DDTree (not just DFlash2 variants) must get
    matches_baseline so Step 9's correctness.csv can report exact-output
    match rates for the whole method matrix."""
    baseline = _result_with_output([1, 2, 3])
    matching_dflash = _result_with_output([1, 2, 3])
    matching_dflash.repetitions = [matching_dflash]
    diverging_ddtree = _result_with_output([1, 2, 9])
    diverging_ddtree.repetitions = [diverging_ddtree]
    response = {
        "baseline": baseline,
        "dflash": matching_dflash,
        "ddtree_tb16": diverging_ddtree,
    }

    apply_baseline_comparison(
        response,
        ["dflash", "ddtree_tb16"],
        comparable_to_baseline=True,
        dataset_index=0,
    )

    assert matching_dflash.matches_baseline is True
    assert diverging_ddtree.matches_baseline is False


def test_apply_baseline_comparison_covers_every_repetition() -> None:
    baseline = _result_with_output([1, 2, 3])
    baseline_second = _result_with_output([4, 5, 6])
    baseline.repetitions = [baseline, baseline_second]
    reps = [_result_with_output([1, 2, 3]), _result_with_output([1, 2, 3])]
    designated = reps[0]
    designated.repetitions = reps
    response = {
        "baseline": baseline,
        "dflash2": designated,
    }

    apply_baseline_comparison(
        response,
        ["dflash2"],
        comparable_to_baseline=True,
        dataset_index=0,
    )

    assert reps[0].matches_baseline is True
    assert reps[1].matches_baseline is False


def test_apply_baseline_comparison_skips_when_not_comparable() -> None:
    """Native-trajectory turns after the first must not get a (meaningless)
    baseline comparison attached."""
    rep = _result_with_output([1, 2, 3])
    designated = rep
    designated.repetitions = [rep]
    response = {
        "baseline": _result_with_output([1, 2, 3]),
        "dflash": designated,
    }

    apply_baseline_comparison(
        response,
        ["dflash"],
        comparable_to_baseline=False,
        dataset_index=0,
    )

    assert not hasattr(rep, "matches_baseline")
