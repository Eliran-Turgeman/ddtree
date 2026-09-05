from dataclasses import replace
from pathlib import Path

import pytest
import torch

from ddtree import compile_generic_tree_for_verifier
from dflash2_tree import (
    DFLASH2_PAIRWISE_K16,
    DFLASH2_UNARY_K16,
    DFLASH2_UNARY_K32,
    DFLASH2_UNARY_K64,
    annotate_candidate_diagnostics,
    build_dflash2_verifier_tree,
    proposal_to_lattice,
)
from model.dflash2 import DFlash2Proposal
from offline_dflash2_trees import (
    PAIRWISE_MASS_PRESERVING,
    UNARY_FULL_MASS,
    build_best_first_tree,
    build_scorers,
    validate_lattice_tensors,
)


TRACE_PATH = Path(
    "traces/2026-09-03_dflash2_a100-40gb_gsm8k-32/"
    "gsm8k__Qwen_Qwen3-4B__"
    "mgoin_Qwen3-4B-speculator.dflash2__seed0__traces.pt"
)


def frozen_rounds() -> list[dict]:
    artifact = torch.load(
        TRACE_PATH,
        map_location="cpu",
        weights_only=False,
    )
    return [
        trace_round
        for prompt in artifact["prompts"][:2]
        for trace_round in prompt["rounds"][:2]
    ]


@pytest.mark.parametrize(
    ("online_method", "offline_method"),
    [
        (DFLASH2_UNARY_K16, UNARY_FULL_MASS),
        (DFLASH2_PAIRWISE_K16, PAIRWISE_MASS_PRESERVING),
    ],
)
@pytest.mark.parametrize("budget", [8, 16, 32, 64])
def test_online_tree_matches_offline_frozen_trace(
    online_method: str,
    offline_method: str,
    budget: int,
) -> None:
    for trace_round in frozen_rounds():
        offline_nodes = build_best_first_tree(
            trace_round["candidate_token_ids"],
            build_scorers(trace_round)[offline_method],
            budget,
        )
        online_nodes, *compiled = build_dflash2_verifier_tree(
            trace_round,
            online_method,
            budget,
        )

        assert online_nodes == offline_nodes
        token_ids, depths, parents, child_maps, visibility = compiled
        assert token_ids.tolist() == [
            node.token_id for node in offline_nodes
        ]
        assert depths.tolist() == [
            node.depth for node in offline_nodes
        ]
        assert parents == [
            -1,
            *[
                0 if node.parent == -1 else node.parent + 1
                for node in offline_nodes
            ],
        ]
        expected_length = len(offline_nodes) + 1
        assert len(child_maps) == expected_length
        assert visibility.shape == (
            expected_length,
            expected_length,
        )


def test_proposal_adapter_preserves_frozen_lattice() -> None:
    trace_round = frozen_rounds()[0]
    candidate_ids = trace_round["candidate_token_ids"].unsqueeze(0)
    unary_scores = trace_round["candidate_unary_logits"].unsqueeze(0)
    proposal = DFlash2Proposal(
        token_ids=trace_round[
            "selected_draft_token_ids"
        ].unsqueeze(0),
        selected_candidate_indices=trace_round[
            "selected_candidate_indices"
        ].unsqueeze(0),
        candidate_ids=candidate_ids,
        unary_scores=unary_scores,
        unary_logsumexp=trace_round["unary_logsumexp"].unsqueeze(0),
        anchor_pairwise_corrections=trace_round[
            "anchor_pairwise_corrections"
        ].unsqueeze(0),
        pairwise_corrections=trace_round[
            "pairwise_corrections"
        ].unsqueeze(0),
        anchor_final_scores=trace_round[
            "anchor_final_scores"
        ].unsqueeze(0),
        pairwise_final_scores=trace_round[
            "pairwise_final_scores"
        ].unsqueeze(0),
        corrected_scores=torch.empty(1, 7, 16),
    )

    lattice = proposal_to_lattice(proposal)

    for key in (
        "candidate_token_ids",
        "candidate_unary_logits",
        "unary_logsumexp",
        "anchor_final_scores",
        "pairwise_final_scores",
    ):
        assert torch.equal(lattice[key], trace_round[key])


def make_wide_proposal() -> tuple[DFlash2Proposal, torch.Tensor]:
    full_unary_logits = torch.arange(
        7 * 80,
        dtype=torch.float32,
    ).reshape(1, 7, 80)
    unary_scores, candidate_ids = full_unary_logits.topk(16, dim=-1)
    proposal = DFlash2Proposal(
        token_ids=candidate_ids[:, :, 0],
        selected_candidate_indices=torch.zeros(
            1,
            7,
            dtype=torch.long,
        ),
        candidate_ids=candidate_ids,
        unary_scores=unary_scores,
        unary_logsumexp=torch.logsumexp(
            full_unary_logits.float(),
            dim=-1,
        ),
        anchor_pairwise_corrections=torch.empty(1, 16),
        pairwise_corrections=torch.empty(1, 6, 16, 16),
        anchor_final_scores=torch.zeros(1, 16),
        pairwise_final_scores=torch.zeros(1, 6, 16, 16),
        corrected_scores=torch.empty(1, 7, 16),
        full_unary_logits=full_unary_logits,
    )
    return proposal, full_unary_logits


def test_wide_unary_lattices_use_true_nested_top_k() -> None:
    proposal, full_unary_logits = make_wide_proposal()
    lattices = {
        candidate_count: proposal_to_lattice(proposal, candidate_count)
        for candidate_count in (16, 32, 64)
    }

    for candidate_count, lattice in lattices.items():
        expected_scores, expected_ids = full_unary_logits.topk(
            candidate_count,
            dim=-1,
        )
        assert torch.equal(
            lattice["candidate_token_ids"],
            expected_ids[0],
        )
        assert torch.equal(
            lattice["candidate_unary_logits"],
            expected_scores[0],
        )
        assert torch.equal(
            lattice["unary_logsumexp"],
            proposal.unary_logsumexp[0],
        )

    assert torch.equal(
        lattices[32]["candidate_token_ids"][:, :16],
        lattices[16]["candidate_token_ids"],
    )
    assert torch.equal(
        lattices[64]["candidate_token_ids"][:, :32],
        lattices[32]["candidate_token_ids"],
    )


@pytest.mark.parametrize(
    ("method", "candidate_count"),
    [
        (DFLASH2_UNARY_K16, 16),
        (DFLASH2_UNARY_K32, 32),
        (DFLASH2_UNARY_K64, 64),
    ],
)
def test_wide_unary_budget_and_verifier_invariants(
    method: str,
    candidate_count: int,
) -> None:
    proposal, _ = make_wide_proposal()
    lattice = proposal_to_lattice(proposal, candidate_count)

    nodes, token_ids, depths, parents, child_maps, visibility = (
        build_dflash2_verifier_tree(
            lattice,
            method,
            budget=32,
        )
    )

    assert len(nodes) == 32
    assert token_ids.shape == (32,)
    assert depths.shape == (32,)
    assert parents[0] == -1
    assert len(child_maps) == 33
    assert visibility.shape == (33, 33)
    for node_index, node in enumerate(nodes):
        if node.parent != -1:
            assert node.parent < node_index
            assert nodes[node.parent].depth == node.depth - 1


def test_candidate_diagnostics_classify_failures_and_censoring() -> None:
    metrics = [
        {"matched_draft_tokens": 1},
        {"matched_draft_tokens": 0},
        {"matched_draft_tokens": 2},
    ]
    candidate_ids = torch.tensor(
        [
            [10, 11],
            [20, 21],
            [30, 31],
            [40, 41],
            [50, 51],
            [60, 61],
            [70, 71],
        ]
    )
    annotate_candidate_diagnostics(
        metrics,
        [candidate_ids, candidate_ids, candidate_ids],
        [0, 2, 6],
        torch.tensor([10, 21, 99, 40, 50, 60, 70, 10]),
    )

    assert metrics[0]["failure_type"] == "ranking_budget_failure"
    assert metrics[0]["target_rank_depth_2"] == 2
    assert metrics[1]["failure_type"] == "candidate_failure"
    assert metrics[1]["target_rank_depth_1"] is None
    assert metrics[2]["failure_type"] == "censored"
    assert metrics[2]["target_available_depth"] == 2


def test_lattice_shape_and_retained_mass_assertions() -> None:
    trace_round = frozen_rounds()[0]
    validate_lattice_tensors(
        trace_round,
        expected_depth=7,
        expected_candidate_count=16,
    )

    malformed = dict(trace_round)
    malformed["pairwise_final_scores"] = trace_round[
        "pairwise_final_scores"
    ][:-1]
    with pytest.raises(ValueError, match="pairwise_final_scores"):
        validate_lattice_tensors(malformed)

    invalid_mass = dict(trace_round)
    invalid_mass["unary_logsumexp"] = (
        torch.logsumexp(
            trace_round["candidate_unary_logits"].float(),
            dim=-1,
        )
        - 1.0
    )
    with pytest.raises(
        ValueError,
        match="probability mass exceeds one",
    ):
        validate_lattice_tensors(invalid_mass)


def test_generic_tree_compiler_rejects_invalid_parent_order() -> None:
    trace_round = frozen_rounds()[0]
    nodes = build_best_first_tree(
        trace_round["candidate_token_ids"],
        build_scorers(trace_round)[UNARY_FULL_MASS],
        budget=8,
    )
    malformed = list(nodes)
    malformed[1] = replace(malformed[1], parent=1)

    with pytest.raises(ValueError, match="invalid parent"):
        compile_generic_tree_for_verifier(
            malformed,
            budget=8,
            depth_limit=7,
        )
