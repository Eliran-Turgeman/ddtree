from dataclasses import replace
from pathlib import Path

import pytest
import torch

from ddtree import compile_generic_tree_for_verifier
from dflash2_tree import (
    DFLASH2_PAIRWISE_K16,
    DFLASH2_UNARY_K16,
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
