import torch
import numpy as np
import itertools
import pytest

from analyze_dflash2_trees import (
    direct_prefix_length,
    right_censored_quantile,
)
from offline_dflash2_trees import (
    PAIRWISE_AFTER_ROOT,
    PAIRWISE_MASS_PRESERVING,
    UNARY_FULL_MASS,
    UNARY_TRUNCATED,
    UnaryScorer,
    build_best_first_tree,
    build_scorers,
    greedy_path_matched_tokens,
    matched_draft_tokens,
    prefix_entry_budgets,
    target_candidate_path,
)


def make_trace_round() -> dict:
    unary_logits = torch.tensor(
        [
            [5.0, 4.0],
            [3.0, 2.0],
            [1.0, 0.0],
        ]
    )
    unary_logsumexp = torch.logsumexp(
        torch.tensor(
            [
                [5.0, 4.0, 0.0],
                [3.0, 2.0, 0.0],
                [1.0, 0.0, -2.0],
            ]
        ),
        dim=-1,
    )
    anchor_scores = torch.tensor([0.0, 2.0])
    pairwise_scores = torch.tensor(
        [
            [[4.0, 0.0], [0.0, 5.0]],
            [[3.0, 0.0], [0.0, 4.0]],
        ]
    )
    return {
        "candidate_token_ids": torch.tensor(
            [[10, 11], [20, 21], [30, 31]]
        ),
        "candidate_unary_logits": unary_logits,
        "unary_logsumexp": unary_logsumexp,
        "anchor_final_scores": anchor_scores,
        "pairwise_final_scores": pairwise_scores,
    }


def test_best_first_tree_is_prefix_closed() -> None:
    trace_round = make_trace_round()
    scorer = build_scorers(trace_round)[UNARY_FULL_MASS]
    nodes = build_best_first_tree(
        trace_round["candidate_token_ids"],
        scorer,
        budget=7,
    )

    for node_index, node in enumerate(nodes):
        if node.parent == -1:
            assert node.depth == 1
        else:
            assert node.parent < node_index
            assert nodes[node.parent].depth == node.depth - 1
            assert node.path_candidate_indices[:-1] == (
                nodes[node.parent].path_candidate_indices
            )


def test_best_first_tree_clamps_tolerated_positive_roundoff() -> None:
    scorer = UnaryScorer(
        UNARY_FULL_MASS,
        torch.tensor(
            [
                [1.3113024e-6, -1.0],
                [1.0e-6, -1.0],
            ]
        ),
    )
    nodes = build_best_first_tree(
        torch.tensor([[10, 11], [20, 21]]),
        scorer,
        budget=2,
    )

    assert nodes[0].log_prefix_score == 0.0
    assert nodes[1].log_prefix_score == 0.0


def test_best_first_tree_rejects_material_positive_extension() -> None:
    scorer = UnaryScorer(
        UNARY_FULL_MASS,
        torch.tensor([[2.0e-5, -1.0]]),
    )

    with pytest.raises(ValueError, match="non-monotonic root"):
        build_best_first_tree(
            torch.tensor([[10, 11]]),
            scorer,
            budget=1,
        )


def test_pairwise_depth_alignment_uses_parent_candidate() -> None:
    trace_round = make_trace_round()
    scorer = build_scorers(trace_round)[
        PAIRWISE_MASS_PRESERVING
    ]
    nodes = build_best_first_tree(
        trace_round["candidate_token_ids"],
        scorer,
        budget=3,
    )

    assert nodes[0].token_id == 11
    assert nodes[0].candidate_index == 1
    assert nodes[1].token_id == 21
    assert nodes[1].parent == 0
    assert nodes[2].token_id == 31
    assert nodes[2].parent == 1


def test_target_prefix_entry_and_matching_depth() -> None:
    trace_round = make_trace_round()
    scorer = build_scorers(trace_round)[
        PAIRWISE_MASS_PRESERVING
    ]
    nodes = build_best_first_tree(
        trace_round["candidate_token_ids"],
        scorer,
        budget=8,
    )
    target_path, representable_depth = target_candidate_path(
        trace_round["candidate_token_ids"],
        torch.tensor([11, 21, 31]),
    )
    entry_budgets = prefix_entry_budgets(nodes, target_path)

    assert representable_depth == 3
    assert entry_budgets == [1, 2, 3]
    assert matched_draft_tokens(entry_budgets, budget=2) == 2
    assert matched_draft_tokens(entry_budgets, budget=3) == 3


def test_greedy_path_respects_prefix_match_and_budget() -> None:
    selected = torch.tensor([10, 20, 30, 40])
    target = torch.tensor([10, 20, 99, 40])

    assert greedy_path_matched_tokens(selected, target, budget=1) == 1
    assert greedy_path_matched_tokens(selected, target, budget=7) == 2


def test_right_censored_quantiles_are_not_reported_as_observed() -> None:
    observed = np.array([1, 2, 3, 4, 5, 6], dtype=np.float64)

    assert right_censored_quantile(observed, 4, 0.5, 256) == 5.5
    assert right_censored_quantile(observed, 4, 0.75, 256) == ">256"


def test_direct_observation_mask_must_be_a_contiguous_prefix() -> None:
    assert direct_prefix_length(
        torch.tensor([True, True, False, False])
    ) == 2
    with pytest.raises(ValueError, match="contiguous prefix"):
        direct_prefix_length(
            torch.tensor([True, False, True, False])
        )


def test_best_first_order_matches_brute_force_prefix_ranking() -> None:
    trace_round = make_trace_round()
    scorer = build_scorers(trace_round)[UNARY_FULL_MASS]
    nodes = build_best_first_tree(
        trace_round["candidate_token_ids"],
        scorer,
        budget=14,
    )

    all_prefixes = []
    for depth in range(1, 4):
        for path in itertools.product(range(2), repeat=depth):
            score = sum(
                scorer.extension_log_score(
                    depth_index,
                    path[depth_index - 1]
                    if depth_index > 0
                    else None,
                    candidate_index,
                )
                for depth_index, candidate_index in enumerate(path)
            )
            all_prefixes.append((score, path))
    expected = [
        path
        for _, path in sorted(
            all_prefixes,
            key=lambda item: (-item[0], item[1]),
        )
    ]

    assert [node.path_candidate_indices for node in nodes] == expected


def test_scorers_match_their_probability_definitions() -> None:
    trace_round = make_trace_round()
    scorers = build_scorers(trace_round)
    unary_logits = trace_round["candidate_unary_logits"].float()
    full_lse = trace_round["unary_logsumexp"].float()
    log_mass = torch.logsumexp(unary_logits, dim=-1) - full_lse

    assert scorers[UNARY_FULL_MASS].extension_log_score(
        1,
        0,
        1,
    ) == float(unary_logits[1, 1] - full_lse[1])
    assert scorers[UNARY_TRUNCATED].extension_log_score(
        1,
        0,
        1,
    ) == float(torch.log_softmax(unary_logits[1], dim=-1)[1])

    pairwise = scorers[PAIRWISE_MASS_PRESERVING]
    expected_depth_1 = (
        log_mass[0]
        + torch.log_softmax(
            trace_round["anchor_final_scores"].float(),
            dim=-1,
        )[1]
    )
    expected_depth_2 = (
        log_mass[1]
        + torch.log_softmax(
            trace_round["pairwise_final_scores"][0, 1].float(),
            dim=-1,
        )[0]
    )
    assert pairwise.extension_log_score(0, None, 1) == float(
        expected_depth_1
    )
    assert pairwise.extension_log_score(1, 1, 0) == float(
        expected_depth_2
    )

    after_root = scorers[PAIRWISE_AFTER_ROOT]
    assert after_root.extension_log_score(0, None, 1) == float(
        unary_logits[0, 1] - full_lse[0]
    )
    assert after_root.extension_log_score(1, 1, 0) == float(
        expected_depth_2
    )
