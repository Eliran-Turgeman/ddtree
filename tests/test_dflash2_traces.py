import torch

from inspect_dflash2_traces import analyze_trace, validate_round
from model.dflash2 import CandidateSelector


def build_trace_round() -> dict:
    selector = CandidateSelector(
        vocab_size=4,
        hidden_size=1,
        rank=1,
        top_k=2,
    )
    with torch.no_grad():
        selector.predecessor_codebook.copy_(
            torch.tensor([[2.0], [3.0], [5.0], [7.0]])
        )
        selector.successor_codebook.copy_(
            torch.tensor([[11.0], [13.0], [17.0], [19.0]])
        )
        selector.hidden_projection.weight.fill_(1.0)

    proposal = selector.select_path(
        torch.tensor(
            [[[0.0, 5.0, 4.0, 1.0], [3.0, 1.0, 2.0, 6.0]]]
        ),
        torch.tensor([[[7.0], [11.0]]]),
        torch.tensor([0]),
        collect_lattice=True,
    )
    return {
        "candidate_token_ids": proposal.candidate_ids[0],
        "candidate_unary_logits": proposal.unary_scores[0],
        "unary_logsumexp": proposal.unary_logsumexp[0],
        "anchor_pairwise_corrections": (
            proposal.anchor_pairwise_corrections[0]
        ),
        "pairwise_corrections": proposal.pairwise_corrections[0],
        "anchor_final_scores": proposal.anchor_final_scores[0],
        "pairwise_final_scores": proposal.pairwise_final_scores[0],
        "selected_draft_token_ids": proposal.token_ids[0],
        "selected_candidate_indices": (
            proposal.selected_candidate_indices[0]
        ),
        "verifier_token_ids": torch.tensor([2, 0]),
        "directly_observed_target_mask": torch.tensor([True, True]),
        "realized_continuation_token_ids": torch.tensor([2, 0]),
        "verifier_matched_draft_tokens": 1,
        "accepted_draft_tokens": 1,
        "committed_tokens_this_round": 2,
    }


def test_trace_round_reconstructs_selected_path() -> None:
    checks = validate_round(build_trace_round())

    assert checks == {
        "shapes_valid": True,
        "candidate_consistent": True,
        "score_consistent": True,
        "path_reproduced": True,
        "direct_matches_realized": True,
    }


def test_trace_analysis_computes_recall() -> None:
    artifact = {
        "format": "ddtree.dflash2_trace",
        "format_version": 1,
        "metadata": {},
        "prompts": [
            {
                "prompt_id": "gsm8k:test:0",
                "rounds": [build_trace_round()],
            }
        ],
    }

    analysis = analyze_trace(artifact)

    assert analysis["prompt_count"] == 1
    assert analysis["round_count"] == 1
    assert analysis["recall_hits"] == [1, 1]
    assert analysis["recall_totals"] == [1, 1]
    assert analysis["conditional_hits"] == [1, 1]
    assert analysis["conditional_totals"] == [1, 1]
    assert analysis["path_reproduction_rate"] == 1
