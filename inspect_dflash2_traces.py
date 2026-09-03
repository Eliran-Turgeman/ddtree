#!/usr/bin/env python3

import argparse
from pathlib import Path
from statistics import mean

import torch


TRACE_FORMAT = "ddtree.dflash2_trace"


def load_trace(path: Path) -> dict:
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    if artifact.get("format") != TRACE_FORMAT:
        raise ValueError(
            f"{path} is not a {TRACE_FORMAT} artifact"
        )
    if not artifact.get("prompts"):
        raise ValueError(f"{path} contains no prompts")
    return artifact


def reconstruct_selected_path(
    trace_round: dict,
) -> tuple[torch.Tensor, torch.Tensor]:
    candidate_ids = trace_round["candidate_token_ids"].long()
    anchor_scores = trace_round["anchor_final_scores"]
    pairwise_scores = trace_round["pairwise_final_scores"]

    selected_indices = [int(anchor_scores.argmax().item())]
    for position in range(1, candidate_ids.shape[0]):
        predecessor_index = selected_indices[-1]
        position_scores = pairwise_scores[
            position - 1,
            predecessor_index,
        ]
        selected_indices.append(int(position_scores.argmax().item()))

    index_tensor = torch.tensor(selected_indices, dtype=torch.long)
    token_tensor = candidate_ids.gather(
        1,
        index_tensor.unsqueeze(-1),
    ).squeeze(-1)
    return index_tensor, token_tensor


def validate_round(trace_round: dict) -> dict[str, bool]:
    candidate_ids = trace_round["candidate_token_ids"]
    unary_scores = trace_round["candidate_unary_logits"]
    unary_logsumexp = trace_round["unary_logsumexp"]
    anchor_corrections = trace_round["anchor_pairwise_corrections"]
    pairwise_corrections = trace_round["pairwise_corrections"]
    anchor_final_scores = trace_round["anchor_final_scores"]
    pairwise_final_scores = trace_round["pairwise_final_scores"]
    selected_ids = trace_round["selected_draft_token_ids"].long()
    selected_indices = trace_round["selected_candidate_indices"].long()

    draft_length, candidate_count = candidate_ids.shape
    shapes_valid = (
        unary_scores.shape == (draft_length, candidate_count)
        and unary_logsumexp.shape == (draft_length,)
        and anchor_corrections.shape == (candidate_count,)
        and pairwise_corrections.shape
        == (
            max(draft_length - 1, 0),
            candidate_count,
            candidate_count,
        )
        and anchor_final_scores.shape == (candidate_count,)
        and pairwise_final_scores.shape
        == (
            max(draft_length - 1, 0),
            candidate_count,
            candidate_count,
        )
        and selected_ids.shape == (draft_length,)
        and selected_indices.shape == (draft_length,)
    )

    scores_descend = bool(
        torch.all(unary_scores[:, :-1] >= unary_scores[:, 1:])
    )
    candidates_unique = all(
        torch.unique(position_ids).numel() == candidate_count
        for position_ids in candidate_ids
    )
    logsumexp_valid = bool(
        torch.all(
            unary_logsumexp.float()
            >= unary_scores.float().amax(dim=-1) - 1e-5
        )
    )

    expected_anchor_scores = (
        unary_scores[0].float() + anchor_corrections.float()
    )
    expected_pairwise_scores = (
        unary_scores[1:].float().unsqueeze(-2)
        + pairwise_corrections.float()
    )
    score_consistent = bool(
        torch.allclose(
            anchor_final_scores.float(),
            expected_anchor_scores,
            rtol=1e-2,
            atol=1e-2,
        )
        and torch.allclose(
            pairwise_final_scores.float(),
            expected_pairwise_scores,
            rtol=1e-2,
            atol=1e-2,
        )
    )

    reconstructed_indices, reconstructed_ids = reconstruct_selected_path(
        trace_round
    )
    path_reproduced = bool(
        torch.equal(reconstructed_indices, selected_indices)
        and torch.equal(reconstructed_ids, selected_ids)
    )

    verifier_ids = trace_round["verifier_token_ids"].long()
    directly_observed_mask = trace_round[
        "directly_observed_target_mask"
    ].bool()
    realized_ids = trace_round[
        "realized_continuation_token_ids"
    ].long()
    direct_count = min(
        verifier_ids.numel(),
        directly_observed_mask.numel(),
        realized_ids.numel(),
    )
    direct_matches_realized = bool(
        torch.equal(
            verifier_ids[:direct_count][
                directly_observed_mask[:direct_count]
            ],
            realized_ids[:direct_count][
                directly_observed_mask[:direct_count]
            ],
        )
    )

    return {
        "shapes_valid": shapes_valid,
        "candidate_consistent": (
            shapes_valid
            and scores_descend
            and candidates_unique
            and logsumexp_valid
        ),
        "score_consistent": score_consistent,
        "path_reproduced": path_reproduced,
        "direct_matches_realized": direct_matches_realized,
    }


def analyze_trace(artifact: dict) -> dict:
    trace_rounds = [
        trace_round
        for prompt in artifact["prompts"]
        for trace_round in prompt["rounds"]
    ]
    if not trace_rounds:
        raise ValueError("trace artifact contains no decoding rounds")

    checks = [validate_round(trace_round) for trace_round in trace_rounds]
    max_depth = max(
        trace_round["candidate_token_ids"].shape[0]
        for trace_round in trace_rounds
    )
    recall_hits = [0] * max_depth
    recall_totals = [0] * max_depth
    conditional_hits = [0] * max_depth
    conditional_totals = [0] * max_depth

    for trace_round in trace_rounds:
        candidates = trace_round["candidate_token_ids"].long()
        realized = trace_round[
            "realized_continuation_token_ids"
        ].long()
        verifier = trace_round["verifier_token_ids"].long()
        direct_mask = trace_round[
            "directly_observed_target_mask"
        ].bool()

        for depth in range(min(candidates.shape[0], realized.numel())):
            recall_totals[depth] += 1
            recall_hits[depth] += int(
                bool(torch.isin(realized[depth], candidates[depth]))
            )

        direct_depth = min(
            candidates.shape[0],
            verifier.numel(),
            direct_mask.numel(),
        )
        for depth in range(direct_depth):
            if not direct_mask[depth]:
                continue
            conditional_totals[depth] += 1
            conditional_hits[depth] += int(
                bool(torch.isin(verifier[depth], candidates[depth]))
            )

    shape_signatures = sorted(
        {
            (
                tuple(trace_round["candidate_token_ids"].shape),
                tuple(trace_round["pairwise_corrections"].shape),
            )
            for trace_round in trace_rounds
        }
    )
    return {
        "prompt_count": len(artifact["prompts"]),
        "round_count": len(trace_rounds),
        "shape_signatures": shape_signatures,
        "mean_accepted_draft_tokens": mean(
            trace_round["accepted_draft_tokens"]
            for trace_round in trace_rounds
        ),
        "mean_verifier_matched_draft_tokens": mean(
            trace_round["verifier_matched_draft_tokens"]
            for trace_round in trace_rounds
        ),
        "mean_committed_tokens": mean(
            trace_round["committed_tokens_this_round"]
            for trace_round in trace_rounds
        ),
        "candidate_consistency_rate": mean(
            check["candidate_consistent"] for check in checks
        ),
        "score_consistency_rate": mean(
            check["score_consistent"] for check in checks
        ),
        "path_reproduction_rate": mean(
            check["path_reproduced"] for check in checks
        ),
        "direct_target_agreement_rate": mean(
            check["direct_matches_realized"] for check in checks
        ),
        "recall_hits": recall_hits,
        "recall_totals": recall_totals,
        "conditional_hits": conditional_hits,
        "conditional_totals": conditional_totals,
    }


def format_rate(hits: int, total: int) -> str:
    if total == 0:
        return "N/A"
    return f"{hits / total:.2%}"


def print_report(path: Path, artifact: dict, analysis: dict) -> None:
    total_bytes = path.stat().st_size
    round_count = analysis["round_count"]
    print(f"Trace: {path}")
    print(f"Prompts: {analysis['prompt_count']}")
    print(f"Decoding rounds: {round_count}")
    print(f"Tensor shape signatures: {analysis['shape_signatures']}")
    print(
        "Mean accepted draft tokens: "
        f"{analysis['mean_accepted_draft_tokens']:.3f}"
    )
    print(
        "Mean raw verifier-matched draft tokens: "
        f"{analysis['mean_verifier_matched_draft_tokens']:.3f}"
    )
    print(
        "Mean committed tokens/round: "
        f"{analysis['mean_committed_tokens']:.3f}"
    )
    print(
        "Candidate consistency: "
        f"{analysis['candidate_consistency_rate']:.2%}"
    )
    print(
        "Pairwise score consistency: "
        f"{analysis['score_consistency_rate']:.2%}"
    )
    print(
        "DFlash2 path reproduction: "
        f"{analysis['path_reproduction_rate']:.2%}"
    )
    print(
        "Direct verifier/realized-token agreement: "
        f"{analysis['direct_target_agreement_rate']:.2%}"
    )
    print(
        f"Trace size: {total_bytes / (1024 ** 2):.2f} MiB "
        f"({total_bytes / round_count:.0f} bytes/round)"
    )
    print()
    print(
        f"{'Depth':>5} {'Recall@16':>12} {'Recall N':>10} "
        f"{'Conditional@16':>16} {'Eligible N_d':>14}"
    )
    print("-" * 65)
    for depth, (
        recall_hits,
        recall_total,
        conditional_hits,
        conditional_total,
    ) in enumerate(
        zip(
            analysis["recall_hits"],
            analysis["recall_totals"],
            analysis["conditional_hits"],
            analysis["conditional_totals"],
        ),
        start=1,
    ):
        print(
            f"{depth:>5} "
            f"{format_rate(recall_hits, recall_total):>12} "
            f"{recall_total:>10} "
            f"{format_rate(conditional_hits, conditional_total):>16} "
            f"{conditional_total:>14}"
        )
    print()
    print(
        "Recall@16 uses the eventual committed DFlash2 continuation. "
        "Conditional Recall@16 uses only target tokens directly observed "
        "while the selected prefix remains matched."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect and validate a DFlash2 trace artifact."
    )
    parser.add_argument("trace_path", type=Path)
    args = parser.parse_args()

    artifact = load_trace(args.trace_path)
    analysis = analyze_trace(artifact)
    print_report(args.trace_path, artifact, analysis)


if __name__ == "__main__":
    main()
