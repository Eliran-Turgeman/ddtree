from types import SimpleNamespace

import pytest
import torch

from analyze_step8_protocol import (
    DRAFT_REVISION,
    TARGET_REVISION,
    collect_cluster_metrics,
    controlled_allocation_values,
    expected_method_keys,
    load_and_validate,
)
from benchmark import parse_dflash2_tree_configs
from dflash2_tree import (
    DFLASH2_ORIGINAL_DDTREE,
    DFLASH2_PAIRWISE_K16,
    DFLASH2_UNARY_K16,
)


def test_parse_protocol_tree_configs() -> None:
    value = (
        f"{DFLASH2_ORIGINAL_DDTREE}:7,16,32,64;"
        f"{DFLASH2_PAIRWISE_K16}:7,16,32,64;"
        f"{DFLASH2_UNARY_K16}:32,64"
    )

    configs = parse_dflash2_tree_configs(value)

    assert configs == [
        (DFLASH2_ORIGINAL_DDTREE, 7),
        (DFLASH2_ORIGINAL_DDTREE, 16),
        (DFLASH2_ORIGINAL_DDTREE, 32),
        (DFLASH2_ORIGINAL_DDTREE, 64),
        (DFLASH2_PAIRWISE_K16, 7),
        (DFLASH2_PAIRWISE_K16, 16),
        (DFLASH2_PAIRWISE_K16, 32),
        (DFLASH2_PAIRWISE_K16, 64),
        (DFLASH2_UNARY_K16, 32),
        (DFLASH2_UNARY_K16, 64),
    ]


@pytest.mark.parametrize(
    "value",
    [
        "",
        f"{DFLASH2_ORIGINAL_DDTREE}:0",
        f"{DFLASH2_ORIGINAL_DDTREE}:seven",
        f"{DFLASH2_ORIGINAL_DDTREE}:7;{DFLASH2_ORIGINAL_DDTREE}:7",
        "unknown:7",
        DFLASH2_ORIGINAL_DDTREE,
    ],
)
def test_parse_protocol_tree_configs_rejects_invalid_values(
    value: str,
) -> None:
    with pytest.raises(ValueError):
        parse_dflash2_tree_configs(value)


def make_result(
    method_key: str,
    *,
    matched: float,
) -> SimpleNamespace:
    is_greedy = method_key == "dflash2"
    budget = 7 if is_greedy else int(method_key.rsplit("_tb", 1)[1])
    candidate_count = budget if "original_ddtree" in method_key else 16
    metric = {
        "prompt_id": f"gsm8k:selected:0:turn-0:{method_key}:abc",
        "matched_draft_tokens": matched,
        "committed_tokens_this_round": matched + 1,
        "tree_build_latency_ms": 0.0,
        "target_verify_latency_ms": 1.0,
    }
    if not is_greedy:
        metric.update(
            {
                "tree_node_count": budget,
                "candidate_count": candidate_count,
                "tree": [],
                "allocation_lattice": {},
            }
        )
    return SimpleNamespace(
        round_metrics=[metric],
        decode_rounds=1,
        time_per_output_token=0.1,
        trace_rounds=[] if is_greedy else None,
    )


def test_protocol_artifact_validation_and_clustering(tmp_path) -> None:
    commit = "a" * 40
    responses = {
        key: make_result(
            key,
            matched=5.0 if "pairwise" in key else 4.0,
        )
        for key in expected_method_keys()
    }
    tree_keys = set(expected_method_keys()) - {"dflash2"}
    artifact = {
        "args": {
            "dataset": "gsm8k",
            "temperature": 0.0,
            "max_new_tokens": 32,
            "max_samples": 1,
            "dflash2_tree_configs": (
                "dflash2_original_ddtree:7;dflash2_pairwise_k16:7"
            ),
        },
        "repository": {"commit": commit, "dirty": False},
        "target_revision": TARGET_REVISION,
        "draft_revision": DRAFT_REVISION,
        "trajectory_mode": "controlled_shared",
        "completed_dataset_indices": [0],
        "responses": [responses],
        "tree_exact_token_matches": {
            key: {"count": 1, "total": 1} for key in tree_keys
        },
    }
    path = tmp_path / "smoke.pt"
    torch.save(artifact, path)

    run = load_and_validate(
        path,
        commit,
        allow_partial=True,
        expected_trajectory_mode="controlled_shared",
    )
    clustered = collect_cluster_metrics(run)

    assert clustered["dflash2_pairwise_k16_tb16"][0]["matched"] == 5.0
    assert clustered["dflash2_original_ddtree_tb16"][0]["matched"] == 4.0


def test_controlled_allocation_uses_shared_greedy_trace() -> None:
    candidate_ids = torch.arange(64).repeat(7, 1)
    unary_logits = -torch.arange(64, dtype=torch.float32).repeat(7, 1)
    top16_ids = candidate_ids[:, :16]
    top16_logits = unary_logits[:, :16]
    greedy = make_result("dflash2", matched=7.0)
    greedy.trace_rounds = [
        {
            "unary_top64_token_ids": candidate_ids,
            "unary_top64_logits": unary_logits,
            "unary_logsumexp": torch.logsumexp(unary_logits, dim=-1),
            "candidate_token_ids": top16_ids,
            "candidate_unary_logits": top16_logits,
            "anchor_final_scores": top16_logits[0],
            "pairwise_final_scores": top16_logits[1:].unsqueeze(1).expand(6, 16, 16),
            "selected_draft_token_ids": torch.zeros(7, dtype=torch.long),
            "realized_continuation_token_ids": torch.zeros(
                7,
                dtype=torch.long,
            ),
        }
    ]
    values = controlled_allocation_values({"responses": [{"dflash2": greedy}]})

    assert values["dflash2"][0] == 7.0
    for budget in (7, 16, 32, 64):
        assert (
            values[f"dflash2_original_ddtree_tb{budget}"][0]
            == values[f"dflash2_pairwise_k16_tb{budget}"][0]
        )
