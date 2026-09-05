import pytest
import torch

from model.dflash2 import (
    CandidateSelector,
    DFlash2DraftModel,
    GroupedDynamicCausalConv,
    load_dflash2_draft_model,
)


def test_dynamic_convolution_starts_as_identity() -> None:
    convolution = GroupedDynamicCausalConv(
        hidden_size=4,
        block_size=3,
        kernel_size=2,
        group_size=2,
    )
    hidden_states = torch.randn(1, 3, 4)

    prepared, output_kernel = convolution.prepare(hidden_states)
    finished = convolution.finish(prepared, output_kernel)

    assert torch.equal(prepared, hidden_states)
    assert torch.equal(finished, hidden_states)


def test_candidate_selector_walks_unary_path_without_corrections() -> None:
    selector = CandidateSelector(
        vocab_size=5,
        hidden_size=2,
        rank=2,
        top_k=2,
    )
    with torch.no_grad():
        selector.predecessor_codebook.zero_()
        selector.successor_codebook.zero_()
        selector.hidden_projection.weight.zero_()

    unary_logits = torch.tensor(
        [[[1.0, 5.0, 3.0, 2.0, 0.0], [4.0, 0.0, 2.0, 1.0, 3.0]]]
    )
    proposal = selector.select_path(
        unary_logits,
        torch.zeros(1, 2, 2),
        torch.tensor([0]),
        collect_lattice=True,
    )

    assert proposal.token_ids.tolist() == [[1, 0]]
    assert proposal.selected_candidate_indices.tolist() == [[0, 0]]
    assert proposal.candidate_ids.shape == (1, 2, 2)
    assert proposal.unary_scores.shape == (1, 2, 2)
    assert proposal.unary_logsumexp.shape == (1, 2)
    assert proposal.anchor_pairwise_corrections.shape == (1, 2)
    assert proposal.pairwise_corrections.shape == (1, 1, 2, 2)
    assert proposal.anchor_final_scores.shape == (1, 2)
    assert proposal.pairwise_final_scores.shape == (1, 1, 2, 2)
    assert torch.count_nonzero(proposal.anchor_pairwise_corrections) == 0
    assert torch.count_nonzero(proposal.pairwise_corrections) == 0
    assert proposal.corrected_scores.shape == (1, 2, 2)


def test_candidate_selector_exposes_raw_pairwise_corrections() -> None:
    selector = CandidateSelector(
        vocab_size=4,
        hidden_size=1,
        rank=1,
        top_k=2,
    )
    with torch.no_grad():
        selector.predecessor_codebook.copy_(torch.tensor([[2.0], [3.0], [5.0], [7.0]]))
        selector.successor_codebook.copy_(
            torch.tensor([[11.0], [13.0], [17.0], [19.0]])
        )
        selector.hidden_projection.weight.fill_(1.0)

    unary_logits = torch.tensor([[[0.0, 5.0, 4.0, 1.0], [3.0, 1.0, 2.0, 6.0]]])
    hidden_states = torch.tensor([[[7.0], [11.0]]])
    normal_proposal = selector.select_path(
        unary_logits,
        hidden_states,
        torch.tensor([0]),
    )
    proposal = selector.select_path(
        unary_logits,
        hidden_states,
        torch.tensor([0]),
        collect_lattice=True,
    )

    assert normal_proposal.anchor_pairwise_corrections is None
    assert normal_proposal.pairwise_corrections is None
    assert torch.equal(normal_proposal.token_ids, proposal.token_ids)
    assert torch.equal(
        normal_proposal.selected_candidate_indices,
        proposal.selected_candidate_indices,
    )
    assert torch.equal(
        normal_proposal.corrected_scores,
        proposal.corrected_scores,
    )
    assert proposal.candidate_ids.tolist() == [[[1, 2], [3, 0]]]
    assert proposal.selected_candidate_indices.tolist() == [[1, 0]]
    assert torch.equal(
        proposal.anchor_pairwise_corrections,
        torch.tensor([[182.0, 238.0]]),
    )
    assert torch.equal(
        proposal.pairwise_corrections,
        torch.tensor([[[[627.0, 363.0], [1045.0, 605.0]]]]),
    )
    assert torch.equal(
        proposal.corrected_scores,
        torch.tensor([[[187.0, 242.0], [1051.0, 608.0]]]),
    )


def test_checkpoint_config_adapter_preserves_dflash2_contract() -> None:
    config = DFlash2DraftModel._convert_checkpoint_config(
        {
            "transformer_layer_config": {
                "attention_bias": False,
                "attention_dropout": 0.0,
                "head_dim": 4,
                "hidden_act": "silu",
                "hidden_size": 8,
                "intermediate_size": 16,
                "layer_types": ["sliding_attention"],
                "max_position_embeddings": 128,
                "max_window_layers": 1,
                "num_attention_heads": 2,
                "num_hidden_layers": 1,
                "num_key_value_heads": 1,
                "rms_norm_eps": 1e-6,
                "rope_parameters": {
                    "rope_theta": 1_000_000,
                    "rope_type": "default",
                },
                "sliding_window": 64,
                "tie_word_embeddings": False,
                "use_cache": True,
                "use_sliding_window": True,
                "vocab_size": 32,
            },
            "aux_hidden_state_layer_ids": [1],
            "block_size": 8,
            "conv_group_size": 4,
            "conv_kernel_size": 2,
            "mask_token_id": 31,
            "sample_from_anchor": False,
            "selector_rank": 4,
            "selector_top_k": 2,
            "_commit_hash": "draft-checkpoint-sha",
        }
    )

    assert config.block_size == 8
    assert config.dflash_config == {
        "target_layer_ids": [1],
        "mask_token_id": 31,
    }
    assert config.rope_parameters["rope_theta"] == 1_000_000
    assert config.selector_top_k == 2
    assert config._commit_hash == "draft-checkpoint-sha"


def test_official_checkpoint_config_adapter() -> None:
    config = DFlash2DraftModel._convert_checkpoint_config(
        {
            "architectures": ["DFlash2DraftModel"],
            "hidden_size": 5120,
            "intermediate_size": 17408,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "num_hidden_layers": 5,
            "head_dim": 128,
            "vocab_size": 248320,
            "layer_types": ["sliding_attention"] * 5,
            "max_position_embeddings": 262144,
            "rms_norm_eps": 1e-6,
            "rope_parameters": {
                "rope_theta": 10_000_000,
                "rope_type": "default",
            },
            "sliding_window": 2048,
            "use_sliding_window": True,
            "dflash_config": {
                "block_size": 8,
                "conv_group_size": 16,
                "conv_kernel_size": 2,
                "mask_token_id": 248070,
                "selector_rank": 256,
                "selector_top_k": 16,
                "target_layer_ids": [5, 19, 33, 47, 61],
            },
            "_commit_hash": "official-draft-sha",
        }
    )

    assert config.hidden_size == 5120
    assert config.num_hidden_layers == 5
    assert config.block_size == 8
    assert config.conv_kernel_size == 2
    assert config.conv_group_size == 16
    assert config.selector_rank == 256
    assert config.selector_top_k == 16
    assert config.sample_from_anchor is False
    assert config.dflash_config == {
        "target_layer_ids": [5, 19, 33, 47, 61],
        "mask_token_id": 248070,
    }
    assert config._commit_hash == "official-draft-sha"


def test_official_checkpoint_config_requires_complete_dflash_config() -> None:
    with pytest.raises(
        ValueError,
        match="dflash_config is missing: target_layer_ids",
    ):
        DFlash2DraftModel._convert_checkpoint_config(
            {
                "hidden_size": 8,
                "num_attention_heads": 2,
                "num_hidden_layers": 1,
                "num_key_value_heads": 1,
                "vocab_size": 32,
                "dflash_config": {
                    "block_size": 8,
                    "conv_group_size": 4,
                    "conv_kernel_size": 2,
                    "mask_token_id": 31,
                    "selector_rank": 4,
                    "selector_top_k": 2,
                },
            }
        )


def test_official_checkpoint_shares_omitted_target_token_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    draft = type(
        "Draft",
        (),
        {
            "embed_tokens": torch.nn.Embedding(32, 8),
            "lm_head": torch.nn.Linear(8, 32, bias=False),
        },
    )()
    target_embeddings = torch.nn.Embedding(32, 8)
    target_lm_head = torch.nn.Linear(8, 32, bias=False)
    target = type(
        "Target",
        (),
        {
            "get_input_embeddings": lambda self: target_embeddings,
            "get_output_embeddings": lambda self: target_lm_head,
        },
    )()
    monkeypatch.setattr(
        DFlash2DraftModel,
        "from_pretrained",
        lambda *args, **kwargs: (
            draft,
            {
                "missing_keys": [
                    "embed_tokens.weight",
                    "lm_head.weight",
                ]
            },
        ),
    )

    loaded, loading_info = load_dflash2_draft_model(
        "official-draft",
        target=target,
    )

    assert loaded.embed_tokens is target_embeddings
    assert loaded.lm_head is target_lm_head
    assert loading_info["shared_target_weight_keys"] == [
        "embed_tokens.weight",
        "lm_head.weight",
    ]


@pytest.mark.parametrize(
    ("missing_keys", "message"),
    [
        (
            ["embed_tokens.weight"],
            "must either provide both token-weight matrices or omit both",
        ),
        (
            ["embed_tokens.weight", "lm_head.weight", "layers.0.weight"],
            "missing unsupported weights: layers.0.weight",
        ),
    ],
)
def test_official_checkpoint_rejects_invalid_missing_weights(
    monkeypatch: pytest.MonkeyPatch,
    missing_keys: list[str],
    message: str,
) -> None:
    draft = type("Draft", (), {})()
    monkeypatch.setattr(
        DFlash2DraftModel,
        "from_pretrained",
        lambda *args, **kwargs: (
            draft,
            {"missing_keys": missing_keys},
        ),
    )

    with pytest.raises(ValueError, match=message):
        load_dflash2_draft_model(
            "official-draft",
            target=object(),
        )


def test_checkpoint_owned_token_weights_are_preserved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    draft = type(
        "Draft",
        (),
        {
            "embed_tokens": torch.nn.Embedding(32, 8),
            "lm_head": torch.nn.Linear(8, 32, bias=False),
        },
    )()
    original_embeddings = draft.embed_tokens
    original_lm_head = draft.lm_head
    monkeypatch.setattr(
        DFlash2DraftModel,
        "from_pretrained",
        lambda *args, **kwargs: (draft, {"missing_keys": []}),
    )

    loaded, loading_info = load_dflash2_draft_model(
        "self-contained-draft",
        target=object(),
    )

    assert loaded.embed_tokens is original_embeddings
    assert loaded.lm_head is original_lm_head
    assert loading_info["shared_target_weight_keys"] == []
