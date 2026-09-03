import torch

from model.dflash2 import (
    CandidateSelector,
    DFlash2DraftModel,
    GroupedDynamicCausalConv,
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
    )

    assert proposal.token_ids.tolist() == [[1, 0]]
    assert proposal.candidate_ids.shape == (1, 2, 2)
    assert proposal.unary_scores.shape == (1, 2, 2)
    assert proposal.corrected_scores.shape == (1, 2, 2)


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
        }
    )

    assert config.block_size == 8
    assert config.dflash_config == {
        "target_layer_ids": [1],
        "mask_token_id": 31,
    }
    assert config.rope_parameters["rope_theta"] == 1_000_000
    assert config.selector_top_k == 2
