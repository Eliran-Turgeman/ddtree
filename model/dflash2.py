"""
MIT License

Copyright (c) 2026 Z Lab

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

DFlash2 architecture adapted from:
https://github.com/z-lab/dflash/blob/07ebd93db9f472af339b644bb70221ad8428328a/dflash/model.py
"""

from dataclasses import dataclass

import torch
from torch import nn
from transformers import PretrainedConfig
from transformers.cache_utils import Cache
from transformers.models.qwen3.modeling_qwen3 import (
    FlashAttentionKwargs,
    Qwen3Config,
)
from typing_extensions import Unpack

from .dflash import DFlashDraftModel, Qwen3DFlashDecoderLayer


@dataclass
class DFlash2Proposal:
    token_ids: torch.Tensor
    selected_candidate_indices: torch.Tensor
    candidate_ids: torch.Tensor
    unary_scores: torch.Tensor
    unary_logsumexp: torch.Tensor | None
    anchor_pairwise_corrections: torch.Tensor | None
    pairwise_corrections: torch.Tensor | None
    anchor_final_scores: torch.Tensor | None
    pairwise_final_scores: torch.Tensor | None
    corrected_scores: torch.Tensor
    full_unary_logits: torch.Tensor | None = None


def load_dflash2_draft_model(
    pretrained_model_name_or_path: str,
    *,
    target: nn.Module,
    **kwargs,
) -> tuple["DFlash2DraftModel", dict]:
    loaded = DFlash2DraftModel.from_pretrained(
        pretrained_model_name_or_path,
        output_loading_info=True,
        **kwargs,
    )
    draft, loading_info = loaded
    missing_keys = set(loading_info["missing_keys"])
    shared_weight_keys = {
        "embed_tokens.weight",
        "lm_head.weight",
    }
    unexpected_missing = missing_keys - shared_weight_keys
    if unexpected_missing:
        raise ValueError(
            "DFlash2 checkpoint is missing unsupported weights: "
            + ", ".join(sorted(unexpected_missing))
        )
    if missing_keys:
        if missing_keys != shared_weight_keys:
            raise ValueError(
                "DFlash2 checkpoint must either provide both token-weight "
                "matrices or omit both for target sharing; missing: "
                + ", ".join(sorted(missing_keys))
            )
        target_embeddings = target.get_input_embeddings()
        target_lm_head = target.get_output_embeddings()
        if target_embeddings is None or target_lm_head is None:
            raise ValueError("target model does not expose input/output embeddings")
        if (
            draft.embed_tokens.weight.shape != target_embeddings.weight.shape
            or draft.lm_head.weight.shape != target_lm_head.weight.shape
        ):
            raise ValueError("target and draft token-weight shapes are incompatible")
        draft.embed_tokens = target_embeddings
        draft.lm_head = target_lm_head
        loading_info["shared_target_weight_keys"] = sorted(shared_weight_keys)
    else:
        loading_info["shared_target_weight_keys"] = []
    return draft, loading_info


def grouped_dynamic_conv(
    hidden_states: torch.Tensor,
    delta_kernel: torch.Tensor,
    base_kernel: torch.Tensor,
    *,
    block_size: int,
    group_size: int,
) -> torch.Tensor:
    hidden_size = hidden_states.shape[-1]
    if hidden_size % group_size:
        raise ValueError(
            f"hidden_size ({hidden_size}) must be divisible by group_size "
            f"({group_size})"
        )

    kernel_size = base_kernel.shape[0]
    if kernel_size > block_size:
        raise ValueError(
            f"kernel_size ({kernel_size}) cannot exceed block_size ({block_size})"
        )

    num_groups = hidden_size // group_size
    expected_delta_shape = (*hidden_states.shape[:-1], kernel_size, num_groups)
    if delta_kernel.shape != expected_delta_shape:
        raise ValueError(
            f"delta_kernel must have shape {expected_delta_shape}, "
            f"got {tuple(delta_kernel.shape)}"
        )

    original_shape = hidden_states.shape
    flat_hidden = hidden_states.reshape(-1, num_groups, group_size)
    flat_delta = delta_kernel.reshape(-1, kernel_size, num_groups)
    positions = torch.arange(
        flat_hidden.shape[0],
        device=hidden_states.device,
    ).remainder(block_size)
    output = torch.zeros_like(flat_hidden)

    for tap in range(kernel_size):
        if tap == 0:
            shifted = flat_hidden
        else:
            padding = flat_hidden.new_zeros(tap, num_groups, group_size)
            shifted = torch.cat([padding, flat_hidden[:-tap]], dim=0)
        coefficient = (
            base_kernel[tap]
            .to(flat_hidden.dtype)
            .view(
                1,
                num_groups,
                group_size,
            )
            + flat_delta[:, tap, :, None]
        )
        valid = positions.ge(tap).view(-1, 1, 1)
        output = output + shifted * coefficient * valid

    return output.reshape(original_shape)


class GroupedDynamicCausalConv(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        *,
        block_size: int,
        kernel_size: int,
        group_size: int,
    ) -> None:
        super().__init__()
        if hidden_size % group_size:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by group_size "
                f"({group_size})"
            )
        if kernel_size > block_size:
            raise ValueError(
                f"kernel_size ({kernel_size}) cannot exceed block_size ({block_size})"
            )

        self.block_size = block_size
        self.kernel_size = kernel_size
        self.group_size = group_size
        self.num_groups = hidden_size // group_size
        self.base_kernel = nn.Parameter(torch.empty(2, kernel_size, hidden_size))
        self.kernel_projection = nn.Linear(
            hidden_size,
            2 * kernel_size * self.num_groups,
            bias=False,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            self.base_kernel.zero_()
            self.base_kernel[:, 0].fill_(1.0)
            self.kernel_projection.weight.zero_()

    def prepare(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        kernels = self.kernel_projection(hidden_states).view(
            *hidden_states.shape[:-1],
            2,
            self.kernel_size,
            self.num_groups,
        )
        prepared = grouped_dynamic_conv(
            hidden_states,
            kernels[..., 0, :, :],
            self.base_kernel[0],
            block_size=self.block_size,
            group_size=self.group_size,
        )
        return prepared, kernels[..., 1, :, :]

    def finish(
        self,
        hidden_states: torch.Tensor,
        delta_kernel: torch.Tensor,
    ) -> torch.Tensor:
        return grouped_dynamic_conv(
            hidden_states,
            delta_kernel,
            self.base_kernel[1],
            block_size=self.block_size,
            group_size=self.group_size,
        )


class Qwen3DFlash2DecoderLayer(Qwen3DFlashDecoderLayer):
    def __init__(
        self,
        config: Qwen3Config,
        layer_idx: int,
        *,
        block_size: int,
        conv_kernel_size: int,
        conv_group_size: int,
    ) -> None:
        super().__init__(config, layer_idx)
        conv_kwargs = {
            "block_size": block_size,
            "kernel_size": conv_kernel_size,
            "group_size": conv_group_size,
        }
        self.attention_conv = GroupedDynamicCausalConv(
            config.hidden_size,
            **conv_kwargs,
        )
        self.mlp_conv = GroupedDynamicCausalConv(
            config.hidden_size,
            **conv_kwargs,
        )

    def forward(
        self,
        target_hidden: torch.Tensor | None = None,
        hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_value: Cache | None = None,
        output_attentions: bool | None = False,
        use_cache: bool | None = False,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> torch.Tensor:
        if hidden_states is None:
            raise ValueError("hidden_states must be provided")

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, output_kernel = self.attention_conv.prepare(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            target_hidden=target_hidden,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )[0]
        hidden_states = self.attention_conv.finish(hidden_states, output_kernel)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states, output_kernel = self.mlp_conv.prepare(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.mlp_conv.finish(hidden_states, output_kernel)
        return residual + hidden_states


class CandidateSelector(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        hidden_size: int,
        rank: int,
        top_k: int,
    ) -> None:
        super().__init__()
        if top_k > vocab_size:
            raise ValueError(f"top_k ({top_k}) cannot exceed vocab_size ({vocab_size})")
        self.top_k = top_k
        self.predecessor_codebook = nn.Parameter(torch.empty(vocab_size, rank))
        self.successor_codebook = nn.Parameter(torch.empty(vocab_size, rank))
        self.hidden_projection = nn.Linear(hidden_size, rank, bias=False)

    def pairwise_correction(
        self,
        hidden_states: torch.Tensor,
        predecessor_ids: torch.Tensor,
        candidate_ids: torch.Tensor,
    ) -> torch.Tensor:
        predecessor = self.predecessor_codebook[predecessor_ids.long()]
        projected_hidden = self.hidden_projection(hidden_states)
        context = predecessor * projected_hidden.to(predecessor.dtype).unsqueeze(-2)
        successors = self.successor_codebook[candidate_ids.long()]
        return (context.unsqueeze(-2) * successors.unsqueeze(-3)).sum(dim=-1)

    def score_candidate_components(
        self,
        unary_logits: torch.Tensor,
        hidden_states: torch.Tensor,
        predecessor_ids: torch.Tensor,
        candidate_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        predecessor = self.predecessor_codebook[predecessor_ids.long()]
        projected_hidden = self.hidden_projection(hidden_states)
        context = predecessor * projected_hidden.to(predecessor.dtype)
        successors = self.successor_codebook[candidate_ids.long()]
        transition_scores = (context.unsqueeze(-2) * successors).sum(dim=-1)
        unary_scores = unary_logits.gather(-1, candidate_ids)
        final_scores = unary_scores + transition_scores.to(unary_scores.dtype)
        return unary_scores, transition_scores, final_scores

    def score_candidates(
        self,
        unary_logits: torch.Tensor,
        hidden_states: torch.Tensor,
        predecessor_ids: torch.Tensor,
        candidate_ids: torch.Tensor,
    ) -> torch.Tensor:
        return self.score_candidate_components(
            unary_logits,
            hidden_states,
            predecessor_ids,
            candidate_ids,
        )[2]

    def select_path(
        self,
        unary_logits: torch.Tensor,
        hidden_states: torch.Tensor,
        anchor_ids: torch.Tensor,
        *,
        collect_lattice: bool = False,
    ) -> DFlash2Proposal:
        candidate_ids = unary_logits.topk(self.top_k, dim=-1).indices
        unary_scores = unary_logits.gather(-1, candidate_ids)
        predecessor_ids = anchor_ids
        path = []
        selected_indices = []
        selected_corrections = []
        corrected_scores = []

        for position in range(hidden_states.shape[1]):
            _, position_corrections, position_scores = self.score_candidate_components(
                unary_logits[:, position],
                hidden_states[:, position],
                predecessor_ids,
                candidate_ids[:, position],
            )
            selected_predecessor_indices = position_scores.argmax(
                dim=-1,
                keepdim=True,
            )
            selected_token_ids = candidate_ids[:, position].gather(
                -1,
                selected_predecessor_indices,
            )[:, 0]
            path.append(selected_token_ids)
            selected_indices.append(selected_predecessor_indices[:, 0])
            selected_corrections.append(position_corrections)
            corrected_scores.append(position_scores)
            predecessor_ids = selected_token_ids

        if collect_lattice:
            unary_logsumexp = torch.logsumexp(
                unary_logits.float(),
                dim=-1,
            )
            anchor_pairwise_corrections = selected_corrections[0]
            anchor_final_scores = corrected_scores[0]
            pairwise_correction_rows = []
            pairwise_final_score_rows = []
            for position in range(1, hidden_states.shape[1]):
                correction_matrix = self.pairwise_correction(
                    hidden_states[:, position],
                    candidate_ids[:, position - 1],
                    candidate_ids[:, position],
                )
                selected_row_mask = (
                    torch.nn.functional.one_hot(
                        selected_indices[position - 1],
                        num_classes=self.top_k,
                    )
                    .bool()
                    .unsqueeze(-1)
                )
                correction_matrix = torch.where(
                    selected_row_mask,
                    selected_corrections[position].unsqueeze(-2),
                    correction_matrix,
                )
                final_score_matrix = unary_scores[:, position].unsqueeze(
                    -2
                ) + correction_matrix.to(unary_scores.dtype)
                final_score_matrix = torch.where(
                    selected_row_mask,
                    corrected_scores[position].unsqueeze(-2),
                    final_score_matrix,
                )
                pairwise_correction_rows.append(correction_matrix)
                pairwise_final_score_rows.append(final_score_matrix)

            pairwise_corrections = (
                torch.stack(pairwise_correction_rows, dim=1)
                if pairwise_correction_rows
                else unary_scores.new_empty(
                    unary_scores.shape[0],
                    0,
                    self.top_k,
                    self.top_k,
                )
            )
            pairwise_final_scores = (
                torch.stack(pairwise_final_score_rows, dim=1)
                if pairwise_final_score_rows
                else unary_scores.new_empty(
                    unary_scores.shape[0],
                    0,
                    self.top_k,
                    self.top_k,
                )
            )
        else:
            unary_logsumexp = None
            anchor_pairwise_corrections = None
            pairwise_corrections = None
            anchor_final_scores = None
            pairwise_final_scores = None

        return DFlash2Proposal(
            token_ids=torch.stack(path, dim=1),
            selected_candidate_indices=torch.stack(
                selected_indices,
                dim=1,
            ),
            candidate_ids=candidate_ids,
            unary_scores=unary_scores,
            unary_logsumexp=unary_logsumexp,
            anchor_pairwise_corrections=anchor_pairwise_corrections,
            pairwise_corrections=pairwise_corrections,
            anchor_final_scores=anchor_final_scores,
            pairwise_final_scores=pairwise_final_scores,
            corrected_scores=torch.stack(corrected_scores, dim=1),
            full_unary_logits=unary_logits if collect_lattice else None,
        )


class DFlash2DraftModel(DFlashDraftModel):
    _no_split_modules = ["Qwen3DFlash2DecoderLayer"]

    @staticmethod
    def _convert_checkpoint_config(config_dict: dict) -> Qwen3Config:
        if "transformer_layer_config" in config_dict:
            transformer_config = dict(config_dict["transformer_layer_config"])
            dflash_config = {
                "block_size": config_dict["block_size"],
                "conv_group_size": config_dict["conv_group_size"],
                "conv_kernel_size": config_dict["conv_kernel_size"],
                "mask_token_id": config_dict["mask_token_id"],
                "selector_rank": config_dict["selector_rank"],
                "selector_top_k": config_dict["selector_top_k"],
                "target_layer_ids": config_dict["aux_hidden_state_layer_ids"],
            }
            sample_from_anchor = config_dict["sample_from_anchor"]
        else:
            transformer_config = dict(config_dict)
            dflash_config = config_dict.get("dflash_config")
            if not isinstance(dflash_config, dict):
                raise ValueError("DFlash2 checkpoint config must contain dflash_config")
            required = {
                "block_size",
                "conv_group_size",
                "conv_kernel_size",
                "mask_token_id",
                "selector_rank",
                "selector_top_k",
                "target_layer_ids",
            }
            missing = sorted(required - dflash_config.keys())
            if missing:
                raise ValueError(
                    "DFlash2 checkpoint dflash_config is missing: " + ", ".join(missing)
                )
            sample_from_anchor = False

        rope_parameters = transformer_config.get("rope_parameters")
        if rope_parameters is not None:
            transformer_config.setdefault(
                "rope_theta",
                rope_parameters["rope_theta"],
            )
        layer_config = Qwen3Config.from_dict(transformer_config)
        layer_config._commit_hash = config_dict.get("_commit_hash")
        layer_config.architectures = ["DFlash2DraftModel"]
        layer_config.block_size = dflash_config["block_size"]
        layer_config.conv_kernel_size = dflash_config["conv_kernel_size"]
        layer_config.conv_group_size = dflash_config["conv_group_size"]
        layer_config.selector_rank = dflash_config["selector_rank"]
        layer_config.selector_top_k = dflash_config["selector_top_k"]
        layer_config.sample_from_anchor = sample_from_anchor
        layer_config.dflash_config = {
            "target_layer_ids": dflash_config["target_layer_ids"],
            "mask_token_id": dflash_config["mask_token_id"],
        }
        if rope_parameters is not None:
            configured_rope = getattr(
                layer_config,
                "rope_parameters",
                None,
            ) or getattr(layer_config, "rope_scaling", None)
            if configured_rope != rope_parameters:
                raise ValueError(
                    "DFlash2 checkpoint RoPE configuration was not preserved: "
                    f"expected {rope_parameters}, got {configured_rope}"
                )
        return layer_config

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        if kwargs.get("config") is None:
            config_kwargs = {
                key: kwargs[key]
                for key in (
                    "cache_dir",
                    "force_download",
                    "local_files_only",
                    "revision",
                    "token",
                )
                if key in kwargs
            }
            config_dict, _ = PretrainedConfig.get_config_dict(
                pretrained_model_name_or_path,
                **config_kwargs,
            )
            kwargs["config"] = cls._convert_checkpoint_config(config_dict)
        return super().from_pretrained(
            pretrained_model_name_or_path,
            *model_args,
            **kwargs,
        )

    def _make_decoder_layer(
        self,
        config: Qwen3Config,
        layer_idx: int,
    ) -> Qwen3DFlash2DecoderLayer:
        return Qwen3DFlash2DecoderLayer(
            config,
            layer_idx,
            block_size=config.block_size,
            conv_kernel_size=config.conv_kernel_size,
            conv_group_size=config.conv_group_size,
        )

    def __init__(self, config: Qwen3Config) -> None:
        if config.sample_from_anchor:
            raise ValueError("DFlash2 requires sample_from_anchor=False")
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.lm_head = nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
        )
        self.candidate_selector = CandidateSelector(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            rank=config.selector_rank,
            top_k=config.selector_top_k,
        )
        self.post_init()

    def propose(
        self,
        hidden_states: torch.Tensor,
        anchor_ids: torch.Tensor,
        *,
        collect_lattice: bool = False,
    ) -> DFlash2Proposal:
        unary_logits = self.lm_head(hidden_states)
        return self.candidate_selector.select_path(
            unary_logits,
            hidden_states,
            anchor_ids,
            collect_lattice=collect_lattice,
        )
