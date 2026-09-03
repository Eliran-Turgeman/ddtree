import time
from types import SimpleNamespace

import torch
from transformers import AutoModelForCausalLM, DynamicCache

from model import DFlash2DraftModel, extract_context_feature


DFLASH2_STAGE_ORDER = ("draft", "verify", "commit")


def cuda_time() -> float:
    torch.cuda.synchronize()
    return time.perf_counter()


@torch.inference_mode()
def dflash2_generate(
    model: DFlash2DraftModel,
    target: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    stop_token_ids: list[int] | None,
    collect_traces: bool = False,
    prompt_id: str | None = None,
) -> SimpleNamespace:
    num_input_tokens = input_ids.shape[1]
    max_length = num_input_tokens + max_new_tokens
    block_size = model.block_size

    output_ids = torch.full(
        (1, max_length + 1),
        model.mask_token_id,
        dtype=torch.long,
        device=target.device,
    )
    position_ids = torch.arange(
        output_ids.shape[1],
        device=target.device,
    ).unsqueeze(0)
    stop_tokens = (
        torch.tensor(
            stop_token_ids,
            dtype=output_ids.dtype,
            device=output_ids.device,
        )
        if stop_token_ids is not None
        else None
    )

    past_key_values_target = DynamicCache()
    past_key_values_draft = DynamicCache()
    stage_times = {stage_name: 0.0 for stage_name in DFLASH2_STAGE_ORDER}

    prefill_start = cuda_time()
    output = target(
        input_ids,
        position_ids=position_ids[:, :num_input_tokens],
        past_key_values=past_key_values_target,
        use_cache=True,
        logits_to_keep=1,
        output_hidden_states=True,
    )
    output_ids[:, :num_input_tokens] = input_ids
    output_ids[:, num_input_tokens] = output.logits[:, -1].argmax(dim=-1)
    target_hidden = extract_context_feature(
        output.hidden_states,
        model.target_layer_ids,
    )
    past_key_values_target.crop(num_input_tokens)
    time_to_first_token = cuda_time() - prefill_start

    decode_start = cuda_time()
    acceptance_lengths = []
    accepted_draft_tokens = []
    committed_tokens_per_round = []
    verifier_bonus_committed_per_round = []
    round_timestamps = []
    round_metrics = []
    trace_rounds = [] if collect_traces else None
    draft_prefill = True
    start = num_input_tokens
    stopped = (
        stop_tokens is not None
        and torch.isin(output_ids[:, start], stop_tokens).any()
    )

    while start + 1 < max_length and not stopped:
        round_start = cuda_time()
        verify_size = min(block_size, max_length - start)
        block_output_ids = output_ids[:, start : start + verify_size].clone()
        block_position_ids = position_ids[:, start : start + verify_size]
        prefix_token_ids = (
            output_ids[0, : start + 1].detach().cpu().to(torch.int32)
            if collect_traces
            else None
        )

        draft_start = cuda_time()
        noise_embedding = model.embed_tokens(block_output_ids)
        draft_hidden = model(
            target_hidden=target_hidden,
            noise_embedding=noise_embedding,
            position_ids=position_ids[
                :,
                start - target_hidden.shape[1] : start + verify_size,
            ],
            past_key_values=past_key_values_draft,
            use_cache=True,
        )[:, 1 - verify_size :, :]
        past_key_values_draft.crop(start)
        proposal = model.propose(
            draft_hidden,
            block_output_ids[:, 0],
            collect_lattice=collect_traces,
        )
        block_output_ids[:, 1:] = proposal.token_ids
        draft_elapsed = cuda_time() - draft_start
        if draft_prefill:
            draft_prefill = False
            decode_start = cuda_time()
        else:
            stage_times["draft"] += draft_elapsed

        verify_start = cuda_time()
        output = target(
            block_output_ids,
            position_ids=block_position_ids,
            past_key_values=past_key_values_target,
            use_cache=True,
            output_hidden_states=True,
        )
        verify_elapsed = cuda_time() - verify_start
        stage_times["verify"] += verify_elapsed

        commit_start = cuda_time()
        posterior = output.logits.argmax(dim=-1)
        acceptance_length = (
            (block_output_ids[:, 1:] == posterior[:, :-1])
            .cumprod(dim=1)
            .sum(dim=1)[0]
            .item()
        )
        bonus = posterior[:, acceptance_length]
        output_ids[:, start : start + acceptance_length + 1] = block_output_ids[
            :,
            : acceptance_length + 1,
        ]
        output_ids[:, start + acceptance_length + 1] = bonus

        produced = min(acceptance_length + 1, max_length - start - 1)
        if stop_tokens is not None:
            stop_indices = torch.isin(
                output_ids[0, start + 1 : start + produced + 1],
                stop_tokens,
            ).nonzero(as_tuple=True)[0]
            if stop_indices.numel() > 0:
                produced = stop_indices[0].item() + 1
                stopped = True

        if collect_traces:
            if (
                proposal.unary_logsumexp is None
                or proposal.anchor_pairwise_corrections is None
                or proposal.pairwise_corrections is None
                or proposal.anchor_final_scores is None
                or proposal.pairwise_final_scores is None
            ):
                raise RuntimeError(
                    "DFlash2 lattice tensors were not collected"
                )
            draft_length = proposal.token_ids.shape[1]
            directly_observed_count = min(
                acceptance_length + 1,
                draft_length,
                produced,
            )
            directly_observed_mask = torch.zeros(
                draft_length,
                dtype=torch.bool,
            )
            directly_observed_mask[:directly_observed_count] = True
            committed_token_ids = output_ids[
                0,
                start + 1 : start + produced + 1,
            ].detach().cpu().to(torch.int32)
            trace_rounds.append(
                {
                    "round_id": len(trace_rounds),
                    "prefix_token_ids": prefix_token_ids,
                    "prefix_length": start + 1,
                    "anchor_position": start,
                    "anchor_token_id": int(block_output_ids[0, 0].item()),
                    "generation_start_position": start + 1,
                    "draft_length": draft_length,
                    "candidate_count": proposal.candidate_ids.shape[-1],
                    "candidate_token_ids": proposal.candidate_ids[0]
                    .detach()
                    .cpu()
                    .to(torch.int32),
                    "candidate_unary_logits": proposal.unary_scores[0]
                    .detach()
                    .cpu(),
                    "unary_logsumexp": proposal.unary_logsumexp[0]
                    .detach()
                    .cpu(),
                    "anchor_pairwise_corrections": (
                        proposal.anchor_pairwise_corrections[0].detach().cpu()
                    ),
                    "pairwise_corrections": proposal.pairwise_corrections[0]
                    .detach()
                    .cpu(),
                    "anchor_final_scores": proposal.anchor_final_scores[0]
                    .detach()
                    .cpu(),
                    "pairwise_final_scores": proposal.pairwise_final_scores[0]
                    .detach()
                    .cpu(),
                    "selected_draft_token_ids": proposal.token_ids[0]
                    .detach()
                    .cpu()
                    .to(torch.int32),
                    "selected_candidate_indices": (
                        proposal.selected_candidate_indices[0]
                        .detach()
                        .cpu()
                        .to(torch.int16)
                    ),
                    "verifier_token_ids": posterior[0, :draft_length]
                    .detach()
                    .cpu()
                    .to(torch.int32),
                    "directly_observed_target_mask": directly_observed_mask,
                    "verifier_matched_draft_tokens": acceptance_length,
                    "accepted_draft_tokens": min(
                        acceptance_length,
                        produced,
                    ),
                    "committed_tokens_this_round": produced,
                    "committed_token_ids": committed_token_ids,
                    "verifier_next_token_id": int(bonus.item()),
                    "verifier_next_token_committed": (
                        produced > acceptance_length
                    ),
                }
            )

        start += produced
        past_key_values_target.crop(start)
        acceptance_lengths.append(produced)
        accepted_draft_tokens.append(
            min(acceptance_length, produced)
        )
        committed_tokens_per_round.append(produced)
        verifier_bonus_committed_per_round.append(
            produced > acceptance_length
        )
        target_hidden = extract_context_feature(
            output.hidden_states,
            model.target_layer_ids,
        )[:, :produced, :]
        commit_elapsed = cuda_time() - commit_start
        stage_times["commit"] += commit_elapsed
        total_round_elapsed = cuda_time() - round_start
        round_timestamps.append(cuda_time() - decode_start)
        round_metrics.append(
            {
                "method": "dflash2_greedy",
                "prompt_id": prompt_id,
                "round_id": len(round_metrics),
                "tree_budget": 7,
                "tree_node_count": proposal.token_ids.shape[1],
                "tree_max_depth": proposal.token_ids.shape[1],
                "nodes_per_depth": [
                    1
                    for _ in range(proposal.token_ids.shape[1])
                ],
                "matched_draft_tokens": min(
                    acceptance_length,
                    produced,
                ),
                "committed_tokens_this_round": produced,
                "verifier_bonus_committed": (
                    produced > acceptance_length
                ),
                "draft_latency_ms": draft_elapsed * 1000,
                "tree_build_latency_ms": 0.0,
                "tree_compile_latency_ms": 0.0,
                "target_verify_latency_ms": verify_elapsed * 1000,
                "commit_latency_ms": commit_elapsed * 1000,
                "total_round_latency_ms": (
                    total_round_elapsed * 1000
                ),
            }
        )

    output_ids = output_ids[:, : min(start + 1, max_length)]
    if trace_rounds is not None:
        for trace_round in trace_rounds:
            continuation_start = trace_round["generation_start_position"]
            continuation_end = min(
                continuation_start + trace_round["draft_length"],
                output_ids.shape[1],
            )
            trace_round["realized_continuation_token_ids"] = output_ids[
                0,
                continuation_start:continuation_end,
            ].detach().cpu().to(torch.int32)
    num_output_tokens = output_ids.shape[1] - num_input_tokens
    total_decode_time = cuda_time() - decode_start

    return SimpleNamespace(
        output_ids=output_ids.cpu(),
        num_input_tokens=num_input_tokens,
        num_output_tokens=num_output_tokens,
        time_to_first_token=time_to_first_token,
        time_per_output_token=total_decode_time / max(num_output_tokens, 1),
        acceptance_lengths=acceptance_lengths,
        accepted_draft_tokens_per_round=accepted_draft_tokens,
        matched_draft_tokens_per_round=accepted_draft_tokens,
        committed_tokens_per_round=committed_tokens_per_round,
        verifier_bonus_committed_per_round=(
            verifier_bonus_committed_per_round
        ),
        decode_rounds=len(acceptance_lengths),
        stage_times=stage_times,
        round_timestamps=round_timestamps,
        round_metrics=round_metrics,
        trace_rounds=trace_rounds,
    )
