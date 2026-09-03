import time
from types import SimpleNamespace

import torch
from transformers import AutoModelForCausalLM, DynamicCache

from model import DFlash2DraftModel, extract_context_feature


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
    start = num_input_tokens
    stopped = (
        stop_tokens is not None
        and torch.isin(output_ids[:, start], stop_tokens).any()
    )

    while start + 1 < max_length and not stopped:
        verify_size = min(block_size, max_length - start)
        block_output_ids = output_ids[:, start : start + verify_size].clone()
        block_position_ids = position_ids[:, start : start + verify_size]

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
        proposal = model.propose(draft_hidden, block_output_ids[:, 0])
        block_output_ids[:, 1:] = proposal.token_ids

        output = target(
            block_output_ids,
            position_ids=block_position_ids,
            past_key_values=past_key_values_target,
            use_cache=True,
            output_hidden_states=True,
        )
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

        start += produced
        past_key_values_target.crop(start)
        acceptance_lengths.append(produced)
        target_hidden = extract_context_feature(
            output.hidden_states,
            model.target_layer_ids,
        )[:, :produced, :]

    output_ids = output_ids[:, : min(start + 1, max_length)]
    num_output_tokens = output_ids.shape[1] - num_input_tokens
    total_decode_time = cuda_time() - decode_start

    return SimpleNamespace(
        output_ids=output_ids.cpu(),
        num_input_tokens=num_input_tokens,
        num_output_tokens=num_output_tokens,
        time_to_first_token=time_to_first_token,
        time_per_output_token=total_decode_time / max(num_output_tokens, 1),
        acceptance_lengths=acceptance_lengths,
        decode_rounds=len(acceptance_lengths),
    )
