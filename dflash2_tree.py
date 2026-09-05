from types import SimpleNamespace

import torch
from transformers import AutoModelForCausalLM, DynamicCache

from ddtree import (
    compact_dynamic_cache,
    compile_ddtree_tree,
    compile_generic_tree_for_verifier,
    follow_verified_tree,
)
from dflash import cuda_time, empty_stage_times
from dflash2 import create_generation_cache, retain_cache_prefix
from model import DFlash2DraftModel, extract_context_feature
from model.dflash2 import DFlash2Proposal
from offline_dflash2_trees import (
    PAIRWISE_MASS_PRESERVING,
    UNARY_FULL_MASS,
    TreeNode,
    build_best_first_tree,
    build_scorer,
    validate_lattice_tensors,
    validate_unary_lattice_tensors,
)


DFLASH2_UNARY_K16 = "dflash2_unary_k16"
DFLASH2_UNARY_K32 = "dflash2_unary_k32"
DFLASH2_UNARY_K64 = "dflash2_unary_k64"
DFLASH2_PAIRWISE_K16 = "dflash2_pairwise_k16"
DFLASH2_UNARY_METHODS = {
    DFLASH2_UNARY_K16: 16,
    DFLASH2_UNARY_K32: 32,
    DFLASH2_UNARY_K64: 64,
}
DFLASH2_TREE_METHODS = (
    *DFLASH2_UNARY_METHODS,
    DFLASH2_PAIRWISE_K16,
)
DFLASH2_TREE_STAGE_ORDER = (
    "draft",
    "candidate_select",
    "tree_build",
    "tree_compile",
    "verify",
    "commit",
)
EXPECTED_DRAFT_DEPTH = 7
EXPECTED_CANDIDATE_COUNT = 16


def proposal_to_lattice(
    proposal: DFlash2Proposal,
    candidate_count: int = EXPECTED_CANDIDATE_COUNT,
) -> dict[str, torch.Tensor]:
    required = {
        "unary_logsumexp": proposal.unary_logsumexp,
        "anchor_final_scores": proposal.anchor_final_scores,
        "pairwise_final_scores": proposal.pairwise_final_scores,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise ValueError(
            "DFlash2 proposal did not materialize the candidate lattice: "
            + ", ".join(missing)
        )
    if proposal.candidate_ids.shape[0] != 1:
        raise ValueError("online DFlash2 tree generation requires batch size 1")

    if candidate_count < EXPECTED_CANDIDATE_COUNT:
        raise ValueError(
            "candidate_count cannot be smaller than the checkpoint "
            f"selector width {EXPECTED_CANDIDATE_COUNT}"
        )
    if candidate_count == EXPECTED_CANDIDATE_COUNT:
        candidate_ids = proposal.candidate_ids
        unary_scores = proposal.unary_scores
    else:
        if proposal.full_unary_logits is None:
            raise ValueError("DFlash2 proposal did not retain full unary logits")
        if candidate_count > proposal.full_unary_logits.shape[-1]:
            raise ValueError(
                f"candidate_count {candidate_count} exceeds vocabulary size "
                f"{proposal.full_unary_logits.shape[-1]}"
            )
        unary_scores, candidate_ids = proposal.full_unary_logits.topk(
            candidate_count,
            dim=-1,
        )

    lattice = {
        "candidate_token_ids": candidate_ids[0],
        "candidate_unary_logits": unary_scores[0],
        "unary_logsumexp": proposal.unary_logsumexp[0],
        "anchor_final_scores": proposal.anchor_final_scores[0],
        "pairwise_final_scores": proposal.pairwise_final_scores[0],
    }
    if candidate_count == EXPECTED_CANDIDATE_COUNT:
        validate_lattice_tensors(
            lattice,
            expected_depth=EXPECTED_DRAFT_DEPTH,
            expected_candidate_count=EXPECTED_CANDIDATE_COUNT,
        )
    else:
        validate_unary_lattice_tensors(
            lattice,
            expected_depth=EXPECTED_DRAFT_DEPTH,
            expected_candidate_count=candidate_count,
        )
    return lattice


def build_dflash2_verifier_tree(
    lattice: dict[str, torch.Tensor],
    method: str,
    budget: int,
    *,
    require_checkpoint_shape: bool = True,
    validate_lattice: bool = True,
) -> tuple[
    list[TreeNode],
    torch.Tensor,
    torch.Tensor,
    list[int],
    list[dict[int, int]],
    torch.Tensor,
]:
    unary_candidate_count = DFLASH2_UNARY_METHODS.get(method)
    expected_candidate_count = (
        unary_candidate_count
        if unary_candidate_count is not None
        else EXPECTED_CANDIDATE_COUNT
    )
    if validate_lattice:
        validator = (
            validate_unary_lattice_tensors
            if unary_candidate_count is not None
            else validate_lattice_tensors
        )
        depth, _ = validator(
            lattice,
            expected_depth=(EXPECTED_DRAFT_DEPTH if require_checkpoint_shape else None),
            expected_candidate_count=(
                expected_candidate_count if require_checkpoint_shape else None
            ),
        )
    else:
        depth = int(lattice["candidate_token_ids"].shape[0])
    scorer_name = {
        **{unary_method: UNARY_FULL_MASS for unary_method in DFLASH2_UNARY_METHODS},
        DFLASH2_PAIRWISE_K16: PAIRWISE_MASS_PRESERVING,
    }.get(method)
    if scorer_name is None:
        raise ValueError(
            f"unsupported DFlash2 tree method {method!r}; "
            f"expected one of {DFLASH2_TREE_METHODS}"
        )
    scorer = build_scorer(lattice, scorer_name, validate=False)
    maximum_extension = scorer.maximum_extension_log_score()
    if maximum_extension > 1e-5:
        raise ValueError(
            f"tree scorer has a positive extension log probability: {maximum_extension}"
        )
    nodes = build_best_first_tree(
        lattice["candidate_token_ids"],
        scorer,
        budget,
    )
    compiled = compile_generic_tree_for_verifier(
        nodes,
        budget=budget,
        depth_limit=depth,
    )
    return (nodes, *compiled)


def _depth_counts(nodes: list[TreeNode], depth_limit: int) -> list[int]:
    counts = [0] * depth_limit
    for node in nodes:
        counts[node.depth - 1] += 1
    return counts


def target_requires_sequential_tree_verification(
    target: AutoModelForCausalLM,
) -> bool:
    config = target.config.get_text_config(decoder=True)
    return "linear_attention" in getattr(config, "layer_types", ())


def verify_target_selected_path(
    target: AutoModelForCausalLM,
    past_key_values_target: DynamicCache,
    root_token: torch.Tensor,
    node_token_ids: torch.Tensor,
    child_maps: list[dict[int, int]],
    start: int,
    target_layer_ids: list[int],
) -> tuple[list[int], int, torch.Tensor]:
    tree_token_ids = torch.cat(
        [root_token.reshape(-1), node_token_ids.to(root_token.device)]
    )
    accepted_indices = [0]
    selected_hidden_states = []
    current_index = 0

    while True:
        output = target(
            tree_token_ids[current_index].reshape(1, 1),
            position_ids=torch.tensor(
                [[start + len(accepted_indices) - 1]],
                dtype=torch.long,
                device=root_token.device,
            ),
            past_key_values=past_key_values_target,
            use_cache=True,
            logits_to_keep=1,
            output_hidden_states=True,
        )
        selected_hidden_states.append(
            extract_context_feature(
                output.hidden_states,
                target_layer_ids,
            )
        )
        next_token = int(output.logits[0, -1].argmax().item())
        child_index = child_maps[current_index].get(next_token)
        if child_index is None:
            return (
                accepted_indices,
                next_token,
                torch.cat(selected_hidden_states, dim=1),
            )
        accepted_indices.append(child_index)
        current_index = child_index


def annotate_candidate_diagnostics(
    round_metrics: list[dict[str, object]],
    round_candidate_ids: list[torch.Tensor],
    round_generation_starts: list[int],
    generated_token_ids: torch.Tensor,
) -> None:
    if not (
        len(round_metrics) == len(round_candidate_ids) == len(round_generation_starts)
    ):
        raise ValueError("candidate diagnostics are not round-aligned")
    generated_token_ids = generated_token_ids.long().cpu()
    for metric, candidate_ids, generation_start in zip(
        round_metrics,
        round_candidate_ids,
        round_generation_starts,
        strict=True,
    ):
        available_depth = min(
            EXPECTED_DRAFT_DEPTH,
            max(int(generated_token_ids.shape[0]) - generation_start, 0),
        )
        prefix_representable = True
        ranks: list[int | None] = []
        for depth_index in range(EXPECTED_DRAFT_DEPTH):
            rank = None
            if depth_index < available_depth:
                target_token_id = generated_token_ids[generation_start + depth_index]
                matches = (candidate_ids[depth_index] == target_token_id).nonzero(
                    as_tuple=True
                )[0]
                if matches.numel():
                    rank = int(matches[0]) + 1
                prefix_representable = prefix_representable and rank is not None
            else:
                prefix_representable = False
            ranks.append(rank)
            metric[f"target_rank_depth_{depth_index + 1}"] = rank
            metric[f"target_prefix_representable_depth_{depth_index + 1}"] = (
                prefix_representable if depth_index < available_depth else None
            )

        matched = int(metric["matched_draft_tokens"])
        metric["target_available_depth"] = available_depth
        if available_depth < EXPECTED_DRAFT_DEPTH and matched >= available_depth:
            failure_type = "censored"
        elif matched >= EXPECTED_DRAFT_DEPTH:
            failure_type = "covered"
        else:
            failure_type = (
                "ranking_budget_failure"
                if ranks[matched] is not None
                else "candidate_failure"
            )
        metric["failure_type"] = failure_type


@torch.inference_mode()
def dflash2_tree_generate(
    model: DFlash2DraftModel,
    target: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    stop_token_ids: list[int] | None,
    tree_budget: int,
    tree_method: str,
    *,
    prompt_id: str | None = None,
    collect_tree_data: bool = False,
) -> SimpleNamespace:
    if tree_method not in DFLASH2_TREE_METHODS:
        raise ValueError(f"unsupported DFlash2 tree method {tree_method!r}")
    if model.block_size != EXPECTED_DRAFT_DEPTH + 1:
        raise ValueError(
            "online DFlash2 tree prototype expects block size "
            f"{EXPECTED_DRAFT_DEPTH + 1}, got {model.block_size}"
        )
    if model.candidate_selector.top_k != EXPECTED_CANDIDATE_COUNT:
        raise ValueError(
            "online DFlash2 tree prototype expects selector top-k "
            f"{EXPECTED_CANDIDATE_COUNT}, got "
            f"{model.candidate_selector.top_k}"
        )
    if tree_budget < 0:
        raise ValueError("tree budget must be non-negative")

    num_input_tokens = input_ids.shape[1]
    max_length = num_input_tokens + max_new_tokens
    max_tree_nodes = tree_budget + 1
    output_ids = torch.full(
        (
            1,
            max_length + max(max_tree_nodes, model.block_size),
        ),
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

    verify_input_ids_buffer = torch.empty(
        (1, max_tree_nodes),
        dtype=torch.long,
        device=target.device,
    )
    verify_position_ids_buffer = torch.empty(
        (1, max_tree_nodes),
        dtype=torch.long,
        device=target.device,
    )
    attention_mask_buffer = torch.zeros(
        (1, 1, max_tree_nodes, max_length + max_tree_nodes),
        dtype=target.dtype,
        device=target.device,
    )
    tree_visibility_buffer = torch.empty(
        (max_tree_nodes, max_tree_nodes),
        dtype=torch.bool,
        device=target.device,
    )

    sequential_verifier = target_requires_sequential_tree_verification(target)
    past_key_values_target = create_generation_cache(target.config)
    past_key_values_draft = create_generation_cache(model.config)
    stage_times = empty_stage_times(DFLASH2_TREE_STAGE_ORDER)

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
    retain_cache_prefix(past_key_values_target, num_input_tokens)
    time_to_first_token = cuda_time() - prefill_start

    decode_start = cuda_time()
    start = num_input_tokens
    round_metrics = []
    round_candidate_ids = []
    round_generation_starts = []
    acceptance_lengths = []
    matched_draft_tokens_per_round = []
    committed_tokens_per_round = []
    verifier_bonus_committed_per_round = []
    round_timestamps = []
    previous_tree_start = 0
    previous_tree_length = 0
    draft_prefill = True
    stopped = stop_tokens is not None and bool(
        torch.isin(output_ids[:, start], stop_tokens).any()
    )

    while start + 1 < max_length and not stopped:
        round_start = cuda_time()
        generation_start = start - num_input_tokens + 1
        root_token = output_ids[:, start : start + 1]
        block_output_ids = output_ids[
            :,
            start : start + model.block_size,
        ].clone()

        draft_start = cuda_time()
        noise_embedding = model.embed_tokens(block_output_ids)
        draft_hidden = model(
            target_hidden=target_hidden,
            noise_embedding=noise_embedding,
            position_ids=position_ids[
                :,
                start - target_hidden.shape[1] : start + model.block_size,
            ],
            past_key_values=past_key_values_draft,
            use_cache=True,
        )[:, 1 - model.block_size :, :]
        retain_cache_prefix(past_key_values_draft, start)
        proposal = model.propose(
            draft_hidden,
            root_token[:, 0],
            collect_lattice=True,
        )
        draft_latency = cuda_time() - draft_start
        if draft_prefill:
            draft_prefill = False
            decode_start = cuda_time()
        else:
            stage_times["draft"] += draft_latency

        candidate_select_start = cuda_time()
        candidate_count = DFLASH2_UNARY_METHODS.get(
            tree_method,
            EXPECTED_CANDIDATE_COUNT,
        )
        lattice = proposal_to_lattice(proposal, candidate_count)
        candidate_select_latency = cuda_time() - candidate_select_start
        stage_times["candidate_select"] += candidate_select_latency
        tree_build_start = cuda_time()
        candidate_ids_cpu = lattice["candidate_token_ids"].long().cpu()
        tree_lattice = {
            **lattice,
            "candidate_token_ids": candidate_ids_cpu,
        }
        (
            nodes,
            node_token_ids,
            node_depths,
            _parents,
            child_maps,
            visibility_cpu,
        ) = build_dflash2_verifier_tree(
            tree_lattice,
            tree_method,
            tree_budget,
            validate_lattice=False,
        )
        tree_build_latency = cuda_time() - tree_build_start
        stage_times["tree_build"] += tree_build_latency

        tree_compile_start = cuda_time()
        if sequential_verifier:
            verify_input_ids = torch.cat(
                [
                    root_token,
                    node_token_ids.to(target.device).unsqueeze(0),
                ],
                dim=1,
            )
            verify_position_ids = None
            verify_attention_mask = None
        else:
            (
                verify_input_ids,
                verify_position_ids,
                verify_attention_mask,
                previous_tree_start,
                previous_tree_length,
            ) = compile_ddtree_tree(
                root_token_id=root_token[0, 0],
                start=start,
                node_token_ids=node_token_ids,
                node_depths=node_depths,
                visibility_cpu=visibility_cpu,
                past_length=start,
                dtype=target.dtype,
                device=target.device,
                verify_input_ids_buffer=verify_input_ids_buffer,
                verify_position_ids_buffer=verify_position_ids_buffer,
                attention_mask_buffer=attention_mask_buffer,
                tree_visibility_buffer=tree_visibility_buffer,
                previous_tree_start=previous_tree_start,
                previous_tree_length=previous_tree_length,
            )
        tree_compile_latency = cuda_time() - tree_compile_start
        stage_times["tree_compile"] += tree_compile_latency

        verify_start = cuda_time()
        if sequential_verifier:
            (
                accepted_indices,
                next_token,
                accepted_target_hidden,
            ) = verify_target_selected_path(
                target=target,
                past_key_values_target=past_key_values_target,
                root_token=root_token,
                node_token_ids=node_token_ids,
                child_maps=child_maps,
                start=start,
                target_layer_ids=model.target_layer_ids,
            )
        else:
            output = target(
                verify_input_ids,
                position_ids=verify_position_ids,
                attention_mask=verify_attention_mask,
                past_key_values=past_key_values_target,
                use_cache=True,
                output_hidden_states=True,
            )
            posterior = output.logits.argmax(dim=-1)
            accepted_indices, next_token = follow_verified_tree(
                child_maps,
                posterior,
            )
            accepted_target_hidden = extract_context_feature(
                output.hidden_states,
                model.target_layer_ids,
            ).index_select(
                1,
                torch.tensor(
                    accepted_indices,
                    dtype=torch.long,
                    device=verify_input_ids.device,
                ),
            )
        verify_latency = cuda_time() - verify_start
        stage_times["verify"] += verify_latency

        commit_start = cuda_time()
        accepted_index_tensor = torch.tensor(
            accepted_indices,
            dtype=torch.long,
            device=verify_input_ids.device,
        )
        accepted_tokens = verify_input_ids.index_select(
            1,
            accepted_index_tensor,
        )
        output_ids[
            :,
            start : start + len(accepted_indices),
        ] = accepted_tokens
        output_ids[:, start + len(accepted_indices)] = next_token

        verifier_matched = len(accepted_indices) - 1
        produced = min(
            verifier_matched + 1,
            max_length - start - 1,
        )
        if stop_tokens is not None:
            stop_indices = torch.isin(
                output_ids[0, start + 1 : start + produced + 1],
                stop_tokens,
            ).nonzero(as_tuple=True)[0]
            if stop_indices.numel() > 0:
                produced = int(stop_indices[0]) + 1
                stopped = True

        matched_draft_tokens = min(verifier_matched, produced)
        verifier_bonus_committed = produced > verifier_matched
        keep_indices = accepted_indices[:produced]
        if sequential_verifier:
            retain_cache_prefix(
                past_key_values_target,
                start + produced,
            )
            target_hidden = accepted_target_hidden[:, :produced]
        else:
            compact_dynamic_cache(
                past_key_values_target,
                start,
                keep_indices,
            )
            target_hidden = accepted_target_hidden[:, :produced]
        start += produced
        commit_latency = cuda_time() - commit_start
        stage_times["commit"] += commit_latency

        total_round_latency = cuda_time() - round_start
        metric = {
            "method": tree_method,
            "prompt_id": prompt_id,
            "round_id": len(round_metrics),
            "tree_budget": tree_budget,
            "candidate_count": candidate_count,
            "verifier_mode": (
                "target_selected_path" if sequential_verifier else "packed_tree"
            ),
            "tree_node_count": len(nodes),
            "tree_max_depth": max(
                (node.depth for node in nodes),
                default=0,
            ),
            "nodes_per_depth": _depth_counts(
                nodes,
                EXPECTED_DRAFT_DEPTH,
            ),
            "matched_draft_tokens": matched_draft_tokens,
            "committed_tokens_this_round": produced,
            "verifier_bonus_committed": verifier_bonus_committed,
            "draft_latency_ms": draft_latency * 1000,
            "candidate_select_latency_ms": (candidate_select_latency * 1000),
            "tree_build_latency_ms": tree_build_latency * 1000,
            "tree_compile_latency_ms": tree_compile_latency * 1000,
            "target_verify_latency_ms": verify_latency * 1000,
            "commit_latency_ms": commit_latency * 1000,
            "total_round_latency_ms": total_round_latency * 1000,
        }
        if collect_tree_data:
            metric["tree"] = [
                {
                    "token_id": node.token_id,
                    "candidate_index": node.candidate_index,
                    "depth": node.depth,
                    "parent": node.parent,
                    "path_candidate_indices": (node.path_candidate_indices),
                }
                for node in nodes
            ]
        round_metrics.append(metric)
        round_candidate_ids.append(candidate_ids_cpu)
        round_generation_starts.append(generation_start)
        acceptance_lengths.append(produced)
        matched_draft_tokens_per_round.append(matched_draft_tokens)
        committed_tokens_per_round.append(produced)
        verifier_bonus_committed_per_round.append(verifier_bonus_committed)
        round_timestamps.append(cuda_time() - decode_start)

    total_decode_time = cuda_time() - decode_start
    output_ids = output_ids[:, : min(start + 1, max_length)]
    num_output_tokens = output_ids.shape[1] - num_input_tokens
    annotate_candidate_diagnostics(
        round_metrics,
        round_candidate_ids,
        round_generation_starts,
        output_ids[0, num_input_tokens:],
    )
    return SimpleNamespace(
        output_ids=output_ids.cpu(),
        num_input_tokens=num_input_tokens,
        num_output_tokens=num_output_tokens,
        time_to_first_token=time_to_first_token,
        time_per_output_token=(total_decode_time / max(num_output_tokens, 1)),
        acceptance_lengths=acceptance_lengths,
        matched_draft_tokens_per_round=matched_draft_tokens_per_round,
        accepted_draft_tokens_per_round=matched_draft_tokens_per_round,
        committed_tokens_per_round=committed_tokens_per_round,
        verifier_bonus_committed_per_round=(verifier_bonus_committed_per_round),
        decode_rounds=len(round_metrics),
        stage_times=stage_times,
        round_timestamps=round_timestamps,
        round_metrics=round_metrics,
    )
