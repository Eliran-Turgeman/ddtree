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
from model import DFlash2DraftModel, extract_context_feature
from model.dflash2 import DFlash2Proposal
from offline_dflash2_trees import (
    PAIRWISE_MASS_PRESERVING,
    UNARY_FULL_MASS,
    TreeNode,
    build_best_first_tree,
    build_scorer,
    validate_lattice_tensors,
)


DFLASH2_UNARY_K16 = "dflash2_unary_k16"
DFLASH2_PAIRWISE_K16 = "dflash2_pairwise_k16"
DFLASH2_TREE_METHODS = (
    DFLASH2_UNARY_K16,
    DFLASH2_PAIRWISE_K16,
)
DFLASH2_TREE_STAGE_ORDER = (
    "draft",
    "tree_build",
    "tree_compile",
    "verify",
    "commit",
)
EXPECTED_DRAFT_DEPTH = 7
EXPECTED_CANDIDATE_COUNT = 16


def proposal_to_lattice(proposal: DFlash2Proposal) -> dict[str, torch.Tensor]:
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
        raise ValueError(
            "online DFlash2 tree generation requires batch size 1"
        )

    lattice = {
        "candidate_token_ids": proposal.candidate_ids[0],
        "candidate_unary_logits": proposal.unary_scores[0],
        "unary_logsumexp": proposal.unary_logsumexp[0],
        "anchor_final_scores": proposal.anchor_final_scores[0],
        "pairwise_final_scores": proposal.pairwise_final_scores[0],
    }
    validate_lattice_tensors(
        lattice,
        expected_depth=EXPECTED_DRAFT_DEPTH,
        expected_candidate_count=EXPECTED_CANDIDATE_COUNT,
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
    if validate_lattice:
        depth, _ = validate_lattice_tensors(
            lattice,
            expected_depth=(
                EXPECTED_DRAFT_DEPTH if require_checkpoint_shape else None
            ),
            expected_candidate_count=(
                EXPECTED_CANDIDATE_COUNT
                if require_checkpoint_shape
                else None
            ),
        )
    else:
        depth = int(lattice["candidate_token_ids"].shape[0])
    scorer_name = {
        DFLASH2_UNARY_K16: UNARY_FULL_MASS,
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
            "tree scorer has a positive extension log probability: "
            f"{maximum_extension}"
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

    past_key_values_target = DynamicCache()
    past_key_values_draft = DynamicCache()
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
    past_key_values_target.crop(num_input_tokens)
    time_to_first_token = cuda_time() - prefill_start

    decode_start = cuda_time()
    start = num_input_tokens
    round_metrics = []
    acceptance_lengths = []
    matched_draft_tokens_per_round = []
    committed_tokens_per_round = []
    verifier_bonus_committed_per_round = []
    round_timestamps = []
    previous_tree_start = 0
    previous_tree_length = 0
    draft_prefill = True
    stopped = (
        stop_tokens is not None
        and bool(torch.isin(output_ids[:, start], stop_tokens).any())
    )

    while start + 1 < max_length and not stopped:
        round_start = cuda_time()
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
        past_key_values_draft.crop(start)
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

        tree_build_start = cuda_time()
        lattice = proposal_to_lattice(proposal)
        (
            nodes,
            node_token_ids,
            node_depths,
            _parents,
            child_maps,
            visibility_cpu,
        ) = build_dflash2_verifier_tree(
            lattice,
            tree_method,
            tree_budget,
            validate_lattice=False,
        )
        tree_build_latency = cuda_time() - tree_build_start
        stage_times["tree_build"] += tree_build_latency

        tree_compile_start = cuda_time()
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
        output = target(
            verify_input_ids,
            position_ids=verify_position_ids,
            attention_mask=verify_attention_mask,
            past_key_values=past_key_values_target,
            use_cache=True,
            output_hidden_states=True,
        )
        verify_latency = cuda_time() - verify_start
        stage_times["verify"] += verify_latency

        commit_start = cuda_time()
        posterior = output.logits.argmax(dim=-1)
        accepted_indices, next_token = follow_verified_tree(
            child_maps,
            posterior,
        )
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
        keep_index_tensor = torch.tensor(
            keep_indices,
            dtype=torch.long,
            device=verify_input_ids.device,
        )
        compact_dynamic_cache(
            past_key_values_target,
            start,
            keep_indices,
        )
        target_hidden = extract_context_feature(
            output.hidden_states,
            model.target_layer_ids,
        ).index_select(1, keep_index_tensor)
        start += produced
        commit_latency = cuda_time() - commit_start
        stage_times["commit"] += commit_latency

        total_round_latency = cuda_time() - round_start
        metric = {
            "method": tree_method,
            "prompt_id": prompt_id,
            "round_id": len(round_metrics),
            "tree_budget": tree_budget,
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
                    "path_candidate_indices": (
                        node.path_candidate_indices
                    ),
                }
                for node in nodes
            ]
        round_metrics.append(metric)
        acceptance_lengths.append(produced)
        matched_draft_tokens_per_round.append(matched_draft_tokens)
        committed_tokens_per_round.append(produced)
        verifier_bonus_committed_per_round.append(
            verifier_bonus_committed
        )
        round_timestamps.append(cuda_time() - decode_start)

    output_ids = output_ids[:, : min(start + 1, max_length)]
    num_output_tokens = output_ids.shape[1] - num_input_tokens
    total_decode_time = cuda_time() - decode_start
    return SimpleNamespace(
        output_ids=output_ids.cpu(),
        num_input_tokens=num_input_tokens,
        num_output_tokens=num_output_tokens,
        time_to_first_token=time_to_first_token,
        time_per_output_token=(
            total_decode_time / max(num_output_tokens, 1)
        ),
        acceptance_lengths=acceptance_lengths,
        matched_draft_tokens_per_round=matched_draft_tokens_per_round,
        accepted_draft_tokens_per_round=matched_draft_tokens_per_round,
        committed_tokens_per_round=committed_tokens_per_round,
        verifier_bonus_committed_per_round=(
            verifier_bonus_committed_per_round
        ),
        decode_rounds=len(round_metrics),
        stage_times=stage_times,
        round_timestamps=round_timestamps,
        round_metrics=round_metrics,
    )
