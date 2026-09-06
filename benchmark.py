import argparse
import hashlib
import platform
import random
import subprocess
from itertools import chain
from pathlib import Path

from loguru import logger
import numpy as np
import torch
import transformers
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import distributed as dist
from model import (
    DFlashDraftModel,
    load_dflash2_draft_model,
    load_and_process_dataset,
)
from dflash import dflash_generate
from dflash2 import dflash2_generate
from dflash2_tree import (
    DFLASH2_TREE_METHODS,
    dflash2_tree_generate,
)
from ddtree import ddtree_generate, maybe_enable_cpp_compact


def parse_dflash2_tree_configs(
    value: str,
    allowed_methods: tuple[str, ...] = DFLASH2_TREE_METHODS,
) -> list[tuple[str, int]]:
    configs = []
    seen = set()
    for entry in value.split(";"):
        if not entry:
            continue
        try:
            method, budgets_text = entry.split(":", maxsplit=1)
        except ValueError as exc:
            raise ValueError(
                "DFlash2 tree configs must use METHOD:BUDGET[,BUDGET];..."
            ) from exc
        if method not in allowed_methods:
            raise ValueError(f"unsupported DFlash2 tree method: {method}")
        for budget_text in budgets_text.split(","):
            try:
                budget = int(budget_text)
            except ValueError as exc:
                raise ValueError(f"invalid tree budget: {budget_text}") from exc
            if budget <= 0:
                raise ValueError("tree budgets must be positive")
            config = (method, budget)
            if config in seen:
                raise ValueError(f"duplicate DFlash2 tree config: {method}:{budget}")
            seen.add(config)
            configs.append(config)
    if not configs:
        raise ValueError("at least one DFlash2 tree config is required")
    return configs


def rotate_method_order(
    method_keys: list[str],
    dataset_index: int,
    turn_index: int,
    repetition_index: int,
) -> list[str]:
    """Deterministically permute the measured-method execution order.

    The permutation is a pure function of ``(dataset_index, turn_index,
    repetition_index)`` so no method is pinned to a position or predecessor
    while the order remains fully reproducible from artifact metadata.
    """
    seed = dataset_index * 1_000_003 + turn_index * 9_176 + repetition_index
    ordered = list(method_keys)
    random.Random(seed).shuffle(ordered)
    return ordered


def attach_repetition_bookkeeping(
    method_repetition_results: dict[str, list],
    designated_index: int = 0,
) -> dict[str, object]:
    """Fold repeated timing measurements into one response per method.

    Every measured repetition is preserved (for paired comparisons and
    variance analysis) by attaching the full list as ``.repetitions`` on the
    designated result. The designated repetition (``designated_index``,
    default the first) is the only one whose output advances conversation
    history and is compared against the baseline, so multi-turn semantics
    are unaffected by the repetition count. With the default of a single
    repetition, the returned object is identical to the pre-existing
    single-result shape plus a ``.repetitions`` list containing itself,
    so downstream consumers that only read the designated result are
    unaffected.
    """
    response = {}
    for method_key, repetition_results in method_repetition_results.items():
        designated = repetition_results[designated_index]
        designated.repetitions = repetition_results
        response[method_key] = designated
    return response


def repository_metadata() -> dict[str, object]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                [
                    "git",
                    "status",
                    "--porcelain",
                    "--untracked-files=no",
                ],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        commit = None
        dirty = None
    return {"commit": commit, "dirty": dirty}


def run_method(
    method_key: str,
    *,
    draft_model,
    target,
    tokenizer,
    input_ids,
    max_new_tokens: int,
    temperature: float,
    block_size: int,
    method_key_to_tree_budget: dict[str, int],
    method_key_to_tree_method: dict[str, str],
    collect_allocation_data: bool,
    prompt_id: str | None = None,
):
    """Dispatch a single generate() call for one method key.

    Shared by warmup (``prompt_id=None``) and measured timing calls so the
    method-to-generator mapping only needs to be maintained in one place.
    """
    if method_key in ("baseline", "dflash"):
        return dflash_generate(
            model=draft_model,
            target=target,
            input_ids=input_ids,
            mask_token_id=draft_model.mask_token_id,
            max_new_tokens=max_new_tokens,
            block_size=1 if method_key == "baseline" else block_size,
            stop_token_ids=[tokenizer.eos_token_id],
            temperature=temperature,
        )
    if method_key.startswith("ddtree_tb"):
        return ddtree_generate(
            model=draft_model,
            target=target,
            input_ids=input_ids,
            mask_token_id=draft_model.mask_token_id,
            max_new_tokens=max_new_tokens,
            block_size=block_size,
            tree_budget=method_key_to_tree_budget[method_key],
            stop_token_ids=[tokenizer.eos_token_id],
            temperature=temperature,
        )
    if method_key == "dflash2":
        return dflash2_generate(
            model=draft_model,
            target=target,
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            stop_token_ids=[tokenizer.eos_token_id],
            prompt_id=prompt_id,
            collect_traces=collect_allocation_data,
        )
    return dflash2_tree_generate(
        model=draft_model,
        target=target,
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        stop_token_ids=[tokenizer.eos_token_id],
        tree_budget=method_key_to_tree_budget[method_key],
        tree_method=method_key_to_tree_method[method_key],
        prompt_id=prompt_id,
        collect_tree_data=collect_allocation_data,
    )


def measure_method_with_peak_memory(
    method_key: str,
    *,
    device: torch.device,
    peak_memory_tracker: dict[str, float],
    **run_method_kwargs,
):
    """Run one measured ``run_method`` call bracketed by CUDA peak-memory
    resets so the returned result's ``peak_allocated_gib``/
    ``peak_reserved_gib`` reflect only this one method/repetition call, not
    everything that ran before it.

    ``torch.cuda.max_memory_allocated``/``max_memory_reserved`` are
    process-wide running maxima that only go up, so isolating a single
    call's peak requires resetting them immediately beforehand. That would
    normally destroy the true whole-run peak (previously read once, without
    resets, at artifact-save time), so this also folds each reading into
    ``peak_memory_tracker`` (a plain ``{"allocated_bytes": ..., "reserved_
    bytes": ...}`` dict owned by the caller) before resetting, preserving
    an accurate running maximum across every reset boundary for the
    existing run-level metadata.
    """
    peak_memory_tracker["allocated_bytes"] = max(
        peak_memory_tracker["allocated_bytes"],
        torch.cuda.max_memory_allocated(device),
    )
    peak_memory_tracker["reserved_bytes"] = max(
        peak_memory_tracker["reserved_bytes"],
        torch.cuda.max_memory_reserved(device),
    )
    torch.cuda.reset_peak_memory_stats(device)

    result = run_method(method_key, **run_method_kwargs)

    call_peak_allocated = torch.cuda.max_memory_allocated(device)
    call_peak_reserved = torch.cuda.max_memory_reserved(device)
    result.peak_allocated_gib = call_peak_allocated / 1024**3
    result.peak_reserved_gib = call_peak_reserved / 1024**3
    peak_memory_tracker["allocated_bytes"] = max(
        peak_memory_tracker["allocated_bytes"], call_peak_allocated
    )
    peak_memory_tracker["reserved_bytes"] = max(
        peak_memory_tracker["reserved_bytes"], call_peak_reserved
    )
    return result


def apply_baseline_comparison(
    response: dict[str, object],
    methods_to_run: list[str],
    *,
    comparable_to_baseline: bool,
    dataset_index: int,
) -> None:
    """Attach ``.matches_baseline`` to every repetition of every measured
    method against the sequential baseline's output ids.

    This covers the whole method matrix -- raw DFlash and original DDTree
    included, not just DFlash2 variants -- because Step 9's correctness.csv
    needs exact-output match rates for every method. A no-op when
    ``comparable_to_baseline`` is False (native-trajectory turns after the
    first, where per-method history has already diverged and a baseline
    comparison would not be meaningful).
    """
    if not comparable_to_baseline:
        return
    for method_key in methods_to_run:
        baseline_repetitions = response["baseline"].repetitions
        method_repetitions = response[method_key].repetitions
        if len(method_repetitions) != len(baseline_repetitions):
            raise ValueError(
                f"{method_key} has {len(method_repetitions)} repetitions but "
                f"baseline has {len(baseline_repetitions)}"
            )
        for repetition_result, baseline_result in zip(
            method_repetitions, baseline_repetitions
        ):
            repetition_result.matches_baseline = torch.equal(
                baseline_result.output_ids,
                repetition_result.output_ids,
            )
        if not response[method_key].matches_baseline:
            logger.warning(
                f"{method_key} output differs from the sequential baseline "
                f"for dataset index {dataset_index}. Inspect early or large "
                "divergences; occasional BF16 argmax/tree-shape differences "
                "are possible."
            )


def atomic_torch_save(value: object, path: Path) -> None:
    temporary_path = path.with_name(f"{path.name}.tmp")
    torch.save(value, temporary_path)
    temporary_path.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name-or-path", type=str, required=True)
    parser.add_argument("--model-revision", type=str, default=None)
    parser.add_argument("--draft-name-or-path", type=str, required=True)
    parser.add_argument("--draft-revision", type=str, default=None)
    parser.add_argument(
        "--draft-type",
        choices=("dflash", "dflash2"),
        default="dflash",
    )
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--tree-budget", type=str, default="16,32,64,128,256,512,1024")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--dataset-revision", type=str, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=16384)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--flash-attn", action="store_true")
    parser.add_argument("--disable-cpp-compact-cache", action="store_true")
    parser.add_argument(
        "--dflash2-tree-methods",
        default=",".join(DFLASH2_TREE_METHODS),
    )
    parser.add_argument(
        "--dflash2-tree-configs",
        help=(
            "Explicit METHOD:BUDGET[,BUDGET];... matrix. When set, this "
            "replaces the cross-product of --dflash2-tree-methods and "
            "--tree-budget."
        ),
    )
    parser.add_argument(
        "--collect-allocation-data",
        action="store_true",
        help="Persist DFlash2 proposal lattices and selected tree nodes.",
    )
    parser.add_argument(
        "--native-method-trajectories",
        action="store_true",
        help=(
            "For multi-turn datasets, feed each method its own previous "
            "assistant response instead of sharing the final method's history."
        ),
    )
    parser.add_argument(
        "--timing-repetitions",
        type=int,
        default=1,
        help=(
            "Number of times to repeat timing measurement per method per "
            "turn from the same input context. Only the first repetition "
            "advances conversation history and is compared against the "
            "baseline; every repetition's timing is preserved in the "
            "artifact for paired comparisons."
        ),
    )
    parser.add_argument("--save-path", type=str, default=None)
    args = parser.parse_args()

    if args.timing_repetitions < 1:
        parser.error("--timing-repetitions must be a positive integer")

    if args.draft_type == "dflash2":
        if args.temperature != 0.0:
            parser.error("DFlash2 benchmarking currently supports greedy decoding only")
        if args.flash_attn:
            parser.error("DFlash2 benchmarking currently supports SDPA mode only")
        requested_tree_methods = args.dflash2_tree_methods.split(",")
        invalid_tree_methods = sorted(
            set(requested_tree_methods) - set(DFLASH2_TREE_METHODS)
        )
        if invalid_tree_methods:
            parser.error(
                "unsupported DFlash2 tree methods: " + ", ".join(invalid_tree_methods)
            )
        try:
            requested_tree_configs = (
                parse_dflash2_tree_configs(args.dflash2_tree_configs)
                if args.dflash2_tree_configs
                else None
            )
        except ValueError as exc:
            parser.error(str(exc))
    else:
        requested_tree_methods = []
        requested_tree_configs = None

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    dist.init()
    if args.draft_type == "dflash2" and dist.size() != 1:
        raise RuntimeError(
            "DFlash2 proof-of-concept benchmarking currently requires one GPU"
        )
    torch.cuda.set_device(dist.local_rank())
    device = torch.device(f"cuda:{dist.local_rank()}")
    maybe_enable_cpp_compact(not args.flash_attn and not args.disable_cpp_compact_cache)

    def has_flash_attn() -> bool:
        try:
            import flash_attn  # noqa: F401

            return True
        except ImportError:
            return False

    installed_flash_attn = has_flash_attn()
    if args.draft_type == "dflash" and not installed_flash_attn:
        raise RuntimeError(
            "flash_attn must be installed because the draft DFlash model always uses FlashAttention"
        )

    target_attn_implementation = "flash_attention_2" if args.flash_attn else "sdpa"
    draft_attn_implementation = (
        "flash_attention_2" if args.draft_type == "dflash" else "sdpa"
    )

    if args.draft_type == "dflash" and not args.flash_attn and installed_flash_attn:
        logger.warning(
            "DDTree uses a custom tree attention mask on the target model. For compatibility, forcing the target verifier to torch.sdpa."
        )

    target = (
        AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            revision=args.model_revision,
            attn_implementation=target_attn_implementation,
            dtype=torch.bfloat16,
        )
        .to(device)
        .eval()
    )

    if args.draft_type == "dflash2":
        draft_model, _draft_loading_info = load_dflash2_draft_model(
            args.draft_name_or_path,
            target=target,
            revision=args.draft_revision,
            attn_implementation=draft_attn_implementation,
            dtype=torch.bfloat16,
        )
    else:
        draft_model = DFlashDraftModel.from_pretrained(
            args.draft_name_or_path,
            revision=args.draft_revision,
            attn_implementation=draft_attn_implementation,
            dtype=torch.bfloat16,
        )
    draft_model = draft_model.to(device).eval()

    if (
        args.draft_type == "dflash2"
        and args.block_size is not None
        and args.block_size != draft_model.block_size
    ):
        parser.error(
            f"DFlash2 block size is fixed by the checkpoint at {draft_model.block_size}"
        )
    block_size = (
        args.block_size if args.block_size is not None else draft_model.block_size
    )
    if args.draft_type == "dflash2" and args.tree_budget == "16,32,64,128,256,512,1024":
        args.tree_budget = "7,8,16,32,64"
    tree_budgets = [int(tree_budget) for tree_budget in args.tree_budget.split(",")]
    methods_to_run = [args.draft_type]
    method_key_to_tree_budget = {}
    method_key_to_tree_method = {}
    if args.draft_type == "dflash" and not args.flash_attn:
        ddtree_method_keys = [f"ddtree_tb{tree_budget}" for tree_budget in tree_budgets]
        methods_to_run.extend(ddtree_method_keys)
        method_key_to_tree_budget.update(
            {f"ddtree_tb{tree_budget}": tree_budget for tree_budget in tree_budgets}
        )
    elif args.draft_type == "dflash2":
        tree_configs = requested_tree_configs or [
            (tree_method, tree_budget)
            for tree_method in requested_tree_methods
            for tree_budget in tree_budgets
        ]
        for tree_method, tree_budget in tree_configs:
            method_key = f"{tree_method}_tb{tree_budget}"
            methods_to_run.append(method_key)
            method_key_to_tree_budget[method_key] = tree_budget
            method_key_to_tree_method[method_key] = tree_method

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        revision=args.model_revision,
    )
    dataset = load_and_process_dataset(args.dataset, revision=args.dataset_revision)

    if args.max_samples is not None and len(dataset) > args.max_samples:
        dataset = dataset.shuffle(seed=0).select(range(args.max_samples))

    warmup_input_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Warmup"}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    warmup_input_ids = tokenizer.encode(warmup_input_text, return_tensors="pt").to(
        target.device
    )
    warmup_max_new_tokens = min(args.max_new_tokens, 16)

    _ = run_method(
        "baseline",
        draft_model=draft_model,
        target=target,
        tokenizer=tokenizer,
        input_ids=warmup_input_ids,
        max_new_tokens=warmup_max_new_tokens,
        temperature=args.temperature,
        block_size=block_size,
        method_key_to_tree_budget=method_key_to_tree_budget,
        method_key_to_tree_method=method_key_to_tree_method,
        collect_allocation_data=args.collect_allocation_data,
    )
    for method_key in methods_to_run:
        _ = run_method(
            method_key,
            draft_model=draft_model,
            target=target,
            tokenizer=tokenizer,
            input_ids=warmup_input_ids,
            max_new_tokens=warmup_max_new_tokens,
            temperature=args.temperature,
            block_size=block_size,
            method_key_to_tree_budget=method_key_to_tree_budget,
            method_key_to_tree_method=method_key_to_tree_method,
            collect_allocation_data=args.collect_allocation_data,
        )

    save_path = Path(args.save_path) if args.save_path is not None else None
    peak_memory_tracker = {
        "allocated_bytes": float(torch.cuda.max_memory_allocated(device)),
        "reserved_bytes": float(torch.cuda.max_memory_reserved(device)),
    }
    partial_path = (
        save_path.with_name(f"{save_path.stem}.partial{save_path.suffix}")
        if save_path is not None and dist.size() == 1
        else None
    )
    responses = []
    completed_dataset_indices: set[int] = set()
    if partial_path is not None and partial_path.exists():
        partial_run = torch.load(
            partial_path,
            map_location="cpu",
            weights_only=False,
        )
        if partial_run.get("args") != vars(args):
            raise ValueError(f"{partial_path} was created with different arguments")
        responses = list(partial_run["responses"])
        completed_dataset_indices = set(partial_run["completed_dataset_indices"])
        logger.info(
            f"Resuming {partial_path} after "
            f"{len(completed_dataset_indices)} completed samples"
        )

    def build_run_data() -> dict:
        dflash2_matches = [
            response["dflash2"].matches_baseline
            for response in responses
            if (
                "dflash2" in response
                and isinstance(
                    getattr(response["dflash2"], "matches_baseline", None),
                    bool,
                )
            )
        ]
        tree_matches = {
            method_key: [
                response[method_key].matches_baseline
                for response in responses
                if (
                    method_key in response
                    and isinstance(
                        getattr(
                            response[method_key],
                            "matches_baseline",
                            None,
                        ),
                        bool,
                    )
                )
            ]
            for method_key in method_key_to_tree_method
        }
        run_data = {
            "responses": responses,
            "completed_dataset_indices": sorted(completed_dataset_indices),
            "block_size": block_size,
            "draft_type": args.draft_type,
            "draft_attn_implementation": draft_attn_implementation,
            "target_attn_implementation": (target_attn_implementation),
            "target_dtype": str(target.dtype),
            "draft_dtype": str(draft_model.dtype),
            "timing_repetitions": args.timing_repetitions,
            "method_order_seed_formula": (
                "dataset_index * 1_000_003 + turn_index * 9_176 + repetition_index, "
                "then random.Random(seed).shuffle(method_keys)"
            ),
            "trajectory_mode": (
                "native" if args.native_method_trajectories else "controlled_shared"
            ),
            "args": vars(args),
            "runtime": {
                "python": platform.python_version(),
                "pytorch": torch.__version__,
                "transformers": transformers.__version__,
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(device),
                "peak_allocated_gib": (
                    max(
                        peak_memory_tracker["allocated_bytes"],
                        torch.cuda.max_memory_allocated(device),
                    )
                    / 1024**3
                ),
                "peak_reserved_gib": (
                    max(
                        peak_memory_tracker["reserved_bytes"],
                        torch.cuda.max_memory_reserved(device),
                    )
                    / 1024**3
                ),
                "decode_timing_excludes_first_draft_prefill": True,
                "total_generation_time_includes_first_draft_prefill": True,
            },
            "repository": repository_metadata(),
            "target_revision": getattr(
                target.config,
                "_commit_hash",
                None,
            )
            or args.model_revision,
            "draft_revision": getattr(
                draft_model.config,
                "_commit_hash",
                None,
            )
            or args.draft_revision,
        }
        if args.draft_type == "dflash2":
            run_data["exact_token_match_count"] = sum(dflash2_matches)
            run_data["exact_token_match_total"] = len(dflash2_matches)
            run_data["tree_exact_token_matches"] = {
                method: {
                    "count": sum(matches),
                    "total": len(matches),
                }
                for method, matches in tree_matches.items()
            }
        return run_data

    indices = list(range(dist.rank(), len(dataset), dist.size()))
    pending_indices = [idx for idx in indices if idx not in completed_dataset_indices]
    progress = tqdm(
        pending_indices,
        disable=not dist.is_main(),
        total=len(indices),
        initial=len(indices) - len(pending_indices),
    )
    for idx in progress:
        instance = dataset[idx]
        shared_messages = []
        native_messages = {
            method_key: [] for method_key in ("baseline", *methods_to_run)
        }
        instance_responses = []
        for turn_index, user_content in enumerate(instance["turns"]):
            all_method_keys = ("baseline", *methods_to_run)

            # Advance conversational context exactly once per turn, before
            # any method runs, so the rotated per-repetition execution
            # order below cannot change which methods see the latest user
            # turn (previously this only worked because "baseline" always
            # ran first and happened to be the one appending it in shared
            # mode).
            if args.native_method_trajectories:
                for method_key in all_method_keys:
                    native_messages[method_key].append(
                        {"role": "user", "content": user_content}
                    )
            else:
                shared_messages.append({"role": "user", "content": user_content})

            method_inputs = {}
            for method_key in all_method_keys:
                messages = (
                    native_messages[method_key]
                    if args.native_method_trajectories
                    else shared_messages
                )
                input_text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                input_ids = tokenizer.encode(
                    input_text,
                    return_tensors="pt",
                ).to(target.device)
                prompt_hash = hashlib.sha256(input_text.encode("utf-8")).hexdigest()
                prompt_id = (
                    f"{args.dataset}:selected:{idx}:turn-{turn_index}:"
                    f"{method_key}:{prompt_hash[:12]}"
                )
                method_inputs[method_key] = (input_ids, prompt_id, prompt_hash)

            # Repeat timing measurement from the same input context
            # (`method_inputs` is built once above, per turn) for every
            # configured repetition, rotating the measured method order
            # deterministically so no method is always first or last.
            method_repetition_results: dict[str, list] = {
                method_key: [] for method_key in all_method_keys
            }
            for repetition_index in range(args.timing_repetitions):
                execution_order = rotate_method_order(
                    list(all_method_keys),
                    dataset_index=idx,
                    turn_index=turn_index,
                    repetition_index=repetition_index,
                )
                for method_key in execution_order:
                    input_ids, prompt_id, prompt_hash = method_inputs[method_key]
                    result = measure_method_with_peak_memory(
                        method_key,
                        device=device,
                        peak_memory_tracker=peak_memory_tracker,
                        draft_model=draft_model,
                        target=target,
                        tokenizer=tokenizer,
                        input_ids=input_ids,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        block_size=block_size,
                        method_key_to_tree_budget=method_key_to_tree_budget,
                        method_key_to_tree_method=method_key_to_tree_method,
                        collect_allocation_data=args.collect_allocation_data,
                        prompt_id=prompt_id,
                    )
                    result.repetition_index = repetition_index
                    result.execution_order = execution_order
                    result.prompt_id = prompt_id
                    result.prompt_hash = prompt_hash
                    method_repetition_results[method_key].append(result)

            # Only the designated (first) repetition advances history;
            # every repetition is compared against the corresponding
            # baseline repetition and kept for paired timing comparisons.
            response = attach_repetition_bookkeeping(method_repetition_results)

            comparable_to_baseline = (
                not args.native_method_trajectories or turn_index == 0
            )
            # Every measured method (original DFlash/DDTree as well as
            # DFlash2 variants) is compared against the sequential baseline
            # so Step 9's correctness.csv can report exact-output match
            # rates for the whole method matrix, not just DFlash2. Only
            # turn 0 is comparable in native-trajectory mode, since later
            # turns intentionally diverge history.
            apply_baseline_comparison(
                response,
                methods_to_run,
                comparable_to_baseline=comparable_to_baseline,
                dataset_index=idx,
            )

            if args.native_method_trajectories:
                for method_key in all_method_keys:
                    result = response[method_key]
                    generated_ids = result.output_ids[
                        0,
                        result.num_input_tokens :,
                    ]
                    native_messages[method_key].append(
                        {
                            "role": "assistant",
                            "content": tokenizer.decode(
                                generated_ids,
                                skip_special_tokens=True,
                            ),
                        }
                    )
            else:
                baseline_response = response["baseline"]
                generated_ids = baseline_response.output_ids[
                    0,
                    baseline_response.num_input_tokens :,
                ]
                shared_messages.append(
                    {
                        "role": "assistant",
                        "content": tokenizer.decode(
                            generated_ids,
                            skip_special_tokens=True,
                        ),
                    }
                )
            instance_responses.append(response)
        responses.extend(instance_responses)
        completed_dataset_indices.add(idx)
        if partial_path is not None:
            partial_path.parent.mkdir(parents=True, exist_ok=True)
            atomic_torch_save(build_run_data(), partial_path)

    if dist.size() > 1:
        responses = dist.gather(responses, dst=0)
        if not dist.is_main():
            return
        responses = list(chain(*responses))

    run_data = build_run_data()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_torch_save(run_data, save_path)
        if partial_path is not None:
            partial_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
