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
    DFlash2DraftModel,
    DFlashDraftModel,
    load_and_process_dataset,
)
from dflash import dflash_generate
from dflash2 import dflash2_generate
from dflash2_tree import (
    DFLASH2_TREE_METHODS,
    dflash2_tree_generate,
)
from ddtree import ddtree_generate, maybe_enable_cpp_compact


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name-or-path", type=str, required=True)
    parser.add_argument("--draft-name-or-path", type=str, required=True)
    parser.add_argument(
        "--draft-type",
        choices=("dflash", "dflash2"),
        default="dflash",
    )
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--tree-budget", type=str, default="16,32,64,128,256,512,1024")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=16384)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--flash-attn", action="store_true")
    parser.add_argument("--disable-cpp-compact-cache", action="store_true")
    parser.add_argument("--save-path", type=str, default=None)
    args = parser.parse_args()

    if args.draft_type == "dflash2":
        if args.temperature != 0.0:
            parser.error("DFlash2 benchmarking currently supports greedy decoding only")
        if args.flash_attn:
            parser.error("DFlash2 benchmarking currently supports SDPA mode only")

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
    maybe_enable_cpp_compact(
        not args.flash_attn
        and not args.disable_cpp_compact_cache
    )

    def has_flash_attn() -> bool:
        try:
            import flash_attn  # noqa: F401
            return True
        except ImportError:
            return False

    installed_flash_attn = has_flash_attn()
    if args.draft_type == "dflash" and not installed_flash_attn:
        raise RuntimeError("flash_attn must be installed because the draft DFlash model always uses FlashAttention")

    target_attn_implementation = "flash_attention_2" if args.flash_attn else "sdpa"
    draft_attn_implementation = (
        "flash_attention_2"
        if args.draft_type == "dflash"
        else "sdpa"
    )

    if args.draft_type == "dflash" and not args.flash_attn and installed_flash_attn:
        logger.warning("DDTree uses a custom tree attention mask on the target model. For compatibility, forcing the target verifier to torch.sdpa.")

    target = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        attn_implementation=target_attn_implementation,
        dtype=torch.bfloat16,
    ).to(device).eval()

    draft_model_class = (
        DFlashDraftModel
        if args.draft_type == "dflash"
        else DFlash2DraftModel
    )
    draft_model = draft_model_class.from_pretrained(
        args.draft_name_or_path,
        attn_implementation=draft_attn_implementation,
        dtype=torch.bfloat16,
    ).to(device).eval()

    if (
        args.draft_type == "dflash2"
        and args.block_size is not None
        and args.block_size != draft_model.block_size
    ):
        parser.error(
            "DFlash2 block size is fixed by the checkpoint "
            f"at {draft_model.block_size}"
        )
    block_size = (
        args.block_size
        if args.block_size is not None
        else draft_model.block_size
    )
    if (
        args.draft_type == "dflash2"
        and args.tree_budget == "16,32,64,128,256,512,1024"
    ):
        args.tree_budget = "7,8,16,32,64"
    tree_budgets = [int(tree_budget) for tree_budget in args.tree_budget.split(",")]
    methods_to_run = [args.draft_type]
    method_key_to_tree_budget = {}
    method_key_to_tree_method = {}
    if args.draft_type == "dflash" and not args.flash_attn:
        ddtree_method_keys = [f"ddtree_tb{tree_budget}" for tree_budget in tree_budgets]
        methods_to_run.extend(ddtree_method_keys)
        method_key_to_tree_budget.update({f"ddtree_tb{tree_budget}": tree_budget for tree_budget in tree_budgets})
    elif args.draft_type == "dflash2":
        for tree_method in DFLASH2_TREE_METHODS:
            for tree_budget in tree_budgets:
                method_key = f"{tree_method}_tb{tree_budget}"
                methods_to_run.append(method_key)
                method_key_to_tree_budget[method_key] = tree_budget
                method_key_to_tree_method[method_key] = tree_method

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    dataset = load_and_process_dataset(args.dataset)

    if args.max_samples is not None and len(dataset) > args.max_samples:
        dataset = dataset.shuffle(seed=0).select(range(args.max_samples))

    warmup_input_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Warmup"}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    warmup_input_ids = tokenizer.encode(warmup_input_text, return_tensors="pt").to(target.device)
    warmup_max_new_tokens = min(args.max_new_tokens, 16)

    _ = dflash_generate(
        model=draft_model,
        target=target,
        input_ids=warmup_input_ids,
        mask_token_id=draft_model.mask_token_id,
        max_new_tokens=warmup_max_new_tokens,
        block_size=1,
        stop_token_ids=[tokenizer.eos_token_id],
        temperature=args.temperature,
    )
    for method_key in methods_to_run:
        if method_key == "dflash":
            _ = dflash_generate(
                model=draft_model,
                target=target,
                input_ids=warmup_input_ids,
                mask_token_id=draft_model.mask_token_id,
                max_new_tokens=warmup_max_new_tokens,
                block_size=block_size,
                stop_token_ids=[tokenizer.eos_token_id],
                temperature=args.temperature,
            )
        elif method_key.startswith("ddtree_tb"):
            _ = ddtree_generate(
                model=draft_model,
                target=target,
                input_ids=warmup_input_ids,
                mask_token_id=draft_model.mask_token_id,
                max_new_tokens=warmup_max_new_tokens,
                block_size=block_size,
                tree_budget=method_key_to_tree_budget[method_key],
                stop_token_ids=[tokenizer.eos_token_id],
                temperature=args.temperature,
            )
        elif method_key == "dflash2":
            _ = dflash2_generate(
                model=draft_model,
                target=target,
                input_ids=warmup_input_ids,
                max_new_tokens=warmup_max_new_tokens,
                stop_token_ids=[tokenizer.eos_token_id],
            )
        else:
            _ = dflash2_tree_generate(
                model=draft_model,
                target=target,
                input_ids=warmup_input_ids,
                max_new_tokens=warmup_max_new_tokens,
                stop_token_ids=[tokenizer.eos_token_id],
                tree_budget=method_key_to_tree_budget[method_key],
                tree_method=method_key_to_tree_method[method_key],
            )

    responses = []
    dflash2_token_matches = []
    tree_token_matches = {}
    indices = range(dist.rank(), len(dataset), dist.size())
    for idx in tqdm(indices, disable=not dist.is_main()):
        instance = dataset[idx]
        messages = []
        for turn_index, user_content in enumerate(instance["turns"]):
            messages.append({"role": "user", "content": user_content})
            input_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            input_ids = tokenizer.encode(input_text, return_tensors="pt").to(target.device)
            prompt_hash = hashlib.sha256(
                input_text.encode("utf-8")
            ).hexdigest()
            prompt_id = (
                f"{args.dataset}:selected:{idx}:turn-{turn_index}:"
                f"{prompt_hash[:12]}"
            )

            response = {}
            response["baseline"] = dflash_generate(
                model=draft_model,
                target=target,
                input_ids=input_ids,
                mask_token_id=draft_model.mask_token_id,
                max_new_tokens=args.max_new_tokens,
                block_size=1,
                stop_token_ids=[tokenizer.eos_token_id],
                temperature=args.temperature,
            )
            for method_key in methods_to_run:
                if method_key == "dflash":
                    response[method_key] = dflash_generate(
                        model=draft_model,
                        target=target,
                        input_ids=input_ids,
                        mask_token_id=draft_model.mask_token_id,
                        max_new_tokens=args.max_new_tokens,
                        block_size=block_size,
                        stop_token_ids=[tokenizer.eos_token_id],
                        temperature=args.temperature,
                    )
                elif method_key.startswith("ddtree_tb"):
                    response[method_key] = ddtree_generate(
                        model=draft_model,
                        target=target,
                        input_ids=input_ids,
                        mask_token_id=draft_model.mask_token_id,
                        max_new_tokens=args.max_new_tokens,
                        block_size=block_size,
                        tree_budget=method_key_to_tree_budget[method_key],
                        stop_token_ids=[tokenizer.eos_token_id],
                        temperature=args.temperature,
                    )
                elif method_key == "dflash2":
                    response[method_key] = dflash2_generate(
                        model=draft_model,
                        target=target,
                        input_ids=input_ids,
                        max_new_tokens=args.max_new_tokens,
                        stop_token_ids=[tokenizer.eos_token_id],
                        prompt_id=prompt_id,
                    )
                    matches_baseline = torch.equal(
                        response["baseline"].output_ids,
                        response[method_key].output_ids,
                    )
                    tree_token_matches.setdefault(method_key, []).append(
                        response[method_key].matches_baseline
                    )
                    if not response[method_key].matches_baseline:
                        logger.warning(
                            f"{method_key} output differs from the "
                            "sequential baseline for dataset index "
                            f"{idx}. Inspect early or large divergences; "
                            "occasional BF16 tree-shape argmax differences "
                            "are possible."
                        )
                    response[method_key].matches_baseline = matches_baseline
                    dflash2_token_matches.append(matches_baseline)
                    if not matches_baseline:
                        logger.warning(
                            "DFlash2 output differs from the sequential baseline "
                            f"for dataset index {idx}. This can occur near BF16 "
                            "argmax ties because block verification uses different "
                            "matrix shapes."
                        )
                else:
                    response[method_key] = dflash2_tree_generate(
                        model=draft_model,
                        target=target,
                        input_ids=input_ids,
                        max_new_tokens=args.max_new_tokens,
                        stop_token_ids=[tokenizer.eos_token_id],
                        tree_budget=method_key_to_tree_budget[method_key],
                        tree_method=method_key_to_tree_method[method_key],
                        prompt_id=prompt_id,
                    )
                    response[method_key].matches_baseline = torch.equal(
                        response["baseline"].output_ids,
                        response[method_key].output_ids,
                    )

            spec_response = response[methods_to_run[-1]]
            generated_ids = spec_response.output_ids[0, spec_response.num_input_tokens :]
            output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            messages.append({"role": "assistant", "content": output_text})
            responses.append(response)

    if dist.size() > 1:
        responses = dist.gather(responses, dst=0)
        if not dist.is_main():
            return
        responses = list(chain(*responses))

    run_data = {
        "responses": responses,
        "block_size": block_size,
        "draft_type": args.draft_type,
        "draft_attn_implementation": draft_attn_implementation,
        "target_attn_implementation": target_attn_implementation,
        "args": vars(args),
        "runtime": {
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device),
            "decode_timing_excludes_first_draft_prefill": True,
        },
        "repository": repository_metadata(),
        "target_revision": getattr(target.config, "_commit_hash", None),
        "draft_revision": getattr(
            draft_model.config,
            "_commit_hash",
            None,
        ),
    }
    if args.draft_type == "dflash2":
        run_data["exact_token_match_count"] = sum(dflash2_token_matches)
        run_data["exact_token_match_total"] = len(dflash2_token_matches)
        run_data["tree_exact_token_matches"] = {
            method: {
                "count": sum(matches),
                "total": len(matches),
            }
            for method, matches in tree_token_matches.items()
        }
    
    if args.save_path is not None:
        save_path = Path(args.save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(run_data, save_path)


if __name__ == "__main__":
    main()
