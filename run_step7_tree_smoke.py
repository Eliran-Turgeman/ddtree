#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from dflash2 import dflash2_generate
from dflash2_tree import (
    DFLASH2_PAIRWISE_K16,
    DFLASH2_UNARY_K16,
    dflash2_tree_generate,
)
from model import load_dflash2_draft_model
from run_step7_vanilla_smoke import DRAFT_REVISION, TARGET_REVISION


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the official 27B DFlash2 tree compatibility smoke."
    )
    parser.add_argument("--target", default="Qwen/Qwen3.8-27B")
    parser.add_argument("--target-revision", default=TARGET_REVISION)
    parser.add_argument(
        "--draft",
        default="incoai/Qwen3.8-27B-DFlash2",
    )
    parser.add_argument("--draft-revision", default=DRAFT_REVISION)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--tree-budget", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def summarize(result) -> dict[str, object]:
    matched = result.matched_draft_tokens_per_round
    committed = result.committed_tokens_per_round
    return {
        "output_tokens": result.num_output_tokens,
        "decode_rounds": result.decode_rounds,
        "mean_matched_draft_tokens": sum(matched) / len(matched),
        "mean_committed_tokens_per_round": sum(committed) / len(committed),
        "full_block_acceptance": (sum(value == 7 for value in matched) / len(matched)),
        "tokens_per_second": 1 / result.time_per_output_token,
        "milliseconds_per_token": result.time_per_output_token * 1000,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
        "round_metrics": result.round_metrics,
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Step-7 smoke run")

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    tokenizer = AutoTokenizer.from_pretrained(
        args.target,
        revision=args.target_revision,
    )
    target = (
        AutoModelForCausalLM.from_pretrained(
            args.target,
            revision=args.target_revision,
            attn_implementation="sdpa",
            dtype=torch.bfloat16,
        )
        .to(device)
        .eval()
    )
    draft, loading_info = load_dflash2_draft_model(
        args.draft,
        target=target,
        revision=args.draft_revision,
        attn_implementation="sdpa",
        dtype=torch.bfloat16,
    )
    draft = draft.to(device).eval()
    prompt = tokenizer.apply_chat_template(
        [
            {
                "role": "user",
                "content": (
                    "If a box has 3 rows of 4 apples, how many apples "
                    "are in the box? Give only the number."
                ),
            }
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    results = {}
    output_token_ids = {}
    methods = (
        ("dflash2_greedy", None),
        (DFLASH2_UNARY_K16, DFLASH2_UNARY_K16),
        (DFLASH2_PAIRWISE_K16, DFLASH2_PAIRWISE_K16),
    )
    for method_name, tree_method in methods:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        if tree_method is None:
            result = dflash2_generate(
                model=draft,
                target=target,
                input_ids=input_ids,
                max_new_tokens=args.max_new_tokens,
                stop_token_ids=[tokenizer.eos_token_id],
                prompt_id="step7-tree-smoke",
            )
        else:
            result = dflash2_tree_generate(
                model=draft,
                target=target,
                input_ids=input_ids,
                max_new_tokens=args.max_new_tokens,
                stop_token_ids=[tokenizer.eos_token_id],
                tree_budget=args.tree_budget,
                tree_method=tree_method,
                prompt_id="step7-tree-smoke",
                collect_tree_data=True,
            )
        torch.cuda.synchronize()
        results[method_name] = summarize(result)
        output_token_ids[method_name] = result.output_ids[
            0, result.num_input_tokens :
        ].tolist()

    reference_tokens = output_token_ids["dflash2_greedy"]
    if any(tokens != reference_tokens for tokens in output_token_ids.values()):
        raise RuntimeError(
            f"tree methods diverged from greedy target output: {output_token_ids}"
        )
    output = {
        "target": args.target,
        "target_revision": args.target_revision,
        "draft": args.draft,
        "draft_revision": args.draft_revision,
        "shared_target_weight_keys": loading_info["shared_target_weight_keys"],
        "tree_budget": args.tree_budget,
        "generated_text": tokenizer.decode(
            reference_tokens,
            skip_special_tokens=True,
        ),
        "methods": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
