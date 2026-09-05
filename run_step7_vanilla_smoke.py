#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from dflash2 import dflash2_generate
from model import load_dflash2_draft_model


TARGET_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
DRAFT_REVISION = "dedf8df68adfb1afeaf7b7480c0a0243108177b4"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one official 27B vanilla DFlash2 smoke prompt."
    )
    parser.add_argument("--target", default="Qwen/Qwen3.8-27B")
    parser.add_argument("--target-revision", default=TARGET_REVISION)
    parser.add_argument(
        "--draft",
        default="incoai/Qwen3.8-27B-DFlash2",
    )
    parser.add_argument("--draft-revision", default=DRAFT_REVISION)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def gibibytes(value: int) -> float:
    return value / 1024**3


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

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    result = dflash2_generate(
        model=draft,
        target=target,
        input_ids=input_ids,
        max_new_tokens=args.max_new_tokens,
        stop_token_ids=[tokenizer.eos_token_id],
        collect_traces=True,
        prompt_id="step7-vanilla-smoke",
    )
    torch.cuda.synchronize()

    if not result.trace_rounds:
        raise RuntimeError("vanilla DFlash2 produced no verification rounds")
    if any(
        trace["selected_draft_token_ids"].numel() != 7 for trace in result.trace_rounds
    ):
        raise RuntimeError("official DFlash2 did not reconstruct seven-token paths")

    mean_matched = sum(result.matched_draft_tokens_per_round) / len(
        result.matched_draft_tokens_per_round
    )
    mean_committed = sum(result.committed_tokens_per_round) / len(
        result.committed_tokens_per_round
    )
    output = {
        "target": args.target,
        "target_revision": args.target_revision,
        "draft": args.draft,
        "draft_revision": args.draft_revision,
        "shared_target_weight_keys": loading_info["shared_target_weight_keys"],
        "prompt_tokens": result.num_input_tokens,
        "output_tokens": result.num_output_tokens,
        "decode_rounds": result.decode_rounds,
        "mean_matched_draft_tokens": mean_matched,
        "mean_committed_tokens_per_round": mean_committed,
        "full_block_acceptance": (
            sum(value == 7 for value in result.matched_draft_tokens_per_round)
            / result.decode_rounds
        ),
        "tokens_per_second": 1 / result.time_per_output_token,
        "milliseconds_per_token": result.time_per_output_token * 1000,
        "stage_times_seconds": result.stage_times,
        "peak_allocated_gib": gibibytes(torch.cuda.max_memory_allocated()),
        "peak_reserved_gib": gibibytes(torch.cuda.max_memory_reserved()),
        "generated_text": tokenizer.decode(
            result.output_ids[0, result.num_input_tokens :],
            skip_special_tokens=True,
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
