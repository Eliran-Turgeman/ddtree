#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from model import load_dflash2_draft_model


EXPECTED_TARGET_LAYERS = [5, 19, 33, 47, 61]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load and inspect the official 27B DFlash2 model pair."
    )
    parser.add_argument(
        "--target",
        default="Qwen/Qwen3.8-27B",
    )
    parser.add_argument(
        "--target-revision",
        default="1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
    )
    parser.add_argument(
        "--draft",
        default="incoai/Qwen3.8-27B-DFlash2",
    )
    parser.add_argument(
        "--draft-revision",
        default="dedf8df68adfb1afeaf7b7480c0a0243108177b4",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def gibibytes(value: int) -> float:
    return value / 1024**3


def memory_snapshot() -> dict[str, float]:
    return {
        "allocated_gib": gibibytes(torch.cuda.memory_allocated()),
        "reserved_gib": gibibytes(torch.cuda.memory_reserved()),
        "peak_allocated_gib": gibibytes(torch.cuda.max_memory_allocated()),
        "peak_reserved_gib": gibibytes(torch.cuda.max_memory_reserved()),
    }


def parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Step-7 compatibility run")

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

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
    torch.cuda.synchronize()
    target_memory = memory_snapshot()
    target_config = target.config.get_text_config(decoder=True)

    torch.cuda.reset_peak_memory_stats()
    draft, loading_info = load_dflash2_draft_model(
        args.draft,
        target=target,
        revision=args.draft_revision,
        attn_implementation="sdpa",
        dtype=torch.bfloat16,
    )
    draft = draft.to(device).eval()
    torch.cuda.synchronize()
    pair_memory = memory_snapshot()

    expected_draft = {
        "vocab_size": 248320,
        "hidden_size": 5120,
        "num_hidden_layers": 5,
        "block_size": 8,
        "conv_kernel_size": 2,
        "conv_group_size": 16,
        "selector_rank": 256,
        "selector_top_k": 16,
        "target_layer_ids": EXPECTED_TARGET_LAYERS,
    }
    actual_draft = {
        "vocab_size": draft.config.vocab_size,
        "hidden_size": draft.config.hidden_size,
        "num_hidden_layers": draft.config.num_hidden_layers,
        "block_size": draft.block_size,
        "conv_kernel_size": draft.config.conv_kernel_size,
        "conv_group_size": draft.config.conv_group_size,
        "selector_rank": draft.config.selector_rank,
        "selector_top_k": draft.config.selector_top_k,
        "target_layer_ids": draft.target_layer_ids,
    }
    if actual_draft != expected_draft:
        raise ValueError(
            "official DFlash2 config does not match the audited contract: "
            f"{actual_draft}"
        )
    if target_config.vocab_size != draft.config.vocab_size:
        raise ValueError(
            "target and draft vocabulary sizes differ: "
            f"{target_config.vocab_size} != {draft.config.vocab_size}"
        )
    if target_config.hidden_size != draft.config.hidden_size:
        raise ValueError(
            "target and draft hidden sizes differ: "
            f"{target_config.hidden_size} != {draft.config.hidden_size}"
        )
    if max(draft.target_layer_ids) >= target_config.num_hidden_layers:
        raise ValueError("draft target hidden-layer ID is out of range")

    layer_types = list(target_config.layer_types)
    result = {
        "target": args.target,
        "target_revision": getattr(
            target.config,
            "_commit_hash",
            None,
        )
        or args.target_revision,
        "target_class": type(target).__name__,
        "target_parameters": parameter_count(target),
        "target_config": {
            "vocab_size": target_config.vocab_size,
            "hidden_size": target_config.hidden_size,
            "num_hidden_layers": target_config.num_hidden_layers,
            "linear_attention_layers": layer_types.count("linear_attention"),
            "full_attention_layers": layer_types.count("full_attention"),
            "max_position_embeddings": (target_config.max_position_embeddings),
            "dtype": str(target.dtype),
        },
        "memory_after_target_load": target_memory,
        "draft": args.draft,
        "draft_revision": getattr(
            draft.config,
            "_commit_hash",
            None,
        )
        or args.draft_revision,
        "draft_class": type(draft).__name__,
        "draft_parameters": parameter_count(draft),
        "draft_shared_target_weight_keys": loading_info["shared_target_weight_keys"],
        "draft_config": actual_draft,
        "memory_after_draft_load": pair_memory,
        "runtime": {
            "gpu": torch.cuda.get_device_name(device),
            "python_torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
