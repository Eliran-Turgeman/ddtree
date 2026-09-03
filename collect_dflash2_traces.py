#!/usr/bin/env python3

import argparse
import hashlib
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import transformers
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from dflash2 import dflash2_generate
from model import DFlash2DraftModel, load_and_process_dataset


TRACE_FORMAT = "ddtree.dflash2_trace"
TRACE_FORMAT_VERSION = 1
TRACE_SOURCE_FILES = (
    "collect_dflash2_traces.py",
    "dflash2.py",
    "inspect_dflash2_traces.py",
    "model/dflash2.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect raw DFlash2 proposal and verifier traces."
    )
    parser.add_argument("output_path", type=Path)
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-4B",
        help="Target model name or path.",
    )
    parser.add_argument(
        "--draft-model",
        default="mgoin/Qwen3-4B-speculator.dflash2",
        help="DFlash2 checkpoint name or path.",
    )
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--draft-revision", default=None)
    parser.add_argument("--dataset", default="gsm8k")
    parser.add_argument("--max-samples", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def set_deterministic_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def git_metadata() -> dict[str, object]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    source_hashes = {}
    for source_path in TRACE_SOURCE_FILES:
        path = Path(source_path)
        source_hashes[source_path] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    return {
        "commit": commit,
        "dirty": bool(status.strip()),
        "trace_source_sha256": source_hashes,
    }


def resolved_model_revision(model: torch.nn.Module) -> str | None:
    return getattr(model.config, "_commit_hash", None)


def build_runtime_metadata(
    args: argparse.Namespace,
    target: torch.nn.Module,
    draft_model: DFlash2DraftModel,
    tokenizer: object,
    dataset: object,
    dataset_split: str,
    dataset_config: str | None,
) -> dict[str, object]:
    device = torch.device(args.device)
    device_properties = torch.cuda.get_device_properties(device)
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository": git_metadata(),
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "pytorch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda": torch.version.cuda,
            "gpu": device_properties.name,
            "gpu_total_memory_bytes": device_properties.total_memory,
            "gpu_compute_capability": [
                device_properties.major,
                device_properties.minor,
            ],
            "dtype": str(next(target.parameters()).dtype),
            "target_attention": "sdpa",
            "draft_attention": "sdpa",
        },
        "target_model": {
            "name_or_path": args.model,
            "requested_revision": args.model_revision,
            "resolved_revision": resolved_model_revision(target),
        },
        "draft_model": {
            "name_or_path": args.draft_model,
            "requested_revision": args.draft_revision,
            "resolved_revision": resolved_model_revision(draft_model),
            "num_hidden_layers": draft_model.config.num_hidden_layers,
            "target_layer_ids": list(draft_model.target_layer_ids),
            "hidden_size": draft_model.config.hidden_size,
            "block_size": draft_model.block_size,
            "speculative_positions": draft_model.block_size - 1,
            "selector_top_k": draft_model.candidate_selector.top_k,
            "selector_rank": draft_model.config.selector_rank,
            "conv_kernel_size": draft_model.config.conv_kernel_size,
            "conv_group_size": draft_model.config.conv_group_size,
            "dynamic_convolutions_per_layer": 2,
        },
        "tokenizer": {
            "name_or_path": getattr(tokenizer, "name_or_path", args.model),
            "eos_token_id": tokenizer.eos_token_id,
        },
        "dataset": {
            "name": args.dataset,
            "config": dataset_config,
            "split": dataset_split,
            "fingerprint_before_selection": getattr(dataset, "_fingerprint", None),
            "selection": "seeded_shuffle_then_first_n",
            "seed": args.seed,
            "max_samples": args.max_samples,
        },
        "decoding": {
            "temperature": 0.0,
            "max_new_tokens": args.max_new_tokens,
            "stop_token_ids": [tokenizer.eos_token_id],
        },
        "target_token_semantics": {
            "verifier_token_ids": (
                "Target argmax tokens from the current block verification. "
                "Only entries marked by directly_observed_target_mask are "
                "conditioned on the true selected prefix."
            ),
            "realized_continuation_token_ids": (
                "The eventual committed DFlash2 output after the round anchor. "
                "Positions after the first mismatch may come from later "
                "verification rounds and are not direct observations from the "
                "current verifier call."
            ),
        },
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("DFlash2 trace collection requires a CUDA GPU")
    if args.max_samples <= 0:
        raise ValueError("--max-samples must be positive")

    set_deterministic_seed(args.seed)
    device = torch.device(args.device)
    torch.cuda.set_device(device)

    target_kwargs = {
        "attn_implementation": "sdpa",
        "dtype": torch.bfloat16,
    }
    if args.model_revision is not None:
        target_kwargs["revision"] = args.model_revision
    target = AutoModelForCausalLM.from_pretrained(
        args.model,
        **target_kwargs,
    ).to(device).eval()

    draft_kwargs = {
        "attn_implementation": "sdpa",
        "dtype": torch.bfloat16,
    }
    if args.draft_revision is not None:
        draft_kwargs["revision"] = args.draft_revision
    draft_model = DFlash2DraftModel.from_pretrained(
        args.draft_model,
        **draft_kwargs,
    ).to(device).eval()

    tokenizer_kwargs = {}
    if args.model_revision is not None:
        tokenizer_kwargs["revision"] = args.model_revision
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        **tokenizer_kwargs,
    )

    dataset = load_and_process_dataset(args.dataset)
    dataset_split = str(getattr(dataset, "split", "unknown"))
    dataset_config = getattr(getattr(dataset, "info", None), "config_name", None)
    dataset = dataset.add_column(
        "_trace_dataset_index",
        list(range(len(dataset))),
    )
    dataset_fingerprint = getattr(dataset, "_fingerprint", None)
    if len(dataset) > args.max_samples:
        dataset = dataset.shuffle(seed=args.seed).select(
            range(args.max_samples)
        )

    metadata = build_runtime_metadata(
        args,
        target,
        draft_model,
        tokenizer,
        dataset,
        dataset_split,
        dataset_config,
    )
    metadata["dataset"]["fingerprint_before_selection"] = dataset_fingerprint

    warmup_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Warmup"}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    warmup_ids = tokenizer.encode(
        warmup_text,
        return_tensors="pt",
    ).to(device)
    dflash2_generate(
        model=draft_model,
        target=target,
        input_ids=warmup_ids,
        max_new_tokens=min(args.max_new_tokens, 16),
        stop_token_ids=[tokenizer.eos_token_id],
    )

    prompts = []
    for instance in tqdm(dataset, desc="Collecting DFlash2 traces"):
        messages = []
        dataset_index = int(instance["_trace_dataset_index"])
        for turn_index, user_content in enumerate(instance["turns"]):
            messages.append({"role": "user", "content": user_content})
            input_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            input_ids = tokenizer.encode(
                input_text,
                return_tensors="pt",
            ).to(device)
            prompt_hash = hashlib.sha256(
                input_text.encode("utf-8")
            ).hexdigest()
            prompt_id = (
                f"{args.dataset}:{dataset_split}:{dataset_index}:"
                f"turn-{turn_index}:{prompt_hash[:12]}"
            )

            result = dflash2_generate(
                model=draft_model,
                target=target,
                input_ids=input_ids,
                max_new_tokens=args.max_new_tokens,
                stop_token_ids=[tokenizer.eos_token_id],
                collect_traces=True,
            )
            generated_ids = result.output_ids[
                0,
                result.num_input_tokens:,
            ]
            output_text = tokenizer.decode(
                generated_ids,
                skip_special_tokens=True,
            )
            messages.append({"role": "assistant", "content": output_text})

            prompts.append(
                {
                    "prompt_id": prompt_id,
                    "dataset_name": args.dataset,
                    "dataset_config": dataset_config,
                    "dataset_split": dataset_split,
                    "dataset_index": dataset_index,
                    "turn_index": turn_index,
                    "prompt_text": user_content,
                    "rendered_prompt_sha256": prompt_hash,
                    "input_token_ids": input_ids[0]
                    .detach()
                    .cpu()
                    .to(torch.int32),
                    "output_token_ids": result.output_ids[0].to(torch.int32),
                    "rounds": result.trace_rounds,
                }
            )

    artifact = {
        "format": TRACE_FORMAT,
        "format_version": TRACE_FORMAT_VERSION,
        "metadata": metadata,
        "prompts": prompts,
    }

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, args.output_path)
    total_rounds = sum(len(prompt["rounds"]) for prompt in prompts)
    print(
        f"Saved {len(prompts)} prompts and {total_rounds} rounds "
        f"to {args.output_path}"
    )


if __name__ == "__main__":
    main()
