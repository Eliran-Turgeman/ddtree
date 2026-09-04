#!/usr/bin/env python3

import argparse
import csv
import hashlib
import importlib.metadata
import json
import re
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoTokenizer


TREE_METHOD_PATTERN = re.compile(
    r"^dflash2_(unary|pairwise)_k16_tb(\d+)$"
)
DATASET_REVISIONS = {
    "math500": "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be",
    "humaneval": "7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544",
}
MATH_VERIFY_VERSION = "0.9.0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export and score frozen Step-6.2 task outputs."
    )
    parser.add_argument("run_path", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--task",
        required=True,
        choices=("math500", "humaneval"),
    )
    return parser.parse_args()


def selected_dataset(task: str, max_samples: int | None):
    if task == "math500":
        dataset = load_dataset(
            "HuggingFaceH4/MATH-500",
            split="test",
            revision=DATASET_REVISIONS[task],
        )
    else:
        dataset = load_dataset(
            "openai/openai_humaneval",
            split="test",
            revision=DATASET_REVISIONS[task],
        )
    if max_samples is not None and len(dataset) > max_samples:
        dataset = dataset.shuffle(seed=0).select(range(max_samples))
    return dataset


def method_identity(method_key: str) -> tuple[str, int | None] | None:
    if method_key == "baseline":
        return "target-only", None
    match = TREE_METHOD_PATTERN.match(method_key)
    if match is None:
        return None
    return match.group(1), int(match.group(2))


def output_text(result: object, tokenizer: object) -> str:
    generated_ids = result.output_ids[
        0,
        result.num_input_tokens :,
    ]
    return tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    )


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def dataset_order_hash(task: str, dataset: object) -> str:
    field = "unique_id" if task == "math500" else "task_id"
    values = "\n".join(str(value) for value in dataset[field])
    return hashlib.sha256(values.encode("utf-8")).hexdigest()


def benchmark_user_content(task: str, sample: dict[str, object]) -> str:
    if task == "math500":
        return (
            f"{sample['problem']}\nPlease reason step by step, and put "
            "your final answer within \\boxed{}."
        )
    return (
        "Write a solution to the following problem and make sure that it "
        f"passes the tests:\n```python\n{sample['prompt']}\n```"
    )


def validate_saved_prompt_hashes(
    run: dict[str, object],
    task: str,
    dataset: object,
    tokenizer: object,
) -> None:
    for sample_index, response in enumerate(run["responses"]):
        input_text = tokenizer.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": benchmark_user_content(
                        task,
                        dataset[sample_index],
                    ),
                }
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        prompt_hash = hashlib.sha256(
            input_text.encode("utf-8")
        ).hexdigest()
        expected_prompt_id = (
            f"{task}:selected:{sample_index}:turn-0:"
            f"{prompt_hash[:12]}"
        )
        for method_key, result in response.items():
            if not method_key.startswith("dflash2"):
                continue
            prompt_ids = {
                metric["prompt_id"]
                for metric in result.round_metrics
                if "prompt_id" in metric
            }
            if prompt_ids != {expected_prompt_id}:
                raise ValueError(
                    f"{task} sample {sample_index} method {method_key} "
                    f"has prompt IDs {sorted(prompt_ids)}, expected "
                    f"{expected_prompt_id}"
                )


def write_metadata(
    path: Path,
    run: dict[str, object],
    task: str,
    dataset: object,
) -> None:
    metadata = {
        "task": task,
        "dataset_revision": DATASET_REVISIONS[task],
        "dataset_order_sha256": dataset_order_hash(task, dataset),
        "target_model": run["args"]["model_name_or_path"],
        "target_revision": run["target_revision"],
        "tokenizer_revision": run["target_revision"],
        "saved_prompt_hashes_validated": True,
        "math_verify_version": (
            MATH_VERIFY_VERSION if task == "math500" else None
        ),
    }
    path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def export_math500(
    run: dict[str, object],
    dataset: object,
    tokenizer: object,
    output_dir: Path,
) -> None:
    installed_version = importlib.metadata.version("math-verify")
    if installed_version != MATH_VERIFY_VERSION:
        raise RuntimeError(
            f"math-verify {MATH_VERIFY_VERSION} is required, "
            f"found {installed_version}"
        )
    from math_verify.metric import math_metric
    from math_verify.parser import (
        ExprExtractionConfig,
        LatexExtractionConfig,
    )

    verify = math_metric(
        gold_extraction_target=(LatexExtractionConfig(),),
        pred_extraction_target=(
            ExprExtractionConfig(),
            LatexExtractionConfig(),
        ),
        aggregation_function=max,
        precision=6,
    )
    detail_rows = []
    summary_rows = []
    method_keys = [
        key
        for key in run["responses"][0]
        if method_identity(key) is not None
    ]
    for method_key in method_keys:
        method, budget = method_identity(method_key)
        correct = 0
        for sample_index, response in enumerate(run["responses"]):
            prediction = output_text(response[method_key], tokenizer)
            gold = dataset[sample_index]["answer"]
            gold_in_latex_environment = f"${gold}$"
            error = ""
            extracted = None
            try:
                grade, extracted = verify(
                    [gold_in_latex_environment],
                    [prediction],
                )
                is_correct = grade == 1
            except Exception as exc:
                is_correct = False
                error = f"{type(exc).__name__}: {exc}"
            correct += int(is_correct)
            detail_rows.append(
                {
                    "sample_index": sample_index,
                    "method_key": method_key,
                    "method": method,
                    "budget": "" if budget is None else budget,
                    "gold": gold,
                    "prediction": prediction,
                    "extracted": repr(extracted),
                    "is_correct": is_correct,
                    "error": error,
                }
            )
        summary_rows.append(
            {
                "method_key": method_key,
                "method": method,
                "budget": "" if budget is None else budget,
                "evaluated": len(run["responses"]),
                "correct": correct,
                "accuracy": correct / len(run["responses"]),
            }
        )
    write_csv(output_dir / "math500_details.csv", detail_rows)
    write_csv(output_dir / "math500_summary.csv", summary_rows)


def export_humaneval(
    run: dict[str, object],
    dataset: object,
    tokenizer: object,
    output_dir: Path,
) -> None:
    manifest_rows = []
    method_keys = [
        key
        for key in run["responses"][0]
        if method_identity(key) is not None
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    for method_key in method_keys:
        method, budget = method_identity(method_key)
        path = output_dir / f"{method_key}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for sample_index, response in enumerate(run["responses"]):
                row = {
                    "task_id": dataset[sample_index]["task_id"],
                    "solution": output_text(
                        response[method_key],
                        tokenizer,
                    ),
                }
                handle.write(json.dumps(row) + "\n")
        manifest_rows.append(
            {
                "method_key": method_key,
                "method": method,
                "budget": "" if budget is None else budget,
                "samples": len(run["responses"]),
                "path": path.name,
            }
        )
    write_csv(output_dir / "humaneval_exports.csv", manifest_rows)


def main() -> None:
    args = parse_args()
    run = torch.load(
        args.run_path,
        map_location="cpu",
        weights_only=False,
    )
    if run["args"]["dataset"] != args.task:
        raise ValueError(
            f"run dataset is {run['args']['dataset']!r}, "
            f"not {args.task!r}"
        )
    dataset = selected_dataset(
        args.task,
        run["args"].get("max_samples"),
    )
    if len(dataset) != len(run["responses"]):
        raise ValueError(
            f"dataset has {len(dataset)} samples but run has "
            f"{len(run['responses'])} responses"
        )
    tokenizer = AutoTokenizer.from_pretrained(
        run["args"]["model_name_or_path"],
        revision=run["target_revision"],
    )
    validate_saved_prompt_hashes(
        run,
        args.task,
        dataset,
        tokenizer,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_metadata(
        args.output_dir / "evaluation_metadata.json",
        run,
        args.task,
        dataset,
    )
    if args.task == "math500":
        export_math500(run, dataset, tokenizer, args.output_dir)
    else:
        export_humaneval(run, dataset, tokenizer, args.output_dir)


if __name__ == "__main__":
    main()
