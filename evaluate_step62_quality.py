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


TREE_METHOD_PATTERN = re.compile(r"^dflash2_(unary|pairwise)_k16_tb(\d+)$")
STEP9_METHOD_PATTERNS = (
    (re.compile(r"^ddtree_tb(\d+)$"), "dflash-original-ddtree"),
    (
        re.compile(r"^dflash2_original_ddtree_tb(\d+)$"),
        "dflash2-original-ddtree",
    ),
    (
        re.compile(r"^dflash2_unary_k16_tb(\d+)$"),
        "dflash2-unary-k16",
    ),
    (
        re.compile(r"^dflash2_pairwise_k16_tb(\d+)$"),
        "dflash2-pairwise-k16",
    ),
)
DATASET_REVISIONS = {
    "gsm8k": "740312add88f781978c0658806c59bc2815b9866",
    "math500": "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be",
    "humaneval": "7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544",
    "mt-bench": "e3a795c5e9a82ee40611c416b8a7786c73198991",
}
MATH_VERIFY_VERSION = "0.9.0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export and score frozen Step-6.2/Step-9 task outputs."
    )
    parser.add_argument("run_path", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--task",
        required=True,
        choices=("gsm8k", "math500", "humaneval", "mt-bench"),
    )
    return parser.parse_args()


def selected_dataset(task: str, max_samples: int | None):
    if task == "gsm8k":
        dataset = load_dataset(
            "openai/gsm8k",
            "main",
            split="test",
            revision=DATASET_REVISIONS[task],
        )
    elif task == "math500":
        dataset = load_dataset(
            "HuggingFaceH4/MATH-500",
            split="test",
            revision=DATASET_REVISIONS[task],
        )
    elif task == "humaneval":
        dataset = load_dataset(
            "openai/openai_humaneval",
            split="test",
            revision=DATASET_REVISIONS[task],
        )
    else:
        dataset = load_dataset(
            "HuggingFaceH4/mt_bench_prompts",
            split="train",
            revision=DATASET_REVISIONS[task],
        )
    if max_samples is not None and len(dataset) > max_samples:
        dataset = dataset.shuffle(seed=0).select(range(max_samples))
    return dataset


def method_identity(method_key: str) -> tuple[str, int | None] | None:
    if method_key == "baseline":
        return "target-only", None
    if method_key == "dflash":
        return "dflash", None
    if method_key == "dflash2":
        return "dflash2", None
    match = TREE_METHOD_PATTERN.match(method_key)
    if match is not None:
        return match.group(1), int(match.group(2))
    for pattern, label in STEP9_METHOD_PATTERNS:
        match = pattern.match(method_key)
        if match is not None:
            return label, int(match.group(1))
    return None


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
    field = {
        "gsm8k": "question",
        "math500": "unique_id",
        "humaneval": "task_id",
        "mt-bench": "prompt",
    }[task]
    values = "\n".join(str(value) for value in dataset[field])
    return hashlib.sha256(values.encode("utf-8")).hexdigest()


def benchmark_user_content(task: str, sample: dict[str, object]) -> str:
    if task == "gsm8k":
        return (
            f"{sample['question']}\nPlease reason step by step, and put "
            "your final answer within \\boxed{}."
        )
    if task == "math500":
        return (
            f"{sample['problem']}\nPlease reason step by step, and put "
            "your final answer within \\boxed{}."
        )
    if task == "humaneval":
        return (
            "Write a solution to the following problem and make sure that it "
            f"passes the tests:\n```python\n{sample['prompt']}\n```"
        )
    raise ValueError("MT-Bench prompt validation is handled per turn")


def math_gold_answer(task: str, sample: dict[str, object]) -> str:
    answer = str(sample["answer"])
    if task == "gsm8k":
        if "####" not in answer:
            raise ValueError("GSM8K gold answer is missing the '####' delimiter")
        return answer.rsplit("####", maxsplit=1)[1].strip().replace(",", "")
    if task == "math500":
        return answer
    raise ValueError(f"unsupported math task: {task}")


def validate_saved_prompt_hashes(
    run: dict[str, object],
    task: str,
    dataset: object,
    tokenizer: object,
) -> int:
    response_index = 0
    validated_pairs = 0
    for sample_index, sample in enumerate(dataset):
        turns = (
            sample["prompt"]
            if task == "mt-bench"
            else [benchmark_user_content(task, sample)]
        )
        messages = []
        for turn_index, user_content in enumerate(turns):
            response = run["responses"][response_index]
            response_index += 1
            messages.append({"role": "user", "content": user_content})
            input_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            prompt_hash = hashlib.sha256(input_text.encode("utf-8")).hexdigest()
            expected_input_ids = tokenizer.encode(
                input_text,
                return_tensors="pt",
            )[0].cpu()
            expected_prefix = f"{task}:selected:{sample_index}:turn-{turn_index}:"
            for method_key, result in response.items():
                round_metrics = getattr(result, "round_metrics", None)
                if not round_metrics:
                    actual_input_ids = result.output_ids[
                        0, : result.num_input_tokens
                    ].cpu()
                    if not torch.equal(expected_input_ids, actual_input_ids):
                        raise ValueError(
                            f"{task} sample {sample_index} method {method_key} "
                            "saved input tokens do not match the pinned dataset prompt"
                        )
                    validated_pairs += 1
                    continue
                prompt_ids = {
                    metric["prompt_id"]
                    for metric in round_metrics
                    if "prompt_id" in metric
                }
                if not prompt_ids:
                    continue
                if len(prompt_ids) != 1:
                    raise ValueError(
                        f"{task} sample {sample_index} method {method_key} "
                        f"has multiple prompt IDs: {sorted(prompt_ids)}"
                    )
                prompt_id = next(iter(prompt_ids))
                if not (
                    prompt_id.startswith(expected_prefix)
                    and prompt_hash[:12] in prompt_id
                ):
                    raise ValueError(
                        f"{task} sample {sample_index} method {method_key} "
                        f"has unexpected prompt ID {prompt_id!r}"
                    )
                validated_pairs += 1
            if task == "mt-bench":
                messages.append(
                    {
                        "role": "assistant",
                        "content": output_text(
                            response["baseline"],
                            tokenizer,
                        ),
                    }
                )
    if response_index != len(run["responses"]):
        raise ValueError(
            f"validated {response_index} responses, "
            f"artifact contains {len(run['responses'])}"
        )
    return validated_pairs


def write_metadata(
    path: Path,
    run: dict[str, object],
    task: str,
    dataset: object,
    validated_prompt_method_pairs: int,
) -> None:
    metadata = {
        "task": task,
        "dataset_revision": DATASET_REVISIONS[task],
        "dataset_order_sha256": dataset_order_hash(task, dataset),
        "target_model": run["args"]["model_name_or_path"],
        "target_revision": run["target_revision"],
        "tokenizer_revision": run["target_revision"],
        "saved_prompt_hashes_validated": validated_prompt_method_pairs > 0,
        "validated_prompt_method_pairs": validated_prompt_method_pairs,
        "math_verify_version": (
            MATH_VERIFY_VERSION if task in ("gsm8k", "math500") else None
        ),
    }
    path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def export_math(
    run: dict[str, object],
    task: str,
    dataset: object,
    tokenizer: object,
    output_dir: Path,
) -> None:
    installed_version = importlib.metadata.version("math-verify")
    if installed_version != MATH_VERIFY_VERSION:
        raise RuntimeError(
            f"math-verify {MATH_VERIFY_VERSION} is required, found {installed_version}"
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
        key for key in run["responses"][0] if method_identity(key) is not None
    ]
    for method_key in method_keys:
        method, budget = method_identity(method_key)
        correct = 0
        for sample_index, response in enumerate(run["responses"]):
            prediction = output_text(response[method_key], tokenizer)
            gold = math_gold_answer(task, dataset[sample_index])
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
    write_csv(output_dir / f"{task}_details.csv", detail_rows)
    write_csv(output_dir / f"{task}_summary.csv", summary_rows)


def export_humaneval(
    run: dict[str, object],
    dataset: object,
    tokenizer: object,
    output_dir: Path,
) -> None:
    manifest_rows = []
    method_keys = [
        key for key in run["responses"][0] if method_identity(key) is not None
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


def export_mtbench(
    run: dict[str, object],
    dataset: object,
    tokenizer: object,
    output_dir: Path,
) -> None:
    rows = []
    method_keys = [
        key for key in run["responses"][0] if method_identity(key) is not None
    ]
    response_index = 0
    for conversation_index, sample in enumerate(dataset):
        for turn_index, prompt in enumerate(sample["prompt"]):
            response = run["responses"][response_index]
            response_index += 1
            for method_key in method_keys:
                method, budget = method_identity(method_key)
                rows.append(
                    {
                        "conversation_index": conversation_index,
                        "turn_index": turn_index,
                        "prompt": prompt,
                        "method_key": method_key,
                        "method": method,
                        "budget": "" if budget is None else budget,
                        "response": output_text(
                            response[method_key],
                            tokenizer,
                        ),
                    }
                )
    write_csv(output_dir / "mt_bench_outputs.csv", rows)


def main() -> None:
    args = parse_args()
    run = torch.load(
        args.run_path,
        map_location="cpu",
        weights_only=False,
    )
    if run["args"]["dataset"] != args.task:
        raise ValueError(
            f"run dataset is {run['args']['dataset']!r}, not {args.task!r}"
        )
    expected_revision = DATASET_REVISIONS[args.task]
    if run["args"].get("dataset_revision") != expected_revision:
        raise ValueError(
            f"run dataset revision is {run['args'].get('dataset_revision')!r}, "
            f"expected {expected_revision!r}"
        )
    dataset = selected_dataset(
        args.task,
        run["args"].get("max_samples"),
    )
    expected_responses = (
        sum(len(sample["prompt"]) for sample in dataset)
        if args.task == "mt-bench"
        else len(dataset)
    )
    if expected_responses != len(run["responses"]):
        raise ValueError(
            f"dataset requires {expected_responses} responses but run has "
            f"{len(run['responses'])} responses"
        )
    tokenizer = AutoTokenizer.from_pretrained(
        run["args"]["model_name_or_path"],
        revision=run["target_revision"],
    )
    validated_prompt_method_pairs = 0
    if args.task != "mt-bench" or run["trajectory_mode"] == "controlled_shared":
        validated_prompt_method_pairs = validate_saved_prompt_hashes(
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
        validated_prompt_method_pairs,
    )
    if args.task in ("gsm8k", "math500"):
        export_math(
            run,
            args.task,
            dataset,
            tokenizer,
            args.output_dir,
        )
    elif args.task == "humaneval":
        export_humaneval(run, dataset, tokenizer, args.output_dir)
    else:
        export_mtbench(run, dataset, tokenizer, args.output_dir)


if __name__ == "__main__":
    main()
