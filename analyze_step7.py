#!/usr/bin/env python3

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch


BOOTSTRAP_SEED = 20260905
BUDGETS = (16, 32, 64)
TARGET_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
DRAFT_REVISION = "dedf8df68adfb1afeaf7b7480c0a0243108177b4"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze the Step-7 official 27B validation."
    )
    parser.add_argument("--gsm8k", type=Path, required=True)
    parser.add_argument("--math500", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def prompt_metrics(result: object) -> dict[str, float]:
    rounds = result.round_metrics
    matched = np.asarray([float(row["matched_draft_tokens"]) for row in rounds])
    committed = np.asarray(
        [float(row["committed_tokens_this_round"]) for row in rounds]
    )
    return {
        "matched": float(matched.mean()),
        "full_block": float((matched == 7).mean()),
        "committed": float(committed.mean()),
        "tokens_per_second": 1 / float(result.time_per_output_token),
        "milliseconds_per_token": (float(result.time_per_output_token) * 1000),
    }


def bootstrap_interval(
    values: np.ndarray,
    samples: int,
    seed_offset: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(BOOTSTRAP_SEED + seed_offset)
    indices = rng.integers(
        0,
        len(values),
        size=(samples, len(values)),
    )
    estimates = values[indices].mean(axis=1)
    return tuple(float(value) for value in np.quantile(estimates, [0.025, 0.975]))


def load_and_validate(path: Path, dataset: str) -> dict:
    run = torch.load(path, map_location="cpu", weights_only=False)
    if len(run["responses"]) != 64:
        raise ValueError(f"Step-7 {dataset} artifact must contain 64 prompts")
    if run["target_revision"] != TARGET_REVISION:
        raise ValueError(f"unexpected target revision: {run['target_revision']}")
    if run["draft_revision"] != DRAFT_REVISION:
        raise ValueError(f"unexpected draft revision: {run['draft_revision']}")
    if run["args"]["dataset"] != dataset:
        raise ValueError(
            f"Step-7 artifact must be {dataset}, got {run['args']['dataset']}"
        )
    if run["args"]["max_samples"] != 64:
        raise ValueError("Step-7 artifact must use 64 prompts")
    expected = {
        f"dflash2_{method}_k16_tb{budget}"
        for method in ("unary", "pairwise")
        for budget in BUDGETS
    }
    for prompt_index, response in enumerate(run["responses"]):
        missing = expected - response.keys()
        if missing:
            raise ValueError(f"prompt {prompt_index} is missing {sorted(missing)}")
    return run


def analyze_run(
    dataset_name: str,
    run: dict,
    bootstrap_samples: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    method_rows = []
    comparison_rows = []
    dataset_offset = 0 if dataset_name == "GSM8K" else 1000
    greedy_prompts = [
        prompt_metrics(response["dflash2"]) for response in run["responses"]
    ]
    greedy_rounds = [
        row
        for response in run["responses"]
        for row in response["dflash2"].round_metrics
    ]
    method_rows.append(
        {
            "dataset": dataset_name,
            "prompts": len(greedy_prompts),
            "budget": 7,
            "method": "DFlash2-greedy",
            "mean_matched_draft_tokens": np.mean(
                [row["matched"] for row in greedy_prompts]
            ),
            "full_block_acceptance": np.mean(
                [row["full_block"] for row in greedy_prompts]
            ),
            "committed_tokens_per_round": np.mean(
                [row["committed"] for row in greedy_prompts]
            ),
            "tokens_per_second": np.mean(
                [row["tokens_per_second"] for row in greedy_prompts]
            ),
            "milliseconds_per_token": np.mean(
                [row["milliseconds_per_token"] for row in greedy_prompts]
            ),
            "tree_build_ms_per_round": 0.0,
            "verify_ms_per_round": np.mean(
                [row["target_verify_latency_ms"] for row in greedy_rounds]
            ),
        }
    )
    for budget in BUDGETS:
        methods = {}
        for method in ("unary", "pairwise"):
            key = f"dflash2_{method}_k16_tb{budget}"
            prompts = [prompt_metrics(response[key]) for response in run["responses"]]
            rounds = [
                row
                for response in run["responses"]
                for row in response[key].round_metrics
            ]
            methods[method] = prompts
            method_rows.append(
                {
                    "dataset": dataset_name,
                    "prompts": len(prompts),
                    "budget": budget,
                    "method": f"{method.title()}-K16",
                    "mean_matched_draft_tokens": np.mean(
                        [row["matched"] for row in prompts]
                    ),
                    "full_block_acceptance": np.mean(
                        [row["full_block"] for row in prompts]
                    ),
                    "committed_tokens_per_round": np.mean(
                        [row["committed"] for row in prompts]
                    ),
                    "tokens_per_second": np.mean(
                        [row["tokens_per_second"] for row in prompts]
                    ),
                    "milliseconds_per_token": np.mean(
                        [row["milliseconds_per_token"] for row in prompts]
                    ),
                    "tree_build_ms_per_round": np.mean(
                        [row["tree_build_latency_ms"] for row in rounds]
                    ),
                    "verify_ms_per_round": np.mean(
                        [row["target_verify_latency_ms"] for row in rounds]
                    ),
                }
            )

        unary = methods["unary"]
        pairwise = methods["pairwise"]
        matched_differences = np.asarray(
            [
                pairwise_row["matched"] - unary_row["matched"]
                for unary_row, pairwise_row in zip(
                    unary,
                    pairwise,
                    strict=True,
                )
            ]
        )
        throughput_differences = np.asarray(
            [
                pairwise_row["tokens_per_second"] - unary_row["tokens_per_second"]
                for unary_row, pairwise_row in zip(
                    unary,
                    pairwise,
                    strict=True,
                )
            ]
        )
        matched_ci = bootstrap_interval(
            matched_differences,
            bootstrap_samples,
            dataset_offset + budget,
        )
        throughput_ci = bootstrap_interval(
            throughput_differences,
            bootstrap_samples,
            dataset_offset + budget * 10,
        )
        comparison_rows.append(
            {
                "dataset": dataset_name,
                "prompts": len(unary),
                "budget": budget,
                "pairwise_minus_unary_matched": (matched_differences.mean()),
                "matched_ci_low": matched_ci[0],
                "matched_ci_high": matched_ci[1],
                "improve_prompts": int((matched_differences > 0).sum()),
                "tie_prompts": int((matched_differences == 0).sum()),
                "hurt_prompts": int((matched_differences < 0).sum()),
                "pairwise_minus_unary_tokens_per_second": (
                    throughput_differences.mean()
                ),
                "throughput_ci_low": throughput_ci[0],
                "throughput_ci_high": throughput_ci[1],
            }
        )
    return method_rows, comparison_rows


def run_provenance(run: dict) -> dict[str, object]:
    return {
        "target_revision": run["target_revision"],
        "draft_revision": run["draft_revision"],
        "repository": run["repository"],
        "runtime": run["runtime"],
        "exact_token_match_count": run["exact_token_match_count"],
        "exact_token_match_total": run["exact_token_match_total"],
        "tree_exact_token_matches": run["tree_exact_token_matches"],
    }


def main() -> None:
    args = parse_args()
    runs = [("GSM8K", load_and_validate(args.gsm8k, "gsm8k"))]
    if args.math500 is not None:
        runs.append(("MATH500", load_and_validate(args.math500, "math500")))

    method_rows = []
    comparison_rows = []
    for dataset_name, run in runs:
        dataset_methods, dataset_comparisons = analyze_run(
            dataset_name,
            run,
            args.bootstrap_samples,
        )
        method_rows.extend(dataset_methods)
        comparison_rows.extend(dataset_comparisons)

    output_dir = args.output_dir
    write_csv(output_dir / "method_metrics.csv", method_rows)
    write_csv(output_dir / "pairwise_comparisons.csv", comparison_rows)
    provenance = {
        "source_artifacts": {
            "GSM8K": args.gsm8k.name,
            "MATH500": (args.math500.name if args.math500 is not None else None),
        },
        "runs": {dataset_name: run_provenance(run) for dataset_name, run in runs},
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_samples": args.bootstrap_samples,
    }
    (output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {"methods": method_rows, "comparisons": comparison_rows},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
