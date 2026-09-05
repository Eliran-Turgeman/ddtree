#!/usr/bin/env python3

import argparse
import csv
import re
from collections import Counter
from pathlib import Path

import numpy as np
import torch


BOOTSTRAP_SEED = 20260905
BUDGETS = (8, 16, 32, 64)
METHODS = (
    "dflash2_unary_k16",
    "dflash2_unary_k32",
    "dflash2_unary_k64",
    "dflash2_pairwise_k16",
)
METHOD_LABELS = {
    "dflash2_unary_k16": "Unary-K16",
    "dflash2_unary_k32": "Unary-K32",
    "dflash2_unary_k64": "Unary-K64",
    "dflash2_pairwise_k16": "Pairwise-K16",
}
EXPECTED_COMMIT = "88774714e931f57d8cc974160352e2b8a587051b"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze the frozen Step-6.3 wider-unary runs."
    )
    parser.add_argument("--gsm8k", type=Path, required=True)
    parser.add_argument("--math500", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    return parser.parse_args()


def method_key(method: str, budget: int) -> str:
    return f"{method}_tb{budget}"


def candidate_count(method: str) -> int:
    match = re.search(r"_k(\d+)$", method)
    if match is None:
        raise ValueError(f"method has no candidate count: {method}")
    return int(match.group(1))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_run(path: Path, dataset: str) -> dict:
    run = torch.load(path, map_location="cpu", weights_only=False)
    if run["repository"] != {"commit": EXPECTED_COMMIT, "dirty": False}:
        raise ValueError(
            f"{path} does not record the frozen clean commit: "
            f"{run['repository']}"
        )
    if run["args"]["dataset"] != dataset:
        raise ValueError(
            f"{path} records dataset {run['args']['dataset']}, "
            f"expected {dataset}"
        )
    if len(run["responses"]) != 128:
        raise ValueError(
            f"{path} has {len(run['responses'])} prompts, expected 128"
        )
    expected_keys = {
        method_key(method, budget)
        for method in METHODS
        for budget in BUDGETS
    }
    for prompt_index, response in enumerate(run["responses"]):
        missing = expected_keys - response.keys()
        if missing:
            raise ValueError(
                f"{path} prompt {prompt_index} is missing {sorted(missing)}"
            )
    return run


def prompt_metrics(result: object) -> dict[str, float]:
    rounds = result.round_metrics
    if not rounds:
        raise ValueError("tree method has no decoding rounds")
    matched = np.asarray(
        [float(row["matched_draft_tokens"]) for row in rounds]
    )
    milliseconds_per_token = float(result.time_per_output_token) * 1000
    return {
        "matched": float(matched.mean()),
        "full_block": float((matched == 7).mean()),
        "tokens_per_second": 1000 / milliseconds_per_token,
        "milliseconds_per_token": milliseconds_per_token,
    }


def bootstrap_interval(
    differences: np.ndarray,
    bootstrap_samples: int,
    seed_offset: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(BOOTSTRAP_SEED + seed_offset)
    sampled = rng.integers(
        0,
        differences.shape[0],
        size=(bootstrap_samples, differences.shape[0]),
    )
    estimates = differences[sampled].mean(axis=1)
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high)


def collect_prompt_values(
    run: dict,
) -> dict[str, list[dict[str, float]]]:
    values = {
        method_key(method, budget): []
        for method in METHODS
        for budget in BUDGETS
    }
    for response in run["responses"]:
        for key in values:
            values[key].append(prompt_metrics(response[key]))
    return values


def analyze_method_metrics(
    dataset: str,
    values: dict[str, list[dict[str, float]]],
) -> list[dict[str, object]]:
    rows = []
    for budget in BUDGETS:
        for method in METHODS:
            prompts = values[method_key(method, budget)]
            rows.append(
                {
                    "dataset": dataset,
                    "budget": budget,
                    "method": METHOD_LABELS[method],
                    "candidate_count": candidate_count(method),
                    "prompts": len(prompts),
                    "mean_matched_draft_tokens": np.mean(
                        [row["matched"] for row in prompts]
                    ),
                    "full_block_acceptance": np.mean(
                        [row["full_block"] for row in prompts]
                    ),
                    "tokens_per_second": np.mean(
                        [row["tokens_per_second"] for row in prompts]
                    ),
                    "milliseconds_per_token": np.mean(
                        [
                            row["milliseconds_per_token"]
                            for row in prompts
                        ]
                    ),
                }
            )
    return rows


def analyze_pairwise_comparisons(
    dataset: str,
    values: dict[str, list[dict[str, float]]],
    bootstrap_samples: int,
) -> list[dict[str, object]]:
    rows = []
    pairwise_method = "dflash2_pairwise_k16"
    for budget in BUDGETS:
        pairwise = values[method_key(pairwise_method, budget)]
        for unary_k in (16, 32, 64):
            unary_method = f"dflash2_unary_k{unary_k}"
            unary = values[method_key(unary_method, budget)]
            matched_difference = np.asarray(
                [
                    pairwise_row["matched"] - unary_row["matched"]
                    for pairwise_row, unary_row in zip(
                        pairwise,
                        unary,
                        strict=True,
                    )
                ]
            )
            throughput_difference = np.asarray(
                [
                    pairwise_row["tokens_per_second"]
                    - unary_row["tokens_per_second"]
                    for pairwise_row, unary_row in zip(
                        pairwise,
                        unary,
                        strict=True,
                    )
                ]
            )
            matched_ci = bootstrap_interval(
                matched_difference,
                bootstrap_samples,
                budget * 100 + unary_k,
            )
            throughput_ci = bootstrap_interval(
                throughput_difference,
                bootstrap_samples,
                budget * 1000 + unary_k,
            )
            unary_throughput = np.mean(
                [row["tokens_per_second"] for row in unary]
            )
            rows.append(
                {
                    "dataset": dataset,
                    "budget": budget,
                    "unary_k": unary_k,
                    "pairwise_matched": np.mean(
                        [row["matched"] for row in pairwise]
                    ),
                    "unary_matched": np.mean(
                        [row["matched"] for row in unary]
                    ),
                    "matched_gain": matched_difference.mean(),
                    "matched_ci_low": matched_ci[0],
                    "matched_ci_high": matched_ci[1],
                    "improve_prompts": int(
                        (matched_difference > 0).sum()
                    ),
                    "tie_prompts": int(
                        (matched_difference == 0).sum()
                    ),
                    "hurt_prompts": int(
                        (matched_difference < 0).sum()
                    ),
                    "pairwise_tokens_per_second": np.mean(
                        [row["tokens_per_second"] for row in pairwise]
                    ),
                    "unary_tokens_per_second": unary_throughput,
                    "throughput_gain": throughput_difference.mean(),
                    "relative_throughput_gain_percent": (
                        100
                        * throughput_difference.mean()
                        / unary_throughput
                    ),
                    "throughput_ci_low": throughput_ci[0],
                    "throughput_ci_high": throughput_ci[1],
                }
            )
    return rows


def analyze_unary_marginals(
    dataset: str,
    values: dict[str, list[dict[str, float]]],
    bootstrap_samples: int,
) -> list[dict[str, object]]:
    rows = []
    for budget in BUDGETS:
        for lower_k, upper_k in ((16, 32), (32, 64)):
            lower = values[
                method_key(f"dflash2_unary_k{lower_k}", budget)
            ]
            upper = values[
                method_key(f"dflash2_unary_k{upper_k}", budget)
            ]
            difference = np.asarray(
                [
                    upper_row["matched"] - lower_row["matched"]
                    for upper_row, lower_row in zip(
                        upper,
                        lower,
                        strict=True,
                    )
                ]
            )
            interval = bootstrap_interval(
                difference,
                bootstrap_samples,
                1_000_000 + budget * 100 + lower_k,
            )
            rows.append(
                {
                    "dataset": dataset,
                    "budget": budget,
                    "comparison": f"U{upper_k}-U{lower_k}",
                    "matched_gain": difference.mean(),
                    "matched_ci_low": interval[0],
                    "matched_ci_high": interval[1],
                    "improve_prompts": int((difference > 0).sum()),
                    "tie_prompts": int((difference == 0).sum()),
                    "hurt_prompts": int((difference < 0).sum()),
                }
            )
    return rows


def analyze_candidate_coverage(
    dataset: str,
    run: dict,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    actual_rows = []
    common_rows = []
    for budget in BUDGETS:
        for method in METHODS:
            support_k = candidate_count(method)
            rounds = [
                metric
                for response in run["responses"]
                for metric in response[
                    method_key(method, budget)
                ].round_metrics
            ]
            for depth in range(1, 8):
                observed = [
                    metric
                    for metric in rounds
                    if int(metric["target_available_depth"]) >= depth
                ]
                included = [
                    metric[f"target_rank_depth_{depth}"] is not None
                    for metric in observed
                ]
                prefixes = [
                    bool(
                        metric[
                            "target_prefix_representable_depth_"
                            f"{depth}"
                        ]
                    )
                    for metric in observed
                ]
                actual_rows.append(
                    {
                        "dataset": dataset,
                        "budget": budget,
                        "trajectory_method": METHOD_LABELS[method],
                        "support_k": support_k,
                        "depth": depth,
                        "observed_rounds": len(observed),
                        "target_token_inclusion": np.mean(included),
                        "prefix_representability": np.mean(prefixes),
                    }
                )

        unary64_rounds = [
            metric
            for response in run["responses"]
            for metric in response[
                method_key("dflash2_unary_k64", budget)
            ].round_metrics
        ]
        for support_k in (16, 32, 64):
            for depth in range(1, 8):
                observed = [
                    metric
                    for metric in unary64_rounds
                    if int(metric["target_available_depth"]) >= depth
                ]
                ranks_by_round = [
                    [
                        metric[f"target_rank_depth_{index}"]
                        for index in range(1, depth + 1)
                    ]
                    for metric in observed
                ]
                included = [
                    ranks[-1] is not None
                    and int(ranks[-1]) <= support_k
                    for ranks in ranks_by_round
                ]
                prefixes = [
                    all(
                        rank is not None and int(rank) <= support_k
                        for rank in ranks
                    )
                    for ranks in ranks_by_round
                ]
                common_rows.append(
                    {
                        "dataset": dataset,
                        "budget": budget,
                        "trajectory_method": "Unary-K64",
                        "support_k": support_k,
                        "depth": depth,
                        "observed_rounds": len(observed),
                        "target_token_inclusion": np.mean(included),
                        "prefix_representability": np.mean(prefixes),
                    }
                )
    return actual_rows, common_rows


def analyze_failures_and_systems(
    dataset: str,
    run: dict,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    failure_rows = []
    system_rows = []
    for budget in BUDGETS:
        for method in METHODS:
            key = method_key(method, budget)
            results = [response[key] for response in run["responses"]]
            rounds = [
                metric
                for result in results
                for metric in result.round_metrics
            ]
            counts = Counter(metric["failure_type"] for metric in rounds)
            classified_total = len(rounds) - counts["censored"]
            failure_rows.append(
                {
                    "dataset": dataset,
                    "budget": budget,
                    "method": METHOD_LABELS[method],
                    "candidate_count": candidate_count(method),
                    "rounds": len(rounds),
                    "classified_rounds": classified_total,
                    "candidate_failure_count": counts[
                        "candidate_failure"
                    ],
                    "candidate_failure_rate": (
                        counts["candidate_failure"] / classified_total
                    ),
                    "ranking_budget_failure_count": counts[
                        "ranking_budget_failure"
                    ],
                    "ranking_budget_failure_rate": (
                        counts["ranking_budget_failure"]
                        / classified_total
                    ),
                    "covered_count": counts["covered"],
                    "covered_rate": counts["covered"] / classified_total,
                    "censored_count": counts["censored"],
                }
            )
            system_rows.append(
                {
                    "dataset": dataset,
                    "budget": budget,
                    "method": METHOD_LABELS[method],
                    "candidate_count": candidate_count(method),
                    "rounds": len(rounds),
                    "candidate_select_ms_per_round": np.mean(
                        [
                            float(
                                metric[
                                    "candidate_select_latency_ms"
                                ]
                            )
                            for metric in rounds
                        ]
                    ),
                    "tree_build_ms_per_round": np.mean(
                        [
                            float(metric["tree_build_latency_ms"])
                            for metric in rounds
                        ]
                    ),
                    "verify_ms_per_round": np.mean(
                        [
                            float(metric["target_verify_latency_ms"])
                            for metric in rounds
                        ]
                    ),
                    "tokens_per_second": np.mean(
                        [
                            1 / float(result.time_per_output_token)
                            for result in results
                        ]
                    ),
                    "milliseconds_per_token": np.mean(
                        [
                            1000 * float(result.time_per_output_token)
                            for result in results
                        ]
                    ),
                }
            )
    return failure_rows, system_rows


def analyze_dataset(
    dataset: str,
    run: dict,
    bootstrap_samples: int,
) -> dict[str, list[dict[str, object]]]:
    values = collect_prompt_values(run)
    actual_coverage, common_coverage = analyze_candidate_coverage(
        dataset,
        run,
    )
    failures, systems = analyze_failures_and_systems(dataset, run)
    return {
        "method_metrics": analyze_method_metrics(dataset, values),
        "pairwise_comparisons": analyze_pairwise_comparisons(
            dataset,
            values,
            bootstrap_samples,
        ),
        "unary_width_marginals": analyze_unary_marginals(
            dataset,
            values,
            bootstrap_samples,
        ),
        "candidate_coverage": actual_coverage,
        "unary_k64_trajectory_coverage": common_coverage,
        "failure_decomposition": failures,
        "systems_cost": systems,
    }


def build_matched_table(
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    by_key = {
        (str(row["dataset"]), int(row["budget"]), str(row["method"])): row
        for row in rows
    }
    return [
        {
            "dataset": dataset,
            "budget": budget,
            "pairwise_k16": by_key[
                (dataset, budget, "Pairwise-K16")
            ]["mean_matched_draft_tokens"],
            "unary_k16": by_key[
                (dataset, budget, "Unary-K16")
            ]["mean_matched_draft_tokens"],
            "unary_k32": by_key[
                (dataset, budget, "Unary-K32")
            ]["mean_matched_draft_tokens"],
            "unary_k64": by_key[
                (dataset, budget, "Unary-K64")
            ]["mean_matched_draft_tokens"],
        }
        for dataset in ("GSM8K", "MATH500")
        for budget in BUDGETS
    ]


def build_pairwise_gain_table(
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    by_key = {
        (str(row["dataset"]), int(row["budget"]), int(row["unary_k"])): row
        for row in rows
    }
    output = []
    for dataset in ("GSM8K", "MATH500"):
        for budget in BUDGETS:
            row: dict[str, object] = {
                "dataset": dataset,
                "budget": budget,
            }
            for unary_k in (16, 32, 64):
                comparison = by_key[(dataset, budget, unary_k)]
                row[f"p16_minus_u{unary_k}"] = comparison["matched_gain"]
                row[f"p16_minus_u{unary_k}_ci_low"] = comparison[
                    "matched_ci_low"
                ]
                row[f"p16_minus_u{unary_k}_ci_high"] = comparison[
                    "matched_ci_high"
                ]
            output.append(row)
    return output


def build_coverage_summary(
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    output = []
    for dataset in ("GSM8K", "MATH500"):
        for support_k in (16, 32, 64):
            for depth in range(1, 8):
                selected = [
                    row
                    for row in rows
                    if (
                        row["dataset"] == dataset
                        and int(row["support_k"]) == support_k
                        and int(row["depth"]) == depth
                    )
                ]
                observed = sum(
                    int(row["observed_rounds"]) for row in selected
                )
                output.append(
                    {
                        "dataset": dataset,
                        "trajectory_method": "Unary-K64",
                        "support_k": support_k,
                        "depth": depth,
                        "observed_rounds": observed,
                        "target_token_inclusion": sum(
                            int(row["observed_rounds"])
                            * float(row["target_token_inclusion"])
                            for row in selected
                        )
                        / observed,
                        "prefix_representability": sum(
                            int(row["observed_rounds"])
                            * float(row["prefix_representability"])
                            for row in selected
                        )
                        / observed,
                    }
                )
    return output


def main() -> None:
    args = parse_args()
    tables: dict[str, list[dict[str, object]]] = {}
    for dataset, path in (
        ("GSM8K", args.gsm8k),
        ("MATH500", args.math500),
    ):
        run = load_run(path, dataset.lower())
        analyzed = analyze_dataset(
            dataset,
            run,
            args.bootstrap_samples,
        )
        for name, rows in analyzed.items():
            tables.setdefault(name, []).extend(rows)

    tables["matched_table"] = build_matched_table(
        tables["method_metrics"]
    )
    tables["pairwise_gain_table"] = build_pairwise_gain_table(
        tables["pairwise_comparisons"]
    )
    tables["candidate_coverage_summary"] = build_coverage_summary(
        tables["unary_k64_trajectory_coverage"]
    )
    for name, rows in tables.items():
        write_csv(args.output_dir / f"{name}.csv", rows)


if __name__ == "__main__":
    main()
