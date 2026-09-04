#!/usr/bin/env python3

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


BOOTSTRAP_SEED = 20260904
BUDGETS = (8, 16, 32, 64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the frozen Step-6.2 cross-domain analysis."
    )
    parser.add_argument("--gsm8k", type=Path, required=True)
    parser.add_argument("--cross-domain", type=Path, required=True)
    parser.add_argument("--mtbench-native", type=Path, required=True)
    parser.add_argument("--math-quality", type=Path, required=True)
    parser.add_argument("--humaneval-quality", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate_clusters(
    rows: list[dict[str, str]],
    turns_per_cluster: int,
) -> dict[str, list[dict[str, float]]]:
    grouped: dict[
        tuple[str, int],
        list[dict[str, str]],
    ] = defaultdict(list)
    for row in rows:
        method_key = row["method_key"]
        if not (
            method_key.startswith("dflash2_unary_k16_tb")
            or method_key.startswith("dflash2_pairwise_k16_tb")
        ):
            continue
        cluster_index = int(row["prompt_index"]) // turns_per_cluster
        grouped[(method_key, cluster_index)].append(row)

    values: dict[str, list[dict[str, float]]] = defaultdict(list)
    for (method_key, cluster_index), cluster_rows in sorted(grouped.items()):
        if len(cluster_rows) != turns_per_cluster:
            raise ValueError(
                f"{method_key} cluster {cluster_index} has "
                f"{len(cluster_rows)} turns, expected {turns_per_cluster}"
            )
        rounds = np.asarray(
            [float(row["decode_rounds"]) for row in cluster_rows]
        )
        output_tokens = np.asarray(
            [float(row["num_output_tokens"]) for row in cluster_rows]
        )
        milliseconds_per_token = np.asarray(
            [
                float(row["milliseconds_per_token"])
                for row in cluster_rows
            ]
        )
        decode_milliseconds = output_tokens * milliseconds_per_token
        values[method_key].append(
            {
                "cluster_index": float(cluster_index),
                "matched_draft_tokens": float(
                    np.average(
                        [
                            float(row["mean_matched_draft_tokens"])
                            for row in cluster_rows
                        ],
                        weights=rounds,
                    )
                ),
                "full_block_acceptance": float(
                    np.average(
                        [
                            float(row["full_block_acceptance"])
                            for row in cluster_rows
                        ],
                        weights=rounds,
                    )
                ),
                "tokens_per_second": float(
                    output_tokens.sum()
                    / (decode_milliseconds.sum() / 1000)
                ),
                "milliseconds_per_token": float(
                    decode_milliseconds.sum() / output_tokens.sum()
                ),
            }
        )
    return values


def analyze_dataset(
    dataset: str,
    analysis_dir: Path,
    cluster_count: int,
    turns_per_cluster: int,
    bootstrap_samples: int,
) -> list[dict[str, object]]:
    values = aggregate_clusters(
        read_csv(analysis_dir / "per_prompt_metrics.csv"),
        turns_per_cluster,
    )
    timing = {
        row["method_key"]: row
        for row in read_csv(analysis_dir / "timing_breakdown.csv")
    }
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    sampled = rng.integers(
        0,
        cluster_count,
        size=(bootstrap_samples, cluster_count),
    )
    rows = []
    for budget in BUDGETS:
        unary_key = f"dflash2_unary_k16_tb{budget}"
        pairwise_key = f"dflash2_pairwise_k16_tb{budget}"
        unary = values[unary_key]
        pairwise = values[pairwise_key]
        if len(unary) != cluster_count or len(pairwise) != cluster_count:
            raise ValueError(
                f"{dataset} B={budget}: expected {cluster_count} clusters"
            )
        metrics = {}
        differences = {}
        intervals = {}
        for metric in (
            "matched_draft_tokens",
            "full_block_acceptance",
            "tokens_per_second",
            "milliseconds_per_token",
        ):
            unary_values = np.asarray(
                [cluster[metric] for cluster in unary]
            )
            pairwise_values = np.asarray(
                [cluster[metric] for cluster in pairwise]
            )
            difference = pairwise_values - unary_values
            estimates = difference[sampled].mean(axis=1)
            metrics[metric] = (
                float(unary_values.mean()),
                float(pairwise_values.mean()),
            )
            differences[metric] = difference
            intervals[metric] = tuple(
                float(value)
                for value in np.quantile(estimates, [0.025, 0.975])
            )
        matched_difference = differences["matched_draft_tokens"]
        throughput_difference = differences["tokens_per_second"]
        unary_throughput, pairwise_throughput = metrics[
            "tokens_per_second"
        ]
        rows.append(
            {
                "dataset": dataset,
                "clusters": cluster_count,
                "turns": cluster_count * turns_per_cluster,
                "budget": budget,
                "unary_matched": metrics["matched_draft_tokens"][0],
                "pairwise_matched": metrics["matched_draft_tokens"][1],
                "matched_gain": float(matched_difference.mean()),
                "matched_ci_low": intervals["matched_draft_tokens"][0],
                "matched_ci_high": intervals["matched_draft_tokens"][1],
                "improve_clusters": int((matched_difference > 0).sum()),
                "tie_clusters": int((matched_difference == 0).sum()),
                "hurt_clusters": int((matched_difference < 0).sum()),
                "unary_full_block": metrics["full_block_acceptance"][0],
                "pairwise_full_block": metrics[
                    "full_block_acceptance"
                ][1],
                "unary_tokens_per_second": unary_throughput,
                "pairwise_tokens_per_second": pairwise_throughput,
                "tokens_per_second_gain": float(
                    throughput_difference.mean()
                ),
                "relative_throughput_gain_percent": float(
                    100 * throughput_difference.mean() / unary_throughput
                ),
                "throughput_ci_low": intervals["tokens_per_second"][0],
                "throughput_ci_high": intervals["tokens_per_second"][1],
                "unary_milliseconds_per_token": metrics[
                    "milliseconds_per_token"
                ][0],
                "pairwise_milliseconds_per_token": metrics[
                    "milliseconds_per_token"
                ][1],
                "milliseconds_per_token_difference": float(
                    differences["milliseconds_per_token"].mean()
                ),
                "pairwise_tree_build_overhead_ms_per_round": (
                    float(
                        timing[pairwise_key][
                            "tree_build_milliseconds_per_round"
                        ]
                    )
                    - float(
                        timing[unary_key][
                            "tree_build_milliseconds_per_round"
                        ]
                    )
                ),
            }
        )
    return rows


def quality_differences(
    math_path: Path,
    humaneval_path: Path,
) -> list[dict[str, object]]:
    rows = []
    for dataset, path, metric in (
        ("MATH500", math_path, "accuracy"),
        ("HumanEval", humaneval_path, "pass_at_1"),
    ):
        source = read_csv(path)
        target = next(row for row in source if row["method"] == "target-only")
        target_value = float(target[metric])
        for budget in BUDGETS:
            unary = next(
                row
                for row in source
                if row["method"] == "unary"
                and int(row["budget"]) == budget
            )
            pairwise = next(
                row
                for row in source
                if row["method"] == "pairwise"
                and int(row["budget"]) == budget
            )
            unary_value = float(unary[metric])
            pairwise_value = float(pairwise[metric])
            rows.append(
                {
                    "dataset": dataset,
                    "budget": budget,
                    "evaluated": int(pairwise["evaluated"]),
                    "metric": metric,
                    "target_only": target_value,
                    "unary": unary_value,
                    "pairwise": pairwise_value,
                    "pairwise_minus_unary": pairwise_value - unary_value,
                    "pairwise_minus_target": pairwise_value - target_value,
                }
            )
    return rows


def main() -> None:
    args = parse_args()
    datasets = (
        ("GSM8K", args.gsm8k, 128, 1),
        ("MATH500", args.cross_domain / "math500", 128, 1),
        ("HumanEval", args.cross_domain / "humaneval", 164, 1),
        (
            "MT-Bench controlled-history",
            args.cross_domain / "mt-bench",
            80,
            2,
        ),
        ("MT-Bench native-trajectory", args.mtbench_native, 80, 2),
    )
    metric_rows = [
        row
        for dataset, path, clusters, turns in datasets
        for row in analyze_dataset(
            dataset,
            path,
            clusters,
            turns,
            args.bootstrap_samples,
        )
    ]
    quality_rows = quality_differences(
        args.math_quality,
        args.humaneval_quality,
    )
    write_csv(args.output_dir / "cross_domain_metrics.csv", metric_rows)
    write_csv(args.output_dir / "task_quality.csv", quality_rows)


if __name__ == "__main__":
    main()
