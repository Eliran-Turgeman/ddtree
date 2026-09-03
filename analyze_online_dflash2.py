#!/usr/bin/env python3

import argparse
import csv
import re
from pathlib import Path

import numpy as np
import torch


BOOTSTRAP_SEED = 20260903
STAGE_NAMES = ("draft", "tree_build", "tree_compile", "verify", "commit")
TREE_METHOD_PATTERN = re.compile(
    r"^dflash2_(unary|pairwise)_k16_tb(\d+)$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze the controlled online DFlash2 tree benchmark."
    )
    parser.add_argument("run_path", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--offline-csv",
        type=Path,
        default=Path(
            "analysis/2026-09-03_dflash2_unary-vs-pairwise/"
            "mean_matched_depth.csv"
        ),
    )
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    return parser.parse_args()


def method_identity(method_key: str) -> tuple[str, int] | None:
    if method_key == "dflash2":
        return "DFlash2-GreedyPath", 7
    match = TREE_METHOD_PATTERN.match(method_key)
    if match is None:
        return None
    method = (
        "Unary-FullMass"
        if match.group(1) == "unary"
        else "Pairwise-MassPreserving"
    )
    return method, int(match.group(2))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def confidence_interval(values: np.ndarray) -> tuple[float, float]:
    low, high = np.quantile(values, [0.025, 0.975])
    return float(low), float(high)


def bootstrap_prompt_mean(
    values: np.ndarray,
    sampled_prompts: np.ndarray,
) -> np.ndarray:
    return values[sampled_prompts].mean(axis=1)


def mean_or_nan(values: list[float] | list[int] | list[bool]) -> float:
    return float(np.mean(values)) if values else float("nan")


def load_offline_means(path: Path) -> dict[tuple[str, int], float]:
    with path.open(newline="", encoding="utf-8") as file:
        return {
            (row["method"], int(row["budget"])): float(row["mean"])
            for row in csv.DictReader(file)
        }


def main() -> None:
    args = parse_args()
    run = torch.load(
        args.run_path,
        map_location="cpu",
        weights_only=False,
    )
    responses = run["responses"]
    if not responses:
        raise ValueError("benchmark artifact contains no responses")
    prompt_count = len(responses)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    sampled_prompts = rng.integers(
        0,
        prompt_count,
        size=(args.bootstrap_samples, prompt_count),
    )

    method_keys = [
        key
        for key in responses[0]
        if method_identity(key) is not None
    ]
    prompt_rows = []
    values_by_method: dict[str, dict[str, np.ndarray]] = {}
    for method_key in method_keys:
        identity = method_identity(method_key)
        if identity is None:
            continue
        method, budget = identity
        metric_lists = {
            "mean_matched_draft_tokens": [],
            "mean_committed_tokens": [],
            "full_block_acceptance": [],
            "tokens_per_second": [],
            "milliseconds_per_token": [],
        }
        for prompt_index, response in enumerate(responses):
            result = response[method_key]
            matched = list(result.matched_draft_tokens_per_round)
            committed = list(result.committed_tokens_per_round)
            full_block = [value == 7 for value in matched]
            milliseconds_per_token = (
                float(result.time_per_output_token) * 1000
            )
            row = {
                "prompt_index": prompt_index,
                "method_key": method_key,
                "method": method,
                "budget": budget,
                "decode_rounds": result.decode_rounds,
                "num_output_tokens": result.num_output_tokens,
                "mean_matched_draft_tokens": mean_or_nan(matched),
                "mean_committed_tokens": mean_or_nan(committed),
                "full_block_acceptance": mean_or_nan(full_block),
                "milliseconds_per_token": milliseconds_per_token,
                "tokens_per_second": 1000 / milliseconds_per_token,
                "matches_sequential_baseline": getattr(
                    result,
                    "matches_baseline",
                    "",
                ),
            }
            for stage in STAGE_NAMES:
                row[f"{stage}_milliseconds_per_round"] = (
                    float(result.stage_times.get(stage, 0.0))
                    * 1000
                    / max(result.decode_rounds, 1)
                )
            total_decode_seconds = (
                float(result.time_per_output_token)
                * result.num_output_tokens
            )
            measured_stage_seconds = sum(
                float(value) for value in result.stage_times.values()
            )
            row["other_milliseconds_per_round"] = (
                max(total_decode_seconds - measured_stage_seconds, 0.0)
                * 1000
                / max(result.decode_rounds, 1)
            )
            prompt_rows.append(row)
            for metric in metric_lists:
                metric_lists[metric].append(float(row[metric]))
        values_by_method[method_key] = {
            metric: np.asarray(values, dtype=np.float64)
            for metric, values in metric_lists.items()
        }

    method_rows = []
    for method_key in method_keys:
        method, budget = method_identity(method_key)
        prompt_values = values_by_method[method_key]
        matched_round_values = [
            value
            for response in responses
            for value in response[
                method_key
            ].matched_draft_tokens_per_round
        ]
        full_block_round_values = [
            value == 7 for value in matched_round_values
        ]
        row = {
            "method_key": method_key,
            "method": method,
            "budget": budget,
            "prompt_count": prompt_count,
            "decode_rounds": sum(
                response[method_key].decode_rounds
                for response in responses
            ),
            "equal_prompt_mean_matched_draft_tokens": float(
                prompt_values["mean_matched_draft_tokens"].mean()
            ),
            "round_weighted_mean_matched_draft_tokens": mean_or_nan(
                matched_round_values
            ),
            "equal_prompt_mean_committed_tokens": float(
                prompt_values["mean_committed_tokens"].mean()
            ),
            "round_weighted_full_block_acceptance": mean_or_nan(
                full_block_round_values
            ),
            "equal_prompt_mean_tokens_per_second": float(
                prompt_values["tokens_per_second"].mean()
            ),
            "equal_prompt_mean_milliseconds_per_token": float(
                prompt_values["milliseconds_per_token"].mean()
            ),
            "sequential_exact_matches": sum(
                bool(
                    getattr(
                        response[method_key],
                        "matches_baseline",
                        False,
                    )
                )
                for response in responses
            ),
        }
        estimates = bootstrap_prompt_mean(
            prompt_values["mean_matched_draft_tokens"],
            sampled_prompts,
        )
        row["matched_ci_low"], row["matched_ci_high"] = (
            confidence_interval(estimates)
        )
        method_rows.append(row)

    paired_rows = []
    budgets = sorted(
        {
            budget
            for key in method_keys
            for method, budget in [method_identity(key)]
            if method == "Unary-FullMass"
        }
    )
    for budget in budgets:
        unary_key = f"dflash2_unary_k16_tb{budget}"
        pairwise_key = f"dflash2_pairwise_k16_tb{budget}"
        if unary_key not in values_by_method:
            continue
        for metric in (
            "mean_matched_draft_tokens",
            "mean_committed_tokens",
            "full_block_acceptance",
            "tokens_per_second",
            "milliseconds_per_token",
        ):
            differences = (
                values_by_method[pairwise_key][metric]
                - values_by_method[unary_key][metric]
            )
            estimates = bootstrap_prompt_mean(
                differences,
                sampled_prompts,
            )
            ci_low, ci_high = confidence_interval(estimates)
            paired_rows.append(
                {
                    "budget": budget,
                    "metric": metric,
                    "mean_paired_prompt_difference": float(
                        differences.mean()
                    ),
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "bootstrap_probability_positive": float(
                        (estimates > 0).mean()
                    ),
                    "pairwise_improves_prompts": int(
                        (differences > 0).sum()
                    ),
                    "ties": int((differences == 0).sum()),
                    "unary_improves_prompts": int(
                        (differences < 0).sum()
                    ),
                }
            )

    timing_rows = []
    for method_key in method_keys:
        method, budget = method_identity(method_key)
        results = [response[method_key] for response in responses]
        total_rounds = sum(result.decode_rounds for result in results)
        total_decode_seconds = sum(
            float(result.time_per_output_token)
            * result.num_output_tokens
            for result in results
        )
        stage_seconds = {
            stage: sum(
                float(result.stage_times.get(stage, 0.0))
                for result in results
            )
            for stage in STAGE_NAMES
        }
        measured_seconds = sum(stage_seconds.values())
        timing_rows.append(
            {
                "method_key": method_key,
                "method": method,
                "budget": budget,
                "rounds": total_rounds,
                **{
                    f"{stage}_milliseconds_per_round": (
                        seconds * 1000 / max(total_rounds, 1)
                    )
                    for stage, seconds in sorted(stage_seconds.items())
                },
                "other_milliseconds_per_round": (
                    max(total_decode_seconds - measured_seconds, 0.0)
                    * 1000
                    / max(total_rounds, 1)
                ),
                "total_milliseconds_per_round": (
                    total_decode_seconds
                    * 1000
                    / max(total_rounds, 1)
                ),
            }
        )

    offline_means = load_offline_means(args.offline_csv)
    offline_online_rows = []
    for row in method_rows:
        key = (str(row["method"]), int(row["budget"]))
        if key not in offline_means:
            continue
        offline_mean = offline_means[key]
        online_mean = float(
            row["round_weighted_mean_matched_draft_tokens"]
        )
        offline_online_rows.append(
            {
                "method": row["method"],
                "budget": row["budget"],
                "offline_frozen_trace_round_weighted_mean": offline_mean,
                "online_observed_round_weighted_mean": online_mean,
                "online_minus_offline": online_mean - offline_mean,
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "per_prompt_metrics.csv", prompt_rows)
    write_csv(args.output_dir / "method_summary.csv", method_rows)
    write_csv(
        args.output_dir / "paired_prompt_comparisons.csv",
        paired_rows,
    )
    write_csv(args.output_dir / "timing_breakdown.csv", timing_rows)
    write_csv(
        args.output_dir / "offline_online_comparison.csv",
        offline_online_rows,
    )

    matched_paired_rows = {
        int(row["budget"]): row
        for row in paired_rows
        if row["metric"] == "mean_matched_draft_tokens"
    }
    moderate_gains = [
        float(matched_paired_rows[budget][
            "mean_paired_prompt_difference"
        ])
        for budget in (16, 32, 64)
        if budget in matched_paired_rows
    ]
    if moderate_gains and min(moderate_gains) >= 0.3:
        recommendation = "STRONG GO"
    elif moderate_gains and np.mean(moderate_gains) > 0:
        recommendation = "MODERATE GO"
    else:
        recommendation = "NO-GO / investigate discrepancy"

    with (args.output_dir / "summary.md").open(
        "w",
        encoding="utf-8",
    ) as file:
        file.write("# Step 5 online DFlash2 tree evaluation\n\n")
        file.write(
            f"- Source commit: `{run.get('repository', {}).get('commit')}`\n"
        )
        file.write(
            f"- GPU: {run.get('runtime', {}).get('gpu')}\n"
        )
        file.write(
            f"- PyTorch/CUDA/Transformers: "
            f"{run.get('runtime', {}).get('pytorch')} / "
            f"{run.get('runtime', {}).get('cuda')} / "
            f"{run.get('runtime', {}).get('transformers')}\n"
        )
        file.write(f"- Prompts: {prompt_count}\n\n")
        file.write("## Online acceptance\n\n")
        file.write(
            "| Method | B | Matched draft tokens | 95% CI | "
            "Full block | Tokens/s | Exact sequential |\n"
        )
        file.write(
            "| :----- | -: | -------------------: | -----: | "
            "---------: | -------: | ---------------: |\n"
        )
        for row in method_rows:
            file.write(
                f"| {row['method']} | {row['budget']} | "
                f"{row['equal_prompt_mean_matched_draft_tokens']:.4f} | "
                f"[{row['matched_ci_low']:.4f}, "
                f"{row['matched_ci_high']:.4f}] | "
                f"{row['round_weighted_full_block_acceptance']:.2%} | "
                f"{row['equal_prompt_mean_tokens_per_second']:.2f} | "
                f"{row['sequential_exact_matches']}/{prompt_count} |\n"
            )
        file.write("\n## Pairwise minus Unary\n\n")
        file.write(
            "| B | Matched-token gain | 95% CI | "
            "Prompts improve/tie/hurt |\n"
        )
        file.write("| -: | -----------------: | -----: | :----------------------- |\n")
        for budget, row in matched_paired_rows.items():
            file.write(
                f"| {budget} | "
                f"{row['mean_paired_prompt_difference']:+.4f} | "
                f"[{row['ci_low']:+.4f}, {row['ci_high']:+.4f}] | "
                f"{row['pairwise_improves_prompts']}/"
                f"{row['ties']}/"
                f"{row['unary_improves_prompts']} |\n"
            )
        file.write("\n## Interpretation\n\n")
        file.write(
            "The online methods encounter different state trajectories. "
            "Paired statistics therefore compare prompt-level averages, "
            "not decoding round N across methods. Offline values are "
            "off-policy frozen-state predictions and are included only to "
            "check whether the relative ordering survives.\n\n"
        )
        file.write(f"**Recommendation: {recommendation}.**\n")

    print(f"Wrote Step-5 analysis to {args.output_dir}")


if __name__ == "__main__":
    main()
