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


def load_and_validate(path: Path) -> dict:
    run = torch.load(path, map_location="cpu", weights_only=False)
    if len(run["responses"]) != 64:
        raise ValueError("Step-7 GSM8K artifact must contain 64 prompts")
    if run["target_revision"] != TARGET_REVISION:
        raise ValueError(f"unexpected target revision: {run['target_revision']}")
    if run["draft_revision"] != DRAFT_REVISION:
        raise ValueError(f"unexpected draft revision: {run['draft_revision']}")
    if run["args"]["dataset"] != "gsm8k":
        raise ValueError("Step-7 primary artifact must be GSM8K")
    if run["args"]["max_samples"] != 64:
        raise ValueError("Step-7 primary artifact must use 64 prompts")
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


def main() -> None:
    args = parse_args()
    run = load_and_validate(args.gsm8k)
    method_rows = []
    comparison_rows = []

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
                    "dataset": "GSM8K",
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
            args.bootstrap_samples,
            budget,
        )
        throughput_ci = bootstrap_interval(
            throughput_differences,
            args.bootstrap_samples,
            budget * 10,
        )
        comparison_rows.append(
            {
                "dataset": "GSM8K",
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

    output_dir = args.output_dir
    write_csv(output_dir / "method_metrics.csv", method_rows)
    write_csv(output_dir / "pairwise_comparisons.csv", comparison_rows)
    provenance = {
        "source_artifact": str(args.gsm8k),
        "target_revision": run["target_revision"],
        "draft_revision": run["draft_revision"],
        "repository": run["repository"],
        "runtime": run["runtime"],
        "exact_token_match_count": run["exact_token_match_count"],
        "exact_token_match_total": run["exact_token_match_total"],
        "tree_exact_token_matches": run["tree_exact_token_matches"],
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_samples": args.bootstrap_samples,
    }
    (output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps({"methods": method_rows, "comparisons": comparison_rows}, indent=2)
    )


if __name__ == "__main__":
    main()
