#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path
from statistics import mean

import torch


BASE_COLUMNS = [
    "sample_index",
    "dataset",
    "model",
    "draft_model",
    "temperature",
    "block_size",
    "method",
    "tree_budget",
    "num_input_tokens",
    "num_output_tokens",
    "time_to_first_token_ms",
    "time_per_output_token_ms",
    "tokens_per_second",
    "total_decode_time_seconds",
    "decode_rounds",
    "mean_acceptance_length",
    "speedup_vs_baseline",
    "matches_baseline",
]


def method_sort_key(method: str) -> tuple[int, int]:
    if method == "baseline":
        return (0, 0)
    if method == "dflash":
        return (1, 0)
    if method.startswith("ddtree_tb"):
        return (2, int(method.removeprefix("ddtree_tb")))
    return (3, 0)


def tree_budget(method: str) -> int | str:
    if method.startswith("ddtree_tb"):
        return int(method.removeprefix("ddtree_tb"))
    return ""


def load_run(path: Path) -> dict:
    run = torch.load(path, map_location="cpu", weights_only=False)
    required_keys = {"responses", "block_size", "args"}
    missing_keys = required_keys - run.keys()
    if missing_keys:
        raise ValueError(f"{path} is missing required keys: {sorted(missing_keys)}")
    if not run["responses"]:
        raise ValueError(f"{path} contains no benchmark responses")
    return run


def build_rows(run: dict) -> tuple[list[dict[str, object]], list[str]]:
    args = run["args"]
    responses = run["responses"]
    stage_names = sorted(
        {
            stage
            for response in responses
            for result in response.values()
            for stage in result.stage_times
        }
    )
    rows = []

    for sample_index, response in enumerate(responses):
        baseline_time = response["baseline"].time_per_output_token
        for method in sorted(response, key=method_sort_key):
            result = response[method]
            time_per_token = float(result.time_per_output_token)
            acceptance_lengths = result.acceptance_lengths
            row = {
                "sample_index": sample_index,
                "dataset": args["dataset"],
                "model": args["model_name_or_path"],
                "draft_model": args["draft_name_or_path"],
                "temperature": args["temperature"],
                "block_size": run["block_size"],
                "method": method,
                "tree_budget": tree_budget(method),
                "num_input_tokens": result.num_input_tokens,
                "num_output_tokens": result.num_output_tokens,
                "time_to_first_token_ms": float(result.time_to_first_token) * 1000,
                "time_per_output_token_ms": time_per_token * 1000,
                "tokens_per_second": 1 / time_per_token if time_per_token > 0 else "",
                "total_decode_time_seconds": time_per_token * result.num_output_tokens,
                "decode_rounds": result.decode_rounds,
                "mean_acceptance_length": mean(acceptance_lengths) if acceptance_lengths else "",
                "speedup_vs_baseline": baseline_time / time_per_token if time_per_token > 0 else "",
                "matches_baseline": (
                    True
                    if method == "baseline"
                    else getattr(result, "matches_baseline", "")
                ),
            }
            for stage_name in stage_names:
                row[f"stage_{stage_name}_seconds"] = float(result.stage_times.get(stage_name, 0))
            rows.append(row)

    return rows, stage_names


def write_csv(path: Path, rows: list[dict[str, object]], stage_names: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = BASE_COLUMNS + [f"stage_{stage_name}_seconds" for stage_name in stage_names]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: list[dict[str, object]]) -> None:
    methods = sorted({str(row["method"]) for row in rows}, key=method_sort_key)
    baseline_rows = [row for row in rows if row["method"] == "baseline"]
    baseline_mean_ms = mean(float(row["time_per_output_token_ms"]) for row in baseline_rows)

    print()
    print(
        f"{'Method':<16} {'ms/token':>10} {'tokens/s':>10} "
        f"{'speedup':>10} {'tokens/round':>12}"
    )
    print("-" * 62)
    for method in methods:
        method_rows = [row for row in rows if row["method"] == method]
        mean_ms = mean(float(row["time_per_output_token_ms"]) for row in method_rows)
        acceptance_rows = [
            row
            for row in method_rows
            if row["mean_acceptance_length"] != ""
            and int(row["decode_rounds"]) > 0
        ]
        total_rounds = sum(int(row["decode_rounds"]) for row in acceptance_rows)
        acceptance = (
            sum(
                float(row["mean_acceptance_length"])
                * int(row["decode_rounds"])
                for row in acceptance_rows
            )
            / total_rounds
            if total_rounds
            else 0
        )
        print(
            f"{method:<16} "
            f"{mean_ms:>10.2f} "
            f"{1000 / mean_ms:>10.2f} "
            f"{baseline_mean_ms / mean_ms:>9.2f}x "
            f"{acceptance:>12.2f}"
        )

    print()
    print(
        "tokens/round is round-weighted and includes the verifier-carried token; "
        "subtract 1 for matched speculative tokens."
    )

    match_rows = [
        row
        for row in rows
        if row["method"] == "dflash2" and row["matches_baseline"] != ""
    ]
    if match_rows:
        matches = sum(row["matches_baseline"] is True for row in match_rows)
        print()
        print(f"DFlash2 exact token matches: {matches}/{len(match_rows)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize a DDTree .pt benchmark artifact and export sample-level CSV data."
    )
    parser.add_argument("run_path", type=Path)
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Output CSV path (default: next to the .pt file)",
    )
    args = parser.parse_args()

    csv_path = args.csv or args.run_path.with_suffix(".csv")
    run = load_run(args.run_path)
    rows, stage_names = build_rows(run)
    write_csv(csv_path, rows, stage_names)

    print(f"Loaded {len(run['responses'])} samples from {args.run_path}")
    print(f"Wrote {len(rows)} sample-method rows to {csv_path}")
    print_summary(rows)


if __name__ == "__main__":
    main()
