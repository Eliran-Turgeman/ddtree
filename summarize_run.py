#!/usr/bin/env python3

import argparse
import csv
import re
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
    "mean_matched_draft_tokens",
    "mean_committed_tokens_per_round",
    "full_block_acceptance",
    "verifier_bonus_commit_rate",
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
    if method == "dflash2":
        return (3, 0)
    unary_match = re.match(r"^dflash2_unary_k(\d+)_tb(\d+)$", method)
    if unary_match:
        return (
            4,
            int(unary_match.group(1)) * 10000
            + int(unary_match.group(2)),
        )
    if method.startswith("dflash2_pairwise_k16_tb"):
        return (5, int(method.rsplit("tb", maxsplit=1)[1]))
    return (6, 0)


def tree_budget(method: str) -> int | str:
    if method.startswith("ddtree_tb"):
        return int(method.removeprefix("ddtree_tb"))
    if "_tb" in method:
        return int(method.rsplit("tb", maxsplit=1)[1])
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
            matched_draft_tokens = getattr(
                result,
                "matched_draft_tokens_per_round",
                [],
            )
            if (
                not matched_draft_tokens
                and method.startswith("ddtree_tb")
            ):
                matched_draft_tokens = [
                    max(int(value) - 1, 0)
                    for value in acceptance_lengths
                ]
            committed_tokens = getattr(
                result,
                "committed_tokens_per_round",
                [],
            )
            bonus_committed = getattr(
                result,
                "verifier_bonus_committed_per_round",
                [],
            )
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
                "mean_matched_draft_tokens": (
                    mean(matched_draft_tokens)
                    if matched_draft_tokens
                    else ""
                ),
                "mean_committed_tokens_per_round": (
                    mean(committed_tokens) if committed_tokens else ""
                ),
                "full_block_acceptance": (
                    mean(
                        value == run["block_size"] - 1
                        for value in matched_draft_tokens
                    )
                    if matched_draft_tokens
                    else ""
                ),
                "verifier_bonus_commit_rate": (
                    mean(bonus_committed) if bonus_committed else ""
                ),
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


def build_round_rows(run: dict) -> list[dict[str, object]]:
    rows = []
    for sample_index, response in enumerate(run["responses"]):
        for method, result in response.items():
            for metric in getattr(result, "round_metrics", []):
                row = {
                    "sample_index": sample_index,
                    "method_key": method,
                    **metric,
                }
                if isinstance(row.get("nodes_per_depth"), list):
                    for depth, count in enumerate(
                        row.pop("nodes_per_depth"),
                        start=1,
                    ):
                        row[f"nodes_at_depth_{depth}"] = count
                row.pop("tree", None)
                rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, object]], stage_names: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = BASE_COLUMNS + [f"stage_{stage_name}_seconds" for stage_name in stage_names]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_round_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0])
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
        f"{'speedup':>10} {'matched/round':>13}"
    )
    print("-" * 62)
    for method in methods:
        method_rows = [row for row in rows if row["method"] == method]
        mean_ms = mean(float(row["time_per_output_token_ms"]) for row in method_rows)
        matched_rows = [
            row
            for row in method_rows
            if row["mean_matched_draft_tokens"] != ""
            and int(row["decode_rounds"]) > 0
        ]
        total_rounds = sum(int(row["decode_rounds"]) for row in matched_rows)
        matched = (
            sum(
                float(row["mean_matched_draft_tokens"])
                * int(row["decode_rounds"])
                for row in matched_rows
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
            f"{matched:>12.2f}"
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
    parser.add_argument(
        "--rounds-csv",
        type=Path,
        default=None,
        help="Round-level CSV path (default: <run>.rounds.csv)",
    )
    args = parser.parse_args()

    csv_path = args.csv or args.run_path.with_suffix(".csv")
    run = load_run(args.run_path)
    rows, stage_names = build_rows(run)
    write_csv(csv_path, rows, stage_names)
    round_rows = build_round_rows(run)
    rounds_csv_path = (
        args.rounds_csv
        or args.run_path.with_suffix(".rounds.csv")
    )
    write_round_csv(rounds_csv_path, round_rows)

    print(f"Loaded {len(run['responses'])} samples from {args.run_path}")
    print(f"Wrote {len(rows)} sample-method rows to {csv_path}")
    if round_rows:
        print(
            f"Wrote {len(round_rows)} round rows to "
            f"{rounds_csv_path}"
        )
    print_summary(rows)


if __name__ == "__main__":
    main()
