#!/usr/bin/env python3

import argparse
import csv
import json
import re
from pathlib import Path


TREE_METHOD_PATTERN = re.compile(
    r"^dflash2_(unary|pairwise)_k16_tb(\d+)"
    r"-sanitized_eval_results$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize sandboxed EvalPlus HumanEval results."
    )
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("output_path", type=Path)
    return parser.parse_args()


def method_identity(stem: str) -> tuple[str, int | None]:
    if stem == "baseline-sanitized_eval_results":
        return "target-only", None
    match = TREE_METHOD_PATTERN.match(stem)
    if match is None:
        raise ValueError(f"unexpected EvalPlus result name: {stem}")
    return match.group(1), int(match.group(2))


def main() -> None:
    args = parse_args()
    rows = []
    for path in sorted(args.results_dir.glob("*_eval_results.json")):
        method, budget = method_identity(path.stem)
        result = json.loads(path.read_text(encoding="utf-8"))
        evaluations = result["eval"]
        statuses = [
            sample["base_status"]
            for task_samples in evaluations.values()
            for sample in task_samples
        ]
        passed = sum(status == "pass" for status in statuses)
        rows.append(
            {
                "method_key": path.stem.removesuffix(
                    "-sanitized_eval_results"
                ),
                "method": method,
                "budget": "" if budget is None else budget,
                "evaluated": len(statuses),
                "passed": passed,
                "pass_at_1": passed / len(statuses),
                "evalplus_result": path.name,
            }
        )
    if not rows:
        raise ValueError(f"no EvalPlus results found in {args.results_dir}")
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
