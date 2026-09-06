#!/usr/bin/env python3

import argparse
import csv
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from offline_dflash2_trees import (
    PAIRWISE_MASS_PRESERVING,
    UNARY_FULL_MASS,
    build_best_first_tree,
    build_scorer,
    greedy_path_matched_tokens,
    matched_draft_tokens,
    prefix_entry_budgets,
    target_candidate_path,
)

TARGET_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
DRAFT_REVISION = "dedf8df68adfb1afeaf7b7480c0a0243108177b4"
BOOTSTRAP_SEED = 20260906
BUDGETS = (7, 16, 32, 64)
ORIGINAL = "dflash2_original_ddtree"
PAIRWISE = "dflash2_pairwise_k16"
FIXED_UNARY = "dflash2_unary_k16"
GREEDY_KEY = "dflash2"
FULL_CONFIGS = (
    "dflash2_original_ddtree:7,16,32,64;"
    "dflash2_pairwise_k16:7,16,32,64;"
    "dflash2_unary_k16:32,64"
)
EXPECTED_SAMPLES = {
    "gsm8k": 128,
    "math500": 128,
    "humaneval": 164,
    "mt-bench": 80,
}
PROMPT_ID = re.compile(r":selected:(\d+):turn-(\d+):")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze the frozen official-27B allocation protocol."
    )
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help=(
            "Artifact label and path, for example "
            "GSM8K=artifacts/step8/.../gsm8k_controlled.pt"
        ),
    )
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Allow smoke artifacts with fewer than the frozen sample count.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def method_key(method: str, budget: int) -> str:
    return f"{method}_tb{budget}"


def expected_method_keys() -> tuple[str, ...]:
    return (
        GREEDY_KEY,
        *(method_key(ORIGINAL, budget) for budget in BUDGETS),
        *(method_key(PAIRWISE, budget) for budget in BUDGETS),
        method_key(FIXED_UNARY, 32),
        method_key(FIXED_UNARY, 64),
    )


def load_and_validate(
    path: Path,
    expected_commit: str,
    allow_partial: bool,
    expected_trajectory_mode: str,
) -> dict:
    run = torch.load(path, map_location="cpu", weights_only=False)
    args = run["args"]
    dataset = args["dataset"]
    if run["repository"] != {"commit": expected_commit, "dirty": False}:
        raise ValueError(f"{path} was not produced by clean {expected_commit}")
    if run["target_revision"] != TARGET_REVISION:
        raise ValueError(f"{path} has unexpected target revision")
    if run["draft_revision"] != DRAFT_REVISION:
        raise ValueError(f"{path} has unexpected drafter revision")
    if run["trajectory_mode"] != expected_trajectory_mode:
        raise ValueError(
            f"{path} has trajectory mode {run['trajectory_mode']!r}, "
            f"expected {expected_trajectory_mode!r}"
        )
    if args["temperature"] != 0.0 or args["max_new_tokens"] != 256:
        if not allow_partial:
            raise ValueError(f"{path} does not use the frozen decode settings")
    if args["dflash2_tree_configs"] != FULL_CONFIGS and not allow_partial:
        raise ValueError(f"{path} does not use the frozen method matrix")
    completed = run["completed_dataset_indices"]
    if len(completed) != args["max_samples"]:
        raise ValueError(f"{path} is incomplete")
    if not allow_partial and len(completed) != EXPECTED_SAMPLES[dataset]:
        raise ValueError(f"{path} has the wrong frozen sample count")
    required = set(expected_method_keys())
    for response_index, response in enumerate(run["responses"]):
        missing = required - response.keys()
        if missing and not allow_partial:
            raise ValueError(
                f"{path} response {response_index} misses {sorted(missing)}"
            )
        if GREEDY_KEY in response and response[GREEDY_KEY].trace_rounds is None:
            raise ValueError(f"{path} response {response_index} misses greedy traces")
        for key in required & response.keys():
            if key == GREEDY_KEY:
                continue
            result = response[key]
            budget = int(key.rsplit("_tb", 1)[1])
            expected_candidates = budget if ORIGINAL in key else 16
            for round_metric in result.round_metrics:
                if round_metric["tree_node_count"] != budget:
                    raise ValueError(f"{path} {key} has the wrong node count")
                if round_metric["candidate_count"] != expected_candidates:
                    raise ValueError(f"{path} {key} has the wrong candidate width")
                if "tree" not in round_metric:
                    raise ValueError(f"{path} {key} misses selected tree data")
                if "allocation_lattice" not in round_metric:
                    raise ValueError(f"{path} {key} misses allocation lattice data")
    tree_exact = run["tree_exact_token_matches"]
    expected_tree_keys = required - {GREEDY_KEY}
    if not allow_partial and set(tree_exact) != expected_tree_keys:
        raise ValueError(f"{path} has an unexpected exact-match method matrix")
    for key, exact in tree_exact.items():
        if exact["count"] != exact["total"]:
            raise ValueError(f"{path} tree method {key} diverged from target")
        if exact["total"] == 0:
            raise ValueError(f"{path} tree method {key} has no exact-match checks")
    return run


def prompt_identity(result: object) -> tuple[int, int]:
    if not result.round_metrics:
        raise ValueError("method has no decoding rounds")
    prompt_id = result.round_metrics[0]["prompt_id"]
    match = PROMPT_ID.search(prompt_id)
    if match is None:
        raise ValueError(f"unexpected prompt id: {prompt_id}")
    return int(match.group(1)), int(match.group(2))


def result_metrics(result: object) -> dict[str, float]:
    rounds = result.round_metrics
    if not rounds:
        raise ValueError("method has no decoding rounds")
    matched = np.asarray([float(row["matched_draft_tokens"]) for row in rounds])
    committed = np.asarray(
        [float(row["committed_tokens_this_round"]) for row in rounds]
    )
    return {
        "matched": float(matched.mean()),
        "committed": float(committed.mean()),
        "full_block": float((matched == 7).mean()),
        "target_calls": float(result.decode_rounds),
        "tokens_per_second": 1 / float(result.time_per_output_token),
        "milliseconds_per_token": float(result.time_per_output_token) * 1000,
        "tree_build_ms": float(
            np.mean([row["tree_build_latency_ms"] for row in rounds])
        ),
        "verify_ms": float(
            np.mean([row["target_verify_latency_ms"] for row in rounds])
        ),
    }


def collect_cluster_metrics(
    run: dict,
) -> dict[str, dict[int, dict[str, float]]]:
    per_turn = defaultdict(lambda: defaultdict(list))
    available = set(expected_method_keys())
    for response in run["responses"]:
        methods = available & response.keys()
        if not methods:
            raise ValueError("response contains no protocol methods")
        reference_method = next(iter(methods))
        cluster, _ = prompt_identity(response[reference_method])
        for method in methods:
            per_turn[method][cluster].append(result_metrics(response[method]))

    clustered = defaultdict(dict)
    for method, clusters in per_turn.items():
        for cluster, turns in clusters.items():
            clustered[method][cluster] = {
                metric: float(np.mean([turn[metric] for turn in turns]))
                for metric in turns[0]
            }
    return clustered


def bootstrap_interval(
    values: np.ndarray,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    estimates = values[indices].mean(axis=1)
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high)


def method_label(key: str) -> str:
    if key == GREEDY_KEY:
        return "DFlash2-Greedy"
    method, budget = key.rsplit("_tb", maxsplit=1)
    if method == ORIGINAL:
        return f"Original-DDTree-K{budget}"
    if method == PAIRWISE:
        return f"Pairwise-K16-B{budget}"
    if method == FIXED_UNARY:
        return f"Unary-K16-B{budget}"
    raise ValueError(f"unknown method key: {key}")


def analyze_methods(
    label: str,
    clustered: dict[str, dict[int, dict[str, float]]],
) -> list[dict[str, object]]:
    rows = []
    for key in expected_method_keys():
        if key not in clustered:
            continue
        prompts = list(clustered[key].values())
        budget = 7 if key == GREEDY_KEY else int(key.rsplit("_tb", 1)[1])
        candidate_count = (
            16 if key == GREEDY_KEY or PAIRWISE in key or FIXED_UNARY in key else budget
        )
        rows.append(
            {
                "dataset": label,
                "prompts": len(prompts),
                "budget": budget,
                "method": method_label(key),
                "candidate_count": candidate_count,
                **{
                    f"mean_{metric}": np.mean([prompt[metric] for prompt in prompts])
                    for metric in prompts[0]
                },
            }
        )
    return rows


def comparison(
    label: str,
    clustered: dict[str, dict[int, dict[str, float]]],
    left: str,
    right: str,
    samples: int,
    seed: int,
) -> dict[str, object]:
    shared = sorted(set(clustered[left]) & set(clustered[right]))
    if not shared:
        raise ValueError(f"{label} has no shared prompts for {left} and {right}")
    matched = np.asarray(
        [
            clustered[left][prompt]["matched"] - clustered[right][prompt]["matched"]
            for prompt in shared
        ]
    )
    throughput = np.asarray(
        [
            clustered[left][prompt]["tokens_per_second"]
            - clustered[right][prompt]["tokens_per_second"]
            for prompt in shared
        ]
    )
    matched_ci = bootstrap_interval(matched, samples, seed)
    throughput_ci = bootstrap_interval(throughput, samples, seed + 1)
    return {
        "dataset": label,
        "prompts": len(shared),
        "left_method": method_label(left),
        "right_method": method_label(right),
        "budget": 7 if left == GREEDY_KEY else int(left.rsplit("_tb", 1)[1]),
        "matched_gain": matched.mean(),
        "matched_ci_low": matched_ci[0],
        "matched_ci_high": matched_ci[1],
        "improve_prompts": int((matched > 0).sum()),
        "tie_prompts": int((matched == 0).sum()),
        "hurt_prompts": int((matched < 0).sum()),
        "throughput_gain": throughput.mean(),
        "throughput_ci_low": throughput_ci[0],
        "throughput_ci_high": throughput_ci[1],
    }


def analyze_comparisons(
    label: str,
    clustered: dict[str, dict[int, dict[str, float]]],
    samples: int,
    dataset_offset: int,
) -> list[dict[str, object]]:
    pairs = [
        (
            method_key(PAIRWISE, budget),
            method_key(ORIGINAL, budget),
        )
        for budget in BUDGETS
    ]
    pairs.extend(
        (
            method_key(PAIRWISE, budget),
            method_key(FIXED_UNARY, budget),
        )
        for budget in (32, 64)
    )
    pairs.append((method_key(PAIRWISE, 7), GREEDY_KEY))
    return [
        comparison(
            label,
            clustered,
            left,
            right,
            samples,
            BOOTSTRAP_SEED + dataset_offset * 100 + index * 2,
        )
        for index, (left, right) in enumerate(pairs)
        if left in clustered and right in clustered
    ]


def controlled_allocation_values(
    run: dict,
) -> dict[str, dict[int, float]]:
    prompt_round_values = defaultdict(lambda: defaultdict(list))
    for response in run["responses"]:
        greedy = response[GREEDY_KEY]
        cluster, _ = prompt_identity(greedy)
        if not greedy.trace_rounds:
            raise ValueError("greedy result has no controlled allocation traces")
        for trace_round in greedy.trace_rounds:
            required = {
                "unary_top64_token_ids",
                "unary_top64_logits",
                "unary_logsumexp",
                "candidate_token_ids",
                "candidate_unary_logits",
                "anchor_final_scores",
                "pairwise_final_scores",
                "selected_draft_token_ids",
                "realized_continuation_token_ids",
            }
            missing = required - trace_round.keys()
            if missing:
                raise ValueError(
                    f"controlled allocation trace misses {sorted(missing)}"
                )
            target_tokens = trace_round["realized_continuation_token_ids"]
            prompt_round_values[GREEDY_KEY][cluster].append(
                greedy_path_matched_tokens(
                    trace_round["selected_draft_token_ids"],
                    target_tokens,
                    7,
                )
            )
            pairwise_lattice = {
                key: trace_round[key]
                for key in (
                    "candidate_token_ids",
                    "candidate_unary_logits",
                    "unary_logsumexp",
                    "anchor_final_scores",
                    "pairwise_final_scores",
                )
            }
            pairwise_scorer = build_scorer(
                pairwise_lattice,
                PAIRWISE_MASS_PRESERVING,
            )
            pairwise_nodes = build_best_first_tree(
                pairwise_lattice["candidate_token_ids"],
                pairwise_scorer,
                max(BUDGETS),
            )
            pairwise_path, _ = target_candidate_path(
                pairwise_lattice["candidate_token_ids"],
                target_tokens,
            )
            pairwise_entries = prefix_entry_budgets(
                pairwise_nodes,
                pairwise_path,
            )
            for budget in BUDGETS:
                prompt_round_values[method_key(PAIRWISE, budget)][cluster].append(
                    matched_draft_tokens(pairwise_entries, budget)
                )

                unary_lattice = {
                    "candidate_token_ids": trace_round["unary_top64_token_ids"][
                        :, :budget
                    ],
                    "candidate_unary_logits": trace_round["unary_top64_logits"][
                        :, :budget
                    ],
                    "unary_logsumexp": trace_round["unary_logsumexp"],
                }
                unary_scorer = build_scorer(
                    unary_lattice,
                    UNARY_FULL_MASS,
                )
                unary_nodes = build_best_first_tree(
                    unary_lattice["candidate_token_ids"],
                    unary_scorer,
                    budget,
                )
                unary_path, _ = target_candidate_path(
                    unary_lattice["candidate_token_ids"],
                    target_tokens,
                )
                unary_entries = prefix_entry_budgets(
                    unary_nodes,
                    unary_path,
                )
                prompt_round_values[method_key(ORIGINAL, budget)][cluster].append(
                    matched_draft_tokens(unary_entries, budget)
                )

    return {
        method: {
            cluster: float(np.mean(round_values))
            for cluster, round_values in clusters.items()
        }
        for method, clusters in prompt_round_values.items()
    }


def analyze_controlled_allocation(
    label: str,
    values: dict[str, dict[int, float]],
    bootstrap_samples: int,
    dataset_offset: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    method_rows = []
    for method, prompts in values.items():
        budget = 7 if method == GREEDY_KEY else int(method.rsplit("_tb", 1)[1])
        method_rows.append(
            {
                "dataset": label,
                "prompts": len(prompts),
                "budget": budget,
                "method": method_label(method),
                "mean_matched_draft_tokens": np.mean(list(prompts.values())),
            }
        )
    comparison_rows = []
    pairs = [
        (method_key(PAIRWISE, budget), method_key(ORIGINAL, budget))
        for budget in BUDGETS
    ]
    pairs.append((method_key(PAIRWISE, 7), GREEDY_KEY))
    for index, (left, right) in enumerate(pairs):
        shared = sorted(set(values[left]) & set(values[right]))
        differences = np.asarray(
            [values[left][prompt] - values[right][prompt] for prompt in shared]
        )
        low, high = bootstrap_interval(
            differences,
            bootstrap_samples,
            BOOTSTRAP_SEED + 10000 + dataset_offset * 100 + index,
        )
        comparison_rows.append(
            {
                "dataset": label,
                "prompts": len(shared),
                "left_method": method_label(left),
                "right_method": method_label(right),
                "budget": 7 if left == GREEDY_KEY else int(left.rsplit("_tb", 1)[1]),
                "matched_gain": differences.mean(),
                "matched_ci_low": low,
                "matched_ci_high": high,
                "improve_prompts": int((differences > 0).sum()),
                "tie_prompts": int((differences == 0).sum()),
                "hurt_prompts": int((differences < 0).sum()),
            }
        )
    return method_rows, comparison_rows


def main() -> None:
    args = parse_args()
    method_rows = []
    comparison_rows = []
    controlled_method_rows = []
    controlled_comparison_rows = []
    provenance = {
        "expected_commit": args.expected_commit,
        "target_revision": TARGET_REVISION,
        "draft_revision": DRAFT_REVISION,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_samples": args.bootstrap_samples,
        "sources": {},
    }
    for dataset_offset, run_argument in enumerate(args.run):
        try:
            label, path_text = run_argument.split("=", maxsplit=1)
        except ValueError as exc:
            raise ValueError("--run must use LABEL=PATH") from exc
        path = Path(path_text)
        expected_trajectory_mode = (
            "native" if label.lower().endswith("-native") else "controlled_shared"
        )
        run = load_and_validate(
            path,
            args.expected_commit,
            args.allow_partial,
            expected_trajectory_mode,
        )
        clustered = collect_cluster_metrics(run)
        method_rows.extend(analyze_methods(label, clustered))
        comparison_rows.extend(
            analyze_comparisons(
                label,
                clustered,
                args.bootstrap_samples,
                dataset_offset,
            )
        )
        controlled_values = controlled_allocation_values(run)
        controlled_methods, controlled_comparisons = analyze_controlled_allocation(
            label,
            controlled_values,
            args.bootstrap_samples,
            dataset_offset,
        )
        controlled_method_rows.extend(controlled_methods)
        controlled_comparison_rows.extend(controlled_comparisons)
        provenance["sources"][label] = {
            "path": str(path),
            "sha256": sha256(path),
            "dataset": run["args"]["dataset"],
            "trajectory_mode": run["trajectory_mode"],
            "completed_samples": len(run["completed_dataset_indices"]),
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "method_metrics.csv", method_rows)
    write_csv(args.output_dir / "paired_comparisons.csv", comparison_rows)
    write_csv(
        args.output_dir / "controlled_allocation_metrics.csv",
        controlled_method_rows,
    )
    write_csv(
        args.output_dir / "controlled_allocation_comparisons.csv",
        controlled_comparison_rows,
    )
    (args.output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
