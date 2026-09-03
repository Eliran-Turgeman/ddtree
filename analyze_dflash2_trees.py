#!/usr/bin/env python3

import argparse
import csv
import math
from pathlib import Path
from statistics import mean, median

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from inspect_dflash2_traces import load_trace
from offline_dflash2_trees import (
    PAIRWISE_AFTER_ROOT,
    PAIRWISE_MASS_PRESERVING,
    UNARY_FULL_MASS,
    UNARY_TRUNCATED,
    build_best_first_tree,
    build_scorers,
    greedy_path_matched_tokens,
    matched_draft_tokens,
    prefix_entry_budgets,
    target_candidate_path,
)


BUDGETS = (1, 2, 4, 7, 8, 16, 32, 64, 128, 256)
METHODS = (
    UNARY_FULL_MASS,
    UNARY_TRUNCATED,
    PAIRWISE_MASS_PRESERVING,
    PAIRWISE_AFTER_ROOT,
)
DFLASH2_GREEDY_PATH = "DFlash2-GreedyPath"
ORACLE = "Oracle-Lattice"
REPORT_METHODS = (*METHODS, DFLASH2_GREEDY_PATH, ORACLE)
BOOTSTRAP_SEED = 20260903


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate unary and pairwise DDTree scoring entirely offline "
            "from a frozen DFlash2 trace."
        )
    )
    parser.add_argument("trace_path", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=10_000,
    )
    parser.add_argument(
        "--max-budget",
        type=int,
        default=max(BUDGETS),
    )
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def competition_rank(scores: torch.Tensor, target_index: int) -> int:
    target_score = scores[target_index]
    return 1 + int((scores > target_score).sum().item())


def cluster_bootstrap_estimates(
    values: np.ndarray,
    prompt_indices: np.ndarray,
    sampled_prompts: np.ndarray,
    prompt_count: int,
) -> np.ndarray:
    prompt_sums = np.bincount(
        prompt_indices,
        weights=values,
        minlength=prompt_count,
    )
    prompt_counts = np.bincount(
        prompt_indices,
        minlength=prompt_count,
    )
    sampled_sums = prompt_sums[sampled_prompts].sum(axis=1)
    sampled_counts = prompt_counts[sampled_prompts].sum(axis=1)
    return sampled_sums / sampled_counts


def confidence_interval(estimates: np.ndarray) -> tuple[float, float]:
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high)


def direct_prefix_length(direct_mask: torch.Tensor) -> int:
    direct_mask = direct_mask.bool().cpu()
    length = int(direct_mask.sum())
    expected = torch.arange(direct_mask.numel()) < length
    if not torch.equal(direct_mask, expected):
        raise ValueError(
            "directly_observed_target_mask must be a contiguous prefix: "
            f"{direct_mask.tolist()}"
        )
    return length


def summarize_distribution(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "p05": float(np.quantile(values, 0.05)),
        "p25": float(np.quantile(values, 0.25)),
        "median": float(np.quantile(values, 0.50)),
        "p75": float(np.quantile(values, 0.75)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(values.max()),
    }


def right_censored_quantile(
    observed: np.ndarray,
    censored_count: int,
    quantile: float,
    max_budget: int,
) -> float | str:
    total = observed.size + censored_count
    if total == 0:
        return ""
    position = quantile * (total - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if upper >= observed.size:
        return f">{max_budget}"
    ordered = np.sort(observed)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(
        ordered[lower] * (1 - weight)
        + ordered[upper] * weight
    )


def plot_mean_depth(
    output_path: Path,
    summary_rows: list[dict],
) -> None:
    colors = {
        UNARY_FULL_MASS: "#1f77b4",
        UNARY_TRUNCATED: "#9467bd",
        PAIRWISE_MASS_PRESERVING: "#d62728",
        PAIRWISE_AFTER_ROOT: "#ff7f0e",
        DFLASH2_GREEDY_PATH: "#8c564b",
        ORACLE: "#2ca02c",
    }
    fig, ax = plt.subplots(figsize=(8, 5))
    for method in REPORT_METHODS:
        rows = [
            row
            for row in summary_rows
            if row["method"] == method
        ]
        x = np.array([row["budget"] for row in rows])
        y = np.array([row["mean"] for row in rows])
        low = np.array([row["ci_low"] for row in rows])
        high = np.array([row["ci_high"] for row in rows])
        ax.plot(
            x,
            y,
            marker="o",
            label=method,
            color=colors[method],
        )
        ax.fill_between(
            x,
            low,
            high,
            alpha=0.12,
            color=colors[method],
        )
    ax.set_xscale("log", base=2)
    ax.set_xticks(BUDGETS, labels=[str(value) for value in BUDGETS])
    ax.set_xlabel("Verification-tree budget B")
    ax.set_ylabel("Mean tree-matched draft tokens")
    ax.set_title("Offline target-prefix coverage by tree budget")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_prefix_coverage(
    output_path: Path,
    coverage_rows: list[dict],
) -> None:
    selected_budgets = (8, 16, 32, 64, 128)
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), sharex=True, sharey=True)
    axes = axes.ravel()
    colors = {
        UNARY_FULL_MASS: "#1f77b4",
        UNARY_TRUNCATED: "#9467bd",
        PAIRWISE_MASS_PRESERVING: "#d62728",
        PAIRWISE_AFTER_ROOT: "#ff7f0e",
        DFLASH2_GREEDY_PATH: "#8c564b",
        ORACLE: "#2ca02c",
    }
    for axis, budget in zip(axes, selected_budgets):
        for method in REPORT_METHODS:
            rows = [
                row
                for row in coverage_rows
                if row["scope"] == "realized"
                and row["budget"] == budget
                and row["method"] == method
            ]
            axis.plot(
                [row["depth"] for row in rows],
                [row["coverage"] for row in rows],
                marker="o",
                label=method,
                color=colors[method],
            )
        axis.set_title(f"B = {budget}")
        axis.grid(alpha=0.25)
        axis.set_ylim(0, 1.02)
    axes[-1].axis("off")
    for axis in axes[:5]:
        axis.set_xticks(range(1, 8))
    axes[3].set_xlabel("Target-prefix depth")
    axes[4].set_xlabel("Target-prefix depth")
    axes[0].set_ylabel("Coverage")
    axes[3].set_ylabel("Coverage")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right")
    fig.suptitle("Realized target-prefix coverage")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_transition_ranks(
    output_path: Path,
    ranking_rows: list[dict],
) -> None:
    depths = np.array([row["depth"] for row in ranking_rows])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(
        depths,
        [row["unary_mean_rank"] for row in ranking_rows],
        marker="o",
        label="Unary",
    )
    axes[0].plot(
        depths,
        [row["pairwise_mean_rank"] for row in ranking_rows],
        marker="o",
        label="Pairwise",
    )
    axes[0].set_ylabel("Mean target-token rank")
    axes[0].set_xlabel("Speculative depth")
    axes[0].set_xticks(depths)
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(
        depths,
        [row["unary_recall_at_1"] for row in ranking_rows],
        marker="o",
        label="Unary",
    )
    axes[1].plot(
        depths,
        [row["pairwise_recall_at_1"] for row in ranking_rows],
        marker="o",
        label="Pairwise",
    )
    axes[1].set_ylabel("Recall@1 within top-16")
    axes[1].set_xlabel("Speculative depth")
    axes[1].set_xticks(depths)
    axes[1].set_ylim(0, 1.02)
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    fig.suptitle(
        "Direct target-transition rank, conditioned on top-16 inclusion"
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_improvement_distribution(
    output_path: Path,
    alpha_by_method: dict[str, np.ndarray],
) -> None:
    categories = ("Helps", "Ties", "Hurts")
    budgets = (32, 64)
    x = np.arange(len(categories))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for offset, budget in enumerate(budgets):
        budget_index = BUDGETS.index(budget)
        differences = (
            alpha_by_method[PAIRWISE_MASS_PRESERVING][:, budget_index]
            - alpha_by_method[UNARY_FULL_MASS][:, budget_index]
        )
        counts = [
            int((differences > 0).sum()),
            int((differences == 0).sum()),
            int((differences < 0).sum()),
        ]
        ax.bar(
            x + (offset - 0.5) * width,
            counts,
            width,
            label=f"B={budget}",
        )
    ax.set_xticks(x, labels=categories)
    ax.set_ylabel("Decoding rounds")
    ax.set_title("Pairwise minus Unary-FullMass matched depth")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.bootstrap_samples <= 0:
        raise ValueError("--bootstrap-samples must be positive")
    if args.max_budget < max(BUDGETS):
        raise ValueError(
            f"--max-budget must be at least {max(BUDGETS)}"
        )

    artifact = load_trace(args.trace_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    prompt_indices = []
    alpha_lists = {
        method: []
        for method in REPORT_METHODS
    }
    retained_mass = [[] for _ in range(7)]
    transition_ranks = [
        {
            "eligible": 0,
            "in_lattice": 0,
            "unary": [],
            "pairwise": [],
        }
        for _ in range(7)
    ]
    round_records = []
    entry_records = []
    maximum_extension_scores = {
        method: -math.inf
        for method in METHODS
    }

    total_rounds = sum(
        len(prompt["rounds"])
        for prompt in artifact["prompts"]
    )
    progress = tqdm(total=total_rounds, desc="Evaluating offline trees")
    round_index = 0
    for prompt_index, prompt in enumerate(artifact["prompts"]):
        for trace_round in prompt["rounds"]:
            candidates = trace_round["candidate_token_ids"].long()
            realized = trace_round[
                "realized_continuation_token_ids"
            ].long()
            direct_mask = trace_round[
                "directly_observed_target_mask"
            ].bool()
            verifier = trace_round["verifier_token_ids"].long()
            direct_length = direct_prefix_length(direct_mask)
            realized_length = min(
                realized.numel(),
                candidates.shape[0],
            )

            unary_logits = trace_round[
                "candidate_unary_logits"
            ].float()
            log_mass = (
                torch.logsumexp(unary_logits, dim=-1)
                - trace_round["unary_logsumexp"].float()
            )
            mass = log_mass.exp()
            if bool(torch.any(mass > 1.0 + 1e-5)):
                raise ValueError(
                    f"retained unary mass exceeds 1 in round {round_index}"
                )
            for depth_index, value in enumerate(mass):
                retained_mass[depth_index].append(float(value))

            target_path, representable_depth = target_candidate_path(
                candidates,
                realized,
            )
            method_entries = {}
            scorers = build_scorers(trace_round)
            for method, scorer in scorers.items():
                maximum_extension_scores[method] = max(
                    maximum_extension_scores[method],
                    scorer.maximum_extension_log_score(),
                )
                if scorer.maximum_extension_log_score() > 1e-5:
                    raise ValueError(
                        f"{method} violates monotonicity in round "
                        f"{round_index}: maximum extension log score "
                        f"{scorer.maximum_extension_log_score()}"
                    )
                nodes = build_best_first_tree(
                    candidates,
                    scorer,
                    args.max_budget,
                )
                entries = prefix_entry_budgets(nodes, target_path)
                method_entries[method] = entries
                alpha_lists[method].append(
                    [
                        matched_draft_tokens(entries, budget)
                        for budget in BUDGETS
                    ]
                )

            alpha_lists[DFLASH2_GREEDY_PATH].append(
                [
                    greedy_path_matched_tokens(
                        trace_round["selected_draft_token_ids"],
                        realized,
                        budget,
                    )
                    for budget in BUDGETS
                ]
            )
            alpha_lists[ORACLE].append(
                [
                    min(representable_depth, budget)
                    for budget in BUDGETS
                ]
            )

            for depth_index in range(candidates.shape[0]):
                if (
                    depth_index >= direct_mask.numel()
                    or not direct_mask[depth_index]
                ):
                    continue
                transition_ranks[depth_index]["eligible"] += 1
                target_token = int(verifier[depth_index])
                target_matches = torch.nonzero(
                    candidates[depth_index] == target_token,
                    as_tuple=True,
                )[0]
                if target_matches.numel() == 0:
                    continue
                target_index = int(target_matches[0])
                transition_ranks[depth_index]["in_lattice"] += 1
                unary_rank = competition_rank(
                    unary_logits[depth_index],
                    target_index,
                )
                if depth_index == 0:
                    pairwise_scores = trace_round[
                        "anchor_final_scores"
                    ].float()
                else:
                    predecessor_token = int(verifier[depth_index - 1])
                    predecessor_matches = torch.nonzero(
                        candidates[depth_index - 1]
                        == predecessor_token,
                        as_tuple=True,
                    )[0]
                    if predecessor_matches.numel() != 1:
                        raise ValueError(
                            "directly observed predecessor is missing or "
                            f"duplicated in round {round_index}, depth "
                            f"{depth_index + 1}"
                        )
                    pairwise_scores = trace_round[
                        "pairwise_final_scores"
                    ][
                        depth_index - 1,
                        int(predecessor_matches[0]),
                    ].float()
                pairwise_rank = competition_rank(
                    pairwise_scores,
                    target_index,
                )
                transition_ranks[depth_index]["unary"].append(unary_rank)
                transition_ranks[depth_index]["pairwise"].append(
                    pairwise_rank
                )

            prompt_indices.append(prompt_index)
            round_records.append(
                {
                    "round_index": round_index,
                    "prompt_index": prompt_index,
                    "prompt_id": prompt["prompt_id"],
                    "round_id": trace_round["round_id"],
                    "realized_length": realized_length,
                    "direct_length": direct_length,
                    "representable_depth": representable_depth,
                }
            )
            for method in METHODS:
                entries = method_entries[method]
                for depth in range(1, realized_length + 1):
                    representable = representable_depth >= depth
                    entry_budget = (
                        entries[depth - 1]
                        if representable
                        else None
                    )
                    entry_records.append(
                        {
                            "round_index": round_index,
                            "prompt_index": prompt_index,
                            "prompt_id": prompt["prompt_id"],
                            "method": method,
                            "depth": depth,
                            "representable": representable,
                            "entry_budget": (
                                entry_budget
                                if entry_budget is not None
                                else ""
                            ),
                            "censored_above_budget": (
                                representable
                                and entry_budget is None
                            ),
                        }
                    )
            round_index += 1
            progress.update(1)
    progress.close()

    prompt_indices_array = np.asarray(prompt_indices, dtype=np.int64)
    alpha_by_method = {
        method: np.asarray(values, dtype=np.int16)
        for method, values in alpha_lists.items()
    }
    direct_lengths = np.asarray(
        [row["direct_length"] for row in round_records]
    )
    direct_alpha_by_method = {
        method: np.minimum(values, direct_lengths[:, None])
        for method, values in alpha_by_method.items()
    }
    prompt_count = len(artifact["prompts"])
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    sampled_prompts = rng.integers(
        0,
        prompt_count,
        size=(args.bootstrap_samples, prompt_count),
    )

    mean_rows = []
    bootstrap_by_method_budget = {}
    for method in REPORT_METHODS:
        for budget_index, budget in enumerate(BUDGETS):
            values = alpha_by_method[method][:, budget_index].astype(
                np.float64
            )
            estimates = cluster_bootstrap_estimates(
                values,
                prompt_indices_array,
                sampled_prompts,
                prompt_count,
            )
            bootstrap_by_method_budget[(method, budget)] = estimates
            ci_low, ci_high = confidence_interval(estimates)
            mean_rows.append(
                {
                    "method": method,
                    "budget": budget,
                    "mean": float(values.mean()),
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                }
            )

    comparison_rows = []
    for baseline in (UNARY_FULL_MASS, UNARY_TRUNCATED):
        for budget_index, budget in enumerate(BUDGETS):
            differences = (
                alpha_by_method[PAIRWISE_MASS_PRESERVING][
                    :, budget_index
                ]
                - alpha_by_method[baseline][:, budget_index]
            ).astype(np.float64)
            estimates = cluster_bootstrap_estimates(
                differences,
                prompt_indices_array,
                sampled_prompts,
                prompt_count,
            )
            ci_low, ci_high = confidence_interval(estimates)
            oracle_headroom = (
                alpha_by_method[ORACLE][:, budget_index]
                - alpha_by_method[UNARY_FULL_MASS][:, budget_index]
            ).mean()
            pairwise_gain = (
                alpha_by_method[PAIRWISE_MASS_PRESERVING][
                    :, budget_index
                ]
                - alpha_by_method[UNARY_FULL_MASS][:, budget_index]
            ).mean()
            comparison_rows.append(
                {
                    "baseline": baseline,
                    "budget": budget,
                    "mean_paired_difference": float(differences.mean()),
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "bootstrap_probability_positive": float(
                        (estimates > 0).mean()
                    ),
                    "pairwise_gain_fraction_vs_unary_full": (
                        float(pairwise_gain / oracle_headroom)
                        if baseline == UNARY_FULL_MASS
                        and oracle_headroom > 1e-12
                        else ""
                    ),
                }
            )

    per_prompt_effect_rows = []
    equal_prompt_rows = []
    for budget_index, budget in enumerate(BUDGETS):
        round_differences = (
            alpha_by_method[PAIRWISE_MASS_PRESERVING][:, budget_index]
            - alpha_by_method[UNARY_FULL_MASS][:, budget_index]
        ).astype(np.float64)
        prompt_effects = np.empty(prompt_count, dtype=np.float64)
        for prompt_index in range(prompt_count):
            prompt_values = round_differences[
                prompt_indices_array == prompt_index
            ]
            if prompt_values.size == 0:
                raise ValueError(
                    f"prompt {prompt_index} has no decoding rounds"
                )
            prompt_effects[prompt_index] = prompt_values.mean()
            per_prompt_effect_rows.append(
                {
                    "prompt_index": prompt_index,
                    "prompt_id": artifact["prompts"][prompt_index][
                        "prompt_id"
                    ],
                    "round_count": int(prompt_values.size),
                    "budget": budget,
                    "mean_paired_difference": float(
                        prompt_effects[prompt_index]
                    ),
                }
            )
        estimates = prompt_effects[sampled_prompts].mean(axis=1)
        ci_low, ci_high = confidence_interval(estimates)
        equal_prompt_rows.append(
            {
                "budget": budget,
                "mean_equal_prompt_paired_difference": float(
                    prompt_effects.mean()
                ),
                "ci_low": ci_low,
                "ci_high": ci_high,
                "bootstrap_probability_positive": float(
                    (estimates > 0).mean()
                ),
                "prompts_positive": int((prompt_effects > 0).sum()),
                "prompts_tied": int((prompt_effects == 0).sum()),
                "prompts_negative": int((prompt_effects < 0).sum()),
            }
        )

    after_root_rows = []
    for budget_index, budget in enumerate(BUDGETS):
        differences = (
            alpha_by_method[PAIRWISE_AFTER_ROOT][:, budget_index]
            - alpha_by_method[PAIRWISE_MASS_PRESERVING][:, budget_index]
        ).astype(np.float64)
        estimates = cluster_bootstrap_estimates(
            differences,
            prompt_indices_array,
            sampled_prompts,
            prompt_count,
        )
        ci_low, ci_high = confidence_interval(estimates)
        after_root_rows.append(
            {
                "budget": budget,
                "mean_paired_difference": float(differences.mean()),
                "ci_low": ci_low,
                "ci_high": ci_high,
                "bootstrap_probability_positive": float(
                    (estimates > 0).mean()
                ),
            }
        )

    path_tree_rows = []
    budget_7_index = BUDGETS.index(7)
    for method in (
        UNARY_FULL_MASS,
        PAIRWISE_MASS_PRESERVING,
        PAIRWISE_AFTER_ROOT,
    ):
        differences = (
            alpha_by_method[method][:, budget_7_index]
            - alpha_by_method[DFLASH2_GREEDY_PATH][:, budget_7_index]
        ).astype(np.float64)
        estimates = cluster_bootstrap_estimates(
            differences,
            prompt_indices_array,
            sampled_prompts,
            prompt_count,
        )
        ci_low, ci_high = confidence_interval(estimates)
        path_tree_rows.append(
            {
                "tree_method": method,
                "budget": 7,
                "mean_paired_difference_vs_greedy_path": float(
                    differences.mean()
                ),
                "ci_low": ci_low,
                "ci_high": ci_high,
                "bootstrap_probability_positive": float(
                    (estimates > 0).mean()
                ),
            }
        )

    cross_budget_rows = []
    for pairwise_budget, unary_budget in (
        (8, 16),
        (16, 64),
        (32, 256),
        (64, 256),
    ):
        differences = (
            alpha_by_method[PAIRWISE_MASS_PRESERVING][
                :, BUDGETS.index(pairwise_budget)
            ]
            - alpha_by_method[UNARY_FULL_MASS][
                :, BUDGETS.index(unary_budget)
            ]
        ).astype(np.float64)
        estimates = cluster_bootstrap_estimates(
            differences,
            prompt_indices_array,
            sampled_prompts,
            prompt_count,
        )
        ci_low, ci_high = confidence_interval(estimates)
        cross_budget_rows.append(
            {
                "pairwise_budget": pairwise_budget,
                "unary_full_budget": unary_budget,
                "mean_paired_difference": float(differences.mean()),
                "ci_low": ci_low,
                "ci_high": ci_high,
                "bootstrap_probability_positive": float(
                    (estimates > 0).mean()
                ),
                "exploratory_noninferiority_margin": 0.1,
                "exploratory_noninferior": ci_low > -0.1,
            }
        )

    direct_mean_rows = []
    direct_comparison_rows = []
    for method in REPORT_METHODS:
        for budget_index, budget in enumerate(BUDGETS):
            values = direct_alpha_by_method[method][
                :, budget_index
            ].astype(np.float64)
            estimates = cluster_bootstrap_estimates(
                values,
                prompt_indices_array,
                sampled_prompts,
                prompt_count,
            )
            ci_low, ci_high = confidence_interval(estimates)
            direct_mean_rows.append(
                {
                    "method": method,
                    "budget": budget,
                    "mean_direct_observation_censored_depth": float(
                        values.mean()
                    ),
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                }
            )
    for baseline in (UNARY_FULL_MASS, UNARY_TRUNCATED):
        for budget_index, budget in enumerate(BUDGETS):
            differences = (
                direct_alpha_by_method[PAIRWISE_MASS_PRESERVING][
                    :, budget_index
                ]
                - direct_alpha_by_method[baseline][:, budget_index]
            ).astype(np.float64)
            estimates = cluster_bootstrap_estimates(
                differences,
                prompt_indices_array,
                sampled_prompts,
                prompt_count,
            )
            ci_low, ci_high = confidence_interval(estimates)
            direct_comparison_rows.append(
                {
                    "baseline": baseline,
                    "budget": budget,
                    "mean_paired_difference": float(differences.mean()),
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "bootstrap_probability_positive": float(
                        (estimates > 0).mean()
                    ),
                }
            )

    retained_mass_rows = []
    for depth, values in enumerate(retained_mass, start=1):
        array = np.asarray(values)
        retained_mass_rows.append(
            {
                "depth": depth,
                "N": array.size,
                **summarize_distribution(array),
            }
        )

    ranking_rows = []
    for depth, values in enumerate(transition_ranks, start=1):
        unary = np.asarray(values["unary"], dtype=np.float64)
        pairwise = np.asarray(values["pairwise"], dtype=np.float64)
        if unary.size == 0:
            raise ValueError(
                f"no in-lattice direct transitions at depth {depth}"
            )
        ranking_rows.append(
            {
                "depth": depth,
                "eligible_N": values["eligible"],
                "in_lattice_N": values["in_lattice"],
                "unary_mean_rank": float(unary.mean()),
                "pairwise_mean_rank": float(pairwise.mean()),
                "unary_median_rank": float(median(unary)),
                "pairwise_median_rank": float(median(pairwise)),
                "unary_recall_at_1": float((unary <= 1).mean()),
                "pairwise_recall_at_1": float((pairwise <= 1).mean()),
                "unary_recall_at_2": float((unary <= 2).mean()),
                "pairwise_recall_at_2": float((pairwise <= 2).mean()),
                "unary_recall_at_4": float((unary <= 4).mean()),
                "pairwise_recall_at_4": float((pairwise <= 4).mean()),
                "unary_mrr": float((1 / unary).mean()),
                "pairwise_mrr": float((1 / pairwise).mean()),
            }
        )

    coverage_rows = []
    failure_rows = []
    realized_lengths = np.asarray(
        [row["realized_length"] for row in round_records]
    )
    representable_depths = np.asarray(
        [row["representable_depth"] for row in round_records]
    )
    for scope, lengths in (
        ("realized", realized_lengths),
        ("direct", direct_lengths),
    ):
        for method in REPORT_METHODS:
            for budget_index, budget in enumerate(BUDGETS):
                alphas = alpha_by_method[method][:, budget_index]
                for depth in range(1, 8):
                    eligible = lengths >= depth
                    total = int(eligible.sum())
                    hits = int(((alphas >= depth) & eligible).sum())
                    coverage_rows.append(
                        {
                            "scope": scope,
                            "method": method,
                            "budget": budget,
                            "depth": depth,
                            "hits": hits,
                            "N": total,
                            "coverage": hits / total if total else "",
                        }
                    )
                    if method == ORACLE:
                        continue
                    candidate_failures = int(
                        (
                            eligible
                            & (representable_depths < depth)
                        ).sum()
                    )
                    ranking_failures = int(
                        (
                            eligible
                            & (representable_depths >= depth)
                            & (alphas < depth)
                        ).sum()
                    )
                    failure_rows.append(
                        {
                            "scope": scope,
                            "method": method,
                            "budget": budget,
                            "depth": depth,
                            "N": total,
                            "covered": hits,
                            "candidate_failures": candidate_failures,
                            "ranking_budget_failures": ranking_failures,
                            "candidate_failure_rate": (
                                candidate_failures / total
                                if total
                                else ""
                            ),
                            "ranking_budget_failure_rate": (
                                ranking_failures / total
                                if total
                                else ""
                            ),
                        }
                    )

    budget_required_rows = []
    for method in METHODS:
        method_records = [
            row for row in entry_records if row["method"] == method
        ]
        for depth in range(1, 8):
            rows = [
                row
                for row in method_records
                if row["depth"] == depth
            ]
            representable_rows = [
                row for row in rows if row["representable"]
            ]
            observed = np.asarray(
                [
                    row["entry_budget"]
                    for row in representable_rows
                    if row["entry_budget"] != ""
                ],
                dtype=np.float64,
            )
            censored_count = len(representable_rows) - observed.size
            budget_required_rows.append(
                {
                    "method": method,
                    "depth": depth,
                    "eligible_N": len(rows),
                    "representable_N": len(representable_rows),
                    "entered_by_max_budget_N": observed.size,
                    "censored_above_max_budget_N": (
                        censored_count
                    ),
                    "max_budget": args.max_budget,
                    "min_entry_budget": (
                        float(observed.min()) if observed.size else ""
                    ),
                    "p25_entry_budget": (
                        right_censored_quantile(
                            observed,
                            censored_count,
                            0.25,
                            args.max_budget,
                        )
                    ),
                    "median_entry_budget": (
                        right_censored_quantile(
                            observed,
                            censored_count,
                            0.50,
                            args.max_budget,
                        )
                    ),
                    "p75_entry_budget": (
                        right_censored_quantile(
                            observed,
                            censored_count,
                            0.75,
                            args.max_budget,
                        )
                    ),
                    "max_observed_entry_budget": (
                        float(observed.max()) if observed.size else ""
                    ),
                }
            )

    full_block_rows = [
        row
        for row in coverage_rows
        if row["depth"] == 7
    ]

    write_csv(args.output_dir / "mean_matched_depth.csv", mean_rows)
    write_csv(args.output_dir / "paired_comparisons.csv", comparison_rows)
    write_csv(
        args.output_dir / "equal_prompt_paired_comparisons.csv",
        equal_prompt_rows,
    )
    write_csv(
        args.output_dir / "per_prompt_paired_effects.csv",
        per_prompt_effect_rows,
    )
    write_csv(
        args.output_dir / "pairwise_after_root_comparisons.csv",
        after_root_rows,
    )
    write_csv(
        args.output_dir / "greedy_path_tree_comparisons.csv",
        path_tree_rows,
    )
    write_csv(
        args.output_dir / "cross_budget_comparisons.csv",
        cross_budget_rows,
    )
    write_csv(
        args.output_dir / "direct_censored_mean_matched_depth.csv",
        direct_mean_rows,
    )
    write_csv(
        args.output_dir / "direct_censored_paired_comparisons.csv",
        direct_comparison_rows,
    )
    write_csv(args.output_dir / "retained_mass_by_depth.csv", retained_mass_rows)
    write_csv(args.output_dir / "transition_ranking.csv", ranking_rows)
    write_csv(args.output_dir / "prefix_coverage.csv", coverage_rows)
    write_csv(args.output_dir / "full_block_coverage.csv", full_block_rows)
    write_csv(args.output_dir / "failure_decomposition.csv", failure_rows)
    write_csv(args.output_dir / "budget_required_summary.csv", budget_required_rows)
    write_csv(args.output_dir / "budget_required_per_round.csv", entry_records)

    torch.save(
        {
            "budgets": BUDGETS,
            "methods": REPORT_METHODS,
            "prompt_indices": torch.from_numpy(prompt_indices_array),
            "alpha_by_method": {
                method: torch.from_numpy(values)
                for method, values in alpha_by_method.items()
            },
            "realized_lengths": torch.from_numpy(realized_lengths),
            "direct_lengths": torch.from_numpy(direct_lengths),
            "representable_depths": torch.from_numpy(
                representable_depths
            ),
        },
        args.output_dir / "per_round_results.pt",
    )

    plot_mean_depth(
        args.output_dir / "mean_matched_depth_vs_budget.png",
        mean_rows,
    )
    plot_prefix_coverage(
        args.output_dir / "prefix_coverage_vs_budget.png",
        coverage_rows,
    )
    plot_transition_ranks(
        args.output_dir / "target_transition_rank.png",
        ranking_rows,
    )
    plot_improvement_distribution(
        args.output_dir / "pairwise_improvement_distribution.png",
        alpha_by_method,
    )

    monotonicity_rows = [
        {
            "method": method,
            "maximum_extension_log_score": maximum_extension_scores[
                method
            ],
            "monotonic": maximum_extension_scores[method] <= 1e-5,
        }
        for method in METHODS
    ]
    write_csv(args.output_dir / "monotonicity.csv", monotonicity_rows)

    representative_differences = {
        budget: (
            alpha_by_method[PAIRWISE_MASS_PRESERVING][
                :, BUDGETS.index(budget)
            ]
            - alpha_by_method[UNARY_FULL_MASS][
                :, BUDGETS.index(budget)
            ]
        )
        for budget in (32, 64)
    }
    with (args.output_dir / "summary.md").open(
        "w",
        encoding="utf-8",
    ) as file:
        file.write("# Offline DFlash2 UnaryTree vs PairwiseTree\n\n")
        file.write(
            "All results use the frozen Step-3 trace and make no model "
            "inference calls.\n\n"
        )
        file.write(
            "Every method is restricted to the saved DFlash2 top-16 "
            "candidate lattice. `Unary-FullMass` uses the full-vocabulary "
            "log-normalizer but cannot select tokens outside that top 16; "
            "it is not unrestricted DDTree for budgets above 16.\n\n"
        )
        file.write("## Mean matched draft tokens\n\n")
        file.write(
            "| Budget | Unary-FullMass | Unary-Truncated | "
            "Pairwise | Pairwise-after-root | DFlash2 greedy | Oracle |\n"
        )
        file.write("| -----: | -------------: | --------------: | ")
        file.write("-------: | ------------------: | --------------: | -----: |\n")
        for budget in BUDGETS:
            values = {
                row["method"]: row["mean"]
                for row in mean_rows
                if row["budget"] == budget
            }
            file.write(
                f"| {budget} | {values[UNARY_FULL_MASS]:.4f} | "
                f"{values[UNARY_TRUNCATED]:.4f} | "
                f"{values[PAIRWISE_MASS_PRESERVING]:.4f} | "
                f"{values[PAIRWISE_AFTER_ROOT]:.4f} | "
                f"{values[DFLASH2_GREEDY_PATH]:.4f} | "
                f"{values[ORACLE]:.4f} |\n"
            )
        budget_7_values = {
            row["method"]: row["mean"]
            for row in mean_rows
            if row["budget"] == 7
        }
        file.write("\n## Seven-node path/tree comparison\n\n")
        file.write("| Method | Mean | Difference vs greedy path | 95% CI |\n")
        file.write("| :----- | ---: | ------------------------: | -----: |\n")
        file.write(
            f"| DFlash2 greedy path | "
            f"{budget_7_values[DFLASH2_GREEDY_PATH]:.4f} | -- | -- |\n"
        )
        for method in (
            UNARY_FULL_MASS,
            PAIRWISE_MASS_PRESERVING,
            PAIRWISE_AFTER_ROOT,
        ):
            row = next(
                item
                for item in path_tree_rows
                if item["tree_method"] == method
            )
            file.write(
                f"| {method} | {budget_7_values[method]:.4f} | "
                f"{row['mean_paired_difference_vs_greedy_path']:+.4f} | "
                f"[{row['ci_low']:+.4f}, {row['ci_high']:+.4f}] |\n"
            )
        file.write("\n## Pairwise-after-root ablation\n\n")
        file.write(
            "This ablation uses Unary-FullMass at depth 1 and pairwise "
            "mass-preserving transitions from depth 2 onward.\n\n"
        )
        file.write("| Budget | Difference vs Pairwise | 95% CI |\n")
        file.write("| -----: | ---------------------: | -----: |\n")
        for row in after_root_rows:
            file.write(
                f"| {row['budget']} | "
                f"{row['mean_paired_difference']:+.4f} | "
                f"[{row['ci_low']:+.4f}, {row['ci_high']:+.4f}] |\n"
            )
        file.write("\n## Pairwise improvement counts\n\n")
        file.write("| Budget | Helps | Ties | Hurts |\n")
        file.write("| -----: | ----: | ---: | ----: |\n")
        for budget, differences in representative_differences.items():
            file.write(
                f"| {budget} | {int((differences > 0).sum())} | "
                f"{int((differences == 0).sum())} | "
                f"{int((differences < 0).sum())} |\n"
            )
        file.write("\n## Equal-prompt-weighted pairwise gain\n\n")
        file.write(
            "The primary mean above is round-weighted. This robustness "
            "estimand first averages the paired effect within each prompt, "
            "then weights all 32 prompts equally.\n\n"
        )
        file.write(
            "| Budget | Equal-prompt mean gain | 95% CI | "
            "Positive prompts |\n"
        )
        file.write("| -----: | ---------------------: | -----: | ")
        file.write("---------------: |\n")
        for row in equal_prompt_rows:
            file.write(
                f"| {row['budget']} | "
                f"{row['mean_equal_prompt_paired_difference']:+.4f} | "
                f"[{row['ci_low']:+.4f}, {row['ci_high']:+.4f}] | "
                f"{row['prompts_positive']}/{prompt_count} |\n"
            )
        file.write("\n## Cross-budget efficiency\n\n")
        file.write(
            "| Pairwise B | Unary-FullMass B | Paired difference | "
            "95% CI |\n"
        )
        file.write("| ---------: | ---------------: | ----------------: | ")
        file.write("-----: |\n")
        for row in cross_budget_rows:
            file.write(
                f"| {row['pairwise_budget']} | "
                f"{row['unary_full_budget']} | "
                f"{row['mean_paired_difference']:+.4f} | "
                f"[{row['ci_low']:+.4f}, {row['ci_high']:+.4f}] |\n"
            )
        file.write(
            "\nThe 0.1-token non-inferiority checks are exploratory and "
            "post hoc; see `cross_budget_comparisons.csv`. These comparisons "
            "are between pairwise and unary allocation on the same top-16 "
            "lattice, not against unrestricted DDTree.\n"
        )
        file.write(
            "\n## Direct-observation-censored robustness analysis\n\n"
        )
        file.write(
            "These values cap each round at the last target position "
            "directly observed under a correct verifier prefix. The paired "
            "differences are robustness estimates, not mathematical lower "
            "bounds on the full realized-continuation differences.\n\n"
        )
        file.write(
            "| Budget | Unary-FullMass | Pairwise-MassPreserving | "
            "Paired gain |\n"
        )
        file.write("| -----: | -------------: | ")
        file.write("------------------------: | ----------: |\n")
        for budget in BUDGETS:
            unary_row = next(
                row
                for row in direct_mean_rows
                if row["method"] == UNARY_FULL_MASS
                and row["budget"] == budget
            )
            pairwise_row = next(
                row
                for row in direct_mean_rows
                if row["method"] == PAIRWISE_MASS_PRESERVING
                and row["budget"] == budget
            )
            comparison_row = next(
                row
                for row in direct_comparison_rows
                if row["baseline"] == UNARY_FULL_MASS
                and row["budget"] == budget
            )
            file.write(
                f"| {budget} | "
                f"{unary_row['mean_direct_observation_censored_depth']:.4f} "
                f"| "
                f"{pairwise_row['mean_direct_observation_censored_depth']:.4f} "
                f"| {comparison_row['mean_paired_difference']:+.4f} |\n"
            )
        file.write("\n## Monotonicity\n\n")
        for row in monotonicity_rows:
            file.write(
                f"- {row['method']}: maximum extension log-score "
                f"`{row['maximum_extension_log_score']:.8f}`\n"
            )
        file.write(
            "\nSee the CSV files and plots in this directory for the full "
            "bootstrap, coverage, ranking, failure, and headroom results.\n"
        )

    print(f"Wrote offline analysis to {args.output_dir}")


if __name__ == "__main__":
    main()
