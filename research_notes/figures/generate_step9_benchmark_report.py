#!/usr/bin/env python3

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
ANALYSIS = ROOT / "analysis" / "step9-4b-throughput"
FIGURES = ROOT / "research_notes" / "figures"
TABLES = FIGURES / "step9_benchmark_tables.tex"
VALUES = FIGURES / "step9_benchmark_values.tex"

DATASETS = [
    "GSM8K",
    "MATH500",
    "HumanEval",
    "MT-Bench-controlled",
    "MT-Bench-native",
]
DATASET_LABELS = {
    "GSM8K": "GSM8K",
    "MATH500": "MATH500",
    "HumanEval": "HumanEval",
    "MT-Bench-controlled": "MT-Bench\ncontrolled",
    "MT-Bench-native": "MT-Bench\nnative",
}

PAIRWISE = "#b42318"
UNARY = "#374151"
ORIGINAL = "#355f8a"
RAW = "#6b7280"
TARGET = "#9ca3af"
FIXED_UNARY = "#d97706"


def configure_plotting() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "figure.dpi": 160,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.alpha": 0.22,
        }
    )


def save_figure(fig: plt.Figure, stem: str) -> None:
    fig.savefig(
        FIGURES / f"{stem}.pdf",
        bbox_inches="tight",
        metadata={"CreationDate": None, "ModDate": None},
    )
    fig.savefig(FIGURES / f"{stem}.png", bbox_inches="tight")
    plt.close(fig)


def load_sources() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    required = {
        "method_metrics.csv",
        "paired_throughput_comparisons.csv",
        "timing_decomposition.csv",
        "correctness.csv",
    }
    missing = sorted(str(ANALYSIS / name) for name in required if not (ANALYSIS / name).is_file())
    if missing:
        raise FileNotFoundError("Missing frozen Step 9 source files:\n" + "\n".join(missing))

    metrics = pd.read_csv(ANALYSIS / "method_metrics.csv")
    comparisons = pd.read_csv(ANALYSIS / "paired_throughput_comparisons.csv")
    timing = pd.read_csv(ANALYSIS / "timing_decomposition.csv")
    correctness = pd.read_csv(ANALYSIS / "correctness.csv")
    return metrics, comparisons, timing, correctness


def primary_rows(comparisons: pd.DataFrame) -> pd.DataFrame:
    rows = comparisons[
        (comparisons["comparison_type"] == "controlled")
        & (comparisons["left_method"] == "DFlash2-Pairwise-K16-B16")
        & (comparisons["right_method"] == "DFlash2-Original-DDTree-B16")
    ].copy()
    rows["dataset"] = pd.Categorical(rows["dataset"], DATASETS, ordered=True)
    return rows.sort_values("dataset")


def plot_primary_gain(comparisons: pd.DataFrame) -> None:
    rows = primary_rows(comparisons)
    mean = rows["mean_tokens_per_second_diff"].to_numpy()
    low = rows["tokens_per_second_diff_ci_low"].to_numpy()
    high = rows["tokens_per_second_diff_ci_high"].to_numpy()
    relative = rows["mean_relative_pct_diff"].to_numpy()
    labels = [DATASET_LABELS[str(value)].replace("\n", " ") for value in rows["dataset"]]

    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    y = np.arange(len(rows))
    ax.errorbar(
        mean,
        y,
        xerr=np.vstack((mean - low, high - mean)),
        fmt="o",
        color=PAIRWISE,
        ecolor=PAIRWISE,
        capsize=3,
        linewidth=1.8,
        markersize=6,
    )
    ax.axvline(0, color="#111827", linewidth=0.8)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Pairwise-K16 minus unary DDTree end-to-end throughput (tokens/s)")
    ax.set_title("Controlled B=16 throughput gain (paired 95% bootstrap CI)")
    ax.grid(axis="y", visible=False)
    for index, (value, pct) in enumerate(zip(mean, relative, strict=True)):
        ax.text(high[index] + 0.25, index, f"+{pct:.2f}%", va="center", fontsize=8)
    ax.set_xlim(min(0, low.min() - 1), high.max() + 3)
    fig.tight_layout()
    save_figure(fig, "step9_primary_gain")


def select_metric(
    metrics: pd.DataFrame, dataset: str, family: str, method_key: str, column: str
) -> float:
    row = metrics[
        (metrics["dataset"] == dataset)
        & (metrics["family"] == family)
        & (metrics["method_key"] == method_key)
    ]
    if len(row) != 1:
        raise ValueError(f"Expected one row for {dataset}/{family}/{method_key}, got {len(row)}")
    return float(row.iloc[0][column])


def plot_system_throughput(metrics: pd.DataFrame) -> None:
    series = [
        ("Sequential target", "dflash2", "baseline", TARGET),
        ("Raw original DFlash", "original", "dflash", RAW),
        ("Original DFlash + DDTree B16", "original", "ddtree_tb16", ORIGINAL),
        ("Raw DFlash2", "dflash2", "dflash2", "#8b5cf6"),
        ("DFlash2 + unary DDTree B16", "dflash2", "dflash2_original_ddtree_tb16", UNARY),
        ("DFlash2 + Pairwise-K16 B16", "dflash2", "dflash2_pairwise_k16_tb16", PAIRWISE),
    ]
    x = np.arange(len(DATASETS))
    width = 0.13
    fig, ax = plt.subplots(figsize=(10.2, 4.2))
    for index, (label, family, method, color) in enumerate(series):
        values = [
            select_metric(
                metrics,
                dataset,
                family,
                method,
                "mean_end_to_end_tokens_per_second",
            )
            for dataset in DATASETS
        ]
        ax.bar(x + (index - 2.5) * width, values, width, label=label, color=color)
    ax.set_xticks(x, [DATASET_LABELS[name] for name in DATASETS])
    ax.set_ylabel("End-to-end output tokens/s")
    ax.set_title("Absolute system throughput at B=16")
    ax.legend(ncol=3, frameon=False, loc="upper right")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    save_figure(fig, "step9_system_throughput")


def budget_values(
    metrics: pd.DataFrame, dataset: str, prefix: str, column: str
) -> tuple[list[int], list[float]]:
    budgets: list[int] = []
    values: list[float] = []
    for budget in (7, 16, 32, 64):
        key = f"{prefix}{budget}"
        row = metrics[
            (metrics["dataset"] == dataset)
            & (metrics["family"] == "dflash2")
            & (metrics["method_key"] == key)
        ]
        if len(row) == 1:
            budgets.append(budget)
            values.append(float(row.iloc[0][column]))
    return budgets, values


def plot_budget_panels(metrics: pd.DataFrame, column: str, stem: str, ylabel: str) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(10.4, 6.0), sharex=True)
    for ax, dataset in zip(axes.flat, DATASETS, strict=False):
        for prefix, label, color, marker, linestyle in [
            ("dflash2_original_ddtree_tb", "Unary DDTree", UNARY, "o", "-"),
            ("dflash2_pairwise_k16_tb", "Pairwise-K16", PAIRWISE, "o", "-"),
            ("dflash2_unary_k16_tb", "Fixed Unary-K16", FIXED_UNARY, "s", "--"),
        ]:
            budgets, values = budget_values(metrics, dataset, prefix, column)
            ax.plot(
                budgets,
                values,
                label=label,
                color=color,
                marker=marker,
                linestyle=linestyle,
                linewidth=1.8,
                markersize=4.5,
            )
        if column == "mean_end_to_end_tokens_per_second":
            raw = select_metric(metrics, dataset, "dflash2", "dflash2", column)
            ax.axhline(raw, color=RAW, linestyle=":", linewidth=1.3, label="Raw DFlash2")
        ax.set_title(DATASET_LABELS[dataset].replace("\n", " "))
        ax.set_xticks([7, 16, 32, 64])
        ax.grid(axis="x", visible=False)
    axes[1, 2].axis("off")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    axes[1, 2].legend(handles, labels, loc="center", frameon=False)
    fig.supxlabel("Verification-node budget B")
    fig.supylabel(ylabel)
    fig.tight_layout()
    save_figure(fig, stem)


def plot_exact_output(correctness: pd.DataFrame) -> None:
    methods = [
        ("DFlash2 unary DDTree B16", "dflash2_original_ddtree_tb16", UNARY),
        ("DFlash2 Pairwise-K16 B16", "dflash2_pairwise_k16_tb16", PAIRWISE),
    ]
    x = np.arange(len(DATASETS))
    width = 0.34
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    for index, (label, key, color) in enumerate(methods):
        values = []
        for dataset in DATASETS:
            row = correctness[
                (correctness["dataset"] == dataset)
                & (correctness["family"] == "dflash2")
                & (correctness["method_key"] == key)
            ]
            values.append(100 * float(row.iloc[0]["exact_output_rate"]))
        ax.bar(x + (index - 0.5) * width, values, width, label=label, color=color)
    ax.set_xticks(x, [DATASET_LABELS[name] for name in DATASETS])
    ax.set_ylim(0, 100)
    ax.set_ylabel("Full-output exact match vs sequential target (%)")
    ax.set_title("BF16 numerical exactness at B=16")
    ax.legend(frameon=False)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    save_figure(fig, "step9_exact_output")


def load_quality() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for family in ("original", "dflash2"):
        for task in ("gsm8k", "math500", "humaneval"):
            path = ANALYSIS / "quality" / family / task / f"{task}_summary.csv"
            frame = pd.read_csv(path)
            frame["family"] = family
            frame["task"] = task
            score_column = "accuracy" if "accuracy" in frame.columns else "pass_at_1"
            frame["score"] = frame[score_column]
            frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def quality_score(quality: pd.DataFrame, task: str, method_key: str) -> float:
    rows = quality[(quality["task"] == task) & (quality["method_key"] == method_key)]
    if method_key == "baseline":
        rows = rows[rows["family"] == "dflash2"]
    if len(rows) != 1:
        raise ValueError(f"Expected one quality row for {task}/{method_key}, got {len(rows)}")
    return 100 * float(rows.iloc[0]["score"])


def plot_quality(quality: pd.DataFrame) -> None:
    tasks = ["gsm8k", "math500", "humaneval"]
    task_labels = ["GSM8K accuracy", "MATH500 accuracy", "HumanEval pass@1"]
    series = [
        ("Sequential target", "baseline", TARGET),
        ("DFlash2 unary DDTree B16", "dflash2_original_ddtree_tb16", UNARY),
        ("DFlash2 Pairwise-K16 B16", "dflash2_pairwise_k16_tb16", PAIRWISE),
    ]
    x = np.arange(len(tasks))
    width = 0.24
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    for index, (label, method, color) in enumerate(series):
        values = [quality_score(quality, task, method) for task in tasks]
        bars = ax.bar(x + (index - 1) * width, values, width, label=label, color=color)
        ax.bar_label(bars, fmt="%.1f", fontsize=7, padding=2)
    ax.set_xticks(x, task_labels)
    ax.set_ylabel("Task score (%)")
    ax.set_ylim(0, 65)
    ax.set_title("Task quality at B=16 (descriptive, not an equivalence test)")
    ax.legend(frameon=False, ncol=3, loc="upper center")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    save_figure(fig, "step9_task_quality")


def tex_escape(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "--"
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(char, char) for char in text)


def fmt(value: object, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "--"
    return f"{float(value):.{digits}f}"


def longtable(
    columns: str,
    headers: list[str],
    rows: list[list[str]],
    caption: str,
    label: str,
) -> str:
    header = " & ".join(headers) + r" \\"
    body = "\n".join(" & ".join(row) + r" \\" for row in rows)
    return rf"""
\begin{{longtable}}{{{columns}}}
\caption{{{caption}}}\label{{{label}}}\\
\toprule
{header}
\midrule
\endfirsthead
\toprule
{header}
\midrule
\endhead
\midrule
\multicolumn{{{len(headers)}}}{{r}}{{Continued on next page}}\\
\endfoot
\bottomrule
\endlastfoot
{body}
\end{{longtable}}
"""


def generate_tables(
    metrics: pd.DataFrame,
    comparisons: pd.DataFrame,
    timing: pd.DataFrame,
    correctness: pd.DataFrame,
    quality: pd.DataFrame,
) -> None:
    merged = metrics.merge(
        correctness[["dataset", "family", "method_key", "exact_output_rate", "truncated_rate"]],
        on=["dataset", "family", "method_key"],
        how="left",
        validate="one_to_one",
    )
    dataset_rank = {name: index for index, name in enumerate(DATASETS)}
    merged["_dataset_rank"] = merged["dataset"].map(dataset_rank)
    merged = merged.sort_values(["_dataset_rank", "family", "method_key"])

    throughput_rows = [
        [
            tex_escape(row.dataset),
            tex_escape(row.family),
            tex_escape(row.method),
            fmt(row.budget, 0),
            fmt(row.mean_end_to_end_tokens_per_second),
            fmt(row.mean_decode_only_tokens_per_second),
            fmt(row.speedup_vs_baseline_end_to_end),
            fmt(row.mean_target_calls, 1),
            fmt(row.mean_ttft_ms),
            fmt(row.peak_allocated_gib),
        ]
        for row in merged.itertuples()
    ]
    acceptance_rows = [
        [
            tex_escape(row.dataset),
            tex_escape(row.family),
            tex_escape(row.method),
            fmt(row.budget, 0),
            fmt(row.mean_matched_tokens_per_round, 3),
            fmt(row.mean_committed_tokens_per_round, 3),
            fmt(100 * row.exact_output_rate, 1),
            fmt(100 * row.truncated_rate, 1),
        ]
        for row in merged.itertuples()
    ]

    comparison_rows = []
    for row in comparisons.itertuples():
        ci = (
            f"{fmt(row.mean_tokens_per_second_diff)} "
            f"[{fmt(row.tokens_per_second_diff_ci_low)}, "
            f"{fmt(row.tokens_per_second_diff_ci_high)}]"
        )
        normalized = (
            "--"
            if pd.isna(row.mean_baseline_normalized_speedup_diff)
            else (
                f"{fmt(row.mean_baseline_normalized_speedup_diff, 3)} "
                f"[{fmt(row.baseline_normalized_speedup_diff_ci_low, 3)}, "
                f"{fmt(row.baseline_normalized_speedup_diff_ci_high, 3)}]"
            )
        )
        comparison_rows.append(
            [
                tex_escape(row.dataset),
                tex_escape(row.comparison_type),
                tex_escape(row.left_method),
                tex_escape(row.right_method),
                fmt(row.left_budget, 0),
                ci,
                fmt(row.mean_relative_pct_diff),
                normalized,
                f"{int(row.improve_prompts)}/{int(row.tie_prompts)}/{int(row.hurt_prompts)}",
            ]
        )

    timing_rows = [
        [
            tex_escape(row.dataset),
            tex_escape(row.left_method),
            tex_escape(row.right_method),
            fmt(row.matched_tokens_gain, 3),
            fmt(row.committed_tokens_gain, 3),
            fmt(row.target_calls_avoided, 2),
            fmt(row.draft_stage_ms_per_call_left, 3),
            fmt(row.draft_stage_ms_per_call_right, 3),
            fmt(row.tree_overhead_ms_per_call, 3),
            fmt(row.verify_stage_ms_per_call_left, 3),
            fmt(row.verify_stage_ms_per_call_right, 3),
            fmt(row.net_tokens_per_second_gain),
        ]
        for row in timing.itertuples()
    ]

    quality = quality.copy()
    task_rank = {"gsm8k": 0, "math500": 1, "humaneval": 2}
    quality["_task_rank"] = quality["task"].map(task_rank)
    quality = quality.sort_values(["_task_rank", "family", "method_key"])
    quality_rows = [
        [
            tex_escape(row.task),
            tex_escape(row.family),
            tex_escape(row.method),
            fmt(row.budget, 0),
            str(int(row.evaluated)),
            str(int(row.correct if hasattr(row, "correct") and not pd.isna(row.correct) else row.passed)),
            fmt(100 * row.score, 2),
        ]
        for row in quality.itertuples()
    ]

    primary = primary_rows(comparisons)
    primary_rows_tex = [
        [
            tex_escape(str(row.dataset)),
            fmt(row.mean_tokens_per_second_diff),
            f"[{fmt(row.tokens_per_second_diff_ci_low)}, {fmt(row.tokens_per_second_diff_ci_high)}]",
            fmt(row.mean_relative_pct_diff) + r"\%",
            f"{int(row.improve_prompts)}/{int(row.tie_prompts)}/{int(row.hurt_prompts)}",
        ]
        for row in primary.itertuples()
    ]

    value_lines = [
        r"\newcommand{\StepNineMinGain}{"
        + fmt(primary["mean_relative_pct_diff"].min())
        + r"\%}",
        r"\newcommand{\StepNineMaxGain}{"
        + fmt(primary["mean_relative_pct_diff"].max())
        + r"\%}",
        r"\newcommand{\StepNinePrimaryTable}{%",
        r"\begin{tabular}{lrrrr}",
        r"\toprule Dataset & $\Delta$ tok/s & 95\% CI & Relative & I/T/H \\",
        r"\midrule",
        *[" & ".join(row) + r" \\" for row in primary_rows_tex],
        r"\bottomrule",
        r"\end{tabular}}",
    ]
    VALUES.write_text("\n".join(value_lines) + "\n", encoding="utf-8", newline="\n")

    sections = [
        longtable(
            "lllrrrrrrr",
            [
                "Dataset",
                "Family",
                "Method",
                "B",
                "E2E tok/s",
                "Decode tok/s",
                "Speedup",
                "Target calls",
                "TTFT ms",
                "Peak GiB",
            ],
            throughput_rows,
            "Complete per-method throughput and resource metrics.",
            "tab:all-throughput",
        ),
        longtable(
            "lllrrrrr",
            [
                "Dataset",
                "Family",
                "Method",
                "B",
                "Matched/round",
                "Committed/round",
                r"Exact \%",
                r"Truncated \%",
            ],
            acceptance_rows,
            "Complete acceptance and exact-output metrics.",
            "tab:all-acceptance",
        ),
        longtable(
            "llllrrrrr",
            [
                "Dataset",
                "Type",
                "Left",
                "Right",
                "B",
                r"$\Delta$ tok/s [95\% CI]",
                r"$\Delta$\%",
                "Baseline-normalized [CI]",
                "I/T/H",
            ],
            comparison_rows,
            "All paired throughput comparisons. I/T/H denotes improved, tied, and hurt prompt clusters.",
            "tab:all-comparisons",
        ),
        longtable(
            "lllrrrrrrrrr",
            [
                "Dataset",
                "Left",
                "Right",
                r"$\Delta$ match",
                r"$\Delta$ commit",
                "Calls saved",
                "Draft L",
                "Draft R",
                "Tree",
                "Verify L",
                "Verify R",
                r"$\Delta$ tok/s",
            ],
            timing_rows,
            "All timing decompositions. Stage values are milliseconds per call.",
            "tab:all-timing",
        ),
        longtable(
            "lllrrrr",
            ["Task", "Family", "Method", "B", "Evaluated", "Passed", r"Score \%"],
            quality_rows,
            "All directly scored task-quality results.",
            "tab:all-quality",
        ),
    ]
    TABLES.write_text("\n".join(sections), encoding="utf-8", newline="\n")


def main() -> None:
    configure_plotting()
    metrics, comparisons, timing, correctness = load_sources()
    quality = load_quality()
    plot_primary_gain(comparisons)
    plot_system_throughput(metrics)
    plot_budget_panels(
        metrics,
        "mean_end_to_end_tokens_per_second",
        "step9_budget_throughput",
        "End-to-end output tokens/s",
    )
    plot_budget_panels(
        metrics,
        "mean_matched_tokens_per_round",
        "step9_budget_acceptance",
        "Mean matched draft tokens per round",
    )
    plot_exact_output(correctness)
    plot_quality(quality)
    generate_tables(metrics, comparisons, timing, correctness, quality)
    print(
        "Generated Step 9 figures, "
        f"{VALUES.relative_to(ROOT)}, and {TABLES.relative_to(ROOT)}"
    )


if __name__ == "__main__":
    main()
