#!/usr/bin/env python3

import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
STEP63_METRICS = ROOT / "analysis/2026-09-05_step63-wide-unary/method_metrics.csv"
STEP63_COMPARISONS = (
    ROOT / "analysis/2026-09-05_step63-wide-unary/pairwise_comparisons.csv"
)
STEP7_COMPARISONS = ROOT / "analysis/2026-09-05_step7-27b/pairwise_comparisons.csv"
STEP7_PROVENANCE = ROOT / "analysis/2026-09-05_step7-27b/provenance.json"
OUTPUT_DIR = Path(__file__).resolve().parent
BUDGETS = (8, 16, 32, 64)
TRANSFER_BUDGETS = (16, 32, 64)
DATASETS = ("GSM8K", "MATH500")
METHODS = ("Unary-K16", "Unary-K32", "Unary-K64", "Pairwise-K16")
COLORS = {
    "Unary-K16": "#9ca3af",
    "Unary-K32": "#6b7280",
    "Unary-K64": "#374151",
    "Pairwise-K16": "#b42318",
    "4B": "#355f8a",
    "27B": "#b42318",
}
MARKERS = {
    "Unary-K16": "o",
    "Unary-K32": "s",
    "Unary-K64": "^",
    "Pairwise-K16": "D",
    "4B": "o",
    "27B": "D",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.6,
            "figure.dpi": 180,
            "savefig.bbox": "tight",
        }
    )


def wider_unary_figure(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    selected = []
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.25), sharey=True)
    for axis, dataset in zip(axes, DATASETS, strict=True):
        for method in METHODS:
            method_rows = sorted(
                (
                    row
                    for row in rows
                    if row["dataset"] == dataset and row["method"] == method
                ),
                key=lambda row: int(row["budget"]),
            )
            values = [float(row["mean_matched_draft_tokens"]) for row in method_rows]
            axis.plot(
                BUDGETS,
                values,
                color=COLORS[method],
                marker=MARKERS[method],
                linewidth=2 if method == "Pairwise-K16" else 1.25,
                markersize=4,
                label=method,
            )
            selected.extend(
                {
                    "dataset": dataset,
                    "budget": int(row["budget"]),
                    "method": method,
                    "mean_matched_draft_tokens": float(
                        row["mean_matched_draft_tokens"]
                    ),
                }
                for row in method_rows
            )
        axis.set_title(dataset)
        axis.set_xticks(BUDGETS)
        axis.set_xlabel("Verification-node budget B")
        axis.set_ylim(4.0, 5.9)
    axes[0].set_ylabel("Mean matched draft tokens")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.subplots_adjust(bottom=0.27, left=0.09, right=0.99, wspace=0.1)
    for suffix in ("pdf", "png"):
        fig.savefig(
            OUTPUT_DIR / f"wider_unary_falsification.{suffix}",
            dpi=300,
        )
    plt.close(fig)
    return selected


def transfer_figure(
    step63_rows: list[dict[str, str]],
    step7_rows: list[dict[str, str]],
) -> list[dict[str, object]]:
    selected = []
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 1.85), sharey=True)
    for axis, dataset in zip(axes, DATASETS, strict=True):
        sources = {
            "4B": [
                row
                for row in step63_rows
                if row["dataset"] == dataset
                and int(row["unary_k"]) == 16
                and int(row["budget"]) in TRANSFER_BUDGETS
            ],
            "27B": [
                row
                for row in step7_rows
                if row["dataset"] == dataset and int(row["budget"]) in TRANSFER_BUDGETS
            ],
        }
        for scale, rows in sources.items():
            rows = sorted(rows, key=lambda row: int(row["budget"]))
            gain_key = (
                "matched_gain" if scale == "4B" else "pairwise_minus_unary_matched"
            )
            values = [float(row[gain_key]) for row in rows]
            low = [float(row["matched_ci_low"]) for row in rows]
            high = [float(row["matched_ci_high"]) for row in rows]
            axis.errorbar(
                TRANSFER_BUDGETS,
                values,
                yerr=[
                    [value - lower for value, lower in zip(values, low)],
                    [upper - value for value, upper in zip(values, high)],
                ],
                color=COLORS[scale],
                marker=MARKERS[scale],
                linewidth=1.7,
                capsize=2.5,
                markersize=4,
                label=scale,
            )
            selected.extend(
                {
                    "dataset": dataset,
                    "model_scale": scale,
                    "budget": int(row["budget"]),
                    "gain": float(row[gain_key]),
                    "ci_low": float(row["matched_ci_low"]),
                    "ci_high": float(row["matched_ci_high"]),
                }
                for row in rows
            )
        axis.axhline(0, color="#111827", linewidth=0.8)
        axis.set_title(dataset)
        axis.set_xticks(TRANSFER_BUDGETS)
        axis.set_xlabel("Verification-node budget B")
    axes[0].set_ylabel("Pairwise - Unary matched tokens")
    axes[1].legend(frameon=False, loc="lower right")
    fig.subplots_adjust(bottom=0.25, left=0.09, right=0.99, wspace=0.1)
    for suffix in ("pdf", "png"):
        fig.savefig(
            OUTPUT_DIR / f"model_scale_transfer.{suffix}",
            dpi=300,
        )
    plt.close(fig)
    return selected


def write_onepager_values(
    step7_rows: list[dict[str, str]],
    step7_provenance: dict,
) -> None:
    rows_by_dataset = {
        dataset: sorted(
            (row for row in step7_rows if row["dataset"] == dataset),
            key=lambda row: int(row["budget"]),
        )
        for dataset in DATASETS
    }
    if any(
        float(row["matched_ci_low"]) <= 0
        for rows in rows_by_dataset.values()
        for row in rows
    ):
        raise ValueError("Step-7 matched-token interval includes zero")
    exact_matches = sum(
        comparison["count"]
        for run in step7_provenance["runs"].values()
        for comparison in run["tree_exact_token_matches"].values()
    )
    exact_total = sum(
        comparison["total"]
        for run in step7_provenance["runs"].values()
        for comparison in run["tree_exact_token_matches"].values()
    )
    if exact_matches != exact_total:
        raise ValueError("Step-7 tree output mismatch found")
    latex_rows = []
    for dataset in DATASETS:
        gains = [
            float(row["pairwise_minus_unary_matched"])
            for row in rows_by_dataset[dataset]
        ]
        latex_rows.append(
            f"{dataset} & " + " & ".join(f"+{gain:.3f}" for gain in gains) + r" \\"
        )
    content = "\n".join(
        [
            r"\newcommand{\StepSevenGainRows}{%",
            *latex_rows,
            "}",
            rf"\newcommand{{\ExactMatchCount}}{{{exact_matches}}}",
            rf"\newcommand{{\ExactMatchTotal}}{{{exact_total}}}",
            "",
        ]
    )
    (OUTPUT_DIR / "onepager_frozen_values.tex").write_text(
        content,
        encoding="utf-8",
    )


def main() -> None:
    configure_style()
    step63_metrics = read_csv(STEP63_METRICS)
    step63_comparisons = read_csv(STEP63_COMPARISONS)
    step7_comparisons = read_csv(STEP7_COMPARISONS)
    step7_provenance = json.loads(STEP7_PROVENANCE.read_text(encoding="utf-8"))
    write_onepager_values(step7_comparisons, step7_provenance)
    figure_data = {
        "sources": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in (
                STEP63_METRICS,
                STEP63_COMPARISONS,
                STEP7_COMPARISONS,
                STEP7_PROVENANCE,
            )
        },
        "wider_unary_falsification": wider_unary_figure(step63_metrics),
        "model_scale_transfer": transfer_figure(
            step63_comparisons,
            step7_comparisons,
        ),
    }
    (OUTPUT_DIR / "figure_data.json").write_text(
        json.dumps(figure_data, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
