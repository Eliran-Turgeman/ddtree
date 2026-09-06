# Reproducing the Pairwise-DDTree research communication

No model inference or benchmark execution is required. The figures and
one-page handout read only committed frozen analysis artifacts.

From the repository root:

```powershell
python research_notes\figures\generate_pairwise_ddtree_figures.py
Set-Location research_notes
latexmk -pdf -interaction=nonstopmode -halt-on-error pairwise_ddtree_onepager.tex
Set-Location ..
```

The PDF is written to:

```text
research_notes/pairwise_ddtree_onepager.pdf
```

Verify that it has exactly one page:

```powershell
pdfinfo research_notes\pairwise_ddtree_onepager.pdf |
  Select-String '^Pages:'
```

## Figure provenance

`research_notes/figures/figure_data.json` records the exact rows plotted and
SHA-256 hashes of every source table.

| Output | Frozen source artifacts |
|---|---|
| `wider_unary_falsification.{pdf,png}` | `analysis/2026-09-05_step63-wide-unary/method_metrics.csv` |
| `model_scale_transfer.{pdf,png}` | `analysis/2026-09-05_step63-wide-unary/pairwise_comparisons.csv`; `analysis/2026-09-05_step7-27b/pairwise_comparisons.csv` |
| `onepager_frozen_values.tex` | `analysis/2026-09-05_step7-27b/pairwise_comparisons.csv`; `analysis/2026-09-05_step7-27b/provenance.json` |

The 4B cross-domain statements are checked against:

```text
analysis/2026-09-04_step62-final/cross_domain_metrics.csv
research_notes/dflash2_step62_cross_domain_validation.md
```

The detailed mathematical and attribution audit is:

```text
research_notes/pairwise_ddtree_formalization.md
```
