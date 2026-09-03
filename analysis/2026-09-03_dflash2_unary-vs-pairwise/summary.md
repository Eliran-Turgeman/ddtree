# Offline DFlash2 UnaryTree vs PairwiseTree

All results use the frozen Step-3 trace and make no model inference calls.

Source snapshot: `c16676718c9f5d450af3b5d817b0b9385fd2b503`.

Every method is restricted to the saved DFlash2 top-16 candidate lattice. `Unary-FullMass` uses the full-vocabulary log-normalizer but cannot select tokens outside that top 16; it is not unrestricted DDTree for budgets above 16.

## Mean matched draft tokens

| Budget | Unary-FullMass | Unary-Truncated | Pairwise | Pairwise-after-root | DFlash2 greedy | Oracle |
| -----: | -------------: | --------------: | -------: | ------------------: | --------------: | -----: |
| 1 | 0.8860 | 0.8860 | 0.8855 | 0.8860 | 0.8855 | 0.9979 |
| 2 | 1.6388 | 1.6404 | 1.6513 | 1.6529 | 1.6383 | 1.9712 |
| 4 | 2.7867 | 2.7841 | 2.8562 | 2.8688 | 2.8019 | 3.7522 |
| 7 | 3.8928 | 3.8860 | 4.1119 | 4.1291 | 3.9059 | 5.9665 |
| 8 | 3.9916 | 3.9822 | 4.2603 | 4.2760 | 3.9059 | 5.9665 |
| 16 | 4.3842 | 4.3837 | 4.8186 | 4.8280 | 3.9059 | 5.9665 |
| 32 | 4.6571 | 4.6487 | 5.1432 | 5.1589 | 3.9059 | 5.9665 |
| 64 | 4.8510 | 4.8453 | 5.3978 | 5.4072 | 3.9059 | 5.9665 |
| 128 | 5.0193 | 5.0099 | 5.5682 | 5.5761 | 3.9059 | 5.9665 |
| 256 | 5.1516 | 5.1396 | 5.6848 | 5.6999 | 3.9059 | 5.9665 |

## Seven-node path/tree comparison

| Method | Mean | Difference vs greedy path | 95% CI |
| :----- | ---: | ------------------------: | -----: |
| DFlash2 greedy path | 3.9059 | -- | -- |
| Unary-FullMass | 3.8928 | -0.0131 | [-0.0638, +0.0387] |
| Pairwise-MassPreserving | 4.1119 | +0.2060 | [+0.1721, +0.2378] |
| Pairwise-after-root | 4.1291 | +0.2232 | [+0.1820, +0.2641] |

## Pairwise-after-root ablation

This ablation uses Unary-FullMass at depth 1 and pairwise mass-preserving transitions from depth 2 onward.

| Budget | Difference vs Pairwise | 95% CI |
| -----: | ---------------------: | -----: |
| 1 | +0.0005 | [-0.0070, +0.0080] |
| 2 | +0.0016 | [-0.0106, +0.0141] |
| 4 | +0.0125 | [-0.0010, +0.0264] |
| 7 | +0.0173 | [+0.0000, +0.0342] |
| 8 | +0.0157 | [-0.0022, +0.0330] |
| 16 | +0.0094 | [-0.0017, +0.0206] |
| 32 | +0.0157 | [+0.0032, +0.0283] |
| 64 | +0.0094 | [-0.0050, +0.0238] |
| 128 | +0.0078 | [-0.0030, +0.0203] |
| 256 | +0.0152 | [+0.0054, +0.0251] |

## Pairwise improvement counts

| Budget | Helps | Ties | Hurts |
| -----: | ----: | ---: | ----: |
| 32 | 574 | 1215 | 124 |
| 64 | 602 | 1227 | 84 |

## Equal-prompt-weighted pairwise gain

The primary mean above is round-weighted. This robustness estimand first averages the paired effect within each prompt, then weights all 32 prompts equally.

| Budget | Equal-prompt mean gain | 95% CI | Positive prompts |
| -----: | ---------------------: | -----: | ---------------: |
| 1 | +0.0010 | [-0.0071, +0.0093] | 11/32 |
| 2 | +0.0136 | [-0.0014, +0.0289] | 16/32 |
| 4 | +0.0701 | [+0.0429, +0.1002] | 24/32 |
| 7 | +0.2204 | [+0.1638, +0.2748] | 29/32 |
| 8 | +0.2712 | [+0.2199, +0.3231] | 31/32 |
| 16 | +0.4307 | [+0.3572, +0.5002] | 31/32 |
| 32 | +0.4805 | [+0.4243, +0.5360] | 32/32 |
| 64 | +0.5411 | [+0.4844, +0.5989] | 32/32 |
| 128 | +0.5450 | [+0.4896, +0.6034] | 32/32 |
| 256 | +0.5342 | [+0.4831, +0.5875] | 32/32 |

## Cross-budget efficiency

| Pairwise B | Unary-FullMass B | Paired difference | 95% CI |
| ---------: | ---------------: | ----------------: | -----: |
| 8 | 16 | -0.1239 | [-0.1812, -0.0707] |
| 16 | 64 | -0.0324 | [-0.0931, +0.0255] |
| 32 | 256 | -0.0084 | [-0.0558, +0.0383] |
| 64 | 256 | +0.2462 | [+0.2020, +0.2897] |

The 0.1-token non-inferiority checks are exploratory and post hoc; see `cross_budget_comparisons.csv`. These comparisons are between pairwise and unary allocation on the same top-16 lattice, not against unrestricted DDTree.

## Direct-observation-censored robustness analysis

These values cap each round at the last target position directly observed under a correct verifier prefix. The paired differences are robustness estimates, not mathematical lower bounds on the full realized-continuation differences.

| Budget | Unary-FullMass | Pairwise-MassPreserving | Paired gain |
| -----: | -------------: | ------------------------: | ----------: |
| 1 | 0.8860 | 0.8855 | -0.0005 |
| 2 | 1.6289 | 1.6513 | +0.0225 |
| 4 | 2.7392 | 2.8359 | +0.0967 |
| 7 | 3.7637 | 3.9963 | +0.2326 |
| 8 | 3.8463 | 4.1025 | +0.2561 |
| 16 | 4.0768 | 4.3194 | +0.2426 |
| 32 | 4.2143 | 4.4161 | +0.2018 |
| 64 | 4.3027 | 4.4757 | +0.1730 |
| 128 | 4.3670 | 4.5076 | +0.1406 |
| 256 | 4.4077 | 4.5269 | +0.1192 |

## Monotonicity

- Unary-FullMass: maximum extension log-score `0.00000000`
- Unary-Truncated: maximum extension log-score `0.00000000`
- Pairwise-MassPreserving: maximum extension log-score `0.00000000`
- Pairwise-after-root: maximum extension log-score `0.00000000`

See the CSV files and plots in this directory for the full bootstrap, coverage, ranking, failure, and headroom results.
