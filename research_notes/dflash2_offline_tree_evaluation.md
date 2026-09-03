# Offline DFlash2 UnaryTree vs PairwiseTree evaluation

## Scope

This evaluation uses only the frozen Step-3 trace:

```text
traces/2026-09-03_dflash2_a100-40gb_gsm8k-32/
  gsm8k__Qwen_Qwen3-4B__mgoin_Qwen3-4B-speculator.dflash2__seed0__traces.pt
```

No target-model or drafter inference was performed. The candidate lattice,
unary logits, pairwise scores, prompts, decoding rounds, and realized
continuations are identical for every method.

The exact source, trace, and generated outputs used for this final Step-4
snapshot are committed at
`c16676718c9f5d450af3b5d817b0b9385fd2b503`.

Every evaluated method is restricted to DFlash2's saved top-16 candidate
lattice at each depth. `Unary-FullMass` means that retained candidate scores
use the full-vocabulary unary log-normalizer; it does not mean the tree can
select tokens outside the saved top 16.

## Methods

For unary candidate logit \(U_t(b)\) and full-vocabulary log normalizer
\(\operatorname{LSE}_t\):

```text
Unary-FullMass:
    log q_t(b) = U_t(b) - LSE_t

Unary-Truncated:
    log q_t(b) = log_softmax over the retained top-16 candidates
```

For Pairwise-MassPreserving:

```text
log m_t = logsumexp(top-16 unary logits) - full-vocabulary LSE_t
r_t(b | a) = softmax over the 16 pairwise final scores for predecessor a
log q_t(b | a) = log m_t + log r_t(b | a)
```

`Pairwise-after-root` uses `Unary-FullMass` for the first speculative token,
then the same pairwise mass-preserving transitions at depths 2-7.

`DFlash2-GreedyPath` is the actual seven-token selected path saved in each
trace round. At budget \(B\), it includes only the first \(\min(B, 7)\) nodes
of that path.

Official DDTree can set its per-depth unary top-k to the tree budget. The
controlled `Unary-FullMass` baseline is therefore close to the relevant
official-style unary search for budgets up to 16, but it is candidate-limited
for budgets above 16. For example, at B=256 an unrestricted unary DDTree could
consider unary ranks 17-256, while this evaluation cannot because those token
IDs and logits were not saved in the frozen trace.

Depth 1 uses the saved anchor final scores. Depths 2-7 use:

```text
pairwise_final_scores[
    depth - 2,
    predecessor_candidate_index,
    successor_candidate_index,
]
```

All extension log-probabilities were non-positive. Prefix-score monotonicity
therefore held for every scorer and trace round.

## Generic tree builder

The builder starts with all 16 depth-1 candidates in a max-priority frontier.
When a node is selected, only then are its 16 children added. It returns the
first \(B\) prefixes by cumulative log score.

This guarantees:

- no node appears without every ancestor
- smaller-budget trees are exact prefixes of larger-budget trees
- with monotonic prefix scores, the selected order matches an exhaustive
  sort of all prefixes
- the root is explicit conceptually but is not counted in the speculative
  node budget

## Retained top-16 unary mass

| Depth | Mean | Median | 5th percentile |
| ----: | ---: | -----: | --------------: |
| 1 | 99.63% | 99.99% | 98.44% |
| 2 | 98.65% | 99.90% | 92.86% |
| 3 | 97.25% | 99.66% | 85.63% |
| 4 | 95.58% | 99.14% | 78.15% |
| 5 | 93.48% | 98.15% | 71.66% |
| 6 | 91.00% | 96.30% | 66.18% |
| 7 | 88.44% | 93.49% | 61.81% |

Top-16 retains most unary probability mass on average, although the lower
tail becomes materially weaker at later depths.

## Direct target-transition ranking

Ranks are computed only when the target transition is directly observed
under a correct verifier prefix and the target token is in top-16.

| Depth | Eligible N | In-lattice N | Unary mean rank | Pairwise mean rank | Unary R@1 | Pairwise R@1 |
| ----: | ---------: | -----------: | --------------: | -----------------: | ---------: | ------------: |
| 1 | 1913 | 1909 | 1.200 | 1.221 | 89.42% | 89.21% |
| 2 | 1690 | 1677 | 1.376 | 1.266 | 84.44% | 86.05% |
| 3 | 1434 | 1411 | 1.549 | 1.301 | 81.57% | 85.19% |
| 4 | 1198 | 1173 | 1.475 | 1.256 | 82.35% | 87.47% |
| 5 | 1023 | 1006 | 1.592 | 1.328 | 77.04% | 84.69% |
| 6 | 848 | 828 | 1.617 | 1.310 | 76.09% | 84.18% |
| 7 | 691 | 668 | 1.711 | 1.362 | 76.80% | 85.18% |

The anchor-conditioned selector is slightly worse at depth 1. Pairwise
conditioning improves target-token ranking consistently at depths 2-7, with
the largest Recall@1 gains at later positions.

## Mean matched draft tokens

Primary results use the continuation eventually realized by DFlash2.
Confidence intervals use 10,000 paired bootstrap resamples over the 32
prompts, not individual rounds. The primary point estimate is nevertheless
round-weighted: prompts with more decoding rounds contribute more observations.

| B | Unary-FullMass | Pairwise | Pairwise-after-root | DFlash2 greedy | Oracle | Pairwise - Unary |
| -: | -------------: | -------: | ------------------: | --------------: | -----: | ---------------: |
| 1 | 0.8860 | 0.8855 | 0.8860 | 0.8855 | 0.9979 | -0.0005 |
| 2 | 1.6388 | 1.6513 | 1.6529 | 1.6383 | 1.9712 | +0.0125 |
| 4 | 2.7867 | 2.8562 | 2.8688 | 2.8019 | 3.7522 | +0.0695 |
| 7 | 3.8928 | 4.1119 | 4.1291 | 3.9059 | 5.9665 | +0.2190 |
| 8 | 3.9916 | 4.2603 | 4.2760 | 3.9059 | 5.9665 | +0.2687 |
| 16 | 4.3842 | 4.8186 | 4.8280 | 3.9059 | 5.9665 | +0.4344 |
| 32 | 4.6571 | 5.1432 | 5.1589 | 3.9059 | 5.9665 | +0.4861 |
| 64 | 4.8510 | 5.3978 | 5.4072 | 3.9059 | 5.9665 | +0.5468 |
| 128 | 5.0193 | 5.5682 | 5.5761 | 3.9059 | 5.9665 | +0.5489 |
| 256 | 5.1516 | 5.6848 | 5.6999 | 3.9059 | 5.9665 | +0.5332 |

Selected paired prompt-bootstrap results:

| B | Mean gain | 95% CI | Bootstrap \(P(\Delta>0)\) |
| -: | --------: | ------: | -------------------------: |
| 8 | +0.2687 | [0.2215, 0.3169] | 100% |
| 16 | +0.4344 | [0.3637, 0.5000] | 100% |
| 32 | +0.4861 | [0.4293, 0.5402] | 100% |
| 64 | +0.5468 | [0.4901, 0.6031] | 100% |
| 128 | +0.5489 | [0.4952, 0.6033] | 100% |
| 256 | +0.5332 | [0.4825, 0.5840] | 100% |

Unary-Truncated is nearly identical to Unary-FullMass. Pairwise gains are
therefore not explained by merely renormalizing over top-16.

## Equal-prompt-weighted robustness

An additional estimand first averages the paired Pairwise-minus-Unary effect
within each prompt, then weights all 32 prompts equally:

| B | Equal-prompt mean gain | 95% CI | Positive prompts |
| -: | ---------------------: | -----: | ---------------: |
| 8 | +0.2712 | [0.2199, 0.3231] | 31/32 |
| 16 | +0.4307 | [0.3572, 0.5002] | 31/32 |
| 32 | +0.4805 | [0.4243, 0.5360] | 32/32 |
| 64 | +0.5411 | [0.4844, 0.5989] | 32/32 |
| 128 | +0.5450 | [0.4896, 0.6034] | 32/32 |
| 256 | +0.5342 | [0.4831, 0.5875] | 32/32 |

The main conclusion is unchanged by equal prompt weighting. The full
per-prompt audit is in `per_prompt_paired_effects.csv`.

## Seven-node path versus tree

At the same seven-node cost:

| Method | Mean matched tokens | Difference vs greedy path | 95% CI |
| :----- | ------------------: | ------------------------: | -----: |
| DFlash2 greedy path | 3.9059 | -- | -- |
| Unary-FullMass tree | 3.8928 | -0.0131 | [-0.0638, 0.0387] |
| Pairwise tree | 4.1119 | +0.2060 | [0.1721, 0.2378] |
| Pairwise-after-root | 4.1291 | +0.2232 | [0.1820, 0.2641] |

Branching with unary scores alone does not improve over DFlash2's greedy path
at this budget. Pairwise-scored branching does, separating the value of the
tree structure from the value of predecessor-conditioned allocation.

## Pairwise-after-root ablation

Using unary scoring only at depth 1 removes the small depth-1 regression and
produces small positive point estimates at every budget. Relative to the
original Pairwise scorer, the gain is +0.0157 tokens at B=32
[0.0032, 0.0283] and +0.0094 at B=64 [-0.0050, 0.0238]. The effect is tiny
compared with the main pairwise-versus-unary gain, so this is a minor default
choice rather than a new tuning direction.

## Budget efficiency

| Pairwise budget | Unary-FullMass budget | Paired difference | 95% CI |
| --------------: | --------------------: | ----------------: | -----: |
| 8 | 16 | -0.1239 | [-0.1812, -0.0707] |
| 16 | 64 | -0.0324 | [-0.0931, 0.0255] |
| 32 | 256 | -0.0084 | [-0.0558, 0.0383] |
| 64 | 256 | +0.2462 | [0.2020, 0.2897] |

Pairwise B=8 does not match Unary B=16; it is significantly worse by 0.124
matched tokens. The substantial budget-efficiency advantage begins at
moderate budgets, not at the smallest tested tree sizes.

Pairwise top-16 B=16 achieved nearly the same offline matched depth as Unary
top-16 B=64:
the estimated difference was -0.032 tokens with a 95% CI of
[-0.093, +0.026]. Pairwise top-16 B=32 similarly achieved nearly the same
matched depth as Unary top-16 B=256, with an estimated difference of -0.008
tokens and a 95% CI of [-0.056, +0.038].

These intervals containing zero do not establish equivalence. Under an
exploratory, post-hoc non-inferiority margin of 0.1 matched tokens, both
comparisons pass because their lower confidence bounds are above -0.1. This
margin was not pre-registered and should be validated on a holdout.

## Full seven-token prefix coverage

Among the 1,887 rounds with a seven-token realized continuation:

| B | Unary-FullMass | Pairwise | Oracle |
| -: | -------------: | -------: | -----: |
| 8 | 24.43% | 28.46% | 68.15% |
| 16 | 31.11% | 39.90% | 68.15% |
| 32 | 36.41% | 46.95% | 68.15% |
| 64 | 40.01% | 53.10% | 68.15% |
| 128 | 43.40% | 56.54% | 68.15% |
| 256 | 46.26% | 58.88% | 68.15% |

## Oracle headroom

The oracle knows the realized target path and spends only the nodes needed to
include that path. It is not deployable.

Pairwise closes this fraction of the Unary-FullMass-to-oracle gap:

| B | Headroom closed |
| -: | --------------: |
| 8 | 13.61% |
| 16 | 27.45% |
| 32 | 37.13% |
| 64 | 49.02% |
| 128 | 57.95% |
| 256 | 65.43% |

## Failure decomposition

At depth 7, candidate failures are identical because all methods use the same
lattice. Pairwise changes only ranking/tree-budget failures.

| B | Method | Candidate failure | Ranking/budget failure | Covered |
| -: | ------ | ----------------: | ---------------------: | ------: |
| 32 | Unary-FullMass | 31.85% | 31.74% | 36.41% |
| 32 | Pairwise | 31.85% | 21.20% | 46.95% |
| 64 | Unary-FullMass | 31.85% | 28.14% | 40.01% |
| 64 | Pairwise | 31.85% | 15.05% | 53.10% |

For the directly observed depth-7 subset, \(N=691\):

| B | Method | Candidate failure | Ranking/budget failure | Covered |
| -: | ------ | ----------------: | ---------------------: | ------: |
| 32 | Unary-FullMass | 3.33% | 16.21% | 80.46% |
| 32 | Pairwise | 3.33% | 4.34% | 92.33% |
| 64 | Unary-FullMass | 3.33% | 12.30% | 84.37% |
| 64 | Pairwise | 3.33% | 1.59% | 95.08% |

## Direct-observation-censored robustness analysis

When each round is capped at the last target position directly observed under
a correct verifier prefix, pairwise gains remain positive:

| B | Mean paired gain | 95% CI |
| -: | ---------------: | -----: |
| 8 | +0.2561 | [0.2237, 0.2906] |
| 16 | +0.2426 | [0.2127, 0.2715] |
| 32 | +0.2018 | [0.1736, 0.2306] |
| 64 | +0.1730 | [0.1521, 0.1931] |
| 128 | +0.1406 | [0.1190, 0.1619] |
| 256 | +0.1192 | [0.1006, 0.1384] |

Each method's censored matched depth is a lower bound on its own full matched
depth. The difference between two censored values is not necessarily a lower
bound on their full difference. These results should therefore be interpreted
as a robustness analysis showing that pairwise gains remain positive when
restricted to target positions directly justified by the verifier trace.

## Recommendation

**GO** to a reviewed online Pairwise-DDTree prototype.

The pairwise selector contains useful information for fixed-budget tree
allocation:

- same-budget gains are approximately +0.49 to +0.55 matched draft tokens at
  B=32-128
- Pairwise top-16 B=16 approximately matches Unary top-16 B=64
- Pairwise top-16 B=32 approximately matches Unary top-16 B=256
- every prompt has a positive mean paired effect at B=32 and B=64
- direct target-transition ranking improves consistently at depths 2-7
- gains persist under the directly observed, censored analysis
- top-16 renormalization alone does not reproduce the improvement

This is a go/no-go result, not an online speedup claim. The next implementation
must still validate exact target acceptance and BF16 behavior with the actual
tree-verification tensor shape.

The recommendation applies to moderate budgets. Pairwise B=8 does not match
Unary B=16, so the offline evidence does not support a universal
budget-reduction claim at very small tree sizes.

It also does not establish that PairwiseTree needs 4x or 8x fewer nodes than
unrestricted DDTree. At budgets above 16, an unrestricted unary DDTree could
recover target tokens outside DFlash2's top-16 lattice. The defensible claim
is narrower: on the same DFlash2 top-16 candidate lattice, pairwise scoring
allocates the fixed node budget better than unary scoring.

## Remaining concerns

- The result is based on 32 prompt clusters, so the bootstrap intervals should
  not be generalized beyond this checkpoint, dataset, and trace.
- `Unary-FullMass` is full-mass-normalized but top-16-candidate-limited. It is
  not an unrestricted official DDTree baseline for budgets above 16.
- This is an off-policy evaluation of local tree quality. All 1,913 states
  were visited by DFlash2's existing greedy decoding policy. An online
  Pairwise-DDTree policy would accept different lengths, begin later rounds at
  different anchors, and therefore induce a different distribution of
  decoding states. Offline expectations over these rounds need not equal the
  online Pairwise-DDTree expectation.
- Realized-continuation metrics are retrospective after mismatches.
- Direct-observation metrics are progressively selected and censored.
- Entry-budget ranks above 256 are right-censored; the report marks affected
  quantiles as `>256`.
- Pairwise scores are selector logits converted into a mass-preserving
  conditional distribution. This objective is principled but was not the
  checkpoint's training loss.
- Offline ranking cannot predict BF16 target argmax changes under a different
  online tree-verification matrix shape.
