# Step 6.3: Pairwise-K16 versus wider Unary DDTree

## Frozen question and setup

This experiment tests whether Pairwise-K16 only beats Unary-K16 because the
unary baseline is restricted to too few candidates.

- Target: `Qwen/Qwen3-4B`
- Target/tokenizer revision:
  `1cfa9a7208912126459214e8b04321603b3df60c`
- Drafter: `mgoin/Qwen3-4B-speculator.dflash2`
- Drafter revision: `e3e7a18e4f541fa3841c2fb0666a7759079ab6fd`
- GPU: one NVIDIA A100-SXM4 40GB
- Temperature: 0
- Verification-node budgets: 8, 16, 32, 64
- Datasets: frozen GSM8K-128 and MATH500-128 subsets
- Frozen benchmark implementation:
  `88774714e931f57d8cc974160352e2b8a587051b`
- Frozen analysis implementation:
  `95f0f30e7b6e2ff5e845e2c1062cb5bb034c01c1`
- Bootstrap: 10,000 paired prompt-level resamples

### Dataset subsets

Both datasets use the test split, shuffled with Hugging Face Datasets
`shuffle(seed=0)`, followed by `select(range(128))`. These are the same frozen
GSM8K-128 and MATH500-128 subsets used in Steps 6.1 and 6.2.

| Dataset | Repository/config/split | Revision | Selected-field SHA-256 |
|---|---|---|---|
| GSM8K | `openai/gsm8k`, `main`, `test` | `740312add88f781978c0658806c59bc2815b9866` | `3c91366d92f64bdf755e8d258f15e0c83f2aa57e15eee1899cadb7c0fff04aa0` |
| MATH500 | `HuggingFaceH4/MATH-500`, `test` | `6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be` | `09cd9c03cd241afad62401ef84a48e5e61ebc243a783aebfdd26721e7a273c05` |

The hashes cover the selected problem text in order. The revisions and order
were independently reconstructed after the run and verified by tokenizing
all 128 prompts with the frozen tokenizer revision and matching every saved
input-token prefix in the raw run artifacts.

### Hardware and software

| Component | Frozen value |
|---|---|
| GPU | NVIDIA A100-SXM4 40GB, one GPU |
| Python | 3.10.12 |
| PyTorch | 2.7.0 |
| CUDA reported by PyTorch | 12.8 |
| Transformers | 5.16.1 |
| Attention implementation | SDPA |
| Cache compaction | Existing C++ tail-cache extension enabled |
| Maximum new tokens | 2048 |

The raw artifacts record the implementation commit as clean (`dirty: false`)
and contain these runtime and model-revision fields.

Pairwise-K16 is unchanged from Step 6.2. Wider unary methods use the existing
full-vocabulary unary logits from the same drafter forward pass. Unary-K32 and
Unary-K64 select the true top-K tokens at each speculative position and score
them with the unchanged full-vocabulary normalization:

```text
log q_t(b) = U_t(b) - LSE_V U_t
```

Candidate support is not renormalized over K. The verification budget still
counts emitted speculative tree nodes, not available candidates.

## Correctness checks

The implementation tests establish that:

- Unary-K16 exactly preserves the frozen online/offline K16 tree behavior.
- K16, K32, and K64 are true score-ordered top-K sets, with K16 a prefix of
  K32 and K32 a prefix of K64.
- All K values reuse the identical full-vocabulary logsumexp.
- A budget of B emits B speculative nodes regardless of candidate width.
- Wider trees remain prefix-closed and best-first ordered.
- The existing verifier compilation format and traversal are unchanged.
- Candidate diagnostics align with each round's realized target continuation.

The full suite passes: 32 tests.

## Main acceptance result

Mean matched draft tokens, averaged at the prompt level:

| Dataset | B | Pairwise-K16 | Unary-K16 | Unary-K32 | Unary-K64 |
|---|---:|---:|---:|---:|---:|
| GSM8K | 8 | 4.339 | 4.083 | 4.091 | 4.089 |
| GSM8K | 16 | 4.906 | 4.473 | 4.469 | 4.475 |
| GSM8K | 32 | 5.248 | 4.739 | 4.741 | 4.743 |
| GSM8K | 64 | 5.491 | 4.946 | 4.952 | 4.948 |
| MATH500 | 8 | 4.616 | 4.329 | 4.313 | 4.308 |
| MATH500 | 16 | 5.197 | 4.732 | 4.730 | 4.732 |
| MATH500 | 32 | 5.531 | 4.976 | 4.969 | 4.984 |
| MATH500 | 64 | 5.750 | 5.190 | 5.188 | 5.196 |

Pairwise-K16 minus each unary baseline, with paired prompt-bootstrap 95% CIs:

| Dataset | B | P16-U16 | P16-U32 | P16-U64 |
|---|---:|---:|---:|---:|
| GSM8K | 8 | +0.255 [0.215, 0.296] | +0.247 [0.207, 0.286] | +0.250 [0.209, 0.290] |
| GSM8K | 16 | +0.433 [0.383, 0.484] | +0.436 [0.385, 0.488] | +0.431 [0.381, 0.481] |
| GSM8K | 32 | +0.509 [0.461, 0.558] | +0.507 [0.460, 0.556] | +0.505 [0.458, 0.553] |
| GSM8K | 64 | +0.545 [0.499, 0.591] | +0.539 [0.494, 0.583] | +0.543 [0.498, 0.589] |
| MATH500 | 8 | +0.287 [0.249, 0.325] | +0.302 [0.259, 0.345] | +0.307 [0.268, 0.347] |
| MATH500 | 16 | +0.465 [0.412, 0.518] | +0.467 [0.419, 0.517] | +0.464 [0.414, 0.514] |
| MATH500 | 32 | +0.555 [0.509, 0.604] | +0.561 [0.510, 0.617] | +0.547 [0.500, 0.601] |
| MATH500 | 64 | +0.560 [0.512, 0.607] | +0.562 [0.518, 0.606] | +0.554 [0.503, 0.603] |

Every Pairwise-K16 versus Unary-K64 CI excludes zero. Pairwise improves over
Unary-K64 on 109/128 to 125/128 prompts depending on dataset and budget.

## Marginal value of wider unary support

Wider unary support has effectively zero acceptance value at fixed tree
budget.

| Dataset | B | U32-U16 | U64-U32 |
|---|---:|---:|---:|
| GSM8K | 8 | +0.008 [+0.000, +0.017] | -0.002 [-0.010, +0.005] |
| GSM8K | 16 | -0.004 [-0.016, +0.009] | +0.005 [-0.004, +0.016] |
| GSM8K | 32 | +0.001 [-0.005, +0.008] | +0.003 [-0.002, +0.008] |
| GSM8K | 64 | +0.007 [-0.007, +0.022] | -0.004 [-0.014, +0.004] |
| MATH500 | 8 | -0.015 [-0.037, +0.005] | -0.005 [-0.031, +0.017] |
| MATH500 | 16 | -0.002 [-0.022, +0.018] | +0.003 [-0.018, +0.025] |
| MATH500 | 32 | -0.006 [-0.036, +0.018] | +0.014 [-0.011, +0.044] |
| MATH500 | 64 | -0.002 [-0.023, +0.019] | +0.008 [-0.014, +0.033] |

Only GSM8K B=8 U32-U16 has a marginally positive interval, and its gain is
only 0.008 matched tokens. All other width increments include zero.

## Full-block acceptance

| Dataset | B | Pairwise-K16 | Unary-K16 | Unary-K32 | Unary-K64 |
|---|---:|---:|---:|---:|---:|
| GSM8K | 8 | 0.300 | 0.262 | 0.264 | 0.264 |
| GSM8K | 16 | 0.421 | 0.339 | 0.339 | 0.339 |
| GSM8K | 32 | 0.490 | 0.383 | 0.386 | 0.385 |
| GSM8K | 64 | 0.541 | 0.418 | 0.419 | 0.419 |
| MATH500 | 8 | 0.368 | 0.324 | 0.320 | 0.320 |
| MATH500 | 16 | 0.493 | 0.407 | 0.405 | 0.406 |
| MATH500 | 32 | 0.564 | 0.444 | 0.444 | 0.447 |
| MATH500 | 64 | 0.612 | 0.484 | 0.483 | 0.487 |

## Candidate coverage by K and depth

The table uses Unary-K64 trajectories as a common state distribution and
aggregates all four budgets. Each cell is target-token inclusion / full-prefix
representability. The complete per-budget and method-native tables are saved
as CSV artifacts.

| Dataset | Depth | K16 | K32 | K64 |
|---|---:|---:|---:|---:|
| GSM8K | 1 | 99.7% / 99.7% | 99.9% / 99.9% | 99.9% / 99.9% |
| GSM8K | 2 | 97.8% / 97.6% | 98.8% / 98.7% | 99.4% / 99.4% |
| GSM8K | 3 | 94.9% / 93.8% | 96.9% / 96.2% | 98.1% / 97.7% |
| GSM8K | 4 | 91.0% / 88.0% | 93.8% / 91.5% | 95.7% / 94.1% |
| GSM8K | 5 | 88.5% / 82.3% | 92.1% / 87.2% | 94.7% / 90.9% |
| GSM8K | 6 | 85.6% / 76.2% | 90.4% / 82.5% | 93.6% / 87.3% |
| GSM8K | 7 | 82.2% / 69.5% | 88.1% / 77.1% | 92.3% / 83.6% |
| MATH500 | 1 | 99.7% / 99.7% | 99.8% / 99.8% | 99.9% / 99.9% |
| MATH500 | 2 | 98.3% / 98.2% | 99.1% / 99.0% | 99.5% / 99.5% |
| MATH500 | 3 | 95.8% / 94.8% | 97.5% / 96.9% | 98.7% / 98.3% |
| MATH500 | 4 | 92.8% / 90.1% | 95.6% / 93.7% | 97.3% / 96.1% |
| MATH500 | 5 | 90.3% / 85.0% | 94.0% / 89.9% | 96.3% / 93.5% |
| MATH500 | 6 | 87.2% / 79.3% | 92.0% / 85.6% | 95.2% / 90.5% |
| MATH500 | 7 | 84.6% / 73.5% | 90.3% / 81.1% | 94.2% / 87.3% |

K64 substantially improves theoretical candidate coverage. At depth 7,
prefix representability rises by 14.1 percentage points on GSM8K and 13.8
points on MATH500 versus K16. This headroom does not translate to actual
acceptance at fixed B.

## Candidate-failure versus ranking/budget-failure decomposition

At B=64, excluding end-of-generation censored rounds:

| Dataset | Method | Candidate failure | Ranking/budget failure | Covered |
|---|---|---:|---:|---:|
| GSM8K | Unary-K16 | 12.5% | 45.1% | 42.4% |
| GSM8K | Unary-K32 | 7.3% | 50.1% | 42.5% |
| GSM8K | Unary-K64 | 3.8% | 53.6% | 42.5% |
| GSM8K | Pairwise-K16 | 20.0% | 25.3% | 54.7% |
| MATH500 | Unary-K16 | 10.3% | 40.3% | 49.4% |
| MATH500 | Unary-K32 | 5.5% | 45.2% | 49.3% |
| MATH500 | Unary-K64 | 2.8% | 47.0% | 50.1% |
| MATH500 | Pairwise-K16 | 17.7% | 20.0% | 62.3% |

Wider unary support does exactly what it should: candidate failures fall.
However, the additional candidates are not ranked into the fixed verification
budget, so ranking/budget failures rise by nearly the same amount and total
coverage stays flat. Pairwise-K16 accepts more full blocks despite more
failures being classified at its first unmatched token because it has far
fewer ranking/budget failures. Pairwise-K16 and Unary-K16 have identical
candidate support; Pairwise's higher candidate-failure rate is
depth-composition confounding because Pairwise generally reaches deeper,
where candidate inclusion is lower. It does not indicate weaker Pairwise
candidate support. The clean causal decomposition is within unary:
K16-to-K64 converts candidate failures into ranking/budget failures without
improving total coverage.

## Systems cost and throughput

At B=64:

| Dataset | Method | Candidate select ms/round | Tree build ms/round | Verify ms/round | tok/s |
|---|---|---:|---:|---:|---:|
| GSM8K | Unary-K16 | 0.209 | 2.259 | 39.719 | 101.54 |
| GSM8K | Unary-K32 | 0.287 | 3.024 | 39.805 | 100.04 |
| GSM8K | Unary-K64 | 0.284 | 5.506 | 39.869 | 95.97 |
| GSM8K | Pairwise-K16 | 0.209 | 2.503 | 39.739 | 110.41 |
| MATH500 | Unary-K16 | 0.210 | 2.316 | 40.182 | 104.04 |
| MATH500 | Unary-K32 | 0.290 | 3.056 | 40.303 | 102.30 |
| MATH500 | Unary-K64 | 0.287 | 5.629 | 40.388 | 98.31 |
| MATH500 | Pairwise-K16 | 0.211 | 2.523 | 40.247 | 112.92 |

Target verification latency remains comparable because B is fixed. Wider
unary support raises tree-construction cost and reduces throughput.
The P16-U64 throughput comparison includes the current wider-support
best-first tree-construction cost; it is not a claim that this Python
construction is maximally optimized. The overhead-neutral comparison is
P16-U16, where candidate selection and tree construction are similar and
Pairwise still gains 4.16 to 9.05 tok/s across the two datasets and budgets.
Throughput point estimates and CIs use equal-weighted per-prompt rates
(`output tokens / prompt decode time`), matching the prompt-level inference
unit rather than pooling all tokens and wall time.

Pairwise-K16 throughput minus Unary-K64:

| Dataset | B | Unary-K64 tok/s | Pairwise-K16 tok/s | Gain | Relative gain | 95% CI |
|---|---:|---:|---:|---:|---:|---:|
| GSM8K | 8 | 88.84 | 93.69 | +4.86 | +5.5% | [+4.14, +5.56] |
| GSM8K | 16 | 95.21 | 103.78 | +8.57 | +9.0% | [+7.67, +9.49] |
| GSM8K | 32 | 98.17 | 109.06 | +10.88 | +11.1% | [+10.03, +11.72] |
| GSM8K | 64 | 95.97 | 110.41 | +14.44 | +15.1% | [+13.60, +15.29] |
| MATH500 | 8 | 91.03 | 96.83 | +5.81 | +6.4% | [+5.08, +6.51] |
| MATH500 | 16 | 97.96 | 107.00 | +9.04 | +9.2% | [+8.11, +9.95] |
| MATH500 | 32 | 100.56 | 112.08 | +11.52 | +11.5% | [+10.65, +12.47] |
| MATH500 | 64 | 98.31 | 112.92 | +14.61 | +14.9% | [+13.68, +15.55] |

All paired prompt-level throughput CIs exclude zero.

## Generation anomalies and caveats

- As in Steps 5 through 6.2, some speculative outputs differ from the
  sequential target-only baseline because different BF16 verification tree
  shapes can resolve near-tied argmax values differently.
- Exact target-only sequence matches range from 63/128 to 74/128 on GSM8K
  and 24/128 to 34/128 on MATH500 across tree methods.
- No crashes, invalid trees, budget violations, or non-monotonic score
  failures occurred.
- Step 6.3 is an acceptance and systems falsification test. It does not repeat
  the task-quality evaluation already completed in frozen Step 6.2.
- Candidate coverage on a shared state distribution is reported from
  Unary-K64 trajectories. Method-native coverage is also preserved in the
  detailed artifact.
- Online methods commit different numbers and sometimes different token
  sequences, so they visit different decoding states. Consequently, the
  method-native candidate-failure and ranking/budget percentages are
  descriptive and must not be interpreted as a perfectly causal frozen-state
  decomposition. The common Unary-K64-trajectory coverage table isolates
  candidate-width representability, while the acceptance comparison remains
  the primary end-to-end result.

## Interpretation and recommendation

**STRONG GO**

The pre-specified decision rule was:

- **STRONG GO:** Pairwise-K16 beats or matches Unary-K64 at useful budgets
  while using much narrower candidate support.
- **GO:** Unary-K64 catches up, but Pairwise-K16 provides similar acceptance
  and better or equal throughput with less search breadth.
- **MIXED:** Wider unary materially narrows the gap and wins at some
  budgets or domains.
- **WEAKENED CLAIM:** Unary-K32 or Unary-K64 clearly dominates Pairwise-K16.

Pairwise-K16 beats Unary-K64 at every budget on both frozen datasets. All
acceptance CIs and all throughput CIs exclude zero. Unary candidate width is
not the explanation for the original Pairwise gain:

1. K64 materially reduces candidate failures and improves theoretical prefix
   representability.
2. Fixed-budget unary ranking cannot use that extra support; ranking/budget
   failures increase and observed acceptance remains flat.
3. Pairwise-K16 uses predecessor-conditioned information to rank a much
   narrower support more effectively.
4. Wider unary support also costs more to build and is slower end to end.

The supported claim is:

> Widening unary candidate support from K=16 to K=64 substantially improves
> target-path representability but produces essentially no improvement in
> fixed-budget acceptance. Pairwise-K16 continues to outperform Unary-K64,
> supporting the hypothesis that predecessor-conditioned ranking—not
> candidate breadth—is responsible for the observed gain.

This claim is scoped to the frozen, unrenormalized full-vocabulary unary
scores and the existing score-ordered best-first allocator at fixed node
budget. It does not rule out a different wider-support algorithm.

## Reproduction

Benchmark:

```bash
cd ~/ddtree-step63
PATH=/lambda/nfs/ddtree/.venv/bin:$PATH \
bash run_benchmark.sh \
  --task gsm8k:128 \
  --task math500:128 \
  --model-draft-pair \
    'Qwen/Qwen3-4B|mgoin/Qwen3-4B-speculator.dflash2' \
  --temperature 0.0 \
  --mode sdpa \
  --draft-type dflash2 \
  --gpus 0 \
  --nproc-per-node 1 \
  --max-new-tokens 2048 \
  --tree-budget 8,16,32,64 \
  --python /lambda/nfs/ddtree/.venv/bin/python \
  --log-dir logs/2026-09-05_step63-wide-unary_a100-40gb \
  --run-dir runs/2026-09-05_step63-wide-unary_a100-40gb
```

Analysis:

```bash
python analyze_step63_wide_unary.py \
  --gsm8k \
    runs/2026-09-05_step63-wide-unary_a100-40gb/gsm8k__Qwen_Qwen3-4B__mgoin_Qwen3-4B-speculator.dflash2__temp0.0__dflash2__sdpa.pt \
  --math500 \
    runs/2026-09-05_step63-wide-unary_a100-40gb/math500__Qwen_Qwen3-4B__mgoin_Qwen3-4B-speculator.dflash2__temp0.0__dflash2__sdpa.pt \
  --output-dir analysis/2026-09-05_step63-wide-unary \
  --bootstrap-samples 10000
```

Prompt and round CSV exports, including exact target-only match indicators:

```bash
python summarize_run.py \
  runs/2026-09-05_step63-wide-unary_a100-40gb/gsm8k__Qwen_Qwen3-4B__mgoin_Qwen3-4B-speculator.dflash2__temp0.0__dflash2__sdpa.pt \
  --csv runs/2026-09-05_step63-wide-unary_a100-40gb/gsm8k-summary.csv \
  --rounds-csv runs/2026-09-05_step63-wide-unary_a100-40gb/gsm8k-rounds.csv

python summarize_run.py \
  runs/2026-09-05_step63-wide-unary_a100-40gb/math500__Qwen_Qwen3-4B__mgoin_Qwen3-4B-speculator.dflash2__temp0.0__dflash2__sdpa.pt \
  --csv runs/2026-09-05_step63-wide-unary_a100-40gb/math500-summary.csv \
  --rounds-csv runs/2026-09-05_step63-wide-unary_a100-40gb/math500-rounds.csv
```

## Files

Implementation and tests:

- `model/dflash2.py`
- `offline_dflash2_trees.py`
- `dflash2_tree.py`
- `summarize_run.py`
- `tests/test_dflash2_tree.py`

Analysis and report:

- `analyze_step63_wide_unary.py`
- `analysis/2026-09-05_step63-wide-unary/`
- `research_notes/dflash2_step63_wider_unary.md`
- `analysis/2026-09-05_step63-wide-unary/artifact_manifest.sha256`

Raw local artifacts:

- `runs/2026-09-05_step63-wide-unary_a100-40gb/`
- `logs/2026-09-05_step63-wide-unary_a100-40gb/`

## Freeze declaration

Step 6.3 is frozen after this report. No scorer, normalization, candidate
support, tree builder, checkpoint, budget, prompt subset, or analysis choice
was tuned based on these results. HumanEval, MT-Bench, wider models, H100
experiments, and further optimization were not started.

The exact benchmark implementation is committed at
`88774714e931f57d8cc974160352e2b8a587051b`; the exact analysis code that
generated the final tables is committed at
`95f0f30e7b6e2ff5e845e2c1062cb5bb034c01c1`. The SHA-256 manifest covers the
implementation, analysis code, report, machine-readable tables, raw run
artifacts, CSV exports, and logs.
