# Step 9: One-H100 Qwen3-4B throughput validation

## Question and decision

This experiment asks whether DFlash2 Pairwise-DDTree's improved tree
allocation translates into end-to-end throughput, and whether the resulting
system is faster than the released original-DFlash baselines.

**Within DFlash2: yes.** At the primary budget \(B=16\), Pairwise-K16 is
5.3%-9.6% faster than DFlash2 with the released unary DDTree allocator on all
tested domains. Every paired prompt-bootstrap 95% confidence interval excludes
zero.

**Across drafter families: no.** The released original-DFlash checkpoint is
faster than the tested DFlash2 checkpoint both as a raw drafter and with
released DDTree. Pairwise improves how DFlash2 spends a fixed verification
budget, but it does not make this DFlash2 system the fastest tested system.

The cross-family result is descriptive rather than an allocator ablation:
the checkpoints use different drafter architectures and different draft
attention backends. No model, scorer, allocation rule, or benchmark parameter
was tuned during this run.

## Frozen setup

| Component | Value |
|---|---|
| Benchmark commit | `1bfb1cb8cf3cb40308d520de608a01fe5a86eafe` |
| Target | `Qwen/Qwen3-4B` at `1cfa9a7208912126459214e8b04321603b3df60c` |
| Original drafter | `z-lab/Qwen3-4B-DFlash-b16` at `b74e3a329c4d963783143b1e970d95b002be72bd` |
| DFlash2 drafter | `mgoin/Qwen3-4B-speculator.dflash2` at `e3e7a18e4f541fa3841c2fb0666a7759079ab6fd` |
| Hardware | One NVIDIA H100 80GB HBM3 |
| Software | Python 3.10.12, PyTorch 2.7.0, CUDA 12.8, Transformers 5.16.1 |
| Precision / decoding | BF16 / greedy (`temperature=0`) |
| Target attention | SDPA for both families |
| Draft attention | FlashAttention2 for original DFlash; SDPA for DFlash2 |
| Output cap | 256 new tokens |
| Timing | Three repetitions per method and prompt/turn |
| Primary endpoint | Paired end-to-end output tokens/s, including prefill |
| Bootstrap | 10,000 prompt/conversation resamples, seed `20260906` |

The run covers 128 GSM8K prompts, 128 MATH500 prompts, all 164 HumanEval
prompts, and 80 two-turn MT-Bench conversations under both controlled shared
history and native per-method history. Dataset revisions are recorded in
`analysis/step9-4b-throughput/provenance.json`.

## Primary controlled result

Pairwise-K16 and released unary DDTree use the same DFlash2 checkpoint,
candidate support \(K=16\), target verifier, prompts, and node budget \(B=16\).
Only tree allocation differs.

| Dataset | Unary tok/s | Pairwise tok/s | Difference (95% CI) | Relative gain | Improve / hurt |
|---|---:|---:|---:|---:|---:|
| GSM8K | 163.32 | 176.79 | +13.47 [12.11, 14.81] | +8.44% | 121 / 7 |
| MATH500 | 168.58 | 184.21 | +15.63 [13.88, 17.42] | +9.60% | 120 / 8 |
| HumanEval | 145.77 | 156.44 | +10.66 [9.08, 12.28] | +7.61% | 144 / 20 |
| MT-Bench controlled | 129.57 | 136.64 | +7.07 [5.53, 8.62] | +5.41% | 67 / 13 |
| MT-Bench native | 130.46 | 137.66 | +7.20 [5.43, 9.02] | +5.32% | 68 / 12 |

The throughput result tracks improved allocation. Mean matched draft tokens
per verification round increase from 4.347 to 4.823 on GSM8K, 4.375 to 4.904
on MATH500, 3.819 to 4.197 on HumanEval, and approximately 3.11 to 3.36 on
both MT-Bench trajectory variants.

The direction also holds at the secondary budgets \(B=7,32,64\). The full
budget-specific results and confidence intervals are in
`paired_throughput_comparisons.csv`.

## Practical system comparison

Absolute end-to-end throughput at \(B=16\) shows that the allocation gain does
not overcome the tested DFlash2 drafter's lower system throughput.

| Dataset | Raw original DFlash | Original DFlash + DDTree | DFlash2 + Pairwise |
|---|---:|---:|---:|
| GSM8K | 232.97 | 261.22 | 176.79 |
| MATH500 | 268.28 | 285.39 | 184.21 |
| HumanEval | 240.60 | 266.87 | 156.44 |
| MT-Bench controlled | 158.29 | 183.57 | 136.64 |
| MT-Bench native | 152.98 | 177.95 | 137.66 |

Pairwise-DFlash2 is slower than both original-DFlash comparisons on every
dataset. The paired cross-drafter confidence intervals remain below zero,
including after normalization by each process's sequential-target baseline.
These comparisons conflate checkpoint architecture, drafter implementation,
and draft attention backend, so they do not contradict the controlled
allocation result.

## Correctness and task quality

BF16 packed/block verification was not token-exact with one-token sequential
target decoding on every prompt. At \(B=16\), exact full-output rates for
unary versus Pairwise were 57.8% versus 57.0% on GSM8K, 46.9% versus 35.2% on
MATH500, 42.7% versus 45.1% on HumanEval, 36.3% versus 36.3% on controlled
MT-Bench, and 36.3% versus 37.5% on native MT-Bench. A full-output exact match
requires every generated token to match, and many outputs reached the
256-token cap, but these rates still mean this run does not establish strict
lossless equivalence to sequential BF16 decoding.

Task-level scoring did not show a consistent Pairwise-B16 regression relative
to unary-B16:

| Task | Sequential target | Unary-B16 | Pairwise-B16 |
|---|---:|---:|---:|
| GSM8K accuracy | 65/128 | 64/128 | 62/128 |
| MATH500 accuracy | 18/128 | 18/128 | 18/128 |
| HumanEval pass@1 | 91/164 | 88/164 | 89/164 |

These are descriptive counts, not a powered quality-equivalence test. The
two-answer GSM8K difference, one-pass HumanEval difference, and exact-output
divergences should not be interpreted as proof of either quality parity or a
quality change. MT-Bench responses were exported for external judging; this
protocol does not assign MT-Bench quality scores.

## What this experiment establishes

- Predecessor-conditioned Pairwise allocation improves both matched tokens and
  end-to-end throughput over released unary DDTree allocation on the same
  DFlash2 checkpoint under the tested one-H100 setup.
- The controlled throughput gain generalizes across math, code, and dialogue
  datasets and both MT-Bench trajectory regimes.
- The tested DFlash2 system remains slower in absolute tokens/s than released
  original DFlash and original DFlash+DDTree.
- Task scores are broadly similar at \(B=16\), but strict decoding equivalence
  and statistical quality equivalence are not established.

## Frozen evidence

Compact analysis:

- `analysis/step9-4b-throughput/method_metrics.csv`
- `analysis/step9-4b-throughput/paired_throughput_comparisons.csv`
- `analysis/step9-4b-throughput/timing_decomposition.csv`
- `analysis/step9-4b-throughput/correctness.csv`
- `analysis/step9-4b-throughput/provenance.json`
- `analysis/step9-4b-throughput/quality/*/{gsm8k,math500,humaneval}/*_summary.csv`
- `analysis/step9-4b-throughput/artifact_manifest.sha256`

Raw `.pt` artifacts and logs are retained locally outside the git commit under
`artifacts/step9_4b/1bfb1cb8cf3cb40308d520de608a01fe5a86eafe/full/` and
`logs/step9_4b/`. Their hashes are recorded in
`analysis/step9-4b-throughput/raw_artifacts_sha256.txt`. The complete local
analysis archive contains the detailed task records, sanitized HumanEval
generations, EvalPlus results, and exported MT-Bench responses; its transfer
was verified against `analysis_sha256.txt`.
