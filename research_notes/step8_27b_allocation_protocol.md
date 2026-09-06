# Step 8: Official 27B allocation benchmark protocol

## Research question

Given the same official DFlash2 proposal and a fixed verification-node budget,
does predecessor-conditioned Pairwise-DDTree allocate nodes more effectively
than DFlash2's greedy path and the released DDTree unary allocation rule?

This benchmark compares allocation rules. Qwen3.8-27B requires the
correctness-first target-selected-path verifier, so it does not reproduce the
parallel packed-tree throughput system used by the original DDTree paper.

## Frozen configuration

| Item | Value |
|---|---|
| Target | `Qwen/Qwen3.8-27B` |
| Target revision | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` |
| Drafter | `incoai/Qwen3.8-27B-DFlash2` |
| Drafter revision | `dedf8df68adfb1afeaf7b7480c0a0243108177b4` |
| GPU | One H100 80 GB |
| Temperature | 0 |
| Maximum new tokens | 256 |
| Budgets | 7, 16, 32, 64 |
| GSM8K | 128 prompts |
| MATH500 | 128 prompts |
| HumanEval | 164 prompts |
| MT-Bench | 80 conversations |

## Methods

- `dflash2`: the checkpoint's native seven-node greedy path.
- `dflash2_original_ddtree`: released DDTree unary scoring with
  \(K=\min(B,|V|)\), giving K7/K16/K32/K64.
- `dflash2_pairwise_k16`: mass-preserving conditional allocation on the
  checkpoint's native top-16 selector lattice.
- `dflash2_unary_k16` at B32/B64: a diagnostic that holds candidate support
  fixed against Pairwise.

The primary endpoint is the paired prompt-level difference in mean matched
speculative tokens between Pairwise-K16 and Original-DDTree-K16 at B=16.

At B=7, Pairwise and DFlash2 greedy use the same number of speculative nodes.
At B=32 and B=64, original DDTree receives wider unary support than Pairwise.

## Required staged execution

Run each stage only after the prior artifact passes inspection and analysis:

```bash
bash run_step8_27b_protocol.sh one
bash run_step8_27b_protocol.sh matrix
bash run_step8_27b_protocol.sh domains
bash run_step8_27b_protocol.sh full
```

The script refuses to run with tracked worktree changes. Artifacts are stored
under `artifacts/step8/<commit>/<profile>/` and use atomic partial checkpoints
that resume only when every argument matches.

The `one` stage checks one GSM8K prompt at B=7. `matrix` checks two GSM8K
prompts across the complete method matrix. `domains` checks four samples from
each dataset, including both controlled-history and native-history MT-Bench.
Only `full` is an empirical result.

## Full-run analysis

Replace `<commit>` with the clean benchmark commit:

```bash
python analyze_step8_protocol.py \
  --expected-commit <commit> \
  --run GSM8K=artifacts/step8/<commit>/full/gsm8k_controlled.pt \
  --run MATH500=artifacts/step8/<commit>/full/math500_controlled.pt \
  --run HumanEval=artifacts/step8/<commit>/full/humaneval_controlled.pt \
  --run MT-Bench-controlled=artifacts/step8/<commit>/full/mt-bench_controlled.pt \
  --run MT-Bench-native=artifacts/step8/<commit>/full/mt-bench_native.pt \
  --output-dir analysis/step8-27b-allocation
```

For smoke profiles, add `--allow-partial`. The analyzer rejects wrong
revisions, dirty commits, incomplete sample sets, missing methods, incorrect
node or candidate counts, missing proposal/tree traces, and any checked tree
output that differs from sequential target decoding.

It writes:

- `method_metrics.csv`
- `paired_comparisons.csv`
- `controlled_allocation_metrics.csv`
- `controlled_allocation_comparisons.csv`
- `provenance.json`, including SHA-256 hashes of all source artifacts

## Interpretation boundary

Matched speculative tokens and committed tokens per target call are the main
outcomes. Throughput is recorded but is not expected to improve under the
sequential recurrent-safe verifier. Published original-DDTree 30B speedups are
external context, not a directly comparable baseline.
