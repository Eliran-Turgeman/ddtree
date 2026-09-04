# Online DFlash2 Unary-K16 vs Pairwise-K16 evaluation

## Scope

This Step-5 experiment tests whether the Step-4 offline allocation gain
survives the real target-model tree verifier.

- Target: `Qwen/Qwen3-4B`
- Drafter: `mgoin/Qwen3-4B-speculator.dflash2`
- Dataset: deterministic 32-prompt GSM8K subset
- GPU: NVIDIA A100-SXM4 40GB
- PyTorch / CUDA / Transformers: 2.7.0 / 12.8 / 5.16.1
- Temperature: 0
- Maximum new tokens: 2048
- Source commit: `660645efbe0cfe7913a218c577290c360cbed5d5`

The primary comparison holds the DFlash2 forward pass, seven speculative
positions, top-16 candidate IDs, and tree budget fixed:

```text
DFlash2 Unary-K16 Tree
vs
DFlash2 Pairwise-K16 Tree
```

## Online insertion point

`dflash2_tree_generate()` replaces only the allocation stage:

```text
DFlash2 proposal tensors
    -> audited Step-4 scorer
    -> build_best_first_tree(...)
    -> compile_generic_tree_for_verifier(...)
```

The existing DDTree `compile_ddtree_tree(...)`, tree attention mask, position
IDs, target forward pass, `follow_verified_tree(...)`, cache compaction, and
commit path are reused unchanged.

Generic `TreeNode.parent` indices exclude the conceptual root. The adapter
converts them to the verifier's root-inclusive indexing, builds token-to-child
maps, and constructs the ancestor visibility matrix. The root is not counted
in the speculative-node budget.

No DFlash2-specific changes were required to the target tree mask, position
IDs, or verifier logit indexing.

## Construction identity and defensive checks

Frozen Step-3 rounds were passed through both the offline and new online
allocation paths at B=8, 16, 32, and 64.

Unary and Pairwise trees matched exactly for:

- token IDs
- node depths
- parents
- candidate-index paths
- node order

Tests also enforce candidate tensor shapes, retained mass at most one,
non-positive extensions within numerical tolerance, prefix closure, valid
parent ordering, depth bounds, and budget bounds.

The production run encountered a mathematically zero extension represented as
`+1.31e-6` in float32. The builder now accepts numerical noise through `1e-5`
and clamps it to zero; larger positive extensions still fail.

## Verification semantics and correctness

Every round records:

- matched draft tokens
- committed tokens
- whether the verifier bonus was committed
- tree size and depth distribution
- CUDA-synchronized draft, tree-build, tree-compile, verify, commit, and total
  latency

Committed paths are always selected by the actual target verifier. Sequential
token identity is not universal because different BF16 verification matrix
shapes can resolve close argmax ties differently:

- DFlash2 greedy: 20/32 exact sequential matches
- Unary trees: 17-21/32 depending on budget
- Pairwise trees: 18-22/32 depending on budget

The inspected early smoke divergence occurred after 51 generated tokens and
was a coherent wording tie (`need to find` versus `are asked to find`), not an
invalid speculative acceptance.

## Online acceptance

Prompt-level means and confidence intervals use 10,000 paired bootstrap
resamples over the 32 prompts.

| Method | B | Mean matched draft tokens | 95% CI | Full-block acceptance |
| :----- | -: | ------------------------: | -----: | --------------------: |
| DFlash2 greedy path | 7 | 3.9117 | [3.7263, 4.1040] | 29.69% |
| Unary-K16 | 7 | 3.9623 | [3.7829, 4.1417] | 23.43% |
| Pairwise-K16 | 7 | 4.1711 | [4.0062, 4.3405] | 27.22% |
| Unary-K16 | 8 | 4.0590 | [3.8814, 4.2393] | 25.49% |
| Pairwise-K16 | 8 | 4.3445 | [4.1845, 4.5053] | 29.33% |
| Unary-K16 | 16 | 4.4820 | [4.2930, 4.6738] | 32.91% |
| Pairwise-K16 | 16 | 4.8562 | [4.6696, 5.0355] | 41.10% |
| Unary-K16 | 32 | 4.7768 | [4.6010, 4.9527] | 37.31% |
| Pairwise-K16 | 32 | 5.1694 | [4.9917, 5.3470] | 46.71% |
| Unary-K16 | 64 | 4.9506 | [4.7769, 5.1216] | 41.63% |
| Pairwise-K16 | 64 | 5.4642 | [5.3029, 5.6191] | 52.95% |

## Paired prompt comparison

| B | Pairwise - Unary matched tokens | 95% CI | Prompts improve / tie / hurt |
| -: | --------------------------------: | -----: | ----------------------------: |
| 7 | +0.2088 | [0.1250, 0.2832] | 26 / 1 / 5 |
| 8 | +0.2855 | [0.2205, 0.3477] | 28 / 3 / 1 |
| 16 | +0.3742 | [0.2724, 0.4758] | 30 / 1 / 1 |
| 32 | +0.3926 | [0.3086, 0.4759] | 29 / 2 / 1 |
| 64 | +0.5136 | [0.4255, 0.5994] | 31 / 1 / 0 |

The relative Pairwise > Unary ordering survives online at every budget.
Regressions are sparse and budget-specific. The worst observed prompt-level
regression was -0.40 tokens at B=7; no prompt regressed at B=64. Because the
methods follow different online trajectories, these cases cannot be attributed
to a single corresponding round without additional debug tracing.

## Offline versus online

The online round-weighted means were slightly higher than the frozen-trace
offline predictions for both methods:

- Unary: approximately +0.05 to +0.10 tokens
- Pairwise: approximately +0.03 to +0.08 tokens

The expected relative ordering survived. The offline and online values are not
expected to match exactly because online acceptance changes subsequent anchors
and state visitation.

## Timing

| Method | B | Tokens/s | ms/token |
| :----- | -: | -------: | -------: |
| DFlash2 greedy | 7 | 90.78 | 11.17 |
| Unary-K16 | 7 | 88.42 | 11.47 |
| Pairwise-K16 | 7 | 92.15 | 10.97 |
| Unary-K16 | 16 | 96.94 | 10.44 |
| Pairwise-K16 | 16 | 102.69 | 9.85 |
| Unary-K16 | 32 | 101.45 | 9.96 |
| Pairwise-K16 | 32 | 108.26 | 9.32 |
| Unary-K16 | 64 | 102.10 | 9.88 |
| Pairwise-K16 | 64 | 110.61 | 9.10 |

The target verification stage remained approximately 39-40 ms per decoding
round. Pairwise tree construction added only:

- +0.18 ms/round at B=7
- +0.17 ms/round at B=8
- +0.20 ms/round at B=16
- +0.20 ms/round at B=32
- +0.24 ms/round at B=64

relative to Unary. The extra allocation cost was more than offset by fewer
online rounds. This is a prototype measurement, not a production serving
claim.

## Recommendation

**STRONG GO to optimization and broader validation.**

The Step-5 scientific success criterion is met: on the exact same generated
DFlash2 top-16 lattice, Pairwise-conditioned allocation improves acceptance
under the real target tree verifier. At moderate budgets the online gain is
+0.37 to +0.51 matched speculative tokens, and the prototype also improves
throughput on this controlled A100 run.

The next work should optimize and validate this implementation without yet
changing model size, checkpoint, or serving framework.

## Reproduction

```bash
bash run_benchmark.sh \
  --gpus 0 \
  --nproc-per-node 1 \
  --task gsm8k:32 \
  --model-draft-pair \
    'Qwen/Qwen3-4B|mgoin/Qwen3-4B-speculator.dflash2' \
  --draft-type dflash2 \
  --tree-budget 7,8,16,32,64 \
  --temperature 0.0 \
  --mode sdpa \
  --max-new-tokens 2048 \
  --run-dir runs/2026-09-04_step5-online_a100_gsm8k-32 \
  --log-dir logs/2026-09-04_step5-online_a100_gsm8k-32

python summarize_run.py \
  runs/2026-09-04_step5-online_a100_gsm8k-32/*.pt

python analyze_online_dflash2.py \
  runs/2026-09-04_step5-online_a100_gsm8k-32/*.pt \
  analysis/2026-09-04_step5-online_a100_gsm8k-32 \
  --bootstrap-samples 10000
```
