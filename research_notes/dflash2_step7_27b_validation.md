# Step 7: Official Qwen3.8-27B DFlash2 validation

## Question and decision

This experiment tests whether the frozen predecessor-conditioned
Pairwise-DDTree allocation from Steps 6.2/6.3 generalizes to the official
27B DFlash2 checkpoint.

**Recommendation: GO.** Pairwise-K16 improves mean matched draft tokens over
Unary-K16 on both GSM8K-64 and MATH500-64 at B=16/32/64. All six paired
prompt-bootstrap 95% confidence intervals exclude zero. Pairwise and unary
throughput are statistically tied in the correctness-first recurrent-target
verifier, so this establishes algorithmic generalization but not a production
throughput win.

No scorer, candidate lattice, tree allocation rule, or benchmark-specific
parameter was tuned.

## Frozen models and configuration

| Component | Value |
|---|---|
| Target | `Qwen/Qwen3.8-27B` |
| Target revision | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` |
| Drafter | `incoai/Qwen3.8-27B-DFlash2` |
| Drafter revision | `dedf8df68adfb1afeaf7b7480c0a0243108177b4` |
| Dtype | BF16 |
| Attention implementation | SDPA |
| Temperature | 0 |
| DFlash2 block size | 8 |
| Speculative positions | 7 |
| Selector top-K / rank | 16 / 256 |
| Dynamic convolution | kernel 2, group size 16 |
| Draft layers | 5 Qwen3 sliding-attention layers |
| Target hidden-layer IDs | `[5, 19, 33, 47, 61]` |
| Draft sliding window | 2048 |

The target is `Qwen3_5ForCausalLM`, with vocabulary 248,320, hidden size
5,120, and 64 layers: 48 recurrent linear-attention layers and 16
full-attention layers.

## Compatibility changes

The official drafter config stores checkpoint metadata under
`dflash_config`, unlike the experimental 4B checkpoint. The adapter now
supports both layouts with required-field validation.

The official drafter omits `embed_tokens.weight` and `lm_head.weight`.
The strict loader verifies that exactly both are omitted, rejects any other
missing weights, verifies their shapes, and shares the target input embedding
and output head. This avoids silently using randomly initialized token
matrices.

Qwen3.5 recurrent caches require `DynamicCache(config=...)`, past-state
recording, and negative token-removal rollback. The shared cache helpers now
provide those semantics while preserving Qwen3 behavior.

The Step-6 packed verifier is not correct for the hybrid target. Its 4D
ancestry mask controls full-attention layers, but recurrent layers would
process unrelated packed branches through one recurrent state. Step 7
therefore retains the exact frozen scorer and tree builder but verifies only
the target-selected path sequentially. This is correctness-first: it preserves
the target trajectory and acceptance measurement, but it does not exploit
parallel tree verification and therefore is not expected to improve
throughput over greedy DFlash2.

## Correctness

- Frozen Unary-K16 and Pairwise-K16 scoring/tree tests remain unchanged.
- Retained mass, non-positive extension scores, prefix closure, parent order,
  depth <= 7, B-node limits, and root exclusion remain enforced.
- The recurrent verifier visits only the root and target-selected child path.
- Every Unary/Pairwise output exactly matched the one-token sequential target
  baseline on all 64 GSM8K and all 64 MATH500 prompts, for every budget:
  768/768 method/prompt comparisons.
- Focused compatibility, cache, and tree tests pass.

## Memory and vanilla DFlash2 smoke

| Point | Allocated GiB | Reserved GiB |
|---|---:|---:|
| Target loaded | 50.10 | 50.10 |
| Target + shared-weight drafter loaded | 53.69 | 53.72 |
| Vanilla generation smoke peak | 53.95 | 53.99 |
| GSM8K benchmark peak | 54.19 | 54.34 |
| MATH500 benchmark peak | 54.64 | 54.84 |

The pair fits safely on one H100 80GB without quantization, tensor
parallelism, or reduced precision.

The one-prompt vanilla smoke generated `12` correctly, matched 2 draft
tokens, committed 2 tokens in its verification round, ran at 11.55 tok/s
including the tiny-run timing effects, and reconstructed all seven selector
positions.

## Datasets

Both runs use the test split, `shuffle(seed=0)`, then `select(range(64))`.
Maximum new tokens is 256.

| Dataset | Repository/config/split | Revision | Selected text SHA-256 |
|---|---|---|---|
| GSM8K | `openai/gsm8k`, `main`, `test` | `740312add88f781978c0658806c59bc2815b9866` | `a42617d4a4af08316ac7efa12bab56272b9afb415a18b8afaa24e838a8a5cb94` |
| MATH500 | `HuggingFaceH4/MATH-500`, `test` | `6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be` | `cfceb02e924f35e38579f86579c76b74c96326210f240b2a9ec99b3e74f646bc` |

The selected-text hash is SHA-256 over the newline-joined `question` or
`problem` field in selected order.

## Acceptance

Mean matched draft tokens and paired prompt-bootstrap differences
(10,000 resamples):

| Dataset | B | Unary-K16 | Pairwise-K16 | Pairwise - Unary (95% CI) | Improve / tie / hurt |
|---|---:|---:|---:|---:|---:|
| GSM8K | 16 | 5.646 | 6.064 | +0.418 [0.353, 0.485] | 56 / 7 / 1 |
| GSM8K | 32 | 5.838 | 6.282 | +0.444 [0.384, 0.505] | 59 / 5 / 0 |
| GSM8K | 64 | 5.993 | 6.430 | +0.436 [0.374, 0.499] | 60 / 4 / 0 |
| MATH500 | 16 | 5.561 | 5.964 | +0.404 [0.351, 0.461] | 61 / 2 / 1 |
| MATH500 | 32 | 5.777 | 6.216 | +0.439 [0.385, 0.495] | 61 / 3 / 0 |
| MATH500 | 64 | 5.948 | 6.382 | +0.434 [0.378, 0.491] | 61 / 3 / 0 |

Relative acceptance gains are approximately 7.2%-7.7% across all six
comparisons.

## Full-block acceptance and committed tokens

| Dataset | B | Unary full block | Pairwise full block | Unary committed/round | Pairwise committed/round |
|---|---:|---:|---:|---:|---:|
| GSM8K | 16 | 0.603 | 0.714 | 6.624 | 7.042 |
| GSM8K | 32 | 0.654 | 0.776 | 6.816 | 7.258 |
| GSM8K | 64 | 0.687 | 0.808 | 6.971 | 7.407 |
| MATH500 | 16 | 0.579 | 0.693 | 6.539 | 6.941 |
| MATH500 | 32 | 0.638 | 0.748 | 6.754 | 7.191 |
| MATH500 | 64 | 0.677 | 0.789 | 6.925 | 7.358 |

## Throughput and latency

| Dataset | B | Unary tok/s | Pairwise tok/s | Unary tree ms | Pairwise tree ms | Unary verify ms | Pairwise verify ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| GSM8K | 16 | 17.296 | 17.278 | 0.589 | 0.769 | 372.75 | 398.19 |
| GSM8K | 32 | 17.296 | 17.291 | 0.894 | 1.083 | 384.26 | 410.43 |
| GSM8K | 64 | 17.254 | 17.316 | 1.467 | 1.674 | 393.76 | 418.14 |
| MATH500 | 16 | 17.552 | 17.555 | 0.583 | 0.758 | 361.35 | 384.78 |
| MATH500 | 32 | 17.546 | 17.559 | 0.890 | 1.070 | 373.88 | 399.45 |
| MATH500 | 64 | 17.542 | 17.579 | 1.459 | 1.657 | 383.71 | 408.54 |

All six paired throughput CIs include zero. Pairwise tree construction adds
only 0.18-0.21 ms/round, negligible beside target verification. Pairwise
verification takes longer because its higher acceptance causes the
correctness-first verifier to execute more sequential target steps.

Greedy-path DFlash2 measured 27.87 tok/s on GSM8K and 27.19 tok/s on
MATH500. The tree methods are slower because the recurrent-safe prototype
does not pack branches into one target forward. This is a verifier limitation,
not evidence against the allocation result.

## 4B versus 27B

| Model | Dataset | B | Unary | Pairwise | Gain | Relative gain | Pairwise-Unary tok/s |
|---|---|---:|---:|---:|---:|---:|---:|
| Qwen3-4B | GSM8K | 16 | 4.473 | 4.906 | +0.433 | 9.7% | +7.38 |
| Qwen3.8-27B | GSM8K | 16 | 5.646 | 6.064 | +0.418 | 7.4% | -0.02 |
| Qwen3-4B | GSM8K | 32 | 4.739 | 5.248 | +0.509 | 10.7% | +8.47 |
| Qwen3.8-27B | GSM8K | 32 | 5.838 | 6.282 | +0.444 | 7.6% | -0.00 |
| Qwen3-4B | GSM8K | 64 | 4.946 | 5.491 | +0.545 | 11.0% | +8.87 |
| Qwen3.8-27B | GSM8K | 64 | 5.993 | 6.430 | +0.436 | 7.3% | +0.06 |

The acceptance effect survives at 27B with slightly smaller relative gains.
The throughput translation does not survive in the current harness because
the hybrid recurrent architecture requires sequential path verification.

## Numerical anomaly

Vanilla block DFlash2 differed from the one-token sequential BF16 baseline on
all 64 prompts in each dataset. The difference is associated with target
forward shape/numerical behavior: the vanilla method verifies an eight-token
block, while the baseline advances one token at a time. It did not cause
invalid output, cache failure, NaNs, or OOM.

The recurrent-safe Unary and Pairwise methods both advance only the
target-selected path and exactly matched the sequential baseline in every
comparison. Therefore the primary Pairwise-versus-Unary result uses identical,
validated target trajectories.

## Reproduction

```bash
python run_step7_compatibility.py \
  --output artifacts/step7/load_compatibility.json

python run_step7_vanilla_smoke.py \
  --max-new-tokens 16 \
  --output artifacts/step7/vanilla_smoke.json

python run_step7_tree_smoke.py \
  --tree-budget 16 \
  --max-new-tokens 16 \
  --output artifacts/step7/tree_smoke_b16.json

python benchmark.py \
  --model-name-or-path Qwen/Qwen3.8-27B \
  --model-revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 \
  --draft-name-or-path incoai/Qwen3.8-27B-DFlash2 \
  --draft-revision dedf8df68adfb1afeaf7b7480c0a0243108177b4 \
  --draft-type dflash2 \
  --dataset gsm8k \
  --max-samples 64 \
  --max-new-tokens 256 \
  --tree-budget 16,32,64 \
  --dflash2-tree-methods dflash2_unary_k16,dflash2_pairwise_k16 \
  --save-path artifacts/step7/gsm8k_64.pt

# Repeat the previous command with:
#   --dataset math500
#   --save-path artifacts/step7/math500_64.pt

python analyze_step7.py \
  --gsm8k artifacts/step7/gsm8k_64.pt \
  --math500 artifacts/step7/math500_64.pt \
  --output-dir analysis/2026-09-05_step7-27b
```

## Files changed

- `model/dflash2.py`
- `model/__init__.py`
- `generation_cache.py`
- `dflash.py`
- `dflash2.py`
- `dflash2_tree.py`
- `benchmark.py`
- `run_step7_compatibility.py`
- `run_step7_vanilla_smoke.py`
- `run_step7_tree_smoke.py`
- `analyze_step7.py`
- `tests/test_dflash2.py`
- `tests/test_dflash2_generate.py`
- `tests/test_dflash2_tree.py`
- `analysis/2026-09-05_step7-27b/`
- `research_notes/dflash2_step7_27b_validation.md`

## Environment

| Component | Value |
|---|---|
| GPU | NVIDIA H100 80GB HBM3, one GPU |
| Python | 3.10.12 |
| PyTorch | 2.7.0 |
| CUDA reported by PyTorch | 12.8 |
| Transformers | 5.16.1 |
| Attention | SDPA |
| Quantization | none |
| Tensor parallelism | none |

Raw artifact hashes are recorded in
`analysis/2026-09-05_step7-27b/raw_artifacts.sha256`. The committed analysis
tables and provenance are covered by
`analysis/2026-09-05_step7-27b/artifact_manifest.sha256`.
