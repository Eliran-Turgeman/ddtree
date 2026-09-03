# DFlash2 trace schema

## Checkpoint configuration

The trace collector targets `mgoin/Qwen3-4B-speculator.dflash2` with
`Qwen/Qwen3-4B` as verifier. The checkpoint configuration specifies:

- five DFlash2 transformer layers
- hidden size `H = 2560`
- vocabulary size `V = 151936`
- block size `8`, giving `L = 7` speculative positions
- top-K candidate count `K = 16`
- selector rank `R = 256`
- dynamic-convolution kernel size `2`
- dynamic-convolution group size `16`
- target hidden-state layers `[1, 9, 17, 25, 33]`

## Actual unary computation

`dflash2_generate` obtains draft hidden states with shape `[B, L, H]`.
`DFlash2DraftModel.propose` applies the checkpoint's `lm_head`:

```text
unary_logits = lm_head(draft_hidden)
```

The resulting full-vocabulary tensor has shape `[B, L, V]`. It is not saved
because it would add roughly 2 MiB per round in BF16. The proposal retains:

```text
candidate_ids     = topk(unary_logits, K).indices  # [B, L, K]
unary_scores      = gather(unary_logits, candidate_ids)  # [B, L, K]
unary_logsumexp   = logsumexp(unary_logits.float(), dim=-1)  # [B, L]
```

The top-K operation is performed directly on the full-vocabulary logits and
returns candidates sorted by descending unary logit.

## Actual pairwise computation

The selector contains:

```text
predecessor_codebook  # [V, R]
successor_codebook    # [V, R]
hidden_projection.weight  # [R, H]
```

For hidden state `h_t`, predecessor token `a`, and successor candidate `b`,
the implementation computes:

```text
projected_t = hidden_projection(h_t)                  # [R]
context_t(a) = predecessor_codebook[a] * projected_t # [R]
correction_t(a, b) =
    dot(context_t(a), successor_codebook[b])          # scalar
final_t(a, b) = unary_t(b) + correction_t(a, b)
```

For the first speculative position, the predecessor is the single anchor
token:

```text
anchor_pairwise_corrections  # [B, K]
anchor_final_scores          # [B, K]
```

For positions `1..L-1`, the collector evaluates every transition from the
previous position's retained candidates to the current position's candidates:

```text
pairwise_corrections  # [B, L - 1, K, K]
pairwise_final_scores # [B, L - 1, K, K]
```

The two K dimensions are `[predecessor_candidate, successor_candidate]`.
No softmax, log-softmax, or other normalization is applied.

DFlash2's own path selection is greedy rather than a global lattice search:

```text
j_0 = argmax_b anchor_final_scores[b]
j_t = argmax_b pairwise_final_scores[t - 1, j_(t - 1), b]
```

The selected candidate indices and token IDs are saved so this path can be
reconstructed offline.

## Dynamic convolutions

The dynamic convolutions are executed, not merely loaded. Every one of the
five `Qwen3DFlash2DecoderLayer` instances calls:

1. `attention_conv.prepare` before self-attention
2. `attention_conv.finish` after self-attention
3. `mlp_conv.prepare` before the MLP
4. `mlp_conv.finish` after the MLP

Each `GroupedDynamicCausalConv` has a learned base kernel with shape
`[2, kernel_size, H]` and a kernel projection that generates per-position,
per-group corrections for its prepare and finish phases.

## Acceptance semantics

The older benchmark field `acceptance_lengths` does not mean exactly the same
thing as DDTree's mathematical matched-prefix length.

- DFlash1 computes the number of matching speculative tokens and stores
  `matching_speculative_tokens + 1`.
- DDTree starts `accepted_indices` with the already available root token and
  stores `len(accepted_indices)`, also
  `matching_speculative_tree_nodes + 1`.
- DFlash2's benchmark stores the number of tokens committed after the anchor
  in that round. Normally this is `matching_speculative_tokens + 1`, but it is
  capped when generation reaches EOS or `max_new_tokens`.

New DFlash2 traces therefore use separate fields:

```text
verifier_matched_draft_tokens
accepted_draft_tokens
committed_tokens_this_round
committed_token_ids
verifier_next_token_id
verifier_next_token_committed
```

`verifier_matched_draft_tokens` is the raw equality prefix from the complete
block-verification output. `accepted_draft_tokens` caps that value at the
actual terminal commit, so matches after EOS or the generation limit are not
treated as meaningful continuation tokens. The latter is the appropriate
direct analogue of DDTree's mathematical matched speculative depth.

## Target-token observability

The verifier processes `[anchor, draft_1, ..., draft_L]`. Its logit at the
anchor predicts the true target token at depth 1. A later verifier logit
predicts the true continuation only while all earlier selected draft tokens
match.

If the first mismatch is at depth `d`, target tokens through depth `d` are
directly observable. Verifier logits after that point are conditioned on an
incorrect draft prefix and are not target-continuation oracles.

Each round stores:

```text
verifier_token_ids               # [L]
directly_observed_target_mask    # [L]
realized_continuation_token_ids  # up to [L]
```

`realized_continuation_token_ids` is annotated after generation from the
eventual committed DFlash2 output. It provides a retrospective continuation
for ordinary Recall@16. Positions after the first mismatch may have been
generated by later verifier calls, so they are explicitly distinguished from
tokens directly observed in the current verification call.

No extra target-model calls are made to obtain a separate length-7
autoregressive oracle.

## Artifact structure

The collector writes a `torch.save` dictionary:

```text
format: "ddtree.dflash2_trace"
format_version: 1
metadata:
    repository and runtime versions
    target and draft checkpoint identifiers/revisions
    checkpoint architecture
    dataset selection and RNG seed
    decoding settings
prompts:
    prompt identity and text
    rendered prompt hash
    input and final output token IDs
    rounds:
        prefix token IDs and absolute positions
        raw unary candidate data and logsumexp
        raw pairwise corrections and final scores
        DFlash2-selected path
        verifier observations and validity mask
        matched and committed token counts
        retrospective realized continuation
```

This is sufficient to implement unary-only and pairwise-conditioned
tree-ranking algorithms offline. It does not contain target logits for
alternative branches and therefore cannot simulate how BF16 target argmaxes
might change under a different verification matrix shape.

## Initial 32-prompt collection

The initial artifact is:

```text
traces/2026-09-03_dflash2_a100-40gb_gsm8k-32/
  gsm8k__Qwen_Qwen3-4B__mgoin_Qwen3-4B-speculator.dflash2__seed0__traces.pt
```

Its SHA-256 is:

```text
158b76739c17243694b6ada4659a640bbcf14d1ffa58bcb666e890b12928d69b
```

The 32 tokenized prompts match the previous DFlash2 GSM8K benchmark exactly.
The trace contains 1,913 decoding rounds and uses these resolved model
revisions:

```text
Qwen/Qwen3-4B: 1cfa9a7208912126459214e8b04321603b3df60c
mgoin/Qwen3-4B-speculator.dflash2:
  e3e7a18e4f541fa3841c2fb0666a7759079ab6fd
```

Validation results:

```text
candidate consistency:                 100%
pairwise score consistency:            100%
DFlash2 selected-path reconstruction:  100%
direct verifier/realized agreement:    100%
trace-mode/non-trace output agreement: 32/32 prompts
```

Candidate recall:

| Depth | Recall@16 | Recall N | Conditional Recall@16 | Eligible rounds \(N_d\) |
| ----: | --------: | -------: | --------------------: | ----------------------: |
| 1 | 99.79% (1909 hits) | 1913 | 99.79% (1909 hits) | 1913 |
| 2 | 97.64% (1864 hits) | 1909 | 99.23% (1677 hits) | 1690 |
| 3 | 93.85% (1785 hits) | 1902 | 98.40% (1411 hits) | 1434 |
| 4 | 90.15% (1711 hits) | 1898 | 97.91% (1173 hits) | 1198 |
| 5 | 87.44% (1657 hits) | 1895 | 98.34% (1006 hits) | 1023 |
| 6 | 83.64% (1585 hits) | 1895 | 97.64% (828 hits) | 848 |
| 7 | 81.61% (1540 hits) | 1887 | 96.67% (668 hits) | 691 |

Recall@16 uses the eventual committed DFlash2 continuation. Conditional
Recall@16 uses only target tokens directly observed by the current verifier
call while the selected draft prefix remains correct. Its denominator \(N_d\)
therefore decreases with depth: a round is eligible at depth \(d\) only when
the first \(d-1\) selected draft tokens matched and generation had not already
terminated.

The artifact averages 3.906 accepted draft tokens and 4.891 committed tokens
per round. The raw verifier equality prefix is slightly higher at 3.916
because it can include matches after EOS; those matches are retained for
auditability but excluded from `accepted_draft_tokens`.

## Limitations for the future tree experiment

- Selector scores are raw additive logits, not normalized transition
  probabilities. A prefix-mass objective must be defined and validated rather
  than assuming the scores can be multiplied directly.
- Unconditional Recall@16 after a mismatch uses the eventual committed output,
  whose later tokens may come from later verifier calls. Conditional recall is
  the stronger directly observed diagnostic.
- The trace cannot predict BF16 argmax changes caused by verifying a different
  tree shape. It can evaluate candidate coverage and ranking against the saved
  continuation, but exact alternative-tree acceptance still requires an
  online verifier experiment.
- Against the realized continuation in this offline trace, candidate
  Recall@16 falls to 81.61% at depth 7. This is the empirical depth-7
  candidate-lattice ceiling for this trace, not an absolute ceiling for an
  online verifier using a different tensor shape.
