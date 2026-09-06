# Pairwise-DDTree: formalization and empirical evidence

## Scope and attribution

This note formalizes the frozen research implementation in this repository.
It does not claim a new drafter, selector, verifier, or target-decoding rule.

- **DDTree prior work** supplies the verification-tree framework, the
  fixed-node-budget objective, prefix-mass best-first construction, and target
  tree verification. See Ringel and Romano, *Accelerating Speculative Decoding
  with Block Diffusion Draft Trees*, Section 4.2, Propositions 1--2
  ([arXiv:2604.12989](https://arxiv.org/abs/2604.12989)).
- **DFlash2 prior work/checkpoints** supply the diffusion drafter, unary logits,
  top-\(K\) candidate lattice, and learned predecessor-conditioned selector.
  We did not train DFlash2 or invent its selector.
- **This repository's extension** converts the frozen DFlash2 selector scores
  into mass-preserving conditional extension scores, uses them to allocate a
  multi-branch DDTree, and evaluates the resulting allocation against
  controlled unary baselines.

The central research question is:

> Given a fixed verification-node budget, can predecessor-conditioned
> candidate scores allocate the tree more effectively than independent
> positional marginals?

Candidate support \(K\) and verification-node budget \(B\) are different.
\(K\) is the number of token choices available at each speculative position;
\(B\) is the maximum number of speculative tree nodes sent to the verifier.
The root/anchor token is excluded from \(B\).

## 1. Notation and original unary DDTree

Let:

- \(\mathcal V\) be the target vocabulary;
- \(x\) be the already verified target context;
- \(y_0\) be the verified anchor/bonus token at the tree root;
- \(L\) be the number of speculative positions;
- \(t\in\{1,\ldots,L\}\) index speculative position;
- \(y_t\in\mathcal V\) be a candidate token at position \(t\);
- \(U_t(y)\in\mathbb R\) be the drafter unary logit for token \(y\);
- \(q_t(y)\) be its full-vocabulary softmax probability.

The unary distribution is

\[
q_t(y)=
\frac{\exp U_t(y)}
{\sum_{v\in\mathcal V}\exp U_t(v)}.
\]

For a nonempty speculative prefix
\(y_{1:d}=(y_1,\ldots,y_d)\), unary DDTree uses the factorized surrogate

\[
Q_U(y_{1:d})=\prod_{t=1}^{d}q_t(y_t),
\qquad
\log Q_U(y_{1:d})=\sum_{t=1}^{d}\log q_t(y_t).
\]

This is a tree-allocation surrogate, not a statement that the target language
model is positionally independent. Its approximation is that the allocation
score for \(y_t\) does not depend on earlier speculative choices. For example,
the depth-2 unary score of `York` is identical after `New` and after `Los`.

The original DDTree paper defines a valid tree \(T\) as a prefix-closed set of
nonempty prefixes and the matched length

\[
\alpha_T(Y_{1:L})=\max\{d:Y_{1:d}\in T\}.
\]

Its ideal objective uses the unavailable target continuation distribution.
DDTree instead maximizes expected matched length under the tractable
factorized drafter surrogate \(Q_U\).

## 2. Exact DFlash2 selector semantics

The frozen implementation is in `model/dflash2.py::CandidateSelector`.
For each position, DFlash2 first selects

\[
C_t=\operatorname{TopK}_{y\in\mathcal V} U_t(y),
\qquad K=16.
\]

For a predecessor token \(a\), successor candidate \(b\in C_t\), and the
position's draft hidden state \(h_t\), the selector computes a low-rank
correction

\[
c_t(a,b)=
\left(e^{\mathrm{pred}}_a\odot W h_t\right)^\top
e^{\mathrm{succ}}_b,
\]

where \(e^{\mathrm{pred}}\) and \(e^{\mathrm{succ}}\) are DFlash2's learned
predecessor and successor codebooks and \(W\) is its hidden projection.
The final selector score is

\[
F_t(a,b)=U_t(b)+c_t(a,b).
\]

The collected tensors have the following semantics:

- `candidate_ids[t, j]`: token ID of candidate \(j\in C_t\);
- `unary_scores[t, j]`: \(U_t(C_{t,j})\);
- `unary_logsumexp[t]`:
  \(\operatorname{LSE}_{v\in\mathcal V}U_t(v)\);
- `anchor_final_scores[j]`: \(F_1(y_0,C_{1,j})\);
- `pairwise_final_scores[t-2, i, j]`:
  \(F_t(C_{t-1,i},C_{t,j})\), for \(t\ge2\).

DFlash2 normally follows one greedy selector path: at each position it chooses
the highest-scoring successor and uses it as the next predecessor. Our work
does not change that selector. It materializes the already defined
predecessor/successor score lattice and uses it to allocate a multi-branch
DDTree.

## 3. Why raw pairwise logits are insufficient

Adding arbitrary unnormalized logits along a path need not produce extension
factors in \([0,1]\). A child could then score above its parent, invalidating
the prefix ordering used by DDTree best-first enumeration. Renormalizing the
top-16 scores to sum to one is also undesirable: it would pretend that the
retained candidate set contains all unary probability mass.

Define the unary probability mass retained by \(C_t\):

\[
m_t=\sum_{b\in C_t}q_t(b),
\]

computed stably as

\[
\log m_t=
\operatorname{LSE}_{b\in C_t}U_t(b)
-
\operatorname{LSE}_{v\in\mathcal V}U_t(v).
\]

For predecessor \(a\), normalize DFlash2's final selector scores only within
the retained successor set:

\[
r_t(b\mid a)=
\frac{\exp F_t(a,b)}
{\sum_{b'\in C_t}\exp F_t(a,b')}.
\]

The frozen **Pairwise-MassPreserving** transition is

\[
q_t^P(b\mid a)=m_t\,r_t(b\mid a),
\]

or in log-space,

\[
\log q_t^P(b\mid a)=
\log m_t+
\log\operatorname{softmax}_{b'\in C_t}F_t(a,b).
\]

The two factors have separate roles:

- \(m_t\) preserves how much full-vocabulary unary mass lies inside the
  retained top-16 support;
- \(r_t(\cdot\mid a)\) redistributes that retained mass using DFlash2's
  predecessor-conditioned preference.

Therefore

\[
\sum_{b\in C_t}q_t^P(b\mid a)=m_t\le1.
\]

The missing mass \(1-m_t\) is not assigned to a retained token. Formally, one
can add an unverified residual symbol \(\bot_t\) with probability \(1-m_t\).
This yields a normalized conditional distribution over
\(C_t\cup\{\bot_t\}\), while every verifiable prefix keeps exactly the score
used by the implementation. This augmented-distribution view is useful when
relating the method to DDTree's expected-acceptance surrogate.

## 4. Exact frozen conditional prefix score

The prompt's provisional formula used unary \(q_1(y_1)\) at depth 1. The
frozen online method is slightly different: it also uses DFlash2's
anchor-conditioned selector at depth 1. With the verified root token \(y_0\),

\[
Q_P(y_{1:d}\mid y_0)
=
\prod_{t=1}^{d}q_t^P(y_t\mid y_{t-1}),
\]

and

\[
\log Q_P(y_{1:d}\mid y_0)
=
\sum_{t=1}^{d}
\left[
\log m_t+
\log\operatorname{softmax}_{b'\in C_t}
F_t(y_{t-1},y_t)
\right].
\]

At \(t=1\), `anchor_final_scores` supplies
\(F_1(y_0,\cdot)\). At \(t\ge2\), `pairwise_final_scores` supplies the full
\(16\times16\) predecessor/successor lattice. The repository also contains an
exploratory `PairwiseAfterRootScorer` with unary depth 1, but that is not the
frozen online Pairwise-K16 method evaluated in Steps 6.2, 6.3, and 7.

\(Q_P\) is a first-order conditional (Markov) **tree-allocation surrogate**.
This does not assert that the target model or its continuation distribution is
Markovian.

## 5. Prefix monotonicity

For every retained transition,

\[
0<r_t(b\mid a)<1,\qquad 0\le m_t\le1,
\]

so

\[
0\le q_t^P(b\mid a)\le1.
\]

Consequently,

\[
Q_P(y_{1:d+1}\mid y_0)
=
Q_P(y_{1:d}\mid y_0)
q_{d+1}^P(y_{d+1}\mid y_d)
\le
Q_P(y_{1:d}\mid y_0).
\]

Thus no descendant has a greater score than its parent. This matters because
a max-priority best-first search can emit high-scoring prefixes while
maintaining prefix closure: before a child can be popped, its at-least-as-high
scoring parent has already been emitted.

The implementation validates retained mass, checks every extension log score
is non-positive within \(10^{-5}\) floating-point tolerance, and tests prefix
closure, parent-before-child ordering, depth limits, and the \(B\)-node budget.

## 6. Algorithms

Both methods use the same generic best-first tree builder and the same target
verifier. Only `ExtensionLogScore` changes.

### Unary-DDTree

```text
Input: unary logits U[1:L, V], candidate sets C[1:L], node budget B
Output: prefix-closed speculative tree T

frontier <- empty max-priority queue
T <- empty list

for candidate index j, token b in C[1]:
    delta <- U[1,b] - LSE(U[1,V])
    push(frontier, node(token=b, depth=1, parent=ROOT,
                        path=(j,), log_score=delta))

while |T| < B and frontier is not empty:
    node <- pop maximum-log-score node from frontier
    append node to T
    if node.depth == L:
        continue
    t <- node.depth + 1
    for candidate index j, token b in C[t]:
        delta <- U[t,b] - LSE(U[t,V])
        push(frontier, node(token=b, depth=t, parent=node,
                            path=node.path + (j,),
                            log_score=node.log_score + delta))

return T
```

### Pairwise-DDTree

```text
Input: unary logits U, top-16 sets C, DFlash2 scores F, root token y0,
       node budget B
Output: prefix-closed speculative tree T

compute log_m[t] = LSE(U[t,C[t]]) - LSE(U[t,V])
frontier <- empty max-priority queue
T <- empty list

for candidate index j, token b in C[1]:
    delta <- log_m[1] + log_softmax(F[1,y0,C[1]])[j]
    push(frontier, node(token=b, depth=1, parent=ROOT,
                        path=(j,), log_score=delta))

while |T| < B and frontier is not empty:
    node <- pop maximum-log-score node from frontier
    append node to T
    if node.depth == L:
        continue
    t <- node.depth + 1
    a <- node.token
    for candidate index j, token b in C[t]:
        delta <- log_m[t] + log_softmax(F[t,a,C[t]])[j]
        push(frontier, node(token=b, depth=t, parent=node,
                            path=node.path + (j,),
                            log_score=node.log_score + delta))

return T
```

The contribution changes tree allocation, not target verification. The target
model still verifies the resulting tree (or, for the hybrid 27B target, the
target-selected path) and alone determines committed output tokens.

## 7. Relationship to the DDTree propositions

### What we can claim mathematically

1. **Bounded extensions and monotone prefixes.** The mass-preserving
   transitions lie in \([0,1]\), hence extension cannot increase prefix score.
2. **Prefix closure of best-first output.** Children enter the frontier only
   after their parent is emitted. The first \(B\) popped nodes therefore form
   a prefix-closed tree.
3. **Top-\(B\) enumeration under the frozen score.** With monotone prefix
   scores, the priority-queue algorithm emits the highest-scoring available
   prefixes in non-increasing score order (up to tie-breaking).
4. **Additive expected matched length is not independence-specific.** For any
   normalized continuation distribution \(Q\) and valid tree \(T\),
   \[
   \mathbb E_Q[\alpha_T(Y_{1:L})]
   =
   \sum_{u\in T}Q(Y_{1:|u|}=u).
   \]
   This is the tail-sum identity:
   \(\alpha_T=\sum_{d=1}^{L}\mathbf1\{Y_{1:d}\in T\}\).
   DDTree Proposition 1 instantiates this identity for its factorized \(Q_U\).
5. **Conditional-surrogate optimum over the allowed support.** Using the
   residual-symbol construction above makes \(Q_P\) a coherent conditional
   distribution whose retained-token prefix probabilities equal the
   implementation scores. Therefore selecting its top \(B\) allowed prefix
   masses maximizes the corresponding additive expected-acceptance surrogate
   among prefix-closed trees over the retained candidate lattice. This is the
   same combinatorial argument as DDTree Proposition 2; independence is a
   sufficient way to obtain prefix probabilities, not the key combinatorial
   assumption.

The fifth claim is intentionally narrow: it concerns the explicitly defined
conditional surrogate and the top-16 lattice, not the target distribution.

### What we should not claim yet

- **Not target-optimal.** Neither unary nor Pairwise-DDTree maximizes expected
  acceptance under the unknown target continuation distribution \(p\). The
  experiments evaluate surrogate quality empirically.
- **Not globally optimal outside the retained support.** Pairwise-K16 cannot
  allocate tokens outside DFlash2's top-16 candidate sets. Its optimum is
  relative to that candidate lattice and budget \(B\).
- **Not a theorem that every conditional scorer helps.** Prefix monotonicity
  makes best-first enumeration valid; it does not imply a conditional
  surrogate ranks target prefixes better than unary marginals.
- **Not a proof of throughput improvement.** Acceptance can translate to
  throughput only when verifier parallelism and system overhead permit it.
- **Not a general proof for arbitrary unnormalized pairwise energies.** The
  mass-preserving normalization is what supplies a coherent, bounded
  extension factor. Raw accumulated selector logits do not have this property.
- **Not a claim that the target is first-order Markov.** Only the allocation
  surrogate conditions on the immediately preceding speculative token.

The defensible general observation is:

> DDTree's best-first construction can operate on efficiently computable
> prefix probabilities from a prefix-monotone conditional surrogate; DFlash2
> provides a practical first-order conditional scorer.

For exact expected-acceptance language, the conditional scores should be
shown to arise from a coherent normalized distribution (possibly with
residual outcomes), as done above.

## 8. Mechanism

Unary allocation knows that `York` is individually likely at position 2, so
it gives `York` the same extension score after `New` and `Los`. Pairwise
allocation can make `New` \(\rightarrow\) `York` strong and
`Los` \(\rightarrow\) `York` weak, while favoring
`Los` \(\rightarrow\) `Angeles`. Under a fixed \(B\)-node budget, this can
spend verifier nodes on coherent paths rather than combinations of
individually plausible but mutually inconsistent tokens.

## 9. Empirical evidence as falsification tests

All values below come from committed frozen analysis tables; no benchmark was
rerun for this document.

### A. Same support and same budget

Step 6.2 compares Pairwise-K16 with Unary-K16 on the same DFlash2 lattice and
the same node budget. At \(B=32\):

| Dataset | Unary | Pairwise | Gain | Throughput gain |
|---|---:|---:|---:|---:|
| GSM8K | 4.739 | 5.248 | +0.509 | +8.62 tok/s |
| MATH500 | 4.976 | 5.531 | +0.555 | +9.34 tok/s |
| HumanEval | 4.196 | 4.615 | +0.419 | +6.94 tok/s |
| MT-Bench, controlled history | 3.450 | 3.762 | +0.311 | +5.17 tok/s |

All corresponding clustered 95% intervals exclude zero. Native-trajectory
MT-Bench also remains positive (+0.350 matched tokens at \(B=32\)). These
results suggest the effect is not specific to one task format.

### B. Wider unary support does not explain the gain

Step 6.3 is the strongest mechanistic falsification test. It holds \(B\)
fixed, leaves Pairwise at its native \(K=16\), and gives unary allocation
\(K=16,32,64\) true top-\(K\) candidates from the same full-vocabulary logits.

At \(B=64\):

| Dataset | Unary-K16 | Unary-K64 | Pairwise-K16 |
|---|---:|---:|---:|
| GSM8K | 4.946 | 4.948 | 5.491 |
| MATH500 | 5.190 | 5.196 | 5.750 |

Widening to \(K=64\) substantially improves target-path representability
(depth-7 prefix representability rises from 69.5% to 83.6% on GSM8K and
73.5% to 87.3% on MATH500) but produces essentially no fixed-budget
acceptance gain. Candidate failures become ranking/budget failures: more
choices are available, but the unary score does not identify which dependent
prefixes deserve the limited nodes.

### C. Official-checkpoint and model-scale transfer

Step 7 uses the official pair
`Qwen/Qwen3.8-27B` and `incoai/Qwen3.8-27B-DFlash2`, with 64 GSM8K and 64
MATH500 prompts:

| Dataset | B=16 | B=32 | B=64 |
|---|---:|---:|---:|
| GSM8K Pairwise - Unary | +0.418 | +0.444 | +0.436 |
| MATH500 Pairwise - Unary | +0.404 | +0.439 | +0.434 |

All six paired prompt-bootstrap 95% intervals exclude zero. All 768
Unary/Pairwise method-prompt outputs exactly match sequential target decoding.
Throughput is statistically tied because the correctness-first recurrent
27B verifier advances only the target-selected path and does not parallelize
tree branches. Step 7 therefore supports algorithm/checkpoint/model-scale
generalization, not a 27B production throughput claim.

## 10. Limitations

- The empirical studies use greedy decoding and two DFlash2 checkpoints.
- Pairwise conditioning is first-order and restricted to the selector's
  top-16 lattice.
- Online methods visit different state trajectories. Step 6.3
  failure-decomposition percentages are diagnostic, not a perfectly causal
  frozen-state decomposition.
- BF16 target calls with different tensor shapes can produce near-tie argmax
  differences in the 4B packed verifier. Step 7's recurrent-safe methods were
  checked against sequential decoding exactly.
- The 4B throughput results are prototype single-GPU measurements. The 27B
  implementation prioritizes correctness over tree-parallel performance.
- Task-quality checks in Step 6.2 rule out a large obvious regression but do
  not establish statistical quality equivalence.

## 11. Potential theoretical questions

1. Can approximation error between target prefix mass and unary versus
   conditional surrogate mass be bounded in a way that predicts acceptance
   gain?
2. How should retained mass depend on the predecessor when candidate sets are
   themselves predecessor-conditioned?
3. Can higher-order or state-conditioned surrogates retain efficient
   best-first enumeration without material tree-build overhead?
4. What is the optimal allocation when the verifier cost is not linear in
   node count or depends on tree shape?
5. Can a recurrent-target verifier recover parallel tree verification while
   preserving branch-specific recurrent state?

## Frozen sources

- `research_notes/dflash2_step62_cross_domain_validation.md`
- `analysis/2026-09-04_step62-final/cross_domain_metrics.csv`
- `research_notes/dflash2_step63_wider_unary.md`
- `analysis/2026-09-05_step63-wide-unary/`
- `research_notes/dflash2_step7_27b_validation.md`
- `analysis/2026-09-05_step7-27b/`
- `model/dflash2.py`
- `offline_dflash2_trees.py`
- `dflash2_tree.py`
