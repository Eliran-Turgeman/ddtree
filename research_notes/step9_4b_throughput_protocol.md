# Step 9.4b: dual-drafter, one-H100 throughput protocol

## Research question

Is DFlash2+Pairwise-K16 faster in tokens/s than each of:

1. raw DFlash2 (the checkpoint's native seven-node greedy path),
2. DFlash2+original (released) unary DDTree allocation,
3. raw original DFlash (no tree, the released single-path drafter), and
4. original DFlash+DDTree (the released unary allocation on the original
   drafter)?

This is a throughput question, not an allocation-quality question: Step 6.2
and Step 6.3 already established that Pairwise-K16 improves matched
speculative tokens over the released unary rule on the DFlash2 checkpoint.
Step 9.4b asks whether that acceptance gain survives as an end-to-end
tokens/s gain once wall-clock stage overhead (draft, candidate selection,
tree build, tree compile, verify, commit) is included, and how the DFlash2
family compares in absolute tokens/s to the older, structurally different
original-DFlash family.

## Two checkpoint families, not four independent systems

"Original DFlash" and "DFlash2" are two unrelated drafter architectures for
the same target. Each family is invoked with its own `benchmark.py
--draft-type` value, and each single invocation covers both the family's raw
method and its DDTree-allocated methods in one process (one warmup, one
target/draft load):

- **Original-DFlash family** — `--draft-type dflash --tree-budget 16,32,64`
  with SDPA left as the default target attention (no `--flash-attn`
  argument) produces method keys `dflash` (raw), `ddtree_tb16`, `ddtree_tb32`,
  `ddtree_tb64` in a single `.pt` artifact.
- **DFlash2 family** — `--draft-type dflash2 --dflash2-tree-configs
  "dflash2_original_ddtree:7,16,32,64;dflash2_pairwise_k16:7,16,32,64;
  dflash2_unary_k16:32,64"` produces method keys `dflash2` (raw),
  `dflash2_original_ddtree_tb{7,16,32,64}`, `dflash2_pairwise_k16_tb{7,16,32,64}`,
  `dflash2_unary_k16_tb{32,64}` in a single `.pt` artifact.

Both families always target the same `Qwen/Qwen3-4B` weights at the same
pinned revision, so the two artifacts for a given `<dataset>_<suffix>` pair
are directly comparable: same prompts (same dataset, same `--max-samples`
seed-0 shuffle, same target checkpoint), different drafters. The launch
script writes them side by side as
`<dataset>_original_<suffix>.pt` and `<dataset>_dflash2_<suffix>.pt` under
one `artifacts/step9_4b/<commit>/<profile>/` directory so they are paired by
construction under commit/profile/dataset.

**Attention implementation is not uniform across families.** The target
model runs SDPA in both families (the default when `--flash-attn` is
omitted). The *draft* model does not: `benchmark.py` hardcodes
`draft_attn_implementation = "flash_attention_2"` whenever
`--draft-type dflash` is selected, independent of `--flash-attn`, so the
original drafter always runs FlashAttention2 and requires `flash_attn` to be
installed. The DFlash2 drafter runs SDPA. Do not describe this protocol as
"both drafts on SDPA" — only the DFlash2 draft is SDPA; the original draft
is FlashAttention2 while its target is SDPA.

## Frozen configuration

| Item | Value |
|---|---|
| Target | `Qwen/Qwen3-4B` |
| Target revision | `1cfa9a7208912126459214e8b04321603b3df60c` (frozen for Step 6.2/6.3; see below) |
| Original drafter | `z-lab/Qwen3-4B-DFlash-b16` |
| Original drafter revision | `b74e3a329c4d963783143b1e970d95b002be72bd` (verified on the Hugging Face Hub for this protocol; see below) |
| DFlash2 drafter | `mgoin/Qwen3-4B-speculator.dflash2` |
| DFlash2 drafter revision | `e3e7a18e4f541fa3841c2fb0666a7759079ab6fd` (frozen for Step 6.3) |
| GPU | One H100 80 GB |
| Target attention (both families) | SDPA |
| Original draft attention | FlashAttention2 (hardcoded by `benchmark.py` for `--draft-type dflash`; not SDPA) |
| DFlash2 draft attention | SDPA |
| Dtype | BF16 |
| Temperature | 0 (greedy) |
| Maximum new tokens | 256 |
| Timing repetitions | 1 (`one`/`matrix`/`domains`), 3 (`stability`), `FULL_TIMING_REPETITIONS` (`full`, default 1) |
| GSM8K | 128 prompts |
| MATH500 | 128 prompts |
| HumanEval | 164 prompts (full test split) |
| MT-Bench | 80 conversations, controlled and native trajectories |

Dataset revisions are pinned by the launcher and recorded in each artifact:
GSM8K `740312add88f781978c0658806c59bc2815b9866`, MATH500
`6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be`, HumanEval
`7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544`, and MT-Bench
`e3a795c5e9a82ee40611c416b8a7786c73198991`.

### Revision provenance

`research_notes/dflash2_step62_cross_domain_validation.md` and
`research_notes/dflash2_step63_wider_unary.md` both record
`Qwen/Qwen3-4B` at revision `1cfa9a7208912126459214e8b04321603b3df60c`, and
`dflash2_step63_wider_unary.md` records `mgoin/Qwen3-4B-speculator.dflash2`
at revision `e3e7a18e4f541fa3841c2fb0666a7759079ab6fd`. Both are pinned
directly in `run_step9_4b_throughput.sh` (overridable via the
`TARGET_REVISION` / `DFLASH2_DRAFTER_REVISION` environment variables, for the
rare case the upstream ref moves).

No revision for `z-lab/Qwen3-4B-DFlash-b16` was ever recorded anywhere in
this repository: `README.md`, `run_benchmark.sh`, and every prior run/trace
artifact reference it unpinned (implicitly "main"). Rather than invent one,
its current Hugging Face Hub revision was verified for this protocol —
`b74e3a329c4d963783143b1e970d95b002be72bd` — and is now pinned directly in
`run_step9_4b_throughput.sh` as `ORIGINAL_DRAFTER_REVISION` (overridable via
the same-named environment variable if the upstream ref moves). The script
still hard-fails if `ORIGINAL_DRAFTER_REVISION` is ever cleared to empty, so
an accidental override cannot silently fall back to an unpinned "main".

## Methods

Original-DFlash family:

- `dflash`: the released drafter's native single-path proposal, no tree.
- `ddtree_tb16` / `ddtree_tb32` / `ddtree_tb64`: released DDTree unary
  allocation at verification-node budgets 16, 32, 64.

DFlash2 family:

- `dflash2`: the checkpoint's native seven-node greedy path.
- `dflash2_original_ddtree_tb{7,16,32,64}`: released DDTree unary scoring
  with \(K=\min(B,|V|)\).
- `dflash2_pairwise_k16_tb{7,16,32,64}`: mass-preserving conditional
  allocation on the checkpoint's native top-16 selector lattice.
- `dflash2_unary_k16_tb{32,64}`: a diagnostic that holds candidate support
  fixed at K=16 against Pairwise-K16 at the wider budgets, isolating the
  allocation-rule effect from the candidate-support effect.

In controlled-history MT-Bench, the sequential target baseline supplies the
assistant turn added to shared history. This keeps every method and both
separately launched drafter families on the same canonical turn-2 prompt.
Native-history MT-Bench instead advances each method with its own output and
is analyzed as a separate trajectory regime.

## Timing repetitions

`benchmark.py --timing-repetitions N` repeats each method's timing
measurement N times per prompt/turn from the same input context. Only the
first repetition advances conversation history; every repetition is compared
with the corresponding sequential-baseline repetition, and every timing is
preserved in the
saved artifact for future variance analysis. `run_step9_4b_throughput.sh`
passes this to both family calls with stage-dependent defaults:

| Stage | Timing repetitions |
|---|---|
| `one` | 1 |
| `matrix` | 1 |
| `domains` | 1 |
| `stability` | 3 |
| `full` | `FULL_TIMING_REPETITIONS` environment variable, default 1 |

The `stability` stage's 3 repetitions per prompt exist specifically to
decide `full`'s repetition count: inspect per-repetition timing variance in
the `stability` artifacts (GPU clock/thermal variance on a shared H100 can
be non-trivial) before launching `full`. If that variance is large relative
to the throughput differences reported in Step 6.2/Step 6.3 for matched-token
acceptance, override `FULL_TIMING_REPETITIONS=2` or `FULL_TIMING_REPETITIONS=3`
before freezing the `full` run — do not launch `full` with the silent
default of 1 repetition if `stability` showed it would be noise-dominated.

## Allocation-trace collection is disabled in every measured artifact

`benchmark.py --collect-allocation-data` records per-step lattice/tree
allocation bookkeeping (the data Step 8's allocation-quality analysis
consumes) via CPU-side copies executed inside the timed generation path.
That instrumentation cost is not symmetric across the two families: only
the DFlash2 tree-construction path has anything to collect, so enabling it
would add CPU overhead to DFlash2/Pairwise's timed loop with no equivalent
overhead added to the original-DFlash family, unfairly depressing DFlash2's
measured tokens/s relative to original DFlash for a reason that has nothing
to do with either family's real inference cost.

**No stage of `run_step9_4b_throughput.sh` (`one`, `matrix`, `domains`,
`stability`, `full`) passes `--collect-allocation-data` to either family's
`benchmark.py` invocation.** Step 9 is a throughput protocol only; Step 8
(`run_step8_27b_protocol.sh` / `research_notes/step8_27b_allocation_protocol.md`)
already owns allocation-trace collection and analysis, on its own hardware
and its own timing budget, and is the correct place to look for
allocation-quality evidence. If a trace diagnostic against the Step 9.4b
checkpoints/datasets is ever wanted, it must be run as a separate, clearly
labeled, non-timed-comparison invocation — e.g. writing to a distinct
`diagnostic/` subdirectory that `analyze_step9_4b_throughput.py`'s `--pair`
paths never point at — and its tokens/s numbers must never be reported
alongside, or substituted into, the `full`-stage throughput comparisons
above.

## Primary and secondary endpoints

Both families report two throughput metrics per response:
`tokens_per_second_end_to_end` (total generated tokens divided by
`total_generation_time`, which includes prefill/time-to-first-token plus
decode) and `decode_only_tokens_per_second` (derived from
`time_per_output_token`, decode phase only, excluding prefill).

**The headline/primary metric is end-to-end tokens/s, including prefill.**
`analyze_step9_4b_throughput.py`'s paired-comparison bootstrap
(`paired_throughput_comparisons.csv`) is computed exclusively on
`tokens_per_second_end_to_end`. Decode-only tokens/s
(`decode_only_tokens_per_second`, reported per method in
`method_metrics.csv` as `mean_decode_only_tokens_per_second` /
`speedup_vs_baseline_decode_only`) is secondary: it isolates the decode
phase from prefill cost, which is useful for understanding *why* two
methods differ (e.g. tree-build/verify stage overhead in
`timing_decomposition.csv`) but is not itself the research answer, since a
method that only improves decode while adding equal or larger prefill
overhead is not actually faster for a user.

**Primary endpoint:** the paired prompt-level difference in end-to-end
tokens/s between `dflash2_pairwise_k16_tb16` and
`dflash2_original_ddtree_tb16` — both against the released top-16 candidate
lattice at equal node budget B=16, `comparison_type=controlled` — the
narrowest, most controlled comparison, isolating the allocation rule with
candidate support and node budget both held fixed. This mirrors the Step 8
primary-endpoint convention. This is the only row in
`paired_throughput_comparisons.csv` with `is_primary=True`.

**Secondary comparisons** (all `is_primary=False`; every row reports
`comparison_type` so `controlled` and `cross_drafter` evidence are never
conflated):

`comparison_type=controlled` (same DFlash2 drafter, only the tree
method/budget differs):

- `dflash2_pairwise_k16_tb{7,32,64}` vs `dflash2_original_ddtree_tb{7,32,64}`
  — does the primary-endpoint direction hold away from B=16? (At B=7 both
  methods use the same node count as DFlash2 greedy; at B=32/64 original
  DDTree gets wider unary candidate support than Pairwise-K16, so a Pairwise
  win there is a stronger claim than at B=16.)
- `dflash2_pairwise_k16_tb7` vs `dflash2` (raw DFlash2 greedy path) — does
  Pairwise beat the checkpoint's own native proposal at matched node count?
- `dflash2_pairwise_k16_tb{32,64}` vs `dflash2_unary_k16_tb{32,64}` —
  candidate-support-controlled allocation-rule comparison.

`comparison_type=cross_drafter` (different drafter checkpoints — these
conflate the allocation rule with the whole drafter architecture, including
the FlashAttention2-vs-SDPA draft attention difference noted above; treat as
descriptive context, not as evidence isolating Pairwise's contribution):

- `dflash2_pairwise_k16_tb{16,32,64}` vs `dflash` (raw original drafter, no
  tree).
- `dflash2_pairwise_k16_tb{16,32,64}` vs `ddtree_tb{16,32,64}` (original
  drafter + released DDTree, matched budget).

The analyzer also emits paired comparisons of every non-baseline method
against the sequential target baseline in its own process, plus original
DDTree versus raw DFlash at B=16/32/64. Cross-drafter rows report both the
raw tokens/s difference and a baseline-normalized speedup difference, using
each process's sequential baseline to reduce process-state bias.

## Required staged execution

Run each stage only after the prior artifact pair passes inspection. Each
stage is a stop/go gate for the next:

```bash
bash run_step9_4b_throughput.sh one
bash run_step9_4b_throughput.sh matrix
bash run_step9_4b_throughput.sh domains
bash run_step9_4b_throughput.sh stability
bash run_step9_4b_throughput.sh full
```

The script refuses to run with tracked worktree changes and refuses to run
without a visible-GPU count of exactly one. Artifacts are stored under
`artifacts/step9_4b/<commit>/<profile>/` as `<dataset>_original_<suffix>.pt`
/ `<dataset>_dflash2_<suffix>.pt` pairs, with atomic partial checkpoints that
resume only when every argument matches (inherited from `benchmark.py`).
Within each dataset/suffix pair, `run_step9_4b_throughput.sh` launches the
two family processes in a documented, deterministic order (original-family
first or DFlash2-family first) rather than always the same order, so that
neither family systematically benefits from launching into a "cold" versus
GPU-clock-ramped/thermally-settled state; see "Cross-family process order"
below for exactly which stages vary the order and how.

- **`one`** — one-prompt model-family smoke. One GSM8K prompt, B=7 only
  (the two methods that share DFlash2's own node count) for the DFlash2 side,
  B=16/32/64 for the original side. Confirms both drafter families load,
  generate, and save without crashing before spending any real budget.
  Stop/go: both `.pt` files exist, contain all expected method keys for the
  smoke config, and `matches_sequential_baseline` (or the equivalent exact
  greedy-match field) is true for every produced response. If either family
  fails to load or produces a mismatched token, stop and fix the environment
  before proceeding.
- **`matrix`** — two-prompt full matrix. Two GSM8K prompts across every
  method/budget in both families. Stop/go: both artifacts contain every
  method key listed above with no missing budgets, and per-method tokens/s
  is positive and finite for every prompt.
- **`domains`** — four-sample all domains. Four samples from GSM8K, MATH500,
  HumanEval, and MT-Bench (both controlled-history and native-trajectory
  MT-Bench) for both families. Stop/go: no dataset is missing either family's
  artifact, MT-Bench controlled and native artifacts are both present and
  distinct, and the primary-endpoint sign (Pairwise-K16 tb16 vs
  Original-DDTree tb16 tokens/s) is at least directionally stable across
  domains before committing GPU time to the 16-prompt stage.
- **`stability`** — 16-prompt timing stability. GSM8K only, at the full
  `max-new-tokens=256`, both families, to check that per-prompt tokens/s
  variance is low enough that the `full` stage's paired bootstrap will be
  informative rather than dominated by run-to-run timing noise. This stage
  runs the pair **twice**, with opposite family launch order and distinct
  `ab`/`ba` suffixes (`gsm8k_original_ab.pt`+`gsm8k_dflash2_ab.pt`,
  original-family launched first; `gsm8k_original_ba.pt`+`gsm8k_dflash2_ba.pt`,
  DFlash2-family launched first) — see "Cross-family process order" below.
  Stop/go: the coefficient of variation of tokens/s within each method
  across the 16 prompts is small enough that a real effect is unlikely to
  be swamped by noise, **and** the AB and BA artifacts do not disagree on a
  method's tokens/s by more than that same noise floor (inspect both
  explicitly, do not just trust a threshold — GPU clock/thermal variance on
  a shared H100 can dominate small speedups, and so can a launch-order
  effect if one exists).
- **`full`** — the empirical run: GSM8K-128, MATH500-128, HumanEval-164,
  MT-Bench-80 (controlled and native), both families, `max-new-tokens=256`,
  family launch order alternated deterministically by dataset (see
  "Cross-family process order" below). Only this stage's output is reported
  as a result.

Every stage before `full` is a smoke or stability check, not a result. Do
not report `one`/`matrix`/`domains`/`stability` throughput numbers as
evidence for the research question; they exist only to catch a broken
environment, a missing method, or excessive timing noise before spending the
full budget.

## Cross-family process order

Each dataset/suffix pair is produced by two separate, sequential
`benchmark.py` processes (one per family). Always launching the same family
first would confound any measured tokens/s difference with whatever
systematic GPU-state drift correlates with launch position on a shared
H100 — clock-boost ramp-up, thermal creep over a long sequential run, or
kernel/cache warmth left over from the previous process. This confound is
secondary to, and does not affect, the primary within-DFlash2 controlled
endpoint (Pairwise-K16-tb16 vs Original-DDTree-tb16), since both methods
there run inside the *same* `benchmark.py` process and therefore share
identical launch-order exposure. It matters specifically for the
`comparison_type=cross_drafter` rows and for any claim about a family's
*absolute* tokens/s, both of which compare across the two separate
processes.

`run_paired_dataset()` in `run_step9_4b_throughput.sh` takes an explicit
`order` argument (`original-first` or `dflash2-first`) controlling which
family's process is launched first for that call. Policy by stage:

- **`one` / `matrix` / `domains`** — stay at the default, `original-first`,
  for every call. These stages are smoke/coverage/sign-stability checks,
  not reported timing results (see above), so the order confound they
  cannot resolve does not need resolving here.
- **`stability`** — launches the GSM8K pair **twice**, once
  `original-first` (suffix `ab`) and once `dflash2-first` (suffix `ba`).
  Pairing `gsm8k_original_ab.pt:gsm8k_dflash2_ab.pt` and
  `gsm8k_original_ba.pt:gsm8k_dflash2_ba.pt` under the same `--pair` label
  (e.g. `--pair GSM8K=...ab.pt:...ab.pt --pair GSM8K=...ba.pt:...ba.pt`)
  lets `analyze_step9_4b_throughput.py` merge them as repeated paired
  artifacts for one label, combining both launch orders into a single
  stability read rather than treating order as a second independent
  variable to analyze separately.
- **`full`** — alternates order deterministically by dataset position so no
  single family is always launched first across the stage:

  | Dataset (suffix) | Order |
  |---|---|
  | GSM8K (`controlled`) | original-first |
  | MATH500 (`controlled`) | dflash2-first |
  | HumanEval (`controlled`) | original-first |
  | MT-Bench (`controlled`) | dflash2-first |
  | MT-Bench (`native`) | original-first |

  This also gives MT-Bench's `controlled` and `native` suffixes opposite
  orders from each other, as requested, so neither trajectory mode is
  confounded with a fixed launch position either.

This alternation bounds, but does not eliminate, an order confound: it is a
single alternating sequence on one GPU, not a randomized or replicated
design. Treat any `cross_drafter` or absolute-tokens/s finding that is only
marginal relative to Step 6.2/6.3 effect sizes as inconclusive rather than
confirmed, and prefer the `controlled`-comparison primary/secondary
endpoints (immune to this confound by construction) wherever the research
question can be answered from those alone.

## Full-run analysis

`analyze_step9_4b_throughput.py` exists and pairs each dataset's two
artifacts via `--pair DATASET_LABEL=ORIGINAL_PATH:DFLASH2_PATH` (repeatable
with the same label to merge multiple artifact pairs for one dataset, e.g.
extra reruns, aggregated within prompt clusters and never treated as
independent prompts):

```bash
python analyze_step9_4b_throughput.py \
  --expected-commit <commit> \
  --pair GSM8K=artifacts/step9_4b/<commit>/full/gsm8k_original_controlled.pt:artifacts/step9_4b/<commit>/full/gsm8k_dflash2_controlled.pt \
  --pair MATH500=artifacts/step9_4b/<commit>/full/math500_original_controlled.pt:artifacts/step9_4b/<commit>/full/math500_dflash2_controlled.pt \
  --pair HumanEval=artifacts/step9_4b/<commit>/full/humaneval_original_controlled.pt:artifacts/step9_4b/<commit>/full/humaneval_dflash2_controlled.pt \
  --pair MT-Bench-controlled=artifacts/step9_4b/<commit>/full/mt-bench_original_controlled.pt:artifacts/step9_4b/<commit>/full/mt-bench_dflash2_controlled.pt \
  --pair MT-Bench-native=artifacts/step9_4b/<commit>/full/mt-bench_original_native.pt:artifacts/step9_4b/<commit>/full/mt-bench_dflash2_native.pt \
  --output-dir analysis/step9-4b-throughput
```

To analyze the `stability` stage's AB/BA artifact pairs together (see
"Cross-family process order" above), repeat `--pair` with the same label for
both orders instead of the single `--pair GSM8K=...` line above:

```bash
python analyze_step9_4b_throughput.py \
  --expected-commit <commit> \
  --allow-partial \
  --pair GSM8K=artifacts/step9_4b/<commit>/stability/gsm8k_original_ab.pt:artifacts/step9_4b/<commit>/stability/gsm8k_dflash2_ab.pt \
  --pair GSM8K=artifacts/step9_4b/<commit>/stability/gsm8k_original_ba.pt:artifacts/step9_4b/<commit>/stability/gsm8k_dflash2_ba.pt \
  --output-dir analysis/step9-4b-stability
```

`--expected-commit` and `--output-dir` are required. `--target-revision`,
`--original-draft-revision`, and `--dflash2-draft-revision` all default to
the same pinned constants used by `run_step9_4b_throughput.sh` (including
`b74e3a329c4d963783143b1e970d95b002be72bd` for
`z-lab/Qwen3-4B-DFlash-b16`), so they only need to be passed explicitly when
overriding for a moved upstream ref. `--bootstrap-samples` defaults to
10,000. For smoke/stability profiles, add `--allow-partial` (relaxes decode
settings, sample-count, and method-matrix checks, and permits a derived
end-to-end-timing fallback where the timing-runner has not yet reached a
given generator — see the module docstring). Derived timing omits the first
draft round, so it underestimates total time and overestimates tokens/s;
never report an `--allow-partial` throughput result as final evidence.

Validation (`validate_artifact` / `validate_cross_family`) rejects: a dirty
or unexpected commit; wrong target/original-draft/DFlash2-draft revision;
`draft_type`/`draft_name_or_path` not matching the expected family; an
unrecognized dataset; a target attention implementation other than SDPA;
a draft attention implementation other than FlashAttention 2 for original
DFlash or SDPA for DFlash2; in strict mode, wrong
temperature/`max_new_tokens`, an incomplete sample count against the frozen
per-dataset counts, a `tree_budget`/`dflash2_tree_configs` argument that
does not match the frozen method matrix, or any response missing a required
method key; and, across the paired original/DFlash2 artifacts, mismatched
completed dataset indices, mismatched `trajectory_mode`, mismatched
runtime-critical settings (Python/PyTorch/Transformers/CUDA/GPU/decode-timing
convention), or diverging prompt hashes/baseline outputs where the
trajectories should be identical.

The analyzer validates the native attention split explicitly: SDPA for both
targets, FlashAttention 2 for the original DFlash drafter, and SDPA for the
DFlash2 drafter.

It writes, under `--output-dir`:

- `method_metrics.csv` — one row per (dataset, family, method): prompt
  count, `mean_end_to_end_tokens_per_second`,
  `mean_decode_only_tokens_per_second`, `speedup_vs_baseline_end_to_end`,
  `speedup_vs_baseline_decode_only`, mean matched/committed tokens per
  round, mean target calls, mean time-to-first-token, mean total generation
  time, per-method peak memory, and the fraction of responses whose end-to-end timing
  came from the real runner versus a derived (`--allow-partial`) fallback.
  `peak_allocated_gib` is the meaningful per-method high-water mark.
  `peak_reserved_gib` also reflects PyTorch's process-wide caching allocator
  pool and must not be interpreted as isolated memory demand for that method.
- `paired_throughput_comparisons.csv` — one row per (dataset, comparison
  pair) for every pair listed under "Primary and secondary endpoints" above
  that both methods have data for: `is_primary`, `comparison_type`
  (`controlled`/`cross_drafter`/`within_family_vs_sequential`), prompt count,
  mean end-to-end tokens/s
  difference with a 10,000-sample paired bootstrap 95% CI, mean relative %
  difference, and improve/tie/hurt prompt counts. Cross-drafter rows also
  include a paired baseline-normalized speedup difference and CI.
- `timing_decomposition.csv` — the same comparison pairs, decomposed into
  matched/committed token gain, target calls avoided, draft-stage
  ms/call on each side, summed candidate-selection/tree-build/tree-compile
  overhead change,
  verify-stage ms/call on each side and its change, and the resulting net
  end-to-end tokens/s gain — for explaining *why* a throughput difference in
  `paired_throughput_comparisons.csv` arose, not for an independent claim.
- `correctness.csv` — one row per (dataset, family, method): prompt count,
  exact-output match rate against that family's own sequential baseline,
  truncated rate (hit `max_new_tokens`), and stopped rate.
- `provenance.json` — generation timestamp, expected commit, all resolved
  revisions, `--allow-partial`/bootstrap settings, one entry per source
  artifact (dataset label, family, path, SHA-256, dataset, trajectory mode,
  sample count), and a count of how many method/cluster metrics used the
  real runner-provided end-to-end timing versus the derived fallback.

## Interpretation boundaries

- **The headline result is end-to-end tokens/s (`tokens_per_second_end_to_end`),
  including prefill/time-to-first-token.** This is a change from earlier
  Step 8/Step 6.x protocols in this repository, which reported decode-only
  throughput as primary; see "Timing repetitions" and "Primary and secondary
  endpoints" above. Decode-only tokens/s (`decode_only_tokens_per_second`,
  derived from `time_per_output_token`, prefill excluded) is reported
  alongside every method and used in `timing_decomposition.csv` to explain
  *why* an end-to-end difference arose, but it is not itself the research
  answer: a method that only speeds up decode while adding equal or larger
  prefill/tree-setup overhead has not actually improved end-to-end tokens/s
  for a user. Do not report decode-only numbers as the primary finding.
- The original-DFlash and DFlash2 families differ in drafter architecture,
  parameter count, block size, and draft attention implementation:
  the original drafter always runs FlashAttention2 (hardcoded by
  `benchmark.py` for `--draft-type dflash`, independent of `--flash-attn`),
  while the DFlash2 drafter runs SDPA. The target model runs SDPA in both
  families. Any `comparison_type=cross_drafter` tokens/s difference reflects
  the whole system, including this attention-implementation difference, not
  just the allocation rule. Treat `cross_drafter` rows in
  `paired_throughput_comparisons.csv` as descriptive context, not as
  evidence isolating Pairwise's allocation-rule contribution — only the
  `comparison_type=controlled` rows (Pairwise vs Original-DDTree vs
  Unary-K16 vs raw DFlash2, same DFlash2 drafter, same candidate lattice
  where noted) isolate that.
- A single H100 is a single hardware sample. Thermal state, clock boost
  behavior, and co-tenancy on a shared machine can shift absolute tokens/s
  run to run; the `stability` stage exists to bound this before trusting the
  `full`-stage paired comparisons, but it does not eliminate the underlying
  variance. Repeat the `full` stage if the `stability` stage's variance
  looks large relative to the effect sizes reported for prior Pairwise
  acceptance gains (Step 6.2, Step 6.3).
- Published original-DDTree 30B throughput numbers are external context for
  the original-DFlash family, not a directly comparable baseline: they were
  measured on different hardware, a different target size, and (per Step 8)
  a different verifier regime for large targets.
- Quality (task correctness) is not measured by `run_step9_4b_throughput.sh`
  or `analyze_step9_4b_throughput.py` themselves (`correctness.csv` only
  reports exact-match-vs-baseline and truncation rates, not task
  correctness). See "Quality evaluation" below for the explicit commands to
  run task scoring on the same `full`-stage artifacts. Step 6.2 already
  found no consistent quality loss tracking the Pairwise acceptance gain on
  the DFlash2 checkpoint; that finding does not by itself license skipping
  quality evaluation here, since Step 9.4b uses a different (dual-family)
  artifact set and a fresh commit.

## Quality evaluation

Throughput is not evidence about task correctness. Run quality evaluation
against the same `full`-stage artifacts as an explicit, separate step,
reusing `evaluate_step62_quality.py` (already parameterized for the Step
9.4b method keys: `dflash`, `ddtree_tb{16,32,64}`, `dflash2`,
`dflash2_original_ddtree_tb{7,16,32,64}`, `dflash2_pairwise_k16_tb{7,16,32,64}`,
`dflash2_unary_k16_tb{32,64}` all resolve via its `STEP9_METHOD_PATTERNS`)
and `summarize_evalplus_results.py`. Run once per family artifact (original
and DFlash2 use separate `.pt` files, so separate invocations):

GSM8K / MATH500 (scored directly with `math-verify`):

```bash
DEPS="$HOME/step9-4b-mathverify-deps"
mkdir -p "$DEPS"
python -m pip install --target "$DEPS" math-verify==0.9.0

for family in original dflash2; do
  for task in gsm8k math500; do
    PYTHONPATH="$DEPS" python evaluate_step62_quality.py \
      "artifacts/step9_4b/<commit>/full/${task}_${family}_controlled.pt" \
      "analysis/step9-4b-throughput/quality/${family}/${task}" \
      --task "${task}"
  done
done
```

HumanEval (exact text exported, then sanitized and executed in the sandboxed
EvalPlus Docker image — never execute generated code directly on the host):

```bash
IMAGE=ganler/evalplus@sha256:26b118098bef281fe8dfe999bf05f1d5b45374b4e6c00161ec0f30592aef4740

for family in original dflash2; do
  Q="$PWD/analysis/step9-4b-throughput/quality/${family}/humaneval"
  python evaluate_step62_quality.py \
    "artifacts/step9_4b/<commit>/full/humaneval_${family}_controlled.pt" \
    "$Q" --task humaneval

  for src in "$Q"/*.jsonl; do
    name=$(basename "$src" .jsonl)
    docker run --rm -v "$Q:/app" "$IMAGE" \
      evalplus.sanitize --samples "/app/$name.jsonl"
    docker run --rm -v "$Q:/app" "$IMAGE" \
      evalplus.evaluate --dataset humaneval \
      --samples "/app/$name-sanitized.jsonl" --base-only
  done
  python summarize_evalplus_results.py "$Q" "$Q/humaneval_summary.csv"
done
```

MT-Bench (response export only; `evaluate_step62_quality.py --task
mt-bench` writes `mt_bench_outputs.csv` per family/trajectory-mode but does
not score it — MT-Bench quality requires an external judge and is out of
scope for this protocol):

```bash
for family in original dflash2; do
  for suffix in controlled native; do
    python evaluate_step62_quality.py \
      "artifacts/step9_4b/<commit>/full/mt-bench_${family}_${suffix}.pt" \
      "analysis/step9-4b-throughput/quality/${family}/mt-bench-${suffix}" \
      --task mt-bench
  done
done
```

Compare the `original` and `dflash2` family quality outputs side by side per
method, the same way `method_metrics.csv` pairs their throughput: a
throughput win from Pairwise-K16 is only actionable if it does not come with
a quality regression relative to that family's own target-only baseline.
