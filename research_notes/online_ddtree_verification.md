# Online DDTree verification path

## Current DFlash1 pipeline

`ddtree_generate()` in `ddtree.py` keeps one pending target token at
`output_ids[:, start]`. The target cache contains positions strictly before
`start`.

Each decoding round performs:

1. **DFlash1 draft**
   - `block_output_ids = output_ids[:, start:start + block_size]`
   - The draft model consumes target hidden states, mask-token embeddings, and
     draft positions.
   - `target.lm_head(...)` produces independent unary draft logits with shape
     `[1, block_size - 1, vocab_size]`.
2. **Tree allocation**
   - `build_ddtree_tree(draft_logits[0], tree_budget)` chooses up to
     `tree_budget` speculative nodes.
   - The conceptual root is not counted in the budget.
3. **Tree compilation**
   - `compile_ddtree_tree(...)` linearizes the root followed by the selected
     speculative nodes.
4. **Target verification**
   - The target model verifies the linearized tree in one forward pass with
     custom position IDs and a tree visibility mask.
5. **Acceptance and commitment**
   - `follow_verified_tree(...)` follows target argmax tokens through the
     tree.
   - The accepted root/path cache entries are compacted into a contiguous
     target cache.
   - The first target token not represented by an accepted child becomes the
     next pending root.

## Tree-builder output

`build_ddtree_tree(...)` returns:

- `node_token_ids`: speculative token IDs in verifier order, excluding root.
- `node_depths`: depths in `1..block_size - 1`.
- `parents`: root-inclusive parent indices. Index `0` is the root and has
  parent `-1`; speculative node `i` is stored at verifier index `i + 1`.
- `child_maps`: root-inclusive maps from token ID to verifier child index.
- `visibility_cpu`: a root-inclusive boolean ancestor matrix.
- tree-build timing subtotals.

Parents always precede children. A row in `visibility_cpu` contains the node
itself and its complete ancestor chain.

## Linearized target inputs

`compile_ddtree_tree(...)` creates:

- verifier token order: `[root, node_0, node_1, ...]`
- root position ID: `start`
- speculative node position ID: `start + node_depth`
- attention visibility:
  - all verifier nodes can attend to the cached prefix
  - within the appended tree block, each node can attend only to itself and
    its ancestors

No DFlash2-specific mask or position-ID rule is required. A DFlash2 tree only
needs to be converted to the existing root-inclusive parents, child maps, and
ancestor visibility representation.

## Logit-to-node mapping

The target output keeps the same linear order as the verifier input.
`posterior[0, i]` is the target argmax successor predicted after verifier node
`i`.

`follow_verified_tree(...)` starts at verifier index `0`. If the target token
is present in `child_maps[current_index]`, it moves to that child and repeats.
The first target token without a represented child is the verifier bonus
token.

If `accepted_indices` has length `k`:

- index `0` is always the pending root
- `k - 1` is the number of matched speculative tree nodes
- `k` new output tokens become available after the previous root:
  `k - 1` matched draft tokens plus one verifier bonus token

EOS or `max_new_tokens` can truncate that final sequence. The online DFlash2
tree prototype must therefore report matched draft tokens, committed tokens,
and bonus commitment separately.

## Cache and hidden-state commitment

The target forward appends every linearized verifier node to the dynamic
cache. `compact_dynamic_cache(...)` selects accepted verifier indices and
packs them after the old prefix. The cache is then cropped to the new pending
root position.

Target hidden states are selected with the same accepted verifier indices and
become the context features for the next draft round.

## Smallest DFlash2 insertion point

The reusable boundary is immediately before `compile_ddtree_tree(...)`.

Replace:

```text
DFlash1 full-vocabulary unary logits
    -> build_ddtree_tree(...)
```

with:

```text
DFlash2 draft hidden states
    -> model.propose(..., collect_lattice=True)
    -> audited build_scorers(...)
    -> audited build_best_first_tree(...)
    -> compile_generic_tree_for_verifier(...)
```

The existing `compile_ddtree_tree(...)`, target forward pass,
`follow_verified_tree(...)`, cache compaction, position IDs, and tree attention
mask can remain shared. Unary-K16 and Pairwise-K16 differ only in the scorer
passed to the audited generic best-first builder.
