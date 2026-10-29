# kq-derived-multiseq — RESOLVED: promoted into block 05 (r172)

**Status: PROMOTED (2026-10-29, r172, block 05).**  PR #247 (`@overdoingism`) extended block 05's derived kq mask to
ubatches holding several sequences; it was folded into **block 05** of the delivery set and released as
`v15-6050b467c-r172`.  This directory is the campaign record: the contributor's original handover (`README.md`, the
`git am` patch, the diagnostic oracle and the randomized counter test) is preserved verbatim, alongside the resolution.

## What shipped

Sequence `k` of a ubatch gets its own position range - its cells and its tokens' `[lo, hi]` both shift by
`k * 2^23` (`KQ_DERIVED_SEQ_STRIDE`) - so the derived mask's single predicate `tok_lo <= cell_pos <= tok_hi` stays
exact for a batch of several sequences and **no FA kernel changes**.  Taken only with one KV stream, every token in
exactly one sequence, **no cell carrying two of the batch's sequences** (a prefix copied with
`llama_memory_seq_cp`) and every position below the stride; a multi-sequence batch of at most **32 tokens** (the
multi-slot verify width) and every other refused shape keep the packed mask.  `llama_kv_cells` gained an O(1)
shared-cell counter (`get_n_multi_seq()`) so the sharing check costs nothing unless something was copied.
`kq_mask_packed_reachable()` lost its `n_seq_max > 1` term, and `decode()` latches `kq_mask_multiseq_seen` on the first
pre-split batch the form cannot serve and re-reserves once with the packed mask (the r160 2-D latch pattern).
`LLAMA_KQ_MASK_DERIVED_MULTISEQ=0` restores the single-sequence rule and the packed reserve.

## Why it was promoted

* **It gives back memory the reserve was wasting**: -1.031 GiB device and -0.937 GiB pinned host on a 2-slot unified
  245K-context server (dense 27B), i.e. the contributor's 1.03 / 0.94 GiB reproduced exactly on this box.
* **It is numerics-neutral by construction and by measurement**: no kernel, and the derived predicate *is* the packed
  one - the author's cell-by-cell oracle reads **0 mismatches** on dense 27B (1 and 2 GPU tensor), Qwen3.5-4B (hybrid),
  Gemma4-12B and gpt-oss-20b (iSWA), over perplexity batches at 4 sequences each; the prefill-logit gate reads
  **0.000000 / 100.000 %** at f16 and q8_0 KV; single-sequence greedy text and `plain == draft-mtp` are byte-identical.
* **Prefill is neutral** in the three shapes where the derived path's batch-wide skip window could have cost: two
  concurrent 16K prefills (26.96 vs 27.23 s at 1 GPU, 17.07 vs 17.09 s at 2 GPU tensor) and a 112K-resident decode
  beside a 16K prefill (13.97 vs 13.96 s).  The removed host fill is the larger term.
* **It is safe where it does not apply**: a memory `kq_mask_kvs()` does not cover keeps the packed reserve, and a graph
  that also holds an `allow_derived == false` builder (qwen4exp/QSA, MLA) creates and fills the packed mask anyway -
  Flash-Next measures identical VRAM and RSS.  Correctness never depends on the reserve latch, because
  `kq_mask_derivable()` re-runs the whole predicate per ubatch.

## What the fold changed (beyond the submitted patch)

* `kq_mask_kvs()` is resolved once and cached instead of returning a fresh `std::vector` (and running 2-3
  `dynamic_cast`s) on every `decode()` batch until a latch fires; `batch_needs_packed_multiseq()`'s per-call
  allocations became fixed-size arrays.
* The block-10 comment block above `kq_mask_packed_reachable()` (which claimed every multi-sequence batch always needs
  the packed mask) was rewritten with the r172 rule, including that the reserve's own measure batch carries
  `n_seq_max` sequences.
* Delivery documentation: `GREEDY-PURITY.md` §70, `ENVIRONMENT.md` (`LLAMA_KQ_MASK_DERIVED_MULTISEQ`, and the two
  existing kq-mask rows corrected), `patches/README.md` (block 05), `MANIFESTS.md` (row 05 + the r172 gate row),
  `AGENTS.md` (the r160 bullet + a new r172 bullet), `benchmarks/2026-10-29-derived-kq-multiseq.md`,
  `WORKLOG.md` 261029.003.

## Re-entry conditions / not covered

* A server that really shares prefixes through `llama_memory_seq_cp` latches and keeps the packed mask for the rest of
  the run - correct, but the memory win is then a one-time re-reserve and then nothing.  Firing the latch could not be
  reproduced on this box (this `llama-server` does not `seq_cp` across slots while a slot is busy, and prompt-split
  children run as single-sequence ubatches); it is documented from the contributor's `llama-parallel -pps` evidence.
* Any new way to put two sequences in one cell must bump `llama_kv_cells::n_multi`, or the derived fast path becomes
  unsound.  The randomized test here (`test-kv-cells-multi.cpp`, run at promotion: 400 000 operations, counter ==
  recount after each) is the check; it is a `wip`-side tool, not part of the delivery.
* The batch-wide skip window means a multi-sequence derived ubatch skips no KV groups.  It measured neutral at 16K-112K
  on gfx1201; a workload where the other slot's cells dominate (a very long resident context beside a small prefill)
  was not measured on a *second* workload shape.
* Not measured here: Windows (the contributor measured it: -0.94 GiB shared GPU memory and a 2.7x prefill on a `-ncmoe`
  MoE) and the CPU backend (the contributor saw no derived path in either arm).

## Files

| file | what |
|---|---|
| `README.md` | the contributor's handover, measurements and caveats (verbatim) |
| `kq-derived-multiseq.patch` | the submitted `git am` patch (sha256 `ecd200608258d0d566a0159df7679ca57f687240621a350132139f77cf460937`), cut against r167 |
| `DIAGNOSTIC-verify-derived-kq-mask-multiseq.patch` | the cell-by-cell oracle (`LLAMA_KQ_MASK_DERIVED_VERIFY=1`), **not for delivery** - re-applied to the folded tree for the promotion gates |
| `test-kv-cells-multi.cpp` | the randomized `get_n_multi_seq()` check, **not for delivery** |
