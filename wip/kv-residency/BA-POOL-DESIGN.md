# B-A — the attendance pool for the attention KV (design + acceptance numbers)

**Status: DESIGN, with every number measured on this box (1 R9700). No pool code yet.**  Plan context:
[`B1-STRATA-PLAN.md`](B1-STRATA-PLAN.md) §3.2-§3.5; instrument and trace: [`qsa-census.py`](qsa-census.py) over a
48-step decode trace (Flash-Next IQ3, `-c 131072`, 102 912 cells, sparse arm, `GGML_CUDA_DISABLE_GRAPHS=1`).

## 1. What is measured (the inputs this design stands on)

| fact | value |
|---|---|
| attended set | **2051 cells per layer per step** (`top_k + r − 1`, r = 4), spread over the whole prefix |
| step-to-step Jaccard (same layer) | median **0.586** |
| recency window of 17-25 % of the KV | **≈ 0 gain** (+3 % over ~0 at 180k) — position says nothing about attendance |
| region's map unit (block 15) at 4 MiB | **3855 cells/layer** → an attendance policy at that granularity hits **6 %** and churns 601 units/step (**90 ms/step**): the region's granularity cannot express a scattered attended set |
| a page-table pool, page = **4 cells** (one indexer block), lag-1 admission, adaptive LRU | **85.1 % at 4096 cells/layer, 90.2 % at 8192** (0.05 ms/step of fills), 90.3 % plateau = churn-bound |
| adaptive (LRU/F-LRU) vs a pre-calculated static ordering, equal capacity | **84-90 % vs 63-68 %** — the expert-cache campaign's verdict, reproduced |
| pool VRAM | 8192 cells/layer × 1088 B × 12 layers = **107 MiB** |

## 2. The design

**One compact pool + one host mirror per KV cache, with the page table as the only indirection.**

* **Page = 4 cells = one indexer block** (the QSA selection is block-granular: `blk_idx`/`blk_cells` name whole
  blocks of `compress_ratio` cells, and `top_k/r` blocks are selected).  This is the *same* page size Strata
  chose, for the same reason: it is the granularity the reader already speaks.
* **The host mirror is authoritative** and holds the whole `-c` context per tensor in pinned RAM (Strata's
  writer rule): a KV *write* goes to the mirror **always**, and to its slot only if the page is resident.  No
  invalidation is ever needed — not after a rejected draft, not on a resume, not on `seq_rm`/`seq_cp`/defrag.
  An eviction therefore costs **nothing** (no stage-out); only a miss costs a copy.
* **The pool is compact**: page `p` of layer `l` lives at slot `table_l[p] ∈ [0, N)`, so a scattered attended
  set packs densely.  VRAM = `N × 4 × cell_bytes` per layer; 8192 cells/layer is 107 MiB for Flash-Next.
* **Resolve runs inside the captured graph**, per layer, before the attention: one small kernel that
  (a) maps the selected block ids through `table_l`, claiming a slot for a miss, (b) copies the missed pages
  from the mirror (host→device, one run per K/V array), (c) rewrites the id list to slot indices.  This is the
  only in-graph piece; the reader (`GGML_OP_FLASH_ATTN_QSA`) is unchanged because it already walks an id list.
* **Admission is lag-1** (the previous step's selected set), because a per-step device→host readback would
  serialise the decode pipeline and a synchronous host remap is not capturable.  The census says lag-1 at page
  granularity still hits 85-90 %.
* **Policy = the delivery's, not a frozen ordering**: LRU / F-LRU over pages, with the heat prior usable exactly
  as the expert cache uses it.  `qsa-census.py` is the replayer that scores a policy offline before it is coded
  (the KV analogue of `scripts/moe-trace-replay.py`).
* **The indexer cache stays fully resident** (`O(depth)`, read in full to score: 165 MiB at 103k, 410 MiB at
  262144).  It is not poolable; that is a design constraint, not an oversight.

## 3. What it is worth (expected, from the measured hit rate)

At 103 899 depth, streamed `W=1` + B2 is **31.4 t/s** against a resident **42.2** (plain) / **58.2** (MTP,
their config's shape).  The pool's fill traffic at 8192 cells/layer is **0.05 ms/step** (2.6 MiB), i.e. the
attention KV's streaming cost disappears; what remains is the indexer's `O(depth)` read and the sparse arm's own
compute — both present in the resident arm too.  **So a pool should land the streamed arm within a few percent
of resident, with 107 MiB of VRAM instead of 2.45 GiB.**  That is Strata's 80+ shape, and it is the acceptance
test: the four-arm gate in `B1-STRATA-PLAN.md` §3.3, with the pool arm added, at 5.3k / 103k / 180k.

## 4. Build order (each step gated on its own)

1. **The mirror + the pool as a KV cache mode** (`LLAMA_KV_REGION_POOL=<cells>`; 0 = off, byte-identical to
   today): a host-mirror tensor pair per layer, a device pool of `N` pages, a page table, and a write path that
   lands in the mirror always (+ the slot when resident).  Gate: text byte-identical on a *resident* arm
   (pool off vs on, no window), and the state/ckpt/park gates (the mirror must make `seq_rm`, `seq_cp`,
   `defrag`, `state_read/write` trivial — that is the point of Strata's writer rule).
2. **The resolve kernel + the id rewrite** for the QSA decode path only (prefill keeps the current window or a
   fully-resident arm: its union is not sparse).  Gate: `test-backend-ops -o FLASH_ATTN_QSA`, then the pool arm's
   text equal to the resident arm's *sparse* hash (`e6080aaf2318` at 51 475, from B2's gate) — the pool changes
   placement, never values.
3. **The policy** (lag-1 LRU, then F-LRU + heat), gated on `qsa-census.py`'s predictive numbers (≥ 84 % at
   4096 cells/layer) and on the four-arm timing gate.
4. **B3** (the arena half at `-n 2000`, full `-ncmoe`, recording arena MiB / residency % / `identity fast path`
   count), then the promotion decision.

## 5. Invariants this owes (from the campaign's list, plus its own)

* **Exactness**: a page is filled from the mirror *before* the attention reads it, and the mirror is written
  first on every write — so the pool can only change *where* bytes are, never which.  The oracle is the resident
  sparse arm's text hash, not a new regime.
* **The mirror is pinned** (an AMD kernel cannot read pageable memory with XNACK off, `AGENTS.md`), and bounded
  by `LLAMA_KV_SPILL_RAM_PCT` — a `-c`-sized mirror is 2.45 GiB at 180k, 3.3 GiB at 262144, per cache.
* **A graph bakes the pool's addresses**: the pool, the table and the mirror's staging are fixed allocations,
  and any re-allocation must bump `ggml_cuda_graph_mem_freed` (the block 09/10 rule).
* **Do not** put the residency decision on the host mid-step (not capturable); **do not** big-pool the indexer;
  **do not** use the region's unit flips for this (measured: 6 % hit, 90 ms/step of churn).
* **Windows portability** if any host file/POSIX call appears (issue #231 discipline).

## 6. The two alternatives this closes

| alternative | verdict |
|---|---|
| block 15's region with the resident set chosen by attendance (no page table) | **refuted by measurement**: the region's 4 MiB unit = 3855 contiguous cells, the attended set is scattered → 6.4 % hit and 601 unit flips/step (90 ms/step) |
| a pre-calculated static ordering (Strata's static arena) | **refuted by measurement on the KV**: 63-68 % hit vs 84-90 % for an adaptive policy at the same capacity, the same verdict the expert-cache campaign reached for the arena |
