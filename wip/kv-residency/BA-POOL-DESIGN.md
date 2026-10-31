# B-A — the attendance pool for the attention KV (design + acceptance numbers)

**Status: STEPS 1 AND 2 ARE BUILT AND GATED (session A10, §7); step 3 (the policy) is not.**  Every number is
measured on this box (1 R9700).  Plan context:
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

## 7. Built and measured (A10) — steps 1 and 2 of §4

Branch **`wip-kv-residency-b`**, tip **`b5d895c59`** (pushed).  Two commits: the pool itself, then the mirror's
host-copy handling in the delivery's async state paths.

### 7.1 What was built

* **`LLAMA_KV_REGION_POOL=<cells>`** (unset/`0` = off, byte-identical): a KV cache mode.  The attention cache's
  cell-indexed tensors become the **pinned host MIRROR** -- authoritative, so every existing path (the
  `set_rows` writers, `state_read`/`state_write`, the K-shift, `seq_park`, `defrag`, `get_k`/`get_v`) works on it
  **unchanged**, which is the whole point of Strata's writer rule -- and a compact **DEVICE POOL** of pages of one
  indexer block (`compress_ratio` = 4 cells) per layer, plus a page table `tbl[block] -> slot`, its inverse, a
  per-slot epoch mark and a counter block.
* **`GGML_OP_KV_POOL`** (CUDA only): mark / claim / map+fill, three launches per layer per step.
  * MARK sets `use[slot] = epoch` for every page this launch reads **and** every page the batch just wrote
    (`wids` = the graph's `k_idxs`), and queues the written ones for re-fill.
  * CLAIM takes a slot for each **distinct missing block** (the table CAS is the dedupe) and never takes a slot
    MARK protected.
  * MAP+FILL writes each selected cell's pool position into the reader's K/V index list and copies the claimed
    and dirty pages from the mirror, K and V (they share slots).
  * The epoch is a device counter incremented once per launch -- the graph bakes pointers, never counters, and
    nothing is allocated per launch (a transient device-pool scratch inside a captured graph is a
    use-after-free; see §7.3).
* **The reader is unchanged in shape**: `ggml_flash_attn_qsa_set_kv_idx()` gives the QSA kernel a second id list,
  so the K/V gather follows the page table while the mask / visibility keys stay indexed by the **cell** number.
* Eligibility is narrow on purpose: one stream, non-MLA, K and V both present, `!v_trans`, a uniform
  `compress_ratio` and cell geometry, no window active, and only the **trunk** context of a qwen4exp model
  (`llama_memory_params.kv_pool`, false for the MTP draft's cache -- its attention is dense).  The indexer cache
  stays fully resident by construction.  Anything else declines with one message.
* The pool is **floored at one decode band's worth of blocks** (`8 x ceil(width/page)` pages) so a verify batch
  cannot exhaust the slot budget, and the graph re-checks the real width (`kv_pool_serves`): a batch too wide
  for the pool takes the cell-indexed path instead.
* A pooled cache reports `is_streaming()` -- the pool exists to serve the sparse arm, so the dense arm must not
  read the mirror over the link.

### 7.2 Gates (Flash-Next GSQ-IQ3_XXS, 1 GPU, `-ncmoe 48`, q8_0, PLE flags, `ROCBLAS_USE_HIPBLASLT=0`)

| gate | result |
|---|---|
| text, pool **off** vs **on** (`-c 65536`, 51 475 depth, `-n 64`, sparse arm both) | **854 chars `43b30de151b9` — identical** |
| text, pool on, **dense** arm forced (`STREAM=0`) | 853 `77a40bf40be1` — identical to the pool-off dense arm (the mirror write path is exact) |
| the pool's own verify kernel (`LLAMA_KV_POOL_VERIFY=1`: every selected page compared byte-for-byte with the mirror at read time) | **0 mismatches** |
| `scripts/gate-prefill-logits.sh --ab "LLAMA_KV_REGION_POOL=8192"` (ctx 8192, chunks 2, q8_0) | **0.000000 KLD / 100.000 % top-p — PASS** |
| `scripts/gate-park-idle-slots.sh` (1 GPU, layer, `-ncmoe 48`, ctx 32768) | **PASS** — 6 park + 2 unpark moves traced (not the save-and-clear fallback), `park == lazyoff` on all four requests |
| `scripts/gate-ckpt-reuse.sh` (same shape) | **PASS** — keep 3 / eager 3, identical text and RSS |
| `scripts/gate-state-pipe.sh` (ctx 16384) | **PASS** — all five arms |
| `test-backend-ops -o FLASH_ATTN_QSA` | **27/27 OK** (`HIP_VISIBLE_DEVICES=0`; the iGPU abort at the end of an unpinned run is the documented gfx trap) |
| `check-gpu-faults.sh` over every log | clean |
| hit rate / fills | **89.66 %** of ids at 4104 pages, 27 762 fills over 768 resolves (~36 pages/resolve) |

Two earlier text discrepancies were **not** defects: the pool's stats line was printed between the generated text
and llama-cli's perf line, and `scripts/extract-generated.py` slices exactly that range.  `LLAMA_KV_POOL_STATS`
now takes a **path** and appends there.

### 7.3 Three defects the text caught (each is now an invariant)

1. **A claim must never evict a page the same launch reads.**  The first cut marked nothing, so the round-robin
   victim could be a page another row still needed: the text diverged ~400 chars in (and a pool larger than the
   whole cache hid it completely).  MARK-then-CLAIM is what makes it exact.
2. **Claim attempts are not distinct blocks.**  A slot was consumed per *id* attempt, so a 4-row batch (4 x 2051
   ids against a 4104-page pool) wrapped the cursor **within one launch**: two blocks ended up in one slot and a
   just-resolved block went unresident.  The table CAS now dedupes first, so only distinct blocks consume a slot,
   and `kv_pool_serves` bounds the distinct count.
3. **No transient scratch in a captured graph.**  The fill list lived in a `ggml_cuda_pool_alloc`, which is freed
   at the end of the op -- and a captured graph bakes the pointer, so a replay wrote into whatever op had been
   handed that block.  The symptom was exactly "values verify clean, text differs run to run".  The list lives in
   the counters tensor now.

### 7.4 What it is worth at 103k depth (the acceptance test)

Flash-Next, 1 GPU, `-ncmoe 48`, q8_0, `-c 131072`, 102 948-token prompt, **MTP depth 4**, `-n 128`:

| arm | gen t/s | accepted | prefill t/s |
|---|---|---|---|
| resident KV, dense arm (the default) | **60.2** | 87 | 519.1 |
| resident KV, **sparse** arm forced (`LLAMA_QSA_DENSE_DECODE_UNTIL=0`) | 47.9 | 79 | 518.9 |
| **pool, 4104 pages = 16 416 cells/layer, 204 MiB** | **48.3** | 79 | 488.1 |

**The pool lands ON the sparse ceiling (+0.8 %, i.e. within noise) — the fills are essentially free (2.7
MiB/token at 90 % hits = 0.05 ms of a 21 ms step).**  Against B2's fully spilled arm (31.4 t/s at this depth)
that is **+54 %**, and it costs 204 MiB instead of the resident KV's 816 MiB.

So the residual **-20 % against resident is entirely the SPARSE ARM's own cost** (the indexer scores every block
and the top-k runs over it -- `B1-STRATA-PLAN.md` §3.2's 47.9 vs 60.2 pair proves it with no pool involved), not
the KV's placement.  That reframes step 3: a better pool policy can only recover the *remaining* ~10 % of fills
(≈0.05 ms), so **the next lever is the sparse decode arm's indexer cost, not the page policy.**  The pool's own
job -- "204 MiB of VRAM instead of 816 MiB, with the streaming tax gone" -- is done.

Pre-split note: prefill reads the mirror over the link (488 vs 519 t/s, -6 % at 103k), the same shape block 15's
windowed arm showed (`W=1` prefill was *faster* there because the freed VRAM went to the expert arena; here the
pool's 204 MiB is small and the mirror is a second copy of the KV, so prefill pays the link).

### 7.5 B3 — the arena half (what the pool's VRAM is worth)

Flash-Next, 1 GPU, `-ncmoe 48` (the model's full layer count), q8_0, `-c 131072`, 102 948-token prompt,
`-n 2000`, **plain** decode (no MTP), arms back to back in one batch:

| arm | arena | slots/table | expert hit / cold | prefill t/s | decode t/s |
|---|---|---|---|---|---|
| resident KV (816 MiB) | 23 579.8 MiB | 295 | 96.26 % / 2.68 % | 562.2 | **43.1** |
| **pool (204 MiB, mirror in host)** | **25 018.6 MiB (+1 438.8)** | **313** | **97.02 % / 2.00 %** | 568.2 | **33.0** |

* **The VRAM is real: +1.44 GiB of arena (+18 slots/table), +0.76 points of expert hit rate, cold reads 42.0 ->
  31.6 GiB.**  `identity fast path ON` count is **0 in both arms** — at 313 of 512 slots per table the r117
  cliff is not in play at this shape, so the per-token traffic figure the report prints (a stale per-token
  divisor) is not usable here.
* **Plain decode is the pool's worst case (-23 %)**: it must take the sparse arm while a resident KV takes the
  dense one on gfx1201 (`B1-STRATA-PLAN.md` §3.2).  With MTP depth 4 -- the product shape -- the same comparison
  is 60.2 (resident-dense) vs **48.3 (pool)**, and the sparse ceiling with a *resident* KV is 47.9, so the pool
  itself costs ~0 (see §7.4).

So B3's answer: **+1.44 GiB of arena for a 612 MiB smaller KV, at the sparse arm's cost, which is only paid
because the pool cannot express the dense arm.**  That is the same trade block 15's window makes; the difference
is that the pool keeps 90 % of the attention's KV *hot* rather than a positional tail.

### 7.6 Not done (the work list)

* **Step 3, the policy.**  Eviction is round-robin + "skip what this launch marked" (§2's lag-1 LRU, F-LRU and
  the heat prior are NOT implemented).  Predicted ceiling: the remaining ~10 % of fills (`qsa-census.py` says an
  LRU at this capacity is 90.2 % vs 90.05 % measured -- already at the plateau, §1's "churn-bound").
* **Step 4 / B3**: the arena half (`-n 2000`, `-ncmoe` = the full layer count), recording arena MiB / residency %
  / the `identity fast path ON` count.
* `--fit` accounting: the pool's 204 MiB is inside the KV buffer's size but the mirror is *pinned host*, invisible
  to a VRAM arithmetic.  A `-c 262144` Flash-Next run pins 3.4 GiB (bounded by `LLAMA_KV_SPILL_RAM_PCT`).
* The mirror is allocated **in full at creation** (a short run pins the whole `-c`): a grow-on-demand mirror is
  the obvious next step if the RAM cost matters.
* The pool declines a `-sm tensor` split (like the window): one page table per device would need a per-device
  pool and a split-aware resolve.
