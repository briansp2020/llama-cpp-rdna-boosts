# Goal B — stream a *sparse attended set* cheaply: plan of record (B0 → B4)

**Status: PLAN + one decisive baseline measurement (B0).  No campaign code written, nothing in `patches/`.**
Goal A (the recency window) is shipped as block 15; Goal B is a *different* mechanism (the campaign's own
words: "a sparse attended set streamed cheaply").  This file is that mechanism's plan, written before any code
(HANDOVER §11: Goal B "needs its own plan before any code").  The reference design is
[`STRATA-KV-STREAMING.md`](STRATA-KV-STREAMING.md); the state of the build is
[`A2-MIXED-BACKING.md`](A2-MIXED-BACKING.md) (§3.7.13 §4 records B0).

Box: **one R9700 (gfx1201, 32 624 MiB)**; every number below is 1 GPU, 55.7 GB/s H2D (2026-10-30 re-cable).

---

## 1. Why this is not Goal A, and why it is worth a session

| | Goal A (shipped, block 15) | Goal B (this plan) |
|---|---|---|
| what decides what stays in VRAM | **recency**: the newest `W` used cells (a watermark, host-side) | **the attended set**: the cells this step's attention actually reads (device-side, per call) |
| VRAM is | `O(W)` where `W >= depth` pays, `W < depth` spills | `O(top_k)` — **independent of depth** |
| cost | a straight line in spilled bytes/token (`A2 §3.7.9`) | the read volume of the attended set + the selection's own read |
| purity | exact (bytes move, values do not) | **exact** (the attended set is exact; residency changes *where* the bytes live, never *which*) |

The point of B: block 15 is a **capacity** trade (it only wins when `W` is at or above the depth you run,
`A2 §3.7.11.4 A/B`).  QSA's attended set is **bounded by construction** — Flash-Next's GGUF says
`attention.indexer.top_k = 2048` with `compress_ratios = 4`, i.e. `top_k + r - 1 = 2051` cells per query per
layer, **3.98 % of a 51 475-token context** — so a residency policy built on it is `O(top_k)` at *any* depth.

## 2. What is already true in this tree (so the plan builds on shipped machinery)

1. **Flash-Next is a hybrid: only 12 of its 48 layers carry an attention KV** (`llama_kv_cache: size = 816.00
   MiB (65536 cells, 12 layers)`, K 408 + V 408 at q8_0; the other 36 are recurrent/GDN, `O(1)` state).  So the
   streamable mass is 12.75 KiB/cell, and everything below is computed on that.
2. **The QSA reader is already sparse and exact.**  `GGML_OP_FLASH_ATTN_QSA` "attend[s] only over the cells the
   indexer's top-k names" and "each warp walks that column's top-k list in tiles"
   (`ggml/src/ggml-cuda/fattn-qsa.cuh:6-27`); the ids are a **device** tensor (`ggml_indexer_top_k`), one list
   per token column.  So no reader change is needed to *read* a sparse set — the open question is only *where
   the bytes live* and *how they get there*.
3. **The selection's own read is the elephant, and it is not sparse.**  Three layers of cost per step: the
   indexer **scores every block** of the prefix (its cache is keys-only, `src/llama-memory-hybrid-idx.cpp:23-30`,
   ~1.6 KiB/cell at `-c 65536` → ~82 MiB/token at depth 51k), the top-k itself is device-local, and the
   attention reads the selected 2051 cells/layer (~26 MiB/token).  Any B policy must keep the **indexer cache
   resident** (it is small: 102 MiB at `-c 65536`, 410 MiB at 262144) — streaming it is the trap.
4. **Block 15's region is already the physical layer B needs**: one VA reservation per tensor, per-mapping
   `kind` (device/host), a flip through a pinned scratch, a run-time pinned-RAM ceiling, and a per-cache
   window.  What B adds is *which* pages are device — the flip mechanism already exists (`ensure_span`).
5. **The decode band is a captured graph**, so a residency **decision taken on the host mid-step is not
   capturable** — the campaign's own Strata mapping says a page-table + copy-kernel resolve is "the only
   capturable form" for a per-call policy (`STRATA-KV-STREAMING.md` §3).
6. **A policy precedent exists** in the expert cache (heat prior, F-LRU, the identity cliff) — but it is a
   *host-side, sizing-time* policy, not a per-call one; B's is per call.

## 3. B0 — the measured starting point (2026-10-30, this session)

Arm: Flash-Next GSQ-RCO-IQ3_XXS, `-ncmoe 48`, `-c 65536`, q8_0 KV, 51 475-token prompt, `-n 2000`,
`ROCBLAS_USE_HIPBLASLT=0`, interleaved 1 rep each (1 GPU).  `W=1` is *the naive streaming arm*: block 15 puts
the whole KV's tail in pinned host RAM (both caches windowed), and the QSA reader then reads its attended set
out of host.

| arm | prefill t/s | decode t/s | ms/token | arena MiB | residency | hit rate |
|---|---|---|---|---|---|---|
| plain (KV fully resident: attention 816 + indexer 102 MiB) | 584.13 | **39.19** | 25.52 | 24 539.0 | 60.0 % | 0.9686 |
| `LLAMA_KV_RESIDENT_CELLS=1` (spilled: attention 576/672 + indexer 48/96 MiB host, `KVREGION_FREE … host=` proved) | **604.49 (+3.5 %)** | **33.67 (−14.1 %)** | 29.70 | **25 338.3 (+799)** | **61.9 %** | 0.9727 |

**Read this as: naive streaming is a 14 % loss for a 0.8 GiB arena gain (+1.9 points of residency, +0.4 points
of hit rate) — a clear loss, and exactly the gap Goal B exists to close.**  Two further facts it pins down:

* **The 4.18 ms/token is not explained by the sparse read alone.**  2051 cells × 12.75 KiB/cell (12 attention
  layers of K+V, q8_0) = **26 MiB/token** of attention read, plus ~82 MiB/token of indexer scoring (every
  block, every layer) ≈ 108 MiB → ~2 ms at the link's peak 55.7 GB/s.  Measured 4.18 ms ⇒ either scattered host reads run at ~**26 GB/s** (≈ half the linear rate;
  `bw-probe.cc` measures linear only) or there is traffic we have not attributed (e.g. the qsa3 prefill path,
  or the indexer's per-layer share being larger than the cache's average).  **B1 must attribute it before any
  policy is designed** — a policy aimed at the wrong half of the traffic is wasted work.
* **Prefill gets faster when the KV leaves VRAM** (+3.5 %): prefill is capacity-bound (a prefill routes every
  expert), so the freed VRAM is worth more there than the spilled read costs.  Decode is where the loss is.

## 4. The three candidate mechanisms

**B-A — page table + resolve inside the graph (Strata's form).**  The KV becomes a bounded pool of `n_slot`
cells per device; `residency[block] -> slot | -1`; a resolve kernel runs before the attention in the same
captured graph, rewrites the top-k list into slot ids, and copies the misses from the (pinned) host copy; a
writer writes the host copy always and the slot only if resident.  *Cost*: a new pool/table/resolve/miss-copy
path, a write-path change, and the reader is **unchanged** (it walks ids).  *Risk*: a per-layer pool and table
(the selection is per layer), plus the epoch/stamp reserve so a slot this call uses is never a victim.

**B-B — compact gather per call.**  The KV stays entirely in pinned host memory; before each attention a
gather kernel copies the selected cells into a compact device buffer (`n_sel` cells), and the attention runs on
identity ids `[0, n_sel)`.  *Cost*: one gather kernel + an id rewrite; the write path is untouched (host
always).  *Risk*: 2× the traffic of B-A (host read + device write per token instead of a direct read), and the
compact buffer is a new allocation in the decode band (it must be reserved like a graph buffer).

**B-C — attended-set residency inside block 15's region (the smallest step).**  Keep block 15's mixed-backing
region and change *only* the policy: instead of a recency watermark, the device pages are the pages the
attended set will touch.  This needs a **device-side** decision (the top-k is device data), which the region's
host-side flip cannot express in a captured graph — so B-C's realistic form is a one-step-lag host policy (a
diagnostic readback of the previous step's ids; legal only if the graph is not captured, or with a small
device→host copy after the step) or an in-graph "prefetch next step's set" kernel.  *Value*: it reuses every
primitive block 15 already ships; *risk*: a lagging policy thrashes (a wrong guess costs a flip, and a flip is
a *copy*, not a mapping change).

**Recommendation: B1 (measure) → B-A if the attended read dominates, B-B if the write path proves costly,
B-C only as a cheap A/B.**  Do **not** start with the page table: B0 shows we cannot yet say whether the prize
is 26 MiB/token or 108 MiB/token.

## 5. The staged plan

**B1 — attribute the cost (no mechanism).**  Reproduce B0 and split the 4.18 ms:
1. a **per-cache window exemption** (`LLAMA_KV_RESIDENT_CELLS_IDX`-style, wip only) so the attention cache can
   spill while the indexer stays resident — the difference isolates the indexer's share;
2. a **scattered-read probe** (extend `bw-probe.cc`: random 4 MiB-unit reads vs linear) — the effective GB/s a
   sparse KV read actually gets;
3. a **census** of the attended set per step (union of blocks per layer, per prefill ubatch), from a diagnostic
   readback in a non-captured arm.
*Gate*: the three numbers add up to the measured `W=1` delta within ~10 %.  Cheap: 3-4 arms, no mechanism.

**B2 — the residency policy on a bounded set.**  Prototype one of B-A/B-B behind `LLAMA_QSA_STREAM` (0 = off,
byte-identical) on the campaign fork branch `wip-kv-residency`.  Invariants: exactness (a page is staged before
it is read; writers write host always), the indexer cache stays resident, prefill may stream its union or stay
resident (decide from B1's census), and the pool is sized from `top_k` + headroom, not from `-c`.
*Gates*: prefill-logit KLD **0.000000 / 100.000 %** on a streaming arm (it must be exact, unlike the drafter
ring), greedy text byte-identical, `FLASH_ATTN_QSA` op test, `check-gpu-faults`, and the B0 arm's t/s.

**B3 — the arena half.**  The freed VRAM goes to the expert cache; measure at `-n 2000` with `-ncmoe` = the
full layer count, and require the identity fast path's residency to be recorded (the r117 cliff).

**B4 — promotion decision.**  Only after B2/B3: a block, `ENVIRONMENT.md`, the default-on call.  Until then
nothing leaves `wip/`.

## 6. Eligibility, risks, and what must not be done

* **Eligible models are the ones whose attended set is bounded**: qwen4exp/QSA (Flash-Next) first, DSA next.
  **Not** full-attention dense/MoE, where the attended set is the whole used context (`STRATA-KV-STREAMING.md`
  §4/§5: Strata never streams a full-attention layer either).
* **The indexer cache is not streamable** in the same policy (it is read in full to score) — keep it resident
  and say so in the VRAM arithmetic (102 MiB at `-c 65536`, ~410 MiB at 262144), or the design trades a
  few hundred MiB of savings for 82 MiB/token of reads.
* **The recurrent/GDN layers are not part of this** (36 of Flash-Next's 48 layers): their state is `O(1)` per
  layer and already resident.  Goal B is a 12-layer (attention-KV) feature on this model.
* **Exactness is the oracle, not the drafter ring's trade.**  The drafter may approximate (the target
  verifies); a *target* KV read may not — a streamed arm owes the prefill-logit gate at 100.000 %.
* **Prefill's union is not sparse** (each query selects its own set), so a policy tuned on decode must have a
  prefill arm; a prefill regression here would likely be the loss.
* **Do not** put the residency decision on the host mid-step (not capturable), do not stream the indexer, do
  not route this through the slab (MoE-only by r37), and do not add host file/POSIX calls without the
  Windows-portability discipline (`AGENTS.md`, issue #231) if new host code appears.
* **Do not** start from Strata's `--kv-grow`/`--vram-elastic` rows: block 15's region already covers the
  first, and the slab already covers the second's shape.

## 7. Open questions for the maintainer

1. **Is B-A's pool-of-slots (a real page table, one per layer, with a resolve kernel in the graph) sanctioned
   work**, or is B-B's compact gather (simpler, 2× traffic) the preferred first cut?
2. **The target context**: does Goal B aim at `-c 262144` Flash-Next on 1 GPU (where the attention KV is
   3.3 GiB and B frees most of it), or at the multi-device `-ncmoe` server shape (where the freed VRAM is
   ~1 GiB/device of arena)?  The measurement arms differ.
3. **B1's per-cache window exemption** is a small, wip-only code change (a second env knob on block 15's
   window).  It is diagnostic, but it does touch the block's file — confirm it stays on the campaign branch
   and never reaches `patches/`.
