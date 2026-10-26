# 2026-10-17 — what limits the expert-cache hit rate: **coverage, not fragmentation**

Exploratory session on the premise of `wip/moe-arena-defrag/` (`TODO.md` #61).  Two questions were
asked: (1) what does Strata actually count as an expert-cache *hit*, and why can it report ~99 % where
we report ~90 %; (2) is the arena being used well, and does the "defrag / repack at idle" premise
deserve to be the campaign?

**Answer in one line: no fragmentation fix can move our number; the gap is (a) accounting, (b) how many
`(layer, expert)` pairs the arena holds, and (c) a live bug that has silently destroyed the heat map
that every heat-driven feature reads.**  Campaign retarget proposed in §8.

All numbers below are from this box unless a source is named.  Evidence files are listed per claim.

---

## 1. What Strata counts as a hit — and what the 99.8 % actually was

`hit_rate` is "the **VRAM share of the experts looked up while answering**" (`docs/DETAILS.md:294`),
measured **per decode request**, over **distinct `(layer, expert)` pairs of one verify window**, with
three exclusions.  Each is in their source:

> **Context: Strata is a one-model engine.**  It runs Qwen3.8-Flash-Next and nothing else
> (`docs/MODELS.md`, the per-model configs, the MTP draft vocab, the hand-written i-quant CPU kernels,
> the `pcie_frac` auto-calibration that refines itself from the CPU pool's own speed, and the 96-swap
> adaptation round are all specific to that model).  Its published hit-rate curve is therefore **the
> coverage curve of that one routing distribution**, not a property of an expert cache — it is not a
> target another engine can reach by copying a mechanism.  What transfers is the *method* (§3, §8 P1),
> not the number.

| # | what | where | effect on the number |
|---|---|---|---|
| 1 | **Per verify window, not per token.**  The window's `n_tok * k` routing entries are collapsed to distinct expert ids (`distinct` / `first_of`) and each distinct id is one lookup. | `src/core/expert_source.cpp:3669-3705` (`expert_pool_dispatch_multi`) | *~0 pts* — see below |
| 2 | **The PCIe share is excluded** from both numerator and denominator (`--pcie-frac`: missed experts the *GPU* reads over PCIe instead of the CPU computing them).  Their own docs warn that raising it raises the hit rate. | `src/core/expert_source.cpp:3713` (`m = nmiss * pcie_num >> 8`), `src/program/generate.cpp:12273-12282` | *0.0 pts* on this run (0.0 % of routed) |
| 3 | **`STRATA_ROUTE_TAIL_SKIP=7` (default ON):** a missed expert that every token of the window routes at rank ≥ 7 is **neither copied nor computed** — its contribution is dropped without renormalisation.  It is not a lookup, so it is not a miss. | `src/core/expert_source.cpp:3702-3712`; log line *"STRATA_ROUTE_TAIL_SKIP=7 is ON (default)"* | **6.5 pts** — the big one |

The window dedup (row 1) turns out **not** to matter: at these routing densities `p_e ≤ 1e-3`, so the
union probability `1-(1-p)^T` is linear in `p` to within 0.1 % (`tools/heatmap-audit.py --all`, window
section: delta ≤ 0.0023 pts at T = 2.6).  It compresses *transfer* cost, not the ratio.  Good to know
before blaming the metric.

### The arithmetic on the maintainer's own Strata run

`~/Strata/strata-iq3_xxs.log` (1 × R9700, GSQ-RCO IQ3_XXS, `--spec 4`, engine 0.1.43).  The last two
per-request lines are cumulative counters, so subtract them:

| | lookups | hits | reported |
|---|---:|---:|---:|
| request A (cumulative) | 9 855 417 | 9 776 048 | 99.2 % |
| request B (cumulative) | 13 890 067 | 13 671 024 | **98.4 %** |
| **request B alone** | **4 034 650** | **3 894 976** | **96.5 %** |

`route tail skip` is also cumulative: 96 213 → 298 191, so **201 978** experts were skipped during
request B.  Counting them as the misses they are:

> **3 894 976 / (4 034 650 + 201 978) = 91.9 %**

The PCIe share for that request was 548 experts (+0.01 pts).  So the same run is **98.4 % as reported,
91.9 % on an apples-to-apples "was the expert resident" basis** — and the experts in that 6.5-point band
were not merely mis-counted: **they were never computed** (a deliberate approximation that changes the
answer, which is why it is off on the reported `--exact`-ish paths).

### Where "99.8 % at 25 % residency" came from

It did not.  In that same log the cache holds **15 667 of 24 576 pairs = 63.7 %** (25.46 GiB — the 25 %
is the *arena size* against the 76 GB model file, not the pair fraction).  Their published curve
(`docs/paper/Strata-Paper.pdf`, Fig. 3, held-out prompts) is 0.50 at ~4 500 of 24 576 pairs (18 %),
raised to ~0.72 by the adaptive tier on a 12 GB card.  The 99.7–99.9 % rows in `docs/DETAILS.md`
(`:1604`, `:1636`) are configs that hold **94.8 %** of the pairs or run a 2-GPU split that holds nearly
all of them.  **63.7 % residency with a static ranked hot set and no eviction is 97–99 % because the
routing distribution is extremely concentrated — not because of anything clever in the transfer path.**

---

## 2. Our number, measured on the same model, on the same basis

The field 1-GPU GSQ run (`archive/work/first-prefill-latency/seed-floor-full-arena-1gpu.log:663,670`):

| | Strata (as reported) | Strata (same basis) | us (r74) |
|---|---:|---:|---:|
| pairs held | 15 667 / 24 576 = **63.7 %** | 63.7 % | 28 224 / 73 728 = **38.3 %** |
| decode lookup unit | distinct per window | distinct per window | `(token, expert)` entry |
| excluded experts | tail-skip 6.5 pts, PCIe 0.0 | none | none (cold reads are misses) |
| arena | 25.46 GiB | 25.46 GiB | **15 666.6 MiB** |
| hit rate | 98.4 % | **91.9 %** | **90.6 %** (264 632 / 292 200) |

**On comparable accounting the two engines are within ~1–2 points on this model**, and the entire
remaining difference is the arena: 25.46 GiB vs 15.30 GiB (15 666.6 MiB), i.e. 63.7 % vs 38.3 % of
the pairs.  Nothing
in this comparison invokes fragmentation, compaction, placement or a prefetcher.

---

## 3. Coverage is the first-order term — measured on our own routing

`tools/heatmap-audit.py` (new, §10) reads an `MHT1` sidecar and reports `H(f)`: the share of routed
mass held by the top `f` of `(layer, expert)` pairs.  Two **healthy** sidecars on this box (same model,
two quants, independent runs — they agree, which is the check that they are not inflated):
`Qwen3.6-35B-A3B-{UD-Q4_K_M,Q8_0}.moe-heatmap`.

| held | 10 % | 20 % | 25 % | 30 % | 40 % | 50 % | 63.7 % | 80 % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| share of routed mass | 47.1 % | 66.5 % | 73.3 % | 78.9 % | 87.4 % | 93.2 % | **97.8 %** | 99.8 % |

* 90 % of the mass needs f = 0.44; 95 % needs f = 0.54; 99 % needs f = 0.71.
* A **static** top-f set therefore *predicts* their number: at 63.7 % held, 97.8 % on a 35B-A3B.
  Their 91.9 %-on-our-basis at 63.7 % is the same order (their distribution differs, and their adaptive
  tier trades some static coverage for conversation locality).
* At **our** 38.3 % the *35B-A3B* map gives 86.1 % (global ranking) / 84.2 % (our uniform per-table slots).
  We measure **90.6 %** on Flash-Next.

### 3.1 Closed 2026-10-17: the clean Flash-Next map (r80 build, field GSQ command)

The poisoned sidecar was deleted and re-recorded in one r80 run (5 669-token prompt + 441 generated
tokens, MTP; `tools/record-gsq-heatmap.sh`; map = 8.38e6 tallies, **0 saturated**, `H(0.5) = 0.952`):

| held | 10 % | 20 % | 25 % | 38.3 % | 50 % | 63.7 % | 80 % |
|---|---:|---:|---:|---:|---:|---:|---:|
| Flash-Next GSQ share of routed mass | 48.5 % | 68.6 % | 75.8 % | **88.7 %** | 95.2 % | **98.7 %** | 99.9 % |

* **Our 90.56 % at 38.3 % residency is +1.8 pts ABOVE the static top-f bound (88.7 %)** — the
  conversation-local F-LRU plus the seed is genuinely beating a cross-run static hot set.  The policy is
  not what is costing us the 10 points.
* **Their 98.4 %-on-our-basis at 63.7 % is 98.7 % on this very curve.**  There is no residual mystery:
  one model's routing concentration, read off one map.  (Their 63.7 % is 25.46 GiB of arena; ours is
  15.30 GiB — that, and nothing else, is the 10 points.)
* **Correction to §6 / the P2 estimate:** on this map every table's routing *sum* is identical
  (0.694 % = 1/144 each — every token routes exactly `k` experts through every layer, by construction),
  so the global-vs-uniform allocation loss is only **0.3–0.5 pts**, not the 1.5–1.9 measured on the
  35B-A3B maps.  Those 35B per-table spreads (0.004 %–1.104 %) are an artifact of the known
  **decode-identity tally gap** (`TODO.md` #64.2: a fully-resident table's decode traffic is not
  tallied), not a routing property.  **P2 is therefore worth ~0.5 points, not 1.5–1.9.**

**Corollary (the thing the campaign must say out loud): on this class of model, coverage is set by how
many pairs the arena holds; policy moves a few points around that line, and transfer batching or
compaction move none.**  Every point above the arena's coverage line has to be bought with VRAM or with
admission quality — there is no third term.

---

## 4. The heat map is broken — and it feeds every heat-driven feature

### 4.1 The bug: the prior is folded in again at every flush (doubling)

`ggml/src/ggml-cuda/moe-expert-cache.cu` (delivery, block 09 / r60):

1. `heat_apply_loaded_locked()` (`:1999`) does `t.heat = g_heat_loaded[key]` at table registration
   (`:3674`) — the persisted prior *becomes* the session vector.
2. The session then adds to the same vector: decode on the host (`moe_cache_update_host`, `:4269`),
   prefill on the device (`heat_pull_locked`, `:3819`).
3. The debounced flush (`moe_cache_get_stats`, `:2949-2958`; once per ≥ 60 s at a server turn) calls
   `heat_map_save_locked()` → `heat_merge_into_loaded_locked()` (`:1719`), which does
   **`loaded[e] += t.heat[e]`**.
4. **`t.heat` is never reset.**  So every flush re-adds the entire running total:

   `loaded_k = 2 * loaded_{k-1} + <new routing this interval>`

   i.e. a doubling per flush, saturating at `0xFFFFFFFF`.

A long server session therefore reaches "everything is 0xFFFFFFFF" in about six flushes (~6 minutes at
the default debounce), and — because the seed and the placement both load that prior back — **both r60
features silently degenerate to expert-id / layer-id order** (`apply_heat_seed_locked`, `:1619`
ties break `a < b`; `alloc_all_locked` orders layers by the same summed value).  The feature logs
success (`heat-seeded 196/196 slots from the persisted map`) while ranking by id.

### 4.2 Evidence on disk (7 sidecars, `tools/heatmap-audit.py --health`)

| sidecar | pairs | saturated | H(0.5) | verdict |
|---|---:|---:|---:|---|
| `35B-A3B/UD-Q4_K_M.moe-heatmap` | 30 720 | 0.00 % | 0.932 | healthy |
| `35B-A3B/Q8_0.moe-heatmap` | 30 720 | 0.00 % | 0.935 | healthy (agrees with the row above) |
| `35B-A3B/UD-Q4_K_M.gguf.moe-heatmap` (retired name) | 30 720 | 0.02 % | 0.997 | inflated |
| `35B-A3B/Q8_0.gguf.moe-heatmap` (retired name) | 30 720 | **97.60 %** | **0.512** | **poisoned** |
| `Flash-Next/GSQ-IQ3_XXS.moe-heatmap` (the one this campaign uses) | 73 728 | **99.05 %** | **0.504** | **poisoned** |
| `Flash-Next/UD-IQ3_XXS-00002.gguf.moe-heatmap` | 72 192 | 24.06 % | 1.000 | mid-cascade |
| `Flash-Next/Q4_K_M-00004.gguf.moe-heatmap` | 73 728 | 0.00 % | 1.000 | inflated |

The signature is unmistakable: the poisoned maps' `H(0.5)` sits **on the uniform line (0.500)** while
healthy ones are 0.93 — a saturated map ranks by id, so its coverage curve is `H(f) = f`.  (The audit
tool now flags exactly this, so a future run cannot miss it.)

**So the campaign's own evidence base is currently invalid**: any measurement of "heat-ordered
placement" or "heat-seeded start" taken against these files measured nothing (and possibly a small loss,
since seeding with experts `0..q-1` of every table is a *fixed arbitrary* set).

### 4.3 Measured proof (gpt-oss-20B MXFP4, `-ncmoe 99`, `MOE_EXPERT_CACHE_MIB=4096`, 3 server turns,
`MOE_EXPERT_HEATMAP_DEBOUNCE_S=0` so every turn flushes)

The same three requests against the same model, reading the sidecar's total after each flush.  Each turn
adds 8 016 routing entries; the exit report's session total is 41 172 in both arms.

| build | after turn 1 | after turn 2 | after turn 3 | sidecar vs session total |
|---|---:|---:|---:|---|
| **as delivered (r79)** | 25 140 | **58 296** | **99 468** | **2.4 x** |
| **with the fix (r80, block 09)** | 25 140 | 33 156 | **41 172** | **1.00 x** |

The delivered numbers are exactly `loaded_k = 2*loaded_{k-1} + new` (25 140 -> 2*25 140 + 8 016 = 58 296;
58 296 + 41 172 = 99 468).  The patched arm is linear and the file *equals* the session total.
Three turns already inflate by 2.4 x; the Flash-Next sidecar on disk (99.05 % saturated) is what a normal
server session does after about six.

The second half of the patch was checked the same way: with a synthesised all-`0xffffffff` sidecar the run
logs `ignoring the heat map ... it is degenerate (2304 of 2304 entries saturated) ... Delete the file to
re-record it` and starts cold instead of pre-filling experts `0..slots-1`.

Reproduce with [`tools/heat-growth-test.sh`](tools/heat-growth-test.sh) (`./heat-growth-test.sh unfixed`
vs `... fixed` — three server turns, `MOE_EXPERT_HEATMAP_DEBOUNCE_S=0`, reading the sidecar's total after
each turn).

**Fix (shipped in r80, block 09)** — the prototype patch that proved it was `heatmap-doubling-fix.patch`
here; it is now superseded by `patches/0009-rdna-boosts-block-09-...patch` and was deleted so nobody
applies the pre-refinement form.  It does:

* **(a) Stop the doubling** — track `heat_merged` per table (the part of `heat` already folded into
  `g_heat_loaded`, seeded to the prior at registration) and merge the **delta**.  `MHT1` version stays 1.
* **(b) Guard the consumer** — at load, if ≥ 2 % of entries are saturated or every held expert is tied,
  log one `WARN` and **ignore the map** (run cold) rather than seed from it.  A poisoned sidecar cannot
  be un-multiplied; a sidecar is a cache, not a record.
* **(c) Repair the files on disk** — delete the three poisoned sidecars and re-record from one clean run.

---

## 5. "Four experts at a time", prefetch, and the Markov idea

**The "four" is not a transfer batch.**  Strata's per-round swap budget is **96 experts** (`adapt_swaps`
default 96, `src/program/generate.cpp:611`; paper §3.4: *"every four rounds it swaps up to 96 experts"* —
the "four" is the **round** period).  Three other things in their tree are plausibly what was seen:

1. **The verify window.**  `--spec 4` drafts 4 tokens; the window verifies up to ~4–6 at once
   (their log: *"window up to 6 tokens"*), and one expert fetch serves **every** token in the window
   that routes it.  So a transfer is "for a window", not "for a token" — the amortisation is real, and
   **we already have it**: our hook dedups fills within the ubatch too (first occurrence fills, the rest
   hit; `moe_cache_update_host` pass 1/pass 2).
2. **The PCIe staging group** — up to 16 staging blobs per layer-window (`kStagingBlobs = 16`,
   `src/core/verify.cpp:672`), not 4.
3. **The copy-engine PCIe share** (`--pcie-frac`) — 20 % of misses on Q2_0, 55 % on the i-quants.

**The router-prediction prefetcher exists, and it lost.**  This is the answer to the "Markov modelling /
pull in what will be needed next" hypothesis — Strata built exactly that and measured it:

* `RouterLookahead` (`include/strata/core/expert_source.hpp:240-300`): applies **layer l+1's own router**
  to layer l's input (*"the residual stream changes little from one layer to the next"*), predicts each
  token's top-k / k+4 / k+10 and a vote-ranked top 4/8/12, and warms/prefetches the ones no cache holds;
  depth up to 4 layers (`STRATA_IO_PREFETCH_DEPTH`).
* `ForesightSwap` (`STRATA_FS_SLOTS`): N spare VRAM slots per layer, refilled on a copy stream with *the
  experts that just missed plus the next layer's predictions*.
* **Measured result** (`docs/DETAILS.md:1679-1691`): decode 0.91 / 0.95 / 0.97 / 0.98 / 1.00 × — i.e.
  it lost or tied on every box (RTX 3060, P100, 5070, 2×3090), because the slots come out of the expert
  cache and at high residency "the misses per layer are below one, so the CPU round trip stays".
  Left off by default.

Their winning mechanisms were all **capacity or admission**, not prediction: size slots in bytes per
layer rather than to the largest blob (**+33 % experts, +13 % decode**, paper finding 8 — we already do
this), and evict-at-once / admit-when-landed so no one waits (finding 10).

---

## 6. Are we making the most of the arena? (what is actually recoverable)

Field GSQ 1-GPU slab layout (`seed-floor-full-arena-1gpu.log:235,245,526`):

```
slab 17.38 GiB (chunk 64 MiB): work 1.75 GiB + arena 15.62 GiB  (8.00 GiB reserve, 4.00 GiB VA spare)
arena 15666.6 MiB of 40925.0 MiB host experts  (38.3% residency)   144 tables, 196 slots/table
narrow-2 (MTP draft compute) pinned at [20.88, 21.38] GiB (0.50 GiB); the arena tops out at 15.62 GiB
```

(15.62 GiB is the slab's arena **region**; the **sized** arena is 15 666.6 MiB = 15.30 GiB after the
per-table slot rounding.  The 8 GiB and the 4 GiB are outside the region: held back from the arena.)

* **Inside the arena: nothing meaningful is wasted.**  Uniform per-table slots truncate 79.9 MiB
  (0.51 %); `head_pad` (`min(expert_bytes, 512)` per table) is covered by the `n_tables × unit` hold-back;
  and at steady state the slots are claimed — the r74 log heat-seeds **196/196 slots in every table above
  the prefill reclaim band** (101 of 144 here; the band's 43 tables are deliberately left empty by
  `heat_seed_above_floor` and fill lazily), and `moe_cache_rearm` restores the arena to the full
  15666.6 MiB after the first prefill's boundary move.  **There are no holes to defragment** — consistent
  with the campaign's own F2.
* **The real unused VRAM is outside the arena**: an **8.00 GiB compute reserve** plus **4.00 GiB VA
  spare** held back from the boundary.  Reclaiming even half of it is ~+4 GiB ⇒ ~+2 000 pairs ⇒
  +4–6 points of coverage at our residency (from §3's curve), i.e. 3–6× everything the campaign's
  phases 2–4 could win.  **That lever is owned by [`../slab-transient-absorption/`](../slab-transient-absorption/README.md)**,
  not by this campaign; it is named here so the two campaigns do not both pretend to it.
* **Capacity allocation (per-table quota vs a global ranking) is worth ~1.5–1.9 points**, measured on
  the clean 35B map (`--all`, `alloc` section: −1.87 pts at f = 0.20, −1.85 at f = 0.38, −1.14 at
  f = 0.637).  It is real, it is the campaign's only strictly-inside-the-arena win, and it is *not*
  defrag — it is how many slots each `(layer, role)` table gets (today: all equal, despite a 275× spread
  in per-table routing mass, `min 0.004 % / max 1.104 %`).  It collides with the per-layer slot
  uniformity invariant (gate/up/down of a layer route the *same* experts), so it has to be expressed as
  a per-*layer* quota, not per-table.

---

## 6.1 Where the ~10 GiB of arena actually is (2026-10-17, measured)

The field GSQ config on one 32 GiB R9700 (`SLAB_INIT dev=0 free=25.38 GiB reserve=8.00 GiB
headroom=4.00 GiB aux=4.00 GiB size(VA)=21.38 GiB mapped=17.38 GiB boundary=1.75 GiB va_spare=4.00 GiB
ring_req=4.11 GiB`; arena 15 666.6 MiB) against Strata's own log on the same card and model ("expert
cache 15667 slots, 25.46 GiB"; "584 MiB of VRAM free with everything loaded"):

| VRAM bucket | us (MiB) | Strata (MiB) | delta |
|---|---:|---:|---:|
| expert arena | 15 667 | 26 071 | **-10 404** |
| dense weights | 3 665 | 3 247 | +418 |
| MTP draft (weights + KV + compute) | 3 447 | 1 252 (835 draft + 417 Q5_K head) | +2 195 |
| KV + recurrent state | 3 025 | ~1 400 (streamed; see below) | +1 625 |
| always-mapped narrow compute | 1 792 (+512 narrow-2) | ~70 (decode buffers) | +2 230 |
| idle free at steady state | ~4 700 | 584 | +4 100 |
| **total** | ~32 600 | ~32 600 | |

**It is five things, not one, and the biggest is our own slack:**

1. **~4.1 GiB of idle VRAM** — `GGML_CUDA_SLAB_HEADROOM_MIB` (4096) + the unspent part of the draft `aux`
   estimate.  `ENVIRONMENT.md` measured ~3.7 GiB of post-slab allocations this headroom really pays for
   (MTP draft buffer 762 MiB, FA-QSA workspace 762 MiB, hipBLASLt Tensile objects + workspace, pool
   high-water), so it is not pure waste — but it is a *policy* number, not a hardware limit.
2. **+2.2 GiB: our MTP draft.**  We load a separate Q8_0 draft GGUF (2 647 MiB weights + 288 KV + 512
   compute); they use 2-bit draft experts + a 417 MiB Q5_K head (1 252 MiB).
3. **+2.2 GiB: an always-mapped narrow compute region.**  Our decode/verify compute view is resident at
   all times (1 792 MiB, plus 512 MiB for the draft context); their decode devices buffers are ~70 MiB
   because *every* buffer, prompt included, is borrowed from the arena.
4. **+1.6 GiB: KV.**  We keep the whole 131 072-cell window in VRAM (q8_0 K/V 1 836 MiB + 901 MiB
   recurrent state + 288 draft KV); they keep **32 768 of 131 072 cells per layer resident** and stream
   the rest from pinned RAM ("KV streaming: 32768 of 131072 cells ... the K/V in 1.55 GiB of pinned
   RAM").  Their KV VRAM at this context is ~1.4 GiB.
5. **+0.4 GiB: dense weights** (3 665 vs 3 247) — quant/pack choice, negligible.

**We already do their borrow trick.**  `ggml-cuda-slab.cu:126-132`: "the always-resident NARROW
(decode/verify) compute view owns `[0, narrow_floor)`, the H2D staging ring owns
`[narrow_floor, narrow_floor + ring_region)`, and the transient WIDE (prefill) compute view owns
`[narrow_floor + ring_region, boundary)`.  The arena is always `[boundary, mapped)` ... When the wide
view is dropped the boundary shrinks back to `narrow_floor` and the ring + wide regions return to the
arena".  Measured on the field run: the first prefill evicts **6150 MiB** (the band) and
`moe_cache_rearm` restores the full 15 666.6 MiB; the r74 seed floor exists so nothing is pre-seeded into
that band.  Strata's "the prompt path borrows 4374 CUDA0 cache slots (7.13 GiB)" is the same mechanism —
the difference is that their *decode-time* buffers are tiny and ours are a permanent 1.8 GiB.

### Measured levers (same card, same model, same command, box idle)

**(a) The outside-slab headroom is worth ~1 pt of hit rate per GiB** — `GGML_CUDA_SLAB_HEADROOM_MIB`,
16 KB prose prompt + 400 generated, MTP, cold start each time:

| headroom | arena | residency | first-prefill eviction | hit rate |
|---|---:|---:|---:|---:|
| 4096 (default) | 15 666.6 MiB | 38.3 % | 6 150 MiB | 0.9056 |
| 3072 | 16 705.7 MiB | 40.8 % | — | 0.9147 |
| 2048 | 17 744.8 MiB | 43.4 % | 6 208 MiB | 0.9223 |

and on a 16 384-token prompt (4x the prose file, 200 generated): 0.9165 -> **0.9380** at 2048, generated
text **byte-identical** (`9ac8737781aa…` at 1x; the 4x pair both start identically and produce the same
answer), arena re-armed to the larger size in both arms.  **The default stays 4096** — `ENVIRONMENT.md`
records that 3072 was rejected on a harsher config (`-ub 6144 -c 204800`, `--fit off`, `-sm tensor`), so
this is a documented per-config tuning value, not a new default.  It is the cheapest ~2 GiB available.

**(b) The heat map itself is worth +1.4 pts of hit rate** — same 16 384-token prompt, same arena
(15 666.6 MiB), same reaches (434 760), only the map differs:

| arm | hit rate | hits / reaches |
|---|---:|---|
| `MOE_EXPERT_HEATMAP=0` (no seed, id-order placement) | 0.9022 | 392 229 / 434 760 |
| `MOE_EXPERT_HEATMAP=load` (clean map: seeded + heat-ordered placement) | **0.9163** | 398 385 / 434 760 |

That is the value r60 was *supposed* to be delivering: pre-r80 the same two arms were the same
configuration (a poisoned map ranked by expert id), which is why the r74 -> r80 field number looked
unchanged.  Turn time was 0.53 s vs 0.52 s — the hit gain is real but the remaining ~2 % of misses still
dominates the step.

**(c) Not measured, but structural:** a smaller MTP draft (Q4/Q2 experts instead of the Q8_0 GGUF) would
hand ~1.5-2 GiB to the arena; KV window streaming (Strata's `--kv-resident`) would hand ~1.3 GiB more.
Both are engine/content changes, and the second is a campaign of its own.

---

## 7. Verdict on the "defrag" premise

| premise | verdict | evidence |
|---|---|---|
| Expert sizes vary ⇒ intra-table holes | **No.** One contiguous `slots * expert_bytes` block per table; eviction frees a *logical* slot.  Already F1/F2. | `table_t`, `access_locked` |
| The slab is fragmented by heterogeneous tables | Holes exist *between* tables, but they are not a hit-rate tax: eviction is whole-table and the boundary move takes the lowest chunks.  The cost of a boundary move is *which tables* it takes — placement (phase 1, shipped), not compaction. | `moe_cache_evict_slab_range`, `ggml_cuda_slab_work_alloc` |
| A repack at idle would recover capacity | It would recover **bytes**, not **slots**: the arena is already fully allocated (0.5 % truncation).  Moving a table's base invalidates captured graphs and the arena address in kernel launch params, for ≤ 80 MiB. | §6 |
| A repack would raise the hit rate at the current residency | The churn it would fix is tiny: over the field turn, **8 753 evictions against 292 200 reaches (3.0 %)**. The transient that costs hit rate is the cold start (**37 146 fills ≈ 1.3 × the arena**), and that is what the seed exists for. | r74 field log |
| Pulling experts in larger batches / predicting the next ones would raise the hit rate | Measured twice: Strata's `ForesightSwap` lost on every box (§5); our own r60 win came from *seeding*, not lookahead.  Grouping also cannot help the *ratio* (the dedup transform is linear, §1). | `docs/DETAILS.md:1679`; `archive/work/moe-heatmap-persistence/` |

**Net: the campaign's name and phases 2–4 are aimed at a tax that does not exist.**  Keeping them would
mean spending multiple sessions on graph-invalidation hazards for ≤ 1 % of arena bytes.

---

## 8. Retarget proposal (APPROVED and done 2026-10-17)

Rename `wip/moe-arena-defrag/` → **`wip/moe-cache-coverage/`** — **done**, with `TODO.md` #61 and
`wip/CAMPAIGNS.md` rewritten in the same change; P0 shipped in r80.  (Alternatives considered:
`moe-expert-cache-policy`,
`moe-cache-hitrate`; "coverage" is the quantity `H(f)` that all of this is about, and it is what the
maintainer's "hit rate" question reduces to).  New phases, in value order:

| phase | what | why | size |
|---|---|---|---|
| **P0** | **Heat-map correctness**: stop the doubling (§4.3a), add the degenerate-map guard (b), delete the poisoned sidecars, re-record one clean map.  Block 09. | Every heat-driven feature is currently ranking by id, silently.  Default-on feature, so this is a *correctness* fix, not an optimisation. | small |
| **P1** | **Offline replay instrument**: one routing-trace dump (`MOE_EXPERT_TRACE=<path>`: per decode token/window, `(layer, expert, rank)`), plus a replayer that scores policies against it (F-LRU as shipped, static top-f, per-layer quota, prefetch-on-miss, window dedup on/off). | We have only an aggregate heat vector, which cannot answer *any* policy question (thrash, prefetch, window effects).  Strata tuned its adaptive tier exactly this way (`--dump-routing` + `tools/replay_cache_policy.py`).  One GPU run then answers a dozen questions for free. | medium |
| **P2** | **Per-layer capacity allocation from heat** (floor + quota, keeping the gate/up/down alignment), A/B on the P1 replayer and then on the field config. | The only strictly-inside-the-arena win — but only **0.3–0.5 pts** on a uniformly tallied map (§3.1); the 1.5–1.9 figure was a tally-gap artifact.  Do it only if P1 shows it is cheap. | small |
| **P3** | **Metric parity**: report, per turn, the per-token `h`, the **covered mass** of the resident set, **expert MiB/token**, and an estimated static bound — plus the accounting definition in `patches/README.md`. | "90 % vs 99 %" cost a session to untangle; a two-number report makes it self-evident next time. | small |
| **P4** | *(hand-off, not this campaign)* **arena size**: the 8 GiB reserve + 4 GiB VA spare → `archive/work/slab-transient-absorption/`. | 3–6× the win of P2. | — |

Parked/closed: the relocate-and-resize repack (old phase 3) and selective eviction (old phase 4) —
kept as history below, **not planned**.

---

## 9. Open questions for the maintainer

1. ~~P0 is a delivery change~~ **DONE 2026-10-17: shipped in `v12-097cf33d9-r80`** (block 09; delta merge +
   degenerate-map guard), released and pushed.
2. ~~Rename + `TODO.md` #61 rewrite~~ **DONE 2026-10-17**: `wip/moe-cache-coverage/`, with `TODO.md` #61
   and `wip/CAMPAIGNS.md` rewritten in the same change.
3. ~~The exact 1-GPU GSQ command~~ **DONE 2026-10-17** — the maintainer's field command was used to
   re-record the Flash-Next map on the r80 build (§3.1).  Field check on that run: arena 15 666.6 MiB,
   38.3 % residency, hit 0.9056 (724 928 / 800 520), identical to r74's 0.9057.
4. The **PCIe share of misses** (Strata finding 9: the copy engine serves 20–55 % of misses beside the CPU
   pool) is an *overlap* idea, not a cache idea.  Point it at `archive/work/moe-cpu-overlap/` rather than here?

---

## 10. Tool added

`tools/heatmap-audit.py` (stdlib only) — reads an `MHT1` sidecar and reports:

* `--health`: saturation / zeros / distinct values **and the degeneracy test** (`H(0.5)` on the uniform
  line ⇒ the map cannot rank).  This is the regression test for §4.
* coverage `H(f)` and the f needed for 90/95/99 % of routed mass (§3).
* capacity allocation: global-top-f vs our uniform per-table slots at the same held fraction (§6).
* window accounting: per-token vs per-window-distinct hit rate, with the linearity check (§1).
* `--arena-mib N`: reproduce `alloc_all_locked`'s `slots = budget / Σ expert_bytes`, the truncation loss,
  and the coverage that residency implies.

```bash
tools/heatmap-audit.py --health /llm/models/*/*/*/*.moe-heatmap   # any run that learned something
tools/heatmap-audit.py --all --arena-mib 15666.6 <sidecar>        # the campaign's full arithmetic
```

---

## 11. Incidental finding (separate from this campaign): `-cmoe` aborts

Found while looking for a cheap model to exercise the heat map with.  On a host-expert MoE,
**`-cmoe` aborts in the `llama_context` constructor**:

```
ggml/src/ggml-backend.cpp:921: GGML_ASSERT(device) failed        (ggml_backend_dev_memory(nullptr, ...))
```

`-cmoe` (`--cpu-moe`, all experts on the CPU) routes the expert tensors through the **CPU host buffer**,
whose `ggml_backend_buft_get_device()` is `nullptr`; the delivery's accounting
(`moe_host_expert_account`, `src/llama-model-loader.cpp:1166-1192`, block 09) then does
`acc[buft_dev] += bytes` — i.e. it inserts a **null device key** — and every consumer that walks
`model.moe_host_expert_bytes` (the H2D-ring sizing at `src/llama-context.cpp:765`, and the later loops)
calls `ggml_backend_dev_memory()`/`ggml_backend_dev_name()` on it.  Reproduced on gpt-oss-20B MXFP4 with
both `GGML_CUDA_SLAB=1` and `=0` and with the cache off, so it is the ring/accounting path, not the
arena.  `-ncmoe 99` is fine (the map key is then a real device), so the field configs are unaffected;
`-cmoe` is broken.  One-line fix candidate: skip (or attribute to the layer device) a host buft with no
device in `moe_host_expert_account`.  **Owed to `TODO.md` as its own item** — it is not this campaign's
scope and was not touched.

---
