# Phase 2b — where the pool's CPU goes, and a plan

Status: **targets 1-5 implemented + landed (2026-10-08)**; Phase 3 remains the open item.
Nothing in `patches/`/`release.json` was touched and nothing was pushed.

**Update 2026-10-08 (target 1 landed -- pool-aware device policy).**  Pooled tables now use the
device-side admission policy (the `!g_pool_enabled` gate is removed), and the kernel's fill sources the
bounded host pool when the expert is pool-resident (a device `expert -> pool_slot` map snapshotted from
the pool LRU each flush) and falls back to the master otherwise.  The per-table promote keeps the pool
warm through the target-2 background worker (a synchronous pre-fill measured far slower: 26.3 vs 43.1).
`MOE_HOST_POOL_POLICY` selects `1` pool-aware (default), `0` host promotion (original), `2` master fill
(1b). 35B-A3B Q4_K_M `-sm layer -ncmoe 40`, `MOE_EXPERT_CACHE_MIB=2048` + `MOE_HOST_POOL_MIB=2048`,
prose `-n 128`: host promotion **41.0** -> pool-aware **43.1** t/s (pool off **45.4**).  A bring-up
kernel page fault was a bug in this work (the fill added `e*host_bytes` on top of the slot base), not a
pool/virtual-memory limit.  Gates green: short `359ff4337837`, long-prefill `12d4fcd10886` (both
splits), dense 4B `1c5d32ac537d`, `MUL_MAT_ID` 931/931, prefill-logit KLD 0.000707 / 98.755 %.

**Update 2026-10-08 (target 3 landed).**  The prefill tally now drives both the arena seed and the pool
ranking, independent of the device-policy gate (`seed_prefill_lazy_locked` iterates the device's tables
and runs outside `if (g_devpolicy)`; `pool_rerank_from_tally_locked` is new and excludes arena residents,
i.e. the pool is the additive complement).  35B-A3B Q4_K_M, prose prompt (5258 tokens), `-n 128`,
`MOE_HOST_POOL_MIB=2048`:

| arm | gen t/s |
|---|---:|
| `-sm layer` pool on, seed **on** | **41.1** |
| `-sm layer` pool on, seed off | 36.3 |
| `-sm layer` pool off (device policy) | 42.0* / 41.5† |
| `-sm tensor` pool on, seed on | 17.3 |
| `-sm tensor` pool on, seed off | 17.5 |

`*` short prompt; `†` `-sm layer` host promotion with `DEVPOLICY=0` (master source) at `-n 64`.  The
seed closes most of the pool penalty on `-sm layer` (the reporter's mode); `-sm tensor` is unaffected.
Gates: short `359ff4337837`, long-prefill `12d4fcd10886` (all four pool/split arms), dense 4B
`1c5d32ac537d`, `MUL_MAT_ID` 931/931, prefill-logit KLD 0.000707 / 98.755 % PASS.

**Maintainer's direction (2026-10-08):** the pool must be **additive** — once the GPU has a weight block
(its arena copy has drained) the pool releases that entry, and when the GPU evicts a block the pool
fetches it from disk so it is resident for re-use.  The only in-transit exception is a pinned block
whose arena copy has not drained.  Budget is **process-wide**, and the implication is a **host-wanted
active prefetch/streaming** pattern that keeps the pool ready with what the devices need next.  Question
3 is answered: the pool need not be ≥ the arena hot set; it is the complement of the arena.

**Update 2026-10-08 (target 4 landed — 4a + 4b).**

* **4a process-wide pool.**  `g_host_pools` is keyed by the host tensor only; `MOE_HOST_POOL_MIB` is the
total non-swappable budget split over all registered host tensors (one whole-expert copy serves every
device).  Per-`(slot, device)` in-flight events; the pool is `cudaHostAlloc(Portable|Mapped)` and the
arena fill reads the UVA device alias with `cudaMemcpyDeviceToDevice`.  Two bugs found and fixed on the
way: (i) a cross-device H2D from a plain `cudaMallocHost` pool is treated as pageable and crashes in the
run-time staging `memmove` (gdb backtrace); (ii) the meta splitter hands the middle device of an axis
split a **degenerate** simple tensor (`ne[split]==0`, `nb[2]==0`) that the cache registered with
`expert_bytes = host_bytes` and a nonzero `slice_off`, so a pool fill read past the slot — the pool
source is now guarded on `src_off + expert_bytes <= host_bytes` and falls back to the master.
* **4b additive lifecycle.**  Per-slot state `0=free / 1=resident / 2=retiring`: an arena eviction
fetches the victim into the pool, and an arena admission marks the entry retiring and frees it
(`pool_reap_locked`, non-blocking `hipEventQuery`) as soon as the copy drains.  Retiring slots are never
reused or evicted while a copy is in flight.

35B-A3B Q4_K_M, prose prompt, `-n 128`, `MOE_HOST_POOL_MIB=6144` process-wide:

| arm | 4a (non-additive) | 4b (additive) | pool h (4b) |
|---|---:|---:|---:|
| `-sm layer` | 41.1 | 39.1 | 0.224 |
| `-sm tensor` | 18.7 | 17.1 | **0.493** |

So the hit rate rose 2-8x, but t/s dipped ~5-9 %: the eviction fetch is still **synchronous and on the
critical path**.  That is target 2's job (overlap / host-wanted streaming) — the additive pool is the
prerequisite that makes an overlap worthwhile.  Note target 1's original 2x `-sm layer` gap (42.0 vs
17.4) is now nearly closed by target 3 + the additive pool (41.1 / 39.1): re-validating it at this
point may show only a residual benefit.  Gates green at 4b: short `359ff4337837`, prose `12d4fcd10886`
(all four pool/split arms), dense 4B `1c5d32ac537d`, `MUL_MAT_ID` 931/931, prefill-logit KLD 0.000707 /
98.755 % PASS.

**Update 2026-10-08 (target 2 landed — background prefetch).**  The eviction fetch now runs on a
detached **nice-19** worker: the eviction enqueues the expert, the worker reserves a slot and opens the
fd under `g_mutex`, reads the GGUF **without** it, then publishes.  Dedupe both ways — the worker drops
an expert the critical path already populated (state 1) or reserved (3/4); the critical path treats a
state-3/4 entry as a miss, reads the master, and cancels the reservation (3 → 4) so the pool never keeps
a duplicate.  `MOE_HOST_POOL_BGFETCH=0` restores the synchronous 4b path.

| arm | 4a | 4b (sync) | 2 (background) | pool h |
|---|---:|---:|---:|---:|
| `-sm layer` | 41.1 | 39.1 | **41.9** | 0.29 |
| `-sm tensor` | 18.7 | 17.1 | **18.7** | 0.46 |

The background queue recovers the 4b regression *and* keeps the additive hit rate, i.e. the pool is now
net-neutral-to-positive rather than a wash.  A deadlock found and fixed en route: the main thread's
`__cxa_finalize` destroyed the static `condition_variable` while the detached worker was still waiting
on it (`pthread_cond_destroy` hang, seen as a stray 27 GiB/GPU process that starved the next runs) — the
queue/cv/mutex are now deliberately heap-leaked, process-lifetime primitives.  Gates green at target 2:
all four coherence arms, dense 4B, `MUL_MAT_ID`, prefill-logit as above.

**Remaining:** target 1 (pool-aware device policy) is now a residual; otherwise the campaign's open item
is Phase 3 (remove the pinned master and the scheduler fallback).

**Separate block-12 bug found while answering "why is `-sm tensor` so slow?":** the default hybrid
all-reduce eagerly initializes NCCL, and that alone makes the internal pipeline ~3x slower for
`-sm tensor` decode (3x R9700 `-ncmoe 0`: 21.6 -> 69.6 t/s after deferring NCCL to the first large
tensor).  Full analysis, the fix and validation: [`SM-TENSOR-AR.md`](SM-TENSOR-AR.md).  It is a block-12
change and needs the maintainer's sign-off before any delivery fold.

Read with [`PHASE2B-HANDOVER.md`](PHASE2B-HANDOVER.md) (targets), [`PHASE2.md`](PHASE2.md) (the pool
design / the page-cache finding) and the repo [`AGENTS.md`](../../AGENTS.md).

## TL;DR

1. **The pool is a net loss at its default bounded size, because it does not hit.**  Pool (L2) hit
   rates measured: **3.8 %** (35B `-sm tensor`, 2048 MiB/device), **9.8 %** (35B `-sm layer`), **20.4 %**
   (Flash-Next reporter, 40 GiB).  Every arena fill is therefore a page-cache→pinned `pread` that the
   pool-off path never does (it reads the already-pinned master).  `strace -c` on 35B `-sm tensor`,
   `-n 16`: pool on = **21262 `pread64` / 0.48 s** syscall time; pool off = **20 / 0.009 s**.
2. **Under `-sm layer` the pool also switches the device-side admission policy off**, and that alone is
   worth **2x** decode: pool off **42.0 t/s** (device policy) vs pool off + `DEVPOLICY=0` **21.1 t/s**
   (host promotion) vs pool on **17.4 t/s**.  This is target 1 and it dominates every other item.
3. **The prefill seed and the pool prewarm are dead whenever the pool is on** (`seed_prefill_lazy_locked`
   is called only from the `g_devpolicy` branch of `moe_cache_policy_flush`, but the pool forces
   `t.policy = false`).  The device tally is collected and never consumed; the pool prewarm runs at
   load-time and ranks from "1 arena-hot" expert.  This is target 3.
4. **The pool stores whole host experts per device**, so under `-sm tensor` it holds ~1/3 the experts the
   arena does for the same bytes, and duplicates that per device.  This is target 4.
5. `tools/cli/cli.cpp:36` sets llama-cli's default verbosity to **`LOG_LEVEL_ERROR`**, so WARN *and* INFO
   are filtered on a plain run.  The correct flags are `-lv 2` (the pool summary, now WARN) and `-lv 4`
   / `-v` (the full `moe_cache_report`).  The handover's `-lv 1` was wrong (level 1 is "error").

## Measurements

35B-A3B Q4_K_M, 3×R9700 (gfx1201), ROCm 7.14, `-ngl 99 -ncmoe 40 -fa 1 -t 8 -n 256`, `MOE_EXPERT_CACHE_MIB=2048`,
same-seed greedy.  All four coherence configs are byte-identical (`359ff4337837`).

| config | gen t/s | prompt t/s | arena h | pool h | pool fills | pool evictions |
|---|---:|---:|---:|---:|---:|---:|
| `-sm tensor` pool off | 6.9 | 16.2 | 0.610 | – | – | – |
| `-sm tensor` pool 2048 | 6.3 | 14.9 | 0.610 | 0.038 | 21242 | 11198 |
| `-sm tensor` pool 2048 `PREWARM=0` | 6.2 | 14.9 | 0.610 | 0.000 | 12045 | 3552 |
| `-sm layer` pool off (device policy) | **42.0** | 26.1 | 0.716 | – | – | – |
| `-sm layer` pool off + `DEVPOLICY=0` (host promotion) | 21.1 | 25.3 | 0.515 | – | – | – |
| `-sm layer` pool 2048 (host promotion) | 17.4 | 24.1 | 0.515 | 0.098 | 15018 | 4873 |

Flash-Next IQ4_NL reporter (2×R9700, `-ncmoe 48 -sm layer -c 16384`, `MOE_HOST_POOL_MIB=20000` = 40 GiB,
5246-token prose prompt, `-n 150`): prompt **127.6 t/s** / gen **8.0 t/s**, halo **0.204** (16150/79267),
63117 pool fills / 17691 evictions, 202 MiB expert H2D per token.  (This run was cold-page-cache: 57 GB
read from disk during the run; the handover's 18.8 t/s was a warm-cache measurement.  Warm-cache
re-measure pending.)

Pool-size curve (same 35B run, `-v` report; the pool `h` includes the prewarm fills, so the decode hit
rate is `pool_fills - pool_evictions` over the arena fills):

| config | gen t/s | pool slots/pool | pool fills | pool evictions | decode pool hits / arena fills |
|---|---:|---:|---:|---:|---:|
| `-sm layer` 2048 | 17.4 | ~85 | 15018 | 4873 | 10145 / 6495 (miss) |
| `-sm layer` 8192 (= **full residency**, 256/256) | 20.9 | 256 | 30720 | 0 | 6495 / 6495 (**100 %**) |
| `-sm tensor` 2048 | 6.3 | ~28 | 21242 | 11198 | 847 / 12045 (**7 %**) |
| `-sm tensor` 8192 | 6.4 | ~113 (of 256) | 48253 | 7477 | 4580 / 12057 (**38 %**) |

Two things fall out.  First, **when the pool hits, the pool source is neutral**: `-sm layer` 8192 is a
full-residency pool and it lands at 20.9 t/s, i.e. on top of the host-promotion pool-off arm (21.1) —
the copy is the same H2D either way.  So the entire pool penalty is misses, not the lookup/copy.
Second, a *partial* pool with an id-order prewarm still misses 62 % of fills (`-sm tensor` 8192), which
is exactly target 3: the ranking, not the size alone, decides whether a bounded pool is useful.

`strace -c -f -e trace=pread64` on the 35B `-sm tensor`, `-n 16`:

| arm | pread64 calls | pread64 sys time |
|---|---:|---:|
| pool on | 21262 | 0.480 s |
| pool off | 20 | 0.009 s |

1329 `pread`s per token at pool-on.  The lookup (`unordered_map`) and the `cudaMemcpyAsync` enqueue are
noise next to this; the synchronous buffered read is the miss path.

## Root causes

### The pool is filled on arena *admission*, so it can never beat the arena it mirrors
`access_locked` calls `pool_slot_locked` only on an arena *fill* (after admission), so the pool holds the
most-recently-admitted experts with an LRU.  A future arena fill for expert `e` only happens after `e`
was evicted from the arena — by which time the pool has evicted it too, because the pool is *smaller in
expert count* than the arena (it stores whole host experts while the arena stores per-device slices).
Net: ~0 % reuse.  A pool equal to or larger than the arena's hot set would hold `e`; a pool smaller than
it is pure cost.  The full-residency pool (27000 MiB) is neutral (6.7 vs 6.9) exactly because it never
misses.

### The prewarm cannot rank the hot set
`pool_prepopulate_locked` ranks by `t.slot_expert` (the arena's residents at pool-build time) then
`t.prefill_count` then id.  The pool is built lazily on the first fill during the load-time warmup, when
the arena has ~1 resident, and `t.prefill_count` is **empty** under `-sm tensor` (block-06 staging
intercepts `moe_cache_update_host`'s prefill branch).  Every prewarm log reads `(1 arena-hot first)`, so
the ~10–300 slots/pool are essentially id-ordered and useless.

### The prefill tally is collected but never consumed under the pool
`moe_cache_tally_prefill` runs (gated `g_prefill_seed && g_devmap`, not `g_devpolicy`) and fills
`prefill_count_dev`, but the only consumer, `seed_prefill_lazy_locked`, is called only inside
`if (g_devpolicy)` in `moe_cache_policy_flush`.  The pool forces `g_devpolicy` tables off, so the tally
is dead work and the arena gets no prompt seed either (no `prompt-routing seed` logs at all).

### The device policy is disabled wholesale by the pool
`moe_expert_cache.cu` (~line 1831): `if (g_devpolicy && !g_pool_enabled && t.devmap && ...)`.  Pool on ⇒
every table uses `moe_cache_promote_host` (per-table D2H used-list + host LFRU + slot-map H2D), which is
2x slower than the one-batched-kernel-per-device device policy.  (Independent of the pool, `-sm tensor`
split tables already default the device policy off via `devpolicy_split`; that is why `-sm tensor` shows
no such cliff.)

### Per-device pools duplicate whole experts
`g_host_pools` is keyed `(t.host, t.device)` and `MOE_HOST_POOL_MIB` is per device.  Under `-sm tensor`
the same host tensor gets one whole-expert pool per device — 3× the pinned bytes for 1× the experts —
and the arena's per-device slice means the pool's usable expert count is `host_bytes/expert_bytes` times
smaller than the arena's for the same byte budget.

## Proposed designs and order

Recommended order: **3 → 4 → 1 → 2** (make the pool useful first, then the policy, then the residual
micro-optimisation); approved by the maintainer.  **Targets 3 and 4 are done** (4a + 4b above).
Remaining: **target 2 is the highest-value next step** (overlap the now-synchronous eviction fetch so the
additive hit rate becomes speed); **target 1 should be re-scoped** — target 3 + the additive pool have
closed almost all of its original 2x `-sm layer` gap, so a pool-aware device policy is now a residual,
not the headline.  Each step is independently gated (coherence, `MUL_MAT_ID`, prefill-logit KLD); per
repo policy every new beneficial path ships default-on with a kill-switch and must clear the full gate
matrix first.

### Target 4 — additive, process-wide pool (maintainer's contract) — DONE (4a + 4b)
The pool becomes the **complement of the arena**, not a duplicate:
* On an arena admission of expert `e`, `e` is released from the pool once its arena copy has drained
  (`slot_ev`); until then it is the in-transit exception and stays pinned.
* On an arena eviction of `e`, `e` is fetched into the pool (buffered `pread`), ready for re-use.
* One pool per host tensor (keyed `t.host`, not `(t.host, t.device)`) with a **process-wide** budget, so
  a whole expert is stored once and every device copies its slice from it; per-slot in-flight tracking
  must be per device (a shared pool + one cross-device event fails `invalid resource handle`).
* A **host-wanted streaming** worker keeps the pool ready with the next-needed experts (the recent
  device routing + the prefill tally are the predictor), shielding the devices from storage latency.
* `seed_prefill_lazy_locked` already produces the wanted order for the seed; the streaming worker is the
  steady-state extension of the same ranking.

### Target 3 — feed the device prefill tally into the seed *and* the pool prewarm (cheap, high leverage)
* Decouple the seed from `g_devpolicy`: call `seed_prefill_lazy_locked` from `moe_cache_policy_flush`
  whenever a device tally is pending, even when no table has `t.policy` (pool/host-promotion path).  It
  already seeds the host mirrors via `apply_prefill_seed_rank_locked`, which the host-promotion path
  reads, so no new machinery.
* Make the pool prewarm use the same tally.  The pool is currently created during the load-time warmup
  (before prefill), so either (a) defer pool creation to the first decode-band pass (after the seed), or
  (b) on `seed_prefill_lazy_locked`, re-rank + refill the pool (`pool_prepopulate_locked` with the
  tally).  (b) is a superset of the current prewarm and keeps the load-time warm.
* Expected effect: pool h rises from ~4 % to the arena's hot-set fraction; the `pread` storm collapses.

### Target 4 — process-wide pool, sized against the arena hot set
* Key one pool by `t.host` (not `(t.host, t.device)`) so a whole expert is stored once and every device
  copies its slice from it.  Per-slot in-flight tracking must become per-device (the Phase-2 note
  already found a shared pool + a single cross-device `cudaEvent_t` fails with `invalid resource
  handle`): either an event per `(slot, device)` or a small per-device ready-flag array.
* Decide budget semantics explicitly (the handover's target 4): per-device (today; `20000` = 40 GiB on
  2 GPUs) vs process-wide (`20000` = 20 GiB).  Suggest process-wide with an explicit
  `MOE_HOST_POOL_MIB_TOTAL`, because that is the non-swappable bound the campaign is trying to cap.
* Size the per-table pool to at least the arena's uniform slot count (the pool is created before
  `alloc_all_locked`, so it needs the sizing to move earlier or a second pass).

### Target 1 — pool-aware device policy (now a residual; re-scope before doing it)
Maintainer's sketch: a device `expert -> pool_slot` map; the host pre-fills the pool for the token's used
experts before `moe_cache_policy_flush`.

* 1a (full): add `pool_base` (the `cudaMallocHost` UVA base), `pool_slot_dev` (`int32[n_experts]`,
  `-1` = not pool-resident) and `pool_host_bytes` to `moe_cache_policy_desc`.  The policy kernel's fill
  phase reads `pool_base + pool_slot[e]*host_bytes + src_off`; a `-1` falls back to `host_dev` (the
  master still exists in Phase 2), so correctness does not depend on the pre-fill being complete.  The
  host pre-fills the pool for the used experts in the per-table promote (it already D2Hs the used list
  there — only the per-expert host LFRU loop and the per-table `slot_dev` H2D are dropped), then the
  flush runs the batched admission as today.  The win is the removal of 120 per-table host calls/token;
  the pool preads remain until target 3/4 raise the hit rate.
* 1b (interim probe): enable the device policy for pooled tables with the master fill (drop
  `!g_pool_enabled`).  Recovers the 2x immediately but leaves the pinned pool unused for `-sm layer`;
  useful only to bound the target-1 win, not as a landing.

### Target 2 — the residual miss path (after 3 + 4)
Once the hit rate is high the `pread` storm is gone.  Residual: the synchronous buffered `pread` under
`g_mutex` (a page-cache copy) and the same H2D the master path does.  Options, in order of value/risk:
overlap the fill with the previous slot's H2D via a small staging ring; move the `pread` out of
`g_mutex` with a per-pool fill mutex (the LRU map stays under `g_mutex`); or a background fill thread.
`g_mutex` itself is uncontended on the decode path (single scheduler thread), so its hold time only
matters as latency, not as waiting.

### Target 5 (done) — pool stats visible
The pool summary is now `GGML_LOG_WARN`; `-lv 2` shows it, `-lv 4`/`-v` shows the full report.

## Open questions for the maintainer

1. **Order:** 3+4 first (make the pool useful, then 1), or 1 first (bank the 2x on `-sm layer` and accept
   a thrashing pool in the interim)?
2. **Budget semantics:** process-wide (`MOE_HOST_POOL_MIB` = the total non-swappable bound) or keep the
   per-device reading?  The measurements favour process-wide for `-sm tensor` (3× the experts for the
   same bytes).
3. **Pool vs arena sizing:** is a pool required to be ≥ the arena hot set to count as "working", or is
   the intended reading that the arena (VRAM) is the hot set and the pool only has to beat disk?  The
   data says the pool must be ≥ the arena hot set to hit at all.
4. **Phase-3 contract:** for 1a, is a per-device expert-in-flight table acceptable overhead, or should
   the pool pre-fill be demand-driven by a device "wanted" list + a host fill thread (keeps the policy
   fully off-host, but reintroduces a background copy)?
