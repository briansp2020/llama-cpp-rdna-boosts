# `slab-ring-region` — make the H2D staging ring a first-class slab region

**Status: P1 IMPLEMENTED (2026-10-09/10), BLOCKED on the decode reclaim.**  The slab-side plumbing, the
`--fit` lockstep and the arm/disarm lifecycle are in (`slab-ring-p1.patch`, from an r38 tree); the standing
gates are green and the field **prefill** win is real, but handing the ring's hole back to the arena during
decode corrupts the MTP draft (NaN).  See §9 for the record.

This campaign exists because of the §5.5 finding in
[`archive/work/fit-slab-accounting/README.md`](../../archive/work/fit-slab-accounting/README.md) §15.9: the G4 free-VRAM cap
refuses the op-offload H2D staging ring on the maintainer's field config and costs ~16 % prefill.  The
maintainer's framing: *the ring is our construction; it should live inside the movable-boundary slab,
where we control placement, instead of fighting the general HIP allocator.*

## 1. The problem, precisely

The ring (`h2d_stage_buffer`, `common.cuh`) is a raw `cudaMalloc` — general HIP VRAM, **outside** the
slab.  So are the FA prefill staging arena (`fattn_stage_try_get`) and the consumers the slab cannot
route (hipBLASLt's Tensile code objects + workspace, the MTP draft buffer, compute-buffer growth).  All
of them draw on the one region the slab leaves outside itself (`GGML_CUDA_SLAB_HEADROOM_MIB` + the draft
aux).  When that region is thin, an optional transient can take it and the next non-routable allocation
aborts (`exit 134`) or corrupts — which is what the G4 cap exists to prevent, at a prefill cost.

The slab is a **two-region** design: `work = [0, boundary)` (the compute buffer, grows by moving the
boundary up) and `arena = [boundary, mapped)` (the evictable MoE expert-cache tables).  There is no
non-evictable partition, so the ring has nowhere to go.

### Why it does not simply use `ggml_cuda_slab_arena_alloc`

`ggml_cuda_slab_work_alloc` (`ggml-cuda.cu:1030`) grows the work region by taking the contiguous range
`[boundary, target)` and calling `moe_cache_evict_slab_range(device, lo, hi)`
(`ggml-cuda.cu:1068` -> `moe-expert-cache.cu:3501`).  That eviction walks **`g_tables` only** (the MoE
expert tables); it cannot relocate or preserve anything else.  An in-flight ring buffer placed in the
arena free list would be silently overlapped by the next boundary move (the compute buffer's base slides
over its VA).  The arena is deliberately "the evictable store"; the ring is a non-evictable transient.

### The hook that was meant for this, unimplemented

`ggml-cuda-vmm.h:63` declares

```c
void * ggml_cuda_slab_arena_alloc_transient(int device, size_t size);
```

There is **no definition and no caller** anywhere in the tree.  It is the vestige of "a slab allocation
the boundary move must not take" — exactly the ring.

## 2. Current slab internals (for the implementer)

`ggml_cuda_slab` (`ggml-cuda.cu:592`):

| field | meaning |
|---|---|
| `base`, `size` | the VA reserved once (`cuMemAddressReserve`); `size` may exceed `mapped` |
| `mapped` | how much of `size` has physical behind it; the **arena top is `mapped`, not `size`** |
| `chunk` | boundary-move granularity; `unit` | finer arena-alloc granularity |
| `boundary` | work `= [0, boundary)`, arena `= [boundary, mapped)` |
| `work_needs` | multiset of live work-view requested sizes (the shrink floor) |
| `arena_free` | arena free runs `offset -> size`; filled **top-down** (`arena_alloc`) |
| `cache_unusable` | the slab declined (`work + min_arena` did not fit): the cache streams |

`ggml_cuda_slab_init_locked` (`ggml-cuda.cu:896`): `want = (free - reserve)/chunk*chunk` (chunk-aligned
physical), `w = up(work_min)`, `min_arena` floor, VA `size = (free - headroom)/chunk*chunk` (or `want`),
`mapped = want`, `arena_free = {w: mapped - w}`.  `ggml_cuda_slab_extend` later maps
`[mapped, min(size, free - headroom))`, growing `mapped` and the arena.

Ring accounting today: `h2d_stage_bound(max_table) = slots * (max_table + 512)` (`common.cuh`, exposed to
`--fit` as `ggml_backend_cuda_h2d_stage_bound`, `ggml-cuda.cu:10384`), counted in the `--fit` **margin**
(`llama_h2d_stage_accounting`), i.e. outside the slab.

## 3. Design

**Recommended: a reserved, non-evictable transient region at the top of the slab, pre-split into fixed
per-slot buffers.**

1. **Region at the top of the VA reservation.**  `ring_off = size - ring_region`; the ring occupies the
   topmost `ring_region` bytes of the slab.  Because the boundary grows from the bottom, the region is
   never in `[boundary, target)`.  The boundary target is **clamped** to `ring_off`; `slab_extend` maps
   `[mapped, ring_off)` only.  The arena is `[boundary, ring_off)`.
2. **Reserve its physical at creation** (it is part of `mapped`/`want`), and **shrink the reserve by
   `ring_region`** so the slab + region together still fit the same budget.  `--fit` no longer counts the
   ring in the margin (it is in the slab); the outside headroom is left for hipBLASLt/the draft.
3. **Pre-split into `GGML_SCHED_STAGE_SLOTS` fixed sub-buffers**, each `max_table + 512`, instead of the
   grow/free-per-slot dance.  The ring's VA becomes **stable**, so `ggml_cuda_graph_mem_freed(device)` is
   no longer needed for ring growth and captured graphs stop being invalidated by it.
4. **Sizing** = the same `h2d_stage_bound(max_table)`, or `GGML_SCHED_STAGE_MAX_MB` when set; `0` when
   `GGML_SCHED_STAGE=0`.  If `w + min_arena + ring_region > want`, decline the slab as today
   (`cache_unusable`), so the ring falls back to `cudaMalloc`.
5. **Fallbacks stay:** `GGML_CUDA_SLAB=0`, a declined slab (`cache_unusable`), or `GGML_SCHED_STAGE=0`
   keep the current `cudaMalloc` ring.  `arena_alloc_transient` gets a real definition (carve from the
   region; return `nullptr` when exhausted), and `h2d_stage_buffer` prefers it, else `cudaMalloc`.

**Alternative considered (not recommended):** a `pinned` flag in `arena_free` that `work_alloc` skips.
Rejected because the boundary must move over a *contiguous* range; a pinned run in the middle would block
growth entirely, and the top-of-arena placement only works until the work region climbs to it.

## 4. Phases

* **P1 — the ring.** Region + fixed slots + `--fit`/margin move + `arena_alloc_transient`.  This removes
  the ring from the G4 tension.
* **P2 — the FA staging arena.** Same mechanism (its size is known from the FA shape); then the G4 cap
  only guards allocations it genuinely cannot predict, or is retired.
* **P3 — retire/keep G4.** If nothing is left outside the slab, the free-VRAM cap becomes a diagnostic
  rather than a guard.

## 5. Invariants to preserve

* The boundary move target is clamped to `ring_off`; a `work_min` that would cross the region is a
  decline, never an overlap.
* The ring region is never inserted into `arena_free`; the arena top is `ring_off`.
* `slab_extend` never maps into the ring region.
* Ring-slot VAs are stable for the life of the slab (pre-split).
* One region per device; the existing `g_slab_mutex` covers it.
* The existing corruption guard stays: no arena chunk may be handed to the work pool while a view that
  was laid out to use it is live (`work_needs` floor); the region change must not weaken it.

## 6. Risks

* **Wasted arena.** The region is a worst-case reservation held for the whole run; a short prefill that
  never fills the ring still pays for it.  Measure the arena delta and consider sizing to the actual
  (bounded) table, not `slots * max + 512`, if the waste is material.
* **Double-counting.** `--fit` and the slab must move the ring bound in lockstep or the fit will either
  under-reserve (abort) or over-reserve (context shrink).  This is the same class as the G1/G2 work.
* **Slab-created-before-model-data.** The ring bound depends on the largest host-resident weight, which
  the preflight knows; the slab must be given it explicitly rather than discovering it lazily.
* **FA staging still outside.** P1 alone does not retire the G4 cap; the field prefill win only lands once
  P2 is in (or the cap is scoped to the FA staging only).  The §5.5 G4 decision is unchanged until then.
* **The two-region boundary contract is load-bearing.**  The history (`ggml-cuda.cu` comments) records a
  real corruption from letting the boundary shrink under a live view; the region work must not disturb
  that.

## 7. Gates (run on gfx1201 3-GPU and 2-GPU, plus `fingon` gfx1100)

* Standing: dense `Qwen3.5-4B` golden, `scripts/gate-prefill-logits.sh` (KLD <= 0.005, top-p >= 98 %),
  `test-backend-ops -o MUL_MAT_ID`, `none == n1 == n3 == n7` width purity, MTP acceptance.
* **Boundary-move stress with the ring active:** `wide1 -> short -> wide2` (the r22 pattern) at
  `-ub 8192` and the §5.5 field config; assert the ring region is never taken, `////` = 0, no
  `optional device allocation refused` for the ring, arena restored.
* **§5.5 field config:** prefill >= stock (~922 t/s), decode, post-rearm arena, 0 aborts.
* Region sizing: `GGML_SCHED_STAGE=0` -> region 0; `GGML_SCHED_STAGE_MAX_MB` honoured; `GGML_CUDA_SLAB=0`
  and a declined slab fall back to `cudaMalloc`.
* Arena-cost measurement: arena MiB with the region vs without, at an identical config.
* `fingon`: the §11.4 staging A/B and the primed-arena server numbers must not regress.

## 8. References

* `ggml/src/ggml-cuda/ggml-cuda.cu`: `struct ggml_cuda_slab` (`:592`), `ggml_cuda_slab_init_locked`
  (`:896`), `ggml_cuda_slab_work_alloc` (`:1030`, eviction at `:1068`), `ggml_cuda_slab_arena_alloc`
  (`:1106`), `ggml_cuda_slab_arena_free` (`:1132`), `slab_extend` (`:823`), `h2d_stage_bound` iface
  (`:10384`).
* `ggml/src/ggml-cuda/ggml-cuda-vmm.h:63`: `ggml_cuda_slab_arena_alloc_transient` (declared, undefined).
* `ggml/src/ggml-cuda/common.cuh`: `h2d_stage_buffer` (`cudaMalloc` at ~`:1985`), `fattn_stage_try_get`,
  `optional_alloc_within_free_cap`.
* `archive/work/fit-slab-accounting/README.md` §15.9 (the §5.5 finding) and §12.1/§12.3 (G3/G4).
* `archive/work/moe-cache-autosize/` (`ARENA-UB-TENSION.md` §13/§14) — the movable-boundary slab design.
* `ENVIRONMENT.md` §1.3 (`GGML_CUDA_SLAB*`), §2 (`GGML_SCHED_STAGE*`).

---

## 9. P1 implementation record (2026-10-09/10, gfx1201)

**Artifact: [`slab-ring-p1.patch`](slab-ring-p1.patch)** (sha256 `de78092acd1b…`, 694 lines) against a clean
r38 tree (`849c04161`, tree `88856410`, `scripts/apply-all.sh`).  Built with
`cmake --build build-rocm-hybrid --target llama-cli llama-server test-backend-ops llama-perplexity -j 16`.
Not applied to any delivery checkout; `patches/` untouched.

### 9.1 What was implemented (the §3 design, with the lifecycle tie-in)

* **The ring is a slab hole** `[ring_off, ring_off + ring_region)` at the top of the mapped slab
  (`ring_off = mapped - ring_region`).  `ggml_cuda_slab_work_alloc` clamps the boundary to `ring_off`;
  `slab_extend` maps above `mapped`, so it never touches the hole; the hole is excluded from
  `arena_free` while armed.
* **`ggml_cuda_slab_arena_alloc_transient()`** (the declaration that had no definition) is now the hole's
  bump allocator; `arena_alloc_transient` recomputes `h2d_stage_bound(max_table)` (`slot_cap =
  h2d_stage_region_bytes(max_table)/slots`) and `h2d_stage_buffer()` allocates one FULL slot capacity per
  slot on first use, so the ring's VA is stable for the life of the slab and the G4 free-VRAM cap is
  bypassed entirely (it is our own reservation).  `h2d_stage_free()` does not `cudaFree` slab slots.
* **Sizing**: `h2d_stage_region_bytes(max_table)` = `min(slots*(max_table+512),
  GGML_CUDA_SLAB_RING_MIB [default 6144], GGML_SCHED_STAGE_MAX_MB when set)`; `0` when `GGML_SCHED_STAGE=0`.
  `max_table` comes from `llama_model::max_host_weight_tensor_bytes()`, which for this model is the
  **merged** host tensor (27 466 MiB), 100x the largest staged slice (270 MiB) — hence the 6144 MiB cap.
* **Arm/disarm** (`ggml_backend_dev_slab_ring_set`): armed for a wide pass (`ubatch.n_tokens >= 64`) and
  disarmed for a narrow one, from `llama_context::process_ubatch` **before** the drop's `moe_cache_rearm`
  (so a post-prefill re-arm sizes against the full arena).  Arming removes the hole from `arena_free` and
  evicts the tables the arena put there (`moe_cache_evict_slab_range`, no slab lock held); disarming syncs
  the device and returns the hole.  `--fit` no longer counts the ring bound (`ggml_backend_cuda_h2d_stage_bound`
  returns 0 when the slab is enabled), so it is not double-counted; the fit n_ctx/arena are unchanged.
* Debug switches used below: `GGML_CUDA_SLAB_RING_MIB`, `GGML_CUDA_SLAB_RING_KEEP=1` (never disarm),
  `GGML_CUDA_SLAB_RING_NEVER=1` (never arm → the ring stays on `cudaMalloc` + G4).

### 9.2 Standing gates (all PASS, ring path live where relevant)

| gate | result |
|---|---|
| dense `Qwen3.5-4B-Q8_0` `-sm tensor` (slab never armed) | **`1c5d32ac537d`** (matches the golden) |
| `scripts/gate-prefill-logits.sh` (gfx1201, 27B Q8_0) | **PASS** mean KLD **0.000707**, same-top-p **98.755 %** |
| `test-backend-ops -o MUL_MAT_ID` | **931/931** |
| width purity 2-GPU Flash-Next IQ4_NL `-ncmoe 48`, `none == n1 == n3 == n7` | **`b00fdf534227`** (329 chars), 0 `////` (different prompt/len than §15.3, hence a different hash) |

### 9.3 §5.5 field config (2 GPU, `--fit on`, 31 482-token prose, 1000 MTP tokens)

The maintainer's suggested before/after.  `srr_*` logs under `/tmp/srr/` (transient); harness
`/tmp/fsa155/field.sh` re-pointed at `/tmp/srr`.

| arm | prefill t/s | decode t/s | acceptance | NaN |
|---|---:|---:|---:|---:|
| **stock r37 reference** (§15.9) | **922-924** | 61.5-61.8 | 1.0 | 0 |
| r38 WIP, G4 default 50 % (§15.9) | **767-769** | 61.8-62.0 | 1.0 | 0 |
| this build, ring never armed (`RING_NEVER`, = the r38 G4 cost) | **781.4 / 782.6** | **62.2** | 1.0 | **0** |
| this build, ring in slab, **hole kept armed** (6 GiB) | **929.3** | 43.8 | 1.0 | **0** |
| this build, ring in slab, **hole kept armed** (2.5 GiB) | **921.2** | 53.8 | 1.0 | **0** |
| this build, ring in slab, **hole disarmed after prefill** (6 GiB) | **973.4** | 28.9 | **0.002** | **8901** |
| this build, ring in slab, **hole disarmed after prefill** (2.5 GiB) | — | — | **0.0027** | **8874** |

**The G4 tension is removed and the prefill win is real** (973 vs 782, and above the stock 922).  But:

* **The decode reclaim corrupts.**  With the disarm, the MTP draft logits go **NaN** ~3 s into the decode
  (8901 NaN lines; acceptance 0.002), and decode collapses to 28.9 t/s.  It is deterministic (2.5 GiB and
  6 GiB holes both do it); it is **not** the arm's eviction (never-arm and keep-armed are clean) and it is
  **not** the ring's use (keep-armed uses the ring and is clean).  It needs the hole to be handed back to
  the arena during decode.  The draft's first few candidates are finite (acceptance 0.667), then NaN — a
  state corruption as the arena fills the hole, consistent with a stale expert-cache alias/remap (the
  G7 class), not a plain staging error.  A small `-c 8192 -ub 2048 -n 300` MTP run does **not** reproduce
  it (both arm modes show one benign `nan` log line, acceptance 0.45).
* **Keeping the hole armed costs decode.**  The hole is removed from the arena for the whole run; a
  2.5 GiB hole costs ~13.5 % decode (62.2 → 53.8), a 6 GiB hole ~30 %.  So the prefill/decode trade is
  back, just moved from the G4 cap to the arena size — the reclaim is what makes it a pure win.

**Conclusion for P1:** the slab-side ring (hole + `arena_alloc_transient` + `h2d_stage_buffer` routing +
`--fit` lockstep + arm/disarm plumbing) is correct and green on the standing gates, and it removes the G4
refusal (field prefill 782 → 921-973).  **The decode reclaim is blocked on the MTP NaN** and P1 is not
promotable until it is root-caused.  Next steps, in order:

1. **Root-cause the NaN.**  `MOE_EXPERT_CACHE_VALIDATE=1` on the field config; diff the `g_alias_to_id` /
   per-table `slot_dirty`/`remap` state across the disarm; check whether the re-arm places a table at a
   VA a live MTP-export consumer still reads.  The tiny repro is `srr_ring_in` (server, 31k prefill →
   1000 MTP decode) — `grep -ac nan` is the oracle.  The G7 condition (`node_n_consumers == 1`) is in
   `ggml-backend.cpp:2694` and is untouched by this change.
2. **Shrink the hole** once the reclaim is safe: it is sized for the worst-case slot (`slots*(270+512)`
   = 6 GiB here) while the ring's live total is ~1 GiB.  The `max_host_weight_tensor_bytes()` merged-tensor
   artifact is the reason the cap defaults to 6 GiB; a real per-table bound (or a `GGML_LOG` warning when
   the cap clamps) would let it be ~2.5 GiB.
3. If the hole cannot be shared safely, P1's fallback is the **keep-armed** configuration (prefill +18 %,
   decode −13.5 %); that is a smaller, honest win, and it still retires the G4 refusal for the field.

