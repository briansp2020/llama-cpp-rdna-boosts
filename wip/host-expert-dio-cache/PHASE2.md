# Phase 2 — the bounded, pinned host tier (L2): a page-cache-backed bounce buffer

Status: **implemented + validated (wip)**.  Inert behind `MOE_HOST_POOL_MIB` unset/0.  Part of the
`changes.patch` working-tree diff; not folded into the delivery.

## Design (the "invisible L2" route)

The GPU never reads the pool.  The pool is a **fill source** for the VRAM arena (`access_locked`'s
host→device copy), so the kernel interface is unchanged and the existing purity anchors hold.

Its purpose: `mmap()` of the GGUF cannot be read by the GPU on a no-XNACK part (issue #116), so the
weights must live in **pinned** host memory to be GPU-readable.  The pool is that pinned window — a
**bounce buffer** — sized to stream weights effectively out of the OS **page cache** (which is
reclaimable, so the non-swappable bound stays the pool).  DIO is only a debug fallback.

* **One pool per `(host tensor, device)`**, keyed by `t.host` + `t.device` (`g_host_pools`).  Per device
  keeps every slot's async copy on that device's stream, so the per-slot `cudaEvent_t` is valid
  (a shared pool + cross-device events fails with `invalid resource handle`).  The budget
  `MOE_HOST_POOL_MIB` is therefore **per device** (handover's "N MiB/device").
* Each slot holds a **whole host expert** (`host_bytes`), so a fill is one contiguous O_DIRECT read and
  every per-device slice is served from the same slot (the arena copy keeps the slice geometry:
  1-D for an unsplit/axis-1 table, `cudaMemcpy2DAsync` for an axis-0 slice).
* **LRU** eviction by per-expert last-use; slots are chosen so the decode working set does not evict
  itself.
* **Fill read**: buffered `pread` **through the page cache** (the default).  `MOE_HOST_POOL_DIO=1` forces
  `open(path, O_RDONLY|O_DIRECT)` + an aligned bounce buffer (`pool_dio_read`) — a **debug fallback** for
  someone who explicitly wants to avoid the page cache; it is much slower for a bounded pool (see below).
  Falls back to the full master if the read fails.
* **Prewarm** (`MOE_HOST_POOL_PREWARM`, default on): when a pool is built, every slot is DIO-filled,
  ranked by the arena's current hot set (`t.slot_expert`, the prefill seed / admission) then the host
  prefill tally then expert id.  This is the point of a DIO tier: spend the SSD bandwidth once, up
  front, not in per-token stalls.
* **In-flight safety**: a slot's arena fill is an async copy, so before a slot is reused its copy is
  drained via `slot_ev`.  A pass that cannot fill more experts than the pool holds cannot evict a slot
  it just filled, and the scheduler synchronises the backend between tokens, so that drain is skipped
  in the common case (no whole-stream sync per eviction).
* **Device policy**: a pooled table must be host-filled (the in-kernel policy fill reads the full
  master and cannot DIO), so `t.policy` is off while the pool is on and pooled tables use the host
  promotion path (`moe_cache_promote_host` → `access_locked`).  Making the policy kernel pool-aware is
  Phase 2b.

## Validation (gfx1201 ×3, ROCm 7.14)

* Output-preserving: 35B-A3B Q4_K_M `-sm tensor -ncmoe 40 -n 48` and `-sm layer -ncmoe 40 -n 48` are
  `359ff4337837` for **pool off == pool on**, no faults.  Dense 4B golden `1c5d32ac537d` both ways.
* `test-backend-ops -o MUL_MAT_ID` 931/931.  `scripts/gate-prefill-logits.sh` PASS (0.000707 / 98.755 %).
* Pool stats (`MOE_HOST_POOL_MIB=2048`, `-n 128`): 120 host tensors, ~28 slots each, 2009.7 MiB pinned
  (host-shared build), **5239 O_DIRECT fills**, h≈0.56, 0 faults.

## Reporter-model end-to-end (Qwen3.8-Flash-Next IQ4_NL, 2×R9700, `-ncmoe 48`)

The reporter's model, 9 shards (~93 GiB), 2 GPUs, `MOE_HOST_POOL_MIB=20000` (40 GiB total, buffered,
prewarm on), `-c 16384`, the 5246-token prose prompt + `-n 1000`:

| split | prompt t/s | generation t/s | peak RSS | fault |
|---|---:|---:|---:|---|
| `-sm layer` | 137.1 | 18.8 | ~130 GB (load) / ~104 GB (run) | 0 |
| `-sm tensor` | 182.2 | 16.3 | ~128 GB / ~105 GB | 0 |

Both produce coherent 2.9k-char completions.  `-sm tensor` prefill is faster, decode a little slower.
This is the config the campaign exists for: the pool is a 40 GiB pinned bounce buffer over a ~50 GiB
pinned master, readings weights through the page cache, on a no-XNACK part where `mmap()` cannot be
read by the GPU.  Phase 3 removes the master so the bound is the pool alone.

## Performance — the pool mechanism is free; a small pool thrashes

35B-A3B Q4_K_M `-sm tensor -ncmoe 40 -fa 1 -n 256` generation:

| pool | t/s |
|---|---:|
| off | 6.9 |
| `MOE_HOST_POOL_MIB=2048` (bounded, ~28 slots/tensor) | 2.4 |
| `MOE_HOST_POOL_MIB=2048` + `PREWARM=0` | 2.4 |
| `MOE_HOST_POOL_MIB=27000` (full residency, `slots == n_experts`) + prewarm | **6.7** |

The full-residency pool matches pool-off, so the pool lookup/copy path costs nothing: the 2.4 t/s is
the fill **miss rate** of a pool too small for the decode's expert set (the 256-token decode touches far
more than 28 experts/tensor, so it evicts continuously).  Prewarm alone does not fix that — it only
warms the first tokens.

## Finding: the DIO fill bypasses the page cache, which is what made a bounded pool slow

On this box the model filesystem is **XFS** and the page cache already holds the model
(`Cached: ~97 GB`).  `O_DIRECT` **bypasses** that cache, so every pool miss was a real disk read even
though the bytes were already in RAM — the observed thrash.  With the fill read buffered
(`MOE_HOST_POOL_DIO=0`) the misses hit the page cache:

| config (`-n 256`) | t/s |
|---|---:|
| pool off | 7.0 |
| pool `2048` MiB/device, buffered (default) | **6.1** |
| pool `2048` MiB/device, `POOL_DIO=1` (debug fallback) | 2.4 |
| pool `27000` MiB/device (full residency), prewarm | 6.7 |

So the page cache is the free, reclaimable L2.5 the campaign was missing: the GPU still reads only the
**pinned** pool (bounded, non-swappable), but the pool is refilled through the reclaimable page cache
instead of from disk.  **The default is buffered** (flipped 2026-10-08); `MOE_HOST_POOL_DIO=1` is the
debug fallback for a strict no-page-cache run.  Remaining 6.1-vs-7.0 is the miss path itself (pool
lookup + copy from the slot vs the master) and is Phase 2b tuning (better prewarm / device-policy
integration).
