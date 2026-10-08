# Expert-cache admission policy on `-sm tensor` (split MoE) tables

Status: **OPEN / scoping** (opened 2026-10-08 from the `host-expert-dio-cache` Phase 2b session).
Nothing here is part of the delivery.  Do not fold without the maintainer's go-ahead.

## The problem

For a MoE decode, `-sm tensor` is slower than `-sm layer` (the per-layer cross-device reduction; see
[`../host-expert-dio-cache/SM-TENSOR-AR.md`](../host-expert-dio-cache/SM-TENSOR-AR.md) for the separate
all-reduce bug).  Turning the expert cache **on** makes `-sm tensor` *worse*:

35B-A3B Q4_K_M, 3×R9700 (gfx1201), ROCm 7.14, `-sm tensor -ncmoe 40 -fa 1 -t 8 -n 128`, greedy, hybrid
all-reduce with the lazy-NCCL fix:

| config | gen t/s | notes |
|---|---:|---|
| cache **off** (`MOE_EXPERT_CACHE_MIB=0`) | 22.3 | |
| cache on, **host promotion** (today's default for split tables) | 10.2 | 2.2x slower than off |
| cache on, **`MOE_EXPERT_CACHE_DEVPOLICY_SPLIT=1`** | **36.5** | 1.6x faster than off |
| cache on, `MOE_EXPERT_CACHE_DEVMAP=0` (eager host path) | 25.9 | |

So the cache *can* be a clear win on `-sm tensor`; the default merely picks the wrong admission engine
for split tables.  `DEVPOLICY_SPLIT=1` is output-preserving (coherence `359ff4337837` == cache-off).

## Root cause

The device-side admission policy is hard-disabled for split tables in `moe-expert-cache.cu`
(`alloc_table_locked`, ~line 1831):

```c
if (g_devpolicy && !g_pool_enabled && t.devmap && (t.split_axis < 0 || devpolicy_split)) { ... }
```

With `devpolicy_split` off (the default), every split table falls back to `moe_cache_promote_host`: a
per-table used-list D2H + host LFRU loop + slot-map H2D, ~120+ host calls per token.  That is the same
path that costs `-sm layer` 2x when the device policy is disabled there (42.0 -> 21.1 t/s).  The device
policy does the admission in one batched GPU kernel per device, with no host readback.

## Why it was disabled (and why that may be stale)

The code comment records the opposite result on a different workload:

> "A SPLIT table … must NOT use the device policy.  Its policy kernel and its prefill seed are tuned for a
> whole, per-device expert and measurably hurt the split case: 2 GPU IQ4_NL `-ncmoe 48`, MTP n3, n=1024
> went 60.3 -> 70.7 t/s once the split tables fell back to the host promotion."

That measurement predates the Phase 2b **target-3 seed fix** (the device prefill tally now drives both the
arena seed and the pool ranking; before, the tally was dead under any non-devpolicy path).  Re-validation
2026-10-08:

* 35B-A3B (3 GPU, Q4_K_M, plain decode): `DEVPOLICY_SPLIT=1` **36.5** vs host promotion **10.2** t/s.
* Reporter (Qwen3.8-Flash-Next IQ4_NL, 2 GPU, `-ncmoe 48`, plain decode, no pool, `-n 64`):
  host promotion **16.3** vs device policy **16.4** t/s — **no regression** reproduced.
* So the 60.3 -> 70.7 regression does not reproduce in plain decode; it may have been **MTP-specific**
  (`n3`), or the target-3 seed fix removed it.  It needs re-measuring under MTP n3 (the reporter's draft
  model is `mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf`).

## Plan

1. **Re-measure the MTP n3 reporter case** (`-sm tensor`, `-ncmoe 48`, MTP, `-n 1024`) with and without
   `DEVPOLICY_SPLIT=1`, both 2 and 3 GPUs, on the current tree (lazy-NCCL AR + target-3 seed).
2. If the regression is gone, **flip the default** so split tables get the device policy (kill-switch
   `MOE_EXPERT_CACHE_DEVPOLICY_SPLIT=0`), and re-run the gate matrix (`-sm tensor` + `-sm layer`, dense 4B,
   `MUL_MAT_ID`, prefill-logit KLD).
3. If it persists, find the mechanism.  Candidate suspects: the prefill seed's provisional slots under
   split (target-3 fixed the path but the split geometry may still misbehave), the policy kernel's
   in-kernel fill from `host_dev` with a strided axis-0 slice, or the 2-GPU band/shape.
4. Integrate with the host pool: the pool currently forces `t.policy=false` too (Phase 2b target 1).  Once
   both are on the device-policy path, the pool-aware policy (device `expert -> pool_slot` map) is the
   end state.

## Open questions

* Is the MTP n3 regression real on the current tree?  If not, is a measured gate needed at all?
* Does the answer depend on GPU count (2 vs 3) or quant (IQ4_NL vs Q4_K_M)?
* Should the default be split-aware automatically (e.g. enabled unless MTP is active), or always on with a
  kill-switch?
