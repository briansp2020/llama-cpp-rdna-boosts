# moe-cache-inplace: prefill reads routed experts in place (arena / pinned host master) instead of staging whole tables

One format-patch on top of `v16-a55e952b8-r33` (strict `git am`; it also applies cleanly on r34).  Block 06 territory:
`moe-expert-cache.cu/.h`, `mmq.cuh`, `ggml-cuda.cu`, the scheduler's staging gate and a new backend hook.  Pinned host
master only.  Switches: `GGML_MOE_CACHE_INPLACE=0` keeps the cache-aware staging copy but reads nothing in place;
`GGML_MOE_CACHE_STAGE=0` restores r33's plain upload.  Greedy output is unchanged.

This is the above-16-token side mentioned in #115.

## What

With host experts (`-ncmoe`) and the expert cache, every prefill ubatch above the band copied each routed expert
table whole: host -> scheduler ring slot, then (`-sm tensor`) `stage_d2d` into the tensor.  The arena already holds
the hot experts on the device, and a pinned host master is GPU-readable.

1. **Cache-aware staging** (`moe_cache_stage_table`): when a whole-table staging copy is exactly a registered table's
   slice, it is filled by one kernel instead, resident experts from the arena and the rest from the pinned host
   alias.  Same bytes.  The copy stream is ordered after the compute stream only when an arena or slot map was
   written since the last staging (`g_write_seq`), so during a prefill the staging keeps overlapping compute.
2. **In-place reads**: by default the staging is *deferred* instead.  `stage_d2d` hands the record from the ring slot
   to the tensor and skips the copy.  The tensor's consumer then either reads the table in place (the routed-compact
   MMQ kernel takes a per-expert source: arena slot via `slot_dev`, else the pinned host alias) or fills the tensor
   first with the bytes the staging would have written (any other consumer, and the fused gate stream).  So every
   path other than the compact kernel sees the original bytes.  The geometry is read live at the consumer, so an
   eviction in between only turns reads cold.
3. **Tail guard**: MMQ reads one K tile past an expert's last row.  Read in place that is harmless, except for a
   table's *last* expert read from the host, where it runs off the end of the host table (found as one NaN in a
   bit-for-bit compare of the in-place and staged kernels).  A routed, non-resident last expert is copied into a
   small zero-padded scratch on the device and read from there.  The arena side is already covered by your head/tail
   guard.
4. **Below the staging width gate** (17-63 tokens): a split still stages when every host weight in it is served in
   place, since staging then copies nothing.  New iface hook `moe_cache_inplace` (CUDA: per device; meta: every
   device holding a slice; CPU/RPC: NULL).
5. **`offload_op`**: above the band, a routed op whose table is served in place goes to the GPU.  17-31-token routed
   batches were running on the CPU (the 32-token offload minimum).

Gated on `cudaHostGetDevicePointer` succeeding for the master (`host_pinned`), on the slot map being current (device
policy, or host policy with no pending upload), and on the geometry matching exactly; anything else falls back to the
r33 path.

## Interaction with `wip/host-expert-dio-cache`

The cold half of the in-place reads uses the full pinned master.  If the bounded host pool replaces that master for a
table, the in-place path would need to read cold experts from the pool, or simply decline for pooled tables (the
`host_pinned` gate is the natural place).  Nothing here depends on the pool; flagging it so the two don't collide.

## Results on r33 (2 x R9700 / gfx1201, ROCm 10.0, Flash-Next GSQ-RCO IQ3_XXS + shared Q8_0 MTP head)

Plain r33 vs r33 + this patch, pinned host experts, `--n-cpu-moe 48`, 256K, q8_0 KV, `-ub 2048`, MTP n-max 3,
default cache settings, one run each:

| | r33 | r33 + patch |
|---|---|---|
| `-sm tensor`, short prompts 31 / 44 / 60 tokens (median of 5) | 63 / 73 / 99 t/s | 136 / 209 / 278 t/s |
| `-sm tensor`, 2.9k-token prefill | 794 t/s | 1754 t/s |
| `-sm tensor`, 61k-token prefill | 1384 t/s | 2422 t/s |
| `-sm tensor`, 4 concurrent streams | 151 t/s | 157 t/s |
| `-sm tensor`, greedy decode | 110.4-110.9 t/s | 110.8-113.1 t/s |
| `-sm layer -ts 59,41`, 2.9k-token prefill | 745 t/s | 1743 t/s |
| `-sm layer -ts 59,41`, 61k-token prefill | 1214 t/s | 1931 t/s |

For reference, with every expert in VRAM plain r33 does 2443-2474 t/s at a 37k-token prompt under `-sm tensor` here.

Output:
- Greedy shas identical to r33: decode `45e68f1bdfeb` (also after the 61k prefill and after the needle), 45-token
  prompt `42a52754c67d`, 2.9k prompt `0a6791b84da2`; `-sm layer` warm-up `9f826c79de59`, 2.9k `3ce4e10a520c`.
- 255k-token needle (depth 0.5, `-sm tensor`) passes at 1651 t/s.
- `test-backend-ops -o MUL_MAT_ID` 932/932.
- 0 GPU faults, 0 out-of-memory, and no "deferred table was never consumed" errors in any run.

The 4-stream outputs are not comparable between builds: four identical concurrent requests already give four
different shas within one r33 run (batch composition).

Pre-existing, not from this patch, in case it is useful: under `-sm layer -ts 59,41 --n-cpu-moe 48` the greedy
decode right after the 61k prefill is broken on **both** builds (sha `43aaeff3595c`, MTP draft 0/1191, where the
warm-up gives `9f826c79de59`).  It is the low-free-VRAM state described in #116; keeping >= 512 MiB free in the
workspace pool's first attempt avoids it on our builds.

## Not measured

RDNA3.5 hardware; RDNA3 / NVIDIA (the compact routed kernel is RDNA3.5/RDNA4-only, so there the consumer fills the
tensor and nothing is read in place); 3+ GPUs; the `GGML_MOE_CACHE_INPLACE=0` and `GGML_MOE_CACHE_STAGE=0` switches
on r33; `-sm layer` at 255k.
