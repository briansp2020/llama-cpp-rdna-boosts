# Implementation log — what exists, in which tree, and what the next slice is

**Nothing here is delivery work.** The campaign's code lives on the fork branch `wip-kv-residency`
(cut from the r174 tip `5bdec04ab`), is **not** in `patches/`, and is applied to no checkout other than the
one it is developed and measured in. Snapshot: `patches/0001-ggml-cuda-grow-only-VMM-region-primitive.patch` (see `patches/README.md`).

## Slice 1 (2026-10-29) — the grow-only VMM region + its selftest

Branch `wip-kv-residency`, commit `c757b5ea8` (413 insertions, 3 files, purely additive):

| file | what |
|---|---|
| `ggml/src/ggml-cuda/ggml-cuda-vmm-region.h` | the contract, and the driver facts + two traps it is built on |
| `ggml/src/ggml-cuda/ggml-cuda-vmm-region.cu` | one VA reservation, physical mapped per `unit` chunk, grown **monotonically**; `grow()` memsets what it maps; owner + best-effort peer access (the slab helper's pattern); fail-soft everywhere |
| `ggml/src/ggml-cuda/ggml-cuda.cu` | one env-gated, one-shot call from `ggml_backend_cuda_init` |

Why a selftest and not a feature: the primitive is internal to the backend library, and the consumer (a KV
buffer type) is the next slice — so `GGML_CUDA_VMM_REGION_SELFTEST=1` is the only way to exercise it in-tree.
Same precedent as `MOE_EXPERT_CACHE_VALIDATE`, `GGML_META_SS_VERIFY`, `LLAMA_RS_ROLLBACK_DEBUG`.

```
device 0: VMM region available (map unit 2 MiB, driver granularity 2 MiB)
device 0: PASS (reserved 384 MiB in 6 x 64 MiB units, grew to 3, watermark contract and clear-on-map verified)
```

The selftest checks, in order: a fresh reservation maps **nothing**; `grow(1)` maps one whole unit and that
unit reads **zeros** (stale-content clear); a pattern in unit 0 **survives** a grow to 3 units; the two new
units read zeros; a smaller `grow` is a no-op; growing past the reservation **refuses** (no abort); drain.

**Gates for the slice** (both green):

* same-seed greedy text is **byte-identical** with the arm off and on — 4B Q8_0, 32 tokens, 1109 bytes each.
  (The transient 384 MiB mapping therefore perturbs nothing.)
* a MoE arm with the slab armed prints an **identical layout** with the selftest on:
  `SLAB_INIT dev=0 free=21.21 GiB ... work 0.56 GiB + arena 19.62 GiB`, arena 11238.0 MiB / 100.0 %
  residency — the same numbers as r174 without the change (`INVENTORY.md` §1.B).

## What the probe settled, and how it shaped this code

`VMM-PROBE.md` is the evidence.  The four decisions that came out of it:

1. **Chunk-granular, not page-granular.**  A sub-range `hipMemUnmap` is refused, so the give-back unit is one
   whole mapping.  `unit` defaults to 64 MiB (driver granularity is 2 MiB; create+map is ~3-4 µs at any
   size), and `grow()` only ever maps whole units.  Nothing ever unmaps part of a chunk, so the refusal can
   never be hit.
2. **Monotone growth.**  Mixed with (1) this means the region never shrinks — which is exactly right for a KV
   whose used-cell watermark only rises, and it removes every "reclaim a hole" question from the design.
3. **Clear on map.**  A fresh mapping can hold the previous owner's bytes (measured, nondeterministically), so
   `grow()` memsets what it just mapped (~0.54 ms per 256 MiB, 490 GB/s — no fault storm).
4. **The mapped watermark is a hard read limit.**  Reading reserved-but-unmapped VA *faults the GPU* rather
   than returning an error.  The public contract is therefore `ggml_cuda_vmm_region_mapped()`, and it lines up
   with what `llama_kv_cache` already does: the graph view is sized by **used** cells (`n_kv`), not `kv_size`.

## Next slice — the KV buffer type (`ggml_backend_cuda_kv_region_buffer_type`)

The shape, so it does not have to be re-derived:

* **`alloc_buffer`** takes the KV's *reserved* size (from the requested `-c`) and returns a buffer backed by a
  region object: `ggml_cuda_vmm_region_create(device, size_k + size_v, unit)` and `grow(0)`.  VA only at this
  point — no physical — so `SLAB_INIT` sees the KB, not the GB, and the arena keeps the difference.
* **the grow call site** is `llama_kv_cache::init_batch()` (or `prepare()`): the used-cell watermark is known
  there (`seq_pos_max + n_tokens`), and it is the last point before the graph is built.  `grow(watermark ×
  bytes_per_cell)` per cache; the region never shrinks, so a decode step after a prompt does nothing.
* **the buffer vtable** needs, at minimum: `get_base` (the reservation base), `get_alloc_size`,
  `init_tensor` (offset within the region), `clear` (a memset over `mapped()`, not `size`), `memset/set/get`
  (bounds-checked against `mapped()` — the watermark rule), `reset` (no-op: a region is not reusable), and
  `usage = GGML_BACKEND_BUFFER_USAGE_ANY`.  `ggml_backend_buft_alloc_buffer` must refuse a tensor that lands
  above the watermark rather than silently handing out unmapped VA.
* **it cannot be the only provider.**  Dense models never arm the slab, and the slab stays MoE-only, so the
  region must not be gated on it — that is provider P1/P2, and P0 (a plain `cudaMalloc` pool + a pinned host
  master) is the fallback for CUDA/Windows and for any device whose driver refuses VMM
  (`ggml_cuda_info().devices[d].vmm` already latches that).
* **`--fit`** (`common/fit.cpp:287-360`) models weights + KV as one per-device pool; the residency term goes
  there so a config that over-reserves is *reported*, not silently repaired.
* **the arena must yield later, not die.**  With the KV reserving VA and mapping as it goes, a grow that
  cannot get physical must not reach `ggml_cuda_device_malloc`'s OOM chain (which frees the arena) — it has to
  check headroom and refuse, or take the slab's boundary.  This is the same rule `vram-resident-slots` §4.2
  records (`TODO` #112).

## Gates the slice and its successors owe

* Same-seed greedy text at 100 % residency vs a forced-under-mapped arm (the residency identity gate).
* `scripts/gate-prefill-logits.sh` — a residency change moves bytes, not values, so it must read
  KLD 0.000000 / 100.000 %.
* MTP Protocol A on the 27B (a verify width + a residency change interact through the arena band).
* `GGML_CUDA_SLAB_STATS=1` layout assertions: with the KV region shrinking the KV's *mapped* bytes at start,
  the arena must be *larger* than the r174 baseline, and the work/arena boundary must still be exact.
* A negative test: a config whose used context exceeds the reservation must refuse/limit, never fault.
