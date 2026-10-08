# Parked bug: `--host-experts pinned` + `-sm tensor` + a partial arena page-faults

Status: **ROOT-CAUSED + FIXED in the wip tree** (`~/llama.cpp`, uncommitted; the isolated fix is
[`parked-bug-fix.patch`](parked-bug-fix.patch), and [`changes.patch`](changes.patch) is the full
working-tree diff including the Phase 1 plumbing).  Not folded into the delivery.  The handover's
"host over-read" diagnosis was **wrong** — it is a slab memory-access bug.

## Symptom

`llama-cli` on `Qwen3.6-35B-A3B-UD-Q4_K_M`, `-ngl 99 -sm tensor -ncmoe 40 -fa 1`, with a partial expert
cache (`MOE_EXPERT_CACHE_MIB=2048`) aborts on a long prefill or the first decode passes:

```
Memory access fault by GPU node-3 (Agent handle: …) on address 0x… . Reason: Page not present or supervisor privilege.
```

The fault address is always `slab_work_base + <small offset>` (0x9000, 0xa000, 0xb000, 0xc000) on the
third R9700.  It does **not** occur with `MOE_EXPERT_CACHE_MIB=0` (cache off) or `GGML_CUDA_SLAB=0`.

## Controls (all reproduced)

| config | result |
|---|---|
| cache on (2048) + slab on | **fault** (deterministic) |
| cache off (0) + slab on | clean |
| cache on (2048) + slab off (`GGML_CUDA_SLAB=0`) | clean |
| cache on + `GGML_CUDA_SLAB_HEADROOM_MIB=0` (no `slab_extend`) | **fault** |
| cache on + `GGML_CUDA_DISABLE_GRAPHS=1` | **fault** |

A host-side probe (`cudaMemset` + sync) at the faulting offsets right after slab creation succeeds, so
the page is mapped at init and the fault is a **later, cross-device access**.

## Root cause

The movable-boundary slab is one VMM mapping per device.  `ggml_cuda_vmm_map_phys` called
`cuMemSetAccess` with a **single** access descriptor for the **owning** device.  The scheduler copies a
split input device-to-device (`hipMemcpyPeerAsync`) into the compute buffer, which is a view of that
same slab mapping; the peer device's copy engine has no access to the destination range, so it faults
`Page not present or supervisor privilege` at the destination offset.

The AMD runtime trace (`AMD_LOG_LEVEL=3 HIP_LAUNCH_BLOCKING=1`) ends with exactly that:

```
hipMemcpyPeerAsync ( 0x7fc8fa809a80, 2, 0x7fd56ac09a80, 0, 16384, … )   # dst = dev2 slab, src = dev0 slab
ShaderName : __amd_rocclr_copyBuffer
… Memory access fault by GPU node-3 … on address 0x7fc8fa80b000
```

Why the cache is required: the cache changes the compute-buffer layout/residency so that a split input
lands in the slab work region and a cross-device copy targets it.  The bug is in the slab's access
grants, not in the expert cache or in the host master (the pinned host masters `0x7f1f…` are never near
the fault address).

## Fix

`ggml_cuda_vmm_map_phys` now calls `cuMemSetAccess` once with a descriptor for the owning device **and
every other CUDA/HIP device** (READWRITE), falling back to the owning device only if the driver rejects
a peer descriptor.  The slab is HIP-scoped (`ggml_cuda_slab_enabled()` is false elsewhere), so no other
backend is affected.

## Secondary findings kept in the same patch (reviewed separately)

* `moe_cache_shrink_arena` and the `moe_cache_update_host` device-migration path freed a table arena
  with raw `cudaFree` without checking `arena_reserved`.  A slab-backed arena must go back through
  `ggml_cuda_slab_arena_free` (`free_arena_backing`); a `cudaFree` of a range inside the slab's single
  VMM mapping can unmap/corrupt the whole slab.  Both now use `free_arena_backing`.  (Neither path
  fired in the reproduced fault — the shrink path is also unreachable while the slab is active — but
  they are latent equivalents of the same hazard.)
* `mul_mat_vec_q_moe`'s row loop read all `c_rows_per_block` rows even when `nrows_x % rpb != 0`; the
  discarded rows still issued `vx_use` loads.  For a cold read that sources the pinned host master in
  place, the tail rows walk past the last expert into the next expert (or off the end of the last host
  tensor).  The item count is now clamped to `min(rpb, nrows_x - row0)`: identical output (the skipped
  results were discarded), and it is a general over-read hardening for the VRAM arena too.

## Validation (final binaries, gfx1201 ×3, ROCm 7.14)

* 35B-A3B Q4_K_M, `-sm tensor -ncmoe 40`, `MOE_EXPERT_CACHE_MIB=2048`: short (`-p … -n 32`) **and** the
  full `prompts/prose-rdna-boosts.txt` (`-n 64`) run to completion, no fault.
* dense `Qwen3.5-4B-Q8_0` `-sm tensor` same-seed = `1c5d32ac537d` (r24–r33 golden).
* `test-backend-ops -o MUL_MAT_ID`: 931/931, ROCm0 OK.
* `scripts/gate-prefill-logits.sh`: mean KLD **0.000707**, same-top-p **98.755 %**, PASS (base is the
  r33-recorded `Qwen3.8-27B-Q8_0`).
* `-sm layer` (the mode the parked bug was NOT reported in): 35B-A3B Q4_K_M `-sm layer -ncmoe 40
  -fa 1 -n 48` is `359ff4337837` for **cache-off == cache-on(2048) == cache-on+pool(2048)**, no fault.

## Repro command

```sh
HIP_VISIBLE_DEVICES=0,1,2 MOE_EXPERT_CACHE_MIB=2048 \
  ~/llama.cpp/build-rocm-hybrid/bin/llama-cli \
  -m /llm/models/Qwen3.6/35B-A3B/Q4_K_M/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  -ngl 99 -sm tensor -ncmoe 40 -fa 1 -t 8 \
  -p "The capital of France is" -n 32 \
  --seed 42 --temp 0 --no-display-prompt --single-turn --reasoning off
```

A `sudo /usr/local/sbin/reset-amd-gpus` reset is required after a fault wedges the third GPU.
