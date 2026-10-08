# Why `-sm tensor` is much slower than `-sm layer` (and the block-12 all-reduce bug)

Status: **root-caused + fixed (wip, 2026-10-08)**.  The fix is a **block-12 all-reduce-init change**
(`ggml/src/ggml-cuda/ggml-cuda.cu`); it needs the maintainer's sign-off before any delivery fold.  The
fix lives in `changes.patch` on the `wip/host-expert-pool` branch only.

Model: 35B-A3B Q4_K_M, 3×R9700 (gfx1201), ROCm 7.14, `-ngl 99 -fa 1 -t 8 -n 128`, greedy.

## Symptom

`-sm tensor` decode is far slower than `-sm layer` — and *slower than a single GPU*.  With the expert
cache **off** (so the pool is not involved):

| device(s) | `-sm tensor` gen t/s | `-sm layer` gen t/s |
|---|---:|---:|
| 1 (gfx1201) | 81.6 | 81.4 |
| 2 | 46.0 | 81.9 |
| 3 | **21.6** | 79.7 |

Single-GPU `-sm tensor` and `-sm layer` are identical, so the per-device compute is fine; `-sm layer`
scales flat (the model shards with no per-layer communication) while `-sm tensor` collapses.  The delta
is the **per-layer cross-device reduction** of the split MoE expert partials — no cache or pool is
involved.

## Root cause: the hybrid all-reduce eagerly initializes NCCL, which degrades the internal pipeline

Switching the all-reduce implementation moves the 3-GPU number by 3x:

| `GGML_CUDA_ALLREDUCE` | 2 GPU | 3 GPU |
|---|---:|---:|
| `hybrid` (default) | 45.9 | 22.0 |
| `nccl` | 46.6 | 21.7 |
| **`internal`** | **78.6** | **66.8** |
| `ce` (2 GPU) | 45.9 | – |
| `none` (butterfly) | 64.8 | 49.4 |

A temporary probe in `ggml_backend_cuda_comm_allreduce_tensor` showed the dispatch is **identical**
under `hybrid` and `internal`: every decode reduce is `small=1` (so it takes the internal branch) with
`small_fail=0`, `large=0`.  So the reduce *algorithm* is not the difference — **the mere presence of an
initialized NCCL communicator makes the internal pipeline ~3x slower** (its proxy threads/side-effects).
`ce` is equally affected because `init_hybrid` (which `ce` builds on) is what calls `ncclCommInitAll`.
Swapping the init order (internal before NCCL) does not help.

Full hybrid measurements table (see [`PHASE2B-FINDINGS.md`](PHASE2B-FINDINGS.md) for the cache/pool
context):

| `-sm tensor` config | before | after lazy fix |
|---|---:|---:|
| `-ncmoe 0` (all on device), cache off | 21.6 | **69.6** |
| `-ncmoe 40` (host experts), cache off | 12.7 | **22.0** |

## Fix

`init_hybrid` now brings up the **internal pipeline only** and defers NCCL to the first tensor too large
for it (a prefill), via `nccl_lazy`/`nccl_tried` on the comm context; `init_ce` clears the lazy flag.
A decode-only run never pays the eager `ncclCommInitAll` cost, and a prefill still gets NCCL for the
bandwidth-bound large tensors.  `GGML_CUDA_ALLREDUCE` semantics are unchanged from the user's view.

## Validation

- Output unchanged (the reduce path was already internal): short `359ff4337837`, long-prefill
  `12d4fcd10886` (pool on/off, both splits), dense 4B `1c5d32ac537d`.
- `MUL_MAT_ID` 931/931; `scripts/gate-prefill-logits.sh` mean KLD 0.000707 / same-top-p 98.755 % PASS.

## Secondary finding: the expert cache is a net loss for `-sm tensor`

Independent of the AR bug: with the fix in place, the expert cache **halves** `-sm tensor` decode —
`-ncmoe 40` is 22.0 t/s cache-off but 10.2 t/s cache-on (pool off) and 9.3 t/s with the pool.  Split
tables use the host-promotion path (the device policy is off for split tables), and the per-device pool
thrashes (see `PHASE2B-FINDINGS.md`).  For `-sm tensor` today, `MOE_EXPERT_CACHE_MIB=0` is the faster
configuration; making the cache/split-table path win is the campaign's remaining work.
