# RESULTS — slab-resident narrow-2 (end-pinned), 2026-10-12 session

**Status: IMPLEMENTED, FUNCTIONALLY VALIDATED. The numeric acceptance gate in the handover §4.2 is
CONFOUNDED by the open #50 (arena-dependent cache-aware fusion selection); see §3.**  Still WIP —
nothing has been folded into `patches/`, and nothing was applied to the fork's committed history.

## 1. What was implemented (all in `~/llama.cpp`, working tree)

`ggml/src/ggml-cuda/ggml-cuda.cu`:
* `ggml_cuda_slab` gained `narrow2_floor`, `narrow2_off`, `narrow2_live`.
* `ggml_cuda_slab_arena_limit()` = `size` before narrow-2, `narrow2_off` after; `arena_top()` =
  `min(mapped, limit)`.  `arena_total`, `slab_extend`'s VA room, and the work-pool growth guard all use
  them, so neither the arena nor a drifting boundary can reach into the pinned region.
* `ggml_cuda_slab_narrow2_floor_locked()` anchors narrow-2 at `[size - floor, size)`, maps it with
  `ggml_cuda_vmm_map_phys`, and is idempotent.  It refuses (→ the draft falls back to cudaMalloc) if the
  region would leave less than `boundary + min_arena + chunk`, if it would overlap already-mapped
  physical (`off < mapped`), or if a later request exceeds the pinned floor.
* `work_alloc` routes the MTP draft's COMPUTE buffers to narrow-2 (no boundary/`work_needs`/arena
  participation); `work_release` early-returns for a narrow-2 base.  The *target's* work path is
  unchanged and is guarded from growing past `narrow2_off`.
* New iface hooks `slab_narrow2_floor(dev,bytes)` and `slab_compute_narrow2(dev,enable)` (renamed from
  the session quick fix's `slab_compute_enable`), forwarded by the meta device.
* The quick fix's `cudaMalloc` bypass is **kept as the A/B arm**: `GGML_CUDA_SLAB_NARROW2=0` routes the
  draft out of the slab exactly as the quick fix did.  Default is the narrow-2 layout (`=1`).
* Kept env-gated diagnostics: `GGML_CUDA_WORK_ALLOC_DEBUG` (SLAB_INIT / WORK_ALLOC / NARROW2_ALLOC /
  NARROW2_RELEASE / COMPUTE_ALLOC / GRAPH_RESERVE) and `GGML_CUDA_CACHE_GUARD_DEBUG`
  (`moe_cache_guard_stats`).  Dropped the proven-negative ones: `GGML_CUDA_SKIP_ALLOC_DEPS`,
  `GGML_CUDA_PREFILL_FUSION_OFF`, `GGML_FORCE_N_RS_SEQ`, `NEXTN_EXPS`.
* `llama-context.cpp` sets the narrow-2 role around the draft's `graph_reserve`'s `sched_reserve` and
  `process_ubatch`'s `alloc_graph`, and re-affirms the floor in `sched_reserve` from
  `backend_buf_exp_size`.

Deliberate deviation from handover §3.4: the region is established **lazily**, on the draft's first
narrow-2 allocation, rather than from an eager `slab_narrow2_floor` call, because the draft's narrow size
is not known before its first allocation.  The first request is the draft's construction reserve (the
widest layout), so the floor never needs to grow; the public hook is still there and re-affirms it.

## 2. Validation — the alias is gone

GSQ-IQ3_XXS + shared Q8_0 draft, 2 GPU, `--fit on -c 204800`, alloc trace (`n2full`, `n2dbg2`):

```
dev 0: target narrow-1  need=1792  ret_off=0            (narrow_floor=2048)
       target wide      need=2816  ret_off=8192
       draft narrow-2   need=768   off=23296  (established)   -> release
       draft narrow-2   need=512   off=23296  (live through the run)
dev 1: target 0 / 8192; draft narrow-2 off=21440
```

* narrow-1 and narrow-2 are at **distinct, stable** offsets; the target never gets a narrow-2 base.
* The draft's 768 is released **before** the 512 is allocated (`NARROW2_RELEASE live=0` then
  `NARROW2_ALLOC live=1`), so narrow-2 does not alias the draft's own two views either.
* `narrow2_off > mapped` for the whole run; `slab_extend` maps only `[mapped, narrow2_off)` and the
  arena tops out below it.

## 3. The acceptance numbers, and why the gate is confounded

Auto arena, identical config (stable across repeats):

| arm | acceptance | decode t/s | arena |
|---|---:|---:|---:|
| pre-fix (aliased, historical) | 0.69318 | 43.1 | — |
| quick-fix A/B (`GGML_CUDA_SLAB_NARROW2=0`) | 0.76316 | 55.4 | 35811 MiB (87.5%) |
| quick-fix, historical `gy_n3` | 0.73835 | 53.53 | 37329 MiB (91.2%) |
| **narrow-2 (auto)** | **0.66284** | 48.6 | 36810 MiB (89.9%) |

The quick fix **itself** is 0.738 in one run and 0.763 in another at the same config, so the acceptance
metric is not stable across runs.  Decisive controls (exact generated token ids, `return_tokens`):

| control | first divergence narrow-2 vs quick-fix |
|---|---:|
| auto arena | idx 9 |
| `GGML_CUDA_DISABLE_FUSION=1` (both arms) | idx 65 |
| `MOE_EXPERT_CACHE_MIB=14000` (both arms, identical resident set) | idx 68 |

With the fusions off, and again with the cache budget pinned, the two arms agree all the way to the
**known plain-vs-MTP verify-width divergence index** (handover §5).  So:

* narrow-2 does not corrupt the MTP or target compute; the divergence at idx 9 is the **cache-aware
  fusion selection flipping with the slab/arena layout**, i.e. exactly `TODO.md` #50 (a global
  `ggml_cuda_cache_blocks_fusion` gate over per-table eviction).
* Pinning the cache budget flips the acceptance (narrow-2 0.75214 vs quick-fix 0.68548), confirming the
  acceptance ordering is a layout artifact, not a property of the buffer placement.

**Consequence:** the handover §4.2 gate (“acceptance >= the quick fix, 0.738”) cannot be met or refuted
by any buffer-placement change while #50 is open.  The structural metric is clean: the derived verify
step is ~61.5 ms (narrow-2) vs ~60.1 ms (quick-fix historical) and ~71.4 ms (pre-fix) — within the
run-to-run arena spread, and far above pre-fix.  No-draft runs never establish narrow-2, so they are
byte-identical by construction.

## 4. Open questions for the maintainer

1. **Accept the durable form now, and treat the acceptance gate as #50-blocked?**  Suggested: yes — the
   alias defect is fixed, the fused paths are untouched, and the acceptance spread is demonstrably the
   #50 non-determinism.  The alternative (matching the quick fix's arena) is not reachable: the quick
   fix's own arena (and acceptance) varies run to run.
2. **Priority: #50 (per-table fusion guard) or item (c) (verify-width divergence)?**  #50 is what makes
   the acceptance metric unstable and it is the campaign's original item.  Item (c) is the follow-on
   after the layout; its first step (handover §5) is the IQ4_NL purity classification
   (2-GPU Flash-Next `-ncmoe 48`, hash `b00fdf534227`).
3. **`narrow2_floor` sizing:** currently pinned at the first request (768 MiB, the draft's construction
   reserve).  The live graph only needs 512 MiB, so 256 MiB/device is pinned but never used.  Freeing it
   would require an unmap/remap of the top region when idle, which the maintainer has said is not needed —
   left as is.

## 5. Reproducers

```bash
cd wip/moe-verify-fusions/tools
export M=/llm/models/Qwen3.8/Flash-Next/GSQ-IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf
export D=/llm/models/Qwen3.8/Flash-Next/IQ4_NL/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf
export PROMPT=/tmp/srr/mixed30k.txt
GPUS=0,1 ./field3.sh ~/llama.cpp/build-rocm-hybrid n2 run30kfit        # narrow-2
GGML_CUDA_SLAB_NARROW2=0 GPUS=0,1 ./field3.sh ~/llama.cpp/build-rocm-hybrid qf run30kfit
# alloc trace: GGML_CUDA_WORK_ALLOC_DEBUG=1 ... (mode `fit` is enough to see the establishment)
# controlled token comparison: /tmp/srr/tokcmp.sh (return_tokens, n_predict=128)
```
