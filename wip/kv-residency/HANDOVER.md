# HANDOVER — `wip/kv-residency` (TODO #117), cold-session start

Written 2026-10-29 at the end of the session that seeded the campaign and landed its first slice.
Read this top to bottom before touching anything; it is deliberately self-contained (do not assume the
previous session's context).  Cross-references: [`README.md`](README.md) (the campaign and its options),
[`INVENTORY.md`](INVENTORY.md) (measurements), [`VMM-PROBE.md`](VMM-PROBE.md) (driver facts),
[`STRATA-KV-STREAMING.md`](STRATA-KV-STREAMING.md) (the reference design), [`IMPLEMENTATION.md`](IMPLEMENTATION.md)
(the code log).

---

## 1. State in one screen

| what | where |
|---|---|
| delivery repo | `~/llama-cpp-rdna-boosts`, branch `main` |
| release the campaign is cut against | **r174**, `v15-6050b467c-r174`, base `6050b467c`, tip `5bdec04ab`, 15 blocks |
| campaign code | **fork branch `wip-kv-residency`** in `~/llama.cpp`, cut from `5bdec04ab`, one commit `c757b5ea8` |
| the release chain | `rdna-boosts` in `~/llama.cpp` — **still at `5bdec04ab`, untouched.** The campaign code is NOT in it, is NOT in `patches/`, and must never be folded in without the promotion rule |
| patch snapshot (the backup) | [`patches/0001-ggml-cuda-grow-only-VMM-region-primitive.patch`](patches/0001-ggml-cuda-grow-only-VMM-region-primitive.patch) + `patches/README.md` |
| what exists | the ROCm VMM probe (`vmm-probe.cc`, 0 failing checks) and slice 1: `ggml/src/ggml-cuda/ggml-cuda-vmm-region.{h,cu}` + one env-gated hook in `ggml_backend_cuda_init` |
| what is next | **slice 2: the KV buffer type** (§7) — the first change that actually moves VRAM |
| scope (decided) | **the KV cache only** (K/V + the qwen4exp/DSA indexer cache + their bookkeeping). The rest of the device-resident non-weight budget is deferred (§5) |

`~/llama.cpp` is left **checked out on `wip-kv-residency`** with a matching build, so the tree is ready to
continue.  `git checkout rdna-boosts` + a rebuild returns it to the release state.  `scripts/fork-refresh.sh`
re-cuts `rdna-boosts` from `release.json.base` and is unaffected by the extra branch (it requires a clean
checkout, and this one is clean).

## 2. This box, right now (it changed)

* **Only ONE R9700 is attached** (`HIP_VISIBLE_DEVICES=0`, gfx1201, 32 624 MiB).  The other HIP device is the
  9950X3D's iGPU (gfx1036, 512 MiB) which the delivery is not built for — pin `HIP_VISIBLE_DEVICES` on every
  arm or `-sm tensor` aborts at load with `ROCm error: invalid kernel file`.
* Every measurement in this campaign is therefore **1-GPU**.  Anything needing 2-3 devices (the `-sm tensor`
  peer-copy path, cross-device KV splits, MTP Protocol A at the reference device counts, the slab's
  three-device peer-grant behaviour) needs the hardware back; say so rather than assuming.
* Build: `~/bin/build-llama-rocm` (ROCm 10.1.0, `build-rocm/`, gfx1201 only, ccache on).  Never pass
  `-DCMAKE_HIP_FLAGS` (see `AGENTS.md`).
* Models used here: `/llm/models/Qwen3.5/4B/Q8_0/Qwen3.5-4B-Q8_0.gguf` (fast smoke),
  `/llm/models/Qwen3.6/35B-A3B/Q4_K_M/…` (MoE + slab), `/llm/models/Qwen3.8/27B/Q4_K_XL/…` (dense + MTP),
  `/llm/models/Qwen3.8/Flash-Next/GSQ-IQ3_XXS/…` (qwen4exp; 75 GB, will **not** fit one 32 GiB card).
* Always `--single-turn` (+ `--no-display-prompt`) with `llama-cli`, always `timeout`, never two GPU jobs at
  once.

## 3. The three answers this campaign already produced (do not re-derive)

1. **Strict metadata is not the problem.**  The one big device-resident term was the packed `kq_mask`
   (960 MiB at `-c 245760 -ub 2048`); r160 made it lazy and r172 extended the derived form to multi-sequence
   batches, so what remains is ~1 MiB (`cell_pos`/`tok_lo`/`tok_hi`) plus the expert cache's MiB-scale tables.
   The byte mass is the **KV data itself**.
2. **Windowed KV streaming only pays where the per-token attended set is bounded.**  The link is 28.8 GB/s
   measured (the card is x16, its switch uplink is x8 — read the *root port*), VRAM reads at 565 GB/s, so
   1 GiB of misses per token = 37 ms.  Full attention attends the whole used context, hence:
   `-nkvo` (pageable all-host KV) on a 4B at depth 4096 = **100.67 → 38.71 t/s**.  The eligible families are
   **qwen4exp QSA** (`indexer.top_k 2048` × `compress_ratio 4` = 8192 cells = 3.1 % of 262144 — and its reader
   already walks a block list) and **DSA**, plus the *drafter* (approximation, target verifies).
3. **The cheap, generic win is placement, not streaming.**  The KV is a plain `cudaMalloc` made *before* the
   slab exists, so `SLAB_INIT` maps `free VRAM - reserve` and every token of context is taken out of the
   expert arena 1:1 (measured: +3840 MiB of `-c` = **−3.75 GiB** of arena; arena 19.62 → 15.88 GiB).  A fixed,
   lazily-mapped KV region removes that coupling with **no kernel change**.  That is slice 2, and it is what
   the implementation is aimed at.

## 4. Design invariants (settled — treat as constraints, not options)

* **Policy lives in the memory module, physical comes from a provider.**  The residency metadata (page table,
  slots, stamps, miss lists) belongs in `llama-kv-cache*`; the physical comes from **P0** (plain `cudaMalloc`
  pool + pinned host master — the CUDA/Windows/no-VMM fallback), **P1** (a private VMM region), or **P2** (a
  slab carve).  This is what makes the work generic over dense/MoE and slab/no-slab.
* **The slab stays MoE-only.**  `ggml_cuda_slab_enabled()` = `env && g_slab_armed`, armed solely by the
  host-expert preflight (`ggml-cuda.cu:7960`); r37 and issues #118/#120 are what happens when it parks a dense
  model's VRAM.  **Nothing in this campaign may require the slab.**
* **Chunk-granular, monotone growth; never a sub-range unmap.**  ROCm refuses it (`VMM-PROBE.md` T5).
* **Clear on map.**  A fresh mapping can contain stale bytes (T8).
* **The mapped watermark is a hard read limit.**  Reading reserved-but-unmapped VA *faults the GPU* (T8/T9
  story in `VMM-PROBE.md`): never let a kernel, a copy or a graph view touch above `mapped()`.
* **A failed grow must never reach the OOM chain.**  `ggml_cuda_device_malloc`'s last resort frees the expert
  arena (the r117 identity cliff is 46.7 → 65.0 t/s).  Check headroom and decline/limit instead — the same rule
  `wip/vram-resident-slots/README.md` §4.2 records (`TODO` #112).
* **Purity.**  Residency moves bytes, never values: the oracle is byte-identical greedy text **and**
  `gate-prefill-logits.sh` at KLD 0.000000 / 100.000 % between the resident and forced-under-mapped arms.

## 5. Deferred by the 2026-10-29 scope decision (do not wander into it)

The expert cache's tables, the F1 layer-input store, the q8_1 arena, the scheduler's graph inputs, the FA
staging arena, the recurrent/GDN state (T3).  They are listed in `INVENTORY.md` §3 only so the boundary is
explicit.  "Reduce the GPU-resident metadata" was read as **the KV cache**; other areas are a later
exploration.

## 6. What exists in code (slice 1) and how to exercise it

Branch `wip-kv-residency`, commit `c757b5ea8`, 413 insertions in 3 files, purely additive:

| file | what |
|---|---|
| `ggml/src/ggml-cuda/ggml-cuda-vmm-region.h` | the contract + the driver facts and traps it is built on |
| `ggml/src/ggml-cuda/ggml-cuda-vmm-region.cu` | `ggml_cuda_vmm_region_{create,grow,mapped,size,unit,base,free,available,selftest}`; one VA reservation, physical mapped per `unit` chunk (default 64 MiB, rounded to the driver's 2 MiB granularity), grown monotonically, memset on map, owner + best-effort peer ACCESS (the slab helper's pattern), fail-soft everywhere |
| `ggml/src/ggml-cuda/ggml-cuda.cu` | one one-shot env-gated call at the end of `ggml_backend_cuda_init` |

```bash
cd ~/llama.cpp && ~/bin/build-llama-rocm          # or: cmake --build build-rocm --target llama-cli -j 16
HIP_VISIBLE_DEVICES=0 GGML_CUDA_VMM_REGION_SELFTEST=1 \
  ./build-rocm/bin/llama-cli -m /llm/models/Qwen3.5/4B/Q8_0/Qwen3.5-4B-Q8_0.gguf \
  -ngl 99 -c 4096 -t 8 -p hi -n 4 --temp 0 --no-display-prompt --single-turn -v 2>&1 |
  grep -E "vmm_region"
# expect:
#   ... ggml_cuda_vmm_region_available: device 0: VMM region available (map unit 2 MiB, driver granularity 2 MiB)
#   ... ggml_cuda_vmm_region_selftest: device 0: PASS (reserved 384 MiB in 6 x 64 MiB units, grew to 3, ...)
```

Gates already run for slice 1 (both green, re-run them after any edit):
same-seed greedy text byte-identical with the arm **off and on** (4B Q8_0, 32 tokens, 1109 bytes each), and a
slab-armed 35B-A3B arm reports an **identical** `SLAB_INIT` (`free=21.21 GiB`, `work 0.56 + arena 19.62 GiB`,
arena 11238.0 MiB / 100.0 %).

## 7. NEXT SLICE — the KV buffer type (`ggml_backend_cuda_kv_region_buffer_type`)

This is the first change that moves VRAM.  Build it in this order; each step is separately testable.

**7.1 The buft.**  Model it on the delivery's own `ROCm_HostMap` buft (`ggml-cuda.cu:3483-3545`,
`ggml_backend_cuda_host_map_buffer_type_dev`) — that is the closest precedent for a device-carrying custom
buffer type in this tree.  Members (`ggml/src/ggml-backend-impl.h:70-91`):
`free_buffer`, `get_base`, `init_tensor`, `memset_tensor`, `set_tensor`, `get_tensor`, `set_tensor_2d`,
`get_tensor_2d`, `cpy_tensor`, `clear`, `reset`; plus the buft vtable's `get_name`, `alloc_buffer`,
`get_alignment`, `get_max_size`, `get_alloc_size` and `is_host`.

**7.2 `alloc_buffer` reserves VA only.**  The requested `size` is the full `-c` KV (K + V for every layer with
`hparams.has_kv`), i.e. `llama_kv_cache::size_k_bytes() + size_v_bytes()`.  Create the region with that size
and `grow(0)`: **no physical at this point**, so `SLAB_INIT` (which runs later, at the first compute-buffer
allocation) sees only the KB and the arena keeps the difference this slice exists to recover.  `init_tensor`
assigns the tensor's offset inside the region exactly as `ggml_backend_alloc_ctx_tensors_from_buft` expects;
`get_alloc_size` must be the *reserved* size so the helper's arithmetic matches.

**7.3 The grow call site.**  `llama_kv_cache::init_batch()` (`src/llama-kv-cache.cpp:1038`) — after the
ubatches are built and before the context is handed to the graph.  **Subtlety that will bite:** grow to the
**padded** `n_kv`, not to the used cells — `get_n_kv(sinfo, n_tokens)` (`:1615`) pads to
`max(n_pad, 256)` and (on a multi-device split) to `LLAMA_KV_N_PAD_PREFILL`; the graph's KV view is
`[0, n_kv)` including the padding, so the mapped watermark must cover `n_kv`, or a kernel reads unmapped VA
and faults the card.  Call it per cache (a hybrid model has the attention cache plus the recurrent/indexer
caches; only the K/V ones are region-backed).

**7.4 Bounds checks in the vtable.**  Every `*_tensor` entry asserts (fail-soft: clamp + one `GGML_LOG_WARN`)
that `tensor->offset + offset + size <= mapped()`.  This is the watermark rule enforced where it matters, and
it turns a would-be GPU hang into a loud, catchable error.

**7.5 It cannot be the only provider.**  Gate on `ggml_cuda_info().devices[d].vmm` and an env switch
(`LLAMA_KV_REGION=1`, default OFF in the WIP set — the promotion rule inverts it *and* runs the gates).  If VMM
is unavailable, or the region cannot be created, fall back to the plain device buft: same allocations, same
numbers as r174, one log line saying why.

**7.6 `--fit`.**  `common/fit.cpp:287-360` models weights + KV as one per-device pool.  The residency term
(the mapped high-water estimate, or simply the *reserved* size when the user pinned `-c`) goes there, so an
over-reserving config is **reported**, never silently repaired.

**7.7 Do not.**  Do not touch `patches/`, `release.json` or any block.  Do not make the region the default in
`wip/`.  Do not let the region's physical grow by calling `cudaMalloc` in a loop.  Do not "simplify" the
sub-range-unmap avoidance away (map in whole `unit` chunks, monotone).

## 8. Gates slice 2 owes (and how to run them)

```bash
# 1. identity: residency at 100 % vs a deliberately under-mapped arm, byte-identical greedy text
HIP_VISIBLE_DEVICES=0 LLAMA_KV_REGION=1 timeout 600 ./build-rocm/bin/llama-cli \
  -m /llm/models/Qwen3.5/4B/Q8_0/Qwen3.5-4B-Q8_0.gguf -ngl 99 -c 4096 -t 8 \
  -p "The capital of France is" -n 32 --seed 42 --temp 0 --no-display-prompt --single-turn
#   ... and the same with the region's unit forced tiny (a new debug knob) so misses are guaranteed

# 2. the layout claim (this is the point of the slice): with region mapping only what a 4-token run
#    reaches, SLAB_INIT must show MORE free VRAM and a BIGGER arena than the r174 numbers in
#    INVENTORY.md §1.B (free 21.21 GiB / arena 19.62 GiB at -c 65536 on 35B-A3B -ncmoe 24)
HIP_VISIBLE_DEVICES=0 GGML_CUDA_SLAB_STATS=1 LLAMA_KV_REGION=1 timeout 900 ./build-rocm/bin/llama-cli \
  -m /llm/models/Qwen3.6/35B-A3B/Q4_K_M/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf -ngl 99 -ncmoe 24 -c 262144 \
  -t 8 -p hi -n 4 --temp 0 --no-display-prompt --single-turn -v 2>&1 |
  grep -E "SLAB_INIT|slab .*work|arena .*of .*host experts|KV buffer"

# 3. prefill purity (a residency path is new, so it owes the strict arm)
./scripts/gate-prefill-logits.sh --ab "LLAMA_KV_REGION=1"

# 4. same-seed coherence on all three families that fit one card (4B dense, 35B-A3B MoE, 27B+MTP)
# 5. a negative test: -c larger than the region's reservation must refuse/limit loudly, never fault
```

Acceptance bar: (1) identical text, (3) KLD 0.000000 / 100.000 %, (2) a strictly better arena/free-VRAM
layout, (5) a clear refusal.  Anything less is a failed slice.

## 9. Pitfalls that cost time last session (do not repeat)

* **Standalone HIP probes need `-isystem /opt/rocm-10.1.0-gfx120X/include`**, or HIP headers resolve from
  Fedora's ROCm 7.1 in `/usr/include` and the probe silently builds against a different driver API.  `cuda.h`
  does not exist in the ROCm 10.1 tree — the tree's `vendors/hip.h` maps `cu*` → `hip*` for HIP builds, and it
  maps only a *subset* of the CUDA macros (`CUresult` and `CU_MEM_ALLOC_GRANULARITY_MINIMUM` are **missing**;
  use `ggml_cuda_info().devices[d].vmm_granularity` instead of querying).
* `hipMemcpy2D` validates the **whole source extent** (`spitch × height`); an under-sized source returns
  `hipErrorInvalidValue` and can look like a VMM hazard.  Always keep a no-mapping baseline arm in a probe.
* A grid-stride kernel over **reserved** VA hangs the GPU.  `setvbuf(stdout, …, _IOLBF, 0)` in a probe, so
  progress is visible before a driver call hangs.
* `llama-cli` without `-v` prints nothing but the completion (the new logger); the accounting lines need `-v`.
* `llama-bench` in this tree has **no `-c`** and sizes its own context.
* Single `ls` of a 5900-iteration loop with a broken increment cost 30 minutes — the probe's mapping-count
  sweep is a fixed list now.

## 10. Open questions still waiting on the maintainer

1. **Priority after slice 2:** P2 indexer diet (q8_0 indexer keys, Flash-Next 3 GiB → 1.5 GiB at 256 K) or
   go straight at O2/P3 (qwen4exp QSA windowed residency, the XL piece)?
2. **Drafter ring** (O3): an acceptance trade as a shipped default (Strata ships `--mtp-window` default-on at
   32768 and reports the cost), or opt-in only?
3. **Host master budget** for O2: `MOE_HOST_POOL_RAM_PCT`-style percentage of `MemAvailable`, or an explicit
   MiB flag?
4. **qwen4exp only, or DSA too?**  No GLM-5 GGUF on this fleet (`TODO` #108), so DSA could only be reasoned.
5. **O1 on dense:** worth shipping?  It changes nothing at a full context and only helps a config that
   over-reserves — which is *most* server configs.

## 11. Records produced by the seeding session

| record | content |
|---|---|
| [`README.md`](README.md) | the campaign: the ask split into 3 axes, measured resident table, eligibility rule, options O1-O6, genericity (P0/P1/P2), phases, blockers, non-goals, open questions |
| [`INVENTORY.md`](INVENTORY.md) | T1-T5 taxonomy with in-scope tags, 4 measured shapes + verbatim log lines, per-token arithmetic (incl. qwen4exp indexer), the metadata table, reproduction commands |
| [`VMM-PROBE.md`](VMM-PROBE.md) | the 9 driver tests, results, and the two traps (unmapped-read fault, stale fresh mapping) |
| [`bw-probe.cc`](bw-probe.cc) | the link/kernel-read probe: H2D/D2H pinned 28.8/28.2 GB/s, kernel read of pinned 29.0, VRAM 565, D2D 291 |
| [`vmm-probe.cc`](vmm-probe.cc) | the VMM probe (reserve/map/unmap/count/latency/staleness) |
| [`STRATA-KV-STREAMING.md`](STRATA-KV-STREAMING.md) | Strata's `kv_stream.hpp` read in full, the 1:1 mapping onto this tree, what transfers and what does not |
| [`IMPLEMENTATION.md`](IMPLEMENTATION.md) | slice 1 + the written-down shape of slice 2 |
| `patches/` | the fork commit as a patch, with apply instructions |

## 12. First 30 minutes of the next session

1. `cd ~/llama-cpp-rdna-boosts && git pull` — the campaign docs may have moved.
2. `cd ~/llama.cpp && git branch --show-current` — expect `wip-kv-residency`; if it is `rdna-boosts`, the
   branch exists but is not checked out (`git checkout wip-kv-residency`, then rebuild).  Verify
   `git log --oneline -2` shows `c757b5ea8` on top of `5bdec04ab`.
3. If the campaign patch is not applied in the tree (fresh clone, or the branch is gone):
   `git am patches/0001-*.patch` from this directory — see [`patches/README.md`](patches/README.md).
4. `~/bin/build-llama-rocm` and run the §6 selftest; it must print PASS before anything else.
5. Re-read §3 and §4, then start §7.1 (the buft) — nothing above it is blocking.
