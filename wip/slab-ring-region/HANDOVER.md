# `slab-ring-region` — P1 handover: finish the ring-in-slab work

Read `README.md` first (the design record, §1-§10), then this.  This file is the *do-this-next* list.
Everything here is WIP: `patches/` is untouched, nothing is promoted, and the ring must not be applied to
any delivery checkout until the gates below pass and the maintainer gives the go-ahead (see AGENTS.md).

## 0. One-paragraph status

The op-offload H2D staging ring is now a **work-region sub-region of the movable-boundary slab** in the
`0 -> narrow -> RB -> wide -> arena` ordering: a fixed always-resident narrow (decode/verify) floor at
base 0, the ring above it, the transient wide (prefill) view above that, and the arena always above the
work.  This removes the G4 free-VRAM tension (the ring no longer competes for the outside headroom) and is
structurally NaN-free (the ring is never arena space, so arming it is an ordinary boundary move that
evicts the **coldest** tables, not an independent top-of-arena eviction of the hottest).  On gfx1201 the
§5.5 field config went from **781 t/s prefill (r38 G4 cap) / 62.2 decode** to **946 t/s prefill (2.5 GiB
region) or 992 (6 GiB region) / 61.1-61.4 decode**, with **0 NaN, acceptance 1.0**, and the standing gates
are green.  The prefill win is real and above the un-capped stock 922; decode is within ~1.5 % of the
baseline.  What remains is cross-arch validation (`fingon` gfx1100, 3-GPU), a reserve-side accounting
decision, the P2/P3 follow-ons, and promotion.

## 1. Artifact and a working reproduction

* **Artifact:** [`slab-ring-p1.patch`](slab-ring-p1.patch) — sha256
  `e0d5fda7d0306866c9e4308534c6253855e416d4d7f710dff454bc9f09c51048`, **927 lines**, against a clean r38
  tree (`scripts/apply-all.sh` from `release.json.base a55e952b8` yields tip `849c04161`, tree `88856410`).
* **Prototype tree:** `~/llama.cpp`, branch `slab-ring-prototype` at `849c04161` **with the patch applied
  (uncommitted `git diff`)**.  If it is lost: stash/`git checkout -- .`, re-apply the patch, rebuild.
* **Build:**
  ```bash
  cd ~/llama.cpp
  cmake --build build-rocm-hybrid --target llama-cli llama-server test-backend-ops llama-perplexity -j 16
  ```
  (`EXTRA_CMAKE_FLAGS="-DCMAKE_HIP_FLAGS="` matters only for a from-scratch configure; the incremental
  build does not.)
* **Field harness:** `/tmp/fsa155/field.sh` is the r38 §5.5 arm; a copy lives at `/tmp/srr/field.sh`
  (`LOGD=/tmp/srr`, `PORT=8942`) with the prompt `/tmp/srr/prose30k.txt`.  Run the prefill/decode arm with:
  ```bash
  cd /tmp/srr
  ./field.sh /home/stew675/llama.cpp/build-rocm-hybrid <tag> run30kfit
  ```
  and read `prompt_tps` / `predicted_tps` from `/tmp/srr/<tag>.timing`, `grep -ac nan` and the acceptance
  line from `/tmp/srr/<tag>.err`.  `/tmp` is transient — **copy any result you want to keep into this
  campaign before the box is rebooted.**
* **Knobs used in this session:** `GGML_CUDA_SLAB_RING_MIB` (region cap; default 6144; 2560 also tested,
  and it is what makes `slot_capacity` big enough for the field's 270 MiB table).  `GGML_CUDA_SLAB_RING_NEVER`
  / `_KEEP` are gone (the arm/disarm is implicit now); `GGML_CUDA_SLAB_RING_NOTRANSIENT` remains in
  `common.cuh` as an A/B debug switch — remove it on promotion.

## 2. What is implemented (the shape to keep)

`ggml/src/ggml-cuda/ggml-cuda.cu`, `struct ggml_cuda_slab`:

```
low  [ 0 ............ narrow )              work, narrow/decode view,      base = s.base
     [ narrow ....... narrow + RB )         work, the H2D staging ring
     [ narrow+RB .... boundary )            work, wide/prefill view,       base = s.base + narrow_floor + ring_region
     [ boundary ..... mapped )              arena (always above the work)
high
```

* `ggml_cuda_slab_work_alloc(device, need)` returns **two stable bases** (`s.base` for a narrow-sized
  request, `s.base + narrow_floor + ring_region` for a wide one) and inserts each view's **boundary
  contribution** (`base_off + up(need, chunk)`) into `work_needs`.  `ggml_cuda_slab_work_release(device,
  off, need)` removes it.  The multiset floor makes a view placed before the narrow floor existed harmless.
* `narrow_floor` is established either by the free `ggml_cuda_slab_set_narrow_floor(device, bytes)` hint or
  from the first reserve smaller than the boundary (fallback).  The hint is only honored while
  `ring_used == 0` (the ring's VA must not move).
* `ggml_cuda_slab_set_narrow_floor` applies the compute-allocation formula
  `(ceil(bytes / C) + 1) * C`, `C = GGML_COMPUTE_BUFFER_CHUNK_MIB` (256), because the context's probe
  reports the **layout** size, not the allocated size.
* `ggml_cuda_slab_arena_total == mapped - boundary` (do **not** change this — see §4.4).
* `ggml_cuda_slab_ring_slot_capacity` / `ggml_cuda_slab_arena_alloc_transient` serve the ring from
  `[narrow_floor, narrow_floor + ring_region)` iff `ring_armed` (`boundary >= narrow_floor + ring_region`).
  `ggml_cuda_slab_ring_set` is a **no-op** now (kept for the generic arm/disarm plumbing).
* `common.cuh` `h2d_stage_buffer` uses the slab slot when `ring_slot_capacity > 0`, else `cudaMalloc`
  (+ the G4 cap).  No `ggml_cuda_graph_mem_freed` on growth (stable VA).

Context plumbing (`src/llama-context.cpp`, in the prefill→decode drop, before the real narrow reserve):

```c
size_t narrow_sizes[16] = {0};
if (graph_reserve(n_rs_batch, n_rs_seqs, n_rs_batch, mctx, /*split_only=*/true,
                  narrow_sizes, kq_mask_packed_reachable()) != nullptr) {
    ... narrow_bytes = max(narrow_sizes) ...
    ggml_backend_dev_slab_narrow_floor(dev, narrow_bytes);   // for the host-expert devices
}
const bool narrow_ok = graph_reserve(..., /*split_only=*/false, ...);
```

New generic optional device hook: `ggml_backend_dev_slab_narrow_floor(dev, bytes)` (impl in
`ggml/src/ggml-cuda/ggml-cuda.cu`, declared in `ggml/include/ggml-backend.h`, wired through
`ggml-backend-impl.h` / `ggml-backend.cpp` / cpu / rpc / meta).

Preflight: `llama_model_moe_cache_preflight` passes `max_host_weight_tensor_bytes()` to
`ggml_backend_dev_moe_cache_preflight`, and the CUDA impl sets the ring region via
`ggml_cuda_slab_set_ring_region(...)`.  NOTE: `max_host_weight_tensor_bytes()` is the **merged** host
tensor (27 GiB on the field model) vs the real 270 MiB staged slice, hence the `GGML_CUDA_SLAB_RING_MIB`
cap in `h2d_stage_region_bytes` (`common.cuh`).  The `--fit` lockstep (`ggml_backend_cuda_h2d_stage_bound`
returns 0 when the slab is enabled) keeps the ring out of the fit's compute.

## 3. Results (gfx1201, 2x R9700, `--fit on`, 31 482-token prose, 1000 MTP tokens)

| arm | prefill | decode | accept | NaN |
|---|---:|---:|---:|---:|
| stock r37 reference (r38 §15.9) | 922-924 | 61.5-61.8 | 1.0 | 0 |
| r38 G4 default 50 %, ring refused | 767-769 | 61.8-62.0 | 1.0 | 0 |
| this patch, ring never armed (baseline) | 781 | 62.2 | 1.0 | 0 |
| **this patch, `GGML_CUDA_SLAB_RING_MIB=2560`** (3 runs) | **945.5-946.8** | 61.2-61.4 | 1.0 | **0** |
| **this patch, default (6144)** | **992.2** | 61.1 | 1.0 | **0** |

Standing gates on this build: dense `Qwen3.5-4B-Q8_0` `-sm tensor` = **`1c5d32ac537d`**;
`scripts/gate-prefill-logits.sh` **PASS** mean KLD **0.000707** / same-top-p **98.755 %**;
`test-backend-ops -o MUL_MAT_ID` **931/931**; width purity 2-GPU Flash-Next `-ncmoe 48`
`none == n1 == n3 == n7` = **`b00fdf534227`**, 0 `////`.

Key timeline to recognize a good run (2.5 GiB region): wide `3072` init -> first small reserves `768`/`512`
-> narrow-floor hint (layout `1350.7` -> `1792`) -> narrow `1792` with **`boundary = 1792`** -> cache
`sized 337/295 slots/table` -> wide `2816` -> ring used -> post-prefill narrow `1792` (region reclaimed).

## 4. Remaining work, in order

### 4.1 Local cleanup (do first, cheap)

* Remove the diagnostic `GGML_LOG_WARN`s added for this session: `ggml_cuda_slab_set_ring_region`
  (the "H2D staging-ring region" line), `ggml_cuda_slab_set_narrow_floor` (the per-hint line),
  `ggml_cuda_slab_arena_alloc_transient` (first-carve), `h2d_stage_buffer` ("new max H2D staging
  request"), and `llama_model_moe_cache_preflight` ("max_host_weight_tensor_bytes = ...").
  Keep the slab-``init``/`extend` warnings (they are the normal delivery logs).
* Remove the `GGML_CUDA_SLAB_RING_NOTRANSIENT` debug switch in `common.cuh`.
* `ring_placed` is now an unused struct field — delete it.
* Run `git diff --check` and a warning-free build (`cmake --build ... 2>&1 | grep -i warning`).

### 4.2 `fingon` (gfx1100) validation

Same procedure as r38 §15.8: `fingon` = RX 7900 XTX (gfx1100) + gfx1036 iGPU, ROCm `/opt/rocm-7.14-gfx1100`,
`HIP_VISIBLE_DEVICES=0`.  Apply this patch to `fingon`'s `~/llama.cpp` (the box was left at the r37/r38
tree), build with `~/bin/build-llama-rocm-...` or `cmake --build build-rocm`, then:

* §11.4 staging A/B (`llama-bench -p 4096 -n 4 -r 1 -lm dio`) vs `GGML_SCHED_STAGE=0` /
  `GGML_CUDA_OPTIONAL_ALLOC_MAX_FREE_PCT={0,1}` — must not regress the r38 numbers (staging 4248.5 vs
  3539.1 off).
* Primed `llama-server` prefill, prompt = the 5246-token prose request.
* gfx1100 width purity (qwen35moe, embedded MTP, `-ncmoe 20`): `none == n1 == n3 == n7` (r38 gave
  `cd5e36218bd2`; this patch may legitimately differ, but must be pure).
* Record the region/`sized` lines and the arena residency.

Note: gfx1100 is a **single 24 GiB card**, so the narrow/wide split and the ring region sizing are smaller;
watch for the `narrow_floor + ring + chunk > mapped` guard disabling the ring (`h2d_stage_region_bytes`
returns 0 when it cannot fit) and for `cache_unusable`.

### 4.3 3-GPU gate

Run the §5.5 field config on `HIP_VISIBLE_DEVICES=0,1,2` (`-sm tensor`) and the standing gates (dense
golden is unaffected; width purity is the 2-GPU one).  Confirm per-device narrow floors and ring regions
are balanced and no device declines the slab.

### 4.4 Reserve-side decision (the maintainer's question)

Before the ring moved inside, it `cudaMalloc`'d from the outside **headroom** (`GGML_CUDA_SLAB_HEADROOM_MIB`,
default 4096, which also protects hipBLASLt).  Now the region is inside the slab, so
`want = free - (headroom + aux)` is unchanged and the region is carved from the arena; in decode it is
reclaimed, so the arena returns to baseline (why decode is ~61 with no reserve change).  The slab does not
**gain** the headroom the outer ring held — that slack is idle outside.

Experiment: reduce the reserve by the ring's *actual* share (not the worst-case region) so `want` grows and
the decode arena gains it.  Constraints:
* Do **not** starve hipBLASLt — keep the outside headroom at least the effective value the outer ring left
  (`headroom - ring_actual`).  A safe first cut: reduce by the ring's *live* use, or by
  `min(region, headroom/2)`, and gate it behind an env var (`GGML_CUDA_SLAB_RESERVE_MIB` already overrides
  the whole reserve, so an A/B can be done today by setting it manually to `headroom - X`).
* `--fit` (G2) uses `ggml_backend_cuda_device_slab_headroom_bytes` for its margin; if the reserve changes,
  the fit's headroom term must change in lockstep or the fit double-counts (see the `--fit` lockstep in
  §2).  The cleanest is to make the getter report `headroom - ring_share` too.
* Measure the §5.5 field arm with the reduced reserve; expect decode to move from 61 toward/above 62 with
  prefill unchanged; watch for hipBLASLt OOM (`hipblaslt.cpp:164`, Tensile `hipModuleLoad failed`) which
  means the headroom is too thin.

### 4.5 P2 — the FA prefill staging arena

The other raw outside allocation is the FA prefill staging arena (`fattn_stage_try_get` in `common.cuh`,
capped by G4).  It has a known per-launch bound (`ggml_backend_cuda_fattn_stage_bound`) and the same shape
as the ring: a transient that could live in the slab work region above the narrow floor.  Apply the same
mechanism:
* a second work sub-region (or share the ring region's unused tail) above the narrow floor;
* route `fattn_stage_try_get` through a slab transient allocator;
* the G4 FA-staging cap in block 15 then only guards what the slab cannot predict (or is retired).
P2 is where the *rest* of the field prefill win (the FA staging refusal on `fingon`) lands.  Keep the
`mtp-adaptive-methodology.md` rule-5 batched-bench gate and the prefill-logit gate in the loop.

### 4.6 P3 — retire / keep G4

Once P2 lands, audit what still allocates outside the slab (hipBLASLt Tensile objects + workspace, the MTP
draft buffer, compute-buffer growth).  If nothing transient is left outside, `GGML_CUDA_OPTIONAL_ALLOC_MAX_FREE_PCT`
becomes a diagnostic rather than a guard; decide with the maintainer whether to default it off, keep it as
a floor, or delete it (`ENVIRONMENT.md` §1.3).

### 4.7 Promotion (only after 4.1-4.4, and with the maintainer's go-ahead)

Follow the r38 `fit-slab-accounting` pattern (`WORKLOG.md` 2026-10-09 (later)):
* Fold the block-06-scoped changes into `patches/0006-…general-system-operations-bucket.patch` and any
  block-15 (`fattn_stage`) changes into `patches/0015`; regenerate with `scripts/make-patches.sh` from a
  canonical fork rebuilt at `release.json.base` via `scripts/apply-all.sh` — **never hand-edit** patches.
* Refresh `release.json` with `scripts/make-release.sh`; `scripts/validate-set.sh` must be green.
* Add a dated `WORKLOG.md` entry (newest first) with the field numbers and the gate results; update the
  `patches/README.md`/`README.md` headers and `ENVIRONMENT.md` (`GGML_CUDA_SLAB_RING_MIB`).
* Refresh the personal fork's `rdna-boosts` branch per the AGENTS.md Pushing policy (clean apply at the
  baseline, build, coherence + MTP + prefill-logit + width purity, `git push --force-with-lease` to
  `git@github.com:stew675/llama.cpp.git` only).  **Never** touch upstream `ggml-org/llama.cpp`.

## 5. Invariants and risks (do not weaken)

* The narrow view and its base are **stable**; the wide view's base is stable at `narrow_floor + ring_region`.
  A base shift under a live view is the corruption class this design exists to avoid.
* `work_needs` must store **boundary contributions**, not raw needs — a view allocated before the narrow
  floor existed otherwise inflates the floor.
* The narrow-floor hint only before `ring_used == 0`; once the ring is carved its VA is fixed.
* `arena_total` stays `mapped - boundary` (what is allocatable now).  Sizing the cap against a future
  settled arena makes `alloc_table_locked` fail and the cache self-disable.
* The ring region is **work**, never `arena_free` while the wide view is live (the ordering guarantees it).
* Keep the boundary-move corruption contract (`work_needs` floor, no chunk handed to work under a live
  view) intact.
* The 6 GiB `GGML_CUDA_SLAB_RING_MIB` default is the merged-tensor cap, not the real need (~1 GiB live):
  it is spent only during prefill (the region is reclaimed in decode), but a smaller cap is cheaper if the
  field tables fit (`2560` works for the field's 270 MiB tables and gives nearly the same prefill).

## 6. Files touched by the artifact

`ggml/src/ggml-cuda/ggml-cuda.cu`, `common.cuh`, `ggml-cuda-vmm.h`, `moe-expert-cache.{cu,h}`;
`ggml/include/ggml-backend.h`, `ggml/src/ggml-backend-impl.h`, `ggml-backend.cpp`,
`ggml-cpu/ggml-cpu.cpp`, `ggml-rpc/ggml-rpc.cpp`, `ggml-backend-meta.cpp`; `src/llama-context.cpp`,
`src/llama-model.cpp`.  13 files, +486/-54 pre-cleanup.

## 7. Pointers

* `README.md` §1-§9 (design, the top-of-arena NaN root cause), §10 (this ordering, the results, the
  sizing fix, the reserve note).
* `archive/work/fit-slab-accounting/README.md` §15.9 (the §5.5 field procedure and r38 numbers),
  §15.8 (`fingon`), §12.1/§12.3 (G3/G4).
* `archive/work/moe-cache-autosize/ARENA-UB-TENSION.md` §13/§14 (the movable-boundary slab), and
  `archive/work/moe-cache-autosize/` generally for the cache.
* `ENVIRONMENT.md` §1.3 (`GGML_CUDA_SLAB*`), §2 (`GGML_SCHED_STAGE*`);
  `benchmarks/mtp-adaptive-methodology.md` (rule 5), `benchmarks/prefill-logit-methodology.md`.
