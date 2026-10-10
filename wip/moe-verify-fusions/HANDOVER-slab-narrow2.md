# HANDOVER — slab-resident narrow-2 (the MTP draft compute view), end-pinned

**For a cold session. Status: READY TO IMPLEMENT.** This document is self-contained; read the campaign
`README.md` first for the #50/#51 framing and the directives (D1/D2), then this.

## TL;DR

The 2026-10-11 session found and fixed a **real defect**: `ggml_cuda_slab_work_alloc` hands out **one**
narrow base (`s.base + 0`) to every NARROW compute view, but with MTP there are **two live compute
buffers** — the target's verify buffer *and* the draft context's buffer — so they aliased at base 0.
The quick fix (allocate the draft's compute buffers with `cudaMalloc` instead) is implemented and
measured: **MTP acceptance 0.693 -> 0.738, decode 43.1 -> 53.5 t/s (~16 % lower verify step) on the GSQ
2-GPU field config.**

The maintainer wants the **long-term** form instead: a **second pinned narrow region, indoors in the
slab**, so the allocator stays "all in the slab" (no split slab/cudaMalloc, no fragmentation). That is
what this session must implement and test. The quick fix is the A/B reference and the fallback.

A **second, independent** defect remains: the plain-vs-MTP **verify-width divergence** (a near-tie flip;
`none != n1 != n3 != n7`) is **not** the aliasing — it survives the quick fix. It is the follow-on item,
*after* the slab-resident layout is in and validated (see "After this session" below).

---

## 1. The defect (confirmed, instrumented)

`ggml_cuda_slab_work_alloc` (`ggml/src/ggml-cuda/ggml-cuda.cu`) implements exactly **two** stable bases:

```
ret_off = narrow ? 0 : wide_off;      // wide_off = narrow_floor + ring_region
```

The design comment on `ggml_cuda_slab_work_release` already notes *"more than one compute buffer is live at
a time (the main context and the MTP draft context each own one)"* — for the **boundary floor**
(`work_needs` is a multiset) — but **base assignment** still aliases them.

Instrumented proof (env `GGML_CUDA_WORK_ALLOC_DEBUG=1`), MTP run, device 0:

```
need=768.0  MiB narrow=1 ret_off=0.0 MiB narrow_floor=768.0    <- draft ctx compute, base 0
need=512.0  MiB narrow=1 ret_off=0.0 MiB narrow_floor=768.0    <- draft ctx second buffer, base 0 (ALIAS)
need=1792.0 MiB narrow=1 ret_off=0.0 MiB narrow_floor=1792.0   <- target verify buffer, base 0 (ALIAS)
```

Plain run, device 0: only the target's `1792` — no alias.

The draft's two COMPUTE buffers (768 + 512 = ~1.25 GiB/device; the draft's *weights* are already
`cudaMalloc`'d — the slab is `GGML_BACKEND_BUFFER_USAGE_COMPUTE`-only) overlap the target's `1792` view.
The draft context is created after the slab init and **drops + lazily re-reserves** its compute buffers
(`llama_context_drop_compute_buffers`), so the alias recurs at the first `alloc_graph`.

---

## 2. The quick fix that is currently in the working tree (and its numbers)

Small, upstream-shaped: give the MTP draft context a way to keep its COMPUTE buffers **out of the slab**
(raw `cudaMalloc` from the slab reserve, which already budgets the draft via
`LLAMA_MOE_CACHE_AUX_RESERVE_MIB_DEFAULT = 4096`).

### 2.1 Functional changes (in `session-quickfix-and-diagnostics.diff`)

* `ggml/include/ggml-backend.h` + `ggml/src/ggml-backend-impl.h` + `ggml/src/ggml-backend.cpp`:
  new optional hook `void (*slab_compute_enable)(ggml_backend_dev_t, bool)` and public
  `ggml_backend_dev_slab_compute_enable(dev, enable)`.
* `ggml/src/ggml-backend-meta.cpp`: the meta device **forwards** `slab_compute_enable` to its
  `simple_devs`.
* `ggml/src/ggml-cuda/ggml-cuda.cu`: `static bool g_slab_compute_enabled = true;`; in
  `ggml_backend_cuda_buffer_type_alloc_buffer_usage`, the COMPUTE branch is gated on
  `ggml_cuda_slab_enabled() && g_slab_compute_enabled`; the CUDA device hook sets the flag.
* `src/llama-context.cpp`: the MTP draft context is identified by `cparams.ctx_other != nullptr`; the flag
  is turned off around **both** allocation paths:
  * `llama_context::graph_reserve`'s real `ggml_backend_sched_reserve` (line ~3169), and
  * `llama_context::process_ubatch`'s `ggml_backend_sched_alloc_graph` (line ~1892).
  The first cut only wrapped `graph_reserve`; the `512` is allocated in the `alloc_graph` path, so it
  still aliased — **both** are required.

### 2.2 Measured result (GSQ-IQ3_XXS + shared Q8_0 draft, 2 GPU, `--fit on -c 204800`)

| metric (n3) | pre-fix | post-fix |
|---|---:|---:|
| draft acceptance | 0.69318 | **0.73835** |
| eval ms/accepted token | 23.20 | **18.68** |
| decode t/s | 43.11 | **53.53** |
| derived verify step | ~71.4 ms | **~60.1 ms** |

Instrumented post-fix: the draft's `768` and `512` are `flag=0` (cudaMalloc); only the target's `1792`
takes the slab (`ret_off=0`, `narrow_floor=1792`). No cross-context alias remains. The plain (no-draft)
run is untouched (`is_draft == false` -> identical slab path).

### 2.3 Diagnostics mixed into the same diff (must be handled)

The saved diff also contains **temporary diagnostics** that are NOT part of the fix. Decide to keep
(env-gated) or revert them — do NOT fold them into `patches/`:

* `GGML_CUDA_WORK_ALLOC_DEBUG` logs in `ggml_cuda_slab_work_alloc` and
  `ggml_backend_cuda_buffer_type_alloc_buffer_usage`, plus a `GRAPH_RESERVE` log in `llama-context.cpp`.
* `GGML_CUDA_CACHE_GUARD_DEBUG` log in `ggml_cuda_cache_blocks_fusion`.
* `moe_cache_guard_stats()` (`moe-expert-cache.cu/.h`) — the counter behind that log.
* `GGML_CUDA_SKIP_ALLOC_DEPS` — skip ONLY the `graph_optimize` alloc-deps pass (separates the
  `GGML_CUDA_DISABLE_FUSION` confound); **proved the deps are not a factor**.
* `GGML_CUDA_PREFILL_FUSION_OFF` — disable fusions for wide graphs only; **proved the prefill fusions are
  not the divergence** (it changes the trajectory only by moving the near-tie).
* `GGML_FORCE_N_RS_SEQ` (`common/common.cpp`) — force `n_rs_seq`/`n_rs_batch`; **proved the recurrent
  snapshot count is not it** (plain RS=3 == plain RS=0 exactly).
* `NEXTN_EXPS` log in `src/llama-model-loader.cpp` — **proved the MTP head's own experts are
  GPU-resident** (`blk.48.ffn_*_exps host=0 buft=Meta()`).

---

## 3. WHAT THIS SESSION MUST DO — the end-pinned narrow-2 layout (option 2)

**Goal.** Keep *both* static compute views **inside the slab**, pinned at the two ends, with everything
dynamic between them:

```
base 0                                                              mapped
| narrow-1 (target verify) | ring | wide (prefill) | arena (top-down) | narrow-2 (draft) |
```

This keeps the slab's single invariant — *every live view's VA never moves* — and removes the
slab/`cudaMalloc` split (no fragmentation). Then **remove the quick fix's bypass** (or leave the hook
unused) so the draft's compute buffer is served by narrow-2.

### 3.1 Key design decision: anchor narrow-2 at the top of the **reserved VA**, not `mapped`

Pin narrow-2 at `[size - n2, size)` (the top of the `cuMemAddressReserve`d range), **not** at the current
`s.mapped`. Reason: `ggml_cuda_slab_extend` maps more physical above `mapped`; if narrow-2 sat at
`mapped`, an extend would either relocate a live view (forbidden) or split the arena. With
`narrow2_off = size - n2`, `slab_extend` maps the arena only up to `narrow2_off`, and narrow-2 is a
separate mapping at the very top — nothing moves and no table is evicted to make room.

### 3.2 Code changes (all in `ggml/src/ggml-cuda/ggml-cuda.cu`)

1. `struct ggml_cuda_slab`: add
   ```cpp
   size_t narrow2_floor = 0;   // second static narrow view (MTP draft), pinned at the TOP
   size_t narrow2_off   = 0;   // = size - narrow2_floor, fixed once established
   ```
   and a helper `arena_top()` = `narrow2_floor ? narrow2_off : mapped`.
2. Replace every arena-top use of `mapped` with `arena_top()`:
   * `ggml_cuda_slab_arena_alloc` (fills top-down from `arena_top`)
   * `ggml_cuda_slab_arena_free`
   * `ggml_cuda_slab_arena_total` (cap = `arena_top - boundary`)
   * the coalescing / "hand the new range to the arena" in `ggml_cuda_slab_extend`
3. `ggml_cuda_slab_extend`: cap the mapped growth at `arena_top() - mapped` (VA), keep new physical in the
   arena. (Do not map above narrow-2's region.)
4. Establish narrow-2 from a **new backend hook** (analogous to `slab_narrow_floor`), e.g.
   `slab_narrow2_floor(dev, bytes)`: map `[narrow2_off, size)` with `ggml_cuda_vmm_map_phys`, set
   `narrow2_floor`/`narrow2_off`. It must be **idempotent** (the draft drops and re-reserves) and must
   refuse (fall back to cudaMalloc) if `size - n2 < boundary + min_arena` or the physical map fails.
5. **Identity in `work_alloc` — the real design bit.** Today `work_alloc` cannot tell narrow-1 from
   narrow-2. Recommended: assign narrow slots by **first-request order** with a small live-view registry
   `{offset, size, live}` in the slab struct: the first live narrow request gets `0`; the next gets
   `narrow2_off`; keep the assignment stable across the target's drop/re-reserve. Alternative: thread an
   explicit role down from `llama_context` (it knows which context is reserving) via a per-device
   "current reserve is the draft" flag — same mechanism as the quick fix's `ctx_other` test.
6. **Establishing narrow-2 physical / accounting.** narrow-2's physical comes from the VRAM
   `GGML_CUDA_SLAB_RESERVE_MIB` already held *outside* the slab for the draft
   (`g_slab_aux_reserve_bytes`, default `LLAMA_MOE_CACHE_AUX_RESERVE_MIB_DEFAULT = 4096 MiB`). Subtract
   `narrow2_floor` from the reserve (or size the reserve so the draft's compute bytes are not held twice)
   so `--fit` and the arena projection stay correct. `--fit` itself needs no change: it already subtracts
   the aux and the slab is sized `free - reserve` before the draft loads.
7. `ggml_cuda_slab_work_release` already receives the per-buffer `slab_off` (`ctx->slab_off` in
   `alloc_buffer_usage`) — use it for the registry; no new plumbing beyond the slot table.

### 3.3 Prune the quick fix

Once narrow-2 serves the draft, remove the `cudaMalloc` bypass (the `slab_compute_enable` hook can stay
as an unused/optional escape hatch, or be deleted). Keep the quick fix patch as the A/B arm.

---

## 4. Validation plan

**Reproduce the divergence and the hit rate before/after.** Use the harness
(`tools/field3.sh`, `field_nospec.sh`) and the exact per-token comparison below.

1. **No alias:** `GGML_CUDA_WORK_ALLOC_DEBUG=1` must show narrow-1 and narrow-2 at **distinct**
   `ret_off`s, both narrow, and the draft's `768`/`512` served by narrow-2 (not cudaMalloc).
2. **Hit rate / step:** MTP acceptance and `eval ms/accepted token` on GSQ 2-GPU must be **>= the quick
   fix** (0.738 / 18.68 ms) and far above the pre-fix (0.693 / 23.20 ms).
3. **No regression on no-draft runs:** the target's slab path must be byte-for-byte unchanged
   (dense golden `1c5d32ac537d`; the plain `none` token stream unchanged vs r39).
4. **Standing gates** (the campaign README's list): `scripts/gate-prefill-logits.sh` (mean KLD 0.005 /
   98 %), `test-backend-ops -o MUL_MAT_ID` 931/931, width purity `none == n1 == n3 == n7`
   (the remaining divergence may still fail on GSQ — that is item 3 below, not this one).
5. **Stability:** a long MTP run (>= 1000 accepted tokens) with no abort/OOM; check
   `slab_extend` leaves `GGML_CUDA_SLAB_HEADROOM_MIB` free and narrow-2's establishment evicts **no**
   arena table.
6. **3-GPU `-sm tensor`** check of the verify step (the original #51 symptom) as the perf confirmation.

**Do not run parallel benches.** Warm up before recording (the first run after a config change pays the
lazy PLE / host-expert read). `llama-cli` MUST use `--single-turn`. `-fa off` is **invalid** with
`-sm tensor` (the server exits; a health-check loop then spins — the runner needs a `kill -0` liveness
check).

---

## 5. After this session — the remaining divergence (item 3)

The plain-vs-MTP **verify-width** divergence is **not** the aliasing. Deterministic evidence on the GSQ
2-GPU harness config, exact generated token ids (`return_tokens: true`), compared to the `--spec-type
none` stream:

| config | first divergence vs plain |
|---|---:|
| n1 (2-row verify) | idx 103 (post-quickfix) / 125 (pre) |
| n3 (4-row verify) | idx 96 (post-quickfix) / 69 (pre) |
| n7 (8-row verify) | idx 64 |

Ruled out (each changed nothing or only moved the flip):

* fusions (`GGML_CUDA_DISABLE_FUSION=1` still flips; individual switches no-op),
* the alloc-deps pass (`GGML_CUDA_SKIP_ALLOC_DEPS`),
* the MMVQ dense/MoE bands (`GGML_CUDA_DISABLE_MMVQ_*_BAND`),
* GDN chunked/BF16, QSA dense, FA-native, norm-Q8_1/add-rms/MMID-512,
* `n_rs_seq` (forced 0/3 — plain RS=3 == plain RS=0 exactly),
* the cache partial residency (plain partial == plain full, bit-identical),
* CPU numerics (MTP-CPU == plain-CPU exactly; the default MTP uses `cold=uva`, not the CPU).

What remains: the cache-band **verify-width kernel arithmetic** (`none == n1 == n3 == n7`) on
**IQ3_XXS**. The documented guarantee is text-level for f16/bf16/q8_0 and logits-level for coarse quants
(`GREEDY-PURITY.md` §36), and GSQ is a coarse i-quant. **Recommendation before any kernel work:** classify
GSQ against the documented purity model (IQ4_NL, 2-GPU Flash-Next `-ncmoe 48`, hash
`b00fdf534227`) — the maintainer is willing to load it once now that the slab fix is (soon) in place. If
IQ4_NL stays pure, GSQ's flip is the §36 coarse-quant band edge, a quality measurement, not a delivery
bug; if IQ4_NL is impure, pursue `wip/mmvq-verify-rows`/the routed-expert MMVQ.

---

## 6. Reproducers / environment

Models (D2): GSQ-IQ3_XXS + the shared MTP draft.
```
M=/llm/models/Qwen3.8/Flash-Next/GSQ-IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf
D=/llm/models/Qwen3.8/Flash-Next/IQ4_NL/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf
PROMPT=/tmp/srr/mixed30k.txt   # README+ENVIRONMENT+CONTAINERS+archive/docs/baseline-history.md,
                               # sha256 69624f4d207f40bd..., ~32k tokens (non-degenerate)
```
Harness: `wip/moe-verify-fusions/tools/` (`field3.sh` MTP, `field_nospec.sh` plain,
`field.sh` forces `GGML_CUDA_ALLREDUCE=ce`), `GPUS=0,1` or `0,1,2`, `LOG=/tmp/srr`.

The exact per-token comparison (server, `--fit on -c 204800`, prime then the 32k prompt,
`return_tokens: true`):
```bash
env HIP_VISIBLE_DEVICES=0,1 LD_LIBRARY_PATH=/opt/rocm-7.14-gfx120X/lib \
  ~/llama.cpp/build-rocm-hybrid/bin/llama-server -m $M \
  --spec-type draft-mtp --spec-draft-model $D --spec-draft-n-max 3 \
  -sm tensor -ncmoe 48 -ub 6144 -b 6144 -c 204800 --no-kv-unified \
  -ctk q8_0 -ctv q8_0 -fa on -t 8 --fit on --host 127.0.0.1 --port 8949 --no-webui --cache-ram 0
# prime {"prompt":"Hello","n_predict":8}; then the mixed30k prompt with "return_tokens":true
```
Build: `cd ~/llama.cpp && cmake --build build-rocm-hybrid --target llama-server llama-cli -j 16`
(fresh configure needs `EXTRA_CMAKE_FLAGS="-DCMAKE_HIP_FLAGS="`).

**Current working tree:** `~/llama.cpp` on `rdna-boosts` (block-15 tip) with the quick fix +
diagnostics above, built into `build-rocm-hybrid`. `git diff` is saved as
`session-quickfix-and-diagnostics.diff` beside this file. Nothing is committed; `patches/` and
`release.json` are untouched.

## 7. Pointers

* `wip/moe-verify-fusions/README.md` — campaign framing, D1/D2, ordered split-tests, gates.
* `TODO.md` #50/#51 — the tracker items.
* `wip/mmvq-verify-rows/` — the multi-row routed MMVQ kernel; the remaining width impurity lives here.
* `GREEDY-PURITY.md` §36 (coarse-quant band edges) and §39 (the cache band).
* `ENVIRONMENT.md` §1.3 (slab knobs) and §7 (fusion switches).
* `archive/work/slab-ring-region/HANDOVER.md` §4.3.1 — the 3-GPU table and the clean-r38 control.
