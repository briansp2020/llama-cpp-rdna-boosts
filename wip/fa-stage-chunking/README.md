# `fa-stage-chunking` — bound the FA prefill staging arena to O(1) by converting per KV chunk

**Status: ACTIVE (2026-10-27). NOT part of the delivery.**  Nothing here is applied to `patches/` or the
fork until the maintainer promotes it (see the WIP rules in `AGENTS.md`).  This directory is the scoping +
measurement record; the decision it exists to produce is stated in §0.

Seeded from the **issue #237 follow-up** (@DanoPTT, `issues/237#issuecomment-6453000019`): on
Windows/WDDM the r156 growth ladder makes a 100k prefill ~22 % slower than r149, because the last 1 GiB
arena step lands in shared memory.  The r156/r149 delta there is *exactly* the arena
(`+899` → `+1923` MiB shared for a f16 draft KV, `+2949` with a q8_0 draft): one and two 1024 MiB arenas
that arrived late.  See §1 for the arithmetic.

> **STATE (2026-10-27): M0 + M1 + M2 done; recommendation is C, pending the maintainer's call.**  The
> staging is worth keeping (+1.5/+2.7/+5.1 % pp, §3), and a chunked prototype works and is numerically sound
> (`FINDINGS-2026-10-27-m2.md`), **but** the MMA prefill kernel's only multi-block KV split is stream-K, so
> a chunked launch currently costs 2.8x prefill.  Tiling therefore requires re-expressing the stream-K
> partial protocol (the same path the decode band's purity depends on) — a deliberate follow-up, not a
> launcher tweak.  Read §0 with `FINDINGS-2026-10-27-m2.md` §"Verdict".

## 0. The decision this campaign must produce

> Is the F16 prefill staging worth its memory at depth — and if it is, can it be **tiled** so its
> footprint is O(one KV chunk) instead of O(the whole prefix)?

Three outcomes, all acceptable:

* **A. Tile it.**  The staging survives (most of the prefill win) at ~16-32 MiB constant, on every
  device.  `#138` (ladder stranding), `#237` (eager over-commit) and the follow-up (late step → WDDM
  shared memory) all disappear at the root, `GGML_CUDA_FA_STAGE_PREALLOC` is deleted, and on host-expert
  configs the bytes return to the MoE arena (the second prize, §5).
* **B. Drop it.**  The staging is worth less than its cost at the depths that matter → the native read
  (already the RDNA3_5 default and the automatic fallback) becomes the default for quantized KV caches,
  the arena goes away entirely, and the answer to the reporter is "you were managing a buffer you should
  not have had".
* **C. Bound it.**  Tiling costs too much in either numerics or prefill speed, but the memory must still
  be bounded → the transient-region/`PREALLOC` route from the issue thread (reserve once, early,
  bounded) stays as the answer.

The campaign is done when one of those is chosen with numbers, recorded in `RESOLUTION.md`.

## 1. What grows, exactly

`ggml_backend_cuda_context::fattn_stage[stream]` — a **temporary F16 copy** of the K and V rows the
current FA op reads, per layer. It is allocated only when the cache is *quantized* (`q8_0`/`q4_0`/`bf16`)
**and** the op is wider than 8 rows (prefill; `native_width = Q->ne[1] <= 8`).  An f16 KV cache never
creates it.

```
stage_req  = F16 bytes of (n_past + n_q) rows, summed over the operands staged
stage_hint = F16 bytes of the K/V view ROOT (= the whole cache), capped per operand by
             GGML_CUDA_FA_STAGE_MAX_MB (default 512 MiB)
```

The reporter's `GGML_CUDA_SLAB_STATS=1` ladder is that formula, per prefill ubatch:
`request=2.0` (512 rows × 1024 elems × 2 B × 2 operands) → `18` → `74` → `298` → `alloc=1024` (the
per-operand cap hit), and the *same* ladder again for the MTP draft context because since r125 the draft
KV follows `-ctk` (`llama-context.cpp`: the draft context gets its own backends, hence its own
`fattn_stage[]`).

So: **not the KV cache** (allocated once at context creation from `-c`/`-ctk`, never grows), and **not an
"FA window"**.  It is a converted copy, proportional to the *used prefix*, per layer, transient.

## 2. Why it is prefix-sized today

`launch_fattn` (`ggml/src/ggml-cuda/fattn-common.cuh`, block 05) hoists the conversion out of the launch:

```
to_fp16(K)  ->  to_fp16(V)  ->  ONE FA kernel launch whose blocks walk the whole KV extent
```

Prefill is a single launch: `blocks_num.x = ntiles_dst` (query tiles), plus the stream-K / split-KV
machinery (`parallel_blocks`, `stream_k`, `kq_blocks`) — but every one of those blocks may touch any KV
tile, over the *whole* `K->ne[1]`.  Therefore the entire converted prefix must be readable for the
duration of that launch.  That is the whole reason the arena is O(prefix), and therefore the whole reason
it ladders, strands, over-commits and lands late.

## 3. What the staging is worth (the prize being defended)

Without staging, each K/V tile is dequantized once per **query** tile (`ntiles_x`); with staging, once.
Recorded in the tree's own comments (block 05, gfx1201, q8_0 KV): **pp150k 691 staged vs 661 native**,
pp32k 1087 vs 1073; on RDNA3_5 it is negative at every depth, which is why gfx1151 already takes the
native read at prefill (`prefill_stages = !GGML_CUDA_CC_IS_RDNA3_5(cc)`).  So the prize is real but
depth-scaled: ~+4.5 % at 150k, ~+1 % at 32k, ~0 at 16k (`fa-stage-prealloc` measured pp16384 within noise).

Two things follow, and both are measured in M0:

1. The number that decides A-vs-B is the *staged minus native* pp gap at the depths we care about, on
   this box, not the historical one.
2. If tiling preserves that gap it is strictly better, because it also removes the serial conversion
   pre-pass: today `to_fp16` over a 100k prefix runs on the compute stream *before* the launch.  Per-chunk
   conversion can overlap the previous chunk's attention on the copy stream — the same effect Strata
   measures in their `KV_PREFETCH` (prefill −5.3 % at 32K, −4.1 % at 64K with an *upload*; ours is a
   conversion).  So outcome A may be a prefill **win**, not a tax.

**Measured (M0a, `FINDINGS-2026-10-27-m0.md` §F2)** — 27B UD-Q4_K_XL, 1 GPU, q8_0 KV, `-ub 512`,
`llama-bench -r 2`, native forced with `GGML_CUDA_FA_STAGE_MAX_MB=1`:

| pp | staged (default) | native | staged vs native |
|---|---|---|---|
| 8192 | **1324.60** | 1304.79 | **+1.5 %** |
| 16384 | **1246.48** | 1213.84 | **+2.7 %** |
| 32768 | **1139.94** | 1085.12 | **+5.1 %** |

So the staging buys real prefill time that grows with depth → **outcome B (drop it) is out**.

> **Measurement trap (cost this campaign an hour):** `llama-bench` installs a null log callback
> (`llama-bench.cpp:2345`) and `llama-cli` needs `-v` to show `GGML_LOG_WARN`, so `FA_STAGE`/`SLAB_*` are
> invisible in a default `llama-bench` run.  Use `llama-bench` for throughput and `llama-cli -v` for state,
> and check the `[ Prompt: … t/s ]` line (a too-long `-f` prompt exits 0 and prefills nothing).

## 4. Proposed design (to be validated by the feasibility read, M1)

```
for each KV chunk [k0, k1) of C rows:            C >= enough to fill the device (nsm * tiles)
    convert K[k0:k1] and V[k0:k1] into slot (i % S) of a small pinned-to-VRAM ring (S = 2 or 3)
    launch the FA kernel for KV range [k0, k1)  -> partials into dst_tmp
combine the partials (the existing fixup kernel)
```

Memory: `S * 2 * C * head_dim * 2 B` — ~16-32 MiB at C=8192 on a dense head, constant in depth.

Feasibility questions the M1 read must answer **before** any code (each is a potential blocker):

* **Which reduction order exists already?**  The decode band and `parallel_blocks` already produce
  per-block partials + a fixup (`dst_tmp`, `dst_tmp_meta`).  Does the prefill path's partial accumulation
  give a clean host-side "process KV range [k0,k1)" entry point, or would the host loop need a new
  `k_KV` parameter threaded through the kernels?  This decides the size of the change.
* **The mask.**  `KV_max` / `kq_blocks` / the fully-masked-group bitmap are computed over the whole mask.
  A per-chunk launch needs the chunk's slice (and the scan must stay cheap — it is per chunk now).
* **Sparse (QSA) path.**  `use_sparse` index lists reference arbitrary KV columns; a chunk must either
  keep the full list (fine, it is small) or filter per chunk.  QSA is block 11 and its own gate
  (`test-backend-ops -o FLASH_ATTN_QSA`) is the oracle.
* **`V_is_K_view`** (V aliasing K) and the `nbatch_fa` tile alignment of `C`.
* **The graph.**  Prefill is never CUDA-graph captured (`ggml_backend_cuda_graph_compute` skips
  multi-token graphs), so a host loop costs only enqueue time; but a loop of N launches × layers must be
  checked against the dispatch-bound shape recorded in r144 (`graph_compute` calls per token are already
  the decode bottleneck; prefill is not).
* **Ring reuse and the second stream.**  A ring slot may only be refilled after the launch that read it
  has consumed it; with one launch per chunk on one stream that is automatic (in-order), and an
  overlapped conversion on the copy stream needs events — the delivery already has this pattern in the
  H2D ring and the async-input ring.

### 4a. M1 result (2026-10-27, code read — no prototype yet): the split-and-combine machinery exists

`launch_fattn` already has **two** N-way KV-split paths, both with a host-visible partial buffer:

* `parallel_blocks` (the non-stream-K path, `fattn-common.cuh` ~2030-2080): `blocks_num = (ntiles_x,
  parallel_blocks, …)`, the kernel takes its KV share from `blockIdx.y`, partials land in
  `dst_tmp`/`dst_tmp_meta`, and the launcher finishes with
  `flash_attn_combine_results<DV>(dst_tmp, dst_tmp_meta, KQV, parallel_blocks)` — **the same combine the
  decode band runs every step**.
* `stream_k` (Ada/`DKQ==64`/low-efficiency shapes): a 1-D grid whose blocks walk KV ranges, combined by
  `flash_attn_stream_k_fixup_uniform` / `_general`.

So the *reduction* side is not the work.  What is missing for a chunked prefill is only:

1. **A KV base offset the kernel honours.**  Today the KV index is derived from `blockIdx.y` (and the
   stream-K arithmetic) and is also the **mask column** and the `cell_pos`/`tok_lo`/`tok_hi` index.
   Threading one `kv_start` scalar (and offsetting `K_data`/`V_data` by `k0` rows, `n_kv = chunk`) keeps
   the causal alignment intact; that is a signature change in the MMA kernel (`fattn-mma-f16.cuh`) and
   its instantiations, not a rewrite.
2. **`dst_tmp`/`dst_tmp_meta` sized by chunk count**, with one `flash_attn_combine_results` call at the
   end (`n_chunks` instead of `parallel_blocks`).  `C` should be a multiple of `nbatch_fa` so no chunk
   ends mid-tile.
3. **The mask scans stay once-per-op.**  `flash_attn_mask_to_KV_max` and `flash_attn_mask_to_KV_blocks`
   are per *tile* already (`KV_max[sequence*ne31 + jt]`, `kq_blocks`), so a chunked loop reads its slice of
   them, and chunks beyond a tile's `KV_max` are skipped outright.  Do **not** rescan the mask per chunk.
4. **The sparse path stays single-launch.**  `use_sparse` (QSA) index lists address arbitrary KV columns;
   chunking it is a separate change with its own oracle (`test-backend-ops -o FLASH_ATTN_QSA`).  Gate the
   loop on `!use_sparse`, which is also block 11's territory rather than block 05's.

Scope this suggests: **MMA-f16 only** (the RDNA4/RDNA3_0 prefill kernel, which is where the arena is
reached); the tile kernel keeps today's single-launch staging (its `prefill_stages` is already false on
RDNA3_5), so the risk stays off the RDNA2/`bf16` arm.  `V_is_K_view` needs no special case (it is a
pointer alias, so one conversion covers both operands).

The two costs to watch in the prototype: (a) the **numerics re-baseline** (chunk boundaries change the
combine order — the prefill-logit gate, §7), and (b) **enqueue count**: `n_chunks ≈ payload/8192` extra
launches per layer per ubatch — irrelevant at `-ub 512` (one chunk), and ~13 at 100k across ~60 layers ≈
800 extra launches per ubatch, which must be measured against the dispatch-bound shape of r144 before it
is called free.

## 5. The second prize: arena bytes on host-expert configs (maintainer's hypothesis) — **measured, and it is not a decode prize**

On a config where the movable-boundary slab is armed, the staging arena is *inside* the slab
(`ggml_cuda_slab_wide_aux_alloc`, r86): it sits above the boundary, so growing it evicts arena tables
(`moe_cache_evict_slab_range`) and shrinking it returns bytes to the expert arena.  The field GSQ figure
is **762 MiB per device** for the FA staging alone, and r114's measurements put ~**1 pt of expert-cache
hit rate per GiB** moved from the headroom into the arena.

**M0b (35B-A3B UD-Q4_K_M, 2 GPU `-sm tensor -ncmoe 40`, `-c 65536`, 46k prompt, q8_0 KV) says the decode
half of that does not happen:** the arena is already slab-backed (`FA_STAGE … slab=1`), its peak slab
footprint is 64 MiB of a 26.5 GiB arena (0.24 %), and it is **released at the prefill → decode
transition** — after the drop both arms read `wide_aux=0, arena=28480 MiB`, i.e. *identical*.  Decode was
75.6 (staged) vs 76.5 (native) t/s, noise; prefill 2496.6 vs 2433.2 (+2.6 % staged).

So the second prize is (a) **prefill-time** arena pressure, not decode residency, and (b) the *dense* and
*Windows* configs, where the arena is a raw `cudaMalloc` outside every manager (no slab on a dense model;
no HIP VMM on WDDM).  It must **not** be claimed as a decode/hit-rate win in any release note.

Tables and the two log-visibility traps: `FINDINGS-2026-10-27-m0.md` §F4/F0.

## 6. Measurement plan

All runs are **sequential** (never two GPU jobs at once), `HIP_VISIBLE_DEVICES` pinned (device 3 is the
iGPU and must never be enumerated), and every arm records the arena bytes from
`GGML_CUDA_SLAB_STATS=1`.

| id | what | config | arms |
|---|---|---|---|
| **M0a** ✅ | what the staging is worth at depth (dense) | Qwen3.8-27B UD-Q4_K_XL, 1 GPU (`0`), `-fa on -ctk/-ctv q8_0`, `llama-bench -p 8192,16384,32768 -n 0 -ub 512 -r 2` | staged **+1.5 / +2.7 / +5.1 %** over native → see `FINDINGS-2026-10-27-m0.md` §F2 |
| **M0b** ✅ | the arena→slab→decode prize | Flash-Next IQ4_NL `-sm tensor -ncmoe 48` on 2 GPUs (`0,2`), fixed `MOE_EXPERT_CACHE_MIB`, MTP draft as in the r144 records | run on 35B-A3B instead (load time): **no decode prize** — the arena is released at the drop, `wide_aux=0 / arena=28480` in both arms.  §F4 |
| **M1** ✅ | the feasibility read (§4) — launcher + kernel entry points; no code | — | **feasible without new machinery** — the split/combine exists (`parallel_blocks` + `flash_attn_combine_results`, and the stream-K fixups); the work is one `kv_start` scalar threaded into the MMA kernel + chunk-sized `dst_tmp` + one combine.  §4a |
| **M2** ✅ | prototype: `C`-chunked conversion + per-chunk launch, single stream first | same as M0a | built (`PROTOTYPE-chunked-staging.patch`): correct (PPL 6.6302 -> 6.6261 at 4 chunks) and the arena is the ring, but **2.8x slower** because the MMA kernel is stream-K-only.  `FINDINGS-2026-10-27-m2.md` |
| **M3** | prototype with the overlapped conversion on the copy stream | same as M0a + M0b | as M2 |

Decision criteria (written before the numbers, so they cannot be rationalised afterwards):

* **A (tile)** if chunked pp is within ~1 % of staged pp at >= 16k **and** the arena becomes O(chunk) —
  regardless of the M0b prize.
* **B (drop)** if staged−native at the depths we care about is under ~1 % on both gfx1201 and gfx1151, or
  if tiling loses more than ~2 % of prefill.
* **C (bound)** if tiling is a numerics or speed loss of > 2 % but the staging is worth > 1 %.

## 7. Risks

* **Numerics.**  Chunking changes the prefill reduction order at the chunk boundaries → prefill logits
  move → `scripts/gate-prefill-logits.sh` is a **re-baseline**, not just a check (mean KLD ≤ 0.005,
  same-top-p ≥ 98 %; `benchmarks/prefill-logit-methodology.md`).  Greedy text may move too (a prefill
  change is not decode-purity-gated, but `GREEDY-PURITY.md` §46 is the rule).
* **The band must not move.**  `native_width = Q->ne[1] <= 8` and the band's geometry (`r134`/issue #224,
  §51) are decode paths; a chunking change must leave them bit-identical.  The `W=1..8` probe row hash is
  the cheap check.
* **Perf regressions from the loop.**  More launches per layer per ubatch; prefill is compute-bound, but
  the chunk must be large enough that each launch still fills the device (measure C=4096/8192/16384).
* **Third-party expectations.**  The reporter is on Windows and on r149; any answer we give them must say
  what happens on WDDM, and the confirmation is their re-test (the WDDM symptom cannot be reproduced on
  Linux/KFD by construction — `fa-stage-prealloc`/`#138` established that).

## 8. Gates for promotion (only if A is chosen)

`gate-prefill-logits.sh` at a **re-recorded base** (`--kv q8_0`, the arm that actually reaches the
arena), the `W=1..8` probe, `test-backend-ops -o FLASH_ATTN_EXT` (the full mask/type matrix) and
`-o FLASH_ATTN_QSA`, the coherence gate on the dense/MoE/Flash-Next arms, MTP Protocol A, and the pp
A/B at 2k/32k/64k.  Promotion target: **block 05** owns `fattn-common.cuh`; the arena's allocation policy
is block 10.  `GGML_CUDA_FA_STAGE_PREALLOC` / `GGML_CUDA_FA_STAGE_MAX_MB` rows in `ENVIRONMENT.md` are
updated or removed by the same change.

## 9. Touchpoints (read-only until M2)

| file | what |
|---|---|
| `ggml/src/ggml-cuda/fattn-common.cuh` | `launch_fattn`: the conversion, `stage_req`/`stage_hint`, the launch, the `KV_max`/`kq_blocks` mask scans |
| `ggml/src/ggml-cuda/common.cuh` | `fattn_stage_try_get`, `fattn_stage_bytes`, `ggml_cuda_fattn_stage_max_bytes` |
| `ggml/src/ggml-cuda/fattn-*.cu` / `template-instances/` | the kernel entry points (`k_KV` threading, if needed) |
| `ggml/src/ggml-cuda/ggml-cuda-slab.cu` | `ggml_cuda_slab_wide_aux_alloc/free` (the arena's slab home) |
| `src/llama-context.cpp` | `--fit` accounting via `ggml_backend_cuda_fattn_stage_bound` (r156 note: it reserves the worst case) |

## 10. Log

* **2026-10-27** — campaign opened from the #237 follow-up.  §1-§3 written from the code and the
  reporter's log.
* **2026-10-27 (later)** — **M0 done** (`FINDINGS-2026-10-27-m0.md`).  Confirmed the staging is the
  default RDNA4 prefill path (MMA_F16 passes `need_f16_K/V = true`; the launcher stages at wide ops and
  reads native in the band), that it is worth **+1.5 / +2.7 / +5.1 %** prefill at 8k/16k/32k (so B is
  out), and that the **decode** half of the slab hypothesis is false (the arena is released at the
  prefill→decode drop; identical arena bytes in both arms).  Arena sizing matches the formula exactly
  (16 → 68 → 160 MiB at 32k on the 27B).  Next: **M1** (feasibility read of `launch_fattn`'s KV split and
  the three kernels' entry points), then the M2 prototype.
* **2026-10-27 (M1)** — feasibility read done (§4a): the KV split + combine already exists for both the
  `parallel_blocks` and the stream-K paths (`flash_attn_combine_results`), so a chunked prefill needs one
  `kv_start` scalar in the MMA kernel, chunk-sized `dst_tmp` and a single combine — no new reduction
  machinery.  Scope: **MMA-f16 only**, sparse/QSA and the tile kernel stay single-launch.  Remaining
  unknowns are the numerics re-baseline and the extra enqueue count at depth.
* **2026-10-27 (M2)** — the prototype is built and measured (`FINDINGS-2026-10-27-m2.md`): the ring + merge
  + finalize path is correct (single-chunk PPL identical to the baseline, 4 chunks -0.06 %) and takes the
  arena to O(chunk), but forcing the per-output-tile grid costs **2.8x pp** because the MMA kernel's work
  decomposition (`kbc = blockIdx.x*total/gridDim.x`) never reads `blockIdx.y/z` — i.e. MMA prefill is
  **stream-K-only**.  The mask-skip control (`GGML_CUDA_FA_MASK_SKIP=0`: ~0 %) rules out the other suspect.
  The remaining work is therefore a **kernel-protocol change**: give the stream-K path a per-row
  `(numerator, max, rowsum)` contract for *every* row (no-seam tiles currently write a final normalized
  value and no meta), plus `kv_start` so the group skip and `KV_max` keep working per chunk.  Estimate
  1-2 days with a prefill-logit re-baseline.  **Recommendation: outcome C (bound it) for this cycle**, keep
  the prototype for that follow-up.
