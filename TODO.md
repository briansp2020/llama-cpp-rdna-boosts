# rdna-boosts TODO / follow-up tracker

Cross-project tracker so important state survives context compaction.

**Forward-looking only.**  This file lists open work, items waiting on someone else, accepted limitations
and parked ideas.  **A closed or resolved item is not kept here** — it is one dated entry in
[`WORKLOG.md`](WORKLOG.md) plus the campaign record that entry points to (`archive/work/<campaign>/`).
Details never live here: they live in `AGENTS.md`, `patches/README.md`, `MANIFESTS.md`, `WORKLOG.md`,
`GREEDY-PURITY.md`, `ENVIRONMENT.md`, `wip/*` and `benchmarks/`.

**Numbers are never reused.**  `#NN` is a permanent handle: an old `WORKLOG.md` entry that cites `TODO #74`
still means the same subject after that item is closed and removed from this file.  New items append.

**Current state:** the **15-block V2.1 set** (block 00 + blocks 01-14) at fork point **`6050b467c`**
(the 2026-10-25 re-base), release **`v15-6050b467c-r150`**.  `release.json` is the source of truth for the
tip/tree and hashes; the top of `WORKLOG.md` is the newest record.  **Block 14 is temporary** (item #104).

| # | subject | kind | status |
|---|---|---|---|
| **102** | the band's width purity has no head-256 **GQA-8** arm | platform validation | open, accepted limitation (r134, issue #224).  The r134 guard is width-only (`np`, `nbatch_fa/K2/V2`, `nstages_target`, `Q_in_reg` on the two band rows), and it is measured at **GQA 6** (dense 27B: `test-logits-width-probe P=4096`, maxdiff 0) and **GQA 12** (Flash-Next `RS=from_w`, 2 GPU `-sm layer -ncmoe 40`, q8_0, maxdiff 0); no head-256 GQA-8 model exists in the fleet.  What to run if one appears: the same probe at `P >= 4096` (`KV=q8_0/f16`) plus `plain == draft-*` at `-n 512` |
| **103** | the ~30x collapse at the last few MiB of free VRAM | diagnosis | open (r135, found in issue #219).  At 14.06 GiB free the same 27B + MTP config runs prefill 35 t/s / decode 1.5-3 t/s instead of 1127 / 53 with the weights still resident; not the FA staging fallback and not graph churn alone.  `GGML_CUDA_MIN_FREE_MIB` makes it unreachable; the cause is unrecorded.  Detail in `TODO.md` §103 |
| **79** | RDNA3_0 verify-band fusions: no multi-device measurement, and gfx1101/gfx1102 untested | platform validation | open (shipped default-on in r95; needs a 2+ card gfx1100 box).  `ssh fingon` is single-RDNA3 (`ROCm0`; its second rocm-smi entry is an iGPU and is not a HIP device), so it does **not** cover the multi-device arm — it validates single-GPU gfx1100 only.  gfx1101/gfx1102 still have no hardware |
| **70** | harden the expert-cache re-arm / wide-aux growth against a starved arena | robustness | open |
| **104** | block 14 (`-sm relaxed-tensor`) is a stop-gap, not a feature | temporary upstream modification | open, transient (r137).  It exists so `flint`/`mellum`/`mistral4` can tensor-split while upstream #30861/#31081 leave the MoE amax row max unsplittable.  **Drop the block** when upstream fixes the amax under a tensor split (a real cross-device MAX all-reduce, or the `ggml_prec_set_src(..., GGML_PREC_BF16, 1)` path the graph TODO names) or when its relaxations stop being needed; nothing else depends on it.  Keep it separate, do not let it grow into general `-sm tensor` support.  Limitations: `patches/README.md` (block 14), `GREEDY-PURITY.md` §58 |
| **106** | issue #214's RDNA2 deep-context prefill residual | platform validation (waiting on the reporter) | open (2026-10-25, r138/r139).  On the reporter's 12 GiB gfx1031 (RDNA2 tile FA, slab declined) the delivery's `pp512` lead over upstream shrinks with depth: +36 % at d0, about even at 80k, -10 % at 120k (his r126 sweep; the r138 re-base does not change his tile path).  Not measurable here: no gfx1030/gfx1031 device.  Candidates are both block 05: the RDNA2 tile table from PR #165/r76 (tuned on a V620 with a different GQA shape), `GGML_CUDA_FATTN_TILE_Q_GLOBAL` (the +30 % figure was shallow), and the `prefill_stages = !RDNA3_5` staging gate.  Next measurement: his r138 `-p 512 -n 0` sweep with `GGML_CUDA_FA_STAGE_MAX_MB=1` as the staging arm.  Detail in §106 |
| **108** | parking a mirroring second cache is validated on qwen4exp (and its k-pool arm is not) | platform validation | open, accepted limitation (r146, issue #227).  `llama_memory_hybrid_idx` is shared with **glm5-next**, whose indexer is the `indexer_kpool > 0` (k-pool, `indexer_kpool_by_order`) variant: the same park code moves it and the same invalidation (`mem_idx_stale_set`, `pool_invalidate_all`) covers the pool layout, but no glm5-next GGUF exists on this fleet, so only the qwen4exp arm is measured (`scripts/gate-park-idle-slots.sh`) |
| **109** | a replayed graph loses to the direct node loop where the RCCL comm layer is initialised | performance | open (2026-10-26, r149).  Now that decode graphs replay (TODO #107 closed) the captured path loses on **3-device `-sm tensor` qwen35 dense plain decode**: 27B Q8_0 `llama-bench tg128` **41.7 -> 38.6** (Q4_K_XL 52.4 -> 49.1, unchanged at depth 16384).  **The axis is the comm layer, not the model's shape and not the per-call algorithm choice:** with `GGML_CUDA_ALLREDUCE=internal` (RCCL **never initialised**) the same arm is **42.2 (+0.9 %)** on the replaying build, while `hybrid` (41.7 -> 38.6) and `ce` (40.7 -> 38.6, and `ce` runs `comm_init_hybrid` first) both regress, and `nccl` is worse still (38.5 -> 35.6).  `GGML_CUDA_AR_SMALL_BYTES` 0 or 1e9 changes nothing, so it is not which arm runs per call: it is **RCCL's presence**, which the eager-init comment in `ggml_backend_cuda_comm_init_hybrid` already records as an r34 decode-only cost of the internal pipeline — r149 makes it visible on this arm because replay now happens at all.  Not capture churn (one shot per key, the deficit survives `-n 512`), not `GGML_CUDA_GRAPH_CHUNK_SIZE`, not `GGML_CUDA_ASYNC_INPUT`, not `GGML_SCHED_EVENTS`.  **It is not a dense-model property and not a 3-device property:** Gemma4 31B Q8_0 (dense, 32.6 GB) is **+0.8 %** at 2 GPUs (26.64 -> 26.86) and **flat** at 3 (34.68 -> 34.58, and only -0.9 % for `internal` vs `hybrid`), 27B at 2 GPUs is +0.5 % with replay working, and a `draft-mtp` draft hides it at 3 devices (42.8 -> 42.9).  Kill switch: `GGML_CUDA_GRAPH_KEY_WEIGHT=0`.  Next: find what RCCL's init leaves behind that costs a *replayed* graph 5-7 % while the direct node loop is immune (an r34 follow-up, not a r149 regression) |
| **110** | the meta backend's out-of-graph views are per chunk, not per tensor | performance / robustness | open (2026-10-26, found in issue #232).  `llama_io_read_device` registers one simple-tensor view per KV cache **chunk** per layer on a state restore (and `llama_kv_cache::park_move` one per move chunk); the container now grows so this is harmless, but a 140k-token restore alone is ~1.6k simple tensors and ~0.6 MiB of host metadata per device, where one view per tensor plus a chunked copy would be ~24.  A llama-context copy-path change with its own risk (the chunked path exists for differing save/restore chunking), so it is parked until it can be measured against the checkpoint/prompt-cache gates.  Record: `WORKLOG.md` 261026.008 |
Accepted limitations that are deliberately **not** fixed are listed at the end — do not re-report them.

---

## 106. Issue #214's RDNA2 deep-context prefill residual — PLATFORM VALIDATION (waiting on the reporter)

**Status: open (2026-10-25, r138/r139).**  On the 12 GiB RX 6700 XT (gfx1031 as gfx1030, RDNA2 tile FA,
`-ncmoe 24`, slab declined so the experts stream) the reporter's r126 depth sweep shows the delivery's
`pp512` lead over upstream shrinking with depth: +36 % at d0, about even at 80k, -10 % at 120k.  r126 cannot
be the cause (his card has no WMMA and takes the tile kernel, so the r126 RDNA4 row never applied to him),
and the r125 campaign's gfx1201 streaming arm leads at every depth.  The remaining candidates are block 05's
RDNA2 tile config table (PR #165/r76, tuned on a V620 with a different GQA shape), the
`GGML_CUDA_FATTN_TILE_Q_GLOBAL` wide-tile Q-from-global read (its +30 % was a shallow measurement), and the
`prefill_stages = !RDNA3_5` FA staging gate (RDNA2 stages; it was never depth-swept).  A near-full card adds
the staging-arena refusal: his peak is 12204 of 12272 MiB, so the deep-prefill staging allocation can be
refused and falls back to the native K/V read, which is a slowdown but is also upstream's path.

This fleet cannot measure it: there is no gfx1030/gfx1031 device, and the only RDNA2 silicon is a gfx1036
iGPU with a 2 GiB BAR and ~1.8 GiB already used by the desktop.  The reporter's r138 sweep, with
`GGML_CUDA_FA_STAGE_MAX_MB=1` (force the native K/V read) as the staging arm, is what separates the staging
gate from the tile table.  The r138 gfx1201 control (no crossover, resident +18..+27 %, streaming +18..+45 %
with default flags) is [`benchmarks/2026-10-25-r138-depth-sweep.md`](benchmarks/2026-10-25-r138-depth-sweep.md).

## 103. The ~30x collapse when the last few MiB of VRAM are in use has no root cause — PERFORMANCE (diagnosis)

**Status: open, not understood (r135, found while investigating issue #219).**  On a single gfx1201 with the
27B GSQ-RCO-IQ3_S + mmproj + MTP at `-c 65536`, reducing the free VRAM at process start from 14.46 GiB to
14.06 GiB turns a 13 s request into a 384 s one (prefill 1127 -> 35 t/s, decode 53 -> 1.5-3 t/s) with
**zero** device memory left, and the next larger prompt then aborts with `ROCm error: out of memory`.
Eliminated so far: the weights are still resident (`rocm-smi --showpids` shows 15.08 GiB for the process);
it is **not** the FA staging stand-down (`GGML_CUDA_FA_STAGE_MAX_MB=0` on a roomy card is still 13.3 s); it
is **not** graph churn alone (`graphs reused` is 0 in one arm and 65 in another, both slow); the workspace
pool's flush + retry and the ECC/pool-floor logic account for a fraction, not 30x.  Next step: `-lv 5` on a
short (2-3 K token) prompt at that budget with the per-op pool debug (`GGML_CUDA_GCDBG`) and a GPU
busy/clock sample, to separate "the GPU is doing 30x the work" from "the host is stalling 30x".

Harmless on Linux (it terminates or degrades) and unreachable with `GGML_CUDA_MIN_FREE_MIB` set, but it is
the state the Windows reporter's card spends its life in, so it is worth understanding.

## 102. The band's width purity has no head-256 GQA-8 model arm — PLATFORM VALIDATION

**Status: open, accepted limitation (r134, issue #224).**  Not a suspected defect.  The r134 fix restores the
band's per-column KV split (`np = nwarps*cols_per_warp/ncols`) as one geometry across query widths 1..8, and
a `static_assert` on the two band rows makes a config divergence a build error.  `ncols2 = 8` folds a GQA
group of up to 8 in one pass (and 9..12 in two), so the geometry does not depend on the ratio; the guard is
width-only.  Measured: `test-logits-width-probe` at `P=4096` on **GQA 6** (dense 27B, q8_0/f16/bf16) and on
**GQA 12** (Flash-Next, `RS=from_w`, 2 GPU `-sm layer`, `-ncmoe 40`, q8_0) — both maxdiff 0.  No head-256
GQA-8 model exists in the fleet, so ratio 8 itself is unmeasured.

What to run if one appears: the probe at `P >= 4096` with `KV=q8_0` and `f16` (plus `SPLIT=tensor` for the
multi-device arm), and `plain == draft-dflash/draft-mtp` at `-n 512`.  A failure would look like a row-0 hash
split between W=4 and W=5 (the shape the r126 geometry produced).

## 79. RDNA3_0 verify-band fusions: the multi-device arm and gfx1101/gfx1102 are unmeasured — PLATFORM VALIDATION


**Status: open, accepted limitation.  Shipped default-on in r95 on the maintainer's decision**
(`WORKLOG.md` 261020.006, record `archive/work/rdna3-verify-fusions/RESOLUTION.md`).  Not a suspected
defect: the change is arch-identical code behind a host predicate and every single-GPU gate is green
(gfx1100 `W=1..8` maxdiff 0 on six model/prefix cells, `none == draft-mtp`, same-binary kill-switch
identity, prefill-logit KLD 0.000000).

Two gaps, both needing hardware the fleet does not have:

1. **RDNA3_0 multi-device.**  The batched f32 `GGML_OP_CPY` fusion is **not** single-device restricted and
   it is an actively used multi-device path: on 3x gfx1201 with the GDN model + MTP,
   `GGML_CUDA_FUSE_CPY_BATCH_DEBUG=1` shows **450 runs under `-sm layer` and 1350 under `-sm tensor`**.
   The RDNA4 multi-device path is therefore heavily exercised, but the same code on a 2+ card gfx1100
   box was never run.  What to run: a GDN model with MTP (`draft-mtp`, n-max 3) under both `-sm tensor`
   and `-sm layer`, plus `test-logits-width-probe` with `SPLIT=tensor`, comparing default against
   `GGML_CUDA_FUSE_CPY_BATCH=0`.  `fingon` is one gfx1100 plus a gfx1036 iGPU and `halo` is gfx1151, so
   neither is a valid host.  Community members running multiple gfx1100 cards are expected to report.
   If a multi-device failure does appear, `GGML_CUDA_FUSE_CPY_BATCH=0` is the immediate workaround and
   restricting that fusion to `get_device_count() <= 1` on RDNA3_0 is the local fix.
2. **gfx1101 / gfx1102.**  `GGML_CUDA_CC_IS_RDNA3_0` covers `0x1100 <= cc < 0x1150`, so gfx1101 (RX
   7800 XT / 7700) and gfx1102 (RX 7600) take the same fusions, but neither the PR author nor the r95
   release ran them.  There is no such card in the fleet.  The same `W=1..8` probe and kill-switch A/B
   is the check if one becomes available.

**Do not re-report as a bug without one of those runs.**

---

## 70. Harden the expert-cache re-arm / wide-aux growth against a starved arena — ROBUSTNESS


Opened 2026-10-19 while fixing r87.  r87 forces the post-prefill drop, so the r86 trigger (the wide layout
was never released and the arena decayed) is gone.  But the failure shape was: a re-arm that had to shrink
one layer to a partial slot count (`layer=47 ... 344 requested, only 202 fit`), a starved wide-aux region
(thousands of `SLAB_WIDE_AUX` attempts), then a GPU page fault in `k_get_rows_float` on the next graph.
The re-arm is supposed to be layer-uniform and safe; make a partial/refused re-arm leave the cache
**wholesale stood down** (the fuse-guard invariant) instead of running a mixed-slot layer, and make the
wide-aux growth refuse once per pass instead of once per allocation.  Repro: the 10-prompt series against
r86 in `WORKLOG.md` 261019.002; validation arm `MOE_EXPERT_CACHE_VALIDATE=1`.

**r94 note:** the issue #187 fix adds the missing *graph* half of this stale-pointer family — a captured
graph is now invalidated when the slab work boundary shrinks (block 10, `WORKLOG.md` 261020.004).  That
covers the observed #187 fault; it does **not** discharge this item, because a re-arm can still change a
table's arena placement under a graph that is not being dropped.  Keep #70 open.

## Documented, deliberately NOT fixed (accepted limitations — do not re-report)

* **Parking a memory whose second cache mirrors the attention cells is validated on qwen4exp only** (TODO
  #108, r146): `llama_memory_hybrid_idx` also serves glm5-next, whose indexer is the k-pool
  (`indexer_kpool > 0`) variant.  The park code, its mirror check and its invalidation are arch-agnostic
  and the k-pool layout is invalidated with everything else, but no glm5-next GGUF exists on this fleet, so
  only the qwen4exp arm is measured.  `llama_kv_cache_dsa` / `_iswa` (deepseek4, glm-dsa) report parking
  unsupported by design and keep #210 — they never half-park.

* **A dense `-sm tensor` split is not guaranteed to produce the same greedy text across device counts**
  (AGENTS "Greedy text is a per-device-count reference" was corrected 2026-10-24).  Qwen3.8-27B
  UD-Q4_K_XL, prose prompt, seed 42: 3 GPU 850 chars vs 2 GPU 876; the r126 build shows the same
  (897 vs 855), so it is the tensor split's rounding class, not a delivery defect.  Every `-sm layer`
  arm *is* text-identical across counts.  Gate a change within a device count.


- **A Windows VRAM overcommit is not catchable: it surfaces as a device loss, not an allocation failure
  (issue #216, accepted; nothing in the delivery can change it).**  On WDDM an allocation set larger than
  the card does not fail, it pages into system memory; the first big prefill (a full `-ub` compute buffer
  plus the FA staging) then loses the device, and the process's *next* rocBLAS touch aborts with
  `rocBLAS error: Could not initialize Tensile host: No devices found` (rocBLAS's `TensileHost` prints
  that when `hipGetDeviceCount()` returns non-success; it allocates nothing on that path, so the message
  is a symptom, not an OOM inside rocBLAS).  Measured budget for the reported config (dense 27B IQ3_S,
  `-c 131072 -ctk q4_0 -ctv q4_0 -b 4096 -ub 1024` + MTP n3, no mmproj): **14.8 GiB** of static device
  memory and a **16.0-16.2 GiB** first-prefill peak on a headless 32 GiB card, against a 16 GiB card that
  also drives the display.  Mitigations shipped/to use: the draft KV follows the main KV type (r125,
  block 02, -368 MiB at `-c 131072`), `-c`, `-ub`, a real `-ngl` offload (`-ngl` is a **layer count** and
  a NextN model has `n_layer + 1` layers, so `-ngl 85` on a 65-layer model is a no-op), and letting
  `--fit` size the context (it declines when `-ngl` or `-c` is set).  Do not re-investigate the rocBLAS
  message as a delivery defect without a Windows event log showing a TDR (`4101`/`4102`) *absent* at the
  abort.

- **`std::min<int64_t>` / `std::max<int64_t>` in other HIP TUs is a latent Windows-only build failure
  (TODO #95, accepted; no Windows toolchain here).**  With HIP on Windows (clang + the MSVC STL) clang's
  force-included `cuda_wrappers/algorithm` and the MSVC STL `<utility>` each define an inline
  `std::min<long long>` under **one** mangled name (the `enable_if` that makes clang's win overload
  resolution is not part of the MSVC mangling), so a translation unit that instantiates both fails to
  codegen with *"definition with same mangled name"* — r123's `moe-expert-cache.cu` was the first to
  trigger it (issue #217, fixed in r124 with the local ternary helpers; see `patches/README.md` block 09).
  The reporter's toolchain builds every other TU, but ~10 of them still carry the int64_t pattern
  (`argmax.cu`, `cpy-batch.cu`, `fattn.cu`, `mmvf.cu`, `fattn-common.cuh`, `gated_delta_net.cu`, `top-k.cu`,
  `ssm-scan.cu`, `ggml-cuda.cu`, `hc-mix.cu`): the collision only fires where the STL's own instantiation is
  emitted too, which is a header-instantiation accident, not a property of the call.  Not proactively
  rewritten because a non-AMD/Windows claim cannot be validated on this box (scope policy: non-AMD backends
  are out of validation scope); if a Windows build log names one of them, apply the same two-line helper
  fix there and bump the release.  Do not re-report as a new defect without a Windows build log.

- **The Meta split-state lookup-count headroom (TODO #80) — retired 2026-10-24, accepted.**  The measured
  defect (issue #184: the pre-warm walk allocating one heap node per visited tensor on every graph
  re-allocation) was fixed in r100 (block 09, `GGML_META_SS_SCRATCH`, default on).  What remained is the
  reporter's ~165k cache hits per re-allocation (30+ per node), and the only lever is the lookup **count**
  (the recursion/pruning structure of `calculate_split_state` and the pre-warm walk) — hashing the cached
  336-byte `ggml_tensor` copy to skip the `memcmp` costs about what the `memcmp` costs.  Not pursued because
  the frequency that makes it worth anything is his workload: it re-allocates **~230** graphs per run, while
  the delivery's own validated shapes re-allocate **3** in a whole 4-stream x 192-token Flash-Next IQ3_XXS
  + MTP window (`-sm tensor -ncmoe 47`, 63 graph reuses per stream; 4B Q8_0 2-GPU tensor: 12 in 4 x 128),
  so even at his own post-r100 16.0 ms per re-allocation the whole walk is at most ~1 % of a 4.2 s window
  and the lookup-count share of that is below this box's noise floor (r100's entire heap-churn removal
  measured +0.29 % on this shape).  Issue #184 is closed (2026-10-21) with the reporter confirming r100 in
  his production build (+1.6 % 4-stream decode, identical greedy output), and the walk's code is unchanged
  since r100.  A future change there still owes the r100 proof set (identical states under
  `GGML_META_SS_VERIFY=1`, identical greedy text and prefill KLD, same-binary A/B with
  `GGML_META_SS_SCRATCH=0`).  Records: `archive/work/issue-184-meta-r80r82/RESOLUTION.md` (addendum) +
  `probe-reallocs.py`, `WORKLOG.md` 261024.003.
- **`mmf32` covers the skinny F32 shapes only (r104, `TODO` #82 closed) — accepted.**  A full-F32 model's FFN
  GEMMs (`m` ≫ 256: dense F32/F16-weights-heavy prefill) still fall through to rocBLAS/hipBLASLt, so they keep
  that path's per-start solution lottery; the kernel's tile (32 output rows) is the wrong shape to be
  competitive there and no such model is validated.  Also untouched: BF16/f16 prefill (Flash-Next keeps both
  hipBLASLt's +51 % and its own lottery — a BF16 counterpart is its own job) and **RDNA3** (gfx1100), where the
  kernel stays opt-in (`GGML_CUDA_MMF32=1`) because it measured −1.3…−2.2 % at `pp2048` there; the lever would be
  a per-arch tile, not the gate.  RDNA3_5 was admitted and is default-on since r105 (`halo`: 12/12
  deterministic, +1.0 %/+1.6 % prefill, family-switch PASS).  Scope: `patches/README.md` (block 06), rule:
  `GREEDY-PURITY.md` §47.
- **The `launch-ledger` remainder (item 5(c)) — measured, not pursued (2026-09-12 (8)).**  The small-pp
  remainder (+38 `scale_f32`/eval, an `rms_norm<256,true>` count diff) is sub-0.2 %, root-cause-only.
- **The mmq mma `sum[]` accumulator-overflow latent defect (item 5(d)) — accepted, no upstream report
  (2026-09-12 (8)).**  `process_tile` sizes the per-thread accumulator as `J*I/(nwarps*32)` while the
  AMD-WMMA vec_dot indexes up to `J/2-1`, so any config with `I < nwarps*16` silently corrupts
  (deterministic for J=128, racy for J=48/24).  Root cause + full evidence matrix:
  `archive/work/wip-archive/qwen4exp/discovery/2026-09-06-strix-halo-gfx1151-mmq-j128-latent-defect.md`.
  Upstream ships **no** violating config (every `mmq-config-*.cuh` row keeps `I >= nwarps*16`) and the
  block-13 rows never violate it either, so there is no upstream reproducer to file.
- **V3 prefill cost is arch-dependent (item 5(g)) — accepted.**  gfx1151 measured −3.2 % at pp20480 (4B,
  q8_0) vs the RDNA4 reference −1.3 %, decode flat; still a large net win (−799 MiB compute + −799 MiB
  host) and on by default.
- **Upstream monitor: ROCm unaligned-width split-load (item 13) — standing, no action (2026-09-12 (8)).**
  Fixed locally in block 13; no PR planned (upstream is busy with its own qwen4exp work).  It resolves
  naturally as a re-base conflict if upstream fixes it; nothing to track.
- **Mixed K/V cache types are HARD-REJECTED.**  Every mixed pair (`bf16`+`q8_0`, `f16`+`q8_0`) measured
  1.7-3.6× slower and never smaller; `params.type_k != params.type_v` fails `llama_init_from_model` with
  a message naming both types (block-14 amendment; upstream's MLA/DeepSeek4-only condition is dropped).
  Both types default to f16, so only an explicit `-ctk`/`-ctv` can trigger it.
- **gemma-4-E4B-it + 3-GPU `-sm tensor`** aborts in the meta splitter because its 2 KV heads are fewer
  than the 3 devices (one device gets a zero-extent share: `[1, 1, 0]` over 2 heads).  The abort is the
  split-state **ratio-consistency** assert in `ggml_backend_meta_get_split_state`
  (`split_state.ne[j]*split_state.nr[0] * tensor->src[i]->ne[src_ss[i].axis] == sum * tensor->ne[split_state.axis]`
  — `ggml-backend-meta.cpp`; 1582 at r101, 1487 at r99: the line number moves, the assert does not, and
  older records that cite `:1177` are one more tree back).  Works on 1/2 GPUs and on 3 GPUs with
  `-sm layer`; maintainer's call: no fix.  r101's split-state work deliberately does not change this
  (`GREEDY-PURITY.md` §45 accepts a slice that is *empty or the whole* extent, and one head of two is
  neither); re-verified on r99 and r101, identical abort.  Every other model is unaffected (a future
  block or upstream report could make the splitter tolerate a zero-extent share).
- **`src/llama-kv-cache.h:274` `-Wunused-private-field` for `v_enabled`** on a full build (the field *is*
  used, in `llama-kv-cache.cpp:232`; clang's per-TU analysis fires).  A `[[maybe_unused]]` one-liner
  silences it; left alone to keep the V5 amendment scoped to the FA kernels.
- **Gemma4's full-attention (head_dim 512) layers stay outside the rope + FWHT + KV-write fusion**
  (documented 2026-10-21, deliberate).  The fused kernel instantiates `fwht_n` in `{64, 128, 256}`, and
  Gemma4 is the only arch with a distinct SWA/full head dim (`key_length_swa` 256 vs `key_length` 512),
  so its 1/6 full-attention layers run `rms_norm_mul_rope_f32<256,true,float,0,false>` + `fwht_cuda<512>`
  for Q and K plus two `set_rows` — 6 launches instead of 2.  Extending it needs 4 new kernel
  instantiations **and** a fix to the fused kernel's V-row write (its `ncols <= block_size` precondition
  would quantize half a 512-wide V row), which edits the shipping 256 fold, for ~0.1 % on gfx1201.
  Accepted as a gap; the analysis (what to change, why the arithmetic is bit-identical by construction)
  is in [`archive/work/fix-gemma-4/ISWA-WRITE-FUSION.md`](archive/work/fix-gemma-4/ISWA-WRITE-FUSION.md) §5.6.
  (The iSWA builder order that made the 256-wide half of this fusion live is r101, `WORKLOG.md` 261021.003;
  `GREEDY-PURITY.md` §45 records the split-state rule that came with it.)
- **3-GPU plain (non-speculative) decode is a few percent slower than 2-GPU — accepted (this closes
  `TODO #51`; 2026-10-23, `archive/work/three-gpu-performance/RESOLUTION.md`).**  −2.0 % with a fully-resident
  arena (the 3-device internal all-reduce is 1.2 % of decode and no backend/size knob beats it), −10.5 % at
  91 % residency (the non-identity per-token routing readback, paid per device; the *marginal*-arena case is
  fixed in r117, an 8 %-short one is not), −10.2 % with
  `-sm layer` (no intra-token parallelism; a property of layer splitting, reproduced on dense 27B).
  Speculative decode does **not** lose: +14…+23 % at 3 devices when resident.  Production rule: 3 GPUs for
  speculative/MTP workloads, 2 GPUs for plain decode at partial residency, never `-sm layer` for decode.
- **Not worth pursuing** (measured, no win): the decode fq-inline-quantize port (wash-to-negative — the
  tree already launches fewer kernels/step and sits at wall parity) and the GDN +72-launch 2-kernel split
  (cosmetic).
- **`--spec-draft-mtp-vocab` two-band subset (closed 2026-10-22, measured, not shipped).**  A second band
  (keep `< N` **and** `>= n_vocab - M`) was meant to catch gemma4's space-piece band (229376..255999): it
  measured 0.610 acceptance at 37 % of the rows and was still 0.4 % *slower* on that payload.  With the
  width rule found in the 3-GPU arm (`N · n_devices < n_vocab`) a second band only makes the **mirrored**
  draft matmul wider, so it is a non-goal, not a parked idea.  Record:
  `archive/work/spec-draft-mtp-vocab/VALIDATION-2026-10-22.md` §5.

## Where the current lists live


- **Open campaigns and scoping notes:** [`wip/CAMPAIGNS.md`](wip/CAMPAIGNS.md) (the index for `wip/`).
- **Closed campaigns, with their evidence:** `archive/work/<campaign>/` (`RESOLUTION.md` where the campaign
  closed with a finding).
- **Remaining gfx1151 work + the 2026-09-12 audit:** `archive/work/strix-halo/HANDOVER-2026-09-12-remaining-gfx1151.md`.
- **QSA sparse-regime width purity (items 4/7 disposition):** `archive/work/strix-halo/RECORD-2026-09-12-qsa-sparse-width.md`.
- **Environment, instruments, reference hashes and the landing procedure for KV/FA work:**
  `archive/work/kv-quant-purity-followups/HANDOVER-2026-09-11-remaining-work.md` (its §0 status and its
  items 1/5 and F3 are **superseded** — see `WORKLOG.md`).
- **Delivery verification contract + dated records:** `MANIFESTS.md`, `patches/README.md`, `WORKLOG.md`,
  `AGENTS.md` headers.
- **Purity/invariant analysis and the instrument rules:** `GREEDY-PURITY.md`.
- **Benchmarks + gates:** `benchmarks/` (the adaptive-MTP baseline gate: `mtp-adaptive-methodology.md`).
- **Upstream PR candidates:** `upstream/README.md`.
