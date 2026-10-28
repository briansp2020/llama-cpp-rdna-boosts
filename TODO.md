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
(the 2026-10-25 re-base), release **`v15-6050b467c-r168`**.  `release.json` is the source of truth for the
tip/tree and hashes; the top of `WORKLOG.md` is the newest record.  **Block 14 is temporary** (item #104).

| # | subject | kind | status |
|---|---|---|---|
| **111** | a multi-slot verify step whose slot drafted NOTHING still splits | performance / accepted limitation | open (2026-10-27, r153/r154, issue #234).  `LLAMA_SPEC_ALIGN` equalizes the draft CAPS of the sequences one `draft-mtp-adaptive` call drafts, but a slot that drafted no tokens this round (the last step of a response, or a drafter that gave up) contributes its sampled row only, so the batch is `n_draft + 1` vs `1` and `split_equal()` still produces a second ubatch there.  Truncating the neighbour's good draft to nothing is the worse trade, so that step keeps splitting (measured: 1 such step per ~200 on the 35B-A3B host-expert arm).  **The cross-drafter case is CLOSED, not deferred**: the generic `min` cut of the finished drafts was shipped in r153 and measured -6 % on the reporter's own n-gram-first chain (65.6 -> 61.7 t/s, prose acceptance 74 % -> 50 %, splits 12 -> 1), so r154 removed it: a length mismatch caused by CONTENT rather than a cap is not worth equalizing.  `scripts/gate-spec-align.sh` now gates that chain (inert within 2 %, identical draft counts).  Record: `archive/work/spec-align-234/` |
| **112** | VRAM-resident slot snapshots for the server | parked idea | **parked** (2026-10-27, r157) — no work implied.  `LLAMA_STATE_SEQ_FLAGS_ON_DEVICE` now copies a sequence state at **14.7 / 17.2 ms** on 2 GPUs (was 2.2 s; `WORKLOG.md` 261027.009), so keeping a slot's snapshot **in VRAM** instead of host RAM is usable for the first time — but nothing calls the arm.  Blockers before a server could: no eviction/budget API on `mem_storage` and no `--fit` accounting, the MoE arena would be the OOM chain's sacrifice, and `llama-memory-recurrent.cpp` refuses the flag for more than one cell range (so a hybrid model's interleaved slot cannot be saved at all).  Detail in §112 |
| **115** | the pair fusion's MMB stand-down and the GLU matcher's MMB predicate disagree | performance / kernel selection | open (2026-10-28, r161, found fixing issue #241).  `mul_mat_q_pair` stands down when `ggml_cuda_mmb_routed_will_take()` says MMB will take both muls (a weak check: `mmb_routed_flag() && mmb_wtype_ok(type)`), while the later `{MUL_MAT_ID, MUL_MAT_ID, GLU}` matcher tests the real `ggml_cuda_mmb_supported_glu()` (which also needs `mmb_glu()`, `K % 64`, `T >= mmb_min_t()`, `n_used <= 64`, a plain unswapped SWIGLU and matching `dst` layout).  When they disagree the pair has already stood down and the MMQ triple takes the gate/up instead of MMB.  Measured neutral-to-positive on gfx1151 (prefill 5421/5425 vs 5397/5398 t/s on LFM2.5-8B-A1B Q6_K), so no change was made: this is a decision to measure, not a known defect.  Detail in §115 |

Accepted limitations that are deliberately **not** fixed are listed at the end — do not re-report them.

---

## 112. VRAM-resident slot snapshots — PARKED IDEA (not scheduled)

**Status: parked (2026-10-27, r157).**  Not a defect and not a plan.  The delivery's "switch away and come
back" story stays the host-side pair (the in-memory prompt cache and the context checkpoints), and r157 does
not change that: nothing in `llama-server`, `llama-cli` or `llama-bench` passes
`LLAMA_STATE_SEQ_FLAGS_ON_DEVICE`, so the arm is reached by `tests/test-llama-context` only.

**What r157 changed for this.**  That arm keeps a snapshot in a *device* buffer instead of handing the caller
a host blob, and it used to be unusable: 2.23 s to save / 2.28 s to restore a 213.6 MiB state on 2 GPUs
(`ggml_backend_tensor_copy`'s pageable bounce per piece), i.e. 58x slower than the host path it replaces.  It
is now 14.7 / 17.2 ms — faster than the host path's 21.9 / 38.9 ms — so the performance objection is gone and
what remains is a memory-placement decision.

**Where it would pay.**  A discrete GPU whose model leaves the state comfortable VRAM headroom (the 32 GiB
cards with a 4B-12B model, i.e. most of this box's own runs): the snapshot costs VRAM instead of host RAM and
the restore never materialises a host blob.  At 213 MiB that is ~7 ms / ~22 ms; at GB scale (a 100k-token
dense KV) both paths are bandwidth-bound and the argument becomes **residency, not latency**.  On UMA
(gfx1151) the latency win is a fraction of that — the pinned staging buffer is the same physical memory — but
a snapshot that lives in VRAM does not add to the non-swappable host footprint, which is the trade
`--host-experts pool` exists to manage.

**What is missing before a server could use it.**

1. `mem_storage` (`src/llama-context.h:508`) is a `std::map<seq_id, llama_memory_buffers>` with no erase
   path, no budget and no participation in VRAM accounting: one snapshot per seq_id, retained until the
   context is destroyed, invisible to `--fit`.
2. The MoE arena interaction: a snapshot allocation that fails reaches `ggml_cuda_device_malloc`'s OOM
   chain, whose last resort is freeing the expert arena — the wrong victim for a cache.
3. `llama-memory-recurrent.cpp:974` refuses `ON_DEVICE` when the recurrent cell ranges are more than one, so a
   hybrid/recurrent model's **interleaved** slot (the #210 lazy-idle-slot shape that `LLAMA_SERVER_PARK_IDLE_SLOTS`
   and `--kv-unified` produce) cannot be saved over this path at all.
4. `--kv-unified` / `seq_cp` sharing: "this slot's state" is not a cell range once two slots share cells (the
   checkpoint work had to answer the same question), and the one-stage-per-context rule means a save and a
   restore must not run concurrently.

**If it is ever picked up**, the oracle is `tests/test-llama-context`'s `cp_d` / `cp_d_s` for the copy itself
(`-sm tensor`, 2 GPU), then the server path it becomes — `scripts/gate-state-pipe.sh` and
`scripts/gate-ckpt-reuse.sh` are the closest existing models for "a restored state moves no token".  Records:
`archive/work/state-device-path/RESOLUTION.md`, `benchmarks/2026-10-27-state-device-path.md`.

---

## 115. The pair fusion's MMB stand-down and the GLU matcher's MMB predicate disagree — PERF/KERNEL SELECTION

**Status: open (2026-10-28, r161).  Found while fixing issue #241; measured neutral, so nothing was changed.**

Two matchers decide who serves a routed (MoE) gate/up/GLU prefill, and their MMB predicates are not the same:

* `ggml_cuda_mul_mat_q_pair_try_fuse` (mmq.cu) **stands down** when
  `ggml_cuda_mmb_active() && ggml_cuda_mmb_routed_will_take(w_gate) && ggml_cuda_mmb_routed_will_take(w_up)`.  That
  predicate is only `mmb_enabled() && mmb_routed_flag() && mmb_wtype_ok(w->type)`.
* The later `{MUL_MAT_ID, MUL_MAT_ID, GLU}` matcher (ggml-cuda-try-fuse.cu) asks
  `ggml_cuda_mmb_supported_glu()`, which additionally requires `mmb_glu()`, `gw->type == uw->type`,
  `K % 64 == 0` (and `K % 256 == 0` for the k-quant/IQ super-block types), `T >= mmb_min_t()`, `n_used <= 64`,
  a plain unswapped `SWIGLU`, and a `dst` layout of `[M, n_used, T, 1]`.

So whenever `mmb_routed_will_take` is true but `mmb_supported_glu` is false (a GLU variant MMB does not fuse, a
shape below `mmb_min_t()`, an expert count above 64, ...), the pair fusion has already stood down for MMB and the
MMQ triple then takes the gate/up: the two decisions are not consistent with each other.

**Why it matters (and did, once).**  This is the hole issue #241 fell through: on gfx1151 (`routed = 1`) the pair
stands down for MMB and MMB's fused GLU normally takes the triple, so the delivery's own arms never reached the
MMQ routed arm, and its dead `!ids` assert went unnoticed for five releases.  With that fixed, the mismatch is
only a kernel-selection question.

**Measured (gfx1151, LFM2.5-8B-A1B-i1-Q6_K, server, `cache_prompt:false`, 3 requests per arm):** fused MMQ triple
**5421 / 5425 t/s** prefill vs the MMB-individuals arm (`GGML_CUDA_DISABLE_MOE_MMQ_FUSION=1`) **5397 / 5398**, decode
identical (128.5-128.7 t/s), output byte-identical (`965 chars sha=d55bc8a8def9`).  No stand-down was added,
because on the one model that reaches the arm the MMQ path is not a loss.

**What to do when this is picked up.**  Make the stand-down ask the question the GLU matcher asks (the pair site
can see `i+2`, so it can evaluate the GLU predicate properly, or share one helper), then measure both arms on the
gfx1151 models that can reach it (an MMB-declined MoE: this LFM2, plus any shape at `T < mmb_min_t()` or
`n_used > 64`) and on the RDNA4 arm (`mmb_cfg().routed = 0` there, so the pair always wins and the change must be
inert).  A kernel-selection change of this kind owes the fused-vs-unfused text and prefill-logit check
(`GGML_CUDA_DISABLE_MOE_MMQ_FUSION=1` as the A/B), per `GREEDY-PURITY.md` §65.

---

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
