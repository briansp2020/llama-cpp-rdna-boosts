# Open WIP campaigns (index)

This is the index `AGENTS.md` refers to.  `wip/` holds **active exploration only** — nothing here is part
of the delivery, and nothing here may be applied to the fork without the maintainer's explicit go-ahead
(see the WIP and promotion rules in `AGENTS.md`).  Each campaign is a self-contained handover under its own
`README.md`.

| directory | what | status |
|---|---|---|
| [`fit-slab-accounting/`](fit-slab-accounting/README.md) | right-size every slab-active VRAM consumer (slab reserve/headroom, arena, H2D staging ring) so no `-ncmoe` crashes on a shortfall, with graceful degradation + guidance; reframed around runtime robustness rather than `--fit` only | **REVIVED / IMPLEMENTED (WIP)** (2026-10-09) — first pass landed as [`revival-2026-10-09.patch`](fit-slab-accounting/revival-2026-10-09.patch): **G3** adaptive slab reserve (`headroom + draft aux`, KV is pre-slab), **G4** free-VRAM cap on the H2D ring + FA staging, a **split-slice SIGSEGV fix**, and **G6** keeps the speculative (nextn/MTP) head weights on-GPU regardless of `-ncmoe` (host speculation verified byte-identical).  Phase 1 (G1/G2 fit-side) stays parked (the `-sm tensor` auto-floor corrupts); promotion blocked on the GSQ `/`-corruption band re-test.  **NEW G7 correctness bug (needs its own session):** the expert cache corrupts a wide MTP-export consumer (Flash-Next draft acceptance 0 + NaN logits); `MOE_EXPERT_CACHE_MIB=0` is correct — README §13, TODO #48.  **G1/G2 RE-OPENED for retest** after the r37 `-sm tensor` changes (auto floor still dead under tensor split) — README §14, TODO #47.  See README §12-14 |
| [`cache-split-admission/`](cache-split-admission/README.md) | expert-cache admission policy on `-sm tensor` split MoE tables (`MOE_EXPERT_CACHE_DEVPOLICY_SPLIT`); today the split tables use the slow host promotion | **OPEN / measured** — re-measured 2026-10-08 on r34 (bounded 2048 MiB arena; at the default AUTO the arena is fully resident and the choice is moot): device policy wins big in plain decode (35B `-sm tensor` 10.2 -> **36.5** t/s), byte-identical, but MTP n3 still favours host promotion (3-GPU `MOE_EXPERT_CACHE_MIB=16384`: **76.5** vs 55.6 t/s), so a blanket default flip would regress MTP |
| [`fp8-support/`](fp8-support/README.md) | native FP8 E4M3 for RDNA4 | PARKED |
| [`host-memory-footprint/`](host-memory-footprint/README.md) | host-memory footprint of GPU-resident weights (gfx1100) | open |
| [`mmvq-verify-rows/`](mmvq-verify-rows/README.md) | faster multi-token mmvq on RDNA4 (bit-exact) | open |
| [`moe-cpu-overlap/`](moe-cpu-overlap/README.md) | genuine CPU/GPU overlap for the expert misses (Strata shape) | OPEN / scoping |
| [`nwarps/`](nwarps/README.md) | per-M `nwarps` MoE candidate — the one deliberate width-purity impurity | ACTIVE (env-OFF) |
| [`strata-amd-kernels/`](strata-amd-kernels/README.md) | compare Strata's AMD decode kernels against block-10/13/15 | OPEN / scoping |

**Closed campaigns** live in `archive/work/`; each left a redirect stub here so historical pointers resolve.
Recently closed:

* [`archive/work/host-expert-dio-cache/`](../archive/work/host-expert-dio-cache/README.md) - the bounded pinned host-expert pool, **folded into block 06 in r35** (`--host-experts pool`, an **option**; the default stays `pinned`).  It sizes to `MOE_HOST_POOL_MIB` (auto = 25 % of the MoE host expert bytes) and runs the device admission policy with a pool-sourced fill; the routing prefill is default-off.  Also folded: the parked slab peer-access fix (block 06) and the `mul_mat_vec_q_moe` row-tail clamp (block 13); the block-12 lazy-NCCL all-reduce bug was found en route and delivered in r34.

* [`archive/work/moe-cache-band16/`](../archive/work/moe-cache-band16/README.md) - PR #115's expert-cache band widening (the decode/verify band now follows the routed-expert MMVQ band, 16 tokens on RDNA4, via a single-owner `moe_cache_band` hook) plus the device-side admission fill-list fix, **folded into block 06 in r31** behind `GGML_MOE_CACHE_MAX_TOK=8`.  On qwen35moe `-ncmoe 40`: the 16-token MoE batch 123.7 -> 702.5 t/s, `draft-mtp n-max 12` 33.9 -> 110.7 t/s, `n_max <= 7` byte-identical and `n-max 8` cache text bit-identical to all-VRAM.

* [`archive/work/gfx12-gdn-accuracy/`](../archive/work/gfx12-gdn-accuracy/README.md) - the gfx1201 (RDNA4) bf16 chunked-GDN accuracy-parity campaign.  **Refuted:** the gfx12 kernel has per-op parity (equal op NMSE, identical layer-0 real-data error, marginally tighter WMMA), is tiling-invariant, and an fp16-operand variant that is 65x more accurate per op still moves the model KLD only 1.5x.  The residual gfx1201-vs-gfx11 KLD is model-level numerical sensitivity, not a GDN defect.  Only artifact: a **test coverage fix folded into block 02** (cache-fusion bf16 gate + the model's exact op shape), pending the next release; no kernel change.

* [`archive/work/gdn-bf16-audit/`](../archive/work/gdn-bf16-audit/README.md) - issue #113's BF16 chunked-GDN prefill KLD: the r29 divergence was an `A_sc` stride aliasing bug for `n_seqs > 1` (not bf16 precision).  **Fixed in r30** (block 02), the bf16 default is back on, and the op test's gates are now realistic with a 1e-4 bf16 tolerance.
* [`archive/work/rdna4-qwen4exp-decode-fusions/`](../archive/work/rdna4-qwen4exp-decode-fusions/README.md) - PR #114's four bit-identical qwen4exp decode fusions (latency-scheduled BF16 `hc_mix` up/collapse, `HC_COMBINE` folded into that norm, the shared-expert `sigmoid`-`mul`-`add` gate, and the GDN `beta` sigmoid), **folded into block 15 in r27** behind four default-on `=0` switches (`GGML_HC_UP_V2`, `GGML_CUDA_FUSE_HC_COMBINE_MIX`, `GGML_CUDA_FUSE_SIGMOID_MUL_ADD`, `GGML_CUDA_FUSE_GDN_BETA_SIGMOID`); +2.4 % decode with byte-identical text.
* [`archive/work/r26-rdna4-dpp-butterflies/`](../archive/work/r26-rdna4-dpp-butterflies/README.md) - PR #110's DPP wave32 warp butterflies, **folded into block 15 in r26** behind the build-time `-DGGML_HIP_NO_DPP_XOR` gate (a runtime device-side gate wedged the FA prefill, so it was dropped).
* [`archive/work/r26-dflash-dev-default-on/`](../archive/work/r26-dflash-dev-default-on/README.md) - PR #107's DFlash F1 warn-and-fall-back + default-on, **folded into block 15 in r26** (TODO #30 closed).
* [`archive/work/moe-cache-autosize/`](../archive/work/moe-cache-autosize/README.md) - the expert-cache /
  arena campaign (arm + auto-size, the movable-boundary slab).
* [`archive/work/expert-cache-split/`](../archive/work/expert-cache-split/README.md) — closed with its
  "mirrored experts" premise **refuted** (the weights are already split per device).
* [`archive/work/host-pinned-buffer-crash/`](../archive/work/host-pinned-buffer-crash/README.md) — the
  `--load-mode none` host-expert page fault; no longer reproduces (14/14 clean on r24).  Only the stale
  `common/common.cpp` warning removal remains (`TODO.md` #45).
* [`archive/work/layer-split-host-experts/`](../archive/work/layer-split-host-experts/README.md) —
  per-device host bufts so `-sm layer` spreads host experts over the GPUs (**delivered in r14**, block 06).
* [`archive/work/moe-mmq-overread/`](../archive/work/moe-mmq-overread/README.md) — the MoE MMQ expert-table
  over-read (**resolved 2026-10-03**; the host gather's prefill "win" was the corruption).

The resolution pattern (investigate -> fold into the owning block -> regenerate + validate -> record +
archive -> ship) is documented in `archive/work/lightning-indexer-fusion/README.md`; promotion rules are in
`AGENTS.md`.
