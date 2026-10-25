# BASELINE - provenance and drift policy

Historical baseline records, checkpoint cuts, drift fixes, validation records, older branches, and the original-source provenance, moved out of BASELINE.md (2026-08-29) to keep the top-level doc lean. Current state: `../../BASELINE.md`.

---

### Re-base (2026-10-25): `3f03b1772` -> `6050b467c` -- 21 upstream commits, two of them duplicating delivery work

Shipped as **`v15-6050b467c-r138`** (base tree `cb3251e177`, tip `abb8f76b2b`, tree `35090ba36c`,
15 blocks, numbering unchanged).  One day of upstream, but dense:

* **#29948 fused FFN gate/up + GLU in MMQ** restructures the same kernel block 08 fuses for RDNA
  (`mmq-load-tiles.cuh` gains `x_gate` and `has_fusion`, `mul_mat_q_process_tile` gains leading
  `x_gate`/`glu_op`/`glu_limit`, `I_dst = I/2`, a new `util_funcs` dispatch and `mul_mat_q_switch_fusion`).
  Its arm is NVIDIA-only (`ggml_cuda_mul_mat_q_fusion_matches` ends in `turing_mma_available()`; the
  write-back assumes the MMA accumulator layout), so **both arms ship, arch-disjoint, on one dispatch**:
  upstream's `has_fusion` and the delivery's `has_gate`, with `static_assert(!(has_fusion && has_gate))`.
* **#31028 qwen merge tensors on convert** merges qwen4exp's `hc_*_down` + `hc_*_inject` into one GGUF
  tensor and one GEMM; the delivery adopted it for that block's HC path (loader keeps the separate-tensor
  fallback), while the MTP branch stays the delivery's own QSA/derived-visibility code.
* Upstream factored the MMQ J-tile choice into `ggml_cuda_mmq_get_J_best()`; the delivery's
  `ggml_cuda_mmq_select_J()` is a strict superset and is now the implementation, with upstream's helper
  delegating to it (`GGML_CUDA_MMQ_J_MAX` moved inside it, so both call sites honour the cap).
* Block 10's moved dispatcher required the 3-way merge with the **pre-move** tree as the base; using the
  rebased block-09 tree drops upstream's new arm silently (0 conflicts, feature missing).

Gates: per-block build **15/15**; prefill-logit **KLD 0.000531 / same-top-p 98.863 %** against the
r137-recorded base (no re-record needed); `FLASH_ATTN_EXT` 6450/6450, `GATED_DELTA_NET` 55/55, all
MUL_MAT/MUL_MAT_ID/QSA/TOP_K_MASK/HC_MIX families pass; `plain == draft-mtp` byte-identical on dense Q8_0
3-GPU tensor, 35B-A3B 2-GPU `-ncmoe 40` and Flash-Next 3-GPU `-ncmoe 48`; MTP Protocol A 1.81x/2.34x/
2.47x/2.71x.  Greedy text differs from the r137 build on every arm -- expected on a base move, and decided
by the prefill-logit gate.  Not re-validated: gfx1151/gfx1100 (open item in `TODO.md`).  Campaign record:
`archive/work/rebase-2026-10b/`.

### Checkpoint cut (2026-08-29): master `cc83d7b48` - cross-architecture consolidation bookend

Master moved 4 commits past `d7bd3bfca` to `cc83d7b48` (sycl --fit
improvement, vulkan fastdiv/duplicate cleanup, metal fa-vec tunings, vulkan
mul_mat_id K-pad change + issue-27873 mul_mat_id test cases). None of the
4 commits touch the block footprint except `tests/test-backend-ops.cpp`
(the issue-27873 test additions in `make_test_cases_eval`, ~line 9367 -
outside every block hunk). `scripts/apply-all.sh` applies all 11 patches
**zero-fuzz to master `cc83d7b48`** (no 3-way fallback), and the applied
tree is byte-identical to the validated `d7bd3bfca` tree for every block
file except `fattn.cu` (the gfx1151-ports RDNA3_5 ternary, which is the
current official block-04 content and is strictly RDNA3_5-gated) and
`tests/test-backend-ops.cpp` (which carries exactly upstream's issue-27873
addition). The 3-arch validation (gfx1100/gfx1151/gfx1201) recorded in
MANIFESTS.md therefore carries over unchanged. Checkpoint:
`baseline/cc83d7b48`.

The same patch files apply **zero-fuzz to master at `192067b72`** (13
upstream commits past `d222767c7`): `scripts/apply-all.sh` applied all 12
patches without 3-way fallback, and the applied tree is byte-identical to
the `d222767c7`-validated tree for every block file (the 13 drift commits
only touch files outside the blocks). A fresh full build
(`~/bin/build-llama-rocm-714`, ROCm 7.14, gfx1201) succeeds, and the
real-model sanity run (Qwen3.8-27B Q4_K_XL, 1 card) matches the validated
numbers (pp512 1330 t/s, tg64 30.9 t/s; llama-cli generation 30.7 t/s).
That checkpoint is `baseline/192067b72`; its patch files are unchanged from
`d222767c7` (regeneration from the new tip produces identical content -
only hunk line hints and blob SHAs shift).


### Drift fix (2026-08-27): block 01 re-based against master `fe235f434`

Master moved 7 commits past `192067b72` to `fe235f434`. Block 01
(`01-adaptive-mtp.patch`) no longer applied: upstream added the
`n_max` parameter to `common_speculative_impl` (`spec: Add
benchmark-only synthetic speculative acceptance options`, `2bb9bddaf`), so
the old hunks in `common/arg.cpp`, `common/speculative.cpp` and
`tests/test-arg-parser.cpp` conflicted (`git apply -3` left conflicts in
`common/speculative.cpp` and `tests/test-arg-parser.cpp`). The fix comes
from the fork's `adaptive-mtp` branch (`/home/stew675/stew675/llama-master-before`,
same 5-commit series re-based on `fe235f434` with the `n_max` constructor
adapted - only `common/speculative.cpp` differs from the old block-01
content, by 2 lines). `01-adaptive-mtp.patch` was regenerated from that
branch; all 11 patches then apply **zero-fuzz to master `fe235f434`** via
`scripts/apply-all.sh` (no 3-way fallback), and the applied block-01 tree is
byte-identical to `adaptive-mtp` for all 10 block files. The other 10 blocks
are untouched (their patch files apply zero-fuzz unchanged).


### Validation record (2026-08-25, AMD Radeon AI PRO R9700 x3, ROCm 7.14)

- `test-backend-ops` (ROCm0/1/2 + CPU): **14883/14883 passed, 0 failures**.
- `GATED_DELTA_NET` `-b ROCm0`: **46/46** in all four dispatch configs
  (default bf16 S_v=128, `GGML_CUDA_GDN_CHUNKED_BF16=0`,
  `GGML_CUDA_GDN_CHUNKED=0`, both force-enabled).
- `test-arg-parser`, `test-speculative-adaptive`: all tests OK.
- The applied tree is byte-identical to the source fork branch `chunked-gdn`
  for every production file (the only intentional divergence is the test
  harness fix described below).


### Validation record (2026-08-26, AMD Radeon AI PRO R9700 x3, ROCm 7.14) - k-quant bundle

Block 09 retired; its Q6_K VDR=2 content folded into block 12 (the combined
k-quant umbrella, incl. RDNA4-gated Q8_0 VDR=4). The set is now 11 patches
(01-08, 10, 11, 12). Re-validated: fresh master `192067b72` checkout,
apply-all 01-08 + 10 + 11 + 12 - zero-fuzz apply, applied tree byte-identical (0 diff
lines) to the pre-bundle validated tree. MUL_MAT q4_K/q5_K/q8_0 54+47,
MUL_MAT_ID 76/76, GDN 46/46. Real-model sanity unchanged (Q4_K_XL decode
+5.5-5.7%, Q8_0 model +0.8%). PPL re-run on the Q8_0 model: 6.3162
(128 chunks) and 6.3563 (full corpus) - bit-identical to the pre-block-12
records (PPL is prefill-only; the VDR changes do not touch the prefill
path).

Greedy-purity note (block 12 is the only patch that changes decode
numerics): compute outputs are not bit-identical to a build without it - max
logit diff 0.184 vs 0.203 for flash-attn on/off; greedy streams are
deterministic within a build but can flip across configs (1 of 3 test
prompts diverged at char 176). Excluding block 12 restores purity; it is
applied last.


## Validation record (2026-08-28, AMD Radeon AI PRO R9700 x3, ROCm 7.14) - mmvq umbrella fold + renumber

Block 06 (old numbering) retired; its gfx1151 (RDNA3_5) mmvq parameter
table folded into block 10 (the k-quant + mmvq-parameter umbrella),
completing the plan from `~/stew675/bundle-k-quant-boosts.md` section 7.
Block 10 now carries the only decode-numerics changes in the set across ALL
architectures (RDNA3, RDNA3_5, RDNA4): the Q6_K/Q4_K/Q5_K/Q8_0 VDR kernels,
the RDNA3_5 nwarps=2 Q8_0 table, and the RDNA4 MoE mmid whitelist. Block
08's RDNA3_5 verify-batch hunk (`ncols_dst <= MMVQ_MAX_BATCH_SIZE`) moved
to block 10 with the table (it modifies block-06 content). The set is 10
patches numbered 01-10 with no gaps (old 07->06, 08->07, 10->08, 11->09,
12->10). Re-validated: fresh master `fe235f434` checkout, apply-all
01-10 - zero-fuzz apply, no 3-way fallback, applied tree **byte-identical
(0 diff lines)** to the pre-fold validated tree (every block's file bytes
unchanged; only the hunk boundaries moved between patches 08 and 10).
Block-01 behavior unchanged (`test-speculative-adaptive`, `test-arg-parser`);
MUL_MAT 1193/1193, GDN 46/46 (4/4 backends), q4_K/q5_K/q8_0 OK on the fresh
build. RDNA3_5 table semantics unchanged from the validated block-06/08
combination.

Greedy-purity note (block 10 is now the ONLY patch that changes decode
numerics on any architecture): compute outputs are not bit-identical to a
build without it - max logit diff 0.184 vs 0.203 for flash-attn on/off;
greedy streams are deterministic within a build but can flip across
configs (1 of 3 test prompts diverged at char 176). This is a different
fp32 rounding path, not a correctness change (see
[`GREEDY-PURITY.md`](../../GREEDY-PURITY.md)); PPL is bit-unchanged. Excluding block 10
restores 100% greedy purity on RDNA3, RDNA3_5 and RDNA4 alike (block 06's
gfx1151 table was the last non-block-10 numerics change and is now inside
block 10); it is applied last.


### Validation record (2026-08-27, AMD Radeon AI PRO R9700 x3, ROCm 7.14)

Master tip `fe235f434` + the 11-block patch set (applied with `apply-all.sh`,
zero-fuzz, block 01 regenerated from fork `adaptive-mtp`):

- `scripts/apply-all.sh` applied all 11 patches in manifest order with no
  3-way fallback (block 10 in context after 03+04, as required).
- The applied block-01 tree is byte-identical to the fork branch
  `adaptive-mtp` for all 10 block files; blocks 02-12 are byte-identical to
the `192067b72`-validated block code (their patch files were not changed).
- Block-01 behavior verified: `./bin/test-speculative-adaptive` and
  `./bin/test-arg-parser` pass (same tests as the `d222767c7` record);
  `--draft-mtp-adaptive` range validation (n_min_adaptive vs n_max,
  incl. the MTP-layer-capped n_max case) matches the fork branch.

### Validation record (2026-08-26, AMD Radeon AI PRO R9700 x3, ROCm 7.14)

Master tip `192067b72` + the 12-block patch set (applied with `apply-all.sh`,
zero-fuzz):

- Fresh full build (`~/bin/build-llama-rocm-714`): clean, all targets
  (incl. llama-server, llama-cli, test-backend-ops).
- Real-model sanity (Qwen3.8-27B Q4_K_XL, 1 card): llama-bench pp512
  1330 t/s, tg64 30.9 t/s; llama-cli --single-turn generates correctly
  (30.7 t/s). Matches the `d222767c7`-validated numbers.
- The block code is byte-identical to the 2026-08-25/26 records below (the
  13 drift commits do not overlap any block file region - proven by the
  zero-fuzz apply).


### Validation record (2026-08-26, AMD Radeon AI PRO R9700 x3, ROCm 7.14)

Blocks 01-11 as above, plus **block 12** (`12-k-quant-boosts.patch`,
`a7d092368`):

- `test-backend-ops` (ROCm0 + CPU): **14883/14883 passed, 0 failures** on the
  block-11+12 tree.
- `MUL_MAT` q4_K/q5_K: **54/54**; `MUL_MAT_ID` q4_K/q5_K: **76/76**.
- Real-model (Qwen3.8-27B Q4_K_XL, 17 GB, mixed Q4_K/Q5_K/Q6_K/Q8_0, 1 card,
  llama-bench): decode +5.5-5.7% (tg32/tg128), prefill +0.6-1.0%
  (pp128/512/1024) vs the block-11 tree.


## Older branches

`baseline/192067b72` holds the previous checkpoint (same patch files as
`d222767c7`, zero-fuzz-validated at `192067b72`, full-suite + real-model
records in the validation sections above). It is superseded because block 01
drifted (see the drift-fix section); it remains the known-good set for that
upstream range.

`baseline/758443071` holds the original patch set (generated against
`758443071`, mirroring the fork `chunked-gdn` exactly). It remains the
known-good set for that upstream range, with two caveats:

1. Its `02-chunked-gdn.patch` still carries the deterministic test-harness
   seed (see above) - the fork's state. Use this branch's fixed patches if
   you are testing against `758443071` and hit the `rms_norm_back` /
   `cross_entropy_loss_back` failures.
2. Its block-10 test hunk is placed at a re-based location in
   `make_test_cases_perf()` (context added by block 04 in the fork).


## Original source commits (chunked-gdn lineage, historical)

| `0001-…-adaptive-MTP…` | fork branch `adaptive-mtp` (re-based): `c0f398ec5` `f3208c5c5` `0bbd7f3c2` `b5c99b890` `b26c775e1` — same 5-commit series as the old `chunked-gdn` lineage (`87ad1db26` `b56926039` `d0d7ff27e` `8d70e21f5` `0cf87e989`), re-based with the `common_speculative_impl` `n_max` constructor fix |
| `0002-…-chunked-gdn…` | `876ef1f0b` `5d2090e96` `b220647b1` `a4982afa2` `659f94987` `2a1e5c5a8` `1da07e19b` `77d51ee28` `abfa24265` `be46c7621` `3441b7d40` `246136122` `05cab3c41` |
| `0003-…-bf16-kv-cache…` | `5485e79e4` `b98265cfd` `07767a88a` `ef3673358` `b6bfa422e` `5e6072558` `bd5bf0ea3` `d33ce1adf` |
| `0004-…-wmma-flash-attn…` | `beaf69fb6` (plus the gfx1151-ports RDNA3_5 consumer-side fix) |
| `0005-…-bit-identical-decode-cpu…` | `89ac4ba1f` |
| `0008-…-fused-core…` | `14e5dd427` `d0e6119a7` `333e8f950` `c11752b18` `10e016df4` `85387ba3a` `8e1300159` `ac08b6d85` `a84112dcf` `9b4554626` `555e79ab2` `00f53040f` `ec09a818e` `bb64338f9` `4c0440841` `3d65d7979` |
| `0006-…-host-buffer-revert…` | `edb8d44c0` |
| `0007-…-meta-device-wrapper-skip…` | `32670eec8` |
| `0009-…-meta-headroom…` | `f2a22a71` (fork branch `rdna-boosts`, NOT on `chunked-gdn`) |
| `0010-…-k-quant-boosts…` | combined k-quant + mmvq-parameter umbrella: `cd35abd19` (Q6_K, ex-block 09) + `a7d092368`, `f1a072dcd` (Q4_K/Q5_K/Q8_0) + `5b320ed94` (RDNA3_5 table, ex-block 06, incl. the verify-batch hunk block 08 used to carry) — all fork branch `rdna-boosts`, NOT on `chunked-gdn` |
| `0011-…-prefill-graph-skip…` | the CUDA prefill-graph skip (fork `rdna-boosts` branch) |
| `12-hybrid-allreduce-hip.patch` | WIP exploration (`wip/12-hybrid-allreduce-hip.md`); no source commit — the fork's working-tree delta vs `f6f8f6778` (allreduce-hip.cu + allreduce.cu + allreduce.cuh + ggml-cuda.cu), RDNA4-gated |


---

## `BASELINE.md` as of 2026-10-20 (preserved verbatim before the V2.0 cleanup)

The 16-block era's provenance narrative, the per-block cut history (`9113cc188` .. `ebbb18522`), the
"two fixes vs the fork" record and the dated re-baseline records.  The current provenance and drift
policy are in `../../BASELINE.md`; nothing here describes the current delivery.

# BASELINE - provenance and drift policy

Current state (V2.0): `main` is the delivery branch carrying the **12-block set** (block 00 + blocks
01-11) generated against the fork point **llama.cpp master `097cf33d9`**; the current release is
**`v12-097cf33d9-r91`** (canonical tip `45f817cc3`, net tree
`90507dfb528cbfe0c688261226d18934653b1b61`).  r91 closes the rollback-boundary campaign (blocks 03/09/10:
the per-sequence tracker folds every ubatch in, the chunked GDN prefill writes its rollback snapshots via a
fixed-`D` sequential tail, and the MoE expert arena is refused on a shared host/device pool) -- same fork
point, no re-base.  r90 fixed the async graph-input nondeterminism and flipped
`GGML_CUDA_ASYNC_INPUT` **default-on** (**block 10**, no new block): the `set_tensor_async` /
`set_tensor_2d_async` paths copied directly from the caller's live pinned host pointer (a CPU split's
output that the next ubatch rewrites), which corrupted a single graph whenever the host ran ahead; both now
snapshot through the pinned upload ring, unconditionally, and the ring's wait/done guard is unconditional.
It is the follow-up to PR #182.  r89 folds PR #182 (`@briansp2020`) into **block 10**: inside
the opt-in `GGML_CUDA_ASYNC_INPUT` path a graph's staged inputs are copied by one kernel at the next
graph start instead of one `cudaMemcpyAsync` + event record per input, and the "previous graph done"
event is recorded by the next upload instead of after every `graph_compute` (two new default-on
kill-switches, `GGML_CUDA_ASYNC_INPUT_BATCH` / `..._LAZY_DONE`; `ASYNC_INPUT` itself stays opt-in because
its capture-time nondeterminism is unchanged).  Before it, r88 folds @slavap's RDNA2 (gfx1030) FA gate
fixes
(PR #181) into **block 05** -- bf16 F16-staging scratch sizing, the decode-band `q_h2` launch argument,
and an RDNA2-gated last-tile guard tail -- with no re-base and no new block.  Before it, r87 fixes the
**r86 second-request regression**: r85's
drop reference recorded the wide prefill region, so an equally-wide second prefill never dropped the wide
compute layout or re-armed the arena (decode ~half speed from request 2, GPU page fault by request 5).
**Block 09** now forces the drop on every wide -> narrow transition; a 10-prompt field series runs
63.5-87.2 t/s with the arena at 41 349 MiB and no fault.  Before it, r86 folded the two promotion
candidates in: the **slab
transient absorption** (**block 09** common half + derived draft reserve, **block 10** CUDA/slab half,
**block 11** loaded-draft ownership handoff) and the **meta hot-path dispatcher** (**block 00**).
Before it, r85 fixes the slab narrow band: it is learned from the real
decode graph (**block 10**) and the prefill->decode drop reference takes the region actually observed
(**block 09**), so an MTP decode view is no longer placed above the H2D staging ring and re-evicting the
expert-cache arena every step (18 -> 1 re-arm, decode 23.6 -> 45.6 t/s, at 16k prefill 1443 -> 1406 t/s).
Before it, r83 promotes **prefill cache staging** from
`archive/work/prefill-arena-residency/` into **block 09** (the engine + the immutable prefill residency
snapshot) and **block 10** (the `stage_from_cache` hook; it is block 10 that creates
`ggml-cuda-moe-cache-hooks.cu`).  A whole-table staging copy now fills each expert from the cache arena
when it is resident and from the pinned host master otherwise, so only the misses cross PCIe; the
staged bytes and every consumer are unchanged.  Default on (`GGML_MOE_CACHE_STAGE=0` is the kill
switch).  Measured on gfx1201 (2x R9700, Flash-Next IQ4_NL `-ncmoe 48`): prefill +32-54 %, greedy text
byte-identical, prefill-logit KLD 0 / 100 %.  Before it, r82 fixes **issue #170** (**blocks 00, 09**:
an unallocated meta alias resolved through its view chain, a failed constituent allocation returns
NULL, and the meta `offload_op` hook is limited to `GGML_OP_MUL_MAT_ID`).  Before it, r81 folds four
patches from three contributor PRs into **block 10**: M-RoPE / IMROPE admitted to the fused rope + FWHT kernel (PR #175 0001), a BF16
`SET_ROWS` refused by the 5-op norm+rope+set_rows fusion so the fused op cannot abort (PR #175 0002; a
latent abort on r80 for a NEOX qk-norm model with `-ctk bf16`), the q8_0 K/V cache writes folded into
that kernel (PR #173, reachable now that the fusion runs on `rope_multi` models), and the GDN q/k l2
norms paired into one launch (PR #179).  All bit-identical: same-seed text byte-identical to r80 on
35B-A3B and dense 27B, width probe maxdiff 0 with the r80 hashes, prefill-logit 0.000000 / 100.000 % on
both.  PR #177 (GDN conv snapshot fold) was rejected -- not run-to-run deterministic on 35B-A3B -- and
its record is `archive/work/gdn-conv-snapshot-fold/`.  Before it, r80 fixes the MoE heat-map doubling
(**block 09**).  Before it, r79 folds PR #167 into **blocks 07/10**: the MoE
expert kernel takes the 32-element dense Q4_K dot product (same layout and VDR 4, one integer chain per
sub-block), **default on RDNA3_0** and a **RDNA4 build-time opt-in** (`-DGGML_HIP_RDNA4_Q4K_MOE_VDR4`,
off by default; the ccache host-visibility constant ships with it).  **Block 10** also makes the PR #168
KV prefill pad **default-on (8192) on a multi-device split**, inactive on a single GPU (the r77
single-GPU -0.35 % shape is now off).  Independent gfx1100 rerun: tg256 +0.83 %, batched B=1/B=4/B=8
+1.4/+1.9/+2.7 %, KLD 0.004544 / 97.124 %, prefill-logit 0.000000 / 99.990 %; the P-axis MTP -2.7 % is
acceptance-driven content drift (plain decode on the same prompt +0.7 %, not 3-GPU-specific).  The
RDNA4 default is byte-identical to r78; 0002 on 2-GPU is flat on tg and +0.5-1.5 % at the verify widths,
so it stays opt-in.  Before it, r78 folds the KV-rotation FWHT fusion and the one-token ADD +
`rms_norm_q8_1` fusion into **block 10** (PRs #172/#174; same-seed text byte-identical to r77,
prefill-logit 0.000000 / 100.000 % on Gemma4 12B).  Before it, r77 folds two contributor PRs into
**blocks 08/10**: the q8_0 `SET_ROWS` K/V-write kernel and the register-quantizing `rms_norm_q8_1` (PR
#171; bit-identical, no `try_fuse`/band change, MTP decode +0.6-0.7 %, no regression at the verify
widths), plus the **then-opt-in** coarse prefill KV pad (`LLAMA_KV_N_PAD_PREFILL`; r79 flips its
default for a multi-device split).  Before
it, r76 folds the RDNA2 (gfx1030) tuned tile-FA + k-quant
work (PR #165) into **blocks 05/07**, gated to RDNA2 (`#if defined(RDNA2)` /
`GGML_CUDA_CC_IS_RDNA2`): gfx1201 / gfx1100 / gfx1151 keep their previous numerics (same-seed greedy
text identical; prefill-logit mean KLD 0.000000 / 99.990-100.000 % on all three).  Before it, r75 fixed
a release blocker exposed by r74: **block 10**
places the wide (prefill) compute view at the end of the **carved** H2D staging ring, not the live
re-armed *request* (pre-fix the wide buffer aliased the ring and the first prefill faulted on 35B-A3B +
host experts + MTP: `k_get_rows_float` on Q4_K_M / `rope_multi` on Q8_0); **block 11** excludes the MTP
draft context from the ring auto-size (its weights are fully resident and it never stages a table).
Before it, r74 sizes the MoE expert-cache arena at the warmup and stops the seed at the prefill reclaim
boundary (`MOE_EXPERT_CACHE_SIZE_AT_PRIME=0` kill switch), and r73 unifies the heat-map sidecar to one
`<base>.moe-heatmap` per model.  **r73 and r74 were internal pre-release betas and were never published**
(r74 failed validation and is superseded by r75).  Before them, the r68-r72 amendments (container work,
`-ncmoe` host placement, the ring-region-guarded prefill drop, the ring auto-size and the H2D-ring
largest-slice fix) are recorded in `patches/README.md`.  Earlier, r67 fixes the r63..r66 host-expert
regressions:
**block 10** restores the batched, duplicate-correct `mm_ids_helper` template path (r66's generic helper
cost ~20-23 % of the fully-resident 35B-A3B prefill: 6430 -> 8243 t/s, r65 8357); **block 09** sizes the
expert cache from the **total** registered host bytes so the uneven `-sm tensor` split no longer leaves
the arena unsized (GSQ `-ncmoe 48` decode 14.8 -> 62 t/s); **block 10** bounces any pageable source on
the weight upload (the `-lm none` + `-sm tensor` SDMA0 fault, issue #137 class); **blocks 10/11** pin the
narrow floor before the first prefill and place a prefill ubatch (`n_tokens >= 64`) above the ring so it
arms (GSQ `-ncmoe 48` prefill 645 -> 687 t/s).  Gates: `validate-set.sh` strict 12/12 green,
`gate-prefill-logits.sh` 0.000707 / 98.755 %, `gate-qwen4exp-quant-coherence.sh` PASS, adaptive-MTP
`none == draft-mtp n3` byte-identical on prose and mixed, width purity `none == n1 == n3 == n7`,
`test-backend-ops -o MUL_MAT_ID -p dup=1` 16/16 (full 947/947).  Before it, r66 folds the upstream
**MUL_MAT_ID duplicate-ids fix** into **block 00** (generic upstream foundation; every repeated expert
occurrence is its own compact row) and caches the hot-path **environment reads**.  Before it, r65 merges
five `@briansp2020` PRs into
their owner blocks:
the pinned output buffer under `-sm tensor` (`LLAMA_OUTPUT_PINNED=0`, **block 09**), the second
scheduler for the MTP draft's output graphs (`LLAMA_DRAFT_SCHED2=0`, **blocks 09 + 10**), the Q4_K/Q6_K
mmvq vecdot instruction reductions (**block 07**), and two gfx1100 ports (the Q4_K-only 128x256 MMB
dense tile in **block 06** and 4 rows per block for a Q4_K verify batch in **block 08**).
Gates: `validate-set.sh` green, `gate-prefill-logits.sh` 0.000707 / 98.755 %, rule-5
`llama-batched-bench` B=1/4/8 no worse than r64, MTP acceptance and text identical.  Before it, r64
folds the r63 residuals: **block 09** splices an
arbitrary segmented Meta write (no `offset % row_stride` classifier, no accumulator; byte-identical on
current callers) and the Gemma4 26B-A4B front/tail drift is proven a near-tie and accepted; **block 10**
owns the CUDA splice-gather ring in `ggml_backend_cuda_context` (torn down with its streams) and drops
the qwen4exp MTP `ubatch.token` assert so mtmd image requests work under `--spec-type draft-mtp`
(PR #160).  Gates: `validate-set.sh`, `gate-prefill-logits.sh` 0.000707 / 98.755 %,
`gate-qwen4exp-quant-coherence.sh`, adaptive-MTP `none == n3` (`-n 3000`), 35B `-ncmoe 20` hang loop
10/10.  Before it, r63 flips `-ncmoe N` to host the **last N** trunk layers
(`llama_model_params.n_cpu_moe_from_end`; `LLAMA_HOST_EXPERTS_FROM_END=0` kill switch; 35B rear >=
front, `tg500` 87.5 vs 82.6, `pp4096` 4264 vs 4120) and lands three block-09 fixes: the MoE-cache
preflight is now universal (in `llama_model_load_from_file`, so llama-bench arms the slab too), the
deferred arena sizing waits for the full host-table set, and `-lm none` + `-sm tensor` uses whole-tensor
Meta uploads (r54 regression, qwen4exp gate re-run green).  The known follow-up
(`archive/work/gemma4-drift-and-meta-async/`) is resolved in r64 (near-tie accepted; segmented write fixed in block
09).  Before it, r62 fixes the `-sm tensor` + `-ncmoe` below-gate upload race in
**block 10**: `ggml_backend_cuda_set_tensor_2d_async`'s splice-gather issued its 1-D H2D on
`copy_stream()`, escaping the scheduler's main-stream `wait_before_overwrite`, so it could overwrite the
reused split-input buffer while the previous split still read it (garbage on 35B-A3B/gemma4/Flash-Next
GSQ, NaN perplexity); the gather H2D now runs on the main stream.  Before it, r61 makes the heat map's
**file mode the freeze signal**:
a read-only sidecar is loaded and used to seed but never overwritten and never relocated, and
`MOE_EXPERT_HEATMAP=load` (aliases `ro`, `read-only`) is the explicit freeze.  Before it, r60 added the
**persistent cross-run MoE expert heat map** to **block 09** (device-side prefill routing tally + on-disk
`<gguf>.moe-heatmap` sidecar + seed-at-sizing + heat-ordered layer placement; default on) -- measured
+34 % decode over a 48-token turn and +10.3 % over 256 on a `MOE_EXPERT_CACHE_MIB=4096` host-expert
config.  Before it, r59
lands issue #139 in **block 01**: the internal
all-reduce's spin budget is wall-clock time (`wall_clock64()` sized by
`hipDeviceAttributeWallClockRate`; the `clock64()` counter steps backwards on gfx1201 while a queue is
stopped), `GGML_CUDA_AR_SPIN_TIMEOUT_MS` (default 2000 ms) is a soft deadline that reports a late peer
and keeps waiting, and `GGML_CUDA_AR_SPIN_HARD_MS` (default 20000 ms) alone poisons and falls back to
the butterfly.  Before it, r58 folded PR #150 (DFlash layer-input preset) into
**block 02** (the draft-setup read of `<arch>.target_layers` and the preset call) and **block 11** (the
preset API/storage, the `llama_context` ctor apply and the `set_embeddings_layer_inp()` early return);
a DFlash draft no longer frees and reallocates the target's compute buffers at the first request
(5 -> 4 reserves).  Before it, r57 scoped the `-Wshadow` gate to `-Wshadow=local` on
GCC (the container host compiler) in block 00 and fixes the two GCC-only sites it exposed (the AMX
buffer-type initializer, block 09; the `size_t >= 0` assert, block 10).  The applied tree
is not tree-identical to the 16-block r50 set (Phase 1 redistributed the concerns; Phase 2 moved each
into a self-contained function/file).  See `WORKLOG.md` 261014.006 (r61), 261014.005 (r60), 261014.004 (r59), 261014.003 (r58), 261014.001 (r57), 261013.007 (r56),
261013.005 (r55), 261013.004 (r54), 261013.003 (r53) and 261013.002 (r52).  The entries below are the
**pre-V2.0 (16-block) provenance history**:

**Previous release `v16-a55e952b8-r49`** (canonical block-15 tip `f0a74f4cc`, net tree
`45ba5da24f872c5f9aa1b70b85470b5eb4d98395`): the mmap host-memory reclaim lands
in **block 06** (the loader stops prefetching the whole file and unmaps each tensor's mapping right after
its device upload, so a `-ncmoe` load never holds the full mapping and the pinned host master at once;
`LLAMA_LOAD_RELEASE_MMAP=0` kill-switch) and two AMD consistency fixes land in **block 15** (the Q1_0
`__byte_perm` -> `v_perm_b32` port and the `cpy-batch.cu` device idiom).  Before it,
**`v16-a55e952b8-r48`** (canonical block-15 tip `0bc12586`, net tree
`3b9f403544169544f312bd24e4de31a554bf189d`): the `--load-mode none` + `-sm tensor` startup GPU fault
(**issue #137**) is fixed in **block 06** (the Meta device reports `host_buffer`/`events` and proxies its
event iface to the simple devices, so the loader's pinned async-upload pipeline engages;
`ggml_backend_meta_set_tensor_async` is generalized to the loader's chunked and segmented ranges;
adapted from upstream PR #22466).  Before it, **`v16-a55e952b8-r47`** (canonical block-15
tip `6daf81569`, net tree `0099dd85130db65cbfbcfd738abfc57566bffe6f`): the `--fit off` slab blackout
(**issue #136**) is fixed (block 06: the host-expert preflight parses its env before the cache gate, so
the slab is always armed when the cache is enabled; a slab-creation failure errors and streams), the
copy-engine all-reduce scratch is fail-soft with an optional `GGML_CUDA_AR_CE_SCRATCH_MIB` budget
(block 12), and block 15 is re-cut on top.  Before it, **`v16-a55e952b8-r46`** (canonical block-15
tip `2e19a9d3`, net tree `a6ca8905f3ce5eb9ac7bf67ff8863a0b43419293`): the r45 **block 16 is folded
back into the 0000-0015 set** (host-side/system work into block 06, small-message all-reduce into
block 12, and the decode/latency remainder plus issue #133 into block 15).  Before
it, **`v16-a55e952b8-r44`** (canonical block-15
tip `4b2874db9`, net tree `fb348f69badfab0bb2a284c97eaed84117ea57d9`): the Windows host-pool portability
fix (issue #130) and a `-Wmissing-field-initializers` cleanup fold into **block 15**, and the per-sequence
recurrent rollback guard (issue #131) folds into **block 02** -- no new block.  Before
that, **`v16-a55e952b8-r41`** (canonical block-15
 tip `a89ffa273`, net tree `46c4ce7a6a00aa6c6931c5a89faf847dde19fd34`): **block 12's hybrid NCCL init is
eager again** — the r34 lazy mid-run `ncclCommInitAll` perturbed the `-sm tensor -ncmoe` split compute and
broke MTP purity under the expert cache; eager init is bit-identical to pure RCCL and restores it (TODO
#53).  Before it, **`v16-a55e952b8-r40`** (canonical block-15 tip `fdecb4d337b2`, net tree
`73c371a2f453489dccdcea73ed22216b27cb9a28`; as shipped the block-15 probe fix moved the tip to
`b8d1e2988`, tree `1745cf9b`): the device-remap
(`DEVMAP`/`DEVPOLICY`/`KSLOT`/`DEV_EAGER`) experiment is removed from the code base (block 06 never
creates it) together with the non-devmap WIP fixes (the per-table fusion guard for issue #50, the
narrow-2 slab MTP-aliasing fix, `-ncmoe` host-expert offload).  Before it, **`v16-a55e952b8-r38`**
(canonical block-15 tip `849c04161`, net tree `888564105e13dd73fefda755ea5b055d65011c16`): the
`fit-slab-accounting` revival is folded into block 06 (G1/G2 `--fit` arena+headroom reservation, G3
adaptive slab reserve, G4
`GGML_CUDA_OPTIONAL_ALLOC_MAX_FREE_PCT` free-VRAM cap, G5 split-slice guard, G6 nextn-offload guard,
G7 scheduler multi-consumer fill, per-device host-expert accounting) plus the G4 FA-staging cap in
block 15.  Before it, **`v16-a55e952b8-r36`** (canonical block-15
 tip `8e28631f6`, net tree `d55aebfe0d7af530aa485ffd7ebf5e1dce331b08`), which folds three contributor
prefill PRs into the existing blocks: the shared BF16 src1 reuse (#119) and the in-place host-expert
prefill reads (#121) into block 15, the pool free-VRAM floor (#122-0001) into block 06, and the 6 GiB
`-sm layer` slab headroom (#122-0002) into block 09.  Before it, **`v16-a55e952b8-r35`** (tip
`645fd4989e542d93`, tree `b2ba2bb32c76e857`) folded the bounded pinned host-expert pool (`--host-experts
pool`, an option; the default stays `pinned`) into block 06 and the `mul_mat_vec_q_moe`
row-tail clamp into block 13.  Before that, **`v16-a55e952b8-r34`** deferred block 12's `ncclCommInitAll`
to the first large tensor, and **`v16-a55e952b8-r33`** removed the pageable host-expert master (issue
#116).  Before that, the r32 state fixed the two bugs found during the PR #115 review: **issue #48** (the post-prefill re-reserve uses the current ubatch's sequence count, so a
multi-sequence decode no longer builds a zero-token attention graph; `llama-batched-bench -npl 4`
aborted) and **issue #49** (the meta split-state computation is pre-warmed bottom-up, so a >1200-node
`src` chain no longer overflows the stack; `llama-cli --spec-type draft-mtp --spec-draft-n-max 3`
segfaulted).  #48 is in block 06, #49 in block 15.  Before it, **`v16-a55e952b8-r31`** folded **PR #115
into block 06**: the MoE expert-cache decode/verify band follows the routed-expert MMVQ band (16 tokens
on RDNA4, 8 on RDNA3/NVIDIA) via a new single-owner `moe_cache_band` backend hook, plus a fix for the
device-side admission fill list; `GGML_MOE_CACHE_MAX_TOK=8` restores the old band.  Before it,
**`v16-a55e952b8-r30`** was the **issue #113 root cause**: the r29 bf16 divergence was an `A_sc` stride aliasing bug for `n_seqs > 1`, not bf16
precision; fixed in both GDN kernel files, so the `GGML_CUDA_GDN_CHUNKED_BF16` default is **on** again at
a near-lossless KLD (0.0007 / 0.00005 / 0.00015 on gfx1201 / gfx1100 / gfx1151).  Before it,
**`v16-a55e952b8-r29`** flipped the path off as a workaround;
**`v16-a55e952b8-r28`** (tip
`2983f72c`, tree `ae5aa3e0`) folded the **`--host-experts` flag into block 06**: the
`-ncmoe`/`-cmoe` host-expert backing becomes a first-class `--host-experts <pinned|mmap|auto>` option
(`llama_model_params.host_experts_mode`), default pinned unchanged, the legacy `LLAMA_MMAP_HOST_EXPERTS`
env kept.  `--host-experts mmap` keeps the expert master in the pageable model mmap instead of the pinned
`ROCm_Host` buffer (~5 % slower prefill, ~0.5 % decode, peak RSS 40.7 -> 22.2 GB).  Before it were
**`v16-a55e952b8-r26`** (PR #107's DFlash F1 default-on and PR #110's DPP butterflies) and
**`v16-a55e952b8-r25`** (eight of the nine PR #106 (issue #105) patches folded into blocks 06/08/14/15 plus
`TODO #45`; patch 0002, the data-pointer graph key, is dropped: it regresses on the r22+ slab).  Before it, **`v16-a55e952b8-r24`** (tip `46701e3ff`, tree `1a580f937`) fixed the early MoE
expert-cache floor and repacked the whole cache + arena/slab subsystem from blocks 13/14/15 into block 06
(no code change).  Before that, **`v16-a55e952b8-r23`** (tip `ef49781df`, tree `f652d71c`) was the diagnostics-hygiene
release and **`v16-a55e952b8-r22`** (tip `562e06f81`, tree `c0927a3ea`) folded the **movable-boundary slab
allocator** into block 15.
 (re-based **2026-10-05** from
`84e76d8a2`, itself re-based 2026-09-24 from
`ebbb18522`, itself re-based 2026-09-17 from
`d1d3c3396`, itself re-based 2026-09-15 from
`790cf51aa`, itself re-based 2026-09-13 from
`9113cc188`, itself re-based 2026-09-08 from `050dde50c`, itself re-based
2026-09-07 from `465e49b9c`, itself
re-based 2026-09-06 from `9cffdcc80`, itself re-based
2026-09-02 from `0eadefebd`). The `baseline/<sha>` branches below
are HISTORICAL checkpoints of the old pre-block-12 structure (patch
numbering 01-11 against older upstream ranges, `git apply` flow); they
remain as known-good records for those upstream versions.

> **Previous release (2026-10-06): `v16-a55e952b8-r21`** - the OPEN 1 safety subset: the r20 arena
> slot-count fix (`t.slots` must be the achieved count), per-layer uniform arena allocation,
> `MTP_DRAFT_N_UBATCH` default 512, and the wide-prefill drop default ON for `llama-cli` only
> (`-ub 8192` cache-auto 16k decode **78.7 t/s** / prefill **1683 t/s**, coherent).  Same fork point
> `a55e952b8`; canonical block-15 tip `94c3eeb89b4530dad9850cb29ce28bf296075b5a`, net tree
> `2cc89dfbe981abe2d858887c28cb9e25550edf99`; strict 16/16 `git am` (`validate-set.sh` green).  The
> `llama-server` half of the DoD is still open (OPEN 2).  See `WORKLOG.md` 2026-10-06 (r21).
>
> **Earlier release (2026-10-06): `v16-a55e952b8-r15`** - blocks 06 + 13: under `-sm tensor` the
> host-resident MoE experts no longer fall back to the CPU (the Meta device's host buft is null once the
> host bufts are per device, so the loader fell back to pageable `CPU_REPACK` and the scheduler's
> offload device pin skipped the Meta backend), and the expert cache's device-side admission policy is
> skipped for split tables (it is tuned for a whole, per-device expert).  2 GPU IQ4_NL
> `-sm tensor -ncmoe 48` MTP n3 `-n 3000`: 30.3 -> **88.0 t/s** (vs `-sm layer` 76.2); 3 GPU `-sm
> tensor` 99.1; byte-identical.  Same fork point `a55e952b8`, new canonical block-15 tip `7e2dcd8f1`,
> net tree `0e9273f846c4b22d0db4297ba84312f158bab088`; strict 16/16 `git am` (`validate-set.sh` green).
> See `WORKLOG.md` 2026-10-06 (r15).  (r13/r14 -- per-device host buffers, MoE-cache auto mode -- are in
> `WORKLOG.md`.)
>
> **Previous release (2026-10-04): `v16-a55e952b8-r11`** - contributor PR #102 in **block 15**: three
> RDNA4 verify-step fusions (GLU -> Q8_1, GDN conv at 2..255 tokens, batched state-snapshot copies),
> each bit-identical and default-on with a kill switch (`GGML_CUDA_FUSE_*`, `=0` off).  A 16,130-case
> geometry sweep is identical across on == off == stock r10, and an alternating A/B (27B UD-Q4_K_XL +
> DFlash2) measured +1.5 / +1.6 / +1.3 percent at 8K / 35K / 110K context with byte-identical output.
> Same fork point `a55e952b8` (tree `3550faf840a88ae652e5ff8d32067f28a836d87b`), new canonical
> block-15 tip `ea86588646930c016d9caa49509154f31e338c56`, net tree
> `38ebce2f738f9486a5fc1a95d26ca5a523bac902`; strict 16/16 `git am` (`validate-set.sh` green).  See
> `WORKLOG.md` 2026-10-04 (r11).
>
> **Previous release (2026-10-04): `v16-a55e952b8-r10`** - contributor PR #98 in **block 01**:
> `LLAMA_MTP_DRAFT_OP_OFFLOAD=0` keeps the MTP draft context's host-resident ops on the host, freeing
> the draft device expert copies when the expert cache is armed (draft compute 2054 -> 1444 MiB,
> post-load VRAM 20027 -> 19417 MiB here; ~1.1 GB on the reporter's larger head), opt-in with no
> default change and no throughput cost.  **Block 06** is a comment-only correction of the
> `SCHED_GATHER_TABLE_MIN_BYTES` gather figures (they came from the corrupted pass).  Same fork point
> `a55e952b8` (tree `3550faf840a88ae652e5ff8d32067f28a836d87b`), new canonical block-15 tip
> `b86854900`, net tree `dab5186bc0527508156507fd323a9109924cb03e`; strict 16/16 `git am`
> (`validate-set.sh` green).  See `WORKLOG.md` 2026-10-04 (r10).
>
> **Previous release (2026-10-04): `v16-a55e952b8-r9`** - issue #86 (block 12: the internal/hybrid HIP
> all-reduce is on by default on non-RDNA4, `GGML_CUDA_AR_ALLOW_NON_RDNA4=0` opts out, and a
> first-call NCCL failure fails over to the internal pipeline) and issue #99 (block 14: the gemma4
> `-sm tensor` guard is relaxed for all-resident and `-ngl`-offloaded loads; only host-resident
> experts and the MTP head are rejected).  Same fork point `a55e952b8` (tree
> `3550faf840a88ae652e5ff8d32067f28a836d87b`), new canonical block-15 tip `6d4ac7a52`, net tree
> `6cf4f5323691e429c69ff2d8a404749eb1f93fad`; strict 16/16 `git am` (`validate-set.sh` green).  See
> `WORKLOG.md` 2026-10-04 (r9), issues #86/#99.
>
> **Previous release (2026-10-05): `v16-a55e952b8-r8`** - issue #97, the H2D staging bandwidth
> calibration no longer runs for a split with no host-resident weight, in **block 06**.  Same fork
> point `a55e952b8` (tree `3550faf840a88ae652e5ff8d32067f28a836d87b`), new canonical block-15 tip
> `05bbd56e0`, net tree `af02d2d4bb9823fefa3a80d4a3e147c6ac5a48cc`; strict 16/16 `git am`
> (`validate-set.sh` green).  `sched_stage_issue()` ran the one-off calibration (512 MiB
> `cudaMalloc`, three timed copies, `cudaFree`) before its host-weight loop; on Windows the freed
> allocation is not returned to the per-process GPU counters, so a dense full-offload run stranded
> ~512 MiB for the session (a 27B Q5 at 161K ctx on a 32 GB R9700 spilled into shared memory and
> decode fell 46.8 -> 16.4 t/s).  `sched_stage_min_tokens_for()` now returns 0 when the split has
> no host weight, before the calibration is reachable; a split with a host weight is unchanged.  See
> `WORKLOG.md` 2026-10-05 (r8) and issue #97.
>
> **Previous release (2026-10-05): `v16-a55e952b8-r7`** - issue #93, the auto-sized H2D staging
> ring and table-size-scaled width gate, in **blocks 06 and 15**.  Same fork point `a55e952b8`
> (tree `3550faf840a88ae652e5ff8d32067f28a836d87b`), new canonical block-15 tip `27b6254e7`, net
> tree `77ee997c9fad231ea64ffb3a4247a1d158819b48`; strict 16/16 `git am` (`validate-set.sh` green).
> qwen4exp's 450 MiB host-resident expert tables overflowed the fixed 2048 MiB budget, so the fifth
> slot growth disabled staging for the run; the budget is now auto-sized when
> `GGML_SCHED_STAGE_MAX_MB` is unset, the gate is scaled by the host table size, a shortfall skips
> the split instead of disabling the ring, and the arena is counted in `--fit`.  The device gather
> is untouched and stays default-off.  On gfx1201 x4 (`-lzm off`): `pp8192 -ub 8192` 1570 -> 2374
> t/s (+51 %), `-ub 2048/4096` gated; staged == serial byte-identical.  See `WORKLOG.md` 2026-10-05
> (r7) and issue #93.
>
> **Previous release (2026-10-05): `v16-a55e952b8-r6`** - issue #95, the dynamic-backend (Docker)
> `-sm tensor` rejection for qwen4exp, fixed in **block 14**.  Same fork point `a55e952b8` (tree
> `3550faf840a88ae652e5ff8d32067f28a836d87b`), new canonical block-15 tip `1d10390a8`, net tree
> `2b57533c8002d11bd047c75a3323b30229f7f526`; strict 16/16 `git am` (`validate-set.sh` green).
> `ggml_add_backend()` only adds `GGML_USE_<backend>` to `ggml` when `GGML_BACKEND_DL=OFF`, and the
> containers build with `-DGGML_BACKEND_DL=ON` (for `GGML_CPU_ALL_VARIANTS`), so
> `src/llama-arch.cpp`'s `#ifdef GGML_USE_HIP` compiled to the HIP-absent branch and the qwen4exp
> gate rejected tensor split although the HIP backend was built.  `ggml-hip/CMakeLists.txt` now also
> publishes `GGML_USE_HIP` on `ggml`, covering both modes; the static build was already correct.
> See `WORKLOG.md` 2026-10-05 (r6) and issue #95.
>
> **Previous release (2026-10-05): `v16-a55e952b8-r5`** - contributor PR #96, the **block 06**
> scheduler re-stage fix for the MoE expert cache.  Same fork point `a55e952b8` (tree
> `3550faf840a88ae652e5ff8d32067f28a836d87b`), new canonical block-15 tip `b5ca42a92`, net tree
> `c5c716e796b29d770902ff2aecfeaba42e80f487`; strict 16/16 `git am` (`validate-set.sh` green).
> `ggml_backend_sched_split_graph` reused a weight copy across splits without re-registering it as
> an input, so a second `MUL_MAT_ID` consumer of host-resident expert weights in a later split read
> the copy the expert cache had taken over for its 1-row decode-band consumer and never filled.
> qwen4exp's unmasked MTP export hit this and NaN-ed `t_h_nextn`, collapsing MTP draft acceptance;
> the fix re-registers the weights as an input of the later split.  Reproduced end to end
> (446-token prefill, probe 141/149 -> **0/591** on r4, 141/149 with the fix).  See `WORKLOG.md`
> 2026-10-05 (r5) and `archive/work/sched-moe-restage/`.
>
> **Previous release (2026-10-05): `v16-a55e952b8-r4`** - the last two r1 follow-ups
> (`qwen4exp-qsa-convergence` + `lightning-indexer-fusion`) are resolved, in **blocks 14 and 15**.
> Same fork point `a55e952b8` (tree `3550faf840a88ae652e5ff8d32067f28a836d87b`), new canonical
> block-15 tip `cd1485fd1`, net tree `714f94f050dfce08c987a8a14467f456fe6e9d60`; strict 16/16
> `git am` (`validate-set.sh` green).  Block 14 keeps the fork's fused QSA graph and adopts
> upstream `10f340d1a`'s `hc_init` split fix
> (`archive/work/qwen4exp-qsa-convergence/DECISION.md`); blocks 14/15 register the fused
> indexer-score nodes as `LLM_FUSED_OP_LIGHTNING_INDEXER` (inert today) and the handover's
> `llama_prefetch_rows` item is dropped as a measured pp512 regression
> (`archive/work/lightning-indexer-fusion/RESULTS.md`).  See `WORKLOG.md` 2026-10-05 (r4).
>
> **Previous release (2026-10-05): `v16-a55e952b8-r3`** - the `mmq-prec-gate-fp4` and
> `shared-expert-fusion-reconcile` r1 follow-ups are resolved in **block 13**.  Same fork point
> `a55e952b8` (tree `3550faf840a88ae652e5ff8d32067f28a836d87b`), new canonical block-15 tip
> `3d1cd47f2`, net tree `25a8e137a585cd9fc2907a74236998f881635b8e`; strict 16/16 `git am`
> (`validate-set.sh` green).  The fused-gate MMQ now threads and asserts its `prec_src1`, and the
> MMQ pair fusion asserts its non-FP4 contract
> (`archive/work/mmq-prec-gate-fp4/RESOLUTION.md`); upstream's fused shared-expert MMVQ and block
> 13's `shexp_down_gate` are proven disjoint
> (`archive/work/shared-expert-fusion-reconcile/RESULTS.md`).  See `WORKLOG.md` 2026-10-05 (r3).
>
> **Previous release (2026-10-05): `v16-a55e952b8-r2`** - the first two r1 re-base follow-ups are
> resolved.
> Same fork point `a55e952b8` (tree `3550faf840a88ae652e5ff8d32067f28a836d87b`), new canonical
> block-15 tip `dbe88ea6e3afd86da26ce766ae8b71d2b26b67ac`, net tree
> `c38ba8f2066f01c3a1f69207a7e0860e5026ef17`; strict 16/16 `git am` (`validate-set.sh` green).
> Every block commit now builds (merge hygiene, `archive/work/rebase-merge-hygiene/RESOLUTION.md`),
> the DFlash device path / `common_sampler_clone` / fast-top-k audit is clean, and a new
> duplicate-value `ARGSORT` test fixed the CPU tie-break oracle
> (`archive/work/rebase-integration-audit/RESULTS.md`).  See `WORKLOG.md` 2026-10-05 (r2).
>
> **Previous release (2026-10-05): `v16-a55e952b8-r1`** - the 16-block set is re-based onto upstream
> master **`a55e952b8`** (203 commits since `84e76d8a2`).  The fork point is `a55e952b8` (tree
> `3550faf840a88ae652e5ff8d32067f28a836d87b`), the canonical block-15 tip `def454e4c` and net tree
> `6a44aa2904772db02dbc88960397efe8138498df`; strict 16/16 `git am` (`validate-set.sh` green).  The
> re-base folds upstream's batch-API migration, probabilistic draft sampling, bitonic-argsort refactor,
> `rms_norm`+`scale` fusion, fused shared-expert MMVQ, BF16 `ggml_cuda_cast` and `ggml_prec prec_src1`
> MMQ parameter, and upstream's own qwen4exp MTP/kpool/mask/indexer/`-sm tensor` work; block 08's
> standalone `rms_norm_scale_f32`/`GGML_CUDA_FUSE_RMS_SCALE` fusion is retired (subsumed).  See
> `WORKLOG.md` 2026-10-05 (r1).
>
> **Previous release (2026-10-05): `v16-84e76d8a2-r37`** - the BF16 hyper-connection mixer fusion
> (contributor PR #91, @briansp2020) is folded into **block 15**.  The ISTA-DASLab GSQ-RCO quants keep
> `hc_{attn,ffn}_{down,up,inject}` in BF16, and block 14's fused `GGML_OP_HC_MIX` was Q8_0-only, so
> those models ran the six-dispatch unfused chain (96 mixers per token) on a dispatch-bound decode.  The
> new BF16 arm replays that chain in three dispatches with the same per-thread K order and reductions
> (bit-identical); no weights are converted and it engages only for GPU-resident BF16 hc weights at
> `hc_lr == 320` (`LLAMA_HC_MIX_BF16=0` keeps the chain).  The CPU `HC_MIX` reference gained the
> matching BF16 arm, and the new `test-backend-ops` BF16 cases (`HC_MIX` 30/30 = 20 Q8_0 + 10 BF16)
> exposed and fixed a pre-existing no-inject dst-stride bug at `nt > 1`.  Output-preserving (fused ==
> unfused byte-identical for `nt` 1/3/5/8; +3.8 % `tg256` on the 4-layer fixture); strict 16/16 `git am`,
> `validate-set.sh` green.  Tip `f39945172993fa4a7517b6a5af8821d7eef36c3a`, net tree
> `ea5f8012f30d1aef94f1b3057ae58897fff0d61a`.  See `WORKLOG.md` (2026-10-05 r37) and
> `archive/work/rdna4-hc-mix-bf16/`.
>
> **Previous release (2026-10-05): `v16-84e76d8a2-r36`** - the issue **#89** indexer top-k block-path
> fix (contributor PR #90) is folded into **block 15**.  The fused indexer top-k's block fast path
> (`indexer_topk_radix_cuda_blocks`) partitioned its block-level radix passes by block range while the
> gather partitions by cell range; the histogram-derived per-range `g_cnt`/`e_cnt` bases only match when
> block `b`'s cells are `[b*r, b*r+r)`.  A unified KV holding several sequences, or a single sequence
> whose KV head has moved past cell 0, broke that: output entries were left unwritten or overwritten,
> and `flash_attn_qsa` then gathered K/V at stale out-of-range indices (GPU page fault, server hang) or
> silently attended to the wrong cells.  A new `indexer_topk_count_cells_grouped` kernel recounts
> `g_cnt`/`e_cnt` over exactly the cell ranges the gather walks after the last radix pass, with the
> gather's key logic; the from-cell-0 output is unchanged and the block-level passes are kept.  A new
> `INDEXER_TOPK` backend-op case (`test_indexer_topk_block`) fails on the unfixed r35 build. 
> Output-preserving: 4B `7386359e5dac`, 35B-A3B `cf7f8b23f404`; `INDEXER_TOPK` 3/3, `TOPK_QSA` 4/4,
> `LIGHTNING_INDEXER` 225/225, `FLASH_ATTN_QSA` 26/26, `MUL_MAT_ID` 929/929; strict 16/16 `git am`,
> `validate-set.sh` green.  Tip `9b8b6f10815d285cd7f8828ab431686937873085`, net tree
> `c595f29253ad70d693793d010f5e5399dadf57ae`.  See `WORKLOG.md` (2026-10-05 r36) and
> `archive/work/issue-89/`.
>
> **Previous release (2026-10-04): `v16-84e76d8a2-r35`** - the issue **#87** synchronous graph-input fix
> is now default-on in **block 06**.  The reporter confirmed the r33 A/B candidate clears their crash
> (the r26 async split-input H2D path copies straight from the host pointer, and the recurrent-state
> copy `rs_s_copy` is always consumed through views that lose `GGML_TENSOR_FLAG_INPUT`, so the copy
> races the host overwrite on the next ubatch).  `GGML_SCHED_SYNC_GRAPH_INPUTS` is now an opt-out: unset
> = enabled, `=0` restores the r26 async behaviour (A/B / bisect).  Host-weight uploads keep the
> async/staged path.  Output-preserving: 4B `7386359e5dac` with unset and `=0`, and 35B-A3B
> `cf7f8b23f404`; `MUL_MAT_ID` 929/929; strict 16/16 `git am`, `validate-set.sh` green.  Tip
> `b01620f2de060d546b945786a2eee4fc04cd0248`, net tree `d08fbaf2ca842ea3c3ce044ac45c0a8d0f11c597`.
> Blocks 07-15 are rebased onto the amended block 06.  See `WORKLOG.md` (2026-10-04 r35).
>
> **Previous release (2026-10-04): `v16-84e76d8a2-r34`** - the issues **#59/#60** qwen4exp fix folded into
> **block 15**, promoting the `archive/work/issues-59-60` gfx1100 candidate (built on r27) onto r33.  **#59:**
> qsa3 is off on RDNA3_0 by default (`GGML_CUDA_QSA3=1` opts in, `=0` force-off), and a packed QSA op's
> support equals the qsa3 predicate, so the qwen4exp graph probes the backend and skips the two
> natural-F16 K/V packs it would otherwise only waste (RDNA3_5/RDNA4 defaults unchanged).  **#60:** the
> 4-head lightning-indexer prefill score is supported on RDNA3_0 again, and `build_qsa_top_k`'s
> `use_wmma` takes the fused op once the score exceeds `LLAMA_QSA_SCORE_WMMA_MB` MiB (default 64, `0` =
> always fused) and keeps the faster chain below it; the chain's `mul_mat+relu` and its chunked form now
> use `ggml_relu_inplace` (bit-identical, removes the 2x-score reserve peak).  Also counts the FA prefill
> staging arena in `llama_get_memory_breakdown`/`--fit` (issue #33 follow-up).  Candidate gates on
> gfx1100: `FLASH_ATTN_QSA` 23/23 (26/26 with `GGML_CUDA_QSA3=1`), `LIGHTNING_INDEXER` 225/225,
> `TOPK_QSA` 4/4, `FLASH_ATTN_EXT` 6354/6354, dense 27B same-seed `1acb04bd9104` identical to r20;
> rebased onto r33 with no conflicts; strict 16/16 `git am`, `validate-set.sh` green.  Tip `33a8c30db`,
> net tree `3c07e1f6e303efa59a92d0d63d2acf5e30666cb2`.  See `WORKLOG.md` (2026-10-04 r34).
>
> **Previous release (2026-10-04): `v16-84e76d8a2-r33`** - a **default-off** block-06 A/B candidate for
> issue #87.  The r26 async split-input H2D path copies straight from the host pointer, and the
> recurrent-state copy `rs_s_copy` is always consumed through views that lose `GGML_TENSOR_FLAG_INPUT`,
> so the copy races the host overwrite on the next ubatch.  `GGML_SCHED_SYNC_GRAPH_INPUTS=1` forces the
> synchronous user-input branch for (views of) graph inputs while leaving host-weight uploads
> asynchronous; unset keeps the r26 behaviour.  Output-preserving: 4B `7386359e5dac` and 35B-A3B
> `cf7f8b23f404` with the variable both unset and `=1`; `MUL_MAT_ID` 929/929; strict 16/16 `git am`,
> `validate-set.sh` green.  Tip `13a3b1353`, net tree `14444e869d75871514d2aa99924264386014d55a`.  Blocks
> 07-15 are rebased onto the amended block 06.  See `WORKLOG.md` (2026-10-04 r33).
>
> **Previous release (2026-10-04): `v16-84e76d8a2-r32`** - four contributor PRs folded into **block 15**:
> **PR #78** (gfx1201 mmvq dispatch-stall workaround: a 1792-block row loop for single-token
> Q4_K/Q5_K/Q6_K/IQ4_XS dense decode, per-type rows-per-block, 2 rows at 4..8 verify tokens for Q8_0
> short-K), **PR #83** (the GSQ-RCO kernels on top of #78: BF16 `mul_mat_vec_f` unroll and
> `mul_mat_vec_f_vb`, IQ2_S/IQ3_S `apply_ksigns`, IQ3_S/IQ2_S in the row loop, IQ2_XXS/IQ2_S/Q2_0
> routed-compact MoE mmq), **PR #84** (x86 AVX2 `ggml_vec_dot_q2_0_q8_0`, bit-identical) and **PR #81**
> (issue #80: the exact top-k fast path plus a clone without the candidate copy, `GGML_LF_FAST_TOPK=0`
> opts out).  All bit-exact/output-identical.  Tip `9d46b0966`, net tree
> `b090750760c58cc4c2271cbf4d260fe0413c52a3`; gates: warning-free build, `MUL_MAT_ID` 929/929, `MUL_MAT`
> 1297/1297, CPU `MUL_MAT` 1323/1323, 4B `7386359e5dac` and 35B-A3B `cf7f8b23f404` identical to r31,
> sampling on == off `c118179c57ec`, strict 16/16 `git am`, `validate-set.sh` green.  See `WORKLOG.md`
> (2026-10-04 r32).
>
> **Previous releases:** `r31` (2026-10-03) fixed the two MMQ `MUL_MAT_ID` tail over-read holes; `r30`
> (2026-10-02) was the **block-13 expert-gather head-pad fix** for
> the repeated-`/` incoherence r29 shipped: the always-on host-resident-expert device gather's one-time
> expert-head zero was hard-coded to 64 bytes (the *IQ4_NL* threshold the beta5 session measured), so
> IQ4_XS over-read further and poisoned the tile (on one GPU as well as multi-GPU; the beta5 gates only
> used IQ4_NL / Q8_0 / Q4_K_M).  It now uses the host path's `min(expert_size, 512)` and keys the zero on
> `(input_cpy buffer, expert_bytes)`; prefill unchanged.  Canonical tip
> `6bba985363599e8dd92290ca32a1fb15876bbaf2`, net tree `0fe48395051775079fb18041142e3f22dbf82a72`; new
> per-quant gate `scripts/gate-qwen4exp-quant-coherence.sh`.  See `WORKLOG.md` (2026-10-02 r30).
>
> **Previous release (2026-10-01): `v16-84e76d8a2-r29`** - the **decode-side MoE expert cache** (the
> `archive/work/moe-expert-cache` campaign, PR #82) folded into blocks 06/13/14/15: a per-device VRAM arena of hot
> experts over the pinned host pool (opt-in `MOE_EXPERT_CACHE_MIB`, unset = inert and bit-identical to r28),
> with the model-aware expert gather (a `>= 224 MiB` table gathers instead of staging the whole shard) and
> the decode-band-gated routed-expert rebalance always on.  Blocks: 06 the generic backend iface + the
> scheduler half, 13 the engine + the `mmvq.cu` slot lookup, 14 the gemma4 `-sm tensor` guard, 15 the CUDA
> consumer glue.  Canonical tip `8e16c882ad8ebe6d7f3498e5940758f2d8802611`, net tree
> `65276106fc5a6f62e1d81f4975c4816012ea4fc4`; gates: strict 16/16 `git am`, warning-free build,
> `MUL_MAT_ID` 929/929, byte-identity to the `-ncmoe 0` oracles, width purity `none == n1 == n3 == n7`, MTP
> `n3` 0.75273.  **gemma4 `-sm tensor` is now rejected** (use `-sm layer`).  See `WORKLOG.md` (2026-10-01
> r29) and `archive/work/moe-expert-cache/` (fold record + the COMMUNITY-CONFIG.md config guide).
>
> **Previous release (2026-09-30): `v16-84e76d8a2-r28`** - a block-15 amendment fixing the VMM pool
> free-order abort (issue #76, PR #77 by overdoingism): `kq_blocks` was declared after
> `dst_tmp`/`dst_tmp_meta` but allocated before them, so a batch that also allocated `dst_tmp_meta` freed
> `kq_blocks` out of stack order and aborted in `ggml_cuda_pool_vmm::free`; the fix only moves the
> declaration.  Reproduced and fixed on gfx1201 / ROCm 7.14 with `-DGGML_HIP_NO_VMM=OFF`
> (`test-backend-ops -o FLASH_ATTN_EXT` aborts before, 6354/6354 after); the default build's 4B
> `-sm tensor` same-seed text is unchanged.  Canonical tip
> `60361cb9f90437f7070e6f6b04ab673c85af7ddd`, net tree
> `dc2decae2a6ec8c95562c0d9a2fe53eb1ac49b63`.  See `WORKLOG.md` (2026-09-30 r28).
>
> **Previously (2026-09-30): `v16-84e76d8a2-r27`** - a block-15 amendment collecting four contributor
> PRs and the issue-#71 RDNA4 rows fix (PR #68 dense SWIGLU -> mmq down projection bit-exact; PR #73 DFlash
> device-resident layer features, `GGML_LF_DFLASH_DEV=1`, opt-in pending a model-level gate; PR #74 ksplit
> mmvq verify recursive-halving reduce, bit-exact; PR #75 qwen4exp `HC_MIX` band kernels + general RDNA4
> decode/prefill kernels, the IQ2_XS/IQ3_XXS routed-compact mmq gated to RDNA4; issue #71 returns 1 row for
> `ncols_dst >= 2 && nwarps > 1`).  Gates on gfx1201 / ROCm 7.14: `MUL_MAT_ID` 929/929, `MUL_MAT`
> 1297/1297, `HC_MIX` 20/20, `GATED_DELTA_NET` 46/46, 4B coherence `1c5d32ac537d` and qwen4exp
> `359ff4337837` identical to r26, 27B q8_0 width probe `P = 4000` byte-identical with `width_purity=PASS`.
> Canonical tip `7fe4fca497f8ef2c6e440d5405a95452cdd3c230`, net tree
> `7427f424fbd3b7e1b2fbf807d81a04fe43caf373`.  See `WORKLOG.md` (2026-09-30 r27).
>
> **Previously (2026-09-30): `v16-84e76d8a2-r26`** - canonical tip
> `0d58404e16aa076521091f1b1e2f8d2d88bff5c3`, net tree `afbdc436059b11b9a18b9ac6e6481c40a28327d9`
> (block-06 amendment: the op-offload prefill H2D staging ring now overlaps - `GGML_SCHED_EVENTS` defaults
> ON and the staging gate counts host-weight inputs; qwen4exp IQ4_NL 8K prefill `-ub 8192` ~870 -> ~1090
> t/s, output-preserving).  See `WORKLOG.md` (2026-09-30 r26).
>
> **Previously (2026-09-29): `v16-84e76d8a2-r25`** - canonical tip
> `81fda69c81a48d48ac386d2f7175ec82cfda23ee`, net tree `c7385cd5f03d16b462ef9b586959188b8f1556e6`
> (block-15 `#pragma clang fp contract(off)` in `rope.cu`, making the address-gated `ROPE -> VIEW ->
> SET_ROWS` fusion bit-transparent; issue #67).  (Issue #67 closed 2026-10-03 as external: the residual
> cross-start flip was hipBLASLt solution selection, `ROCm/rocm-libraries#12126`, workaround
> `ROCBLAS_USE_HIPBLASLT=0` - see `WORKLOG.md` 2026-10-03 and `GREEDY-PURITY.md` §41.)
>
> **Previously (2026-09-29): `v16-84e76d8a2-r24`** - two default-off kill switches for the two
> address-overlap-selected rope fusions (`GGML_CUDA_DISABLE_ROPE_SET_ROWS=1`,
> `GGML_CUDA_DISABLE_RMS_NORM_MUL_ROPE=1`), added to block 15 for issue-#58 item D.  Default path
> byte-identical; the switches only bisect the address-gated fusion that decides the W=1 decode logits on
> the reporter's model.  Canonical tip `667ff09476e55f3ddeed4fd56e6ba8305b990a2c`, net tree
> `94b60ec74e8ebc230c87b0b600b7cc9aa59b8a49`.  See `WORKLOG.md` (2026-09-29 r24) and `GREEDY-PURITY.md` §41.
>
> **Previously (2026-09-29): `v16-84e76d8a2-r23`** — canonical tip
> `eb567e04ba79c773c096e4ced8ad2dfeda1df87d`, net tree `7fa881011c7794b3cbdf2a6fd041bdb85aaddb80`
> (PR #64 by @briansp2020 folded into block 15: a wide FA-band block for query widths 5..8, an `ncols`
> template for `ssm_gate_beta_fused_q8_0`, and the residual ADD folded into `rms_norm_q8_1` for 2..8 tokens;
> each bit-exact and default-on with its own kill switch).  Re-verified on ROCm 7.14 / gfx1201.  See
> `WORKLOG.md` (2026-09-29 r23).

> **Previously (2026-09-29): `v16-84e76d8a2-r22`** — canonical tip
> `c0356818289975b8eccd9fb70314cf9c5bdb35f7`, net tree `c63060dc5dfd17a72cd697d70279db38c8d6ec8c`
> (the issue-#65 block-15 amendment caching the `getenv()` lookups on the fusion and staging hot paths:
> `LLAMA_HC_CN_DEBUG` on every candidate fusion window, `GGML_CUDA_DISABLE_CONV_FUSION` on every conv
> launch, plus the other per-op/per-graph debug gates).  Weak `getenv` on Windows cost ~15 ms per graph
> and ~10 % decode; behaviour is unchanged and the clean-build warnings are fixed.  See `WORKLOG.md`
> (2026-09-29 r22).

> **Previously (2026-09-28): `v16-84e76d8a2-r21`** — canonical tip
> `feefecfbcd4ddaec32895dd67a9ea48b8e44eaba`, net tree `9975a333d3d785da662dfcc9b601c442d3be8104`
> (r20's `6f8369bf…` + three contributor PRs by @briansp2020: #57 blocks 10+13 multi-row mmvq verify
> blocks + exact `__mul24`, #62 block 15 the RDNA4 GQA-6 FA band's 64-wide K/V batches + 8 warps, #63
> blocks 08+14 five bit-exact verify-band fusions).  Each was re-verified on ROCm 7.14 / gfx1201; see
> `WORKLOG.md` (2026-09-28 r21, PR #57/#62/#63).

> **Previously (2026-09-27): `v16-84e76d8a2-r15`** — canonical tip
> `e40c70ec326a533592758bc0bdb58cd7f4733340`, net tree `d609d34d1d78ddf21c00c5b6b119ab29693aa3b8`
> (r14's `7790b606…` + the r15 block-06 host-expert pinning fix: `MUL_MAT_ID` host weights are no longer
> downgraded to the mmap, so op-offload expert uploads read pinned memory — **+83 %** on `-sm tensor
> -ncmoe` pp8192, bit-identical output).  See `WORKLOG.md` (2026-09-27 r15).

> **In `main` since 2026-09-25 (release `v16-84e76d8a2-r8`; then r9, 2026-09-26):**
> the fork point is unchanged at **`84e76d8a2`**, but the 16 patches now absorb the 28
> `archive/work/mmb-general` patches, so the applied tree is r8's campaign tree
> **`24bb0f5acb3e866abd4cad8c0de1bad45a20cb47`** plus r9's issue-#47 typed-store fix ->
> **`a3dc4bbb680bf9dd8bcb5949ec833dec2a892aeb`** (canonical tip `b48fb3f686fe2681f55aa406a8ed52313ad80875`,
> release `v16-84e76d8a2-r9`).  See `archive/work/beta-integration/integration.md` and `WORKLOG.md` (2026-09-26 r9).

> **Current baseline marker:** the current delivery baseline is **`84e76d8a2`** (2026-09-24),
> the fork point of release **`v16-84e76d8a2-r7`** (canonical tip
> `596a22dbfbde571728e93acf986a02200aaf46ee`, tree
> `7726e514284ea7393bb9097ce305dc5b6dacdb11`; r7 = the 2026-09-25 block-14 Meta-tensor-split scheduler
> race fix under `-sm tensor` - the multi-device no-sync gallocr re-reserve guard now treats a `META`
> backend as multi-device, fixing the intermittent `quantize_q8_1` fault on a secondary GPU;
> 28-patch beta re-cut onto r7, applied tree `24bb0f5acb…`)
> on top of r6 = the 2026-09-25 block-15 bf16 native default flip,
> which makes native bf16 the default so a bf16 cache takes the RDNA4 GQA-6 decode/verify band
> (`GGML_CUDA_FA_KV_NATIVE=0` is the single kill-switch; bf16 kv 16384 verify widths 1.5-2.3x faster,
> +14 % `draft-mtp n3` at ~30k; 28-patch beta re-cut onto r6, applied tree `1df5769c…`)
> on top of r5 = the 2026-09-25 block-15 follow-up that extends the
> RDNA4 head-256 GQA-6 decode/verify FA band (issue #45) to f16 (and bf16 through its native arm)
> with a per-K/V-element-size config - 2-byte types `ncols1 = 2` / `P = max(2, 3*nsm/4)`,
> native-quantized `ncols1 = 4` / `P = nsm`; f16 verify widths 2.1-3.2x faster, +13 % `draft-mtp n3`
> at ~30k, plain f16 decode -2.5..-4.2 %
> on top of r4 = the 2026-09-25 block-15 RDNA4 head-256 GQA-6
> decode/verify FA band (issue #45: the whole `n_q <= 8` band runs the WMMA kernel with the GQA group
> folded into `ncols2 = 8` and a round-robin KV split over a fixed `P` blocks, so decode and
> every verify width reduce identically; all native quantized K/V types, default on, prefill untouched)
> on top of r3 = the 2026-09-25 block-14 `hc_combine`
> CPU-reference fix (issue #44: the CPU reference used `t*ne[1]`/`t*hc` row strides instead of the
> tensors' own `nb[1]`, corrupting every multi-token fused ubatch on a CPU-resident qwen4exp layer -
> now mirrors the CUDA kernel, bit-identical at nt == 1) plus the 28-patch `archive/work/mmb-general` re-base
> onto r3 (applied tree `0daefe22…`) and its gfx1100 routed-band fix (the 16-wide `mul_mat_vec_q_moe`
> band is now RDNA4-only - it failed `MUL_MAT_ID` 23/929 on RDNA3_0 and is a measured loss on
> RDNA3_5).  On top of r2 = the 2026-09-25 block-10 amendment that scopes the wide-VDR MoE expert
> entry points to RDNA4/RDNA3_0 with one arch gate, so RDNA3_5 (gfx115x) uses the dense VDR as the
> block-10 comment intended - the per-quant gate had let the Q4_K/Q6_K experts leak.  On top of r1 =
> the 2026-09-24 re-base onto upstream master `84e76d8a2`, 149 upstream commits past `ebbb18522`,
> blocks 10/14/15 resolved manually - see `WORKLOG.md`).  `baseline/ebbb18522` was the previous (2026-09-17) delivery baseline, last
> released as **`v16-ebbb18522-r13`** (canonical tip
> `8491bf2bff8eb3a56e5120c3c9c17533a94ea6bf`, tree
> `bb7b6d07b05ad8e23ab6e770172e7f597cfb3c12`; r13 = the 2026-09-22 block-00 amendment that gates the
> MTP `is_mem_shared` inference on the `gemma4-assistant` arch, so shared-NextN heads
> (`nextn_shared_target_tensors`, e.g. the qwen4exp shared sidecar) keep their own KV instead of
> dying on the M-RoPE `X < Y` check - upstream bug `04eb4c446`, #23398 - on top of
> r12 = the 2026-09-21 block-06 amendment that makes `--fit`
> support `-sm tensor`, promoted from `beta/tensor-fit-fix/` - upstream threw "not implemented for
> SPLIT_MODE_TENSOR" and swallowed it, so the default-on `--fit` was a silent no-op under tensor split.
> Re-validated on r11 first: fit decisions, seven end-to-end loads with no out-of-memory and no
> compute-buffer growth, byte-identical same-seed gate - on top of
> r11 = the 2026-09-20 block-15 amendment that makes the
> compute reserve account for the *reachable* (packed) kq mask, issue #42 — V3's derived form is
> per-*batch*, so a 2-D M-RoPE or multi-sequence batch allocates a mask the reserve did not contain,
> and the growth died under the default `--fit-target 256`; `kq_mask_packed_reachable()` decides where
> the mask is reserved (M-RoPE / `n_seq_max > 1`), so same-seed output stays byte-identical, throughput
> is unchanged, non-M-RoPE single-sequence models keep V3's reserve, and the reporter's M-RoPE model
> pays -8960 tokens (-4.4 %) — on top of
> r10 = the 2026-09-20 block-11 pre-fill token-count
> fix + the HIP `hipGraphExecUpdate` leak guard, issue #41, on top of r9's block-15 V3 tile-kernel
> derived mask, which sits on r8/r7 (block 15), r6 (FA build time), r5 (block-04 head cap) and r4
> (block-04 RDNA3_0 tensor-split `ncols2`) — base `ebbb18522` unchanged from r2).  `baseline/790cf51aa` remains the
> (2026-09-13) marker.  None of these is one of the historical pre-block-12
> checkpoints listed below; those remain frozen records.

> **Naming collision warning:** in the OLD records below, "block 12"
> sometimes means the old *k-quant umbrella* (folded into what is now block
> 10) and sometimes the *hybrid all-reduce* (the current block 12). In the
> current delivery, block 12 = the hybrid all-reduce, period.

Older branches (historical): `baseline/fe235f434` (the gfx12/gfx11
segregation baseline, validated on gfx1100, gfx1151, gfx1201),
`baseline/192067b72` (same patch files as `d222767c7`, zero-fuzz-validated
at `192067b72`), `baseline/d222767c7` (validated against `d222767c7`) and
`baseline/758443071` (the original set for the older upstream range).

## Baseline (current delivery)


All 16 patches are generated against **llama.cpp upstream master at
`ebbb18522`** (re-based **2026-09-17** from `d1d3c3396`, itself re-based 2026-09-15 from `790cf51aa`,
itself re-based 2026-09-13 from `9113cc188`, itself re-based 2026-09-08 from `050dde50c`, itself re-based
2026-09-07 from `465e49b9c`, itself re-based
2026-09-06 from `9cffdcc80`, itself re-based
2026-09-02 from `0eadefebd`; dated records at the
bottom of this file): block 00 = the structural/architecture fixes added
2026-09-10 (FA small-batch KV-split width invariance + Vulkan masked-V), and
blocks 01-15 = the fork's `rdna-boosts` block
commits.  The canonical 16-block chain for the `ebbb18522` base is tip
`ba9e18cacfa3f97f13a822dded971eeb2cce2480`, tree
`b84b1783f7207e25600403df5a8e98c183b9f80a` (release **`v16-ebbb18522-r4`**, the 2026-09-18 block-04 RDNA3_0 tensor-split `ncols2` fix for issue #30 on top of r3's 2026-09-18 block-01 `--fit` fix for issue #38 on the same base;
r2 was the 2026-09-17 re-base of the 16-block set onto upstream master `ebbb18522`, 37 commits past
`d1d3c3396`, whose three resolved blocks were block 02's Vulkan GATED_DELTA_NET check-results clone
(moved upstream to `ggml-vulkan-debug.cpp`), block 12's upstream HIP AllReduce enablement (the delivery
keeps its HIP split: `allreduce.cu` CUDA-only, HIP hybrid in `allreduce-hip.cu`), and block 14's
upstream qwen4exp hc ops (the delivery's decode-band fused hc ops keep `nt <= 8`, upstream's
`ggml_dsv4_hc_pre_gated`/`post` serve prefill) plus the pair-fusion `ncols_opt` RDNA3 consistency fix;
see `WORKLOG.md` 2026-09-18 (r3) and 2026-09-17).
The previous canonical chain (base `d1d3c3396`, 2026-09-15) was tip
`8465f08b9efb26c60e992b48b7d2857d9ffcaf7a`, tree
`3bb7c223c60570978d1bbf996a03808fe31f2842`.  Before that, the `790cf51aa` base:
tip
`6f76c1cb1d80c7ecbf176f939a351bc385ff33fc`, tree
`d735d6c11258ae939cfd392511e3f29ac22a7686` (the 2026-09-15 **build-time** block-15 amendment: the tile
kernel's native-KV type axis is instantiated in the generated instance TUs again instead of implicitly
in the dispatch TU — a clean `-j16` backend build 538 s -> 330 s, `fattn-tile.cu` 509 s -> < 10 s, no
runtime change; `archive/work/build-time-regression/`; release `v16-790cf51aa-r5`).  The previous canonical tip
was `b19c70b341f9ed439bcda2a636fe6e5fa4fa634b`, tree
`7fab975d9518b29aa7d890c1163f13a6c393c5df` (the 2026-09-15 block-15 amendment — issue #30's second
round: the mixed-K/V kernel contract, the `get_alloc_size` q4_0 fix, the prefill band split + staging
arena + the RDNA3_5 arch gate, and the `q4_1`/`q5_0`/`q5_1`/`iq4_nl` native arms; release
`v16-790cf51aa-r4`).  Before that, `a2c8d06a7931c9f6bec8542fe10149c615853be7`, tree
`eb5b7583d14b30b7610fac53acf2fc52bc806ce4` (the 2026-09-13 re-base + the block-08 (sixth)
`iq4_nl` `GET_ROWS` sub-`QK_K` amendment, TODO item 3, and the block-08 (seventh) MoE-router
bit-identity amendment, TODO item 19; the four
upstream clashes and the FA head-to-head are recorded in `WORKLOG.md` and
`patches/README.md`).  The previous base was `9113cc188` (canonical tip block 15
`0f4f83f9e`, promoted 2026-09-12 from `archive/work/block-15-campaign-wins/`, on top of the 2026-09-12 (13) block-14 QSA indexer-score decode/verify band-uniformity fix -> `d306d4b4b`; block 02 amended 2026-09-11 with the
K-independent whole-batch chunked GDN prefill and again 2026-09-12 with the rollback-bounded
chunked threshold (`n_rs_batch`, long-draft speculators) + the pre-batch snapshot slots (free; the
`GGML_CUDA_GDN_ALIGN_BOUNDARY` gate and its K-dependent branches removed; +
rollback guard), block 08 amended 2026-09-11 with the decode/verify FA kernel-family fix, the
quantized-KV-type enablement (`q4_1`/`q5_0`/`q5_1`) and the `iq4_nl` enablement (predicate, the 15
new vec instances, `dequantize_q4_nl`, the non-contiguous converters), and block 13 amended
2026-09-11 with the dense ncols==1 ksplit alignment (decode/verify bit-identity), again
2026-09-12 with the RDNA3_5 single-token-only mmvq fusion skip, and block 14
amended 2026-09-11 with the hyper-connection decode/verify band fix, the QSA decode arm, the QSA
quantized-KV enablement + K/V-head chunking fix and the `iq4_nl` QSA/CPU-oracle/test entries;
**block 15, the attention-memory campaign, is the last delivery patch** -- promoted 2026-09-12 from `archive/work/block-15-campaign-wins/`,
see `patches/README.md` and `WORKLOG.md`); the reference `~/llama.cpp`
`rdna-boosts` branch is *disposable* and had at cut time drifted two
upstream master commits past the fork point — `f3f1a8f27` (iGPU lazy-
load default) and `304665fe7` (SYCL IQ-type-for-MoE), both dated after
`9113cc188` — so `format-patch 9113cc188..<that branch's tip>` would
wrongly export those two upstream commits as patches 0001/0002.
**Always regenerate from a canonical fork rebuilt at `9113cc188` via
`scripts/apply-all.sh`** (that is what `make-patches.sh`'s default tip
`0f4f83f9e` refers to).  The two commits' content is 106 lines in 3 files
(`ggml/src/ggml-sycl/mmvq.cpp`, `ggml/src/ggml-sycl/vecdotq.hpp`,
`src/llama-model.cpp`) and is deliberately **not** in the delivery — it
is upstream code past the recorded fork point; it does not touch any
validated path; block 01 refreshed 2026-09-09 to the llama.cpp PR
#27210 review head `d236d41a2` and block 14 amended 2026-09-10 with the
kernel-side masked-V fixes — the 2026-09-09 gfx1151-only freed-cell
KV-row-zeroing host gate it replaces is removed — see the WORKLOG entries; block 06 now
carries only the host-buffer
rationale marker — upstream #28604 reverted #24233 on 2026-09-08,
matching its end state; block 12 amended 2026-09-04 with the runtime
NCCL-failure fallback (issue #13) and 2026-09-11 so the hybrid dispatch's
small/large crossover does not change the reduction algorithm with the batch
width (`-sm tensor`, 2-device `32768` -> `131072` elements); block 13 amended
2026-09-02 with
two MTP regression fixes, 2026-09-05 with the RDNA3.5/RDNA3.0 gate
relaxations and 2026-09-06 with the model-neutral Strix MoE mmq folds,
block 14 (qwen4exp support) promoted from `beta/qwen4exp` 2026-09-07
and amended 2026-09-07 with the QSA quantized-KV decode gate + the
derived-cache pool gate
— see `MANIFESTS.md` / `patches/README.md`
block-12/13/14 notes; the previous `465e49b9c`-based
regeneration `45bf4d291..c261553a1` is superseded and preserved on the
fork remote's history).
`scripts/make-patches.sh` regenerates both. Verified 2026-09-02 and
re-verified 2026-09-02 after the block-13 amendment, 2026-09-04
after the block-12 amendment, 2026-09-06 on the `465e49b9c` re-base
and 2026-09-07 on the `050dde50c` re-base + block 14, and 2026-09-07
after the block-14 QSA quantized-KV gate + pool gate amendment: clean
apply (`git am` 01-15) on a fresh checkout at
`050dde50c`, full build clean, llama-cli same-seed coherence IDENTICAL
(hybrid vs RCCL) — and the
apply is **whitespace-free** (zero git warnings).  Re-verified 2026-09-08
on the `9113cc188` re-base: strict clean apply (14/14 `git am`, zero
warnings, applied tree == fork tip `78e67a3d8`) on a fresh `9113cc188`
checkout, full ROCm build clean, llama-cli same-seed coherence IDENTICAL
to the `72f0ee944` build on gfx1151 (tensor/layer x f16/q8_0/bf16 KV,
depth 16384), dense MTP adaptive gate green (acceptance 0.833; draft-mtp
20.3 vs plain 7.9 t/s).  Re-verified 2026-09-09 after the block-01
refresh to the PR #27210 review head: strict clean apply (14/14 `git
am`, zero whitespace warnings, applied tree == fork tip `0f2b7a4e1`)
on a fresh `9113cc188` checkout, rebuilt unit tests
(`test-arg-parser`, `test-speculative-adaptive`) pass, llama-cli
same-seed coherence IDENTICAL to the known-good `050ec89ce` build on
gfx1201.  Re-verified 2026-09-09 after the block-14 gfx1151-zeroing-gate
amendment: strict clean apply (14/14 `git am`, zero whitespace warnings,
applied tree == fork tip `27485f1ca`) on a fresh `9113cc188` checkout;
content built + validated on gfx1201 (stall A/B + zeroing-off determinism
gate) and the gate enable path on the gfx1151 Halo box.  Re-verified
2026-09-10 after the block-14 kernel-side masked-V amendment (host
freed-cell zeroing removed): strict clean apply (14/14 `git am`, zero
whitespace warnings, applied tree == fork tip `ff2b35f49`) on a fresh
`9113cc188` checkout; rebuilt ROCm tree passes the 16/16 determinism
gate on the gfx1151 Halo box (all supported KV types, both backends —
see the WORKLOG 2026-09-10 entry).

## Two fixes vs the fork

The patch set carries two fixes that are NOT on `chunked-gdn`; both come from
the fork's `rdna-boosts` branch (the production lineage) or from this
validation:

1. **Test-harness seeding (folded into block 02).** The fork's chunked-GDN
   work carried a deterministic seed into `init_tensor_uniform` in
   `tests/test-backend-ops.cpp` (added while debugging the bf16 GDN kernel):

   ```cpp
   // fork (chunked-gdn): static std::atomic<unsigned> g_seed(12345); (void) g_seed;
   // thread_local std::default_random_engine gen(12345 + (unsigned) start * 101);
   // this branch:         thread_local std::default_random_engine gen(std::random_device{}());
   ```

   The fixed seed correlates tensor data across rows and deterministically
   exposes a pre-existing numerical fragility in `rms_norm_back` and
   `cross_entropy_loss_back` on RDNA4 (CPU/GPU comparison fails with fixed
   seeds; passes with `random_device` seeding). GDN results are unaffected
   (46/46 in all configs either way). Full diagnostic: fixed seeds 12345 and
   54321 both fail those 9 cases; only the seeding line differs in the
   passing build.

2. **Meta-buffer compute-container headroom (block 09, `f2a22a71`).**
   `compute_headroom` 16x -> 128x in `ggml-backend-meta.cpp`. Hybrid
   recurrent models (GDN/SSM) create ~2*(n_rs_seq+1) conv-state snapshot
   views per recurrent layer during graph allocation, exceeding 16x and
   aborting with "not enough space in the context's memory pool"
   (ggml.c:1804). Without it, speculative MTP drafting under
   `--split-mode tensor` crashes on first decode. Source commit is on the
   fork branch `rdna-boosts`, not `chunked-gdn`; upstream has not fixed it
   either (reproduced on pristine `d222767c7`).

## Per-block provenance

The CURRENT delivery patches (0000-0015) are the fork's `rdna-boosts` block
commits exported with `git format-patch` (one commit per block; the
current 16-block set against `9113cc188`:
block 00 = the structural/architecture fixes added 2026-09-10 (FA
small-batch KV-split width invariance for issue #25 + Vulkan masked-V);
blocks 01-13 = the `7c4d9c4e0`-based series (block 01 refreshed 2026-09-09 to the llama.cpp
PR #27210 review head `d236d41a2`, squash — see the WORKLOG entry) with
block 03 amended 2026-09-10 (HIP masked-V fixes, re-homed from block 14)
and block 14 amended 2026-09-10 (the kernel-side masked-V fixes were
re-homed — Vulkan to block 00, HIP to block 03; the freed-cell host zeroing
is removed; tip `daf32f804`; block 02 amended 2026-09-11 with the
K-independent whole-batch chunked GDN prefill (free, no gate, + rollback guard)
and block 13 amended
2026-09-11 with the dense ncols==1 ksplit alignment; block 14 = the qwen4exp-support delta promoted
from `beta/qwen4exp`; re-based 2026-09-08 from the `050dde50c` set
`90a816a68..3bebffd6b` (block 06 reduced to a marker — see the WORKLOG
re-base entry); previously the
re-based set against `465e49b9c`: `45bf4d291..c261553a1`, against
`9cffdcc80`: `04122bfb5..8f2838d1`, block 13
amended 2026-09-02 with the two MTP regression fixes; previously the
re-based set against `0eadefebd`: `b25bc8a9c..a14257996`;
re-based regeneration `4c0f30dec..8fbf10e5b` against `a7cc83bba`; the
whitespace-clean regeneration `3209e83b4..cc985ba9a` against
`17252c769`; originally `2b7a135cb..f6f8f6778`; the original fork history
is preserved on `old-rdna-boosts`). Block 12 is the hybrid HIP all-reduce delta over four
files (RDNA4-gated). The ORIGINAL source commits on the fork branch
`chunked-gdn` (the pre-consolidation lineage) and the old `baseline/*`-branch
checkpoint history moved to `archive/docs/baseline-history.md`.

## Drift policy

The patches are static against the fork point `9113cc188`. If a patch fails
to apply against a newer upstream master:

1. Try `git am -3` / `git apply -3` (3-way merge against the baseline blobs).
2. If 3-way fails (or the fork-state pre-image blobs are not in the local
   clone — the normal case for a fresh puller), rebase the failing hunks
   manually against the current
   master and continue.
3. Do NOT hand-edit the committed patches as the permanent fix: when more
   than one block needs manual re-base hunks, regenerate the whole set from
   the fork with `scripts/make-patches.sh` (re-exports blocks 00-15 from
   `9113cc188..<blocks-tip>`; defaults target
the current 16-block tip `0f4f83f9e`), then re-verify the clean-apply
simulation (fresh worktree at the new fork point, `scripts/apply-all.sh`,
build, coherence) and update the fork point + verification numbers in
`patches/README.md` and `README.md`.

---

## Re-baseline to a7cc83bba (2026-08-30, dated record)

Upstream master moved 24 commits past the fork point `17252c769` (6 of
them touching ggml-cuda). The fork's `rdna-boosts` branch was rebuilt from
the delivery patches on the new base (blocks 01-07 and 09-12 applied
cleanly; block 08 needed a manual merge) and the set regenerated with
`scripts/make-patches.sh` (base `a7cc83bba`, blocks tip `8fbf10e5b`,
block 12 committed as `4fa92f0ae`).

The one real conflict: upstream's **SWIGLU_CLAMP (#27930)**, landed one
day after the old fork point, added `glu_limit` plumbing to the same
mm-fusion machinery block 08 rewrites (the `ggml_cuda_mm_fusion_args_*`
structs in `common.cuh`; four regions of `mmvq.cu` — the `active_glu`
decls, the fusion-assign block, the GLU-switch/result-write restructure,
and the `fusion_local` copy). Resolution: upstream's `glu_limit`/
`SWIGLU_CLAMP` additions were kept alongside block 08's fields, with the
SWIGLU_CLAMP case relocated inside block 08's restructured switch
(`result_val`). Verified by diffing the merged files against block 08's
post-image blobs: the difference is exactly upstream's additions, nothing
else.

Verified end-to-end 2026-08-30: clean-apply sim on a fresh clone at
`a7cc83bba` (`scripts/apply-all.sh`, zero conflicts + zero whitespace
warnings), full build clean, llama-cli same-seed coherence IDENTICAL to
the pre-re-base known-good build. tg64 38.12 / tg512 41.08 (sim build)
unchanged — the re-base is code-identical to the 2026-08-29 set plus
upstream's SWIGLU_CLAMP additions.

---

## Cross-version apply to 0eadefebd (2026-09-01, dated record)

Upstream master moved **22 commits** past the fork point `a7cc83bba`; only
**3 touched ggml-cuda** — `e4b9af007` (XOR-swizzle flash-attn K/V smem
fp16 tiles, #25635), `f8dbcd618` (ROCm radix TOP_K for long rows,
#27466), `41ef91f7c` (MOE fusion extended to specdec, #27621) — all in
block 08 / block 10 territory. Applied the 12-patch set to a fresh clone
checked out at `0eadefebd` (branch `rdna-boosts`):

- Blocks 01-07, 09-11: `git am` clean. Block 12: `git apply` clean.
- Block 08 (fused core): the ONE conflict — `git am -3` 3-way merge
after fetching the fork's blobs (the clone lacked the patch's index
blobs); **auto-resolved, zero manual hunks**. Verified per the
post-image-blob protocol: the two merged files (`ggml-cuda.cu`,
`mmvq.cu`) diff vs block 08's post-image blobs = exactly upstream's
additions (content-identical after stripping index/hunk headers).

**Full-tree zero-drift check:** `fork-tip → HEAD` differs from
`a7cc83bba → 0eadefebd` in exactly the same **51 files**, and all 51
diffs are content-identical — the applied tree is byte-faithful to the
fork delivery tip `4fa92f0ae` (blocks tip `8fbf10e5b` + block 12
`4fa92f0ae`) plus exactly the upstream drift.

**Verified end-to-end 2026-09-01:** full build clean (ROCm 7.14 gfx1201,
`GGML_HIP_RCCL=1`, graphs+native; zero patch-related compiler warnings);
llama-cli same-seed coherence **IDENTICAL between hybrid and RCCL**
(3-GPU tensor split, `GGML_CUDA_ALLREDUCE=nccl` comparison) — the
coherence gate passes on the new master.

**Fork point decision:** the delivery set **remains static against
`a7cc83bba`** — per the drift policy, regeneration / formal re-baseline
is triggered only when *more than one* block needs manual re-base hunks;
here only block 08 needed a 3-way merge and it auto-resolved with zero
drift. The verified applied state is preserved on the `rdna-boosts`
branch of the `~/prs/llama.cpp` clone (upstream `0eadefebd` + 12 blocks,
tag `rdna-boosts-0eadefebd`). If a future drift event ever needs the fork
point moved, follow the regeneration path above (`scripts/make-patches.sh`
with base `0eadefebd`, blocks tip `9c2463ff8`/`221b0c804` in that clone).

---

## Re-baseline to 0eadefebd (2026-09-01, dated record)

**Maintainer decision (same day): move the fork point to `0eadefebd`** —
`scripts/apply-all.sh` must apply cleanly against a fresh upstream
checkout, and with the old base's patch context it does not (block 08
fails with plain `git am`; only `git am -3` works). The record above's
"keep static" recommendation is superseded. Full re-baseline performed
per the drift policy step 3:

- **Fork rebuild:** `~/llama.cpp`'s `rdna-boosts` was deleted and
  rebuilt on `0eadefebd` (worktree; blocks 01-07 + 09-12 `git am`
  clean, block 08 `git am -3` auto-3way, block 12 `git apply` +
  commit). New commits: blocks `217e33ba4..d7bdd0a91`, block 12
  `ce9182473`. The rebuilt tree is **byte-identical** to the verified
  2026-09-01 cross-version apply above. The old `a7cc83bba`-based fork
  state (tip `4fa92f0ae`) is preserved on the `rdna-boosts-a7cc83bba`
  branch.
- **Set regenerated:** `scripts/make-patches.sh` (base `0eadefebd`,
  blocks tip `d7bdd0a91`) re-exported blocks 01-11 + the block-12
  delta; `rdna-boosts-all.patch` regenerated as
  `git diff 0eadefebd..ce9182473`. Folding upstream's changes into the
  patch context means the regenerated block 08 now applies with plain
  `git am` — **`apply-all.sh` is clean again on fresh master**.
- **Re-verified 2026-09-01:** clean-apply sim on a fresh clone at
  `0eadefebd` (`scripts/apply-all.sh`: **zero conflicts, zero
  whitespace warnings**; sim tree byte-identical to the fork tip), full
  build clean (ROCm 7.14 gfx1201, RCCL+graphs+native), llama-cli
  same-seed coherence IDENTICAL to the pre-re-base known-good build,
  tg64 38.12 / tg512 41.08 (numbers unchanged — code-identical
  content). Re-measured 2026-09-01 on the sim build (27B Q8_0,
  3-GPU tensor, r2): tg64 36.87 ± 4.83 / tg512 40.72 ± 1.02 — matches
  the documented numbers within noise (prs build: 37.92 ± 4.67 /
  40.90 ± 0.86).
- **Tooling fix:** `scripts/make-patches.sh`'s checkout check now
  accepts git worktrees (`[ ! -e "$FORK/.git" ]` instead of `-d`),
  which is how the fork rebuild is hosted.

## AR_PROFILE devices[] init fix + fork re-sync (2026-09-01)

- **Fix:** block 12's `allreduce-hip.cu` now fills `p->devices[]` from the
  caller list before the per-device profiler hipMallocs (PR #8).  With
  `GGML_CUDA_AR_PROFILE=1` the buffers were allocated while `devices[]`
  was still zero-filled, so all landed on GPU 0 and MTP's second pipeline
  (draft context) faulted/hung GPU 1 on gfx1201.  Pre-fix A/B reproduced
  the fault on 3x R9700 (2-GPU, internal AR, MTP n-max 3, `-c 32768`);
  post-fix runs clean with profiler teardown dumps on every device;
  coherence IDENTICAL to the pre-fix golden.  Integrated into the fork's
  block-12 commit and regenerated into
  `patches/0012-…-hybrid-HIP-all-reduce-RDNA4-gat.patch` + `rdna-boosts-all.patch`.
- **Fork re-sync:** the fork branch was rebuilt as a 12-commit branch
  directly on `0eadefebd` (block 01 `b25bc8a9c` .. block 11 `43f5ab71d`,
  block 12 `7d5d3f77b`, block 13 `a14257996`), dropping the upstream
  `kleidiai` docs commit
  `518b76236` that had crept into the previous rebuild (upstream-only;
  remains in `origin/master`).  The delivery contract is unchanged: 13
  blocks applied to a fresh checkout at `0eadefebd`.
- **Re-verified 2026-09-01:** clean-apply sim (`apply-all.sh` on a fresh
  clone at `0eadefebd`): zero whitespace warnings, applied tree
  byte-identical to the fork tip (`d42fc80…`).  Fork build clean (ROCm
  7.14 gfx1201, `cmake --build build-rocm --config Release -j 16 --
  VERBOSE=1`) + coherence + the A/B above.

## MTP chunked-GDN prefix folded into block 02 (2026-09-01, PR #9)

- **Change:** block 02 now also runs its chunked WMMA GDN on long
  single-sequence MTP prefills (`K > 1`): chunked on the prefix
  (`n_tokens - K`), sequential GDN only on the last K snapshot slots
  (PR #9; folded into the block-02 commit, NOT a new patch block).  The
  chunked ops take an `n_tokens_limit`; opt out `GGML_CUDA_GDN_CHUNKED=0`.
- **Verified 2026-09-01 (3x R9700, 2-GPU, internal AR, Qwen3.8-27B Q8,
  ubatch 1024, MTP n-max 3):** path fire `n=1024 K=4 prefix=1020`;
  prefill +7.5% (~5.5k) / +7.7% (~38k) vs sequential; 64-token
  same-seed output token-identical; non-MTP coherence unchanged.
- **Fork state:** block 02 amended (`cbc219af4`), blocks 03-12 replayed
  unchanged; blocks tip `43084332f`, block 12 `7d5d3f77b`.  Set
  regenerated (blocks 01, 03-11 content-identical; 0002 = old 0002 +
  PR #9 hunks); clean-apply sim re-verified (applied tree byte-identical
  to the fork tip `b90eb525e`).  Full clean build passes.


## Re-baseline to 9cffdcc80 (2026-09-02, dated record)

Upstream master moved **42 commits** past the fork point `0eadefebd`. This
re-base was performed as a **clean-room exercise** (simulating a first-time
puller of this repo): the fork was rebuilt in `~/llama.cpp` (a fresh
upstream clone at `9cffdcc80`) by applying the old 13-patch set with plain
`git am` — no fork-blob 3-way merges, no access to the previous fork
checkouts. Failed hunks were resolved **by hand** per the drift policy step
2. Three blocks needed manual re-base hunks (all in fattn-tile.cuh /
ggml-cuda.cu):

- **Block 03 vs upstream #27970 (sparse-fa, `8e93a9773`):** a 4th bool
  (`use_sparse`) was added to `launch_fattn` and the fattn-tile call sites
  updated; block 03 rewrites the same sites (type_KV threading + runtime
  `need_f16_K`/`need_f16_V`). Merged: `need_f16_K, need_f16_V, false,
  false, warp_size` per site (upstream's `stream_k`/`use_sparse` stay
  false); the `launch_fattn_tile_switch_ncols2` template gained `type_KV`.
- **Block 08 vs upstream #25952 (fused MoE expert reduction,
  `3466812d1`):** upstream's new `GGML_OP_MUL` weighted-reduction arm in
  `ggml_cuda_try_fuse` shifted block 08's rms_norm->mmvq quantize-fold arm
  context; the arm now sits after upstream's (order-independent — arms are
  mutually exclusive on `node->op`).  A second, latent issue was caught by
  the coherence gate, not the build: block 08's spec-verify `launch_fattn`
  call site in fattn-tile.cuh still passed the pre-#27970 3-bool arg list,
  binding the `warp_size` int into the new `use_sparse` bool slot
  (compiles silently; `use_sparse=true`) -> runtime
  `GGML_ASSERT(n_kv_max > 0)` in fattn-common.cuh. Fixed to the 4-bool
  form and folded into the block-08 commit.
- **Block 13 vs upstream #25952:** the `disable_moe_mmq` opt-out static +
  `const int cc` decls at the top of `ggml_cuda_try_fuse` (context shifted
  by upstream's inserted arm) restored after the MoE arm.

**Set regenerated:** `scripts/make-patches.sh` (base `9cffdcc80`, blocks
tip `92f09e80a`) re-exported blocks 01-13; `rdna-boosts-all.patch`
regenerated as `git diff 9cffdcc80..92f09e80a`. Folding upstream's
changes into the patch context means **`scripts/apply-all.sh` applies all
13 blocks with plain `git am` — zero conflicts, zero whitespace warnings**
on a fresh checkout at `9cffdcc80`.

**Re-verified end-to-end 2026-09-02:** clean-apply sim on a fresh worktree
at `9cffdcc80` (applied tree byte-identical to the fork tip `92f09e80a`),
full build clean (ROCm 7.14 gfx1201, `GGML_HIP_RCCL=1`, graphs+native,
zero errors — build note: `EXTRA_CMAKE_FLAGS="-DCMAKE_HIP_FLAGS="` is
required with CMake >= 4.3, whose HIP compiler test injects
`--cuda-host-only` directly after the build script's bare `-mllvm`),
llama-cli same-seed coherence **IDENTICAL between hybrid and RCCL**
(3-GPU tensor split, Qwen3.5-4B Q8_0). tg64 38.12 / tg512 41.08 unchanged
— the re-base is content-identical to the `0eadefebd` set plus upstream's
additions. The previous `0eadefebd`-based fork state (tip `482837e5a`)
remains on the `stew675/llama.cpp` fork remote (`rdna-boosts`); older
reference checkpoints are preserved in `~/prs/llama.cpp`.

## Re-baseline to 465e49b9c (2026-09-06, dated record)

Upstream master moved **18 commits** past the fold-verified base
`8b4b3558f` (57 past the old delivery fork point `9cffdcc80`; the
ggml-cuda-touching ones were `73a43d1f6` mmid/mmf race fixes #28475 and
`5fdfa6282` GDN l2-norm fix #28068).  The `~/llama.cpp` fork
(`rdna-boosts`) was rebuilt from `patches/` via `scripts/apply-all.sh`
on the fresh master tip: 13/13 `git am` clean, **zero conflicts, zero
whitespace warnings** — both upstream ggml-cuda commits landed in
disjoint hunks; no manual merges.  Applied-tree content check on all 112
upstream-touched files passed (deltas == old-fork + upstream drift); the
14 extra differing files are exactly the 2026-09-06 Strix fold delta.
Set regenerated with `scripts/make-patches.sh` (base `465e49b9c`, blocks
tip `c261553a1`; canonical am-commits `45bf4d291..c261553a1`);
`rdna-boosts-all.patch` refreshed (45 files — the previous copy was
stale at 41, pre-fold).  Two prerequisites: restored the format-patch
mail headers the 0044cfe fold had stripped from `0002/0004/0008/0013`
(delivery commit 0610b75) and re-dated the block-13 message's
fold-amendment trailer to the fold's true date (block-13 tip amended
`b4b760eb8` -> `c261553a1`).  Re-verified 2026-09-06: clean-apply sim on
a fresh checkout at `465e49b9c` (zero conflicts/whitespace warnings;
applied tree byte-identical to the fork tip `c261553a1`).  The
`qwen4exp` fork branch was rebuilt on the new base (`465e49b9c` + blocks
+ the consolidated `beta/qwen4exp/qwen4exp-support.patch`, clean 3-way,
46 files, zero conflicts — fork tip `627506c1c`).  Campaign date
re-stamp: the gfx1151 campaign docs had run a week ahead of the real
calendar; all `wip/`/`beta/`/archive dates were collapsed onto the real
git dates (2026-09-05/06) and the moved benchmark records'
`benchmarks/2026-09-*` references repointed at
`archive/work/wip-archive/qwen4exp/discovery/`.

## Re-baseline to 050dde50c + block 14 (2026-09-07, dated record)

Upstream master moved **22 commits** past `465e49b9c` to the 2026-09-07
master tip `050dde50c` (the ggml-cuda-touching ones: `b74f590ea` f16
flash-attention divergent-barrier fix #27870, `73ab7599b` branchless
Q4_K/Q5_K mmvq unpack + L2 prefetch #26705, `473599738` gfx90c HIP
support #26454).  The `~/llama.cpp` fork was rebuilt on the new base:
blocks 01-13 `git am -3` — 12 auto-merged, **one manual conflict** in
`tests/test-backend-ops.cpp` (block 04's Q6_K/WMMA-flash-attn perf cases
vs upstream's new LEAKY_RELU perf cases at the same spot; both kept).
Then **block 14 (qwen4exp support) was promoted from
`beta/qwen4exp/qwen4exp-support.patch`** — the squashed fork delta
`c261553a1..dd4301fb4` — applied with `git apply --3way`:
**one manual conflict** in `ggml-cuda/common.cuh` (upstream's gfx90c
GCN-APU arch macros vs the block's exact-SKU
`GGML_CUDA_CC_IS_GFX1151` predicate; both kept).  Canonical am-commits
on the new base: `90a816a68..3bebffd6b` (block-14 tip `3bebffd6b`).
Set regenerated with `scripts/make-patches.sh` (base `050dde50c`, blocks
tip `3bebffd6b`); `rdna-boosts-all.patch` refreshed (87 files).
Re-verified 2026-09-07: clean-apply sim on a fresh checkout at
`050dde50c` (`scripts/apply-all.sh` 14/14 `git am`, zero conflicts /
whitespace warnings; applied tree byte-identical to the fork tip
`3bebffd6b`), full build clean (ROCm 7.14 gfx1201, RCCL+graphs+native),
test-backend-ops 6759/6759 (MUL_MAT / MUL_MAT_ID / FLASH_ATTN_EXT),
test-llama-archs 617 OK / 0 fail incl. qwen4exp (GPU 9.21e-14 / CPU
0.00), llama-cli same-seed coherence (3x R9700 gfx1201; dense 27B Q8_0
and qwen4exp IQ4_XS — numbers in `patches/README.md` block-14 notes).
Block 08 was amended same-day with the **PR #15** mul_mat+add
through-view shape guard (community report + fix, DanoPTT — single-seq
fusion untouched; author-validated on their single R9700, deployed to
production 2026-09-07): the fix was folded into the block-08 commit
and the set regenerated again (fork tip moved to `3bebffd6b`);
clean-apply sim re-verified (tree byte-identical), full build clean,
test-backend-ops 6759/6759, dense 27B same-seed byte-identical pre vs
post fix, 3-GPU hybrid == RCCL IDENTICAL, parallel 2-slot llama-server
decode clean on the dense 27B and qwen4exp IQ4_XS (no asserts).

## Re-base to 3f03b1772 (2026-10-24, r127)

Upstream (the maintainer's fork `master`) moved **261 commits / 868 files (+96 173 / −48 196)** past
`097cf33d9` (2026-10-12) to **`3f03b1772`** (2026-10-24).  Unlike every `rN` since V2.0 this was a
**re-base**, not a block amendment: the 14 blocks were replayed with `git rebase --onto` in nine rungs
(checkpointed `rebase-ckpt-1..9`, one per upstream cluster), each rung resolved semantically and gated
before the next.  The per-rung conflict tables, the SUBSUMES/CORDON/INTEGRATE decisions and the halo
recipes are the campaign record at `archive/work/rebase-2026-10/` (`STATE.md`, `DECISIONS.md`,
`GOTCHAS.md`, the two surveys); the release note is `WORKLOG.md` 261024.008.

What upstream had implemented in parallel, and what was decided:

* **chunked GDN prefill** (`62df1815b`, #29353) — kept as a *fallback behind* the delivery's chunked
  kernel; on gfx1151 both are eligible at ≥2048 tokens and ours measured +3.2 % (pp2048 interleaved) with
  prefill-logit KLD 0.000074 / 99.637 % against the MMA arm.  One chunked kernel per device class.
* **MoE cache SLRU** (`33d6135cb`, #30637) — SUBSUMES for the function body; the delivery's own
  `MOE_EXPERT_CACHE_*` engine is independent and stays.
* **4 GiB pinned-host cap** (`b6334e37e`, #30449) — INTEGRATE: the cap now applies to the delivery's
  per-device host buft too, so a `-ncmoe 40` master allocates as 5 pinned parts in one multi_buffer.
* **chunked CUDA graphs** (`280141e6d`, #30804) — INTEGRATE: the AR-stage hook moved to the last chunk's
  capture and the HIP `hipGraphExecUpdate` workaround took upstream's new signature; 3-GPU 27B MTP decode
  84.2/84.8 vs 79.8 t/s.
* **fused QSA mask op** (`d454a204e`, #30640) and **QSA without kq_mask** (`5c3114268`, #30965) — CORDON
  both times: qwen4exp keeps the delivery's derived-visibility selection (no mask tensor, −800 MiB), and
  the QSA oracle (`FLASH_ATTN_QSA` 26/26, sparse-vs-dense PPL 2.2399/2.2429 on gfx1151) is intact.
* **`hc_stream_mean`** (`d9ee007a0`, #30927) — SUBSUMES the delivery's `graph::build_hc_mean()` for
  qwen4exp (identical node list); deepseek4/glm5-next keep theirs.
* **rms_norm float4** (`623c16393`) — adopted; it moves the greedy text at a near-tie (prefill-logit
  0.000657 / 98.755 % against the pre-r127 base: inside tolerance) and the prefill-logit base was
  re-recorded from the r127 build once, as the campaign README required.

Chain hygiene: the campaign's four fixup commits were folded into the blocks that own them and dropped, so
the delivered chain is exactly 14 commits again; `scripts/gate-per-block-build.sh` is **14/14** (it caught
a cross-block helper definition, a stray brace and the `ml`/`ml_l` rename fallout, all fixed in their
blocks).  Canonical chain rebuilt by `scripts/apply-all.sh` at `3f03b1772`: tip `0c28ef86c`, tree
`f46264b1fe565a016940e6fc7858cf50e3b11a9d`, 14/14 strict `git am`; `validate-set.sh` green.
Set regenerated with `scripts/make-patches.sh` (base `3f03b1772`, tip `0c28ef86c`) and `release.json`
written by `scripts/make-release.sh` (`v14-3f03b1772-r127`).

Gates re-run at the new base: per-block build 14/14; dense 27B / 35B-A3B / Flash-Next / DFlash2 coherence
(`plain == <speculator>` byte-identical; 58.0 vs 106.7 t/s on Flash-Next MTP); prefill-logit PASS then
re-recorded; op tests FLASH_ATTN_EXT 6402/6402, FLASH_ATTN_QSA 26/26, GATED_DELTA_NET 55/55, HC_MIX 30/30,
TOP_K_MASK 28/28, MUL_MAT 2445/2445, MUL_MAT_ID 931/931; `test-llama-archs` qwen4exp/deepseek4/glm5-next
5/5 each; band gate in the rung-3 band.  **gfx1151 (Strix Halo, halo) got its first full validation**:
build 0 errors, the GDN A/B and prefill-logit above, the QSA sparse-vs-dense ratio, Flash-Next
`plain == draft-mtp` (34.8 → 57.8 t/s), DFlash2 `plain == draft-dflash` (61.6 → 104.8 t/s) and the five
op tests at the same counts as gfx1201.

One documented claim was corrected on the evidence: a **dense** `-sm tensor` split is *not* guaranteed to
produce the same greedy text across device counts (27B UD-Q4_K_XL: 3 GPU 850 chars vs 2 GPU 876; r126
shows 897 vs 855, so it predates the re-base).  `AGENTS.md` and `TODO.md` now say so; every `-sm layer` arm
and every within-count comparison is unaffected.
