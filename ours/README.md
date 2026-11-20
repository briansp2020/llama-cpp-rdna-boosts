# ours: our production stack on top of the current rdna-boosts release

This branch holds the patch series we run in production, on top of the latest rdna-boosts release. It is
regenerated for each new release; patches the release has taken are dropped. It is not a PR.

**Base:** `v17-4f9dfa0e3-r321` (upstream llama.cpp `4f9dfa0e3` + this repo's release).
The rest of this branch is that release as published; only `ours/` is added.

**New layout (2026-11-08).** Up to r228 this was a 55-patch series (0000-0054) that replaced the release's
`moe-expert-cache.cu` with its own r206 copy. From r235 on, the release's `moe-expert-cache.cu` stays as released, and our
`--host-experts mmap` code (RAM tier, pin budget, KV lending, fill fence, batched admissions, device remap for sparse
tables, ...) lives in two included files, `moe-expert-tier.inc` and `moe-expert-tier-state.inc`, called from about 130
hook lines in the release's file. Pool and pinned tables therefore run the release's own code; mmap tables run ours.
The old series is kept as the tag `ours-r228-series`; the mapping table below shows where each of its patches went.

**Apply:**

```
git checkout 4f9dfa0e3
bash <path-to-this-repo>/scripts/apply-all.sh .     # the r321 release, branch rdna-boosts
git am <path-to-this-repo>/ours/*.patch              # 48 patches (0001-0048)
```

Result: tree `2196617d7a6d`. This is the production build from 2026-11-20 (2 GPUs, `-sm tensor`, ROCm 10.2).

## Patches

The release has taken the r251 series' 0018 (DeepSeek-V4 cache size probes) and 0021 (DeepSeek-V4.1 kq-mask probe, issue
#300, fixed in r254); they are dropped here. Earlier releases took the r236 series' 0009, 0011, 0014-0015, 0017 and 0018.
r272 took our copy-engine all-reduce single exchange (PR #304, r265 0012); it is dropped here. r279 fixed the
`-sm tensor --load-mode none` load race (issue #308, `GGML_META_SEG_DRAIN`), so our drain (r277 0037) is dropped too.
r291 took our qwen4exp sequence parallelism (PR #315: on by default, bf16 all-gathers by default, at least 128 tokens per
split); the r287 series' 0022-0024, 0027 and 0028 are dropped and the later patches move up.
r302 took our QSA indexer top-k fast path (PR #320, r299 0044) and r304 our GDN gate/beta epilogue (PR #322, r299
0038-0040, with the release's own Q8_1-activation fix); they are dropped here and the later patches move up.
r313 took our Q8_0 hc mixer (PR #326, r305 0032, 0036 and 0037); they are dropped here and the later patches move up.
The numbers below are the r321 series: the r305 series rebased onto r321 (one conflict, `gdn-conv.cu`: our pair/match
split of the conv+norm fusion kept, r321's context parameter added), plus 0042-0048.

| | what | env switch (off) |
|---|---|---|
| 0001 | everything of the r228 series outside `moe-expert-cache.cu`, squashed (scheduler, server, MoE / GDN / shared-expert fusions, dedup, slab, KV lending hooks in llama, ...; see the mapping table) | per feature, as before |
| 0002 | our mmap expert-cache code in `moe-expert-tier.inc` / `moe-expert-tier-state.inc`, not yet called | - |
| 0003 | the hooks: the release's `moe-expert-cache.cu` calls the tier code for sparse (mmap) tables | per feature, as before |
| 0004 | the device remap declines when a fill failed during its own policy pass | - |
| 0005 | a declined decode-band sparse gate+up+GLU runs unfused | `GGML_MOE_FUSED_DECLINE_UNFUSE=0` |
| 0006 | review fixes: the tier remap predicate refuses a `fill_failed` table; the down fold also runs unfused for a declined sparse table | - |
| 0007 | header comment | - |
| 0008 | layer split with pipeline parallelism overlaps the devices on ROCm: prompt ubatches rebuild their graph instead of reusing it, and host inputs go through a pinned per-(backend, copy) ring uploaded on the split's stream; opt-in `GGML_GALLOC_FP_SHAPES=0` and `LLAMA_KV_N_PAD_PREFILL_POW2=1` | `LLAMA_PP_PREFILL_NO_REUSE=0`, `GGML_SCHED_INPUT_STAGE=0` |
| 0009 | a table whose fill failed also runs gate+up+GLU unfused | - |
| 0010 | drops the 0005 workaround (`GGML_MOE_FUSED_DECLINE_UNFUSE`): the release's fusion overlap check fixes the cause | - |
| 0011 | the dual-output mmvq fusion skips weights that live in a compute buffer | `GGML_CUDA_FUSION_CHECK_LEAVES=0` |
| 0012 | the checkpoint pool uses `common_state_data` (no-init allocator); the pool thread still touches every page | - |
| 0013 | CUDA graph eviction counts active time only (idle gaps capped at 1 s), and keeps the rule that the looked-up graph is never evicted | - |
| 0014 | qwen4exp: PLE page hints (`MADV_WILLNEED`) only when a `mincore` sample says the table is cold (~13 ms per 4096-token ubatch saved) | `LLAMA_PLE_ADVISE=1` (always advise) |
| 0015 | multi-device: `alloc_splits` skips the drain when no buffer grows | `GGML_SCHED_DEFER_ALLOC_SYNC=0` |
| 0016 | `-sm tensor`: graph inputs in pinned memory (the meta device's first child host buffer type) and uploaded asynchronously per device; host compute buffers kept across the drop | `LLAMA_META_HOST_BUFT=0`, `GGML_SCHED_ASYNC_USER_INPUT=0`, `GGML_GALLOC_DROP_HOST=1` |
| 0017 | meta: the split-input copy of a split host weight is sized at the largest device slice, not the whole tensor | `GGML_META_SLICE_ALLOC=0` |
| 0018 | moe cache: a device-remap call does not re-batch a sibling already taken over in the same round | `GGML_MOE_DEVREMAP_REBATCH=1` |
| 0019 | qwen4exp qsa3: Q8_0 cache -> F16 packs in one copy; the gate copy runs before attention on prefill-sized graphs | `LLAMA_QSA3_DIRECT_F16=0`, `LLAMA_GATE_EARLY=0` |
| 0020 | qwen4exp qsa3: the packs read the Q8_0 cache views directly, no F16 copies in the graph | `LLAMA_QSA3_PACK_Q8=0` |
| 0021 | `-sm tensor` ubatch overlap: asynchronous graph-input views, asynchronous CPU-split output upload, deferred input wait | `GGML_SCHED_ASYNC_INPUT_VIEWS=0`, `GGML_SCHED_ASYNC_CPU_OUT=0`, `LLAMA_DEFER_INPUT_WAIT=0` |
| 0022 | decode MoE gate+up+GLU fusion when the GLU output overlaps only the unfilled weight copy (41 of 47 layers were blocked) | `GGML_MOE_FUSE_COPY_ALIAS=0` |
| 0023 | moe cache: identity (fully resident) tables are served in place above the band in mmap mode too; fixes NaN at small ubatches since r258, with a warn-once `MOE_INPLACE_MISS` tripwire | `MOE_EXPERT_PREFILL_IN_PLACE=0` |
| 0024 | moe cache: a table is pool-backed only when the pool was requested | `GGML_MOE_POOL_TABLE_REQ=0` |
| 0025 | mmq: RDNA4 IQ routed-compact experts with 65-96 rows per expert use J=64 (+1-2 % prefill at `-ub 4096`, bit-identical) | `GGML_MMQ_IQ_J64_WIDE=0` |
| 0026 | moe cache: the release's in-place prefill stands down when a sparse table exists (only reachable with a GTT arena and mixed tables; inert on discrete GPUs) | - |
| 0027 | moe cache: KV segment -- under `GGML_MOE_KV_LEND=1`, the lent KV rows past the context in use are an expert-cache tier below the arena (heat-seeded, then demand-filled from the RAM tier; the arena evolves as without it); every device lends its own K/V slices under `-sm tensor` | opt-in `GGML_MOE_KV_SEG=1` (production sets it) |
| 0028 | KV segment: one take-back per request (the server hints the prompt length) instead of stepwise takes during long prefills | `GGML_MOE_KV_SEG_TAKE1=0` |
| 0029 | KV segment: small prefill ubatches (decode band < n < 512) read non-resident experts from their KV copies | `GGML_MOE_KV_SEG_PF=0` |
| 0030 | KV segment: review fixes (source counters, the hint returns to 0 when idle) | - |
| 0031 | KV segment: 0029 on by default with the segment | `GGML_MOE_KV_SEG_PF=0` |
| 0032 | loader: open the MoE heat map in `--host-experts mmap` mode (the map was never loaded or saved there: the only opener ran after mmap mode had returned) | `LLAMA_MMAP_HEAT_OPEN=0` |
| 0033 | expert cache: the persisted-heat seed of a sparse (mmap) table fills through the tier code (it used a pageable copy from the mapping; a warm start took more than 15 min) | - |
| 0034 | expert cache: a sparse table's staged fill at sizing takes the heat map's hottest experts; the per-expert heat seed leaves sparse tables to it | `GGML_MOE_STAGE_HEAT_RANK=0` |
| 0035 | MoE decode: the down + weighted-reduction fold takes precedence over the MUL-only down fold, and its sum runs without fma contraction (exact against the unfused graph) | `GGML_CUDA_MWR_FOLD_FIRST=0`, `GGML_CUDA_MWR_FOLD_CONTRACT=1` (old sum) |
| 0036 | the standalone MoE weighted-reduction kernels without fma contraction (exact against the unfused graph; changes output at rounding level) | `GGML_CUDA_MWR_EXACT=0` |
| 0037 | r292's arena settle skips layers that hold an mmap-tier table (re-sizing such a layer frees the table, which turned our RAM tier off for the rest of the process) | `GGML_MOE_SETTLE_SPARSE=1` (release behaviour) |
| 0038 | r299 merge fixes for the mmap tier: our prefill resolve runs only when the fused prefill serves nothing in place; our in-place-miss check only when the release's repair copied nothing; gate/up lanes with different tier remaps count as mixed (the fused kernels read one ids copy); our tier serves and take-overs clear the release's take-over promise | - |
| 0039 | our namespace-scope env-flag initializers are named functions instead of lambdas (see the r299 validation) | - |
| 0040 | the Q8_1 copy of a layer's input quantized for the attention / shared-expert matvecs is carried across the routed-MoE graph split, so the routed experts do not quantize the same input again; bit-identical | `GGML_CUDA_Q8_1_CARRY=0` |
| 0041 | prompt GEMMs off hipBLASLt: on ROCm, rocBLAS hands GEMMs to hipBLASLt by default, and hipBLASLt returned one of two results for bit-identical inputs, chosen once per process (the same prompt gave two different outputs across server starts). The F32 split tile of our MMB GEMM gets its own token floor (33 on RDNA4, above the matvec band) so the MoE router's whole prompt band runs on it, and `ggml_cuda_init` sets `ROCBLAS_USE_HIPBLASLT=0` (rocBLAS's own Tensile path) unless the variable is already set. Changes output at rounding level | `GGML_CUDA_HIPBLASLT=1`, `GGML_CUDA_MMB_F32SPLIT_MIN_T=512` (old floor) |
| 0042 | MMB: BF16 dense weights (on the production model the BF16 hyper-connection, router and indexer GEMMs) run on our bf16-WMMA GEMM on RDNA4 from 33 tokens while hipBLASLt is off (0041), instead of rocBLAS's Tensile path, which was 25-35 % slower than hipBLASLt on these shapes; deterministic. Changes output at rounding level | `GGML_CUDA_MMB_BF16_DENSE=0` |
| 0043 | MMB: no BF16 side copy of the hc gate on RDNA4 (nothing read it); bit-identical | `GGML_CUDA_MMB_HC16=1` (old) |
| 0044 | MMB: warp-per-token tiny-M kernel for BF16 weights with M <= 8 (hc inject / shared-expert gate) on RDNA4. Changes output at rounding level | `GGML_CUDA_MMB_TINY_M_BF16=0` |
| 0045 | MMB: the tiny-M BF16 kernel reads the cached bf16 activation copy instead of the F32 one; bit-identical to 0044's dense-tile reference rounding | `GGML_CUDA_MMB_TINY_BF16_XH=0` |
| 0046 | MMB: take the bf16 activation the producers already wrote (`bf16_src1_cache`: hc combine+norm, dsv4 hc_pre, hc_post RMS norm) instead of converting again; same rounding, bit-identical | `GGML_CUDA_MMB_PRODUCER_BF16=0` |
| 0047 | MMB: BF16 dense weights whose RDNA4 256x128 grid is below 64 workgroups use a 128x64 or 64x32 tile (each output still sums K in the same order); bit-identical | `GGML_CUDA_MMB_SMALL_TILE=0` |
| 0048 | hipBLASLt by version: `ggml_cuda_init` reads hipBLASLt's version (already-loaded `libhipblaslt.so.1`, one temporary handle). 1.5.0 and later (ROCm 10.2) keep a stable solution order (ROCm/rocm-libraries PR #11340: an untuned shape fell back to a solution list ordered by heap address), so hipBLASLt stays on there; older or unreadable versions get 0041's `ROCBLAS_USE_HIPBLASLT=0`, and 0042's MMB BF16 path applies only then. One log line states the decision. Source builds of hipBLASLt between 2026-09-09 and the fix also report 1.5.0 | `GGML_CUDA_HIPBLASLT=0` / `=1` (force off / on) |

0005, 0006 (down-fold part) and 0009 were workarounds for the bug the release's fusion overlap check fixes at the source;
0010 removes 0005. The others only run after a decline or a failed fill.

Production also sets `GGML_HIP_FA_BAND_WMMA_SPLIT=32` (r316's fixup period 16 costs 4-9 % long-context decode under `-sm tensor`; issue #333, fixed in r322) and `MOE_HOST_POOL_TRANSPORT=map` (pins the host-read transport the r262 adaptive mode would otherwise pick).

## Validation on r321 (2026-11-19 and 2026-11-20, Radeon AI PRO R9700 x 2, ROCm 10.1 and 10.2)

On the production configuration (described in the next section) unless another model is named.

- Rebase: every hash identical to the r317 series (r306-r317 changed some hashes themselves: r315's H256 gate and r316's
  fixup period, see #333).
- 0042-0047 on ROCm 10.1 (hipBLASLt off), Flash-Next GSQ IQ3_S (484 BF16 tensors): prefill +23 % (small prompts) to
  +48 % (31.5k) against the r305 series' rocBLAS path, within 1-3 % of hipBLASLt, one output per input. KLD against
  the r305 path on the production model: 0.0100 (ubatch 4096).
- 0048 on ROCm 10.2 (hipBLASLt 1.5.0, on): 10 of 10 server starts give one output, on IQ3_S and on the production
  model. Production midpf prefill +4 % over the same tree on 10.1; KLD against it 0.0097 (ubatch 4096) / 0.0108 (256),
  perplexity within error. IQ3_S prefill 1-3 % above our MMB path.
- Nightly on the final tree (ROCm 10.2): greedy references from the r305 series changed on four test models; the
  10.1 build of the same tree reproduces each new hash (code changes since r305, not 10.2). Four stress sessions,
  0 faults, every round the same hash, needle correct; byte checks (`GGML_MOE_VERIFY`) 0 differences.
- Production since 2026-11-20 (ROCm 10.2): greedy decode 117-139 t/s, midpf prefill about 2720 t/s, 259.6k needle
  prefill about 3210 t/s.

## Validation on r305 (2026-11-17 to 2026-11-19, Radeon AI PRO R9700 x 2, ROCm 10.1)

On the production configuration (described in the next section) unless another model is named.

- Rebase: greedy, prompt-set, mid-length, depth and needle hashes identical to the r299 series.
- 0043: bit-identical; the duplicate quantize launch is gone, decode gain within noise.
- 0044: 10 of 10 server starts give one output (before: two outcomes, deterministic per process; first seen in the
  router GEMM of 41-266 token prompts, then in the next hipBLAS GEMM in graph order once the router was moved). KLD
  against the old path on the production model: 0.0109 (ubatch 256) / 0.0100 (ubatch 4096), top token the same in
  about 96.8 %, perplexity within error. `GGML_CUDA_HIPBLASLT=1` brings back the hipBLASLt outcome.
- Nightly on the final tree: the 7-model comparison with the release passes; the q30 fault-injection arms change hash,
  0 faults: in mmap mode from 0044's floor (it moves the q30 router GEMM; the old floor gives the old hash); in the
  release's pool mode the text already differs from mmap on r305 with 0043 and 0044 off (coherent, near-tie), which
  we did not chase since production runs mmap. Four stress sessions, 48 2-GPU
  rounds, 0 faults, every round the expected hash, needle correct; byte checks (`GGML_MOE_VERIFY`) 0 differences.
- Production since 2026-11-19: greedy decode 125-138 t/s; in the nightly, 31.5k prefill about 3430 t/s, 159k about
  3338 t/s.

## Validation on r299 (2026-11-16 and 2026-11-17, Radeon AI PRO R9700 x 2, ROCm 10.1)

On the production configuration (described in the next section) unless another model is named.

- 0046: in this translation unit (our `.inc` files are included into the release's `moe-expert-cache.cu`) clang gave
  namespace-scope lambda initializers in different `namespace {}` blocks the same name and reused the first body. On r299
  this left our in-place prefill off (wrong output on the first build); on the r292 series `GGML_MOE_PF_PREFETCH` and
  `GGML_MOE_PREFILL_INPLACE` read `GGML_MOE_FILL_FENCE` / `GGML_MOE_TIER_PRECOPY` (all on by default, so output was
  unaffected, but those two switches did nothing). With 0046 every one of our 11 env flags is bound to its own variable
  (checked on the compiled object).
- Greedy, prompt-set, mid-length and depth hashes identical to the r292 series, 259.6k needle correct, speed equal, byte
  checks (`GGML_MOE_VERIFY`) 0 differences, 6 of 6 loads identical under `-sm tensor --load-mode none`.
- Nightly on the final tree: the 7-model comparison with the release passes with 0 problems; three 2-GPU stress sessions,
  57 rounds, 0 faults, every round the expected hash, needle correct; byte checks 0 differences.
- Production since 2026-11-17: greedy decode about 133 t/s, 31.5k prefill about 3480 t/s, 159k about 3378 t/s.

## Validation on r290-r292 (2026-11-15 and 2026-11-16, Radeon AI PRO R9700 x 2, ROCm 10.1)

On the production configuration (described in the next section) unless another model is named.

- Each rebase (r290, r291, r292): greedy, prompt-set, mid-length and depth hashes identical to the series before it, 259.6k
  needle correct, byte checks (`GGML_MOE_VERIFY`) 0 differences, 6 of 6 loads clean under `-sm tensor --load-mode none`.
  r291's sequence parallelism gives the same output as our dropped patches.
- r292's settle does not fire on this configuration (no decode-band prior; no capacity growth after sizing), so 0043 does not
  change output here.
- 0039-0040, on models with Q4_K / MXFP4 `ssm_alpha` / `ssm_beta`: KLD on vs off at the level of off vs off; decode
  Qwen3.6-35B-A3B +1.4 %, Qwen3.8-27B +0.6 %. No effect on the production model (BF16 / F32 projections).
- 0041: Qwen3.6-35B-A3B decode +2.5 %, Qwen3-30B-A3B and gemma-4-26B-A4B flat; KLD against the unfused graph 0 (was 0.0018
  on Qwen3.6-35B-A3B and 0.0021 on Qwen3-30B-A3B).
- 0042 changes the production model's output at rounding level, toward the unfused graph: KLD against the unfused graph
  0.011 (96.7 % same top token) before, 0 after; perplexity 2.5558 -> 2.5504. Decode speed unchanged.
- 0044: QSA top-k about 4.7 ms per call before; 259.6k needle prefill 2915 -> 3185 t/s (+9.3 %), 31.5k +1.5 %, every hash
  identical with the switch on and off.
- Nightly on the final tree: 0 faults. The 7-model comparison with the release differs only on the three MoE models
  (Qwen3-30B-A3B, Qwen3.6-35B-A3B, gemma-4-26B-A4B), as 0041-0042 intend: KLD 0.005 and 0.009; gemma's 0.54 is measured
  where its raw-wikitext perplexity is about 17,000 (the release too), and its multi-turn text is coherent. The dense
  models are identical. Four stress sessions: 61 rounds, 0 faults, every round the expected hash, needle correct.
- Production since 2026-11-16: 31.5k prefill 3456-3521 t/s, 159k 3362-3384 t/s, greedy decode about 135 t/s.

## Validation on r283 and r287 (2026-11-14, Radeon AI PRO R9700 x 2, ROCm 10.1)

All on the production configuration (unsloth `Qwen3.8-Flash-Next-UD-IQ4_XS`, `--n-cpu-moe 48`, expert-Q4_K MTP head,
2-GPU tensor, `-ub 4096`, `--host-experts mmap`, `GGML_MOE_KV_LEND=1 GGML_MOE_KV_SEG=1`).

- r277 series on r283 (0001-0037): outputs identical to the r277 series (greedy, prompt-set, mid-length prompts, depth
  texts, 259.6k needle), byte checks (`GGML_MOE_VERIFY`) 0 differences, speed equal; 6 of 6 loads clean under
  `-sm tensor --load-mode none` with the async arm on (r279's fix).
- 0038-0040, warm start from a saved heat map: ready in 26 s (cold 29 s; the first version of 0039 hung for more than
  15 min), all hashes identical to the cold start, byte checks 0 differences.
- 0041: up kernel 20.9 -> 18.4 us per call. 0042: per 4-token hc site 36.1 -> 31.9 us (hc kernels -11.6 % per call,
  about 1.7 % of GPU busy time). Both with their switches on and off: greedy, prompt-set, mid-length and depth hashes
  identical.
- 0043, same binary with the switch on and off: greedy, prompt-set, mid-length and depth hashes identical; 22 fewer
  kernel launches per decode step (731 -> 709), about 0.5 % decode.
- r287 (0001-0042): outputs identical to the same series on r283 (all hashes above, 259.6k needle
  correct), byte checks 0 differences, 6 of 6 loads clean; 31.5k prefill 3447-3470 t/s, greedy decode 117-125 t/s.
- Not yet: the nightly comparison and stress on this tree.

## Validation on r277 (2026-11-13 and 2026-11-14, Radeon AI PRO R9700 x 2, ROCm 10.1)

- r273 series on r277 (0001-0037): production outputs identical to the r273 series (greedy, prompt-set, mid-length
  prompts, depth texts, 259.6k needle), byte checks (`GGML_MOE_VERIFY`) 0 differences; speed equal within noise
  (31.5k prefill 3939-3957 t/s, greedy decode 129-137 t/s on the IQ3_S production model). 0037: 0 of 6 loads off the
  clean state (stock r277 shows the race about 1 load in 24).
- 0038, same binary with its four switches on and off: greedy, prompt-set, mid-length and depth hashes identical on
  UD-IQ4_XS and IQ3_S; 51 of 55 compared kernels byte-identical, including every BF16 one. UD-IQ4_XS decode +3.5-7.5 %
  (greedy 107.7 -> 112.4 t/s, two concurrent requests 162 -> 174 t/s).
- Nightly on the final tree: the 7-model comparison passed (KLD 0 on 6 of 6, test-backend-ops only the known cases),
  the 1-GPU production arms (5) 0 faults, the 2-GPU layer and tensor comparisons clean; gate 0 problems. Stress: only
  the 2-GPU production session ran (30 min, 10 rounds, 0 greedy mismatches, 0 faults); the tensor and layer-split
  stress sessions did not run this time.
- Production moved to unsloth `Qwen3.8-Flash-Next-UD-IQ4_XS` (UD 3.1) with `--n-cpu-moe 48`, the expert-Q4_K MTP
  head and `--n-predict 32768` (2-GPU tensor, `-ub 4096`, `--host-experts mmap`, `GGML_MOE_KV_LEND=1 GGML_MOE_KV_SEG=1`):
  perplexity 2.98 vs 3.14 for IQ3_S, no repetition loops on a prompt where IQ3_S looped on 7 of 24 seeds; 31.5k prefill
  3386-3431 t/s, 159k 3169-3178, 259.6k needle correct; greedy decode 110-112 t/s (IQ3_S: 129-137).

## Validation on r273 (2026-11-12 and 2026-11-13, Radeon AI PRO R9700 x 2, ROCm 10.1)

- Rebase of the r269 series onto r273 without conflicts after dropping r265 0012 (= PR #304, in r272); production outputs
  identical to the r269 series (greedy, prompt-set, mid-length prompts, depth texts, 259.6k needle).
- KV segment (0032): decode +1.4-2.5 % at 4.2k / 101k context and on mid-length prompts, prefill equal. 0033-0036, same
  binary on and off: all output hashes identical, byte checks (`GGML_MOE_VERIFY`) 0 differences; small-prompt prefill
  +3-4 %, 31.5k prefill and the needle +0.3-0.6 %, decode flat.
- Nightly on the final tree: the 7-model comparison passed (KLD 0 on 6 of 6, test-backend-ops only the known `MUL_MAT` f16
  case); one arm of the 2-GPU tensor comparison (all-VRAM IQ3_XXS with `GGML_CUDA_HC_POST_RMS=0`, segment off) gave a
  different greedy hash once and matched on a re-run. Three stress sessions (2-GPU production config with the segment,
  2-GPU tensor, pipelined layer split): 57 rounds, 0 greedy mismatches, 0 faults.
- Production (2-GPU tensor, `-ub 4096`, `--n-cpu-moe 47 --host-experts mmap`, MTP, `GGML_MOE_KV_LEND=1 GGML_MOE_KV_SEG=1`):
  31.5k prefill 3880-3925 t/s, 159k 3578, 259.6k needle 3272 t/s (correct); greedy decode 126-138 t/s.

## Validation on r265 (2026-11-11 and 2026-11-12, Radeon AI PRO R9700 x 2, ROCm 10.1)

- Rebase of the r261 series onto r265: two conflicts in the cache header / cache file, both resolved by keeping both sides.
  KLD against the r261 series 0 (seqpar off and on); the bf16 gather 0.0138 against a rounding-only control of 0.0138.
- 0027: `llama-perplexity` at `-ub 512 -c 8192` on the production config gave `nan` before the fix and 2.2453 after (= the
  kill switch); KLD against the staged build 0. 0030, 0031: all output hashes identical.
- Nightly on the final tree: the 7-model comparison and the gate passed (0 problems), test-backend-ops only the known
  `MUL_MAT` f16 case; four stress sessions (1-GPU byte-verify, 2-GPU tensor, pipelined layer split, 2-GPU production
  config): 56 rounds, 0 greedy mismatches, 0 faults, 0 byte differences in the expert cache.
- Production (2-GPU tensor, `-ub 4096`, `--n-cpu-moe 47 --host-experts mmap`, MTP): 31.5k prefill 3926-3998 t/s, 101k
  3808, 259.6k needle 3296 (correct); greedy decode 131-139 t/s. Against the r251 series: prefill +11-14 %, decode equal or
  slightly up. 2-GPU layer split on the same build is 38-69 % slower than tensor (prefill 2040-2379 t/s).

## Validation on r251 (2026-11-10, Radeon AI PRO R9700 x 2, ROCm 10.1)

- Rebase of the r248 series onto r251 without conflicts; production outputs identical to the r248 series (greedy, prompt-set,
  101k-token prompt, needle), 2-GPU layer split identical, fault-injection arms as before.
- 0022-0023, same binary on and off: all hashes identical; prefill +0.5-1.8 % (`-sm tensor` / layer split).
- Nightly on the final tree: the 7-model comparison and the gate passed (0 problems), then four stress sessions (1-GPU
  byte-verify, 2-GPU tensor, pipelined layer split, 2-GPU prefill rounds): 62 rounds, 0 greedy mismatches, 0 faults, 0
  byte differences in the expert cache. Production 31.5k prefill 3420-3460 t/s, 159k 3196-3201 t/s.

## Where the r228 series went

"in 0001" / "in 0002-0003" means the patch's code is in the new series (checked line by line and by its identifiers; a
few lines differ where the merge renamed helpers or the release changed the surrounding code).

| r228 patch | now |
|---|---|
| 0000 (keep `moe-expert-cache.cu` at r206) | not needed: the release's file is kept, our code is hooked in (0002-0003) |
| 0001 (release without #266's device remap / pool cold read) | not needed: the release's code stays whole; sparse tables use our remap from 0002 |
| 0002 (earlier stack, squashed) | in 0001 and 0002-0003, except two options removed: `GGML_MOE_TURN_SWAPS`, `GGML_MOE_SREG_NOMLOCK` |
| 0003 (`<thread>` port fix) | in 0002-0003 |
| 0004 (pool-backed staging only in pool mode) | superseded: pool tables run the release's own pool path |
| 0005-0007, 0009-0013, 0015-0019, 0047, 0049, 0052, 0053 | in 0001 |
| 0008 (heat-map debounce stamp) | in 0002-0003 for mmap tables; a frozen heat map in pool mode no longer stamps it (a few ms per turn, pool only) |
| 0014, 0021-0027, 0029-0033, 0035, 0037-0044, 0046, 0048, 0051, 0054 | in 0002-0003 |
| 0020, 0028, 0036, 0045 (cache file and other files) | in 0001 and 0002-0003 |
| 0034 (`GGML_MOE_TIER_INCL`, inclusive tier, default off) | removed (debug-only, never used in production) |
| 0050 (fill rollback) | in 0002-0003 for sparse tables; pool and pinned tables use the release's own rollback (r219 issue #274, r235) |

## Validation on r236 (2026-11-08 night, Radeon AI PRO R9700 x 2, ROCm 10.1)

The r235 series rebased (0001-0018), plus the two r236 ports (0019-0020). Built without ccache.

- Against r236 itself, seven 1-GPU models (as below): greedy server replies identical (14/14), KLD 0 on all six checks.
  test-backend-ops: only the known `MUL_MAT` f16 case fails, as on the release.
- Production model on 1 GPU: greedy, prompt-set, long-decode hashes and the 259.6k needle the same as the r235 series.
- 2 GPUs, `-sm tensor`: greedy identical, KLD 0, no faults. `GGML_CUDA_BF16_PRODUCER=2`: 0 mismatching BF16 elements.
  Two one-hour stress sessions (all-VRAM IQ3_XXS, and the production config), rounds of MTP decode, 31.5k and 159k
  prefills and the 259.6k needle: 38/38 rounds with the expected greedy hash, no faults. Production 31.5k prefill 3270-3300 t/s.
- **Not passing yet:** 2-GPU layer split of the full production model (with and without pipeline parallelism) runs out of
  memory on device 0 in a concurrent test (`ggml_cuda_mul_mat_q_pair` pool allocation); the r235 series passes the same
  test. Use `-sm tensor` with this series until that is fixed.

## Validation on r235 (2026-11-07 and 2026-11-08, Radeon AI PRO R9700 x 2, ROCm 10.1)

Built without ccache. Three configurations.

- 1 GPU, production model and config (Qwen3.8-Flash-Next IQ3_S, `--host-experts mmap`, `--n-cpu-moe 47`, MTP): greedy, prompt
  and post-needle output identical to the r228 series; byte checks (`GGML_MOE_VERIFY_AT`) 0 differences; cache-tier variants
  (8 GB / 16 GB pin budgets, 1 spare) no faults. Fill-fault arms handled; the pool fill-fault arm that printed "!!!!" before
  0011 now gives coherent text. Against r235 itself: greedy server replies identical on seven 1-GPU models (Llama-3.2-1B,
  Qwen3-30B-A3B, Qwen3.6-35B-A3B, gemma-4-26B-A4B, Qwen3.8-27B, Qwen3.8-27B NVFP4-MTP, Qwen3.5-4B, plus q8_0 / q4_0 KV and MTP),
  KLD 0 on all six non-MTP models, long-context depth texts identical on Qwen3-30B-A3B and Qwen3.6-35B-A3B. test-backend-ops
  21679/21680: the known `MUL_MAT` f16 case on `ROCBLAS_USE_HIPBLASLT=0` (the release fails it the same way, checked on r229).
- 2 GPUs, layer split with pipeline parallelism (0008): KLD 0 against plain layer split, identical greedy and needle output,
  prefill 1.39-1.46x one GPU in the server (half-depth Flash-Next, which fits one GPU).
- 2 GPUs, `-sm tensor` (the production configuration from 2026-11-08): KLD 0 with 0009 on and off; production model with
  `-ub 4096`, `GGML_CUDA_SLAB_WORK_MIN_MIB=2304`, `GGML_CUDA_SLAB_HEADROOM_MIB=3072`: greedy 127-135 t/s (one GPU: 69), 259.6k-token
  needle correct at 2673 t/s prefill with a VRAM peak of 32059 of 32624 MiB, no faults.
- 0011 alone, same binary on and off (`GGML_CUDA_FUSION_CHECK_LEAVES=0`), cache on, 1 GPU and 2-GPU tensor: all output hashes
  identical, prefill and decode within run-to-run noise. Cache off: Qwen3.6-35B-A3B and the production model give coherent
  text with it, "!!!!" without.

The headroom matters for `-sm tensor` with mmap host experts: at the default 2048 MiB the slab leaves only ~100 MiB at the
259.6k needle, and `--n-cpu-moe 24` runs out of memory there.
