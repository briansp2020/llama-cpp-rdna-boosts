# ours: our production stack on top of the current rdna-boosts release

This branch holds the patch series we run in production, on top of the latest rdna-boosts release. It is
regenerated for each new release; patches the release has taken are dropped. It is not a PR.

**Base:** `v17-a691b2345-r265` (upstream llama.cpp `a691b2345` + this repo's release).
The rest of this branch is that release as published; only `ours/` is added.

**New layout (2026-11-08).** Up to r228 this was a 55-patch series (0000-0054) that replaced the release's
`moe-expert-cache.cu` with its own r206 copy. From r235 on, the release's `moe-expert-cache.cu` stays as released, and our
`--host-experts mmap` code (RAM tier, pin budget, KV lending, fill fence, batched admissions, device remap for sparse
tables, ...) lives in two included files, `moe-expert-tier.inc` and `moe-expert-tier-state.inc`, called from about 130
hook lines in the release's file. Pool and pinned tables therefore run the release's own code; mmap tables run ours.
The old series is kept as the tag `ours-r228-series`; the mapping table below shows where each of its patches went.

**Apply:**

```
git checkout a691b2345
bash <path-to-this-repo>/scripts/apply-all.sh .     # the r265 release, branch rdna-boosts
git am <path-to-this-repo>/ours/*.patch              # 31 patches (0001-0031)
```

Result: tree `d6ece8b5f7be`. This is the production build from 2026-11-12 (2 GPUs, `-sm tensor`).

## Patches

The release has taken the r251 series' 0018 (DeepSeek-V4 cache size probes) and 0021 (DeepSeek-V4.1 kq-mask probe, issue
#300, fixed in r254); they are dropped here. Earlier releases took the r236 series' 0009, 0011, 0014-0015, 0017 and 0018.
The numbers below are the r265 series.

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
| 0012 | copy-engine all-reduce: one exchange for 2 ranks | `GGML_CUDA_AR_CE_SINGLE=0` |
| 0013 | the checkpoint pool uses `common_state_data` (no-init allocator); the pool thread still touches every page | - |
| 0014 | CUDA graph eviction counts active time only (idle gaps capped at 1 s), and keeps the rule that the looked-up graph is never evicted | - |
| 0015 | qwen4exp: PLE page hints (`MADV_WILLNEED`) only when a `mincore` sample says the table is cold (~13 ms per 4096-token ubatch saved) | `LLAMA_PLE_ADVISE=1` (always advise) |
| 0016 | multi-device: `alloc_splits` skips the drain when no buffer grows | `GGML_SCHED_DEFER_ALLOC_SYNC=0` |
| 0017 | `-sm tensor`: graph inputs in pinned memory (the meta device's first child host buffer type) and uploaded asynchronously per device; host compute buffers kept across the drop | `LLAMA_META_HOST_BUFT=0`, `GGML_SCHED_ASYNC_USER_INPUT=0`, `GGML_GALLOC_DROP_HOST=1` |
| 0018 | meta: the split-input copy of a split host weight is sized at the largest device slice, not the whole tensor | `GGML_META_SLICE_ALLOC=0` |
| 0019 | moe cache: a device-remap call does not re-batch a sibling already taken over in the same round | `GGML_MOE_DEVREMAP_REBATCH=1` |
| 0020 | qwen4exp qsa3: Q8_0 cache -> F16 packs in one copy; the gate copy runs before attention on prefill-sized graphs | `LLAMA_QSA3_DIRECT_F16=0`, `LLAMA_GATE_EARLY=0` |
| 0021 | qwen4exp qsa3: the packs read the Q8_0 cache views directly, no F16 copies in the graph | `LLAMA_QSA3_PACK_Q8=0` |
| 0022 | `-sm tensor` ubatch overlap: asynchronous graph-input views, asynchronous CPU-split output upload, deferred input wait | `GGML_SCHED_ASYNC_INPUT_VIEWS=0`, `GGML_SCHED_ASYNC_CPU_OUT=0`, `LLAMA_DEFER_INPUT_WAIT=0` |
| 0023 | qwen4exp sequence parallelism: in 2-GPU `-sm tensor` prefill the hyper-connection work (mixers, residual streams, norms) is split by token instead of computed on both GPUs; reduce-scatter + all-gather around attention and the FFN | opt-in `LLAMA_SEQPAR=1` (production sets it) |
| 0024 | seqpar review fix: the all-gather never grows an existing copy-engine scratch | `GGML_SEQPAR_CE_GATHER=0` (generic gather) |
| 0025 | seqpar debug switch: token-split only layers a..b | `LLAMA_SEQPAR_LAYERS=a-b` (default all) |
| 0026 | decode MoE gate+up+GLU fusion when the GLU output overlaps only the unfilled weight copy (41 of 47 layers were blocked) | `GGML_MOE_FUSE_COPY_ALIAS=0` |
| 0027 | moe cache: identity (fully resident) tables are served in place above the band in mmap mode too; fixes NaN at small ubatches since r258, with a warn-once `MOE_INPLACE_MISS` tripwire | `MOE_EXPERT_PREFILL_IN_PLACE=0` |
| 0028 | seqpar: the two per-layer `hc_mixed` all-gathers send bf16 (rounding-level KLD, 0.013) | opt-in `LLAMA_SEQPAR_GATHER_BF16=1` (production sets it) |
| 0029 | seqpar: odd token counts split ceil/floor | `LLAMA_SEQPAR_ODD=0` |
| 0030 | moe cache: a table is pool-backed only when the pool was requested | `GGML_MOE_POOL_TABLE_REQ=0` |
| 0031 | mmq: RDNA4 IQ routed-compact experts with 65-96 rows per expert use J=64 (+1-2 % prefill at `-ub 4096`, bit-identical) | `GGML_MMQ_IQ_J64_WIDE=0` |

0005, 0006 (down-fold part) and 0009 were workarounds for the bug the release's fusion overlap check fixes at the source;
0010 removes 0005. The others only run after a decline or a failed fill.

Production also sets `MOE_HOST_POOL_TRANSPORT=map` (pins the host-read transport the r262 adaptive mode would otherwise pick).

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
