# ours: our production stack on top of the current rdna-boosts release

This branch holds the patch series we run in production, on top of the latest rdna-boosts release. It is
regenerated for each new release; patches the release has taken are dropped. It is not a PR.

**Base:** `v16-6050b467c-r235` (upstream llama.cpp `6050b467c` + this repo's release; resulting tree `ef5dcbcb0ae7`).
The rest of this branch is that release as published; only `ours/` is added.

**New layout (2026-11-08).** Up to r228 this was a 55-patch series (0000-0054) that replaced the release's
`moe-expert-cache.cu` with its own r206 copy. From r235 on, the release's `moe-expert-cache.cu` stays as released, and our
`--host-experts mmap` code (RAM tier, pin budget, KV lending, fill fence, batched admissions, device remap for sparse
tables, ...) lives in two included files, `moe-expert-tier.inc` and `moe-expert-tier-state.inc`, called from about 130
hook lines in the release's file. Pool and pinned tables therefore run the release's own code; mmap tables run ours.
The old series is kept as the tag `ours-r228-series`; the mapping table below shows where each of its patches went.

**Apply:**

```
git checkout 6050b467c
bash <path-to-this-repo>/scripts/apply-all.sh .     # the r235 release, branch rdna-boosts
git am <path-to-this-repo>/ours/*.patch              # 11 patches (0001-0011)
```

Result: tree `0113783a8a1e`.

## Patches

| | what | env switch (off) |
|---|---|---|
| 0001 | everything of the r228 series outside `moe-expert-cache.cu`, squashed (scheduler, server, MoE / GDN / shared-expert fusions, dedup, slab, KV lending hooks in llama, ...; see the mapping table) | per feature, as before |
| 0002 | our mmap expert-cache code in `moe-expert-tier.inc` / `moe-expert-tier-state.inc`, not yet called | - |
| 0003 | the hooks: the release's `moe-expert-cache.cu` calls the tier code for sparse (mmap) tables (+128/-18 in that file) | per feature, as before |
| 0004 | the device remap declines when a fill failed during its own policy pass (the consumer refuses a `fill_failed` table and would read a scheduler copy nobody filled) | - |
| 0005 | a declined decode-band sparse gate+up+GLU runs unfused | `GGML_MOE_FUSED_DECLINE_UNFUSE=0` |
| 0006 | review fixes: the tier remap predicate refuses a `fill_failed` table; the down fold also runs unfused for a declined sparse table | - |
| 0007 | header comment | - |
| 0008 | layer split with pipeline parallelism overlaps the devices on ROCm: prompt ubatches rebuild their graph instead of reusing it (reuse forces a full scheduler sync), and host inputs go through a pinned per-(backend, copy) ring uploaded on the split's stream; opt-in `GGML_GALLOC_FP_SHAPES=0` and `LLAMA_KV_N_PAD_PREFILL_POW2=1` | `LLAMA_PP_PREFILL_NO_REUSE=0`, `GGML_SCHED_INPUT_STAGE=0` |
| 0009 | `-sm tensor` prefill on qwen4exp: one kernel for DSV4_HC_POST -> RMS_NORM -> MUL (bit-identical; falls back on overlap) | `GGML_CUDA_HC_POST_RMS=0` |
| 0010 | a table whose fill failed (r235 declines it itself) also runs gate+up+GLU unfused | - |
| 0011 | the CUDA fusion overlap check no longer skips leaves that live in a compute buffer. A scheduler copy of a host expert weight (`-ncmoe` + op offload) is a leaf in the compute buffer, and the allocator can give its memory to a fused output once the last unfused consumer is done; the fused routed gate+up+GLU then wrote its output over the gate weights it was still reading (NaN, "!!!!" output) whenever a layer's experts were served from the scheduler copy: cache off, a declined or fill-failed table. The same check is in upstream ggml | `GGML_CUDA_FUSION_CHECK_LEAVES=0` |

0005, 0006 (down-fold part) and 0010 were workarounds for the bug 0011 fixes at the source. They stay for now; they only run
after a decline or a failed fill, and removing them needs its own A/B.

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
