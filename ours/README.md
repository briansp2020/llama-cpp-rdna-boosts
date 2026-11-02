# ours: our production stack on top of the current rdna-boosts release

This branch holds the patch series we run in production, on top of the latest rdna-boosts release. It is
regenerated for each new release; patches the release has taken are dropped. It is not a PR.

**Base:** `v16-6050b467c-r204` (upstream llama.cpp `6050b467c` + this repo's release; resulting tree
`6260370949c1`). The rest of this branch is that release as published; only `ours/` is added. r180 took #252's 0003 (the
GDN small-projection gather), r182 took #253 (the 1-token router fold and its RDNA4 gate), r183 took #256 (the GDN
decode folds), r187 took #258 (the quantized-KV rotations), r188 took #259 (the shared-expert GLU Q8_1), r198 took #260
(the mixed K-quant x Q6_K row concat), r199 took #262 (the q4_0 KV write fold) and r203 took #266 (host-expert decode
without the per-op routing sync), so they are no longer in this series.

0001 takes #266's device remap and pool cold read back out of the release: this series has carried its own variant of
that remap since before #266 (in 0002, with the review fixes in 0021: freshness rounds, and the remap forwarded through
the Meta backend), and the two cannot both run. After 0001, `moe-expert-cache.cu` is the r200 file again.

**Apply:**

```
git checkout 6050b467c
bash <path-to-this-repo>/scripts/apply-all.sh .     # the r204 release, branch rdna-boosts
git am <path-to-this-repo>/ours/*.patch              # 23 patches
```

Result: tree `9851e4c9335a`.

## Patches

| | what | env switch (off) |
|---|---|---|
| 0001 | the release without #266's device remap and pool cold read (see above) | - |
| 0002 | our earlier stack, squashed (expert cache, MoE / GDN / shared-expert fusions, scheduler and server items) | per feature |
| 0003-0004, 0014 | port fixes for the expert cache and graph recapture | - |
| 0005-0006 | `LLAMA_UB_SMALL_SPLIT` (off by default) | - |
| 0007 | `LLAMA_CKPT_USER_MERGE=K` (off by default) | - |
| 0008-0011 | heat-map debounce, split-state hit copy, threaded large GET_ROWS, device token-embedding gather | - |
| 0012-0013 | `GGML_SCHED_UPFRONT_INPUTS` (opt-in) | - |
| 0015 | restore the GLU -> Q8_1 mark call that 0002 drops from `ggml_cuda_try_fuse` | `GGML_CUDA_FUSE_GLU_Q8_1=0` |
| 0016 | 0002's pinned input-snapshot ring and lazy split events are on by default only with more than one GPU (each cost about 1 % decode on a single GPU) | `GGML_CUDA_SNAP_RING=0/1`, `GGML_SCHED_LAZY_EVENTS=0/1` force off / on |
| 0017-0018 | MoE expert dedup at verify widths (2-4 tokens): (slot, token) pairs that share a cold expert are computed by one owner warp, which loads each weight once (at most 2 pairs per owner); the raw-load IQ2_S / IQ3_XXS / IQ3_S gate/up and IQ4_NL down paths of 0002 | `GGML_CUDA_MOE_DEDUP=0` (`=2` also dedups resident experts, A/B only) |
| 0019 | the fused MoE gate/up+GLU publishes its Q8_1 at verify widths (2-8 tokens) too, per pair under the dedup | `GGML_CUDA_MOE_GLU_Q8_MULTI=0` |
| 0020 | cleanup of 0002's mmap expert-cache code: shared helpers for the copy kernels, the expert-source choice, the device-map upload, the seed fill and the pinned-chunk upload; an empty function removed (no behaviour change) | - |
| 0021 | review fixes for 0002's decode device remap: unloading a model drops the pending sibling batches (they hold table indices); a sibling table joins a remap launch only when its own last routing buffer and stream match; the lag-flag store is a system-scope atomic (`__hip_atomic_store`, `atomicExch` on CUDA) | `GGML_MOE_DEVREMAP=0` / `GGML_MOE_DEVREMAP_BATCH=0` |
| 0022-0023 | `--host-experts mmap` RAM tier: K spare RAM slots, and the likely victims copied back ahead on a side stream, so an admission rarely waits for its copy-back (`GGML_MOE_TIER_SPARES`, default 4; we run 8 with `MOE_EXPERT_CACHE_TOUCH=3`); 0023 hardens its error paths (a failed event record drains the side stream, a failed wait fails the swap) | `GGML_MOE_TIER_PRECOPY=0` |

The work the release has taken keeps its switches there: `GGML_CUDA_QK_ROTATE=0`, `GGML_CUDA_MMVQ_ROW_CONCAT_RDNA4=0`,
`GGML_CUDA_MMVQ_CONCAT_REORDER=0`, `GGML_CUDA_ROUTER_TOPK=0`, `GGML_CUDA_DISABLE_GDN_CONV_SNAP=1`,
`GGML_CUDA_FUSE_GDN_CONV_DECODE=0`, `GGML_CUDA_FUSE_SHEXP_GATE_SIGMOID=0`, #258's / #259's (`GGML_CUDA_QK_ROTATE_HADAMARD=0`,
`GGML_CUDA_FUSE_V_FWHT=0`, `GGML_CUDA_FA_COMBINE_FWHT=0`, `GGML_CUDA_QK_PAIR_FWHT=0`, `GGML_CUDA_FUSE_ROPE_FWHT=0`,
`GGML_CUDA_MOE_SHARED_GLU_Q8=0`), `GGML_CUDA_MMVQ_ROW_CONCAT_MIXED=0` (#260) and `GGML_CUDA_KV_FOLD_Q4_0=0` (#262).

## Validation (2026-11-01, Radeon AI PRO R9700, 1 GPU, ROCm 10.1)

r204 vs r204 + this series, both built and run in `rocm/dev-ubuntu-24.04:10.1.0-full`, no env on either side,
`ROCBLAS_USE_HIPBLASLT=0`. The series replays onto 0001 without conflicts.

- Greedy server replies (12 multi-turn replies per model) are byte-identical to r204 on Llama-3.2-1B, Qwen3-30B-A3B,
  Qwen3.6-35B-A3B, gemma-4-26B-A4B, Qwen3.8-27B, Qwen3.8-27B NVFP4-MTP (plain and MTP `--spec-draft-n-max 3`) and Qwen3.5-4B;
  also with `-ctk q8_0 -ctv q8_0` and `-ctk q4_0 -ctv q4_0`. No faults.
- KLD on wikitext-2 `-c 512`, 24 chunks, against r204's logits: 0 on all six non-MTP models (same top token 99.98-100 %).
- test-backend-ops (ROCm0, with `ROCBLAS_USE_HIPBLASLT=0`): 21675 / 21676 passed; the one failure,
  `MUL_MAT(type_a=f16,type_b=f32,m=16,n=2105344,k=32)`, comes from that env (the release shows the same with it).
- Our production model (Qwen3.8-Flash-Next IQ3_S, `--host-experts mmap`, MTP draft) on one GPU: greedy replies
  identical to stock r204, the 259.6k-token needle answered correctly, no faults. Against our previous build (r196 + the
  previous series): greedy decode 77.0 vs 74.5 t/s (+3.4 %), two concurrent streams 96.9 vs 97.6 t/s (4 alternating runs).

llama-bench `-fa 1 -p 512 -n 128 -r 5`, 2 alternating rounds, tg128 means:

| model | r204 -> r204 + series |
|---|---|
| Llama-3.2-1B UD-Q4_K_XL | 432.8 -> 432.7 (=) |
| Qwen3-30B-A3B Q4_K_M | 167.1 -> 167.6 (=) |
| Qwen3.6-35B-A3B Q4_K_M | 138.1 -> 139.8 (+1.2 %) |
| gemma-4-26B-A4B MXFP4_MOE | 93.9 -> 93.7 (=) |
| Qwen3.8-27B Q4_K_M | 31.94 -> 32.06 (=) |
| Qwen3.8-27B NVFP4-MTP-HIGH | 30.55 -> 30.12 (-1.4 %) |
| Qwen3.5-4B Q8_0 | 102.1 -> 102.0 (=) |

The dedup (0017-0018) and the copy-back ahead (0022) help only a model whose routed experts are partly in host memory:
our production model.
