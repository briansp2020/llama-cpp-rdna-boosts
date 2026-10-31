# ours: our production stack on top of the current rdna-boosts release

This branch holds the patch series we run in production, on top of the latest rdna-boosts release. It is
regenerated for each new release; patches the release has taken are dropped. It is not a PR.

**Base:** `v16-6050b467c-r188` (upstream llama.cpp `6050b467c` + this repo's release; resulting tree
`61a89235716a`). The rest of this branch is that release as published; only `ours/` is added. r180 took #252's 0003 (the
GDN small-projection gather), r182 took #253 (the 1-token router fold and its RDNA4 gate) and r183 took #256 (the GDN
decode folds), r187 took #258 (the quantized-KV rotations) and r188 took #259 (the shared-expert GLU Q8_1), so they
are no longer in this series. 0016 (#260) and 0020 (#262) are our open PRs, exactly as posted.

**Apply:**

```
git checkout 6050b467c
bash <path-to-this-repo>/scripts/apply-all.sh .     # the r188 release, branch rdna-boosts
git am <path-to-this-repo>/ours/*.patch              # 21 patches
```

Result: tree `0ad4bc8ed9e4`.

## Patches

| | what | env switch (off) |
|---|---|---|
| 0001 | our earlier stack, squashed (expert cache, MoE / GDN / shared-expert fusions, scheduler and server items) | per feature |
| 0002-0003, 0013 | port fixes for the expert cache and graph recapture | - |
| 0004-0005 | `LLAMA_UB_SMALL_SPLIT` (off by default) | - |
| 0006 | `LLAMA_CKPT_USER_MERGE=K` (off by default) | - |
| 0007-0010 | heat-map debounce, split-state hit copy, threaded large GET_ROWS, device token-embedding gather | - |
| 0011-0012 | `GGML_SCHED_UPFRONT_INPUTS` (opt-in) | - |
| 0014 | restore the GLU -> Q8_1 mark call that 0001 drops from `ggml_cuda_try_fuse` | `GGML_CUDA_FUSE_GLU_Q8_1=0` |
| 0015 | 0001's pinned input-snapshot ring and lazy split events are on by default only with more than one GPU (each cost about 1 % decode on a single GPU) | `GGML_CUDA_SNAP_RING=0/1`, `GGML_SCHED_LAZY_EVENTS=0/1` force off / on |
| 0016 | #260: RDNA4 1-token Q / K / V row concat across a K-quant / Q6_K pair | `GGML_CUDA_MMVQ_ROW_CONCAT_MIXED=0` |
| 0017-0018 | MoE expert dedup at verify widths (2-4 tokens): (slot, token) pairs that share a cold expert are computed by one owner warp, which loads each weight once (at most 2 pairs per owner); the raw-load IQ2_S / IQ3_XXS / IQ3_S gate/up and IQ4_NL down paths of 0001 | `GGML_CUDA_MOE_DEDUP=0` (`=2` also dedups resident experts, A/B only) |
| 0019 | the fused MoE gate/up+GLU publishes its Q8_1 at verify widths (2-8 tokens) too, per pair under the dedup | `GGML_CUDA_MOE_GLU_Q8_MULTI=0` |
| 0020 | #262: fused K / V cache write for a q4_0 KV cache too | `GGML_CUDA_KV_FOLD_Q4_0=0` |
| 0021 | cleanup of 0001's mmap expert-cache code: shared helpers for the copy kernels, the expert-source choice, the device-map upload, the seed fill and the pinned-chunk upload; an empty function removed (no behaviour change) | - |

The work the release has taken keeps its switches there: `GGML_CUDA_QK_ROTATE=0`, `GGML_CUDA_MMVQ_ROW_CONCAT_RDNA4=0`,
`GGML_CUDA_MMVQ_CONCAT_REORDER=0`, `GGML_CUDA_ROUTER_TOPK=0`, `GGML_CUDA_DISABLE_GDN_CONV_SNAP=1`,
`GGML_CUDA_FUSE_GDN_CONV_DECODE=0`, `GGML_CUDA_FUSE_SHEXP_GATE_SIGMOID=0`, and #258's / #259's (`GGML_CUDA_QK_ROTATE_HADAMARD=0`,
`GGML_CUDA_FUSE_V_FWHT=0`, `GGML_CUDA_FA_COMBINE_FWHT=0`, `GGML_CUDA_QK_PAIR_FWHT=0`, `GGML_CUDA_FUSE_ROPE_FWHT=0`,
`GGML_CUDA_MOE_SHARED_GLU_Q8=0`).

## Validation (2026-10-31, Radeon AI PRO R9700, 1 GPU)

r188 vs r188 + this series, no env on either side, `ROCBLAS_USE_HIPBLASLT=0`. The release took #258 and #259 exactly as posted
(r186 + the posted #258 gives the r187 tree and r187 + the posted #259 the r188 tree, byte for byte), so dropping them leaves the rest unchanged; the rebase
reproduces the three-way merge tree exactly.

- Greedy server replies (12 multi-turn replies per model) are byte-identical to r188 on Llama-3.2-1B, Qwen3-30B-A3B,
  Qwen3.6-35B-A3B, gemma-4-26B-A4B, Qwen3.8-27B, Qwen3.8-27B NVFP4-MTP (plain and MTP `--spec-draft-n-max 3`) and Qwen3.5-4B;
  also with `-ctk q8_0 -ctv q8_0` (Llama-3.2-1B, Qwen3-30B-A3B, Qwen3.6-35B-A3B) and `-ctk q4_0 -ctv q4_0` (Llama-3.2-1B,
  Qwen3-30B-A3B, Qwen3.6-35B-A3B). No faults.
- KLD on wikitext-2 `-c 512`, 24 chunks, against r188's logits: 0 on all six non-MTP models.
- test-backend-ops (ROCm0, with `ROCBLAS_USE_HIPBLASLT=0`): 21674 / 21675 passed; the one failure,
  `MUL_MAT(type_a=f16,type_b=f32,m=16,n=2105344,k=32)`, comes from that env (the release shows the same with it).
- Our production model (Qwen3.8-Flash-Next IQ3_S, host experts, MTP draft) on one GPU: greedy replies unchanged, the
  259.6k-token needle answered correctly, no faults.

llama-bench `-fa 1 -p 512 -n 128 -r 5`, 2 alternating rounds, tg128 means:

| model | r188 -> r188 + series |
|---|---|
| Llama-3.2-1B UD-Q4_K_XL | 423.0 -> 434.0 (+2.6 %) |
| Qwen3-30B-A3B Q4_K_M | 164.2 -> 169.4 (+3.2 %; r188's first round ran cold, second round +1.3 %) |
| Qwen3.6-35B-A3B Q4_K_M | 138.5 -> 138.8 (=) |
| gemma-4-26B-A4B MXFP4_MOE | 94.3 -> 92.5 (=; one slow ours repetition, otherwise 94.0-94.1) |
| Qwen3.8-27B Q4_K_M | 31.96 -> 32.17 (=) |
| Qwen3.8-27B NVFP4-MTP-HIGH | 30.65 -> 30.21 (-1.4 %; in the noise at r185, where 12 alternating runs measured it equal) |
| Qwen3.5-4B Q8_0 | 102.0 -> 102.0 (=) |

With a q4_0 KV cache (0020, #262): Qwen3-30B-A3B tg128 +9.4 % (+9.7 % at depth 4096), Llama-3.2-1B +5.0 %, Qwen3.6-35B-A3B
flat. The dedup (0017-0018) helps only a model whose routed experts are partly in host memory: our production model,
+2.6 % greedy decode on one GPU.
