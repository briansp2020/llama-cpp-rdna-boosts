# ours: our production stack on top of the current rdna-boosts release

This branch holds the patch series we run in production, on top of the latest rdna-boosts release. It is
regenerated for each new release; patches the release has taken are dropped. It is not a PR.

**Base:** `v16-6050b467c-r185` (upstream llama.cpp `6050b467c` + this repo's release; resulting tree
`f8eaf8f005b4`). The rest of this branch is that release as published; only `ours/` is added. r180 took #252's 0003 (the
GDN small-projection gather), r182 took #253 (the 1-token router fold and its RDNA4 gate) and r183 took #256 (the GDN
decode folds), so they are no longer in this series. 0016-0022 are our open PRs #258, #259 and #260, exactly as posted.

**Apply:**

```
git checkout 6050b467c
bash <path-to-this-repo>/scripts/apply-all.sh .     # the r185 release, branch rdna-boosts
git am <path-to-this-repo>/ours/*.patch              # 22 patches
```

Result: tree `d75f9e27bd93`.

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
| 0016-0020 | #258: decode with a quantized KV cache. QK rotate across the KV rotation (0016), V's rotation folded into the K cache write (0017), the output rotation folded into the FA combine (0018), the Q / K norm+rope pair (0019), rope + rotation in one launch for attention without Q / K norm (0020) | `GGML_CUDA_QK_ROTATE_HADAMARD=0`, `GGML_CUDA_FUSE_V_FWHT=0`, `GGML_CUDA_FA_COMBINE_FWHT=0` / `GGML_CUDA_FA_COMBINE_Q8=0`, `GGML_CUDA_QK_PAIR_FWHT=0`, `GGML_CUDA_FUSE_ROPE_FWHT=0` |
| 0021 | #259: the shared-expert lane of the 1-token fused MoE gate/up+GLU publishes its Q8_1 too | `GGML_CUDA_MOE_SHARED_GLU_Q8=0` |
| 0022 | #260: RDNA4 1-token Q / K / V row concat across a K-quant / Q6_K pair | `GGML_CUDA_MMVQ_ROW_CONCAT_MIXED=0` |

The work the release has taken keeps its switches there: `GGML_CUDA_QK_ROTATE=0`, `GGML_CUDA_MMVQ_ROW_CONCAT_RDNA4=0`,
`GGML_CUDA_MMVQ_CONCAT_REORDER=0`, `GGML_CUDA_ROUTER_TOPK=0`, `GGML_CUDA_DISABLE_GDN_CONV_SNAP=1`,
`GGML_CUDA_FUSE_GDN_CONV_DECODE=0`, `GGML_CUDA_FUSE_SHEXP_GATE_SIGMOID=0`.

## Validation (2026-10-31, Radeon AI PRO R9700, 1 GPU)

r185 vs r185 + this series, no env on either side, `ROCBLAS_USE_HIPBLASLT=0`. r184 -> r185 changes only `common/speculative.cpp`
(the verify width bound); the rebase reproduces the three-way merge tree exactly, and 0016-0022 apply without conflicts and are
identical to the posted PRs.

- Greedy server replies (12 multi-turn replies per model) are byte-identical to r185 on Llama-3.2-1B, Qwen3-30B-A3B,
  Qwen3.6-35B-A3B, gemma-4-26B-A4B, Qwen3.8-27B, Qwen3.8-27B NVFP4-MTP (plain and MTP `--spec-draft-n-max 3`) and Qwen3.5-4B;
  also with `-ctk q8_0 -ctv q8_0` on Llama-3.2-1B, Qwen3-30B-A3B and Qwen3.6-35B-A3B. No faults.
- KLD on wikitext-2 `-c 512`, 24 chunks, against r185's logits: 0 on all six non-MTP models (same top token 100 %, or 99.984 %
  on Qwen3-30B-A3B and Qwen3.5-4B, which is r185 against itself).
- test-backend-ops (ROCm0, with `ROCBLAS_USE_HIPBLASLT=0`): 21674 / 21675 passed. The one failure,
  `MUL_MAT(type_a=f16,type_b=f32,m=16,n=2105344,k=32)`, comes from that env; the release shows the same with it.
- Our production model (Qwen3.8-Flash-Next IQ3_S, host experts, MTP draft) on one GPU: greedy replies unchanged from the
  r184-based series, no faults.

llama-bench `-fa 1 -p 512 -n 128 -r 5`, 2 alternating rounds, tg128 means:

| model | r185 -> r185 + series |
|---|---|
| Llama-3.2-1B UD-Q4_K_XL | 422.6 -> 434.3 (+2.8 %) |
| Qwen3-30B-A3B Q4_K_M | 168.2 -> 169.9 (+1.0 %) |
| Qwen3.6-35B-A3B Q4_K_M | 136.7 -> 139.6 (r185's first round ran cold; second round equal) |
| gemma-4-26B-A4B MXFP4_MOE | 94.3 -> 94.1 (=) |
| Qwen3.8-27B Q4_K_M | 31.8 -> 31.8 (=; 12 alternating runs) |
| Qwen3.8-27B NVFP4-MTP-HIGH | 30.6 -> 30.5 (=; 12 alternating runs) |
| Qwen3.5-4B Q8_0 | 102.0 -> 102.0 (=) |

With a q8_0 KV cache, #258 is where most of the gain is: Qwen3-30B-A3B tg128 150.3 (r184) -> 167.5 with #258 -> 169.8 with
#258 + #260 (R9700; the PRs carry the per-patch numbers). pp512 is unchanged on every model except where the release's
first round ran cold.
