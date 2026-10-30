# ours: our production stack on top of the current rdna-boosts release

This branch holds the patch series we run in production, on top of the latest rdna-boosts release. It is
regenerated for each new release; patches the release has taken are dropped. It is not a PR.

**Base:** `v16-6050b467c-r182` (upstream llama.cpp `6050b467c` + this repo's release; resulting tree
`bdbcb1964e57`). The rest of this branch is that release as published; only `ours/` is added. r180 took #252's 0003 (the
GDN small-projection gather) and r182 took #253 (the 1-token router fold and its RDNA4 gate), so they are no longer in
this series.

**Apply:**

```
git checkout 6050b467c
bash <path-to-this-repo>/scripts/apply-all.sh .     # the r182 release, branch rdna-boosts
git am <path-to-this-repo>/ours/*.patch              # 17 patches
```

Result: tree `a2eb4147c3fc`.

## Patches

| | what | env switch (off) |
|---|---|---|
| 0001 | our earlier stack, squashed (expert cache, MoE / GDN / shared-expert fusions, scheduler and server items) | per feature |
| 0002-0003, 0013 | port fixes for the expert cache and graph recapture | - |
| 0004-0005 | `LLAMA_UB_SMALL_SPLIT` (off by default) | - |
| 0006 | `LLAMA_CKPT_USER_MERGE=K` (off by default) | - |
| 0007-0010 | heat-map debounce, split-state hit copy, threaded large GET_ROWS, device token-embedding gather | - |
| 0011-0012 | `GGML_SCHED_UPFRONT_INPUTS` (opt-in) | - |
| 0014 | GDN direct conv for 1-token decode (superseded by 0016) | - |
| 0015 | fixes: restore the GLU -> Q8_1 mark an earlier rebase dropped; shared-expert gate fold only for BF16 / F32 weights | `GGML_CUDA_FUSE_GLU_Q8_1=0` |
| 0016 | the reviewed GDN series, as posted in #256 (snapshot fold, conv-state gather fold, 1-token direct conv only where it also folds the q / k norm and the prescan does not match, shared-expert gate fold with the MMB check), replacing the older forms in 0001 / 0014 | `GGML_CUDA_DISABLE_GDN_CONV_SNAP=1`, `GGML_CUDA_FUSE_GDN_CONV_DECODE=0`, `GGML_CUDA_FUSE_SHEXP_GATE_SIGMOID=0` |
| 0017 | 0001's pinned input-snapshot ring and lazy split events are on by default only with more than one GPU (each cost about 1 % decode on a single GPU) | `GGML_CUDA_SNAP_RING=0/1`, `GGML_SCHED_LAZY_EVENTS=0/1` force off / on |

The work the release has taken keeps its switches there: `GGML_CUDA_QK_ROTATE=0`, `GGML_CUDA_MMVQ_ROW_CONCAT_RDNA4=0`,
`GGML_CUDA_MMVQ_CONCAT_REORDER=0`, `GGML_CUDA_ROUTER_TOPK=0`.

## Validation (2026-10-30, Radeon AI PRO R9700, 1 GPU)

r182 vs r182 + this series, no env on either side, `ROCBLAS_USE_HIPBLASLT=0`. llama-bench `-fa 1 -p 512 -n 128 -r 5`,
2 alternating rounds, means; KLD on wikitext-2 `-c 512`, 24 chunks, against r182's logits.

| model | KLD | tg128 r182 -> series | pp512 |
|---|---|---|---|
| Llama-3.2-1B UD-Q4_K_XL | 0, top 100 % | 423.0 -> 423.5 (=) | = |
| Qwen3-30B-A3B Q4_K_M | 0, top 99.984 % (= r182 vs itself) | 164.6 -> 165.4 (+0.5 %, noisy) | noisy |
| Qwen3.6-35B-A3B Q4_K_M | 0, top 100 % | 130.5 -> 136.0 (+4.2 %) | noisy |
| gemma-4-26B-A4B MXFP4_MOE | 0, top 100 % | 94.3 -> 94.2 (=) | -0.4 % |
| Qwen3.8-27B Q4_K_M | 0, top 100 % | 31.4 -> 32.1 (+2.2 %) | +1.2 % (second round; the first r182 round ran cold) |
| Qwen3.8-27B NVFP4-MTP-HIGH | - | 30.0 -> 30.2 (=) | +2.4 % (second round, same) |
| Qwen3.5-4B Q8_0 | 0, top 99.984 % (= r182 vs itself) | 101.1 -> 102.1 (+1.0 %) | = |

- Greedy server replies (12 multi-turn replies per model) match r182 byte for byte on all seven models, including MTP
  drafting on (`--spec-type draft-mtp --spec-draft-n-max 3`) and off; no faults.
- test-backend-ops (ROCm0, with `ROCBLAS_USE_HIPBLASLT=0`): 21674 / 21675 passed. The one failure,
  `MUL_MAT(type_a=f16,type_b=f32,m=16,n=2105344,k=32)` (error 0.0043, limit 0.0005), comes from that env: it passes
  without it, and r182 itself shows the same failure with it and passes without it.
- With our production `LLAMA_CKPT_USER_MERGE=256`, replies on the hybrid and sliding-window models differ, by design:
  the follow-up turn restarts from a different checkpoint.
- 0017 on its own (on the r177-based series, compared with forcing both features on): Llama-3.2-1B tg128 +1.8 %,
  Qwen3-30B-A3B +1.9 %, Qwen3.6-35B-A3B within noise, replies identical. With more than one GPU (including
  `-sm tensor`) nothing changes.
