# ours: our production stack on top of the current rdna-boosts release

This branch holds the patch series we run in production, on top of the latest rdna-boosts release. It is
regenerated for each new release; patches the release has taken are dropped. It is not a PR.

**Base:** `v15-6050b467c-r176` (upstream llama.cpp `6050b467c` + this repo's release; resulting tree
`e990c8611b71`).

**Apply:**

```
git checkout 6050b467c
bash <path-to-this-repo>/scripts/apply-all.sh .     # the r176 release, branch rdna-boosts
git am <path-to-this-repo>/ours/*.patch              # 19 patches
```

Result: tree `45df8d1f2fe6`.

## Patches

| | what | env switch (off) |
|---|---|---|
| 0001-0003 | graph orderings for the row concat and the Q / K pair (as posted in #252) | `GGML_CUDA_QK_ROTATE=0`, `GGML_CUDA_MMVQ_ROW_CONCAT_RDNA4=0`, `GGML_CUDA_MMVQ_CONCAT_REORDER=0` |
| 0004 | 1-token router MUL_MAT folded into topk-moe (as posted in #253) | `GGML_CUDA_ROUTER_TOPK=0` |
| 0005 | 0004 steps aside on RDNA4 with more than 128 experts (slower there) | (with 0004) |
| 0006 | our earlier stack, squashed (expert cache, MoE / GDN / shared-expert fusions, scheduler and server items) | per feature |
| 0007-0008, 0018 | port fixes for the expert cache and graph recapture | - |
| 0009-0010 | `LLAMA_UB_SMALL_SPLIT` (off by default) | - |
| 0011 | `LLAMA_CKPT_USER_MERGE=K` (off by default) | - |
| 0012-0015 | heat-map debounce, split-state hit copy, threaded large GET_ROWS, device token-embedding gather | - |
| 0016-0017 | `GGML_SCHED_UPFRONT_INPUTS` (opt-in) | - |
| 0019 | GDN direct conv for 1-token decode on RDNA4 / RDNA3_0 | `GGML_CUDA_FUSE_GDN_CONV_VERIFY=0` |

## Validation (2026-10-29, Radeon AI PRO R9700, 1 GPU)

r176 vs r176 + this series. The series ran with our production env (`GGML_SCHED_UPFRONT_INPUTS=1`,
`LLAMA_CKPT_USER_MERGE=256`, `GGML_HIP_GRAPH_UPDATE_CAP=4`, `LLAMA_SERVER_LAZY_IDLE_SLOTS=0`).
llama-bench `-fa 1 -p 512 -n 128 -r 5`, 2 alternating rounds; KLD on wikitext-2 (`-c 512`, 24 chunks) against r176's
logits.

| model | KLD | tg128 r176 -> series | pp512 |
|---|---|---|---|
| Llama-3.2-1B UD-Q4_K_XL | 0 | 422.5 -> 419.0 (-0.8 %) | = |
| Qwen3-30B-A3B Q4_K_M | 0 | 156.4 -> 166.9 (+6.7 %) | = |
| Qwen3.6-35B-A3B Q4_K_M | 0 | 127.1 -> 134.7 (noisy run) | +3 % (noisy) |
| gemma-4-26B-A4B MXFP4_MOE | 0 | 93.6 -> 94.0 (=) | = |
| Qwen3.8-27B Q4_K_M | 0 | 31.3 -> 31.9 (+1.7 %) | +1.3 % |
| Qwen3.8-27B NVFP4-MTP-HIGH | - | 30.0 -> 30.4 (+1.5 %) | +2.5 % |

- test-backend-ops (ROCm0): 21675 / 21675 passed.
- No faults on any model.
- Greedy server replies (12 multi-turn replies per model):
  - Without our env, they match r176 byte for byte on all six models. On the MTP model this holds with MTP drafting on (`--spec-type draft-mtp --spec-draft-n-max 3`) as well as off.
  - With `LLAMA_CKPT_USER_MERGE=256`, replies on the hybrid and sliding-window models differ, by design: the follow-up turn restarts from a different checkpoint.
