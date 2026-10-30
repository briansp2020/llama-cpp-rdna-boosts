# ours: our production stack on top of the current rdna-boosts release

This branch holds the patch series we run in production, on top of the latest rdna-boosts release. It is
regenerated for each new release; patches the release has taken are dropped. It is not a PR.

**Base:** `v15-6050b467c-r176` (upstream llama.cpp `6050b467c` + this repo's release; resulting tree
`e990c8611b71`).

**Apply:**

```
git checkout 6050b467c
bash <path-to-this-repo>/scripts/apply-all.sh .     # the r176 release, branch rdna-boosts
git am <path-to-this-repo>/ours/*.patch              # 21 patches
```

Result: tree `835100458fb5`.

## Patches

| | what | env switch (off) |
|---|---|---|
| 0001-0003 | graph orderings for the row concat and the Q / K pair (as posted in #252) | `GGML_CUDA_QK_ROTATE=0`, `GGML_CUDA_MMVQ_ROW_CONCAT_RDNA4=0`, `GGML_CUDA_MMVQ_CONCAT_REORDER=0` |
| 0004 | 1-token router MUL_MAT folded into topk-moe (as posted in #253) | `GGML_CUDA_ROUTER_TOPK=0` |
| 0005 | 0004 steps aside on RDNA4 with more than 128 experts (slower there; as posted in #253's 0002) | (with 0004) |
| 0006 | our earlier stack, squashed (expert cache, MoE / GDN / shared-expert fusions, scheduler and server items) | per feature |
| 0007-0008, 0018 | port fixes for the expert cache and graph recapture | - |
| 0009-0010 | `LLAMA_UB_SMALL_SPLIT` (off by default) | - |
| 0011 | `LLAMA_CKPT_USER_MERGE=K` (off by default) | - |
| 0012-0015 | heat-map debounce, split-state hit copy, threaded large GET_ROWS, device token-embedding gather | - |
| 0016-0017 | `GGML_SCHED_UPFRONT_INPUTS` (opt-in) | - |
| 0019 | GDN direct conv for 1-token decode (superseded by 0021) | - |
| 0020 | fixes: restore the GLU -> Q8_1 mark the rebase dropped; shared-expert gate fold only for BF16 / F32 weights | `GGML_CUDA_FUSE_GLU_Q8_1=0` |
| 0021 | the reviewed GDN series, as posted in #256 (snapshot fold, conv-state gather fold, 1-token direct conv only where it also folds the q / k norm and the prescan does not match, shared-expert gate fold with the MMB check), replacing the older forms in 0006 / 0019 | `GGML_CUDA_DISABLE_GDN_CONV_SNAP=1`, `GGML_CUDA_FUSE_GDN_CONV_DECODE=0`, `GGML_CUDA_FUSE_SHEXP_GATE_SIGMOID=0` |

## Validation (2026-10-30, Radeon AI PRO R9700, 1 GPU)

r176 vs r176 + this series, no env (the arms ran in the same session against the same r176 build; llama-bench `-fa 1 -p 512`,
interleaved rounds; KLD on wikitext-2 `-c 512`, 24 chunks, against r176's logits).

| model | KLD | decode r176 -> series | pp512 |
|---|---|---|---|
| Qwen3-30B-A3B Q4_K_M | - | tg128 156.9 -> 165.2 (+5.3 %) | = |
| Qwen3.6-35B-A3B Q4_K_M | 0 | tg128 129.1 -> 138.2 (+7.0 %) | = (noisy) |
| Qwen3.8-27B Q4_K_M | - | tg64 31.5 -> 32.4 (+2.9 %) | +1.6 % |
| Qwen3.5-4B Q8_0 | at the base file's floor (= r176 vs itself) | tg256 102.8 -> 102.5 (-0.3 %) | - |
| Llama-3.2-1B UD-Q4_K_XL | - | (2026-10-29: -0.8 %) | = |

- test-backend-ops (ROCm0): 21675 / 21675 passed.
- Greedy server replies (12 multi-turn replies per model) match r176 byte for byte on all five models above.
- 2026-10-29, before 0020 / 0021: gemma-4-26B-A4B MXFP4_MOE and Qwen3.8-27B NVFP4-MTP-HIGH were identical to r176 as well, including MTP drafting on (`--spec-type draft-mtp --spec-draft-n-max 3`) and off.
- With our production `LLAMA_CKPT_USER_MERGE=256`, replies on the hybrid and sliding-window models differ, by design: the follow-up turn restarts from a different checkpoint.
