# ours: our production stack on top of the current rdna-boosts release

This branch holds the patch series we run in production, on top of the latest rdna-boosts release. It is
regenerated for each new release; patches the release has taken are dropped. It is not a PR.

**Base:** `v15-6050b467c-r177` (upstream llama.cpp `6050b467c` + this repo's release; resulting tree
`b7adbd125921`).  r177 took #252's 0001 / 0002 (graph orderings), so they are no longer in this series.

**Apply:**

```
git checkout 6050b467c
bash <path-to-this-repo>/scripts/apply-all.sh .     # the r177 release, branch rdna-boosts
git am <path-to-this-repo>/ours/*.patch              # 20 patches
```

Result: tree `f65594740407`. Through 0019 it is the same tree as the r176-based series this replaces (`835100458fb5`), so the validation below applies to it; 0020 has its own check, listed at the end.

## Patches

| | what | env switch (off) |
|---|---|---|
| 0001 | small projections over one input gathered for the row concat (#252's 0003, held by the release for now) | `GGML_CUDA_MMVQ_CONCAT_REORDER=0` |
| 0002 | 1-token router MUL_MAT folded into topk-moe (as posted in #253) | `GGML_CUDA_ROUTER_TOPK=0` |
| 0003 | 0002 steps aside on RDNA4 with more than 128 experts (slower there; as posted in #253's 0002) | (with 0002) |
| 0004 | our earlier stack, squashed (expert cache, MoE / GDN / shared-expert fusions, scheduler and server items) | per feature |
| 0005-0006, 0016 | port fixes for the expert cache and graph recapture | - |
| 0007-0008 | `LLAMA_UB_SMALL_SPLIT` (off by default) | - |
| 0009 | `LLAMA_CKPT_USER_MERGE=K` (off by default) | - |
| 0010-0013 | heat-map debounce, split-state hit copy, threaded large GET_ROWS, device token-embedding gather | - |
| 0014-0015 | `GGML_SCHED_UPFRONT_INPUTS` (opt-in) | - |
| 0017 | GDN direct conv for 1-token decode (superseded by 0019) | - |
| 0018 | fixes: restore the GLU -> Q8_1 mark the rebase dropped; shared-expert gate fold only for BF16 / F32 weights | `GGML_CUDA_FUSE_GLU_Q8_1=0` |
| 0019 | the reviewed GDN series, as posted in #256 (snapshot fold, conv-state gather fold, 1-token direct conv only where it also folds the q / k norm and the prescan does not match, shared-expert gate fold with the MMB check), replacing the older forms in 0004 / 0017 | `GGML_CUDA_DISABLE_GDN_CONV_SNAP=1`, `GGML_CUDA_FUSE_GDN_CONV_DECODE=0`, `GGML_CUDA_FUSE_SHEXP_GATE_SIGMOID=0` |
| 0020 | 0004's pinned input-snapshot ring and lazy split events are on by default only with more than one GPU (each cost about 1 % decode on a single GPU) | `GGML_CUDA_SNAP_RING=0/1`, `GGML_SCHED_LAZY_EVENTS=0/1` force off / on |

The graph orderings now in the release keep their switches there: `GGML_CUDA_QK_ROTATE=0`, `GGML_CUDA_MMVQ_ROW_CONCAT_RDNA4=0`.

## Validation (2026-10-30, Radeon AI PRO R9700, 1 GPU)

r176 vs r176 + the r176-based series (same final tree as r177 + this series), no env (the arms ran in the same session against the same r176 build; llama-bench `-fa 1 -p 512`,
interleaved rounds; KLD on wikitext-2 `-c 512`, 24 chunks, against r176's logits).  Against r177 itself the Qwen3-30B gain is smaller, since r177 already has #252's 0001 / 0002 (about +5.9 % of it on the R9700).

| model | KLD | decode r176 -> series | pp512 |
|---|---|---|---|
| Qwen3-30B-A3B Q4_K_M | - | tg128 156.9 -> 165.2 (+5.3 %) | = |
| Qwen3.6-35B-A3B Q4_K_M | 0 | tg128 129.1 -> 138.2 (+7.0 %) | = (noisy) |
| Qwen3.8-27B Q4_K_M | - | tg64 31.5 -> 32.4 (+2.9 %) | +1.6 % |
| Qwen3.5-4B Q8_0 | at the base file's floor (= r176 vs itself) | tg256 102.8 -> 102.5 (-0.3 %) | - |
| Llama-3.2-1B UD-Q4_K_XL | - | (2026-10-29: -0.8 %) | = |

- test-backend-ops (ROCm0): 21675 / 21675 passed.
- Greedy server replies (12 multi-turn replies per model) match r176 byte for byte on all five models above.
- 2026-10-29, before 0018 / 0019: gemma-4-26B-A4B MXFP4_MOE and Qwen3.8-27B NVFP4-MTP-HIGH were identical to r176 as well, including MTP drafting on (`--spec-type draft-mtp --spec-draft-n-max 3`) and off.
- With our production `LLAMA_CKPT_USER_MERGE=256`, replies on the hybrid and sliding-window models differ, by design: the follow-up turn restarts from a different checkpoint.
- 0020, 2026-10-30, 1x R9700. Compared with forcing both features on (the behaviour before 0020), tg128 over 3 interleaved rounds: Llama-3.2-1B 429.6 -> 437.3 (+1.8 %), Qwen3-30B-A3B 162.4 -> 165.5 (+1.9 %), Qwen3.6-35B-A3B 135.7 -> 136.0 (noise). Greedy server replies are identical on all three. With more than one GPU (including `-sm tensor`) nothing changes.
