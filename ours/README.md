# ours: our production stack on top of the current rdna-boosts release

This branch holds the patch series we run in production, on top of the latest rdna-boosts release. It is
regenerated for each new release; patches the release has taken are dropped. It is not a PR.

**Base:** `v16-6050b467c-r206` (upstream llama.cpp `6050b467c` + this repo's release; resulting tree
`1bac1001f178`). The rest of this branch is that release as published; only `ours/` is added. r180 took #252's 0003 (the
GDN small-projection gather), r182 took #253 (the 1-token router fold and its RDNA4 gate), r183 took #256 (the GDN
decode folds), r187 took #258 (the quantized-KV rotations), r188 took #259 (the shared-expert GLU Q8_1), r198 took #260
(the mixed K-quant x Q6_K row concat), r199 took #262 (the q4_0 KV write fold) and r203 took #266 (host-expert decode
without the per-op routing sync), so they are no longer in this series. r205 and r206 took none of it; the series
replays onto r206 unchanged.

0001 takes #266's device remap and pool cold read back out of the release: this series has carried its own variant of
that remap since before #266 (in 0002, with the review fixes in 0021: freshness rounds, and the remap forwarded through
the Meta backend), and the two cannot both run. After 0001, `moe-expert-cache.cu` is the r200 file again.

**Apply:**

```
git checkout 6050b467c
bash <path-to-this-repo>/scripts/apply-all.sh .     # the r206 release, branch rdna-boosts
git am <path-to-this-repo>/ours/*.patch              # 45 patches
```

Result: tree `b03c0e279f6f`.

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
| 0022-0023 | `--host-experts mmap` RAM tier: K spare RAM slots, and the likely victims copied back ahead on a side stream, so an admission rarely waits for its copy-back (`GGML_MOE_TIER_SPARES`; 0024 makes 8 spares and the 3rd-touch admission the default); 0023 hardens its error paths (a failed event record drains the side stream, a failed wait fails the swap) | `GGML_MOE_TIER_PRECOPY=0` |
| 0024 | RAM tier defaults: 8 spares, admission from the 3rd touch with the copy-back ahead | `GGML_MOE_TIER_SPARES`, `GGML_MOE_TIER_TOUCH` |
| 0025-0026 | `--host-experts mmap` pin budget: one budget for the RAM tier and the sparse registration (hottest tables first, a partial tier instead of all-or-nothing, the warm-up bounded too); a declined read of a sparse master goes through the per-table pinned bounce | `GGML_MOE_PIN_BUDGET_MIB` |
| 0027-0032 | RAM tier at a tight pin budget: a partial tier per layer; a fused MoE op whose lanes cannot be redirected together runs unfused when a lane was taken over; fill-through (a routed expert outside a partial tier is copied in place of its coldest unrouted member, so a partial table never declines) with review fixes; hybrid share (the hottest layers keep a full tier, every other layer at least 30 % of its own) | `GGML_MOE_TIER_SHARE=0`, `GGML_MOE_FUSED_LANE_GUARD=0`, `GGML_MOE_TIER_FILL=0`, `GGML_MOE_TIER_SHARE_MIN_PCT=0` |
| 0033-0035 | debug and analysis tools, off by default: `GGML_MOE_ROUTE_LOG=<file>` (the routing each decode policy call sees, for offline replay; never changes residency), an inclusive RAM tier (`GGML_MOE_TIER_INCL=1`), and byte checks of the arena, RAM tier, device map and cold pointers against the file (`GGML_MOE_VERIFY_AT` / `GGML_MOE_VERIFY_EVERY`) | - |
| 0036-0038 | KV lending, off by default (`GGML_MOE_KV_LEND=1`): the KV cache rows past the context in use hold VRAM copies of evicted experts, so a later miss reads or refills them over VRAM instead of PCIe; the rows go back to the KV cache when the context grows; with lending on, a miss on a victim-held expert is read from it without evicting (`GGML_MOE_KV_LEND_STAY`, default on) | - |
| 0039 | a sparse (mmap) table without a RAM tier admits a routed expert it cannot register (filled through the bounce) instead of declining it per token; the decline could be captured into a CUDA graph and replayed for other routing (wrong output at a tight pin budget) | `GGML_MOE_FORCE_ADMIT=0` |
| 0040-0041 | fill fence: in a decode take-over, the RAM-tier swaps, the device-map upload and the copy-back-ahead waits are ordered by the remap kernel's lag flag instead of one event record each (each record costs ~4-10 us of compute-queue idle on RDNA4) | `GGML_MOE_FILL_FENCE=0` |
| 0042-0043 | review fixes for 0029's fill-through: with every member routed a spare RAM slot takes the expert; the trim keeps a partial tier's registered non-member aliases; a fill waits for the last sparse staging launch on the device (its kernel reads tier slots through source pointers taken at launch); `GGML_MOE_TIER_FILL=0` also turns the share off; 0043 makes the victim-copy check of 0036 skip the spare case | - |
| 0044 | 0039's forced admission also covers a partial-tier table, for shapes that fit its arena (larger shapes keep the cold reads): with every member routed, no spare free and the pin budget refusing the registration, a tight share otherwise read an unregistered alias (wrong output, then a GPU fault); a force-evicted victim with a copy-back ahead waits for it and keeps that RAM copy, otherwise its cold pointer goes back to the file | `GGML_MOE_FORCE_ADMIT=0` |
| 0045 | a captured CUDA graph keeps each MoE op's choice between the expert-cache arena (through the slot remap) and the scheduler copy; when a table declined after the capture, the replay read the previous token's remap (wrong experts). The graph now records which ops read the arena and recaptures when that changes; in normal use it never forces a recapture | `GGML_CUDA_GRAPH_MOE_SIG=0` |

The work the release has taken keeps its switches there: `GGML_CUDA_QK_ROTATE=0`, `GGML_CUDA_MMVQ_ROW_CONCAT_RDNA4=0`,
`GGML_CUDA_MMVQ_CONCAT_REORDER=0`, `GGML_CUDA_ROUTER_TOPK=0`, `GGML_CUDA_DISABLE_GDN_CONV_SNAP=1`,
`GGML_CUDA_FUSE_GDN_CONV_DECODE=0`, `GGML_CUDA_FUSE_SHEXP_GATE_SIGMOID=0`, #258's / #259's (`GGML_CUDA_QK_ROTATE_HADAMARD=0`,
`GGML_CUDA_FUSE_V_FWHT=0`, `GGML_CUDA_FA_COMBINE_FWHT=0`, `GGML_CUDA_QK_PAIR_FWHT=0`, `GGML_CUDA_FUSE_ROPE_FWHT=0`,
`GGML_CUDA_MOE_SHARED_GLU_Q8=0`), `GGML_CUDA_MMVQ_ROW_CONCAT_MIXED=0` (#260) and `GGML_CUDA_KV_FOLD_Q4_0=0` (#262).

## Validation (2026-11-03, Radeon AI PRO R9700, 1 GPU, ROCm 10.1)

r206 vs r206 + this series, both built and run in `rocm/dev-ubuntu-24.04:10.1.0-full`, no env on either side,
`ROCBLAS_USE_HIPBLASLT=0`.

- Greedy server replies (12 multi-turn replies per model) are byte-identical to r206 on Llama-3.2-1B, Qwen3-30B-A3B,
  Qwen3.6-35B-A3B, gemma-4-26B-A4B, Qwen3.8-27B, Qwen3.8-27B NVFP4-MTP (plain and MTP `--spec-draft-n-max 3`) and Qwen3.5-4B;
  also with `-ctk q8_0 -ctv q8_0` and `-ctk q4_0 -ctv q4_0`. No faults.
- KLD on wikitext-2 `-c 512`, 24 chunks, against r206's logits: 0 on all six non-MTP models (same top token 99.98-100 %).
- test-backend-ops (ROCm0, with `ROCBLAS_USE_HIPBLASLT=0`): 21675 / 21676 passed; the one failure,
  `MUL_MAT(type_a=f16,type_b=f32,m=16,n=2105344,k=32)`, comes from that env (the release shows the same with it).
- Our production model (Qwen3.8-Flash-Next IQ3_S, `--host-experts mmap`, MTP draft, `GGML_MOE_KV_LEND=1`) on one GPU:
  this series on r206 vs the same series on r204, 3 alternating fresh loads: greedy replies identical, the needle and the
  256 tokens after it identical, greedy decode 83.5 / 81.7 vs 83.2 / 81.2 t/s, two concurrent streams 107.6 vs 107.6 t/s.
  On r204, 3 alternating fresh loads against our previous production build (0001-0032): greedy replies identical, the 259.6k-token needle answered
  correctly and the 256 tokens after it identical, no faults; the byte checks found 0 differences; greedy decode 83.6 / 81.8
  vs 80.1 / 77.2 t/s (first / second pass, +4.3 / +6.0 %), two concurrent streams 107.4 vs 99.1 t/s (+8.4 %). Of that,
  KV lending is +3.8 / +4.9 % and +9 %; the series with lending off is +0.5 / +1.1 %.
- 0039 was validated on the #267 branch, without 0027-0032: at a 28 GB pin budget, 16 fresh loads all gave the correct
  greedy output (13 of 31 were wrong without it). With 0027-0032 every table gets at least a partial tier at that budget,
  so 0039 is a backstop here.
- 0042-0043 at a 28 GB pin budget (51 partial-tier tables, ~215k fill-throughs per load), 2 fresh loads against 0001-0041
  and the default budget once: greedy output identical, no faults, 0 declines, speed the same; at the default budget no
  table has a partial tier, so they do not run there.
- 0044 at an 8 GB pin budget with `GGML_MOE_TIER_SPARES=1` and `GGML_MOE_TIER_SHARE_MIN_PCT=5`: without it, 2 of 2 fresh loads
  gave wrong greedy output from the first request and then an illegal memory access in the MoE GEMV; with it, correct output
  (9335 forced admissions), no faults. At 20 GB, 16 GB and the default budget it never runs and changes nothing.
- 0045 with a debug-only switch that makes every 7th host-path call of a RAM-tier table decline: without it every pass gave wrong
  greedy output, with CUDA graphs off or with 0045 every pass was correct. In the production config (2 fresh loads): greedy
  output identical, the 259.6k-token needle answered correctly and the 256 tokens after it identical, no faults; graph
  captures 532 vs 543 per load, speed within noise (12 loads per side: greedy -0.4 %, two streams -1.9 %, pass-to-pass
  spread ~4 %). Seven 1-GPU models identical to the r206 release, KLD 0.

llama-bench `-fa 1 -p 512 -n 128 -r 5`, 2 alternating rounds, tg128 means (three single rounds with a +-9..45 t/s spread
left out: r206 Qwen3-30B and Qwen3.5-4B round 2, the series' gemma-4 round 2):

| model | r206 -> r206 + series |
|---|---|
| Llama-3.2-1B UD-Q4_K_XL | 432.8 -> 433.3 (=) |
| Qwen3-30B-A3B Q4_K_M | 164.9 -> 168.6 (+2.2 %) |
| Qwen3.6-35B-A3B Q4_K_M | 138.0 -> 138.9 (+0.7 %) |
| gemma-4-26B-A4B MXFP4_MOE | 93.9 -> 93.7 (=) |
| Qwen3.8-27B Q4_K_M | 31.88 -> 31.96 (=) |
| Qwen3.8-27B NVFP4-MTP-HIGH | 30.51 -> 30.20 (-1.0 %) |
| Qwen3.5-4B Q8_0 | 102.0 -> 102.0 (=) |

The dedup (0017-0018), the RAM-tier work (0022-0032), KV lending and the fill fence help only a model whose routed
experts are partly in host memory: our production model.
