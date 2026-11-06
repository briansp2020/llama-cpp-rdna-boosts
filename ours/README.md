# ours: our production stack on top of the current rdna-boosts release

This branch holds the patch series we run in production, on top of the latest rdna-boosts release. It is
regenerated for each new release; patches the release has taken are dropped. It is not a PR.

**Base:** `v16-6050b467c-r228` (upstream llama.cpp `6050b467c` + this repo's release; resulting tree
`44b8fc4d8b92`). The rest of this branch is that release as published; only `ours/` is added. r180 took #252's 0003 (the
GDN small-projection gather), r182 took #253 (the 1-token router fold and its RDNA4 gate), r183 took #256 (the GDN
decode folds), r187 took #258 (the quantized-KV rotations), r188 took #259 (the shared-expert GLU Q8_1), r198 took #260
(the mixed K-quant x Q6_K row concat), r199 took #262 (the q4_0 KV write fold) and r203 took #266 (host-expert decode
without the per-op routing sync), so they are no longer in this series. r205-r228 took none of it.

0000 keeps `moe-expert-cache.cu` at its r206 state: r207 rewrote the host pool path in that file (the pool-tier gather,
per-table pool sizing, the `MOE_HOST_POOL_RAM_PCT` ceiling), and that code sits where this series replaces the release's
remap and changes the bounce, gather and report paths for `--host-experts mmap`. So with this series `--host-experts pool`
behaves as in r206, and `moe_cache_table_is_pool` answers false, so the scheduler's new default gathers nothing (r206's
default). Everything else in r207-r228 (RDNA2 flash attention, Windows host memory, the scheduler and loader changes) is
in, as released, and so are r210-r228, except the host pool changes of r219, r223, r224, r227 and r228 in that file (r219's rollback rule is ported in 0050).

0001 takes #266's device remap and pool cold read back out of the release: this series has carried its own variant of
that remap since before #266 (in 0002, with the review fixes in 0021: freshness rounds, and the remap forwarded through
the Meta backend), and the two cannot both run. After 0001, `moe-expert-cache.cu` is the r200 file again.

**Apply:**

```
git checkout 6050b467c
bash <path-to-this-repo>/scripts/apply-all.sh .     # the r228 release, branch rdna-boosts
git am <path-to-this-repo>/ours/*.patch              # 55 patches (0000-0054)
```

Result: tree `a1032ded184f`.

## Patches

| | what | env switch (off) |
|---|---|---|
| 0000 | keeps `moe-expert-cache.cu` at r206 (r207's host-pool-speed is not carried: pool mode as in r206); `moe_cache_table_is_pool` answers false | - |
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
| 0046 | lends the prefill prefetch's two staging buffers (2 x ~957 MiB on the production model, kept for the whole run) to KV lending between prefills: while no prefill runs they hold copies of experts the arena evicted, like the KV rows past the context in use. The first prefetched op of a prefill takes them back (one device sync, the copies dropped); a few decode steps without a prefill lend them again. Single device only | `GGML_MOE_PF_LEND=0` |
| 0047 | `GGML_CUDA_SLAB_WORK_MIN_MIB` sizes the slab's work band when the model loads. With `GGML_CUDA_SLAB_RING_PREFILL=0`, a prefill view wider than the band learned from the reserves (`-ub 4096`: 2304 MiB vs 1792) was placed above the unused ring candidate; that moved the boundary ~6.6 GB after the arena was filled, evicted arena tables and switched the RAM tier off for the run (decode 83 -> 26 t/s on the production model). Production sets it to 2304 with `-ub 4096 -b 4096` | unset (default): unchanged |
| 0048 | the forced admission's victim cold pointer kernel launches with its device current: the eager hook runs with whichever device computed last, so on two GPUs the launch could go to the wrong one. One device: no change | none |
| 0049 | KV lending also covers the main indexer key cache. It holds one raw key row per KV cell on the attention cache's slot layout (12 tensors, 408 MiB on the production model), and rows past the context in use are idle, so they follow the attention cache's lend line and hold evicted-expert copies too. Models with a k-pool indexer (glm5-next) stay out | `GGML_MOE_KV_LEND_IDX=0` |
| 0050 | a failed host-source fill rolls the admission back and stands the table down, so the scheduler serves it from the host master. Before this, a slot whose bytes never arrived stayed resident and served stale data: wrong output, only a warning. Same rule as r219 (issue #274), on our r206 file, plus a decline right after the first pass of the same call. The RAM tier and sparse paths already rolled back and production tables only use those | `GGML_MOE_FILL_ROLLBACK=0` |
| 0051 | decode admissions are batched: inside one remap call, a layer's victim saves, fills, cold pointer writes and map uploads are queued and launched just before the remap, one layer step = two kernels. Same bytes and pointers, fewer dispatches. On the production config, greedy +1.6 % and the first answer after a cold load +12.5 % | `GGML_MOE_FILL_BATCH=0` |
| 0052 | keeps the GDN conv inputs alive to the end of its q/k norm pair, so the allocator can no longer place the norm outputs over them and decline the fold depending on the graph layout. 1..255 tokens only. Neutral on the production model, a gain on Qwen3.6-35B-A3B (28 of 30 GDN layers fold again) | `GGML_CUDA_GDN_CONV_NORM_HOLD=0` |
| 0053 | the q8_0 `flash_attn_tile` converts int8 to f16 through `v_perm`, on RDNA3 and RDNA4. Same product of the same operands, fewer instructions. This is the tile kernel only: Qwen3-30B-A3B at depth 32768 +3.8 %, the production model never launches it | `GGML_CUDA_FA_TILE_PERM=0` |
| 0054 | `GGML_MOE_VERIFY_AT` and `GGML_MOE_VERIFY_EVERY` flush the fill batch before they compare. A check that fell inside a layer saw the host bookkeeping of the admissions queued by 0051, ahead of the device, and reported differences that were not real. Debug path only | none |

The work the release has taken keeps its switches there: `GGML_CUDA_QK_ROTATE=0`, `GGML_CUDA_MMVQ_ROW_CONCAT_RDNA4=0`,
`GGML_CUDA_MMVQ_CONCAT_REORDER=0`, `GGML_CUDA_ROUTER_TOPK=0`, `GGML_CUDA_DISABLE_GDN_CONV_SNAP=1`,
`GGML_CUDA_FUSE_GDN_CONV_DECODE=0`, `GGML_CUDA_FUSE_SHEXP_GATE_SIGMOID=0`, #258's / #259's (`GGML_CUDA_QK_ROTATE_HADAMARD=0`,
`GGML_CUDA_FUSE_V_FWHT=0`, `GGML_CUDA_FA_COMBINE_FWHT=0`, `GGML_CUDA_QK_PAIR_FWHT=0`, `GGML_CUDA_FUSE_ROPE_FWHT=0`,
`GGML_CUDA_MOE_SHARED_GLU_Q8=0`), `GGML_CUDA_MMVQ_ROW_CONCAT_MIXED=0` (#260) and `GGML_CUDA_KV_FOLD_Q4_0=0` (#262).

## Validation on r228 (2026-11-06, Radeon AI PRO R9700, 1 GPU, ROCm 10.1)

For a Linux build, r227 and r228 reach this series only as Windows-guarded lines in `wddm-mem.cu`, because their other change is in
the file 0000 holds at r206. So the source is the same as the r226 validation below, plus 0054, and that run was not repeated in
full. The binaries do differ from r226, the cause not established, so these checks ran on the r228 binary: the varied-prompt test on
Qwen3-30B-A3B and Qwen3.6-35B-A3B at 15K and 30K tokens, same text as r226 (before 0054). On the production config with 0054: greedy,
prompt-test and post-needle output identical, no faults or evictions, `GGML_MOE_VERIFY_AT=300000` 0 differing slots.
`GGML_MOE_VERIFY_EVERY=997` over short traffic: 354 passes, none with differences; 35 of 361 had differences without 0054.

## Validation on r226 (2026-11-06, Radeon AI PRO R9700, 1 GPU, ROCm 10.1)

r226 vs r226 + this series, built without ccache, same method as the r223 run below: greedy server replies byte-identical on all
seven models, KLD 0, no faults; the varied-prompt test at four depths gives the same text as stock on Qwen3-30B-A3B and
Qwen3.6-35B-A3B. test-backend-ops 21678/21680: the known `MUL_MAT` f16 case on `ROCBLAS_USE_HIPBLASLT=0`, and one
`MUL_MAT_ID` q4_1 case (`m=512, n=1, k=256`, error 0.000605 against 0.0005). That second one did not repeat: `MUL_MAT_ID` passed
931/931 in 3 more runs on stock r226 and 3 on this series. The test draws random data each run, so this is probably a borderline
draw, not proven. On the production config against the r223 series, 2 interleaved loads per side: greedy 85.6 vs 84.9 t/s, longdec
72.7 vs 72.1 t/s, midpf decode 93.8 vs 92.1 t/s, two concurrent streams 117.7 vs 112.2 t/s, the last with the r223 side low against
its earlier runs today. Greedy, prompt-test and post-needle output identical, no faults or evictions,
`GGML_MOE_VERIFY_AT=300000` 0 differing slots. Batch and fill fault arms as below: handled, same output.

## Validation on r223 (2026-11-06, Radeon AI PRO R9700, 1 GPU, ROCm 10.1)

r223 vs r223 + this series, built without ccache: greedy server replies byte-identical on all seven models, KLD 0, no faults;
test-backend-ops 21675/21676 (the known `MUL_MAT` f16 case on `ROCBLAS_USE_HIPBLASLT=0`). New since r220, because short replies
missed a text change in an earlier form of 0053: a varied, non-repeating prompt at four depths, 3.7K to 30K tokens, q8_0 KV.
Qwen3-30B-A3B, Qwen3.6-35B-A3B and Llama-3.2-1B give the same text with 0053 on and off, and the first two the same as stock
r223. An earlier form of 0053 that also fetched the next load phase ahead changed the text at 7.4K tokens and deeper, on both
gfx1201 and gfx1100, so it is not in this series; the cause is not known. On the production config: greedy, prompt-test and post-needle
output identical, no faults or evictions, `GGML_MOE_VERIFY_AT=300000` 0 differing slots. 0051 with
`GGML_MOE_FILL_BATCH_FAULT=2000`: handled, same output. `GGML_MOE_FILL_FAULT` on Qwen3-30B-A3B (`--host-experts` pool and pinned,
fills failing from the 1st or the 200th): all replies identical to the run without faults.

0053, `llama-bench` tg128, Qwen3-30B-A3B q8_0 KV, off vs on: depth 32768 103.2 vs 107.1 t/s, depth 4096 148.1 vs 149.6 t/s;
Qwen3.6-35B-A3B at depth 32768 within noise. `flash_attn_tile` median 75.2 vs 66.9 us.

The release's RDNA4 band fetch-ahead, which production does run, was checked the same way with `GGML_HIP_FA_BAND_FETCH_AHEAD=0`
(a trace confirms the kernel changes): the same text at varied prompts of 30K, 123K and 237K tokens, longdec 72.2 vs 70.5 t/s
with it on, short decode unchanged.

## Validation on r220 (2026-11-05, Radeon AI PRO R9700, 1 GPU, ROCm 10.1)

r220 vs r220 + this series, same method as the runs below: greedy server replies byte-identical on all seven models, KLD 0, no
faults; test-backend-ops 21675/21676 (the known `MUL_MAT` f16 case on `ROCBLAS_USE_HIPBLASLT=0`). On the production config
against the r218 tree, 2 interleaved loads per side: greedy 85.4 vs 84.5 t/s, two concurrent streams 112.5 vs 113.8 t/s, longdec
73.5 vs 71.8 t/s, prefill unchanged, all within the load spread; greedy, prompt-test and post-needle output identical; no
faults or evictions. `GGML_MOE_VERIFY_AT=300000`: 0 differing slots. 0050 with `GGML_MOE_FILL_FAULT` on Qwen3-30B-A3B
(`--host-experts` pool and pinned, fills failing from the 1st or the 200th): all replies identical to the run without
faults. Without the decline after the first pass, pool mode at the 200th fill gave NaN-like output.

## Validation on r218 (2026-11-05, Radeon AI PRO R9700, 1 GPU, ROCm 10.1)

r218 vs r218 + this series, same method as the r209 run below: greedy server replies are byte-identical on all seven models
(plain, quantized KV and MTP where that run had them), KLD 0, `llama-bench` within noise, no faults. test-backend-ops
21675/21676; the one failure is the known `MUL_MAT` f16 case on `ROCBLAS_USE_HIPBLASLT=0`, the same as on r209.

0049 on the production config at `-c 262144`, against `GGML_MOE_KV_LEND_IDX=0`, 2 interleaved loads per side: lent slots 6232 vs
5788, cold reads served from VRAM +7.4 %, greedy decode 84.9 vs 83.6 t/s, two concurrent streams 113.9 vs 110.7 t/s, within
the load-to-load spread. Greedy, prompt-test and post-needle output identical, no faults or evictions. One load with
`GGML_MOE_VERIFY_AT=300000`: 0 differing arena, RAM-tier, map, cold-pointer or KV-victim slots (5956 KV victims). Prefill is
unchanged.

## Validation on r209 (2026-11-04, Radeon AI PRO R9700, 1 GPU, ROCm 10.1)

r209 vs r209 + this series, both built and run in `rocm/dev-ubuntu-24.04:10.1.0-full`, no env on either side,
`ROCBLAS_USE_HIPBLASLT=0`.

- Greedy server replies (12 multi-turn replies per model) are byte-identical to r209 on Llama-3.2-1B, Qwen3-30B-A3B,
  Qwen3.6-35B-A3B, gemma-4-26B-A4B, Qwen3.8-27B, Qwen3.8-27B NVFP4-MTP (plain and MTP `--spec-draft-n-max 3`) and Qwen3.5-4B;
  also with `-ctk q8_0 -ctv q8_0` and `-ctk q4_0 -ctv q4_0`. No faults.
- KLD on wikitext-2 `-c 512`, 24 chunks, against r209's logits: 0 on all six non-MTP models (same top token 99.98-100 %).
- test-backend-ops (ROCm0, with `ROCBLAS_USE_HIPBLASLT=0`): 21675 / 21676 passed; the one failure,
  `MUL_MAT(type_a=f16,type_b=f32,m=16,n=2105344,k=32)`, comes from that env (the release shows the same with it): rocBLAS's f16 -> f32
  `gemm_ex` then leaves every output column from 2^21 on unwritten, with a success status; hipBLASLt (the default) is correct.
- Our production model (Qwen3.8-Flash-Next IQ3_S, `--host-experts mmap`, MTP draft, `GGML_MOE_KV_LEND=1`,
  `LLAMA_SPEC_BATCH_BUDGET=6`) on one GPU: this series on r209 vs the same series on r206, 3 alternating fresh loads:
  greedy replies identical, the 259.6k-token needle answered correctly and the 256 tokens after it identical, no faults,
  greedy decode 83.7 / 82.9 vs 82.7 / 80.9 t/s, two concurrent streams 110.2 vs 110.2 t/s. On r206 (2026-11-03),
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
- 0046 on the production config (`GGML_MOE_KV_LEND=1`), 8 interleaved pairs against 0000-0045: greedy decode +3.5 % (95 % CI
  +2.3..+4.7), two concurrent streams +4.6 % (+2.7..+6.6), fresh 4.4k-token prompts -0.6 % (the take-back, ~32 ms per
  prompt), greedy and prompt-test output identical in all 16 loads, no faults; one load with the 259.6k-token needle: same
  prefill speed (1841 t/s), correct answer, the 256 tokens after it identical.
- 0047 on the production config, `-ub 4096 -b 4096` + `GGML_CUDA_SLAB_WORK_MIN_MIB=2304` against 0000-0046 at `-ub 2048`, 2 loads
  per side: fresh 101k-token prompt +10.6 %, 259.6k-token needle prefill +12.4 % (125 vs 140 s), 4.4k-token prompts +7.1 %,
  greedy decode -1.9 %, two concurrent streams -3.3 %, arena 12.7 vs 13.3 GB; no evictions, RAM tier stayed on, no faults, needle
  correct in every load. This is a batch size change, so it is not output-identical: short-prompt greedy output and the 101k
  reply are identical, but replies to prompts longer than one 2048-token batch can change. With the env unset, the code path
  does not change. The cause of the decode loss is not established; the smaller arena is the likely part.

llama-bench `-fa 1 -p 512 -n 128 -r 5`, 2 alternating rounds, tg128 means:

| model | r209 -> r209 + series |
|---|---|
| Llama-3.2-1B UD-Q4_K_XL | 432.8 -> 433.8 (=) |
| Qwen3-30B-A3B Q4_K_M | 166.8 -> 167.4 (+0.4 %) |
| Qwen3.6-35B-A3B Q4_K_M | 137.8 -> 138.6 (+0.6 %) |
| gemma-4-26B-A4B MXFP4_MOE | 93.9 -> 93.6 (=) |
| Qwen3.8-27B Q4_K_M | 31.70 -> 32.01 (+1.0 %) |
| Qwen3.8-27B NVFP4-MTP-HIGH | 30.59 -> 30.20 (-1.3 %) |
| Qwen3.5-4B Q8_0 | 102.0 -> 102.0 (=) |

The dedup (0017-0018), the RAM-tier work (0022-0032), KV lending and the fill fence help only a model whose routed
experts are partly in host memory: our production model.
