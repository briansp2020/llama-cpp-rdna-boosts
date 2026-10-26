# MANIFESTS — apply order and verification contract

**What this file answers:** in what order the blocks apply, what "verified" means for each one, and what to
do when an apply fails.  The current release identity (fork point, tip/tree, block count, per-artifact
sha256) is [`release.json`](release.json); the dated record of every release and re-validation is
[`WORKLOG.md`](WORKLOG.md).  Dated validation records from the pre-V2.0 era (the 16-block set, its
per-block table and its release-by-release narrative) are preserved in
[`archive/docs/validation-history.md`](archive/docs/validation-history.md) and do **not** describe the
current delivery.

## Current delivery

15 blocks (`block 00` + `blocks 01-14`) against `release.json.base`, canonical chain `release.json.tip` /
`.tree`.  `scripts/validate-set.sh` is green (checksums, then a strict 15/15 apply on a fresh tarball,
tree equality).  **Block 14 is temporary** (`-sm relaxed-tensor`, a local relaxation of an upstream rule,
not a supported feature) -- keep it separate and expect to drop it.  Each block is **one** `git am` commit and the series is **linear**: apply in order, never
reorder, never insert.

## Apply order

| # | patch | subject | notes |
|---|---|---|---|
| 00 | `0000-…-block-00-foundation-…` | foundation: band uniformity, Meta wrapper, allocator | assumed by every later block |
| 01 | `0001-…-block-01-hybrid-HIP-all-reduce-…` | hybrid HIP all-reduce | RDNA4 gate, RCCL elsewhere |
| 02 | `0002-…-block-02-adaptive-MTP-draft-depth` | adaptive MTP draft depth | `--spec-type draft-mtp-adaptive` |
| 03 | `0003-…-block-03-chunked-gated-delta-net-prefill` | chunked GDN prefill | two arch-segregated kernels + rollback tracker |
| 04 | `0004-…-block-04-BF16-KV-cache-…` | BF16 KV + native-BF16 FA | |
| 05 | `0005-…-block-05-flash-attention-band-…` | FA band / native quantized K-V / derived kq mask | every kq-mask fill is guarded on `tensor && tensor->buffer` (r131, PR #222) — the derived form leaves the packed mask unallocated at one slot; the decode/verify band also takes the fully-masked group skip (r133, PR #223), with decode-width packed-hole op cases as its oracle |
| 06 | `0006-…-block-06-MMB-quantized-prefill-GEMM` | MMB quantized prefill GEMM + `mmf32` (the F32 skinny prefill GEMM, r104; default-on for **RDNA4 and RDNA3_5** since r105, opt-in on RDNA3) | |
| 07 | `0007-…-block-07-k-quant-boosts-…` | k-quant boosts (`mmvq` VDR, Q6_K MMQ, Q1_0, grid pad) | **the decode-numerics block** — see `GREEDY-PURITY.md` §2/§9 |
| 08 | `0008-…-block-08-fused-kernels-…` | fused kernels | fusions + fused MoE gate+up+GLU + per-shape `mmvq` `nwarps`; also the `-sm tensor` **model-load staging buffer** (r116, TODO #90) |
| 09 | `0009-…-block-09-host-resident-MoE-…` | host-resident MoE | slab, expert cache, prefill cache staging |
| 10 | `0010-…-block-10-slab-fusion-dispatcher-async-in` | infrastructure: slab, fusion dispatcher, async input ring | the TUs split out of `ggml-cuda.cu`; GDN/rope/MMB fusions |
| 11 | `0011-…-block-11-qwen4exp-…` | qwen4exp | model, hyper-connections, QSA/indexer, lazy reader, indexer cache (r109 split of the old block 10) |
| 12 | `0012-…-block-12-campaign-remainder-…` | campaign remainder | CPU wins, backend iface fills, DFlash (incl. the r130 prompt-ingest tail), FA build grouping, mmap reclaim, batched state restore, server prompt-cache residency, idle-slot parking (r136) |
| 13 | `0013-…-block-13-MTP-draft-vocabulary-subset-…` | MTP draft vocabulary subset | **opt-in** (`--spec-draft-mtp-vocab N`), default off; draft-only |
| 14 | `0014-…-block-14-relaxed-tensor-split-mode-TEMPO` | `-sm relaxed-tensor` (TEMPORARY) | a local relaxation of upstream #31081: tensor split with a per-device MoE amax scalar, for `flint`/`mellum`/`mistral4`.  **Not a supported feature**; delete this patch to return to the r136 tree |

`rdna-boosts-all.patch` (repo root) is the entire net as ONE patch.  It is a convenience for a one-off
build at the fork point; it is **not** the reviewable increment and not a substitute for the per-block
flow.

## Apply (fresh checkout at the fork point)

```bash
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
git checkout "$(jq -r .base /path/to/rdna-boosts/release.json)"
bash /path/to/rdna-boosts/scripts/apply-all.sh .     # strict 15/15 git am, tree checked vs release.json
```

`scripts/apply-all.sh` is the only supported flow.  **Do not `git apply` the concatenated series** — it
silently drops hunks (measured: 30 files / 2483 lines instead of 35 / 6094).  The script:

1. refuses a dirty tree or an existing `rdna-boosts` branch;
2. applies the whole series with strict `git am` (one commit per block, original subjects);
3. on failure, aborts and retries the series **once** with `git am -3` (3-way against the recorded blob
   ids), warning that the merged tree may differ from the canonical one;
4. on the strict path, asserts the applied tree equals `release.json.tree` — a stale fork point or a
   partially regenerated set fails here rather than at the build;
5. `--include-beta` then applies `beta/*/*.patch` on top of the *verified* tree (one commit each).  The
   checkout is then **not** the release tree, which the script reports.

## Release gates (all mandatory, recorded in the release's `WORKLOG.md` entry)

`scripts/validate-set.sh` covers patch integrity only (checksums + a fresh-tarball strict apply) and runs
in CI.  The model-level gates are manual:

1. **Coherence** — same-seed `llama-cli` output identical to a known-good build (dense and, for a change
   that touches the MoE path, a host-expert config).  `GGML_CUDA_ALLREDUCE=nccl` is not a bit-identical
   reference under `-sm tensor`; treat it as a smoke comparison.
2. **MTP** — the adaptive-MTP four-axis gate, `-n 3000` (2000 floor), `--reasoning on` for R and `off` for
   P/C/K, plus the stock-relative `llama-batched-bench -npl 1,4,8` arm.  Protocol:
   [`benchmarks/mtp-adaptive-methodology.md`](benchmarks/mtp-adaptive-methodology.md).
3. **Prefill logits** — `scripts/gate-prefill-logits.sh`: mean KLD <= 0.005 and same-top-p >= 98 % against
   a recorded known-good base.  **Required for any change to a prefill kernel** (GDN/SSM chunked, FA,
   MMQ/MMVF/MMB, MoE prefill) or to a default-on approximate path.  Protocol:
   [`benchmarks/prefill-logit-methodology.md`](benchmarks/prefill-logit-methodology.md).
   A **kernel-family switch** (a new tile/WMMA head cap, e.g. block 05's RDNA3_0 head-512 rows) uses the
   script's `--family-switch` mode, which judges the same absolute limits **at a point inside the gate's
   calibrated domain** and always prints a same-kernel reordering control measured on the spot:
   `--record --ab "<feature-off env>" --family-switch --model <head-256/512 model> --corpus
   prompts/fa-family-switch-gemma4-26B-A4B.txt --ctx 2048 --chunks 10`, then the same command without
   `--record` (`--candidate-env` selects the arm, so a single switch can be judged against a fully-tile
   base).  The absolute limits stay the criterion; `--allow-control-pass` is an explicit opt-in for a
   switch that provably cannot meet them, so an insensitive point cannot pass a bad change.  The mode refuses a point whose base PPL is above `--family-max-ppl` (default 10): at a
   point the model is not confident about, *every* reordering measures far outside the limits (a benign
   `-ub 512` vs `-ub 256` same-kernel change measures 0.105 / 85.0 %, so such a point cannot decide the
   question — `GREEDY-PURITY.md` §46, `archive/work/fa-rdna3-head512-purity/RESOLUTION.md`).
   The `FA_DBG_NMSE` op-level accuracy reference (`test-backend-ops` error magnitude vs the CPU backend at
   the model's own prefill shape) is the second, deterministic leg of a family-switch judgement.
4. **Per-block build** — `scripts/gate-per-block-build.sh` on a clean `all` build per commit (bisection and
   review walk the blocks individually, so every block must build on its own).  Use `--changed-from` to
   rebuild only the blocks a release changed.  **Run it against the long-lived checkout** (its default
   `--repo`, or `scripts/scratch-tree.sh gate`): ccache keys on the source path, so a per-arm worktree makes
   the first block a full cold build (measured: 5m56s / 85 CPU-min at a new path vs 8.6s for the same
   content at the established one).  The gate prints each commit's build time and the run's total; a first
   block that takes minutes on an otherwise unchanged tree means the path was new.
5. **GPU faults** — `scripts/check-gpu-faults.sh` on the run's journal + log (never grep a log by hand).
6. **Idle-slot parking** — `scripts/gate-park-idle-slots.sh`: drives a park/unpark request sequence
   (a long prompt on slot 0, a short one on slot 1, a return to slot 0, a launch on slot 2) and requires the
   parking arm to be **byte-identical** to `LLAMA_SERVER_LAZY_IDLE_SLOTS=0`, with the moves traced and no
   refusal.  Required for any change to the parking path: the KV-cache park API, a memory type that carries
   a second cache, or the server's park scheduling.  **Gates 1-3 never reach parking** — it is inert in
   `llama-cli`/`llama-bench`/`llama-perplexity`, so a server-only defect is invisible to them (issue #227
   shipped in r136 and survived r136-r145 on this fact).

Run the GPU gates **alone**: two concurrent GPU jobs change the arena/band layout and invent failures.

## Verification per block

The per-block command is a *targeted* check for a block's own surface; the release gates above still run
on the tip.

| block | verify | expected |
|---|---|---|
| 00 | `test-backend-ops -o FLASH_ATTN_EXT`; `test-logits-width-probe` W=1..8 | pass; one hash across the band |
| 01 | same-seed coherence on 2 and 3 GPUs (`-sm tensor`), MTP `none == draft-mtp` | identical |
| 02 | `tests/test-speculative-adaptive`, `tests/test-arg-parser`; `llama-server --spec-type draft-mtp-adaptive` smoke | pass |
| 02 | **(r118) MTP prompt-tail readback skip**: `llama-server` with `LLAMA_MTP_INGEST_TAIL=4096` on a >4k prompt, greedy 300 tokens, `LLAMA_MTP_TAIL_READBACK` unset and `=0` | greedy text and draft counts identical to the pre-r118 build in every arm; prefill at the new rate with the switch on and back to the old rate with `=0`; with `LLAMA_MTP_INGEST_TAIL` unset, unchanged (the skip cannot trigger) — `archive/work/mtp-tail-readback/RESOLUTION.md` |
| 03 | `test-backend-ops -o GATED_DELTA_NET`; `test-recurrent-state-depth`, `test-recurrent-state-rollback` | per-arm FAIL/PASS set unchanged from the recorded baseline (the counts differ per box — never gate on zero) |
| 04 | `test-backend-ops` bf16-KV FA cases + a bf16-cache model run | pass; coherence identical to f16 |
| 05 | `test-backend-ops -o FLASH_ATTN_EXT` (full matrix); width probe; prefill-logit gate | 6389/6389 on the r102 tree (gfx1100); probe PASS; KLD <= 0.005 for within-family changes, and the RDNA3_0 head-512 row set passes the absolute limit in `--family-switch` mode (0.001778 / 98.967 % at a base-PPL-2.65 point) — `GREEDY-PURITY.md` §46 |
| 05 | **(r131) the kq-mask fill guards**: `test-llama-archs -a bailingmoe3`; a `bailingmoe3` `llama-cli` at `-np 1` (the hybrid-K + derived-form arm that aborted) and at `-np 2`; the same-seed text against `LLAMA_KQ_MASK_DERIVED=0`; `gate-prefill-logits.sh --model <bailingmoe3 gguf> --ab "LLAMA_KQ_MASK_DERIVED=0"` | 2/2; `-np 1` works and its greedy text is byte-identical to `-np 2`/`-np 4` and to the packed arm; KLD 0.000000 / same-top-p 100.000 % (r131: Ling-3.0-tiny IQ4_NL, `-c 32768`, `-n 128`, seed 42, gfx1201) |
| 05 | **(r133) the band's fully-masked group skip (PR #223)**: `test-backend-ops -o FLASH_ATTN_EXT` (the new decode-width packed-hole cases at `nb` 1/2/3/8, f16/bf16/q8_0, plus `GGML_HIP_FA_BAND_WIDE=0` and `GGML_HIP_FA_BAND_FETCH_AHEAD=0` so all four band kernel variants are covered); the same suite with the kernel's group test **mutated** to skip one visible group (must FAIL, proving the coverage); `test-logits-width-probe` with `KV=q8_0/f16/bf16` and `GGML_CUDA_FA_BAND_SKIP=0/1` (row-0 hashes must match); `plain == draft-mtp` at `-n 3000` on 1 GPU and 2 GPU `-sm tensor`; the four-axis MTP acceptance; the 2-slot `--kv-unified` server cell (a resident ~105K conversation + an active 9.8K one) with `GGML_CUDA_FA_BAND_SKIP=0/1`, and the same with a DFlash2 draft; the no-foreign-cells control (single slot 9.8K and 105K); **(r134, issue #224)** `test-logits-width-probe` with **`P=4096`** (the wide block is only reachable above the launcher's 4096-cell gate; `P=256` never reaches it) on `KV=q8_0/f16/bf16`, `plain == draft-dflash` at `n_max` 3/5/7 x `KV=q8_0/f16/bf16` and at `-c 131072`, and the band's own two-tile arm (`GGML_HIP_FA_BAND_WIDE=0`) as the text-identical control | 6414/6414 both arms of every env variant; the mutation fails 14/14 hole cases; probe **PASS, worst maxdiff 0** (hashes identical with the skip on and off for all three KV types); text identical in all arms; acceptance identical to the last digit (R 0.52697 / P 0.80144 / C 0.86491 / K 0.99200); 50.8 -> 41.7 ms/step (**-18 %**, token ids identical) and 18.6 -> 14.8 with the draft (**-20 %**, ids identical); -0.6 % at 9.8K and neutral at 105K single-slot — `GREEDY-PURITY.md` §37 |
| 06 | `test-backend-ops -o MUL_MAT` (MMQ cases) + prefill A/B; **(r134, issue #224)** `test-logits-width-probe` with **`P >= 4096`** (the probe's `n_ctx` is `P + W + 8`; `K->ne[1]` is the context's *used* length, so a smaller prefix never reaches the RDNA4 band's 64-column block and reports PASS regardless) | pass; prefill not slower at depth; probe `width_purity=PASS`, worst maxdiff 0 on `KV=q8_0/f16/bf16` (the r126 geometry fails it at 0.199764) |
| 06 | **(r132) per-device MMB arch dispatch**: a single-arch coherence + prefill-logit re-run (the change is a no-op when every device has the same cc), and — on a **mixed-arch** box — a 2-GPU `-sm layer` perplexity with the RDNA2 card as the secondary and with it as the primary, plus an `-ot output=<healthy device>` arm | single arch: byte-identical text, KLD 0.000000 (r132: 12/12 configs, both recorded points); mixed arch (the reporter's box, not this fleet): PPL == vocab -> ~2.98 and the ~917 garble -> ~2.98 — `GREEDY-PURITY.md` §56 |
| 06 | **(r104) `mmf32`**: `test-backend-ops -o MUL_MAT` (F32 cases; with `LLAMA_TEST_OP_NMSE=1 LLAMA_TEST_SEED=1` the intercepted cases must be no worse than the rocBLAS path's NMSE) + fresh-start determinism (`llama-perplexity -c 512 --chunks 1`, 12 starts, 1 class; plus 2 GPU `-sm tensor`) + `llama-bench` A/B against `GGML_CUDA_MMF32=0` | 2444/2444 and 11x closer to the CPU reference than rocBLAS (6.1e-14 vs 6.9e-13, threshold 5e-4); 12/12 one class, 6/6 at 2 GPU; `+4.2 % pp512 / +4.1 % pp2048`, decode flat — `GREEDY-PURITY.md` §47.  **(r105) the same three legs on each arch the default is claimed for**; on gfx1151 (RDNA3_5, now default-on) 2444/2444, 12/12 one class, `+1.0 % pp512 / +1.6 % pp2048`, and the `--family-switch` judgement at the shipped confident point (0.013016 / 98.673 % vs a 0.019125 / 97.975 % same-kernel control) |
| 07 | `test-backend-ops -o MUL_MAT`/`MUL_MAT_ID` (K-quant cases); `llama-perplexity` before/after | pass; **PPL must be unchanged** (it is a numerics change, not a quality change) |
| 08 | `test-backend-ops -o MUL_MAT_ID`; MoE decode + `llama-batched-bench -npl 1,4,8`; width probe on a MoE model | pass; one hash across the band; verify widths not slower than stock-relative |
| 08 | **(r145) the rms_norm reduction-order selector (issue #229)**: the new default must reproduce the previous release **with the strided reduction**, i.e. it is compared against a reference binary of the previous tip by rebuilding that tip in the same build dir and running (a) the previous tip as-is vs the new build with `GGML_CUDA_RMS_NORM_VEC=1` (**byte-identical**, proving the opt-in arm did not drift) and (b) the new default's text against the recorded arm, plus the decode-width per-fusion KLD probe (`scripts/gate-prefill-logits.sh --ab "<fusion env>" --ub 1`) for the fold / GDN norm pair / `rms_norm_mul_rope` (must read 0.000000 / 100.000 %), the prefill-logit gate, `test-backend-ops -o RMS_NORM`, a qwen4exp `GGML_CUDA_DISABLE_HC_COMB=1` A/B, MTP Protocol A at `-n 3000` and the per-block build gate | (a) 4B 3-GPU tensor and 27B 1 GPU: text byte-identical (`1c5d32ac537d`, `d55055a5ec80`); (b) 27B plain n=256 `ab94eb7db4d4` = the strided arm, MTP `draft-mtp n3` acceptance 0.79204 = the strided arm; per-fusion KLD **−0.000000 / 100.000 %** (was 0.0005 / 99.2 % on the float4 default); prefill-logit KLD 0.000737 / 98.784 % vs the float4 base, then re-recorded to 0.000000 / 100.000 %; `test-backend-ops -o RMS_NORM` OK; Flash-Next `GGML_CUDA_DISABLE_HC_COMB=1` text identical; MTP Protocol A (27B Q4_K_XL, 1 GPU, `-n 3000`) prose 29.2 → 68.8 t/s and code 29.5 → 73.1 t/s with `plain == draft-mtp` byte-identical on both axes; `validate-set.sh` strict 15/15; per-block build 4/4 (`--fast --changed-from 438cbb3c7`) |
| 08 | **(r132) the Q8_1 cache identity + source guard**: `test-backend-ops -o MUL_MAT`/`-o MUL_MAT_ID` with `LLAMA_TEST_SEED` fixed (unseeded, a marginal `q4_1`/`q5_0` tolerance case flakes on the base build too), the four-axis adaptive gate at `-n 3000` (acceptance and text), and — on a config that reaches the path — a decode that used to fault | 2445/2445 and 931/931; acceptance identical on all four axes and greedy text byte-identical; 0 offset-view keys on the maintainer's seven configs, so the fault arm is the reporter's (`GREEDY-PURITY.md` §55) |
| 09 | **(r144) the traffic report's decode-band census + the `GGML_CUDA_GRAPH_TRACE` graph-call census** (diagnostics only, so the gate is "nothing numeric moved"): a `-ncmoe` host-expert arm long enough for the arena to warm (2000-token decode) must print a non-zero `over ~N tokens = X MiB/token` with `N` matching the realised decode length, and `GGML_CUDA_GRAPH_TRACE=1` must print the `graph_compute calls: direct=... replay=... capture=... properties_changed=...` line; plus same-seed greedy text before/after (`ROCBLAS_USE_HIPBLASLT=0`, generated text only — the banner spinner frame count varies) and the prefill-logit gate | measured `over ~194 tokens = 25.38 MiB/token` on a 200-token run (was `over ~0 tokens = 0.00 MiB/token` on every run); text byte-identical (`8d89080db6735a30`); prefill-logit KLD **0.000531 / 98.863 %** (identical to the recorded base); `validate-set.sh` strict 15/15; per-block build 3/3 |
| 09 | **(r132) split-state cache hit path**: `-sm tensor` coherence (the arm that exercises it) plus `GGML_META_SS_VERIFY=1` on a cache-heavy run; compare the built function's size/codegen against the previous build | text byte-identical; no stale-entry abort; `GGML_META_SS_VERIFY=1` clean (r132: 12/12 configs, 6 of them `-sm tensor`).  The change is codegen-neutral on clang 23/24 at `-O3` — recorded so nobody re-reads it as a speedup |
| 09 | host-expert coherence (`-ncmoe`/`-cmoe`), `test-llama-archs`, `MOE_EXPERT_CACHE_VALIDATE=1`; for the batched f32 CPY: `test-logits-width-probe` default vs `GGML_CUDA_FUSE_CPY_BATCH=0`; for the verify band (r123/r129): an ngram+host-expert arm with a **live arena** must log `draft length 64 -> 15` and the same text as `--spec-ngram-mod-n-max 15`; a **slab-declined** arm (e.g. `GGML_CUDA_SLAB_MIN_ARENA_MIB=999999`) must log **no** clamp and print `llama_context: n_rs_batch = 65` (both with `ROCBLAS_USE_HIPBLASLT=0`); for the Windows HIP build (r124, issue #217): `grep -c 'std::\(min\|max\)<int64_t>' ggml/src/ggml-cuda/moe-expert-cache.cu` | pass; no `////`, no fault; cache-on == cache-off text; probe maxdiff 0 and identical hashes; clamped text == unclamped text, and a device-resident arm logs no clamp; 0 (the file uses `moe_cache_min_i64`/`_max_i64` — an int64_t `std::min`/`std::max` there collides with clang's `cuda_wrappers/algorithm` on Windows) |
| 10 | `test-backend-ops -o FLASH_ATTN_QSA` (18/18 minimum); `test-llama-archs -a qwen4exp`; sparse-vs-dense perplexity ratio (`LLAMA_QSA_SPARSE_FA=0`); `scripts/gate-qwen4exp-quant-coherence.sh`; for the verify-band fusions: `test-logits-width-probe` default vs `GGML_CUDA_FUSE_Q8_1_VERIFY=0` + `GGML_CUDA_FUSE_GDN_CONV_VERIFY=0`; for the async-input ring (r105): `GGML_CUDA_ASYNC_INPUT=0` A/B (identical text + prefill logits) and the two-thread origin count (`GGML_CUDA_ASYNC_INPUT_DEBUG=1`, `archive/work/async-input-per-origin/repro/`) | pass; ratio close to 1; probe maxdiff 0 and identical hashes (`GREEDY-PURITY.md` §11); ring: 2 origins on one device from two threads, `=0` unchanged, no perf regression vs the ring-off arm |
| 10 | **(r108) gate/beta fusion, multi-sequence**: `build/bin/test-fusion-gate-beta-multiseq` (built by default; `ctest -R test-fusion-gate-beta-multiseq`) - it compares the fused launch against the same chain with the fusion made ineligible for `(n_seq_tokens, n_seqs)` = (1,1)..(2,4) and proves the fusion fired; plus a real multi-sequence decode A/B (`GGML_CUDA_FUSE_GATE_BETA_BF16=0`, e.g. two concurrent `llama-server` slots or a lockstep probe) on a BF16-`ssm_alpha` model | PASS (11/11 fused, 0 failed, exit 0); the fusion-off arm bit-identical to the fusion-on arm at every sequence; single-stream text and prefill logits unchanged (`GREEDY-PURITY.md` §49) |
| 10 | **(r132)** the block-10 arms of the Q8_1 fix (the fused norm+gate and GLU writes) take the source-range guard, and `graph_optimize` selects its own device: a GDN + MTP verify-band decode (the norm arm), a MoE decode (the GLU arm) and a `-sm tensor`+`-sm layer` pair | text and acceptance identical to the previous build; no `q8_1 source out of range` line in any run (r132: 12/12 configs, four-axis acceptance identical) |
| 10 | **(r103)** bf16 K/V prefill runs and is bit-identical to the unfused chain: `llama-bench -ctk bf16 -ctv bf16` (must not abort) + a prefill-logit compare against a base recorded with `GGML_CUDA_DISABLE_ROPE_SET_ROWS=1`; `test-backend-ops -o ROPE -o SET_ROWS` | 5136.70 t/s pp512; KLD −0.000000 / 100.000 %; ROPE+SET_ROWS green |
| 10 | **(r139) the compute-buffer chunk follows a live slab (issue #214)**: a slab-declined host-expert arm (`GGML_CUDA_SLAB_MIN_ARENA_MIB=999999`) and a live-arena arm (`MOE_EXPERT_CACHE_MIB=4096`) on 35B-A3B `-ncmoe 24`, reading `sched_reserve: ROCm0 compute buffer size`; `GGML_COMPUTE_BUFFER_CHUNK_MIB=0` and `GGML_COMPUTE_BUFFER_MARGIN_PCT=0` as the controls; same-seed greedy text on the streamed, arena and dense cells | declined and live arms **561 MiB** (was 768), 561 / 510 with the knobs; text byte-identical (streamed and arena sha `4ac3a523160f`, dense 4B 3-GPU `7e6229b260f6`); streaming and arena throughput at parity (r139: 954.8 vs 957.4 t/s at 120k; live arena `pp512` 1143.7 vs 1141.1, `tg128` 63.6 vs 64.4) -- `WORKLOG.md` 261025.012 |
| 11 | dense coherence + the block's own arm (DFlash/CPU/mmap): `-ncmoe` peak-RSS check with and without the reclaim | identical text; RSS as recorded in `patches/README.md` |
| 11 | **(r106) DFlash2 + `-sm tensor`**: a 2- and 3-GPU `--spec-type draft-dflash` run on a selector drafter (coherence, deterministic text, draft acceptance equal to the same config under `-sm layer`), `--parallel 2`, and `-sm tensor` + draft + `-ncmoe` (the slab narrow-2 arm) | all coherent with no override env; acceptance identical to `-sm layer`; `test-backend-ops` 3/3 and `gate-prefill-logits.sh` at KLD <= 0.005 (r106: 2 GPU 84.5 vs 63.4 t/s, 3 GPU 61.1, KLD 0.000000 / 100.000 %) |
| 11 | **(r128) the HC op oracles**: `test-backend-ops -o HC_COMBINE` (the op against its CPU reference; both block_out/inject broadcast shapes, hc 2/4/8, `nt` up to the CUDA band bound) and `-o HC_COMBINE_MIX` (the whole-graph combine -> BF16 mixer pair, i.e. the default-on `GGML_CUDA_FUSE_HC_COMBINE_MIX` arm; `GGML_CUDA_FUSE_HC_COMBINE_MIX=0` is the unfused control) | 20/20 and 6/6 on gfx1201.  `-o HC_COMBINE` ran **0/0 — no oracle at all — before r128** (TODO #100); `GREEDY-PURITY.md` §52 |
| 11 | **(r140, issue #226) the combine+norm fusion's inject read**: a pre-#31028 (separate `hc_*_down`/`hc_*_inject`) qwen4exp GGUF with `LLAMA_FUSED_DSV4_HC_POST=0` — `llama-perplexity` on a 512-token corpus at `-sm tensor` **and** `-sm layer -ts 59,41`; `LLAMA_HC_CN_DEBUG=1` (the window fires, and a declined one names its alias term); the same 4-chunk PPL against the reporter's one-line `ggml_set_output(inject->src[0])` patch; a `--kl-divergence` pair fused vs `GGML_CUDA_DISABLE_HC_COMB=1` | PPL 2690.39 -> **2.4538** (`-sm layer` 2.4235; the default `DSV4_HC_POST` arm unchanged at 2.4409); **24/24 chunk PPLs identical** to the reporter's patch; 6/7 same-seed greedy cells byte-identical between the r139 and r140 builds (the 7th is the chain arm the fix changes).  The KLD pair was **not** a pass at r140: 0.018523 / 95.588 % fused-vs-chain, which the pre-r138 r135 build reproduced (0.021338 / 95.882 %), so it was pre-existing rather than an r140 regression — and **(r141) it is fixed**, not waived: the fused kernel now reduces the RMS sum in `rms_norm_f32`'s vec4 grouping, so the pair is **−0.000000 / 100.000 %** on `-sm tensor` and **0.000000 / 100.000 %** on `-sm layer -ts 59,41`, and the fused arm's greedy text is byte-identical to `GGML_CUDA_DISABLE_HC_COMB=1` (182 chars sha `a99ccbd97ecb`).  r141 also makes the fusion reachable from a **merged-tensor** GGUF (`ggml_cont` inject + a matcher that accepts any pinned, `view_src == nullptr` tensor), validated with a temporary merged-shape emulation arm (fused-vs-chain 0.000000, windows firing) |
| 11 | **(r142) the separate-tensor mixer's inject GEMM is adjacent to the down GEMM**: `llama-bench -p 2048,8192,31500 -n 0` on Flash-Next GSQ-IQ3_XXS (3 GPU `-sm tensor`, all-VRAM, `--load-mode none --lazy-mode off`), the same build with `GGML_CUDA_BF16_SRC1_CACHE=0` as the control, a 2-GPU (`0,2`, x8+x8) `-ncmoe 24` arm, the graph node order (`GGML_SCHED_DEBUG=2 -v`: `hc_attn_down` and `hc_attn_inject` adjacent), the `convert_unary_cont_vec4<float, bf16>` launch count from a `rocprofv3 --kernel-trace`, plus PPL (`-sm tensor`, both `HC_POST` arms), the reporter's `--chunks 24` command, and 6 same-seed greedy cells | **pp2048 2285.9 -> 2357.2 (+3.1 %), pp8192 2206.1 -> 2282.0 (+3.4 %), pp31500 2156.3 -> 2228.6 (+3.4 %)**; xn conversions **4560 -> 2280 launches** (274 -> 85 ms in one pp2048 pass); the control (`SRC1_CACHE=0`) is 2248.7 t/s, i.e. the mechanism, and 2 GPU `-ncmoe 24` 835.5 -> 841.7 (+0.7 %); **PPL 2.4392 / 2.4370 unchanged to the last digit on both arms, all 24 chunk PPLs of the reporter's command identical (2.4202), 6/6 greedy cells byte-identical** (`benchmarks/2026-10-25-qwen4exp-hc-inject-adjacency.md`) |
| 12 | **(r143) the expert cache at a model teardown**: a run that frees and reloads the model between two instances — `llama-bench -p 512 -n 0 -r 1` with a `-ts` *list* (one test per value, so the params differ and llama-bench reloads), on Flash-Next GSQ-IQ3_XXS 2 GPU `-sm layer -ts 59,41 -ncmoe 24 --load-mode none --lazy-mode off`, and on 35B-A3B Q4_K_M 2 GPU `-sm layer -ts 59,41 -ncmoe 40 --host-experts pool MOE_HOST_POOL_MIB=4096`; plus a single-model run's cache log lines (arena, residency, identity fast path) and same-seed greedy text before/after, and a dense 27B run (the forget must be a no-op) | both shapes **SIGSEGV'd before** the fix (exit 139 after the first test: a freed host master filled by the deferred sizing) and **both tests complete** after (572.54 / 943.73 t/s `-ncmoe 24`; 735.92 / 766.12 t/s pool mode); single-model cache lines identical apart from the two new teardown lines, greedy text unchanged (`13406dc51399`; pool `e2dde701fc04`), pp4096 570.20 -> 569.91 and tg64 40.35 -> 40.12 t/s; `scripts/validate-set.sh` strict 15/15, per-block build 2/2, prefill-logit KLD 0.000531 / 98.863 %, `test-backend-ops` HC 88/88 + QSA 26/26 |
| 12 | **(r107) MTP draft vocabulary subset**: `-sm tensor` 2-GPU run with `--spec-draft-mtp-vocab N` on (the model prints `mtp_draft_vocab = N: MTP draft head uses R of V rows` under `-v`); greedy text at `--spec-draft-n-max 7` against `--spec-type none` (prose **and** code, `-n 3000`); the four-axis adaptive gate (R/C/K/P, `-n 3000`) with the flag off vs on; `test-backend-ops`; `test-logits-width-probe SPLIT=tensor`; `scripts/gate-prefill-logits.sh`; a row-range read check for the `ggml-backend-meta` half | text identical inside the purity band; all four axes not slower (r107: R +1.3 %, P +1.7 %, C +2.4 %, K +0.8 %); `test-backend-ops` 21179/21179; probe W=1..8 maxdiff 0; KLD 0.000000 / 100.000 %; `-sm tensor` works without the trim aborting — `GREEDY-PURITY.md` §48 |
| 12 | **(r119) batched state restore**: `test-save-load-state`, `test-state-restore-fragmented`; a server prompt-cache restore vs a fresh `cache_prompt:false` run (48 greedy tokens) on 35B-A3B (plain and DFlash2) and Flash-Next GSQ, 2 GPU `-sm tensor`; four-axis MTP A/B at `-n 3000`; prefill-logit gate | restored text == fresh in every arm; MTP text/acceptance identical stock vs patched; KLD 0.000000 / 100.000 %; `LLAMA_STATE_ASYNC_READ=0` keeps the per-tensor path; one GPU is ~2 ms slower (accepted for the multi-device win) |
| 12 | **(r130) DFlash prompt-ingest tail**: a DFlash2 draft (all layers windowed) on a long prompt, tail on vs `LLAMA_DFLASH_INGEST_TAIL=0`, at two ubatches and one and two slots | greedy text **and draft counts identical**; prefill faster with the tail (r130: +8-12 % at 6K/26K/88K; the draft's share of the prefill 10.2 % -> 1.4 %) and back to the old rate with `=0`; a mixed-pattern draft keeps every row (no window line; the path is by construction, untested here as no mixed-pattern draft was available) (`GREEDY-PURITY.md` §53) |
| 12 | **(r130) server prompt-cache residency**: `tools/server/tests/unit/test_kv_keep_only_active.py` (both arms: the default lazy behaviour and `LLAMA_SERVER_LAZY_IDLE_SLOTS=0`), plus a 2-slot unified-KV ping-pong and the KV-full restore over the server API (a second conversation that overflows the unified cache, then the first again) | 3/3 tests green; the ping-pong switch does not copy state out and back (0.5 -> 0.2-0.3 s at 11.5K, same text); the overflow case restores (4 tokens evaluated) instead of recomputing (8.3 s) and the restored text equals the original; `--no-kv-unified` and `--cache-ram 0` unchanged |
| 12 | **(r146) parking a memory with a mirroring second cache (issue #227)**: `scripts/gate-park-idle-slots.sh` on qwen4exp Flash-Next GSQ-RCO IQ3_XXS + MTP (4 slots, 10.6k-token prompt, `-sm tensor` 3 GPU all-VRAM, `-sm layer` 3 GPU, **and 2 GPU `-ncmoe 44`** in `-sm tensor` and `-sm layer -ts 59,41`), plus the same sequence on the pre-fix build as the negative control (3 GPU and 2 GPU) | park **==** `LLAMA_SERVER_LAZY_IDLE_SLOTS=0` byte for byte on all four requests (A parked, B on the freed low cells, C unparks A, D with two slots parked) in every arm; `LLAMA_SERVER_PARK_IDLE_SLOTS=0` differs on 1-2 of the 4 (config-dependent); both caches log each move under `LLAMA_KV_PARK_DEBUG=1`; no GPU fault; the pre-fix build fails the gate with the reported `GGML_ASSERT` — `GREEDY-PURITY.md` §57 |
| 12 | **(r136) idle-slot parking**: 4-slot `--kv-unified` server, 16 varied prompts two rounds with `cache_prompt: false`, parking on / `LLAMA_SERVER_PARK_IDLE_SLOTS=0` (#210) / `LLAMA_SERVER_LAZY_IDLE_SLOTS=0`, on 27B UD-Q4_K_XL + DFlash2 (gfx1201); plus the shared-prefix `cache_prompt: true` arm, `-fa off`, a sliding-window model, hybrid Qwen3.5-4B and 2-GPU `-sm tensor` | parking **16/16** rounds and **32/32** equal to lazy-off; #210 **13/16** and **13/32**; shared-prefix arm falls back by design (12/16 both arms, 32/32 equal); unsupported caches text-identical on/off; hybrid 16/16 equal to lazy-off; 2-GPU tensor works; prefill KLD 0.000000 / 99.980 % — `GREEDY-PURITY.md` §57 |

## Failure handling (agent instruction)

Apply in the order above, to a branch from `release.json.base`, with `scripts/apply-all.sh`.  If a block
fails to apply:

1. Let the script's `git am -3` retry run; if the merged tree differs from `release.json.tree`, **stop** —
   the checkout was not at the recorded base, or the set is stale.
2. For a true conflict, resolve the hunks **in the block that owns the code** (never in the last block),
   then regenerate the set (`scripts/make-patches.sh`) rather than leaving a hand-edited patch.
3. Do not skip a block.  The series is linear and each block assumes its predecessors; there is no
   supported "apply a subset" flow (including for the k-quant block — see `GREEDY-PURITY.md` §8).
4. Record what happened in `WORKLOG.md`, with the base/tip/tree you ended up with.

## Known notes (working)

- **`test-backend-ops.cpp` is shared by several blocks** (03/04/05/07/08/10/11).  Their hunks sit in different
  case regions, so a re-base is local to the failing file — but grep every patch that touches the file
  before concluding a case is missing.
- **Block 08 is one diff on purpose.**  Its matchers and the `mmvq` / `mmq` / `norm` / `quantize` kernels
  they dispatch to co-developed over one kernel region; do not try to split it.  (The `ggml_cuda_try_fuse`
  translation unit itself is block 10's.)
- **Blocks own their code.**  A fix goes into the block whose surface it changes, even if that means
  amending an early block; appending it to the last block is what made the per-block build gate necessary.
- **The set must apply whitespace-clean.**  `apply-all.sh` prints no git whitespace warnings; a warning is
  a regeneration bug, not a cosmetic issue.
- **Keep UPSTREAM's identifiers in `src/llama-model.{cpp,h}`** — most blocks touch that file, so a local
  rename is paid on every re-base.  Since r128 the loader parameter is `ml` again (the delivery had
  renamed it `ml_l` to silence clang's `-Wshadow`, which re-skinned every conflict in the file — TODO #101);
  the 9 standing `declaration shadows a field` warnings that buys are documented in `src/CMakeLists.txt`
  and are expected.
- **Block-number collisions in old records:** documents written before 2026-10-13 use the
  pre-V2.0 16-block numbering ("block 12" = hybrid all-reduce there, but "block 10" = the k-quant
  umbrella).  `WORKLOG.md`, `archive/` and the dated notes in `archive/docs/patch-notes-history.md` are
  historical and keep their numbering; `patches/README.md` and this file use the current (V2.1) numbers,
  and the block subjects are the table above.  **r109 split the old block 10 in two:** the infrastructure
  stayed block 10 and the qwen4exp architecture became block 11, so an older note that says "block 10" is
  block 10 for slab/dispatcher/async-input and block 11 for qwen4exp/QSA/HC/lazy-reader.

## History

Dated validation records, the retired block-structure churn and the pre-V2.0 per-block verification table:
[`archive/docs/validation-history.md`](archive/docs/validation-history.md).  Baseline/re-base records:
[`archive/docs/baseline-history.md`](archive/docs/baseline-history.md).
