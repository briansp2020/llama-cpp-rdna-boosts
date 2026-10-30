# PR #253 assessed — both patches folded into block 10 (r182)

**Status: PROMOTED (both patches).** Contributor PR
[#253](https://github.com/stew675/llama-cpp-rdna-boosts/pull/253) (`@briansp2020`), filed as
`wip/router-topk-fold/` against r176 and applying with `git am` unchanged to r181.  This directory keeps the
submission (both patch files, its `README.md` verbatim) as the record; the delivery carries the change inside
**block 10** (the fusion dispatcher block, which owns `ggml-cuda-try-fuse.cu` and the topk-moe matcher; the
r177/r180 `PR #252` precedent), and nothing here is applied by `scripts/apply-all.sh`.

| patch | verdict | why |
|---|---|---|
| `0001` the 1-token router MUL_MAT folded into topk-moe (`GGML_CUDA_ROUTER_TOPK`) | **taken** | bit-identical logits, the whole win, on both RDNA generations where it measures positive |
| `0002` step aside on RDNA4 above 128 experts | **taken** | reproduces on our R9700 (fold on there is a small loss at 256 experts); RDNA3 keeps the fold at 256 |

**One local amendment on top of `0001`** (shipped, ~5 lines, `ggml_cuda_router_topk_moe_ok`): the predicate now
also refuses a `GGML_HINT_SRC0_IS_HADAMARD` router (the FWHT path in `ggml_cuda_mul_mat`) and a compute-buffer
`src0` whose padding needs clearing (the cuBLAS path), i.e. it mirrors the dispatcher's chain in its own order,
the same discipline `ggml_cuda_mul_mat_takes_mmq` documents.  Both are unreachable for a router (the hint is
only emitted by `llama_mul_mat_hadamard`, a router weight is never a compute-buffer view), so the amendment is
defensive, not a fix.

## What it is

For a 1-token MoE decode, the router (`ffn_gate_inp`) is a `MUL_MAT` whose `src1` has one column, so the
dispatcher runs it as `mul_mat_vec_f<float, float, 1, 256>` and the fused topk-moe runs as a second launch.
`0001` gives `topk_moe_cuda` a `router` mode: block `b` first computes logit `b` with the **same** thread map,
`float2` loads, `#pragma unroll 4`, `ggml_cuda_mad` calls and warp + cross-warp butterfly as `mul_mat_vec_f`,
writes it to the (still allocated, still written) logits tensor, and the block that finishes last (a
`__threadfence()` then one atomic on a self-resetting counter, so no launch crosses another) runs the
**unchanged** selection.  One launch replaces two.

The predicate re-checks the dispatcher's route rather than trusting it: F32 weight/activation/output,
contiguous, `x->ne[1]` in 64/128/256, wave32, `!ggml_cuda_mmb_supported_mm`, `ggml_cuda_should_use_mmvf`, and
the same block-size choice `launch_mul_mat_vec_f_cuda` makes (fewest K iterations, smallest block on a tie) —
and `router == logits` with one row.  It refuses a bias, a delayed softmax, a Hadamard hint and a padded
compute-buffer `src0`.

## Measured (1 GPU, `llama-bench -r 5`, two interleaved rounds, fold on vs off)

| arm | on | off | delta |
|---|---|---|---|
| gfx1100 Qwen3-30B-A3B Q4_K_M tg128 (128 experts) | 212.17 / 211.92 | 205.07 / 206.49 | **+3.1 %** |
| the same, d4096 | 183.08 / 183.17 | 180.19 / 180.11 | **+1.7 %** |
| gfx1100 Qwen3.6-35B-A3B Q4_K_M tg128 (256 experts) | 142.77 / 142.79 | 140.97 / 141.15 | **+1.2 %** |
| the same, d4096 | 139.04 / 138.97 | 137.53 / 137.56 | **+1.1 %** |
| gfx1201 Qwen3-30B-A3B Q4_K_M tg128 | 158.0 / 161.7 / 160.8 | 159.1 / 158.5 / 154.7 | **+1.7 %** (noisy, ±4.4 % sd) |
| gfx1201 Qwen3.6-35B-A3B Q4_K_M (stepped aside by 0002) | 127.8 / 128.7 | 128.6 / 128.5 | flat |
| gfx1201 `0001` only, 35B-A3B (fold on at 256) | 127.8 / 128.7 / 127.5 | — | **-0.3 / -0.5 %** → 0002 is right |
| gfx1100 35B-A3B `--spec-type ngram-mod` | 137.1 / 135.6 | 136.2 / 134.5 | flat (a verify batch is > 1 token) |

The submitter's `+2.0 %` (128 experts) and `+1.1 %` (256) on the 7900 XTX reproduce.  Kernel census
(`rocprofv3`): the per-layer router GEMV is gone; `topk_moe_cuda<128,false,true>` (gfx1201) /
`<256,false,true>` (gfx1100) replaces it.  The remaining `mul_mat_vec_f<float,float,1,256>` in the trace is
the **shared-expert** gate (one row), not the router.

## Purity / gates (r182 release flow)

* Greedy text byte-identical on/off: 30B-A3B (gfx1201 and gfx1100) and 35B-A3B (gfx1100).
* Decode-band logits, i.e. the shapes the fold actually changes:
  `gate-prefill-logits.sh --ab "GGML_CUDA_ROUTER_TOPK=0" --ub 1` (every token a 1-token graph, the fold
  firing) — **0.000000 KLD / 100.000 % same-top-p** on 30B-A3B (gfx1201) and 35B-A3B (gfx1100).
* Canonical dense 27B Q8_0 prefill-logit gate: **0.000000 / 100.000 %**.
* `test-backend-ops -o TOPK_MOE`: **608/608** on all three R9700s (the kernel and `topk_moe_config` refactor).
* MTP Protocol A, dense 27B UD-Q4_K_XL, 3 devices, `-sm layer`: plain 24.98 -> draft-mtp **51.53 t/s**,
  acceptance **0.62134** (= the recorded r181 reference 51.7 / 0.62).
* rule-5 `llama-batched-bench -npp 512 -ntg 128 -npl 1,4,8`: 144.50 / 340.95 / 425.39 -> 139.62 / 344.29 /
  420.85 (the fold is single-token only, so npl>1 is inert).
* per-block build 2/2, `validate-set.sh` green, no GPU faults.

## Facts a future edit must keep

* **The fold is a 1-token shape, so the width is the gate**: `ggml_nrows(logits) == 1` in the dispatcher, and
  the kernel is launched with `n_rows = 1`.  A verify batch (`n_tokens > 1`) must not fold — `mul_mat_vec_f`
  at `ncols_dst > 1` accumulates in a different order and the topk fusion's rows are per-row blocks.
* **The counter is the election, and it is the last `moe_red_counters()` slot** (`GGML_CUDA_MOE_RED_CNT - 1`),
  which the MoE down fold cannot reach (`mmvq-moe.cu` uses `red_cnt + blockIdx.x` under
  `dst->ne[0] <= 2*CNT - 2`).  It is per (device, stream), zeroed once at allocation, and every kernel that
  uses it must reset it before it returns; a launch aborted mid-kernel would leave it non-zero (the same
  caveat the down fold carries).
* **The election is a device-wide barrier inside one launch** (threadfence, one atomic, the winner
  re-reads with a non-temporal load and its own fence).  It is correct only because the logits tensor is
  still written and still allocated: the fold elides the *launch*, never the tensor, so any other consumer of
  `logits` keeps working (a consumer always comes later in node order).  Do not turn the logits write into a
  private scratch buffer.
* **`0002` is a per-device tuning step-aside, not a correctness rule**: RDNA4 at more than 128 experts
  measured slower as one launch (our R9700 reproduces the direction at -0.3..-0.5 %, the submitter's at
  -0.8..-1.0 %), so the fold steps aside there and RDNA3 keeps it.  A new arch needs its own measurement.
* **The fold is unreachable in the delivery's recommended speculative configs** (any speculator's verify
  batch is wider than one token).  It is a plain-decode win, plus a 1-token-draft win that no model we hold
  can exercise (no MoE GGUF here carries an MTP head).
