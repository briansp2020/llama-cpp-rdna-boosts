# PR #259 — the shared-expert lane publishes its GLU output's Q8_1 — PROMOTED in r188

Submitter: `briansp2020`.  One patch (`0001`, +33/-6 lines) cut against r184, delivered as
`wip/shared-glu-q8-r184/` in their fork.  **Accepted and folded into block 10** — released as
`v16-6050b467c-r188`, block 10 amended, no block renumbered, blocks 11-15 re-cut unchanged.

## What it does

The 1-token fused MoE gate/up+GLU kernel (`mul_mat_vec_q_moe`, block 10) already publishes the **routed**
lanes' GLU output as Q8_1 for the down projection (`fusion.glu_q8`, r175/PR #250): each block adds its rows
to a self-resetting `(lane, 32-row group)` counter and the block that completes a group quantizes it with one
warp exactly as `quantize_q8_1` does.  The **shared-expert** lane (the last `blockIdx.y`, writing
`shared_dst`) did not, so on a model with a shared expert the shared down projection still ran its own
`quantize_q8_1`.

The patch has that lane publish too, into its own cache entry keyed on the **shared GLU node**
(`fusion.shared_dst`) — the key the shared down `MUL_MAT` looks up — with its counters in the lane **after**
the routed ones (`lane = gridDim.y - 1 = nchannels_dst`), the same 32-lane reductions/rounding and the same
zeroed padding blocks.  `GGML_CUDA_MOE_SHARED_GLU_Q8=0` turns it off; it is nested inside the routed arming
block, so `GGML_CUDA_MOE_GLU_Q8=0` (or any condition that stands the routed publish down: not a 32-lane wave,
not one token, `n_ff % QK8_1 != 0`, a non-contiguous `dst`, the counter budget) stands both down — which is
what makes the entry the kernel fills and the entry the consumer looks up the same thing.

## Assessment (gfx1201 3x R9700 + gfx1100 1x 7900 XTX, ROCm 10.1.0; `benchmarks/2026-10-31-shared-glu-q8.md`)

* **Claims reproduce.**  Launches/token **748 -> 708 on gfx1201** and **688 -> 648 on gfx1100** (exactly one
  per layer, 40 layers, the submitter's 761 -> 721 shape); Qwen3.6-35B-A3B Q4_K_M tg128 **139.54 -> 141.10
  (+1.1 %)** and @4096 **136.47 -> 137.53 (+0.8 %)** against the PR's +1.3 % on an R9700, and on the 7900 XTX
  tg128 **177.90 -> 182.56 (+2.6 %)** / @4096 **178.16 -> 182.28 (+2.3 %)** against the PR's +2.6 % / +2.0 %.
* **Reach is a *quant* property, not only a model one — the one thing the submission could not see.**  The
  delivery's own `shexp_down_gate` fusion (block 08/10, `GGML_CUDA_DISABLE_SHEXP_DOWN_GATE`) already consumes
  the shared GLU + shared down when `ffn_down_shexp` is **Q8_0**, and every Unsloth **UD** quant keeps that
  tensor at Q8_0 (`UD-Q4_K_M`, `UD-Q3_K_M`, `UD-Q5_K_M`, `Q8_0` here) — on all of those the shared down node
  does not exist, so this publish is inert (measured: **839 launches/token before and after**, no perf delta,
  `shexp_down_gated_q8_0` in the census).  It fires where the shared down is its own node, i.e. on a
  conventional quant (`Q6_K`/`Q4_K` shared tensors — the LMStudio `Q4_K_M`, the `UD-IQ3_S`), which is the
  file class the submitter measured.  Quote the file with any number.
* **Purity.**  The `--ub 1` A/B (`gate-prefill-logits.sh --ab "GGML_CUDA_MOE_SHARED_GLU_Q8=0"`) is
  **0.000000 / 100.000 %** — the published blocks are bit-identical to `quantize_q8_1`'s.  Greedy text
  byte-identical at 1 GPU, 2 GPU `-sm tensor` (both x8 pairs), 3 GPU `-sm tensor`, 3 GPU `-sm layer`, a
  `-ncmoe 20` arm and on gfx1100.  MTP: identical text **and** acceptance on a dense 27B in both split modes
  (inert — a verify batch is wider than one token, so nothing publishes) and on a **firing** MoE arm
  (35B-A3B UD-Q4_K_M with the shexp fusion switched off, 7061 chars `f80eab6cf390` and acceptance 0.72137 on
  both builds).  `test-backend-ops -o MUL_MAT_ID` 931/931; `llama-batched-bench -npl 1,4,8` +1.1 % at 1 slot
  and flat at 4/8; no GPU faults.
* **Gaps in the submission, covered here**: it never ran multi-GPU/`-sm tensor` (both split modes, 2 and 3
  devices, are covered above), never ran the prefill-logit gate (the `--ub 1` A/B above is the instrument it
  owes), and reported "Qwen3.6-35B-A3B Q4_K_M" without naming which quant file — which is exactly the
  distinction that decides whether the patch does anything at all.
* **No code change was needed.**  The submission's arming predicate, its counter lane, its padding write and
  its cache key hold up; the fold is content-identical to the validated arm (`git rev-parse HEAD^{tree}` ==
  the applied tree the measurements were taken on).

## Fold mechanics

The patch applied to the r187 tree with offsets only.  In the block chain it was applied on top of **block 10**
(then squashed into that commit with its author/date preserved), blocks 11-15 replayed on top; their trees and
patch bodies are unchanged (only the `From <sha>` line differs, as every re-cut does).  Resulting tree
`61a8923…`, canonical re-cut tip `0e42c6bc2` (`v16-6050b467c-r188`).

Rules a future edit must keep: `GREEDY-PURITY.md` §75.  Variable catalogue: `ENVIRONMENT.md` §7.4.  Block
reference: `patches/README.md` (block 10, "The shared-expert lane's Q8_1 publish").  Release record:
`WORKLOG.md` 261031.006.
