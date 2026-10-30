# PR #256 assessed — all four patches folded into block 10 (r183)

**Status: PROMOTED (all four).** Contributor PR
[#256](https://github.com/stew675/llama-cpp-rdna-boosts/pull/256) (`@briansp2020`), filed as
`wip/gdn-decode-folds-r176/` against r176 and applying with `git am` unchanged to r182.  This directory keeps
the submission (all four patch files, its `README.md` verbatim) as the record; the delivery carries the change
inside **block 10** (the fusion dispatcher block, the r177/r180/r182 precedent), and nothing here is applied
by `scripts/apply-all.sh`.

It is the follow-up to PR #249's declined `0002`: the maintainer's review there asked for the 1-token direct
conv to be gated on `gdn_conv_norm_match() > 0`, to check the decode-only `ssm_prescan` first, to fix the
`return 0`/`continue` slip, and to measure it on both gfx1100 and gfx1201.  All four are done here, and the win
comes from the two folds the gated path then enables.

| patch | verdict | why |
|---|---|---|
| `0001` the direct conv also writes the recurrent snapshot tails (`GGML_CUDA_DISABLE_GDN_CONV_SNAP=1` off) | **taken** | bit-identical, +2.2 % of the composed win on gfx1201 |
| `0002` the conv-state `GET_ROWS` is deferred into the direct kernel (`GGML_CUDA_FUSE_GDN_CONV_GATHER=0` off) | **taken** | bit-identical, +1.0 % |
| `0003` the 1-token direct conv, gated (`GGML_CUDA_FUSE_GDN_CONV_DECODE=0` off) | **taken** | neutral on its own (+0.3 %), but it is what lets 0001/0002 fire at one token; the dense Q8_0 case that cost −1.1 % in #249 is neutral again |
| `0004` the shared-expert gate GEMV + sigmoid/mul/add in one launch (`GGML_CUDA_FUSE_SHEXP_GATE_SIGMOID=0` off) | **taken** | bit-identical, +0.6 % on Flash-Next (the only model we hold whose gate weight is BF16/F32); inert on every quantized model |

## Measured (1 GPU, `llama-bench -r 5`, two interleaved rounds, all four switches on vs all four off)

| arm | on | off | delta |
|---|---|---|---|
| gfx1100 Qwen3.6-35B-A3B Q4_K_M tg128 | 150.70 / 150.48 | 142.27 / 142.17 | **+5.9 %** |
| the same, d4096 | 146.77 / 146.96 | 139.01 / 139.02 | **+5.6 %** |
| gfx1201 Qwen3.6-35B-A3B Q4_K_M tg128 | 108.32 / 107.81 | 105.00 / 104.98 | **+3.2 / +2.7 %** |
| the same, d4096 | 106.04 / 106.27 | 102.82 / 103.05 | **+3.1 %** |
| gfx1201 dense 27B UD-Q4_K_XL tg128 | 31.00 / 31.07 | 30.49 / 30.48 | **+1.7 %** |
| gfx1201 Qwen3.5-4B Q8_0 tg128 (the #249 −1.1 % case) | 103.57 / 104.19 | 104.08 / 104.16 | neutral |
| gfx1201 Flash-Next GSQ-IQ3_XXS, `-sm layer -ncmoe 48`, `0004` alone | 45.63 / 45.67 | 45.34 / 45.34 | **+0.6 %** |

Decomposition (gfx1201, 35B-A3B): base 104.98 → `0003` 105.34 (+0.3 %) → `+0001` 107.39 (+2.3 %) →
`+0002` 108.19 (+3.1 %) → `+0004` 108.43 (inert here; this model's gate weight is quantized).

The submitter's own numbers (+7.2 % on a 7900 XTX, +5.6 % on an R9700, both against r176) reproduce in
direction; against r182 on our box they measure +5.9 % (gfx1100) and +3.3 % (gfx1201).

## Purity / gates (r183 release flow)

* Greedy text byte-identical on/off: 35B-A3B Q4_K_M (1 GPU), 4B Q8_0 (1 GPU), 2-GPU `-sm tensor -ncmoe 40`,
  and Flash-Next `-sm layer` (the `0004` arm).
* `gate-prefill-logits.sh --ab "<all four off>" --ub 1` (every token its own graph, which is the only width
  `0003` fires at) — **0.000000 KLD / 100.000 % same-top-p** on 27B Q8_0.
* Canonical 27B Q8_0 prefill-logit gate against the r182 base: **0.000000 / 100.000 %**.
* MTP, 35B-A3B, `--spec-type draft-mtp --spec-draft-n-max 7`: draft/accept **362 / 203** on both arms
  (identical), 182.8 vs 182.4 t/s in the single-run arms.
* server `--parallel 2 --kv-unified`: clean, no assert (a multi-sequence batch declines the gather deferral).
* per-block build 3/3 `--fast`, `validate-set.sh` green (tree `3eb7762043`), no GPU faults.

## Facts a future edit must keep

* **The folds are inert on a multi-device graph.**  They sit behind the pre-existing `gdn_conv_enabled()`
  (single device unless `GGML_CUDA_CONV_FUSION_MULTI=1`), so the recommended `-sm tensor -ncmoe` server shape
  gets none of them.  That is the same gate the direct conv has had since r95 (its tensor-split numerics were
  never reconciled), not a new restriction.
* **`0003` is an enabler, not a win.**  On its own it measures +0.3 % (gfx1201) and the dense Q8_0 case is
  neutral only because it now requires `gdn_conv_norm_match() > 0` (the decline keeps the `ssm_conv_in`
  epilogue and its free CONCAT write).  Do not relax that gate to "win" the decode case on dense models.
* **The compute-time matchers and `graph_optimize` must keep deciding the same way.**  The decode gate is
  data-pointer dependent (it asks the norm matcher and the prescan matcher), so it lives in
  `ggml_cuda_gdn_conv_match_at_concat` / `_at_conv`; `graph_optimize` uses the structural
  `ggml_cuda_gdn_conv_pattern_at_concat`, which is a **superset**, so its alloc deps cover every case the
  compute-time matchers take.  A new compute-time caller must go through those two matchers.
* **The gather fold's tripwire is deliberate.**  A CONCAT over a deferred gather that the direct conv does not
  take aborts (`write_tail` and the CONCAT both assert) instead of reading an unwritten `GET_ROWS`; the two
  decisions share one predicate, so a mismatch is a code error, not a config.
* **The snapshot fold needs the runtime byte-span check.**  It may only fire when nothing between the CONCAT
  and the conv writes over `state`/`x` or touches a snapshot destination, and the destinations are disjoint;
  the batched-copy run must stop at a folded snapshot CPY (else `cpy_batch` swallows it).
* **The direct kernel reads the state row and writes it back**, so it preloads each channel's three state
  values into registers before the snapshot writes.  Keep that order.
* **`0004` needs BF16/F32 gate weights** (F16 can take a half accumulator, which the fused path does not
  support) and may only fire where the gate, the sigmoid and the product have no other reader.  It elides the
  *stores*, not the tensors' identities, so a model change that reads the gate value disables the fold.
