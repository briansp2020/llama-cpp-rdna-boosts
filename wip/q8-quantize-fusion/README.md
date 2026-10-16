# q8-quantize-fusion: fewer `quantize_q8_1` launches in `-sm tensor` decode

One format-patch on top of `v12-097cf33d9-r68` (tree `ffe83269`; applied tree `d4123914`), for `git am`.  It touches
`ggml/src/ggml-cuda/{common.cuh,ggml-cuda.cu,hc-mix.cu}` and `src/models/qwen4exp.cpp` (+155 / -19).

## What we saw

Qwen3.8-Flash-Next, `-sm tensor`, decode with the MTP draft: each GPU runs about 225 `quantize_q8_1` launches per
verify step on r68.  A debug build (on r62) that logged the src1 producer of every quantize the mmvq path does itself
showed that most of them quantize an HC mixer output, and several quantize the same activation twice:

- **HC mixer output -> shared expert `ffn_gate`, again.**  The shared expert is mirrored (not split) while the MoE
  output is a partial sum, so the meta backend ends the segment at the MoE AllReduce.  The shared expert came after
  that boundary, in a new `graph_compute`, and the Q8_1 cache is cleared per segment.  So the activation the MoE
  gate/up had just quantized was quantized again.
- **The cache key compares raw shapes.**  The MUL_MAT_ID view `[K, 1, n]` and the MUL_MAT view `[K, n]` of one
  activation are the same bytes, but never matched.
- **The HC mixer output** feeds mmvq matmuls (the attention / GDN input projections, the MoE gate/up), each of which
  quantizes it in a separate launch.

## The patch

Three parts, each with its own switch for A/B; all three are bit-identical:

1. `q8_1_cache_get` matches on the canonical row layout (size-1 dims dropped, a dim merged into the previous one
   when it continues it contiguously).  The quantized buffer holds the rows in `(i13, i12, i11)` order, so two
   descriptions with the same canonical rows are byte-identical.  `GGML_CUDA_Q8_1_CACHE_EXACT=1`: the old exact-shape
   match.
2. qwen4exp builds the shared expert before the MoE and expands it first, so it sits in the same meta-backend
   segment as the MoE gate/up and shares their Q8_1.  Node order only.  `LLAMA_SHEXP_FIRST=0`: the old order.
   One side effect we could not measure: `ggml_cuda_shexp_down_gate_try_fuse` needs the shared expert's down matmul
   and its gate chain as six consecutive nodes, which the new order splits.  That fusion needs a Q8_0 `ffn_down_shexp`
   and an F32 `ffn_gate_inp_shexp`; our model has IQ4_NL / BF16, so it fires in neither build here.  On a model where
   it does fire, `LLAMA_SHEXP_FIRST=0` keeps it; we can also make the reorder skip that case if you prefer.
3. `hc_mix_bf16_up_collapse2` also writes the Q8_1 of its output into the quantize cache, which the mmvq consumers
   then find.  With the epilogue on, each block computes a contiguous run of rows (instead of interleaved rows) so a
   32-row Q8_1 group spans few blocks; every row is computed the same way either way.  Each block adds its finished
   row count per group to a per-group counter (one atomic per block and group); the block that completes a group
   quantizes it, one warp per group, with the same 32-lane reductions and rounding as `quantize_q8_1`, and resets
   the counter for the next launch.  The counters are one `GGML_CUDA_HC_Q8_CNT` (1024)-int slice per stream,
   allocated and zeroed once.  It is used only when the up_v2 kernel runs, `n_embd` is a multiple of
   `MATRIX_ROW_PADDING` and `n_embd / 32 <= 1024`; otherwise nothing changes.  `GGML_HC_MIX_Q8=0`: the old path and row
   mapping.

An earlier version of (3) on its own (interleaved rows, one atomic per row) cost as much in the up kernel as it
saved, because the FFN side's second quantize never hit the cache; (1) and (2) are what make it pay.

## Tests (2 x R9700 32 GB, gfx1201)

Qwen3.8-Flash-Next GSQ-RCO IQ3_XXS, all experts in VRAM, + MTP draft (shared Q8_0, `--spec-draft-n-max 3`), no tuning
env.  Each run is a fresh server after `llama-warmup.py`, in the order base, patch, three rounds.

```
llama-server -m <IQ3_XXS> -c 262144 -ngl 99 -sm tensor --n-cpu-moe 0 -ctk q8_0 -ctv q8_0 -ub 2048 -b 2048 \
  -ot per_layer_token_embd.*=CPU -lm none -lzm off \
  --spec-type draft-mtp -md <mtp Q8_0> -ngld 99 --spec-draft-n-max 3 --spec-draft-p-min 0 -ctkd q8_0 -ctvd q8_0
```

| t/s (rounds 1 / 2 / 3) | r68 | r68 + patch |
|---|---|---|
| greedy 400 tokens, 1st request | 146.1 / 142.6 / 142.9 | 147.3 / 147.9 / 144.5 |
| greedy 400 tokens, 2nd request | 148.3 / 148.0 / 148.2 | 149.8 / 149.9 / 149.7 |
| seeded chat (temperature 0.7) | 131.0 / 130.3 / 130.6 | 132.5 / 131.5 / 132.1 |
| 4 concurrent requests, aggregate | 174.9 / 180.2 / 181.9 | 177.9 / 173.4 / 180.0 |

Single-request decode is about 1 % faster (2nd request 148.2 -> 149.8, seeded 130.6 -> 132.0), in every round; the
4-concurrent numbers are within their run-to-run spread.

Kernel trace (rocprofv3 `--kernel-trace`, same config, one GPU, the last 120 verify steps of 3 x 400-token greedy
requests):

| per verify step | r68 | r68 + patch |
|---|---|---|
| `quantize_q8_1` launches | 224.7 | 69.2 |
| `hc_mix_bf16_up_collapse2` avg | 15.0 us | 16.5 us |
| kernel busy | 17.03 ms | 17.08 ms |
| step | 25.57 ms | 25.06 ms |

GPU busy time is about the same (the epilogue costs about what the quantizes did); the gain is the ~155 launches
and the gaps between them.

Outputs are the same with and without the patch: greedy `a7a81f01b3bc` (draft acceptance 274/374 in every run), 12 short prompts `527dd2d14a60`, seeded `ff0607be0b97`, and in the traced runs greedy `196294b5c3b4`.  No faults or aborts.

The fusion is not architecture-specific (none of it is gated on RDNA4), but parts 2 and 3 only apply to qwen4exp;
part 1 applies wherever a MUL_MAT_ID and a MUL_MAT read the same activation in one segment.

No urgency on our side.  Happy to re-run anything on your config, or split it up if you'd rather take the parts
separately.
