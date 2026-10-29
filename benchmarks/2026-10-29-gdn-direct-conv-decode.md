# 2026-10-29 — PR #249: M-RoPE powf hoist (taken) and a 1-token GDN direct conv (declined)

Assessment of [PR #249](https://github.com/stew675/llama-cpp-rdna-boosts/pull/249) (`@briansp2020`), two
patches cut against r171: `0001-rope-compute-the-M-RoPE-section-power-once` and
`0002-GDN-direct-conv-for-1-token-decode`.  Both applied to r173 unchanged.  Outcome: **0001 folded into
block 10 (r174)**, **0002 declined**.  Release record: `WORKLOG.md` 261029.005.

## Arms

One warm tree (`scripts/scratch-tree.sh`, so ccache and the `GGML_CUDA_FA_QUANTS` cache string are
identical for every arm), built four states out of the same build directory:

| arm | what |
|---|---|
| `stock` | r173 release tree |
| `+0001` | r173 + the rope hoist |
| `+0001+0002` | r173 + both patches |
| `VERIFY=0` | the `+0001+0002` build with `GGML_CUDA_FUSE_GDN_CONV_VERIFY=0`, i.e. the stock decode path back |

Boxes: gfx1201 (3× R9700, 1-GPU arms at `HIP_VISIBLE_DEVICES=0`, ROCm 10.1.0-gfx120X) and gfx1100
(1× RX 7900 XTX, ROCm 10.1.0-gfx1100, `-DAMDGPU_TARGETS=gfx1100`, the submitter's own arch/flags).
`llama-bench -fa 1 -t 8`, alternating rounds (the first arm of a cold page cache is dropped).

## Throughput

gfx1100, tg128 (three rounds, `-r 5`):

| model | stock | +0001 | +0001+0002 | VERIFY=0 |
|---|---|---|---|---|
| Qwen3.6-35B-A3B Q4_K_M | 137.38 / 137.80 | 138.52 / 138.62 | 138.65 / 138.70 | 138.87 / 138.68 |
| Qwen3.8-27B Q4_K_M | 42.59 / 42.66 | 42.66 / 42.73 / 42.74 | 42.68 / 42.70 / 42.80 | 42.66 / 42.70 / 42.76 |
| Qwen3.5-9B Q8_0 (tg256) | 84.81 / 84.85 / 84.61 | 85.03 / 84.90 / 85.03 | 84.97 / 84.89 / 84.85 | 84.95 / 85.09 / 84.85 |

gfx1201, 1 GPU:

| model | stock | +0001 | +0001+0002 | VERIFY=0 |
|---|---|---|---|---|
| Qwen3.5-4B Q8_0, tg256 | 103.89 / 103.40 / 103.57 | 103.65 / 103.58 / 103.53 | **102.34 / 102.43 / 102.50** | 103.39 / 103.57 / 103.56 |
| Qwen3.6-35B-A3B Q4_K_M, tg128 | 101.83 | 101.73 | 101.71 | 101.34 / 101.78 |
| Qwen3.8-27B UD-Q4_K_XL, tg128 | 30.93 / 30.86 | 30.85 / 30.87 | 30.91 / 30.92 | 30.88 / 30.85 |

So: **0001 is +0.25 to +0.65 % on gfx1100 and inside the noise on gfx1201**; **0002 is 0 % on every arm
except the dense Q8_0 4B on gfx1201, where it is −1.10 %** and the `VERIFY=0` control restores stock
exactly.

## Kernel census (`rocprofv3 --kernel-trace`)

gfx1201, Qwen3.5-4B Q8_0, `-n 32 -r 2` (per-launch times):

| kernel | stock | +0001+0002 |
|---|---|---|
| `rms_norm_mul_rope_f32<256,false,__half,…>` | 520 @ 3.20 µs | 520 @ **2.85 µs** |
| `rms_norm_mul_rope_f32<256,false,float,…>` | 520 @ 3.06 µs | 520 @ **2.70 µs** |
| `ssm_conv_f32<true,128,4>` | 1560 @ 1.43 µs | — |
| `gdn_conv_direct_kernel<true,128,false>` | — | 1560 @ 1.30 µs |
| `gdn_concat_tail` | — | 1560 @ **1.44 µs (new launch)** |
| `rms_norm_scale_pair_f32<256>` | 1560 @ 1.50 µs | 1560 (not folded) |
| total kernel rows | 32 864 | 34 424 (**+1560**) |

On 35B-A3B **MoE** the conv's q/k norm fold does fire (`gdn_conv_direct_kernel<true,128,true>`, and
`rms_norm_scale_pair_f32` drops to prefill-only), so there the pair replaces two launches with two —
hence "same launch count" — and the arm measures neutral.  On the dense Q8_0 models
`gdn_conv_norm_match()` refuses (its allocator-overlap guard), so the `ssm_conv_in` epilogue's free CONCAT
write is replaced by a separate `gdn_concat_tail` launch and the fused conv is not faster than
`ssm_conv_f32`: that is the whole −1.1 %.

`ssm_conv_l2_gatebeta` (the decode-only `ssm_prescan` 15-node fusion) appears in **no** trace on either
box, and `GGML_CUDA_DISABLE_SSM_PRESCAN=1` changes nothing on stock — so the submitter's note that the
direct conv preempts prescan is latent here, not observed.

## Purity

* Greedy text, same seed / temp 0, 128 tokens: byte-identical for every arm pair on 4B Q8_0, 27B
  UD-Q4_K_XL and 35B-A3B Q4_K_M (gfx1201) and 9B Q8_0 (gfx1100).
* `scripts/gate-prefill-logits.sh --ab "GGML_CUDA_FUSE_GDN_CONV_VERIFY=0" --ub 1` (the decode-width
  probe, 27B Q8_0, gfx1201): **KLD 0.000000 / same-top-p 100.000 %** — the whole T=1 conv swap is
  bit-identical.
* `scripts/gate-prefill-logits.sh` at the default ubatch against the r173-recorded base: 0.000000 /
  100.000 %.
* `test-backend-ops -o ROPE` 450/450.
