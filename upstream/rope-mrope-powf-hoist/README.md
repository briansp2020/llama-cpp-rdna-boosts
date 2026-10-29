# rope-mrope-powf-hoist — CUDA: compute the M-RoPE section power once, not once per section arm

`UPSTREAM-PR-rope-mrope-powf-hoist.patch` is cut against plain `ggml-org/llama.cpp` **`6050b467c`** (the
delivery base; applied tree `1753a0c2aa70a6be9f3f4803e73d7e9d9f3949e2`), sha256
`07f11aee22446a7174b03a2824cba58fb573fe53da18a9636d55c90e6e451fb8`.  It touches
`ggml/src/ggml-cuda/rope.cu` (+21 / −16).  It is the `rope_multi` half of the delivery's block-10 fold
(contributor PR #249, `@briansp2020`, landed as r174) **without** the second half, which edits the fused
`rms_norm_mul_rope_f32` — a kernel the delivery adds the M-RoPE arm to, so that hunk has no upstream
counterpart.

**What.**  `rope_multi` selects one of four M-RoPE sections (or the non-IMROPE four arms) per lane and
multiplied the picked position index by `powf(theta_scale, iw/2)` **inside each arm**.  The power is the
same in every arm — only the position index differs — and adjacent lanes take different arms
(`sector % 3`), so the divergent arms were serialized within a wave and `powf` executed once per arm taken.
It is hoisted once before the branch: same input, same value, so the result is bit-identical, and the
"no arm matched, `theta_base` stays 0" case is untouched (the mrope == 0 path of the fused kernel keeps its
own single `powf`, unchanged).

**Why it matters.**  `powf` is not cheap on RDNA and the kernels that carry this math — `rope_multi` and the
fused norm+rope — sit in every attention layer of every Qwen3.x model (they are IMROPE even for text, where
the position triplet is degenerate).  Traced per launch on gfx1201 the fused kernel goes **3.20 -> 2.85 µs**
(`__half` destination, i.e. the K cache write) and **3.06 -> 2.70 µs** (f32, the Q path), −11 %/−12 %.

**Measured** (`llama-bench -fa 1`, one GPU, three alternating rounds; greedy text byte-identical on every
model, `llama-perplexity --kl-divergence` KLD 0.000000 / 100.000 % same-top-p, `test-backend-ops -o ROPE`
450/450):

| box / model | tg before | tg after |
|---|---|---|
| gfx1100 7900 XTX, Qwen3.6-35B-A3B Q4_K_M, tg128 | 137.38 / 137.80 | 138.52 / 138.62 (**+0.65 %**) |
| gfx1100, Qwen3.8-27B Q4_K_M, tg128 | 42.59 / 42.66 | 42.66 / 42.73 (+0.25 %) |
| gfx1100, Qwen3.5-9B Q8_0, tg256 | 84.81 / 84.85 / 84.61 | 85.03 / 84.90 / 85.03 (+0.25 %) |
| gfx1201 R9700, Qwen3.5-4B Q8_0, tg256 | 103.89 / 103.40 / 103.57 | 103.65 / 103.58 / 103.53 (noise) |

The wall-clock win is bounded by how many such launches a token has (8 in a 4B), so it is small but never
negative; the kernel-level win is the honest number and reproduces on both RDNA4 and RDNA3.0.

**Not validated** on NVIDIA/CUDA hardware (the change is backend-agnostic CUDA C++ and the same kernel is
compiled for both); reviewers there may want a trace, but nothing about the transform is AMD-specific.
