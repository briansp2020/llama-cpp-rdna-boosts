# kv-rot-fwht-fusion: fold the KV-rotation FWHT into the fused rope and into the output projection's Q8_1 input

Thank you for the `fp contract(off)` fix in `rope.cu` (issue #67) and for the Q8_1 quantize cache; these two patches build
on both.  With a quantized KV cache, the attention rotation (`llama_mul_mat_hadamard`, run as `fwht_cuda`) adds four
short launches per layer on decode.  These patches remove two of them and the output projection's `quantize_q8_1`
launch, bit-identically.

Two format-patches on top of `v12-097cf33d9-r71` (tree `dc2cc150`, applied tree `b52c21d2`), for `git am`.
5 files (`rope.cu`, `rope.cuh`, `fwht.cu`, `fwht.cuh`, `ggml-cuda-try-fuse.cu`), +284 / -52.  Not arch-specific; each
fusion has a kill switch (`GGML_CUDA_DISABLE_ROPE_FWHT`, `GGML_CUDA_DISABLE_FWHT_Q8_1`).

**Hardware: RX 7900 XTX (gfx1100, RDNA3, 24 GB)**, Ryzen 7 9800X3D, Windows 11 + WSL2 (Ubuntu 24.04), ROCm 10.2 nightly
`20261013-38325106112` (HIP 7.17.26405) in a container.  All speed numbers below are from this box under WSL2, except the RDNA4 section (labelled there).

## 0001: the Q/K rotation inside rms_norm_mul_rope

Q and K each run `RMS_NORM, MUL, ROPE, RESHAPE, MUL_MAT(Hadamard)`: the fused rope kernel, then `fwht_cuda`.  A new
`fwht_n` instantiation of `rms_norm_mul_rope_f32` (one head per block, `ncols == n` for n = 64 / 128 / 256) stages
the roped row in shared memory and runs the FWHT in warp 0, then writes only the rotated row.

- Same arithmetic: `fwht_cuda`'s load-scale-and-butterfly moved into a helper in `fwht.cuh` (`ggml_cuda_fwht_warp`),
  used by both kernels.  `fwht.cuh` is included before `rope.cu`'s pragma, so the helper compiles under the same
  contraction mode in both files.  ISA check on gfx1100: `fwht_cuda<…, float>` has no FMA, and the fused rope
  kernel has the same 59 FMAs as the unfused one (the transform adds only multiplies and adds).
- Matcher (`ggml_cuda_should_fuse_rms_norm_mul_rope_fwht`): `ggml_can_fuse` for the norm / mul / rope, the RESHAPE
  checked by its own use count (it's a view, which `ggml_node_has_n_uses` always rejects), the Hadamard identified by
  `ggml_cuda_op_mul_mat_use_fwht`.
- One memory case differs from `ggml_cuda_check_fusion_memory_ranges`: the allocator usually hands the dead Q / K
  projection to the rotation output, so dst aliases the norm input x.  That alias is allowed only when it is exact
  (same pointer, both contiguous, same layout): each block reads its whole row of x before a barrier and only then
  writes the same row of dst.  Any other overlap rejects the fusion.

## 0002: the attention-output FWHT also writes the projection's Q8_1 input

The attention output goes `FLASH_ATTN_EXT, RESHAPE, MUL_MAT(Hadamard), RESHAPE, MUL_MAT(output projection)`; on the
mmvq path the projection first runs `quantize_q8_1` over the FWHT output.  When the projection takes mmvq and its row
length is a multiple of `MATRIX_ROW_PADDING` (no padded blocks), `fwht_cuda<…, true>` also writes those Q8_1 blocks
into the quantize cache under the key the mmvq launcher uses (view root of src1, src1's shape), with
`quantize_q8_1`'s reduction and rounding.  The projection finds them and skips its quantize launch.  The F32 output is
still written.  ISA: the extra FMAs in `fwht_cuda<…, true>` are 10 per element, the same as `quantize_q8_1`'s
division sequence.

The V rotation (after the V projection) is left as is.

## Tests

Qwen3-30B-A3B Q4_K_M (MoE, 48 layers) and Qwen3.8-27B Q4_K_M (dense), your README build (`-DGPU_TARGETS=gfx1100`), no
tuning env, `-ngl 99 -fa 1 -ctk q8_0 -ctv q8_0` (rotation on, as by default for a quantized cache).  Plain r71
against r71 + 0001..0002.

**Kernel time** (rocprofv3, one 400-token greedy MoE request per build), per call and launches per layer:

| | r71 | + 0001..0002 |
|---|---|---|
| rms_norm_mul_rope, Q and K | 6.9 us x 2 | 7.4 us x 2 (with the FWHT) |
| fwht_cuda, Q and K rotation | 3.3 us x 2 | gone |
| fwht_cuda, attention-output rotation | 3.1 us | 3.9 us (also writes Q8_1) |
| quantize_q8_1 for the output projection | 3.4 us | gone |
| fwht_cuda, V rotation (unchanged) | 3.1 us | 2.9 us |
| total GPU time per request | 4111 ms | 3766 ms (-8.4 %) |

The total falls by more than the removed kernel time, so part of it is run-to-run variation; the launch counts and
the decode numbers below are the firmer evidence.

**Decode** (`llama-bench -p 0 -n 256 -r 10`, two rounds, alternated): MoE

| | round 1 | round 2 | mean |
|---|---|---|---|
| r71 | 131.38 | 129.44 | 130.4 t/s |
| r71 + 0001..0002, both kill switches set | 131.36 | 131.74 | 131.6 t/s |
| r71 + 0001..0002 | 136.97 | 135.57 | 136.3 t/s (+4.5 % vs r71, +3.6 % vs kill switches) |

Dense 38.39 -> 38.60 t/s (+0.5 %, one round, within noise).

**Correctness.**
- Greedy hashes, 400 tokens, identical to r71: MoE plain decode prose `0116af00d69e` / code `8fd18d2f0b83`; dense
  with MTP (draft-mtp, n-max 3, so multi-token verify batches go through 0001 too) prose `8345c992d1fa` / code
  `47114faa2042`.
- Your `test-logits-width-probe` (q8_0 KV, FA): PASS with worst maxdiff 0 at prefix 256 and 3584, on both models.
- `test-backend-ops -o ROPE` and `-o MUL_MAT`: pass (they don't build the fused pattern; the hashes and the probe do).

## RDNA4

**Hardware: 1 x Radeon AI PRO R9700 (gfx1201)**, native Linux, ROCm 10.0; Qwen3-30B-A3B Q4_K_M and Qwen3.8-27B Q4_K_M,
llama-server `-c 65536 -ngl 99 -fa on -ctk q8_0 -ctv q8_0 -ub 2048 -b 2048`, plain r71 against r71 + 0001..0002.

- Greedy hashes identical to plain r71 on both models (MoE prose `744db56ea1c8` / code `6bb5966d061f`, dense prose
  `d12cdeede326` / code `e8a93c816237`); width probe PASS (maxdiff 0) at 256 and 3584; `test-backend-ops` ROPE
  450/450, MUL_MAT 2444/2444.
- Trace (MoE decode): `fwht_cuda<128>` is gone; `rms_norm_mul_rope_f32` with the FWHT takes 2.02 us per call (1.80 us
  without); `quantize_q8_1` calls halve (153.5k -> 76.8k); the attention-output `fwht_cuda<64, float, true>` takes 1.34
  us (1.10 us for the plain V-rotation one).
- MoE decode (two alternated rounds): r71 133.1 / 133.0, + 0001..0002 138.8 / 138.8 (+4.3 %), same binary with both
  kill switches 132.7 / 132.6 t/s; dense 31.2 t/s in every arm; 31.5k prefill unchanged.

## Limitations

- Measured on gfx1100 (WSL2) and gfx1201 (native Linux); the code is generic and was not run elsewhere.
- 0001 needs one rotation block per head (`n == head dim`, 64 / 128 / 256); 0002 needs an unpadded projection row
  and the mmvq path.  Otherwise the existing launches run.
- If the projection ends up off mmvq at run time, the pre-filled Q8_1 entry is unused (correct output, one wasted
  quantize), as with the existing norm and GLU pre-fills.

No urgency on our side.  Happy to re-run anything on your config.
