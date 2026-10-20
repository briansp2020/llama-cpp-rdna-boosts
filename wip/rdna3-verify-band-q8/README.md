# rdna3-verify-band-q8: the verify-band Q8_1 fusions on RDNA3_0 too

Thank you for carrying the verify-band Q8_1 fusions in the release behind `GGML_CUDA_FUSE_Q8_1_VERIFY`, which this
builds on, and for `test-logits-width-probe`, which let us check this one at every verify width in a few minutes.
This patch opens the existing 2..8-token verify-band Q8_1 pre-quantize fusions, RDNA4-only so far, to RDNA3_0
(gfx1100-1102).  Bit-identical.

One format-patch on top of `v12-097cf33d9-r91` (tree `90507dfb`, applied tree `042644f2`), for `git am`.  3 files
(`mmvq.cu`, plus one comment each in `ggml-cuda.cu` and `ggml-cuda-try-fuse.cu`), +5 / -4.

**Hardware: RX 7900 XTX (gfx1100, RDNA3, 24 GB)**, Ryzen 7 9800X3D, Windows 11 + WSL2 (Ubuntu 24.04), ROCm 10.2 nightly
`20261013-38325106112` (HIP 7.17.26405) in a container.  All numbers below are from this box under WSL2.

## What it changes

`ggml_cuda_should_fuse_mul_mat_vec_q(tensor, verify_band = true)` admits a 2..8-row matmul only when
`GGML_CUDA_CC_IS_RDNA4(cc)` (`mmvq.cu`).  Its `verify_band` callers only pre-fill the mmvq Q8_1 activation cache and
leave the matmul itself as it is: the residual ADD + `rms_norm_q8_1` fold (`norm.cu`), `rms_norm` -> Q8_1, and the
gated unary -> Q8_1 marking (`ggml-cuda-try-fuse.cu`, `mmvq.cu`).  As the comment in `ggml-cuda.cu` says, their
kernels are row-generic and write the values `quantize_q8_1` would write, and none of those matchers has an arch
check of its own, so on gfx1100 the verify step ran them unfused only because of this one condition.  The patch
admits `GGML_CUDA_CC_IS_RDNA3_0(cc)` as well.  RDNA3_5 stays out because we have not measured it.
`GGML_CUDA_FUSE_Q8_1_VERIFY=0` still turns the band fusions off, and RDNA4 is unaffected.

## Tests

Qwen3.8-27B Q4_K_M (dense `qwen35`, MTP) and Qwen3-30B-A3B Q4_K_M (MoE, no MTP head, as a control), your README build
(`-DGPU_TARGETS=gfx1100`), no tuning env, `-ngl 99 -fa 1 -ctk q8_0 -ctv q8_0`.  Plain r91 against r91 + 0001.

**Kernel trace** (rocprofv3, one 400-token greedy request with `draft-mtp`, n-max 3, per build): standalone
`quantize_q8_1` 42772 -> 1694 launches, the residual `k_bin_bcast` add 20798 -> 575, `rms_norm_f32<1024, true>`
22815 -> 2276 with 20224 `rms_norm_q8_1_f32<1024, true>` in their place, and the gated SiLU now runs as
`unary_gated_q8_1_op_kernel`.  Total GPU time 5356 -> 5202 ms (-2.9 %).

**Speed**, dense MTP decode (`draft-mtp` n-max 3, 400 tokens, two runs per request):
- r91 against r91 + 0001, four alternated rounds, the patched build ahead in all four: prose 80.26 / 79.89 / 79.77 /
  79.60 -> 81.72 / 81.38 / 81.41 / 81.45 t/s, code 95.41 / 95.15 / 95.02 / 94.87 -> 97.17 / 96.58 / 96.87 / 96.88 t/s
  (first run of each pair; about +1.9 %).
- Same binary, `GGML_CUDA_FUSE_Q8_1_VERIFY=0` against the default, three alternated rounds: prose 79.90 / 79.42 /
  79.40 -> 81.30 / 81.17 / 80.94, code 95.08 / 94.41 / 94.43 -> 96.80 / 96.53 / 96.32 t/s (about +2.0 %).
- A first single-round check before the alternated rounds showed no difference (prose 81.05 -> 80.90, code
  96.40 -> 96.32 t/s); the alternated rounds above were run afterwards to settle it.
- Plain (one-token) decode does not take the verify path; one round each was within noise (dense prose
  40.18 -> 39.54, code 40.11 -> 39.88 t/s; MoE prose 158.97 -> 159.77, code 159.21 -> 157.77 t/s).

**Correctness.**
- Greedy hashes, 400 tokens, identical to r91: dense with MTP and plain decode prose `8345c992d1fa` / code
  `47114faa2042` (MTP acceptance unchanged, 0.672 / 0.857); MoE prose `bf70e0f571ea` / code `6b19db0bf7d8`.
- Your `test-logits-width-probe` (q8_0 KV, FA): PASS with worst maxdiff 0 at prefix 256 and 3584 on both models,
  for r91 and for r91 + 0001.
- `test-backend-ops -o RMS_NORM` 53/53, `-o ADD` 154/154, `-o MUL_MAT` 2444/2444.

## Limitations

- Measured on gfx1100 (WSL2), single GPU, `draft-mtp` only.  gfx1101 / gfx1102 are included by the RDNA3_0 macro
  but not tested here.
- Larger MTP draft depths or other verify-heavy modes should behave the same way (the band is 2..8 rows), but we
  only timed n-max 3.

No urgency on our side.  Happy to re-run anything on your config.
