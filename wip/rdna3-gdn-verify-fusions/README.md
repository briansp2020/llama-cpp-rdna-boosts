# rdna3-gdn-verify-fusions: the GDN conv verify path and the batched f32 CPY on RDNA3_0 too

Thank you for carrying the GDN conv verify path and the batched CPY in the release behind their own switches
(`GGML_CUDA_FUSE_GDN_CONV_VERIFY`, `GGML_CUDA_FUSE_CPY_BATCH`), which this builds on.  This opens both,
RDNA4-only so far, to RDNA3_0 (gfx1100-1102).  Bit-identical.

One format-patch on top of `v12-097cf33d9-r91` (tree `90507dfb`, applied tree `37a5cef8`), for `git am`.  3 files
(`gdn-conv.cu`, `cpy-batch.cu`, and the comment in `cpy-batch.cuh`), +6 / -4.  Independent of #192.

**Hardware: RX 7900 XTX (gfx1100, RDNA3, 24 GB)**, Ryzen 7 9800X3D, Windows 11 + WSL2 (Ubuntu 24.04), ROCm 10.2 nightly
`20261013-38325106112` (HIP 7.17.26405) in a container.  All numbers below are from this box under WSL2.

## What it changes

- `gdn-conv.cu`: the direct GDN conv (in place of `CONCAT + SSM_CONV + SILU`, same 4-tap reduction order and silu,
  `gdn_concat_tail` still materializing the snapshot tail) takes the 2..255-token batches of the speculative verify
  step only when `GGML_CUDA_CC_IS_RDNA4(cc)`; on other GPUs only T >= 256 (prefill) takes it.
- `cpy-batch.cu`: the run of up to 16 consecutive same-layout f32 `GGML_OP_CPY` nodes in one launch returns early
  unless `GGML_CUDA_CC_IS_RDNA4(cc)`.

Neither kernel nor either matcher depends on anything RDNA4-specific, so the patch admits
`GGML_CUDA_CC_IS_RDNA3_0(cc)` in both.  RDNA3_5 stays out because we have not measured it, RDNA4 is unaffected, the
single-device restriction on the conv fusion is kept, and both switches still turn the fusions off.

## Tests

Qwen3.8-27B Q4_K_M (`qwen35`, GDN hybrid, MTP) and Qwen3-30B-A3B Q4_K_M (MoE, no GDN, as a control), your README
build (`-DGPU_TARGETS=gfx1100`), no tuning env, `-ngl 99 -fa 1 -ctk q8_0 -ctv q8_0`.  Plain r91 against r91 + 0001.

**Kernel trace** (rocprofv3, one 400-token greedy request with `draft-mtp`, n-max 3, per build): `cpy_scalar`
33515 -> 3179 launches with 7680 batched-copy launches in their place, `ssm_conv_f32` 7680 -> 0 with 7680
`gdn_conv_direct_kernel` in its place, and the generic CONCAT before it replaced by 7680 tail-only `gdn_concat_tail`
launches.  Total GPU time 5356 -> 5251 ms (-2.0 %); one trace per build, the r91 one from an earlier run of the same
binary.

**Speed**, dense MTP decode (`draft-mtp` n-max 3, 400 tokens), r91 against r91 + 0001, six alternated rounds with
the order flipped every round (first run of each pair):

| round | prose r91 -> patched (t/s) | code r91 -> patched (t/s) |
| --- | --- | --- |
| 1 | 80.65 -> 81.45 | 95.85 -> 96.98 |
| 2 | 80.20 -> 81.42 | 95.43 -> 96.85 |
| 3 | 80.28 -> 78.89 | 95.66 -> 93.61 |
| 4 | 78.05 -> 79.24 | 92.84 -> 94.17 |
| 5 | 78.18 -> 78.99 | 92.56 -> 94.24 |
| 6 | 77.88 -> 79.05 | 92.85 -> 93.92 |

The patched build is ahead in five of the six rounds, by +1.0 to +1.8 %; the median over all twelve runs, round 3
included, is +1.3 %.  Round 3 is the exception (-1.7 / -2.1 %): the box appears to have stepped down by about 3 % around
it (r91 80.28 -> 78.05 t/s by round 4, and both builds stay at the lower level from then on), so we read it as drift
rather than as the patch, but it is in the table.  An earlier four-round A/B with each
switch off in turn was too noisy on this box to separate the two parts (patched against r91 ranged from about -2.6 % to
about +4 % across its four rounds, and r91 itself drifted by about 4 %), so we don't claim a split between the conv
and the copy batch.

**Correctness.**
- Greedy hashes, 400 tokens, identical to r91: dense with MTP and plain decode prose `8345c992d1fa` / code
  `47114faa2042` (MTP acceptance unchanged, 0.672 / 0.857); MoE prose `bf70e0f571ea` / code `6b19db0bf7d8`.
- Your `test-logits-width-probe` (q8_0 KV, FA): PASS with worst maxdiff 0 at prefix 256 and 3584 on both models.
- `test-backend-ops -o CPY` 251/251, `-o CONCAT` 183/183, `-o SSM_CONV` 45/45.

## Limitations

- Measured on gfx1100 (WSL2), single GPU, `draft-mtp` only.  gfx1101 / gfx1102 are included by the RDNA3_0 macro
  but not tested here.
- The gain is small and this box drifts, hence the six rounds; happy to run more if a firmer number would help.

No urgency on our side.  Happy to re-run anything on your config.
