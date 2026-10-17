# add-rms-norm-q8-decode: the ADD + rms_norm_q8_1 fusion for one token too

Thank you for the verify-band ADD + `rms_norm_q8_1` fusion (`ggml_cuda_add_rms_q8_1_try_fuse`); this one-line patch
lets it run for single-token decode as well.  Bit-identical.

One format-patch on top of `v12-097cf33d9-r71` (tree `dc2cc150`, applied tree `73ca3e25`), for `git am`.  1 file
(`norm.cu`), +1 / -1.  Independent of #171 / #172 / #173.  `GGML_CUDA_FUSE_ADD_RMS_Q8=0` still turns the fusion off.

**Hardware: RX 7900 XTX (gfx1100, RDNA3, 24 GB)**, Ryzen 7 9800X3D, Windows 11 + WSL2 (Ubuntu 24.04), ROCm 10.2 nightly
`20261013-38325106112` (HIP 7.17.26405) in a container.  All speed numbers below are from this box under WSL2, except the RDNA4 section (labelled there).

## What it changes

`ggml_cuda_add_rms_q8_1_try_fuse` required `node->ne[1] > 1`, so one-token decode was excluded.  For one token the
attention residual ADD is already folded into the output projection's mmvq epilogue, so it never reaches this
matcher, but the FFN residual ADD after an MoE block is not: it follows `moe_weighted_reduction` and ran as its own
`k_bin_bcast` launch in every layer, ahead of the next layer's `rms_norm_q8_1`.  The patch drops the `ne[1] <= 1`
exclusion; the rest of the matcher and the kernel (`rms_norm_q8_1_f32<…, true>`, the values of the unfused
`k_bin_bcast` + `rms_norm_q8_1` pair) are unchanged.

## Tests

Qwen3-30B-A3B Q4_K_M (MoE, 48 layers) and Qwen3.8-27B Q4_K_M (dense), your README build (`-DGPU_TARGETS=gfx1100`), no
tuning env, `-ngl 99 -fa 1 -ctk q8_0 -ctv q8_0`.  Plain r71 against r71 + 0001.

**Kernel trace** (rocprofv3, one 400-token greedy MoE request per build): the per-layer `k_bin_bcast` add (3.9 us per
call, 22.6k calls) is gone; the ADD-carrying `rms_norm_q8_1_f32<1024, true>` takes 5.9 us per call, about the same as
the plain one (5.7 us here, 6.1 us in the r71 trace).

**Decode** (`llama-bench -p 0 -n 256 -r 10`, MoE, two alternated rounds): r71 129.90 / 129.90, r71 + 0001 with
`GGML_CUDA_FUSE_ADD_RMS_Q8=0` 132.06 / 130.32, r71 + 0001 132.60 / 133.48 t/s.  About +1.4 % against the switch and
+2.4 % against r71; the switch arm itself sat above r71 in round 1, so the size is within the noise on this box and
the trace is the firmer evidence.  Dense 38.24 -> 38.70 t/s (one round, within noise).

**Correctness.**
- Greedy hashes, 400 tokens, identical to r71: MoE plain decode prose `0116af00d69e` / code `8fd18d2f0b83`; dense with
  MTP (draft-mtp, n-max 3) prose `8345c992d1fa` / code `47114faa2042`; dense plain decode `8345c992d1fa`.
- Your `test-logits-width-probe` (q8_0 KV, FA): PASS with worst maxdiff 0 at prefix 256 and 3584, on both models.
- `test-backend-ops -o RMS_NORM` and `-o ADD`: pass.

## RDNA4

**Hardware: 1 x Radeon AI PRO R9700 (gfx1201)**, native Linux, ROCm 10.0; same models, llama-server
`-c 65536 -ngl 99 -fa on -ctk q8_0 -ctv q8_0`, plain r71 against r71 + 0001 (applied tree `73ca3e25`).

- Greedy hashes identical to plain r71 on both models (MoE prose `744db56ea1c8` / code `6bb5966d061f`, dense prose
  `d12cdeede326` / code `e8a93c816237`); width probe PASS (maxdiff 0) at 256 and 3584; `test-backend-ops` RMS_NORM
  53/53, ADD 154/154.
- Trace (MoE decode): the per-layer `k_bin_bcast` add (1.50 us per call) is gone; the ADD-carrying
  `rms_norm_q8_1_f32<1024, true>` takes 2.84 us against 2.73-2.76 us for the plain one.
- MoE decode (two alternated rounds, every request within 0.2 t/s): r71 132.6, r71 + 0001 with
  `GGML_CUDA_FUSE_ADD_RMS_Q8=0` 132.7, r71 + 0001 134.9 t/s (+1.7 %).  31.5k prefill flat.

## Limitations

- Measured on gfx1100 (WSL2) and gfx1201 (native Linux).
- Only graphs where the ADD feeds a norm whose product goes to an mmvq matmul are affected (the existing matcher).

No urgency on our side.  Happy to re-run anything on your config.
