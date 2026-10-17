# rope-mrope-fusion: the fused rms_norm_mul_rope kernel for MROPE / IMROPE (bit-identical, r80)

Thank you for folding #172 and #174 into r78, and for the trace work on #173 that pointed at the MROPE gap
(`TODO.md`, "Rejected PR #173 re-entry, and the MROPE gap in PR #172").  This takes that suggestion: the fused
`RMS_NORM + MUL + ROPE` kernel, and with it the rope + KV-rotation FWHT fusion, now also runs for
`GGML_ROPE_TYPE_MROPE` / `IMROPE`, so it fires on the Qwen3.5 / 3.6 / 3.8 family.  Bit-identical.

Two format-patches on top of `v12-097cf33d9-r80` (tree `873d0335`, applied tree `379c6629`), for `git am`.
3 files (`rope.cu`, `ggml-cuda-try-fuse.cu`, `test-backend-ops.cpp`), +51 / -14.

**Hardware: RX 7900 XTX (gfx1100, RDNA3, 24 GB)**, Ryzen 7 9800X3D, Windows 11 + WSL2 (Ubuntu 24.04), ROCm 10.2 nightly
`20261013-38325106112` (HIP 7.17.26405) in a container.  All speed numbers below are from this box under WSL2.

## What it changes

**0001, a guard (applies on its own).**  `ggml_cuda_should_fuse_rope_set_rows` admits a BF16 `SET_ROWS`
destination, which rope's own set_rows fusion writes, but the 5-op `RMS_NORM, MUL, ROPE, VIEW, SET_ROWS` fusion
shares that check and `ggml_cuda_op_rms_norm_mul_rope_fused` writes F32/F16 only, so it reaches its
`GGML_ABORT`.  On plain r80 that happens today for a NEOX model with q/k norm whose K cache write follows the K
rope: Qwen3-30B-A3B with `-ctk bf16 -ctv bf16` stops at `rope.cu:1050: fatal error` on the first decode.  With
0002 alone the same would happen on the Qwen3.x models (IMROPE, and the `attn_kv` builder expands K last), which
is how this turned up.  0001 leaves BF16 rows to the existing rope + set_rows path; `test-backend-ops` gains two
BF16 `RMS_NORM_MUL_ROPE` cases (they abort without the guard).

**0002, MROPE / IMROPE.**  The fused kernel learns `rope_multi`'s position lookup: each rotated pair picks its
position row (t / h / w / e) from the mrope sections with the same tests as `rope_multi`, and pairs are
neox-style as in `rope_multi`.  The rest (norm, mul, `rope_yarn`, the pass-through channels of a partial
rotary, the in-block FWHT) is the existing code, under the same `fp contract(off)`.  Vision rope and
`n_offs != 0` stay unfused.  `GGML_CUDA_DISABLE_RMS_NORM_MUL_MROPE=1` turns the new modes off (the old modes are
unaffected).  `test-backend-ops` `RMS_NORM_MUL_ROPE` adds MROPE next to the IMROPE cases it already had; both
now go through the fused kernel.

On Qwen3.8-27B (IMROPE, head 256, 64 rotary dims, sections `[11, 11, 10, 0]`, full attention every 4th layer),
each attention layer's Q and K paths go from three launches (`rms_norm_f32<256, true>` with the mul,
`rope_multi`, `fwht_cuda<256>`) to one `rms_norm_mul_rope_f32<256, false, float, 256>`.

## Tests

Qwen3.8-27B Q4_K_M (dense, `qwen35`) and Qwen3-30B-A3B Q4_K_M (MoE, NEOX, as a control), your README build
(`-DGPU_TARGETS=gfx1100`), no tuning env, `-ngl 99 -fa 1 -ctk q8_0 -ctv q8_0`.  Plain r80 against r80 + 0001 + 0002.

**Kernel trace** (rocprofv3, one 400-token greedy dense request per build): `rope_multi` 14976 -> 0,
`fwht_cuda<256>` 14976 -> 0, `rms_norm_f32<256, true>` 37440 -> 22464, and 14976
`rms_norm_mul_rope_f32<256, false, float, 256>` in their place.

**Speed** (same binary, `GGML_CUDA_DISABLE_RMS_NORM_MUL_MROPE=1` against the default, alternated rounds):
- Dense `llama-bench -p 0 -n 256 -r 5`, two rounds: 38.92 / 38.89 -> 39.29 / 39.09 t/s (+0.7 %); plain r80
  38.76 / 38.86.
- Dense MTP (`draft-mtp`, n-max 3, 400 tokens), two rounds: prose 75.16 / 74.79 -> 76.18 / 75.58 t/s, code
  88.62 / 86.82 -> 89.74 / 88.55 t/s (about +1 %), acceptance unchanged (0.672 / 0.846).
- Dense pp512, two rounds: 1291.5 / 1296.1 -> 1298.9 / 1297.1 t/s (flat).

**Correctness.**
- Greedy hashes, 400 tokens, identical to plain r80: dense with MTP (draft-mtp, n-max 3) and plain decode prose
  `8345c992d1fa` / code `47114faa2042`; MoE prose `bf70e0f571ea` / code `6b19db0bf7d8`.
- Your `test-logits-width-probe` (q8_0 KV, FA): PASS with worst maxdiff 0 at prefix 256 and 3584, on both models.
- `test-backend-ops`: `RMS_NORM_MUL_ROPE` 298/298, `ROPE` 450/450, `MUL_MAT_HADAMARD` 32/32, `RMS_NORM` 53/53.
- BF16 KV (`-ctk bf16 -ctv bf16`, tg64): Qwen3-30B-A3B aborts on plain r80 and runs with the patches;
  Qwen3.8-27B runs on both.

## Limitations

- Measured on gfx1100 (WSL2) only so far, and on one GPU; we have no 2- or 3-GPU setup or prefill-logit base.
- We don't have Qwen3.6-35B-A3B here, so we haven't checked how often its graph takes the fused path.
- Not rebased #173 on top yet.  The `attn_kv` builder used by `qwen35` already expands Q, V, K, so the K cache
  write is adjacent there; happy to follow up once this one is settled.

No urgency on our side.  Happy to re-run anything on your config.
