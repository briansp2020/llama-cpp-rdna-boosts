# wip/mrope-rope-fusion

The fused `rms_norm_mul_rope_f32` kernel from #172 now also handles `GGML_ROPE_TYPE_MROPE` and `GGML_ROPE_TYPE_IMROPE`, the item recorded in `TODO.md` ("the MROPE gap in PR #172").
- On the Qwen3.x models, this puts the `RMS_NORM, MUL, ROPE (+ KV-rotation FWHT)` chain on the fused kernel.
- That chain is currently three launches per rope (`rms_norm_f32`, `rope_multi`, `fwht_cuda<256>`); with the patch it is one.

One format-patch on `v12-097cf33d9-r80` (tree `873d0335`); applied tree `c80bf818`.

## What changes

- **Matcher** (`ggml_cuda_should_fuse_rms_norm_mul_rope`):
  - Admits MROPE and IMROPE as well as NORMAL and NEOX.
  - Every other check is unchanged. That includes `n_offs == 0`, so the rotated offset `iw` equals `i0`.
  - VISION is still rejected.
  - `GGML_CUDA_DISABLE_ROPE_MROPE_FUSE=1` restores the previous matcher, so M-RoPE goes back to the unfused chain.
- **Kernel**:
  - M-RoPE and IMROPE rotate in NEOX order, so the pairing branch is taken for them too.
  - θ is computed with `rope_multi`'s own section selection and float expressions, copied branch for branch: the `sect_dims` / `sec_w` / `sector` arithmetic, the IMROPE `% 3` interleave, `pos[channel + nchannels*k]` and `powf(theta_scale, i0 / 2.0f)`.
  - `nchannels` is `rope_multi`'s `ne02` in this layout.
  - The NORMAL / NEOX path computes exactly what it did before.
- **Op entry**: reads `mrope_sections` from `op_params + 11`, as `ggml_cuda_op_rope_impl` does, and passes the mode kind down.
- **test-backend-ops**: `GGML_ROPE_TYPE_MROPE` is added to the `RMS_NORM_MUL_ROPE` mode list, next to the existing IMROPE cases (72 new cases).

## Gates

Setup:
- R9700 (gfx1201), native Linux, HIP runtime 7.15.26333.
- Stock r80 (`s80s`) against r80 + this patch (`s80p`), built from the same tree in the same build dir.
- Models: Qwen3.6-35B-A3B Q4_K_M and Qwen3.8-27B Q4_K_M.
- Server settings: `-c 65536 -ngl 99 -fa on -ctk q8_0 -ctv q8_0 -ub 2048 -b 2048`.

**Greedy hashes**
- Setup: 400 tokens, prose and code prompts.
- Result: identical across stock, patched and patched + kill switch, on both models. That holds on:
  - 1 GPU, with and without `GGML_CUDA_GRAPH_OPT=1`;
  - 2 GPU `-sm tensor`;
  - 2 GPU `-sm layer`.

| model | 1 GPU / 2 GPU layer | 2 GPU tensor |
|---|---|---|
| 35B-A3B | `0eedd580608a` / `365e3058e68f` | `c0cf0739bdab` / `930bcf1f4405` |
| 27B | `d12cdeede326` / `e8a93c816237` | `7cfb359d5c5d` / `dcd93c8965ce` |

**`test-logits-width-probe`** (1 GPU, q8_0 KV)
- PASS with worst maxdiff 0 at P=256 and P=3584, on both models and both builds.
- The W8 and row-0 hashes are identical between stock and patched:

| model | P | W8 hash | row-0 hash |
|---|---|---|---|
| 35B-A3B | 256 | `251c193d7b72cf34` | `22016c482fa2909e` |
| 35B-A3B | 3584 | `f172e67372ec2698` | `e6053b6b368d025f` |
| 27B | 256 | `498b73a62646b535` | `4edad63ea637c4e5` |
| 27B | 3584 | `aa42818af2333c50` | `4c9e9852c26c92d6` |

**Prefill-logit**
- Setup: `llama-perplexity --kl-divergence`, wikitext-2 test, 16 × 2048, 1 GPU, against a base recorded from stock r80.

| model | run | mean KLD | max KLD | same top p |
|---|---|---|---|---|
| 35B-A3B | patched | 0.000000 | 0.000058 | 99.994 % |
| 35B-A3B | stock vs its own base | 0.000000 | 0.000058 | 99.994 % |
| 27B | patched | 0.000000 | 0.000065 | 99.994 % |
| 27B | stock vs its own base | 0.000000 | 0.000065 | 99.994 % |

- Patched + kill switch gives the same numbers as stock vs its own base.
- So on this box the non-zero max and the 99.994 % are the floor of the stored base, and the patch sits exactly on it.

**`test-backend-ops`**
- `-o RMS_NORM_MUL_ROPE` 296/296, including the 72 new MROPE and the 72 existing IMROPE cases.
- `-o ROPE` 450/450.
- Both pass on stock and patched libraries.

**Faults**: 0 memory-access faults or asserts in any run.

## Launch counts

Kernel trace with `rocprofv3 --kernel-trace`: 35B-A3B, 1 GPU decode, the same 3 × 400-token greedy work on both builds.

| kernel | stock r80 | r80 + patch |
|---|---|---|
| `rope_multi<true, false, float, float>` | 32,120 × 2.05 µs | — |
| `fwht_cuda<256, float, false>` | 32,120 × 1.32 µs | — |
| `rms_norm_f32<256, true, false, false>` | 80,300 × 1.67 µs | 48,180 × 1.71 µs |
| `rms_norm_mul_rope_f32<256, false, float, 256>` | — | 32,120 × 3.15 µs |

- 64,240 fewer launches.
- The three kernels took about 162 ms of GPU time; the fused kernel takes 101 ms.
- The other `rms_norm*`, `quantize_q8_1`, `set_rows` and `fwht_cuda<64>` counts are unchanged.

## Throughput

Two alternated rounds per arm. Decode is the mean of 4 × 400-token greedy runs per round. Prefill is one 31.5k-token prompt.

| config | stock r80 | r80 + patch | change |
|---|---|---|---|
| 35B-A3B, 1 GPU, decode | 123.8 | 125.3 | +1.2 % |
| 35B-A3B, 1 GPU, decode, patch + kill switch | | 123.9 | same as stock |
| 35B-A3B, 2 GPU `-sm tensor`, decode | 130.6 | 132.2 | +1.2 % |
| 35B-A3B, 2 GPU `-sm layer`, decode | 113.7 | 115.0 | +1.2 % |
| 27B, 1 GPU, decode | 31.2 | 31.4 | +0.6 % |
| 35B-A3B, 2 GPU `-sm tensor`, prefill 31.5k | 5641 / 5635 | 6277 / 6324 | +11.7 % |
| 35B-A3B, 1 GPU, prefill 31.5k | 4568 / 4585 | 4595 / 4589 | flat |
| 35B-A3B, 2 GPU `-sm layer`, prefill 31.5k | 6161 / 6163 | 6151 / 6152 | flat |

- Prefill cells list the two rounds. We did not dig into why only tensor-split prefill moves; we're happy to trace it if that's useful.
- We only have 2 GPUs here, so there is no 3-GPU number.

## Limits

- `n_offs != 0` and VISION stay on the unfused path, as before.
- Partial rotation is not unit-tested. The `RMS_NORM_MUL_ROPE` cases rotate the whole row (`n_dims == ne0`), so the untouched tail beyond `n_dims` is never exercised there. Both test models do exercise it: they rotate 64 of 256 head dims (sections `[11, 11, 10, 0]`), so the hash, probe and KLD gates above go through that tail.
- Only `-ub 2048` / `-b 2048` and q8_0 KV were measured end to end; f16 KV goes through the same rope code.
- With this in, #173's K/V cache write fold would see the M-RoPE chain too. That PR is not part of this one.
