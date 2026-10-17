# wip/gdn-l2-norm-pair

`build_gdn_l2_norm` normalizes the GDN q and k with an `RMS_NORM` + `SCALE` each. Both read views of the same conv output with the same shape, strides, eps and scale, so every GDN layer ran two identical `rms_norm_f32<256, false, false, true>` launches. This patch runs both in one launch.

One format-patch on `v12-097cf33d9-r80` (tree `873d0335`); applied tree `a297c909`.

## What changes

- **`rms_norm_scale_pair_f32<256>`** (`norm.cu`):
  - Blocks `[0, nrows)` take the first `(x, dst)` pair, and blocks `[nrows, 2*nrows)` take the second.
  - The per-row body is `rms_norm_f32<256, false, false, true>`'s: the same `block_reduce`, `rsqrtf(mean + eps)` and `scale_out * (scale * x)`. So each output is bit-identical to its own launch.
- **`ggml_cuda_op_rms_norm_scale_pair_fused`**: the launcher, `ncols < 1024` only.
- **Matcher**, in the existing `{RMS_NORM, SCALE}` arm of `ggml-cuda-try-fuse.cu`. A second `RMS_NORM` + `SCALE` is taken when all of these hold:
  - only no-op nodes (VIEW / RESHAPE / PERMUTE / TRANSPOSE / NONE) sit between the two pairs;
  - both inputs are F32 views of the same `view_src`, with equal `ne` / `nb`;
  - eps and the scale's op params are equal;
  - the second input does not read the first pair's output.

  Otherwise the single fusion runs as before.
- **Kill switch**: `GGML_CUDA_DISABLE_NORM_PAIR=1` restores two launches.

## Gates

Setup:
- R9700 (gfx1201), native Linux, HIP runtime 7.15.26333.
- Stock r80 (`s80s`) against r80 + this patch (`s80np`), same build dir.
- Models:
  - Qwen3.8-27B Q4_K_M with its own MTP head (`--spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0`);
  - Qwen3.6-35B-A3B Q4_K_M, plain decode.
- Server settings: `-c 65536 -ngl 99 -fa on -ctk q8_0 -ctv q8_0 -ub 2048 -b 2048`.

**Greedy hashes** (400 tokens, prose / code): identical to stock in every arm.

| model | config | prose / code |
|---|---|---|
| 27B | 1 GPU: stock / patched / patched + kill / patched + `GGML_CUDA_GRAPH_OPT=1`; 2 GPU `-sm layer` | `d12cdeede326` / `e8a93c816237` |
| 27B | 2 GPU `-sm tensor` | `7cfb359d5c5d` / `dcd93c8965ce` |
| 35B-A3B | 1 GPU: the same four arms; 2 GPU `-sm layer` | `0eedd580608a` / `365e3058e68f` |
| 35B-A3B | 2 GPU `-sm tensor` | `c0cf0739bdab` / `930bcf1f4405` |

**`test-logits-width-probe`** (1 GPU, q8_0 KV): PASS with worst maxdiff 0 at P=256 and P=3584 on both models. The W8 and row-0 hashes are identical to stock (27B `498b73a62646b535` / `aa42818af2333c50`; 35B-A3B `251c193d7b72cf34` / `f172e67372ec2698`).

**Prefill-logit** (wikitext-2 test 16 × 2048, 1 GPU, against bases recorded from stock r80): mean KLD 0.000000 on both models. Max KLD (27B 0.000065, 35B-A3B 0.000058) and same top p (99.994 %) are exactly what stock r80 gives against its own base.

**`test-backend-ops`**: `-o RMS_NORM` 53/53, `-o SCALE` 7/7.

**Faults**: 0 memory-access faults or asserts in any run.

## Launch counts

`rocprofv3 --kernel-trace`, 35B-A3B, 1 GPU decode, the same greedy work on both builds:

| kernel | stock r80 | r80 + patch |
|---|---|---|
| `rms_norm_f32<256, false, false, true>` | 96,360 × 1.47 µs | — |
| `rms_norm_scale_pair_f32<256>` | — | 48,180 × 1.36 µs |

48,180 fewer launches. GPU time for the norms goes from 142 ms to 65 ms.

## Throughput

Two alternated rounds per arm, each the mean of 8 × 400-token greedy runs.

| config | stock r80 | r80 + patch | change |
|---|---|---|---|
| 35B-A3B, 1 GPU | 123.54 t/s | 124.98 t/s | +1.2 % |
| 27B MTP, 1 GPU | 67.35 t/s | 67.50 t/s | +0.2 % (kill switch: 67.30) |
| 27B MTP, 2 GPU `-sm tensor` | 99.39 t/s | 99.69 t/s | +0.3 % |

- 31.5k prefill is flat.
- On our hybrid Flash-Next model (2 GPU `-sm tensor`, `--n-cpu-moe 47`), a paired 8-round A/B gave decode +0.37 % (95 % CI +0.04 to +0.70 %) with identical output in all 16 server loads.

## Limits

- Pairs only (q and k). It does not batch other back-to-back norms.
- Rows shorter than 1024 only, which is the 256-thread `rms_norm_f32` path. Longer rows keep two launches.
