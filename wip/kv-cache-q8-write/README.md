# kv-cache-q8-write: fold the q8_0 K and V cache writes into the rope + FWHT kernel

Thank you for the issue #67 work that made the address-selected rope fusions round exactly like the chains they
replace; this patch keeps to that rule.  It does for a q8_0 cache with the KV rotation on what the existing rope ->
set_rows fusion does for f16 / f32 caches, in the rope kernel that already runs the FWHT (#172).  Bit-identical.

**Stacked (updated for r80):** one format-patch on top of `v12-097cf33d9-r80` + #175 (mrope-rope-fusion, 0001..0002), applied in that order (applied tree `67245faf`).  #171 and #172 are in r78; #175 brings the fused rope + FWHT kernel to the M-RoPE / IMROPE models, which is what lets this fold fire on Qwen3.x.  Same 5 files, +188 / -34, unchanged in substance from the r71 version (rebased).  `GGML_CUDA_DISABLE_ROPE_FWHT_SET_ROWS` turns it off.

The r71 measurements below are kept for the record; the r80 results on Qwen3.x are in the next section.

**Hardware: RX 7900 XTX (gfx1100, RDNA3, 24 GB)**, Ryzen 7 9800X3D, Windows 11 + WSL2 (Ubuntu 24.04), ROCm 10.2 nightly
`20261013-38325106112` (HIP 7.17.26405) in a container.  All speed numbers below are from this box under WSL2, except the RDNA4 section (labelled there).

## r80 on Qwen3.x (on #175)

Setup:
- R9700 (gfx1201), native Linux, HIP runtime 7.15.26333.
- r80 + #175 (`s80p`) against r80 + #175 + this patch (`s80kv`), same build dir.
- Models: Qwen3.6-35B-A3B Q4_K_M, plain decode; Qwen3.8-27B Q4_K_M with its own MTP head (`draft-mtp` n-max 3).
- Server settings: `-c 65536 -ngl 99 -fa on -ctk q8_0 -ctv q8_0 -ub 2048 -b 2048`.

These are the re-entry conditions from your review.

**The matcher fires.** A `rocprofv3` trace of 1-GPU 35B-A3B decode, the same greedy work on both builds:

| kernel | r80 + #175 | + this patch |
|---|---|---|
| `k_set_rows_q8_0_group` | 32,120 × 1.46 µs | — |
| `rms_norm_mul_rope_f32<256, false, float, 256>` | 32,120 × 3.15 µs | Q: 16,060 × 3.23 µs; K with the cache writes: 16,060 × 4.91 µs |

All 32,120 cache-write launches are gone. qwen35 / qwen35moe build attention through the non-iSWA `build_attn`, which expands K last, so the K cache write follows the K rotation, as the matcher requires.

**Greedy hashes** (400 tokens, prose / code) are identical to r80 + #175 and to plain r80:

| model | config | prose / code |
|---|---|---|
| 35B-A3B | 1 GPU: patched / kill switch / `GGML_CUDA_GRAPH_OPT=1`; 2 GPU `-sm layer` | `0eedd580608a` / `365e3058e68f` |
| 35B-A3B | 2 GPU `-sm tensor` | `c0cf0739bdab` / `930bcf1f4405` |
| 27B MTP | 1 GPU: the same arms; 2 GPU `-sm layer` | `d12cdeede326` / `e8a93c816237` |
| 27B MTP | 2 GPU `-sm tensor` | `7cfb359d5c5d` / `dcd93c8965ce` |

The gates above ran with this patch on #175 0001. The final stack, on #175 0001 + 0002 (the BF16 guard), was re-checked with greedy hashes on 1 GPU for q8_0, f16 and bf16 KV on both models plus Qwen3-30B-A3B. All matched plain r80, apart from the 30B-A3B bf16 case, which plain r80 cannot run.

**`test-logits-width-probe`** (1 GPU): PASS with worst maxdiff 0 at P=256 and P=3584 on both models. The W8 and row-0 hashes are identical to plain r80 (35B-A3B `251c193d7b72cf34` / `f172e67372ec2698`; 27B `498b73a62646b535` / `aa42818af2333c50`).

**Prefill-logit** (wikitext-2 test 16 × 2048, against bases recorded from stock r80): mean KLD 0.000000 on both models, at the stock-vs-stock floor (max 0.000058 / 0.000065, same top p 99.994 %).

**`test-backend-ops`**: `SET_ROWS` 159/159, `ROPE` 450/450, `RMS_NORM_MUL_ROPE` 296/296.

**Decode**, two alternated rounds, each the mean of 8 × 400-token greedy runs:

| config | r80 + #175 | + this patch | change |
|---|---|---|---|
| 35B-A3B, 1 GPU | 125.03 t/s | 125.70 t/s | +0.5 % (kill switch: 125.01) |
| 27B MTP, 1 GPU | 67.17 t/s | 67.61 t/s | +0.7 % |
| 35B-A3B, 2 GPU `-sm tensor` | 131.97 t/s | 132.55 t/s | +0.4 % |

31.5k prefill is flat. We only have two GPUs, so there is no 3-GPU number.

## What it does

With a q8_0 cache, each layer's K path ends `..., MUL_MAT(Hadamard), RESHAPE, VIEW, SET_ROWS(K)`, and the V write
`VIEW, SET_ROWS(V)` of the V row (rotated earlier) follows right after.  On that pattern the fused
`rms_norm_mul_rope_f32` (+ FWHT) writes both cache rows itself:

- K: warp 0 quantizes the rotated row straight from its FWHT registers into the K cache row.
- V: each block (one KV head) also quantizes its slice of the token's V row into the V cache row.

Both go through `ggml_cuda_quantize_q8_0_group`, which is `k_set_rows_q8_0_group`'s body moved into `set-rows.cuh`,
so the cache bytes are the same as the two SET_ROWS nodes write.  The two `set_rows` launches per layer go away; the
F32 rotation output of K is no longer written (its only reader was the K SET_ROWS).

Matcher: the #172 rope + FWHT match, then the RESHAPE / VIEW / SET_ROWS chain with single-use, non-output nodes, one
contiguous F32 row of `n_head_kv * head_dim` per token, I64 indices, q8_0 cache views; the V part needs the same row
shape.  The cache writes must not overlap anything the kernel reads.  Inside a concurrent-stream region
(`GGML_CUDA_GRAPH_OPT=1`) the V row comes from another stream, so only the K write is folded there.

## Tests

Qwen3-30B-A3B Q4_K_M (MoE, 48 layers) and Qwen3.8-27B Q4_K_M (dense), your README build (`-DGPU_TARGETS=gfx1100`), no
tuning env, `-ngl 99 -fa 1 -ctk q8_0 -ctv q8_0`.

**Kernel trace** (rocprofv3, one 400-token greedy MoE request): no `set_rows` launches remain; the K-path
`rms_norm_mul_rope_f32<…, 128>` with the cache writes takes 9.4 us per call (7.4 us without them), against two
`k_set_rows_q8_0_group` launches of 3.7 us each removed (about 5.4 us net per layer).

**Decode** (`llama-bench -p 0 -n 256 -r 10`, MoE, four alternated rounds over two builds of the same tree):

| | mean |
|---|---|
| r71 | 130.6 t/s |
| r71 + #172 + #171 + this, `GGML_CUDA_DISABLE_ROPE_FWHT_SET_ROWS=1` | 136.8 t/s |
| r71 + #172 + #171 + this | 140.2 t/s (+2.4 % vs the switch, +7.4 % vs r71) |

Dense: 38.76 / 38.78 -> 38.63 / 38.85 t/s (flat, within noise).

**Correctness.**
- Greedy hashes, 400 tokens, identical to r71: MoE plain decode prose `0116af00d69e` / code `8fd18d2f0b83`; dense with
  MTP (draft-mtp, n-max 3) prose `8345c992d1fa` / code `47114faa2042`.  With `GGML_CUDA_GRAPH_OPT=1` (K-only fold):
  MoE hashes again identical to r71's.
- Your `test-logits-width-probe` (q8_0 KV, FA): PASS with worst maxdiff 0 at prefix 256 and 3584, on both models.
- `test-backend-ops -o SET_ROWS`, `-o ROPE`, `-o MUL_MAT`: pass (they don't build the fused pattern; the hashes and
  the probe do).

## RDNA4

**Hardware: 1 x Radeon AI PRO R9700 (gfx1201)**, native Linux, ROCm 10.0; same models, llama-server
`-c 65536 -ngl 99 -fa on -ctk q8_0 -ctv q8_0`, the same stack (applied tree `01067997`).

- Greedy hashes identical to plain r71 on both models (MoE prose `744db56ea1c8` / code `6bb5966d061f`, dense prose
  `d12cdeede326` / code `e8a93c816237`), also with `GGML_CUDA_GRAPH_OPT=1`; width probe PASS (maxdiff 0) at 256 and
  3584; `test-backend-ops` SET_ROWS 159/159, ROPE 450/450, MUL_MAT 2444/2444.
- Trace (MoE decode): no `set_rows` launches remain; the K-path kernel with the cache writes takes 3.25 us per call
  (2.19 us with the switch set), against two `k_set_rows_q8_0_group` launches of 1.34 us each removed per layer
  (about 1.6 us net).
- MoE decode (two alternated rounds, consistent within 0.2 t/s): r71 132.7, the stack with
  `GGML_CUDA_DISABLE_ROPE_FWHT_SET_ROWS=1` 139.2, the stack 143.2 t/s (+2.9 % vs the switch, +7.9 % vs r71).  31.5k
  prefill within 0.5 %.

## Limitations

- Needs #175 first (and #171 / #172, which are in r78).
- q8_0 caches with the rotation only; other cache types keep the existing rope -> set_rows fusion or the plain path.
- Measured on gfx1100 (WSL2) and gfx1201 (native Linux).

No urgency on our side.  Happy to re-run anything on your config.
