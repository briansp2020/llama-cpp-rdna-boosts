# rdna3-mmvq-q4k-vdr4: 32-element dense Q4_K dot product and verify rows per block on gfx1100 (RDNA4 opt-in)

Thank you for the multi-row verify block decomposition in your RDNA4 weight rule (`calc_rows_per_block_weight`) and
for the MoE `vec_dot_q4_K_q8_1_vdr4`, whose two-chunk layout 0002 reuses for dense Q4_K.  These patches build
directly on both, and the row-block structure, which keeps every width bit-identical, is what made them easy to check.

Four format-patches on top of `v12-097cf33d9-r67` ("r67" below; tree `3ccabe15`, applied tree `14953d89`), for
`git am`.  4 files (`mmvq.cu`, `mmvq-rdna-params.cuh`, `mmvq-vecdot.cuh`, `vecdotq.cuh`), +106 / -9.  Everything is
under RDNA3_0 (gfx1100-1103), except 0004, an RDNA4 build-time opt-in that is off by default; RDNA4 without the flag,
RDNA3_5, CUDA and the other targets are untouched by construction (RDNA4 checked below; RDNA3_5 and CUDA not built).

**Hardware: RX 7900 XTX (gfx1100, RDNA3, 24 GB)**, Ryzen 7 9800X3D, Windows 11 + WSL2 (Ubuntu 24.04),
ROCm 10.2 nightly `20261013-38325106112` (HIP 7.17.26405) in a container.  Also run on one **R9700 (gfx1201)**,
with and without 0004's flag: see "Also on gfx1201" below.

## Why

After the vecdot patches in r62 (#159), the gfx1100 verify-width mmvq kernels (W = 2..8 columns, the MTP verify
step) were still the bulk of decode: in a rocprofv3 trace of one MTP request, Q4_K kernels at W = 4 were about 47 %
of all GPU time, at about 670 GB/s against about 825 GB/s for the same weights at W = 1.  The W = 4 Q4_K hot loop
was 475 instructions per 16 elements, of which 64 were `v_dot4`; most of the rest was the per-(row, column) scale,
convert and fma and the per-row scale/min unpack.

## 0001: two rows per block for Q6_K verify batches on gfx1100 (output unchanged)

The gfx1100 multi-row rule covered only Q4_K; Q6_K verify batches ran one row per 8-warp block.  Two rows, the same
row-block decomposition as the Q4_K and RDNA4 rules (each row's accumulation unchanged).  Kernel time at W = 4:

| Q6_K tensor (Qwen3.8-27B) | 1 row | 2 rows | 4 rows |
|---|---|---|---|
| output (lm_head) 248320 x 5120 | 1457 us | 1191 us (-18 %) | 1208 us |
| attn_qkv 10240 x 5120 | 64.5 us | 55.7 us (-14 %) | 61.5 us |
| ffn_down 5120 x 17408 | 92.4 us | 89.3 us (-3 %) | 92.2 us |

8 rows spills to scratch; 0 scratch at 2.

## 0002: 32-element dense Q4_K dot product on gfx1100

`vec_dot_q4_K_q8_1_vdr4_dense` takes 32 elements per call (VDR 4), in the layout of `vec_dot_q4_K_q8_1_vdr4`.
The 32 elements cover the same two sub-blocks, 16 each, so each sub-block's 16 products are summed in one integer
`dp4a` chain before its single scale / convert / fma (|dot1| <= 16*15*127, times the 6-bit scale < 2^21: exact).
The mins term keeps the exact-product form from #159, and the scale words are packed as in `vec_dot_q4_K_q8_1`.
RDNA3_0 dense Q4_K only: MoE keeps `vec_dot_q4_K_q8_1_vdr4`, other targets `vec_dot_q4_K_q8_1`.

`get_vdr_mmvq_dense()` is the one VDR the dense kernels and the host-side K split both read, so they agree on the
override.

**Output can change.**  The order in which K is summed changes, so greedy text can differ from r67 (here the prose
and 26k prompts did, the code prompt did not); measured under Correctness, it is at rounding level (mean KLD 0.0006
against r67).  Single-token decode and every verify width share the dot product, so
widths stay bit-identical (width probe and MTP == plain below).  `vecdotq.cuh` notes that VDR 4 once lost the dense
verify widths to multi-column register pressure (issue #30), in case it's useful: here, with 0003's two rows, the
W = 4 kernel uses 92 VGPRs and W = 8 121, without scratch, and the A/B below is a gain.

gfx1100 W = 4 hot loop: 599 instructions per 32 elements (was 475 per 16).

## 0003: two rows per block for Q4_K verify batches with the 32-element dot product (output unchanged)

The 4-row rule was chosen with the 16-element dot product; with 0002, 2 rows is faster.  Kernel time at W = 4, 4 -> 2
rows: ffn_gate / ffn_up 71.4 -> 69.4 us, attn_qkv 41.8 -> 39.3, attn_q 48.3 -> 46.5, attn_gate 27.1 -> 26.1.  8 rows
is slower everywhere (VGPR 190).  Rows are independent: same text as 0002.

## 0004: RDNA4 build-time opt-in for the 32-element dot product (default off)

`-DGGML_HIP_RDNA4_Q4K_DENSE_VDR4` (e.g. `-DCMAKE_HIP_FLAGS=-DGGML_HIP_RDNA4_Q4K_DENSE_VDR4`) makes RDNA4 use 0002
for dense Q4_K too, for an A/B.  One flag gates both sides (device dot-product choice and `get_vdr_mmvq_dense`).
gfx1201 compile check (`mmvq.cu`, your README flags plus `-DGPU_TARGETS=gfx1201`), Q4_K W = 4 at RDNA4's 4 rows:
loop 478 instructions per 16 elements -> 591 per 32, VGPR 94 -> 118 (W = 8: 166 -> 189), 0 scratch either way.

## Tests

Qwen3.8-27B Q4_K_M (lmstudio-community), your README build
(`-DGGML_HIP=ON -DGGML_HIP_RCCL=1 -DGPU_TARGETS=gfx1100 -DCMAKE_BUILD_TYPE=Release`), no tuning env.  Fresh server
per run, alternating off/on/off/on, greedy, one slot:

```
llama-server -m <model> -c 16384 -np 1 -ngl 99 -ctk q8_0 -ctv q8_0 --spec-type draft-mtp --spec-draft-n-max 3
```

Short prompts, 400 generated tokens, 3 requests per prompt with the first discarded; 26k-token prompt (the first
~95 KB of `ggml-cuda/*.cu` plus "summarize", `-c 40960`), two 300-token generations on the cached prompt.

**On r67** (plain r67 vs r67 + 0001..0004, default build, i.e. no RDNA4 flag), t/s, means of 4 runs x 2 requests:

| | r67 | + 0001..0004 | change |
|---|---|---|---|
| MTP, code (Python LRU cache) | 82.26 | 86.92 | +5.7 % |
| MTP, prose (hash table explanation) | 69.18 | 73.99 | +7.0 % |
| MTP, 26k prompt | 54.94 | 59.90 | +9.0 % |
| plain decode, prose | 37.52 | 38.47 | +2.6 % |

Every "on" request is faster than every "off" request in all four rows.  The code prompt's text and acceptance are
the same in both builds (`47114faa2042`, 0.846), so its +5.7 % is kernel time alone.  The prose and 26k text change
with 0002 and their draft acceptance moved up (0.666 -> 0.672, 0.638 -> 0.668), so part of those two gains is
acceptance rather than kernel time; plain decode (no draft) is +2.6 %.

**Per patch.**  Measured while developing, on r62 + #155 + #156 + #159 (what r67 contains of ours; the four files
these patches touch are identical there and in r67), two separate off/on/off/on A/Bs:

0001 against that baseline:

| | baseline | + 0001 | change |
|---|---|---|---|
| MTP, code (Python LRU cache) | 82.52 | 83.11 | +0.7 % |
| MTP, prose (hash table explanation) | 70.12 | 70.18 | flat |
| MTP, 26k prompt | 55.33 | 55.96 | +1.1 % |
| plain decode, prose | 37.80 | 37.72 | within noise |

0002 + 0003 against + 0001:

| | + 0001 | + 0001..0003 | change |
|---|---|---|---|
| MTP, code | 83.48 | 87.33 | +4.6 % |
| MTP, prose | 70.60 | 74.19 | +5.1 % |
| MTP, 26k prompt | 55.68 | 59.61 | +7.1 % |
| plain decode, prose | 37.63 | 38.32 | +1.8 % |

0002 alone (with 4 rows) was +3.7 % / +4.5 % / +6.4 % / +1.9 % in its own A/B.

**Correctness.**
- Plain r67: prose `75cbd6a6b11c` / code `47114faa2042` / 26k `1a6343387a08`.  0001 leaves them unchanged; with
  0002 they are `8345c992d1fa` / `47114faa2042` / `18ec678cd873` (decoded and checked: a hash-table explanation, a
  Python `LRUCache`, a file-by-file summary of the CUDA sources), and 0003 and 0004 change nothing further.
- MTP text equals plain-decode text in every run.
- Your `test-logits-width-probe` (widths 1..8 bit-identical): PASS at prefix 256 and 3584 on r67 + 0001..0004.
- r67 + 0001..0004: 0 scratch in all 1572 mmvq kernels.
- **How much 0002 moves the logits.**  `llama-perplexity` with `-ub 8`, so every matmul goes through mmvq (at the
  default 512-token ubatch it would go through MMQ and never reach 0002), `-c 512 --chunks 24 -ctk q8_0 -ctv q8_0`,
  on the README / docs `*.md` of llama.cpp (458 KB of prose).  Base logits: plain r67 at `-ub 8` (PPL 4.954).  For
  scale, the same comparison for plain r67 at `-ub 512` (your MMQ prefill path against your mmvq path, an existing
  rounding-level difference):

  | vs r67 `-ub 8` | r67 + 0001..0004, `-ub 8` | r67, `-ub 512` (for scale) |
  |---|---|---|
  | mean KLD | 0.00063 ± 0.00002 | 0.00349 ± 0.00006 |
  | 99.9 % KLD | 0.023 | 0.039 |
  | same top token | 98.84 ± 0.14 % | 96.67 ± 0.23 % |
  | PPL ratio | 0.9993 ± 0.0005 | 1.0044 ± 0.0012 |
  | RMS Δp | 0.71 % | 1.91 % |

  So the change is at rounding level, about a fifth of the prefill-vs-decode difference, and perplexity is unchanged
  within its error.

## Also on gfx1201

One R9700 (HIP device 0 of a 2x R9700 box, native Linux), the same model file (same size, 16,810,714,336 bytes),
the server line above, no tuning env, fresh server per run.  Builds: plain r67; r67 + 0001..0004 without the flag;
the same with `-DCMAKE_HIP_FLAGS=-DGGML_HIP_RDNA4_Q4K_DENSE_VDR4`.  Greedy, 400 tokens, 3 requests per prompt with
the first discarded (prose: a hash-table collision explanation in C; code: a Python `LRUCache`).  These runs used the
previous revision of the patches (applied tree `16fc7f31`), which differs from this one only in a code comment.

- **Without the flag (gating check):** same text and acceptance as r67, speed within 0.02 t/s.  MTP prose
  `d12cdeede326`, MTP code `3416991d20a0`, plain decode the same hashes, in both builds; acceptance 0.612 / 0.733 in
  both; 63.39 vs 63.40 and 71.97 vs 71.95 t/s (MTP), 31.13 vs 31.13 t/s (plain).
- **Width probe:** PASS at prefix 256 and 3584, with and without the flag.
- **With the flag**, off/on/off/on (2 rounds, each the mean of 2 requests; run-to-run spread about 0.05 t/s):

  | | without flag | with flag | change |
  |---|---|---|---|
  | MTP, prose (text unchanged, `d12cdeede326`) | 63.41 / 63.38 | 64.23 / 64.23 | +1.3 % |
  | MTP, code (text `3416991d20a0` -> `729020c7aa2d`) | 71.98 / 71.93 | 75.77 / 75.78 | +5.3 % |
  | plain decode, prose (text unchanged) | 31.15 / 31.14 | 31.66 / 31.63 | +1.6 % |
  | plain decode, code (text changed, as MTP) | 31.16 / 31.12 | 31.65 / 31.63 | +1.6 % |

  On gfx1201 the prose text happened not to change with the flag, and its MTP acceptance went slightly down
  (0.612 -> 0.599), so the prose +1.3 % is kernel time, if anything understated.  The code text changed and its
  acceptance moved 0.733 -> 0.759, so part of its +5.3 % is acceptance.  MTP text equals plain-decode text in every
  arm.  No faults in any server log.

If you'd rather have it on by default on RDNA4, it's a small change in 0004 (the two `#if`s on the flag); we left it
off since that's your call, and it can change RDNA4's greedy text.

## Tried and dropped

- Q6_K with the same integer pre-sum: no kernel change (Q6_K mmvq is memory-bound on gfx1100).
- Hoisting wave-uniform row / column base pointers so the loads use scalar-base addressing: the compiler kept
  64-bit per-lane addresses and the loop got longer.
- Q4_K at 8 rows per block, with either dot product: slower.

## Limitations

- RDNA4 on one R9700 only, and without a kernel trace (the gfx1201 instruction / VGPR numbers in 0004 are from the
  compile check).
- One model (Q4_K / Q6_K weights).  The gfx1100 numbers are under WSL2.
- RDNA3_5 and CUDA were not built; they are untouched by construction (everything is under RDNA3_0 or the RDNA4
  flag).

No urgency on our side.  Happy to re-run anything on your config.
