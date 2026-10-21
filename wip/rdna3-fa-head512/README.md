# rdna3-fa-head512: gfx1100 head-512 WMMA FA rows, and head 512 on WMMA prefill again on RDNA3_0

Thank you for the issue #30 work in r5 (WORKLOG 260918.005): the head-512 measurements on gfx1100 and the cap back
to 256 made the cause easy to find, and `GGML_CUDA_FA_WMMA_MAX_HEAD` made the before / after a same-binary switch.
This gives RDNA3_0 its own head-512 rows so that head 512 can take WMMA prefill there again.

One format-patch on top of `v12-097cf33d9-r99` (tree `72d8b21d`, applied tree `e929ed67`), for `git am`.  2 files
(`fattn-mma-f16.cuh`, `fattn.cu`), +21 / -1.

**Hardware: RX 7900 XTX (gfx1100, RDNA3, 24 GB)**, Ryzen 7 9800X3D, Windows 11 + WSL2 (Ubuntu 24.04), ROCm 10.2 nightly
`20261013-38325106112` (HIP 7.17.26405) in a container.  All numbers below are from this box under WSL2.

## What we found

The head-512 rows of `ggml_cuda_fattn_mma_get_config_rdna` are shared by RDNA4 and RDNA3_0 and keep Q in registers
(`Q_in_reg = true`).  On gfx1100 every head-512 WMMA kernel then sits at the 256-VGPR limit and spills: on plain r99,
994-1362 spilled VGPRs per lane (2.1-3.3 KB scratch) for the `ncols2 = 8` kernels Gemma 4's global layers use.  We think
that is the WMMA-vs-tile gap of the r5 table: with `GGML_CUDA_FA_WMMA_MAX_HEAD=512` on r91, gemma-4-26B-A4B
`pp2048 @ d32768` q8_0 was 1320 t/s against 1826-1892 for the tile kernel.

## What it changes

- `fattn-mma-f16.cuh`: RDNA3_0 (`!is_rdna3_5 && !is_rdna4`) gets its own rows for DKQ = DV = 512 at 16 / 32 / 64
  columns: 128 threads, occupancy 1, Q in shared memory, 64-wide K / V / combine batches (`nbatch_fa` 64 at 16
  columns, 32 at 32 / 64).  Spills drop to 46-149 per lane.  (AMD prefill takes at most 32 columns for heads above 256,
  see `ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1`, so the 64-column row is only there for consistency.)
  RDNA4 and RDNA3_5 rows are unchanged.
- `fattn.cu`: DKQ == DV == 512 on RDNA3_0 takes the WMMA prefill path on top of the 256 cap.  The cap itself stays at
  256, so other heads above 256 (320, 576) keep the tile kernel there, as we could not measure them.  The decode /
  verify band (`n_q <= 8`) is untouched and stays on the tile kernel.  `GGML_CUDA_FA_RDNA3_WMMA_512=0` turns the new
  path off; `GGML_CUDA_FA_WMMA_256=0` and `GGML_CUDA_FA_WMMA_MAX_HEAD` also take it out, as before.

How the rows were picked: a sweep of 14 row sets at `pp2048 @ d32768` (forced WMMA) on r91; six did not compile and two
more failed at run time on this box.  The fastest variants all keep Q in shared memory (no variant isolated that one
change); 128 threads with 64-wide batches was the fastest point (q8_0 2793 / f16 2841 t/s).  32-wide batches spill
less but were slower (2334 / 2412), 128-wide batches spilled more (2473 / 2490), and the 256- and 512-thread variants
with 32- or 64-wide K/V blocks do not compile.

## Tests

gemma-4-26B-A4B-it UD-Q4_K_M (unsloth GGUF, sha256 `f2c28b3d…`; 30 layers, 5 global layers at head 512 / GQA 8,
25 SWA layers at head 256), your README build (`-DGPU_TARGETS=gfx1100`), `-ngl 99 -fa 1`.

**Prefill on r99** (`llama-bench -p 2048 -n 0`, t/s, plain r99 against r99 + 0001, two rounds, order flipped):

| KV | d 32768 | d 98304 |
| --- | --- | --- |
| q8_0 | 1853 / 1813 -> 2736 / 2700 (+48 %) | 822 / 808 -> 1426 / 1426 (+75 %) |
| bf16 | 1877 / 1856 -> 2754 / 2703 (+46 %) | 845 / 835 -> 1449 / 1433 (+72 %) |

**Prefill on r91** (same patch, same-binary switch, two rounds alternated, t/s, tile -> WMMA):

| KV | d 0 | d 32768 | d 65536 | d 98304 |
| --- | --- | --- | --- | --- |
| q8_0 | 4650 / 4620 -> 4578 / 4839 | 1894 / 1812 -> 2507 / 2681 | 1126 / 858 -> 1725 / 1841 | 832 / 826 -> 1439 / 1429 |
| f16 | 4594 / 4554 -> 4615 / 4651 | 1779 / 1780 -> 2546 / 2522 | 1116 / 1100 -> 1742 / 1730 | 804 / 879 -> 1455 / 1322 |
| bf16 | 4692 / 4726 -> 4892 / 4850 | 1864 / 1861 -> 2738 / 2701 | 1160 / 1157 -> 1881 / 1882 | 835 / 839 -> 1434 / 1433 |

Depth 0 is within noise (+1.3 to +3.4 % on the means, less than the round-to-round spread).  The q8_0 tile run at
d65536 in round 2 (858) looks like an outlier against round 1 (1126).  The d98304 q8_0 pairs are from a separate
re-run; in the first run the tile kernel read 751 / 737 there (and one WMMA run gave no result).  The f16 d98304 WMMA
runs spread 1455 / 1322.

**Decode** (r91, `tg64 @ d32768`, q8_0, six rounds order flipped, switch off / on): 93.53 / 94.44, 94.13 / 94.41,
93.67 / 95.12, 93.62 / 94.77, 94.55 / 94.95, 94.28 / 94.27 t/s.  Unchanged, as expected: decode does not take this
path.  (An earlier two-round check read 0.8 % lower with the switch on; the six-round re-run did not reproduce it.)

**Correctness.**
- `test-backend-ops -o FLASH_ATTN_EXT`: 6389/6389 on r99 + 0001 (and on r91 + 0001).
- Greedy text, 400 tokens: with `GGML_CUDA_FA_RDNA3_WMMA_512=0` the patched r99 build gives exactly plain r99's text on
  gemma-4-26B-A4B (prose `901d6a4c708e`, code `e2d6a4b7528a`).  With the new path on the text changes (prose
  `f94471e907a6`, code `5b02ff4954b6`), as it does whenever the prefill kernel changes.  Qwen3.8-27B (head 256) and
  Qwen3-30B-A3B (head 128) are identical to plain r99 (`8345c992d1fa` / `47114faa2042`, `bf70e0f571ea` /
  `6b19db0bf7d8`).
- KLD of the WMMA prefill against the tile prefill (r91, `llama-perplexity -c 2048 -b 2048 -ub 512 --chunks 16`,
  q8_0 KV, the llama.cpp docs as text): mean 0.0959, 99.9 % 7.96.  For scale, the tile kernel against itself at
  `-ub 256` vs `-ub 512` on the same text: mean 0.326, 99.9 % 16.4.  (PPL on this text is high for an instruct
  model, about 82, so both figures are large in absolute terms; the comparison is the point.)  We do not have your
  prefill-logit gate here.

## Limitations

- Measured on gfx1100 (WSL2), single GPU, one head-512 model.  gfx1101 / gfx1102 are included by the RDNA3_0 macros but
  not tested.  Heads 320 / 576 are left on the tile kernel.
- #195 also edits `fattn-mma-f16.cuh` (the RDNA4 band rows), in a different part of the file.

No urgency on our side.  Happy to re-run anything on your config.
