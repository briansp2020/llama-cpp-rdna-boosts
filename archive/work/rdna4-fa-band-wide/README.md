# wip/rdna4-fa-band-wide — one pass over the KV cache for 5..8-token verify batches

One `git am` patch for r21 (`fattn-mma-f16.cuh` + `fattn.cu`, +15/-1). Bit-exact.

## Why

The band folds GQA 6 into `ncols2 = 8` and runs `ncols1 = 4` (32 columns) per block. A 5..8-token
verify batch therefore launches two column tiles, and each streams the whole KV cache. With q8_0 KV
the band is ~70 % slower at `n_q = 5` than at `n_q = 4` (gfx1201, head 256, 4 KV heads, us per call):

| kv | n_q 1..4 | n_q 5..8 |
|---|---|---|
| 51200 | 287-294 | 476-484 |
| 98304 | 490-500 | 852-862 |

A DFlash2 / MTP verify at n-max 4 is `n_q = 5`, so this is the width it runs at on every step.

## The change

- **Row:** `ggml_cuda_fattn_mma_get_config_rdna()` gets a band row for `ncols == 64`:
  `fattn_mma_config(512, 1, 64, 64, 64, 32, 1, true)`. That is the 32-column band row with 16 warps
  instead of 8, so each Q column group still has `np = 4` warps. Every column gets the same KV-row
  split and the same combine order as in the 32-column block, so the result is bit-identical.
  `nbatch_combine` is 32 to keep the 16-warp combine buffer within 64 KiB of LDS. It sums the same
  partials in the same order, in narrower DV chunks.
- **Dispatch:** the band sends `n_q > 4` to the `<8, 8>` instance when the band runs 4-column tiles
  (the q8_0-style types) and the KV length is at least 4096. Below ~2k the 16-warp block is mostly
  idle and slower (kv 1024: +5..8 %). Both paths give the same result, so the gate cannot change an
  output. The 2-column f16/bf16 mode is untouched.
- **Kill switch:** `GGML_HIP_FA_BAND_WIDE=0` restores the two-tile launch.

## Results (gfx1201, R9700, ROCm 10.0, q8_0 KV)

`test-backend-ops perf`, us per call, `n_q` 5..8 (two-tile -> one block):

| kv | 4096 | 8192 | 16384 | 32768 | 51200 | 98304 |
|---|---|---|---|---|---|---|
| change | -11..-12 % | -15 % | -10 % | -11 % | -12 % | -13 % |

`n_q` 1..4 and f16 are unchanged.

Server, 27B UD-Q4_K_XL, q8_0 KV, DFlash2 n-max 4, greedy, 768 tokens, off/on/off/on (ms per verify
step, repeats within 0.1 ms):

| | doc 50k | code 97k |
|---|---|---|
| two tiles | 55.9 | 61.95 |
| one block | 55.1 (-1.4 %) | 60.3 (-2.6 %) |

All texts were byte-identical between the arms.

## Gates

- **`FLASH_ATTN_EXT`:** 6354/6354.
- **`test-logits-width-probe`:** PASS, with per-W and row0/row1 hashes identical on vs off, on 6
  runs: 27B UD-Q4_K_XL q8_0 (`RS=from_w`, `RS=0`), 27B f16, 27B IQ4_XS q8_0, 35B-A3B q8_0, and
  gemma-4-26B q8_0. These were run before the KV gate was added, so the wide path was exercised.

## Tried, didn't help

Two things I tried on top of the wide block did not pay off.

**Register-staged prefetch.** I fetched the next K/V batch's raw q8_0 bytes into registers during
the current batch's MMA. It was bit-exact but slower: kv 98304 `n_q = 1` went +7 %. The kernel is
already at 256 VGPRs, so the prefetch registers raised the 32-column block's scratch from 408 to
648 B per thread.

**LDS DMA.** A register-free copy (`__builtin_amdgcn_global_load_lds`) is not available on gfx1201.

The `GGML_HIP_FA_BAND_WMMA_SPLIT` values 48, 64 and 96 were all slower than the default of 32, with
the wide block too.

## Status after r126/r129 (2026-10-24) -- measured again, declined (TODO #99 retired)

r126 **deleted** the `ncols == 64` band row from the delivery, because it was also the prefill
instantiation (deep-context prefill -26 % on the FA kernel at `pp512 @ d98304`; `WORKLOG.md` 261024.007).
The W=5..8 verify band therefore runs the 256-thread table row `(256, 2, 64, 128, 128, 64, 1, true)` at
KV >= 4096, and the two-tile 16-warp path (`ncols1 = 4`, `GGML_HIP_FA_BAND_WIDE=0`) below it.

`TODO.md` #99 kept "restore the W=5..8 band geometry via a compile-time band instantiation, **only if a
decode-bound workload shows it pays**".  r129 measured the premise and it does not hold on this fleet:

* **There is no decode-bound head-256/GQA-8 workload here.**  The only models that reach the band are MoE
  (Flash-Next, Gemma4-A4B) and are measured with host experts (`-ncmoe`), i.e. FA is a minority of the
  step -- r126's own arm already showed the row swap was noise end to end (248.8/239.6 vs 241.2/249.7 t/s).
* **The two available geometries, end to end** (gfx1201, 3 GPU `-sm tensor`, Flash-Next IQ4_NL +
  shared MTP head, `-ncmoe 48`, q8_0 K/V, `-c 32768`, `prompts/recall.txt`, `-n 200`, `--spec-draft-n-max 7`
  so the verify is W=8, `ROCBLAS_USE_HIPBLASLT=0`, 2 reps each): **default 110.1 / 110.2 t/s vs
  `GGML_HIP_FA_BAND_WIDE=0` 108.9 / 107.6 t/s** -- the shipping arm is the faster of the two by ~1.6 %, and
  greedy text is byte-identical across all four runs (`883f4c55ece2`), consistent with the "bit-identical
  per column" claim for the two-tile path.  So the 16-warp geometry's kernel-level ~17 % does not reach the
  wall clock even on the arm that reaches W=8 most often.

Cutting the instantiation would therefore ship an unvalidated FA variant for no measured gain: it needs a
new `bool band` template parameter threaded through the kernel/launcher **and** a fresh
`template-instances/*.cu` set for the KV-type axis (a shape that is already the build-time hot spot), and
the prefill-side purity gate.  Declined.  If a decode-bound head-256/GQA-8 model ever appears, start from
`GGML_HIP_FA_BAND_WIDE=0` (the two-tile 16-warp path already exists and is bit-identical) and only then
consider the one-pass 16-warp variant; the r126 measurement and the two-arm A/B above are the baseline it
would have to beat.

## Status after r134 (2026-10-25) — implemented, because the deletion broke width purity

The "Declined" verdict above is superseded by issue #224.  Deleting the `ncols == 64` band row left query
widths 5..8 on prefill's 8-warp row, whose per-column KV split is `np = 2` against the 32-column block's
`np = 4` — a different reduction, so widths 5..8 were no longer bit-identical to widths 1..4 and greedy decode
with a DFlash2 draft diverged from plain decode on dense head-256 GQA-6 (Qwen3.8-27B, 8K and 110K prompts).
The instantiation this record declined is now the fix: the band's 64-column block runs
`fattn_mma_config(512, 1, 64, 64, 64, 32, 1, true)` through a compile-time `band_wide` flag (its own case +
fetch-ahead translation units), prefill keeps the 256-thread table row, and a `static_assert` on the two band
rows makes a repeat a build error.  The "no measured gain" premise held: end to end the wide launch measures
at parity with the two-tile arm (8K, a 131072-cell FA range, and the W=8 Flash-Next arm above), and the reason
to ship it is purity, not throughput.  Numbers, the probe's `P >= 4096` coverage fix and the gates:
`benchmarks/2026-10-25-fa-band-width-purity.md`, `WORKLOG.md` 261025.004.
