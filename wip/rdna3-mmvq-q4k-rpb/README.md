# rdna3-mmvq-q4k-rpb: 4 rows per block for Q4_K verify batches on gfx1100

One format-patch on top of `v12-097cf33d9-r62` (tree `386a6a3d`; applied tree `a82acbba`), for `git am`.
2 files (`ggml/src/ggml-cuda/mmvq-rdna-params.cuh`, `ggml/src/ggml-cuda/mmvq.cu`), +17 / -4.  RDNA3_0
(gfx1100/1101/1102) only; RDNA4 keeps its table and guard as before.  Independent of our open #153 / #154.
(The binaries below were built from the same code; only a comment and the commit message's numbers were
updated afterwards.)

**Hardware: RX 7900 XTX (gfx1100, RDNA3, 24 GB)**, Ryzen 7 9800X3D, Windows 11 + WSL2 (Ubuntu 24.04),
ROCm 10.2 nightly `20261013-38325106112` (HIP 7.17.26405) in a container.  Not the gfx1201 box.

## What it changes

It ports the RDNA4 verify rows-per-block rule from #57 (`wip/mmvq-verify-rows`) to gfx1100, for Q4_K only.

- `calc_rows_per_block_weight`: on `MMVQ_PARAMETERS_RDNA3_0`, a Q4_K matrix-vector product with 2..8 columns
  (an MTP/DFlash verify batch) computes 4 weight rows per block, at the table's nwarps (1).  The q8_1
  activation columns are then loaded once per block instead of once per row.  Each row keeps its own thread
  mapping, K order and warp reduction, so the result is bit-identical to 1 row per block.
  `GGML_MMVQ_RDNA3_Q4K_WEIGHT_RPB` (compile-time, default 4) sets the count.
- `mmvq.cu`: the multi-row guard (row count must divide, at least 512 blocks, else the 1-row launch) now also
  applies to RDNA3_0, and computes the rows with the device's own `table_id` instead of the RDNA4 table.  On
  RDNA4 the value is the same as before.  The guard is compiled only for types where RDNA4 also takes more
  than one row (Q4_K is one), so adding another RDNA3_0 type needs the guard extended first (noted in the
  comment).

Single-token decode (1 column) is unchanged.

Sweep on gfx1100 (r59, same model): nwarps 1/2/4/8 x rows 1/2/4.  nwarps > 1 changes the K-split order (greedy
text changes) and was no faster, so rows 4 at nwarps 1 is the fastest bit-identical point we found.

## Tests

Qwen3.8-27B Q4_K_M (lmstudio-community), your README build
(`-DGGML_HIP=ON -DGGML_HIP_RCCL=1 -DGPU_TARGETS=gfx1100 -DCMAKE_BUILD_TYPE=Release`).  "off" is r62, "on" is
r62 + this patch (two binaries: the row count is a compile-time constant).  Fresh server per run, in the
order off, on, off, on.  Greedy, one slot.

```
llama-server -m <model> -c 16384 -np 1 -ngl 99 -ctk q8_0 -ctv q8_0 --spec-type draft-mtp --spec-draft-n-max 3
```

**Short prompts**, 400 generated tokens, 3 requests per prompt with the first discarded; each cell is the
two measured requests (t/s):

| | off | on | off | on |
|---|---|---|---|---|
| prose (hash table explanation) | 60.72 / 61.08 | 62.75 / 62.58 | 57.39 / 57.46 | 64.30 / 64.98 |
| code (Python LRU cache) | 71.70 / 71.79 | 73.49 / 73.37 | 67.65 / 67.42 | 76.71 / 76.63 |

Means of the four requests per arm: prose 59.2 -> 63.7 (+7.6 %), code 69.6 -> 75.1 (+7.8 %).

**26k-token prompt** (the first ~95 KB of `ggml-cuda/*.cu` plus "summarize"), `-c 40960`, two 300-token
generations on the cached prompt:

| | off | on | off | on |
|---|---|---|---|---|
| MTP n-max 3 | 49.30 / 49.16 | 49.94 / 49.98 | 46.60 / 46.52 | 52.11 / 51.86 |
| plain decode (control) | 34.22 / 34.09 | 34.09 / 34.12 | 34.16 / 34.11 | 34.35 / 34.31 |

Means: MTP 47.9 -> 51.0 (+6.4 %); plain decode unchanged, as expected for 1 column.

The spread is wide: the off runs moved 5-6 % between themselves, while plain decode stayed within 1 %, so the
noise is in the MTP path rather than the clocks.  The gain within each off/on pair was +1.5 % to +2.9 % for the first pair and +11.7 % to +13.5 % for the second.  Earlier
on r59 (same ROCm, model and prompts) it measured +7.1 % to +8.5 % (code, 26k, prose).

At 26k both arms ran with `GGML_CUDA_MMB_TILE=0`: r62 otherwise hits an intermittent prefill fault on this
setup (separate PR, `wip/rdna3-mmb-tile-spill`).  MMB is prefill-only, so it doesn't touch the verify batches
this patch changes.

**Correctness**: output identical with it on and off, and identical with and without MTP: prose
`75cbd6a6b11c`, code `47114faa2042`, 26k `1a6343387a08` in every run (each decoded and checked: a hash-table
explanation, a Python `LRUCache` class, a file-by-file summary of the CUDA sources).  Draft acceptance is the
same in every run (0.666 / 0.846 / 0.638).  `test-backend-ops -o MUL_MAT -p type_a=q4_K`: 120/120 pass, but
its shapes (m = 16) are below the multi-row guard, so they run the 1-row fallback; the model outputs above
are the real check of the 4-row path.

## Limitations

- One model and one quant (Q4_K_M; its Q6_K tensors are unchanged).  Other RDNA3_0 types not swept.
- WSL2 only; not tested on native Linux.  gfx1101/gfx1102 take the rule by family; only gfx1100 measured.
- Not tested: DFlash, multi-slot / concurrent requests, multi-GPU.

No urgency on our side.  Happy to re-run anything on your config.
