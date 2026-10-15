# rdna3-mmb-tile-spill: 128x256 MMB dense tile only for Q4_K on gfx1100

One format-patch on top of `v12-097cf33d9-r62` (tree `386a6a3d`; applied tree `0a43a66c`), for `git am`.
1 file (`ggml/src/ggml-cuda/mmb.cu`), +14 / -3.  It changes RDNA3_0 (gfx1100/1101/1102) only; RDNA3_5 and
RDNA4 take the same path as before.  Independent of our open #153 / #154.  (The binaries below were built
from the same code; only the numbers in the patch's comment were updated afterwards.)

**Hardware: RX 7900 XTX (gfx1100, RDNA3, 24 GB)**, Ryzen 7 9800X3D, Windows 11 + WSL2 (Ubuntu 24.04, kernel
6.18.54.1), ROCm 10.2 nightly `20261013-38325106112` (HIP 7.17.26405) in a container, GPU through `/dev/dxg`.
Not the gfx1201 box, and not native Linux: see Limitations.

## What it changes

On gfx1100 the MMB dense prefill GEMM's 128x256 tile does not fit in 256 VGPRs for most weight types and
spills to scratch.  Read from the code object's metadata (`llvm-readelf --notes`,
`.private_segment_fixed_size`) for `mmb_dense_kernel<128, 256, 64, 64, WTYPE>` on gfx1100:

| WTYPE | Q6_K | Q5_K | Q3_K | BF16 | Q4_1 | MXFP4 | Q5_0 | Q4_K, Q8_0, IQ4_NL, ... |
|---|---|---|---|---|---|---|---|---|
| scratch, bytes/thread | 148 | 104 | 56 | 48 | 24 | 20 | 16 | 0 |

The 128x128 tile (`mmb_dense_kernel<128, 128, 32, 64, WTYPE>`) uses no scratch for any of these.  The head-256
WMMA flash-attention prefill kernel also spills on gfx1100 (`flash_attn_ext_f16<256, 256, 32, 2>`: 1016
bytes/thread).

With both a spilling MMB kernel and the spilling FA kernel in one prefill, the queue aborts with
`HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION` on this setup.  We don't know the mechanism; the patch only
avoids the condition.  It adds one per-arch field to `mmb_arch_cfg`, `big_q4k_only`, set for RDNA3_0: there
the 128x256 tile is kept for Q4_K weights (which don't spill) and every other weight type takes the 128x128
tile.  `GGML_CUDA_MMB_TILE` still overrides, and the field is printed in the `GGML_CUDA_MMB_CFG=1` dump.

MMB is prefill-only (`mmb_min_t`, 512 tokens), so decode and verify batches are untouched, and the tile
geometry doesn't change the result (the existing S10 note in `mmb.cu`); the output hashes below agree.

## Tests

Qwen3.8-27B Q4_K_M (lmstudio-community), your README build
(`-DGGML_HIP=ON -DGGML_HIP_RCCL=1 -DGPU_TARGETS=gfx1100 -DCMAKE_BUILD_TYPE=Release`), no tuning env.
"off" is r62, "on" is r62 + this patch (two binaries: the old tile choice has no env toggle), alternating
off/on.

**Prefill**, `llama-bench -m <model> -ngl 99 -fa 1 -ctk q8_0 -ctv q8_0 -p 2048 -n 0 -d 0,16384 -r 2`:

| run | off | on: pp2048 @ d0 | on: pp2048 @ d16384 |
|---|---|---|---|
| 1 | aperture fault | 1259.1 | 998.8 |
| 2 | aperture fault | 1273.3 | 1014.8 |
| 3 | aperture fault | 1257.1 | 946.0 |
| 4 | aperture fault | 1263.0 | 952.9 |

r62 can't be timed here, so for speed: r62 with `GGML_CUDA_MMB_TILE=0` (128x128 for every type, no MMB
scratch) vs this patch, alternating, same command:

| run | `MMB_TILE=0`: d0 | d16384 | this patch: d0 | d16384 |
|---|---|---|---|---|
| 1 | 1172.6 | 952.1 | 1236.9 | 983.6 |
| 2 | 1187.2 | 955.5 | 1230.9 | 993.7 |
| 3 | 1184.0 | 951.2 | 1236.8 | 990.2 |
| mean | 1181.3 | 952.9 | 1234.9 (+4.5 %) | 989.2 (+3.8 %) |

No faults in either arm.  The difference is the Q4_K GEMMs keeping the 128x256 tile.

**Output**, 26k-token prompt (the first ~95 KB of `ggml-cuda/*.cu` plus "summarize"), then two 300-token
greedy generations on the cached prompt, MTP draft depth 3, fresh server per run:

```
llama-server -m <model> -c 40960 -np 1 -ngl 99 -ctk q8_0 -ctv q8_0 --spec-type draft-mtp --spec-draft-n-max 3
```

| run | off | on | off | on |
|---|---|---|---|---|
| decode t/s | 47.0 | 47.1 | 46.9 | 46.9 |
| text sha | `1a6343387a08` | `1a6343387a08` | `1a6343387a08` | `1a6343387a08` |

Same text in all four (checked: a file-by-file summary of the CUDA sources), equal to plain greedy decoding,
acceptance 0.638 in every run.  The server path faults less often than llama-bench: these two off runs got
through, while two other r62 server runs at the same depth faulted in prefill.

**Controls** (earlier, on r59 with the same ROCm and model; interleaved server soak at 26k depth, 8 rounds per
arm):

| arm | aperture faults |
|---|---|
| default | 3 / 8 |
| `GGML_CUDA_FA_WMMA_256=0` | 0 / 8 |
| `GGML_CUDA_MMB=0` | 0 / 8 |
| both | 0 / 8 |

- Either switch alone avoids it, at -13 % / -14 % prefill; this patch keeps both kernels.
- `HSA_NO_SCRATCH_RECLAIM=1` did not help (still faulted).
- Also tried making the FA kernel spill-free instead (Q from shared memory, `nbatch_fa` 32, K/V batches 32 on
  RDNA3_0): it reached 0 scratch but cost ~7 % prefill at 16k, and with the 128x256 MMB tile still spilling
  one of three runs hung.  Not pursued.
- Peak VRAM was 19.4 of 24 GB; no Windows TDR / WHEA events.

## Limitations

- WSL2 only.  Under WSL the HSA runtime handles scratch differently (`disable_scratch()` for the PM4
  emulation path), so this may not reproduce on native Linux gfx1100; not tested there.  The tile change
  itself is cheap either way.
- One model.  Other weight types are covered by the scratch table, not by runs.
- gfx1101/gfx1102 take the same rule by family; only gfx1100 was measured.
- Not tested: RDNA3_5 / RDNA4 (unchanged code path), multi-GPU.

No urgency on our side.  Happy to re-run anything on your config.
