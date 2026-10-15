# rdna3-mmvq-vecdot-alu: fewer instructions in the Q4_K / Q6_K mmvq dot products (output unchanged)

Two format-patches on top of `v12-097cf33d9-r62` (tree `386a6a3d`; applied tree `b47d163b`), for `git am`.
1 file (`ggml/src/ggml-cuda/vecdotq.cuh`), +25 / -17.  They don't depend on our open #155 / #156 and apply to
plain r62; the measurements below were taken on r62 + #155 + #156 (see Tests).

**Hardware: RX 7900 XTX (gfx1100, RDNA3, 24 GB)**, Ryzen 7 9800X3D, Windows 11 + WSL2 (Ubuntu 24.04),
ROCm 10.2 nightly `20261013-38325106112` (HIP 7.17.26405) in a container.  Also run on one **R9700 (gfx1201)**:
see "Also on gfx1201" below.

## Why

On gfx1100 the verify-width mmvq kernels (2..8 columns, the MTP/DFlash verify step) are instruction-bound
rather than bandwidth-bound.  Q4_K at 4 columns, 4 rows per block: 566 instructions per loop iteration for 1152
bytes of weights, about 650 GB/s in a model trace against about 800 GB/s for 1-column decode.  Most of what isn't
the dot products themselves is integer unpacking that doesn't depend on the column, so both patches only remove
instructions without changing a result, as far as we could check (see the contraction note under 0001).

## 0001: Q4_K (`vec_dot_q4_K_q8_1`, VDR 2)

- **Mins term.**  `sumf_m += d8 * float(dot2 * m)` becomes `sumf_m += (d8 * float(dot2)) * float(m)`.  `d8` (a
  half, 11 significant bits) times `|dot2| <= 1024` (11 bits) is exact in fp32, so both forms multiply the same
  exact value.  The result is identical as long as the compiler treats both alike: contracted into one fma
  (hipcc does, on gfx1100 and gfx1201, and the hashes below agree) or both left as a multiply and an add.  The
  first factor depends only on the q8_1 column and the second only on the weight row, so in a multi-row,
  multi-column block the compiler computes each once instead of an integer multiply, a convert and an fma per
  (row, column).
- **Scales / mins.**  The two 6-bit scales and two mins of the lane's sub-block pair are packed into one word
  (bytes sc0, sc1, m0, m1), built 32 bits at a time for both the `j < 2` and `j >= 2` forms and then selected
  per lane, instead of two 16-bit `aux` halves each built both ways.  The mins come out of it straight as floats
  (`v_cvt_f32_ubyte`).

This is shared code (every target built from `ggml-cuda`: CUDA, HIP, MUSA; every caller of `vec_dot_q4_K_q8_1`).  `vec_dot_q4_K_q8_1_impl_vmmq` had
this one caller; its `sc`/`m` pointer arguments become the packed word.

## 0002: Q6_K (`vec_dot_q6_K_q8_1_impl_mmvq_vdr2`, HIP only)

Each weight byte is rebuilt as `(vil | 16*h) - 32` with a shift, a shift/mask and `__vsubss4`, which has no
packed-byte instruction on AMD (about four 16-bit ops per word on gfx1100).  `16*h - 32` is one of 0xE0, 0xF0,
0x00, 0x10, all with a zero low nibble, so `vi = vil | table[h]`: one `v_perm_b32` lookup of the four 2-bit `h`
values and an OR (the compiler fuses it into `v_and_or_b32`).  Same bytes; the values (-32..31) never saturate.
Under `GGML_USE_HIP`; CUDA keeps `__vsubss4`.  The VDR-2 function is only reached on RDNA4 / RDNA3_0 (gfx12xx /
gfx1100), where it is the Q6_K mmvq path for dense and MoE; every other target uses VDR 1 (`mmvq-vecdot.cuh`).

## Instruction counts (gfx1100 hot loop of `mul_mat_vec_q_ksplit`, from the disassembly)

| kernel | before | after |
|---|---|---|
| Q4_K, 4 columns (4 rows/block) | 566 | 475 |
| Q4_K, 1 column | 174 | 169 |
| Q6_K, 4 columns | 213 | 160 |
| Q6_K, 1 column | 129 | 86 |

Kernel time in a model trace (rocprofv3, the 400-token MTP prose request below; weights stream from DRAM),
average per call:

| mmvq group | before | after | change |
|---|---|---|---|
| Q4_K ffn_gate / ffn_up, 4 columns | 77.4 us | 74.9 us | -3.2 % |
| Q4_K ffn_down / ssm_out / attn_output, 4 columns | 46.6 us | 43.9 us | -5.8 % |
| Q6_K ffn_down, 4 columns | 100.5 us | 93.2 us | -7.3 % |
| Q6_K attn_qkv, 4 columns | 72.0 us | 64.9 us | -9.9 % |
| Q6_K output, 4 columns | 1642 us | 1490 us | -9.3 % |
| Q6_K output, 1 column (MTP drafts) | 1212 us | 1129 us | -6.8 % |
| all mmvq in the request | 4091 ms | 3875 ms | -5.3 % |

## Tests

Qwen3.8-27B Q4_K_M (lmstudio-community; Q4_K and Q6_K weights), your README build
(`-DGGML_HIP=ON -DGGML_HIP_RCCL=1 -DGPU_TARGETS=gfx1100 -DCMAKE_BUILD_TYPE=Release`), no tuning env.  "off" is
r62 + #155 + #156, "on" is the same plus these two patches (two binaries; nothing to toggle).  Fresh server per
run, off/on/off/on, greedy, one slot.

```
llama-server -m <model> -c 16384 -np 1 -ngl 99 -ctk q8_0 -ctv q8_0 --spec-type draft-mtp --spec-draft-n-max 3
```

Short prompts, 400 generated tokens, 3 requests per prompt with the first discarded; 26k-token prompt (the first
~95 KB of `ggml-cuda/*.cu` plus "summarize", `-c 40960`), two 300-token generations on the cached prompt.  Each
cell is the two measured requests (t/s):

| | off | on | off | on | mean change |
|---|---|---|---|---|---|
| MTP, prose (hash table explanation) | 64.82 / 65.10 | 68.03 / 68.48 | 64.66 / 64.93 | 68.07 / 68.61 | 64.88 -> 68.30, +5.3 % |
| MTP, code (Python LRU cache) | 76.29 / 76.45 | 81.09 / 80.73 | 76.70 / 76.84 | 80.90 / 80.77 | 76.57 -> 80.87, +5.6 % |
| MTP, 26k prompt | 50.66 / 50.65 | 53.16 / 54.49 | 52.10 / 52.15 | 53.44 / 53.50 | 51.39 -> 53.65, +4.4 % |
| plain decode, prose | 37.34 / 37.22 | 37.51 / 37.40 | 37.24 / 37.30 | 37.15 / 37.15 | within noise |

Every "on" request is faster than every "off" request in the three MTP rows.  Split by patch (an earlier
off/on/off/on pair on the same tree): 0001 alone +3.5 % / +3.4 % / +2.6 %, then 0002 on top +1.4 % / +1.9 % /
+2.0 % (prose / code / 26k).

**Correctness.**  Output identical with them on and off and with and without MTP: prose `75cbd6a6b11c`, code
`47114faa2042`, 26k `1a6343387a08` in every run (decoded and checked: a hash-table explanation, a Python
`LRUCache` class, a file-by-file summary of the CUDA sources); draft acceptance unchanged (0.666 / 0.846 /
0.638).  `test-backend-ops`: MUL_MAT q4_K 120/120, MUL_MAT_ID q4_K 84/84, MUL_MAT q6_K 62/62, MUL_MAT_ID q6_K
3/3 (only three cases, so the model hashes are the stronger check for Q6_K).

## Also on gfx1201

One R9700, same model (`-c 32768 -ngl 99`, no tuning env), plain r62 vs r62 + these two patches (applied tree
`b47d163b`), your README build with `-DGPU_TARGETS=gfx1201`:

- Output: greedy 400-token hash `9ed70fb699ad` (decoded: real text) and a 12-short-prompt combined hash
  `34008188ba1d`, identical for both builds, with and without MTP (`--spec-type draft-mtp --spec-draft-n-max 3
  --spec-draft-p-min 0`); draft acceptance 261/413 in every MTP run; no faults.
- `test-backend-ops`: the same four counts as above, all passing on both builds.
- MTP decode, fresh servers off/on/off/on, 400-token greedy, first / second request per server:

  | | off | on | off | on |
  |---|---|---|---|---|
  | t/s | 62.8 / 64.5 | 66.1 / 66.0 | 64.7 / 64.6 | 66.1 / 66.0 |

  Means 64.15 -> 66.05 (+3.0 %); second requests only, +2.2..2.3 % (the first off request, 62.8, looks like a
  warm-up).  Smaller than on gfx1100, possibly because of RDNA4's different rows-per-block table (not checked).  Without MTP
  only one run per build (30.0 / 31.6 vs 31.8 / 31.8 t/s), so treat that as unconfirmed.
- Hot-loop instructions (`mmvq.cu` compiled with the build's flags, `-x hip --cuda-device-only -S`, largest
  loop of `mul_mat_vec_q_ksplit<type, ncols, false, false, false, 0, false>`): Q4_K 4 columns 567 -> 485,
  1 column 179 -> 174; Q6_K 4 columns 654 -> 478, 1 column 209 -> 167.

**A note on `test-backend-ops perf`.**  Its Q4_K / Q6_K cases at these sizes (up to ~48 MB of weights) stay in the
7900 XTX's 96 MB Infinity Cache across runs, so it overstates any ALU saving: it shows +5..8 % (Q4_K) and
+27..34 % (Q6_K) at 1 column, where the model's 1-column decode, streaming 16 GB from DRAM, is bandwidth-bound and
doesn't move.  At 4 columns it shows +11..12 % (Q4_K) and +17 % (Q6_K).

## Limitations

- **CUDA not checked.**  0001 is shared code; we didn't compile or run a CUDA build, so its bit-identity there
  rests on nvcc treating both forms of the mins term alike (see 0001).  Happy to check it if useful.
- One model; Q4_K / Q6_K only.  The VDR-4 Q4_K variant (MoE) and Q3_K's `__vsubss4` have the same patterns and
  are left alone.
- RDNA3_5 not run (0001 applies there; 0002 doesn't reach it).  The gfx1100 numbers are under WSL2.

No urgency on our side.  Happy to re-run anything on your config.
