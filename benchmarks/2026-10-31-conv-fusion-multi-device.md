# The GDN/PLE conv fusion on multiple devices — r186 (3x R9700, gfx1201)

**Question.**  `gdn_conv_enabled()` / `ple_conv_enabled()` gated the depthwise conv1d fusion to
single-device graphs since 2026-09-23 (r13), when the direct kernel produced a different greedy
continuation from its reference under a 3x R9700 `-sm tensor` split
(`archive/work/closing-the-gap/2026-09-23-gfx1201-conv-fusion-tensor-split.md`).  Never root-caused.
Is it still there?

**Answer: no.**  Removing the gate is bit-identical everywhere we can measure, and worth a large
decode win on the recommended multi-GPU server shape.

## Purity (all arms: fused default == `GGML_CUDA_DISABLE_CONV_FUSION=1`)

`test-logits-width-probe MODEL prompts/prose-rdna-boosts.txt 4096 512`, `W8`/row-0 logit hashes.
The direct conv was confirmed to run in the fused arm (`GGML_CUDA_CONV_DEBUG=1` →
`GDN_CONV_DIRECT C=2560` for 3 devices, `C=5120` for 2, with `snap=1`), and to not run at all in
the reference arm.

| model | devices / split | `W8` / row-0 |
|---|---|---|
| 27B IQ3_S | 1 / layer | `8bb890b8a89303c8` / `9bbb4af6819e78b7` |
| 27B IQ3_S | 2 (`0,2`) / tensor | `8c0fde7e2f2edc07` / `0decc6329f229ec2` |
| 27B IQ3_S | 3 / tensor | `c7358e61f003f775` / `c86873f3774ada7d` (3/3 runs) |
| 27B IQ3_S | 3 / layer | `8bb890b8a89303c8` / `9bbb4af6819e78b7` |
| 35B-A3B Q3_K_M | 3 / tensor | `930668430bf6c432` / `8bc2462726b575a1` |
| 35B-A3B Q3_K_M | 3 / layer | `b2ba7832a589aa4b` / `98d90306b32cbdc5` |
| 35B-A3B Q4_K_M `-ncmoe 40` | 3 / tensor | `36f526ba4f3b899c` / `6eaf29f6f3c53d4a` |
| … same `+ CACHE=1` (arena) | 3 / tensor | same |

`width_purity=PASS` (worst maxdiff 0) in every arm.  Re-checked with `ROCBLAS_USE_HIPBLASLT=0`: same
result (that switch matters for the MoE `-ncmoe` shape's absolute hash — it moves between processes
without it — but fused and reference agree in both cases).

Greedy text (prose, `-n 200`, seed 42, temp 0), fused == reference: 4B Q8_0 1 GPU and 3 GPU tensor
`ae0ca6cd1b53`; 27B Q4_K_XL 3 GPU tensor `e4eb3bf92d94`, 3 GPU layer `b934bb22a82d`; 35B-A3B Q4_K_M
3 GPU tensor `740847e57559`, `-ncmoe 40` `b557442b27d1`.

MTP Protocol A (27B Q4_K_XL, `draft-mtp n_max 7`, bf16 KV, `-n 3000`): 3 GPU tensor text
`0d8932debda4` acceptance **0.63244** (2445/3866); 3 GPU layer text `735f237bdfcd` acceptance
**0.58861** (1850/3143) — identical text *and* acceptance on both arms.

DFlash2 draft (35B-A3B Q4_K_M + `draft-dflash n_max 5`, 3 GPU tensor, `-n 2000`): text
`d5b6f7cb2b73`, acceptance **0.52806** (1449/2744), identical.

Prefill-logit gate, `--sm tensor --device 0,1,2 --ab "GGML_CUDA_CONV_FUSION_MULTI=1"` (pre-change
build, where the fused arm was the opt-in): mean KLD **0.000000**, same-top-p **99.990 %**.

## Throughput (3 GPU `-sm tensor`, `llama-bench -r 5`, three interleaved rounds, `ROCBLAS_USE_HIPBLASLT=0`)

| arm | reference | fused | Δ |
|---|---|---|---|
| 27B IQ3_S tg128 | 58.52 / 58.54 / 58.72 (mean 58.59) | 61.85 / 61.90 / 62.01 (mean 61.92) | **+5.7 %** |
| 27B IQ3_S pp2048 | 1938.9 / 1937.6 / 1938.7 | 1969.1 / 1956.5 / 1952.3 | **+1.1 %** |
| 35B-A3B Q4_K_M tg128 | 118.72 / 118.57 / 118.28 (mean 118.52) | 122.05 / 122.16 / 122.19 (mean 122.13) | **+3.0 %** |
| 35B-A3B Q4_K_M pp2048 | 5163.6 / 5175.6 / 5196.6 (median 5175.6) | 5075.6 / 5175.1 / 5204.1 (median 5175.1) | flat (median −0.0 %, one low first round) |

Rule-5 `llama-batched-bench -npp 512 -ntg 128 -npl 1,4,8` (total t/s, 3 rounds): B=1 250.45 vs
249.46 (**+0.4 %**), B=4 673.89 vs 675.45 (−0.2 %), B=8 929.98 vs 931.02 (−0.1 %) — flat.

## Op oracles / build

`test-backend-ops` SSM_CONV 60/60, CONCAT 183/183, CPY 251/251, 3/3 backends;
`gate-per-block-build.sh --changed-from 5639690d2 --fast` 2/2; `validate-set.sh` green;
`check-gpu-faults.sh` clean.

## Caveat

The 2026-09-23 divergence was never root-caused, so this is an **empirical** result, not a proof.
The most likely fix is r183's rework of exactly the suspect area (the snapshot write moved into the
conv kernel; the conv-state gather was deferred into it), but nothing establishes that.  Configs
outside the matrix above are unproven; release gate 14 therefore carries a multi-device arm, and a
future divergence should be gated on the config that shows it, not on a device count.
