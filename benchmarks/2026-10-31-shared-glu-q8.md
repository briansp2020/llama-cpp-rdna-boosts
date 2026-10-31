# 2026-10-31 — the shared-expert lane's Q8_1 publish (PR #259, r188)

**Question:** does the shared-expert lane of the 1-token fused MoE gate/up+GLU kernel (`mul_mat_vec_q_moe`)
publishing its GLU output as Q8_1 let the shared down `MUL_MAT` skip its `quantize_q8_1`, and is that shift
free of any numerics change?

**Answer:** yes, and it is bit-identical.  Worth **+1.1 % tg128 / +0.8 % @4096** on gfx1201 and **+2.6 % /
+2.3 %** on gfx1100, exactly **one launch per layer** (40 on Qwen3.6-35B-A3B) — **but only on a quant whose
`ffn_down_shexp` is not Q8_0**, because the delivery's own `shexp_down_gate` fusion already consumes the
shared down there.  Folded into block 10 as `v16-6050b467c-r188`.

## Method

Two builds in **one** build dir (`~/llama-scratch/build-rocm`, same `GGML_CUDA_FA_QUANTS` set), switched with
`scripts/scratch-tree.sh` (`tip` then `arm 0001-*.patch`), binaries kept as `bin-r187` / `bin-pr259` and run
from those, so no cross-build-dir artefact can enter a text comparison.  `ROCBLAS_USE_HIPBLASLT=0` on every
timing (issue #67), one GPU job at a time, `llama-bench -fa 1 -p 0 -n 128 -d 0,4096 -r 10`, three interleaved
rounds.

Model: `/llm/models/Qwen3.6/35B-A3B/LMStudio-Q4_K_M` (`ffn_{up,gate}_shexp` **Q4_K**, `ffn_down_shexp`
**Q6_K**) — the file class the submitter measured.  The Unsloth **UD** files in the same tree keep all three
shared tensors at Q8_0 and behave differently (see "Reach").

## Launch census (`rocprofv3 --kernel-trace`, 40-token slope from `-n 10` vs `-n 50`)

| arm | gfx1201 R9700 | gfx1100 7900 XTX |
|---|---:|---:|
| r187 (`bin-r187`) | 748 / token | 688 / token |
| r187 + PR #259 | **708** (-40) | **648** (-40) |
| `GGML_CUDA_MOE_GLU_Q8=0` on r187 (routed publish off, gfx1100 only) | — | 724 (+36) |

One per layer (Qwen3.6-35B-A3B has 40).  The submitter's 761 -> 721 on an R9700 is the same -40 on a
different baseline.  The routed publish this one extends was measured on the same arm for reference
(`GGML_CUDA_MOE_GLU_Q8=0` on r187: 688 -> 724, i.e. +36/token — the shared lane's 40 covers every layer).

## Throughput

gfx1201 (1 GPU, 3 interleaved rounds, mean of 3):

| test | r187 | r188 | delta |
|---|---:|---:|---:|
| tg128 | 139.19 / 139.62 / 139.81 (**139.54**) | 141.05 / 141.18 / 141.08 (**141.10**) | **+1.12 %** |
| tg128 @ d4096 | 136.68 / 136.33 / 136.41 (**136.47**) | 137.34 / 137.60 / 137.65 (**137.53**) | **+0.78 %** |

Every patched run is above every base run.  The submitter's R9700 table said 141.61 -> 143.42 (+1.3 %), the
same shape on our file.

gfx1100 (1 GPU, 7900 XTX via `ssh fingon`, 2 clean rounds — round 3 was contaminated by a foreign CPU
benchmark on that host, load 36, and is dropped; both of its arms collapsed together):

| test | r187 | r188 | delta |
|---|---:|---:|---:|
| tg128 | 178.37 / 177.42 (**177.90**) | 182.36 / 182.76 (**182.56**) | **+2.62 %** |
| tg128 @ d4096 | 178.04 / 178.27 (**178.16**) | 182.33 / 182.23 (**182.28**) | **+2.31 %** |

The submitter reported +2.6 % / +2.0 % on their 7900 XTX — reproduced.

`llama-batched-bench -npp 512 -ntg 128 -npl 1,4,8` (gfx1201, 2 rounds): npl 1 tg 135.07/134.70 -> 136.61/136.22
(**+1.1 %**, the decode win), npl 4 and npl 8 flat within noise (a multi-token batch never publishes).

## Purity

* **`--ub 1` prefill-logit A/B** (`gate-prefill-logits.sh --ab "GGML_CUDA_MOE_SHARED_GLU_Q8=0" --ub 1`, the
  shape that reaches the publish): **mean KLD 0.000000 / same-top-p 100.000 %**, base PPL 6.2078.  The
  published blocks are `quantize_q8_1`'s arithmetic verbatim.
* **Default-ub prefill-logit** against a base recorded from `bin-r187`: **-0.000000 / 99.967 %** (the
  stored-logit harness floor — r187's own recording reads the same), i.e. prefill is untouched.
* **Greedy text, byte-identical** in every arm: 1 GPU `782 chars sha=92d0837f148d`; 2 GPU `-sm tensor` on
  (0,1) and (0,2) `793 / adf5c8f08334`; 3 GPU `-sm tensor` `794 / f03cb34fd6ad`; 3 GPU `-sm layer`
  `782 / 92d0837f148d`; `-ncmoe 20 -sm tensor` 1 GPU `786 / d4e150984a25`; gfx1100 1 GPU
  `831 / 56fb588a388f`.
* **MTP** (`prompts/prose-rdna-boosts.txt`, sha256 `fabdec65…`): dense 27B UD-Q4_K_XL at 3 devices, both split
  modes — text `a3ac07cc4b8a` (tensor) / `b6a520169bc3` (layer) on **both** builds, acceptance
  0.60816 / 0.62134 **identical to the last digit**, t/s identical (a verify batch is wider than one token, so
  nothing publishes).  A **firing** MoE arm (35B-A3B UD-Q4_K_M with `GGML_CUDA_DISABLE_SHEXP_DOWN_GATE=1`,
  3 GPU tensor, `-n 2000`): text `7061 chars sha=f80eab6cf390` and acceptance **0.72137** on both builds.
* `test-backend-ops -o MUL_MAT_ID` **931/931**; `scripts/check-gpu-faults.sh` clean.

## Reach (the finding the submission could not see)

The delivery already has a fusion that consumes the shared down: `ggml_cuda_shexp_down_gate_try_fuse`
(`shexp_down_gated_q8_0`, block 08's epilogue on the shared-expert chain) is armed only when
`ffn_down_shexp` is **Q8_0**.  Where it is armed the shared down `MUL_MAT` node does not exist, so there is no
`quantize_q8_1` for this patch to remove:

| file | `ffn_down_shexp` | launches/token r187 -> r188 | delta |
|---|---|---:|---:|
| `UD-Q4_K_M` | Q8_0 | 839 -> 839 | **0** (inert) |
| `UD-Q3_K_M`, `UD-Q5_K_M`, `Q8_0` | Q8_0 | (same class) | **0** (inert) |
| `LMStudio Q4_K_M` | Q6_K | 748 -> 708 | +1.1 % |
| `UD-IQ3_S` (gfx1100) | Q6_K | — (its shared-expert *pair* fusion is declined there, so the publish is not armed) | 0 |

So a claim about this patch must name the file: the win is real for conventional quants and for any model
whose shared tensors are not Q8_0, and it is nothing at all on an Unsloth UD quant.

## Verdict

Accepted.  Folded into block 10 (it owns `common.cuh`, `mmvq.cu` and `mmvq-moe.cu`); no code change was
needed, and the fold's tree is byte-identical to the measured arm (`61a8923…`).

Record: `archive/work/shared-glu-q8/`, `GREEDY-PURITY.md` §75, `patches/README.md` (block 10),
`ENVIRONMENT.md` §7.4, `WORKLOG.md` 261031.006.
