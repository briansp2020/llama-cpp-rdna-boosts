# 2026-10-30 — PR #256 (GDN decode folds + shared-expert gate fold) and the 2-D copy guard

Assessment of [PR #256](https://github.com/stew675/llama-cpp-rdna-boosts/pull/256) (`@briansp2020`), four
patches cut against r176 that apply unchanged to r182.  Outcome: **all four folded into block 10**, released
as `v16-6050b467c-r183`.  The same release carries a fix for a load abort found while setting up the
Flash-Next arm (below); that abort is not the PR's (it reproduces on stock r182).

## Arms

One warm tree (`scripts/scratch-tree.sh`), four extractions of the same build directory; every comparison is
**on vs off inside one build** (each fold has its own kill switch), so the arm is its own control.
Boxes: gfx1201 (3x R9700, 1-GPU arms at `HIP_VISIBLE_DEVICES=0`, ROCm 10.1.0-gfx120X) and gfx1100
(1x RX 7900 XTX, ROCm 10.1.0-gfx1100, `-DAMDGPU_TARGETS=gfx1100`).

`llama-bench -fa 1 -t 8 -p 0 -n 128 -d 0,4096 -r 5`, two interleaved rounds.

## Throughput (all four on vs all four off)

| model / box | on | off | delta |
|---|---|---|---|
| gfx1100 35B-A3B Q4_K_M tg128 | 150.70 / 150.48 | 142.27 / 142.17 | **+5.9 %** |
| gfx1100 35B-A3B d4096 | 146.77 / 146.96 | 139.01 / 139.02 | **+5.6 %** |
| gfx1201 35B-A3B tg128 | 108.32 / 107.81 | 105.00 / 104.98 | **+3.2 / +2.7 %** |
| gfx1201 35B-A3B d4096 | 106.04 / 106.27 | 102.82 / 103.05 | **+3.1 %** |
| gfx1201 27B UD-Q4_K_XL tg128 | 31.00 / 31.07 | 30.49 / 30.48 | **+1.7 %** |
| gfx1201 Qwen3.5-4B Q8_0 tg128 | 103.57 / 104.19 | 104.08 / 104.16 | neutral |

Decomposition (gfx1201, 35B-A3B, one round per arm, all `llama-bench -r 5`):

| arm | tg128 | d4096 |
|---|---|---|
| all off | 104.98 | 102.93 |
| `0003` only | 105.34 (+0.3 %) | 103.36 (+0.4 %) |
| `+0001` | 107.39 (+2.3 %) | 105.16 (+2.2 %) |
| `+0002` | 108.19 (+3.1 %) | 106.20 (+3.2 %) |
| `+0004` | 108.43 (+3.3 %) | 106.39 (+3.4 %) |

`0004` alone, on the only model we hold whose gate weight is BF16/F32 (Flash-Next GSQ-IQ3_XXS, 3 GPU,
`-sm layer -ncmoe 48`): tg128 **45.63 / 45.67 vs 45.34 / 45.34 (+0.6 %)**.

## Kernel census / fold reachability

* `GGML_CUDA_CONV_DEBUG` on 35B-A3B 1 GPU: the direct conv fires at `T = 1` for **29 of 30** GDN layers
  (the 30th fails the q/k-norm match and keeps the stock path), `snap=1`, and the deferred-gather list grows
  by 2 per layer inside one graph evaluation (the GDN state gather and the conv-state gather).
* `GGML_CUDA_SHEXP_DEBUG` on Flash-Next: 10 folds per step at `ne0=2560`, `w=bf16`; inert on 35B-A3B, 4B,
  27B and gpt-oss (quantized gate weight).

## Purity

* Greedy text byte-identical on/off: 35B-A3B Q4_K_M (1 GPU), 4B Q8_0 (1 GPU), 2-GPU `-sm tensor -ncmoe 40`
  (35B-A3B), Flash-Next `-sm layer`.
* `gate-prefill-logits.sh --ab "<all four off>" --ub 1` (the decode width, where `0003` fires) on 27B Q8_0:
  **0.000000 KLD / 100.000 % same-top-p**.  Canonical gate at the default ubatch against the r182-recorded
  base: **0.000000 / 100.000 %**.
* MTP, 35B-A3B, `n_max 7`: draft/accept **362 / 203** on both arms, identical.
* MTP Protocol A (dense 27B UD-Q4_K_XL, 3 devices, `-sm layer`, `-c 102400 -fa 1`, `-n 3000`): plain **25.9**
  -> draft-mtp **66.1 t/s**, acceptance **0.85057**, and `plain == draft-mtp` **byte-identical** (1806 chars
  each).  The plain rate matches the r181/r182 reference (24.98), so the MTP path is unchanged while the
  snapshot fold runs on the verify band (`T = 2..8`).
* server `--parallel 2 --kv-unified`: clean (no assert from the gather fold's tripwire; a multi-sequence
  batch declines the deferral).

## The Flash-Next `-sm tensor` load abort (separate, fixed in the same release)

Stock r182 aborts before the first token on `Flash-Next GSQ-IQ3_XXS -sm tensor -ncmoe 48`:
`hipMemcpy2DAsync ... invalid argument` in `ggml_cuda_cpy`, device 1.  `-sm layer` and `GGML_CUDA_SLAB=0`
both load, so the trigger is the slab: the qwen4exp PLE conv copies a *column view of a mirrored model
weight* through upstream's 2-D memcpy fast path, and on the last device of the split that view sits at the
end of its slab region.  `ggml_cuda_cpy_as_memcpy_2d` now keeps the fast path only while the padded span
fits the tensor's own allocation (scalar kernels, bit-identical, are the fallback).  Verified on the release
build: the command loads and decodes, and its text matches the `GGML_CUDA_SLAB=0` arm.  Full diagnosis:
`archive/work/cpy-2d-memcpy-guard/README.md`.
