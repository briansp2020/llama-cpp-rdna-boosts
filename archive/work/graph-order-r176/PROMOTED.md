# PR #252 assessed — 0001 + 0002 folded into block 10 (r177), 0003 held off

**Status: PROMOTED (2 of 3 patches).**  Contributor PR [#252](https://github.com/stew675/llama-cpp-rdna-boosts/pull/252)
(`@briansp2020`), filed as `wip/graph-order-r176/` against r176 and applying with `git am` unchanged.  This
directory keeps the submission (all three patch files, `README.md` verbatim) as the record; the delivery
carries 0001 and 0002 inside **block 10**, and nothing here is applied by `scripts/apply-all.sh`.

| patch | verdict | why |
|---|---|---|
| `0001` V/K projections ahead of the Q projection (`GGML_CUDA_QK_ROTATE`) | **taken** | the whole win, on both RDNA generations; bit-identical |
| `0002` dense row concat on RDNA4 (`GGML_CUDA_MMVQ_ROW_CONCAT_RDNA4`) | **taken** | +2.6 % on RDNA4 on top of 0001; sound row-count argument, even-row-segment condition |
| `0003` gather small GDN projections into the row concat (`GGML_CUDA_MMVQ_CONCAT_REORDER`) | **held off** (`TODO` #119) | cannot fire on any GGUF we hold: it needs a quantized `<= 256`-row weight and skips `Q8_0`, and the GDN `ssm_alpha`/`ssm_beta` are F32 in the UD quants and Q8_0 in the Q8_0 quant.  Measured neutral to −0.7 % on the 35B on either arch |

## Measured (1 GPU, alternating rounds, `-sm layer -t 8`, or as noted)

| arm | gfx1201 (R9700) | gfx1100 (7900 XTX) |
|---|---|---|
| Qwen3-30B-A3B Q4_K_M tg128 | 152.0 -> **160.9 (+5.9 %)** | 194.3 -> **205.8 (+5.9 %)** |
| the same, d4096 | 131.2 -> 137.2 (+4.5 %) | 170.0 -> 180.2 (+6.0 %) |
| 30B, `GGML_CUDA_QK_ROTATE=0` | 152.2 (≡ base) | 194.3 (≡ base) |
| 30B, `GGML_CUDA_MMVQ_ROW_CONCAT_RDNA4=0` | 156.9 (so 0001 alone is +3.2 %) | inert by construction |
| 30B, 2-GPU `-sm tensor` tg128 | 141.2 -> **147.9 (+4.7 %)** | — |
| Qwen3.6-35B-A3B Q4_K_M tg128 | 102.5 -> 104.4 (+1.8 %) | 140.1 -> 141.1 (+0.7 %, noise) |
| dense Qwen3.8-27B UD-Q4_K_M tg128 | 32.62 -> 32.72 (flat) | — |

The submitter's claims (+5.9 % / +5.5 % on the R9700, +7.5 % on the 7900 XTX for the 30B) reproduce in
direction and, for `d0` on RDNA4, in magnitude.  Their 35B figures are much faster in absolute terms than
either of our boxes reads, which is a box/tuning difference, not a patch difference.

## Purity / gates (r177 release flow)

* Greedy text byte-identical to r176 on 30B-A3B (1 GPU and 2-GPU `-sm tensor`), 35B-A3B, dense 27B,
  Gemma4-31B and Flash-Next (`-ncmoe 48`, 2 GPU tensor).
* Prefill-logit KLD **0.000000 / 100.000 %** at the default ubatch and at `-ub 1` (the decode-width probe —
  0001's rotate runs at every ubatch size, so prefill graphs are reordered as well).
* MTP Protocol A, dense 27B UD-Q4_K_XL, 3 devices: layer plain 25.0 -> draft-mtp **51.7 t/s** (acceptance
  **0.621**), tensor plain 50.8 -> **101.2** (acceptance **0.608**), `plain == draft-mtp` text byte-identical
  in both split modes.
* rule-5 `llama-batched-bench -npp 512 -ntg 128 -npl 1,4,8` (27B, q8_0 KV): 31.91 / 102.11 / 156.90 ->
  31.94 / 102.20 / 157.18 (flat).
* `gate-per-block-build.sh --changed-from 19dec99a0 --fast` 2/2, `validate-set.sh` green (tree
  `b7adbd1259…`), no GPU faults.

## What the submitter should know

* 0003 needs the exact GGUF (name + sha256) and command line where it was seen to help; with a standard
  (non-UD) k-quant whose GDN small projections stay quantized it may well fire, but we cannot construct that
  case from the models we have, and it is the one patch of the three with a positional-safety constraint.
