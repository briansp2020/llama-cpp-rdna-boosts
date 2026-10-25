# 2026-10-25 — the RDNA4 band's 64-column block: width purity, its own geometry (r134, issue #224)

Box: gfx1201 (R9700), ROCm 10.1.0, `~/bin/build-llama-rocm` on the release tree.  Every fresh-process text
comparison ran with `ROCBLAS_USE_HIPBLASLT=0`; every arm is sequential (no parallel benches).

Models: `/llm/models/Qwen3.8/27B/Q4_K_XL/Qwen3.8-27B-UD-Q4_K_XL.gguf` (dense, head 256, 24 query / 4 KV heads =
GQA 6) + `Qwen3.8-27B-DFlash2-Q4_K_M.gguf`; `/llm/models/Qwen3.6/35B-A3B/Q4_K_M/...` and
`/llm/models/Qwen3.8/Flash-Next/IQ4_NL/...` + its shared MTP head for the prefill/MoE arms.

Prompt: `prompts/prose-rdna-boosts.txt` (sha256 `fabdec65…`, 5246 tokens) unless stated.

## The defect

`r126` deleted the `ncols == 64` RDNA4 band row (it was also the prefill instantiation) and left query widths
5..8 on prefill's 8-warp table row `(256, 2, 64, 128, 128, 64, 1, true)`.  A band block splits each Q column's
KV rows `np = nwarps*cols_per_warp/ncols` ways, and `cols_per_warp` is 16 for the RDNA4 256-dim WMMA tiles:
the 32-column block is `8*16/32 = 4`, the table-row 64-column block is `8*16/64 = 2`.  Widths 5..8 therefore
stopped reducing like widths 1..4.

Greedy text, 512 tokens, `-c 12288`, `-fa on -ctk/-ctv q8_0`, 1 GPU, `--temp 0 --seed 42`:

| arm | chars | sha (first 12) |
|---|---:|---|
| `--spec-type none` (plain, width 1 only) | 1506 | `7181badffbcc` |
| `--spec-type draft-dflash --spec-draft-n-max 5` (r133) | 1619 | `7a3819b47df4` |
| the same with `GGML_HIP_FA_BAND_WIDE=0` | 1506 | `7181badffbcc` |

That is the reporter's reproduction on the maintainer's box: with the draft the text diverges from plain
decode, and the two-tile launch (which is width-pure) restores it.  The first difference is in the opening
sentence at 8K on this box, i.e. the near-tie the reporter hit at token 325 is not required to see it.

## The guard was blind, not absent

`tests/test-logits-width-probe` (the delivery's W=1..8 logits probe) reported `width_purity=PASS` on every
r126..r133 arm.  It could not do otherwise: the wide launch's gate is `K->ne[1] >= 4096`, `K->ne[1]` is the
context's **used** length, and the probe prefilled `P=256` tokens into a fixed `n_ctx = 4096`.  Its `n_ctx` is
now `P + W + 8` and a purity run must pass `P >= 4096`:

| build | `KV=q8_0`, `P=256` | `KV=q8_0`, `P=4096` | `KV=f16`, `P=4096` |
|---|---|---|---|
| r126 geometry (8-warp table row for widths 5..8) | PASS (blind) | **FAIL**, worst maxdiff **0.199764**, row-0 hash splits between W=4 and W=5 | **FAIL**, 0.206944 |
| r134 (band-only 16-warp row) | PASS | **PASS**, worst maxdiff **0** | **PASS**, 0 |

`KV=bf16` at `P=4096` passes with maxdiff 0 on r134 (W=8 hash `0f2bd74ffe4268bd`; q8_0 `6c3835e0a9eacccb`,
f16 `1fbee7ab473371c0`).  The `static_assert` added next to the band rows (same `np`, `nbatch_fa/K2/V2`,
`nstages_target`, `Q_in_reg`) refuses to compile a build with the r126 row re-selected, so the config half of
this regression is now a build error.

## Width purity after the fix (greedy text, plain vs DFlash2 `n_max` 3/5/7)

Nine arms, `-n 512`, `-c 12288`: `KV` f16 / bf16 / q8_0 x `n_max` 3, 5, 7 — **every arm identical to plain
decode** (f16 `2416d85188b1`, bf16 `bb0612faf7f5`, q8_0 `7181badffbcc`; `n_max 3` never reaches the 64-column
block and is the control).  At long context (2 GPUs, `-sm layer`, `-c 131072`, 200 tokens) plain and draft
also agree, and so do the wide and two-tile launches (`664 chars, 63b6b7cf6fdf` both).

## Cost of the pure geometry

End to end, `GGML_HIP_FA_BAND_WIDE=1` (r134 default) vs `=0` (two-tile), interleaved repeats:

| arm | wide | two-tile |
|---|---|---|
| 27B + DFlash2 `n_max 5`, 1 GPU, `-c 12288`, 512 tok (gen t/s) | 75.2 / 74.6 | 75.0 / 74.9 |
| the same at `-c 131072`, 2 GPU `-sm layer`, 200 tok (gen t/s) | 61.0 / 60.6 | 60.9 / 60.8 |
| Flash-Next IQ4_NL + shared MTP, 3 GPU `-sm tensor`, `-ncmoe 48`, `-c 32768`, `n_max 7` (W=8 every step) | 164.5 / 166.6 | 166.3 / 164.9 |

Text is byte-identical inside every pair (`80309046174e` on the Flash-Next arm).  Kernel level
(`test-backend-ops perf`, head 256 / GQA 6 / q8_0 / `kv 16384`, us per call): `n_q 5` 163.3 (wide) vs 167.3
(two-tile); `n_q 8` 174.0 vs 170.0.  So the single pass is worth ~+-2 % at the kernel and nothing measurable
end to end on this box — the reason to keep it is the purity-compatible geometry (and the reporter's
2.2-2.5 % at 110K on their box), not a throughput claim.

## Prefill is untouched (r126's win stays)

`llama-bench`, 35B-A3B UD Q4_K_M, 1 GPU, `-p 512 -n 0 -fa 1 -ctk/-ctv q8_0`: **d0 4764 t/s**, **d98304 1641
t/s** (r126 measured 4624 / 1632).  The prefill instantiation is still `(256, 2, 64, 128, 128, 64, 1, true)`;
the band's row lives in its own case/kernel translation unit.

## Gates run

`test-backend-ops -o FLASH_ATTN_EXT` 6414/6414; `test-logits-width-probe` as above; the nine-arm text sweep;
the long-context and Flash-Next text/throughput arms; `llama-bench` prefill; `scripts/validate-set.sh` green
(strict 14/14 `git am`, tree == `release.json.tree`); `scripts/gate-per-block-build.sh --changed-from
<r133 tip>` 4/4 (blocks 05, 06, 10, 13).  Prefill-logit gate not run: no prefill kernel, dispatch row or
default changed (the prefill config row is byte-identical and its throughput is unchanged).
