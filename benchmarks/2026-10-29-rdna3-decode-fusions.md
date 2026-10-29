# PR #250 — the seven RDNA3 decode fusions (block 10, r175)

Source: contributor PR [stew675/llama-cpp-rdna-boosts#250](https://github.com/stew675/llama-cpp-rdna-boosts/pull/250)
(`@briansp2020`), "Seven decode fusions for RDNA3 (Qwen3-30B-A3B tg128 +20 % on 7900 XTX), on top of
#249".  The series was filed as `wip/decode-fusions-rdna3/` (7 `git format-patch` files + a README),
cut against the r174 release tree — it applies with `git am -3` unchanged (it does **not** need #249's
declined `0002`).

**Outcome:** taken in full except the submitter's `qwen3moe.cpp` graph reorder (see "What we changed"),
folded into **block 10** (the block that owns the fusion dispatcher, the rope matcher and `mmvq-moe`),
default-on with a `=0` switch each.  Released as `v15-6050b467c-r175`.

## Method

* Two boxes, both at ROCm 10.1.0 and the release build flags:
  * **gfx1100** (RX 7900 XTX, 1 GPU, 16-core 7950X) — the submitter's target arch, via `ssh fingon`;
  * **gfx1201** (Radeon AI PRO R9700, 1-3 GPUs, 9950X3D).
* Stock arm = the **r174 release tree** built in the same build dir as the candidate (same
  `GGML_CUDA_FA_QUANTS` set — a differing instantiation set changes greedy text on its own, r138).
* Candidate arm = the r174 tree + the 7 patches + our two adjustments (`git am -3`, or the release
  block-10 tree once folded).
* `llama-bench -fa 1 -r 5 -t 8`, 2-3 **alternating** rounds (never parallel).  Greedy text compared with
  `scripts/extract-generated.py` (never a raw `diff` of the log: the terminal's backspace corrections and
  the perf footer move).
* Per-patch contribution = the tg128 **loss when only that patch is switched off** from the full series.

## Throughput (1 GPU)

**gfx1100, Qwen3-30B-A3B Q4_K_M** (the submitter's headline model), 3 alternating rounds:

| arm | tg128 | tg128 @ d4096 | pp512 |
|---|---|---|---|
| stock r174 | 172.31 / 172.37 / 172.32 | 153.74 / 153.39 | 4881 / 4939 |
| **as submitted** (with the `qwen3moe` reorder) | 206.13 / 206.11 / 206.32 (**+19.6 %**) | 179.86 / 179.38 (**+17.0 %**) | 4897 / 4917 |
| **as shipped** (reorder dropped) | 192.78 / 193.43 / 193.61 (**+12.2 %**) | — | 4934 / 4931 |

**gfx1100, dense Qwen3.8-27B UD-Q4_K_M**: stock 42.74 / 42.85 -> 43.03-43.11 t/s (**+0.5 %**), pp512 flat.

**gfx1100, Qwen3.6-35B-A3B Q4_K_M**: stock 138.19 -> 140.97 t/s (**+2.0 %**, submitter claimed +2.5 %).

**gfx1201, 1 GPU** (the submitter could not test RDNA4; `0006`/`0009` are RDNA3_0-gated and inert there):

| model | stock | shipped | delta |
|---|---|---|---|
| Qwen3-30B-A3B Q4_K_M | 131.07 / 131.69 | 147.43 / 143.88 | **+10.9 %** |
| Qwen3.6-35B-A3B Q4_K_M | 102.05 / 102.08 | 104.79 / 104.47 | **+2.5 %** |
| Qwen3.8-27B UD-Q4_K_XL dense | 30.77 / 30.81 | 30.88 / 30.91 | +0.4 % |
| 27B, `pp8192` | 1315.8 | 1317.5 | flat (peak VRAM 20.17 GiB **both**) |

### Per-patch contribution (Qwen3-30B-A3B, 1 GPU)

Loss from the full series when one switch is turned off, gfx1100 / gfx1201 (`-` = the patch is inert on
that arch):

| switch | gfx1100 | gfx1201 |
|---|---|---|
| `GGML_CUDA_FA_COMBINE_Q8=0` | 2.5 % | 2.2 % |
| `GGML_CUDA_MOE_GLU_Q8=0` | 2.1 % | 1.0 % |
| `GGML_CUDA_ROPE_SET_ROWS_V=0` | 2.2 % | — |
| `GGML_CUDA_MOE_RPB_WIDE=0` (RDNA3_0) | 3.6 % | inert |
| `GGML_CUDA_MWR_FOLD=0` | 1.0 % | — |
| `GGML_CUDA_ROPE_QK_PAIR=0` | 3.7 % | 3.0 % |
| `GGML_CUDA_MMVQ_ROW_CONCAT=0` (RDNA3_0) | 2.8 % | inert |
| `LLAMA_QK_ADJACENT=0` (dropped; shown for the record) | 6.0 % | 3.3 % |

Every switch is live and every patch is individually positive.  The submitter's own table (relative to
r171 + #249) reads 3.1 / 2.7 / 2.9 / 5.3 / 1.2 / 7.2 / 4.2 — same shape, slightly larger magnitudes
because their baseline is r171 without r174's M-RoPE hoist.

## Purity and gates (all green)

* **Greedy text byte-identical** stock == candidate and `plain == draft-mtp`: 30B-A3B (gfx1100 and
  gfx1201), 35B-A3B, dense 27B; 1500-token arms; **1 GPU and 2-GPU `-sm tensor`**; 35B-A3B
  `-sm tensor -ncmoe 40` (host experts, expert cache).  An 8192-token single arm cross-checks the same.
* **Prefill-logit gate: PASS** — mean KLD **-0.000000**, same-top-p **100.000 %** (35B-A3B Q4_K_M,
  40 chunks `-c 512`, against the recorded r174 base; also the same before the padding fix and after it).
* **MTP Protocol A** (dense 27B UD-Q4_K_XL, 3 GPU, `-c 102400`, bf16 KV, `-n 3000`, prose prompt):
  layer **51.6 -> 51.6** t/s, tensor **100.5 -> 100.8** t/s; `n_accepted` 1951 (layer) / 1937 (tensor)
  **identical** in both arms, acceptance 0.51; draft >= plain (plain 25.0 / 50.9) as required.
* **Rule 5** `llama-batched-bench -npp 512 -ntg 128 -npl 1,4,8` (4B Q8_0): tg 479.8 / 1449.8 / 2018.0 ->
  480.0 / 1458.0 / 2015.5 — the verify band is unchanged.
* `test-backend-ops -o ROPE,MUL_MAT_ID`: OK in both arms (the `row_cat` template instance leaves the
  plain ksplit kernels untouched).
* `scripts/check-gpu-faults.sh`: clean on both boxes.

## What we changed relative to the PR

1. **The `src/models/qwen3moe.cpp` reorder is dropped** (`LLAMA_QK_ADJACENT` does not exist in the
   delivery).  It makes the Q and K norm+rope chains adjacent so `GGML_CUDA_ROPE_QK_PAIR` fires on
   qwen3moe; it is worth ~3.3 % of the 30B's 12-20 %, but it edits **upstream model code for every
   backend** to enable a CUDA fusion, for a retired arch family.  Without it Qwen3.6 still pairs and
   qwen3moe pairs rarely; the row concat is unaffected.
2. **The MoE GLU Q8_1 publish zeroes the padding tail** of each published row (the submitter flagged
   the gap): the arena is reused across graphs, so a reader that ever walks a padded row would find stale
   blocks.  `n_ff` is 768 on the 30B, i.e. 8 padding blocks per row are now written as the zeros
   `quantize_q8_1` writes.  Text and prefill KLD unchanged.
3. **The matcher order is kept as submitted** (row concat ahead of the dual-output fusion) after
   measuring that it matters only *with* the model reorder: with the reorder dropped, both orders are
   identical (30B: 192.78/193.43/193.61 vs 192.80/192.51/192.90; dense 27B: 43.11/43.03 vs 43.06/43.03).
   The FFN-pair-preemption question is recorded in `patches/README.md` (block 10) rather than reordered.

## Not covered (honest limits)

* `0006`/`0009` are RDNA3_0-only and the only gfx1100 box here has **one** card, so their multi-GPU
  behaviour is unverified; on RDNA4 they are inert and the 2-GPU arms are clean.  `-sm row` cannot even
  load a model on either box (pre-existing, both arms), so the submitter's "-sm row not guarded" caveat
  is untestable and moot here.
* The submitter's `0007` note about the reduction output overlapping the expert ids is implemented as an
  overlap refusal; it is code-review-validated (no config here produced the overlap).
