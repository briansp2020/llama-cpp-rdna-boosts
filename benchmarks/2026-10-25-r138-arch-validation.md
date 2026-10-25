# r138 arch validation — gfx1100 and gfx1151 (2026-10-25)

Closes `TODO.md` #105: the `v15-6050b467c-r138` re-base was gated on gfx1201 only.  This is the same set run
on the other two validated arches, from a **fresh apply of the released patch set** on each box (not from the
fork branch), so it also independently re-verifies `scripts/apply-all.sh` + `release.json` on both.

| | fingon | halo |
|---|---|---|
| GPU | RX 7900 XTX (gfx1100, 24.5 GiB) + gfx1036 iGPU | Radeon 8060S (gfx1151 APU), 123 GiB unified |
| ROCm | `/opt/rocm-10.1.0-gfx1100` | `/opt/rocm-10.1.0-gfx1151` |
| apply at `6050b467c` | strict 15/15, tree **`35090ba36c`** == `release.json.tree` | same |
| r138 build (`build-r138`) | 7m08s, 0 errors | 8m59s, 0 errors |
| r137 reference (`build-r137`, from `e8bd52069`) | 2m45s, 0 errors | 2m53s, 0 errors |

Both boxes were reset (`git checkout 6050b467c`, local `rdna-boosts` deleted, `apply-all.sh` re-run) and both
build dirs were configured by that box's own `~/bin/build-llama-rocm` flags, so each A/B pair is
configuration-identical.  `build-rocm` was left alone on both (on halo it is the live server build).
`HIP_VISIBLE_DEVICES=0` everywhere — the iGPU has no kernels in these builds and aborts `test-backend-ops`.

## Op tests (the arch oracle, r138)

| op | gfx1100 | gfx1151 | gfx1201 |
|---|---|---|---|
| `FLASH_ATTN_EXT` | 6450/6450 | 6450/6450 | 6450/6450 |
| `GATED_DELTA_NET` | 55/55 | 55/55 | 55/55 |
| `FLASH_ATTN_QSA` | 23/23 | 26/26 | 26/26 |
| `MUL_MAT` | 2445/2445 | 2445/2445 | 2445/2445 |
| `MUL_MAT_ID` | 931/931 | 931/931 | 931/931 |
| `TOP_K_MASK` | 28/28 | 28/28 | 28/28 |
| `HC_MIX` | 30/30 | 30/30 | 30/30 |

`FLASH_ATTN_QSA` has 23 cases on gfx1100 vs 26 on the other two (arch-gated case list); all pass, well above
the 18/18 floor.

## `test-llama-archs` (synthetic-tensor graph coverage: HC / QSA / MoE / Meta)

| arch case | gfx1100 | gfx1151 |
|---|---|---|
| `qwen4exp` | all 3 passed | all 3 passed |
| `deepseek4` | all 3 passed | all 3 passed |
| `glm5-next` | all 2 passed | all 2 passed |

This is what covers the r138 **#31028 one-GEMM HC** adoption on these arches (it exercises the HC_MIX /
HC_COMBINE graph and the Meta/tensor-split path against a CPU reference: NMSE 9.08e-14 on gfx1151).

## Throughput A/B, r137 vs r138

**gfx1100** — 27B UD-Q4_K_M, 1 GPU, `-c 8192`, `-npp 512 -ntg 128` (no `-sm tensor`: one card):

| npl | r138 pp | r137 pp | r138 tg | r137 tg |
|---|---|---|---|---|
| 1 | 1143.18 | 1145.63 | 42.55 | 42.52 |
| 4 | 1144.98 | 1143.13 | 135.23 | 135.32 |
| 8 | 1147.21 | 1145.34 | 178.01 | 177.77 |

`llama-bench -p 2048 -n 128`: pp2048 **1143.38** vs 1141.83 (+0.1 %), tg128 **42.80** vs 42.70 (+0.2 %).
**Parity, no regression.**

**gfx1151** — 4B Q8_0, 1 GPU (APU), same shapes:

| npl | r138 pp | r137 pp | r138 tg | r137 tg |
|---|---|---|---|---|
| 1 | 3167.20 | 3136.16 | 43.62 | 43.62 |
| 4 | 3211.65 | 3176.79 | 144.96 | 145.05 |
| 8 | 3170.03 | 3187.16 | 241.84 | 242.31 |

`llama-bench -p 2048 -n 128`: pp2048 **3254.22** vs 3252.73 (+0.05 %), tg128 **43.96** vs 43.98.
**Parity, no regression.**

## Block 03 (chunked GDN) on gfx1151 — the arch where both kernels are eligible

Ours (default) vs upstream's MMA arm (`GGML_CUDA_GDN_CHUNKED=0`), qwen35 4B Q8_0, `pp2048`, interleaved:

| rep | ours | upstream MMA |
|---|---|---|
| 1 | **3274.02** | 3206.63 |
| 2 | 3162.72 | 3165.44 |
| 3 | **3297.87** | 3191.81 |

Median **+2.6 %** for ours (the r127 campaign measured +3.2 % on the same box).  The APU's clocks drift inside a
run block, so the interleaved pairs are the evidence, not the spread.

**Prefill-logit A/B** (qwen35 9B Q8_0, ctx 512 × 40, base arm = `GGML_CUDA_GDN_CHUNKED=0`, r138 binary):
**mean KLD 0.000056 / same-top-p 99.657 %** — PASS (limits 0.005 / 98 %).  So the delivery's chunked kernel
still agrees with upstream's MMA kernel at the logits level on RDNA3.5.  The base file
(`~/.cache/rdna-boosts/gates/prefill-logit-gfx1151-Qwen3.5-9B-Q8_0.kld`) was re-recorded by `--ab` at the r138
build, as that mode requires.

## Not covered, and why

* **Flash-Next `plain == draft-mtp` on gfx1151** — that box carries only `Q4_K_M` (130 GiB) and `IQ4_XS`
  (96 GiB) Flash-Next quants against 113 GiB available RAM; loading it for a byte-identical-text check was not
  worth the wall time, especially as the change in question (#31028) is arch-independent graph code.  Covered
  instead by `test-llama-archs -a qwen4exp` on both arches (which exercises the HC graph) and by the
  byte-identical `plain == draft-mtp` arm on gfx1201 and on Flash-Next there.
* **NVIDIA** — out of scope by policy; upstream's `has_fusion` arm is its own code there.
* Multi-GPU on these boxes (one GPU each) — the tensor-split arms are gfx1201-only.

## Gate tooling fix found here

`scripts/gate-prefill-logits.sh` labelled a run with the **enclosing checkout's HEAD**, not the binary, so a
halo run of the r138 binary was labelled `e8bd52069` (the r137 chain, still checked out there from the baseline
build).  It now asks the binary (`llama-perplexity --version`, which writes to stderr) for its compiled-in
commit and only falls back to the checkout HEAD.  Re-verified on this box: the same run now reports
`build=6d3f1983d` (the r138 binary) and still passes at KLD 0.000531 / 98.863 %.
