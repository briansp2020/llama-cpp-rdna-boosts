# 2026-10-30 — PR #253 (`router-topk-fold`) on r181: the 1-token router fold ships as r182

Contributor `@briansp2020`, filed as `wip/router-topk-fold/` against r176, applying with `git am` unchanged to
r181.  Folded into **block 10** (the fusion dispatcher) as **r182**.  Record:
[`archive/work/router-topk-fold/`](../archive/work/router-topk-fold/PROMOTED.md).

## What was measured, and how

`GGML_CUDA_ROUTER_TOPK` is its own A/B arm (`=0` restores the two-launch path), so every arm below is one
build with the switch flipped.  `llama-bench -r 5`, two interleaved rounds, 1 GPU, `-fa 1 -t 8`, d0 and d4096.

### gfx1100 (RX 7900 XTX, `fingon`) — 24 GB, model fully on device

| model | on (fold) | off | delta |
|---|---|---|---|
| Qwen3-30B-A3B Q4_K_M (128 experts) tg128 | 212.17 / 211.92 | 205.07 / 206.49 | **+3.1 %** |
| the same, d4096 | 183.08 / 183.17 | 180.19 / 180.11 | **+1.7 %** |
| Qwen3.6-35B-A3B UD-Q4_K_M (256 experts) tg128 | 142.77 / 142.79 | 140.97 / 141.15 | **+1.2 %** |
| the same, d4096 | 139.04 / 138.97 | 137.53 / 137.56 | **+1.1 %** |
| 35B-A3B `--spec-type ngram-mod`, `-n 400` (llama-cli) | 137.1 / 135.6 | 136.2 / 134.5 | flat |

### gfx1201 (R9700, this box)

| model | on | off | delta |
|---|---|---|---|
| Qwen3-30B-A3B Q4_K_M tg128 (3 interleaved rounds) | 158.0 / 161.7 / 160.8 | 159.1 / 158.5 / 154.7 | **+1.7 %** mean (per-run sd ±4.4 %) |
| Qwen3.6-35B-A3B Q4_K_M tg128 (`0002` steps the fold aside) | 127.8 / 128.7 | 128.6 / 128.5 | flat |
| the same with `0001` only (fold live at 256 experts) | 127.8 / 128.7 / 127.5 (d0) | — | **-0.3 %** d0, **-0.5 %** d4096 vs `0002` |

The `0001`-only arm is what justifies the contributor's `0002`: the one launch really is slower at 256 experts
on RDNA4, by a smaller margin here than on their box.  Single-GPU arms; the router is per layer and per token,
so expert placement (`-ncmoe`, `-sm tensor`) does not change the shape.

### Kernel census (`rocprofv3 --kernel-trace`, 4 generated tokens, 35B-A3B)

`mul_mat_vec_f<float,float,...>` for the router is gone.  gfx1100: `topk_moe_cuda<256,false,true>` 123
dispatches (one per MoE layer per token) where gfx1201 with `0002` shows none and the router runs as
`mul_mat_vec_f<float,float,1,256>` (grid 257, the padded 256 rows) again.  The `mul_mat_vec_f<...,1,256>` that
remains on gfx1100 (grid 1) is the **shared-expert** gate, not the router.

## Purity

| gate | result |
|---|---|
| `test-backend-ops -o TOPK_MOE` | 608/608 on ROCm0/1/2 (gfx1201) |
| `gate-prefill-logits.sh` (canonical dense 27B Q8_0, default ubatch) | **0.000000 KLD / 100.000 %** |
| `gate-prefill-logits.sh --ab "GGML_CUDA_ROUTER_TOPK=0" --ub 1`, Qwen3-30B-A3B (gfx1201) | **0.000000 / 100.000 %** |
| the same on Qwen3.6-35B-A3B (gfx1100, the fold live at 256 experts) | **0.000000 / 100.000 %** |
| greedy text, on vs off (30B-A3B on both devices, 35B-A3B on gfx1100) | byte-identical |
| MTP Protocol A, dense 27B UD-Q4_K_XL, 3 devices, `-sm layer`, `-n 3000` | plain 24.98 -> draft-mtp **51.53 t/s**, acceptance **0.62134** (r181 reference 51.7 / 0.62) |
| rule-5 `llama-batched-bench -npp 512 -ntg 128 -npl 1,4,8` (1 GPU, 30B-A3B) | 144.50 / 340.95 / 425.39 -> 139.62 / 344.29 / 420.85 |

The `-ub 1` arm is the one that matters: the fold only exists at a one-token width, so a prefill-width gate
alone would never execute it.  At `-ub 1` every token is its own graph, the fold fires (verified by the census
on the same build), and the resulting logits are bit-identical to the two-launch path.

The MTP arm is inert by construction (a verify batch is more than one token wide, so the fold cannot fire) and
no MoE GGUF we hold carries an MTP head, so the fold's 1-token draft-context path is unmeasured here; it is
the same graph shape as plain decode.

## Release-flow remainder

Per-block build `--changed-from <r181 tip> --fast`: 2/2 (block 10 and the tip), 16 s.  `validate-set.sh` green
(strict fresh-tarball apply, tree `bdbcb1964…`, Windows portability scan 284 files clean).  No GPU faults.
