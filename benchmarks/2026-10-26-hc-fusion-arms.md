# qwen4exp hyper-connection fusion arms — equivalence, and what the fusions are worth (issue #228, r148)

Date: 2026-10-26 · Host: **halo** (Strix Halo, gfx1151, 128 GB UMA, single device) · ROCm 10.1.0
(`/opt/rocm-10.1.0-gfx1151`) · build `df41660ce` (r148) / the pre-fix build `6ff4499ce` (r147) as the
control · release: `v15-6050b467c-r148`.

Two questions, both about `GGML_OP_HC_MIX` / `GGML_OP_HC_COMBINE`:

1. **Equivalence** — the model's HC chain is emitted by `build_hc_mix` / `build_hc_combine`; each half can
   be replaced by a fused op **independently** (`LLAMA_FUSED_HC_MIX`, `LLAMA_FUSED_HC_COMBINE`), so there
   are four arms.  Through r147 two of them were wrong (the fused mixer published a *raw* inject logit
   while the fused combine divided by `hc` again — the v14 convention, which the `6050b467c` re-base
   inverted under the ops): the chain-mixer + fused-combine arm double-scaled, the fused-mixer +
   chain-combine arm did not scale at all.
2. **Cost** — does fixing the convention (and keeping the fusions) cost anything?

The oracle is `scripts/gate-hc-fusions.sh` (it compares the arms with the **unfused chain**, which
`test-backend-ops -o HC_MIX,HC_COMBINE` cannot: its CPU reference can encode the same mistake).

## Equivalence — greedy text, `-n 96 --seed 42 --temp 0 -c 8192 -fa on -ngl 99`

Three GGUFs, one per hc-weight class (the hc-down type decides whether the fused mixer is even eligible):

| class (hc_attn_down) | model | pre-fix `LLAMA_FUSED_HC_COMBINE=0` | pre-fix **default** |
|---|---|---|---|
| Q5_K + F32/F32/Q8_0 — chain mixer | Flash-Next Q5_K_M (Unsloth trunk, 86 GiB) | 445 chars, coherent (the chain) | **445 chars, greedy repetition loop** |
| IQ4_NL — chain mixer | Flash-Next IQ4_NL (93 GiB) | 459 chars, coherent | **degenerate** ("The user asked is a simple request. The prompt is is not encoded.") |
| BF16 — fused mixer | Flash-Next GSQ-RCO IQ3_XXS (71 GiB) | **empty generation (1 char)** | 454 chars, coherent |

After r148, on the same three GGUFs, all four mix/combine arms + `GGML_CUDA_DISABLE_HC_FUSION=1` +
`LLAMA_FUSED_DSV4_HC_PRE=0` are **byte-identical to the unfused chain**:

| model | chars | sha256[:12] | note |
|---|---:|---|---|
| Q5_K_M (chain) | 445 | `c8e579144ff1` | = the pre-fix chain arm's text |
| IQ4_NL (chain) | — | `d9e037d0aa03` (arm.sh prompt) | = the pre-fix chain arm's text |
| GSQ-RCO IQ3_XXS (fused) | 449 | `6a102dbc62d6` | **= the pre-fix default** (`246463129328` on the shorter prompt): the arm the delivery validated does not move |

`LLAMA_FUSED_DSV4_HC_POST=1` is a *reported* arm, not a hard one: the op is documented as rounding-level
vs its chain on one device (r97 / §59), and on the Q5_K model it differs by one near-tie (446 vs 445 chars)
while the BF16-hc model matches on all 7 arms.

## Cost — the fusions are still worth what they were worth

`llama-bench -p 512 -n 64 -r 2 -ngl 99 -fa on`, all device-resident (UMA), default vs both arms off
(identical numerics after r148, so this is a clean A/B of the fused kernels against the chain):

| model | default (fused/fused where eligible) | `LLAMA_FUSED_HC_MIX=0 LLAMA_FUSED_HC_COMBINE=0` | |
|---|---:|---:|---|
| GSQ-IQ3_XXS (BF16 hc, fused mixer) pp512 | 766.16 | 764.50 | flat |
| GSQ-IQ3_XXS tg64 | **34.78** | 31.73 | **+9.6 %** default |
| GSQ-IQ3_XXS tg64, `LLAMA_FUSED_HC_COMBINE=0` only | 33.31 | | combine fusion +4.4 % |
| GSQ-IQ3_XXS tg64, `LLAMA_FUSED_HC_MIX=0` only | 32.00 | | mixer fusion (vs that) +8.7 % |
| IQ4_NL (chain mixer) pp512 / tg64 | 973.81 / 34.49 | 969.56 / 34.35 | parity (the mixer is not eligible) |
| Q5_K_M (chain mixer) pp512 / tg64 | 879.03 / 33.80 | 887.49 / 33.57 | parity |
| Q5_K_M / IQ4_NL — the *corrected* default vs the pre-fix default | same kernels, same work | | no delta by construction: r148 moves one multiply per inject row from the combine kernel into the mixer kernel |

So: prefill is untouched (the fused mixer only serves the `nt <= HC_BF16_MAX_TOKENS` band, prefill uses the
chain), decode keeps the fused win on the class that can take both halves, and the chain-mixer class is at
parity either way — i.e. the fix is a numerics fix with no throughput trade.

## Op-level oracle (the leg that still has to pass)

`test-backend-ops` on the tip, `gfx1151`: `HC_MIX` 51/51, `HC_COMBINE` 20/20, `HC_COMBINE_MIX` 6/6,
`DSV4_HC_PRE`/`_POST`/`_COMB` 11/11 — every op against the **corrected** CPU reference.  That set agreed
with the ops before r148 too (both encoded the raw-inject convention), which is the lesson recorded in
`GREEDY-PURITY.md` §61: an op/CPU pair is not an oracle for an op/chain contract.

## Environment

`ROCBLAS_USE_HIPBLASLT` was not needed: every arm here is a fresh `llama-cli`/`llama-bench` process on a
single device, and the arms agree to the byte.  Logs: `/tmp/repro228/` (this session), the gate keeps its
own per-arm logs under `KEEPOUT` (default `mktemp -d`).
