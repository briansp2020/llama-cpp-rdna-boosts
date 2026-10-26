# 2026-10-25 — qwen4exp HC mixer: the inject GEMM must sit beside the down GEMM (r142, issue #226 follow-up)

**Question.** The r138 re-base adopted upstream #31028's merged-tensor `qwen4exp` HC path.  A reporter
measured **prefill ~3.4 % slower than r137** on their separate-tensor (pre-#31028) Flash-Next GGUF
(`IQ3_XXS`, all-VRAM, `-sm tensor`, 31.5k-token prefill: 3058-3062 t/s on r137 vs 2954-2958 on r138) and
attributed it to the mixer's emission order: r137 emitted the `hc_inject` GEMM right after the `hc_down`
GEMM, #31028 emits it after the *gate* GEMM, and the two BF16 GEMMs that read the same F32 `xn` therefore
no longer share the CUDA BF16 src1 conversion.  Is that the whole difference, and does restoring the
adjacency recover it?

**Outcome: yes, and it is bit-identical — +3.1 % / +3.4 % / +3.4 % prefill (pp2048 / pp8192 / pp31500).**

## Root cause, seen three ways

1. **The code.**  `bf16_src1_cache` (`ggml/src/ggml-cuda/common.cuh`, block 10) holds **one**
   `(root, data, stream, n)` entry: the last F32 src1 converted for a BF16 cuBLAS GEMM.  Its comment says
   it exists because "the hc mixer feeds one normalized `[hc_dim, T]` tensor to two BF16 GEMMs (`hc_down`
   and the 4-wide `hc_inject`), which converted all of it twice; the second GEMM now reuses the first
   copy".  That only works if the pair is *adjacent in the graph*.  `build_hc_mix` emitted
   `*inject = scale(build_lora_mm(w_inject, xn), 1/hc)` at the very end (after `gate = mm(w_up, silu(lo))`),
   and `gate`'s src1 — the silu output — is itself cached, evicting `xn`.
2. **The graph.**  `GGML_SCHED_DEBUG=2 -v` (`llama-bench -p 32`, one prefill ubatch) puts
   `blk.0.hc_attn_down` at node **#10** and `blk.0.hc_attn_inject` at node **#72** in r141; the r137 build
   had them at **#9** and **#10**.  Same node *set* and count (3873) in both — a pure permutation, which is
   why nothing else moved and why the regression is invisible to every correctness gate.
3. **The kernel count.**  `rocprofv3 --kernel-trace` over one pp2048 pass,
   `convert_unary_cont_vec4<float, __hip_bfloat16>` grouped by element count:

| tensor | r137 | r141 | r142 |
|---|---:|---:|---:|
| `xn` = `[10240, 512]` (5.24 M elements) | 2280 calls / 97.3 ms | **4560 calls / 273.9 ms** | 2280 calls / 85.3 ms |
| `lo` = `[320, 512]` (0.16 M) | 2280 / 4.2 ms | 2280 / 3.9 ms | 2280 / 4.2 ms |
| `[2560, 512]` (the MoE/router src1) | 5616 / 79.9 ms | 5616 / 82.2 ms | 5616 / 82.2 ms |

## The change

`src/models/qwen4exp.cpp`, `build_hc_mix`, the `else` (separate-tensor) branch: build `lo`'s GEMM, expand
it, build the inject GEMM, expand it, then scale/silu `lo` as before; the trailing
`if (inject && !w_down_inject)` block is folded in there.

`ggml_build_forward_expand` order — not node creation order — is what lands a node in the graph, so the
explicit expansion is the whole fix.  (The r137 build reached the same order through the pre-#31028
call graph; nothing in `qwen4exp.cpp`'s `build_hc_mix` differs between r137 and r141 apart from the merged
path's `ggml_cont`, which a separate-tensor GGUF does not take.)

## Prefill A/B

2x/3x R9700 (gfx1201, ROCm 10.1.0), `llama-bench -p ... -n 0 -r 2 -ngl 99 -sm tensor -t 8
--load-mode none --lazy-mode off`, interleaved (A/B/A/B), rep 2 quoted, rep 1 agrees:

| test | r141 (base) | r142 (fix) | Δ |
|---|---:|---:|---:|
| pp2048 | 2287.32 | **2359.53** | **+3.2 %** |
| pp8192 | 2208.12 | **2283.70** | **+3.4 %** |
| pp31500 | 2158.08 | **2227.83** | **+3.2 %** |
| pp2048 / 8192 / 31500 (rep 1) | 2284.39 / 2203.99 / 2154.48 | 2354.96 / 2280.18 / 2229.44 | +3.1 / +3.5 / +3.5 % |

Same table on the 2-GPU shape (`HIP_VISIBLE_DEVICES=0,2`, the two x8 uplinks, `-ncmoe 24`):
835.5 -> **841.7 t/s (+0.7 %)** — the hc mixers are a small share beside host-expert traffic.

**Mechanism control.** The same r142 build with `GGML_CUDA_BF16_SRC1_CACHE=0` (no reuse at all) measures
**2248.7 t/s** — between the base and the fix, i.e. adjacency still pays ~1.5 % even with the cache off
(the second conversion re-reads a hot `xn` instead of a cold one), and the cache reuse is worth the rest.

**The `LLAMA_FUSED_DSV4_HC_POST=0` arm** (the reporter's other number): pp8192 2268.0 -> **2321.6 t/s
(+2.4 %)**.

## Correctness (all bit-identical)

| check | result |
|---|---|
| PPL, 4 chunks, `-sm tensor`, default `DSV4_HC_POST` arm | 2.4392 (base) = 2.4392 |
| PPL, 4 chunks, `LLAMA_FUSED_DSV4_HC_POST=0` | 2.4370 = 2.4370 |
| the reporter's `--chunks 24` command (`HC_POST=0`) | 24/24 chunk PPLs identical, final **2.4202** both arms |
| fused combine+norm vs `GGML_CUDA_DISABLE_HC_COMB=1` (`--kl-divergence`, `HC_POST=0`, 4 chunks) | **−0.000000 KLD / 100.000 % same-top-p** (the fusion still fires) |
| same-seed greedy cells (prose prompt, `-n 32 --temp 0 --seed 42`) | Flash-Next GSQ `-sm tensor` `487888b20ddb`; GSQ + `draft-mtp n3` **same hash** (plain == draft); GSQ 2 GPU `-ncmoe 24` `13406dc51399`; Flash-Next IQ4_NL `9bef5c01369c`; 4B Q8_0 `d2ffb97ccb76`; 27B UD-Q4_K_XL `a7aecbf5c5c1`; 35B-A3B `-ncmoe 40` (2 GPU) `e2dde701fc04` — **7/7 identical base vs fix** |
| op oracles | `-o HC_MIX` 30/30, `-o HC_COMBINE` 20/20, `-o HC_COMBINE_MIX` 6/6, `-o FLASH_ATTN_QSA` 26/26, `-o INDEXER_TOPK` 3/3 |
| `gate-prefill-logits.sh` | PASS 0.000531 / 98.863 % (unchanged) |
| `validate-set.sh` | strict 15/15, tree `810570cad` |

The prefill-logit gate alone would *not* have caught the regression (logits are identical, only the number
of conversions changes) — that is what the trace and the `-p` A/B are for.

## The reporter's exact model (measured 2026-10-25, after r142 shipped)

`/llm/models/Qwen3.8/27B/GSQ-RCO-IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S` (77.86 GiB, 48 blocks, pre-#31028
separate-tensor HC — the reporter's production layout), 3 GPU `-sm tensor`, all-VRAM,
`--load-mode none --lazy-mode off`, interleaved A/B against r141, 2 reps per side:

| test | r141 | r142 | |
|---|---:|---:|---:|
| pp2048 | 2265.84 / 2264.24 | **2338.01 / 2338.91** | **+3.2 %** |
| pp8192 | 2184.88 / 2178.89 | **2254.81 / 2256.62** | **+3.2 %** |
| pp31500 | 2133.90 / 2129.21 | **2196.73 / 2199.35** | **+2.9 %** |
| pp8192, `LLAMA_FUSED_DSV4_HC_POST=0` | 2243.76 / 2239.16 | **2296.00 / 2284.48** | **+2.2 %** |
| pp31500, `HC_POST=0` | 2193.88 / 2187.62 | **2238.04 / 2238.65** | **+2.2 %** |

PPL **2.2057** (default arm) and **2.2141** (`HC_POST=0`) in both builds; greedy text 172 chars
sha `e82e1992943a` identical.  The reporter's own numbers were −3.4 % (IQ3_XXS) and −2.5 % / −4.5 %
(IQ3_S) for r138-vs-r137; we measure the recovery as +2.9-3.2 % / +2.2 %.

## Not covered

* **The reporter's `IQ3_S` production arm** — closed after the release, see "The reporter's exact model"
  above (it landed on this box the same day).
* **A merged-tensor (post-#31028) GGUF** — it runs one GEMM for both products, so there is no pair to
  keep adjacent and nothing to fix; still no such GGUF on this fleet (r141 note).
* The remaining **2280 `xn` conversions** (85 ms in one pp2048 pass) are the single copy the *first* GEMM
  needs.  The fused combine+norm could hand it over directly (`ggml_cuda_hc_combine_norm_set_bf16_src1`),
  but on this model the MMB path already claims the kernel's `out_xn_bf16` slot
  (`ggml_cuda_hc_combine_norm_set_bf16`), so the cuBLAS cache would need a second slot.  Not attempted —
  measured as the residual, not as a win.
