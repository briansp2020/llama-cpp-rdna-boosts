# 2026-10-29 — MMB owns what MMB claims: the pair fusion's stand-down predicate (r171, `TODO` #115)

**Question.** `mul_mat_q_pair_try_fuse` stood down for MMB on a weight-*type* test
(`ggml_cuda_mmb_dense_will_take` / `_routed_will_take`) while the fused-GLU and mmq triple matchers decide on the
*shape* (`ggml_cuda_mmb_supported_mm` / `_supported_mmid`).  The type answer is a superset, so on any shape MMB
declines (`T < mmb_min_t()`, `n_used > 64`, a `K` that is not a whole super-block count, ...) the pair had already
stood down and the **mmq triple** took the gate/up pair instead of the **pair** (one shared activation quantize).
Both belong to `mul_mat_q`, so the question is which wrapper is faster, and (since r171) also which one the
optimizer's BF16-only activation marking is consistent with.

## What was measured, and on what

| box | device | ROCm | build |
|---|---|---|---|
| `soar` (primary) | 3x R9700, gfx1201 | 10.1.0-gfx120X | r170 vs r171, same tree path, two build dirs |
| `halo` | Strix Halo 8060S, gfx1151 | 10.1.0-gfx1151 | r170 vs r171, same tree path (`~/m115/src`), two build dirs |

Models: `LFM2/Huihui-LFM2.5-8B-A1B-abliterated.i1-Q6_K` (lfm2moe, 32 experts, `n_used` 4 — the model `TODO` #115
names), `Qwen3.6/35B-A3B/Q4_K_M` (qwen35moe), `Qwen3.8/27B/Q8_0`, `Qwen3.8/Flash-Next/IQ4_NL` (qwen4exp).

The arm is only reachable where MMB's **routed** path is on: gfx1151 accepts the whole type set, gfx1201 accepts
only the IQ family + Q4_1/Q5_0 and sets `routed = 0`, so **gfx1201 is inert by construction** (verified, below).

## 1. Which matcher actually runs on gfx1151 (`GGML_CUDA_MMB_LOG=1`, LFM2 Q6_K, pp512)

| arm | pp512 t/s | MMB log |
|---|---|---|
| default | 4758 / 4894 | `MMB_GLU fused` x2 (one per MoE layer) |
| `GGML_CUDA_MMB_GLU=0` | 4857 / 4858 | no GLU — the mmq **triple** |
| `GGML_CUDA_DISABLE_MOE_MMQ_FUSION=1` | 4913 / 4915 | `MMB_GLU fused` (the env is **inert** here) |
| `GGML_PAIR_OFF=1` | 4874 / 4929 | `MMB_GLU fused` (inert here too) |

So the "MMB-declined MoE" premise of `TODO` #115 does not hold for this model on the current tree: MMB's fused GLU
accepts (`M=1792 K=2048 T=512 n_used=4`, Q6_K is in gfx1151's type mask), which also means the r161 A/B
(fused 5421 vs `GGML_CUDA_DISABLE_MOE_MMQ_FUSION` 5397) compared **two MMB-GLU arms**.  The decision to reproduce is
the *predicate*, not the number.

## 2. Pair vs triple (the arm the change selects), gfx1151, pp512, interleaved 2 rounds, `-r 3`

`GGML_CUDA_MMB_ROUTED=0` on both arms (so the routed pair is the MMQ pair and `GGML_PAIR_ROUTED_OFF=1` restores the
triple), which leaves the dense MMB path — most of this box's prefill — untouched.

| model | pair (r170 = r171 here) | triple (`GGML_PAIR_ROUTED_OFF=1`) |
|---|---|---|
| LFM2 Q6_K | **3379 / 3399** | 3348 / 3407 |
| 35B-A3B Q4_K_M | **2304 / 2312** | 2296 / 2313 |

Neutral.  (Absolute levels are ~35 % below the default arm because `ROUTED=0` also disables MMB's routed experts,
which are a large part of this box's MoE prefill win.)

## 3. The configuration the change actually moves, and its purity

A shape MMB declines with the routed path otherwise on: `GGML_CUDA_MMB_MIN_T=1024` at `-ub 512` (T=512).  r170 hands
it to the mmq triple; r171 fuses the pair (`GGML_CUDA_PAIR_DEBUG=1` confirms `fused routed=1 T=512 type=q6_K
n_experts=32`).  Greedy text, 48 tokens from a 4 kB wikitext prompt, `--temp 0 --seed 42`:

| arm | LFM2 Q6_K | 35B-A3B Q4_K_M | Flash-Next IQ4_NL (`-ub 256`) |
|---|---|---|---|
| r170 default (triple) | 765 chars `d120b56017c2` | 723 chars `44cbb654f6db` | 682 chars `488f8aa2f2e8` |
| r171 default (pair) | **765 chars `d120b56017c2`** | **723 chars `44cbb654f6db`** | **682 chars `488f8aa2f2e8`** |
| `GGML_PAIR_ROUTED_OFF=1` (triple) | 765 `d120b56017c2` | 723 `44cbb654f6db` | — |
| unfused (`+GGML_CUDA_DISABLE_MOE_MMQ_FUSION=1`) | 765 `d120b56017c2` | 723 `44cbb654f6db` | — |
| `GGML_CUDA_MMB=0` | 765 `d120b56017c2` | 723 `44cbb654f6db` | — |

All arms agree: the pair and the mmq triple accumulate the same dots with the same `J` tile and apply the GLU
elementwise, so this is a *kernel-selection* change with no numerics footprint (not a re-baseline).  Flash-Next at
`-ub 256` also exercises the routed-SWIGLU->MMQ matcher, whose stand-down became exact in the same change; its text
is unchanged as well.  Its prompt throughput is unchanged (543.6 -> 546.2 t/s).

## 4. gfx1201 (the maintainer's daily arm) — inert

Same-seed greedy text, 128 tokens, r170 vs r171 (`/tmp/bin-r170` vs `/tmp/bin-r171`):

| arm | r170 | r171 |
|---|---|---|
| 27B Q8_0, 1 GPU | 110 chars `da2e2d192e21` | **identical** |
| 35B-A3B Q4_K_M, 1 GPU | 518 `5d485275db6d` | **identical** |
| LFM2 Q6_K, 1 GPU | 621 `0b02fbdc4bd3` | **identical** |
| 35B-A3B Q4_K_M, 3 GPU `-sm tensor -ncmoe 40` | 518 `5d485275db6d` | **identical** |

`llama-bench -p 2048 -n 0 -r 2`, 1 GPU: 35B-A3B Q4_K_M 4856 -> 4894 t/s, LFM2 Q6_K 9177 -> 9166 (noise).
Prefill-logit gate (`scripts/gate-prefill-logits.sh`, the recorded gfx1201 bases): 27B Q8_0 **0.000000 / 100.000 %**,
35B-A3B Q4_K_M **-0.000000 / 100.000 %**.

## 5. The bug this exercise found (and the reason the arm is worth gating)

The first cut of the exact-predicate stand-down called `ggml_cuda_mmb_supported_mm()` on whatever node pair sat at
`i, i+1`.  The matcher runs on **every** node, so an op without `src[1]` followed by a `MUL_MAT` dereferenced null:
Flash-Next at `-ub 256` on gfx1151 **SIGSEGV'd during `common_init_from_params`'s warm-up decode** (2/2 runs, GDB
backtrace in `ggml_cuda_mmb_supported_mm` under `ggml_cuda_mul_mat_q_pair_try_fuse`), while every other model on
both boxes was fine.  Fixed at the call site (all source slots checked, both nodes required to be the same op) and in
the predicates themselves (null-safe).  A type-only predicate could not have crashed — it took a weight and returned
a bool.

## Reproduce

```bash
# gfx1201: nothing changes
HIP_VISIBLE_DEVICES=0 ./build-rocm/bin/llama-bench -m <27B Q8_0> -ngl 99 -p 2048 -n 0 -r 2

# gfx1151: the arm the change selects
GGML_CUDA_MMB_ROUTED=0 GGML_CUDA_PAIR_DEBUG=1 ./build/bin/llama-bench -m <LFM2 Q6_K> -ngl 99 -p 512 -n 0 -r 3
GGML_CUDA_MMB_ROUTED=0 GGML_CUDA_PAIR_ROUTED_OFF=1 ./build/bin/llama-bench -m <LFM2 Q6_K> -ngl 99 -p 512 -n 0 -r 3

# which matcher owns a routed pair (gfx1151)
GGML_CUDA_MMB_LOG=1 ./build/bin/llama-bench -m <MoE> -ngl 99 -p 512 -n 0 -r 1   # "MMB_GLU fused"?
```
