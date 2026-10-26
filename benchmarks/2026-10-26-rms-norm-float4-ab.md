# rms_norm reduction order — float4 (upstream #29720) vs the pre-#29720 strided reduction (r145, issue #229)

**Question.** The r127 re-base adopted upstream [ggml-org/llama.cpp#29720](https://github.com/ggml-org/llama.cpp/pull/29720),
which gave `rms_norm_f32` a float4 reduction (a lane owns the four contiguous columns `4*i .. 4*i+3` of
`i = tid + k*block_size`) and took it unconditionally for `ncols % 4 == 0` with 16-byte-aligned operands.
Issue #229 reports that this costs 7-9 % of MTP decode on a 1x R9700 (Windows, ROCm 10.0, graphs off,
Qwen3.8-27B UD-Q5_K_XL, `draft-mtp n3`), entirely through draft acceptance, and that
`GGML_CUDA_RMS_NORM_VEC=0` restores r126 bit-for-bit.  This record measures both arms on the delivery's
box to decide the default.

**Box.** gfx1201 (R9700), ROCm 10.1.0, `~/bin/build-llama-rocm`; nothing else running.  Every arm is a
fresh process; `ROCBLAS_USE_HIPBLASLT=0` on every run (fresh-process comparison rule, `AGENTS.md`).  The
two arms are the *same binary*: `GGML_CUDA_RMS_NORM_VEC=1` forces the float4 grouping,
`GGML_CUDA_RMS_NORM_VEC=0` the strided one.

## 1. Prefill / decode throughput (`llama-bench`, `-r 3`)

**Qwen3.8-27B UD-Q4_K_XL, 1 GPU** (`-ngl 99 -t 8`):

| test | float4 | strided | Δ |
|---|---:|---:|---:|
| pp512 | 1397.21 ± 14.14 | 1370.94 ± 2.84 | **+1.9 %** |
| pp2048 | 1402.84 ± 2.47 | 1365.61 ± 1.30 | **+2.7 %** |
| pp8192 | 1340.85 ± 5.13 | 1316.02 ± 1.42 | **+1.9 %** |
| tg128 | 30.89 ± 0.02 | 30.87 ± 0.01 | ~0 |

**Qwen3.8-27B UD-Q4_K_XL, 3 GPU `-sm tensor`:**

| test | float4 | strided | Δ |
|---|---:|---:|---:|
| pp512 | 1981.31 ± 35.63 | 1960.85 ± 25.21 | +1.0 % |
| pp2048 | 1990.07 ± 2.99 | 1971.61 ± 17.33 | +0.9 % |
| pp8192 | 1886.14 ± 1.35 | 1866.09 ± 2.68 | +1.1 % |
| tg128 | 54.18 ± 0.36 | 54.39 ± 0.49 | ~0 |

**Qwen3.6-35B-A3B UD-Q4_K_M, 1 GPU (all in VRAM):**

| test | float4 | strided | Δ |
|---|---:|---:|---:|
| pp512 | 4477.64 ± 630.66 | 5011.92 ± 23.21 | noisy (first arm cold) |
| pp2048 | 4975.69 ± 17.77 | 4922.05 ± 37.02 | +1.1 % |
| pp8192 | 4733.81 ± 28.06 | 4667.86 ± 27.91 | +1.4 % |
| tg128 | 101.71 ± 0.62 | 101.38 ± 1.25 | ~0 |

So the float4 grouping is a **prefill** optimization: **+1.0-2.7 % prefill, ~0 % plain decode**, everywhere.

## 2. MTP (dense 27B, 1 GPU, greedy, `q8_0` K/V, `draft-mtp n3`)

`llama-server` + one `/completion` request (`cache_prompt: false`, `temperature 0`, `seed 42`), prompt
`prompts/prose-rdna-boosts.txt` (5.2 k tokens) or a 94 k-token documentation corpus
(`sha256 ddaa0f16…`, built from this repo's own docs).  Both arms repeat **exactly** (bit-identical
acceptance counts and text on a second run).

**Default arm (all fusions on):**

| context | float4 acceptance / tok/s | strided acceptance / tok/s | Δ |
|---|---|---|---|
| 5.2 k, n=512 | 0.90097 (373/414) / 76.68 | 0.88756 (371/418) / 75.41 | +1.3 pts / +1.7 % |
| 94 k, n=512 | 0.75855 (355/468) / 58.94 | 0.75159 (354/471) / 58.51 | +0.7 pts / +0.7 % |

**The reporter's shape — SSM decode fusions disabled**
(`GGML_CUDA_DISABLE_NORM_PAIR=1 GGML_CUDA_FUSE_GDN_NORM_GATE=0 GGML_CUDA_DISABLE_CONV_FUSION=1
GGML_CUDA_DISABLE_SSM_PRESCAN=1 GGML_CUDA_FUSE_GDN_GATE=0`, 5.2 k prompt):

| n | float4 | strided | Δ |
|---|---|---|---|
| 256 | 0.61111 (165/270) / 58.02 | 0.79204 (179/226) / 68.72 | **−18 %**, −18 pts |
| 512 | 0.77273 (357/462) / 68.01 | 0.88756 (371/418) / 74.55 | −8 % , −11 pts |

Two things follow.

* The **sign is payload-dependent** (+1.3 pts here on the default arm, −18 pts with the fusions off, and
  the reporter measures −7..−9 % tg on his box).  The mechanism is chaotic amplification: the RMS sum's
  last bit moves every layer by ~1 ulp, and acceptance is a discrete function of the logits.  Nothing is
  wrong numerically (PPL agrees to 4 dp, the prefill-logit gate passes at 0.0005 mean KLD) and the arms are
  **pure**: `--spec-type none` and `--spec-type draft-mtp` produce byte-identical greedy text in *every*
  arm above, so this is not a draft/verify inconsistency.
* Under the **strided** reduction, disabling the SSM fusions changes nothing: the fusions-off arm at
  n=512 is byte-identical to the default arm at n=512 (same 371/418 acceptance, same text hash
  `63a2391b2208`).  Under float4 it changes everything.  That is the fusion-equivalence contract
  (`GREEDY-PURITY.md` §60) showing up end to end.

## 3. Per-fusion equivalence at decode widths (`scripts/gate-prefill-logits.sh --ab … --ub 1`)

Qwen3.5-4B Q8_0, `--ctx 512 --chunks 6`, `-ub 1` (decode shapes: the rms_norm→Q8_1 fold, the GDN q/k
norm pair and `rms_norm_mul_rope` all fire).  Each row compares a fusion arm against the chain it stands
in for.  The probe's fresh-process noise floor is ~1e-6 (measured: `0.000001 / 100.000 %` on arms that do
not fire at this width).

| fusion arm | float4 (r144 default) | strided (r145 default) |
|---|---|---|
| `GGML_CUDA_DISABLE_NORM_Q8_1=1` (rms_norm→Q8_1 fold) | **0.000511 / 99.216 %** | **−0.000000 / 100.000 %** |
| `GGML_CUDA_DISABLE_NORM_PAIR=1` (GDN q/k l2 norm pair) | **0.000529 / 98.889 %** | **−0.000000 / 100.000 %** |
| `GGML_CUDA_DISABLE_RMS_NORM_MUL_ROPE=1` | **0.000441 / 99.085 %** | **−0.000000 / 100.000 %** |
| `GGML_CUDA_FUSE_ADD_RMS_Q8=0` | 0.000001 / 100.000 % (did not fire) | — |
| `GGML_CUDA_FUSE_GDN_NORM_GATE=0` | 0.000001 / 100.000 % (did not fire) | — |
| `GGML_CUDA_DISABLE_L2_NORM_PAIR=1` (a different op) | 0.000001 / 100.000 % | — |

i.e. the re-base had silently made three kernels that are documented *bit-identical* to `rms_norm_f32`
merely *approximate*.  They are exact again on the strided default.

## 4. Decision (r145)

The **strided reduction is the default** (`GGML_CUDA_RMS_NORM_VEC` unset; `=1` opts into float4):

* every recorded MTP/acceptance number, the adaptive-MTP controller's tuning and the decode-band fused
  norm kernels were built on it — it is the delivery's reference, not a preference;
* the float4 grouping's payoff is +1.0-2.7 % **prefill** against a **±10 points of MTP acceptance**
  lottery, and MTP decode is what this delivery exists for;
* with it on, the fused norm kernels are approximate (0.0005 KLD) unless they are all re-aligned to the
  vec4 grouping, which is a much larger numerics change for a diagnostic arm (issue #229's follow-up
  discussion; the alignment is *not* implemented).

`GGML_CUDA_RMS_NORM_VEC=1` keeps the float4 path available for a prefill-only workload (and for the A/B in
`ENVIRONMENT.md` §7.5).  Rule: `GREEDY-PURITY.md` §60.

## Reproduce

```bash
# prefill/decode (same binary, env selects the arm)
for v in 1 0; do
  HIP_VISIBLE_DEVICES=0 ROCBLAS_USE_HIPBLASLT=0 GGML_CUDA_RMS_NORM_VEC=$v \
    ~/llama.cpp/build-rocm/bin/llama-bench -m /llm/models/Qwen3.8/27B/Q4_K_XL/Qwen3.8-27B-UD-Q4_K_XL.gguf \
    -ngl 99 -p 512,2048,8192 -n 128 -r 3 -t 8
done

# MTP: llama-server + one greedy /completion request; read "draft acceptance" from the slot log
HIP_VISIBLE_DEVICES=0 ROCBLAS_USE_HIPBLASLT=0 GGML_CUDA_RMS_NORM_VEC=$v \
  ~/llama.cpp/build-rocm/bin/llama-server -m <27B gguf> -ngl 99 -c 16384 --fit false -fa on \
  -b 1024 -ub 512 -ctk q8_0 -ctv q8_0 -t 8 --seed 42 --temp 0 --top-k 1 --repeat-penalty 1.0 \
  --reasoning off --spec-type draft-mtp --spec-draft-n-max 3 --port 8100
# POST /completion {prompt: prompts/prose-rdna-boosts.txt, n_predict: 512, temperature: 0, seed: 42,
#                  cache_prompt: false, stream: false}

# per-fusion equivalence at decode widths (each fusion arm vs its chain)
MODEL=/llm/models/Qwen3.5/4B/Q8_0/Qwen3.5-4B-Q8_0.gguf \
  ./scripts/gate-prefill-logits.sh --ab "GGML_CUDA_DISABLE_NORM_Q8_1=1" --ub 1 --ctx 512 --chunks 6
```
