# wip/gdn-decode-fusions

Four small fusions around the GDN layers and the attention gate, in three patches. Each is bit-identical and has its own off switch. Two match only the qwen4exp (Flash-Next) graph; the other two also fire on qwen35 / qwen35moe.

Three format-patches on `v12-097cf33d9-r100` (tree `d572cbd6`); applied tree `4955cf16`.

## What changes

**0001 — GDN alpha/beta BF16 projections and the softplus gate chain in one launch** (`mmvf.cu`, `ggml-cuda-try-fuse.cu`)
- qwen4exp projects alpha and beta from the same input with two BF16 matrices, then computes `gate = softplus(alpha + dt) * a`. On the decode band that was three launches: two `mul_mat_vec_f` and the fused `add_softplus_mul_bcast`.
- `ggml_cuda_gate_beta_bf16_try_fuse` runs both projections in one `mul_mat_vec_f` launch: alpha rows first, then beta rows. The gate chain is applied to the alpha rows before the store. The beta sigmoid stays where it is, for the GATED_DELTA_NET fusion.
- Each row runs the same instantiation the unfused MUL_MAT takes (the short-row `vb` kernel is excluded), so both outputs are bit-identical. The Q8_0 pair keeps `ssm_gate_beta`.
- Off switch: `GGML_CUDA_FUSE_GATE_BETA_BF16=0`.

**0002 — the q / k l2 norm pair folded into the direct conv kernel** (`gdn-conv.cu`)
- After the GDN conv, `build_gdn_l2_norm` runs RMS_NORM + SCALE over the 128-wide q and k heads of the conv output (one `rms_norm_scale_pair_f32<256>` launch since #179).
- The direct conv kernel's 256-thread block already holds two whole heads (warps 0–3 and 4–7), so it now also writes both normalized outputs. Each thread puts `0 + y*y` through the same warp butterfly, and the head's four warp sums go through the second butterfly in lanes 0..3 with 0 elsewhere. Those are the values and lanes `rms_norm_scale_pair_f32` reduces, so mean, scale and outputs are bit-identical.
- Taken only when all of these hold:
  - only no-op nodes sit between the conv and the pair;
  - both views have the standard q / k head layout;
  - the SCALE has no bias;
  - nothing else reads the RMS_NORM outputs;
  - neither SCALE output overlaps the conv's `x`, state, weight or output. The allocator may place them over the state or `x`: both are dead after the CONCAT in graph order, but the direct kernel still reads them. In a separate run that logged each rejection, the outputs overlapped the state on 124 and `x` on 40 of 240 direct convs on 35B-A3B, and the state on 192 of 1,392 on 27B. Without this check, 35B-A3B gave different, run-to-run varying greedy output; with it, the output is identical to stock. Stock's own conv output was disjoint from `x` in every launch we logged.
- Only fires where the direct conv runs: on one GPU, or on two with `GGML_CUDA_CONV_FUSION_MULTI=1`.
- Off switch: `GGML_CUDA_FUSE_GDN_CONV_NORM=0`.

**0003 — gate → Q8_1: the attention gate view read in place, and the GDN output norm computed in the gate kernel** (`unary.cu`, `ggml-cuda-try-fuse.cu`)
- **Attention output gate:** CONT(the gate half of each Q head) → SIGMOID → MUL → matmul. The gate → Q8_1 kernel already takes row strides, so it reads the row-strided view directly and the CONT launches nothing (it is an exact copy). Off switch: `GGML_CUDA_FUSE_CONT_GATE_Q8_1=0`.
- **GDN output norm (qwen4exp):** `RMS_NORM(xn) * w` is skipped at its node and computed by the gate kernel (`unary_gated_q8_1_norm_kernel`), one 128-wide row per 256-thread block exactly as `rms_norm_f32<256, true>` runs it.
  - Both nodes take the decision from the same predicate, `ggml_cuda_gdn_norm_gate_match`.
  - The context records the skipped norm, and a skip that is never consumed aborts.
  - The match also checks that no node in between writes over the norm's input or weight.
  - Off switch: `GGML_CUDA_FUSE_GDN_NORM_GATE=0`.
- A two-rows-per-block version of the norm kernel was 1 ulp off in some rows, so it stays one row per block.

## Where each fusion fires

`rocprofv3 --kernel-trace`, 1 GPU decode, the same greedy work on stock r100 and r100 + patches:

| fusion | 27B (MTP) | 35B-A3B |
|---|---|---|
| 0001 alpha/beta | — (no match) | — (no match) |
| 0002 conv + norm | 25,920 of 26,112 direct-conv launches; `rms_norm_scale_pair_f32` 26,208 → 288 | 114 of 300 direct-conv launches, all on prompt batches (no MTP, so decode does not take the direct conv); the other 186 fall back on the overlap check |
| 0003 attention gate | `cpy_scalar` 10,977 → 164 | `cpy_scalar` 64,240 → 48,220 |
| 0003 GDN output norm | — (no match) | — (no match) |

0001 and the GDN-norm half of 0003 match only qwen4exp's graph. On our Flash-Next build, 0001 turns three launches per GDN layer into one, and the 0003 norm removes one. The Flash-Next throughput below covers them on stock r100.

## Gates

Setup:
- R9700 (gfx1201), native Linux, HIP runtime 7.15.26333.
- Stock r100 (`s100st`) against r100 + these patches (`s100up2`), same build dir.
- Models:
  - Qwen3.8-27B Q4_K_M with its own MTP head (`--spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0`);
  - Qwen3.6-35B-A3B Q4_K_M, plain decode;
  - Qwen3.8-Flash-Next GSQ-RCO IQ3_XXS, 2 GPU `-sm tensor --n-cpu-moe 10`, shared Q8_0 MTP head, stock env.
- 27B / 35B server settings: `-c 65536 -ngl 99 -fa on -ctk q8_0 -ctv q8_0 -ub 2048 -b 2048`. The 2 GPU arms set `GGML_CUDA_CONV_FUSION_MULTI=1`.

**Greedy hashes** (400 tokens, prose / code): identical to stock in every arm.

| model | config | prose / code |
|---|---|---|
| 27B | 1 GPU: stock / patched / patched with all four switches off / patched + `GGML_CUDA_GRAPH_OPT=1`; 2 GPU `-sm layer` | `d12cdeede326` / `e8a93c816237` |
| 27B | 2 GPU `-sm tensor` | `7cfb359d5c5d` / `dcd93c8965ce` |
| 35B-A3B | 1 GPU: the same four arms; 2 GPU `-sm layer` | `0eedd580608a` / `365e3058e68f` |
| 35B-A3B | 2 GPU `-sm tensor` | `c0cf0739bdab` / `930bcf1f4405` |
| Flash-Next IQ3_XXS | 2 GPU `-sm tensor`, 6 server loads per arm | `a7a81f01b3bc` / `4bb43e3339c8` |

**`test-logits-width-probe`** (1 GPU, q8_0 KV): PASS with worst maxdiff 0 at P=256 and P=3584 on 27B and 35B-A3B. All W8 and row-0 hashes are identical to stock (27B P=256 row 0 `a31712e2ed95b812`, P=3584 `044e7fb409d64485`; 35B-A3B `76cd46de7a1c41df` / `a64aa01b578d2654`).

**Prefill-logit** (wikitext-2 test 16 × 2048, 1 GPU, against bases recorded from stock r100): mean KLD 0.000000 on both models. Max KLD 0.000065 and same top p 99.994 % on both.

**`test-backend-ops`** (stock and patched): `-o SSM_CONV` 45/45, `-o RMS_NORM` 53/53, `-o SCALE` 7/7, `-o MUL_MAT` 2444/2444.

**Faults**: 0 memory-access faults or asserts in any run.

The gates above ran on the patches before one last guard in 0003, which keeps the attention-gate fusion off a pending GDN norm and aborts if a skipped norm's gate node is ever passed. On the final patches the greedy hashes were re-checked: 27B and 35B-A3B 1 GPU, 35B-A3B 2 GPU `-sm tensor`, and Flash-Next 2 GPU `-sm tensor`, all identical to the table above.

## Throughput (decode)

27B / 35B-A3B: two alternated rounds per arm, each 2 × (400-token prose + code) greedy runs; median of the 4. Flash-Next: 3 rotating rounds, the mean of 6 servers × (prose + code).

| config | stock r100 | r100 + patches | change |
|---|---|---|---|
| Flash-Next IQ3_XXS, 2 GPU `-sm tensor`, MTP | 137.80 t/s | 139.97 t/s | +1.6 % (every round) |
| 35B-A3B, 1 GPU | 131.05 t/s | 131.70 t/s | +0.5 % (switches off: 131.10) |
| 35B-A3B, 2 GPU `-sm tensor` | 142.50 t/s | 143.60 t/s | +0.8 % |
| 27B MTP, 1 GPU | 68.95 t/s | 69.10 t/s | +0.2 % (switches off: 68.80) |
| 27B MTP, 2 GPU `-sm tensor` | 106.15 t/s | 107.00 t/s | +0.8 % |

- 31.5k prefill is flat on 1 GPU (27B 1148 → 1153, 35B-A3B 4575 → 4582 t/s). The 2 GPU prefill numbers moved by more between rounds than between arms.
- On our own Flash-Next build (IQ3_S, 2 GPU `-sm tensor`, `--n-cpu-moe 47`), paired 5-round A/Bs gave decode +0.56 % for 0001 + 0002 and +0.89 % (95 % CI +0.50 to +1.28 %) for 0003, with identical output. Those runs predate 0002's overlap check.

## Limits

- 0001 and 0003 act on the decode / verify band (the `mul_mat_vec_f` and gate → Q8_1 paths); larger batches keep their launches.
- 0001 is BF16 only; a Q8_0 alpha/beta pair keeps `ssm_gate_beta`.
- 0002 needs the direct conv path and 128-wide q / k heads, and falls back to the separate pair launch when an output overlaps the conv's buffers.
