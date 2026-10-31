# Quantized-KV decode: the rotation folds (r187, PR #258) — measurement record

Dated record for the r187 release (`WORKLOG.md` 261031.005).  The change: five default-on fusions in **block 10**
that restore the delivery's decode fusions on a `-ctk/-ctv q8_0` KV cache, where `build_attn`'s Hadamard
rotations (Q/K after their ROPE, V before the cache write, the attention output before its projection) used to
turn them all off.  Kills: `GGML_CUDA_QK_ROTATE_HADAMARD`, `GGML_CUDA_FUSE_V_FWHT`,
`GGML_CUDA_FA_COMBINE_FWHT`, `GGML_CUDA_QK_PAIR_FWHT`, `GGML_CUDA_FUSE_ROPE_FWHT` (`ENVIRONMENT.md` §7.4).
Claims/rules: `GREEDY-PURITY.md` §74.

Source: `github.com/stew675/llama-cpp-rdna-boosts` PR #258 (submitter `bsp2020`), five patches cut against
r184, applied to the r186 tree with offsets only and folded into block 10 (block 11 takes the one-line
`flash_attn_combine_results` call-site update, since `fattn-qsa.cu` is created there).

## Environment

* 3x Radeon AI PRO R9700 (gfx1201), ROCm 10.1.0 (`/opt/rocm-10.1.0-gfx120X`), `~/bin/build-llama-rocm`
  build flags, one job at a time (**no parallel benches**).
* **Both arms are the same build directory** (`~/llama-scratch/build-rocm`, `GGML_CUDA_FA_QUANTS` identical in
  the canonical and scratch caches), each arm's `bin/` kept: `bin-r186` = the r186 release tree, `bin-pr258` =
  r186 + the five patches (tree-identical to the released r187 tree).
* Perf: `llama-bench -fa 1 -p 0 -n 128 -d 0,4096 -ctk q8_0 -ctv q8_0 -r 3`, **3 interleaved rounds** (stock,
  patched, …), mean quoted.  Purity: `ROCBLAS_USE_HIPBLASLT=0` (issue #67), fresh processes.

## Decode throughput (1 GPU, q8_0 KV)

Qwen3-Coder-30B-A3B-Instruct Q4_K_M (the 30B-A3B MoE the PR measured on its 7900 XTX):

| arm | tg128 | @ d4096 |
|---|---:|---:|
| stock r186 | 139.50 / 142.02 / 142.19 (**141.24**) | 127.74 / 127.32 / 126.92 (**127.33**) |
| + the five folds | 160.12 / 159.82 / 160.23 (**160.06**) | 140.59 / 140.10 / 139.84 (**140.18**) |
| delta | **+13.3 %** | **+10.1 %** |

The PR claimed +10.8 % / +10.7 % on the same card class; the direction and magnitude reproduce (this model is
the Coder variant, and the submitter's own table is on r183's binary).

**F16 KV is unchanged** (the folds are q8_0-only and the plain combine instantiation is untouched):

| arm | tg128 | @ d4096 |
|---|---:|---:|
| stock r186, f16 KV | 162.34 / 163.49 / 162.18 (**162.67**) | 138.60 / 138.66 / 138.91 (**138.72**) |
| + the five folds, f16 KV | 163.09 / 163.06 / 163.14 (**163.10**) | 139.04 / 138.90 / 138.52 (**138.82**) |

The submitter flagged a -0.8 % F16 decode cost they attributed to code layout (and could make come and go by
adding an unused `__global__` to `rope.cu`); it does **not** reproduce on gfx1201 (+0.3 % / +0.1 %).

**The rope-only form (`GGML_CUDA_FUSE_ROPE_FWHT`) on a no-Q/K-norm model** — gpt-oss-20B MXFP4 (1 GPU, q8_0
KV, 2 interleaved rounds; the submitter's arm was Llama-3.2-1B at +8.9 %):

| arm | tg128 | @ d4096 |
|---|---:|---:|
| stock r186 | 148.05 / 149.93 (**148.99**) | 142.96 / 143.43 (**143.20**) |
| + the five folds | 155.19 / 154.82 (**155.01**) | 151.92 / 151.84 (**151.88**) |
| delta | **+4.0 %** | **+6.1 %** |

## Kernel census (the PR's headline)

`rocprofv3 --kernel-trace` on Qwen3-Coder, q8_0 KV, 1 GPU, launch count = `(rows(-n 60) - rows(-n 20)) / 40`
(`--ignore-eos`, fixed prompt):

| arm | launches/token |
|---|---:|
| stock r186 | **726** |
| + the five folds | **510** |

Exactly the submitter's number.  Per layer the V/K/Q projections become one row-concat GEMV, Q and K norm+rope
one pair launch, the combine kernel does the output rotation and the projection's Q8_1, and both `fwht_cuda`
launches are gone (trace: `fwht_cuda<64>` 960 -> 288, `rms_norm_mul_rope_f32_qk_pair<..,128,true>` 336,
`flash_attn_combine_results<128,true>` 336 with `quantize_q8_1` unchanged).

## Batched decode (rule 5)

`llama-batched-bench -npp 512 -ntg 128 -npl 1,4,8`:

| arm | npl 1 | npl 4 | npl 8 |
|---|---:|---:|---:|
| gpt-oss-20b q8_0, stock | 145.40 / 146.17 | 327.49 / 328.50 | 414.50 / 413.31 |
| gpt-oss-20b q8_0, folds | 151.61 / 150.81 (**+3.7 %**) | 334.18 / 332.95 (+1.7 %) | 411.52 / 411.25 (-0.6 %) |
| 35B-A3B Q4_K_M f16 KV, 3 GPU tensor, stock | 116.50 / 116.59 | 357.85 / 359.11 | 578.02 / 579.81 |
| 35B-A3B Q4_K_M f16 KV, 3 GPU tensor, folds | 117.43 / 116.87 | 358.10 / 359.29 | 579.35 / 580.83 |

Prefill (`S_PP`) flat in every row.

## Purity

**Greedy text, byte-identical stock vs the folds** (`--seed 42 --temp 0`, `ROCBLAS_USE_HIPBLASLT=0`, 4 prompts
per cell, `prompts/recall.txt`, `code-python.txt`, `prose-rdna-boosts.txt`, `reasoning.txt` /
`code-reasoning-mixed.txt`):

| model (all `-ctk/-ctv q8_0`) | arm | result |
|---|---|---|
| Qwen3-Coder-30B-A3B Q4_K_M | 1 GPU | 4/4 identical |
| gpt-oss-20B MXFP4 | 1 GPU | 4/4 identical (the rope-only form fires here) |
| Gemma4-12B Q8_0, Ling-3.0-tiny IQ4_NL, LFM2.5-8B-A1B Q6_K, Qwen3.6-35B-A3B Q4_K_M | 1 GPU | identical |
| Qwen3-Coder-30B-A3B | 3 GPU `-sm tensor`, 3 GPU `-sm layer` | identical |
| gpt-oss-20B | 3 GPU `-sm tensor`, 3 GPU `-sm layer` | identical |
| Qwen3.6-35B-A3B `-ncmoe 16` (host experts) | 3 GPU `-sm tensor` | identical |
| Qwen3-Coder `-c 768` context fill/trim run | 1 GPU | identical |

**MTP, four axes at `-n 3000`** (Qwen3.8-27B UD-Q4_K_XL, 1 GPU, f16 KV, `--reasoning on` for R and off for
P/C/K, `--spec-draft-n-max 3`, `-v`/`-lv 4` for the count):

| axis | stock t/s (plain / mtp) | folds t/s (plain / mtp) | acceptance (both builds) |
|---|---|---|---|
| R | 31.1 / 61.2 | 31.2 / 61.2 | **0.61351** |
| P | 30.7 / 74.5 | 30.7 / 74.5 | **0.83360** |
| C | 31.1 / 79.6 | 31.1 / 79.7 | **0.89815** |
| K | 31.3 / 86.2 | 31.3 / 85.9 | **0.99200** |

Text at `-n 3000` (no `-lv 4`, which interleaves timing lines into the echoed stream and invalidates a text
hash) is byte-identical on all four axes: R `ac2655b79fbc`, P `957e65cb56e8`, C `6536aa147f62`, K
`a87c4318b649`, and a plain-decode control on K the same `a87c4318b649`.

**Prefill logits** (`gate-prefill-logits.sh`, 27B Q8_0, wikitext-2, 40 chunks):

| arm | result |
|---|---|
| `--kv f16` vs the recorded base | mean KLD **0.000000** / same-top-p **100.000 %** |
| `--kv q8_0` vs the recorded base | mean KLD **0.000000** / same-top-p **100.000 %** |
| `--kv q8_0 --ab "<all five>=0" --ub 1` (Qwen3-Coder) | mean KLD **-0.000000** / same-top-p 99.990 % |
| off-vs-off control on that base (the harness floor) | mean KLD -0.000000 / same-top-p **99.990 %** |
| gpt-oss-20b `--kv q8_0 --ab "GGML_CUDA_FUSE_ROPE_FWHT=0"` (default ub — the rope-only form runs in prefill) | mean KLD **0.000000** / same-top-p 99.990 % |

(`99.990 %` is the stored-logit resolution, not a difference: the same-build control reads it.)

**Width probe** (`test-logits-width-probe`, `KV=q8_0`, 3 runs per arm, the reference arm is the five switches
off):

| model / arm | P | W8 hash | row-0 hash | vs reference |
|---|---|---:|---|---|
| gpt-oss-20B, 1 GPU | 3584 | `81b0f0ab5f0b535b` | `d53ebc29b3bb61e0` | identical, 3/3 runs |
| gpt-oss-20B, 1 GPU | 4096 | `b75df2f40180dd0a` | `d3595398241d10bf` | identical |
| gpt-oss-20B, 3 GPU `-sm tensor` | 4096 | `b88c5d63af1bce5f` | `e3c4c9931a3bffdf` | identical |
| gpt-oss-20B, 3 GPU `-sm layer` | 4096 | `b75df2f40180dd0a` | `d3595398241d10bf` | identical (= 1 GPU) |
| Qwen3-Coder-30B-A3B, 1 GPU | 3584 | `146bfdf445740a1c` | `86c922a350bab991` | identical, 2/2 runs |
| Qwen3-Coder-30B-A3B, 1 GPU | 4096 | `b2ae0cc87bee5fec` | `9011d94cb085aa3a` | identical, 2/2 runs |

`width_purity=PASS`, worst `maxdiff_vs_W8` 0 in every arm.

## Other gates

* `test-backend-ops -b ROCm0 -o ROPE,SET_ROWS,FLASH_ATTN_EXT`: **7059/7059** both builds.
* Per-block build gate `--changed-from 8f9dad0be --fast`: **3/3** (block 10 315 s, block 11 315 s, tip 6 s);
  the tip is built with `all` by `~/bin/build-llama-rocm`, which is where the op tests above ran.
* `scripts/validate-set.sh`: OK (16 patches, applied tree `7cd527bde8…`, Windows portability scan clean).
* `scripts/check-gpu-faults.sh`: journal + run logs clean.
* Not run, and not applicable: parking (gates 6), HC fusions (7), state (8), spec-align (9), graph-cache
  census (10), checkpoint reuse (11/12), spec-width (13) — the change touches no server, state, HC or graph-cache
  code.  Gate 14's width-probe arm is covered above for every affected model.
