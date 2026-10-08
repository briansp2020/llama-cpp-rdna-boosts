# bf16-src1-reuse: convert a shared F32 src1 to BF16 once for the prefill hc mixer GEMMs

One format-patch on top of `v16-a55e952b8-r33` (applies with `git am`, no conflicts).  Switch:
`GGML_CUDA_BF16_SRC1_CACHE=0` restores r33.  Bit-identical.  Prefill only (graphs with >= 64 tokens); decode and
verify are untouched.

## What

On RDNA4 the BF16 hyper-connection weights of Flash-Next GSQ-RCO take the cuBLAS/hipBLASLt BF16 path at prefill (MMB
does not take BF16 weights there, `mmb_dense_flag` off; its HC16 BF16-producer marking is RDNA3_5-only).  That path
converts the whole F32 src1 to BF16 for every GEMM.  The hc mixer feeds one normalized `[hc_dim, T]` tensor (`hc_norm`)
to two of them, `hc_down` and the 4-wide `hc_inject`, so a 2048-token ubatch converted ~84 MB tensors twice per mixer:
in a pp2048 trace, 43.3 ms of the 56.6 ms of `convert_unary_cont` per ubatch per GPU was this conversion.

Three parts:

1. `ggml_backend_cuda_context::bf16_src1_cache` (`common.cuh`): a single-entry cache of the BF16 copy, keyed like the
   Q8_1 input cache (view root, data, stream, element count), cleared at graph start next to `q8_1_cache_clear()`.  The
   cuBLAS BF16 impl looks it up for a contiguous F32 src1 with >= 64 columns and otherwise converts into it.  Declared
   after `pools`, so its pool buffer is released before the pools are destroyed.
2. `src/models/qwen4exp.cpp`: `build_hc_mix` emits the inject GEMM right after the down GEMM.  It used to be emitted
   at the combine, and the block's own BF16 GEMMs in between evicted the single entry.
3. `hc_combine_norm` (multi-token, F32 `xn` stored, no `out_xn_bf16` already set) writes the BF16 copy of `xn` in the
   same pass, through its existing `out_xn_bf16` output, straight into the cache entry, so the first GEMM finds it too.
   `hc_f2bf32` rounds to nearest even like the conversion it replaces; the F32 `xn` is still stored for `dsv4_hc_pre`.

Worth a careful look: the key (root, data pointer, stream, size) does not notice an in-place write into the root
between two GEMMs that read it.  Nothing writes `hc_norm` in between in the qwen4exp graph, and the entry is dropped
at every graph start, but a future graph that did that would read a stale copy.  The Q8_1 cache has the same
property, which is why I followed its keying.

## Results on r33 (2 x R9700 / gfx1201, ROCm 10.0, Flash-Next GSQ-RCO IQ3_XXS + shared Q8_0 MTP head)

Plain `v16-a55e952b8-r33` (applied tree `13479e6`) vs r33 + this patch, same session, all experts in VRAM, production
server flags (256K, q8_0 KV, `-ub 2048`, MTP n-max 3), one run per arm unless noted:

| | r33 | r33 + patch |
|---|---|---|
| `llama-bench` pp2048, `-sm tensor` (5 reps) | 2495 ± 16 t/s | 3173 ± 28 t/s (+27 %) |
| `llama-bench` pp2048, `-sm layer -ts 59/41` (5 reps) | 2412 ± 31 t/s | 2519 ± 36 t/s (+4.5 %) |
| server prefill, 2.9k-token prompt, fresh server, 3 runs | 2322 / 2233 / 2413 t/s | 2468 / 2334 / 2529 t/s |
| server prefill, 9.4k / 32.6k natural text, `-sm tensor` | 2297 / 2391 t/s | 2383 / 2473 t/s |
| server prefill, 37k / 151k, `-sm tensor` | 2443-2474 / 2022 t/s | 2582-2595 / 2063 t/s |
| server prefill, 9.4k / 32.6k, `-sm layer -ts 59,41`, 128K | 1821 / 2038 t/s | 1879 / 2108 t/s |
| greedy 400-token decode | 122.8-125.3 t/s | 122.6-124.8 t/s |
| 259.6k-token prompt VRAM peak, card1 / card2, fresh server | 32,596 / 32,605 MiB | 32,592 / 32,592 MiB |

Output:
- Greedy shas identical between the two builds: decode `45e68f1bdfeb`, 2.9k prompt `0a6791b84da2`, 45-token prompt
  `42a52754c67d`, after the 9.4k / 32.6k prompts `483f91653cdf` / `05a5d6f5e8a9` (the same under `-sm layer`).
  The 36k needle passes on both.
- `llama-perplexity` wikitext-2 test, 8 x 4096, `-sm tensor`, KLD vs r33: mean 0.000000, max 0.000062, same top
  token 100 %.  r33 against itself gives the same numbers (max 0.000062), so that is the run-to-run floor here, not
  the patch.
- `test-backend-ops` HC_COMBINE 1/1, HC_MIX 31/31, MUL_MAT 1305/1305.
- 0 GPU faults, 0 reallocations, 0 out-of-memory in every run.

Where the time went, from a trace when I first wrote it (r19, `-sm tensor`, pp2048): `convert_unary_cont` per ubatch
per GPU 56.6 -> 13.1 ms (parts 1+2 alone: 35.4 ms), GPU busy per ubatch 567 -> 532 ms.

It has been in our production builds since r20; the current one is r33 + this patch (plus our other local patches).

## Not measured

RDNA3 / RDNA3_5 (where MMB may take these GEMMs and the cache would not fire); NVIDIA; other models with BF16 weights on
the cuBLAS path; `test-fusion` (our model does not fit on one card).
