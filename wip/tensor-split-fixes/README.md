# tensor-split-fixes (issue #105): page faults, memory and host overhead under `-sm tensor` + MTP

Nine small fixes found on 2 × R9700 (gfx1201, PCIe 5.0 x8 each) with qwen4exp under `-sm tensor` + MTP.  They are
format-patches on top of `v16-a55e952b8-r17` (applied tree `04764deb`), applied after `patches/00*.patch` with `git am`.
Full report: issue [#105](https://github.com/stew675/llama-cpp-rdna-boosts/issues/105).

| patch | files | what | switch |
|---|---|---|---|
| [`0001`](0001-ggml-cuda-keep-retired-Q8_1-arenas-alive-for-capture.patch) | `ggml-cuda/common.cuh`, `ggml-cuda.cu` | keep a grown-out Q8_1 arena alive until the context is destroyed - fixes a page fault in `quantize_q8_1` | `GGML_CUDA_Q8_1_ARENA_FREE_OLD=1` restores the old behaviour |
| [`0002`](0002-ggml-cuda-key-captured-graphs-by-their-data-pointers.patch) | `ggml-cuda/common.cuh`, `ggml-cuda.cu` | hash node/source data pointers into the graph-cache key - stops warm graphs being invalidated as MTP alternates layouts | `GGML_CUDA_GRAPH_KEY_NO_DATA=1` |
| [`0003`](0003-ggml-cuda-recapture-graphs-after-the-memory-they-cap.patch) | `ggml-cuda/common.cuh`, `ggml-cuda.cu` | recapture a graph when memory it captured (pool temporaries, FA/H2D staging) was freed since | `GGML_CUDA_GRAPH_MEM_GEN=0` |
| [`0004`](0004-ggml-meta-keep-several-split-state-cache-versions-in.patch) | `ggml-backend-meta.cpp` | keep 8 split-state cache versions per buffer instead of clearing the cache on every mismatch | none; `GGML_META_SS_VERIFY=1` checks every hit |
| [`0005`](0005-llama-LLAMA_KV_N_PAD_MIN-raises-the-n_kv-padding-flo.patch) | `src/llama-kv-cache.cpp` | `LLAMA_KV_N_PAD_MIN` raises the n_kv padding floor (opt-in; default 256 unchanged) | unset = old behaviour |
| [`0006`](0006-qwen4exp-stop-pinning-block_out-for-the-whole-graph-.patch) | `models/qwen4exp.cpp`, `ggml-cuda.cu` | stop pinning every layer's block output as a graph output in prefill (~1.9 GiB of the 3.5 GiB compute buffer at `-ub 2048`); the fused combine+norm reads a pool copy if its outputs reuse the freed buffer | `LLAMA_HC_PIN_BLOCK_OUT=1` |
| [`0007`](0007-ggml-meta-compact-split-state-cache-entries-hash-map.patch) | `ggml-backend-meta.cpp` | compact split-state cache entries (only the `n_segments x n_bufs` used values of the ~2.2 KiB struct) and hash maps for both meta tensor caches | none; `GGML_META_SS_VERIFY=1` checks every hit |
| [`0008`](0008-qwen4exp-copy-the-conv-state-tail-straight-into-each.patch) | `models/qwen4exp.cpp` | copy the strided conv-state tail straight into each rollback slot instead of `cont` + `cpy` (~144 kernels per verify graph) | `LLAMA_CONV_TAIL_CONT=1` |
| [`0009`](0009-ggml-planar-HC_MIX-output-so-the-verify-band-needs-n.patch) | `ggml.h`, `ggml.c`, `ggml-cuda/hc-mix.cu`, `ggml-cpu/ops.cpp`, `models/qwen4exp.cpp` | `ggml_hc_mix_set_planar()`: HC_MIX writes the mixed rows, then the inject rows, so the verify band needs no `ggml_cont` of the mixed head (~96 kernels per verify graph); BF16 CUDA path only | `LLAMA_HC_MIX_PLANAR=0` |

0001 and 0002 are the two patches from the r12 version of this PR, rebased unchanged; 0003-0005 came with the r15
update and 0006-0009 are new in this one.  All nine rebase onto r17 without conflicts; the numbers below were measured on r16 (not re-run on r17, whose
changes are also on the host-expert path).  Every patch has a switch that restores the old behaviour.

## 0001: Q8_1 arena freed under captured graphs (page fault)

**Symptom.** `-sm tensor -ub 2048`, two greedy 400-token decodes, then a ~1.8k-token prompt: `Memory Fault Error ...
kernel: quantize_q8_1`, "page not present", on both GPUs.

**Cause.** The per-context Q8_1 input arena (`q8_1_cache_get`) grows by allocating a larger buffer and `cudaFree`-ing the
old one.  Captured decode/verify graphs keep pointers into the old arena, and the multi-token graph that grows it is never
captured, so the next decode replay writes freed memory.  The fault address is the old 32 MiB arena's base.

**Fix.** Retired arenas go into `q8_1_arena_retired` and are freed in `~ggml_backend_cuda_context()`.  The retained memory
is bounded by the final arena size (the arena doubles).

## 0002: graph cache keyed only by (first node, n_tokens)

**Symptom.** With MTP, `-sm tensor` decode gained much less than `-sm layer`.  ~20 % of decode-band `graph_compute` calls
ran eagerly or recaptured under `-sm tensor` + MTP, against ~1.5 % without MTP.

**Cause.** As the verify width moves between 1 and 4 tokens, the per-device allocations alternate between a few layouts
(`hc_inject` views, `ffn_shexp`, `ffn_gate` data pointers).  With one graph per (first node, n_tokens), each switch costs
an eager run, then a recapture.

**Fix.** Add an FNV-1a hash of every node's and source's `data` pointer to `ggml_cuda_graph_key`, so each layout keeps its
own captured graph.  The existing 10 s idle eviction bounds the cache.

## 0003: graphs replayed after memory they captured was freed

**Symptom.** With 0002 and `LLAMA_KV_N_PAD_MIN=1024`, a server warm-up followed by a request on a fresh slot faulted 5/5
(`Memory access fault ... Page not present`).

**Cause.** A captured graph bakes in addresses that are not graph tensors, so neither the key nor the property check sees
them change: pool temporaries and two per-context scratch buffers that grow by free + malloc, the FA prefill staging arena
(`fattn_stage_try_get`) and the H2D staging ring.  A hipMalloc/hipFree log placed the fault inside a freed FA staging
buffer.  0002 makes it reachable more often (graphs live longer), but the hazard does not depend on it.

**Fix.** A per-device generation counter, bumped by every one of those frees (`clear_pool`, pool-full free, FA/H2D
staging regrowth).  A graph records it at capture and is recaptured when it changed.  With it: 4/4 survive.

## 0004: meta split-state cache cleared on every mismatch

**Cause.** The meta buffer's split-state cache is cleared whenever a tensor's bytes differ from its cached copy.  With
MTP the verify and draft graphs reuse the same tensor addresses with different shapes, so it was cleared on nearly every
graph and ~1,000 split states were recomputed per `graph_compute` (~21 % of the main thread in
`ggml_backend_meta_get_split_state`).

**Fix.** 8 versions per buffer: on a mismatch switch to the version in which the tensor matches, else recycle the least
recently used one.  A hit still requires identical tensor bytes, as before.  `GGML_META_SS_VERIFY=1` recomputes every
outermost hit and aborts on a difference; it found none.  Decode 77.7 -> 84.6 t/s (measured on r12), greedy output
identical, verify-step scheduler alloc 2.43 -> 1.56 ms.

## 0005: `LLAMA_KV_N_PAD_MIN`

n_kv is padded to 256 cells, so every 256 tokens the KV views change size and the whole decode graph's layout shifts.
Under `-sm tensor` that invalidates ~100 per-device graphs per token at each step.  `LLAMA_KV_N_PAD_MIN=1024` keeps them
constant 4x longer: fresh-prompt decode 77.7 -> 83.3 t/s (measured on r12), greedy output identical, `-sm layer`
unaffected; 4096 gave nothing more.  Default unchanged.

## 0006: every layer's block output pinned for the whole prefill graph

**Symptom.** `-sm tensor -c 262144 -ub 2048` runs out of memory on the first 2048-token ubatch of a long prompt: the
prefill compute buffer is 3.69 GB per device and leaves ~0.5 GB free.

**Cause.** `build_hc_combine` calls `ggml_set_output(block_out)` so the fused combine+norm kernel never sees
`block_out` aliased by its outputs.  The graph allocator never frees outputs, so every layer's block output
(attention, linear-attention and MoE outputs, 20 MiB each at `-ub 2048`) stays allocated to the end of the graph:
an allocator peak dump showed 96 of them live at the 3.5 GiB peak (same pattern under `-sm layer`).

**Fix.** Drop the pin.  In the repeat-anchored CUDA matcher the window nodes are skipped when fused, so only `add` and
`mulg` are written; when either reuses the freed `block_out` buffer the kernel reads `block_out` from a pool copy
(20 MiB D2D) and the fusion is still taken.  Compute buffer 3509 -> 1573 MiB per device; the 259.6k prompt fits at
`-ub 2048`.  KL divergence against the pinned build (wikitext 16 x 2048, same split and `-ub`): 0.000000 for
`-sm tensor` and `-sm layer`.

## 0007: split-state lookups cost ~11 % of the main thread

With MTP, `ggml_backend_meta_get_split_state` is called ~10k times per decode step.  A split state is ~2.2 KiB
(`ne[16*16]`), the cache entries were `std::map` nodes of ~2.5 KiB and every hit copied the whole struct (perf:
10.9 % of the main thread, with ~0 % recomputes).  Storing only the used part and switching both caches to
`std::unordered_map` cut it to ~4 %: warm greedy 91.0-91.9 -> 94.2-94.9 t/s (p-min 0.5).

## 0008, 0009: copy kernels in the verify graph

With the graph-capture churn gone, a kernel trace shows the GPUs ~58 % busy and most of the idle time in ~3 us gaps
between dependent kernels (~84k kernels/s per GPU), 15 % of them runtime memcpy blits.  Two of those copy groups are
not needed:

- 0008: every linear-attention layer copied the conv-state tail into a contiguous temporary and then into each
  rollback slot; `ggml_cpy` takes the strided view directly (one 2D memcpy).
- 0009: the fused HC_MIX output interleaves the mixed head and the inject tail per token, so the verify band made
  the mixed head contiguous with `ggml_cont`; the planar layout makes both outputs contiguous views.  Byte-identical
  at one token; only the BF16 CUDA kernels take it (they already address the output through a base and a stride),
  the Q8_0 and CPU paths assert it is off.

Together ~240 kernels fewer per verify graph (84k -> 76k kernels/s), +2 % each; greedy output unchanged, KLD of 0008
against the old graph 0.000000.

## MTP draft p-min under `-sm tensor` (configuration, no patch)

With `--spec-draft-p-min 0.5` the draft stops early on low-confidence tokens, so the verify width varies (2-4) and
llama.cpp, which keeps one previous graph per context, reuses only ~76 % of the verify graphs; under `-sm tensor` a
rebuilt verify graph costs ~7 ms (graph build, scheduler allocation and the meta split).  With `--spec-draft-p-min 0`
the draft always runs n-max 3, the verify is always 4 wide, ~98 % of the verify graphs are reused and acceptance goes
up too.  n-max 3 stays the best (2, 4, 5, 6 measured).  This is lossless for greedy output.

## Results on r16

2 × R9700, ISTA-DASLab Qwen3.8-Flash-Next GSQ-RCO IQ3_XXS + shared Q8_0 MTP head, all experts in VRAM
(`--n-cpu-moe 0 -ot per_layer_token_embd.*=CPU`), `-c 262144 -ub 2048 -ctk q8_0 -ctv q8_0`, draft-mtp n-max 3
**p-min 0**, stock ROCm 10.0 for both.  One session, one model load per mode and test, r16 + all nine patches.

- **layer**: our production settings, `-sm layer -ts 59,41`.
- **tensor**: `-sm tensor LLAMA_KV_N_PAD_MIN=1024`, no `GGML_HIP_GRAPH_FORCE_UPDATE` (see the ROCm section below).
- Both builds also carry two local knobs we use in production (`LLAMA_MTP_DRAFT_UBATCH=256`, and
  `GGML_CUDA_POOL_CACHE_MB` = 256 for layer, 128 for tensor).

| t/s | layer | tensor | tensor vs layer |
|---|---|---|---|
| greedy 400-token decode | 94.6 - 96.6 | 113.9 - 114.5 | +19 % |
| 6 requests A,B,A,B,C,A (C = new prompt) | 97.1 94.2 97.8 95.0 94.0 98.2 | 111.6 109.6 115.8 113.6 102.7 114.9 | +17 % (C +9 %) |
| 1500-token generation x3 | 109.8 86.0 110.3 | 122.3 97.3 122.1 | +11 - 13 % |
| chat decode | 90.1 | 98.7 | +10 % |
| 2 concurrent requests (sum) | 122.3 - 123.7 | 142.4 - 147.0 | +15 - 20 % |
| prefill 1.8k / 37k / 155k | 1787 / 2032 - 2054 / 1709 | 1863 / 2428 - 2447 / 2022 | +4 / +19 / +18 % |
| 259.6k prompt: prefill / decode | 1473 / 30.6 | 1754 / 47.2 | +19 % / +54 % |
| VRAM peak per card | 30.6 / 31.8 GB | 30.9 / 31.4 GB | |

Long-context suite (passkey @10/50/90 % of 256k, last-log-entry) 4/4 and the warm-up + fresh-slot sequence 2/2 in both
modes; 0 GPU faults or OOMs; greedy output identical to the same configs on r15 (`f2471f845142` layer, `45e68f1bdfeb`
tensor).  On r15 the same tensor config measured greedy 112.5 - 114.0 and the layer config 93.6 - 97.4, so r16 changes
nothing here (blocks 16 and 17 are on the host-expert path, which these runs don't use).  The PR branch on its own (r15 +
0001-0009, without our two local knobs) builds and gives the same greedy output at 64K.

We also asked the 259.6k prompt's question ("how many buildings are mentioned?", answer 6,358) once greedy and 8 times
sampled at temp 0.7 per mode: both modes answered right 7/8 sampled and wrong greedy (1,578 tensor, 1,760 layer); our
older layer-split runs were wrong about 40 % of the time too, so that answer is a model limit, not a split difference.

The earlier round (patches 0001-0005, p-min 0.5, tensor at `-ub 1024`), for reference:

| t/s | layer, r15 + patches | tensor, stock r15 | tensor, r15 + patches |
|---|---|---|---|
| greedy 400-token decode | 87.5 - 91.1 | 78.9 - 83.0 | 87.1 - 92.9 |
| 6 requests A,B,A,B,C,A (C = new prompt) | 89.9 86.6 92.8 87.9 88.7 93.8 | 79.4 77.3 81.6 78.0 68.8 81.6 | 87.0 88.2 91.9 89.3 78.8 91.4 |
| 1500-token generation x3 | 102.9 80.0 102.6 | 94.0 64.1 95.1 | 101.9 74.5 102.8 |
| 2 concurrent requests (sum) | 99.1 - 105.1 | 87.9 - 89.4 | 104.3 - 110.6 |
| prefill 1.8k / 37k / ~153k | 1769 / 2054 - 2059 / 1708 | not run | 1566 / 2112 - 2168 / 1814 |
| 259.6k prompt: prefill / decode | 1474 / 27.2 | 1498 / 45.6 | 1533 / 46.6 |

Long-context suite (passkey @10/50/90 % of 256k, last-log-entry) 4/4 with the patches in both modes (not run on stock); warm-up + fresh-slot sequence
survives 2/2 in all three; 0 GPU faults or OOMs in any run.  The gaps that round left (a brand-new prompt -11 %, chat -5 %, short-prompt prefill) are what 0006-0009 and p-min 0 closed.

**Correctness.** Greedy output is deterministic in each mode but differs between them (`f2471f845142` vs
`45e68f1bdfeb`), so we compared logits with `llama-perplexity --kl-divergence` against the production layer config
(measured on r15; r16 gives the same greedy output in both modes):

| vs layer `-ub 2048` | mean KLD | same top-1 | PPL |
|---|---|---|---|
| tensor `-ub 2048`, all nine patches | 0.0106 | 96.63 % | +0.06 % |
| tensor `-ub 1024`, patches 0001-0005 | 0.0107 | 96.48 % | -0.11 % |
| layer `-ub 1024` (noise floor) | 0.0107 | 96.65 % | +0.11 % |

Tensor split differs from layer split by as much as layer split differs from itself at another `-ub`.

**TODO #41.** These runs keep every expert in VRAM, so the block-06 H2D staging ring never activates (no staging in the
logs) and #41 does not reach this config.  0003 does touch the ring (its regrowth now bumps the generation) but does not
change what the ring copies.

## ROCm and `GGML_HIP_GRAPH_FORCE_UPDATE`

Under tensor + MTP the decode graphs change layout often, and on ROCm 10.0 recapturing them (destroy + instantiate) is
what makes a new prompt slow.  Your existing `GGML_HIP_GRAPH_FORCE_UPDATE=1` path updates the exec in place instead and
closes most of that gap, but on ROCm 10.0 `hipGraphExecUpdate` leaks kernarg memory, ~0.33 MiB per update for a
100-kernel graph (ROCm/rocm-systems#10713).  The partial fix (ROCm/rocm-systems#11069, merged to develop 2026-09-22) is
in the TheRock nightly `10.2.0a20261005` but not in 10.0/10.1: there, VRAM stayed flat over a 60-request server soak.
The r15 round measured tensor with that nightly and `FORCE_UPDATE`.  With p-min 0 the graphs are reused ~98 % of the time, so the
nightly is no longer required: on stock ROCm 10.0 without `FORCE_UPDATE` (and `GGML_CUDA_POOL_CACHE_MB=128`, which keeps
the 259.6k prompt peak at 30.9 / 31.4 GB), the same nine patches give the r16 numbers above, and VRAM stayed flat over a 12-round (~150
request) soak on r15.  `FORCE_UPDATE` on the nightly is still worth ~6-7 % on brand-new prompts and
concurrent requests.

## Not measured

1-GPU and non-RDNA4 layouts; more than 2 GPUs; host-resident experts (`-ncmoe`) with these patches; 0005 with values
other than 1024 and 4096; 0009 on the Q8_0 HC_MIX path (it stays off there).
