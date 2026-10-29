# backend-sampler-reserve-once — llama: reserve the compute buffers once per sequence, and again only for a wider chain

`UPSTREAM-PR-backend-sampler-reserve-once.patch` is cut against plain `ggml-org/llama.cpp` **`6050b467c`**
(the delivery base; applied tree `c87e8593`), sha256
`c7a8999b5b2c9a25634a35eded1d9776df143744316c172c88bdd4eaf9133bd5`.  It touches
`src/llama-context.cpp` and `src/llama-context.h` (+26 / -7).  It is the delivery's block-12 form
(`wip/backend-sampler-reserve-once/` in contributor PR #245, landed as r169) **without** the
`LLAMA_BACKEND_SAMPLER_RESERVE_ONCE` kill switch.  Contributor: overdoingism (PR #245).

**What.** `llama_context::set_sampler()` sets `sched_need_reserve = true` whenever an offloadable sampler
chain is set for a sequence **and** whenever one is removed; `llama-server --backend-sampling` sets a new
chain for every request and removes it when a slot is released (and for a grammar request, whose chain
cannot be offloaded), so every request re-creates the scheduler — freeing and re-allocating every compute
buffer, at the same size — before its first token.

The patch records what the buffers were reserved with (`sampling.reserved`: per sequence, the chain's
`llama_sampler_backend_n_nodes()`) in `sched_reserve()`, and `set_sampler()` asks for a reserve only when
the sequence has no entry or the new chain is **wider**.  A chain set again, a narrower one, or a removal
needs none: the graph is rebuilt anyway (the samplers are part of the graph-reuse check), the output
buffers are sized per decode, and the allocator grows a buffer if a chain needs more.

**Why the key is the node count and not just the sequence id.**  `graph_max_nodes()` is recomputed *only*
by a reserve, and it is what sizes the graph's node capacity (`llm_graph_result::max_nodes` feeds
`ggml_new_graph_custom`; `ggml_build_forward` asserts `cgraph->n_nodes < cgraph->size`), the scheduler's
node/leaf arrays and the compute-metadata arena.  The allocator's on-demand growth covers buffer *sizes*,
so a chain that is merely *bigger in bytes* is safe — but a chain with **more nodes** than the reserved
one is not.  A seq-id-only key misses exactly that: `tests/test-backend-sampler.cpp`'s `test_set_sampler`
replaces a 1-node `dist` chain with a `top_k + dist` chain on the same sequence and would then build into
a graph capacity sized for the first chain.

**Validation** (gfx1201 R9700, Linux, ROCm 10.1.0, delivery tree r169 = this change on `6050b467c` +
block set): plain vs patched `llama-server`, Qwen3.8-27B-UD-Q4_K_XL, `--backend-sampling -ngl 99 -c 65536
-ub 2048 -b 2048 --parallel 2 --kv-unified -fa on -ctk q8_0 -ctv q8_0`, a 2396-token prompt with
`cache_prompt: false`, seven sampled requests plus a grammar request and one more sampled request, two
interleaved starts: median prompt time **1963.9 → 1895.2 ms (-68.7 ms, -3.5 %)**, client time to first token
2750 → 2682 ms.  `sched_reserve` lines over three requests: **4 → 2**.  `test-backend-sampler
--test=test_set_sampler`: **PASS**, and it reserves **2** times (the initial set and the widening) where the
seq-id-only key reserved once.  Identical request text in every arm (greedy, seeded temperature 0.7, and
the grammar fallback, `ROCBLAS_USE_HIPBLASLT=0`), dense 4B/27B coherence byte-identical, prefill-logit KLD
0.000000 / same-top-p 100.000 %.

**Not validated:** other backends (CUDA, Vulkan, Metal) — the change is in `llama_context`, not in a
backend; Windows; multi-GPU (`-sm tensor` refuses backend sampling outright); a draft context with
`--spec-draft-backend-sampling` (same function, not measured).  Re-run the validation on current master
before filing, and check whether upstream has meanwhile changed the reserve/sampler contract.
