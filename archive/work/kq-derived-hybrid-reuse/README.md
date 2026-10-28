# wip/kq-derived-hybrid-reuse — the derived kq mask input can be reused on hybrid models

One `git am` file (one commit), made on `v15-6050b467c-r156`; it applies unchanged on `v15-6050b467c-r159`
(`git apply --check`, no offsets).  It touches only `src/llama-graph.cpp`.

**Why.** `llm_graph_input_kq_derived::can_reuse()` rejects the reuse whenever the graph's memory context is not a
`llama_kv_cache_context`, on the grounds that the derived form only exists for prefill-shaped batches, which are rebuilt
anyway.  A hybrid model (Qwen3.x: `llama_memory_hybrid_context`) also takes the derived form for single-sequence decode
batches of more than 8 tokens - the verification of a long n-gram draft (`--spec-ngram-mod-n-max 15`: 9-16 rows) - whose
shape repeats step after step, so each of those steps rebuilt its graph.

**What changes.** The input was built from the hybrid context's attention cache (`build_inp_mem_hybrid()` →
`build_attn_inp_kv_impl(mctx->get_attn())`), so `can_reuse()` now looks through `llama_memory_hybrid_context::get_attn()`
and then runs the same checks as for a plain kv cache (`n_kv`, stream count, token count, `kq_mask_derivable()`).  Other
wrappers (iSWA, where the input may belong to the base or the SWA cache) still reject.  `LLAMA_KQ_MASK_DERIVED_REUSE=0`
restores the rejection.  No arithmetic changes.

**Results.** gfx1201 (R9700).  Linux: Ryzen 9 3900X, ROCm 10.0.0, Qwen3.8-27B-UD-Q4_K_XL + Qwen3.8-27B-DFlash2-Q4_K_M,
`--parallel 2 --kv-unified -c 245760 -ub 2048 -fa on -ctk q8_0 -ctv q8_0 --spec-type ngram-mod,draft-dflash
--spec-ngram-mod-n-max 15 --spec-ngram-mod-n-min 8 --spec-draft-n-max 5`.  Windows 11: TheRock ROCm 10, driver
32.0.32015.7029, Qwen3.8-27B Q5_K + the same draft, 1 slot, `-c 131072`.  r156 with our out-of-tree patches switched off
(closest to the release; n-gram drafts first).  A whole-file rewrite of a 7 KB Python file (the model copies most of it,
so most steps verify 16-row n-gram drafts), sent twice, greedy; order on, off, off, on; means of two runs:

| | graphs reused (on / off) | ms per step (on / off) | decode |
|---|---|---|---|
| Linux, first rewrite (227 steps) | 202 / 38 | 67.41 / 70.14 | +4.0 % |
| Linux, second rewrite (194 steps) | 174 / 0 | 72.63 / 75.20 | +3.6 % |
| Windows, first rewrite (225 steps) | 200 / 36 | 64.60 / 66.35 | +2.7 % |
| Windows, second rewrite (217 steps) | 189 / 19 | 64.42 / 66.65 | +3.5 % |

Requests without long n-gram drafts (an ERP-style tool-call turn, an 8K continuation) reuse the same number of graphs
and run at the same speed.

Same text: every request was identical with the switch on and off, in each of: the runs above (also with our patches on);
2 slots with two requests at once (single- and two-sequence batches alternating); 4 slots at `-c 131072` with four
requests at once; Qwen3.6-35B-A3B with its DFlash draft; n-gram drafts only (no model draft); `-ub 512`; the rewrite after
a 100K-token prompt.

**Not covered:** multi-GPU, vision input (2-D positions; the derived form is not taken there), iSWA models (unchanged).

*Written by Claude, an AI model by Anthropic, from measurements on the contributor's machines.*
