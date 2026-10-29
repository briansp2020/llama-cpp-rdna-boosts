# wip/kq-derived-multiseq — the derived kq mask for batches of several sequences

`kq-derived-multiseq.patch`: one `git am` file (one commit), made on `v15-6050b467c-r167` (applied tree `400ae52a`); it applies
unchanged on `v15-6050b467c-r168` (`git am`, tree `78e10270`; r167 -> r168 did not touch these files).  It touches
`src/llama-kv-cache.{h,cpp}`, `src/llama-kv-cells.h` and `src/llama-context.{h,cpp}` (+251 / -16).  No kernel changes.
sha256 `ecd200608258d0d566a0159df7679ca57f687240621a350132139f77cf460937`.

`DIAGNOSTIC-verify-derived-kq-mask-multiseq.patch` (on top of it, +42, **not for delivery**): `LLAMA_KQ_MASK_DERIVED_VERIFY=1`
rebuilds each derived ubatch's packed mask with the packed fill itself and compares the visibility cell by cell - the
oracle the V3 plan used for phase 1, extended to several sequences - and recounts the shared cells against the counter
below.  sha256 `4b164d0b5916953b8142ddfa5aeed9fc3cb2ef5fe542bbeb063115285f0548e4`.

`test-kv-cells-multi.cpp` (not for delivery either): a randomized check of that counter against a recount after every
`llama_kv_cells` operation; from a patched tree, `g++ -std=c++17 -Iinclude -Iggml/include -Isrc test-kv-cells-multi.cpp`.

**Why.** `llama_kv_cache::kq_mask_derivable()` accepts single-sequence ubatches only, so a context with `--parallel > 1`
and `--kv-unified` keeps the packed mask for its multi-sequence ubatches: two slots prefilling together, one slot's
prefill next to another's decode.  And because the reserve measures with `n_seq_max` sequences (`sched_reserve()` ->
`graph_reserve(n_tokens, n_seqs = n_seq_max, ...)`), such a context always reserves the packed mask, `n_kv x n_ubatch x 2`
bytes on the device and again in pinned host memory, even for traffic that never has two sequences in one ubatch.  (The
comment above `kq_mask_packed_reachable()` says the reserve's measure batch is single-sequence; with `n_seq_max > 1` it
is not, which is why the release reserves the packed mask on the server shape below whatever the predicate says.)

**What changes.**

* *Encoding.*  Sequence `k` of the ubatch (`seq_idx`) has its cells (`cell_pos`) and its tokens' `[tok_lo, tok_hi]`
  shifted by `k * 2^23`.  The predicate the kernels evaluate, `tok_lo <= cell_pos <= tok_hi`, then keeps every token
  inside its own sequence's range, and `flash_attn_kq_derived_blocks`' batch-wide window stays conservative (the union of
  the ranges).  `2^23 x LLAMA_MAX_SEQ` fits int32.  One sequence keeps offset 0 and the unbounded non-causal window, so its
  inputs are byte-identical to before.
* *When.*  One KV stream (unchanged), more than 32 tokens, every token in one sequence (`n_seq_id == 1`), no cell
  carrying two of the batch's sequences (`llama_memory_seq_cp`), every position below `2^23`; the M-RoPE degeneracy check
  now covers every sequence of the batch.  Multi-sequence ubatches of at most 32 tokens - several slots verifying drafts
  in one step (2 x 6, 4 x 6, 2 x 16 rows), decode steps of up to 32 clients - keep the packed mask: at that size it is a
  few MiB, so the reserve's slack covers it whichever form the reserve picked, and those small, frequent batches skip the
  checks below.
* *Sharing check.*  `llama_kv_cells` now counts the cells that carry more than one sequence (`get_n_multi_seq()`, kept up
  to date in every operation that changes a cell's sequence set: `rm`, `seq_rm`, `seq_keep`, `seq_add`, `pos_add`, both
  `cp` and both `set` forms, `reset`).  While it is 0 - always, unless something was copied with `llama_memory_seq_cp` -
  the check is O(1); otherwise it scans the used cells for one carrying two of the batch's sequences (about 2 ms at
  245760 cells, in place of an O(n_kv x n_tokens) packed fill).
* *Reserve.*  `kq_mask_packed_reachable()` no longer reserves the packed mask for `n_seq_max > 1` when the multi-sequence
  form is available (a single-stream attention cache, the attention part of a hybrid memory, or both caches of an iSWA
  memory; derived mask on).  Without `--kv-unified` (one stream per sequence) nothing changes: the release already turns
  the derived mask off for a multi-stream KV cache (`resolve_fused_ops`), so that context reserves and uses the packed
  mask as before.  A batch above 32 tokens that does not qualify is detected in `decode()` before it is split,
  latches `kq_mask_multiseq_seen` and re-reserves once with the packed mask, exactly as the 2-D position latch does - so
  no later batch grows the compute buffer (issue #42).
* *Switch.*  `LLAMA_KQ_MASK_DERIVED_MULTISEQ=0` restores the single-sequence rule (and the packed reserve).

## Results

Linux: Ryzen 9 3900X, one R9700 (gfx1201), Ubuntu 24.04.5, ROCm 10.0.0 (TheRock pip); `ROCBLAS_USE_HIPBLASLT=0` in every arm
(GREEDY-PURITY §44).  **S** = r167 (with an unrelated server patch, `wip/backend-sampler-reserve-once`, that does not touch
attention), **D** = r167 + this patch, **O** = D with `LLAMA_KQ_MASK_DERIVED_MULTISEQ=0`.

The patch was built up in steps during the measurements: v1 (reserve decision for plain and hybrid caches), v2 (+ iSWA
memories), v3 (+ the shared-cell counter in place of an unconditional scan), v4 (+ the 32-token floor) = this file.  Each
step only narrows where the derived form is taken or how its condition is computed; the rows say which version they ran.

Server shape: Qwen3.8-27B-UD-Q4_K_XL + Qwen3.8-27B-DFlash2-Q4_K_M, `-c 245760 -ub 2048 -b 2048 --parallel 2 --kv-unified
-fa on -ctk q8_0 -ctv q8_0 --spec-type draft-dflash --spec-draft-n-max 5`.

| | S (release) | O | D |
|---|---|---|---|
| device memory after load (v1; v4) | 28.63 GiB | 28.63; 28.69 GiB | **27.60; 27.66 GiB (-1.03)** |
| server RSS after load, pinned host buffers (v1) | | 2.47 GiB | **1.54 GiB (-0.94)** |
| two 16K prompts sent at once, both slots prefill (v1) | 25.8 / 25.9 s | 25.9 / 26.0 s | 25.1 / 25.1 s |
| two slots verifying at once, ms per step at 8K / 32K, order O D D O (v4) | | 64.9 / 94.1 | 65.6 / 92.4 |
| one-slot 8K greedy continuation (v1) | same text, 32.5 t/s | same | same |

The packed mask at this shape is `245760 x 2048 x 2 B = 960 MiB` on each side, which is what D gives back.  The same
model alone (no draft), `-c 245760 -ub 2048 --parallel 2` (v4 and a v4 build with an extra stream check that turned out to
be dead code): with `--kv-unified` 25.63 -> 24.59 GiB after load; without it 25.11 GiB in O and D alike (the multi-stream
case above); two 16K prompts sent at once gave the same text in all five runs.  The verify row
is noise either way: those 12-row ubatches take the packed mask in both arms in v4.  (v3, which still derived them, gave
65.7 / 94.0 for D against 65.5 / 92.1 for O; the extra M-RoPE degeneracy scan over the batch's cells is what the 32-token
floor keeps off the per-step path.)

**Same results.**

* llama-perplexity, XL, `-c 2048 -b 8192 -ub 2048 --kv-unified` (4 chunks = 4 sequences per batch), 16 chunks of a Chinese
  text (v1): per-chunk PPL identical in O and D, two runs each.  rocprofv3: D launches `flash_attn_kq_derived_blocks` 64
  times and `flash_attn_mask_to_KV_blocks` 16 times (the warm-up decode of a few tokens, which keeps the packed mask), O
  launches only the latter (80): the multi-sequence ubatches did take the derived path.
* The oracle (diagnostic patch), **0 mismatching cells** in every derived ubatch: llama-perplexity (8 ubatches of 4
  sequences), the server shape above (v1: 155 derived ubatches, 129 of them multi-sequence: 12 tokens = both slots
  verifying, 192 / 768 = a prefill next to a decode or two prefills), llama-parallel (`-np 4`, 8 multi-sequence ubatches of
  up to 4 sequences, 18-512 tokens; v4 `-np 8`: 12 multi-sequence derived ubatches, outputs O == D).
* The fallback: llama-parallel `-pps` (the system prompt shared through `llama_memory_seq_cp`) latched once ("a batch of
  several sequences the derived kq mask cannot serve - re-reserving with the packed kq mask"); the next graph
  reallocation reported `changed = 0`, and the output was identical to O's (v1; v3 and v4 the same: one latch, no
  multi-sequence ubatch derived afterwards, output O == D).
* Other caches (v1 / v2), llama-perplexity `-c 512 -b 2048 -ub 2048 --kv-unified` (4 sequences in every ubatch), O vs D:
  **gpt-oss-20b MXFP4** (iSWA: sliding-window and full layers, sinks) - 8 multi-sequence derived ubatches, 0 mismatches,
  per-chunk PPL identical; **Qwen3-30B-A3B Q4_K_M** (a plain attention cache, not hybrid; `split_simple` ubatches) - 4
  multi-sequence derived ubatches, 0 mismatches, per-chunk PPL identical, and the run took 10 s instead of 19 s (the packed
  fill it no longer does).  gpt-oss-20b as a server (`-c 131072 -ub 2048 --parallel 2 --kv-unified -fa on`): device memory
  after load 15374 -> 14799 MiB (-575); llama-parallel `-np 8` (16 multi-sequence derived ubatches, 0 mismatches) gave the
  same output in O and D.
* Decode-sized multi-sequence ubatches (v1, before the 32-token floor, so they were derived): llama-parallel `-np 16` on
  the 27B (up to 16 one-token sequences per ubatch, a deterministic batch composition) gave the same output in O and D.
* The counter (v3 / v4): in every derived ubatch the diagnostic's recount equalled `get_n_multi_seq()`; with `-pps` the
  latch itself shows the counter non-zero while the prompt was shared (at 0 the check would have passed); and
  `test-kv-cells-multi.cpp` ran 400000 random operations of every kind on 512 cells x 6 sequences with the counter equal to
  the recount after each one, shared cells present in 96 % of the states.

### Windows, and host-resident experts with the expert-cache arena

Windows 11, Ryzen 7 5700G, one R9700, TheRock ROCm 10, driver 32.0.32015.7029; the patch (and the diagnostic) applied
unchanged on `v15-6050b467c-r164`, with `ROCBLAS_USE_HIPBLASLT=0`.  Memory is the process's dedicated / shared GPU memory
(Windows performance counters).

| | O | D |
|---|---|---|
| llama-perplexity, XL, 4 sequences per batch, 8 chunks | 4.0521 | **4.0521, every chunk equal**; 8 multi-sequence derived ubatches, 0 mismatching cells, counter = recount |
| server, XL alone, `-c 245760 -ub 2048 --parallel 2 --kv-unified`: dedicated / shared after load | 25.58 / 1.26 GiB | **24.54 / 0.32 GiB (-1.03 / -0.94)** |
| same, two 8K prompts at once, 128 greedy tokens each | | same text |

**`-ncmoe 99` with the expert-cache arena (auto)**, Ornith-1.5-35B-A3B Q5_K_M (a 35B-A3B MoE, 40 layers x 256 routed
experts, every routed expert host-resident and served through the arena), same server shape, two 8K prompts at once,
128 greedy tokens each; starts in the order O D, then D O:

| | O | D |
|---|---|---|
| slab at load | work 1.75 GiB + arena 24.88 GiB | **work 0.69 GiB + arena 25.94 GiB** |
| arena residency | 21800 of 21800 MiB (100 %) | the same: this model's experts fit either way |
| shared GPU memory after load | 22.47 GiB | **21.54 GiB (-0.94)** |
| the pair, wall time (two starts) | 31.0 / 27.1 s | **9.7 / 10.1 s** |
| slot 0's 8000-token prompt, slot 1 prefilling alongside (the D O starts) | 15.2 s (526 t/s) | 5.6 s (1426 t/s) |
| text | `2e079f8c3d` in both starts | `2e079f8c3d` in the start without the oracle |

The freed work region goes to the arena (here it has nothing left to hold; a model whose experts do not fit would get
1 GiB more of them resident).  The pair's prefill ran 2.7x faster: the multi-sequence prefill ubatches take the packed
mask in O.  We did not profile where the packed form spends that time on this config; it is the same kind of gap as
Qwen3-30B-A3B's llama-perplexity above (19 -> 10 s), on a model whose own compute per ubatch is small, and a Windows
number - our Linux runs of the dense 27B saw 3 % on the pair.  The D start with the oracle on batched the pair
differently (the oracle's host work shifted when the second request joined: one multi-sequence ubatch, 0 mismatching
cells) and produced another text, as concurrent pairs do from start to start; the three starts without it agreed.

**Not covered:** non-causal attention, a live 2-D M-RoPE batch with several sequences, `-sm tensor` / multi-GPU (we have
one GPU), a host-expert model whose experts do not fit the arena (where the freed GiB would become residency), the CPU
backend (a CPU-only run, Mistral-7B `--device none`, never took the derived path in either arm, so it was not exercised),
positions >= 2^23 (falls back by construction).

*Written by Claude, an AI model by Anthropic, from measurements on the contributor's machines.*
