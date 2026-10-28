# todo70-rearm-wideaux — a re-arm is layer-uniform or the layer is stood down, and a starved wide-aux growth is decided once per wide pass

Closed **TODO #70** (opened 2026-10-19 while fixing r87).  Release `v15-6050b467c-r168`
(r168), 15 blocks, **blocks 09 and 10 amended**.  No kernel, dispatch row, numerics path or
environment variable changed, so the prefill-logit gate reads `0.000000 / 100.000 %` and every
gate's greedy text is unchanged.

## What #70 was

r86's field failure (2x R9700, Flash-Next IQ4_NL, `-sm tensor -ncmoe 46 -b/-ub 6144`, MTP
adaptive n6, `--cache-idle-slots`): the second and later chat requests decoded at half speed and
request 5 died with `HSA_STATUS_ERROR_MEMORY_FAULT` in `k_get_rows_float`.  The record
(`WORKLOG.md` 261019.002) noted the shape at the end: a re-arm that had to shrink one layer to a
partial slot count (`layer=47 ... 344 requested, only 202 fit`) with the wide-aux region starved,
then a fault on the next graph.  r87 fixed the *trigger* (a wide -> narrow compute-layout
transition now always drops, so the arena is restored at every prefill -> decode transition), and
`#70` asked for the two hardening halves so a future starvation sequence cannot reach a fault.

## 1. A re-arm is layer-uniform or the layer is stood down (block 09, `moe-expert-cache.cu`)

`moe_cache_rearm()` re-allocates the tables a slab boundary move evicted, to the slot count the
sizing pass settled on (`g_rearm_slots[layer]`).  Its retry loop could exit with a **mixed-slot
layer**: on the last attempt it set `target = min(achieved)` and the loop ended, leaving the
tables it had just allocated at their *own* counts (`ffn_up` at 202, `ffn_gate` at 344).  The
sizing pass forbids that state -- the gate+up fused kernel seeds the gate lane from the **up**
table's remap -- and answers it by disabling the cache outright.  A re-arm must not do that (the
graphs are already built against the tables, and a late `g_enabled = false` desynchronises the MTP
draft from the target), so the answer is local to the layer:

* `moe_cache_rearm_layer_locked()` converges a layer on ONE count (retrying at the largest count
  any table reached, halving when a table cannot hold even one slot) and, if the retries run out,
  stands the **whole layer** down -- every table of it at 0 slots, never a mixed layer.
* A layer that fits a smaller uniform count is accepted with a warning: a uniformly smaller
  arena is a legal state (it is what the sizing pass decides when the budget is short).
* A final sweep re-checks every touched layer and stands down any that is not uniform -- the
  sizing pass's safety net, made **local** instead of a whole-cache disable.

A stood-down table makes `moe_cache_has_arena_locked()` false (the fusions stand down for every
table) and every per-table consumer declines, which is the fuse-guard invariant.

**Latent, not live:** today's fusion guard (`moe_cache_table_serves`) additionally requires the
table to be `identity`, i.e. fully resident, so the fused gate+up path cannot run against a
partially resident pair even if the layer were mixed.  The hardening closes the state anyway --
the guard that saves us is not the invariant the code documents, and the re-arm's own contract
("layer-uniform and safe") should hold on its own.

## 2. The wide-aux growth is refused once per wide pass (block 10, `ggml-cuda-slab.cu`)

`ggml_cuda_slab_wide_aux_alloc()` (the FA prefill staging arena's home in the slab) is called on
every FA op of a prefill.  A refusal was returned per allocation, so a starved region turned into
hundreds/thousands of decisions, each of which can evict a band of arena tables -- or, worse,
evict one and *then* refuse on a boundary move that raced the MTP draft context -- while a graph
that still held the previous placement was queued.

The two refusal classes are **not** the same decision, and treating them the same would be wrong in both
directions:

* a **VA-limit** refusal (`target > aux_limit`: the pinned narrow-2 region, or the top of the mapping) is
  structural -- the limit cannot grow within the pass, and any later grow only raises the extent -- so it is
  latched;
* a **snapshot-race** refusal (the boundary moved while `g_slab_mutex` was dropped for the eviction) is
  recoverable: the racing thread is the MTP draft context's reserve and it has normally finished by the next
  request, so a retry succeeds.  It is retried, but bounded to `GGML_CUDA_SLAB_WIDE_AUX_RACE_MAX` (4)
  attempts per pass, because every attempt costs another arena-band eviction.  (A blanket latch here would
  write a pass off after one benign race; an unbounded retry is the pre-r168 behaviour that faulted.)

So `wide_aux_denied_want` records the smallest request refused *for the rest of the pass* (a latched VA-limit
refusal, or a race that exhausted its retries) and any request at least that large is refused without
touching the layout again.  It is cleared

* by `ring_set()` on a pass transition (arm or drop) -- the next wide pass gets its own decision,
* by a boundary **shrink** in `work_alloc` (the VA above the boundary grew, so a grow may fit),
* and on slab (re-)init.

A smaller request than the refused one is still allowed (a grow only ever raises the extent, so a
smaller request is the only thing that can fit), so the pass is only written off when the reason cannot
change before the pass ends.  One `SLAB_WIDE_AUX ... refused for this wide pass` WARN is printed per pass
(not gated on `GGML_CUDA_SLAB_STATS`), and the per-pass refusal count is reported as `aux_refused=` in the
`SLAB_STATS` line.  The `SLAB_WIDE_AUX` growth
diagnostic itself is unchanged -- it is the tool the campaign used to size the region.

**Not reproducible on the current tree, by design:** the r86 shape had the wide compute layout
never released, so the boundary sat at the 6.9 GiB wide view and the region had nowhere to go.
r87 forces the drop, so no config on this box starves the region: measured `aux_refused=0` in
most prefill passes and `1-2` in a pass whose first FA op ran before the ring armed (the cheap,
deliberately unlatched `ring_armed=0` path).  The latched refusals are therefore validated by
code review plus the healthy-run no-op below, exactly like the other "not reachable on the
maintainer's configs" items in the rulebook.

## 3. Two further defects found while validating

* **`stand_down_table_locked()` kept a slab-backed arena pointer after returning the range.**  A
  slab stand-down hands the range back to the slab's free list
  (`ggml_cuda_slab_arena_free`), where the work pool or another table can take it, but the table
  kept `t.arena` (and `t.arena_reserved`) on the theory that "a VMM arena keeps its reserved VA".
  `alloc_table_locked()` reads a non-null `t.arena` as *already allocated* and reuses it at the
  requested slot count, so a **re-arm of a table an allocation-OOM stand-down had dropped claimed
  slots in a range it no longer owned**.  Seen live as
  `[validate device-malloc-yield] ... layer=46 slots=0..512` immediately before an eviction and a
  re-arm of 162 tables (`moe_cache_shrink_step` stands ONE table down on purpose, so the mixed
  layer itself is legitimate).  Fixed by clearing `t.arena`/`t.arena_reserved` there, exactly as
  `free_table_buffers_locked()` and `moe_cache_evict_slab_range()` already do.
* **The eviction hook did not invalidate captured graphs.**  `moe_cache_evict_slab_range()` frees
  table arenas through `ggml_cuda_slab_arena_free` directly, bypassing `free_arena_backing()`,
  which is where the `ggml_cuda_graph_mem_freed()` bump lives -- and a captured graph bakes a
  table's arena address into the MoE kernels' launch parameters (a raw device pointer, not a
  tensor source, so `ggml_cuda_graph_update_required` cannot see it).  A wide prefill always ends
  in a boundary shrink that bumps anyway, so this was covered in practice; a **narrow** boundary
  move (a wider verify view, a wide-aux grow) evicts with no other bump.  The hook now bumps when
  it releases anything.
* `MOE_EXPERT_CACHE_VALIDATE=1` now reports layer slot uniformity: `MIXED-LAYER ... (a stood-down
  role; the served tables still agree)` as a WARN, and `NONUNIFORM-LAYER ... (two DIFFERENT
  non-zero counts)` as an inconsistency.  That is the check that would have named the r86 layer
  directly instead of inferring it from generated text.

## Measurements (this box: 3x R9700 gfx1201, 2 used, `HIP_VISIBLE_DEVICES=0,2`)

Two builds of the **same** tree: `build-r167ref` (r167, `38aac1e49`) and the patched tree.

| # | config | result |
|---|---|---|
| A | dense Qwen3.5-4B Q8_0, 1 GPU, `-n 48 --seed 42 --temp 0` | output identical (only the load spinner differs) |
| B | Flash-Next GSQ-IQ3_XXS, `-sm tensor -ncmoe 48 -c 32768 -ub 8192 -b 8192 -ctk q8_0 -ctv q8_0`, auto cache | 288/288 tables resident, arena 40 925.0 MiB; **5 evictions, re-armed 12** tables; identical text |
| C | as B + `GGML_CUDA_SLAB_RESERVE_MIB=18000` (slab mapped 11 264 MiB) | arena 14 227.8 MiB = **34.8 % residency**; 6-7 evictions (~8.6 GiB); `re-armed 162 ... (0 stood down)`; `validate after-rearm: 288 resident, 0 down, inconsistent=0`; identical text |
| D | as C + MTP draft (`-md mtp-...Q8_0.gguf --spec-type draft-mtp`), `-c 16384 -ub 4096` | narrow-2 pinned (draft compute 0.50 GiB, arena tops out at 8.06 / 7.56 GiB); 7 evictions; `re-armed 150` (0 stood down); draft acceptance **0.493**; no abort |
| E | MTP **Protocol A** dense 27B UD-Q4_K_XL, 3 GPU `-sm tensor`, `--spec-type draft-mtp`, `-n 3000` | generation **100.8 t/s**, draft acceptance **0.60816** (1937/3185), mean len 2.82 -- the recorded r167 reference is 90.7 t/s / 0.60 |
| F | `scripts/gate-prefill-logits.sh` on the final build `9cb468e55` | mean KLD **0.000000**, same-top-p **100.000 %** PASS |
| G | `scripts/check-gpu-faults.sh --since 15:00` | clean (every run above) |

Text comparisons in B/C were made with **`ROCBLAS_USE_HIPBLASLT=0`**.  Without it the same two
builds produced *different* text on arm C (576 vs 580 chars) with an identical arena, identical
eviction/re-arm counts and identical acceptance -- the documented per-process hipBLASLt solution
choice (`GREEDY-PURITY.md` §44), confirmed here on a third config.  A cross-build text comparison
on this box is only meaningful with it set.

## The measurement trap this campaign hit first

`scripts/extract-generated.py` slices a `llama-cli` log between the `> ` prompt echo and the
`[ Prompt:` footer, so on a **`-v`** run its hash covers the prompt echo as well.  Two builds
"matching" that way proves nothing about the generation; two builds differing that way may only
differ in the trailing log lines.  Compare the *clean* (non-`-v`) log, or diff the whole filtered
output.

## Files

* `ggml/src/ggml-cuda/moe-expert-cache.cu` (block 09): `moe_cache_rearm_layer_locked()` (new),
  `moe_cache_rearm()` (transactional), `stand_down_table_locked()` (clear the arena),
  `moe_cache_evict_slab_range()` (bump the graph generation), `moe_cache_validate()` (layer
  uniformity).
* `ggml/src/ggml-cuda/ggml-cuda-slab.cu` (block 10): `wide_aux_denied_want` / `wide_aux_denied_n`,
  the latch in `ggml_cuda_slab_wide_aux_alloc()`, the clears in `ring_set()` / the boundary
  shrink / `slab_init`, and `aux_refused=` in the `SLAB_STATS` line.
* No new environment variable; `ENVIRONMENT.md` documents the new `SLAB_STATS` field.  The race-retry bound
  is a `#define` (`GGML_CUDA_SLAB_WIDE_AUX_RACE_MAX`), not a knob: it is not a tuning axis.

## Why a uniformly smaller re-arm count is kept (and never retried later)

The re-arm's target is re-read from `g_rearm_slots[layer]` at **every** drop and nothing ever lowers that
value, so there is no arithmetic ratchet and no "fragmentation down to the small layer size".  A layer that
converges on a smaller uniform count is not revisited (only 0-slot tables form the work list, which was
already true before this change), and that is deliberate: the re-arm runs at the prefill -> decode drop, i.e.
with the arena at its per-request maximum, so a shortfall means a later context's KV or compute layout really
did take the bytes -- and those never give them back.  Retrying such a layer on every drop would zero its
residents (the re-arm rebuilds a layer from scratch) for a count it cannot hold.  In every measured arm the
post-drop arena returned to exactly the sizing value and the recorded counts fitted (`re-armed 162`, `0 stood
down`), so the smaller-uniform branch never fired.

## Was the graph invalidation a correctness bug before?

Yes, latent.  A captured graph bakes a table's arena pointer into the MoE kernels' launch parameters (a raw
device pointer, not a tensor source, so `ggml_cuda_graph_update_required` cannot see it).  Freeing that arena
while the capture stays live means the next replay reads memory the slab has handed to the work pool or to
another table: a wrong answer or a VM fault.  Every *wide* prefill is followed by a boundary shrink that
bumps `ggml_cuda_graph_mem_freed` (the issue #187 fix), so the ordinary sequence was covered; a **narrow**
boundary move (a wider verify view, a wide-aux grow) evicts with no other bump, and no repro was obtained on
this box.  Measured cost of the added bump: **zero** in every arm here (it fires in the prefill band, where
the drop's shrink bumps anyway; text and eviction/re-arm counts unchanged).  A narrow-band eviction would
cost one recapture round (each graph is recaptured on its next use), which is the price of not replaying a
dangling pointer, and `GGML_CUDA_GRAPH_MEM_GEN=0` remains the escape hatch.
