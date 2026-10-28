# llama: reserve a DFlash draft's KV injection and noise-block graphs at their own widths

> **Re-base status (2026-10-28):** cut against the current fork point `6050b467c`; applies clean there
> (`git apply --check`).  Not part of the delivery (nothing here blocks a release).

**Status:** upstream-PR candidate, cut against `6050b467c`; **in the delivery as block 12** (r162, issue #239).
`UPSTREAM-PR-dflash-draft-reserve.patch` applies clean to unadulterated `6050b467c`.

## What it is

A DFlash/DFlash2 draft context alternates two graph shapes by batch kind:

* the target-feature **KV injection** (`ubatch.embd`, or the F1 `lf_dev_inject` token rows) — up to
  `n_ubatch` rows;
* the **noise-block decoder** — at most `n_seq × block` rows.

`sched_reserve()`'s measure graph is a generic token ubatch at `n_ubatch`, which selects the decoder branch,
and that branch applies the lm_head to **every** row (DFlash2's selector reads every block position; it has
no `build_inp_out_ids`).  So `result_output` is reserved as `[n_vocab, n_ubatch]` f32 — 1940 MiB at
`-ub 2048` on a 248k vocabulary — for a width no token batch ever reaches.  It is the whole reason the draft
context's compute buffer is ~2.4 GiB at `-ub 2048`.

This patch:

* adds `llama_batch_allocr::ubatch_reserve(..., with_embd)`, which marks a measure ubatch as an embedding
  batch (a non-null `embd` marker; nothing reads the rows during a reserve);
* adds `llama_context::graph_reserve(..., reserve_injection)`, which builds the measure graph through the
  injection branch;
* for a DFlash draft (`model.arch == LLM_ARCH_DFLASH && cparams.ctx_other != nullptr`) reserves the injection
  at `n_ubatch` and the decoder at `n_seq_max × max(dflash.block_size, n_outputs_max_per_seq)`, then leaves
  the layout at the injection graph.

Both shapes are reserved into the same scheduler, so the buffer keeps the larger of the two: neither can
grow it mid-run (the failure mode issue #42 records).  The width bound uses `dflash.block_size` (lilicorr
decodes a whole block) and `n_outputs_max_per_seq` (DSpark/DFlash decode `n_max + 1` rows per sequence);
a model that does not declare `dflash.block_size` falls back to the default 16.

## Why it matters

The draft context is created before the target's first prefill, so this memory is outside the `--fit` /
slab estimate (the draft compute tier there is 512/768 MiB).  On a 16 GiB card the over-reserve alone can be
the difference between fitting a long-context DFlash draft and not.  The fix is a pure reserve change: no
kernel, no arithmetic and no graph built at runtime changes.

## Validation (delivery tree, gfx1201 = RDNA4, 1 and 2 GPUs)

27B UD-Q5_K_M + DFlash2-Q4_K_M, `-ub 2048`, q8_0 KV, F1 on unless stated:

| check | result |
|---|---|
| draft `sched_reserve` | device compute **2444.41 → 264.15 MiB (−2.18 GiB)**, free VRAM 8622 → 10802 MiB; host compute 40.18 → 200.14 MiB (the injection's F32 feature input) |
| greedy text, `-n 200`, seed 42 | 1 GPU `941 chars sha=c62b6901495a`; 2 GPU `-sm tensor` `899 / 97fb92617a9b`; `LLAMA_DFLASH_DEV=0` `592 / 6e3fd559c6a8` — **byte-identical before/after** |
| server, 1 slot unified, `ngram-mod,draft-dflash`, deterministic run | identical content and draft counts (189 generated / 88 accepted) before/after |

The delivery also has `LLAMA_DRAFT_SCHED2`, which moves the decoder to a second scheduler; this patch
reserves the decoder width on the main scheduler, which is what upstream needs (one scheduler) and is a
harmless upper bound there.

## What was NOT validated

* Not built or run on pristine upstream master; the identical logic is built and validated in the
  delivery's block 12.  A PR branch should re-run the two greedy-text arms at the base.
* Only DFlash2 was available here.  Plain DFlash (no selector) and DSpark share the branch and the bound,
  and lilicorr's full-block decode is covered by `dflash.block_size`, but none of those models was run.
* Not tested with a vision/embd target batch, nor with the draft's host injection path under `-sm tensor`
  beyond the `LLAMA_DFLASH_DEV=0` text arm above.
* No non-AMD backend was exercised (the change is backend-agnostic `llama_context`/`llama_batch` code).
