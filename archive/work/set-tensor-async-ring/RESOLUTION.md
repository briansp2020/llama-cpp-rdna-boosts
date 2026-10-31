# RESOLUTION — `set-tensor-async-ring` (PR #194) — DECLINED at r98, CLOSED 2026-10-31

**Status: reviewed 2026-10-20 and NOT promoted; re-assessed on r186 (2026-10-31) and the PR is CLOSED as
not planned** — see "Final assessment" at the bottom, which supersedes the r98 attribution below.  The
submitted patch plus README are kept here because the diagnosis is worth keeping.

## What was proposed

Keep r90's `set_tensor_async` snapshot guarantee (the copy reads bytes taken when the upload is issued) but
take the snapshot into a pinned two-half ring owned by the backend context and issue the copy on
`cuda_ctx->stream()`, so the upload no longer joins the graph-input upload ring's queue.

## What was verified

* The r90 fix survives: `llama-perplexity --kl-divergence-base` self-comparison (27B UD-Q4_K_M, `-c 512`,
  `-b 8`, 24 chunks, `GGML_CUDA_ASYNC_INPUT=1`) reads **100.000 %** at `-ub 8` and at `-ub 1`, same as the
  r97 base.
* Greedy text is byte-identical to r97 in every config tried (dense 27B at 8K and 110K, 27B 2 GPU tensor,
  35B-A3B 2 GPU tensor, Flash-Next 2 GPU tensor, gemma4 12B).  No GPU faults.

## Why it was declined

It is a **regression on this tree**, because of one line: `ggml_cuda_async_input_drain(cuda_ctx->device)`
before the copy drains the whole graph-input queue, so queued uploads are issued one at a time
(`cudaMemcpyAsync` on `cudaStreamPerThread` plus an event record each) instead of being folded into one
`ggml_cuda_ain_batch_kernel` launch (PR #182).  The `-ncmoe` host-expert uploads use that same ring.

3 interleaved pairs each, seed 42, temp 0, `-n 400`:

| config | r97 | r97 + PR #194 |
|---|---:|---:|
| Flash-Next GSQ IQ3_XXS + shared Q8_0 MTP, 2 GPU tensor, `-ncmoe 10`, q8_0 K/V | 131.4 t/s | 129.8 t/s (-1.2 %) |
| 35B-A3B UD-Q4_K_M, 2 GPU tensor, `-ncmoe 10`, q8_0 K/V, MTP n3 | 184.6 t/s | 180.3 t/s (-2.3 %) |
| 27B UD-Q4_K_XL, 1 GPU, plain, 8K | 29.35 t/s | 29.2 t/s (neutral) |
| 27B UD-Q4_K_XL, 2 GPU tensor, MTP n3 | 95.0 t/s | 95.1 t/s (neutral) |

Mechanism confirmation: with `GGML_CUDA_ASYNC_INPUT_BATCH=0` (per-copy path everywhere) both builds read
**exactly 129.1 t/s** on the Flash-Next cell.

Two further notes given to the submitter: the per-context ring takes no lock (the per-device ring it
replaces takes `d.mtx`), and `cudaEventSynchronize` on a half switch is a host-blocking wait inside the
upload path that r90's ring does not have.

## Re-entry conditions (r98)

Scope the drain to the destination-overlap case only (`ggml_cuda_async_input_set` already does exactly
that) and re-run this table plus the self-KLD control.  Nothing else in the patch needs to change.

---

## Final assessment — r186, 2026-10-31 (PR CLOSED)

The submitter never replied (no comment, no new commit; the branch stayed at `325224590`).  One last pass
re-cut the patch on the current tree (it applies cleanly to `8f9dad0be`, offsets only) and also built the
r98 re-entry condition, i.e. the drain scoped to a destination overlap with the copy **fenced** on the
event that issue recorded (issuing only the overlapping origins, so the rest of the queue keeps batching).
Both arms were measured against the same baseline, from the same build directory (bin dirs kept per arm).

**Set-up.**  gfx1201, ROCm 10.1.0, one job at a time, `ROCBLAS_USE_HIPBLASLT=0` on every fresh process,
3 interleaved pairs, seed 42, temp 0, `prompts/prose-rdna-boosts.txt`, `-n 400`, `llama-cli -lv 3`.

| config | base | PR as submitted | PR + scoped drain |
|---|---:|---:|---:|
| Flash-Next GSQ-RCO-IQ3_XXS + shared Q8_0 MTP, 2 GPU `-sm tensor -ncmoe 10`, MTP n3, q8_0 K/V | **101.12** | 99.65 (**-1.5 %**) | 99.27 (**-1.8 %**) |
| 35B-A3B UD-Q4_K_M, 2 GPU `-sm tensor -ncmoe 10`, MTP n3, q8_0 K/V | **199.66** | 199.15 (-0.3 %) | 196.95 (-1.4 %) |
| dense 27B UD-Q4_K_XL, 1 GPU, plain, 8K | 30.69 | 30.66 | 30.72 |

**Correctness holds** in every arm: greedy text byte-identical across all three arms on the Flash-Next
cell (1775 chars `52ccd3394a63`), the 35B-A3B cell (1401 chars `aaef5122fe19`) and a dense 4B cell (889
chars `fbb8c4ac4d98`); no GPU faults.  The r90 snapshot guarantee is not at risk, exactly as the
submitter measured.

**The r98 attribution was wrong, and this pass corrects it.**  The drain is the *trigger*, not the cost:
scoping it does not recover the loss (it is slightly worse on the 35B-A3B cell, where the extra
compute-stream fence and a queue that no longer drains early are both paid).  The cost is that these
uploads leave the graph-input ring's **batched** path (r89 / PR #182) and become one `cudaMemcpyAsync`
per upload.  Isolated with the same build and only `GGML_CUDA_ASYNC_INPUT_BATCH=0` on the Flash-Next cell:
`base` 101.12 · `PR` 99.65 · `base + BATCH=0` (the per-copy path the PR replaces) **96.96** ·
`PR + BATCH=0` 98.91.  So the submitter's `+2.7 %` is real — it is measured against a tree whose other
host-side work drains that queue — but it lands *between* the per-copy and the batch path, and the batch
path is what this delivery ships.

**Two defects in the patch itself** (found while building the scoped arm):

1. **The pinned ring can be freed under an in-flight copy.**  `snap_half_ev[h]` is recorded when the ring
   *leaves* half `h`, so the half currently being filled has copies that no event covers; the destructor
   synchronises the two stored events and then `cudaFreeHost(snap_ring)`s a buffer an H2D copy may still
   be reading.  Fix (used in the scoped arm): record `snap_half_ev[snap_half]` on `stream()` before the
   syncs, or synchronise the stream first.
2. **No lock on the per-context ring**, where the per-device ring it replaces takes `d.mtx`; and
   `cudaEventSynchronize` on a half switch is a host-blocking wait inside the upload path (r98 noted
   both).

**Outcome.**  CLOSED as not planned (2026-10-31) with the table above on the PR.  What would reopen it is
a measurement showing the queue is actually drained between upload and graph **on a delivery config**
(`GGML_CUDA_ASYNC_INPUT_DEBUG=1` / `rocprofv3` counting per-token drains that issued queued uploads) — and
the fix for that is to stop whatever drains it, not to bypass the ring, because both forms of this change
land below the batch path.
