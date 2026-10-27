# RESOLUTION -- pipelined batched state save / restore (PR #233) promoted into block 12, r151

**Status:** promoted 2026-10-27 in `v15-6050b467c-r151` (block 12).  The submission is preserved verbatim in
this directory (`README.md`, the format-patch the PR carried).  The release record is `WORKLOG.md`
261027.002.

## What was reviewed

The patch pipelines the batched checkpoint copy that r96 (PR #189, save) and r119 (PR #209, restore) built:
the host memcpy into/out of the pinned `state_stage` and the device copies ran back to back, and the pieces
are now cut into up to `LLAMA_STATE_PIPE_GROUPS` (default 8, `1` = the old copy) size-balanced groups.

Code review, in the order the failure modes matter:

* **Stage ranges.**  A restore packs group g into `[off_g, off_g + size_g)` and queues its async sets, then
  packs group g+1 at a **different** offset, so a queued copy is never overwritten by the next group's
  memcpy.  The `off = off_g` reset before the queueing loop is what makes the pack/queue pair cover the same
  range; it is ugly but correct (and the groups are monotonic, so the accumulated `off` ends at the same
  place).  A single shared `off` would be wrong, which is why the submitter's form is kept as-is.
* **Save events.**  One event **per (backend, group)** is recorded after that group's `get_async` calls are
  queued and synchronized before the group is copied out of the stage.  Distinct events are required (a
  re-recorded event moves past the earlier one), the planning loop creates them before any copy is queued,
  and the `piped` flag short-circuits the whole planning loop if any `ggml_backend_event_new` fails.  Every
  piece belongs to exactly one group, so the per-group event syncs replace the single `llama_state_sync_all`
  completely (which is why the unpiped path still calls it).
* **Backends without events.**  `ggml_backend_event_new` returns NULL when the *device* iface has no
  `event_new` (the CPU backend, BLAS, OpenCL, OpenVINO, ...), and the code then takes the old single-sync
  path for the whole copy.  Every backend that DOES expose `event_new` also fills `event_record` /
  `event_wait` in its backend iface (checked across CANN, hexagon, Metal, RPC, SYCL, Vulkan, WebGPU, the
  meta backend), so `ggml_backend_event_record`'s `GGML_ASSERT` is not reachable.  On this box the meta
  (tensor-split) backend proxies events to its simple devices, so `-sm tensor` pipelines on both sides.
* **Stream identity.**  On HIP the async getters/setters and `event_record` all use the backend's main
  stream (`cuda_ctx->stream()`), so the event really does cover the copies it is recorded after.  A copy
  stream would have been a silent race (the class of bug the r62 note in `AGENTS.md` records).
* **Teardown.**  The events are cached per backend for the context's lifetime and freed in
  `~llama_context()`.  No copy can still be in flight when the free runs (both directions synchronize before
  returning), and `ggml_backend_event_free` only needs the device, which outlives the context.
* **Fallbacks are unchanged**: no context, an unresolvable tensor backend, a state above
  `LLAMA_STATE_ASYNC_MAX_MIB`, `LLAMA_STATE_ASYNC=0`, `LLAMA_STATE_ASYNC_READ=0` and `LLAMA_STATE_PIPE_GROUPS=1`
  all keep the pre-existing path.  The new allocations are four small `std::vector`s on a path that already
  allocated two.

Two submission nits were noted to the submitter (neither affects the applied tree as folded):

* the format-patch carries an **empty `wip/state-pipe/README.md`** in the commit body, so `git am` of it as
  submitted creates a stray empty file (the delivery fold applies `src/llama-context.{cpp,h}` only);
* `n_state_pipe_events` is a `static constexpr` member of `llama_context` even though only the file-local
  group splitter reads it.  Left as submitted (it is the bound of a declared member too).

## Reproduced (2x/1x R9700 gfx1201, ROCm 10.1, r150 release tree vs r150 + the patch)

`scripts/gate-state-pipe.sh` (written for this promotion; nothing else reaches the batched state path -- the
file-based `POST /slots/N?action=save|restore` goes through `llama_io_write_file`/`read_file`, which call
`ggml_backend_tensor_get`/`set` per tensor).  Prompt-cache restore of a 5246-token hybrid state,
`prompts/prose-rdna-boosts.txt` + a 3-token suffix, medians of 5 cycles, `prompt_ms` of the restored request:

| config | `PIPE_GROUPS=1` (old) | default (8) | delta | server save+load trace |
|---|---:|---:|---:|---:|
| Flash-Next GSQ-RCO IQ3_S + shared Q8_0 MTP head, 2 GPU `-sm tensor -ncmoe 40` | 134.5 ms | **129.7 ms** | **-3.6 %** | 81.7 -> 78.4 ms |
| 35B-A3B UD-Q4_K_M, 1 GPU `-ncmoe 10` | 69.4 ms | **67.4 ms** | **-2.9 %** | 44.9 -> 39.9 ms |
| 35B-A3B UD-Q4_K_M, 2 GPU `-sm tensor -ncmoe 10` | 74.4 ms | 74.1 ms | within noise | - |
| Flash-Next arm, `LLAMA_STATE_ASYNC=0` (pre-r96/r119 reference) | 168.5 ms | | | 115.5 ms |

The 1 GPU arm is now **below** the pre-r119 synchronous path (71.3 vs 67.4 ms), i.e. pipelining removes the
single-device staging cost the r119 record accepted as a limitation.  The submitted numbers (G1 72.6 -> 66.8,
G1L 79.6 -> 77.4, G2 37.7 -> 32.4, G3 41.2 -> 38.3 on r148) are their box and their checkpoint sizes; the
direction and the ~5 ms Flash-Next magnitude reproduce here, their 2 GPU 35B-A3B gain does not (ours is
inside the noise of that arm).

## Gates

* `scripts/gate-state-pipe.sh`, both the 1 GPU and the Flash-Next + MTP 2 GPU arms: **every** restored cycle
  of **every** arm byte-identical to `LLAMA_STATE_ASYNC=0`; the save bytes isolated by
  `LLAMA_STATE_ASYNC_READ=0` (`read1` == `sync`) and the restore bytes by `pipe` == `read1`; every cycle
  really restored (`prompt_n` 10 of 5256); no abort; `check-gpu-faults.sh` clean on all four logs.
* `test-llama-context` (all 15 state cases) on 35B-A3B 1 GPU and 2 GPU tensor: PASS/FAIL set identical to
  the **stock r150 build** in every arm (the failures are `rollback` on 1 GPU and `rf` + `rollback` on 2 GPU,
  a slab/VRAM property of this box present in r150 too).
* Coherence: 4B Q8_0 3 GPU tensor `b065f37593b3`, 35B-A3B 2 GPU `-ncmoe 10` `6229d4e6ce30` -- identical to
  r150 (`ROCBLAS_USE_HIPBLASLT=0`, `extract-generated.py`; without the getter pin a fresh-process text
  comparison is meaningless, AGENTS.md).
* `scripts/gate-prefill-logits.sh`: mean KLD **0.000000**, same-top-p **100.000 %**.
* Per-block build `--fast --changed-from f0060e159`: block 12 and the tip build (2/2).
* `scripts/validate-set.sh`: strict 15/15, applied tree `e13514007`.

## Where it lives

Block 12 (`src/llama-context.{cpp,h}`) with a `Fold (r151, PR #233, briansp2020)` paragraph in the commit
body, `LLAMA_STATE_PIPE_GROUPS` in `ENVIRONMENT.md`, the block-12 bullet in `patches/README.md`, the
verification row in `MANIFESTS.md`, the invariant in `AGENTS.md`, and `scripts/gate-state-pipe.sh` (indexed in
`scripts/README.md`) as the state-path gate.

## Finding recorded, not fixed (no TODO filed)

The **file-based** slot save/restore (`POST /slots/N?action=save|restore`, `--slot-save-path`) still pays one
pageable `temp_buffer` + `ggml_backend_tensor_get`/`set` per tensor (`llama_io_write_file` / `read_file`), so
none of the r96/r119/r151 work applies to it.  Routing it through the same stage would be a separate change
with its own gate (the file format interleaves metadata between the pieces, so it needs the same
deferred-flush shape rather than a memcpy of the whole stage).  Not measured against the in-memory path here;
the statement is structural (it is a different code path that never touches `state_stage`).
