# State device path — closed (r157, TODO #110)

The `LLAMA_STATE_SEQ_FLAGS_ON_DEVICE` save/restore path (`llama_io_write_device` / `llama_io_read_device`,
block 12) copied every piece through `ggml_backend_tensor_copy`, which on a meta (tensor-split) buffer falls
back to a pageable `malloc` + a synchronous D2H + a synchronous H2D per piece.  A 1024-token, 2-slot state
(213.6 MiB, 128 pieces) measured **2.23 s to save / 2.28 s to restore** on 2 GPUs against **3.0/3.2 ms** on
one GPU and **21.9/38.9 ms** for the same bytes over the host path — the "fast" device path was 58x slower
than the path it exists to beat.

Two findings decided the fix:

1. the per-piece `ggml_view_1d` + `ggml_backend_view_init` cost was **0.17 ms** for 128 views, not the
   item's assumed metadata cost;
2. the *enqueue* of a meta read/set of a 2 MiB **view** took 3.4-5.1 ms (it walks the piece's inherited
   split state row by row), while the same window handed to the backend as `(tensor, offset, size)` costs
   0.15 ms.

The fix stages one band of pieces through the context's pinned `state_stage` (async reads, one synchronize,
async writes, one synchronize per band; `LLAMA_STATE_STAGE_MIB`, default 64, `0` = the old path) and hands
the backend the tensor plus an offset instead of a view.  Result: **14.7/17.2 ms**, i.e. 150x, and faster than
the host path.  The single-device arm never reaches the new code and is unchanged.

Evidence, method and the full table: `benchmarks/2026-10-27-state-device-path.md`.  `llama-stateio.cpp` here
is the one-off harness that produced it (build it against `build-rocm/bin`; `HOSTIO=1` selects the host path,
the last `frag|single` argument the cell layout).

Oracle: `tests/test-llama-context` (`cp_d`, `cp_d_s` compare the restored KV blob byte for byte and the
replayed logits), plus the coherence, prefill-logit, state-pipe and checkpoint gates.
