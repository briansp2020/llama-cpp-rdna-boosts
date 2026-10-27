# 2026-10-27 — the `LLAMA_STATE_SEQ_FLAGS_ON_DEVICE` state path: where its time actually went

Measurement record for r157 (TODO #110, block 12).  The item said the device-state save/restore "creates one
simple-tensor view per KV cache chunk per layer ... where one view per tensor plus a chunked copy would be
~24".  Measured, the views were never the cost (0.17 ms for 128 of them).  The cost was the *copy*:
`ggml_backend_tensor_copy` between two meta buffers falls back to a pageable `malloc` + a synchronous
device read + a synchronous device write **per piece**, and the meta backend serves a *view* of a split
tensor by walking its split state row by row (4.5 ms for 2 MiB against 0.15 ms for the same bytes handed to
it as the tensor plus an offset).

## Method

One-off harness (not part of the delivery): load a model under `-sm tensor`/`-ngl 99`, decode `-c`
proportional to the target, then time `llama_state_seq_get_data_ext` / `set_data_ext` with
`LLAMA_STATE_SEQ_FLAGS_ON_DEVICE` (`best of 3`, memory cleared between).  Two shapes:

* **contiguous** — one sequence, one cell run per (layer, K/V): 128 pieces of ~1.7 MiB.
* **fragmented** — two sequences decoded one token at a time so the saved sequence's cells interleave:
  49 152 pieces of ~3 KiB (the #210 lazy-idle-slots shape).

`llama-stateio.cpp` source is in `archive/work/state-device-path/` with the exact flags.

## Results (gfx1201, ROCm 10.1.0, Qwen3.8-27B-UD-Q4_K_XL, `-c 2112`, state = 213.6 MiB)

| arm | save | restore |
|---|---|---|
| 2 GPU `-sm tensor`, contiguous — **before** | 2227 ms | 2269 ms |
| 2 GPU `-sm tensor`, contiguous — **after** | **14.7 ms** | **17.2 ms** |
| 2 GPU `-sm tensor`, fragmented — before | 2237 ms | 2282 ms |
| 2 GPU `-sm tensor`, fragmented — after | 16.1 / **14.7 ms** (rep 0 pays the first-use stage fault: 497 ms) | 21.1 / **17.3 ms** (rep 0: 355 ms) |
| 1 GPU (device-to-device `cpy_tensor` arm) — before | 3.01 ms | 3.18 ms |
| 1 GPU — after | 3.02 ms | 3.22 ms |
| same bytes over the **host** path (`FLAGS_NONE`), 2 GPU | 21.9 ms | 38.9 ms |

So the device path was **58x slower than the host path it exists to beat**; it is now **1.5x/2.3x faster**
than it, and the single-device arm (a true device-to-device copy, which never reaches the new code) is
unchanged.

`LLAMA_STATE_STAGE_MIB` sensitivity (2 GPU, contiguous, save/restore): `0` (the old path, the A/B arm)
2231/2278 ms, `8` 15.3/18.4, `32` 14.8/17.7, `64` (default) 14.7/17.2, `512` 14.6/17.5.  Flat from 8 MiB up,
so the band size only has to amortise the per-band synchronize.

## What the phases were (instrumented, 2 GPU, contiguous, before the fix)

* view creation / `ggml_backend_view_init`: **0.01 ms** (write side) / **0.17 ms** (read side) for 128
  tensors — not the problem; the item's "0.6 MiB of host metadata" was real but worthless.
* the copy loop: **2210 / 2269 ms** — i.e. all of it.
* per piece, the meta read of a *2 MiB view* took 3.4-5.1 ms at **enqueue**, and only the V-cache tensors
  were slow (K's were under 2 ms): the view's inherited split state makes the reader splice the piece row by
  row.  Handing the same bytes to the backend as `(tensor, offset, size)` costs 0.15 ms.

## Oracle

* `tests/test-llama-context` (`-sm tensor`, 2 GPU, Qwen3.5-4B-Q8_0): `cp_d` / `cp_d_s` / `rt` / `ps` /
  `frag` / `rot` / `seq_rm` / `state_load` / `rollback` / `replay` / `shared` all **OK**; `rf` fails on
  **both** this build and the r153 build because that case creates its own context without flash attention,
  which `-sm tensor` requires — a harness limitation, not a state-path regression (`rf` passes on 1 GPU).
* Coherence unchanged: Qwen3.5-4B-Q8_0, seed 42, 3 GPU `1c5d32ac537d`.
* `scripts/gate-state-pipe.sh` PASS (the host path, untouched, still byte-identical in every arm).
* `scripts/gate-prefill-logits.sh` 0.000000 / 100.000 %.
* `scripts/gate-ckpt-reuse.sh` PASS.
