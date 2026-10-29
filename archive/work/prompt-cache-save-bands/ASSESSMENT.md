# prompt-cache-save-bands — assessment (r173): the huge-page buffer half shipped, the banded-copy half was declined

Submitted as PR [#248](https://github.com/stew675/llama-cpp-rdna-boosts/pull/248) (`overdoingism`), cut on
r170, applied to r172 unchanged.  Two independent commits; the delivery took **commit 2** (huge-page
prompt-cache buffers, folded into block 12) and did **not** take **commit 1** (banded batched state copies
above `LLAMA_STATE_ASYNC_MAX_MIB`).

`README.md` here is the submitter's document, unmodified, and
`prompt-cache-save-bands.patch` is the submission as received.  The measurements, the abort reproduction and
the reasoning are in [`benchmarks/2026-10-29-prompt-cache-hugepage.md`](../../benchmarks/2026-10-29-prompt-cache-hugepage.md);
`WORKLOG.md` 261029.004 is the release record.

## Outcome

* **Commit 2 — taken.**  `server_prompt_cache::take()` reserves a new buffer of 64 MiB or more, advises its
  2 MiB-aligned part `MADV_HUGEPAGE` and only then zero-fills it (Linux; `LLAMA_SERVER_CACHE_HUGEPAGE=0`
  restores 4 KiB pages).  Measured on this box: 3.47 GiB alloc+zero **470 → 125 ms**, and the server's
  `prompt cache update took` for a 100K-token slot saving into a new 3.4 GiB buffer **828.1 → 497.0 ms**
  (−40 %); the recycled-buffer case is unchanged, and nothing copied differs.
* **Commit 1 — declined.**  Cutting a piece at a byte offset hands the meta backend a window it refuses
  (`GGML_ASSERT(size % chunk_size_full == 0)`), so a 2-device `-sm tensor` context **aborts** on a save whose
  state holds a piece larger than one band (`-c 266240`, 250K-token slot, q8_0 KV: exit 134, control with
  `LLAMA_STATE_ASYNC_BANDS=0` completes with identical logits).  It is also not a win here — the per-tensor
  path already runs at 8-14 GiB/s on a whole-slot state's 1-27 MiB pieces, and the batched path's extra host
  copy costs more than the syncs it removes (324 MiB, 1 GPU: save 32.0 → 51.9 ms, restore 23.1 → 53.9 ms;
  3.47 GiB: save 304 → 278, restore 242 → 281).

## What a re-cut of commit 1 needs

1. Bands of **whole** pieces — an oversized piece takes a band of its own, never a cut — which is the rule
   `llama_state_copy_staged()` (r157, the device-state class) already follows; or a cut at a multiple of the
   tensor's own layout unit, which `llama_io_write_host` does not currently know.
2. A 2- and 3-device `-sm tensor` arm with a state above one band **and** one whose pieces exceed the band
   (a lowered `LLAMA_STATE_ASYNC_MAX_MIB` is the cheap way) in the validation.
3. A Linux measurement where the banded path actually beats the per-tensor path at the sizes that occur
   (0.3-3.5 GiB), on a host whose per-tensor state copy is not already ≥ 8 GiB/s.
