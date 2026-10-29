# wip/prompt-cache-save-bands — prompt cache saves and restores of long contexts: banded batched copies, huge-page buffers

`prompt-cache-save-bands.patch`: one `git am` file with two commits, made on `v15-6050b467c-r170` (the release's tree
`c6c0d8e2`; applied tree `86eab721`).  It touches `src/llama-context.cpp` and `tools/server/server-task.cpp` (+159 / -34).
sha256 `adfa91037b27a56a3fa6bbc2eb80f652f30f239167cae6d431366c40e48fbfd3`.  `hugepage-bench.cpp`: the allocation measurement below, standalone.

**Why.**  When a request takes over a slot (LRU, `get_available_slot`), the server first saves the slot's old context to
the host prompt cache, and the new request's first token waits for it; coming back to that conversation restores it.  For
a long context that is a state of gigabytes (100K tokens of Qwen3.8-27B + its DFlash2 draft: 3.5 GiB), and two things
make it slow, both already described in the code:

* *The copy.*  The batched state path (pinned `state_stage`, pipelined groups, streaming host copies) takes states up to
  `LLAMA_STATE_ASYNC_MAX_MIB` (256) only - "so a whole slot save or load never pins gigabytes" - so a slot of a long
  context takes the per-tensor synchronous copies into or out of pageable memory.
* *The buffer.*  A save needs a buffer as large as the state whenever no recycled one is large enough
  (`server_prompt_cache::take`), and - as the comment on the spare buffers says - zero-filling and faulting in a large
  buffer costs about as much as the copy.  On Linux it costs more: in 4 KiB pages, 3.5 GiB took 1.6-1.7 s to zero-fill on
  a Ryzen 9 3900X, against ~0.3-0.45 s for the copy itself.

On our 2-slot server (`-c 245760 --kv-unified`) the takeover happened 49 times on one working day, 17 of them on a slot
holding 50K tokens or more.

**What changes.**

1. *libllama (`llama_io_write_host` / `llama_io_read_host`).*  A state above one band (`LLAMA_STATE_ASYNC_MAX_MIB`,
   default 256) takes the batched path too, one band of at most that many bytes at a time through the same stage, so it
   never pins more than one band.  A piece above a band is cut at byte offsets.  States up to one band (the context
   checkpoints) are copied exactly as before.  `LLAMA_STATE_ASYNC_BANDS=0` restores the per-tensor path for the states
   above a band.
2. *server (`server_prompt_cache::take`, Linux).*  A new buffer of 64 MiB or more is reserved, its 2 MiB-aligned part
   advised `MADV_HUGEPAGE`, and only then zero-filled, so it is faulted in 2 MiB pages (transparent huge pages in
   `madvise` mode, Ubuntu's default; with THP off the advice is a no-op).  Recycled buffers keep their pages.
   `LLAMA_SERVER_CACHE_HUGEPAGE=0` restores 4 KiB pages.

No VRAM change; generation and prefill are not touched.  The pinned stage grows to one band after the first large copy
(on this model it already held ~160-176 MiB for the checkpoints).

## Results

Linux: Ryzen 9 3900X (DDR4), one R9700 (gfx1201), Ubuntu 24.04.5 (THP `madvise`), ROCm 10.0.0 (TheRock pip), plain r170 +
this patch, `ROCBLAS_USE_HIPBLASLT=0`.  Qwen3.8-27B-UD-Q4_K_XL + Qwen3.8-27B-DFlash2-Q4_K_M, `-c 131072 -ub 2048
--parallel 1 -fa on -ctk q8_0 -ctv q8_0 --spec-type draft-dflash --cache-ram 24576`.  One slot, so every new conversation
takes it over.  Requests: A = a 100K-token zh prompt + 32 greedy tokens (state 3494.6 MiB, draft 21.3 MiB); B = a 1K-token
en prompt (193.3 MiB); then A again (restored from the cache) and B again.  The server's own `prompt cache update took`
(`--verbosity 4`), one server start per arm:

| prompt cache update | release path | + bands | + bands + huge pages |
|---|---|---|---|
| B arrives: save A (a new 3.5 GiB buffer) | 2282 ms | 2491 ms | **1451 ms** |
| A again: save B, restore A | 824 ms | 643 ms | **614 ms** |
| B again: save A (a recycled buffer), restore B | 749 ms | 694 ms | 721 ms |
| B's first request, wall clock (1K prefill included) | 3.05 s | 3.26 s | **2.22 s** |

("release path" = this build with `LLAMA_STATE_ASYNC_BANDS=0`, before the huge-page commit; "+ bands" = the default, same
build.)  The banded copies ran at 6.8-7.7 GiB/s per band.  The new buffer was most of the first row: the time between the
save's log line and its first band fell from 1.875 s to 0.79 s with huge pages.  `hugepage-bench.cpp` on the same box, 3.5
GiB: `resize(n)` 1527-1677 ms in 4 KiB pages, 389-396 ms with the advice (1040 ms the first time, before the kernel had
free huge pages at hand), `AnonHugePages` 3492 MiB.

Windows 11 (Ryzen 7 5700G, one R9700, TheRock ROCm 10; the same change on r164, where this code is identical): a 60K-token
state (2165.6 MiB) saves in 508 ms against 526 ms on the per-tensor path and restores in 390 against 428 ms - its pageable
copies were already fast, and a new 2.1 GiB buffer costs ~0.25 s there; the huge-page part is Linux-only.

**Same results.**  The restored A produced the same 32 greedy tokens as the first A, and the restored B the same token,
in every start on both machines (Linux four starts, Windows seven), including one Windows start with
`LLAMA_STATE_ASYNC_MAX_MIB=64`: bands of 64 MiB, so the KV pieces were cut (31 single-piece bands each way) and the
context checkpoint restored through bands as well.

**Not covered:** `-sm tensor` / several devices (the meta-buffer path keeps its own staging and was not run), other
Linux THP settings than `madvise` (with `never` the advice does nothing; with `always` it was not needed), Windows large
pages (they need the lock-pages privilege).

*Written by Claude, an AI model by Anthropic, from measurements on the contributor's machines.*
