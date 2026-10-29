# 2026-10-29 — PR #248 (prompt-cache saves/restores of long contexts): the huge-page buffer half ships, the banded-copy half does not

Assessment of contributor PR [#248](https://github.com/stew675/llama-cpp-rdna-boosts/pull/248)
(`overdoingism`, `wip/prompt-cache-save-bands/`), cut on r170 and applied to r172 unchanged (`git am`, both
commits, no fuzz).  Box: gfx1201, 3× R9700 (arms below use `HIP_VISIBLE_DEVICES=0` or `0,2`), Ryzen 9
9950X3D2, DDR5, `transparent_hugepage/enabled = [madvise]`, ROCm 10.1.0, `~/bin/build-llama-rocm`.
Model: `Qwen3.8/27B/Q4_K_XL/Qwen3.8-27B-UD-Q4_K_XL.gguf` (dense hybrid, GDN layers), `-fa on`.

The PR is two independent commits, and they came out differently:

| commit | what | verdict |
|---|---|---|
| 1 | states above `LLAMA_STATE_ASYNC_MAX_MIB` also take the batched (pinned-stage) path, **one band at a time**, a piece larger than a band being **cut at a byte offset** | **not taken** — aborts on a multi-device `-sm tensor` context, and not a win here |
| 2 | `server_prompt_cache::take()`: a new buffer of 64 MiB or more is reserved, `MADV_HUGEPAGE`d on its 2 MiB-aligned part, then zero-filled (Linux) | **taken** (block 12) — −40 % on the save that has to allocate a new multi-GiB buffer |

## Instrumentation

* `state_bench` (a ~120-line standalone tool, `/tmp/state_bench.cpp` in the session; links the build's
  `libllama`): prefill N tokens from a prompt file, `llama_state_seq_get_size_ext` +
  `llama_state_seq_get_data_ext` (timed), then a second context + `llama_state_seq_set_data_ext` (timed), a
  one-token decode on both contexts and a **byte comparison of the full logit vector** plus the argmax —
  so "same logits" is the oracle, not a text hash.
* `hugepage-bench.cpp` (from the PR): `resize(n)` vs `reserve(n)` + `madvise(MADV_HUGEPAGE)` + `resize(n)`,
  printing the time and `AnonHugePages`.
* The server's own `prompt cache update took` (SRV_TRC, `--verbosity 4`), one server start per arm,
  `-c 131072 -np 1 --cache-ram 30000 -ctk/-ctv q8_0 -t 8`: request A = a 100K-token prose prompt, request B
  = 5 tokens (takes the slot over), then A again (restored from the cache).  Arms are
  `LLAMA_SERVER_CACHE_HUGEPAGE=0` and `LLAMA_STATE_ASYNC_BANDS=0` on the PR build, i.e. the release path.

## Commit 2 — huge-page prompt-cache buffers (accepted)

`hugepage-bench 3472` (the size of a 100K-token Qwen3.8-27B + DFlash2 slot state):

| mode | alloc+zero | AnonHugePages |
|---|---|---|
| 0 = `resize(n)` (release) | 468 / 470 / 471 / 472 ms | 0 MiB |
| 1 = reserve + madvise + `resize(n)` | **125 / 124 / 126 / 124 ms** | 3470 MiB |

Server, 100K-token slot, 1 GPU, `prompt cache update took` (save + restore in one number):

| arm | B arrives: save A into a **new** 3.4 GiB buffer + restore B | A again: save B (recycled) + restore A |
|---|---|---|
| release (both switches off) | 828.12 ms | 292.61 ms |
| bands only | 799.76 ms | 274.09 ms |
| **huge pages only** | **497.00 ms** | 291.01 ms |
| both (PR as submitted) | 473.71 ms | 264.41 ms |

The new-buffer save is the cell the change is about, and it moves by 331 ms — the 345 ms the allocation
measurement predicts.  The recycled-buffer cell is unchanged, as designed (spares keep their pages).  The
user-visible wall clock of request B (which waits on the save) followed: 0.93 s → 0.60 / 0.57 s.  Nothing is
copied differently, so no text or logit can move (the change is an allocation-time `madvise`).

On the contributor's 3900X (DDR4) the same step is bigger (1.6-1.7 s → 0.39 s), which is why the PR's
headline is 2.3 s → 1.45 s: the win is page-fault physics and scales with how slow the host's 4 KiB fill is.

### Same-binary A/B on the shipped r173 build

The table above switches both PR commits on and off across one build.  After the banded half was dropped, the
shipped build was re-measured the same way (one server start per arm, `LLAMA_SERVER_CACHE_HUGEPAGE` the only
difference), two pairs:

| pair | huge pages on | off | Δ |
|---|---|---|---|
| 1 | 610.35 ms | 814.44 ms | −204 ms |
| 2 | **494.67 ms** | **833.60 ms** | −339 ms |
| 1 / 2, recycled buffer | 315.93 / 305.50 ms | 305.29 / 305.66 ms | 0 |

The spread on the on-arm is huge-page *availability*: the PR's own bench reports 1040 ms for the first cold
allocation, "before the kernel had free huge pages at hand", and pair 1 ran shortly after several 3.5 GiB
allocations.  The mechanism measurement (four repetitions each) has no such spread.

## Commit 1 — banded batched copies (not taken)

### It aborts on a multi-device tensor split

`llama_io_write_host`/`llama_io_read_host` hand each state piece to the backend that owns the tensor.  Under
`-sm tensor` that is the meta backend, whose `get_tensor_async`/`set_tensor_async` splice a window that is
aligned to the tensor's own layout (`ggml-backend-meta.cpp`: `GGML_ASSERT(offset % chunk_size_full == 0)` /
`GGML_ASSERT(size % chunk_size_full == 0)`, or `% row_stride` for the row-range path).  A band cut at a byte
offset is not such a window:

```
HIP_VISIBLE_DEVICES=0,2 state_bench … 27B-UD-Q4_K_XL /tmp/long2.txt 250000 266240 tensor q8_0
  bands on (default band)           -> GGML_ASSERT(size % chunk_size_full == 0) failed   (exit 134)
  LLAMA_STATE_ASYNC_BANDS=0         -> save 988.81 ms, restore 2106.72 ms, logits BYTE-IDENTICAL
```

Also reproducible with `LLAMA_STATE_ASYNC_MAX_MIB=1` at a 4,000-token state (2 GPUs).  **A cut needs one
piece larger than the band**: a piece is one layer's contiguous used-cell run × row bytes, so q8_0 (row 1088
on this 27B) needs > 246,723 contiguous cells — i.e. `-c 262144` with a full slot is enough, `-c 266240` is
comfortable.  f16 happens to be *accidentally* aligned (2048 divides 256 MiB), so it cuts without aborting —
which is why the single-device Windows check in the PR saw the cut path work.  A 1-GPU arm can never abort
(the CUDA backend's get/set is a raw byte copy), which is consistent with the PR's "not covered: `-sm tensor`
/ several devices".

This is not a new rule: `llama_state_copy_staged()` (r157, the device-state class) already documents it —
"a band is a run of whole pieces: a piece is only ever a window the backends accept (a tensor or a view of
it), so it is never cut.  A single piece larger than the band gets a band of its own."  `patches/README.md`
(block 12, `LLAMA_STATE_STAGE_MIB`) states the same constraint for the meta backend.

### It is not a win on this box

`state_bench`, both arms in one binary (`LLAMA_STATE_ASYNC_BANDS`), logits byte-identical in every arm:

| state | direction | bands | release | |
|---|---|---|---|---|
| 282.5 MiB (4,000 tok, 1 GPU) | save | 52.12 ms | **27.98 ms** | 1.9x worse |
| | restore | 49.82 ms | **21.14 ms** | 2.4x worse |
| 323.9 MiB (5,246 tok, 1 GPU) | save | 51.86 ms | **31.95 ms** | 1.6x worse |
| | restore | 53.87 ms | **23.09 ms** | 2.3x worse |
| 323.9 MiB (5,246 tok, 2 GPU tensor) | save | 68.08 ms | **55.54 ms** | 1.2x worse |
| | restore | 108.22 ms | **49.03 ms** | 2.2x worse |
| 3472.2 MiB (100K tok, 1 GPU) | save | **278.47 ms** | 303.95 ms | 1.09x better |
| | restore | 281.32 ms | **242.30 ms** | 1.16x worse |

The premise (the per-tensor path is slow above 256 MiB — "about 2.4 GB/s … 1.8 s for a 110K-token slot")
does not hold here: it runs at **8.3-14 GiB/s** on these sizes, because a whole-slot state's pieces are
1-27 MiB, so the one-sync-per-tensor cost the batched path removes is already amortised — while the batched
path adds a full host copy (pageable → pinned → device, and back) on top of the DMA.  The batched path's
real case is the opposite shape: a context checkpoint's ~75 *small* pieces (r96), which is why its gate is
`total <= LLAMA_STATE_ASYNC_MAX_MIB`.

The server end-to-end confirms the isolated numbers (bands-only vs release: 799.76/274.09 vs 828.12/292.61,
i.e. tens of milliseconds, matching the save deltas above).

## What a re-cut of commit 1 would need

1. Bands of **whole** pieces (a piece above the band gets its own band), never a cut — or a cut at a
   multiple of the tensor's own layout unit, which the caller does not currently know.
2. A 2- and 3-device `-sm tensor` arm with a state above one band in the gate, plus a state whose pieces
   are cut (a lowered `LLAMA_STATE_ASYNC_MAX_MIB` is the cheap way) — this is the arm that aborts today.
3. A Linux measurement showing the banded path beating the per-tensor path at the sizes that actually occur
   (0.3-3.5 GiB), on a host where the per-tensor path is not already ≥ 8 GiB/s.
