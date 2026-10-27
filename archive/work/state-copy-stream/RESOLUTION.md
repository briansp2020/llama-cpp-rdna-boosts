# RESOLUTION -- threaded streaming state copy (PR #235) promoted into block 12, r152

**Status:** promoted 2026-10-27 in `v15-6050b467c-r152` (block 12).  The submission is preserved verbatim in
this directory (`README.md`, `state-copy-stream.patch`, cut on r144 and reported as applying on r150).  The
release record is `WORKLOG.md` 261027.003.

## What was reviewed

`llama_io_write_host::flush_async` copies the pinned stage into the caller's buffer with one `memcpy` per
piece.  The PR replaces that with 1 MiB chunks copied by SSE2 streaming stores on `LLAMA_STATE_COPY_THREADS`
threads (default 2, `0` = the plain memcpy), and adds a `LLAMA_STATE_TIMING` diagnostic.

Code review, in the order the failure modes matter:

* **It does not apply on r151.**  The patch is cut on r150 and its hunk covers the copy-out loop that r151
  (PR #233) rewrote: the copy is now per size-balanced group, interleaved with that group's event
  synchronisation.  A 3-way apply conflicts; the fold is a re-cut against the current tree, not a cherry-pick.
* **The submitter's whole-state threshold is the wrong shape after r151.**  `total < 16 MiB` was meant to skip
  small states, and with one unpipelined copy that mapped to "the copy is small".  A pipelined copy runs the
  loop eight times, so the decision belongs to the state (16 MiB, kept) and the per-run decision to the piece
  (64 KiB, new: a piece smaller than that never covers whole cache lines and is copied with `memcpy`).
* **Threading has to respect the group boundary, which is the one real hazard.**  r151's copy-out may only
  read a group's stage range after that group's event, so a worker that took pieces from the *next* group
  would read a stage range whose device copy is still in flight: a torn state, not a crash, and invisible to
  every gate that does not compare restored text byte for byte.  The fold therefore splits *within* one group
  and joins at the end of that group, which is also why it is not "threads over the whole copy".
* **Alignment.**  The submitted inner loop aligns the destination to 16 bytes (the store width).  A run that
  starts mid-line makes two half-lines each carry a partial write-combining buffer, so the fold aligns to 64
  and keeps a `memcpy` tail.
* **The fence.**  `_mm_sfence()` is inside the worker lambda in the submission, i.e. only the thread that ran
  `copy_chunks` fences its own stores, which is correct but fragile: any later caller that copied with
  `llama_copy_stream` and did not fence would hand a device DMA or the state API's caller stale bytes.  The
  fold puts the fence in the run copier, so a run is self-fencing and the thread join is the only other
  ordering it needs.
* **Restore side.**  The PR touches the save side only.  The restore's pack (caller's buffer -> stage) has the
  same shape and the same cost (it is the larger half of a restore), so the fold applies the same copy there;
  that also let the pack resolve each piece's stage offset once instead of re-deriving it in the queue loop
  (the `off = off_g` reset r151 kept is gone - same addresses, one less way to get them wrong).
* **Portability.**  x86-only is right (`__SSE2__` / `_M_X64`, MSVC x64 included); other targets keep the
  memcpy and gain only the threads.  No POSIX call is added, so `check-windows-portability.sh` is unaffected,
  and `std::thread`/`<atomic>`/`<emmintrin.h>` are all available to the Windows HIP build.

Two submission nits were noted to the submitter (neither affects the folded tree):

* the measurements are quoted as one "host copy" number per checkpoint, but the save's copy-out and the
  restore's pack are different code paths with different destinations (a heap buffer vs the pinned stage) - a
  change that helps one need not help the other, so they should be reported apart;
* `LLAMA_STATE_COPY_THREADS=0` is documented as "the previous plain memcpy", which is the right kill-switch
  semantics, but the PR's own numbers make the thread count look like a 5 % afterthought when on a modern
  box it is most of the win (see below).

## Reproduced and re-measured (gfx1201, ROCm 10.1.0)

`scripts/gate-state-pipe.sh`, which now also carries a **`copy0`** arm (`LLAMA_STATE_COPY_THREADS=0`) gated
byte-for-byte like every other arm.  Prompt-cache restore of a 5246-token hybrid state, 3 cycles per arm, the
server's own save+load trace and the per-op timings from `LLAMA_STATE_TIMING=1`:

| config | `copy0` (plain) | default (2 threads) | delta |
|---|---:|---:|---:|
| 35B-A3B UD-Q4_K_M, 1 GPU `-ncmoe 10` | 38.43 ms | **33.37 ms** | **-13.2 %** |
| 35B-A3B UD-Q4_K_M, 2 GPU `-sm tensor -ncmoe 10` | 44.15 ms | **40.49 ms** | **-8.3 %** |
| Flash-Next GSQ-RCO IQ3_XXS + shared Q8_0 MTP head, 2 GPU `-sm tensor -ncmoe 40` | 79.30 ms | **74.92 ms** | **-5.5 %** |

The `copy0` arm lands on r151's recorded numbers (39.9 / 78.4 ms), which is what attributes the difference to
this change.  The copy itself (per size class, medians over the run):

| op | piece total | plain | 2 threads | delta |
|---|---:|---:|---:|---:|
| save (stage -> caller) | 62.8 MiB | 3.65 ms | 3.19 ms | -12 % |
| save | 73.2 MiB | 4.76 ms | 3.75 ms | -21 % |
| save | 165.3 MiB | 10.21 ms | 7.29 ms | -29 % |
| restore pack (caller -> stage) | 62.8 MiB | 4.07 ms | 2.85 ms | -30 % |
| restore pack | 73.2 MiB | 5.29 ms | 3.92 ms | -26 % |
| save (Flash-Next) | 112.6 MiB | 7.47 ms | 5.87 ms | -21 % |
| save (Flash-Next) | 250.9 MiB | 15.37 ms | 10.90 ms | -29 % |
| restore pack (Flash-Next) | 250.9 MiB | 23.64 ms | 19.27 ms | -18 % |

## Why the mechanism, not the headline, had to be re-derived

The PR attributes the win to streaming stores.  On a 164 MiB copy, in a 35B-A3B checkpoint's real piece sizes
(30 x 80 KiB, 30 x 2.6 MiB, 20 x 4.2 MiB), this box measures:

| arm | time | GiB/s |
|---|---:|---:|
| plain `memcpy`, 1 thread | 6.45 ms | 24.8 |
| streaming stores, 1 thread | 6.47 ms | 24.8 |
| plain `memcpy`, 2 threads | 5.17 ms | 31.1 |
| streaming stores, 2 threads | **3.56 ms** | **45.0** |
| streaming stores, 3 / 4 / 8 threads | 4.45 / 4.27 / 3.98 ms | 36-40 |

So on a 9950X3D (192 MiB of L3) the streaming store alone is **0 %** - a 164 MiB destination's lines are
mostly resident, so the read-for-ownership a plain store pays hits cache - and the threads are where the win
is, with two the knee on a dual-channel DDR5 controller.  On the submitter's 3900X + DDR4 (64 MiB of L3) the
same change is 19.4 -> 11.8 ms with ONE thread, i.e. there it really is the streaming store.  Both halves are
kept because they are each worth most on a different box, and neither costs anything on the other; the
recorded conclusion is "do not drop the threads on the strength of the streaming-store measurement, and do
not drop the streaming stores on the strength of this box's".

A cold destination (a freshly allocated buffer, so page faults dominate) is ~28 ms for the same 164 MiB
either way - the first batched copy into a newly allocated stage is fault-bound and was noisy in both arms of
every run.  That is a once-per-context cost and is not attributable to the copy.

## Gates

* `scripts/gate-state-pipe.sh` on all three arms above: every restored cycle of **every** arm byte-identical
  to `LLAMA_STATE_ASYNC=0` (including the new `copy0` arm), the save side isolated by `LLAMA_STATE_ASYNC_READ=0`
  and the restore side by `pipe` vs `read1`, every cycle really restored (`prompt_n` 10 of 5256), no abort,
  `check-gpu-faults.sh` clean on all five logs of every arm.
* `test-llama-context` (all 15 state cases), 1 GPU and 2 GPU tensor, on the r151 build and this one: PASS/FAIL
  set **identical** (`rf` + `rollback` fail on both, the slab/VRAM property of this box r151 recorded).
* Coherence: 4B Q8_0 3 GPU `1c5d32ac537d`, 35B-A3B 2 GPU `-ncmoe 10` `627b2ebd7792` - identical to the r151
  build measured in the same session with the same commands.
* `gate-prefill-logits.sh`: KLD **0.000000 / 100.000 %**.  `gate-park-idle-slots.sh`: PASS.
  `gate-hc-fusions.sh` (Flash-Next GSQ-RCO IQ3_XXS, BF16-hc class): PASS, 7/7 arms identical.
* Per-block build `--fast --changed-from dc6a78d5c`: 2/2.  `validate-set.sh`: strict 15/15.
