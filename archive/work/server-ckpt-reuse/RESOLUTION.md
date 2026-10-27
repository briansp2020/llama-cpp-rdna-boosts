# server-ckpt-reuse — assessment and resolution

**Status: promoted into the delivery as r155 (block 12).**  Release `v15-6050b467c-r155`, tip `0ba2387f6`,
tree `5c941ec661ac1b8c0b4fd0636a495c38a9cbd981`.  Source: PR #236 by **overdoingism** (`wip/server-ckpt-reuse/`,
one `git am` file, `tools/server/server-context.cpp` only).  The contributor's own record is kept verbatim in
`README.md` and the patch as submitted in `server-ckpt-reuse.patch`; this file is the maintainer-side
assessment, what shipped and what the gates measured.

## What it claims

1. A prompt that resumes from a **context checkpoint** (hybrid/recurrent model, cached prefix) restores the
   checkpoint and then saves the *identical* state again at the same position before its first batch (log:
   `superseding context checkpoint`), plus one more near the prompt end.  Each save allocates, zero-fills and
   faults in a fresh host buffer of the recurrent state's size.
2. Keeping the restored checkpoint when nothing has been decoded since removes the first save; recycling the
   buffers of erased/superseded/invalidated checkpoints turns the others into a plain copy.
3. No arithmetic changes and no KV cell moves, so the output is unchanged in every setting.
   `LLAMA_SERVER_CKPT_REUSE=0` restores the old behaviour.

## Verdict

**Accepted.**  The mechanism is exactly as described, the change is gated and default-on with a kill switch
(the repo's default-on policy), and the output is byte-identical in every arm we could construct (36
requests over 6 configurations, plus a slot save/restore round trip).  Two small hardenings were added on top
(below).  One claim needed qualifying rather than fixing: the keep path is **hybrid/recurrent only** (see 3).

## Assessment

**1. Why the keep is sound (and why it is safe to keep rather than re-save).**  `common_prompt_checkpoint`
stores the target's `PARTIAL_ONLY` state (for a hybrid memory that is the recurrent state, *not* the attention
cells), the draft context's state and the drafter's stashed spec state.  The restore path calls
`state_seq_set_data()` with exactly that payload and then `seq_rm(seq, pos_next, -1)`; the only mutation is a
truncation **beyond** the checkpoint's own extent, so the live state after the restore is the state the
checkpoint carries.  A fresh save at that position would therefore write the same bytes back (a
`set_data`/`get_data` round trip on those fields), which is what makes "keep instead of save" a no-op for the
data and a win for the cost.  Nothing else can have run: the first `create_checkpoint()` of a task is called
before the batch it accompanies is decoded, and any decode in between advances `n_tokens_new` past the
remembered value, so the guard cannot match late.

**2. Why the guard can fire at all (and why the PR is right that it is a *hybrid* fix).**  The restore sets
`n_past = min(size_up_to_pos(pos_next), it->n_tokens)` with `pos_next = min(pos_next, max(pos_min + 1,
pos_max))`.  `llama_memory_hybrid::seq_pos_min()` is the **max** of the attention and recurrent caches' minima,
and the recurrent cache holds one cell per sequence whose `pos` is the last processed position - so a hybrid
sequence reports `pos_min == pos_max == n_tokens - 1`, `pos_next` becomes `n_tokens`, and the first checkpoint
of the resumed task lands at exactly the restored `n_tokens` (the re-save we measured in the eager arm).  For a
dense or SWA memory `pos_min` is the oldest cell (usually 0), so `pos_next = pos_max = n_tokens - 1` and the new
checkpoint is one token short of the restored one: the keep cannot match and only the buffer recycling applies.
That is a property of upstream's restore arithmetic, not of this PR, but it does mean **the feature's headline
win is hybrid/recurrent-only** - which happens to be the case the PR was written for and the case where
checkpoints are unavoidable.

**3. The buffer recycling.**  `ckpt_recycle()` moves `data_tgt`/`data_dft` out of a dropped checkpoint into a
flat spare list (at most 4 vectors, the largest kept), and `ckpt_take(n)` hands back one of exactly `n` bytes so
`update_tgt()`'s `resize(n)` is a no-op - no allocation, no zero fill, no page faults.  Reuse of a dirty buffer
is safe because `llama_io_write_host` writes the full `ckpt_size` contiguously (`n != ckpt_size` aborts), so no
stale byte survives.  The spares are **not** counted against `--cache-ram` (they are not prompt-cache state);
the retained host memory is bounded by 4 buffers: 4 x (149.6 + 21.3) MiB for 27B + DFlash2, 4 x 50.3 MiB for
4B.  Measured RSS: **+0.15 GiB** (4B) and **+0.45 GiB** (27B).

## Findings reported to the submitter / changes made here

1. **`n_tokens_ckpt_restored` rested on one task-entry path.**  The PR resets it in the `SLOT_STATE_STARTED`
   branch and immediately before the restore, so today's code is correct, but a future task-entry path (a TTS
   task already skips `STARTED`) would inherit a stale value whose only consequence is a wrong "keep".  Fixed by
   also clearing it in `server_slot::reset()` - the invariant no longer depends on the state machine.
2. **A checkpoint with an empty `data_tgt` could be kept.**  `load_tgt()` returns `true` (a no-op) for an empty
   buffer, so such a checkpoint never restored anything and cannot stand in for the live state.  Unreachable
   today (a completion task always has a memory, so `update_tgt()` always writes), but cheap to close: the keep
   path now requires `!cur.data_tgt.empty()`.
3. **`data_spec` is not recycled** (eagle3-only, one `n_embd` row).  Left as is - not worth a spare slot.
4. **The PR lists multi-GPU, vision, and the slot save/restore endpoints as uncovered.**  We covered multi-GPU
   (3 GPU `-sm tensor`) and the `POST /slots/N?action=save|restore` round trip (identical saved file size,
   identical restored text, no abort on the checkpoint appendix); vision was not run.
5. **The win is host-dependent, as the PR says.**  It is page-fault bound: 25-100 ms per resuming request here
   (9950X3D, DDR5, THP) against the submitter's 195-223 ms (3900X, DDR4, 4 KiB pages).  The mechanism reproduces
   exactly - same log transitions, same checkpoint counts, same direction.

## Measurements (gfx1201, ROCm 10.1.0, r155 tree)

Every arm's greedy text compared byte for byte between `LLAMA_SERVER_CKPT_REUSE=1` (default) and `=0` on the
same binary.  Prompt time of the request that resumes from a checkpoint, ms off -> on:

| arm | requests |
|---|---|
| Qwen3.5-4B Q8_0 (hybrid, 50.3 MiB checkpoint), 1 GPU | 45.1 -> **31.4**, 40.0 -> **28.7**, 46.4 -> **32.4** |
| the same, 2 slots `--kv-unified` + idle-slot parking | 43.4/42.2/44.6/42.2/43.9 -> **29.3/28.7/28.3/27.3/27.7** |
| Qwen3.8-27B UD-Q4_K_XL + DFlash2 n-max 5 (149.6 + 21.3 MiB), 2 GPU `-sm layer` | 197.1 -> **163.2**, 186.9 -> **161.9**, 144.1 -> **112.0**, 187.6 -> **126.7** |
| the same, 128-token generations | 198.7 -> **162.3**, 193.1 -> **147.2**, 142.2 -> **110.7**, 189.4 -> **128.3** |
| the same, 3 GPU `-sm tensor`, 64-token generations | 407.9 -> **373.6**, 291.2 -> **236.2**, 182.4 -> **142.0**, 251.7 -> **150.6** |
| slot `save` + `erase` + `restore` round trip in between (4B) | 51.1 -> **38.8**, 39.4 -> **29.0** |

Mechanism counters in the same runs: "superseding" **3** (eager) vs "keeping" **3** (keep), checkpoints
created **9** vs **6**.  Cold (non-resuming) requests are unchanged within noise; decode throughput is
unchanged (identical greedy text and generation times on the 27B arms).

Correctness arms: 36 requests over the 6 configurations above, **all byte-identical**, plus the slot file
round trip (same size, same restored text), `LLAMA_SERVER_CKPT_REUSE=0` reproducing the pre-r155 log
transitions exactly, and the unpatched r154 build reproducing the switch-off text.

## Gates (release suite, r155 tree)

* Coherence: 4B Q8_0 3 GPU `-sm tensor` `1c5d32ac537d`, 35B-A3B UD-Q4_K_M 2 GPU `-sm tensor -ncmoe 10`
  `627b2ebd7792` - both == r154 (`ROCBLAS_USE_HIPBLASLT=0`, `scripts/extract-generated.py`).
* The r155 tree differs from r154 **only** in `tools/server/server-context.cpp` (`git diff --stat 08c471959
  0ba2387f6` = 1 file, 87 insertions), which no decode path and no `libllama`/`libggml` TU reaches, so the
  decode/MTP kernels are unchanged by construction.  MTP Protocol A was still run: prose `-n 3000`,
  `plain == draft-mtp` byte-identical (**11066 chars, sha `33885f8c27bf`**), and the four-axis adaptive arms
  (reasoning 54.4, prose 70.0, code 74.1, recall 81.4 t/s; mixed 56.0) plus the phase-switching prompt.  A
  `llama-server` four-axis acceptance check (1 GPU, `-c 32768 -fa on -ctk/-ctv bf16`, adaptive cap 3, 3000
  tokens per axis) reads R 70.2 t/s accept 0.895 mean len 3.32, P 80.4 / 0.990 / 3.58, C 73.0 / 0.934 / 3.54,
  K 84.1 / 1.000 / 3.65, i.e. healthy and in the expected order; it is a health check, not a like-for-like
  comparison with the r154 record (different invocation, and `llama-cli` never prints the speculator's stats).
* `scripts/gate-ckpt-reuse.sh` **PASS** (new gate; 3 keeps, 5 vs 8 checkpoints, text identical, no fault).
* `scripts/gate-park-idle-slots.sh` **PASS** (park == lazy-off 4/4, 6 park + 2 unpark moves).
* `scripts/gate-state-pipe.sh` **PASS** 5/5 arms (every restored cycle byte-identical to `LLAMA_STATE_ASYNC=0`).
* `scripts/gate-spec-align.sh` **PASS** (2 GPU 109.4 -> 142.6 t/s, 3 GPU 87.6 -> 135.5, host experts
  117.3 -> 149.1, splits 58 -> 0, mixed arm inert 100.5 -> 100.9).
* `scripts/gate-hc-fusions.sh` **PASS** 7/7 arms, BF16-hc class.
* `scripts/gate-prefill-logits.sh` **PASS** (mean KLD **0.000000**, same-top-p **100.000 %**).
* Per-block build `--fast --changed-from 08c471959`: **2/2** (block 12, tip).
* `scripts/validate-set.sh` strict 15/15; `scripts/check-gpu-faults.sh` clean.

## What shipped in the delivery

Block 12, as submitted plus the two hardening lines described above (both no-ops on every configuration
measured).  `LLAMA_SERVER_CKPT_REUSE` is catalogued in `ENVIRONMENT.md` §15, the block note is in
`patches/README.md` (block 12), the gate is `scripts/gate-ckpt-reuse.sh`, and the release record is
`WORKLOG.md` 261027.006.
