# 2026-10-30 — PR #254: the target's speculative checkpoint in device memory (`LLAMA_SERVER_SPEC_CKPT_DEVICE`, r184)

Assessment and release numbers for [PR #254](https://github.com/stew675/llama-cpp-rdna-boosts/pull/254)
(`wip/spec-ckpt-device/`, one 78-line patch on r176, applying unchanged to r183).  Every number here is from
this box: gfx1201 (one R9700 of three), ROCm 10.1.0, the canonical `build-rocm` tree, ccache warm.

## What the patch does

On a recurrent or hybrid target the server saves a checkpoint of the target's recurrent state before a verify
whose draft is longer than `n_rs_seq` (`use_ckpt_tgt`) and restores it when only part of the draft is accepted.
It went to host memory (Qwen3.8-27B: 149.6 MiB).  The patch passes `LLAMA_STATE_SEQ_FLAGS_ON_DEVICE` for it, so
save and restore become device copies into one staging buffer per slot that the context keeps.  Device memory
only when the target is on exactly one GPU with the state's size + 512 MiB free; decided once per slot at its
first checkpoint and logged.  `LLAMA_SERVER_SPEC_CKPT_DEVICE=0` = host always.  Context checkpoints (the prompt
cache) are not touched.

## Correctness (the delivery's gates)

| check | result |
|---|---|
| strict apply on r183 / r184 | clean |
| `tests/test-llama-context -c 8192 --test 'baseline\|cp_h\|cp_d\|cp_h_s\|cp_d_s'` (27B, 1 GPU) | **5/5 OK** (the io class the server now uses) |
| `scripts/gate-ckpt-reuse.sh` (27B + DFlash2, `ngram-mod,draft-dflash`, 1 GPU) | **PASS** (keep 3 kept / 0 superseded, 5 vs 8 checkpoints, 4/4 requests identical) |
| `scripts/gate-spec-ckpt-device.sh` (new, 27B, `ngram-mod`) | **PASS** — 18 created / 5 restored per arm, text and drafted/accepted identical, kill switch clean, `-nkvo` on the host path with its own reason |
| `scripts/check-gpu-faults.sh` over every arm log + the journal | clean |

The device arm is not a byte image (`data_tgt` = metadata, the payload = the context's per-seq staging buffer),
which `gate-spec-ckpt-device.sh` is the only gate to exercise: `cp_d` covers the io class, not the server's use
of it.  Text identity is the oracle for a *state* copy, so the gate compares greedy text plus the drafted and
accepted token counts.

## Throughput (one binary, kill switch vs default, greedy, `ROCBLAS_USE_HIPBLASLT=0`)

`-ub 2048 -b 2048 -fa on -ctk q8_0 -ctv q8_0 --parallel 1`, one request at a time, order `0 / A / A / 0`.
Requests: `edit` = "add a one-line comment above every function" over a 3.0K-token Python file (n-gram heavy),
`story` = a zh short story, `cont` = an English continuation of 24K characters of llama.cpp's own `docs/*.md`.

| surface | request | host `=0` | default | delta | text, drafts, accepted |
|---|---|---|---|---|---|
| 27B UD-Q4_K_XL + DFlash2 + `ngram-mod,draft-dflash` | edit | 310.2 | **316.8** | **+2.1 %** | identical |
| | story | 28.5 | 28.5 | 0.0 % | identical |
| | cont | 65.2 | 65.3 | +0.2 % | identical |
| 27B UD-Q4_K_XL, `ngram-mod` only | edit | 202.1 | **205.4** | **+1.6 %** | identical |
| 35B-A3B UD-Q4_K_M `-ncmoe 40` + DFlash2 (arena on, 850 MiB free) | edit | 422.1 | **440.4** | **+4.3 %** | identical |
| | story | 69.4 | 69.3 | -0.1 % | identical |
| | cont | 143.3 | 143.5 | +0.1 % | identical |
| 27B, 2 slots `--kv-unified`, request on slot 1 | edit(700) | 235.2 | **241.4** | **+2.6 %** | identical |

Each arm's per-start spread is under 1.5 % (edit: host 309.0 / 311.5, device 316.7 / 316.9).  VRAM +150 MiB
(27B) / +63 MiB (35B-A3B) per slot that takes a checkpoint; RSS -150 MiB on the dense arm (the state is in VRAM).
The slot's decision log reads `speculative checkpoints of the target in device memory (149.6 MiB; 11456.5 MiB
free on ROCm0)` and, on the 35B-A3B arena, `(62.8 MiB; 850.5 MiB free on ROCm0)`.

## Why the PR's +10.0 % / +15.8 % is not reproduced

The win is exactly `(host save cost) x (checkpoints taken)`, and both terms are smaller or different here:

* **The checkpoint fires rarely.**  A 3039-token rewrite takes **47** speculative checkpoints and 8 restores
  (`-v` run), i.e. ~6 % of decode steps, not "most steps": it needs `draft.size() > n_rs_seq` (7 with
  `--spec-draft-n-max 7`), and most n-gram drafts are shorter.  Accepted steps never restore.
* **The host save costs 4.5 ms here, not 16 ms.**  310.2 -> 316.8 t/s over that request is 0.21 s of the 47
  saves.  The r96/r119 batched async reads, r151 pipelining and r152 two-thread streaming stores already took
  most of a 150 MiB host copy; the submitter's 16 ms is a slower CPU and 0.76 s of their 8.8 s request, which
  is their +10 %.  So the feature is worth as much as the host is slow, which is the right direction for a
  delivery whose users are not all on a 9950X3D.

## Findings from the review, fixed in the delivery

1. **`-nkvo` / `-ngl 0` claimed device memory for a host-resident state.**  With `--no-kv-offload` the
   recurrent/attention state lives in a host buffer, so the io class stages on the CPU buft and copies
   host-to-host while the log said `device memory (149.6 MiB; 13502.5 MiB free on ROCm0)` - and the free-VRAM
   test was about a GPU the context does not use.  The decision now requires `!no_kv_offload && n_gpu_layers
   != 0`, with its own log line, and `gate-spec-ckpt-device.sh`'s third arm asserts it.  Measured before the
   fix: host 121.5 vs "device" 122.7 t/s, i.e. no harm, only a wrong claim and a pointless allocation.
2. **The device arm's real invariant is documented** (the code comment, `patches/README.md`, `AGENTS.md`): the
   checkpoint is metadata in `data_tgt` plus a payload in `llama_context::mem_storage`, so the save/restore
   pairing is part of its correctness and it must never be persisted.

Latent, not reachable today: `llama_memory_recurrent::state_write` hard-aborts on `cell_ranges.size() > 1` under
`LLAMA_STATE_SEQ_FLAGS_ON_DEVICE`.  The server's recurrent memory gives each sequence exactly one cell
(`mem_size = max(1, n_seq_max)`, and `prepare()` moves a sequence into a fresh cell rather than letting it own
two), so a save is always one range - but a future change that lets a sequence own two cells would turn this
into a server abort.

## Scope notes

* Single GPU only, by design: under `-sm tensor` or a layer split the state is spread over devices (and the
  r157 device io class bounces through the host anyway), so the delivery's recommended 3-GPU server shape is
  inert.  Enabling it there needs a change and a multi-GPU run; the PR lists it as uncovered.
* +150 MiB per used slot is not in the `--fit` estimate (the buffer is allocated lazily, at the first
  checkpoint, after the expert-cache arena sized itself).
* MTP Protocol A was not re-run for this change: it moves no arithmetic (the same bytes are copied from
  another place), MTP drafts never take this checkpoint (`draft.size() <= n_rs_seq` by construction) and the
  multi-GPU spec path is inert.  The server spec path's text identity is gated by
  `gate-spec-ckpt-device.sh`, which is stronger than a text comparison at one width.
