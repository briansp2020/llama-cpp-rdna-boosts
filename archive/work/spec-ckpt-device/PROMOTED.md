# PROMOTED — PR #254, the target's speculative checkpoint in device memory (r184, block 12)

Promoted into the delivery as **block 12** in `v16-6050b467c-r184`.  The campaign is closed; this directory
keeps the submitted material for reference only (nothing here is applied).

## What shipped

The patch's mechanism, unchanged in substance: on a recurrent or hybrid target the target's **speculative**
checkpoint (saved before a verify whose draft is longer than `n_rs_seq`, restored on a partial acceptance)
carries `LLAMA_STATE_SEQ_FLAGS_ON_DEVICE`, so a save and a restore are device copies into one staging buffer per
slot that the context keeps.  Decided once per slot at its first checkpoint and logged; host memory otherwise.
`LLAMA_SERVER_SPEC_CKPT_DEVICE=0` keeps the host path.

## What the delivery changed on top of the submission

1. **The decision requires an offloaded state** (`!no_kv_offload && n_gpu_layers != 0`, with its own log line).
   As submitted, `-nkvo` / `-ngl 0` logged `device memory (149.6 MiB; 13502.5 MiB free on ROCm0)` for a state
   that lives in a host buffer: the io class would stage on the CPU buft and copy host-to-host, and the
   free-VRAM test was about a GPU the context does not use.  Measured before the fix: host 121.5 vs "device"
   122.7 t/s — no harm, a wrong claim and a pointless allocation.
2. **The real invariant is documented** (the code comment, `patches/README.md`, `AGENTS.md`): the device arm is
   *not a byte image*.  `data_tgt` holds only the metadata; the payload is the context's per-seq staging buffer
   (`llama_context::mem_storage`), which the next save overwrites.  The spec path keeps the save→restore pairing
   (a draft round saves once and only ever restores that save; the replay rounds save nothing; a host-flagged
   save does not touch the stage), and a checkpoint of this kind must never be written to anything persistent.
3. **A gate** (`scripts/gate-spec-ckpt-device.sh`): text/draft identity across host and device arms, the kill
   switch, the `-nkvo` fallback, no abort, no GPU fault.  As submitted, nothing in the repo reached this code:
   `tests/test-llama-context`'s `cp_d` covers the io class, not the server's use of it.
4. The comment's cost figure corrected (16 ms is a slow host; 4.5 ms on this box).

## Numbers (this box, gfx1201, one GPU, one binary, kill switch vs default, greedy)

| surface | request | host | default | delta |
|---|---|---|---|---|
| 27B UD-Q4_K_XL + DFlash2 + `ngram-mod,draft-dflash` | rewrite (edit) | 310.2 | 316.8 | **+2.1 %** |
| 27B, `ngram-mod` only | rewrite | 202.1 | 205.4 | **+1.6 %** |
| 35B-A3B UD-Q4_K_M `-ncmoe 40` + DFlash2 | rewrite | 422.1 | 440.4 | **+4.3 %** |
| all surfaces | prose story / continuation | — | — | 0 ± 0.3 % |

Text, drafted and accepted tokens identical in every arm.  The submitter's +10.0 % / +15.8 % is the same
mechanism on a slower host: the win is (host save cost) × (checkpoints taken), 47 saves at 4.5 ms here against
16 ms there.  See `benchmarks/2026-10-30-spec-ckpt-device.md`.

## Files

* `spec-ckpt-device-as-submitted.patch` — the contributor's `git am` patch verbatim (as filed).
* `specdev176.py` — the contributor's measurement harness (paths are placeholders).
