# Cold-session prompt — `wip/kv-residency` (TODO #118)

Paste the block below into a fresh session.  It assumes only what is in the repo (nothing from the previous
session's context).

---

Continue the windowed-KV campaign (TODO **#118**) in `~/llama-cpp-rdna-boosts` and `~/llama.cpp`.

**The purity blocker is closed and the fix is gated (A4, 2026-10-30).**  A spilled KV is numerically transparent
— prefill-logit KLD −0.000000 / 100.000 % on a spilling 27B arm — and the cause was the VMM *primitive*, not the
KV: a `cuMemUnmap` of a device mapping followed by `cuMemCreate`(HOST)/`cuMemMap` at that VA **zeroes a
different, live host-location mapping** in the same reservation.  `vmm_map` now holds the released physical for
the duration of its create (`LLAMA_KV_REGION_QUARANTINE`, default **on**; `=0` reproduces the bug), and
`GGML_CUDA_VMM_REGION_SELFTEST=1` includes a case that FAILS if the hold stops working.  **This session's job is
the two correctness corners that fix reopened, then the re-measurement: HANDOVER §7.2 items 1 → 2, then 4.**

START BY READING, in this order:
  1. `~/llama-cpp-rdna-boosts/wip/kv-residency/HANDOVER.md` — self-contained.  **§7.2 is the ordered work list**,
     §3 the invariants (each cost a session; **#10 is the new hazard rule**), §5 the measurement rules (**#6:
     never time a `LLAMA_KV_REGION_DEBUG` arm**), §6 the gates, **§8.1 the absolute references for this box**
     (so nothing has to be re-derived before an A/B), §9 the re-baselined box.
  2. `.../A2-MIXED-BACKING.md` **§3.7** — §3.7.6 is CLOSED and carries a banner listing the three readings that
     were **wrong**; **§3.7.8 is the hazard and its hold** (with the gate table); **§3.7.9 is what the spill
     costs** (a straight line, ~61 GB/s effective, no cliff); §3.7.10 is what is left.
  3. `.../MIXED-BACKING.md` — the design, and the hazard row in §4 (background only).

STATE (verify, don't assume):
  - `~/llama.cpp` is on branch `wip-kv-residency` at **`dc122f338`**, pushed to the personal fork
    `git@github.com:stew675/llama.cpp.git`.  The superseded A1 fallback is `wip-kv-residency-a1-two-base`.
  - `~/llama-cpp-rdna-boosts` main = the record.  `patches/0001`…`0012` apply in order to `5bdec04ab` and
    reproduce the branch tip's tree exactly (verified).
  - `rdna-boosts` is NOT the campaign's branch: never commit campaign code there.  Nothing from the campaign is
    in `patches/`, `release.json` or any block, and nothing may be, without the maintainer's explicit go-ahead.
  - ONE R9700 (gfx1201), 32 624 MiB.  Pin `HIP_VISIBLE_DEVICES=0`; never claim a multi-GPU result.
  - The link is 55.7 GB/s H2D (re-cabled 2026-10-30).  Every pre-swap t/s number in the campaign is at 28.8 and
    must not be compared with a new one; the cost model is `bytes / 55.7 GB/s`.

WHAT IS DONE — do not rebuild, do not re-gate:
  - The windowed KV works and slides; `--fit` sizes and reports it (`LLAMA_KV_RESIDENT_CELLS=auto`);
    `LLAMA_KV_SPILL_RAM_PCT` caps the pinned spill; the window apparatus is free (zero-spill arms measure the
    same as plain); the spill's cost is linear (~16.3 µs per MiB per token, ~61 GB/s effective).
  - A4 closed the purity blocker and left the instruments in place (byte-exhaustive restore verify, stage-out
    cross-check, per-mapping fingerprint, region dump, canary, `LLAMA_KV_REGION_TRACE`), plus
    `hostmap-hazard-probe.cc` (the isolated reproducer) and the selftest case.
  - Worst case measured and pure: 27B Q8_0 `-c 262144`, `W=1` → 128 MiB VRAM / 8576 MiB pinned host,
    prefill 944.8 t/s, decode 12.34 t/s, text byte-identical to the fully-resident arm.

DO NOT trust these any more (each was measured on an arm that never actually spilled — check `FLIP > 0` with
`LLAMA_KV_REGION_DEBUG=1` before believing any windowed result):
  - "the 4B is immune to the spill" — at `-c 4096` it never reaches a flip; at `-c 16384` it corrupted like the
    27B (and the same hold fixed it);
  - the A3 `gate-ckpt-reuse.sh` / `gate-state-pipe.sh` / MTP-verify-width PASSes: 4B arms at `-c 4096`, `FLIP=0`.
    Re-running them on a provably spilling arm is your first job;
  - the identity trio at `W=2048`/`4096` `-n 128` **passed while the spill was corrupted**: it is not the
    regression test.  The sensitive arms are the `W=1024` `-n 64` text arm and the prefill-logit gate.

CONSTRAINTS:
  - Build `~/bin/build-llama-rocm` (never pass `-DCMAKE_HIP_FLAGS`); `LD_LIBRARY_PATH=/opt/rocm-10.1.0-gfx120X/lib`.
  - Hash text from a run WITHOUT `-v` via `~/llama-cpp-rdna-boosts/scripts/extract-generated.py`; read timings
    and the cache/arena lines from a separate `-v` run.  Always `--single-turn --no-display-prompt`, always
    `timeout`, never two GPU jobs at once.
  - A fresh-process numeric comparison needs `ROCBLAS_USE_HIPBLASLT=0`.
  - Do not judge anything on a greedy text hash alone: the prefill-logit gate on a spilling arm is the sensitive one.
  - `-n 2000` for anything about the expert arena, and `-ncmoe` must be the model's full layer count.
  - If you need a gate arm that spills, remember the floor: the slide is quantised to a 4 MiB map unit, so
    `W=1` == `W≈3855` and the smallest real spill is one unit per tensor (128 MiB on a 27B).
  - Do not pick scope.  If something cannot be measured, say so and stop.

GATES to re-run if you change code: the VMM selftest (`GGML_CUDA_VMM_REGION_SELFTEST=1` — must print **PASS**,
and again with `LLAMA_KV_REGION_QUARANTINE=0`, where it must print **FAIL**), the §6 identity trio
(`5ba44647188f` with the feature off), the prefill-logit gate on a spilling arm (must read −0.000000 /
100.000 %), `check-gpu-faults.sh` over every log, and the OOM config still OOMing with the feature off.  Re-run
`hostmap-hazard-probe.cc` (default FAIL, `--grab 4` PASS) if you touch the VMM primitive.  Exact commands:
HANDOVER §6.
