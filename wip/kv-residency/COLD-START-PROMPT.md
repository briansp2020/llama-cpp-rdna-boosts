# Cold-session prompt — `wip/kv-residency` (TODO #118)

Paste the block below into a fresh session.  It assumes only what is in the repo (nothing from the previous
session's context).  Rewritten 2026-10-30 at the end of session A6, the promotion.

---

Continue the windowed-KV campaign (TODO **#118**) in `~/llama-cpp-rdna-boosts` and `~/llama.cpp`.

**Goal A is SHIPPED — do not re-measure it and do not re-promote it.**  Release **`v16-6050b467c-r179`** (r178 shipped the same block with a fit/probe regression that made it unusable with `--fit on`; r179 fixed it)
(16 blocks) carries the windowed KV as **block 15**: an **opt-in** fixed-VRAM resident window with the KV's
tail in pinned system RAM (`LLAMA_KV_RESIDENT_CELLS=<cells>|auto`; unset = the plain device buffer, so a run
without the knob is byte-identical to r177 including its OOM).  The fork's `rdna-boosts` is the re-cut tip
`898103305` (tree `9d2e5c28`), `validate-set.sh` is green, the shipped tree's gates are green (coherence
`5ba44647188f`, spilling-arm prefill-logit −0.000000 / 100.000 %, VMM selftest PASS and FAIL with the hold
disabled, no GPU faults), and the container images are built off-host by the tag watcher.  The shipped
description is `patches/README.md` "Block 15", `ENVIRONMENT.md` §17 and `WORKLOG.md` **261030.002**.

**This session's job is the campaign's remainder, not Goal A.**  In order:

1. **One owed fix in the DELIVERY (small, needs the maintainer's go-ahead):**
   `scripts/gate-park-idle-slots.sh` HARD-fails a *legitimate* outcome.  When the park arm's server logs
   *"cannot park the idle slot: saving it to the prompt cache and clearing it"* — what a windowed/region-backed
   KV now triggers, and what already happens for any slot whose cells are not one run (a prefix shared through
   `seq_cp`) — the gate should require `park == lazyoff` and report *"parking declined → save-and-clear
   fallback"* instead of failing on *"parking moved nothing"*.  With no window the gate passes today (verified).
   Rationale and shape: `wip/kv-residency/A2-MIXED-BACKING.md` §3.7.11.3.
2. **Goal C decisions** (`O3-DRAFTER-RING.md` §7): the MTP drafter ring (`LLAMA_SPEC_DRAFT_WINDOW`) is
   **implemented and measured but not promoted** — it is free at a usable window (46k context, `W=32768`:
   acceptance −2.3 % relative, generation flat) and costs 7-9 % at a window far below the working context; the
   VRAM prize is one draft layer (~1 GiB at `-c 262144`), and it does not currently reach the arena because the
   draft KV lives in the slab's `aux_reserve`.  The three open questions are default-on-or-not, one window size
   or per-model guidance, and MTP-only or DFlash2 too.
3. **Goal B** (a sparse attended set streamed cheaply — `STRATA-KV-STREAMING.md`) is **not started**.  It is a
   *different mechanism* and needs its own plan before any code; it does not depend on Goal A.

START BY READING, in this order:
  1. `wip/kv-residency/HANDOVER.md` — self-contained.  **§11 is this work list**, §1 the state, §3 the
     invariants (each cost a session), §5 the measurement rules, §6 the gates, §8.1 the absolute references
     for this box.
  2. `wip/kv-residency/A2-MIXED-BACKING.md` **§3.7.11** — the measurements block 15's claims rest on (the
     spilling-arm gates, the Flash-Next arm, the two bugs and their fixes, the arena re-measurement).
  3. `WORKLOG.md` **261030.002** (what shipped in r178 and how the shipped form differs from the campaign
     tree) and `patches/README.md` "Block 15" (the user-facing description).

STATE (verify, don't assume):
  - `~/llama.cpp`: **`rdna-boosts`** at **`38ef3a767`** is the shipped tree; **`wip-kv-residency`** at
    **`7c7769e50`** is the *campaign/measurement* tree.  They differ deliberately: the shipped one has **A1
    stripped** (the superseded two-base FA read), **no O1 fallback** (the region buffer is only taken when a
    window is active — O1 `GGML_ABORT`s on a slab-armed MoE config where the plain arm completes), and the
    **provider on by default** (`LLAMA_KV_REGION=0` kills it).  Never commit campaign code to `rdna-boosts`.
  - `~/llama-cpp-rdna-boosts` `main` = the record (delivery r178).  Campaign code snapshot:
    `wip/kv-residency/patches/0001`…`0014`, cut against `5bdec04ab`.
  - ONE R9700 (gfx1201), 32 624 MiB.  Pin `HIP_VISIBLE_DEVICES=0`; never claim a multi-GPU result.
  - Link 55.7 GB/s H2D (re-cabled 2026-10-30).  No pre-swap t/s number may be compared with a new one.

WHAT IS DONE — do not rebuild, do not re-gate, do not re-derive:
  - The window slides, `--fit` sizes and reports it, the pinned-RAM ceiling holds, the window apparatus is
    free, the spill's cost is a straight line, purity is proven on spilling arms (27B and Flash-Next), the
    state/checkpoint/MTP gates pass on spilling arms, and the arena half is measured (+2.2…+2.9 GiB when `W`
    is at or above the run depth; 11-26 % lost when it is 35 k cells below it).
  - `W` stays a **cell count** (numeric or `auto`); a percentage-of-slab form is deferred by the maintainer —
    do not re-propose it.  `auto`'s ordering (weights → arena reserve → `W` from the residue, capped at `-c`)
    is settled policy.

CONSTRAINTS:
  - Build `~/bin/build-llama-rocm` (never pass `-DCMAKE_HIP_FLAGS`); `LD_LIBRARY_PATH=/opt/rocm-10.1.0-gfx120X/lib`.
  - Hash text from a run WITHOUT `-v` via `scripts/extract-generated.py`; timings from a separate `-v` run.
    Always `--single-turn --no-display-prompt`, always `timeout`, never two GPU jobs at once.
  - **`ROCBLAS_USE_HIPBLASLT=0` for every fresh-process numeric comparison** (two identical commands otherwise
    differ; this cost a false alarm in A6).
  - Never judge on a greedy text hash alone; the prefill-logit gate on a spilling arm is the sensitive one.
    Never quote a timing from a `LLAMA_KV_REGION_DEBUG` arm.
  - Prove a spill before believing an arm: `LLAMA_KV_REGION_DUMP=/dev/null` → `KVREGION_FREE … host=` > 0 (or
    `FLIP > 0` with `LLAMA_KV_REGION_DEBUG=1`).  A spill needs `used > W + unit/nb1` cells — the shipped prose
    prompt is not enough on the 4B (use prose twice) — and the floor is one 4 MiB map unit per tensor.
  - `/tmp` is a 47 GB tmpfs: a prefill-logit base is ~7 GB, so clean old `.kld` files before a gate run
    (a full tmpfs truncates the base and the next compare fails on a read).
  - `-n 2000` for anything about the expert arena, and `-ncmoe` must be the model's full layer count.
  - Do not pick scope.  If something cannot be measured, say so and stop.

GATES if you touch the window's code: `GGML_CUDA_VMM_REGION_SELFTEST=1` (**PASS**, and **FAIL** with
`LLAMA_KV_REGION_QUARANTINE=0`), the coherence trio (`5ba44647188f`, `ROCBLAS_USE_HIPBLASLT=0`), the
prefill-logit gate on a **spilling** arm (−0.000000 / 100.000 %), `check-gpu-faults.sh` over every log, and the
OOM baseline with the knob unset.  Touch the VMM primitive and you also owe
`wip/kv-residency/hostmap-hazard-probe.cc` (default FAIL, `--grab 4` PASS).  Exact commands: HANDOVER §6.
