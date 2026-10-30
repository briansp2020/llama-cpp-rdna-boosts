# Cold-session prompt — `wip/kv-residency` (TODO #118)

Paste the block below into a fresh session.  It assumes only what is in the repo (nothing from the previous
session's context).  Rewritten 2026-10-30 at the end of session A8 (the gate fix + the split-window design).

---

Continue the windowed-KV campaign (TODO **#118**) in `~/llama-cpp-rdna-boosts` and `~/llama.cpp`.

**Goal A is SHIPPED — do not re-measure it and do not re-promote it.**  Release **`v16-6050b467c-r181`**
(16 blocks) carries the windowed KV as **block 15**: an **opt-in** fixed-VRAM resident window with the KV's
tail in pinned system RAM (`LLAMA_KV_RESIDENT_CELLS=<cells>|auto`; unset = the plain device buffer, so a run
without the knob is byte-identical to r177 including its OOM).  The shipped tree is
`~/llama.cpp`'s `rdna-boosts` at **`83d2c391a`** (= `release.json.tip`); the campaign/measurement tree is
`wip-kv-residency` at `7c7769e50` and does **not** carry r181's fixes.  Shipped description:
`patches/README.md` "Block 15", `ENVIRONMENT.md` §17, `WORKLOG.md` **261030.002/.003/.005**, and
`A2-MIXED-BACKING.md` **§3.7.11** (measurements) + **§3.7.12** (r181's analysis).

**What r181 fixed (know this before touching the block):**

* **Multi-device `--fit` aborted at model load.**  The block reaches its backend entry points through
  `ggml_backend_dev_backend_reg(dev)`; a **Meta** (tensor-split) device has no reg (`device->reg == NULL`) and
  the proc-address helper ASSERTS on a null reg — three paths (the fit's KV probes on any run, the runtime buft
  lookup with a numeric `W`, the spill budget/link lookups).  Both selectors now resolve the reg through a
  reg-less-tolerant helper (`llama_kv_region_dev_reg`, `fit_dev_reg` — the latter walks a Meta device's simple
  child), and a reg-less device means "the backend does not answer": **the window is declined on a `-sm tensor`
  split with one message** (HANDOVER invariant 13).  The campaign's box is single-GPU, which is why nothing
  caught it.
* **Three `auto` wiring defects**, all reproducible on one device: a decided "no window needed" was reported as
  "the fit did not run" (a decided-0 flag now exists); every run *without* the knob logged a KV warning (now
  silent); and the `LLAMA_KV_SPILL_RAM_PCT` clamp could be taken past the VRAM budget (invariant 14 — an
  intermediate form planned a **235 GiB** pinned spill and OOM-killed the box, so **bound the spill PLAN**).

**WHAT SESSION A8 (2026-10-30) DID — do not redo it:**

1. **The `-sm tensor` design note exists**: `wip/kv-residency/A6-TENSOR-SPLIT-WINDOW.md` (+
   `A2-MIXED-BACKING.md` §3.7.13 §2).  Design only, nothing built.  It lists the five changes, the two code
   facts that make it smaller than it looks (the cell axis is never split; the Meta buft already fans allocation
   and init per child) and **four `common/fit.cpp` findings a blind implementation would get wrong** — read §4
   of it before touching the block: `margins[0]` is device 0's margin not the sum; `kv_bytes/cell_bytes` are
   already aggregate; the pinned-spill **plan** under-counts `n_devices×` for a MIRRORED cache; the split
   `estimate()` counts the KV `ntd×`.  Implementing it is the maintainer's call and needs a 2/3-device host.
2. **The parking-gate test gap is fixed** (`scripts/gate-park-idle-slots.sh`): it now counts park MOVES (not the
   region refusal warning logged under the same `seq_park` prefix) and grades the save-and-clear fallback as a
   PASS with `park == lazyoff` still required.  Verified on a windowed Flash-Next arm and the no-window control.
   **It is UNCOMMITTED, waiting for the maintainer's go-ahead.**

**WHAT IS LEFT — in order:**

1. **`auto` policy** (the maintainer's call; unchanged by r181).  `auto` sizes the window to
   `free − margin − everything-but-the-KV`, where the margin includes the **arena/slab reserve** — so on a
   `-ncmoe` config the arena wins the VRAM and the window comes out small *by design* — and it only spills a
   `-c` the user pinned.  The deciding measurements are `A2-MIXED-BACKING.md` §3.7.11.4 A–C.  `W` stays a cell
   count; a percentage-of-slab form is deferred — do not re-propose it.
2. **Goal C decisions** (`O3-DRAFTER-RING.md` §7): the MTP drafter ring (`LLAMA_SPEC_DRAFT_WINDOW`) is
   measured, not promoted — free at a usable window (46k context, `W=32768`: acceptance −2.3 % relative,
   generation flat), 7-9 % lost far below the working context; the prize is one draft layer (~1 GiB at
   `-c 262144`) and it does not reach the arena.  Open: default-on or not, one size or per-model, MTP-only or
   DFlash2 too.
3. **The split window** — implement A6 only on the maintainer's go-ahead and only with a 2/3-device host.
4. **Goal B** (a sparse attended set streamed cheaply — `STRATA-KV-STREAMING.md`) is **not started**: a different
   mechanism, needs its own plan before any code, independent of Goal A.

START BY READING, in this order: `wip/kv-residency/HANDOVER.md` (**§11 = this work list**, §1 the state, §3 the
invariants — **13** reg-less Meta devices, **14** bound the planned spill — §5 the measurement rules, §6 the
gates, §8.1 the absolute references); then `A2-MIXED-BACKING.md` §3.7.11 + §3.7.12; then `WORKLOG.md` 261030.005.

STATE (verify, don't assume):
  - `rdna-boosts` `83d2c391a` = shipped; `wip-kv-residency` `7c7769e50` = campaign/measurement (no r181 fixes).
    Never commit campaign code to `rdna-boosts`.
  - ONE R9700 (gfx1201), 32 624 MiB.  **Check `rocm-smi --showmeminfo vram` first** — the maintainer runs a
    Flash-Next server here that holds ~34 GB, and every fit/gate then reads garbage (`free = 2 MiB`) or OOMs.
    Never run a second GPU job; if the card is in use, say so and stop.
  - Link 55.7 GB/s H2D (re-cabled 2026-10-30).  No pre-swap t/s number may be compared with a new one.

CONSTRAINTS:
  - Build `~/bin/build-llama-rocm` (never pass `-DCMAKE_HIP_FLAGS`); `LD_LIBRARY_PATH=/opt/rocm-10.1.0-gfx120X/lib`.
  - Hash text from a run WITHOUT `-v` via `scripts/extract-generated.py`; timings from a separate `-v` run.
    Always `--single-turn --no-display-prompt`, always `timeout`, never two GPU jobs at once.
  - **`ROCBLAS_USE_HIPBLASLT=0` for EVERY fresh-process numeric comparison** (unset, two identical commands
    differ — that nearly derailed the promotion).
  - Never judge on a greedy text hash alone; the prefill-logit gate on a **spilling** arm is the sensitive one.
    Never quote a timing from a `LLAMA_KV_REGION_DEBUG` arm.
  - Prove a spill before believing an arm: `LLAMA_KV_REGION_DUMP=/dev/null` → `KVREGION_FREE … host=` > 0 (or
    `FLIP > 0`).  A spill needs `used > W + unit/nb1` cells (the shipped prose prompt is not enough on the 4B —
    use it twice); the floor is one 4 MiB map unit per tensor.
  - `/tmp` is a 47 GB tmpfs and a prefill-logit base is ~7 GB — clean old `.kld` files before a gate run.
  - `-n 2000` for anything about the expert arena, with `-ncmoe` = the model's full layer count.
  - Do not pick scope.  If something cannot be measured, say so and stop.

GATES if you touch the block: `GGML_CUDA_VMM_REGION_SELFTEST=1` (**PASS**, and **FAIL** with
`LLAMA_KV_REGION_QUARANTINE=0`), the coherence trio (`5ba44647188f`, with `ROCBLAS_USE_HIPBLASLT=0`), the
prefill-logit gate on a **spilling** arm (−0.000000 / 100.000 %), `check-gpu-faults.sh` over every log, and the
OOM baseline with the knob unset.  Plus, for this block specifically: no un-checked
`ggml_backend_dev_backend_reg` (invariant 13), the planned-spill bound (invariant 14: `llama-fit-params` with a
big `-c` and `LLAMA_KV_SPILL_RAM_PCT=1` must REPORT and drop the window), and the three `auto` wiring arms in
HANDOVER §6.  Touch the VMM primitive and you also owe `wip/kv-residency/hostmap-hazard-probe.cc` (default
FAIL, `--grab 4` PASS).
