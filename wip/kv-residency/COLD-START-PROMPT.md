# Cold-session prompt — `wip/kv-residency` (TODO #118)

Paste the block below into a fresh session.  It assumes only what is in the repo (nothing from the previous
session's context).  Rewritten 2026-10-30 at the end of session A5.

---

Continue the windowed-KV campaign (TODO **#118**) in `~/llama-cpp-rdna-boosts` and `~/llama.cpp`.

**The measurement work is finished; what is left is a promotion decision plus one owed gate change.**  Goal A's
mechanism, its purity and its cost are all measured on arms that **provably spill**: a spilled KV is numerically
transparent (prefill-logit KLD −0.000000 on a spilling 27B *and* on a spilling Flash-Next, text byte-identical),
the spill is a straight line (~8-11 µs per MiB per token at the 55.7 GB/s link), the state/checkpoint/MTP gates
pass on spilling arms, and the arena half is measured.  Session A5 also found and fixed the last two bugs: a
windowed region's tensors were not placed on the map grid (the second tensor's first slide was refused and the
process aborted on Flash-Next — patch `0013`), and an idle-slot **park** on a region-backed KV faulted the GPU
(a park writes above the region's mapped watermark through the copy engine — a region-backed KV now declines it
and the server save-and-clears instead, patch `0014`).  **Do not re-run or re-derive any of that**; the records
carry the numbers and the exact commands.

START BY READING, in this order:
  1. `~/llama-cpp-rdna-boosts/wip/kv-residency/HANDOVER.md` — self-contained.  **§11 is the next session's work
     list**, §7.2 is the campaign's work list (items 1-4 all closed in A5), §3 the invariants (each cost a
     session; **#10** is the A4 hazard hold, **#11** the tensor-placement rule, **#12** the park/copy-engine
     rule), §5 the measurement rules (#6: never time a `LLAMA_KV_REGION_DEBUG` arm), §6 the gates, §8.1 the
     absolute references for this box, §9 the re-baselined box.
  2. `.../A2-MIXED-BACKING.md` **§3.7.11** — the A5 session: the spilling-arm gates, the Flash-Next arm, **both
     bugs and their fixes with their gate tables**, and the arena re-measurement (tables A-C).  Then §3.7.8 (the
     A4 hazard and its hold) and §3.7.9 (the spill's cost).  §3.7.6's banner lists readings that were **wrong**.
  3. `.../MIXED-BACKING.md` — the design, and the residual-risk table in §4 (background only).

STATE (verify, don't assume):
  - `~/llama.cpp` is on branch `wip-kv-residency` at **`7c7769e50`**, pushed to the personal fork
    `git@github.com:stew675/llama.cpp.git`.  The superseded A1 fallback is `wip-kv-residency-a1-two-base`.
    `rdna-boosts` is **NOT** this campaign's branch: never commit campaign code there.
  - `~/llama-cpp-rdna-boosts` `main` (pushed) = the record.  `patches/0001`…`0014` apply in order to
    `5bdec04ab` (r174) and reproduce the branch tip's tree exactly; the delivery itself is at **r177**.
  - ONE R9700 (gfx1201), 32 624 MiB.  Pin `HIP_VISIBLE_DEVICES=0`; never claim a multi-GPU result.
  - The link is 55.7 GB/s H2D (re-cabled 2026-10-30).  No pre-swap t/s number may be compared with a new one.

WHAT IS DONE — do not rebuild, do not re-gate:
  - The window slides, `--fit` sizes and reports it (`LLAMA_KV_RESIDENT_CELLS=auto`), `LLAMA_KV_SPILL_RAM_PCT`
    caps the pinned spill, and the window apparatus is free (zero-spill arms measure the same as plain).
  - Purity on spilling arms: 27B `-c 8192` q8_0 `W=1024` → KLD −0.000000 / 100.000 %; Flash-Next `-ncmoe 48`
    `-c 65536` `W=1024` (two KV caches, 576 + 48 MiB spilled) → text identical, KLD −0.000000 / 99.994 %,
    `FLASH_ATTN_QSA` 26/26, 156/156 flips verified.
  - The three correctness corners on spilling arms: `gate-ckpt-reuse.sh` PASS (128 MiB host spilled),
    `gate-state-pipe.sh` PASS (all five arms, 80 MiB host), MTP verify widths one hash across
    plain/window × decode/draft-mtp.
  - The arena half, `-n 2000`, `-ncmoe` = full layer count, `-c 262144`, interleaved: +2.2 to +2.9 GiB of arena
    (+3.9 to +6.7 points of residency) whichever way the window is sized, and it costs nothing when `W` is at or
    above the depth actually run (122B-A10B: +3.8 % decode / +4.7 % prefill); a `W` 35 k cells below the depth
    costs 11-26 %.
  - `--fit`'s `auto` ordering is settled policy: weights → arena reserve → `W` from the residue, capped at the
    user's `-c`.  **`W` as a percentage of the slab is deferred by the maintainer — do not re-propose it.**

WHAT IS LEFT (all of it is a decision, not a measurement):
  1. **The promotion conversation** — a delivery block, `ENVIRONMENT.md`, and the default-on call for the window.
     The campaign's evidence is complete; the maintainer decides whether it ships, and a promotion re-cuts the
     patch series against the then-current release (now r177).
  2. **One owed DELIVERY change (needs the maintainer's go-ahead):**
     `scripts/gate-park-idle-slots.sh` HARD-fails a legitimate outcome.  When the park arm's server logs *"cannot
     park the idle slot: saving it to the prompt cache and clearing it"* — the fallback a region-backed KV now
     triggers, and which also exists for a slot whose cells are not one run (a `seq_cp` shared prefix) — the gate
     should require `park == lazyoff` and report *"parking declined → save-and-clear fallback"* instead of failing
     on *"parking moved nothing"*.  Shape: `A2-MIXED-BACKING.md` §3.7.11.3.
  3. Goal B (a sparse attended set streamed cheaply) is not started; multi-device is not measurable here.

CONSTRAINTS:
  - Build `~/bin/build-llama-rocm` (never pass `-DCMAKE_HIP_FLAGS`); `LD_LIBRARY_PATH=/opt/rocm-10.1.0-gfx120X/lib`.
  - Hash text from a run WITHOUT `-v` via `~/llama-cpp-rdna-boosts/scripts/extract-generated.py`; read timings
    and the cache/arena lines from a separate `-v` run.  Always `--single-turn --no-display-prompt`, always
    `timeout`, never two GPU jobs at once.  `ROCBLAS_USE_HIPBLASLT=0` for a fresh-process numeric comparison.
  - Never judge anything on a greedy text hash alone; the prefill-logit gate on a spilling arm is the sensitive
    one.  Never quote a timing from a `LLAMA_KV_REGION_DEBUG` arm.
  - If an arm must provably spill, check `FLIP > 0` (`LLAMA_KV_REGION_DEBUG=1`) or `KVREGION_FREE … host=` > 0
    (`LLAMA_KV_REGION_DUMP=/dev/null`) first: a spill needs `used > W + unit/nb1` cells, so the shipped prose
    prompt is **not** enough on the 4B (use prose twice).  The spill floor is one 4 MiB unit per tensor.
  - `-n 2000` for anything about the expert arena, and `-ncmoe` must be the model's full layer count.
  - Do not pick scope.  If something cannot be measured, say so and stop.

GATES if you change code: the VMM selftest (`GGML_CUDA_VMM_REGION_SELFTEST=1` — **PASS**, and **FAIL** with
`LLAMA_KV_REGION_QUARANTINE=0`), the §6 identity trio (`5ba44647188f` with the feature off), the prefill-logit
gate on a spilling arm (−0.000000 / 100.000 %), `check-gpu-faults.sh` over every log, and the OOM config still
OOMing with the feature off.  Touch the VMM primitive and you also owe `hostmap-hazard-probe.cc` (default FAIL,
`--grab 4` PASS).  Exact commands: HANDOVER §6.
