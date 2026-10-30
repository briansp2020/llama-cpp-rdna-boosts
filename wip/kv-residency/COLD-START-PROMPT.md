# Cold-session prompt — `wip/kv-residency` (TODO #118)

Paste the block below into a fresh session.  It assumes only what is in the repo (nothing from the previous
session's context).

---

Continue the windowed-KV campaign (TODO **#118**) in `~/llama-cpp-rdna-boosts` and `~/llama.cpp`.

**This session has ONE job: close the blocker.**  The maintainer's decision (2026-10-30) is that everything
else waits for it.

START BY READING, in this order:
  1. `~/llama-cpp-rdna-boosts/wip/kv-residency/HANDOVER.md` — self-contained; **§7.1 is the block and the
     plan**, §3 the invariants (each cost a session), §5 the measurement rules, §9 the re-baselined box.
  2. `.../A2-MIXED-BACKING.md` **§3.7** — this is the session's record: §3.7.1 the hardware re-baseline,
     §3.7.2/§3.7.3 what `--fit` and the RAM gate now do, §3.7.4 the ABORT the state gate found,
     **§3.7.6 the block (repro, the bisect already done, the two hypotheses)**, §3.7.7 the three slide bugs
     that are already fixed — do not reintroduce them.
  3. `.../MIXED-BACKING.md` — the design + the driver probe (background only).
Do not skip #1.  Do not re-derive anything §7.1 or §3.7.6 says is already excluded.

STATE (verify, don't assume):
  - `~/llama.cpp` is on branch `wip-kv-residency` at **`bbd69cda2`** (A3), pushed to the personal fork
    `git@github.com:stew675/llama.cpp.git`.  The superseded A1 fallback is `wip-kv-residency-a1-two-base`.
  - `~/llama-cpp-rdna-boosts` `main` = the record (HANDOVER / A2 / INVENTORY / TODO / patches 0001-0009).
  - `rdna-boosts` is NOT the campaign's branch: never commit campaign code there.
  - Nothing from the campaign is in `patches/`, `release.json` or any block, and nothing may be, without the
    maintainer's explicit promotion go-ahead.
  - ONE R9700 (gfx1201).  Pin `HIP_VISIBLE_DEVICES=0` on every arm; never claim a multi-GPU result.
  - **The box was re-cabled on 2026-10-30: the link is now 55.7 GB/s H2D / 56.8 D2H** (it was 28.8 when every
    campaign number was measured).  Do not compare a new t/s number with an old one; the cost model is
    `bytes / 55.7 GB/s`.  The probes are re-run and recorded in `A2-MIXED-BACKING.md` §3.7.1.

WHAT IS ALREADY DONE (do not rebuild, do not re-gate):
  - The windowed KV works and slides: one KV tensor in one VA reservation, the cells below the window in pinned
    system RAM, the window in VRAM, no kernel or op change.
  - `--fit` **sizes and reports** the window (`LLAMA_KV_RESIDENT_CELLS=auto`), the spill is capped by
    `LLAMA_KV_SPILL_RAM_PCT` (creation-time clamp + run-time refusal), `gate-ckpt-reuse.sh` and
    `gate-state-pipe.sh` **pass on a windowed arm** (the latter found and fixed an abort in the batched state
    save), and MTP verify widths are byte-identical on a windowed arm (4B).
  - Therefore: the owed list's items 2-4 are **closed**.  Item 1 (`--fit`) is **closed**.  What is open is the
    purity blocker below.

THE BLOCKER (`A2-MIXED-BACKING.md` §3.7.6, HANDOVER §7.1) — a spilled KV is not numerically transparent on a
27B:
  - a **single 4 MiB tensor** spill changes the greedy text (`305 chars e2cea6fa6670` plain → `310
    ccdb98b7b254` with only tensor 0 spilled), and the prefill-logit gate reads **KLD 0.004973 / 98.761 %** at
    `--ctx 8192 --chunks 4 --kv q8_0 --ab "LLAMA_KV_REGION=1 LLAMA_KV_RESIDENT_CELLS=1024"`;
  - with the window on but **nothing spilled** it is `0.000000 / 100.000 %` and the plain hash — so it is the
    spilled cells, not the buffer type;
  - a 4B arm with the same spill shape (f16 *and* q8_0 KV) is byte-identical, so it is not the quant type;
  - it is deterministic (two `W` values with the same boundary give the same text) and the flip's byte
    round-trip verifies.

  The bisect tool already exists: `LLAMA_KV_REGION_SPILL_SKIP=<tensor indices>` (DIAGNOSTIC, unset = shipped
  behaviour) keeps the listed tensors fully resident.  Work §7.1's plan in its stated order — (1) is the
  stage-OUT blind spot in the verify, (2) is a lost kernel store into host-backed memory (`--chunks 1` vs
  `--chunks 2`), (3) is why the 4B is immune (`nb1` 1088 vs 2048; 1088 is not a multiple of 128).
  Report findings to the maintainer **before** committing; if it cannot be closed, say so and stop.

CONSTRAINTS:
  - Build `~/bin/build-llama-rocm` (never pass `-DCMAKE_HIP_FLAGS`); `LD_LIBRARY_PATH=/opt/rocm-10.1.0-gfx120X/lib`.
  - Hash text from a run **without** `-v` via `~/llama-cpp-rdna-boosts/scripts/extract-generated.py`; read
    timings and the cache/arena lines from a separate `-v` run.  Always `--single-turn --no-display-prompt`,
    always `timeout`, never two GPU jobs at once.
  - A fresh-process numeric comparison needs `ROCBLAS_USE_HIPBLASLT=0` (hipBLASLt picks its solution per
    process start — it will otherwise invent a ~0.005 KLD difference and waste your session; `AGENTS.md`).
  - Do not judge anything on a greedy text hash alone: a logit-level probe decides.
  - `-n 2000` for anything about the expert arena, and `-ncmoe` must be the model's full layer count.
  - Do not pick scope.  If something cannot be measured, say so and stop.

GATES to re-run only if you change code: the VMM selftest (`GGML_CUDA_VMM_REGION_SELFTEST=1`, must print PASS,
it now covers the window-boundary flip), the §6 identity trio (`5ba44647188f` with the feature off),
`check-gpu-faults.sh` over every log, and the OOM config still OOMing with the feature off.  The exact commands
are HANDOVER §6.
