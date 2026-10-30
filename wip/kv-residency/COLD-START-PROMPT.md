# Cold-session prompt — `wip/kv-residency` (TODO #118)

Paste the block below into a fresh session.  It assumes only what is in the repo (nothing from the previous
session's context).

---

Continue the windowed-KV campaign (TODO **#118**) in `~/llama-cpp-rdna-boosts` and `~/llama.cpp`.

START BY READING, in this order:
  1. ~/llama-cpp-rdna-boosts/wip/kv-residency/HANDOVER.md   (self-contained; §3 the invariants, §5 the
     measurement rules, §9 the pending hardware change)
  2. ~/llama-cpp-rdna-boosts/wip/kv-residency/A2-MIXED-BACKING.md  (the measured state of the build)
  3. ~/llama-cpp-rdna-boosts/wip/kv-residency/MIXED-BACKING.md     (the design + the driver probe)
Do not skip #1.  Its §5 (measurement rules) and §3 (invariants) each cost a session to learn.

STATE (verify, don't assume):
  - delivery repo ~/llama-cpp-rdna-boosts is pushed; r176 is the current release.
  - ~/llama.cpp is on branch wip-kv-residency at ba5c7205e (campaign code, pushed to the personal fork);
    branch wip-kv-residency-a1-two-base at e49e1cef4 is the superseded A1 fallback.
  - rdna-boosts is NOT the campaign's; never commit campaign code there.
  - The campaign's code is NOT in patches/, release.json or any block.  Nothing may be.
  - ONE R9700 (gfx1201) is attached: pin HIP_VISIBLE_DEVICES=0 on every arm, and never claim a multi-GPU result.
  - The maintainer is re-doing the PCIe layout for ~56 GB/s H2D (from the 28.8 GB/s every campaign number is
    calibrated on) and a faster model SSD.  If the swap has happened, RE-MEASURE the link constant
    (wip/kv-residency/bw-probe.cc) before quoting or comparing any t/s, and re-run mixed-backing-probe.cc.

WHAT IS ALREADY DONE (do not rebuild it):
  A2 is implemented and measured: one KV tensor in one VA reservation, the cells below the window backed by
  pinned system RAM and the window by VRAM, the window SLIDING with the used-cell watermark.  No kernel or op
  change.  Measured: the 27B Q8_0 -c 262144 OOM config runs (18.6 t/s at 51 475 tokens = the fully-resident
  reference), and slab mode's arena gains ~3-5 GiB which is +7.9…+10.1 % decode where the arena is starved
  (122B-A10B) and +8.6…+13.1 % prefill everywhere.  Gates green: text byte-identical to the fully resident
  arm, prefill logits 0.000000 / 100.000 %, no GPU faults.

THE ASK NOW — in priority order:
  1. --fit SIZING of the window.  The agreed shape: the window slides (done), `W` is sized from the depth, and
     the default stays W = -c (no spill).  The break-even is measurable and is written down in HANDOVER §7.1
     and A2-MIXED-BACKING §3.5-§3.6 (Flash-Next at -c 65536 frees 0.39 GiB for -14.6 % decode; at -c 262144 it
     frees 3.63 GiB for 0 % with +13.1 % prefill).  --fit must also REPORT the split (resident MiB of total,
     spilled MiB, window in cells, modelled cost at the current depth) instead of warning and running anyway.
  2. scripts/gate-ckpt-reuse.sh and scripts/gate-state-pipe.sh on a windowed arm — the two paths the slide
     touches that the identity arms do not (a restore writes arbitrary cell indices; grow_for maps the right
     backing kind by position).
  3. The MemAvailable gate before pinning the spill (MOE_HOST_POOL_RAM_PCT pattern; up to 8.5 GiB at -c 262144
     on a 27B, non-swappable).
  4. MTP verify widths on a windowed arm (cheap; A1 gated them).
  Goal B (QSA/DSA sparse attended set) is NOT next.  Multi-device is not measurable here.

CONSTRAINTS:
  - Build: ~/bin/build-llama-rocm (never pass -DCMAKE_HIP_FLAGS).
    LD_LIBRARY_PATH=/opt/rocm-10.1.0-gfx120X/lib
  - llama-cli prints accounting only with -v, and -v interleaves log text into the generation -- hash text from a
    run WITHOUT -v via ~/llama-cpp-rdna-boosts/scripts/extract-generated.py, and read timings from a separate -v
    run.  Always --single-turn --no-display-prompt, always timeout, never two GPU jobs.
  - -n 2000 for anything about the expert arena (shorter runs are drowned out by the cache warm-up).
  - -ncmoe must be the model's FULL layer count when the point is the arena (35B-A3B 48, 122B-A10B 48,
    Flash-Next 48).
  - Read a delta only from arms interleaved back-to-back in one batch (this box's clocks drift).
  - Do NOT pick scope.  If something cannot be done or measured, say so and stop; never substitute something
    that is measurable instead.
  - Surface findings and open questions to the maintainer BEFORE any commit that affects the delivery, and do
    not fold a finding silently into a commit message.

GATES to report before committing (the exact commands are HANDOVER §6): the VMM selftest, text identity
plain-vs-window (hash without -v), gate-prefill-logits.sh with --ab "LLAMA_KV_REGION=1
LLAMA_KV_RESIDENT_CELLS=<W>" reading 0.000000 / 100.000 %, check-gpu-faults.sh over every log, and the OOM
config still OOMing with the feature off.
