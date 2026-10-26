# Next-session prompt — `prefill-arena-residency` (R2) → beta-12 integration + delivery prep

You are resuming `wip/prefill-arena-residency` in `~/llama-cpp-rdna-boosts` (the delivery repo) and
`~/llama.cpp` (the fork checkout on `rdna-boosts`, `24380e5c7`).

**Read, in order:**
1. `AGENTS.md` — delivery policy (report findings before shipping, default-on policy, WIP rules,
   pushing policy, gates).
2. `wip/prefill-arena-residency/README.md` — design, prior art (PR #121), measured results.
3. `wip/prefill-arena-residency/CONCURRENCY-2026-10-18.md` — the two-slot stress and the 75/25 freeze
   negative result.
4. `wip/prefill-arena-residency/HANDOVER.md` — current checkpoint, beta-12 integration, the full gate
   list, the delivery checklist, and the exact repro commands.

**State.** The R2 prototype (cache-aware whole-table staging + immutable PRS, no freeze) is checkpointed
as `wip/prefill-arena-residency/r2-prototype.patch` (227 insertions / 3 files, sha256
`2bfdb8263b0b…`), applied **uncommitted** in `~/llama.cpp` and built.  The 75/25 freeze was implemented,
measured as a net prefill loss with no decode gain, and **dropped** — do not re-add it.  The serialized
scheduler is the prefill/decode coexistence limit (it is upstream, not ours); the per-layer pin is for
the future overlap work (`archive/work/moe-cpu-overlap/`), not for this cache.

**Your tasks, in order:**

1. **Confirm the checkpoint.**  Verify `r2-prototype.patch` sha (`2bfdb826…`), that
   `git diff` in `~/llama.cpp` matches it, that the build is current, and that the patch is R2-only
   (no freeze / activity / device-map code).
2. **Beta-12 integration.**  Beta candidate:
   `beta/0012-slab-transient-absorption/0012-rdna-boosts-beta-12-slab-transient-absorption.patch`
   (sha `9facba49…`).  Both apply orders already `git apply --check` clean on the r82 tree; now build
   and run in the canonical order (release → beta 12 → R2) and, if cheap, the reverse.  Run the 16k
   stage 0 / stage 1 A/B with beta 12, then 15.8k/`-ub 2048` and 31.6k.  Compare the arena size, the
   staged H2D split, `unusable`/`geom_decline`, the slab evict/re-arm counts, and the greedy hashes.
   Report the beta delta; do not assume neutrality.
3. **Gates** (with and without beta 12): coherence (dense 4B same-seed + Flash-Next host-expert
   stage 0 == stage 1), `scripts/gate-prefill-logits.sh` (expect KLD 0 / 100 %),
   `scripts/gate-qwen4exp-quant-coherence.sh`, the adaptive-MTP 4 axes + phase probe,
   `test-backend-ops -o MUL_MAT_ID`, `-sm layer` + `-sm tensor`, redirect mode
   (`GGML_SCHED_STAGE_MODE=1`), the concurrency harness A/B, `scripts/check-gpu-faults.sh`, and no
   `////`.
4. **Only if all green: prepare the delivery** per `HANDOVER.md` §3.3 — amend **block 09**, flip
   `GGML_MOE_CACHE_STAGE` default-on with `0` as the kill switch, fix the stale `ENVIRONMENT.md`
   `GGML_MOE_CACHE_INPLACE`/`STAGE` rows, update `patches/README.md` and `WORKLOG.md`, regenerate from
   a canonical `release.json.base` tree, `scripts/validate-set.sh` green, personal-fork `rdna-boosts`
   refresh with `--force-with-lease`, and the off-host worker container images.  **Never push to
   upstream.**  Surface any findings to the maintainer before any delivery-affecting commit or tag.

Start with the checkpoint verification and the beta-12 apply check exactly as in `HANDOVER.md` §4.

> Maintainer: add any focus for this session here.
