# Open WIP campaigns (index)

This is the index `AGENTS.md` refers to.  `wip/` holds **active exploration only** — nothing here is part
of the delivery, and nothing here may be applied to the fork without the maintainer's explicit go-ahead
(see the WIP and promotion rules in `AGENTS.md`).  Each campaign is a self-contained handover under its own
`README.md` (there is no `wip/README.md`).

**A closed or promoted campaign does not live here — not even as a redirect stub.**  Its record moves to
`archive/work/<campaign>/` and `WORKLOG.md`; the history in dated records refers to the path that was
correct at the time.  Keeping finished work under `wip/` (or leaving a "MOVED — see archive" placeholder)
makes the directory lie about what is in flight, so the stub habit is retired.  See
`archive/work/lightning-indexer-fusion/README.md` for the resolution pattern
(investigate -> fold into the owning block -> regenerate + validate -> record -> archive -> ship).

An entry below is one of exactly two things: a campaign with **work in flight**, or a **scoping note for
work not yet started** (a plan, not a result).  Both stay.  The distinction matters when tidying: a
"MOVED — archived/closed/promoted" redirect is finished work and is deleted, while an unstarted scoping
note is open work and stays.  When in doubt, look for the word MOVED at the top of the `README.md` — that
is the signature of a stub, not of a campaign.

| directory | what | status |
|---|---|---|
| [`moe-verify-fusions/`](moe-verify-fusions/README.md) | **the MoE verify step tracked as ONE campaign**: the cache-aware fusion guard's global-vs-per-table gating (#50), the 3-GPU MTP verify regression (#51), and the plain-vs-MTP recurrent-rewind divergence | **PARTIALLY RESOLVED (2026-10-10)** — **#50 is RESOLVED and DELIVERED in `v16-a55e952b8-r40`** (the removal + the non-devmap WIP fixes are folded into the block set): DEVMAP was dropped entirely and the eager `-sm tensor` split path is MTP-pure on qwen4exp (`none == n3`), budget-sweep-pure, and within noise of the old `DEVMAP=0` perf; the divergence was a layout-sensitive **hipBLASLt** flip (issue #67), not a cache bug ([`FINDINGS-devmap-removal.md`](moe-verify-fusions/FINDINGS-devmap-removal.md), `remove-devmap.diff`).  The slab-resident **narrow-2** layout is implemented + functionally validated ([`RESULTS-slab-narrow2.md`](moe-verify-fusions/RESULTS-slab-narrow2.md)).  Open: **#51** (3-GPU MTP) and **#52** (multi-sequence MTP).  **#53** — the 35B-A3B (`qwen35moe`) `-sm tensor` `+` `-ncmoe` impurity — is **RESOLVED and DELIVERED in `v16-a55e952b8-r41`**: the cause was block 12's **lazy `ncclCommInitAll`** (r34), not the internal AR (which is bit-exact); the hybrid init is eager again, so `-ncmoe 99` + auto cache is `none == n1 == n3 == n7` and `none == n3` on the long field, `-sm layer` and qwen4exp pure, perf neutral.  The **probe is fixed and shipped** (folded into block 15: the cache-aware down-fold guard no longer stands down when the table is un-serveable; blocks 00-14 unchanged) and the personal fork's `rdna-boosts` is at the folded tip.  The earlier root-cause text ("what remains is the field, now root-caused to the internal host-staged all-reduce") is **superseded** — the AR is a symptom, not the site; see [`moe-verify-fusions/HANDOVER-35b-internal-ar.md`](moe-verify-fusions/HANDOVER-35b-internal-ar.md) and `WORKLOG.md` 2026-10-12 (r41). |
| [`cache-split-admission/`](cache-split-admission/README.md) | expert-cache admission policy on `-sm tensor` split MoE tables (`MOE_EXPERT_CACHE_DEVPOLICY_SPLIT`); today the split tables use the slow host promotion | **OPEN / measured** — re-measured 2026-10-08 on r34 (bounded 2048 MiB arena; at the default AUTO the arena is fully resident and the choice is moot): device policy wins big in plain decode (35B `-sm tensor` 10.2 -> **36.5** t/s), byte-identical, but MTP n3 still favours host promotion (3-GPU `MOE_EXPERT_CACHE_MIB=16384`: **76.5** vs 55.6 t/s), so a blanket default flip would regress MTP |
| [`fp8-support/`](fp8-support/README.md) | native FP8 E4M3 for RDNA4 | PARKED (2026-09-26) — rebased and building, fully RDNA4-gated, but **gate 2 failed** (the fp8 4B is 0.69x Q8_0 where cllm had 1.16x, because the delivery's MMB/GEMM work made Q8_0 +32 % and the fp8 WMMA path did not follow).  Resume actions are in `PLAN.md`.  Not closed — paused behind the delivery work |
| [`host-memory-footprint/`](host-memory-footprint/README.md) | host-memory footprint of GPU-resident weights (gfx1100) | OPEN (seeded 2026-10-02) |
| [`mmvq-verify-rows/`](mmvq-verify-rows/README.md) | faster multi-token mmvq on RDNA4 (bit-exact, 2-8-token verify width) | OPEN — **not integrated**; the patch series is validated against `v16-84e76d8a2-r17`, so it needs a re-base and a re-validation before it can be promoted |
| [`moe-cpu-overlap/`](moe-cpu-overlap/README.md) | genuine CPU/GPU overlap for the expert misses (the Strata shape) | OPEN / scoping (TODO #36) — **work in flight**: the note refutes the earlier "cannot be done" verdict for the interleaved shape and sets out the pipeline to try |
| [`nwarps/`](nwarps/README.md) | per-M `nwarps` MoE candidate — the one deliberate width-purity impurity | ACTIVE (env-OFF) |
| [`strata-amd-kernels/`](strata-amd-kernels/README.md) | compare Strata's AMD decode kernels against block-10/13/15 (TODO #35) | **scoping note — not yet started**: no implementation yet, just the candidate list and the comparison method |
