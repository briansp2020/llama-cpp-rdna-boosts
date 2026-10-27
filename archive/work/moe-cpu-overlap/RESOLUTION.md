# RESOLUTION — the MoE-miss CPU/GPU-overlap campaign is retired (2026-10-26, r144)

**Verdict: retired, no delivery change.**  `TODO #36` is closed.  O0 — the campaign's own decision gate
("if the cold read is < 20 % of the decode token, park this") — measured the cold pressure at **0.1-0.4 %
of a decode token** at the delivered arena size, and **7 %** at the smallest arena the cache permits
(2 048 MiB/device, 6 % residency).  A CPU worker pool computing the cold experts beside the GPU could
therefore recover ~0.1 ms of a 28 ms token at a realistic arena, and ~2 ms in the deliberately starved
arm — for a change that would have to break graph capture at every `MUL_MAT_ID`, mix CPU and GPU numerics
into the same layer (so greedy purity could no longer be promised), and add a per-op host hand-off on the
latency-critical path.  The maintainer's decision (2026-10-26) was to retire the item outright rather
than park it, and to *not* implement the overlap.

**Evidence:** [`FINDINGS-2026-10-26.md`](FINDINGS-2026-10-26.md) — the full measurement record (scope,
config, the four arena arms, the traffic conversion, the kernel timeline, and the two findings the
profile produced instead).

**What replaces it.**  The same profile showed the decode token is dispatch-bound, not memory-bound: 297
`graph_compute` calls and ~12 000 node dispatches per token, ~13.5 ms of host time, and the CUDA/HIP
graph never replays on this shape because the Meta rebuild moves node/source identities between
otherwise-identical tokens.  That work is the new campaign
[`archive/work/graph-replay-placement/`](../graph-replay-placement/README.md) (`TODO #107`, closed by r149), and it is
where the same 2× R9700 / Flash-Next IQ4_NL shape continues.

**Delivered in r144 from this session** (both are diagnostics; no kernel, no numerics, no default):

* the expert cache's `expert H2D traffic … MiB/token` report now divides by a real decode-band op census
  snapshotted at sizing, instead of `g_takeover / n_tables` (which printed `over ~0 tokens = 0.00
  MiB/token` on every run that took the identity fast path, or after the r143 teardown dropped the
  tables — so the one number this campaign needed was unusable);
* `GGML_CUDA_GRAPH_TRACE` now also reports how every `graph_compute` call was served
  (`direct` / `replay` / `capture` / `properties_changed`), which is the instrument that made the
  dispatch finding visible.

**Relates to:** `archive/work/moe-expert-cache/` (the cache itself and the earlier negative
"CPU-computes-the-misses" arm this campaign was going to revisit), `archive/work/moe-cache-autosize/`,
`archive/work/three-gpu-performance/` (the `-sm layer` pipeline-stage loss), `archive/work/graph-key-churn/`
(the r115 graph-key fix this finding is the sibling of).
