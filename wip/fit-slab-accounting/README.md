# MOVED — this campaign is archived (promoted)

`wip/fit-slab-accounting/` was **promoted in `v16-a55e952b8-r38` (2026-10-09) and moved to
[`archive/work/fit-slab-accounting/`](../../archive/work/fit-slab-accounting/README.md)** — its work is
folded into **block 06** (G1/G2 `--fit` arena+headroom reservation, G3 adaptive slab reserve, G4
`GGML_CUDA_OPTIONAL_ALLOC_MAX_FREE_PCT` free-VRAM cap, G5 split-slice guard, G6 nextn-offload guard, G7
scheduler multi-consumer fill, per-device host-expert accounting) plus the G4 FA-staging cap in **block
15**.  Validation record: the archive README §12-§15 and `WORKLOG.md` r38.

This stub exists so the historical pointers to `wip/fit-slab-accounting/...` — including the comments in
the shipped sources and the dated `WORKLOG.md` entries — still resolve.

**Follow-up (active):** move the op-offload H2D staging ring **inside** the movable-boundary slab, so it
stops competing with the non-routable post-slab consumers and the G4 free-VRAM cap is no longer needed for
it — see [`../slab-ring-region/README.md`](../slab-ring-region/README.md) (TODO #49).
