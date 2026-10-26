# deep-prefill-depth-scaling — issue #214's long-prompt prefill regression

**Status: RDNA4 CLOSED (r126); the RDNA2 half is NOT confirmed and is open as `TODO.md` #106.**
The 2026-10-24 resolution below was based on an interim relay; the reporter's posted 2026-10-25 r126 sweep
shows the crossover on his gfx1031 card.  See "Correction" at the end.

The reporter's box never had an expert-cache arena, so the r123 band clamp is not their symptom; what is left
is a deep-context prefill regression.  This campaign reproduced the shape on gfx1201, separated our patch set
from upstream drift, and traced it to an RDNA4 FA config row: our decode/verify **band** row is also selected
by wide prefill (they share the `ncols = 64` instantiation), costing 26 % of the FA kernel at
`pp512 @ d98304`.

* Finding + numbers: [`RESULTS-2026-10-24.md`](RESULTS-2026-10-24.md)
* Repro harness: [`sweep.sh`](sweep.sh) (three builds x resident/streaming, one run at a time)
* Items: `TODO.md` **#106** (RDNA2 residual, open) and **#98** (the instantiation split, shipped in r126)
* Doctrine: `GREEDY-PURITY.md` §51; block reference: `patches/README.md` block 05; records: `WORKLOG.md`
  261024.006 (the campaign) and 261025.012 (the r138 sweep and the correction).
* r138 depth sweeps on gfx1201, with and without the load mode pinned:
  [`benchmarks/2026-10-25-r138-depth-sweep.md`](../../../benchmarks/2026-10-25-r138-depth-sweep.md).

Nothing here is part of the delivery.

## Resolution (r129, 2026-10-24) -- the RDNA4 half

The RDNA4 half shipped in r126 (`WORKLOG.md` 261024.007): the `ncols == 64` band row is no longer the
prefill row, and gfx1201 deep-context prefill is +20 % over upstream again.  `TODO.md` #97 was retired on
that change.

## Correction (2026-10-25, r138/r139) -- the RDNA2 half is not confirmed

The 2026-10-24 note above said the reporter's RDNA2 sweep "works well" (a maintainer relay).  His posted
2026-10-25 r126 sweep says otherwise: on the 12 GiB gfx1031 card the delivery's lead over upstream shrinks
with depth, is about even at 80k and reaches -10 % at 120k.  The cause is not r126 (his card has no WMMA and
takes the tile kernel, so the r126 row never applied to him) and it is not the FA staging fallback alone
(upstream has no staging; the delivery's own gfx1201 streaming arm leads at every depth).  It is the
**RDNA2 tile path** at depth, which this fleet cannot measure (no gfx1030/gfx1031 device; the only RDNA2
silicon is the gfx1036 iGPU with no room for a model, a KV cache or an FA staging arena).

`TODO.md` **#106** now tracks it as an open external-validation item.  The candidates, all block 05, are
block 05's RDNA2 tile table from PR #165/r76 (tuned on a V620 with a different GQA shape), the
`GGML_CUDA_FATTN_TILE_Q_GLOBAL` wide-tile Q-from-global read (the +30 % figure was a shallow measurement),
and the `prefill_stages = !RDNA3_5` staging gate (RDNA2 stages; it was never depth-swept).  The reporter's
own `-p 512 -n 0` sweep on r138, with `GGML_CUDA_FA_STAGE_MAX_MB=1` (force the native K/V read) as the
staging arm, is the measurement that would separate them.

The r138 validation on gfx1201 (2026-10-25) is in
[`benchmarks/2026-10-25-r138-depth-sweep.md`](../../../benchmarks/2026-10-25-r138-depth-sweep.md): resident
+18..+27 % at every depth; streaming +18..+45 % with default flags; and with `-lm none -lzm off` pinned on
both arms, parity at d0 and +20 % from 32k on.  There is no crossover on r138 on the architecture we can
measure.
