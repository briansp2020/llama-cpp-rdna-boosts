# RESOLUTION -- multi-slot verify-batch alignment (issue #234) promoted into blocks 02/09, r153

**Status:** promoted 2026-10-27 in `v15-6050b467c-r153` (blocks 02 + 09) and **corrected the same day in
`v15-6050b467c-r154`** (block 02), which removed the generic cut after it measured -6 % on the reporter's other
chain; see "The r153 cut, and why r154 removed it" below; the one residual split step (`TODO.md` #111) was
**investigated and declined** on 2026-10-28 -- see the closing section.  `release.json` carries the r154 tip and
tree.  Reported by **@overdoingism** in
[issue #234](https://github.com/stew675/llama-cpp-rdna-boosts/issues/234) with a measurement, an analysis and a
patch (`0015-6055-spec-align-v1`) cut on two drafters of their own.  The release record is `WORKLOG.md`
261016.001; the rule is `GREEDY-PURITY.md` §63.

## The defect

`llama_memory_hybrid::init_batch()` splits a decode batch with
`llama_batch_allocr::split_equal(n_ubatch, !unified, n_rs_seq + 1)` (the same call in
`llama-memory-hybrid-idx.cpp` and `-iswa.cpp`).  `split_equal()` takes **one token per sequence set per round
and stops when the shortest set runs out**, so on a two-slot server a step in which the slots verify different
draft lengths becomes **two ubatches**: the shorter slot's rows in the first, the longer slot's tail in a
second.  The target model runs a full extra pass for that step.

The reporter's diagnosis was right and the mechanism is upstream, not a delivery defect: a fixed
`--spec-draft-n-max` gives every slot the same length for free, so nothing in stock exposes it.  What makes it
the normal case here is block 02: `draft-mtp-adaptive` holds a **per-sequence** depth
(`common/speculative.cpp`, `n_cap[seq_id] = adaptive ? adaptive_ctrl[seq_id].n_cur : params.n_max`), so two
concurrently verifying slots routinely sit a step apart.

Reproduced on the r152 tree (`b9418b679`), 27B UD-Q4_K_XL, 2 GPU `-sm tensor`, `--parallel 2 --kv-unified
-c 32768 -fa on`, greedy, `n_predict 400`, per-slot `predicted_per_second` and pair aggregate:

| arm | per-slot t/s (2 GPU) | pair agg | per-slot t/s (3 GPU) | pair agg |
|---|---|---:|---|---:|
| each prompt alone | 143.5 / 105.7 | -- | 123.6 / 93.6 | -- |
| adaptive n5, both slots (depths diverge) | 73.5 / 57.5 | **108.1** | 55.3 / 45.5 | **86.9** |
| fixed n3, both slots (equal by construction) | 98.5 / 78.5 | 145.6 | 92.4 / 71.1 | 132.2 |
| adaptive n5, two identical prompts (equal) | 116.0 / 116.0 | 184.2 | 118.8 / 118.8 | 183.4 |

Rows 2 and 3 are the same two prompts: **+35 % at 2 GPUs, +52 % at 3 GPUs**, purely from whether the draft
lengths match.  The direct evidence is the recurrent memory's own trace (`LLAMA_RS_ROLLBACK_DEBUG=1`): a decode
step with depths 3 and 4 prints `RSPREP prep=1/2 n_tokens=4 ... seq_ids=[0]` followed by
`RSPREP prep=2/2 n_tokens=5 ... seq_ids=[1]` -- one decode, two prepared ubatches.

## What shipped

`LLAMA_SPEC_ALIGN` (default **on**, `0` disables), two pieces, both inert with a single drafting sequence:

1. **Block 02, `align_draft_caps()`**: `draft-mtp-adaptive` gives every sequence it drafts in one call the same
   cap: the deepest cap among them, bounded by the smallest **hard** cap in the call (`--spec-draft-n-max`, the
   MTP layer cap and the caller's per-slot `n_draft_max`, which the server lowers as a response runs out of
   tokens) and by the expert cache's band.  No sequence is ever asked for more than it would draft when it runs
   alone, and the alignment never touches another implementation's draft (see the r153 cut below).
2. **Block 09, the band bound**: the aligned depth is `min(deepest cap in the call, band / n_drafting - 1)`
   rows per sequence, read once from `llama_model_moe_cache_band()` after the target context exists.  The band
   is a property of the **whole** verify batch, so a two-slot verify on RDNA4 (band 16) aligns to 7 rows each,
   not 15 each.  A sequence already deeper than that keeps its own cap: the bound may only *widen* a shallow
   draft, never shorten a deep one, which is what keeps a single-sequence run bit-identical.  It lives in
   block 09 because that is where the band API is -- a block may not use a later block's symbol, which
   `scripts/gate-per-block-build.sh` enforces.

## What was declined, and why (so nobody re-proposes it)

* **Padding the shorter sequence to equal length (the reporter's suggestion (c)) is dead for this model
  class.**  A padded row advances the recurrent state in `mem_recr->prepare()`, so it corrupts the sequence
  the padding was meant to help.  Model classes without recurrent memory could use it; the fix must not.
* **Cutting the adaptive depth down to the shallowest slot (the subject of suggestion (b) read the other
  way) is the wrong direction.**  Adaptive depth is worth **+13 %** over a fixed depth when the lengths match
  (184.2 vs 163.0 on the same pair), so the deep slot's proposals are what we want to keep.  The fix therefore
  raises the shallow slot instead.
* **No change to `cparams.n_rs_batch`.**  It stays `common_speculative_n_max() + 1` (the correctness bound
  for a partial rollback and the floor of the chunked-GDN threshold).  An align-up can never exceed the
  per-slot cap, which is itself bounded by that `n_max`, so the bound is unchanged by construction -- and
  verified: the rollback / recurrent-state gates are unaffected.
* **The reporter's in-tree patch does not apply here.**  `0015-6055-spec-align-v1` sits on two drafters of
  their own and this tree has none of them, so this is a port of the *idea* against the current
  `common/speculative.cpp`, together with the two pieces their patch does not have: the per-implementation
  "your draft was cut" feedback and the expert-cache band bound.

## The r153 cut, and why r154 removed it

r153 shipped a second half, the reporter's suggestion (a): a generic "cut the finished drafts to the shortest"
in `common_speculative_draft()`, with `draft_truncated()` telling the implementation so acceptance counted only
verified positions.  The gate only exercised `draft-mtp-adaptive` (where the align-up makes the cut a no-op), so
nothing measured it on the case it was written for.  Measured afterwards on the reporter's chain (a)
(`--spec-type ngram-mod,draft-dflash` + `Qwen3.8-27B-DFlash2-Q4_K_M`, 27B, 2 GPU, recall + prose, `-n 200`,
`--spec-ngram-mod-n-max 15 --spec-ngram-mod-n-min 8`):

| arm | pair per-slot | pair agg | split steps |
|---|---|---:|---:|
| `LLAMA_SPEC_ALIGN=0` (r152 behaviour) | 132.9 / 87.6 | **65.6** | 12 |
| `LLAMA_SPEC_ALIGN=1` (r153, with the cut) | 122.3 / 76.2 | **61.7 (-6 %)** | 1 |
| `LLAMA_SPEC_ALIGN=1` (r154, cut removed) | 126.7 / 82.6 | 64.6 | 12 |

The `min` is the problem: the two drafts are not comparable (a long n-gram match on one slot, a DFlash block on
the other), and the cut fires on every step whose n-gram match is shorter than the block, truncating the OTHER
drafter's good draft.  `RSPREP` shows the verify groups collapsing to 6 rows (both slots at ~2 drafted tokens)
against 6-and-16 in r152, the prose slot's acceptance falling 74 % -> 50 % (231 -> 284 draft tokens for fewer
accepts), and the removed pass costing far less than the lost proposals.

So r154 removed the cut and `draft_truncated()`, and `gate-spec-align.sh` now gates the mixed-drafter chain as
**inert** (within 2 %, identical draft counts) so it cannot come back.  The recorded rule: a length mismatch is
worth equalizing when a **cap** caused it; when the **content** did, the second ubatch is the cheaper arm.

## Measured after the fix

`scripts/gate-spec-align.sh` (2 GPUs = the x8 root-port pair, 3 GPUs = all three, plus a 35B-A3B `-ncmoe 40`
host-expert arm), aggregate t/s of the concurrent pair, split decode steps from the `RSPREP` trace:

| arm | knob off | knob on | split steps off -> on |
|---|---:|---:|---|
| 27B, 2 GPU | 103.8-108.3 | **137.9-140.9** (**+30..35 %**) | 58 -> 0 |
| 27B, 3 GPU | 86.1-87.0 | **133.0-133.7** (**+54..55 %**) | 58 -> 0 |
| 35B-A3B `-ncmoe 40`, 2 GPU | 115.7-116.3 | **147.7-149.7** (**+28..29 %**) | 58 -> 0 |
| `ngram-mod,draft-dflash` + DFlash2, 2 GPU | 99.7 | 99.5 (**-0.2 %, inert**) | 24 -> 24 |

The equal-length arm (two identical prompts) is unchanged: -0.4 / -1.4 / -2.1 / +0.2 % (that arm's spread
between two identical runs is ~12 %, so the gate bounds it at 10 %).  The final gate run also measures a
`--spec-type draft-mtp -n 3` arm -- equal lengths by construction, a different drafter config -- as the
checkpoint the fix has to reach: pair **141.3 (aligned adaptive) vs 143.9 (fixed n3)**, within 2 %, while the
aligned adaptive arm keeps its advantage where acceptance is high (two identical prompts: 171-185 vs 161.2).  Per slot, 2 GPU: 72-74 / 57-60 -> 112 / 76 t/s;
3 GPU: 55 / 45 -> 110 / 72.  The single host-expert split that remains is the last step of a slot that drafted
nothing, which is deliberate.

## Purity: what the oracle is, and what it is not

The fix changes which positions are verified per step, and **changing a slot's verify width moves greedy text
by itself**.  Evidence, in this order:

* **`n_seq == 1` is byte-identical** (knob on vs off, `llama-cli`, same seed): one drafting sequence has
  nothing to align, so the code is inert by construction.  Same for every **single-slot server request** in the
  gate, at both device counts and on the host-expert arm.
* **Cross-process text is stable** with `ROCBLAS_USE_HIPBLASLT=0`: two runs of the same arm agreed character
  for character in every slot and every phase, which is what makes the pair's delta attributable at all
  (`GREEDY-PURITY.md` §44).
* **The two-slot pair's low-acceptance slot diverges at a near-tie** (char 220 of 1202 at 2 GPUs, char 627 of
  1234 at 3).  This is the width, not the change: a **fixed `draft-mtp` pair at depth 5 against depth 4** (both
  slots equal length in both arms, no alignment involved) flips the same slot at the same kind of point
  (char 259, "Ada not on Monday" vs "but Ada not on Monday").  Recorded as `GREEDY-PURITY.md` §63; the gate
  reports the divergence point and does not gate on it.
* **Four-axis MTP (`-n 3000`, 1 GPU, adaptive n3 so the controller is pinned at the floor)**: acceptance
  R 0.5425 / P 0.8091 / C 0.8722 / K 0.9920, mixed prompt 0.6348.  The prose `draft-mtp n3` Protocol A arm is
  **bit-identical to the r152 build**: 1449 of 1791 draft tokens, mean accepted length 3.43, acc/pos
  (0.906, 0.819, 0.702), 68.7 vs 68.9 t/s.

## Reproducing

```bash
# the gate (2 and 3 GPUs + the host-expert arm; ~12 min)
LLAMA_SERVER=~/llama.cpp/build-rocm/bin/llama-server ./scripts/gate-spec-align.sh
# one pair by hand: see the arm loop in the gate, or the harness in the issue's report
# the split itself, before/after
LLAMA_RS_ROLLBACK_DEBUG=1 <server> ... 2>&1 | grep -c 'RSPREP prep=1/[2-9]'
```

## 2026-10-28 -- the residual single-row split (TODO #111): analysed and DECLINED

The one split step per response that survived r154 is the step on which a slot runs out of generation budget:
`get_n_draft_max()` is `n_remaining() - 1`, so that slot never enters the draft call and presents its sampled row
alone while its neighbour presents `n_draft + 1`.  It is a **cap** case, not content, so it *can* be equalized --
and a candidate fix was implemented and measured: decide the step's verify width before any draft is set up, and
if any generating slot in that step cannot draft, let nobody draft (one row per sequence, the drafter resuming on
the next step).

**Declined.**  There is no throughput gain: that second ubatch carries the neighbour's **genuine draft rows**, so
split (`2 passes / ~6 tokens`) and suppressed (`1 pass` now + the next step's `1 pass`, `~7 tokens`) are two passes
either way -- measured **-0.3 %**, below the floor.  There is no correctness gain either: a verify width is a
numerics/config axis, nothing aborts, and the split stays reachable through content-caused mismatches, prompt/decode
mixes, the mixed-drafter chain and the `LLAMA_SPEC_ALIGN=0` arm.  And the fix would widen a **cross-slot coupling**
(one slot's draft suppressed by a neighbour's budget) for a neutral result.  Shipping it would be change for
change's sake.

What did change is the **instrument**: `scripts/gate-spec-align.sh` now reports `full+tail` split counts and bounds
the single-row kind with `TAIL_MAX` (default 2 = one per response boundary), where it previously excluded those
groups from its count entirely -- so a future change that *manufactures* single-row ubatches fails the gate
instead of being invisible.

**Determinism correction (a real fix, unrelated to the candidate).**  The same session found that the two-slot
*aligned* pair is not run-to-run deterministic: four runs of one unchanged binary at 3 GPUs landed in two text
basins (1237 and 1232 characters, 159 vs 174 t/s), and the candidate build landed in the same basins as the
baseline.  Per-config determinism is therefore a **single-slot** statement, and a two-slot text diff is not an
instrument at all.  `GREEDY-PURITY.md` §63 carries the correction.

Details and numbers: `benchmarks/2026-10-28-spec-align-111.md`.
