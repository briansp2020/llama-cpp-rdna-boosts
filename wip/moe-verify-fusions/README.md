# `moe-verify-fusions` — the MoE verify step: cache-aware fusion selection (#50) and 3-GPU scaling (#51)

**Status: OPEN (opened 2026-10-10).**  Not part of the delivery; nothing here may be applied to the fork
without the maintainer's go-ahead (see the WIP and promotion rules in `AGENTS.md`).

> **NEXT SESSION: read [`HANDOVER-single-seq-rollback.md`](HANDOVER-single-seq-rollback.md) end to end.**
> It narrows the single-sequence plain-vs-MTP divergence to the **qwen4exp recurrent/conv rewind with a
> live expert cache**, refutes the old verify-width-kernel hypothesis, records the full negative matrix,
> and gives the code map + the teacher-forced-replay instrument to use next.  The sibling **multi-sequence
> non-determinism** is `TODO.md` #52.
>
> **2026-10-12 session: the slab-resident narrow-2 layout is IMPLEMENTED and functionally validated** —
> no alias, and the exact token streams agree with the quick fix once the cache/fusion confound is
> controlled.  The numeric acceptance gate (handover §4.2) turned out to be confounded by the **open
> #50** (arena-layout-dependent cache-aware fusion selection): the quick fix itself measures 0.738 in one
> run and 0.763 in another, and pinning the cache budget flips the ordering (narrow-2 0.752 vs quick-fix
> 0.685).  Full evidence, numbers and open questions: [`RESULTS-slab-narrow2.md`](RESULTS-slab-narrow2.md).
> The layout is in the `~/llama.cpp` working tree; `patches/` and `release.json` are untouched.
>
> **2026-10-12 (second session) result: the single-sequence "rewind restores wrong state" lead is
> REFUTED.**  A single-context rewind probe (`tools/rrewind.cpp`) is **PURE** (0 logit / 0 serialized-state
> mismatches) on the exact divergent config (2 GPU `-sm tensor` + qwen4exp + partial cache), and the
> controlled matrix localises the plain-vs-MTP divergence to **cache-residency/graph-planning**
> (qwen4exp + 2 devices + `-sm tensor` + a partial cache): it disappears on 1 GPU, 3-GPU full residency,
> `-sm layer`, and cache-off, and the 35B-A3B control is pure even at 2-GPU tensor partial.  The plain
> arm is itself cache-sensitive, so the divergence is `TODO.md` **#50**, not the recurrent rewind.  See
> [`FINDINGS-single-seq-rollback.md`](FINDINGS-single-seq-rollback.md) and the amended
> [`HANDOVER-single-seq-rollback.md`](HANDOVER-single-seq-rollback.md).  `TODO.md` #52 (multi-sequence)
> remains a separate recurrent-rollback-boundary bug.
>
> **2026-10-11 session result:** the cache-band routed-MMVQ path's problem was located: `ggml_cuda_slab_work_alloc`
> hands a single NARROW base to **every** narrow compute view, and with MTP there are **two** live compute
> buffers (the target's verify view *and* the draft context's), so they aliased at slab base 0.  A quick fix
> (allocate the draft's COMPUTE buffers with `cudaMalloc` instead — upstream-shaped, via a new optional
> `slab_compute_enable` hook keyed off `cparams.ctx_other`) is implemented and measured on GSQ 2-GPU:
> **MTP acceptance 0.693 -> 0.738, decode 43.1 -> 53.5 t/s (~16 % lower verify step)**.  The maintainer wants
> the durable form instead — a second static narrow region **inside the slab**, end-pinned
> `| narrow-1 | ring | wide | arena (top-down) | narrow-2 |` — so the allocator stays all-in-the-slab (no
> fragmentation).  That implementation + validation is the handover:
> [`HANDOVER-slab-narrow2.md`](HANDOVER-slab-narrow2.md) (read it first; it is self-contained).  The quick fix
> and the session diagnostics are saved as `session-quickfix-and-diagnostics.diff` beside this file.
>
> **The plain-vs-MTP verify-width divergence is a SECOND, independent defect** (it survives the aliasing fix;
> it is not CPU, not the partial cache, not fusions, not `n_rs_seq`, not the MMVQ bands, not GDN/FA).  It is
> the follow-on item *after* the slab-resident layout is in and validated — see the handover §5.  Recommend
> classifying GSQ against the IQ4_NL purity model (`b00fdf534227`) before any kernel work.

## Directives (maintainer, 2026-10-10 — binding for this campaign)

### D1 — Purity and the fused paths come first; never disable a fusion as the *fix*

The objective is to **keep the fused paths intact and enabled** and to restore determinism — not to trade
the fusions away.  A drift that depends on some condition must be:

1. **Bisected to the exact condition first.**  The deliverable is a *condition*, stated precisely, plus the
   minimal switch or metric that identifies it — not "the fusions are bad".  Example shape: "the fused
   gate+up+GLU decode path drifts when a `MUL_MAT_ID` table is partially evicted *and* the batch width is
   in the cache band".
2. **Repaired for that condition if at all possible.**  E.g. make `ggml_cuda_cache_blocks_fusion`'s gate
   per-table (the code's own `OPEN 2`) so a partial cache keeps serving, or make the two paths agree
   numerically.  Repair is the preferred outcome, full stop.
3. **If it cannot be repaired yet, scoped to exactly that condition** — stood down only **while the
   condition is true**, and enabled everywhere else.  A per-run `GGML_CUDA_DISABLE_FUSION=1`, or a blanket
   `GGML_CUDA_DISABLE_*` for a whole run, is **not** an acceptable fix.
4. **Blast radius proven, not asserted.**  Ship a matrix (width × prompt × model, and the placements/configs
   that trigger the condition) showing the fusions are ON everywhere the condition does not hold, and that
   the disabled scope cannot be reached in the normal path.

Cost reference for why this matters: a *global* fusion disable is **~9-10 % prefill / ~6 % decode** and
lands *below* the unpatched stock 922 (`ENVIRONMENT.md` §7).  The whole point of D1 is that the fix must be
scoped to the condition, not to the run.

### D2 — A/B on the GSQ-IQ3_XXS model, which is checked and warm-cacheable

Use this set as the **model of choice** for A/B iteration:

```
/llm/models/Qwen3.8/Flash-Next/GSQ-IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf
```

Rationale (verified 2026-10-10): the box has **184 GiB RAM / ~104 GiB page cache**, the GSQ set is
**70.6 GiB (2 shards)** and stays resident between runs, while the IQ4_NL set is **96 GiB** and evicts — so
a run loads roughly an order of magnitude faster.  This campaign is A/B-heavy, so iteration speed is the
point; keep IQ4_NL for the periodic cross-check and for anything already recorded against it.

**Verified 2026-10-10: the GSQ-IQ3_XXS model has NO embedded MTP head.**  It carries `blk.0..47` only —
there is no `blk.48` and no `nextn.*` tensor (the separate MTP head *is* a `blk.48.*` layer file).  It must
therefore be paired with the shared draft, exactly as the field harness already does for the IQ4_NL target:

```
--spec-draft-model /llm/models/Qwen3.8/Flash-Next/IQ4_NL/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf
```

Confirm on the first run that the draft is accepted (the head is Q8_0, the target IQ3_XXS; both are
48-layer Flash-Next configs).  A mismatched draft fails loudly at load, so this is a one-run check rather
than an assumption — and if it *is* rejected, say so and fall back to the IQ4_NL target rather than
silently changing the experiment.

This campaign tracks **two open findings together**, at the maintainer's request, because they are
suspected of sharing a root cause:

* **`TODO.md` #50 — the cache-aware fusion guard is global while the slab's eviction is per-table.**
  `ggml_cuda_cache_blocks_fusion` is gated on `moe_cache_has_arena()`, a **global** predicate, but the
  movable-boundary slab evicts cache tables **per table** ("a PARTIAL cache is the normal state").  So any
  arena change flips the cache-aware fusions for the whole model mid-run, and because the fused and
  non-fused paths are not bit-identical, the greedy output depends on the *timing* of the eviction.
* **`TODO.md` #51 — 3-GPU MTP decode is 2.6x SLOWER than plain decode.**  On 3 GPUs the MoE verify step
  costs ~190 ms against ~57 ms on 2 GPUs, while plain (1-token) decode is fine on both.

Both live on the **same code**: the MoE `MUL_MAT_ID` served at a small batch width (the cache band), where
the expert-cache take-over redirects the weight to the arena and the routed-expert **MMVQ** kernel reads it
through the slot remap.  #50 is about *which kernel* that path selects, #51 is about *how that path scales
to 3 devices*.  If the cache-aware fused verify path is both the non-bit-identical one (#50) and the
mis-scaling one (#51), a single fix closes both.

## The two symptoms, as measured

### #50 — the fused/non-fused selection flips with the eviction state

Field model (Qwen3.8-Flash-Next IQ4_NL + the shared Q8_0 MTP head), `-sm tensor -ncmoe 48`, 32 358-token
prompt, greedy, `--spec-type none`, `GGML_CUDA_ALLREDUCE=ce` (deterministic per config; the default AR is
not):

| config | ring-in-slab | ring-outside |
|---|---|---|
| default (all fusions on) | `cdd73eb8e3ea` / `233382632053` (per region size) | `ef91112ab636` |
| **every individual fusion switch off, `GGML_CUDA_DISABLE_FUSION` unset** | **`7b53871fb520`** | **same** |
| `GGML_CUDA_DISABLE_FUSION=1` | `a702e03f374e` | same |
| `GGML_CUDA_DISABLE_SHEXP_DOWN_GATE=1` | `233382632053` | `ef91112ab636` |
| cache off (`MOE_EXPERT_CACHE_MIB=0`) | `ef91112ab636` | `ef91112ab636` |

What that matrix establishes (full analysis: `archive/work/slab-ring-region/FINDINGS-numerics.md` F3):

* It is the **fusion path**, not corruption: 0 NaN, 0 `////`, coherent text, and the **prefill logits are
  bit-identical** between the two placements (mean KLD `0.000000`, same-top-p 100 % at `-ub 512`, `-ub 2048`
  and `-ub 6144`).  The generated stream is identical for the first 528 chars; the divergence is a
  **near-tie at the first novel decode token**.
* It is **not a single bad kernel**: disabling either the qwen4exp HC group *or* the idx/concat/DSV4 group
  alone removes it (at *different* agreed hashes), so it is a cumulative rounding shift.
* It is **not the alloc dependencies**: `GGML_CUDA_DISABLE_FUSION=1` is triply confounded — it also skips
  the `graph_optimize` alloc-deps pass and its shared-expert `std::rotate` (PR #27301).  With the deps pass
  explicitly left **on** and the individual fusions off, the placements already agree, which excludes them.
  (The deps do cost **+512 MiB** of compute buffer, 2816 vs 2304 MiB, charged to the arena like any work.)
* The **trigger** is the slab's eviction volume: ring-in-slab evicts **8514 + 9108 MiB** and re-arms 107
  tables, ring-outside 2442 + 2970 MiB and 32.  A global guard + a per-table eviction is the mismatch.

### #51 — 3-GPU MTP is a loss, 3-GPU plain is fine

| config | prefill t/s | decode t/s | acceptance |
|---|---:|---:|---:|
| 2-GPU plain (`--spec-type none`) | 546 | 41.9 | — |
| 2-GPU MTP | 961 | **64.8** | 0.91 |
| 3-GPU plain | — | **52.2** | — |
| 3-GPU MTP | 1043 | **19.5** | 0.90 |
| 3-GPU MTP, ring off (`GGML_CUDA_SLAB_RING_MIB=0`) | 759 | 21.4 | 0.91 |
| **3-GPU MTP, clean r38 (no patch)** | 768 | **20.1** | — |
| **3-GPU plain, clean r38** | 927 | **53.0** | — |

Established: the verify step is ~190 ms on 3 GPUs vs ~57 ms on 2; it **reproduces on clean r38**, so it is
not the r39 H2D-ring work; `GGML_CUDA_ALLREDUCE=ce` vs the hybrid default makes no difference (and `ce` is
2-GPU-only, `ENVIRONMENT.md` §6); it is not cache residency (3-GPU logs 99.1 %).  It hits the production
config (servers run 3-GPU, `AGENTS.md`).

## The shared-ground hypothesis (to be confirmed or refuted)

The verify batch is 4 tokens (`--spec-draft-n-max 3`) — inside the cache band (`MOE_EXPERT_CACHE_MAX_TOK`,
16 on RDNA4).  Per `GREEDY-PURITY.md` §39 the cache serves such a batch **only through the routed-expert
MMVQ kernel**, reading the arena via the slot remap.  Under `-sm tensor` a layer's tables are **split
across devices** (`GGML_META_SPLIT_COPY`), so that path is a per-device gather + a remap that spans devices.

So the candidate shared cause is: **the cache-aware fused verify path does not behave the same on a
multi-device split table** — it is the path whose selection is guarded globally (#50) and the path that
mis-scales at 3 devices (#51).

Ordered tests, cheapest first — each is designed to *split* the hypothesis, not to confirm it:

1. **Does #51 survive `MOE_EXPERT_CACHE_MIB=0`?**  That removes the take-over and uses the CPU expert path.
   If the 3-GPU verify recovers, the regression is in the cache-aware path (shared with #50); if not, it is
   in the plain split verify (allreduce / gather) and the two are independent.
2. **Does #51 survive `GGML_CUDA_DISABLE_FUSION=1`?**  (Also removes the non-cache fusions and the deps —
   note the confound.)  A recovery points straight at the fused verify kernel.
3. **Does #51 survive `GGML_CUDA_DISABLE_SHEXP_DOWN_GATE=1`** and the other individual switches?  This is
   the knob set that implicated the fusions in #50; if one of them fixes #51, the same kernel is in both.
4. **`-sm layer` on 3 GPUs**: no split tables.  If MTP recovers under `-sm layer`, the split layout is the
   differentiator and #51 is a split-table problem; if not, it is the 3-device allreduce in the verify.
5. **Width sweep**: MTP `n-max {1,3,7}` and `--spec-draft-n-max 0` (plain) on 3 GPUs.  #51 shows a cliff
   (20 vs 52), so the *width* at which it appears says whether it is the 1-token path or the multi-row
   routed path (`mmvq-verify-rows`, the neighbouring WIP campaign, is about the multi-row kernel).
6. **`GGML_MOE_CACHE_MAX_TOK=8`** (restore the pre-r31 band) and the band-16 path: is the cliff at the band
   edge?
7. **Where the time goes**: `GGML_CUDA_OP_TIMING=1` on the 3-GPU verify, plus the graph-reuse counters
   (`graphs reused` was 267 on 3 GPUs vs 246/270 on 2 — a recapture loop would explain a per-step cost).

**Deliverable shape (per D1).**  Success is *not* "disable X".  It is: (1) the **condition** named
precisely; (2) the **minimal switch or metric** that identifies it at runtime; (3) either a **repair** so
the fused path is correct under that condition, or a stand-down **scoped to that condition only**; and
(4) a **blast-radius matrix** (verify width × prompt × model × slab placement) showing the fusions on
everywhere the condition does not hold.  A change that turns a fusion off for a whole run is a failed
outcome even if the output matches.

**Then** the #50 fix: make `ggml_cuda_cache_blocks_fusion`'s gate **per table** (as the code's own
`OPEN 2` marker asks: *"the wholesale-fallback invariant … must hold for EVERY consumer, not just the
fusion guard and the take-over hook"*), so a partial cache keeps serving; or make the fused and non-fused
paths bit-identical.  Either must be gated on the batch-width matrix and the prefill-logit gate.

## Reproducers

Field: the harness is committed under [`tools/`](tools/README.md) (so it survives `/tmp` being cleared) —
`field.sh` (the r38 §5.5 arm, forces the deterministic `GGML_CUDA_ALLREDUCE=ce`), `field3.sh` (same, no
forced AR = the 3-GPU production default) and `field_nospec.sh` (plain, no MTP = `#51`'s control).  All three
take `M` / `D` / `PROMPT` / `GPUS` from the environment:

```bash
cd wip/moe-verify-fusions/tools
export M=/llm/models/Qwen3.8/Flash-Next/GSQ-IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf
export D=/llm/models/Qwen3.8/Flash-Next/IQ4_NL/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf
export PROMPT=/tmp/srr/mixed30k.txt
# (re)create the prompt: cat README.md ENVIRONMENT.md CONTAINERS.md archive/docs/baseline-history.md > $PROMPT
PROMPT=$PROMPT GPUS=0,1,2 ./field3.sh      ~/llama.cpp/build-rocm-hybrid t3  run30kfit   # MTP
PROMPT=$PROMPT GPUS=0,1,2 ./field_nospec.sh ~/llama.cpp/build-rocm-hybrid t3n run30kfit  # plain control
PROMPT=$PROMPT GPUS=0,1   ./field3.sh      ~/llama.cpp/build-rocm-hybrid t2  run30kfit   # 2-GPU reference
```

**Warm up before recording any number.**  The GSQ set should be warm from the previous run; the *first* run
after a config change still pays the lazy PLE / host-expert disk read (measured: 563 vs 968 t/s prefill on
the same config), so discard it.

`/tmp/srr/mixed30k.txt` is `README.md + ENVIRONMENT.md + CONTAINERS.md + archive/docs/baseline-history.md`
(99 174 B, sha256 `69624f4d207f40bdf7357e74ec978af5bfb4725a0968faf4eea4aab46c2ae241`, ~32 k tokens).  It is
deliberately **non-repetitive** — the older `prompts/prose-rdna-boosts.txt ×6` field prompt is degenerate
(the model copies it, so acceptance is 1.0 and the prefill is inflated; see
`archive/work/slab-ring-region/FINDINGS-numerics.md` F1/F2).  Always **warm up** a config before recording a
number: the first pass pays the lazy PLE / host-expert disk read.

`#50`'s A/B matrix, per `archive/work/slab-ring-region/FINDINGS-numerics.md` F3:

```bash
M=/llm/models/Qwen3.8/Flash-Next/IQ4_NL/Qwen3.8-Flash-Next-IQ4_NL-PROJFIX-00001-of-00009.gguf
CLI=~/llama.cpp/build-rocm-hybrid/bin/llama-cli
B="HIP_VISIBLE_DEVICES=0,1 LD_LIBRARY_PATH=/opt/rocm-7.14-gfx120X/lib"
run() { env $1 $B $CLI -m $M -sm tensor -ncmoe 48 -ctk q8_0 -ctv q8_0 -fa on -t 8 -c 40960 \
  -b 6144 -ub 6144 -f /tmp/srr/mixed30k.txt -n 16 --seed 42 --temp 0 --no-display-prompt --single-turn; }
GGML_CUDA_ALLREDUCE=ce run                                    # cdd73eb8e3ea (ring in slab)
GGML_CUDA_ALLREDUCE=ce GGML_CUDA_SLAB_RING_MIB=0 run          # ef91112ab636 (ring outside)
GGML_CUDA_ALLREDUCE=ce GGML_CUDA_DISABLE_FUSION=1 run         # a702e03f374e both ways
```

## Gates and invariants to keep in the loop

* **Prefill-logit KLD** (`scripts/gate-prefill-logits.sh`, limit 0.005 / 98 %) — the instrument that sees a
  uniform prefill shift; note it is **blind to a decode-only change**, which is what both items are.
* **Width purity** `none == n1 == n3 == n7` on 2-GPU Flash-Next `-ncmoe 48` (`b00fdf534227` at r39) — the
  natural first screen for anything touching the verify width.
* **`test-backend-ops -o MUL_MAT_ID`** (931/931) and the qwen4exp quant-coherence gate
  (`scripts/gate-qwen4exp-quant-coherence.sh`) for the gather/slot-remap side.
* **Dense golden** `1c5d32ac537d` — must not move.
* `GGML_CUDA_ALLREDUCE=ce` is the only **deterministic** AR on 2 GPUs; the default is not, so A/B text
  comparisons must either pin `ce` or be repeated.  `ce` is 2-GPU-only, so on 3 GPUs the deterministic
  reference does not exist yet — prefer a *magnitude* metric (verify ms/step) over a text hash there.
* Do not "fix" #50 by turning the fusions off: measured cost is **~9-10 % prefill / ~6 % decode** and the
  fusion-off prefill (845-890) is *below* the unpatched stock 922.  Fusions stay default-on, and per **D1**
  any stand-down must be scoped to the exact condition, with the blast radius shown — the campaign's
  success criterion is *purity restored with the fusions intact*, not a disabled feature.

## Pointers

* `TODO.md` #50, #51 — the tracker items (this campaign is their working home; close them here first, then
  move their records to `WORKLOG.md` and archive this directory per the WIP rule).
* `archive/work/slab-ring-region/FINDINGS-numerics.md` — F1/F2 (the degenerate prompt and the cold-run
  caveat, i.e. how *not* to measure) and F3 (#50's full evidence and matrix).
* `archive/work/slab-ring-region/HANDOVER.md` §4.3.1 — the 3-GPU table and the clean-r38 control.
* `GREEDY-PURITY.md` §39 — the cache band and the routed-expert MMVQ take-over ("the band-16 cache path is
  bit-identical to all-VRAM at the widths it newly serves" — the claim #50's evidence qualifies).
* `ENVIRONMENT.md` §7 — the fusion switches, the measured cost of a full disable and the re-enable recipe;
  §1.3 for `GGML_CUDA_SLAB_*` / `GGML_CUDA_OPTIONAL_ALLOC_MAX_FREE_PCT`; §6 for the AR choices.
* `wip/mmvq-verify-rows/` — the neighbouring campaign on the multi-row routed MMVQ kernel (bit-exact,
  2-8-token verify).  A #51 fix that changes the verify kernel must not regress it, and it may *be* its
  re-basis.
