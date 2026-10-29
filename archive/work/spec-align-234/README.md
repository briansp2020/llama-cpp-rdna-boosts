# spec-align-234 -- multi-slot verify-batch alignment (issue #234)

Issue: <https://github.com/stew675/llama-cpp-rdna-boosts/issues/234> ("Speculative decoding with
--parallel >= 2 on hybrid models: verify batches whose drafts differ in length are split into ubatches
(n-gram-first chain, draft-mtp-adaptive)"), reported by **@overdoingism**.

**Status: promoted in `v15-6050b467c-r153`** (blocks 02 + 09, `LLAMA_SPEC_ALIGN`) and corrected in `r154` (the
generic cut removed).  The one residual -- a step whose slot cannot draft at all, `TODO.md` #111 -- was
investigated on 2026-10-28 and **declined**: its alternative measures neutral and there is no correctness at
stake, so the delivery does not change for it; the gate now counts and bounds those splits instead.  See
`RESOLUTION.md` for the analysis, the measurements, the declined directions and the reproduction commands, and
`benchmarks/2026-10-28-spec-align-111.md` for the #111 record.

The reporter's own in-tree patch (`0015-6055-spec-align-v1`) is not archived here: it sits on two drafters of
their own that this tree does not contain, so it could not be applied and the fold is a port of the idea.  Its
text is in the issue thread.

Delivery-side artefacts:
* `common/speculative.cpp` -- `spec_align_enabled()`, `align_draft_caps()` (block 02), the band bound read from
  `llama_model_moe_cache_band()` (block 09).
* `scripts/gate-spec-align.sh` -- the gate (2 GPUs, 3 GPUs, host experts, `n_seq == 1`, GPU faults), whose
  `splits.awk` reports draft-length and single-row split steps separately: the draft-length kind must be zero
  with the knob on, the single-row kind (this item) is counted and bounded by `TAIL_MAX` (default 2).
* `GREEDY-PURITY.md` §63, `ENVIRONMENT.md` §3, `patches/README.md` (blocks 02 and 09), `MANIFESTS.md`
  (gate 9 + the per-block rows), `AGENTS.md` (critical fact), `benchmarks/2026-10-28-spec-align-111.md`.
