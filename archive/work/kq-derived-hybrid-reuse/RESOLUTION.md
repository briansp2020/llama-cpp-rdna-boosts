# kq-derived-hybrid-reuse — assessment and resolution

**Status: promoted into the delivery as r164 (block 05).**  Release `v15-6050b467c-r164`, tip
`6a8b807b39712c37292df9ca994c67cb6cc44e08`, tree `59f6b98ebd5f7585eb33ca38a09c81e3abb8c0a7`.  Source: PR #242
by **overdoingism** (`wip/kq-derived-hybrid-reuse/`, one `git am` file, `src/llama-graph.cpp` only).  The
contributor's own record is kept verbatim in `README.md` and the patch as submitted in
`kq-derived-hybrid-reuse.patch`; this file is the maintainer-side assessment, what shipped and what the gates
measured.

## What it claims

1. `llm_graph_input_kq_derived::can_reuse()` rejected the reuse whenever the graph's memory context was not a
   `llama_kv_cache_context`, on the grounds that the derived form only exists for prefill-shaped batches.
2. On a hybrid model (`llama_memory_hybrid_context`, every Qwen3.x) the derived form is *also* taken by a
   single-sequence decode batch of more than 8 tokens - a long n-gram draft's verification - so each such step
   rebuilt its graph.
3. Looking through `get_attn()` (the cache `build_inp_mem_hybrid()` built the input from) and then running the
   existing checks restores the reuse; other wrappers (iSWA) still reject.  No arithmetic changes.
4. Measured +4.0 % / +3.6 % (Linux, 2 slots) and +2.7 % / +3.5 % (Windows, 1 slot) decode on Qwen3.8-27B with
   n-gram drafting, with identical text in every arm they ran.

## Verdict

**Accepted.**  The mechanism is exactly as described, the patch applies unchanged on r163 (not only on r156/
r159 as the README says), the change is default-on with a kill switch (the repo's default-on policy), and no
arithmetic moves: greedy text and draft acceptance are byte-identical in every arm we built.

Two things were added or corrected on the way in, neither of them a defect in the submission:

* **It understates its own finding.**  The regression is not only about n-gram drafts.  An **MTP verify step at
  `--spec-draft-n-max >= 8`** is a 9-16 token single-sequence batch too, and there the cost is far larger than
  4 %: with a fixed depth 15 the shipped r163 reused **0** graphs over 300 decode tokens, i.e. it was 7.6 %
  slower than its own `LLAMA_KQ_MASK_DERIVED=0` at 3 GPU `-sm tensor` and 5.4 % slower at 2 GPU.  That is the
  release's headline number, not the n-gram one.
* **iSWA must keep rejecting for a reason the README only gestures at.**  `build_inp_mem_hybrid_iswa()`
  registers **two** derived inputs, one bound to the base cache and one to the SWA cache, so there is no single
  cache to look through and a naive look-through would validate one input against the other's state.  Fixing
  that needs the input to record which cache it belongs to; no model we maintain is hybrid+SWA (`qwen35`
  reports `n_swa = 0`), so it is a doc note.

## Assessment

**1. Why the look-through is exactly right and not merely plausible.**  `build_inp_mem_hybrid()` (and
`build_inp_mem_hybrid_k()`) call `build_attn_inp_kv_impl(mctx_cur->get_attn())`, and the derived input is
constructed with that same `llama_kv_cache_context` - so `get_attn()` at reuse time is the cache the tensors in
this input were built from, and `set_input()` already ran against it before the patch (the member `mctx` is
the attn cache all along; only `params.mctx` is the wrapper).  The reuse decision therefore only ever needed
the wrapper peeled, and every existing guard then applies unchanged: `cell_pos->ne[0] == mctx->get_n_kv()`,
`n_stream == 1`, the token-count match, and `cparams.kq_mask_derived && mctx->kq_mask_derivable(ubatch)`.

**2. Why the derived form is reachable in decode at all (the part that made this a regression rather than dead
code).**  The creation gate is `n_kv % LLM_KQ_MASK_DERIVED_KV_STRIDE == 0` (256) plus `ubatch.n_tokens > 8`,
`n_seqs_unq == 1`, no alibi and degenerate positions.  `llama_kv_cache::get_n_kv()` pads `n_kv` to
`max(cparams.n_pad, LLAMA_KV_N_PAD_MIN = 256)`, so the stride condition holds for the whole pad window and a
9-16 token verify batch takes the derived form on every step of it.  Neither condition is exotic: a draft that
is 8 tokens deep is enough, which is the shipped `--spec-draft-n-max 7`+1 boundary on one side and the
`--spec-draft-n-max 15` cap on the other.

**3. Text purity.**  Reuse vs rebuild changes no kernel, no width and no reduction order - the same graph
objects are re-filled by `set_input()` - so the expected result is byte-identical text, and that is what
measured (four arms: r163/r164 x derived/packed, `ROCBLAS_USE_HIPBLASLT=0`, `1866 chars sha=b1f1903bc51c`;
draft acceptance identical to five decimals on 300-3000-token arms).

**4. Where it does not help, and why that is still fine.**  3 GPU prefill restores reuse (15 -> 22 on a
26 286-token prompt at `-ub 2048`) but moves no time (14 255.76 vs 14 276.62 ms) - a prefill graph rebuild is
cheap next to a 2048-token ubatch.  Flash-Next (qwen4exp, whose `llama_memory_hybrid_idx_context` *does* derive
from the hybrid context) takes the new path but shows the same reuse count in both builds, because another
input already limits its reuse; its text is identical.  So the change is a targeted fix, not a general one.

## What shipped

Block 05, folded into `llm_graph_input_kq_derived::can_reuse()` with `LLAMA_KQ_MASK_DERIVED_REUSE=0` as the A/B
arm and a fold paragraph in the block commit message crediting the contributor.  The block-05 note in
`patches/README.md` also lost its claim that the derived form is "only for prefill-shaped batches
(`n_tokens > 8`), so decode/verify are untouched" - that wording is what hid this class of regression.

Measured on the release tree (gfx1201, ROCm 10.1.0):

| arm | graphs reused (derived -> packed) | result |
|---|---|---|
| n-gram `n_max 15`, `prompts/recall.txt`, 1 GPU | 58 -> **77** (= packed 77) | 127.04 -> 127.82 t/s (+0.6 %) |
| MTP fixed depth 15, 3 GPU `-sm tensor` | **0 -> 97** (= packed 97) | 6820.18 -> 6337.97 ms / 300 (**+7.6 %**) |
| MTP fixed depth 15, 2 GPU `(0,2)` `-sm tensor` | **0 -> 76** (= packed 76) | 5328.70 -> 5041.15 ms / 300 (**+5.7 %**) |
| `draft-mtp-adaptive,ngram-mod` defaults, 3 GPU | 16 -> 19 (= packed 19) | 1331.68 -> 1308.44 ms / 400 (+1.7 %) |
| 3 GPU prefill, 26 286 tokens, `-ub 2048` | 15 -> 22 (= packed 22) | 14.28 s either way |

The 3900X figures in the contributor's README (+2.7 to +4.0 %) are consistent with a per-rebuild cost of
~2.7 ms; our 9950X3D pays ~1.0 ms, which is why the same n-gram arm moves ~0.6 % here.

Gates: per-block build (block 05 + tip), coherence, prefill-logit **KLD 0.000000 / 100.000 %**, MTP Protocol A
at 3 GPU on both split modes (tensor 90.2 vs plain 48.3 t/s, acceptance 0.604; layer 51.8 vs plain 25.0,
acceptance 0.621 - every figure within noise of the recorded reference), `gate-park-idle-slots.sh`,
`gate-ckpt-reuse.sh`, Flash-Next text identity, `validate-set.sh`.  `WORKLOG.md` 261028.006.
