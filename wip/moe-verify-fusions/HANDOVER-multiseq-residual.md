# HANDOVER - multi-sequence non-determinism, the residual (issue #134 / TODO #52, component C)

**For a cold session. Scope is the residual after `v16-a55e952b8-r42`.** Read this in full, then
`AGENTS.md` ("Report findings before shipping", "Pushing policy", "WIP and promotion rules", "Default-on
policy"), then the campaign `README.md`. Do **not** reopen what is already fixed (below). Report the root
cause and the proposed fix to the maintainer before any `patches/`/`release.json`/docs-header change.

---

## 1. Status when this handover was written (2026-10-12, r42)

`release.json` = `v16-a55e952b8-r42`, base `a55e952b8`, tip `a3c13e9ca76717d8dd9b6020fca2b35bcd2c46b6`,
tree `eafc03875b2c894bda7d8081d362b234a55a9c54`. The fork `rdna-boosts` branch is refreshed. The set is
green in `scripts/validate-set.sh`, coherence `1c5d32ac537d`, prefill-logit KLD 0.000707 / same-top-p
98.755 % PASS, `test-recurrent-state-depth` unchanged (130 baseline failures, phase A 1..7 PASS).

### Fixed and delivered in r42 (block 02) - do not reopen

The GDN dispatch took the **whole-batch chunked** kernel for `n_seqs > 1 && K == 1` **unconditionally**,
while the same sequence decoded alone with `n_tokens <= GDN_CHUNKED_MIN_TOKENS` took the **sequential**
kernel (not bit-identical), so a sequence's logits depended on its co-resident sequences. The
`n_seqs > 1` branch now shares the single-sequence `n_tokens > GDN_CHUNKED_MIN_TOKENS` gate. See
`WORKLOG.md` 2026-10-12 (r42) and the `0002` row in `patches/README.md`.

### Held, NOT shipped: the batching change (call it B)

A `split_equal` change that forces **one sequence per ubatch whenever a sequence contributes more than one
token** (single-token decode keeps its cross-sequence weight reuse). It is **decode-cost-free**
(`llama-batched-bench -npp 16 -ntg 32 -npl 1,4,8` on GSQ: B=4 TG 141 -> 141, B=8 138-152 -> 138-152;
prefill *faster*, B=4 pp 48 -> 165, B=8 122 -> 198) and it makes the probe bit-identical solo vs co-batch
for K=1/4/16/32. **Why held:** it breaks `llama-perplexity` at `n_seq >= 4` (27B Q8_0 and 4B Q8_0: PPL
5.78 / 8.51 -> 3425 / 2097), i.e. the prefill-logit release gate, and the output-index mapping was shown
to be **identical** with and without it (`out_ids` debug), so the PPL break is in the forward, not the
logits indexing. Reproduce:

```bash
# 4B, B OFF (correct) vs B ON (garbage)
env LLAMA_MULTI_SEQ_UBATCH=0 <canon>/build-rocm-hybrid/bin/llama-perplexity \
   -m /llm/models/Qwen3.5/4B/Q8_0/Qwen3.5-4B-Q8_0.gguf \
   -f /llm/models/wikitext-2-raw/wiki.test.raw -c 512 -b 2048 --chunks 4 -ngl 99 -fa on -t 8
# PPL 8.51 (MULTI=0) vs 2097 (MULTI=1); -b 512/1024 (n_seq 1/2) are fine.
```

The WIP knob is `GGML_FORCE_SEQ_UBATCH` (1 = split all, 2 = split only multi-token). The delivery form
would be `LLAMA_MULTI_SEQ_UBATCH`, default on. The saved instrumentation diff is
`session-multiseq-instrumentation.diff`; the probe is `tools/test-recurrent-state-multiseq.cpp`.

---

## 2. What remains (C)

With the r42 fix, **B disabled, cache off, and no cross-sequence ubatching at all**
(`GGML_FORCE_SEQ_UBATCH=1`), the server gate is **still not byte-reproducible** under **unified KV**.
So C is a state/layout dependence, not ubatch co-batching. The gate matrix (GSQ + MTP head, 2x R9700,
canonical r42 build, `tools/multiseq.sh`; `HANDOVER-multiseq-mtp.md` section 7 is the reproducer):

| arm (`tools/multiseq*.sh`) | repetitions equal? | conc == solo? |
|---|---|---|
| plain, auto np=4, unified | no | no |
| plain, `LLAMA_MULTI_SEQ_UBATCH=0` | no | no |
| plain, force=1 (no co-batch), unified | no | no |
| plain, force=2, `-np 2 --no-kv-unified` | yes | no (single-token decode co-batch) |
| plain, force=1, `-np 2 --no-kv-unified` | yes | **yes** (fully green) |
| plain, `-ub 12288` | no | no |
| plain, `MOE_EXPERT_CACHE_MIB=0` | no | no |

`GGML_FORCE_N_RS_SEQ=0` and `--spec-type none` both still diverge, so C is independent of the recurrent
snapshot path and of MTP. `-np 2 --no-kv-unified` is the only thing that stabilizes repetitions, so the
**unified KV cache pool** is central to C.

**Best current hypothesis:** under unified KV the per-query attention result depends on the cache's total
extent / cell layout (`n_kv`, `k_idxs`), which changes with the concurrent sequence's presence and across
repetitions. The FA chooser is `Q->ne[1]`-dependent by construction (`ggml/src/ggml-cuda/fattn.cu`), and
the delivery's dense mmvq/mmq band is only uniform to about M=16 (`GGML_CUDA_DISABLE_MMVQ_DENSE_BAND=1`
makes the probe's K=16 co-batch bit-identical). The minimal probe could **not** reproduce C (a companion
prefill of up to 256 tokens, and a KV-length disparity, are bit-exact), so C needs the server's real KV
scale.

### Open questions for C

1. With r42 + force=1 + unified + cache off, diff **solo vs concurrent** (and two concurrent repetitions)
   with the per-node dump and name the first divergent tensor on the real model.
2. Is it the FA kernel's `n_kv`/tile choice, the KV cache cell allocation/defrag, or the `k_idxs` layout?
3. Fix B's `llama-perplexity` break (the multi-ubatch output path) - then force=2 + non-unified KV is
   green and decode-cost-free, and only the unified-KV piece of C is left.

---

## 3. Tooling + how to rebuild

```bash
# canonical tree at the r42 tip (the probe needs the fork tree)
git clone https://github.com/stew675/llama.cpp && cd llama.cpp
git fetch origin rdna-boosts && git checkout rdna-boosts     # or: base + scripts/apply-all.sh
BUILD_DIR=build-rocm-hybrid EXTRA_CMAKE_FLAGS="-DCMAKE_HIP_FLAGS=" ~/bin/build-llama-rocm-714
export LD_LIBRARY_PATH=/opt/rocm-7.14.1-gfx120X/lib
```

- **Probe:** copy `wip/moe-verify-fusions/tools/test-recurrent-state-multiseq.cpp` to `tests/`, add
  `llama_build(test-recurrent-state-multiseq.cpp)` to `tests/CMakeLists.txt`, build it. Env:
  `MSQ_ARM` (0 = solo ref, 1 = co-batch, 2 = separate call, 3 = co-batch seq0 first), `MSQ_RS` (`n_rs_seq`),
  `MSQ_K` (verify width), `MSQ_FILL`, `MSQ_PRE`, `MSQ_UB` (`n_ubatch`), `MSQ_UNI` (`kv_unified`),
  `MSQ_SAMEFILL`. Any `ARM1/ARM3 != ARM0` digest is the bug.
- **Server gate:** `wip/moe-verify-fusions/tools/multiseq.sh <build_dir> <tag>` (and the `_plain`,
  `_np2*` variants used above). Note `multiseq.sh` never passes `-np`, so the server auto-sets `-np 4` and
  forces `kv_unified=true`; pass `-np 2` explicitly to test non-unified.
- **Aliasing-free per-node dump:** in `session-multiseq-instrumentation.diff`, `LLAMA_DUMP_CUDA=<substr>`
  hashes each named node's columns right after its launch (it syncs the compute stream first; the plain
  `ggml_backend_tensor_get` copies on `cudaStreamPerThread` and aliases). Use with
  `GGML_CUDA_DISABLE_FUSION=1` so every node is computed and named.
- **Env knobs used:** `GGML_CUDA_GDN_CHUNKED`, `GGML_CUDA_DISABLE_MMVQ_DENSE_BAND`,
  `GGML_CUDA_DISABLE_FUSION`, `GGML_CUDA_DISABLE_GRAPHS`, `GGML_FORCE_N_RS_SEQ`, `MOE_EXPERT_CACHE_MIB`,
  `LLAMA_MULTI_SEQ_UBATCH` (B). One arm at a time; never push to upstream.

## 4. Gates to keep green

`scripts/validate-set.sh`, coherence `1c5d32ac537d` (Qwen3.5-4B-Q8_0, seed 42),
`scripts/gate-prefill-logits.sh` (KLD <= 0.005 / same-top-p >= 98 %; it **will** fail if B is enabled -
re-record only for an intentional, understood shift), `test-recurrent-state-depth` (baseline) and
`test-recurrent-state-rollback`, `tools/multiseq.sh`, `MUL_MAT_ID` 931/931, `FLASH_ATTN_QSA` >= 26/26, and a
`llama-batched-bench -npp 16 -ntg 32 -npl 1,4,8` A/B (the rule-5 gate).

## 5. Pointers

- Issue #134 (the report that started this), the (now partly superseded) `HANDOVER-multiseq-mtp.md`,
  `FINDINGS-single-seq-rollback.md` section 7.
- `WORKLOG.md` 2026-10-12 (r42) and (r41); `patches/README.md` `0002`; `TODO.md` #52.
- `archive/work/gdn-rs-rollback/`, `archive/work/issue-25-mtp-batch-width/` for earlier rollback work.
