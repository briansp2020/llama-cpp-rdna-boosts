# HANDOVER - multi-sequence non-determinism follow-ups: B (the batching change) and C (the unified-KV residual)

**For a cold session. This is the working home for issue #134 / TODO #52 after `v16-a55e952b8-r42`.**
Read this in full, then `AGENTS.md` ("Report findings before shipping", "Pushing policy", "WIP and
promotion rules", "Default-on policy"), then the campaign `README.md`. Report the root cause and the
proposed fix to the maintainer **before** any `patches/`/`release.json`/docs-header change, and wait.
There are **two jobs**: **B** (a delivered-direction batching change that is held because it breaks
`llama-perplexity`) and **C** (the residual server non-determinism that survives everything fixed so far).

---

## 0. Cold-start prompt

See the companion file [`PROMPT-multiseq-bc.md`](PROMPT-multiseq-bc.md). Paste it verbatim.

---

## 1. Status when this was written (2026-10-12, r42)

- `release.json` = `v16-a55e952b8-r42`, base `a55e952b8`, tip
  `a3c13e9ca76717d8dd9b6020fca2b35bcd2c46b6`, tree `eafc03875b2c894bda7d8081d362b234a55a9c54`, 16 blocks.
- The personal fork `rdna-boosts` is at `a3c13e9ca`. The delivery repo `main` is pushed.
- Green on this tree: `scripts/validate-set.sh`; coherence `1c5d32ac537d` (Qwen3.5-4B-Q8_0, seed 42);
  `scripts/gate-prefill-logits.sh` KLD 0.000707 / same-top-p 98.755 % PASS;
  `test-recurrent-state-depth` unchanged (130 baseline failures, phase A 1..7 PASS).
- **Do not reopen A** (below). B is prototyped and held. C is open.

## 2. Already fixed and delivered (A) - do not reopen

The GDN dispatch (`ggml_cuda_op_gated_delta_net_impl`, `ggml/src/ggml-cuda/gated_delta_net.cu`) took the
**whole-batch chunked** kernel for `n_seqs > 1 && K == 1` **unconditionally**, while the same sequence
decoded alone with `n_tokens <= GDN_CHUNKED_MIN_TOKENS` took the **sequential** kernel. The two are not
bit-identical, so with `--spec-type none` (or `GGML_FORCE_N_RS_SEQ=0`) a sequence's logits depended on the
sequences that shared its ubatch. Fixed in **block 02** in r42: the `n_seqs > 1` branch now uses the same
`n_tokens > GDN_CHUNKED_MIN_TOKENS` gate as the single-sequence branch. See `WORKLOG.md` 2026-10-12 (r42),
`patches/README.md` `0002`, `TODO.md` #52. The probe (`tools/test-recurrent-state-multiseq.cpp`) confirms
solo == co-batch for K=1/4/16/32 on Qwen3.5-4B and the Flash-Next GSQ target.

This does **not** fix issue #134's exact path (that uses `--spec-type draft-mtp`, `n_rs_seq = 3`, K = 4) and
it does **not** fix the residual C.

---

## 3. Job B - the held batching change (multi-token co-batch split)

### What it is

A one-block change in `llama_batch_allocr::split_equal` (`src/llama-batch.cpp`): when a ubatch would carry
**more than one token for any sequence**, emit **one sequence per ubatch** instead. Single-token decode
keeps its cross-sequence weight reuse. Saved patch: `held-B-multi-seq-ubatch.patch` (applies to the r42
tree; the delivery form would use a `LLAMA_MULTI_SEQ_UBATCH=0` kill-switch). The WIP equivalent is
`GGML_FORCE_SEQ_UBATCH` (1 = split all, 2 = split only multi-token) - see the instrumentation diff.

### Why it is wanted

It makes a sequence's forward shape independent of its co-resident sequences for the multi-token band,
which is the direct cause of the co-batch divergence for the widths whose kernels are not uniform (the
dense mmvq/mmq band, the FA `Q->ne[1]` chooser, the GDN chunked/sequential switch). Measured:

- The probe is bit-identical solo vs co-batch for K=1/4/16/32 (`test-recurrent-state-multiseq`).
- It is **decode-cost-free** and prefill-positive on GSQ + `-ncmoe 48`, 2x R9700,
  `llama-batched-bench -npp 16 -ntg 32 -npl 1,4,8`:

  | arm | B=1 TG | B=4 TG | B=8 TG | B=4 PP | B=8 PP |
  |---|---|---|---|---|---|
  | baseline (co-batched) | 5.77 | 141 | 138-152 | 48 | 122 |
  | B (multi-token split) | 5.76 | 141-142 | 138-152 | 165 | 198 |

  (For contrast, splitting **all** ubatches including single-token decode, `GGML_FORCE_SEQ_UBATCH=1`,
  costs about 3x decode: B=4 47, B=8 45.)

### Why it is held: it breaks `llama-perplexity` at `n_seq >= 4`

`llama-perplexity` builds `n_seq = n_batch / n_ctx` sequences per batch and requests per-token logits.
With B the whole batch is split into one ubatch per sequence. The perplexity is then **garbage**:

```bash
# 4B (hybrid), B OFF (correct) vs B ON (garbage); the 27B Q8_0 shows the same
env LLAMA_MULTI_SEQ_UBATCH=0 <canon>/build-rocm-hybrid/bin/llama-perplexity \
   -m /llm/models/Qwen3.5/4B/Q8_0/Qwen3.5-4B-Q8_0.gguf \
   -f /llm/models/wikitext-2-raw/wiki.test.raw -c 512 -b 2048 --chunks 4 -ngl 99 -fa on -t 8
# PPL 8.51 (B=0) vs 2097 (B=1).  -b 512 (n_seq=1) and -b 1024 (n_seq=2) are both fine.
# 27B Q8_0: 5.78 vs 3425 at -b 2048, 5.78 at -b 512/1024.
```

**What is already excluded.** The logits **index mapping is identical** with and without B: a
`LLAMA_DEBUG_OUTIDS` print of `balloc->get_out_ids()` after the output-swap block gives the same
`[256..511, 768..1023, 1280..1535, 1792..2047]` for both. So it is not the `output_ids`/`output_swaps`
path; the **forward itself** differs. The probe (2 sequences) is exact, so the failure needs the
`n_seq >= 4` shape or the perplexity's many-outputs-per-sequence batch.

### Hypotheses / next steps for B

1. Run `llama-perplexity` with B on and off at `-b 2048 --chunks 4`, print the per-chunk `[i]ppl` line. If
   every chunk is garbage, it is systematic; if only later chunks are, it is a state/cell-reuse issue.
2. Use the aliasing-free per-node dump (`LLAMA_DUMP_CUDA`, below) to compare one sequence's forward in the
   4-sequence batch with B on vs off, and name the first divergent tensor.
3. Check the per-sequence ubatch state handling: with B, sequence `s` is a single 512-token ubatch, and
   sequences are processed back-to-back in one `llama_decode`. Verify the recurrent (hybrid) and KV cell
   state for sequence `s` survives the interleaved other-sequence ubatches, and that `find_slot`'s
   gather/reorder does not move it under a stale `src0`. A per-sequence single-ubatch decode (one sequence
   per `llama_decode`) should equal the multi-ubatch decode; if it does not, that is the bug.
4. Check whether `n_unused` in the B predicate is right for a batch whose sequences have different token
   counts (it should split only when a sequence contributes more than one token; verify it never splits a
   single-token-only batch, and never leaves tokens unprocessed).
5. If it is a genuine library limitation rather than a B bug, the alternative is to make the offending
   kernel families shape-uniform instead (see C), which is the maintainer's preferred direction but larger.

---

## 4. Job C - the residual server non-determinism (unified-KV state dependence)

### What it is

With A fixed, **B disabled, the expert cache off, and no cross-sequence ubatching at all**
(`GGML_FORCE_SEQ_UBATCH=1`), the server gate (`tools/multiseq.sh`) is **still not byte-reproducible**
under **unified KV**. So C is a state/layout dependence, not ubatch co-batching. It is independent of MTP
(`--spec-type none` diverges), of the recurrent snapshot path (`GGML_FORCE_N_RS_SEQ=0` diverges), of the
expert cache (`MOE_EXPERT_CACHE_MIB=0` diverges), and of the AR (`GGML_CUDA_ALLREDUCE=ce` is identical).

Gate matrix (GSQ + shared Q8_0 MTP head, 2x R9700, canonical r42 build; `tools/multiseq.sh` and its
`_plain` / `_np2*` variants):

| arm | repetitions equal? | conc == solo? |
|---|---|---|
| plain, auto `-np 4`, unified | no | no |
| plain, `LLAMA_MULTI_SEQ_UBATCH=0` | no | no |
| plain, force=1 (no co-batch), unified | no | no |
| plain, force=2, `-np 2 --no-kv-unified` | yes | no (single-token decode co-batch) |
| plain, force=1, `-np 2 --no-kv-unified` | yes | **yes** (fully green) |
| plain, `-ub 12288 -b 12288` | no | no |
| plain, `MOE_EXPERT_CACHE_MIB=0` | no | no |

`tools/multiseq.sh` never passes `-np`, so the server auto-sets `-np 4` and forces `kv_unified=true`; pass
`-np 2` explicitly to test non-unified. That harness quirk is why an early `KVUNI=0` run was a no-op.

### Leading hypothesis

Under unified KV the per-query attention result depends on the cache's total extent / cell layout
(`n_kv`, `k_idxs`), which changes with the co-resident sequence and across repetitions. Supporting
evidence: the FA chooser is `Q->ne[1]`-dependent by construction (`ggml/src/ggml-cuda/fattn.cu`), and the
delivery's dense mmvq/mmq band is only uniform to about M=16 (`GGML_CUDA_DISABLE_MMVQ_DENSE_BAND=1` makes
the probe's K=16 co-batch bit-identical). The probe could **not** reproduce C (a companion prefill of up to
256 tokens, and a KV-length disparity, are bit-exact), so C needs the server's real KV scale.

### Why this is not the `seq_rm` warning

The `seq_rm: rollback crossed a batch boundary` guard is a **false positive**: a sequence's rollback reads
the snapshot plane its own verify batch wrote, and the probe is bit-exact at the production verify width
(K = 4, `n_rs_seq = 3`). The guard compares the global last-ubatch token count, not the plane's owner, and
it also fires on the reference build. Do not chase it as the cause.

### Next steps for C

1. With r42 + force=1 + unified + cache off, diff **solo vs concurrent** (and two concurrent repetitions)
   with the per-node dump on the real model, and name the first divergent tensor.
2. Decide: is it the FA `n_kv`/tile choice, the KV cache cell allocation/defrag, or the `k_idxs` layout?
3. If B is fixed, force=2 + non-unified KV is green and decode-cost-free, leaving only the unified-KV
   piece of C. The maintainer's preferred end state is to make the affected kernels shape-uniform so
   unified KV also stays green, with cross-sequence batching intact.

---

## 5. Tooling and how to rebuild

```bash
# canonical tree at the r42 tip
git clone https://github.com/stew675/llama.cpp && cd llama.cpp
git fetch origin rdna-boosts && git checkout rdna-boosts
BUILD_DIR=build-rocm-hybrid EXTRA_CMAKE_FLAGS="-DCMAKE_HIP_FLAGS=" ~/bin/build-llama-rocm-714
export LD_LIBRARY_PATH=/opt/rocm-7.14.1-gfx120X/lib
```

- **Probe:** copy `wip/moe-verify-fusions/tools/test-recurrent-state-multiseq.cpp` into `tests/`, add
  `llama_build(test-recurrent-state-multiseq.cpp)` to `tests/CMakeLists.txt`, build. Env: `MSQ_ARM`
  (0 = solo ref, 1 = same-batch co-batch, 2 = separate call, 3 = co-batch seq 0 first), `MSQ_RS`
  (`n_rs_seq`), `MSQ_K` (verify width), `MSQ_FILL`, `MSQ_PRE`, `MSQ_UB` (`n_ubatch`), `MSQ_UNI`
  (`kv_unified`), `MSQ_SAMEFILL`. Any `ARM1`/`ARM3` digest that differs from `ARM0` is the bug.
- **Server gate:** `wip/moe-verify-fusions/tools/multiseq.sh <build_dir> <tag>` (plus the `_plain` and
  `_np2` variants used above).
- **Aliasing-free per-node dump:** in `session-multiseq-instrumentation.diff`, `LLAMA_DUMP_CUDA=<substr>`
  in `ggml-backend-cuda` hashes each named node's columns right after its launch. It synchronizes the
  compute stream first: the plain `ggml_backend_tensor_get` copies on `cudaStreamPerThread` and aliases
  otherwise. Use with `GGML_CUDA_DISABLE_FUSION=1` so every node is computed and named, and
  `GGML_STREAMDBG=1` to see the per-island compute calls.
- **Env knobs used:** `GGML_CUDA_GDN_CHUNKED`, `GGML_CUDA_DISABLE_MMVQ_DENSE_BAND`,
  `GGML_CUDA_DISABLE_FUSION`, `GGML_CUDA_DISABLE_GRAPHS`, `GGML_FORCE_N_RS_SEQ`, `MOE_EXPERT_CACHE_MIB`,
  `GGML_CUDA_ALLREDUCE`, `LLAMA_MULTI_SEQ_UBATCH` / `GGML_FORCE_SEQ_UBATCH`.
- One arm at a time. Never parallel GPU benches. `llama-cli` needs `--single-turn`.

## 6. Gates to keep green

`scripts/validate-set.sh`; coherence `1c5d32ac537d`; `scripts/gate-prefill-logits.sh` (KLD <= 0.005 /
same-top-p >= 98 %; **B currently fails this** and must not ship until it passes or is re-baselined for an
understood, intentional shift); `test-recurrent-state-depth` (baseline) and
`test-recurrent-state-rollback`; `tools/multiseq.sh`; `MUL_MAT_ID` 931/931; `FLASH_ATTN_QSA` >= 26/26;
`llama-batched-bench -npp 16 -ntg 32 -npl 1,4,8` (the rule-5 gate).

## 7. Pointers

- Issue #134 (the report), the r42 release note in `WORKLOG.md`, `TODO.md` #52, `patches/README.md` `0002`.
- `HANDOVER-multiseq-mtp.md` (the original #52 handover; A came out of it), `FINDINGS-single-seq-rollback.md`
  section 7 (why #52 is separate from #50), `HANDOVER-single-seq-rollback.md` section 6.
- `archive/work/gdn-rs-rollback/`, `archive/work/issue-25-mtp-batch-width/` for earlier rollback work.
- `held-B-multi-seq-ubatch.patch`, `session-multiseq-instrumentation.diff` in this directory.
