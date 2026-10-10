# Cold-start prompt - multi-sequence non-determinism follow-ups: B (batching) and C (unified-KV residual)

Paste the block below into a fresh session. It is self-contained; the handover has the full detail.

---

You are picking up the follow-ups to a multi-sequence non-determinism investigation in
`/home/stew675/llama-cpp-rdna-boosts` (the rdna-boosts delivery repo for the `stew675/llama.cpp` fork).
The upstream report is issue #134. One root (the multi-sequence `K == 1` GDN kernel-choice inconsistency)
is **fixed and delivered** in `v16-a55e952b8-r42`; do not reopen it. Two jobs remain:

**B. The held batching change.** A `split_equal` change that forces one sequence per ubatch for
multi-token batches. It makes the two-sequence probe bit-identical solo vs co-batch for K=1/4/16/32 and is
decode-cost-free (and prefill-positive) on `llama-batched-bench`, but it **breaks `llama-perplexity` at
`n_seq >= 4`** (PPL 8.51 -> 2097 on the 4B, 5.78 -> 3425 on the 27B), so it was held. The logits index
mapping is identical with and without it, so the break is in the forward. Saved as
`wip/moe-verify-fusions/held-B-multi-seq-ubatch.patch`. Your first task is to find why it breaks and fix
it (or prove it is a library limitation and say so).

**C. The residual server non-determinism.** With the r42 fix, B disabled, the expert cache off, and **no
cross-sequence ubatching at all**, `llama-server -np 2 --kv-unified` is still not byte-reproducible across
identical repetitions, and concurrent differs from solo. It is independent of MTP, the snapshot path, the
expert cache and the all-reduce. The only fully green configuration found is no cross-sequence ubatching
plus `--no-kv-unified`, which points at a unified KV cache layout/state dependence in the attention path.
The `seq_rm: rollback crossed a batch boundary` warning is a **false positive**; do not chase it.

Read `wip/moe-verify-fusions/HANDOVER-multiseq-bc.md` **first, in full**, then `AGENTS.md`
("Report findings before shipping", "Pushing policy", "WIP and promotion rules", "Default-on policy") and
`wip/moe-verify-fusions/README.md`.

Reproduce on a canonical build of the **r42** tree (`release.json` names the base/tip; `git checkout
rdna-boosts` on the personal fork, or a fresh `release.json.base` + `scripts/apply-all.sh`, then
`BUILD_DIR=build-rocm-hybrid EXTRA_CMAKE_FLAGS="-DCMAKE_HIP_FLAGS=" ~/bin/build-llama-rocm-714`). The
reproducers are `tools/multiseq.sh` (and its `_plain` / `_np2` variants) for C and
`tools/test-recurrent-state-multiseq.cpp` for the probe; the aliasing-free per-node dump
(`LLAMA_DUMP_CUDA`, in `session-multiseq-instrumentation.diff`) is the instrument to diff forward.
Note `tools/multiseq.sh` never passes `-np`, so the server auto-sets `-np 4` and forces `kv_unified=true`.

Do B first (it is smaller and may collapse C to the unified-KV piece alone). Find the exact condition and
the fix for each. The maintainer's preferred end state is shape-uniform kernels with cross-sequence
batching and unified KV both intact, not a feature stand-down. Report the root cause and the proposed fix
to the maintainer **before** any `patches/`/`release.json`/docs-header change, and wait.

Constraints: WIP stays WIP; never push to upstream `ggml-org/llama.cpp` (only this repo and the personal
fork `git@github.com:stew675/llama.cpp.git`); no parallel GPU benches, one arm at a time; `llama-cli` needs
`--single-turn`; `scripts/gate-prefill-logits.sh` must stay green for any change that moves prefill logits
(B currently fails it).
