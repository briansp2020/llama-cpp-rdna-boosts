# Cold-start prompt - multi-sequence non-determinism, the residual (issue #134 / TODO #52, component C)

Paste the block below into a fresh session. It is self-contained; the handover has the full detail.

---

You are picking up the **residual** of a multi-sequence non-determinism investigation in
`/home/stew675/llama-cpp-rdna-boosts` (the rdna-boosts delivery repo for the `stew675/llama.cpp` fork).
The upstream report is issue #134. One root (the multi-sequence `K == 1` GDN kernel-choice
inconsistency) is **fixed and delivered** in `v16-a55e952b8-r42`; do not reopen it. A companion batching
change was prototyped, is decode-cost-free, but breaks `llama-perplexity` at `n_seq >= 4`, so it was
**held**. The **residual (component C)** is still open.

Read `wip/moe-verify-fusions/HANDOVER-multiseq-residual.md` **first, in full**, then `AGENTS.md`
("Report findings before shipping", "Pushing policy", "WIP and promotion rules", "Default-on policy") and
`wip/moe-verify-fusions/README.md`.

Reproduce C first, on a canonical build of the **r42** tree (`release.json` names the base/tip;
`git checkout rdna-boosts` on the personal fork, or a fresh `release.json.base` + `scripts/apply-all.sh`,
then `BUILD_DIR=build-rocm-hybrid EXTRA_CMAKE_FLAGS="-DCMAKE_HIP_FLAGS=" ~/bin/build-llama-rocm-714`).
The reproducer is `wip/moe-verify-fusions/tools/multiseq.sh <build_dir> <tag>` (and its `_plain`,
`_np2*` variants). Two concurrent greedy requests A (long) and B (short); any `A_conc != A_solo` or
`A_concN != A_conc1` is the bug.

C is **not** ubatch co-batching: the gate still diverges with no cross-sequence ubatch at all
(`GGML_FORCE_SEQ_UBATCH=1`), with the expert cache off, and with the r42 fix. It **is** unified-KV
related: `-np 2 --no-kv-unified` plus no co-batching is the only fully green configuration found, and
`-ub 12288` does not change it. The leading hypothesis is that under unified KV a query's attention
result depends on the cache's total extent / cell layout (`n_kv`, `k_idxs`), which changes with the
co-resident sequence and across repetitions; the FA chooser is `Q->ne[1]`-dependent and the delivery's
dense mmvq/mmq band is only uniform to about M=16. The minimal probe does **not** reproduce C, so work on
the server and use the aliasing-free per-node dump (`LLAMA_DUMP_CUDA`, saved in
`session-multiseq-instrumentation.diff`) to diff solo vs concurrent and name the first divergent tensor.
There is also an open question of **why the held batching change breaks `llama-perplexity` at
`n_seq >= 4`** (the output-index mapping is identical; the forward is not) - solving that would let the
free fix land.

Then find the exact condition and the fix. A complete fix of C is the goal; report the root cause and the
proposed fix to the maintainer **before** any `patches/`/`release.json`/docs-header change, and wait.
Constraints: WIP stays WIP; never push to upstream `ggml-org/llama.cpp` (only this repo and the personal
fork `git@github.com:stew675/llama.cpp.git`); no parallel GPU benches, one arm at a time;
`llama-cli` needs `--single-turn`; the prefill-logit gate (`scripts/gate-prefill-logits.sh`) must stay
green for any change that moves prefill logits.
