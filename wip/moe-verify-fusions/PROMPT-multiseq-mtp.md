# Cold-start prompt — multi-sequence MTP non-determinism (TODO #52)

Paste the block below into a fresh session (cwd `/home/stew675/llama-cpp-rdna-boosts`).

---

You are picking up a single-issue investigation in `/home/stew675/llama-cpp-rdna-boosts` (the
rdna-boosts delivery repo for the `stew675/llama.cpp` fork). Read
`wip/moe-verify-fusions/HANDOVER-multiseq-mtp.md` **first, in full**, then `AGENTS.md` ("Report findings
before shipping", "Pushing policy", "WIP and promotion rules", "Default-on policy") and
`wip/moe-verify-fusions/README.md`.

The task: **multi-sequence MTP is non-deterministic** (`TODO.md` #52). With
`llama-server -np 2 --kv-unified --spec-type draft-mtp` and two concurrent greedy requests (A long keeps
decoding after B short finishes), the output is **not reproducible across identical repetitions**, and A
differs when concurrent vs solo. The run logs:

```
seq_rm: rollback crossed a batch boundary: seq 1 rollback=2 but the last batch decoded 6136 tokens
        (last pos 18419, n_rs_seq=3, n_rs_batch=4). ... the restored state is wrong
```

i.e. a verify-rejection rollback of sequence 1 lands while the last decoded batch was sequence 0's large
prefill, so the recurrent (GDN) snapshot plane it restores was never written. It reproduces on our r41
tree with no PR #124 skip (so it is not PR #124), is identical under `GGML_CUDA_ALLREDUCE=ce`, and
`GGML_CUDA_GDN_CHUNKED=0` does **not** fix it (there is a second, batching-level component).

What is already resolved — **do not reopen**: the single-sequence plain-vs-MTP divergence (#50) was the
device-remap path / lazy NCCL init, fixed and delivered in r40/r41; the 35B-A3B `-sm tensor -ncmoe`
impurity (#53) was the lazy `ncclCommInitAll`, fixed in r41. #52 is a **separate root**: the §27
invariant ("every batch that can be rolled back into must run the kernel that writes the snapshots it
will read"), violated **across sequences**.

Reproduce first with `wip/moe-verify-fusions/tools/multiseq.sh` (HANDOVER §7); any `A_concN != A_solo` or
`A_concN != A_conc1` is the bug. Build the **canonical** tree from a fresh checkout at `release.json.base`
+ `scripts/apply-all.sh` — `~/llama.cpp` is NOT the canonical chain (stale, dirty). Then find the exact
condition that lets the rollback cross the batch boundary and repair it; the preferred fix is to make a
sequence's rollback read only snapshots its own verify batch wrote (or convert to a full checkpoint when
the last ubatch was another sequence's non-verify prefill). The full code map, the invariant, fix
directions and the gate list are in HANDOVER §5/§6/§8/§9.

Constraints: the fix is a delivery change — report the root cause + proposed fix to the maintainer
**before** any `patches/`/`release.json`/docs-header change, and wait. WIP stays WIP; do not fold
anything into the delivery unasked. Never push to upstream `ggml-org/llama.cpp`; only this repo
(`github.com:stew675/llama-cpp-rdna-boosts`) and the personal fork
(`git@github.com:stew675/llama.cpp.git`). No parallel GPU benches; one arm at a time. Do not "fix" it with
a blanket run-wide `GGML_CUDA_GDN_CHUNKED=0` (perf regression, and the instability survives it).
