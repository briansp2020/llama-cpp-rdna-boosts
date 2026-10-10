# Cold-start prompt — 35B-A3B `-sm tensor -ncmoe` field impurity (TODO #53, internal AR)

Paste the block below into a fresh session (cwd `/home/stew675/llama-cpp-rdna-boosts`).

---

You are picking up a single-issue investigation in `/home/stew675/llama-cpp-rdna-boosts` (the
rdna-boosts delivery repo for the `stew675/llama.cpp` fork). Read
`wip/moe-verify-fusions/HANDOVER-35b-internal-ar.md` **first, in full**, then `AGENTS.md`
("Report findings before shipping", "Pushing policy", "WIP and promotion rules", "Default-on policy")
and `wip/moe-verify-fusions/README.md`.

The task: `Qwen3.6-35B-A3B-Q8_0` (`qwen35moe`, embedded nextn MTP) is **not bit-pure** under
`-sm tensor -ncmoe` in the **field** (`--spec-type none` vs `--spec-type draft-mtp --spec-draft-n-max 3`,
2 GPU, `-ncmoe 99`, `MOE_EXPERT_CACHE_MIB=0`, `mixed30k`, `NP=128`, `--fit on` → first-diff 8). The
probe (`test-logits-width-probe`) is already fixed and shipped in block 15; **do not reopen it**.

The root cause is already found: the **internal host-staged all-reduce**
(`ggml/src/ggml-cuda/allreduce-hip.cu`) produces a different result from the RCCL/NCCL path; with
`GGML_CUDA_ALLREDUCE=nccl` the field is **pure** (verified on the short reproducer and the real long
field). Making the internal AR F32 (`GGML_CUDA_AR_BF16_THRESHOLD=0` or `=131072`) is **not**
sufficient, so it is a correctness/state bug in the internal pipeline, not a precision difference.

Do the decisive instrument first (HANDOVER §7): for one `linear_attn_out-*` all-reduce, dump the
**local input hash** and the **post-AR output hash** on both devices for both the `none` and `n3`
arms at a shared token position (the first ~10 generated tokens are identical). If the inputs are equal
and the outputs differ, it is unambiguously the pipeline — fix the host-staged slot/arrival handshake
so the internal AR is bit-identical to RCCL, graph-shape-invariant. If the inputs differ, the AR is a
symptom and the split compute is the real site — say so and pivot.

Reproducer and tooling are in HANDOVER §9; the acceptance and the full gate list are in §1/§10.
Fallback (only with the maintainer's explicit approval): route the affected GDN/MTP small-tensor ARs
through RCCL — verified pure, ~6 % decode / +4.8 % prefill.

Constraints: the fix is a delivery change — report the root cause + proposed fix to the maintainer
**before** any `patches/`/`release.json`/docs-header change, and wait for a decision. WIP stays WIP;
do not fold anything into the delivery unasked. Never push to upstream `ggml-org/llama.cpp`; only this
repo (`github.com:stew675/llama-cpp-rdna-boosts`) and the personal fork
(`git@github.com:stew675/llama.cpp.git`). No parallel GPU benches; one arm at a time. The scratch
worktree `/tmp/rdna-fold15` holding the session instrumentation is disposable — reset it
(`git checkout -f b8d1e2988`) and re-instrument as needed.
