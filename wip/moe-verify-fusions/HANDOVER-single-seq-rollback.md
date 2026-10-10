# HANDOVER — the single-sequence plain-vs-MTP divergence (qwen4exp recurrent rewind)

**For a cold session.  Status: OPEN, sharply narrowed.  Read this end to end before touching anything.**
This continues `wip/moe-verify-fusions/` (campaign README) and supersedes the "verify-width kernel
arithmetic" hypothesis in `HANDOVER-slab-narrow2.md` §5 — that hypothesis is **refuted** (see §3).

The sibling multi-sequence defect found on the way is tracked as **`TODO.md` #52**; its handover lives in
this same directory (section §6 below).  It is a *different* manifestation but the same recurrent-rewind
surface.

---

> **2026-10-12 (second session) RESULT — the rewind hypothesis below is REFUTED.**  A single-context
> rewind probe (`tools/rrewind.cpp`) compares, at the same position and starting state, a plain
> single-token decode against a `[t_k, junk x3]` verify batch + rejection rollback, checking both the
> row-0 logits and the serialized state bytes.  It is **PURE (0 logit / 0 state mismatches)** on the exact
> configuration the divergence needs — 2 GPU `-sm tensor`, partial expert cache (`-sm tensor` + qwen4exp),
> and even with the cache resident set moved between the two paths.  The divergence is therefore **not**
> the KV/GDN-ssm/GDN-conv/QSA rewind.  The controlled matrix shows it is a **cache-residency / graph-
> planning** effect: qwen4exp + 2 devices + `-sm tensor` + a partial cache; it disappears on 1 GPU, on 3
> GPU full residency, on `-sm layer`, and with the cache off, and it is qwen4exp-specific (the 35B-A3B
> control is pure even at 2-GPU tensor partial).  The plain arm is itself cache-sensitive (`none`
> cache-on vs cache-off first-diff 103), so there is no cache-invariant "plain reference".  This is
> `TODO.md` **#50**, not a rewind bug.  Full evidence, matrix, tools and reproducers:
> [`FINDINGS-single-seq-rollback.md`](FINDINGS-single-seq-rollback.md).  Section §6 (#52) below still
> stands as a separate, genuine recurrent-rollback-boundary bug.
>
> **Follow-on (same session): the divergence is the target's `n_rs_seq` (0 vs 3) graph × the expert cache.**
> A temporary `GGML_FORCE_N_RS_SEQ=<n>` pins the snapshot depth independently of `--spec-type`; with
> everything else identical, `n_rs_seq=0` vs `3` is **bit-identical with the cache off** and **flips the
> argmax (first-diff 59, ~1.6-nat top-2 shift) with the cache on**, independent of the fusions
> (`DISABLE_FUSION` still diverges) and the slab (`GGML_CUDA_SLAB=0` still diverges).  The plain arm is
> `n_rs_seq=0`, the MTP arm is `n_rs_seq=3`.  So the lead is not the rollback but **the snapshot-armed
> graph's cache-band MoE arithmetic not being bit-identical to the snapshot-free graph's** — the repair
> target.  The directed `#50` per-table fusion guard was implemented this session (see
> `FINDINGS-single-seq-rollback.md` §6) and is a no-op for this divergence.

---

## 1. TL;DR

The original campaign item (c) was "the plain-vs-MTP verify-width divergence (`none != n1 != n3 != n7`)".
This session proved:

1. **The model forward pass is width- and `n_rs_seq`-pure** (the dedicated probe passes on every config),
   so it is NOT a verify-width kernel arithmetic problem.
2. **The divergence requires rejected draft tokens** (acceptance < 1).  With a degenerate prompt
   (acceptance 1.0) the streams are byte-identical; only rejections (which trigger a recurrent-state
   rollback) expose it.
3. **It is qwen4exp-specific.**  A non-qwen4exp MoE with the cache engaged and MTP is byte-identical.
4. The remaining surface is the **qwen4exp recurrent/conv rewind combined with the live expert cache** —
   qwen4exp is the only tested model with recurrent (GDN) layers, and the cache is the only differentiator
   the clean control removes.

**Next action:** build a **teacher-forced per-step logits replay around a rejection** (solo, one sequence)
to see *which* qwen4exp memory child (KV, GDN ssm, GDN conv, or QSA idx) restores the wrong state.  That is
the instrument that caught §28/§29; it is seconds per config instead of a 15-minute 32k-token text run.

---

## 2. The problem, exactly

Field config (D2): **GSQ-IQ3_XXS + the shared Q8_0 MTP head**, 2 GPU, `-sm tensor -ncmoe 48`, 32k
`/tmp/srr/mixed30k.txt`, greedy.  Comparing the plain `--spec-type none` stream to the MTP streams at
`--spec-draft-n-max 1/3/7`:

| arm (current build, narrow-2 default) | none vs n1 | none vs n3 | none vs n7 |
|---|---|---|---|
| GSQ, `mixed30k` | idx 89 | idx 9 | identical |
| GSQ, `prose30k` (degenerate, acceptance 1.0) | **identical** | **identical** | **identical** |
| IQ4_NL, `mixed30k` | idx 82 | idx 151 | idx 185 |

The divergence index is **not stable** across configs (it is a near-tie flip driven by the trajectory, so
treat "which width and which index" as noise — the *presence* of a divergence at acceptance < 1 is the
signal).

---

## 3. What is PROVEN (do not re-derive)

### 3.1 The forward pass is pure (`tests/test-logits-width-probe`, extended this session)

The probe prefills a prefix, then decodes a W-token batch for W=1..8 and requires row 0 (and every shared
row) to hash identically across W.  On GSQ, `NCMOE=48` / `SPLIT=tensor` extensions added this session:

| config | P | result |
|---|---|---|
| all-GPU, layer split | 256 | **PASS** (maxdiff 0) |
| `NCMOE=48` host experts, layer split | 256 | **PASS** |
| `NCMOE=48`, **`-sm tensor`** | 256 | **PASS** |
| `NCMOE=48`, tensor, `RS=0/1/3/7/from_w` | 256 | **PASS**, identical hash for every RS |
| `NCMOE=48`, tensor, RS=from_w | **2500** (> the 2051 QSA-indexer threshold) | **PASS** |

**Watch-out:** a P below ~2051 takes the **dense QSA shortcut** and tests nothing (see the
`archive/work/kv-quant-purity-followups/tools/qsa-text-gate.sh` note).  Always use P>2051 when probing the
QSA path.

### 3.2 It needs rejections

`prose30k.txt` (the degenerate copy prompt) gives acceptance **1.00000** and `none == n1 == n3 == n7`
**byte-identical**.  `mixed30k` (non-degenerate) gives acceptance < 1 and diverges.  So the differentiator
is the recurrent-state **rollback** that a rejection triggers, not the verify width.

### 3.3 It is qwen4exp-specific (the decisive control)

Non-qwen4exp MoE, cache engaged, MTP, **one GPU**: Qwen3.6-35B-A3B **Q4_K_M** (embedded `qwen35moe.nextn`
MTP head), `-ncmoe 20`, `CTX=40960`, same 32k prompt:

```
none vs n1: first-diff None
none vs n3: first-diff None
none vs n7: first-diff None        (256 tokens each; acceptance 0.94 / 0.80 / 0.57)
```

**Byte-identical.**  The expert cache, the rejection rollback, the hybrid/spec machinery and the
embedded-MTP path are sound in general.  The bug is in the qwen4exp layers.

**Watch-out:** the 35B-A3B **Q8_0** with `-ncmoe 48` on one GPU **crashes** MTP with an HSA
`MEMORY_APERTURE_VIOLATION` (`k_get_rows_float_vec`) — use the Q4_K_M/`-ncmoe 20` control (or a smaller
context).  Also `D=${D:-default}` in a harness silently substitutes the default draft head when `D` is set
**empty**; to mean "use the embedded MTP head" the harness must use `D=${D-default}` (already fixed in
`tools/widthsweep.sh`).

---

## 4. Negative matrix (all on GSQ, `mixed30k`, 2 GPU; "idx" = first divergence vs plain)

| control | none vs n1 | none vs n3 | none vs n7 | read |
|---|---|---|---|---|
| narrow-2 default | 89 | 9 | None | baseline |
| quick-fix A/B (`GGML_CUDA_SLAB_NARROW2=0`) | 9 | 64 | 64 | slab layout matters (#50); not the root |
| `GGML_CUDA_DISABLE_FUSION=1` | 116 | 65 | None | not the fusions (this also kills all PR #114 fusions) |
| `GGML_CUDA_FUSE_GDN_BETA_SIGMOID=0` (PR #114 0004) | 89 | 9 | None | byte-identical → the beta-sigmoid fusion is exact |
| `GGML_CUDA_GDN_CHUNKED=0` | 112 | 59 | None | not the chunked-GDN kernel (single-seq) |
| `LLAMA_QSA_SPARSE_FA=0` (dense QSA) | 64 | 84 | 112 | not the sparse indexer |
| `MOE_EXPERT_CACHE_MIB=0` (cache off) | None | None | 147 | cache is required for n1/n3 |
| `GGML_CUDA_SLAB=0` (cache on, slab off) | None | (run failed) | 9 | slab interacts |
| `MOE_EXPERT_CACHE_DEVPOLICY=0` (host policy) | 89 | 9 | None | not the device policy kernel |

`test-recurrent-state-rollback` **PASSES** on GSQ (both cache fills, max diff 0), but note its own TODO in
`tests/test-recurrent-state-rollback.cpp:402`:

> *"this test is invalid because RS rollback is only correct once after a ubatch with more than n_rs_seq
> tokens; this is not the case here. add asserts and guardrails to prevent such attempts"*

So the plain recurrent rollback is correct **for the scenario the test covers**; the MTP/qwen4exp
combination is not covered by it.

---

## 5. The lead: qwen4exp recurrent/conv rewind + live cache

### 5.1 Code map (start here)

| file | what to read |
|---|---|
| `src/llama-memory-recurrent.cpp:185-255` | `seq_rm` partial rollback: sets `rs_idx`, and emits the **"rollback crossed a batch boundary"** warning (once per process, `static`) |
| `src/llama-memory-recurrent.cpp:430-460` | `set_rs_idx` (`idx <= n_rs_seq`) |
| `src/llama-memory-recurrent.cpp:835-880` | `state_write` — how `rs_idx` selects the snapshot row |
| `src/models/delta-net-base.cpp:470-640` | the GDN **ssm** snapshot slots + the **pre-batch plane** (`0 < n_seq_tokens < K`) — §27's invariant |
| `src/models/qwen4exp.cpp:2840-2870` | the GDN **conv**-state snapshot (`n_slots = n_rs_seq + 1`, slot `s` = `s` tokens back) |
| `ggml/src/ggml-cuda/gated_delta_net.cu:290-430` | the sequential vs chunked GDN dispatch, the `GDN_CHUNKED_MIN_TOKENS` bound, the snapshot-slot mapping (`target_slot = n_tokens-1-t`), the beta-sigmoid fusion |
| `tools/server/server-context.cpp:4170-4250` | the MTP accept/rejection and the target rollback: `n_rollback = draft+1-accepted`; `slot.mem.seq_rm(slot.id, pos_next, -1)` |
| `ggml/src/ggml-cuda/moe-expert-cache.cu:1311-1470` | `access_locked` + the intra-op `protect` list (already pins every expert of the current op: `n = n_used * n_tok`) |
| `ggml/src/ggml-cuda/moe-expert-cache.cu:2796-2940` | the device policy kernel (same protect, "never a victim used earlier in this same token") |

### 5.2 The invariant to test

`GREEDY-PURITY.md` **§27**: *every batch that can be rolled back into must run the kernel that writes the
snapshots it will read.*  The MTP verify batch is `n_tokens = n_max+1 = n_rs_seq+1 = K`, so the pre-batch
plane (`0 < n_tokens < K`) is **not** written, and only slots written by the sequential kernel are valid.
The single-sequence failure mode to look for: a rollback that lands on a slot the last batch did not write.

### 5.3 Why the cache could matter

The clean control removes the cache and the divergence (for n1/n3); qwen4exp is the only recurrent model
tested.  Do NOT assume the cache is "just timing" — establish whether an eviction can change **which
kernel/graph** runs (cache-aware fusions) *between a verify and its rollback*, and whether the rollback's
snapshot slot depends on that.  #50 (global fusion gate over per-table eviction) is the documented
mechanism for the fused/non-fused arithmetic flip and is still open; the acceptance gate for the narrow-2
work is confounded by it (see `RESULTS-slab-narrow2.md`).

### 5.4 Recommended next steps, in order

1. **Teacher-forced per-step logits replay around a rejection.**  The instrument that caught §28/§29 is
   `archive/work/strix-halo/qsa-item4/` (`mstep`, `NEXTN=1`).  Generalise it: feed the plain token stream,
   at one step decode a verify batch (last accepted token + draft proposals), force a rejection, roll back,
   then decode the *plain* next token and compare its logits to the plain run's.  Run it on GSQ with the
   cache on.  If the post-rollback logits differ, bisect which state child is wrong (KV vs ssm vs conv vs
   idx) by dumping the snapshot slot read vs written.
2. **Instrument the snapshot read.**  Add a debug env that logs, per GDN layer and per rollback, the
   written slot set (`last_ubatch_nseq_tokens`, `K`) and the `rs_idx` read.  Look for reads of
   unwritten slots in the **single-sequence** case (the warning never fires there, so check the raw values,
   not just the warning).
3. **Check the first verify after the chunked prefill.**  The single-sequence first verify follows a
   6144-token chunked prefill.  Verify that its rollback slot is one the prefill wrote (the pre-batch
   plane / slot-0 logic).
4. Only after the state child is identified: decide whether to force the sequential GDN for that batch,
   write the pre-batch plane for the chunked prefill, or repair the specific snapshot.

---

## 6. Sibling defect: multi-sequence non-determinism (`TODO.md` #52)

Our build (no PR #124 skip) is **non-deterministic across identical concurrent repetitions** with
`-np 2 --kv-unified --spec-type draft-mtp`, and the log emits the invalid-rollback condition above.  It is
identical under `GGML_CUDA_ALLREDUCE=ce`, and `GGML_CUDA_GDN_CHUNKED=0` does **not** fix it.  Reproducer:
`tools/multiseq.sh`.  This is a **correctness** item, independent of PR #124, and likely shares the §5
recurrent-rewind surface.  Keep it in the same investigation, but do not conflate it with the
single-sequence divergence (they have different triggers).

---

## 7. State of the delivery/fork and what must not be disturbed

* `~/.pi`/delivery: `~/llama-cpp-rdna-boosts` at `main`; **`patches/` and `release.json` are untouched**
  by this campaign (the WIP rule).  This session added only `wip/moe-verify-fusions/` docs/tools and a
  `TODO.md` #52 entry.
* `~/llama.cpp`: branch `rdna-boosts` (block-15 tip), **uncommitted** working tree containing:
  * the **slab-resident narrow-2** layout (the durable fix for the MTP draft/target compute-buffer
    aliasing) — see `RESULTS-slab-narrow2.md`; it is functionally validated, and the acceptance gate is
    #50-confounded;
  * the **A/B arm** `GGML_CUDA_SLAB_NARROW2=0` (the quick fix: the draft's COMPUTE buffers via
    `cudaMalloc`); default is narrow-2 (`=1`);
  * env-gated diagnostics: `GGML_CUDA_WORK_ALLOC_DEBUG` (SLAB_INIT / WORK_ALLOC / NARROW2_* /
    COMPUTE_ALLOC / GRAPH_RESERVE) and `GGML_CUDA_CACHE_GUARD_DEBUG` (`moe_cache_guard_stats`);
  * `tests/test-logits-width-probe` extended with `NCMOE` / `SPLIT`;
  * the whole diff is saved at `wip/moe-verify-fusions/session-narrow2.diff`.
* **Do not commit the fork tree to the canonical chain or push anything** without the maintainer's
  go-ahead (AGENTS.md Pushing policy; the only push targets are this delivery repo and the personal fork).

### Build / environment (the handover's older paths are STALE)

```bash
cd ~/llama.cpp
EXTRA_CMAKE_FLAGS="-DCMAKE_HIP_FLAGS=" ~/bin/build-llama-rocm-714   # fresh configure (CMake >= 4.3)
cmake --build build-rocm-hybrid --target llama-server llama-cli test-logits-width-probe -j 16
export LD_LIBRARY_PATH=/opt/rocm-7.14.1-gfx120X/lib   # NOT /opt/rocm-7.14-gfx120X
```

`llama-cli` needs `--single-turn`; `-fa off` is invalid with `-sm tensor`; never run parallel benches; warm
up before recording; `/tmp/srr/mixed30k.txt` is `README.md + ENVIRONMENT.md + CONTAINERS.md +
archive/docs/baseline-history.md` (sha256 `69624f4d207f40bd…`), `/tmp/srr/prose30k.txt` is the degenerate
copy prompt (acceptance 1.0 — never measure on it), `/tmp/srr/probe9k.txt` is the probe text.

---

## 8. Side notes / watch-outs (things that cost us time)

* **The probe's P matters** — P<2051 tests the dense QSA shortcut (see §3.1).
* **`GGML_CUDA_ALLREDUCE`** — the default hybrid AR is not deterministic on 2 GPUs; `ce` is the only
  deterministic one (2-GPU only).  A/B text comparisons must pin `ce` or be repeated; the multi-seq **hash**
  result was identical under `ce`, which is what proved the AR was not the cause.
* **The "rollback crossed a batch boundary" warning is once per process** (static) — absence from a later
  run's log does not mean the condition did not occur.
* **`GGML_CUDA_DISABLE_FUSION=1` also skips the `graph_optimize` alloc-deps pass** (PR #27301), so it is a
  confounded A/B; the individual switches are the clean ones.  All of PR #114's fusions are inside
  `ggml_cuda_try_fuse` and are therefore killed by `DISABLE_FUSION` too (verified at `ggml-cuda.cu:6078-6081`).
* **PR #114 / PR #115 are both folded** into our r39 (`MMVQ_MOE_MAX_BATCH_SIZE 16`, `MOE_CACHE_POLICY_MAX_FILL
  256` + guard, the qwen4exp HC/GDN fusions).  They are ruled out for this divergence (PR #114 by the
  switches, PR #115 because our verify width is 4, inside the old band too).
* **PR #124 is NOT to be merged as-is.**  It is the unmerged GDN **state-gather skip**
  (`GGML_CUDA_FUSE_GDN_STATE_GATHER`); the community found a 2→1-sequence non-determinism with it.  Its two
  in-tree fixes are a **real lifetime fix** our tree lacks — `src/llama-graph.cpp` should mark `rs_s_copy`
  as an **output**, and `ggml/src/ggml-backend.cpp` (`ggml_backend_sched_split_graph`, the `n_copies == 1`
  branch) should keep the source's OUTPUT flag on the split-input copy.  Our tree has the `n_copies > 1`
  case only (`ggml-backend.cpp:1858`).  This is a **generic latent hazard** for any OUTPUT tensor copied
  across devices under `-sm tensor` (e.g. the MTP `t_h_nextn` export, §28) — audit it, but as a separate,
  maintainer-approved change (it touches the shared scheduler; cross-backend consistency per AGENTS.md).
* **The narrow-2 acceptance gate is unusable while #50 is open** — see `RESULTS-slab-narrow2.md`: the quick
  fix itself measures 0.738-0.763 across arena layouts, and pinning the cache budget flips the ordering.
  Do not "fix" the layout to chase that number.
* **`TODO.md`'s "Current state" header lags** (it still says r32 while `WORKLOG.md` is at r39).  Trust
  `release.json` + the top of `WORKLOG.md`.

---

## 9. Pointers

* `wip/moe-verify-fusions/README.md` — campaign framing, D1/D2.
* `wip/moe-verify-fusions/FINDINGS-width-divergence.md` — the raw findings log from this session.
* `wip/moe-verify-fusions/RESULTS-slab-narrow2.md` — the narrow-2 layout + the #50 confound.
* `wip/moe-verify-fusions/session-narrow2.diff` — the full `~/llama.cpp` working-tree diff.
* `wip/moe-verify-fusions/tools/widthsweep.sh` — the plain-vs-MTP width sweep (return_tokens).
* `wip/moe-verify-fusions/tools/multiseq.sh` — the multi-sequence control (TODO #52).
* `GREEDY-PURITY.md` **§27** (rollback bound), **§28** (export tail), **§29** (QSA indexer `ne11` width).
* `tests/test-recurrent-state-rollback.cpp` — the recurrent rollback test (and its TODO at :402).
* `archive/work/strix-halo/qsa-item4/` — the `mstep` teacher-forced replay (§28/§29's instrument).
* `archive/work/issue-25-mtp-batch-width/` — the earlier FA `parallel_blocks` width fix (block 00).
* `TODO.md` **#50** (per-table fusion guard), **#51** (3-GPU MTP), **#52** (multi-seq non-determinism).
* `AGENTS.md` — Pushing policy, WIP/promotion rules, default-on policy.
