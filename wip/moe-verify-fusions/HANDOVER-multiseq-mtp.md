# HANDOVER — multi-sequence MTP non-determinism (TODO #52): a rollback reads a recurrent snapshot the last batch did not write

**For a cold session. Take this to completion.** Scope is **only** `TODO.md` #52. The sibling
single-sequence plain-vs-MTP divergence (`TODO.md` #50) is **RESOLVED and delivered** (r40 + r41) — it was
the device-remap path / the lazy NCCL init, not the recurrent rewind; **do not reopen it**. The 35B-A3B
`-sm tensor -ncmoe` impurity (`TODO.md` #53) is also **RESOLVED and delivered** (r41). What remains is the
multi-sequence defect below.

---

## 0. The prompt to paste into the cold session

See the companion file [`PROMPT-multiseq-mtp.md`](PROMPT-multiseq-mtp.md) (paste it verbatim). In short:
read this handover first, then `AGENTS.md` ("Report findings before shipping", "Pushing policy", "WIP and
promotion rules", "Default-on policy"), reproduce #52 with `tools/multiseq.sh`, find the exact condition
that lets a rejection rollback cross a batch boundary, repair it (the preferred fix is to make the rollback
read only snapshots its own batch wrote), and report the root cause + fix to the maintainer **before** any
`patches/`/`release.json`/docs-header change.

---

## 1. Deliverable / acceptance

**The multi-sequence `-np 2 --kv-unified` MTP scenario is deterministic and sequence-independent.**
Concretely, on the reconstructed delivery tree:

- `tools/multiseq.sh` (two concurrent greedy requests A long / B short, 3 repetitions each, plus solo
  runs): **every concurrent repetition is byte-identical to the solo run and to each other** for both A and
  B (currently A_conc differs across repetitions and from A_solo).
- the `rollback crossed a batch boundary` warning **never fires** (currently it does);
- the scenario stays deterministic with the expert cache **on and off**, on **1, 2 and 3 GPUs**, and with
  `GGML_CUDA_GDN_CHUNKED` **both** 1 and 0 (the second, batching-level component identified below survives
  `GDN_CHUNKED=0`, so a fix that only touches the chunked kernel is incomplete);
- **no regression**: the r41 gates stay green — `test-recurrent-state-rollback` PASS,
  `test-recurrent-state-depth` no new failures, single-sequence `none == draft-mtp n3` on qwen4exp and the
  35B, `scripts/gate-prefill-logits.sh` (KLD ≤ 0.005 / same-top-p ≥ 98 %), `MUL_MAT_ID` 931/931,
  `FLASH_ATTN_QSA` ≥ 26/26, dense coherence `1c5d32ac537d`, and a ≥1000-token MTP+plain perf A/B.

## 2. Current delivery state (as of this handover)

- `release.json`: release **`v16-a55e952b8-r41`**, base `a55e952b8`, canonical tip
  **`a89ffa2735d2141293ed5472beab6746a47fbbf7`**, tree **`46c4ce7a6a00aa6c6931c5a89faf847dde19fd34`**,
  `n_blocks` 16. `validate-set.sh` green. **Do not hand-edit hashes** — regenerate with
  `scripts/make-release.sh`.
- The delivery no longer contains the device-remap machinery (`DEVMAP`/`KSLOT`/`DEVPOLICY` is gone; the
  eager host-routing path is the only `-sm tensor` split policy) and block 12 now initialises NCCL
  **eagerly** (r41). The single-sequence rewind is therefore pure on the current tree; **#52 is the only
  open item on this surface.**
- `/tmp/srr/msqB.txt` exists (a repo-doc prompt, 16 000 B). `/tmp/srr/mixed30k.txt` is the field prompt
  (sha256 `69624f4d207f40bd…`). `/tmp/srr/prose30k.txt` is the degenerate copy prompt — **never measure on
  it** (acceptance 1.0).
- **`~/llama.cpp` is NOT the canonical chain** (its `rdna-boosts` branch is stale, on a dirty WIP tree).
  Build the canonical tree from a fresh clone/worktree at `release.json.base` + `scripts/apply-all.sh`.

## 3. The bug, exactly

With `llama-server -np 2 --kv-unified --spec-type draft-mtp` and two concurrent greedy requests (A long
keeps decoding after B short finishes), **the output is not reproducible across identical repetitions**, and
the long request differs when concurrent vs solo. Reproduced on **our r41 tree with no PR #124 skip**, so it
is **not** caused by PR #124.

The run logs the invalid-rollback condition, verbatim (once per process — see §11):

```
seq_rm: rollback crossed a batch boundary: seq 1 rollback=2 but the last batch decoded 6136 tokens
        (last pos 18419, n_rs_seq=3, n_rs_batch=4). The recurrent snapshot path assumes only verify
        batches (<= n_rs_batch tokens) are rolled back into; if that batch exceeded the chunked-GDN
        threshold it wrote no usable snapshot and the restored state is wrong -- set
        GGML_CUDA_GDN_CHUNKED=0 for the sequential kernel.
```

i.e. a **verify-rejection rollback of sequence 1 lands while the last decoded batch was a large prefill
batch** (6136 tokens — sequence 0's prefill). The recurrent (`GDN`) snapshot path keeps
`K = n_rs_seq + 1 = 4` planes and only the **sequential** GDN kernel writes them (slot `s` = the state `s`
tokens back); the whole-batch **chunked** prefill kernel writes slot 0 only. So restoring snapshot
`rollback = 2` reads a plane that batch never wrote → a wrong (finite-but-silent) recurrent state → the
sequence drifts. This is a **correctness** bug, not a perf one.

## 4. What is proven / ruled out (do not re-derive)

- **Not PR #124.** Reproduces on our tree with no `GGML_CUDA_FUSE_GDN_STATE_GATHER` skip.
- **Not the all-reduce.** Identical under `GGML_CUDA_ALLREDUCE=ce` (the deterministic AR).
- **Not only the chunked kernel.** `GGML_CUDA_GDN_CHUNKED=0` removes the snapshot *substance* but the
  non-determinism **survives**, so there is a **second, batching-level component** (the state is then
  correct but the batch/rollback bookkeeping still differs). A fix must explain both.
- **Not the single-sequence rewind.** The single-context rewind probe (`tools/rrewind.cpp`) is PURE (0 logit
  / 0 state mismatch) on the exact single-sequence configuration — rollbacks *inside* a verify batch are
  correct. #52 is specifically a rollback that crosses into a batch that is not a verify batch. The #50 fix
  (device-remap removal / eager init) does not address it.
- **The cache is not required** for the invariant violation (it is a batching/recurrent-state ordering bug),
  but the reproducer uses the production cache config; test both.

## 5. Code map (start here)

| file | what to read |
|---|---|
| `src/llama-memory-recurrent.cpp:202-240` | `seq_rm` partial rollback, the `rollback <= n_rs_seq` bound, and the **"rollback crossed a batch boundary"** warning (`last_ubatch_nseq_tokens > n_rs_batch`). This is where the condition is detected — extend it with real instrumentation. |
| `src/llama-memory-recurrent.cpp:470-480` | `split_equal(..., n_rs_seq > 0 ? n_rs_seq + 1 : 0)` — the ubatch split that keeps a sequence's trailing `1 + n_rs_seq` tokens together. Check how it behaves when another sequence's **prefill** shares the ubatch. |
| `src/llama-memory-recurrent.cpp:835-940` | `state_write`/`state_read` — how `rs_idx` selects a snapshot row. |
| `src/models/delta-net-base.cpp` (GDN ssm snapshot slots + the **pre-batch plane**) | §27's invariant: a rollback of the whole batch needs slot `n_tokens`, which the sequential kernel does not write; the graph copies the pre-batch ssm/conv state there when `0 < n_tokens < K`. |
| `src/models/qwen4exp.cpp` (GDN conv-state snapshot) | `n_slots = n_rs_seq + 1`, slot `s` = `s` tokens back. |
| `ggml/src/ggml-cuda/gated_delta_net.cu` | the sequential vs chunked GDN dispatch, `GDN_CHUNKED_MIN_TOKENS = max(K > 16 ? K : 16, n_rs_batch)`, the snapshot-slot mapping (`target_slot = n_tokens-1-t`). |
| `tools/server/server-context.cpp:3236-3260` | the MTP accept/rejection rollback: `use_ckpt_tgt = seq_rm_type == FULL \|\| (RS && draft.size() > llama_n_rs_seq(ctx_tgt))` — the fallback to a full checkpoint. The **multi-sequence** path is exactly where a rollback can land after another sequence's prefill batch. |
| `src/llama-context.cpp:193-207` | `n_rs_seq` / `n_rs_batch = max(n_rs_seq + 1, params.n_rs_batch)`, and the `GGML_FORCE_N_RS_SEQ` TEMP diagnostic (A/B the snapshot depth independently of `--spec-type`). |
| `tests/test-recurrent-state-rollback.cpp` | `multi_seq_split_replay` (passes on our tree, single-context shape). Its own TODO at :402: *"RS rollback is only correct once after a ubatch with more than n_rs_seq tokens; this is not the case here."* |
| `tests/test-recurrent-state-depth.cpp` | Phase A (verify-batch shape) / Phase B (**deep draft**, `n_tokens = n_rs_batch > K`). A useful oracle for the batch-boundary invariant. |

## 6. The invariant (`GREEDY-PURITY.md` §27)

> **Every batch that can be rolled back into must run the kernel that writes the snapshots it will read.**

The recurrent rollback keeps `K = n_rs_seq + 1` planes and only the sequential GDN kernel writes them; the
whole-batch chunked prefill writes slot 0 only, so it is safe for a batch only if that batch is **never**
rolled back into. The bound must come from the **speculator's maximum draft**, not the snapshot depth:
`n_rs_batch = common_speculative_n_max() + 1`, consumed as `GDN_CHUNKED_MIN_TOKENS`.

For #52 the violation is *cross-sequence*: sequence 1's verify rollback is evaluated while the last decoded
batch belongs to sequence 0's prefill. The `seq_rm` guard compares the **last ubatch's** per-seq token count
against `n_rs_batch` — so it catches the case but does not prevent it; the state is already wrong. The fix
must ensure sequence 1's rollback never depends on a plane sequence 0's prefill did not write.

## 7. Reproducer / tooling

```bash
cd /home/stew675/llama-cpp-rdna-boosts/wip/moe-verify-fusions/tools
export LD_LIBRARY_PATH=/opt/rocm-7.14.1-gfx120X/lib      # NOT /opt/rocm-7.14-gfx120X
export M=/llm/models/Qwen3.8/Flash-Next/GSQ-IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf
export D=/llm/models/Qwen3.8/Flash-Next/IQ4_NL/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf
GPUS=0,1 ./multiseq.sh <build_dir> msq
```

`multiseq.sh` starts `llama-server -np 2` (KV unified by default, `KVUNI=1`), runs A and B solo, then 3
concurrent repetitions, and prints each stream's sha and the first-diff of every concurrent A against solo
A and against A_conc1. **Any `A_concN != A_solo` or `A_concN != A_conc1` is the bug.** Env: `NPA` (256),
`NPB` (96), `NCMOE` (48), `CTX` (204800), `KVUNI` (1), `GPUS` (0,1), `M`/`D`, `PROMPT_A`/`PROMPT_B`.

Fast probes to build:
- a **minimal** reproducer: two sequences where sequence 0 runs a `> n_rs_batch` prefill ubatch and sequence
  1 rejects a draft in the same decode call. `tests/test-recurrent-state-depth.cpp`'s harness is the natural
  base (it already constructs `n_rs_seq`/`n_rs_batch`/`n_ubatch` combinations and compares logits).
- a snapshot-slot logger (env-gated): per GDN layer and per rollback, dump the written slot set
  (`last_ubatch_nseq_tokens`, `K`) and the `rs_idx` read, so a read of an unwritten slot is visible
  directly (the warning is once-per-process and only compares counts — do not rely on it).

Build (canonical tree — a fresh clone/worktree at the base):

```bash
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
git checkout a55e952b8
bash /home/stew675/llama-cpp-rdna-boosts/scripts/apply-all.sh .
EXTRA_CMAKE_FLAGS="-DCMAKE_HIP_FLAGS=" ~/bin/build-llama-rocm-714
export LD_LIBRARY_PATH=/opt/rocm-7.14.1-gfx120X/lib
# fast loop: cmake --build build-rocm-hybrid --target llama-server test-recurrent-state-rollback test-recurrent-state-depth -j 16
```

`llama-cli` needs `--single-turn`; `-fa off` is invalid with `-sm tensor`; never run parallel benches (one
arm at a time); `GGML_CUDA_ALLREDUCE=ce` is the deterministic 2-GPU AR (use it for A/B text comparisons, or
repeat).

## 8. Fix directions (in order of preference)

1. **Make the rollback self-contained.** Ensure a sequence's verify rollback reads only snapshots its own
   verify batch wrote — e.g. refuse/convert to a full checkpoint when the last ubatch contained another
   sequence's non-verify (prefill) batch, or pin the rollback into the batch's own snapshot. The server
   already has the `use_ckpt_tgt` fallback (`draft.size() > n_rs_seq` → full checkpoint); the missing case
   is *another sequence's prefill sharing/owning the last ubatch*. Look at the scheduler's ubatch split
   (`split_equal`) and the server's accept/reject loop together.
2. **Write the pre-batch plane for a batch a rollback can follow** (the §27 fix already does this for
   `0 < n_tokens < K`; extend it to the cross-sequence / chunked-prefill case).
3. **Never present such a rollback** (server/batching): keep a sequence that may be rolled back out of a
   ubatch with another sequence's large prefill, or force its verify into its own ubatch.
4. Explain and repair the **second, batching-level component** (`GDN_CHUNKED=0` still diverges): even with
   correct state, the concurrent schedule must not change a sequence's output.

Do **not** "fix" it with a blanket `GGML_CUDA_GDN_CHUNKED=0` run-wide disable — that is a perf regression
and does not even remove the instability.

## 9. Gates (all on the reconstructed delivery tree)

- `tools/multiseq.sh` (the #52 gate): concurrent == solo == repeated, cache on and off, 1/2/3 GPU,
  `GGML_CUDA_GDN_CHUNKED` 1 and 0;
- `test-recurrent-state-rollback` (PASS, max diff 0) and `test-recurrent-state-depth` (no new failures vs
  the current tree's baseline);
- single-sequence purity: qwen4exp `none == draft-mtp n3` and the 35B-A3B `-sm tensor -ncmoe 99` field;
- `scripts/validate-set.sh`, `scripts/gate-prefill-logits.sh`, `scripts/gate-qwen4exp-quant-coherence.sh`,
  `test-backend-ops -o MUL_MAT_ID` (931/931) / `-o FLASH_ATTN_QSA` (≥18/18);
- coherence (`Qwen3.5-4B-Q8_0`, seed 42) = `1c5d32ac537d`;
- a ≥1000-token MTP+plain perf A/B (no regression).

## 10. Pointers

- `TODO.md` #52 — the tracker item (this is its working home).
- `wip/moe-verify-fusions/FINDINGS-single-seq-rollback.md` §7 — why #52 is a separate root from #50.
- `wip/moe-verify-fusions/HANDOVER-single-seq-rollback.md` §6 — the original #52 note (its §1-§5 lead is
  refuted; §6 still holds).
- `GREEDY-PURITY.md` §27 (the rollback bound), §39 (the cache band).
- `tools/multiseq.sh` — the reproducer. `tests/test-recurrent-state-depth.cpp`,
  `tests/test-recurrent-state-rollback.cpp` — the multi-`n_rs_batch` oracles.
- `archive/work/issue-25-mtp-batch-width/`, `archive/work/gdn-rs-rollback/` — earlier rollback work.
- `AGENTS.md` — Pushing policy, WIP/promotion rules, default-on policy.

## 11. Gotchas

- **The warning is emitted once per process** (a `static` flag), so a later occurrence is silent — do not
  rely on the log alone; instrument the raw `last_ubatch_nseq_tokens` / `rs_idx` values.
- **`GGML_CUDA_DISABLE_FUSION=1` is confounded** (it also skips the `graph_optimize` alloc-deps pass);
  prefer the individual switches.
- **`GGML_FORCE_N_RS_SEQ=<n>`** (a TEMP diagnostic in the delivery) pins the snapshot depth independently
  of `--spec-type`, useful to A/B `n_rs_seq` 0 vs N with the cache armed.
- **The probe / QSA paths have a P threshold** (P < 2051 tests the dense QSA shortcut only) — any width
  probe must use P > 2051 when touching the QSA path.
- **Never push to upstream `ggml-org/llama.cpp`.** Only this repo
  (`github.com:stew675/llama-cpp-rdna-boosts`) and the personal fork
  (`git@github.com:stew675/llama.cpp.git`), and only with the maintainer's go-ahead.
- The fix is a **delivery change**: report the root cause + proposed fix to the maintainer **before** any
  `patches/`/`release.json`/docs-header change, and wait. WIP stays WIP; do not fold anything in unasked.
