# HANDOVER — 35B-A3B (`qwen35moe`) `-sm tensor -ncmoe` MTP field impurity: the internal host-staged all-reduce (TODO #53)

**For a cold session. Take this to completion.** Scope is **only** the `Qwen3.6-35B-A3B-Q8_0`
(`qwen35moe`) field impurity. This **supersedes** `HANDOVER-35b-tensor-impurity.md` for the field part:
the probe is already fixed and shipped; what remains is the field, now root-caused to the internal
all-reduce. `qwen4exp` is fixed in r40 — do not reopen it.

---

## 0. The prompt to paste into the cold session

> You are picking up a single-issue investigation in `/home/stew675/llama-cpp-rdna-boosts` (the
> rdna-boosts delivery repo for the `stew675/llama.cpp` fork). Read
> `wip/moe-verify-fusions/HANDOVER-35b-internal-ar.md` **first, in full**, then `AGENTS.md`
> ("Report findings before shipping", "Pushing policy", "WIP and promotion rules", "Default-on policy")
> and `wip/moe-verify-fusions/README.md`.
>
> The task: `Qwen3.6-35B-A3B-Q8_0` (`qwen35moe`, embedded nextn MTP) is **not bit-pure** under
> `-sm tensor -ncmoe` in the **field** (`none` vs `draft-mtp n3`). The root cause is already found: the
> **internal host-staged all-reduce** (`ggml/src/ggml-cuda/allreduce-hip.cu`) produces a different
> result from the RCCL/NCCL path; with `GGML_CUDA_ALLREDUCE=nccl` the field is **pure**. Making the
> internal AR F32 (matching RCCL's FP32 small-tensor reduction) is **not** sufficient, so it is a
> correctness/state bug in the internal pipeline, not a precision difference.
>
> Do the decisive instrument first (HANDOVER §7): for one `linear_attn_out-*` all-reduce, dump the
> **local input hash** and the **post-AR output hash** on both devices for both the `none` and `n3`
> arms at a shared token position. If inputs are equal and outputs differ, it is unambiguously the
> pipeline; fix it so the internal AR is bit-identical to RCCL. If inputs differ, the AR is a symptom
> and the split compute is the real site — say so and pivot.
>
> Constraints: the fix is a delivery change — report the root cause + fix to the maintainer **before**
> any `patches/`/`release.json`/docs-header change, and wait. WIP stays WIP. Never push to upstream
> `ggml-org/llama.cpp`; only this repo and `git@github.com:stew675/llama.cpp.git`. No parallel GPU
> benches; one arm at a time. Verify with the §10 acceptance gates.

---

## 1. Deliverable / acceptance

The 35B is bit-pure under `-sm tensor -ncmoe`:

* **field**, 2 GPU, `M=Qwen3.6/35B-A3B/Q8_0/Qwen3.6-35B-A3B-Q8_0.gguf`, `D=""` (embedded MTP),
  `NCMOE=99`, `MOE_EXPERT_CACHE_MIB=0`, `mixed30k`, `NP=128`, `--fit on`:
  `--spec-type none` vs `--spec-type draft-mtp --spec-draft-n-max 3` = **`None`**, and also n1 and n7;
* **no regression**: 35B `-sm layer` = `None`; `qwen4exp` `-sm tensor -ncmoe 48` `none == n3` = `None`
  and its probe PASS;
* the standard gates (§10) green on the reconstructed delivery tree.

## 2. Current delivery state (as of this handover)

* `release.json`: release **`v16-a55e952b8-r40`**, base `a55e952b8`, tip
  **`b8d1e2988a08a15552c8a49fee3767f9195b4f06`**, tree `1745cf9b759e890474f1bfaa130e1c46d24cbe4b`,
  `n_blocks` 16. **Do not hand-edit hashes**; regenerate with `scripts/make-release.sh`.
* This repo `main` = `7355a8d` ("block 15: never stand the MoE down-fold down when the cache table is
  un-serveable (#53)"). The personal fork's `rdna-boosts` = `b8d1e2988`. **No new release was cut** —
  the release label is still `r40` and no `WORKLOG.md` entry was added for the fold.
* **The probe is fixed and shipped.** The fix folded into **block 15** (`ggml_cuda_cache_blocks_fusion`,
  `ggml/src/ggml-cuda/ggml-cuda.cu`): the down-fold is no longer stood down when
  `moe_cache_table_serves()` is false; its own `moe_cache_redirect_fused` call site redirects a served
  table and falls back to the raw scheduler-copied table, and that fallback is bit-identical to the
  per-op chain. Blocks 00–14 are byte-identical; only `patches/0015` and `rdna-boosts-all.patch`
  changed. Probe: 35B `-sm tensor -ncmoe 99` = **PASS**, `W=1..8` all `9577d960c75cb4bb`.
* The **field** is the open item and is the subject of this handover.

## 3. The bug (field)

`none` vs `draft-mtp n3`, 2 GPU, `-sm tensor -ncmoe`:

* long field (`mixed30k`, NP=128, `--fit on`): **first-diff 8**;
* short reproducer (`probe9k.txt`, NP=64, CTX 8192, UB 2048): **first-diff 10**;
* `-sm layer` is `None`; cache off or on both diverge; `Q4_K_M` diverges too (token 54).

## 4. Root cause (found)

The 35B's `ffn_down_exps` is `[512, 2048, 256]` and the GDN `ssm_out` / MTP outputs are split on the
**reduction dim**, so those ops need a **cross-device all-reduce**. Those tensors are all "small"
(`ne < 131072`), so by design they go through the **internal host-staged AR pipeline**
(`ggml/src/ggml-cuda/allreduce-hip.cu`), not RCCL.

`GGML_CUDA_ALLREDUCE=nccl` (RCCL for every size) makes the field **pure** (verified on both the short
reproducer and the real long field). The default 2-GPU mode ("hybrid": internal for small tensors,
NCCL for large) is impure. The internal AR's result **differs from RCCL by a token flip**, not a
rounding: `none` decodes `220` with the internal AR and `258` with RCCL, on the very first token.

## 5. Evidence matrix (all measured this session)

| config | field `none` vs `n3` |
|---|---|
| `GGML_CUDA_ALLREDUCE` default (hybrid) | token 8 (long) / 10 (short) |
| `ce` | 8 / 10 |
| `internal` | 1 (short) |
| **`nccl`** | **None** (short **and** long) |
| `GGML_CUDA_AR_BF16_THRESHOLD=0` (internal F32, all sizes) | 1 |
| `GGML_CUDA_AR_BF16_THRESHOLD=131072` (small internal F32, large unchanged) | 1 |
| `GGML_CUDA_AR_BF16_THRESHOLD=20000` | 1 |
| `GGML_CUDA_AR_COPY_THRESHOLD=1` (force copy-engine for all sizes) | 10 |
| `GGML_CUDA_AR_COPY_THRESHOLD=999999999` (force chunked kernel) | 10 |
| `--fit off` | None |
| `GGML_META_SPLIT_COPY=0` (mirror host-expert copies) | None |
| `-sm layer` | None |

**AR call trace** (`GGML_AR_DBG`, short field, `n3`): the failing ARs are
`linear_attn_out-<il>` (the GDN output projection) at `ne=8192`/`2048`/`4096`, and
`mtp_ffn_moe_out-40` / `mtp_attn_out-40` at `ne=2048`; every one is
`small=1 pipeline=1` (the internal host-staged path). `a_pipeline` is non-null in every
default/hybrid/`ce`/`internal` run.

**Precision is not the cause**: with `GGML_CUDA_AR_BF16_THRESHOLD=0` the ARHIP trace shows
`use_bf16=0` for every call (genuine F32), and the field is still impure. So the internal pipeline has
a bug beyond the F32→BF16 round-trip.

**Perf** (long field MTP, from the runs above): default internal prefill 2756.7 t/s / decode 37.32 t/s;
`nccl` prefill 2888.6 / decode 35.24. RCCL is faster on prefill, ~6 % slower on decode — nearly a wash.

## 6. Refuted / ruled out (do not re-derive)

* **Embedded vs separate MTP head.** Extracting `blk.40.*` into a separate GGUF and running the target
  with `--spec-draft-model` produces the **identical** first-diff 10 (same acceptance 0.54286). The
  embedded-MTP/shared-model hypothesis is **refuted**. (Split files kept at `/home/stew675/mtp-split/`,
  ~38 GB, optional.)
* **Quantization.** qwen4exp is pure under the *exact* 35B config (cache off, `-ncmoe 99`, short
  prompt) for both GSQ-IQ3_XXS and Q4_K_M. The 35B is impure for both Q8_0 and Q4_K_M. Not quant.
* **Config.** qwen4exp cache-off + `-ncmoe 99` is pure → the config is exhaustive.
* **Split-state cache.** `GGML_META_SS_VERIFY=1` does not abort on the 35B MTP arm.
* **`n_outputs_max` / `n_rs_seq` / `embeddings_nextn` / `n_rs_batch`.** Forcing each (or all) on the
  `none` arm does not change it; forcing the MTP graph shape on `none` does not reproduce the drift.
* **`GGML_CUDA_DISABLE_FUSION` / `DISABLE_MWR` / `DISABLE_MOE_DOWN_FOLD` / `DISABLE_MOE_DOWN_FOLD` /
  `DISABLE_ROPE_SET_ROWS` / `GDN_CHUNKED=0` / `SCHED_STAGE=0` / `DISABLE_GRAPHS=1` /
  `COMPUTE_BUFFER_MARGIN_PCT=25` / `DROP_COMPUTE_BUFFERS=0` / `ROCBLAS_USE_HIPBLASLT=0` / `ce`**: none
  make it pure (the fusion ones only move it).
* **Staging / copy path**: both the forced copy-engine and forced chunked internal sub-paths are
  impure (token 10), so it is the pipeline as a whole, not one sub-path.
* **BF16 round-trip alone**: disproven (see §5).

## 7. The decisive instrument (do this first)

Goal: decide whether the internal AR **arithmetic/state** is wrong, or its **inputs** (the split
compute) differ.

For one tensor the arms share — `linear_attn_out-<il>` (the GDN output projection) is ideal — dump,
on **both** devices, for the **`none`** and **`n3`** arms:

1. `in_hash`: FNV/xxhash of `tensors[i]->data[0 .. min(ne,64)] * sizeof(float)` **before** the AR runs
   (the tensor is still the local split partial);
2. `out_hash`: the same **after** the AR has completed (sync the stream(s) first);
3. plus `name`, `ne`, `device`, and a per-name call counter.

The first ~10 generated tokens are byte-identical between the arms, so the same `(layer, token
position)` AR must see equal `in_hash`; if `in_hash` differs, the AR is a symptom and the split
compute is the site. If `in_hash` is equal and `out_hash` differs, it is the pipeline.

Implementation notes:
* Put the dump in `ggml/src/ggml-cuda/allreduce-hip.cu` inside/around `ggml_cuda_ar_allreduce`
  (line ~1300) **or** in its caller `ggml_backend_cuda_comm_allreduce_internal`
  (`ggml/src/ggml-cuda/ggml-cuda.cu`, ~line 2538) which has the `backends[]` and can sync.
* The AR is **in-place** on `tensors[i]->data` (partial → sum), and it is async: for `out_hash` you
  must `hipStreamSynchronize` (the chunked path runs on the caller's compute stream; the copy-engine
  path uses the AR stream + events). Debug syncing is fine for a handful of calls.
* **Mutex-guard** prints/state; `ggml_cuda_try_fuse`/AR run one thread per device (see §9 gotcha).
* Env-gate (`GGML_AR3_DBG`), and cap the number of dumps per name.
* To align the two arms, filter by name and by `ne` (the decode widths are 2048/4096/8192) and compare
  the first few calls after the prefill; the first 10 generated tokens match, so the decode AR
  sequence for that prefix must match.

If the pipeline is at fault, the likely bug is in the host-staged slot/arrival handshake:
`GGML_CUDA_AR_POOL_SIZE = 2`, `ggml_cuda_ar_acquire_slot`, the per-block arrival token
(`ggml_cuda_ar_signal_set/get`), `__threadfence_system`, and the `host_buf`/`dev_tmp` reuse across the
two graphs (the `none` and `n3` graphs have different AR counts/sizes, so a missing sync shows up as a
stale peer read). The copy-engine path's `dev_tmp_kernel_done` / `host_large_read_done` events and the
chunked path's `ev.ker` are the places to audit.

**Fix target**: the internal AR must be bit-identical to the RCCL path for the tensors it serves
(i.e. `out = local + peer`, exactly), graph-shape-invariant. Repair `allreduce-hip.cu`; do **not**
just route through RCCL unless the maintainer approves the ~6 % decode cost (see §8).

## 8. Fallback (only with maintainer approval)

`GGML_CUDA_ALLREDUCE=nccl` (or an equivalent "RCCL for the GDN/MTP small-tensor ARs") is verified pure
at a near-wash perf cost (prefill +4.8 %, decode −5.6 %, long field MTP). Do not ship this as the fix
without the maintainer's go-ahead — the internal AR exists for a reason (block 12, 3-GPU paths).

## 9. Reproducers, tooling, gotchas

Build (scratch worktree `/tmp/rdna-fold15`, detached at `b8d1e2988`, **has session instrumentation
uncommitted** — reset it first: `cd /tmp/rdna-fold15 && git checkout -f b8d1e2988`):

```bash
cd /tmp/rdna-fold15 && BUILD_DIR=build-rocm-hybrid JOBS=16 ~/bin/build-llama-rocm-714   # ~7 min, ccache
export LD_LIBRARY_PATH=/opt/rocm-7.14.1-gfx120X/lib
```

Short reproducer (fast, ~1-2 min/arm):

```bash
cd /home/stew675/llama-cpp-rdna-boosts/wip/moe-verify-fusions/tools
export M=/llm/models/Qwen3.6/35B-A3B/Q8_0/Qwen3.6-35B-A3B-Q8_0.gguf D="" \
       PROMPT=/tmp/srr/probe9k.txt NP=64 NCMOE=99 CTX=8192 UB=2048 GPUS=0,1 \
       MOE_EXPERT_CACHE_MIB=0 LD_LIBRARY_PATH=/opt/rocm-7.14.1-gfx120X/lib
B=/tmp/rdna-fold15/build-rocm-hybrid
./arm.sh $B t_none -- --spec-type none
./arm.sh $B t_n3   -- --spec-type draft-mtp --spec-draft-n-max 3
# diff /tmp/srr/t_{none,n3}.tokens  -> first-diff 10
```

Long acceptance field (the `#53` config, ~7 min/arm): same but
`PROMPT=/tmp/srr/mixed30k.txt NP=128 CTX=40960 UB=6144`; first-diff 8.

Pure reference: prefix every arm with `GGML_CUDA_ALLREDUCE=nccl` → `None`.

Useful envs: `GGML_CUDA_ALLREDUCE` (`hybrid` default | `nccl` | `internal` | `ce` | `none`),
`GGML_CUDA_AR_BF16_THRESHOLD` (default 1), `GGML_CUDA_AR_COPY_THRESHOLD` (default 1 MB),
`GGML_CUDA_AR_FUSED` (default 0), `GGML_CUDA_AR_PROFILE`, `GGML_CUDA_AR_*` (chunk/sleep/spin),
`GGML_OP_OFFLOAD_MIN_BATCH`, `GGML_META_SPLIT_COPY`, `LLAMA_DROP_COMPUTE_BUFFERS`,
`GGML_COMPUTE_BUFFER_MARGIN_PCT`. `ENVIRONMENT.md` documents them.

Gotchas:
* **Never** use an unsynchronized `static std::map` dedup in `try_fuse`/AR traces — one thread per
  device; it hangs/segfaults. Mutex (or thread-local/atomic) guard.
* Run arms under `timeout`, stderr to a file. A hung probe shows 0 % GPU (`rocm-smi --showuse`).
* `pkill -f test-logits-width-probe` kills your shell; use
  `ps -eo pid,comm | awk '/test-logits-wid/{print $1}' | xargs -r kill -9`.
* `git checkout -f <sha>` (not plain) before trusting a scratch build.
* One arm at a time (no parallel GPU runs).
* The 35B load is ~30 s; the 2-shard qwen4exp ~1-2 min.

## 10. Gates (all on the reconstructed delivery tree)

* `scripts/validate-set.sh` (strict `git am`, applied tree == `release.json.tree`);
* `scripts/gate-prefill-logits.sh` (KLD ≤ 0.005, same-top-p ≥ 98 %) and
  `scripts/gate-qwen4exp-quant-coherence.sh`;
* `test-backend-ops -o MUL_MAT_ID` (931/931) and `-o FLASH_ATTN_QSA` (≥18/18);
* `test-recurrent-state-rollback`;
* coherence (`Qwen3.5-4B-Q8_0`, seed 42) and the 35B `-sm tensor -ncmoe 99` coherence;
* the §1 field acceptance (`none` == n1 == n3 == n7 = `None`) and the 35B `-sm layer` reference;
* a ≥1000-token MTP + plain perf A/B (no regression).

## 11. Pointers

* `TODO.md` #53 (this item), #50/#51/#52 (separate).
* `wip/moe-verify-fusions/HANDOVER-35b-tensor-impurity.md` — the earlier handover (probe part, done).
* `wip/moe-verify-fusions/FINDINGS-single-seq-rollback.md`, `FINDINGS-width-divergence.md` — context.
* `ggml/src/ggml-cuda/allreduce-hip.cu` — the internal host-staged AR (the suspect).
* `ggml/src/ggml-cuda/ggml-cuda.cu` — `ggml_backend_cuda_comm_is_small` (~2392), the RCCL small
  (`ncclFloat`) / large (`ncclBfloat16`) paths (~2465 / ~2512), the hybrid dispatch (~2985).
* `/home/stew675/mtp-split/` — the 35B split into no-nextn target + separate nextn draft (refutation of
  the embedded-MTP hypothesis; optional, ~38 GB).
* `benchmarks/mtp-adaptive-methodology.md`, `benchmarks/prefill-logit-methodology.md` — the gates.
