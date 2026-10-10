# HANDOVER — 35B-A3B (`qwen35moe`) `-sm tensor` + `-ncmoe` MTP impurity (issue #53)

**For a cold session. Take this to completion.**  Scope is **only** the Qwen3.6-35B-A3B Q8_0
(`qwen35moe`) `-sm tensor` divergence.  **qwen4exp is already fixed in r40 — do not reopen it** (it was
the devmap / hipBLASLt flip, now removed; its r40 gate is `none == draft-mtp n3` = `None`).

Current delivery: **r40** — `release.json` base `a55e952b8`, tip
`fdecb4d337b2e945e204e7217873a1eeea20d361`, tree `73c371a2f453489dccdcea73ed22216b27cb9a28`
(14 files devmap-free; see `WORKLOG.md` 2026-10-10 (r40)).  The personal fork's `rdna-boosts` is at that
tip.

---

## 1. The bug

`Qwen3.6-35B-A3B-Q8_0.gguf` under `-sm tensor` + `-ncmoe`, 2 GPUs, embedded `qwen35moe.nextn` MTP head
(so **no** `--spec-draft-model`):

* field: `--spec-type none` vs `--spec-type draft-mtp --spec-draft-n-max 3` → **first-diff 8**;
* the same model under **`-sm layer` is bit-identical (`None`)** — that is the reference arm.

It is **cache-independent** (`MOE_EXPERT_CACHE_MIB=0` still diverges) and it is **not** the MTP machinery
per se: the plain **forward pass** is width-impure under `-sm tensor` + `-ncmoe` (the probe below has no
MTP and no cache).  The `-sm tensor` Meta split of the **host-resident** expert weights is required:
`GGML_META_SPLIT_COPY=0` (mirrored host experts) is pure, and 1 GPU / all-GPU / `-sm layer` are pure.

## 2. Model / field config

```
M=/llm/models/Qwen3.6/35B-A3B/Q8_0/Qwen3.6-35B-A3B-Q8_0.gguf      # 37.8 GB, embedded MTP
D=                                                                 # empty = embedded MTP head
PROMPT=/tmp/srr/mixed30k.txt                                       # sha256 69624f4d...
-sm tensor -ncmoe 99 -ub 6144 -b 6144 -c 40960 --no-kv-unified -ctk q8_0 -ctv q8_0 -fa on -t 8 --fit on
GPUS=0,1  MOE_EXPERT_CACHE_MIB=0   NP=128   GGML_CUDA_ALLREDUCE=ce
```
Geometry (from the trace): `n_embd` 2048, MoE FF 256, `n_expert_used` 8, ~40 MoE layers × 2 devices.
Q8_0 experts.  The CUDA sub-graphs at width W (2..8):

```
[MUL_MAT_ID[256,8,W], MUL_MAT_ID[256,8,W], GLU[256,8,W], MUL_MAT_ID[2048,8,W], MUL[2048,8,W], VIEW...]
```

## 3. Reproducers (both fast and deterministic on the r40 build)

### 3a. Width probe — the primary instrument (~1-2 min, no MTP, no cache)

```
cd /tmp/rdna-fold
export LD_LIBRARY_PATH=/opt/rocm-7.14.1-gfx120X/lib HIP_VISIBLE_DEVICES=0,1 NCMOE=99 SPLIT=tensor
./build-rocm-hybrid/bin/test-logits-width-probe \
  /llm/models/Qwen3.6/35B-A3B/Q8_0/Qwen3.6-35B-A3B-Q8_0.gguf /tmp/srr/probe9k.txt 256 512
```
Usage: `<model> <text> [P=256] [ubatch=512]`; env `NCMOE`, `SPLIT=tensor|layer|row`, `KV`, `CACHE`,
`RS`, `FA`.  It prefills `P` tokens, then decodes a W-token batch for W=1..8 and requires row 0 (and every
shared row) to hash identically across W.  `probe9k.txt` is 3130 tokens; the file must be < 8192 tokens
(the tokenizer buffer).

Clean r40 result: `width_purity=FAIL (worst maxdiff 0.888169)`, W=1 hash `705d290898cf0699`, W=2..8
`9577d960c75cb4bb`.

### 3b. Field arm (the acceptance test)

```
cd /home/stew675/llama-cpp-rdna-boosts/wip/moe-verify-fusions/tools
export M=/llm/models/Qwen3.6/35B-A3B/Q8_0/Qwen3.6-35B-A3B-Q8_0.gguf D="" \
       PROMPT=/tmp/srr/mixed30k.txt NP=128 NCMOE=99 CTX=40960 UB=6144 GPUS=0,1 \
       MOE_EXPERT_CACHE_MIB=0 LD_LIBRARY_PATH=/opt/rocm-7.14.1-gfx120X/lib
B=/tmp/rdna-fold/build-rocm-hybrid
./arm.sh $B t35_none -- --spec-type none
./arm.sh $B t35_n3   -- --spec-type draft-mtp --spec-draft-n-max 3
# then diff /tmp/srr/t35_{none,n3}.tokens
```

## 4. Evidence matrix

| A/B | result |
|---|---|
| probe: `-sm tensor` + `-ncmoe 99` (2 GPU) | **FAIL** (0.888169) |
| probe: `-sm layer` + `-ncmoe 99` | PASS |
| probe: `-sm tensor`, all-GPU (no `-ncmoe`) | PASS |
| probe: 1 GPU, `-ncmoe 99` (tensor or layer) | PASS |
| probe: `-sm tensor -ncmoe 99`, `GGML_META_SPLIT_COPY=0` (mirrored) | PASS |
| probe: `GGML_CUDA_DISABLE_MWR=1` | **PASS** (all W = `ac5b55f2e14b9cdb`) |
| probe: `GGML_CUDA_DISABLE_MOE_DOWN_FOLD=1` | **PASS** |
| probe: `GGML_CUDA_DISABLE_SHEXP_DOWN_GATE=1` | FAIL, reduced (0.172252) |
| probe: `DISABLE_FUSION=1` | PASS (instrumented build; re-confirm) |
| probe: `DISABLE_MMVQ_MOE_BAND` / `MOE_MMQ_FUSION` / `MMQ_ROUTED` / `WEIGHTED_DOWN` | FAIL, unchanged |
| probe: `FUSE_GLU_Q8_1=0` / `DISABLE_NORM_Q8_1` / `DISABLE_NORM_ROWS` / `FORCE_MMQ` / `MMQ_PREC=q8` / `DISABLE_MMID_512` | FAIL, unchanged |
| field `-sm tensor`: default | **first-diff 8**, acc 0.79464 |
| field `-sm tensor`: `DISABLE_MWR=1` | **first-diff 8**, acc 0.84259 |
| field `-sm tensor`: `DISABLE_FUSION=1` | **first-diff 72**, acc 0.81081 |
| field `-sm layer` | **None**, acc 0.74359 |

**Read this carefully:** the probe impurity is *fusion-selected* (MWR or the down fold off ⇒ PASS), but the
**field** divergence is **not** fixed by `DISABLE_MWR` and only *moves* (8 → 72) with `DISABLE_FUSION=1` —
the same "fusions off only relocate it" signature the qwen4exp hunt saw.  So expect **two overlapping
`-sm tensor` + `-ncmoe` effects**: a fusion-selected width impurity (probe) and a broader Meta-split
arithmetic drift (field).  A fix must make the field `None`, not just the probe PASS.

## 5. Structural findings (from `try_fuse` instrumentation, instrumented build)

* **W=1:** there is **no `MUL_MAT_ID` node in any CUDA sub-graph** — the routed-down MoE runs *off-GPU*.
  The `MUL[2048,8,1]` + 8 `VIEW[2048,1,1]` + 7 left-fold `ADD[2048,1,1]` nodes are on the GPU and the
  **`moe_weighted_reduction` (MWR)** fusion matches them (`node_count == 16 = 2*n_expert_used + mul_count - 1`).
* **W≥2:** the down `MUL_MAT_ID` is in the CUDA graph and the **mmvq down fold** (`x_scale_channel_dst`
  epilogue, `ggml-cuda.cu` `ggml_cuda_try_fuse`) consumes `[MUL_MAT_ID, MUL]`, so MWR cannot match.
* So the **width selects the reduction implementation**: MWR (W=1, and prefill) vs down fold + ADD tree
  (W=2..8).  `GGML_CUDA_DISABLE_MWR` and `GGML_CUDA_DISABLE_MOE_DOWN_FOLD` each force one implementation at
  all widths and make the probe PASS.
* **qwen4exp has the identical structure** (MWR `count=20` at W=1, down fold at W≥2) and is pure — so the
  structure alone is not the bug; something about the 35B's shapes/quant makes the two paths disagree.
* MWR (`ggml/src/ggml-cuda/moe-weighted-reduction.cu`, `ggml_cuda_op_moe_weighted_reduction`) reads the
  **already-computed** `experts` F32 `[n_embd, n_expert_used, n_tokens]` and applies
  `sum = e0*w0; sum += e_i*w_i` — a strict **left fold**, matching the graph's ADD tree by construction.
  So if MWR and the ADD tree genuinely differ, the difference is *not* the fold order; it is more likely
  the values (which expert tensor / weights are read, or a plan-dependent producer).
* MWR is **not covered by any gate**: `test-backend-ops -o MUL_MAT_ID_FUSION` (28/28 PASS) builds only
  `MUL_MAT_ID (+ MUL)` — it never builds the `VIEW×n + ADD×(n-1)` reduction pattern, so the MWR path is
  untested for bit-identity.  The mmvq down fold *is* covered there.

## 6. Ruled out

* The expert cache (`MOE_EXPERT_CACHE_MIB=0` still diverges) and the cache arena.
* The MTP/draft machinery *as the forward-pass cause* (the probe shows the plain forward is impure).
* The chunked GDN, the sparse QSA/indexer, the shared-expert down gate alone (reduces, does not fix).
* mmvq MoE band, MMQ routed dispatch, MMQ fusion, weighted-down, GLU→Q8_1, norm→Q8_1, `MMQ_PREC`.
* Any single GPU / layer-split / all-GPU / mirrored configuration (all pure).

## 7. Suggested plan

1. Rebuild a scratch tree at the r40 tip and reconfirm §3a/§3b (≈10 min).  Use the **probe** as the loop
   (1-2 min); only run the 4-arm field A/B when the probe is green.
2. **Find the first op whose values differ between a W=1 decode and a W=4 verify** on the same
   `-sm tensor -ncmoe` context (dump per-node outputs; the same-seed `llama-cli` diff is too coarse).  A
   per-op address/plan dump (the r25 §41 canonical dump) plus per-op output hashes will localise it.
3. Compare the `-sm tensor` vs `-sm layer` per-op execution for the same input — `-sm layer` is the pure
   reference, so the first divergence between the two split modes is the bug site.
4. Prime suspects, in order:
   * the Meta backend's per-op handling of a **split host expert** (`ggml-backend-meta.cpp`, the
     `GGML_META_SPLIT_COPY` splice in `llama-model.cpp`, `set_tensor_2d`, the H2D staging ring / device
     gather) — is any of it width-gated or sharing a buffer across widths?
   * the **compute-buffer plan**: the probe uses one context across W=1..8, so the W=1 graph's fusion set
     (MWR) can fix a plan that later widths reuse.  Test by running W=4 **before** W=1 and by running each
     width in a fresh context; if the impurity follows the plan, make the fusion set width-independent.
   * whether MWR should be allowed to match when the same expert tensor would be served by the down fold
     (i.e. make the choice width-independent by construction).
5. Fix, then re-gate (see §8).  Report the root cause + fix to the maintainer **before** any
   delivery-affecting commit (AGENTS.md "Report findings before shipping").

## 8. Acceptance / gates

* probe (35B, `-sm tensor -ncmoe 99`, 2 GPU): **PASS**, W=1..8 identical.
* field (35B, `-sm tensor -ncmoe 99`): `none` vs `draft-mtp n3` = **None**; also n1 and n7.
* no regression: 35B `-sm layer` = `None`; **qwen4exp** (`-sm tensor -ncmoe 48`, 2 GPU) `none == n3` =
  `None` and its probe PASS.
* then the standard release gates on the reconstructed delivery tree: `scripts/validate-set.sh`,
  `scripts/gate-prefill-logits.sh` (KLD ≤ 0.005, top-p ≥ 98 %), `test-backend-ops -o MUL_MAT_ID`
  (931/931) and `-o FLASH_ATTN_QSA` (≥18/18), `test-recurrent-state-rollback`, coherence, and a ≥1000-token
  MTP + plain perf A/B (no regression).

## 9. Tooling gotchas (learned the hard way this session)

* **Never instrument `try_fuse` with unsynchronized `static std::map` dedup.**  `try_fuse` runs on one
  thread per device; the raced map caused intermittent **hangs and one segfault** that looked like
  delivery bugs.  Guard traces with a `std::mutex` (or use thread-local/atomic state).
* Always run probes under `timeout` and send stderr to a file; a hung probe shows 0 % GPU and low load
  (`rocm-smi --showuse`) — kill it, do not wait.
* `pkill -f test-logits-width-probe` **kills your own shell** (the pattern matches the `bash -c` line).
  Use: `ps -eo pid,comm | awk '/test-logits-wid/{print $1}' | xargs -r kill -9` (`pgrep -x` also fails:
  the name is > 15 chars).
* Build: `cd /tmp/rdna-fold && BUILD_DIR=build-rocm-hybrid JOBS=16 ~/bin/build-llama-rocm-714`
  (~7 min with ccache; the script `rm -rf`s the build dir).  `LD_LIBRARY_PATH=/opt/rocm-7.14.1-gfx120X/lib`.
  Use `git checkout -f fold` to guarantee the scratch tree is clean before trusting a build (a plain
  `git checkout fold` carries uncommitted edits across).
* Scratch worktree: `/tmp/rdna-fold` (branch `fold`) — a worktree of `~/llama.cpp`, currently clean at
  `fdecb4d33`.  Do **not** disturb `~/llama.cpp`'s uncommitted WIP tree.
* No parallel GPU benches; one probe/arm at a time.

## 10. Pointers

* `TODO.md` #53 (this item) and #51/#52 (separate, still open).
* `wip/moe-verify-fusions/FINDINGS-width-divergence.md` — the earlier width-divergence hunt (qwen4exp; the
  probe/method this handover reuses).
* `wip/moe-verify-fusions/FINDINGS-devmap-generation-fix.md` §5 — the first 35B split observation
  (tables already split; `none` vs `n3` = 8).
* `wip/moe-verify-fusions/tools/` — `arm.sh`, `widthsweep.sh`, `perf.sh`, `probs.sh`, `multiseq.sh`.
* `tests/test-logits-width-probe.cpp` — env `NCMOE`/`SPLIT`/`CACHE`/`RS`/`FA`.
* `benchmarks/mtp-adaptive-methodology.md`, `benchmarks/prefill-logit-methodology.md` — the gates.
