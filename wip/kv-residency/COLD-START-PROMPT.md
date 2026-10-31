# Cold-session prompt — `wip/kv-residency` (TODO #118)

Paste the block below into a fresh session.  It assumes only what is in the repo (nothing from the previous
session's context).  **Rewritten 2026-10-31 at the end of session A12**: the branch is rebased onto r186, a
delivery regression the rebase exposed is fixed, and the indexer chain's root cause is identified with a plan.

---

Continue the windowed-KV campaign (TODO **#118**) in `~/llama-cpp-rdna-boosts` and `~/llama.cpp`.

**Goal A is SHIPPED — do not re-measure it and do not re-promote it.**  Release `v16-6050b467c-r186` (16 blocks)
carries the windowed KV as **block 15**: an **opt-in** fixed-VRAM resident window with the KV's tail in pinned
system RAM (`LLAMA_KV_RESIDENT_CELLS=<cells>|auto`; unset = byte-identical to r177 including its OOM).
Shipped description: `patches/README.md` "Block 15", `ENVIRONMENT.md` §17, `WORKLOG.md` 261030.002/.003/.005,
`A2-MIXED-BACKING.md` §3.7.11 + §3.7.12.  **A window across a `-sm tensor` split is designed only and parked**
(`wip/kv-residency/A6-TENSOR-SPLIT-WINDOW.md`) — the maintainer picks it up on the multi-GPU host.

**Goal B's B-A (the attendance pool for the attention KV) is BUILT AND GATED — do not rebuild it, do not
re-derive its design, and do not re-run its gates.**  Read [`BA-POOL-DESIGN.md`](BA-POOL-DESIGN.md) **§7** (what
was built, every gate, the three defects), **§8** (A11: the acceptance table at three depths, the sparse arm's
census, the MTP-draft re-measure) and **§9** (A12: the rebase, the r183 copy regression and its fix, and the
q8_0 fused-path gate) before touching anything.

* Branch **`wip-kv-residency-b`**, tip **`f8e47e392`**, **rebased onto r186**: `rebase-r186` is the 16 blocks
  re-cut from `patches/` at `release.json.base` (tip `8f9dad0be`, tree == `release.json`); the pre-rebase tip is
  kept as branch **`wip-kv-residency-b-r182`** (`b5d895c59`).  Both pushed.
* **`LLAMA_KV_REGION_POOL=<cells>`** (unset/`0` = off, byte-identical): the attention KV becomes a pinned host
  **MIRROR** (authoritative; every existing path runs on it unchanged) plus a compact **DEVICE POOL** of pages of
  one indexer block (4 cells), a page table `tbl[block] -> slot`, and **`GGML_OP_KV_POOL`** (mark / claim /
  map+fill, 3 launches per layer per step).  Trunk-only (the MTP draft's cache is not pooled), one stream,
  non-MLA, no window active; otherwise it declines with one message.
* **Three invariants** (each cost a session — keep them): a claim may never take a slot this launch marked
  (mark-then-claim); claims are per **distinct block** (the table CAS dedupes); and **no transient scratch in a
  captured graph** (the fill list lives in the counters tensor).

**The measured state (use these; they are the current numbers).**  Flash-Next GSQ-IQ3_XXS, 1 GPU, `-ncmoe 48`,
q8_0 KV, PLE flags, `ROCBLAS_USE_HIPBLASLT=0`, static MTP depth 4, **`-n 512`**, arms interleaved:

| depth (`-c`) | resident dense | resident sparse | **pool** (204.4 MiB) | tokens/step (dense / sparse) |
|---|---|---|---|---|
| 5 298 (`16384`) | **70.94** | 70.65 (−0.4 %) | **70.30** (−0.9 %) | 2.65 / 2.78 |
| 102 899 (`131072`) | **58.63** | 54.26 (−7.0 %) | **54.52** (−7.0 %) | 2.83 / 2.84 |
| 180 035 (`196608`) | **48.73** | 42.93 (−11.9 %) | **43.09** (−11.6 %) | 2.75 / 2.71 |

(The 103k row is **post-fix** — §9.2.1.  On the rebased tree before the fix it read 58.92 / 50.42 / 50.63, and
that −14 % was a *delivery regression*, not the arm.  The 180k row is pre-fix and must be re-measured on the
current build before it is quoted again; expect ~+1.3 t/s on its sparse/pool arms.)

* **The pool lands ON the sparse ceiling at every depth** (+0.35…+0.6 % = noise): placement is free, and what
  remains is the sparse decode arm's own cost.  Prefill: neutral at 5.3k, −5 % (103k) / −8 % (180k) — the mirror
  read over the link.  Pool hit rate 99.25 / 92.5 / 93.4 % of ids; pinned mirror 0.2 / 1.6 / 2.4 GiB.
* **A10's "−20 %" was the `-n 128` measurement window**, not placement: inside 128 tokens the two arms accept
  55 % vs 37 % because they generate different text, and t/s is (tokens/step) × (step rate).  **Rule (HANDOVER
  §5 rule 9): match the acceptance before reading a t/s delta, and use `-n 512`.**
* **The sparse arm's per-step attribution** (rocprofv3, MTP-4 at 103k; §8.4): indexer score chain **+6.7 ms**,
  radix top-k +0.9, **FA −5.8**, pool resolve **+0.3**, copies +2.8, text/expert drift +3.4.  **The pool's own
  machinery is 0.10-0.11 ms/token and its fills 0.03-0.05 ms/step: the page policy (step 3) is dead.**
* **Step 3's other half is closed:** `LLAMA_MTP_SPARSE_DECODE=1` under a pooled target is **−7.8 %** with
  identical text and acceptance, so that knob's default-OFF stands.
* **A12 (§9): the chain's root cause is a quantization gate.**  With a **quantized indexer key cache**
  (`-ctk q8_0`, the delivery's large-context type) the fused `INDEXER_SCORE` + derived `INDEXER_FILL` path is
  skipped wholesale (`idx_key_float` in `qwen4exp.cpp`; the log says *"derived indexer cache pool skipped (q8_0
  keys, derived cache enabled)"*) and the per-op chain re-materialises the whole indexer key cache in F32 every
  step.  With float keys (the bf16 arm) the fused path gives plain decode **27.0 → 17.8 ms/token** of GPU
  kernels (**+23.5 %** wall, 29.3 → 36.2 t/s) and MTP-4 15.3 → 14.2 (flat on the wall: the 2× KV bytes cost the
  arena).  **So the item is "let the fused scorer read quantized keys", not "tune the chain".**

**THIS SESSION'S JOB, in order (the maintainer has agreed with all of the above):**

1. **Ship / re-home the A12 copy fix (a DELIVERY fix, not campaign code).**  `cpy_2d_vec16` in
   `ggml/src/ggml-cuda/cpy.cu` (commit `f8e47e392` on the campaign branch) belongs in **block 10**, because it
   repairs a regression r183 introduced for qwen4exp + quantized KV + the sparse arm — which block 15's windowed
   KV takes by construction.  It is measured and bit-identical (§9.2.1); the maintainer's call is *where* it
   lands (block-10 amendment + `WORKLOG` entry + `patches/README.md` note, or a separate micro-block).
2. **A13-1: a quantized-key load path for `indexer_score_kernel` / `indexer_fill_kernel`** (§9.4) so the fused +
   derived path serves `-ctk q8_0` — the delivery's own configuration and the campaign's.  Use the same
   dequantize the chain's `get_rows` uses (`dequantize_q8_0`), keep `indexer-score.cu`'s **byte-toggle contract**
   (a pool row must equal what the score pools from the raw cache), and note the derived fill's range leaf
   advances only at `n_tokens == 1` (`llama-memory-hybrid-idx.cpp`), so an MTP verify batch fills nothing.
   Gates: the `GGML_CUDA_QSA_INDEXER_CACHE=0` per-op chain's text vs the fused arm's (they must agree — that is
   the toggle contract), prefill-logit `--ab "<env>" --ub 1` = 0.000000 / 100.000 %, `test-backend-ops -o
   FLASH_ATTN_QSA`, and a census pair (expect the indexer bucket 5.31 → ~2.0 ms/token plain decode).
3. **A13-2 (only if A13-1 is rejected): keep the *indexer* cache bf16 while the attention KV stays q8_0** — the
   same fused path for `O(depth)` VRAM (keys-only: 158 → 316 MiB at 103k) plus `-ctk` plumbing, `--fit` and docs.
4. **Then re-measure the 180k row (and 5.3k) on the final build**, and the promotion DECISION is the
   maintainer's (§8.6 item 4: a block in `patches/`, `ENVIRONMENT.md` §, `patches/README.md`, the pool's VRAM in
   `--fit`, a grow-on-demand mirror, the release-flow gates, default-off).

**Do not redo:** Goal A; B1/B2/B1(c); B-A's design, implementation or gates; the acceptance table's method or the
census (§8); the "is the pool exact?" question; the page policy (dead); the MTP draft's sparse-decode re-measure
(default-OFF holds); the rebase or the r183-regression hunt (§9.1-§9.2).

START BY READING, in this order: [`BA-POOL-DESIGN.md`](BA-POOL-DESIGN.md) **§9** (where the work is), then §7-§8,
then `HANDOVER.md` (**§11 = the work list**, §3 the invariants, §5 the measurement rules — rule 9 is the
`-n 512`/acceptance rule, §6 the gates), then `B1-STRATA-PLAN.md` §3.2-§3.6.

STATE (verify, don't assume):
  - **Campaign branch `wip-kv-residency-b`**, tip **`f8e47e392`** (rebased onto `rebase-r186` = `8f9dad0be`, the
    r186 delivery tree; pre-rebase tip kept as `wip-kv-residency-b-r182` = `b5d895c59`).  `wip-kv-residency`
    (`7c7769e50`) is Goal A's measurement tree — do not develop on it.  Never commit campaign code to
    `rdna-boosts`; **never touch `patches/` or `release.json`** without the maintainer's go-ahead.
  - ONE R9700 (gfx1201), 32 624 MiB.  **Check `rocm-smi --showmeminfo vram` first** — the maintainer runs a
    Flash-Next server here that holds ~34 GB, and every fit/gate then reads garbage (`free = 2 MiB`) or OOMs.
    Never run a second GPU job; if the card is in use, say so and stop.
  - Link 55.7 GB/s H2D.  **`/tmp` is a 47 GB tmpfs** (every byte is RAM): keep the working set small, move a
    prefill-logit base (`~7.6 GB`) to `/var/tmp/kv-gate/` as A12 did, and delete rocprof `*_kernel_trace.csv`
    files after analysis (only the `*_kernel_stats.csv` inputs matter).
  - **Push the campaign branch after every step** — these arms can OOM this box, so the work must be durable.
  - `LLAMA_KV_POOL_STATS=<path>` appends the pool's accounting to a **file**; never print it to stderr on a
    text-hash arm.  `LLAMA_KV_POOL_VERIFY=1` is the pool's oracle (0 mismatches expected).

CONSTRAINTS:
  - Build `~/bin/build-llama-rocm` (never pass `-DCMAKE_HIP_FLAGS`); `LD_LIBRARY_PATH=/opt/rocm-10.1.0-gfx120X/lib`.
  - **Every Flash-Next arm must carry `-ot per_layer_token_embd.*=CPU -lm none -lzm off`** (§5 rule 7): the PLE
    table is 27 465 MiB and is otherwise lazy-mapped, which moved identical runs by **±20 %**.
  - **Flash-Next's MTP head is a SEPARATE file**: `-md /llm/models/Qwen3.8/Flash-Next/GSQ-IQ3_XXS/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf`.
  - **Static MTP: never `--spec-draft-n-max` > 4** (§5 rule 8).  **`-n 512`, and check that the arms' tokens/step
    match before reading a delta** (§5 rule 9).
  - Hash text from a run WITHOUT `-v` via `scripts/extract-generated.py`; timings from a separate `-v` run.
    Always `--single-turn --no-display-prompt`, always `timeout`, never two GPU jobs at once.
  - **`ROCBLAS_USE_HIPBLASLT=0` for every fresh-process numeric comparison** — see the note below.
  - Never quote a timing from a `LLAMA_KV_REGION_DEBUG` arm, nor from a `GGML_CUDA_DISABLE_GRAPHS=1` census arm.
  - A census is two `-n` lengths of the SAME arm, subtracted, filtered to kernels whose call *count* scales with
    tokens (`kv-kernel-census.py`); `rocprofv3 --kernel-trace --stats -f csv` does capture graph-replayed kernels.
  - `-n 2000` for anything about the expert arena, with `-ncmoe` = the model's full layer count.
  - Do not pick scope.  If something cannot be measured, say so and stop.

**WHY `ROCBLAS_USE_HIPBLASLT=0` IS STILL IN EVERY RECIPE (asked and answered in A12).**  It is not a stale rule:
the maintainer's bypass **shipped** — block 06's `mmf32` (r104, `archive/work/hipblaslt-bypass/`) serves the
**F32** prefill GEMM shapes that used to reach hipBLASLt, so the 35B-A3B / gemma4 F32 prefill is deterministic by
construction and +4.2 % faster.  What it deliberately did **not** cover is **BF16 prefill**, and Flash-Next's
prefill is BF16: its GEMMs still dispatch through rocBLAS → hipBLASLt, which picks a solution **per process
start** (issue #67 / `rocm-libraries#12126`, fixed upstream by `rocm-libraries#11340`, which no ROCm release we
run contains).  Every text gate in this campaign is Flash-Next, so the flag stays.  (It also means this
campaign's *absolute* prefill numbers are conservative by up to ~1.5x — hipBLASLt is worth ~+51 % prefill on
Flash-Next — while every *delta* between arms is unaffected.)

GATES: `GGML_CUDA_VMM_REGION_SELFTEST=1` (PASS, and FAIL with `LLAMA_KV_REGION_QUARANTINE=0`) if you touch the
VMM primitive (then `hostmap-hazard-probe.cc` too: default FAIL, `--grab 4` PASS); the coherence trio
(`5ba44647188f`, `ROCBLAS_USE_HIPBLASLT=0`); the prefill-logit gate on a **spilling** arm (−0.000000 / 100.000 %);
`check-gpu-faults.sh` over every log; the OOM baseline with the knob unset.  For anything that touches the pool:
the resident-arm text identity, the pool arm == the resident **sparse** hash, `test-backend-ops -o FLASH_ATTN_QSA`,
and the three state/park gates.  **A change to the indexer score / derived cache / a copy kernel owes the
prefill-logit gate at `--ub 1`** (0.000000 / 100.000 %), the text identity of every arm it touches (at 5.3k **and**
at a deep pool arm — 51 475 depth `854 43b30de151b9` is the campaign's deep-text reference), and a census.
