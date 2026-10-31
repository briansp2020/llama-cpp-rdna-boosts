# Cold-session prompt — `wip/kv-residency` (TODO #118)

Paste the block below into a fresh session.  It assumes only what is in the repo (nothing from the previous
session's context).  Rewritten 2026-10-31 at the end of session A10 (B-A steps 1+2 built and gated; the policy and
B3 are what is left).

---

Continue the windowed-KV campaign (TODO **#118**) in `~/llama-cpp-rdna-boosts` and `~/llama.cpp`.
**This phase is Goal B: B-A, the attendance pool for the attention KV — steps 1 and 2 are BUILT and GATED; the
policy (step 3) and B3 (the arena half) are what is left.**

**Goal A is SHIPPED — do not re-measure it and do not re-promote it.**  Release `v16-6050b467c-r181` (the set is
now at r184) carries the windowed KV as **block 15**: an **opt-in** fixed-VRAM resident window with the KV's tail
in pinned system RAM (`LLAMA_KV_RESIDENT_CELLS=<cells>|auto`; unset = byte-identical to r177 including its OOM).
Shipped description: `patches/README.md` "Block 15", `ENVIRONMENT.md` §17, `WORKLOG.md` 261030.002/.003/.005,
`A2-MIXED-BACKING.md` §3.7.11 + §3.7.12.  **A window across a `-sm tensor` split is designed only and parked**
(`wip/kv-residency/A6-TENSOR-SPLIT-WINDOW.md`) — the maintainer picks it up on the multi-GPU host.

**Goal B so far (all measured on this one card; do not redo):**

* **B-A steps 1 and 2 — BUILT AND GATED (A10, 2026-10-31).**  Branch `wip-kv-residency-b`, tip **`b5d895c59`**
  (pushed).  `LLAMA_KV_REGION_POOL=<cells>`: the attention KV becomes a pinned host MIRROR (authoritative,
  cell-indexed -- every existing path works on it unchanged) + a compact DEVICE POOL of indexer-block pages +
  a page table + an in-graph resolve (`GGML_OP_KV_POOL`: mark / claim / map+fill) and a second id list for the
  QSA reader's K/V gather.  Full detail and every gate: [`BA-POOL-DESIGN.md`](BA-POOL-DESIGN.md) **§7**.  The
  headline numbers at 103k depth with MTP depth 4: resident-dense **60.2**, sparse+resident **47.9**, **pool
  48.3** (4104 pages = 16 416 cells/layer, **204 MiB**), hit rate **89.7 %**, fill cost ~0.05 ms/step.  **So the
  pool lands ON the sparse ceiling and the remaining -20 % is the SPARSE ARM, not the KV's placement** -- a
  better page policy can recover ~0.05 ms/step, no more.  **B3 (the arena half) is measured as well** (`-n 2000`,
  `-c 131072`, `-ncmoe 48`): arena 23 579.8 -> 25 018.6 MiB (**+1.44 GiB**, +18 slots/table, expert hit 96.26 ->
  97.02 %, cold reads 42.0 -> 31.6 GiB), plain decode 43.1 -> 33.0 t/s (the sparse arm's cost; `identity fast path
  ON` = 0 in both arms).  What is left: step 3 (the policy, expected to be a no-op) and the maintainer's
  promotion decision.


* **B1 — the placement matrix.**  A **resident** KV wants the **dense** decode arm (VRAM is ~10x the link, and the
  sparse arm adds the `O(depth)` indexer score + top-k); a **streamed** KV wants the **sparse** arm (+34 % at
  103k, +47 % at 180k).  **MTP x depth frames every number**: resident + MTP is 75.4 t/s at 5.3k depth, 58.2 at
  103k, 53.6 at 180k; streaming costs 3.5 % / 25 % / 32 % at those depths (`B1-STRATA-PLAN.md` §3.4).
* **B2 — SHIPPED TO THE CAMPAIGN BRANCH** (`wip-kv-residency-b`): `llama_kv_cache::is_streaming(n_kv)` +
  `qwen4exp_kv_streams()` make a windowed KV take the sparse decode arm; `LLAMA_QSA_DENSE_DECODE_STREAM=0` is the
  A/B arm.  Text is inert on a resident cache and equal to the forced-sparse arm when streaming.
* **B1(c) — the attendance census.**  `qsa-census.py` over a 48-step decode trace: **2051 attended cells per layer
  per step**, step-to-step Jaccard **0.586**; an LRU pool hits **84.5 % at 4096 cells/layer** and **90.2 % at
  8192** (2.5 MiB/token vs 25.5 with no residency), **adaptive beats a pre-calculated static ordering** (84-90 %
  vs 63-68 %, the expert-cache campaign's verdict reproduced).  **Two forms are refuted by measurement**: block
  15's region with attendance-chosen units (its 4 MiB unit = 3855 contiguous cells vs a scattered attended set →
  **6 % hit, 601 flips/step ≈ 90 ms/step**) and any static ordering.

**THIS SESSION'S JOB (was: build B-A; steps 1 and 2 are now DONE — see above and `BA-POOL-DESIGN.md` §7) — what
is left of `wip/kv-residency/BA-POOL-DESIGN.md`:**

1. ~~**The mirror + the pool + the page table as a KV cache mode**~~ — **DONE (A10)**, gated: text, state, park. (`LLAMA_KV_REGION_POOL=<cells>`, 0 = off and
   byte-identical to today): a host-mirror tensor pair per layer (pinned, authoritative), a device pool of
   `N` pages (page = **4 cells = one indexer block**), a page table `table_l[block] -> slot`, and a write path that
   lands in the mirror **always** and in the slot only when the page is resident (Strata's writer rule: no
   invalidation, ever — `seq_rm`/`seq_cp`/`defrag`/state become trivial, and an eviction costs nothing).
   **Gate:** text byte-identical with the pool on vs off on a **resident** arm, plus `gate-state-pipe.sh`,
   `gate-ckpt-reuse.sh` and `gate-park-idle-slots.sh`.
2. ~~**The resolve kernel + the id rewrite**, QSA decode path only~~ — **DONE (A10)**: `GGML_OP_KV_POOL` +
   `ggml_flash_attn_qsa_set_kv_idx`; the pool's verify kernel reports 0 mismatches, prefill-logit 0.000000.
   Original spec, QSA decode path only (prefill's union is not sparse — leave it on the
   current window or resident).  **Gate:** `test-backend-ops -o FLASH_ATTN_QSA`, then the pool arm's text equal to
   the resident **sparse** hash (`e6080aaf2318` at 51 475 depth).
3. **The policy** (lag-1 LRU first, then F-LRU + heat), gated on `qsa-census.py`'s predicted hit rate
   (>= 84 % at 4096 cells/layer) and on the four-arm timing gate (`B1-STRATA-PLAN.md` §3.3) with the pool arm
   added at 5.3k / 103k / 180k.  **Current eviction is round-robin + "skip what this launch marked", and it
   already measures 89.7 % -- the census's LRU plateau is 90.2 %, so a policy is expected to buy ~1 point of
   hits (0.05 ms/step).  Judge it against that, and note the 103k timing gate is already measured (48.3 vs
   the sparse ceiling 47.9).**
4. **B3** — the arena half at `-n 2000` with `-ncmoe` = the full layer count, recording arena MiB, residency % and
   the `identity fast path ON` count (the r117 cliff).  Then the promotion decision (the maintainer's).

**Numbers to beat** (Flash-Next GSQ-IQ3_XXS, 1 GPU, `-ncmoe 48`, q8_0, `-c 131072`, 102 899 depth, MTP depth 4):
resident **58.24**, `W=32768` (recency, 25 %) **49.58**, `W=1` + B2 **47.26** t/s.  With a pool of 8192
cells/layer (**107 MiB** of VRAM) the streamed arm should land within a few percent of resident — that is the
acceptance test.

START BY READING, in this order: `wip/kv-residency/BA-POOL-DESIGN.md` (the spec), then `HANDOVER.md` (**§11 = the
work list**, §3 the invariants, §5 the measurement rules, §6 the gates), then `B1-STRATA-PLAN.md` §3.4-§3.6 (why
the pool is the only form left), then `A2-MIXED-BACKING.md` §4.

STATE (verify, don't assume):
  - **Machine note: `git push` over the `git@` remote FAILS on this box** (no ssh-agent, no working key).
    `git push https://github.com/stew675/llama.cpp.git <branch>` works (gh's credential helper) and is the same
    personal fork.  Never push to upstream.
  - `LLAMA_KV_POOL_STATS=<path>` appends the pool's accounting to a file; do **not** print it to stderr on a
    text-hash arm (`scripts/extract-generated.py` slices from the generated text to llama-cli's perf line and
    would include it -- that cost most of A10's "nondeterminism" hunt).
  - `LLAMA_KV_POOL_VERIFY=1` is the pool's own oracle: every selected page is compared byte-for-byte with the
    mirror at read time (mismatches land in the stats' `bad` field).  Ground truth before any hash.
  - **Campaign branch `wip-kv-residency-b`** (pushed), based on the shipped tip, tip **`c045bf866`** = B2 + the
    census diagnostic.  `wip-kv-residency` (`7c7769e50`) is Goal A's measurement tree — do not develop on it.
    Never commit campaign code to `rdna-boosts`; **never touch `patches/` or `release.json`**.
  - ONE R9700 (gfx1201), 32 624 MiB.  **Check `rocm-smi --showmeminfo vram` first** — the maintainer runs a
    Flash-Next server here that holds ~34 GB, and every fit/gate then reads garbage (`free = 2 MiB`) or OOMs.
    Never run a second GPU job; if the card is in use, say so and stop.
  - Link 55.7 GB/s H2D (re-cabled 2026-10-30).  No pre-swap t/s number may be compared with a new one.
  - **Push the campaign branch after every step** — these arms can OOM this box (the maintainer lost a session
    to a 235 GiB pinned-spill plan), so the work must be durable at every point.

CONSTRAINTS:
  - Build `~/bin/build-llama-rocm` (never pass `-DCMAKE_HIP_FLAGS`); `LD_LIBRARY_PATH=/opt/rocm-10.1.0-gfx120X/lib`.
  - **Every Flash-Next arm must carry `-ot per_layer_token_embd.*=CPU -lm none -lzm off`** (HANDOVER §5 rule 7):
    the PLE table is 27 465 MiB and is otherwise lazy-mapped, which moved identical runs by **±20 %** (with the
    flags: ±0.2 %).
  - **Static MTP: never `--spec-draft-n-max` > 4** (§5 rule 8; above 4 is almost always worse).
  - Hash text from a run WITHOUT `-v` via `scripts/extract-generated.py`; timings from a separate `-v` run.
    Always `--single-turn --no-display-prompt`, always `timeout`, never two GPU jobs at once.
  - **`ROCBLAS_USE_HIPBLASLT=0` for EVERY fresh-process numeric comparison.**
  - Never quote a timing from a `LLAMA_KV_REGION_DEBUG` arm, nor from a `GGML_CUDA_DISABLE_GRAPHS=1` census arm
    (the census *needs* graphs disabled — a readback cannot be captured — and its timings are meaningless).
  - Prove a spill / a pool before believing an arm (`LLAMA_KV_REGION_DUMP=/dev/null` → `KVREGION_FREE … host=`
    > 0), and a pool arm must show its own accounting (hits / fills per step / VRAM).
  - `/tmp` is a 47 GB tmpfs and a prefill-logit base is ~7 GB — clean old `.kld` files before a gate run.
  - `-n 2000` for anything about the expert arena, with `-ncmoe` = the model's full layer count.
  - Do not pick scope.  If something cannot be measured, say so and stop.

GATES: `GGML_CUDA_VMM_REGION_SELFTEST=1` (PASS, and FAIL with `LLAMA_KV_REGION_QUARANTINE=0`) if you touch the
VMM primitive (then `hostmap-hazard-probe.cc` too: default FAIL, `--grab 4` PASS); the coherence trio
(`5ba44647188f`, `ROCBLAS_USE_HIPBLASLT=0`); the prefill-logit gate on a **spilling** arm (−0.000000 / 100.000 %);
`check-gpu-faults.sh` over every log; the OOM baseline with the knob unset.  For B-A specifically: the resident-arm
text identity, the pool arm == the resident sparse hash, `test-backend-ops -o FLASH_ATTN_QSA`, and the three
state/park gates (the mirror must make them trivial — that is the point of the writer rule).
