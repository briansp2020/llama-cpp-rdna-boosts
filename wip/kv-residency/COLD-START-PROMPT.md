# Cold-session prompt — `wip/kv-residency` (TODO #118)

Paste the block below into a fresh session.  It assumes only what is in the repo (nothing from the previous
session's context).  **Rewritten 2026-10-31 at the end of session A10**: Goal B's B-A is **built and gated**
(steps 1, 2 and B3), so the next session finishes the acceptance table, attributes the one remaining loss, and
prepares the promotion decision.

---

Continue the windowed-KV campaign (TODO **#118**) in `~/llama-cpp-rdna-boosts` and `~/llama.cpp`.

**Goal A is SHIPPED — do not re-measure it and do not re-promote it.**  Release `v16-6050b467c-r181` (the set is
now at r184) carries the windowed KV as **block 15**: an **opt-in** fixed-VRAM resident window with the KV's tail
in pinned system RAM (`LLAMA_KV_RESIDENT_CELLS=<cells>|auto`; unset = byte-identical to r177 including its OOM).
Shipped description: `patches/README.md` "Block 15", `ENVIRONMENT.md` §17, `WORKLOG.md` 261030.002/.003/.005,
`A2-MIXED-BACKING.md` §3.7.11 + §3.7.12.  **A window across a `-sm tensor` split is designed only and parked**
(`wip/kv-residency/A6-TENSOR-SPLIT-WINDOW.md`) — the maintainer picks it up on the multi-GPU host.

**Goal B's B-A (the attendance pool for the attention KV) is BUILT AND GATED — do not rebuild it, do not
re-derive its design, and do not re-run its gates.**  Read
[`BA-POOL-DESIGN.md`](BA-POOL-DESIGN.md) **§7** (what was built, every gate, the three defects, the 103k table,
B3) before touching anything.

* Branch **`wip-kv-residency-b`**, tip **`b5d895c59`** (pushed).
* **`LLAMA_KV_REGION_POOL=<cells>`** (unset/`0` = off, byte-identical): the attention KV becomes a pinned host
  **MIRROR** (authoritative; every existing path — writers, `state_*`, the K-shift, `seq_park`, `defrag`,
  `get_k/get_v` — runs on it unchanged) plus a compact **DEVICE POOL** of pages of one indexer block (4 cells),
  a page table `tbl[block] -> slot`, and **`GGML_OP_KV_POOL`** (mark / claim / map+fill, 3 launches per layer per
  step).  `ggml_flash_attn_qsa_set_kv_idx()` gives the QSA reader a second id list, so the K/V gather follows the
  table while the mask / visibility keys stay **cell**-indexed.
* Trunk-only (the MTP draft's cache is deliberately not pooled: its attention is dense), one stream, non-MLA,
  no window active, uniform `compress_ratio`/cell geometry; otherwise it declines with one message.  The pool is
  floored at one decode band's worth of blocks and the graph re-checks the real width (`kv_pool_serves`).
* **Three invariants** (each cost a session — keep them): a claim may never take a slot this launch marked
  (mark-then-claim); claims are per **distinct block** (the table CAS dedupes — per-id attempts wrapped the
  round-robin cursor on a 4-row batch); and **no transient scratch in a captured graph** (the fill list lives in
  the counters tensor).
* Gates already green, all on the pool arm: text identical pool on/off (854 `43b30de151b9`, 51 475 depth),
  the pool's own verify kernel 0 mismatches (`LLAMA_KV_POOL_VERIFY=1`), prefill-logit 0.000000 / 100.000 %,
  `gate-park-idle-slots.sh` / `gate-ckpt-reuse.sh` / `gate-state-pipe.sh` PASS, `FLASH_ATTN_QSA` 27/27,
  `check-gpu-faults.sh` clean.

**The measured state (use these; they are the current numbers):**

| at 103k depth, MTP depth 4, `-n 128` | gen t/s | accepted |
|---|---|---|
| resident KV, dense arm (the default) | **60.2** | 87 |
| resident KV, sparse arm forced | 47.9 | 79 |
| **pool, 4104 pages = 16 416 cells/layer, 204 MiB** | **48.3** | 79 |

**So the pool lands exactly ON the sparse arm's ceiling (+0.8 %, noise) and the remaining -20 % against
resident-dense is the SPARSE ARM's own cost — the indexer scoring every block plus the radix top-k — NOT the KV's
placement.**  B3 (`-n 2000`, `-c 131072`, `-ncmoe 48`, plain decode): arena 23 579.8 -> 25 018.6 MiB
(**+1.44 GiB**, +18 slots/table, expert hit 96.26 -> 97.02 %, cold reads 42.0 -> 31.6 GiB), plain decode
43.1 -> 33.0 t/s.  Hit rate 89.7-90.05 %, fills ~0.05 ms/step (`qsa-census.py`'s LRU plateau is 90.2 %, so the
selection's ~10 % per-step novelty is the floor — an entropy bound, not a policy weakness).

**THIS SESSION'S JOB, in this order (the maintainer has agreed with all of the above):**

1. **Finish the acceptance table at the other two depths** (`B1-STRATA-PLAN.md` §3.3/§3.4 shape, pool arm added),
   one batch, arms interleaved, `ROCBLAS_USE_HIPBLASLT=0`, MTP depth 4:
   * **5.3k depth**: `-c 16384`, prompt `prompts/prose-rdna-boosts.txt` (5 256 tokens), arms
     resident / sparse-forced / pool.  Expect the pool ~equal to resident here (streaming cost 3.5 % at this
     depth) — if it is not, say so.
   * **180k depth**: a ~180k-token prompt (`cat prompts/kld-corpus-35b-in-domain.txt` **seven** times, ~180k
     tokens) with `-c 196608`; the mirror is 2.45 GiB of pinned RAM.  Arms resident / pool (+ `W=32768` as the
     recency control if time allows).  This is where the pool's ratio is best (204 MiB of pool against 2.45 GiB
     of KV) and where the sparse arm's cost is largest.
2. **Attribute the sparse arm's -20 %.**  At 103k, dump a `rocprofv3` kernel census for the pool arm and for the
   resident-dense arm; split the difference into (a) the indexer score (the `INDEXER_SCORE`/pool chain, every
   block), (b) the radix top-k (`indexer-topk.cu`), (c) the FA kernel.  **This decides the next engineering
   item and it lifts every streamed arm, including block 15's** — a page policy cannot recover more than
   ~0.05 ms/step, so do not spend the session there.  Report the split before changing anything.
3. **The MTP draft's sparse-decode re-measure** (`B1-STRATA-PLAN.md` §3.3's follow-up): `qwen4exp_mtp_sparse_arm`
   is default-OFF on measurements taken with a **resident** draft KV; under a pooled/streamed target the argument
   may invert (`LLAMA_MTP_SPARSE_DECODE=1`).  Two arms, acceptance + text identity.
4. **Then the promotion DECISION is the maintainer's** (do not build the package unasked).  What it would need:
   a block in `patches/` (the code is ~1 200 lines over 19 files), `ENVIRONMENT.md` §, `patches/README.md`
   block entry, the pool's VRAM in `--fit`'s arithmetic (the mirror is host, the pool must be *sized* by the fit
   like block 15's `auto`), a **grow-on-demand mirror** (today the whole `-c` is pinned at creation), the gates
   in the release flow, and a default-off call (it forces the sparse arm, so it is a fittability feature like
   block 15, not a speed one).

**Do not redo:** Goal A; B1/B2/B1(c); B-A's design, implementation or gates; the "is the pool exact?" question
(verify kernel + the sparse hash answer it); step 3's LRU policy **unless the maintainer asks** (expected ~1 point
of hits for ~0.05 ms/step — it is the entropy floor, not a policy gap).

START BY READING, in this order: [`BA-POOL-DESIGN.md`](BA-POOL-DESIGN.md) **§7** (the state), then
`HANDOVER.md` (**§11 = the work list**, §3 the invariants, §5 the measurement rules, §6 the gates), then
`B1-STRATA-PLAN.md` §3.2-§3.6 (the placement matrix and why the pool was the only form left), then
`A2-MIXED-BACKING.md` §4.

STATE (verify, don't assume):
  - **Campaign branch `wip-kv-residency-b`**, tip **`b5d895c59`** (pushed; `git@` works again — `ssh -T
    git@github.com` says hi).  `wip-kv-residency` (`7c7769e50`) is Goal A's measurement tree — do not develop on
    it.  Never commit campaign code to `rdna-boosts`; **never touch `patches/` or `release.json`**.
  - ONE R9700 (gfx1201), 32 624 MiB.  **Check `rocm-smi --showmeminfo vram` first** — the maintainer runs a
    Flash-Next server here that holds ~34 GB, and every fit/gate then reads garbage (`free = 2 MiB`) or OOMs.
    Never run a second GPU job; if the card is in use, say so and stop.
  - Link 55.7 GB/s H2D (re-cabled 2026-10-30).  No pre-swap t/s number may be compared with a new one.
  - **Push the campaign branch after every step** — these arms can OOM this box (the maintainer lost a session
    to a 235 GiB pinned-spill plan), so the work must be durable at every point.
  - `LLAMA_KV_POOL_STATS=<path>` appends the pool's accounting to a **file**; never print it to stderr on a
    text-hash arm (`scripts/extract-generated.py` slices from the generated text to llama-cli's perf line and
    would include it — that cost most of A10's "nondeterminism" hunt).  `LLAMA_KV_POOL_VERIFY=1` is the pool's
    oracle (0 mismatches expected).

CONSTRAINTS:
  - Build `~/bin/build-llama-rocm` (never pass `-DCMAKE_HIP_FLAGS`); `LD_LIBRARY_PATH=/opt/rocm-10.1.0-gfx120X/lib`.
  - **Every Flash-Next arm must carry `-ot per_layer_token_embd.*=CPU -lm none -lzm off`** (HANDOVER §5 rule 7):
    the PLE table is 27 465 MiB and is otherwise lazy-mapped, which moved identical runs by **±20 %** (with the
    flags: ±0.2 %).
  - **Flash-Next's MTP head is a SEPARATE file**: `-md /llm/models/Qwen3.8/Flash-Next/GSQ-IQ3_XXS/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf`
    (without it the draft context fails to load: "model doesn't contain MTP layers").
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
`check-gpu-faults.sh` over every log; the OOM baseline with the knob unset.  For anything that touches the pool:
the resident-arm text identity, the pool arm == the resident **sparse** hash, `test-backend-ops -o
FLASH_ATTN_QSA`, and the three state/park gates (the mirror must make them trivial — that is the point of the
writer rule).  A change to the sparse decode arm or the indexer owes the same prefill-logit gate.
