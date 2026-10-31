# Cold-session prompt — `wip/kv-residency` (TODO #118)

Paste the block below into a fresh session.  It assumes only what is in the repo (nothing from the previous
session's context).  **Rewritten 2026-10-31 at the end of session A11**: the acceptance table is complete at three
depths, the sparse decode arm has been attributed, the page policy (step 3) is measured dead, and the next
engineering item is named.  The promotion decision is the maintainer's.

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
B3) and **§8** (A11: the acceptance table, the census, the MTP-draft re-measure) before touching anything.

* Branch **`wip-kv-residency-b`**, tip **`b5d895c59`** (pushed; unchanged by A11 — that session changed no code).
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
| 102 899 (`131072`) | **58.27** | 53.92 (−7.5 %) | **54.20** (−7.0 %) | 2.83 / 2.84 |
| 180 035 (`196608`) | **48.73** | 42.93 (−11.9 %) | **43.09** (−11.6 %) | 2.75 / 2.71 |

* **The pool lands ON the sparse ceiling at every depth** (+0.35…+0.6 %, i.e. noise): placement is free, and
  what remains is the sparse decode arm's own cost.  Prefill: neutral at 5.3k, −5.1 % (103k) / −8.0 % (180k) —
  the mirror read over the link.  Pool hit rate 99.25 / 92.5 / 93.4 % of ids; pinned mirror 0.2 / 1.6 / 2.4 GiB.
* **A10's "−20 %" was the `-n 128` measurement window**, not placement: inside 128 tokens the two arms accept
  55 % vs 37 % because they generate different text, and t/s is (tokens/step) × (step rate).  **Rule (§5 rule 9):
  match the acceptance before reading a t/s delta, and use `-n 512`.**
* **The sparse arm's attribution (rocprofv3, MTP-4 at 103k, per verify step):** indexer score chain **+6.7 ms**,
  radix top-k +0.9, **FA −5.8** (3.14 → 1.09), pool resolve **+0.3**, copies +2.8, text/expert drift +3.4.
  Plain decode (1 token = 1 step) shows the chain is the *preparation* of the indexer keys — dequantizing key
  read 2.17 ms/token, a whole-cache `copyBufferRect` 1.83, bias add 1.18, rope 1.08, `rms_rows` 0.40 — while the
  score matvec itself is only 0.28.  **The pool's own machinery is 0.10-0.11 ms/token and its fills
  0.03-0.05 ms/step: the page policy (step 3) is dead.**
* **Step 3's other half is closed too:** `LLAMA_MTP_SPARSE_DECODE=1` under a pooled target is **−7.8 %**
  (54.25 → 50.00 t/s at 103k) with identical text (`854 43b30de151b9`) and identical acceptance (332/715), so
  that knob's default-OFF stands.

**THIS SESSION'S JOB, in order (the maintainer has agreed with all of the above):**

1. **The indexer score chain** — the only item with real mass left (§8.4/§8.6): ~6.7 ms of a ~35-44 ms verify step,
   and on a plain decode two thirds of it is the per-step re-derivation of the indexer keys rather than the
   score.  **Report the finding and a proposal to the maintainer before changing any kernel.**  Concrete first
   questions, all cheap: (a) *why* is the derived indexer key form not already in play at `n_tokens == 1` — the
   derived bias is gated on `ubatch.n_tokens > 1` (`qwen4exp.cpp`), the plain-decode arm takes a different top-k
   family (`indexer_topk_radix_histogram`/`write_blocks` rather than the `*_grouped`/`histogram_pass1` set), and
   `indexer-score.cu` already has an `indexer_fill_kernel` + an `RNG [from, lim)` range leaf that writes
   "the normed+rotated row into the pool buffer"; (b) whether the MTP/decode band can share one pre-roped key
   form with the prefill path; (c) what the score matvec alone would cost (`mul_mat_vec_f` 0.28 ms/token at 103k
   — i.e. the floor this can be taken to).  Judge any change with `scripts/gate-prefill-logits.sh --ab "<env>"`
   at `--ub 1` (must read 0.000000 / 100.000 %), the pool-arm text identity, and a fresh census.
2. **The radix top-k** (0.9 ms/step under MTP, 2.2 ms/token plain) — second item, same shape of question, only
   if 1 turns out to be immovable.
3. **Do not build the pool package unasked.**  The promotion DECISION is the maintainer's; what it would need is
   unchanged (§8.6 item 4): a block in `patches/` (~1 200 lines over 19 files), `ENVIRONMENT.md` §,
   `patches/README.md` block entry, the pool's VRAM in `--fit`'s arithmetic, a **grow-on-demand mirror** (today
   the whole `-c` is pinned at creation: 0.2 / 1.6 / 2.4 GiB at 16k / 131k / 196k), the gates in the release
   flow, and a default-off call (it forces the sparse arm, so it is a fittability feature like block 15).

**Do not redo:** Goal A; B1/B2/B1(c); B-A's design, implementation or gates; the acceptance table at 5.3k/103k/
180k or the census (§8); the "is the pool exact?" question; the page policy (step 3 — measured dead); the MTP
draft's sparse-decode re-measure (default-OFF holds).

START BY READING, in this order: [`BA-POOL-DESIGN.md`](BA-POOL-DESIGN.md) **§7 and §8** (the state and the
attribution), then `HANDOVER.md` (**§11 = the work list**, §3 the invariants, §5 the measurement rules — rule 9
is new, §6 the gates), then `B1-STRATA-PLAN.md` §3.2-§3.6, then `A2-MIXED-BACKING.md` §4.

STATE (verify, don't assume):
  - **Campaign branch `wip-kv-residency-b`**, tip **`b5d895c59`** (pushed).  `wip-kv-residency` (`7c7769e50`) is
    Goal A's measurement tree — do not develop on it.  Never commit campaign code to `rdna-boosts`; **never
    touch `patches/` or `release.json`**.
  - **The campaign tree is behind the delivery**: `b5d895c59` sits on the r182-era tip, the set is now at r186.
    A11's numbers were all measured on that one build (so the arm comparisons are like-for-like), and r183-r186
    are 1-GPU-neutral for these arms (r183's GDN folds already fire at one device, r185's verify-width bound is
    inert at one sequence, r186 only removes a multi-device-only conv gate) — but **re-base the campaign branch
    onto `release.json.tip` before promotion**, and re-measure rather than carrying a number across a base.
  - ONE R9700 (gfx1201), 32 624 MiB.  **Check `rocm-smi --showmeminfo vram` first** — the maintainer runs a
    Flash-Next server here that holds ~34 GB, and every fit/gate then reads garbage (`free = 2 MiB`) or OOMs.
    Never run a second GPU job; if the card is in use, say so and stop.
  - Link 55.7 GB/s H2D.  No pre-re-cable t/s number may be compared with a new one.
  - **Push the campaign branch after every step** — these arms can OOM this box, so the work must be durable at
    every point.
  - `LLAMA_KV_POOL_STATS=<path>` appends the pool's accounting to a **file**; never print it to stderr on a
    text-hash arm (`scripts/extract-generated.py` slices from the generated text to llama-cli's perf line and
    would include it).  `LLAMA_KV_POOL_VERIFY=1` is the pool's oracle (0 mismatches expected).

CONSTRAINTS:
  - Build `~/bin/build-llama-rocm` (never pass `-DCMAKE_HIP_FLAGS`); `LD_LIBRARY_PATH=/opt/rocm-10.1.0-gfx120X/lib`.
  - **Every Flash-Next arm must carry `-ot per_layer_token_embd.*=CPU -lm none -lzm off`** (§5 rule 7): the PLE
    table is 27 465 MiB and is otherwise lazy-mapped, which moved identical runs by **±20 %**.
  - **Flash-Next's MTP head is a SEPARATE file**: `-md /llm/models/Qwen3.8/Flash-Next/GSQ-IQ3_XXS/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf`.
  - **Static MTP: never `--spec-draft-n-max` > 4** (§5 rule 8).  **`-n 512`, and check that the arms' tokens/step
    match before reading a delta** (§5 rule 9).
  - Hash text from a run WITHOUT `-v` via `scripts/extract-generated.py`; timings from a separate `-v` run
    (llama-cli's internal server prints them as a JSON `timings` object).  Always `--single-turn
    --no-display-prompt`, always `timeout`, never two GPU jobs at once.
  - **`ROCBLAS_USE_HIPBLASLT=0` for EVERY fresh-process numeric comparison.**
  - Never quote a timing from a `LLAMA_KV_REGION_DEBUG` arm, nor from a `GGML_CUDA_DISABLE_GRAPHS=1` census arm.
  - A census is two `-n` lengths of the SAME arm, subtracted, filtered to kernels whose call *count* scales with
    tokens (§8.4); `rocprofv3 --kernel-trace --stats -f csv` does capture graph-replayed kernels here.
  - `/tmp` is a 47 GB tmpfs and a prefill-logit base is ~7 GB — clean old `.kld` files before a gate run.
  - `-n 2000` for anything about the expert arena, with `-ncmoe` = the model's full layer count.
  - Do not pick scope.  If something cannot be measured, say so and stop.

GATES: `GGML_CUDA_VMM_REGION_SELFTEST=1` (PASS, and FAIL with `LLAMA_KV_REGION_QUARANTINE=0`) if you touch the
VMM primitive (then `hostmap-hazard-probe.cc` too: default FAIL, `--grab 4` PASS); the coherence trio
(`5ba44647188f`, `ROCBLAS_USE_HIPBLASLT=0`); the prefill-logit gate on a **spilling** arm (−0.000000 / 100.000 %);
`check-gpu-faults.sh` over every log; the OOM baseline with the knob unset.  For anything that touches the pool:
the resident-arm text identity, the pool arm == the resident **sparse** hash, `test-backend-ops -o FLASH_ATTN_QSA`,
and the three state/park gates.  **A change to the indexer score / derived cache / sparse decode arm owes the
prefill-logit gate at `--ub 1`** (0.000000 / 100.000 %), the pool-arm text identity and a fresh census.
