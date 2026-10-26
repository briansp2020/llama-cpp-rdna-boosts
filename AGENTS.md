# AGENTS.md — working in this repo

This guide is for humans AND LLM coding agents. Read it before changing anything in
`~/llama-cpp-rdna-boosts/` (or acting on its behalf). It is deliberately short: policies, hard
invariants and pointers only. The dated record lives in `WORKLOG.md` (newest first) and `archive/`;
the pre-2026-10-04 AGENTS.md is preserved verbatim in `archive/docs/AGENTS-HISTORY.md`.

**Current state:** `release.json` is the single source of truth (fork point, canonical tip/tree,
block count, per-artifact sha256); the top of `WORKLOG.md` is the newest dated record. Do not restate
release history here — this file is the policy layer, not the log.

## What this repo is

A **delivery repo**. It packages the RDNA/ROCm work of the [`stew675/llama.cpp`](https://github.com/stew675/llama.cpp)
fork (`rdna-boosts` branch) as a **15-block set (V2.1)** (block 00 + blocks 01-14) that applies to a clean
llama.cpp checkout at the fork point named in `release.json`. This repo is NOT the fork: the fork
lives at `~/llama.cpp`; its `rdna-boosts` branch is re-cut from `patches/` on every release and
force-pushed to the personal fork so it never goes stale (see the Pushing policy). `patches/` is the
deliverable; `scripts/` is the apply/regenerate/validate flow; `README.md`
is the consumer overview; `MANIFESTS.md` is the apply order + verification contract, `BASELINE.md` the
provenance + drift policy (both are contracts now — their dated history is in `archive/docs/`);
`GREEDY-PURITY.md` is
the purity rulebook.

## Report findings before shipping (MANDATORY)

Surface review findings, open questions and small nits to the maintainer **before** any
delivery-affecting commit and **before** any release or tag, and wait for a decision. Do not bury them
in a commit message, a post-push summary or a GitHub reply. If something turns up mid-flight, stop and
ask instead of finishing the release and mentioning it afterwards. The maintainer decides whether a
finding is fixed now, deferred or dropped, and the decision is recorded (a `TODO.md` item, a block
amendment, or a note in the relevant record).

## Pushing policy (MANDATORY — read before any `git push`)

**Never push to upstream llama.cpp.** The only permitted push targets are this repo (the delivery,
`github.com:stew675/llama-cpp-rdna-boosts`) and the maintainer's personal fork
(`git@github.com:stew675/llama.cpp.git`). A bare `git push` in `~/llama.cpp` targets upstream
`ggml-org/llama.cpp` — never acceptable; repeated attempts can get the account banned.

- All deliverable changes live in THIS repo (`llama-cpp-rdna-boosts`) as the `patches/` set. That is
  the primary thing that gets pushed.
- **Every release MUST refresh the fork's `rdna-boosts` branch.**  Consumers clone the personal fork
  and check out `rdna-boosts` instead of applying the patch set, so a stale branch is what produced
  the 2026-10 Reddit "rdna-boosts is slow" incident (the branch was a month behind).  On every
  release (and whenever the block set changes), do a clean apply and push the branch:
  1. In a clean `~/llama.cpp`, check out upstream `master` at the **baseline point** — the `.base`
     commit in `release.json`, never the tip of `master`.
  2. Delete the existing `rdna-boosts` branch and re-create it with `scripts/apply-all.sh` (strict
     `git am`; the applied tree must equal `release.json`'s `.tree`).
  3. Build it and run the coherence + MTP gates and `scripts/gate-prefill-logits.sh`; `scripts/validate-set.sh` must be green.
  4. `git push --force-with-lease origin rdna-boosts` — the **personal fork only**, never upstream;
     never a bare `--force`.
- **Every release MUST also publish the ROCm container images.**  They are built and pushed
  **off-host** by the worker watcher (`scripts/container-watch.sh` on `radiance`), which polls for the
  new release tag and runs `scripts/build-containers.sh --rocm "7.14.1 10.1" --push --prune`; see
  `CONTAINERS.md` → "Offload the image build to a worker".  **Do not run the full container image build
  (all targets, both ROCm lines) on the primary box during a release**: it pins all cores for tens of
  minutes and contaminates any benchmark running beside it.  Nothing here changes the normal local
  **source** build for testing and development (`~/bin/build-llama-rocm`, Common tasks → "Build the
  fork"), which stays on the primary box.  The local `scripts/build-containers.sh` is for a smoke test or
  a manual worker-less container build only.  The Actions *gating* workflow no longer builds them (the tag-push matrix took ~5 h and its failure mode was a cold
  retry).  The fork-branch refresh and the worker image push are both part of tagging a release.
- The `~/llama.cpp` checkout exists to host the block commits and to apply/test the diff set locally.
  Apart from the release-time branch refresh above it is disposable: delete the pre-patched branch and
  re-apply rather than editing history in place.
- Confirm the exact branch name and remote before any push: the branch is always `rdna-boosts`, the
  permitted target is always `git@github.com:stew675/llama.cpp.git` (the personal fork).
- Repeated attempts to push directly to llama.cpp can result in an account ban. When in doubt: don't
  push, ask.

## Layout

| path | what |
|------|------|
| `README.md` | consumer overview + workflow (start here) |
| `ENVIRONMENT.md` | **every environment variable** the delivery adds or repoints — default, and whether it is a kill-switch (default-on, set to disable), an opt-in, a tuning value or a diagnostic. Read it before adding a variable or changing a default |
| `release.json` | **delivery single source of truth** (fork point, canonical tip/tree, block count, per-artifact sha256) — read by `apply-all.sh`, `validate-set.sh` and CI; regenerate with `scripts/make-release.sh`, never hand-edit hashes |
| `MANIFESTS.md` | apply order + the verification contract (release gates, per-block verification, failure handling). Its dated validation history is in `archive/docs/validation-history.md` |
| `BASELINE.md` | fork point, what the patches are, how they are cut, drift policy. Its dated baseline/re-base records are in `archive/docs/baseline-history.md` |
| `TODO.md` | **forward-looking only**: open items, items waiting on someone else, accepted limitations, parked ideas. A closed item is one `WORKLOG.md` entry — never a TODO entry. `#NN` numbers are permanent handles and are never reused |
| `GREEDY-PURITY.md` | the purity rulebook (index + invariants + per-finding claims); dated narratives/evidence are in `archive/docs/GREEDY-PURITY-FINDINGS.md` under the same `§` numbers |
| `patches/` | **the delivery set** (0000-0014: block 00 + blocks 01-14) + `README.md`, the per-block reference (what each block contains, its knobs, its purity notes). **Block 14 is TEMPORARY** (`-sm relaxed-tensor`, a local relaxation of upstream #31081, no upstream candidate, tracked as `TODO` #104) — keep it separate and expect to drop it. The dated amendment notes moved to `archive/docs/patch-notes-history.md` |
| `scripts/` | `apply-all.sh` (the tested apply flow, strict tree check + `git am -3` fallback), `gate-prefill-logits.sh` (the prefill-logit KLD release gate, issue #113), `gate-per-block-build.sh` (a clean build per block commit — `all` by default, `--fast` for the release profile), `gate-qwen4exp-quant-coherence.sh`, `check-gpu-faults.sh` (reads the kernel journal and scans a llama log for **real** GPU faults, filtering the echoed prompt lines — never grep a log for `page fault`/`slab` by hand), `make-patches.sh`, `make-release.sh`, `validate-set.sh` (checksums + fresh-tarball strict apply, runs in CI), `build-containers.sh` (the image build itself), `container-watch.sh` (the off-host release watcher, run on `radiance`), `cleanup-ghcr.py` (GHCR version retention; called by `build-containers.sh --push`), `extract-generated.py` (hashes llama-cli generated text; a naive sed/grep slice does not reproduce the hashes), `check-windows-paths.sh` (fails on a path that Windows cannot check out). **Index: `scripts/README.md`** |
| `rdna-boosts-all.patch` | the entire 12-patch net as ONE patch (fork point only) |
|  `benchmarks/` | dated bench records + methodology; **`mtp-adaptive-methodology.md` = the adaptive-MTP baseline gate** (run before shipping any decode/fusion change) and **`prefill-logit-methodology.md` = the prefill-logit KLD gate** (run before shipping any prefill-kernel change or release) |
| `prompts/` | versioned, hash-stable test prompts; sizes/token counts/**sha256** in `prompts/README.md`. A shipped prompt is **never edited in place** (add a new file); a result is only valid against the prompt hash it names |
| `wip/` | **ACTIVE** exploration docs/tools — **NOT part of the delivery**; see the WIP rule. Each campaign is a self-contained handover under its own `README.md` (no `wip/README.md`); the open set is indexed in `wip/CAMPAIGNS.md` |
| `beta/` | **promotion-candidate patches** staged for a beta test — **NOT part of the delivery**. Each candidate is `<name>/` with `README.md` (what/why/status), `BETA-TESTING.md` (the tester's apply + gate checklist) and `<name>.patch` cut against the current `release.json` tree. Nothing here is applied by default: `scripts/apply-all.sh --include-beta` applies `beta/*/*.patch` on top of the verified release tree for a tester. A confirmed or refuted candidate moves to `archive/work/` and is recorded in `WORKLOG.md` |
| `upstream/` | **upstream-PR candidates** (self-contained changes for unadulterated `ggml-org/llama.cpp`), one sub-directory per candidate (`README.md` + `UPSTREAM-PR-<name>.patch`); the index is its `README.md`. None is part of the delivery |
| `archive/docs/` | moved-out historical records (validation/baseline history, the pre-cleanup AGENTS.md) — reference only |
| `archive/work/` | closed experiments and campaigns, preserved for future re-evaluation (mmb/qsa3, moe-expert-cache, issues/PRs, …) — reference only |
| `baseline/*` branches, `block/*` tags | **historical** pre-block-12 checkpoints — do not use for the current delivery |
| `.github/workflows/` | `validate.yml` (per-push delivery validation), `release.yml` (tag-triggered GitHub Release assets) and `docker-ghcr.yml` (manual-only ROCm image fallback); fork point read from `release.json`. **The release container images are built and pushed off-host by the worker watcher (`scripts/container-watch.sh` on `radiance`), not in CI and not on the primary box** — see `CONTAINERS.md` |

## Scope policy — RDNA first, other backends uninjured

This repo is **RDNA/ROCm-specific**: validation, tuning and claims cover the AMD devices the
maintainer runs (gfx1201 = RDNA4, plus validated gfx1151/RDNA3_5 and gfx1100/RDNA3_0). The patch set
is generic llama.cpp, so it must not *break* other backends (NVIDIA/CUDA, MUSA, SYCL, Vulkan, CPU):
keep the shared CMake lists, dispatch tables and predicates mutually consistent even where a change
is unreachable on AMD. **Behaviour and performance on non-AMD backends are explicitly out of scope** —
no tuning, no validation, no waiting on hardware there.

- An AMD-only fix may land without its non-AMD counterpart as long as the non-AMD paths stay
  consistent (no aborts, no uninstantiated pairs) and the difference is documented (see the F1 chooser
  example in `upstream/fa-decode-verify-kernel-family/`).
- New KV-cache types / instances / predicates **are** kept cross-backend consistent — an inconsistent
  set is a crash on whichever backend reaches it (`GREEDY-PURITY.md` §20). That is correctness, not
  scope creep.
- "Not validated on NVIDIA" is an acceptable, documented state — never a blocker for an RDNA win.

## Default-on policy — beneficial features are ON; env vars only disable

A feature that improves performance (or correctness) and has passed the relevant QA gates is
**enabled by default**. An environment variable for such a feature exists **only to disable it** — for
A/B testing, bisection or debugging — never to enable it.

**Why:** the maintainer's validation boxes (Strix Halo / gfx1151 in particular) are much slower than
gfx1201, so a gate that silently runs without a beneficial opt-in wastes wall-clock time and reports
slow numbers as if they were the product (the `mmb`/HC16 campaign measured ~835 t/s for a session
when the full set is ~1136 t/s, +36 %).

- When a gate passes, flip the default **in the same change** and turn the old opt-in into an opt-out
  (`FEATURE=0`). Keep reading the env var; just invert the default.
- A feature that is beneficial but not fully gated is a reason to **gate it**, not to leave it off.
  Default-on with the risk documented, then run the gate.
- A kill-switch for a *correctness* risk (a bisect knob, a known-bad-state workaround) may stay
  default-off — the distinction is "on because it helps" vs "off because it is known-risky".
- Before any benchmark, use the full feature set. If you write `FOO=1 <bench>`, ask why the default is
  not already `1`.

## WIP and promotion rules (MANDATORY)

Everything under `wip/` and `archive/work/` (including loose patches/diffs and experiment trees) is
**experimental work, NOT part of the delivery**. Never apply a `wip/`/`archive/work/` item to the fork
or any llama.cpp checkout, and never fold its content into `patches/`, **unless the user explicitly
asks for that specific item**. WIP stays env-gated OFF and excluded from `patches/`; the sanctioned way
out is promotion: collect *validated* wins, give each an env kill-switch, re-validate the
**combination** (individual validations do not carry over), then cut a new block with the maintainer's
go-ahead. Anything also applicable to unadulterated upstream gets a copy under `upstream/`.

**`beta/` is the external-validation holding area, not a third kind of source.**  `scripts/apply-all.sh
--include-beta` is the sanctioned way to *test* one (it applies `beta/*/*.patch` after the release blocks and
reports the resulting tree); nothing in `beta/` is ever picked up without that flag.  A candidate that
cannot sit in `wip/` because it edits a file a block owns, but is not yet validated on the reporter's
platform, goes to `beta/<name>/` (a `README.md`, a `BETA-TESTING.md` and a patch cut against the
current `release.json` tree; nothing in it is applied by `apply-all.sh`).  On confirmation or
refutation it moves to `archive/work/<name>/` and is recorded in `WORKLOG.md` — same no-stub rule as
`wip/`.

**A finished campaign leaves `wip/` — it does not become a stub.**  The moment a campaign is promoted
(folded into `patches/`), closed, or refuted, its record moves to `archive/work/<campaign>/` and the
result is recorded in `WORKLOG.md`, then the `wip/` entry is **deleted** and `wip/CAMPAIGNS.md` is
updated.  Do **not** leave a "MOVED — see archive" placeholder behind: a redirect stub makes `wip/` claim
work is in flight when it is not, which is exactly the state this rule exists to prevent.  Dated records
(`WORKLOG.md`, `MANIFESTS.md`, `BASELINE.md`, `archive/`) keep the path that was correct when they were
written; when a campaign moves, fix those paths to the new `archive/work/` location rather than keeping a
stub alive to serve them, and treat a `wip/` path in an old record as a historical location.

## Critical facts (do not re-derive)

- **`llama-cli` MUST be run with `--single-turn`** (plus `--no-display-prompt` for scripted output).
  Without it, the interactive chat loop blocks forever. Wrap potentially blocking commands in `timeout`.
- **Apply method:** all 15 blocks with **`git am`**; `scripts/apply-all.sh` is the tested path. Plain
  `git apply` of the concatenated series **silently drops hunks**. Every block is a normal commit.
- **Naming (V2.1):** the current set is **15 blocks, 00-14** (block 14 is the temporary `-sm relaxed-tensor`
  block, see `patches/README.md`; release `release.json .release` —
  `v15-6050b467c-r138` at the 2026-10-25 re-base; the value lives in `release.json`, never here).  Every `block NN`
  reference elsewhere in this file that is not one of the V2.1 numbers below is the **pre-V2.1 16-block
  numbering** and is kept as a historical handle.  The V2.1 map: **01** all-reduce, **02** adaptive MTP,
  **03** chunked GDN, **04** BF16 KV, **05** FA band/native/derived, **06** MMB, **07** k-quant,
  **08** fused kernels (fused MoE), **09** host-resident MoE (slab/cache), **10** slab + fusion dispatcher
  + async input ring (infrastructure), **11** qwen4exp (model/QSA/HC/lazy reader), **12** campaign
  remainder, **13** MTP draft vocabulary subset (opt-in).  V2.1 split r109's old block 10 in two: the
  **infrastructure stays 10**, the **qwen4exp architecture is 11** (QSA went with it by decision), so
  every pre-r109 "block 10" note about slab / dispatcher / async-input is still block 10 while a note
  about qwen4exp / QSA / HC / the lazy reader is now block 11.
- **Throughput is NOT a correctness signal.** Verify coherence (same-seed `llama-cli` diff, or
  `archive/work/tools/ar_kernel_unit.cpp`). Decode perf work is validated at **depth-16384**, not
  shallow `llama-bench`. **Never run parallel/background benches.**
- **A fresh-process comparison is only valid with `ROCBLAS_USE_HIPBLASLT=0`** (set it for that run, do not
  ship it).  On ROCm before the `rocm-libraries` fix (Linux 10.x included — 10.1.0's pin predates it),
  hipBLASLt picks its solution for an untuned FP32 GEMM size **per process start**, so two identical
  commands can produce different logits and a different gen t/s.  This is *not* a defect in this patch set
  (issue #67, external) but it **will** be misread as one: it was the entire residual of the 2026-10 MTP
  determinism hunt (3 logit classes in 18 fresh starts of one gemma4 MTP config; 10/10 with the env set),
  and it made every graph/capture arm look like a cause.  Attribute a per-start difference to the library
  before touching code; **never judge run-to-run determinism on a greedy text hash** — use a logit-level
  probe.  **Keep hipBLASLt enabled in the delivery** (disabling it costs up to -15 % decode; the switch is
  for reproducible runs and bisection only).  `GREEDY-PURITY.md` §44,
  `archive/work/fix-gemma-4/evidence/MEASUREMENTS-2026-10-21d.md`.
- **`--lazy-mode off` overrides a mapped load, and `-sm tensor` + `--load-mode none` uses the chunked async ring (r122, block 00).**
  An explicit `off` loads the `TENSOR_READ_LAZY` PLE/engram table into its own host buffer instead of
  serving it from the `--load-mode mmap` mapping (`auto` below 4 GiB stays mapped, so default loads are
  unchanged); `--load-mode mmap --lazy-mode off` is the "fast mapped bulk + resident PLE" combination.
  The stale `ggml_backend_dev_is_meta(dev) -> nullptr` loader bail is gone: the Meta async setter serves a
  mid-row chunk (`write_partial_row`) and the event proxy fans the 4-slot staging ring
  (`LLAMA_LOAD_STAGE_MB`, default 4 MiB) out to the simple devices, so `--load-mode none` under `-sm tensor`
  is ~2x faster (cold Flash-Next load 32.2 -> 13.6 s).  Both are **block 00**; do not reintroduce either
  workaround.  Same-seed greedy text is byte-identical across the load modes on Gemma4-E4B and Flash-Next.
- **A layer-input tap needs the HC-averaged state on qwen4exp (`build_hc_mean`, r121, TODO #94 closed).**
  `qwen4exp` runs `hc` parallel residual streams, so `res_hc` is `hc * n_embd` wide (10240 on Flash-Next)
  while `extract_layer_inp()` demands exactly `n_embd` per row: exporting the wide residual made any
  DFlash2/DSpark head abort at `GGML_ASSERT(row_floats == model.hparams.n_embd)` (llama-context.cpp:3437;
  the F1 device path asserts `t->ne[1] == n_tokens`).  The tap is `graph::build_hc_mean()` (mean over the
  streams, the same 8 lines as deepseek4/glm5-next and the head author's own `hidden_states.mean(dim=-2)`
  capture), **gated on `cparams.embeddings_layer_inp[il]`** so a graph that taps nothing (plain decode, MTP,
  QSA) gains no node — plain decode is byte-identical and the prefill logits are unchanged (KLD −0.000000).
  `target_layers` is `completed + 1` (the capture is a layer *output*), no index shift.  **Since the
  `3f03b1772` re-base the helper is upstream's `hc_stream_mean()`** (its own #30927 fix for the same
  problem, node-for-node identical); the delivery dropped its `graph::build_hc_mean()` for qwen4exp
  (deepseek4/glm5-next keep theirs).  Measured on
  Flash-Next: +40 % code / +91 % recall over plain decode at 3 GPU tensor + `-ncmoe 48`, six
  `plain == draft-dflash` text-identical arm pairs.  Index `n_layer` (the last layer's output) is **not**
  exported — it needs the pre-gather residual; no head taps the final layer.  Record:
  `archive/work/qwen4exp-dflash2/`.
- **A cross-device copy must ask `ggml_cuda_slab_peer_copy_ok()` first (r121).**  The slab's
  peer-access grant is per-device and **best effort**: with three ~29 GiB slabs the driver refuses it for the
  second and third device (the peers have no free VRAM to build a 29 GiB peer mapping), and a
  `hipMemcpyPeerAsync` into an ungranted slab range **faults the GPU** (`HSA_STATUS_ERROR_MEMORY_FAULT` in
  `__amd_rocclr_copyBuffer`) or returns `invalid argument` — never anything catchable.  The query is
  process-wide for slab destinations: a copy between two fully granted devices still fails while a third
  slab's grants are incomplete.  When it says no, the copy stages through a grow-only pinned buffer
  (`ggml_cuda_peer_copy_staged`, D2H on the src stream, H2D on the dst stream).  Do not add a raw peer copy
  anywhere without the guard; `-sm layer` + `-ncmoe` warm-up aborted on Flash-Next before this (reproduced on
  the r120 base with no drafter).  No env knob.
- **DFlash2 under `-sm tensor` is supported (r106, TODO #78 closed):** the selector's `ggml_top_k` /
  `ggml_get_rows` over the draft logits -- which are the *target's* `output.weight` projection, placed
  vocabulary-split -- are served by a new general collective **`GGML_OP_ALL_GATHER`** (`ggml_all_gather()`)
  that `build_dflash2_selector()` emits when the draft is a >= 2-device tensor split.  On the Meta backend it
  is `MIRRORED`, its own subgraph boundary, and dropped from every per-device child graph; the collective
  copies one row (of every dim above the split axis) at a time because a slice is contiguous in its own copy
  but strided in the full tensor.  **Only a plain axis-0 split is implemented** (anything else aborts by
  design), and there is no env knob: `-sm layer` is the escape hatch.  Mirroring the *drafter* does not work
  (the offending tensor is the target's), and neither does demoting only the drafter to `-sm layer` (a
  layer-split context cannot read the target's Meta-buffer weights).  Two related fixes ship with it: the F1
  `extract_layer_inputs()` device copy cannot express a Meta buffer (host path instead; it aborted for **any**
  DFlash drafter with `dflash.target_layers`), and the **slab narrow-2** pin must yield from the arena when
  the draft's compute buffer exceeds the unmapped VA spare (a tensor-split drafter's is 1280 MiB against the
  estimator's 512 MiB tier; without it `-sm tensor` + draft + `-ncmoe` aborts in `graph_reserve`).  Rule:
  `patches/README.md` (block 12, "Facts that span blocks").
- **The draft context's second scheduler is selected per arch (`sched_alt_for`): `n_outputs` for MTP, the
  batch kind for DFlash/DFlash2 (r120, issue #208).**  MTP's two graphs differ by outputs; a DFlash2
  draft's do not (the lattice is read from `h_nextn`, so `output = !is_dflash2`), so selecting on
  `n_outputs` put the KV injection and the draft step on one scheduler *and* one `gf_res_prev` slot: 0 %
  graph reuse, a rebuild + re-allocate on every verify step (r65-r119).  DFLASH now sends the token batch
  (no `embd`, no `lf_dev_inject`) to the second scheduler and keeps the injection on the main one; every
  other arch keeps `n_outputs`, where the batch kind would *merge* MTP's ingest and step graphs.  Nothing
  else may pick a draft graph slot independently: `get_gf_res_prev()` shares the decision.  Reuse
  0 -> 97.8 %, decode +1.4 % (27B 2 GPU tensor) to +4.0 % (35B-A3B `-ncmoe 40`), text and acceptance
  identical.  Record: `archive/work/dflash2-draft-sched2/`.
- **The expert cache's routing trace is opt-in and is the policy instrument (r110, TODO #61 P1).**
  `MOE_EXPERT_TRACE=<path>` writes every `mul_mat_id`'s raw routing (`scripts/moe-trace-replay.py` scores
  policies against it offline: the shipped F-LRU, static top-f, per-layer quota, window/prefix
  pre-admission).  DECODE records come from the scheduler hook (the ids are already on the host);
  PREFILL records come from the device entry and pay one readback per op (prefill is never captured, so
  that is legal).  An IDENTITY (fully resident) table's decode routing is not traced -- it cannot miss.
  The trace is a *diagnostic* (never default-on) and its absolute hit rate is a model, not the cache:
  judge a policy by its delta.  `MOE_EXPERT_TRACE_MIB` (default 1024) caps the file.
- **Two host-tally gaps are closed and must stay closed (r110, TODO #64).**  (1) The host pool's warm
  order comes from the *device* prefill tally, folded in once at the first decode-band pass
  (`pool_rank_from_prefill_locked`): the eager `prefill_count` path never runs under the staged prefill,
  which left `pool_rerank_from_tally_locked` uncalled and warmed the pool in expert-id order.  (2) A
  fully-resident (identity) table's decode routing is tallied device-side, **sampled**
  (`MOE_EXPERT_HEATMAP_DECODE_SAMPLE`, default 64, 0 off) and weighted by the period.  The heat mirror
  is pre-allocated at sizing: the decode band is a captured graph and a `cudaMalloc` there aborts the
  capture (`operation failed due to a previous error during capture`).
- **The expert cache's *identity fast path* is an all-or-nothing cliff worth ~40 % of decode; r117 makes the
  cache prefer whole-table identity (block 09, TODO #93 closed).**  A table is identity only at **100 % of
  its own slots** ("the decode band reads the arena directly; no per-token routing readback"); one slot short
  and it takes the remap path and pays the readback exactly like a table with no arena.  Each table's arena is
  rounded up to the slab arena unit, so a budget "just enough" in aggregate landed one unit per table short
  and **every** table dropped off: measured (GSQ `-ncmoe 48 -sm tensor`, plain decode) 3 devices
  `MOE_EXPERT_CACHE_MIB=13800` → arena 40845.1 of 40925.0 MiB, 0/432 identity, **46.7 t/s** vs `15000` →
  432/432, **65.0 t/s**; the same cliff cost the 2-GPU arm 46.3 vs 66.4 t/s at `-c 204800`.  **r117:** when
  ≥ 96 % of the layers can hold their full slot count, those get full residency and the coldest ≤ 4 % are
  dropped to 0 slots (they run through the scheduler's host copy) — +27.6 % on the arm above, bit-identical
  text/acceptance, `MOE_EXPERT_CACHE_IDENTITY_PREF=0` restores the uniform placement, and the placement never
  exceeds `MOE_EXPERT_CACHE_MIB`.  The cap matters: a dropped layer is not free (1 of 48 = +27.6 %, 6 of 48 =
  −30 %), so an arena that is *percentages* short still pays the readback.  **Any 2-vs-3-device (or decode)
  comparison must still record the arena, the residency % and the count of `identity fast path ON` lines.**
- **Greedy text is a per-device-count reference on the MoE tensor split (r116).**  2 vs 3 GPUs on
  `-ncmoe` + `-sm tensor` produce different text (`PPL 10.8452` vs `10.8623`, 0.16 %, the tensor split's
  own rounding class, `GREEDY-PURITY.md` §45) — it is **not** the AR's BF16 wire (`GGML_CUDA_AR_BF16_THRESHOLD=0`
  does not reconcile it).  *Every* `-sm layer` arm is text-identical across device counts, and the recorded
  **dense** cross-count pairs were too — **but a dense tensor split is not guaranteed to be** (measured
  2026-10-24 on Qwen3.8-27B UD-Q4_K_XL: 3 GPU 850 chars vs 2 GPU 876, and r126 shows the same 897 vs 855;
  pre-existing, not a re-base defect).  So gate a build change **within** each device count (as r116's own
  gates did); use `-sm layer` when a cross-count text comparison is required; never read a 2-vs-3 text
  difference as a purity defect.
- **The disk tier is bounded by this box's SSD at ~1.8 GB/s, so the cache's currency is `L3 MiB/token`, not
  `h`.**  1 MiB/token of reads that neither VRAM nor the pinned pool holds is ~0.55 ms/token of unavoidable
  stall (~1800 t/s of pure read budget at a 1 MiB/token traffic level, ~100 t/s at 18 MiB/token).  Every
  tiering claim in the records is therefore quoted in MiB/token beside `h` (`scripts/moe-trace-replay.py`
  reports both): r114's band-split heat prior took the long-prompt arm from 48.1 to 17.6 MiB/token, i.e.
  **~26 ms → ~9.5 ms of disk per token**, which is what a decode on that config (46 t/s = 22 ms/token) can
  and cannot afford.  Use the same conversion before proposing a prefetch/eviction policy: the pool-layout
  and topic-shift candidates were all declined because their whole prize was 0.3-0.5 MiB/token.
- **The heat prior is band-split (r114, block 12): the ARENA seed ranks by the decode-only counter, the host
  pool's layout by the combined one.**  `apply_heat_seed_locked` + the heat-ordered layer placement call
  `heat_rank_vec()` (the `heat_dec` vector, falling back to `heat` when a map has no decode mass); the pool's
  `pool_rank_experts_locked` / `pool_rerank_from_heat_locked` keep `heat`, because the pool stages whole
  tables *for a prefill* and there the prefill mass is signal, not noise.  Sidecar format **MHT2** (both
  arrays per table); a **MHT1** file loads with `dec = all` (pre-r114 behaviour), never an error.  Do not
  "simplify" this back to one counter: on a long prompt the prefill outweighs the decode 41x and the arena
  ends up seeded from a nearly uniform distribution (+1.2 to +2.5 pts of `h` and ~3x less disk traffic on
  the measured arms).  `ENVIRONMENT.md` (heat-map note), `patches/README.md` (block 12).
- **Thread sizing is not optional.** `/usr/local/bin/pin_gpu_irqs.sh` puts the GPU IRQs on the top
  `NUM_GPUS` cores (13,14,15 with 3 GPUs); the default `-t` puts the worker pool there and starves the
  GPU (Q4_K_M `-ncmoe 99` tg1024: `-t 16` 19.6 vs `-t 8` 38.9). Size `-t` (or `--cpu-mask` with `-t`
  *inside* the mask) to leave those cores free; one CCD (8) is within 1-2 % of best. r19's block-06
  cap only covers `-ncmoe` offloaded-MoE decode; override with `GGML_CPU_MOE_OFFLOAD_THREADS=N`.
- **The host link is PCIe 5.0, and it is not uniform: 2 cards at x8 and 1 at x4 (since the 2026-10 top-up).**
  Each R9700 sits behind its own PCIe switch (`Navi 10 XL Upstream Port`): the GPU side always negotiates
  **x16 @ 32 GT/s**, but the switch **uplink to the host** is what varies — read the *root port* behind the
  card, not the card: GPU 0 (`03:00.0`) -> `00:01.1` **x8**, GPU 1 (`06:00.0`) -> `00:01.2` **x4**, GPU 2
  (`09:00.0`) -> `00:01.3` **x8** (all `current_link_speed` = 32 GT/s).  That is ~32 GB/s raw for GPUs 0/2
  and ~16 GB/s for GPU 1.  Consequences for measurement, measured 2026-10-17 on 35B-A3B Q4_K_M q8_0
  `pp31500` when every card was x4: 1 GPU 3614 t/s, 2 GPU tensor **4482 (+24 %)**, 3 GPU tensor **4347
  (-3 % vs 2 GPU)**; the same +17 % / +1 % shape is in `archive/work/three-gpu-performance/` (1065 / 1246 /
  1258 t/s).  So **multi-GPU prefill on this box is link-bound and now heterogeneous** — a 2-GPU arm is not
  the same shape at `HIP_VISIBLE_DEVICES=0,1` (x8+x4) as at `0,2` (x8+x8), and a tensor-split prefill claim
  from a contributor host with x8/x16 everywhere can only be *under-measured* here, never refuted: report it
  as "not measurable on our link", not as "does not reproduce".  Decode claims do carry over (the per-token
  AR traffic is tiny): #175's decode gain reproduced at 2-GPU tensor while its prefill claim did not.
- **`-ncmoe`/`-cmoe` host experts are pinned (`ROCm_Host`) in the DEFAULT mode; `--load-mode mmap` alone
  does NOT change that.** The loader rewrites the `-ncmoe` CPU override for a host `MUL_MAT_ID` weight to
  the layer device's **pinned** host buffer, and the "avoid a host buffer when using mmap" downgrade is
  skipped for it. The pageable `--host-experts mmap` mode and the legacy `LLAMA_MMAP_HOST_EXPERTS` env
  were **removed in r33** (issue #116: a no-XNACK GPU -- gfx1201 reports `XNACK enabled: NO` -- cannot
  read a pageable host master in a kernel, and the pageable fill faults `hipMemcpy2DAsync` on ROCm 7.14).
  `--host-experts` (`llama_model_params.host_experts_mode`) selects **`pinned`** (default),
  **`pool`** (r35/r111) or `mmap` (logs "unsupported" and falls back to pinned).  Pinned is deliberate --
  the per-ubatch op-offload H2D upload and the expert-cache fill both read this master in place, and under
  `-sm tensor` a pageable master would land in the non-`is_host` `CPU_REPACK` (no device access, cache
  inert).  Cost of the default: the host expert set is non-swappable RAM (Q4_K_M 35B-A3B `-ncmoe 40`,
  cache off: `ROCm_Host` 18662 MiB; peak RSS ~41.7 GB).  `--lazy-mode` governs the mmap *reader*, not
  this. `ENVIRONMENT.md` §§1.2.1/4; `patches/README.md` (blocks 09/12).
- **`--host-experts pool` (r111) has NO pinned master: the master IS the read-only model mapping, and
  every GPU read of a pooled host expert goes through pinned memory.**  The tensors live in the model
  mapping (host buft `ROCm_HostMap`, device-carrying, r111, **block 12**) and are never read by a device
  copy or a kernel; the L2 tier is the bounded pool (`MOE_HOST_POOL_MIB`/`FRAC`, default **80 %** of the
  host expert bytes since r112, *per tensor*, process-wide, capped at **`MOE_HOST_POOL_RAM_PCT` = 85 %** of
  `MemAvailable` since r114) and the L3 tier is the GGUF, read with O_DIRECT **only when the page cache
  cannot hold the set** (`MOE_HOST_POOL_DIO` is **`auto`** since r114: decided once at the first pool build
  from `MemAvailable` vs `planned pool + host expert set`; `0`/`1` still force either arm).  The
  mode is only for the case where the set does **not** fit in RAM: when it fits, `pinned` is faster on every
  axis.  80 % is the measured knee (90 % was slower and swapped).  The invariants:
  (1) a pooled table must never be served by a **pageable** `cudaMemcpyAsync` -- ROCm stages a pageable
  upload through a shared internal bounce and back-to-back copies tear (this is what made the first cut
  nondeterministic; `moe_cache_upload_pageable`, the staging bounce and the `stage_gather` host-gather
  shape all exist for it); (2) the cold-UVA serve and the device gather require a *pinned* master and
  stand down (`host_dev == nullptr`) -- a miss becomes a pool fill; (3) any new reader of a host expert
  master must go through `expert_host_src` (pool slot / per-table pinned bounce) or be added to
  `moe_cache_upload_pageable`; (4) the mode needs a **mapped** load (`--load-mode none/dio` keeps the
  pinned master and warns).  Measured: `RssShmem` 18.6 GiB -> 6.8 GiB on 35B-A3B `-ncmoe 40`, and a
  12 GiB or 8 GiB cgroup that OOM-kills the `pinned` arm runs this one at ~93 t/s.  Cost: one extra host
  memcpy per staged byte on the host-expert prefill path (single-ubatch `pp512` on 2-GPU tensor -34 %),
  which is why the staging copy itself must stay batched (`MOE_HOST_POOL_STAGE_BATCH`, r114).  Record:
  `archive/work/host-expert-master-drop/`.
- **The host pool is a LAYOUT, not a churning cache (r113, block 12; TODO #87's disk-I/O half).**  The
  pool's **content** is what decides the disk traffic — measured offline on a real trace: pool ranked by
  expert id = 3571 MiB of L3 reads over 451 decode tokens, ranked by the **persisted heat prior** as the
  arena's *complement* = 237 MiB, oracle = 237 MiB — while **every** eviction/hand-over policy lands within
  0-246 MiB of the others.  So: (1) `pool_rank_experts_locked` / `pool_rerank_from_heat_locked` lay the pool
  out as the arena's complement, in bulk, at sizing and at the first decode-band pass (the old flat
  prompt-tally re-rank is gone — a prefill routes every expert, so its tally cannot rank); (2) the **demand
  path never evicts and never copies the arena victim into the pool** (`pool_peek_locked` /
  `pool_fill_empty_locked`; the background worker and `MOE_HOST_POOL_BGFETCH` are gone) — do not reintroduce
  one: the literal swap cost **2x decode** for 0 MiB of L3; (3) the whole-table prefill staging reads the
  pool and fills empty slots (it used to bypass it and re-read the whole host set per ubatch); (4)
  `MOE_HOST_POOL_RAM_PCT` (default **85** since r114) caps the whole pinned pool at that % of `MemAvailable` — `FRAC`
  is a percentage of the model's host expert set, not of RAM, and an oversized pool fails each table's
  `cudaHostAlloc` *silently*.  Field config (70K prompt, 128-token decode, 2x R9700): pool hit 0.80 -> 0.92,
  evictions 4600 -> 0, prefill 39 -> 287 t/s (`DIO=1`) / 581 -> 733 t/s (`DIO=0`), decode 6.3 -> 23.1 /
  30.6 -> 53.7, against `pinned` 836 / 57.5.  The tool that decided it (`scripts/moe-trace-replay.py
  --pool-frac`, two-tier, reads a real `.moe-heatmap`): use it before proposing another pool policy.  Record:
  `archive/work/host-pool-layout/`.
- **POOL mode: the fill mode is automatic and the staging is one launch (r114, block 12, TODO #87).**
  `MOE_HOST_POOL_DIO` = `auto` (decided once at the first pool build from `MemAvailable` vs `planned pool +
  host expert set`: buffered when the set fits beside the pool, O_DIRECT when it does not; `0`/`1` force),
  and `moe_cache_stage_kernel` fills **both** tiers (arena slot / pool slot) for a pool-backed table, so a
  prefill staging is one launch instead of one `cudaMemcpyAsync` per expert.  Measured (2x R9700, 35B-A3B
  Q4_K_M `-ncmoe 40 --host-experts pool`, 70K prompt, `-ub 2048`): prefill **2103 t/s** vs **1564** with
  `MOE_HOST_POOL_STAGE_BATCH=0` (+34 %), decode 46.4; forced `DIO=1` 2088/**22.7**; `pinned` 2099/49.3.
  Text is byte-identical across every arm and `MOE_HOST_POOL_VERIFY=1` is clean.  **Long-session adaptation
  was measured and declined**: the oracle-layout ceiling is 7.45 MiB/token of L3 vs 8.05 for a
  previous-topic layout (~6 % per topic shift), and every churning policy costs several times that.
  Record: `archive/work/host-pool-swap-and-prefill/`.
- **MTP gates must be long and reasoning-pinned.** Use `-n 3000` (`-n 2000` floor), `--reasoning on`
  for R and `--reasoning off` for P/C/K. Short runs are a correctness smoke test only. Rule 0 in
  `benchmarks/mtp-adaptive-methodology.md`.
- **Prefill-logit purity is a release gate (issue #113).** Same-seed coherence and MTP are blind to a
  uniform prefill shift. Run `scripts/gate-prefill-logits.sh` before shipping any change to a prefill
  kernel (GDN/SSM chunked, FA, MMQ/MMVF/MMB, MoE prefill) or a default-on approximate path: it compares
  against a recorded known-good base and requires mean KLD <= 0.005 and same-top-p >= 98 %. The r29
  example: the default-on BF16 chunked GDN measured 0.032 / 93.6 % and was flipped off.  A
  **kernel-family switch** (a tile→WMMA head cap) uses `--family-switch` instead: it refuses a point
  outside the gate's calibrated domain (base PPL > 10; at a point the model finds hard even a benign
  same-kernel ubatch change measures 0.105 / 85 %) and always prints a same-kernel control, so the
  absolute limits stay the criterion without an insensitive point waving a bad change through.
  Protocol: `benchmarks/prefill-logit-methodology.md`, `GREEDY-PURITY.md` §46.
- **The draft context's KV cache type follows the main `-ctk` / `-ctv` (r125, block 02).**  Upstream gives
  the speculative draft its own type pair (`-ctkd` / `-ctvd`, default f16) and applies it unconditionally;
  since r125 the draft takes the main model's type unless `-ctkd` / `-ctvd` (or
  `LLAMA_ARG_SPEC_DRAFT_CACHE_TYPE_K` / `_V`) pins it.  At `-c 131072` on a dense 27B that is 512 -> 144 MiB
  of draft KV.  No generated token can change (the draft only proposes; the target verifies) and greedy text
  is byte-identical between the two draft types, but **any VRAM/`--fit` arithmetic for an MTP or `-md` run
  must use the main KV type for the draft context too**, and a *small, fast* model under `-sm tensor` pays
  1.5-3 % decode for it (the quantized KV write costs a few host-side node evaluations per step; GPU kernel
  time is identical).  Record: `WORKLOG.md` 261024.005, issue #216.
- **Mixed K/V cache types are HARD-REJECTED** at context creation (`params.type_k != params.type_v`);
  every mixed pair measured 1.7-3.6x slower and never smaller. Pass matching `-ctk`/`-ctv`.
- **The MTP draft-vocabulary trim is opt-in and draft-only (block 13, r107).**  `--spec-draft-mtp-vocab N`
  copies the LM-head rows for ids `< N` (+ control/user ids) into the draft context, scores the draft over them
  and scatters into a full-vocab row of `-inf`; the target still verifies the full vocabulary, so the trim can
  only lose proposals.  `0` = off = today's code path.  **N is payload-specific** (98304 on Qwen3.8-27B Q8_0,
  64000 is the reporter's gfx1151 value and a *loss* on our 27B, gemma4's only lossless cut is 245760/262144 =
  a no-op), it costs `N/n_vocab` of the head in VRAM (510 MiB at 98304, per device under `-sm tensor`) and it
  is **not** in the `--fit` estimate.  Gate purity at **`n_max <= 7`** (`GREEDY-PURITY.md` §48: the trim moves
  acceptance, which moves the adaptive controller's depth, which moves the verify widths); the shipped
  recommendation is `--spec-draft-n-max 7 --spec-draft-mtp-vocab 98304` (leave `--spec-draft-n-start` at its
  default 3 — it is the *initial* depth, not a ceiling).  **Check the device count before enabling it:** the
  subset head is *mirrored* per device under `-sm tensor` while the untrimmed head is *split*, so it pays only
  when **`N * n_devices < n_vocab`** — on the 27B 98304 is +1.4 % at 2 GPU (share 124160) and **-2.2 % at
  3 GPU** (share 82773, the server shape), and no N pays at 3 GPU.  The `ggml` half is the
  `ggml-backend-meta` row-range read that makes a vocabulary-split LM head readable piecewise.
  Record: `archive/work/spec-draft-mtp-vocab/VALIDATION-2026-10-22.md`.
- **`--spec-draft-n-max` is capped at 15** (the recurrent snapshot bound); purity is promised only to
  **7** (the kernel-family switch at 8 rows). `LLAMA_SPEC_DRAFT_N_MAX_CLAMP=0` escapes the clamp.
- **A draft length is a verify width *and* a memory-tier parameter (r123, TODO #59 closed).**  With host
  experts served by the expert-cache arena, a verify batch wider than the band (`moe_cache_max_tok()`: **16**
  on RDNA4/RDNA2, **8** on RDNA3/NVIDIA) leaves the arena and streams the routed experts from the pinned
  host master on *every* step — one extra row measured **159.2 -> 55.0 t/s** on 2x R9700 `-ncmoe 36`, and the
  README's `draft-mtp-adaptive,ngram-mod` pick 73.9 -> 168.3 t/s once the draft fits.  `common` clamps the
  **draftless** n-gram speculators (`ngram_mod.n_max` 64 / `n_min` 48 default, the ngram-map `size_m`) to
  `band - 1` whenever the arena is active — `LLAMA_SPEC_NGRAM_MOD_BAND_CLAMP=0` opts out, and a
  `draft.n_max` above the band warns (not clamped: its 15 is a correctness bound).  A device-resident model
  or the CPU expert path reports band 0 and is **never** clamped (wide drafts are the win there: 144.7 ->
  378.7 t/s; a naked `--spec-ngram-mod-n-max` below `n_min` silently disables ngram-mod instead).  **The
  clamp runs AFTER `llama_init_from_model`, on the arena's real state (r129, TODO #96):** the model-load band
  only says the slab was *armed*, so clamping there clamped slab-declined (streaming, no-arena) configs.  It
  therefore no longer lowers `cparams.n_rs_batch` — that stays the **unclamped** safety bound (a partial
  rollback can land in any batch up to an unclamped draft; the value also floors the chunked-GDN threshold,
  so an ngram speculator's prefill *tail* ubatches take the sequential bit-identical GDN path) — and the
  slab's narrow verify reserve takes the band-limited width instead (`llama_context::verify_batch_width()`),
  so a live arena reserves exactly what it did before.  Rule:
  `GREEDY-PURITY.md` §50; record: `archive/work/ngram-mod-host-experts/EVIDENCE-2026-10-24.md`.
- **The mmvq band-uniform knobs are a purity requirement *and* a perf trap.** `nwarps` and the vec_dot
  VDR participate in the mmvq K-split order, so one value must hold across the whole band (`W=1..8`)
  and must be tuned at the **verify widths**. RDNA4 dense `nwarps=1`; dense VDR=2, MoE-expert VDR=4.
  The **pinned fusion ops (GDN/SSM, shared-expert, gate fusions) MUST keep plain `calc_nwarps`** — they
  are single-token reduction-order anchors. Run the stock-relative
  `llama-batched-bench -npl 1,4,8` gate (`mtp-adaptive-methodology.md` rule 5) before any decode/mmvq
  change. `GREEDY-PURITY.md` §§19/24/25.
- **The adaptive-MTP controller is the tuned bucketed one** (`v16-d1d3c3396-r2`): `delta = n_accepted -
  depth`, `climb_budget(d) = 20 + 6*(d-1)`, `drop_pressure(d) = max(60, 10*d)`, cold start
  `max(floor, cap-3)`, `--spec-draft-n-start N`. Judge any change on the four axes **and** the
  phase-switching `prompts/code-reasoning-mixed.txt`. Records:
  `benchmarks/2026-09-15-adaptive-mtp-tuning.md`, `archive/work/adaptive-mtp-ceiling-scaling/`.
- **FA instantiation discipline:** a FA kernel's KV *type* axis must be instantiated in the generated
  `template-instances/*.cu` files, never left implicitly in the dispatch TU (it gates the build).
  Check with `nm -C <obj> | grep -c <case>`: the dispatch TU shows `U`, the instance TUs `T`/`W`.
  Detail: `archive/work/build-time-regression/`.
- **Every kq-mask fill is guarded on `tensor && tensor->buffer` (r131, PR #222).**  Block 05's derived kq
  form (`build_attn_inp_kq_mask(..., allow_derived = true)`) leaves the packed mask with **no consumer** at
  one slot, and ggml-alloc gives an unconsumed input tensor no buffer, so an unconditional fill hits
  `llama_kv_cache::set_input_kq_mask`'s `GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer))` and the
  process aborts before the first token (`bailingmoe3` at `-np 1` was the live case).  The guard is the
  file's own idiom (`llm_graph_input_attn_kv` is the origin); three sites lacked it and were fixed in
  `llm_graph_input_mem_hybrid_k` (reachable) and `llm_graph_input_attn_k_dsa` (hardening, `allow_derived`
  false for its `mla`/`lid` masks).  A new fill site, a new `allow_derived = true` builder or a new hybrid
  input class must carry it; `can_reuse` needs nothing (`can_reuse_kq_mask()` reads `ne[]`, never
  `buffer`).  `LLAMA_KQ_MASK_DERIVED=0` is the work-around and the packed reference arm.
- **A Q8_1 quantize-cache key walk stops at the first pointer change, and every fill validates its read
  range (r132, PR #218).**  The cache is keyed on the **root** tensor but the arena is filled from the
  caller's (possibly offset) **view** pointer, so an offset view that walked to the root stored its rows
  under the root's identity and a later op with the same canonical row layout could take them; a stale
  pointer after a compute-buffer reuse then faulted (`MEMORY_APERTURE_VIOLATION`) or read reused memory.
  Every walk is `while (key->view_src && src->data == key->data) key = key->view_src;` (offset-free views
  keep the shared entry — that is the common case and unchanged), and `q8_1_src_range_ok()` (block 08's
  `common.cuh`) aborts with the layout logged if a source's extent leaves its root buffer.  A new cache
  site or a new fused writer that fills the cache for a matmul carries both; there is no knob.  Not
  reachable on the maintainer's configs (0 offset-view keys across seven configs), so its validation is
  code review + the identity gates, not a repro.  `GREEDY-PURITY.md` §55.
- **A per-arch kernel-policy predicate keys on the device it runs on, never on `devices[0].cc` (r132).**
  MMB's `mmb_enabled`/`mmb_cfg`/`mmb_dense_flag`/`mmb_wtype_mask`/`mmb_dense_tmask` read
  `ggml_cuda_info().devices[ggml_cuda_get_device()].cc`; the primary-keyed form froze one arch's policy for
  every device, so a gfx1030 secondary behind a gfx1100 primary ran RDNA3-tuned MMB kernels and zeroed its
  output-projection logits (PPL == vocab).  A table that is per-arch is cached per architecture *class*
  (`mmb_arch_defaults()` discriminates only RDNA4/RDNA3_0, so the fallback table is cc-independent — add a
  branch if that changes).  A pass that consults such a predicate without launching selects its own
  context's device first: `ggml_backend_cuda_graph_optimize` does, because the Meta backend forwards its
  marking half per child and the current device is then still the previously computed one.  On a
  single-arch box the change is a no-op.  `GREEDY-PURITY.md` §56.
- **The host-resident-expert device gather's head pad must match the host path**
  (`min(expert_size, 512)`; **64 is IQ4_NL-only**), keyed on `(input_cpy buffer, expert_bytes)`. Run
  `scripts/gate-qwen4exp-quant-coherence.sh` before changing `moe_cache_gather_host` or the head zero.
  Needs no `MOE_EXPERT_CACHE_MIB`.
- **Never write `std::min`/`std::max` on `int64_t` in `moe-expert-cache.cu` (r124, issue #217).**  With HIP on
  Windows (clang + the MSVC STL) clang's force-included `cuda_wrappers/algorithm` and the MSVC STL
  `<utility>` both define an inline `std::min<long long>` under **one** mangled name, so a TU that
  instantiates both fails to codegen (*"definition with same mangled name"*) — that broke r123's Windows
  build.  Use the file's `moe_cache_min_i64`/`moe_cache_max_i64` (block 09); `std::min` on `int`/`size_t` is
  fine.  Other HIP TUs still contain the int64_t pattern and compile on the reporter's toolchain, so sweeping
  them is **not** sanctioned (TODO #95, accepted limitation — fix a named TU only from a Windows build log).
- **An over-budget config is reported, never silently repaired, and Windows cannot be fixed from here
  (r135, issue #219).**  `--fit` prints the arithmetic at ERROR when it cannot meet its target and the user
  pinned `-c` / `-ngl` (`common/fit.cpp`), and `GGML_CUDA_MIN_FREE_MIB` is the only mechanism that *enforces*
  a device budget: an opt-in free-VRAM floor at `ggml_cuda_device_malloc` that fails an allocation like a
  driver OOM so the fail-soft chain still runs.  It is **skipped while the movable-boundary slab is armed**
  (the slab parks nearly all free VRAM by design), so no `-ncmoe` MoE config changes.  On Windows/WDDM an
  over-committing device allocation is not refused at all (it pages to shared memory and the GPU faults,
  `dxgkrnl` bugcheck); on Linux the same state aborts with `ROCm error: out of memory` or, in a narrow band,
  collapses to ~1/30 speed (TODO #103).  Never read a Linux "it works at 0 MiB free" run as evidence that a
  16 GiB Windows card can run the config.
- **A compute buffer that cannot grow must never abort; the MoE arena is what yields (r20).**
  `GGML_COMPUTE_BUFFER_MARGIN_PCT` (default **10**, `0` disables) pads each COMPUTE buffer allocation so a
  runtime graph's slightly larger layout (+3.3 % measured) does not force a free-then-allocate-larger
  **contiguous** block -- that needs a block *bigger than the one just released*, so free VRAM elsewhere
  (including a smaller arena) does not substitute. The slack is an opt-in buffer-type capability
  (`get_compute_margin_pct`), set only by the RDNA/ROCm path and aggregated (min) by the meta buffer type,
  so no other backend's allocation sizes change. `ggml_cuda_device_malloc` is the single allocation choke
  point: on OOM it frees the largest arena table, then the whole arena, and retries; a failed ramp leaves
  the cache disabled rather than aborting. Keep the two together. Related invariant: a partially-failed
  cache **falls back wholesale** (the fusion guard and `moe_cache_take_over` must agree), and no arena may
  be freed while a kernel that carries its address in its launch parameters is in flight
  (`moe_cache_sync_devices_locked`). `GREEDY-PURITY.md`/`archive/work/moe-cache-autosize/ARENA-UB-TENSION.md`
  §13-§14.
- **The decode/verify band skips fully masked KV groups (r133, PR #223, block 05).**  `GGML_CUDA_FA_BAND_SKIP`
  (default on) lets the band consume the packed fully-masked-group bitmap the prefill path already builds, so
  the band keeps its tuned round-robin KV split (`block i` owns iterations `i, i+P, ...`) and only leaves out
  the iterations whose logits are all `-INF` for every query row of its tile: each block's accumulation order
  over the rest is unchanged, a block that owns no unmasked iteration writes the same neutral `-FLT_MAX/2`
  partial it already had to represent, and the result is bit-identical.  The gate is the band's own predicate,
  so it is RDNA4 + head 256 + GQA 5..12 + `n_q <= 8`; the tile-kernel decode that every other model uses keeps
  the `Q->ne[1] > 8` gate, and `GGML_CUDA_FA_MASK_SKIP=0` disables the whole mechanism (this is its band arm).
  This is the cost `LLAMA_SERVER_LAZY_IDLE_SLOTS` (r130) made common: with a ~105K conversation resident in a
  unified cache, an active 9.8K conversation decodes at **50.8 -> 41.7 ms/step (-18 %)**, token ids identical
  (with a DFlash2 draft at `n_max 5`: 18.6 -> 14.8).  With nothing to skip it is not a tax (the padded tail
  group is skipped too: -0.6 % at 9.8K, neutral at 105K single-slot) — which supersedes the 2026-09-26
  "prefill only, break-even" reading, measured on a 2-slot batched cell with a small foreign fraction.  The
  op-test oracle is the **decode-width packed-hole** case set (`mask_hole=1`, `nb` 1/2/3/8, f16/bf16/q8_0);
  mutating the kernel's group test to skip one visible group fails all 14 of those cases.  Do not widen the
  gate to the tile kernel without that family's own hole cases and a measurement.  `GREEDY-PURITY.md` §37.
- **Idle-slot parking is the sequential-reproducibility fix for #210's lazy idle slots (r136, PR #225, block 12).**
  `LLAMA_SERVER_PARK_IDLE_SLOTS` (on unless `0`, with the lazy idle slots and `--kv-unified`) parks idle slots at
  the top of a unified cache and moves the running slot to the low cells, so `n_kv` and the cell indices match the
  save-and-clear layout; new cells and `get_n_kv()` stop below the parked stack.  It requires one stream, cells no
  other cache shares, V not transposed and no sliding window; anything else keeps #210 and logs once.  A slot whose
  cells are not one run (shared through `seq_cp`, e.g. a cached prefix) falls back to save-and-clear, or stays
  resident with `--no-cache-idle-slots`.  It does **not** make concurrent decoding deterministic (that is §38).
  `LLAMA_KV_PARK_DEBUG=1` logs every move and refusal.  `GREEDY-PURITY.md` §57.
- **The band's two query-width blocks must share one geometry (r134, issue #224, block 05).**  A band block
  splits each Q column's KV rows `np = nwarps*cols_per_warp/ncols` ways (`cols_per_warp` is 16 for the RDNA4
  256-dim WMMA tiles), so the 32-column block (widths 1..4, 8 warps) is `np = 4` and the 64-column block
  (widths 5..8) is `np = 4` only at 16 warps.  r126 deleted the 16-warp row because it was also the prefill
  instantiation; leaving widths 5..8 on prefill's 8-warp table row (`np = 2`) is a different reduction and
  greedy decode with a draft then differs from decode without one (dense head-256 GQA-6, 8K and 110K).  The
  64-column row is back **band-only**, via the compile-time `band_wide` flag + its own case/fetch-ahead
  translation units; prefill keeps the 256-thread table row.  A `static_assert` on the two band rows (`np`,
  `nbatch_fa/K2/V2`, `nstages_target`, `Q_in_reg`) makes a repeat a build error -- do not "simplify" it away.
  And `test-logits-width-probe` must run at **`P >= 4096`**: the wide block's gate is `K->ne[1] >= 4096` and
  `K->ne[1]` is the context's used length, so the historical `P=256` run reported PASS without reaching the
  block (that is how r126..r133 passed).  `GREEDY-PURITY.md` §51,
  `benchmarks/2026-10-25-fa-band-width-purity.md`.
- **The expert cache's tables must not outlive the model that registered them (r143).**  `g_tables` is
  process-global and keyed by host-master pointers the MODEL owns, so `llama_model_free` must drop them
  before the masters go: `llama_model_moe_cache_forget()` -> the `moe_cache_forget` device iface entry ->
  `moe_cache_forget_all(device)` frees each dropped table's arena slice, remap staging, heat mirror, bounce
  ring and host pool, re-points the id maps, forgets the per-model preflight accounting (`g_preflight_device_bytes`,
  the unregisterable set, the latched floor) and un-sizes the cache.  Without it the deferred arena sizing —
  which fills **every** registered table — reads a freed master: a host memcpy **SIGSEGV in
  `fill_slot_from_host`**, measured on Flash-Next GSQ 2 GPU `-sm layer -ts 59,41 -ncmoe 24` (llama-bench
  frees and reloads the model between tests whose params differ!) and in the same shape under
  `--host-experts pool` (whose master is the model mapping, so a host-buffer-free hook cannot cover it).
  Un-sizing matters as much as dropping: a second model used to inherit `g_sized` and stayed uncached for
  its whole run.  A new iface field that appends to `ggml_backend_device_i` goes at the **end** (the
  initializers are positional); a hook that must run at teardown needs a matching call in
  `llama_model_free`/the model lifecycle, not a buffer-free hook.  Single-model runs are unchanged.
- **The expert cache's per-token traffic figure and the graph-replay census are the two instruments for
  "where does the time go" (r144); use them before proposing a memory-side optimisation.**  The exit
  report's `expert H2D traffic … MiB/token` divides a **decode-band op census** (`g_band_ops`) by a table
  count snapshotted **at sizing** (`g_band_n_tables`) — never by a live `g_tables` walk (`llama_model_free`
  drops the tables before the report runs, r143) and never by `g_takeover` (0 on the identity fast path).
  Quote the **2000-token** number: a short run's MiB/token is mostly warmup (25.4 vs 3.3 MiB/token for the
  same config at 200 vs 2000 tokens).  `GGML_CUDA_GRAPH_TRACE=1` additionally reports
  `graph_compute calls: direct/replay/capture/properties_changed` — a **captured graph that never replays
  still logs captures**, so read the census, and confirm with `GGML_CUDA_DISABLE_GRAPHS=1` measuring the
  same speed.  Measured 2026-10-26 (2x R9700 x8, Flash-Next IQ4_NL, `-sm tensor -ncmoe 48`, 85 %
  residency): the MoE miss traffic is 3.3 MiB/token = 0.12 ms of a 28.4 ms token (2.0 ms at the 2 048
  MiB/device floor) — so `TODO #36` (CPU/GPU overlap for the misses) was **retired**, while the token is
  dispatch-bound: 297 `graph_compute` calls / 12 258 node dispatches per token at 1.1 us/node = 13.5 ms of
  host time, with 93 % of calls refused replay (`direct=56171 replay=3152 properties_changed=55321` over
  200 tokens) because the Meta rebuild moves node `src[]` object addresses, then inline `name`, then a
  genuine source `data` pointer.  Do **not** "fix" that by relaxing the property comparison alone: it
  enables recaptures (`hipGraphExecDestroy`+`Instantiate` per token on HIP) and measures **slower**
  (28.3 vs 35.2 t/s).  Campaign, gates and traps: `wip/graph-replay-placement/`, `TODO #107`; records
  `archive/work/moe-cpu-overlap/`, `WORKLOG.md` 261026.001.
- **On 2+ devices `-sm tensor` is not just "the tuned path", it is worth tens of percent at low residency
  (r144).**  2x R9700 on the x8 root ports, Flash-Next IQ4_NL `-ncmoe 48`, 2000-token decode: **35.5 t/s
  (`tensor`) vs 25.7 (`layer`) = +38 %** at the same expert hit rate (99.74 % vs 99.48 %), so the cost is
  the per-layer cross-device split and its split/event/handoff boundaries (~50 graph splits/token), not
  expert residency.  `layer` is the CLI default, so a multi-device `-ncmoe` config must pass
  `-sm tensor` explicitly; a *dense* cross-device-count text comparison still needs `-sm layer`
  (see "Greedy text is a per-device-count reference").  `COMMUNITY-CONFIG.md` (the split axis).
- **The movable-boundary slab is MoE-only and must stay that way (r37).** `ggml_cuda_slab_enabled()`
  returns `env_on && g_slab_armed`; `g_slab_armed` is set **only** by the MoE preflight
  (`ggml_backend_cuda_device_moe_cache_preflight`), which `llama_model_moe_cache_preflight` calls only for
  models with a non-empty `model->moe_host_expert_bytes` (`-ncmoe`/`-cmoe` > 0).  The slab exists to let the
  work region and the expert-cache arena coexist; with no arena it parks nearly all free VRAM in an unused
  region (measured ~18 GiB/GPU on a dense 27B).  The **compute chunk** (`GGML_COMPUTE_BUFFER_CHUNK_MIB`) is
  slab-motivated and is gated on the same condition.  Do **not** re-enable either for dense models; the
  dense regression is issues #118 (slab before the weights OOMs a 48 GiB card under `--mmproj`) and #120
  (the chunk cost a dense config +739 MiB and tipped it into shared memory).  The percentage margin
  (`GGML_COMPUTE_BUFFER_MARGIN_PCT`) is deliberately *not* gated (kept for dense).
- **An upload that writes a scheduler-managed buffer goes on the main stream (r62, block 10).**  The
  scheduler's `wait_before_overwrite()` (a wait on the previous split's event) guards the graph
  allocator's reuse of a split-input buffer and is enqueued on the **main** stream.  The compact
  splice-gather's 1-D H2D ran on `cuda_ctx->copy_stream()` and only ordered the *consumer*, so it could
  overwrite the buffer while the previous split still read it -- garbage/NaN for `-sm tensor` + `-ncmoe`
  on the below-gate path (35B-A3B, gemma4, Flash-Next GSQ).  A side stream is only safe with an explicit
  wait on the main stream recorded *before* this upload; the gather has no such need (it must wait for
  the reused-buffer read anyway).
- **The async graph-input ring (`GGML_CUDA_ASYNC_INPUT`) is opt-in and a run is not start-to-start
  deterministic when the consumer is a captured HIP graph** (r69; re-characterised 2026-10-19).  `-ub 8`
  and `-ub 1` self-KLD comparisons diverge with `n_seq = 1`; `-ub 512` and `GGML_CUDA_DISABLE_GRAPHS=1`
  are bit-identical.  **The cause is NOT the ring**: the uploads verify byte-correct at the graph's first
  read, HIP's wait-before-replay / event-after-replay ordering is correct, and consecutive graphs do not
  overlap — so the async path is the *trigger* (it removes the per-copy host block), not the defect.
  **FIXED in r90 (TODO #73 closed)**: the defect was `ggml_backend_cuda_set_tensor_async` /
  `_set_tensor_2d_async`, which issued a pinned-source `cudaMemcpyAsync` **straight from the caller's live
  host pointer** while the scheduler uses that path for a host-resident split input -- a CPU split's
  output that the next ubatch rewrites in place -- so with the host running ahead the copy could read the
  next ubatch's bytes (a rare single-graph corruption that propagated to the end of the chunk).  Both now
  snapshot through the pinned upload ring, unconditionally, and the ring's wait/done guard is
  unconditional; `GGML_CUDA_ASYNC_INPUT` is **default-on** (`=0` disables) and the batched uploads
  (`GGML_CUDA_ASYNC_INPUT_BATCH`, r89/PR #182) stay default-on inside it.  Self-KLD control 91.0-95.1 %
  -> 99.967-100.000 %; decode +9.3 % (35B-A3B 2 GPU tensor tg400) / +5.2 % (27B MTP n3 end-to-end).
  Evidence: `archive/work/async-input-graph-ordering/` (`RESOLUTION.md`, `FINDINGS-2026-10-19b.md`).
- **The batched state save and restore share one pinned `state_stage` per context (r96/r119, PRs #189/#209).**
  `LLAMA_STATE_ASYNC` covers saves, `LLAMA_STATE_ASYNC_READ` (default on) covers restores, and
  `LLAMA_STATE_ASYNC_MAX_MIB` bounds both.  Saves and restores on one context must not run concurrently.
  The restore is a multi-device win (2 GPU `-sm tensor`: 35B-A3B 49 -> 40 ms, Flash-Next 89 -> 75 ms);
  one GPU pays about 2 ms for the extra staging copy, so do not read the single-GPU number as a defect.
- **The op-offload H2D staging ring (V2.0 block 09; old block 06, r12) has three invariants:** (1) prefill is never
  CUDA-graph captured, so a redirected split input that reaches a copy path aborts (tripwire asserts
  this); (2) the width gate is floored at **64 tokens**; (3) the arena lives **outside** the
  compute-graph reserve, so a failed allocation disables staging instead of aborting, and a partially
  staged ubatch is never allowed. Under `-sm tensor` the meta backend owns one ring per device and
  `MIRRORED` tensors must serve arbitrary byte ranges. The expert weights themselves are **split** per
  device, not mirrored (`GGML_META_SPLIT_COPY=1`; each cache table holds `expert_bytes = host_bytes/2`,
  measured 2026-10-07), so any `-sm tensor -ncmoe` vs `-sm layer` gap is the per-op upload/pruning
  machinery, not weight duplication. (`TODO.md` #44 was closed on that finding; see
  `archive/work/expert-cache-split/`.)
- **Cross-device split inputs (V2.0 block 09; old block 06, r12):** a device-to-device input copy runs on the SOURCE
  backend's stream, so `wait_before_overwrite` (which only orders the destination stream) is not
  enough; the block records a fresh event on the destination backend and makes the source wait on it
  before the copy. Without it, `-sm layer` on 2 GPUs with host experts (`--n-cpu-moe` >= 29) can
  overwrite an earlier split's output before its outbound copy reads it (garbage prefill);
  `GGML_SCHED_EVENTS=0` is the old workaround. Issue #103, contributor PR #104,
  `archive/work/2gpu-sched-fixes/VERIFICATION.md`.
- **The QSA op has a CPU oracle** (`test-backend-ops -o FLASH_ATTN_QSA`, 18/18 minimum). The `W=1..8`
  matrix is **blind to a width-uniform corruption**, and MTP acceptance is not a quality signal when
  draft and target share the defect; use the perplexity ratio vs the dense masked path
  (`LLAMA_QSA_SPARSE_FA=0`). A fused op with several heads sharing a staging buffer must keep the block
  homogeneous in every index the staging reads (QSA: the K/V head). `GREEDY-PURITY.md` §21.
- **Block 03 (GDN; old block 02):** the chunked prefill is K-independent with threshold `max(K>16?K:16, n_rs_batch)`,
  `n_rs_batch = common_speculative_n_max()+1`; `GGML_CUDA_GDN_CHUNKED=0` forces the sequential kernel
  (correct, bit-identical, slow). The pure `none == draft-mtp` range is `n_max <= 7`.
  **Our chunked kernel is the only one the delivery dispatches, on every device class** (r127 gfx1151
  A/B): upstream's `gated_delta_net_mma` (in the 2026-10 window) *is* eligible on RDNA3.5 at
  `n_tokens >= 2048` (qwen35's `H_k = ssm.group_count = 16` satisfies its gate) but our branch returns
  first — measured gfx1151 pp2048 interleaved **3232 vs 3131 t/s (+3.2 %)** for ours, prefill-logit
  **KLD 0.000074 / 99.637 %**.  Upstream's MMA only runs behind our stand-down (`kda`, `S_v` outside
  {16,32,64,128}, or the `=0` kill switch): never add a second chunked kernel per device class.
  `GREEDY-PURITY.md` §27, `patches/README.md` (block 03), `WORKLOG.md` 2026-09-12.
- **The rollback bound is enforced by `n_rs_batch`, not observed — do not re-derive it.**  `common`
  sets `cparams.n_rs_batch = common_speculative_n_max() + 1` over **every** speculator type (draft
  simple/mtp/adaptive/eagle3/dflash/dspark *and* ngram simple/map/mod/cache, so ngram-mod's default
  n_max 64 gives 65), `llama_context` floors it at `n_rs_seq + 1`, and `delta-net-base` passes it as
  GDN op param 1.  The op's threshold is floored at it, so a **speculative verify batch can never take
  the chunked path** and always restores snapshots it wrote (r91; the campaign's "ngram is an exposure"
  claim was wrong).  The r91 sequential tail (`GGML_CUDA_GDN_TAIL`, block 10) then covers the remaining
  source of a partial rollback: a **non-speculator** rewind into a prefill batch (server cancel /
  `--cache-reuse`).  The per-sequence tracker is block 03's (`prepare()` folds in every ubatch;
  `LLAMA_RS_ROLLBACK_DEBUG=1` shows `RSPREP`/`RSROLL`).  Record + probe:
  `archive/work/rs-rollback-boundary/`.
- **Test baselines are per box *and* per model: never compare to a number in this file.**  E.g.
  `test-recurrent-state-depth` on Qwen3.6-35B-A3B-UD-Q3_K_M: **130** failures on gfx1201, **153** on
  gfx1151; the gate is that the per-arm FAIL/PASS set is unchanged, not that the count is zero.  Run it
  (and any other GPU test) **alone** — two GPU jobs at once changes the arena/band layout and invents
  failures.
- **Block 01 (all-reduce; old block 12):** the `GGML_CUDA_AR_PROFILE` device-list init fix; on the first NCCL runtime failure the
  comm layer warns once, disables NCCL and re-routes to the internal pipeline/butterfly.
  `GGML_CUDA_ALLREDUCE=nccl` is **NOT bit-identical** under `-sm tensor` (the internal AR BF16-round-trips).
- **Block 08 MTP regression fixes (old block 13):** the dense K-split kernel for verify batches, and the
  rms_norm->mmvq Q8_1 fold gated to single-token MMID + plain `MUL_MAT`. **Purity ranks above raw
  non-MTP throughput.** `GREEDY-PURITY.md` §19.
- **MoE decode/verify is byte-identical by default:** the fused shared-expert band serves
  `1 <= nt <= 8` with `nwarps` pinned; `GGML_CUDA_DISABLE_SHEXP_DOWN_GATE=1` is the A/B kill-switch.
- **The RMS-norm reduction order is a numerics reference, not a knob (r145, issue #229).**
  `ggml_cuda_rms_norm_vec()` (`common.cuh`) is the single selector: **default OFF = the pre-#29720 strided
  scalar reduction**, `GGML_CUDA_RMS_NORM_VEC=1` = upstream #29720's float4 grouping (which the r127 re-base
  had made unconditional).  A reduction-order change moves every layer by ~1 ulp and greedy MTP draft
  acceptance by a payload-dependent amount (issue #229: -7..-9 % tg on the reporter's box; +0.7..+1.3 points
  here, but 0.611 vs 0.792 acceptance / -18 % t/s with the SSM decode fusions disabled), so float4 is worth
  **+1.0-2.7 % prefill and ~0 % decode** and is not a shipping default.  The strided reduction is also what
  every decode-band fused norm kernel is documented bit-identical to (the rms_norm->Q8_1 fold,
  `rms_norm_scale_pair_f32`, `rms_norm_mul_rope_f32`, gdn-conv's folded q/k norms, the GDN norm gate,
  `hc_combine_norm_f32`) - under the float4 default three of them were silently *approximate*
  (0.0005 mean KLD at `-ub 1` where the contract says 0).  A new fused norm kernel owes that probe
  (`scripts/gate-prefill-logits.sh --ab "<fusion env>" --ub 1`, must read 0.000000 / 100.000 %), and every
  kernel standing in for `rms_norm_f32` must consult the selector rather than hard-code a grouping.
  `GREEDY-PURITY.md` §60; `ENVIRONMENT.md` §7.5; `benchmarks/2026-10-26-rms-norm-float4-ab.md`.
- **The r25 tensor-split fold (PR #106 minus its patch 0002) has three invariants to keep together:** a
  retired **Q8_1 arena must outlive captured graphs** (`GGML_CUDA_Q8_1_ARENA_FREE_OLD=1` restores the
  `quantize_q8_1` page fault); a graph is **recaptured when the pool / FA-staging / H2D-ring memory it
  captured is freed** (`GGML_CUDA_GRAPH_MEM_GEN=0` disables); and the **meta split-state cache keeps 8
  versions with compact entries** (`GGML_META_SS_VERIFY=1` verifies hits).  **PR #106 patch 0002 (the
  data-pointer graph key) is deliberately NOT in the delivery**: on the r22+ movable-boundary slab the
  per-device allocations vary between otherwise-identical calls, so the extra key invalidates warm graphs
  and recaptures (measured -19 % `-sm tensor` MTP at `--spec-draft-p-min 0.5`, with identical text and
  acceptance).  Re-cut it against the slab before revisiting.  `TODO.md` #46, `WORKLOG.md` 2026-10-07 (r25).
- **A captured graph is only valid while the slab work region's layout is unchanged.**  The work region is a
  movable boundary: at the post-prefill drop the narrow re-reserve shrinks it and the bytes above the new
  boundary go back to the MoE arena.  A captured CUDA/HIP graph bakes the work-region addresses into its
  executable graph, so the shrink **must** bump `ggml_cuda_graph_mem_freed(device)` (block 10,
  `ggml_cuda_slab_work_alloc`) — otherwise a later replay reads released memory.  Issue #187: a 4-stream MTP
  decode replayed a graph whose KV `set_rows` source indices pointed into the released wide region and
  faulted with `HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION`; `GGML_CUDA_DISABLE_GRAPHS=1` hid it.  Every
  shrink path taken at a drop still needs this; add it wherever a new one is introduced.
- **Under `-sm tensor` the HIP graph cache must be keyed on a *stable* identity — the Meta backend
  re-creates its per-device simple tensors on every rebuild (r115, TODO #51).**  `ggml_cuda_graph_get_key()`
  is `{ first node's `data` pointer, n_tokens }` and must stay that way (`GGML_CUDA_GRAPH_KEY_DATA=0`
  restores the old object-address key, A/B only).  With the object address the cache never warmed: 3954
  keys / 1206 captures / 484 ms of `hipGraphInstantiate` per 300-token decode at 3 devices vs 67 keys / 9 ms
  with `-sm layer` — the user-visible 200-450 ms 3-GPU stutter, and the reason 3 GPUs decoded slower than 2.
  Keyed on `data`: 144 keys / 15 ms and **+41 %** at 3 devices (+6 % at 2, unchanged at 1), identical text
  and acceptance.  Any new *fingerprint* added to the key (an object address, a generation counter, a
  per-allocation pointer) re-introduces the churn if it moves across a rebuild — the r25 PR #106-0002
  data-fingerprint key that was dropped for the slab is the same trap from the other direction.
  **Since the `3f03b1772` re-base the captured graph is a *vector* of per-chunk executables**
  (upstream's chunked CUDA graphs, #30804): `ggml_cuda_graph` has no `graph`/`instance` members, capture
  ends inside `ggml_cuda_graph_launch_chunk()`, and anything that used to run "just before
  `cudaStreamEndCapture`" (the WIP AR-stage hook) now runs before the **last** chunk's launch.  The HIP
  `hipGraphExecUpdate` leak workaround (`GGML_HIP_GRAPH_FORCE_UPDATE`) lives inside upstream's
  `ggml_cuda_graph_update_executable(cudaGraphExec_t &, cudaGraph_t)`, i.e. it re-instantiates one chunk
  per recapture.  3-GPU 27B MTP decode measured 84.2/84.8 t/s at rung 8 vs 79.8 before it (+5.5 %),
  text unchanged.
  Record: `archive/work/graph-key-churn/`.
- **There is NO periodic per-device decode stall left after r115 (TODO #89 closed, 2026-10-22).  Do not
  re-chase it.**  The ~300 ms that `CUDAGC_SLOW` still reports at 3 devices is a **one-time lazy HIP
  code-object load per (module, device)** (~55 ms of it is the `mul_mat_q` object, which holds mmq *and*
  mmvq), paid inside `common_init_from_params`'s existing 2-token warm-up for the decode families and
  inside the first real prefill for the prefill-only ones.  The count is identical at 300, 1000 and 1500
  generated tokens — it does not scale with tokens, only with device count, and `--no-warmup` moves the
  same cost onto the user's first step (which is what the warm-up exists to prevent).  A first use
  inflates the whole window (`RMS_NORM` 4.7 ms right after the FA load) because further enqueues block
  behind the load; `GGML_CUDA_DISABLE_GRAPHS=1` changes nothing.  `CUDAGC_SLOW`'s 5 ms threshold is per
  child-graph launch and is **not** device-count normalised, so on 1 GPU ordinary 4 000-node steps are
  reported — judge per node (~1-2 us) or by `FIRSTUSE`.  Also measured there: `hipFuncGetAttributes` in
  these runs is **librccl** (NCCL init), not ours.  Tooling (probe extension + an `LD_PRELOAD` HIP
  call-site interposer): `archive/work/decode-warmup-stall/`.
- **A `-sm tensor` model load uploads tensors through a pinned staging buffer, one alloc/free per
  tensor** (`ggml_backend_cuda_buffer_set_tensor_2d`, 678 pairs = 2.2 s + 1.1 s of a ~10.7 s load on a 27B
  Q8_0 at 3 devices).  This is *load* time, not decode.  Since **r116** the staging buffer is grow-only
  per device (~5 allocations, 12.7 s -> 9.5 s, block 08; `GGML_CUDA_STAGE_PIN=0` is the kill switch).
- **The deepseek4/glm5-next `DSV4_HC_POST` arm is split-mode dependent (r97).**  `cparams.fused_dsv4_hc_post`
  is the fused **op** whenever a Meta (tensor-split) device is in play and the primitive
  `hc_combine_norm` chain on one device.  The chain is faster on one device (+0.9..+1.8 % prefill,
  gfx1151) but is **not correct across a tensor split**: `test-llama-archs -a deepseek4` on 3x gfx1201
  read NMSE 1.74e-01 through the chain vs 6.5e-06 through the op.  Do not make the chain unconditional
  again without fixing its split-state handling; `LLAMA_FUSED_DSV4_HC_POST` forces either arm.
- **A host buft with no device is never accounted as host experts (r97).**  `ggml_backend_cpu_buffer_type()`
  has `.device = NULL` upstream; the account map, the slab, the H2D ring and the expert cache all need a
  **pinned** master read in place by a device kernel, so a pageable placement (CPU-only, or
  `--override-tensor exps=CPU`) is left to the upstream `llama_moe_cache` engine.  Keying a null device
  aborted in `ggml_backend_dev_memory()`; keying the layer's device reserves a ring/arena the path cannot
  use.  `patches/README.md` (block 09).
- **Dense greedy purity:** `plain == draft-mtp` greedy text is kept for **f16/bf16/q8_0**; the coarse
  quants (q4_0/q4_1/q5_0/q5_1/iq4_nl) are relaxed to the **logits level** (argmax preserved, top-2
  margin >= 2.2). The FA chooser is TILE across the whole band. Re-run the 8-type x 5-length grid when
  a single-token-tuned kernel changes. `GREEDY-PURITY.md` §36.
- **The high-power pin regressed** (`~/bin/high-power`: tg -5-7 %, pp -15-18 % on RCCL/hybrid paths).
  Servers run **unpinned**, 3-GPU (`HIP_VISIBLE_DEVICES=0,1,2`), hybrid default.
- **There are TWO fused gate/up+GLU MMQ arms in one dispatch, and they are arch-disjoint (r138, block 08).**
  Upstream #29948's arm (`has_fusion`: gate rows interleaved into the weight tile by the loader, GLU in the
  MMA write-back, `I_dst = I/2`) is **NVIDIA-only by construction** — its predicate ends in
  `turing_mma_available()` and its write-back depends on the MMA accumulator layout — and the delivery's
  arm (`has_gate`: the matmul runs twice into two accumulators, GLU in registers, `J` capped per type) is
  the **RDNA** one.  Both live in `mul_mat_q_process_tile`/`mul_mat_q`/`mul_mat_q_case` as *separate*
  template parameters, with `static_assert(!(has_fusion && has_gate))` making "one fused path at a time" a
  build error, and a single J selector (`ggml_cuda_mmq_select_J`, which upstream's
  `ggml_cuda_mmq_get_J_best` now delegates to).  Do not "simplify" this to one arm: enabling the interleaved
  arm on RDNA is unvalidated and order-dependent, and dropping the delivery's arm loses the RDNA prefill win
  (pp2048 +5.3 % gfx1151 / +9.4 % gfx1100).  `ggml_cuda_should_fuse_mul_mat_q` is non-static and declared in
  `ggml-cuda-try-fuse.cuh` because block 10's moved dispatcher calls it.  Record:
  `archive/work/rebase-2026-10b/DECISIONS.md`.
- **A combine+norm style fusion must read the tensor the model PINS, never an unpinned ancestor (r140, issue #226).**
  The repeat-anchored `hc_combine_norm` matcher starts its window at the `REPEAT`, so `scale1 -> sigmoid ->
  scale2` run standalone *before* it; a tensor read from before the window is only safe while the graph keeps
  it alive, i.e. only when it is a graph output.  r138's #31028 merge turned `inject` into a `SCALE` node and
  the matcher descended to `scale1->src[0]`, whose buffer the allocator then handed to the chain's own
  SIGMOID — a pre-#31028 (separate-tensor) GGUF read a sigmoid output as `inject` and its PPL went to 2690.
  The fix reads `inject = scale1` (the pinned tensor) with identity params (`s1 = 1`, `b1 = 0`) and requires
  `GGML_TENSOR_FLAG_OUTPUT`.  The **scale1-anchored A-chain** matcher is deliberately different: its window
  starts *at* `scale1`, so that node is elided and `scale1->src[0]` is the live read — do not unify the two.
  A new emitter that hands a `SCALE` to such a matcher must pin the tensor the matcher reads, or lose the
  fusion.  **r141 closes the two halves r140 left open:** the fused kernel now owns the same columns in the
  RMS sum as `rms_norm_f32`'s vec4 path (it was 0.0185 KLD / 95.6 % away from its own chain, now
  `0.000000 / 100.000 %`, and the fused arm's greedy text is byte-identical to
  `GGML_CUDA_DISABLE_HC_COMB=1`), and a **merged-tensor** (post-#31028) GGUF takes the fusion again because
  `build_hc_mix` materialises the inject slice with `ggml_cont` and the matcher accepts a pinned
  `view_src == nullptr` tensor instead of requiring a `SCALE` (a pin on a *view* does not keep the root
  alive — test that any new inject shape satisfies it).  `GREEDY-PURITY.md` §59; records `WORKLOG.md`
  261025.014 / 261025.015.
- **The qwen4exp HC mixer's two BF16 GEMMs share one converted src1, so the model must emit them
  adjacent (r142, issue #226 follow-up).**  `build_hc_mix`'s separate-tensor branch (pre-#31028 GGUF) runs
  `hc_down` and the 4-row `hc_inject` over the same F32 `xn`, and the CUDA BF16 GEMM path caches **one**
  converted src1 (`bf16_src1_cache`, block 10) — so the reuse only happens while nothing converts in
  between.  Upstream #31028's emission order puts the inject product *after* the gate product (whose src1
  is the silu output, which evicts the entry), which converted the whole `[hc*n_embd, n_tok]` tensor twice
  per mixer: measured on Flash-Next GSQ-IQ3_XXS (3 GPU `-sm tensor`, all-VRAM) as 4560 vs 2280
  `convert_unary_cont_vec4<float, bf16>` launches and **-3.4 % prefill**; the fix emits `lo` and the inject
  product back to back and expands **both** there (`ggml_build_forward_expand` order, not node creation
  order, is what lands a node in the graph).  Bit-identical (PPL, all 24 chunks of the reporter's command,
  6/6 greedy cells), `GGML_CUDA_BF16_SRC1_CACHE=0` is the A/B control.  The merged path is one GEMM and is
  untouched.  Record: `benchmarks/2026-10-25-qwen4exp-hc-inject-adjacency.md`,
- **qwen4exp's `hc_*_down`/`hc_*_inject` pairs are ONE tensor since r138 (upstream #31028).**
  `create_merged(LLM_TENSOR_HC_{ATTN,FFN}_DOWN_INJECT, …)` accepts the converter's merged weight and
  `build_hc_mix(…, w_down_inject, …)` does **one** GEMM for both products; the old separate tensors are
  still accepted (the loader falls back when the merged tensor is absent, which is why pre-r138 GGUFs keep
  loading).  A `build_hc_mix` caller must pass `layer.hc_*_down_inject` — a call that omits it compiles
  against the old 7-argument form and silently reads the wrong weight.
- **The set applies whitespace-clean** (`apply-all.sh`, no git whitespace warnings).

## Common tasks

### Apply the set to a fresh llama.cpp checkout

```bash
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
git checkout <release.json.base>
bash <this-repo>/scripts/apply-all.sh .     # creates branch rdna-boosts, 14 commits

# with the beta candidates on top (NOT the release tree; for a beta test only)
bash <this-repo>/scripts/apply-all.sh --include-beta .   # 14 blocks + beta/*/*.patch (15 commits)
```

### Verify (the coherence gate — mandatory after any change)

Local models live under `/llm/models/` (e.g. `Qwen3.5/4B/Q8_0/...`, `Qwen3.8/27B/Q4_K_XL/...`,
`Qwen3.6/35B-A3B/Q8_0/...`, the Gemma4 tree). Paths in older docs are host-specific; resolve them
against `/llm/models/`.

```bash
HIP_VISIBLE_DEVICES=0,1,2 ./build/bin/llama-cli -m /llm/models/Qwen3.5/4B/Q8_0/Qwen3.5-4B-Q8_0.gguf \
  -ngl 99 -sm tensor -mg 0 -p "The capital of France is" -n 20 \
  --seed 42 --temp 0 --no-display-prompt --single-turn
```

Diff against a known-good build: same-seed output must be IDENTICAL. `GGML_CUDA_ALLREDUCE=nccl` is not
a bit-identical reference under `-sm tensor` (see Block 01 above); treat it as a smoke comparison.

The coherence gate is a decode/greedy check and is **blind to a uniform prefill-logit shift**. For any
change to a prefill kernel (GDN/SSM, FA, MMQ/MMVF/MMB, MoE prefill) or a default-on approximate path,
run the **prefill-logit gate** as well:

```bash
./scripts/gate-prefill-logits.sh          # compare the current build against the known-good base
./scripts/gate-prefill-logits.sh --record # (re)record the base from a known-good build
./scripts/gate-prefill-logits.sh --ab "GGML_CUDA_GDN_CHUNKED_BF16=0"  # exact-fallback A/B
```

It requires mean KLD <= 0.005 and same-top-p >= 98 %. Issue #113 (the lossy default-on BF16 chunked
GDN) passed the coherence gate and was only visible to this one; protocol in
`benchmarks/prefill-logit-methodology.md`.

### Regenerate the patches (after fork changes)

`scripts/make-patches.sh` (defaults read from `release.json`) `git format-patch --start-number 0`s the
block commits (block 00 keeps prefix `0000`; `git diff <base>..<tip>` yields `rdna-boosts-all.patch`).
Always regenerate from a canonical fork rebuilt at `release.json.base` via `scripts/apply-all.sh`:
the working `~/llama.cpp` branch is not the canonical chain and a raw range there can export upstream
commits. Then re-verify the clean-apply simulation (fresh worktree, apply, build, coherence, and
`scripts/gate-prefill-logits.sh`).

### Build the fork

```bash
cd ~/llama.cpp && ~/bin/build-llama-rocm          # default BUILD_DIR=build-rocm, ROCm 10.1.0
# fast loop: cmake --build build-rocm --target llama-cli llama-bench -j 16
# runtime libs: LD_LIBRARY_PATH=/opt/rocm-10.1.0-gfx120X/lib
```

**Keep the ccache large enough for a per-block build or it thrashes.**  The tool's default
`max_size = 5 GiB` is far too small once one source tree carries several `GGML_CUDA_FA_*` instantiation
sets: `scripts/gate-per-block-build.sh` then misses on nearly every FA translation unit and evicts the
entries the next block needs (measured 2026-10-21: block 05 went from ~45 min at 5 GiB to ~7 min after
`ccache --max-size=30G`).  Set it once per box — and note it is **not** the whole story (r138): the
cache sits at ~14/30 GiB with hundreds of cleanups, so the previous release's block trees are evicted
before the next one and every re-base pays a cold ladder.  On a box with the disk, 60 GiB keeps two
release ladders resident.

- **A re-base is a decode AND prefill change: measure throughput before the tag, not only correctness
  (r138).**  The per-block build, the prefill-logit KLD and the op tests prove the kernels are *faithful*;
  none of them measures speed, and a re-base can change launch geometry without changing logits (r138's
  hand-merged MMQ tiling, the qwen4exp one-GEMM HC path, rope).  Before tagging, run the stock-relative band
  gate (`llama-batched-bench -npp 512 -ntg 128 -npl 1,4,8`) **and** an rNN-1-vs-rNN prefill A/B
  (`llama-bench -p 2048,8192 -n 0`) on a build pair configured identically — including the same
  `GGML_CUDA_FA_QUANTS` set (see the next bullet; a differing set alone changes greedy text).  Keep the
  previous release's binaries (`build-<tag>/bin`) so the A/B does not need a rebuild.  Record in
   `benchmarks/` + a `WORKLOG.md` entry.  r138's A/B:
  `benchmarks/2026-10-25-r138-rebase-ab.md` (no regression; dense prefill +1.4-2.3 %).
- **The per-block ladder's cost is ONE translation unit, and it is a test (r138, measured).**
  `tests/test-backend-ops.cpp` is 14.4k lines in a single TU and takes **5m13s** to compile cold at
  `-j20` (a `src/*.cpp` TU is ~5 s).  The gate's default target set is `all`, and a re-base changes that
  TU through the base and through blocks 04/05/07, so the ladder sees ~4 *distinct* contents of it:
  ~21 of the ~35 minutes are spent compiling a test binary that says nothing about whether a block's
  delivery code is sound.  Measured full ladder: **2101 s cold, 95 s warm**; `--fast`
  (`ggml-hip llama llama-cli llama-server llama-bench llama-perplexity`) is **78 s warm** and skips the
  cold test-TU compiles entirely.  **The release flow uses `--fast`, and covers the rest at the tip**:
  build the tip with `all` once (the normal `~/bin/build-llama-rocm` build does) and run
  `test-backend-ops -o …` there.  `--fast` is *not* the r2 "every block builds `all`" claim and its
  result line says so — use plain `all` when a commit touches `tests/` or a shared header, or when the
  strongest statement is wanted.  An ordinary release does not pay the ladder at all: pass
  `--changed-from <previous tip>` (r137: 1 block, **8 s**).

- **Never give an arm its own source path: ccache keys on it (measured 2026-10-25).**  The same r132
  content rebuilt in a **new** worktree path costs **5m56s wall / 85 CPU-min, all misses**; the same
  content rebuilt at the **same** path in a **fresh build dir** costs **8.6s, all hits**.  A wiped build
  dir is free, a new source path is a full cold build, so a release that assesses two PRs and folds them
  must not use four worktrees (that is what turned ~20 min of work into ~1-2 h of compiling in r132).
  Use ONE long-lived checkout for every arm: `scripts/scratch-tree.sh` switches states inside it
  (`state <rev>`, `blocks`, `arm <patch>...`, `tip`, `keep <tag>` for an A/B, `gate` to run the per-block
  gate there), and `~/llama.cpp` stays the canonical path so the gate's default `--repo` shares the cache.
  The honest budget with that rule: one `.cu` delta 15-30 s, a widely included header (`common.cuh`,
  `ggml-impl.h`) 3-5 min for its dependents, a whole new tree 5-6 min, and a per-block gate run is the
  sum of the selected blocks' own deltas (r132, at the long-lived path: block 06 took 34 s).
- **A scratch build dir and `~/llama.cpp/build-rocm` can hold different `GGML_CUDA_FA_QUANTS` sets, and
  that changes greedy text (r138).**  The FA instantiation set is a CMake **cache STRING**, so it keeps
  whatever the tree's default was when that build dir was first configured: at r138 the two dirs held
  `q4_0;q4_1;q5_0;q5_1;q8_0;bf16;f16;iq4_nl` (canonical) vs `q4_0;q8_0;f16;bf16` (scratch) from the same
  source tree.  Different instances ⇒ a different FA kernel for a shape ⇒ different greedy text, with
  **identical prefill logits** (the gate read KLD 0.000531 / 98.863 % on both).  So: compare text only
  between builds of the **same** build dir, run the prefill-logit gate with the **canonical** build
  (`gate-prefill-logits.sh` defaults to `~/llama.cpp/build-rocm/bin/llama-perplexity` for exactly this
  reason), and never read a cross-build text difference as a numeric change without checking
  `grep GGML_CUDA_FA_QUANTS <build>/CMakeCache.txt` on both sides.

**`~/bin/build-llama-rocm` (ROCm 10.1.0, `/opt/rocm-10.1.0-gfx120X`) is the canonical build tree.**
`~/bin/build-llama-rocm-714` is the older ROCm 7.14 tree (its own `BUILD_DIR`, `build-rocm-714`); it is
still valid for a 7.14-specific check, but do not present it as the default. **Do not pass
`EXTRA_CMAKE_FLAGS="-DCMAKE_HIP_FLAGS="`** (or any `-DCMAKE_HIP_FLAGS` override): both scripts pass a
joined `-isystem<rocm>/include` there and abort the configure if the flag does not reach the HIP compile
line (the include-order guard). That guard exists because the failure is otherwise silent -- HIP headers
resolve from Fedora's ROCm 7.1 in `/usr/include`, and the resulting binary still produces identical
greedy text on 7.14. The old CMake >= 4.3 problem (a bare `-mllvm` swallowing the injected
`--cuda-host-only`) is already handled by not passing `-mllvm` at all. **ccache is on by default** when
present (the build dir is reused incrementally unless you pass `clean`; ccache makes even a wiped
rebuild of unchanged sources a cache hit).

The FA instances are deliberately force-inlined — do **not** outline the FA loader
(measured: a universal 1.5-2.5 % prefill loss). This is the sanctioned answer to "the build is slow";
detail in `archive/work/build-time-regression/`.

### Build the release container images (worker by default; local only for smoke)

The ROCm container images are built and pushed **off-host** by the worker watcher
(`scripts/container-watch.sh` on `radiance`), which polls the delivery repo for a new `v12-*` tag and
runs `scripts/build-containers.sh --rocm "7.14.1 10.1" --push --prune` on that box.  This keeps the
primary box free for benchmarks, and a tag-push CI build is not used either (~5 h for two ROCm lines,
and a failed/timed-out run wrote no registry cache so every retry was cold).  **After `git push` of
the tag, nothing else is needed: the worker picks it up within 10 minutes.**  The commands below are
for the worker's one-time setup, or for a smoke build on the primary box; **do not run the full
`--rocm "7.14.1 10.1" --push` on the primary box during a release.**  The fast CI workflows stay:
`validate.yml` (per-push patch validation) and `release.yml` (the tag-triggered GitHub Release
carrying `rdna-boosts-all.patch`, `patches.tar.gz`, `release.json` and `SHA256SUMS`).

**Scope: this is the container image build only.**  The normal local source build used for testing,
benchmarking and the release gates (`~/bin/build-llama-rocm`, `build-pr`/`build-r69`/..., plus the
coherence, MTP and prefill-logit runs) is unaffected and is still done on the primary box.  "Do not run
the full build locally" means the `podman` image build below, not `cmake --build`.

**Worker (`radiance`) one-time setup** (full detail in `CONTAINERS.md`):

```bash
sudo loginctl enable-linger "$(id -un)"     # cron + rootless podman survive logout/reboot
gh auth login -h github.com                 # classic PAT needs write:packages + delete:packages
install -m 755 <repo>/scripts/container-watch.sh ~/bin/rdna-container-watch.sh
install -m 755 <repo>/scripts/validate-gpu.sh    ~/bin/rdna-validate-gpu.sh
( crontab -l 2>/dev/null; echo '*/10 * * * * $HOME/bin/rdna-container-watch.sh >> $HOME/.cache/rdna-containers/watch.log 2>&1' ) | crontab -
( crontab -l 2>/dev/null; echo '*/30 * * * * $HOME/bin/rdna-validate-gpu.sh >> $HOME/.cache/rdna-validate/watch.log 2>&1' ) | crontab -
# ccache is required for the ladder (no root needed: the static 4.12.3 release binary)
curl -fsSL -o /tmp/cc.tar.xz https://github.com/ccache/ccache/releases/download/v4.12.3/ccache-4.12.3-linux-x86_64.tar.xz \
  && tar -xJf /tmp/cc.tar.xz -C /tmp && install -m755 /tmp/ccache-4.12.3-linux-x86_64/ccache ~/.local/bin/ccache \
  && ~/.local/bin/ccache --max-size=30G
```

**CI is three tiers, and "CI" on the worker means the gating job, not the image build.**  Which job a
change hits depends on the trigger, and the split is what keeps the cheap gate cheap:

| tier | where | what | trigger |
|---|---|---|---|
| 1 | `.github/workflows/validate.yml` (hosted, no GPU) | `validate-set.sh`: strict apply on a fresh tarball, tree/count, per-artifact sha256, Windows-path check, **and a scan of every block tree for conflict markers / subject drift** | every push and PR, ~20 s |
| 2 | `radiance` (`scripts/validate-gpu.sh`, cron every 30 min) | re-applies the set at `origin/main` from scratch, `gate-per-block-build.sh --fast`, the seven op oracles, then `gate-prefill-logits.sh` (records a base on a fresh box, compares after) | every new `origin/main` commit, unattended |
| 3 | a GPU box, by hand | the release flow: `all` build, coherence, MTP Protocol A, arch sweeps | before a tag |

Tier 2 exists because almost every block is a CUDA/HIP kernel change and a hosted runner has no ROCm, so
tier 1 cannot compile any of it.  It is the automation behind "a re-base is a decode AND prefill change".
It holds its own `flock` and **skips while a container build holds the watcher's lock**, so the two worker
jobs never contend; summaries land in `~/.cache/rdna-validate/runs/<sha>/summary.txt`, and a rev is retried
at most `MAX_ATTEMPTS` times (3) before the script stops and names the `failed-<sha>` file to remove.

**Primary box: manual container build (only if not using the worker) or a fast smoke:**

```bash
./scripts/build-containers.sh --prepare-only               # reconstruct + verify the release tree
./scripts/build-containers.sh --rocm "7.14.1 10.1" --push --prune   # full build; prefer the worker
./scripts/build-containers.sh --rocm 10.1 --arch gfx1201 --targets server   # fast smoke, no push
```

`scripts/build-containers.sh` reads `release.json`, materialises the canonical patched tree
(`release.json.tip`, asserting `tree`) into `$HOME/llama-src`, writes the build-context
`.dockerignore` (from `.devops/dockerignore`), builds `full`/`light`/`server` per ROCm line with the
same build args the CI matrix used, applies the same registry tags (`rocm-<v>`,
`server-/light-/full-rocm-<v>`, the immutable `-<base>` variants and `latest` on 10.1) and (`--push`)
pushes to `ghcr.io/stew675/llama-cpp-rdna-boosts`.  `docker-ghcr.yml` remains only as a manual
`workflow_dispatch` fallback.

**ccache is on.**  `.devops/rdna-rocm.Dockerfile` installs the upstream static ccache **4.12.3** (the
version the host validates, not Ubuntu's older package), sets `CCACHE_DIR=/root/.ccache` and adds
`-DCMAKE_{HIP,C,CXX}_COMPILER_LAUNCHER=ccache` (mirroring `~/bin/build-llama-rocm`), with
`/root/.ccache` as a build cache-mount — never an image layer, so the published images are unchanged.
Measured single-arch: cold compile 10m01s → cache-warm rebuild 1m28s (835/836 direct hits).  A release
whose sources mostly match the previous one recompiles only the delta.

**Prune only the built images, never the base images.**  `--push --prune` publishes and then removes the
just-built `ghcr.io/...` images and dangling build layers, keeping the ROCm base images and the ccache
store so the next build neither re-pulls ~20 GB per line nor recompiles from cold.  Do **not** use
`podman system prune -a` for this (it takes the base images and ccache with it).

**The push also prunes the GHCR package.**  GHCR keeps every manifest ever pushed (an index plus one
version per architecture, plus whatever a moving tag left behind), so the package grows without bound;
`scripts/cleanup-ghcr.py` runs automatically on `--push` and deletes everything outside a rolling
`--keep-days` window (default **7**, one release cycle) while **always keeping the current base's
`*-<base>` tags** — the images just pushed can never be deleted.  `--no-cleanup` skips it.  A GitHub
bulk-delete throttle (secondary rate limit) is a warning, not a failure; re-run the tool to finish.

**Only ROCm 7.14.1 and 10.1 are built by default.**  7.2 (`-complete`) is much slower and is manual-only.

## What NOT to do

- Do not `git apply` the concatenated block series (drops hunks); use `scripts/apply-all.sh`.
- Do not hand-edit the committed patches as a permanent drift fix — regenerate and re-verify.
- Do not mix the historical `baseline/*` branches or `block/*` tags with the current `patches/`.
- Do not push anything from the `~/llama.cpp` checkout except the release-time `rdna-boosts` refresh
  to the personal fork (see the Pushing policy). Never push to upstream `ggml-org/llama.cpp`.
- Do not present old docs as current: `MANIFESTS.md`/`BASELINE.md` are contracts, their dated history
  is in `archive/docs/`; current claims are their current-state headers + `patches/README.md` +
  `release.json` + the newest `WORKLOG.md` entry.
- Do not add new WIP experiments to the delivery set, and **never apply anything from `wip/`** or
  `archive/work/` unless the user explicitly asks for that specific item.

## Editing the docs

The docs have a freshness problem by design: historical records are kept, and the CURRENT state lives
in the current-state headers (`patches/README.md`, `README.md`, `MANIFESTS.md`, `BASELINE.md`,
`release.json`) plus the newest `WORKLOG.md` entry. When you change the delivery, update those headers
and add a new dated `WORKLOG.md` entry (newest first) — never edit dated validation records in place.

**Each top-level doc has one job — keep it that way.**  `README.md` describes the *set* (what it is, how
to apply it, what a consumer needs); `MANIFESTS.md` is the apply order + verification contract;
`BASELINE.md` is provenance + the drift policy; `GREEDY-PURITY.md` is the rulebook (claim, rule, numbers
per `§N`, with the narratives in `archive/docs/GREEDY-PURITY-FINDINGS.md`); `TODO.md` is
**forward-looking only** (open, waiting, accepted-limitation, parked — a closed item is one `WORKLOG.md`
entry, never a TODO entry); `ENVIRONMENT.md` is the variable catalogue.  **Release narrative belongs in
`WORKLOG.md` and nowhere else** — do not paste a release summary into `README.md`, `BASELINE.md` or
`MANIFESTS.md`; point at `WORKLOG.md` instead.

**WORKLOG dated headers use `YYMMDD.VVV`** (e.g. `## 261011.005 (r47) -- ...`).  `YYMMDD` is the
real local calendar date and `VVV` is a per-date, monotonically increasing 3-digit sequence number in
file order (the topmost/newest entry of a date carries the highest number).  The date is only a
date — never treat it as a version; the revision is the explicit `(rNN)` tag.  A new entry takes
today's real date and the next free number for that date.  Do not invent or extrapolate a future
date: if the work is done today, the header says today. Keep this file terse: policies, invariants and pointers only. Session/dev handovers belong
under `wip/` or `archive/docs/`, not at the repo top level.
