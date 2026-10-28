# `fa-stage-peak` — BETA (issue #237 follow-up): one early, correctly-sized FA staging allocation

> **STATUS (2026-10-27): BETA — NOT part of the delivery set.**  Cut against `v15-6050b467c-r158` for
> @DanoPTT's Windows/WDDM re-test (the #237 follow-up: the r156 growth ladder makes a 100k prefill ~22 %
> slower than r149 because the last 1 GiB step lands in shared memory).  Nothing here is folded into a
> block; see "When promoted" for where it would go.

## Artifact

| field | value |
|---|---|
| patch | `fa-stage-peak.patch` (in this directory) |
| sha256 | `257fe48ba4bfc802f578203be5505ba7f4a6843df493360d530cfc6041c82de4` |
| base | delivery `v15-6050b467c-r158`, tip `217b1c999`, tree `8851bb0024e547142ec533bd76ec18912539edad` |
| applies | `git apply --check` clean on a fresh `scripts/apply-all.sh` checkout of that tree (verified here) |
| files | 7, +81/-4 (`common.cuh`, `fattn-common.cuh`, `ggml-cuda.cu` + hooks, `llama-context.{h,cpp}`) |
| knobs | `GGML_CUDA_FA_STAGE_PREALLOC` (`0`/`1`/`2`), `GGML_CUDA_FA_STAGE_DRAFT_MB` (new), `GGML_CUDA_FA_STAGE_MAX_MB` (unchanged) |

## The problem this addresses

On Windows/WDDM a freed device allocation stays committed, and an allocation that does not fit dedicated
VRAM silently lands in shared memory.  So an arena that *grows* is wrong there twice: r156's ladder strands
~1.4 GiB of commitment for a 1 GiB arena (#138) and its last step — taken ~1:30 into the prefill, when KV
and compute buffers have filled the card — lands in the aperture.  The reporter measured the two failure
modes as an exact 1024/2048 MiB shift from dedicated to shared against r149:

| arm (their box) | dedicated + shared after a 100k prefill | pp t/s |
|---|---|---|
| r149, `-ctkd f16` | 32115 + **899** | **782** |
| r156, `-ctkd f16` | 31476 + **1923** | 602 |
| r156, `-ctkd q8_0` | 31381 + **2949** | 595 |

`899 + 1024 = 1923` and `899 + 2048 = 2947` — i.e. the entire regression is "the arena arrived late, once
for the target context and once for the MTP draft context".

## What the patch does

1. **`GGML_CUDA_FA_STAGE_PREALLOC` gains a third mode, and the default becomes platform-aware.**
   * `0` = the plain 1.25x ladder (pre-r53), `1` = r156's ceiling (a growth aims for at most 4x the
     request), `2` = **eager**: allocate the hint in one go at the first request (the r149 behaviour).
   * Default: **`2` on Windows/WDDM, `1` elsewhere**.  A ladder cannot be made safe on WDDM — the only
     moment you can have dedicated VRAM is the first one — so Windows takes the peak up front and Linux
     keeps the cheaper-on-average growth.
2. **`GGML_CUDA_FA_STAGE_DRAFT_MB`** (new, default unset = off): caps the staging arena of a *speculative
   draft* context.  The draft stages the same prefix as the target while its staging win is one layer's
   amortised conversion rather than sixty, which is the #237 half of the cost.  `64` is the suggested
   value; it only ever *lowers* the hint, so it cannot allocate more than the uncapped arm.
3. Nothing else.  The `--fit` projection is unchanged (it deliberately reserves the per-context worst case,
   which a full-context prefill really can reach).

## Measured on Linux (gfx1201, ROCm 10.1, build `217b1c999` + this patch)

**The ladder becomes one allocation** (27B UD-Q4_K_XL, 1 GPU, q8_0 KV, 32k prompt, `-b 1024`,
`GGML_CUDA_SLAB_STATS=1`):

| arm | `FA_STAGE` allocations |
|---|---|
| `PREALLOC=1` (r156, POSIX default) | `alloc=16.0` → `68.0` → `160.0` MiB (3 driver allocations) |
| `PREALLOC=2` (r149, WDDM default) | **`alloc=160.0` MiB, first request, one line** |

4B Q8_0, `-c 131072`, 6k prompt, `PREALLOC=2`: a single `alloc=512.0 MiB` (the root-based hint) instead of
16 → 24 → 32.5 MiB.  That is the trade stated plainly: **Windows gets r149's peak back**; the reporter's
`-c 262144` case is 1 GiB per context, and `GGML_CUDA_FA_STAGE_DRAFT_MB` / `GGML_CUDA_FA_STAGE_MAX_MB` are
how a user bounds it.

**Numerics are untouched** (an allocation size cannot move a logit):

* `llama-perplexity`, 4B Q8_0, q8_0 KV, `-c 4096 --chunks 4`: **PPL 6.6302** in every arm (`PREALLOC`
  0/1/2) — identical to the unpatched r158 build.
* `scripts/gate-prefill-logits.sh --kv q8_0 --ab "GGML_CUDA_FA_STAGE_PREALLOC=2"` (27B Q8_0, 10 chunks,
  its own `--base-dir` so the shipped bases are untouched): **mean KLD 0.000000, same-top-p 100.000 %** —
  the eager and ceiling arms are bit-identical at the logits.

## What was tried and dropped (C2)

"Size the arena by the decode about to run" (`min(hint, F16 K+V of n_past + n_tokens)`) **does not work in
this codebase**: `llama_context::decode()` sees one *batch*, not the request, and the caller loops (the
reporter's `-b 1024` over a 100k prompt = 100 calls, `llama-cli` = 32 for 32k).  Measured: the projection
grows per call, so with eager mode it produced **10 separate allocations** for one 32k prefill — the same
ladder, just driven by `decode()` instead of by the FA ubatch.  It is recorded here so nobody re-derives it:
a request-sized projection needs the *caller* to pass the request length (`common`/server), which is a
delivery change, not a beta.

## The draft cap is implemented but NOT measured here

No MTP arm was run on this box with the cap (`GGML_CUDA_FA_STAGE_DRAFT_MB`) in this session, so its effect
on the draft context's *prefill* speed is unknown — it is an allocation-only change, but a 64 MiB arena could
push the draft's prefill onto the native read for deep prompts.  That comparison is arm B in
`BETA-TESTING.md`, and it is the one thing this beta most needs from the tester.

## When promoted

* Move the `PREALLOC` modes + default into the block that owns the arena policy (**block 10**,
  `common.cuh`), the hint clamp into **block 05** (`fattn-common.cuh`) and the draft cap into **block 12**
  (`llama-context.cpp`).
* `ENVIRONMENT.md`: `GGML_CUDA_FA_STAGE_PREALLOC` gains the mode list and the platform default;
  `GGML_CUDA_FA_STAGE_DRAFT_MB` gets its own row.
* Re-run `validate-set.sh`, the coherence gate, `gate-prefill-logits.sh` (f16 + `--kv q8_0`), the MTP gates
  and the per-block build; then record the tester's numbers in `WORKLOG.md` and move this directory to
  `archive/work/fa-stage-peak/`.
