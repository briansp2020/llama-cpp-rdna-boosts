# BETA test — `fa-stage-peak` (issue #237 follow-up)

**Who this is for:** @DanoPTT, on the Windows/WDDM box from issue #237 (1x R9700, ROCm 10.0, Qwen3.8-27B
UD-Q5_K_XL, `-c 262144 -np 2 --kv-unified -ctk q8_0 -ctv q8_0 -fa on -ub 512 -b 1024
--spec-type draft-mtp --spec-draft-n-max 3`, graphs off).

**What we want to know:** does the prefill-speed regression against r149 go away, and does it cost the peak
VRAM that issue #237 was about?

## 1. Apply

On top of a clean checkout of the delivery tree (base `6050b467c`, `apply-all.sh` already run):

```bash
cd /path/to/llama.cpp
git apply /path/to/fa-stage-peak.patch      # 7 files, +81/-4
# ...or: bash <delivery>/scripts/apply-all.sh --include-beta .   (applies beta/*/*.patch last)
```

Build with your usual flags.  Nothing else is needed: on Windows the patch's default is already the
eager-allocation arm.

Kill switches, if an arm misbehaves: `GGML_CUDA_FA_STAGE_PREALLOC=1` restores r156's growth ladder,
`=0` restores the plain 1.25x ladder (pre-#138).

## 2. Arms to run

One `/completion` with the same 100607-token prompt (no cache) on a fresh server per arm, with
`GGML_CUDA_SLAB_STATS=1` in the environment.  Record, as you did for #237: dedicated + shared after the
request, the pp t/s, and every `FA_STAGE …` line in stderr.

| arm | extra env | what it tests |
|---|---|---|
| **A** (new default) | *(none)* | eager: one allocation, at the first request — the r149 shape |
| **B** (draft cap) | `GGML_CUDA_FA_STAGE_DRAFT_MB=64` | the MTP draft context no longer holds its own 1 GiB arena (#237's half), at the cost of a smaller draft arena |
| **C** (bounded peak, optional) | `GGML_CUDA_FA_STAGE_MAX_MB=256` | if A's peak is still too much: bound the arena and accept the native read above it |
| control | `GGML_CUDA_FA_STAGE_PREALLOC=1` | the r156 behaviour you measured, for a same-build comparison |

Expected in arm A, mirroring r149:

```
FA_STAGE dev=0 stream=0 request=2.0 MiB cached=0.0 MiB alloc=1024.0 MiB slab=0
```

— **one line, at the start of the prefill** (not `16 → 72 → 296 → 1024` at 1:30 into it).  In arm B the
draft context should add a second, capped line (`alloc=64.0 MiB`), where arm A shows a second 1024 MiB one.

## 3. Purity (should be a no-op)

The patch only changes *allocation timing and size*, so:

* same-seed greedy output must be **byte-identical** to the same config on r156 (and to r149 if you kept
  that comparison), for every arm;
* `llama-perplexity` PPL must be identical to 4 decimals.

Our side: PPL 6.6302 in every arm, and `gate-prefill-logits.sh --kv q8_0` reads mean KLD 0.000000 /
same-top-p 100.000 % between the eager and ladder arms.  If your text or PPL moves, stop — that would be a
different bug and worth a log.

## 4. What to send back

For each arm: the four numbers from #237's table (dedicated + shared MiB, pp t/s, the `FA_STAGE` lines), and
one sentence on whether the q8_0 draft-KV config is now preferable to `-ctkd f16` for you.  A server stderr
log with `GGML_CUDA_SLAB_STATS=1` is ideal — it contains all of it.

## Known limits (so they are not surprises)

* **The peak comes back.**  On WDDM there is no safe time to take VRAM except the first, so arm A reserves
  the arena's full size up front (1 GiB per context at `-c 262144`).  That is r149's memory shape; arm B and
  arm C are how it is bounded.  This patch does *not* reduce the peak — making the arena genuinely small
  (a per-KV-chunk staging ring) needs a kernel change we have prototyped but not shipped (`wip/fa-stage-chunking/`
  in the delivery repo).
* The draft cap (arm B) is **unmeasured on our side**: we have no Windows MTP arm.  If the draft's prefill
  gets slower in arm B than arm A, say so — the cap is a knob, not a decision.
* `--fit`'s projection is unchanged (it still reserves the worst case).
