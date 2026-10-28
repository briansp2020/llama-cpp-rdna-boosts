# vram-limit-103 — the "~30x collapse at the last few MiB of VRAM" (TODO #103)

**Status: CLOSED (r167, 2026-10-28).**  The collapse is **not a llama.cpp code path**.  At the card's
limit the delivery is either at full speed or it refuses the allocation cleanly, and the only measurable
degradation is **driver page-migration work for the *other* client's buffers**, which tracks how much
memory has to be moved, not how llama.cpp placed its own.

The durable outcomes (all in block 10, r167): the free-VRAM low-water mark is tracked on **every** config
(it previously existed only for slab-armed MoE runs, printed only under `GGML_CUDA_SLAB_STATS`), the new
default-on `GGML_CUDA_LOW_FREE_MIB` warning names the state when a run reaches it, and a refused
allocation now reports free/total plus the run's low-water mark.  `GGML_CUDA_MIN_FREE_MIB` (r135, opt-in)
remains the mechanism that turns the state into a deterministic refusal *before* the cliff.

## What #103 said

From the r135 record (issue #219): on one gfx1201, with the 27B GSQ-RCO-IQ3_S + mmproj + adaptive MTP at
`-c 65536`, dropping the free VRAM at process start from 14.46 GiB to 14.06 GiB turned a 13 s request into a
384 s one (prefill 1127 -> 35 t/s, decode 53 -> 1.5-3 t/s) with device memory exhausted, and the next
larger prompt aborted with `ROCm error: out of memory`.  Ruled out at the time: the FA staging fallback
and "graph churn alone"; the workspace pool's flush + retry accounted for "a fraction".  Next step as
recorded: separate "the GPU is doing 30x the work" from "the host is stalling 30x".

## Method

A holder process (`tools/vram_fill`, `hipMalloc` + `memset`) pins part of a 32 GiB R9700 so the free VRAM
a run starts with matches the r135 cell, then the r135 config runs with the reporter's exact model and
flags and a 15738-token prompt (`prompts/prose-rdna-boosts.txt` x3; 15790 tokens after the chat template).
`tools/run-arm.sh` drives `llama-cli`, `tools/evict-arm.sh` drives a real `llama-server`; both sample
VRAM/GTT/clock every 0.3-1 s.  `tools/vram_probe{,2}` are the placement oracle.

**Machine:** 3x R9700 gfx1201 (this work is single-device, device 0), ROCm 10.1.0, 9950X3D, 94 GiB.
Model `/llm/models/Qwen3.8/27B/GSQ-RCO-IQ3_S/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf` (11 159.69 MiB model
buffer) + `mmproj-...-BF16.gguf`, `-ngl 99 -c 65536 -b 4096 -ub 1024 -fa on -ctk q4_0 -ctv q4_0 -t 8
--spec-type draft-mtp-adaptive`, KV 1152 MiB, RS 2394 MiB, compute 431.54 MiB, roomy peak **16 203 MiB**.

## The arms

Free VRAM at process start is what the r135 record varies; "holder" is how that free VRAM was taken.

| free at start (MiB) | holder | prefill t/s | decode t/s | GTT peak | note |
|---|---:|---:|---:|---:|---|
| 32 554 (none) | — | **1117.9 / 1120.1** | 62.1 / 62.5 | 20 MiB | reference |
| 16 854 | one 16.4 GB BO | 1102.1 | 61.9 | 26 MiB | fits beside the holder, nothing moves |
| 16 154 | one 16.4 GB BO | 1107.5 | 62.0 | 26 MiB | same |
| 15 954 | one 16.4 GB BO | 626.1 | 58.6 | **27.6 GB** | migration lands inside the prefill |
| 15 754 | one 16.4 GB BO | **577.7** | 58.7 | **16.7 GB** | the "slow band" — reproduced |
| 15 754 | 64 MiB BOs | **1077.3 / 1074.4** | — | **26 MiB** | *identical free VRAM, no migration* |
| 15 754 | 64 MiB BOs, live | 972.1 | 49.2 | 26 MiB | a competitor that touches its pages |
| 15 704 | 64 MiB BOs | **abort** | — | 26 MiB | `ROCm error: out of memory` |
| 15 654 / 15 604 / 15 554 | any | **abort** | — | — | same, up to 887.99 MiB refused at load |
| 16 400 | 64 MiB BOs, r167 binary | 1113.2 | 62.2 | 26 MiB | fires the new low-free warning at 386.50 MiB free |

The cliff is **~50 MiB wide**: 15 704 MiB free aborts, 15 754 MiB free runs at full speed.  Between them
the run's peak sits exactly at the budget and nothing else changes.

## Findings

**F1. HIP refuses an allocation that does not fit; it does not over-commit, and a resident working set is
never demoted.**  With 1 048 MiB free, `hipMalloc(2048 MiB)` fails `out of memory`
(`evidence/probe-3-refuses.txt`).  In one process, A=12 GiB then B=18 GiB leaves 1 684 MiB free and A
still reads **506-513 GB/s** (versus 513 GB/s roomy) — a 30 GiB resident working set at 1.7 GiB of
headroom is at full VRAM bandwidth (`evidence/probe-2-no-demotion.txt`).  So the r135/r168 narrative that
"the driver evicts resident weight pages to GTT" is wrong for the requester: *nothing of the run's own is
moved*, and `rocm-smi`'s process figure (which r135 cited as proof the weights stayed resident) cannot
distinguish VRAM from GTT anyway.

**F2. The "slow band" is the driver migrating the *other* client's memory.**  TTM evicts a **whole BO**:
the holder's single 16.4 GB BO had to be migrated in full to make room for a few hundred MiB, and the
migration lands inside the measured prefill when it happens then — 577.7 t/s at 15 754 MiB free.  The
same free VRAM with the same total hold split into 64 MiB BOs (so the driver only gives up what is
needed) leaves GTT at 26 MiB and measures **1074-1077 t/s**.  That is the whole difference; the two runs
are otherwise identical (same binary, same model, same flags, same free VRAM).

**F3. Nothing in the delivery is the trigger at a tight budget.**  On the tight-but-viable cell:
`GGML_CUDA_FA_STAGE_MAX_MB=0` on a roomy card is 1106 t/s (the FA staging arena is *not* worth 2x, so its
stand-down cannot be the collapse), and the tight arm's 589 t/s came with GTT 16.7 GB, i.e. F2.  The graph
census is *identical* in a 588 t/s arm and a 1117 t/s arm — `direct=57 replay=51 capture=3
properties_changed=1`, `graphs reused = 10` — so it is not graph churn.  Clocks are at boost in both
(GFX 2764-3427 MHz, `GPU use 100 %`, mem 1258 roomy / 1124 tight — at most 11 %, not 30x).  The workspace
pool's recovery is `alloc … failed, flushing … and retrying` followed by a clean `ROCm error: out of
memory`, never a slow path.  And r135's *eager* FA staging hint (`GGML_CUDA_FA_STAGE_PREALLOC=2`, the
behaviour of that tree) turns the tight cell into an earlier OOM, not into a slow run.

**F4. So the r135 30x does not reproduce on the current tree, and its shape is not reachable from
llama.cpp.**  64 runs across four squeeze axes (free VRAM at load, free VRAM taken away after load,
holder granularity, live vs idle holder) produced *full speed*, *a bounded 0-48 % migration loss*, or *a
clean abort* — never a 30x run.  The r135 cell (384 s) is compatible with F2 at a much larger migration
volume than an idle holder produces; it is a driver/OS effect (on Windows it is WDDM paging, which the
r135 budget knob exists to avoid), not a delivery path.  What the delivery can do, it now does: name the
state (`GGML_CUDA_LOW_FREE_MIB`), report it on a refusal, and refuse deterministically one allocation
early if the operator asks (`GGML_CUDA_MIN_FREE_MIB`).

## Reproduce

```bash
cd tools && for f in vram_fill vram_info vram_probe vram_probe2; do
  /opt/rocm-10.1.0-gfx120X/bin/hipcc --offload-arch=gfx1201 -O3 -o $f $f.cpp; done

cat ../../../prompts/prose-rdna-boosts.txt{,,} > /tmp/prompt-3x.txt      # 15738 tokens

# the "slow band": one 16.4 GB holder BO, identical free VRAM to the next arm
CHUNK_MIB=0   ./run-arm.sh 15900      # 577-983 t/s, GTT 16.7 GB
CHUNK_MIB=64  ./run-arm.sh 15900      # 1074-1077 t/s, GTT 26 MiB
./run-arm.sh 0                        # reference 1117-1120 t/s
./run-arm.sh 15850                    # clean abort: ROCm error: out of memory

# the placement oracle (no llama.cpp involved)
./vram_probe2 12288 18432             # resident buffer keeps full bandwidth at 1.7 GiB free
./vram_fill 31500 60 & sleep 4; ./vram_probe 2048   # hipMalloc refuses instead of over-committing
```

`tools/clock-sample.sh` samples GFX/MEM clock and utilisation; `tools/evict-arm.sh` runs a real
`llama-server` and can take VRAM away *after* the model is resident (`PRE_HOLD=0`) or before it
(`PRE_HOLD=1`), with `KILL_AFTER_LOAD=1` and `NREQ=n` for the "is the loss one-time?" question.

## Records

`WORKLOG.md` 261028.006 (r167), `ENVIRONMENT.md` (`GGML_CUDA_LOW_FREE_MIB`, `GGML_CUDA_MIN_FREE_MIB`),
`patches/README.md` (block 10), `AGENTS.md` (the over-budget fact), `WORKLOG.md` 261025.005 (r135, where
#103 was opened).
