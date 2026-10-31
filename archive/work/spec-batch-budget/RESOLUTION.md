# RESOLUTION -- `spec-batch-budget` (PR #169, @briansp2020) -- REJECTED

**Status: REJECTED (2026-10-16), upheld on re-assessment (2026-10-17, r79).  Not shipped; no block
change.**  The contributor's format-patch, README and the multi-slot probe used for the re-assessment
are preserved in this directory so the measurement is not re-litigated from scratch.  The PR is
answered on GitHub with the numbers and with what a re-entry would need.

## What it proposed

`LLAMA_SPEC_BATCH_BUDGET=N` (server only, `tools/server/server-context.cpp`): while more than one slot
is generating, cap each slot's draft at `max(1, N / n_generating - 1)`, so the verify batch stays within
`N` tokens.  Unset = off, one draft token per slot minimum, single request never capped.  The reporter
measured +8.7 % aggregate at 4 concurrent requests on 2x R9700 (IQ3_XXS Flash-Next, 300-token requests)
and no change to single-request speed or output.

## Why it was rejected

It regresses on this box, in the reporter's own 2-GPU topology as well as at 3 GPUs.  Same binary
(r76 + PR #171 + PR #168 + PR #169, so the accepted PRs are held constant), fresh server per arm, warmup
first, q8_0 KV, `--spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0`, four concurrent
300-token requests:

| config | budget unset (cap 3) | `=8` (cap 1) |
|---|---|---|
| MoE 35B-A3B UD-Q4_K_M, 3 GPUs tensor | 279.5 / 259.1 / 278.0 t/s | 232.8 / 211.9 / 213.0 t/s (**-16 %**) |
| MoE 35B-A3B UD-Q4_K_M, **2 GPUs** tensor | 314.6 / 320.7 / 326.9 | 304.7 / 308.7 / 323.4 (**-2.7 %**) |
| dense 27B UD-Q4_K_XL, 3 GPUs tensor | 134.3 / 142.4 / 151.9 | 128.9 / 126.2 / 122.6 (**-12 %**) |
| single request (same arms) | 130.7 / 104.9 / 130.2 | 129.7 / 102.3 / 130.1 |

Budget sweeps show the shape of it: 8 (cap 1) is the losing arm, 12 (cap 2) is neutral, and 16/32 (cap 3,
i.e. no cap) reproduce the unset column.  On this hardware a 16-token verify batch (4 slots x 4) is
**cheaper per accepted token** than an 8-token one, so the premise ("at that width the extra draft
tokens cost more step time than they save") does not hold here.

## Second finding (structural, independent of the numbers)

The cap is applied to `common_speculative_get_draft_params(...).n_max`, i.e. to the *per-call* bound,
but the adaptive controller (`--spec-type draft-mtp-adaptive`) is fed `n_accepted` against its own
`n_cur` (`common_speculative_adaptive::update`), not against the load-capped depth.  During a multi-slot
burst the controller therefore sees a miss of `n_cur - n_accepted` every round and drains to the floor;
when the burst ends, a single-request run has to climb back from the floor (about 20-26 net full accepts
per step).  The PR's measurements are all fixed-depth (`draft-mtp`), so this interaction is untested.

## Re-entry conditions

1. Reproduce a win on a config this project can gate: the 2-GPU row above is the reporter's topology and
   it is -2.7 % here; a model/quant/GPU combination where the cap actually pays would need to be named,
   with the before/after per-step step time and accepted tokens (not only aggregate t/s).
2. Make the cap interact correctly with `draft-mtp-adaptive`: feed the controller the effective
   (capped) depth, or suspend its update while the load cap is active, so a burst does not drain it.
3. Keep it an opt-in tuning value (`0`/unset = off) -- a hard-coded 8 is device- and model-dependent.
4. Gate it like any other change: `llama-batched-bench -npl 1,4,8` at the verify widths plus acceptance
   from multi-slot servers, since "4 concurrent chat requests" is the only shape that can move.

## Record

`WORKLOG.md` 261016.012 (r77) records the rejection and the table; 261017.004 records the re-assessment.
No `ENVIRONMENT.md` entry, because nothing shipped.

## Re-assessment on r79 (2026-10-17) -- the verdict stands, the rejection's SCOPE was too narrow

Re-opened because five releases landed after the rejection (r77 #171, r78 #172/#174, r79 #167/#168), with
the question "was #169 dependent on commits that were missing?".  **No.**  Nothing between the r77
tree and the r79 tip changes the multi-slot verify path on gfx1201:

| change | relevance to an 8-16 row verify batch, gfx1201 |
|---|---|
| r77 #171 q8_0 `SET_ROWS` group kernel + register `rms_norm_q8_1` | **already in the rejection build** (it was `r76 + #171 + #168 + #169`) |
| r78 #172 rope + FWHT fusion | matcher requires NORMAL/NEOX rope -> **inert on Qwen3.x** (`rope_multi`) |
| r78 #174 ADD folds into `rms_norm_q8_1` | gated to `ne[1] == 1` (single token), so not a verify batch |
| r79 #167 MoE `vec_dot_q4_K_q8_1_vdr4_dense` | RDNA3_0 default; **RDNA4 is a build-time opt-in, off by default** |
| r79 #168 `LLAMA_KV_N_PAD_PREFILL` default-on | prefill only |

**Reproduced on r79** with one binary (r79 + #169), env unset vs `=8`, so the accepted PRs are held
constant.  `-np 4`, `--spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0`, q8_0 K/V, four
concurrent 300-token greedy requests, 2 runs per arm:

| config | unset (cap 3) | `=8` (cap 1) |
|---|---|---|
| MoE 35B-A3B UD-Q4_K_M, 3 GPU tensor, all VRAM | 305.9 / 320.5 t/s | 294.3 / 273.0 t/s (**-4 % / -15 %**) |
| MoE 35B-A3B UD-Q4_K_M, 2 GPU tensor, all VRAM | 370.1 / 413.9 t/s | 379.2 / 369.9 t/s (neutral) |
| dense 27B UD-Q4_K_XL, 3 GPU tensor | 194.0 / 205.5 t/s | 158.5 / 159.4 t/s (**-18 % / -22 %**) |

The r77 verdict reproduces, so it was not an artifact of a missing commit.  What the r77 review did get
wrong is coverage: **every arm was all-in-VRAM**.  The contributor's own config is the opposite shape
(host-resident experts), and that is where the cap pays.

**The contributor's shape, reproduced locally** (their IQ3_S is not on this box; Flash-Next GSQ-RCO
IQ3_XXS is the same model class): 2 GPUs, `-sm tensor --n-cpu-moe 47`, shared
`mtp-Qwen3.8-Flash-Next-shared-Q8_0` head, q8_0 K/V, otherwise as above:

| 4 concurrent, n_max 3 | aggregate t/s | step ms | accepted/step | ms per accepted token |
|---|---|---|---|---|
| unset (cap 3) | 76.5 / 87.0 / 84.5 | 144-165 | 12.5-12.7 | 11.5-13.0 |
| `=8` (cap 1) | **145.7 / 145.6 / 155.2** | 47-51 | 7.4-7.5 | 6.4-6.9 |
| `=12` (cap 2) | 81.9 / 86.8 | 122-126 | 10.4-10.6 | 11.5-12.2 |
| 2 slots, unset vs `=4` (cap 1) | 123.4 vs 113.0 | 50.1 vs 33.5 | 6.2 vs 3.8 | 8.1 vs 8.8 |

So **+80-90 %** on that shape at 4 slots, and still a small loss at 2 slots (the author measures the same
shape as a +3.5 % win on Flash-Next GSQ IQ3_S; see the note below).  Note that plain decode on the same
config is 167.5 t/s, so the default (cap 3) is running multi-slot MTP *slower than no speculation at all*
here, which is the real defect this exposes.

**Mechanism: a verify-width cliff at `MMVQ_MAX_BATCH_SIZE == 8`, not corruption.**  Per-step cost against
verify width on the GSQ hybrid (2 GPU, 4 slots):

| verify width | step ms | accepted/step | ms per accepted token |
|---|---|---|---|
| 4 (plain decode) | 23.9 | 4.0 | 5.97 |
| 8 (cap 1) | 47-51 | 7.4 | 6.4-6.9 |
| 12 (cap 2) | 122-126 | 10.5 | 11.5-12.2 |
| 16 (cap 3) | 144-165 | 12.5 | 11.5-13.0 |

The knee sits between 8 and 12 rows, i.e. exactly the `MMVQ_MAX_BATCH_SIZE == 8` boundary (matmuls switch
from the MMVQ/MMVF decode families to MMQ and FA switches to WMMA above it).  It is **not** generic
host-expert cost and **not** qwen4exp alone:

| control | w = 8 | w = 16 | ms/token |
|---|---|---|---|
| Flash-Next GSQ-IQ3_XXS, **3 GPU, all VRAM** (`-ncmoe 0`) | 39.7 ms | 65.6 ms | 5.29 vs 5.17 (flat) |
| MoE 35B-A3B, 2 GPU, **host experts** (`-ncmoe 40`) | 21.4 ms | 32.5 ms | 2.85 vs 2.71 (flat) |
| Flash-Next GSQ-IQ3_XXS, 2 GPU, host experts | 47-51 ms | 144-165 ms | 6.4 vs 12.4 (**cliff**) |

Disabling the fused shared-expert epilogue (`GGML_CUDA_DISABLE_SHEXP_DOWN_GATE=1`) does not remove it
(73.9 vs 142.7 t/s), so the >8-row path on **qwen4exp + host-resident experts + 2-GPU tensor** is the
thing to chase.  That is a delivery defect in its own right and it affects the default config with no env
var; #169's cap is a workaround for it, not a fix.

**Correctness (the +2x is not degenerate output).**  Output length is fixed by the harness (300 tokens
per request, `ignore_eos`, 1200 tokens per arm, every slot reporting `predicted_n == 300`), so no arm was
truncated by a crash.  Text is coherent technical prose in both arms.  At **2 slots, cap 1 and cap 3
produce byte-identical text** (`sha a59941368946` / `a2c9e2955a1c`), i.e. the cap preserves greedy output
where the verify width does not change kernel family.  At 4 slots the text differs between arms *and
between two runs of the same arm*, which is the pre-existing concurrent-batching nondeterminism
(`wip/kv-cell-offset-nondeterminism/`), not the cap.  `scripts/check-gpu-faults.sh` over the kernel
journal and every run log: clean.  The only abort in the session was an invalid `-md` with
`--spec-type none`, which is documented to abort.

**Contributor's reply** (PR comment 6234999977, same day): they retract the +8.7 % as unpaired (+-4 %
between fresh servers on the same binary) and report a paired 8-round design: `=8` **+3.47 %
[+2.03, +4.92]** on 2x R9700 with Flash-Next GSQ **IQ3_S**, `--n-cpu-moe 47`, `=12` neutral, greedy text
byte-identical in all 24 loads.  They agree it is not shippable as it stands and offer to do the
controller work.  The numbers differ from ours in magnitude (+3.5 % vs +80-90 %) on the same shape, with
different quant (IQ3_S vs IQ3_XXS) and possibly vocabulary, so the shape is confirmed and the size of the
win is not.

**Updated re-entry conditions.**  Unchanged 2-4 above (controller feed, opt-in, the width gate), and:

1. Frame it as a **verify-width cap**, not a token budget: `n_draft <= max(1, 8 / n_generating - 1)`
   keeps the batch at or under the `MMVQ_MAX_BATCH_SIZE` boundary, which is the thing the measurements
   say matters.  A fixed 8 is one instance of that rule at 4 slots only.
2. Name the winning shape in the PR (qwen4exp Flash-Next with host-resident experts, 2-GPU tensor), and
   report per-step accepted tokens and step time next to aggregate t/s, so the width effect is visible.
3. Say what happens at 2 and 3 slots (ours: -8.4 % at 2 slots even on this shape).

**Probe / retest kit.**  `retest/README.md` is the reproduction and re-test guide (build, every arm's
exact command line via `retest/run-arms.sh`, the pass criteria, the pitfalls, and where a revised patch
should be re-measured).  `multislot-mtp-probe.py` in this directory is the harness used for every number
above: it
starts one `llama-server` per arm, primes it, runs N concurrent `/completion` calls, and reports
aggregate t/s, per-slot t/s, per-step ms, accepted/step, mean accepted length and per-slot text sha256
(parsed from the server's own `draft acceptance` and `eval time` lines).  It is a record, not a delivery
script.

## Re-assessment on r184 (2026-10-31) -- CLOSED, superseded by the delivery-side width bound

The PR was re-opened a third time and re-measured on **r184** (one binary = r184 + the patch, the env var
selecting the arm; the contributor's shape is 2x R9700 `-sm tensor` with the PLE table resident via
`--lazy-mode off`; 4 concurrent 300-token greedy requests, fresh server per arm, 2-4 reps, this kit's probe as
the harness).  Two things had changed since r79: 148 releases of kernel/graph work, and `LLAMA_SPEC_ALIGN`
(r153) turning multi-slot drafting into a per-sequence-depth problem.  Full tables:
`benchmarks/2026-10-31-multislot-verify-width.md`; release: `WORKLOG.md` 261031.001 (r185).

**The premise is refuted.**  Fixed depth 3, 4 slots, width 16 (uncapped) vs width 8 (`=8`): **35B-A3B 3 GPU
432.7 vs 389.8 t/s, 2 GPU 387.2 vs 359.6, dense 27B 3 GPU 240.1 vs 211.2** -- the cap LOSES 7-12 % on the shapes
this delivery recommends -- while the contributor's shape wins (qwen4exp Flash-Next GSQ-IQ3_XXS 2 GPU + PLE
204.6 vs **220.8**, IQ3_S 196.7 vs **205.8**).  At 2 slots on that shape the same knob is **-22 %**.  Width 16
beats width 8 on 3 of 4 fixed-depth shapes, so "above ~8 rows each extra draft token costs more than it gains"
does not hold as a rule.  The structural point (the controller is fed `n_accepted` against its own `n_cur`, not
the capped depth) was never addressed, and since r153 a per-call cap also has to interact with the alignment.

**The r79 "cliff" is gone, and the real defect is elsewhere.**  The qwen4exp + host-expert + 2-GPU arm that
measured 11.5-13.0 ms per accepted token at width 16 on r79 now reads 5.46 at 16 and 5.03 at 8 (i.e. the
workaround's 80-90 % is gone with it).  What is left is a *different* and larger defect that this PR pointed at:
on a band-0 config the **adaptive** depth is unbounded (r176 made the arena band answer only for host-resident
experts), so 4 slots at `--spec-draft-n-max 7` present ~28-32 rows and the default recommended speculator loses
**9.5-31.8 %** against the same arm bounded at 16 rows (35B-A3B 3 GPU 317.1 -> 418.0, qwen4exp 2 GPU 166.6 ->
212.4, dense 27B 3 GPU 205.7 -> 225.3).  That is fixed in the delivery (block 02, r185,
`LLAMA_SPEC_ALIGN_WIDTH_BAND`, default 16) -- not by a user-facing budget, and **not** by a model/shape whitelist:
the sign flips on the same model across quant, device count, slot count and speculator, so the rule is expressed
on the verify width `n_slots x (1 + depth)` and evaluated per draft call.

**Status: CLOSED (2026-10-31), not merged.**  The PR's files are a `wip/` copy of what this directory already
holds (byte-identical), so there was nothing to take; its finding is credited in the r185 block-02 amendment and
`benchmarks/2026-10-31-multislot-verify-width.md`.
