# The multi-slot verify width: why PR #169 was not the fix, and what was (r184 -> r185)

Dated: 2026-10-31 (local), gfx1201 (3x R9700 34.2 GB), ROCm 10.1.0 (`/opt/rocm-10.1.0-gfx120X`), ccache on.
Release: `v16-6050b467c-r185` (block 02 amended, `LLAMA_SPEC_ALIGN_WIDTH_BAND`).
PR: [stew675/llama-cpp-rdna-boosts#169](https://github.com/stew675/llama-cpp-rdna-boosts/pull/169) (`wip/spec-batch-budget`, `@briansp2020`) -- **closed, not merged**.

## The question and the shape

#169 proposed an opt-in `LLAMA_SPEC_BATCH_BUDGET=N` that caps each slot's draft at `max(1, N / n_generating - 1)`
while more than one slot generates, so the verify batch stays inside what it called "the small-batch paths"
(~8 rows).  It had been rejected twice (r77 all-in-VRAM; r79 re-assessment on the contributor's host-expert shape,
where the *default* was measured at 76-87 t/s against 145-155 for cap 1).  The r79 note also recorded the
suspicion that the cap was a **workaround** for a delivery defect, a cliff at `MMVQ_MAX_BATCH_SIZE == 8` on
qwen4exp + host-resident experts + a 2-GPU tensor split.

Two things had changed by r184, so the verdict was re-measured rather than restated: 148 releases of kernel and
graph work since r79, and `LLAMA_SPEC_ALIGN` (r153) had since made multi-slot drafting a *per-sequence-depth*
problem.  Everything below is one binary per table, one fresh `llama-server` per arm, greedy
(`temperature 0`, `top_k 1`), 300 tokens per request (`ignore_eos`, `cache_prompt false`), a distinct prompt per
slot, one untimed priming request, `q8_0` K/V, `--spec-draft-p-min 0`, `--lazy-mode off` (the PLE table resident
in host RAM -- that is what makes the contributor's shape fit in two cards and removes mmap-read variance), and
`archive/work/spec-batch-budget/multislot-mtp-probe.py` as the harness.  Run-to-run spread at 4 slots on this box
is **±7 %** (measured as two arms taking the *identical* code path, below), so only deltas outside that are read
as real.

## 1. PR #169's premise is not the rule: width 16 vs width 8

Fixed depth (`--spec-type draft-mtp --spec-draft-n-max 3`), 4 slots, so the verify width is 16 uncapped and 8 at
`LLAMA_SPEC_BATCH_BUDGET=8` (cap 1).  Two to three reps per arm.

| shape | width 16 | width 8 (cap 1) | result |
|---|---|---|---|
| 35B-A3B UD-Q4_K_M, 3 GPU tensor, all-VRAM | 432.7 | 389.8 | **-9.9 %** |
| 35B-A3B UD-Q4_K_M, 2 GPU tensor, all-VRAM | 387.2 | 359.6 | **-7.1 %** |
| dense 27B UD-Q4_K_XL, 3 GPU tensor | 240.1 | 211.2 | **-12.0 %** |
| qwen4exp Flash-Next GSQ-IQ3_XXS, 2 GPU + resident PLE | 204.6 | 220.8 | **+7.9 %** (3/3 reps) |
| qwen4exp Flash-Next GSQ-IQ3_S (the contributor's quant), 2 GPU + PLE | 196.7 | 205.8 | +4.6 % |

So the contributor's shape is the exception, not the rule, and per accepted token the picture is the same
(35B-A3B 3 GPU: 2.32 vs 2.58 ms; dense 27B: 4.17 vs 4.74; qwen4exp: 4.89 vs 4.52).

Slot count moves the width too, on the qwen4exp shape at fixed depth 3:

| slots | widths compared | t/s | result |
|---|---|---|---|
| 2 | 8 (uncapped) vs 4 (`=4`) | 189.7 vs 148.3 | **-22 %** for the cap |
| 3 | 9 (uncapped) vs 6 (`=8` -> cap 1) | 186.5 vs 188.8 | neutral (+1.2 %) |
| 4 | 16 vs 8 | 204.6 vs 220.8 | +7.9 % for the cap |

The r79 record said a 2-slot pair was **byte-identical** between cap 1 and cap 3 (`a59941368946`).  On r184 the
uncapped 2-slot text still hashes `a59941368946`, and the capped one does not (`7f8f820b9789`) -- a verify-width
change is a numerics axis (`GREEDY-PURITY.md` §63) and something between r79 and r184 made the two widths
diverge in text.  It does not affect the verdict (a width comparison is judged by ms/accepted and by the
single-slot arms), but the old claim should not be quoted as current.

Coherence: every arm generated its full 300 tokens per slot, no degenerate text, `scripts/check-gpu-faults.sh`
clean over the kernel journal and all 60+ arm logs.

## 2. What the PR actually exposed: an unbounded multi-slot ADAPTIVE depth

`--spec-type draft-mtp-adaptive --spec-draft-n-max 7`, 4 slots -- the **shipped recommended** speculator.  r153's
alignment bounds the aligned depth by `band / n_drafting - 1`, and r176 made `band` answer only for a model with
host-resident experts, so on a band-0 config nothing bounds it: four slots climb to depth 7 and present ~28-32
rows.  4 reps per arm here except cap 1 (2 reps).

| shape (4 slots, adaptive n7) | uncapped | cap 3 (width 16) | cap 1 (width 8) |
|---|---|---|---|
| 35B-A3B, 3 GPU | 317.1 | **418.0 (+31.8 %)** | 405.8 (+27.9 %) |
| qwen4exp GSQ-IQ3_XXS, 2 GPU + PLE | 166.6 | **212.4 (+27.5 %)** | 206.0 (+23.6 %) |
| dense 27B, 3 GPU | 205.7 | **225.3 (+9.5 %)** | 219.9 (+6.9 %) |

Per-step on 35B-A3B 3 GPU: uncapped 51.5 ms/step for 16.1 accepted (3.18 ms per accepted token) against 31.5 ms
for 13.2 (2.39) at cap 3 -- i.e. the uncapped arm is *not* getting more acceptance for its time.  cap 3 lands
within 3-6 % of a fixed depth 3 on the same shape, so the width is the whole effect and the controller is fine.

Slot sweep on the adaptive arm (35B-A3B 3 GPU): 2 slots cap 7 vs cap 1 = 147.8 vs 146.5 (neutral, width 16 vs 4),
3 slots 173.6 vs 179.1 (+3 %), 4 slots 317.1 -> 418.0.  The defect is a *four-slot* property, which is exactly
why every existing gate -- coherence, prefill-logit, the four-axis MTP protocol, `gate-spec-align.sh` (2 slots) --
is blind to it.

## 3. What shipped instead: a verify-width bound (r185)

`align_draft_caps()` now uses `width / n_drafting - 1`, `width` = the expert-cache band where an arena is active
(unchanged) and `LLAMA_SPEC_ALIGN_WIDTH_BAND` (default **16**, `0` = off) otherwise.  One binary, band on vs off,
2 reps per arm, same shape as above:

| arm | band off | band on | result |
|---|---|---|---|
| 35B-A3B 3 GPU, 4 slots, adaptive n7 | 330.8 | 434.7 | **+31.4 %** |
| qwen4exp IQ3_XXS 2 GPU, 4 slots, adaptive n7 | 175.5 | 208.9 | **+19.0 %** |
| dense 27B 3 GPU, 4 slots, adaptive n7 | 218.5 | 249.6 | **+14.2 %** |
| 35B-A3B 3 GPU, 4 slots, **fixed** `draft-mtp n_max 7` | 310.9 | 362.8 | +16.7 % |
| qwen4exp 2 GPU, 4 slots, fixed n7 | 175.4 | 199.6 | +13.8 % (r1 flat, r2 +27 %) |
| 35B-A3B 3 GPU, 3 slots, adaptive n7 | 302.7 | 353.9 | +16.9 % |
| 35B-A3B 3 GPU, 2 slots, adaptive n7 | 274.3 | 280.4 | inert (text identical) |
| 35B-A3B 3 GPU, 2 slots, adaptive **n9** (cap 9 -> 7) | 282.1 | 277.8 | neutral |
| qwen4exp 2 GPU, 2 slots, adaptive n9 | 145.4 | 148.5 | neutral |
| qwen4exp 2 GPU `-ncmoe 47`, 4 slots (arena band 16) | 191.9 | 184.5 | identical code path; the ±7 % noise floor |

The last row is the noise control: on a host-expert config `band > 1`, so both arms run the same code (the logs
show `width band 16` in both) and the difference is measurement noise -- which is the scale every delta above is
read against.

## 4. Gates (r185 tree, canonical build `5639690d2`)

* `scripts/gate-spec-width-band.sh` (new) -- **PASS**: 4 slots 363-366 -> 425-432 t/s, the bound fires as
  `aligned draft depth seq n: 4 -> 3 (width band 16, drafting 4)`, never on the knob-off arm; 1 slot 250 both
  (`fa36ef316479`), 2 slots 265 both (`e846c29f763b`/`b02d535e449d`), no abort, GPU journal clean.
* `scripts/gate-spec-align.sh` -- **PASS**, r176 numbers unchanged: 2 GPU 113.8 -> 146.7 (+29 %), 3 GPU 139.4 ->
  178.3 (+28 %), host-expert 124.5 -> 151.1 (+21 %), split decode steps 58+1 -> 0+1, equal-length arms inert,
  mixed-drafter chain inert (+0.5 %), single-slot text byte-identical on/off (`b8775233a93a`).
* Coherence against a kept r184 build (same seed, `ROCBLAS_USE_HIPBLASLT=0`): **byte-identical** on all five arms
  -- 4B Q8_0 3 GPU and 1 GPU `8d89080db673`, dense 27B 3 GPU and `--spec-type draft-mtp n3` `bda6ce08c7e1`,
  35B-A3B `-ncmoe 40` `265c381dbc11`.
* MTP Protocol A (27B UD-Q4_K_XL, 3 devices, prose, `-n 3000`, bf16 KV): layer plain 25.0 -> **61.7** t/s,
  tensor plain 50.8 -> **121.0**; r176 recorded 25.0/50.9 -> 62.1/120.9 on the same protocol at `-n 2000`.
  Adaptive R axis (recall, `--reasoning on`, `-n 3000`, 1 GPU): 67.0 t/s and `893b1d5a3322` on **both** builds.
* `scripts/gate-prefill-logits.sh`: **mean KLD 0.000000 / same-top-p 100.000 %** (no prefill kernel touched).
* `scripts/gate-per-block-build.sh --fast`: **16/16** built (896 s).
* `scripts/validate-set.sh`: strict 16/16 `git am` from a fresh tarball, tree `f8eaf8f005b44a11bad1a619f0c9b07b0559f64b`,
  checksums and both Windows scans clean.

## 5. Why #169 is not the vehicle, and the rule that replaces it

1. Its premise ("above ~8 tokens each extra draft token costs more than it gains") is **false as a rule**: at 16 vs
   8 rows the sign is shape-dependent and 16 wins on 3 of the 4 fixed-depth shapes; on the adaptive arm the best
   cap is 3 (width 16), not 1.
2. Its knob is a **per-shape gamble** the user has to tune blind: `=8` is -7..-12 % on the shapes this delivery
   recommends, -22 % at 2 slots on the contributor's own model, and +4..+8 % on that model at 4 slots.
3. Its **structural** point is still unaddressed in the patch (the adaptive controller is fed `n_accepted`
   against its own `n_cur`, not against the capped depth), and since r153 a per-call cap also interacts with the
   alignment, which the patch does not touch.
4. The value it was reaching for is **delivered inside** the drafter, default-on, with no user tuning: a width
   bound, killable with `LLAMA_SPEC_ALIGN_WIDTH_BAND=0`.

And the tempting shortcut is wrong: **a model or shape whitelist must not be added.**  The sign of the effect
flips on the *same* model across quant (IQ3_XXS +7.9 %, IQ3_S +4.6 %), device count, slot count and speculator
(fixed 3 vs adaptive 7), so a whitelist would be a coincidence fitted to this box.  The variable that governs it
is the verify width, `n_slots x (1 + depth)` -- so the rule is expressed on that, evaluated per draft call.

## 6. Reproduce

```bash
# the PR's own arms, and the width sweep (the archived kit; BIN = a build of the CURRENT tree + the PR patch)
cd ~/llama-cpp-rdna-boosts/archive/work/spec-batch-budget/retest
BIN=<build>/bin/llama-server REPS=2 ./run-arms.sh gsq-hybrid      # the contributor's shape
BIN=<build>/bin/llama-server REPS=2 ./run-arms.sh all-vram         # the shapes that rejected v1

# the shipped fix (one binary; the env var is the only arm selector)
./scripts/gate-spec-width-band.sh --keep
#   LLAMA_SPEC_ALIGN_WIDTH_BAND=0 reproduces the unbounded arm on any band-0 multi-slot run
```
