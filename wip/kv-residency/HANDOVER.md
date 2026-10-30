# HANDOVER — `wip/kv-residency` (TODO #117), cold-session start

**Rewritten 2026-10-29 at the end of the fourth session.**  Read this top to bottom before touching anything;
it is self-contained.  **The plan of record is [`O2-PLAN.md`](O2-PLAN.md)** — read it immediately after this
file, then [`README.md`](README.md) §4.0 (the goal) and §4.1 (the speed rule).

> **The one thing to know:** three sessions were spent building options off this campaign's own menu that the
> maintainer never asked for (`O1 --kv-grow`, `O3` drafter ring).  The ask is a **windowed KV**, Goal A.  It has
> **not been started**.  Do not substitute.  See §9.

---

## 1. State in one screen

| what | where |
|---|---|
| delivery repo | `~/llama-cpp-rdna-boosts`, branch `main`, pushed to `origin` (`git@github.com:stew675/llama-cpp-rdna-boosts.git`) |
| release the campaign is cut against | **r174**, `v15-6050b467c-r174`, base `6050b467c`, tip `5bdec04ab`, 15 blocks |
| campaign code | **fork branch `wip-kv-residency`** in `~/llama.cpp`, local only: `973e166e1` (O1 slice 1) → `6bf445b34` (O1 slice 2) → `a93dee7a9` (O3) |
| the release chain | `rdna-boosts` in `~/llama.cpp` — **still `5bdec04ab`, untouched.** Nothing in this campaign is in `patches/`, `release.json` or any block |
| patch snapshot (the backup) | `patches/0001`…`0003` + `patches/README.md` + `patches/diffstat.txt` (1076 insertions, 6 files) |
| **Goal A (the ask)** | **not started.**  Build spec: [`O2-PLAN.md`](O2-PLAN.md) §6; the mechanism is corrected (§6.3) and A1's edit points are known (§5 below) |
| O1 (`--kv-grow`) | built, measured, **PARKED** — it regresses a config that uses most of `-c` on a slab-armed MoE model (`IMPLEMENTATION.md` "KNOWN LIMIT") |
| O3 (drafter ring) | built, pure, measured, **Goal C / last** — `O3-DRAFTER-RING.md` |
| box | **one R9700 (gfx1201), 32 624 MiB.**  Pin `HIP_VISIBLE_DEVICES=0` on every arm.  Every number below is 1-GPU; say so, never claim multi-GPU |

## 2. The ask, in the maintainer's words (2026-10-29)

> a **Windowed KV** … for **fully resident dense/MoE models**, because doing so would allow users to run large
> contexts with a Q8_0 quantization of Qwen3.8-27B, or a Q6_K quantization of 35B-A3B, on a 32 GB card, and
> **not run out of VRAM** … and for **host-resident MoE's (slab mode)**, because it would free up more VRAM for
> the decode arena, allowing for higher expert hit rates, leading to higher decode speeds.  Once that is in
> place, we could then look at improving the drafter via a ring too.

| # | goal | mechanism | status |
|---|---|---|---|
| **A** | **fittability** on dense/full attention (+ free VRAM for the slab arena) | **static split**: recent `W` cells resident, the tail read in place from a pinned device-mapped host master | **the ask, next** |
| **B** | a *sparse* attended set streamed cheaply (QSA/DSA) | residency **page table** (the attended set is scattered, so reuse exists) | later; needs Flash-Next + 2-3 GPUs |
| **C** | the drafter ring | built (O3) | done, last priority |

**Opt-in semantics (agreed):** default **off** = byte-identical to today **including the OOM**; the user opts in
to "the tail may spill into system RAM"; the warning is loud and always printed; `--fit` reports the split
instead of refusing.

## 3. Why Goal A is not a speed feature (do not re-derive this)

For **full attention** every decode step reads the whole used context, so:

* the streaming cost is **deterministic**: `spilled_bytes / 28.8 GB/s` per token (link constant measured,
  `INVENTORY.md` §1.D; a kernel reading pinned host gets 29.0 GB/s, VRAM 565 GB/s);
* the comparison is **"windowed vs OOM"**, not "windowed vs resident";
* **there is no temporal reuse**, so an LRU/CLOCK page table is pointless for Goal A — *which* cells are
  resident does not change the traffic.  Hence a **static split**.  The page table is Goal B.
* **no staging ring**: a copy to VRAM costs the same one link crossing and adds VRAM pressure (the scarce
  resource).  Read the tail in place.  Overlap hides latency, not bandwidth — the link is ~93 % of the wall
  time at a deep tail, so overlap's ceiling is the compute fraction (~7 %).

**The failure this fixes, measured on this box** (reproduce it before building):

| config | KV | result |
|---|---|---|
| 27B Q8_0 (27.1 GiB) `-c 262144`, `-ctk/-ctv q8_0` | 8704 MiB | **`cudaMalloc failed: out of memory`** |
| 27B Q8_0 (27.1 GiB) `-c 131072`, f16 KV | 8192 MiB | **`cudaMalloc failed: out of memory`** |
| 27B Q8_0 (27.1 GiB) `-c 131072`, q8_0 KV | 4352 MiB | fits, ~nothing left over |

Expected envelope at `-c 131072` on the 27B (34 KiB/cell, modelled): resident window 98304 → ~16 t/s,
65536 → ~10 t/s, 32768 → ~7 t/s; full speed below the window.  **The claim is "runs where it used to OOM", not
"fast at 262 K".**

## 4. The mechanism (corrected 2026-10-29 — the combine route is dead)

**`ggml`/CUDA FA exposes NO log-sum-exp** (no `lse`/`softmax_lse` under `ggml/src/ggml-cuda/fattn*`), and two
normalised attention halves cannot be combined without each half's row sum.  "Two FA calls + a combine op"
would need an `lse` epilogue in every FA family — numerics changes in the most tuned code in the tree.  **Dead;
do not revive it.**

**Instead: a two-base K/V read inside the existing kernels.**  FA already takes one `KV` base plus a stride and
indexes rows as `KV + i*stride_KV`, with the *caller* forming the per-tile base — so a split is an
**addressing** change:

* the op carries `k_tail`, `v_tail` and the tail's cell count `n_tail`;
* where a tile's base is formed, rows `< n_tail` take the tail base, the rest the head base;
* **the KV index space stays GLOBAL** ⇒ the mask, `cell_pos`, `tok_lo`/`tok_hi` and the derived-mask form are
  untouched.  No mask slicing, no combine, no second graph node;
* `n_tail = 0` indexes exactly as today (**bit-identical by construction**), and `n_tail > 0` keeps the
  accumulation **order** — same rows, same order, only a different base ⇒ A1's oracle is a real equality.

**A1 needs no cache-representation change:** the two segments are two **views of the same KV tensor** — tail =
cells `[0, s)`, head = cells `[s, n_kv)`.  State io, defrag, `clear`, `seq_rm` and every cell-range path stay
untouched.  Only **A2** moves the tail's storage (and only then do state save/restore and defrag need a second
pass).

**Coverage:** implement the two-base read in the **RDNA band** (`fattn-rdna.cuh`; RDNA4 + head 256 + GQA 5..12 +
`n_q <= 8` is the 27B's decode — the case that matters) and the **tile** family (`fattn-tile.cuh`, ~103 KB, the
kernel body; `fattn-tile.cu` is only the instantiation TU).  Any other family/shape **declines the split**, and
declining = **no spill** = today's fit-or-fail path.

## 5. A1 — the next work, with its anchors

| step | where | note |
|---|---|---|
| 1. op plumbing | `ggml/include/ggml.h` (`ggml_flash_attn_ext`, line ~2524; the `_add_kq_derived`/`_set_n_kv_max` setters are the pattern), `ggml/src/ggml.c`, `ggml/src/ggml-cuda/fattn.cu` | `k_tail`/`v_tail` as extra srcs + `n_tail` in `op_params`; **inert when 0** |
| 2. the two views | `llama_kv_cache::get_k/get_v` (`src/llama-kv-cache.cpp:1712`, `:1735`) and `llama_kv_cache_context::get_k/get_v` (`:3560`) | return the head view as today plus an optional tail view; the split point `s` comes from a knob |
| 3. pass them through | `llm_graph_context::build_attn_mha` (`src/llama-graph.cpp:3139`) — the single funnel both `build_attn` variants use (`:3325`, `:3416`) | only place the op is created |
| 4. tile family | `fattn-tile.cuh:456-715` (`KV + i*stride_KV` indexing) | prove **byte-identical text** at a forced split |
| 5. band | `fattn-rdna.cuh` | re-prove; this is the 27B decode path |

**Oracle at every step:** same-seed greedy text **identical** to the unsplit arm at several forced split widths
(use `scripts/extract-generated.py`, and *not* with `-v`).  Plus `scripts/gate-prefill-logits.sh`.

**A2** = move the tail view's storage to the pinned host master (a buffer-type change) → same text, measured
cost.  **A3** = auto-size the window from free VRAM, the opt-in flag, the loud warning, `--fit` reporting, and
the `MemAvailable` gate.  Details: `O2-PLAN.md` §6.

## 6. What is already measured (do not re-derive)

**Link constants** (`INVENTORY.md` §1.D, `bw-probe.cc`): H2D/D2H pinned 28.8/28.2 GB/s, kernel read of pinned
**29.0**, VRAM **565**, D2D 291.

**O1, slice 2 (PARKED)** — `IMPLEMENTATION.md`: same-seed text identical off/on/tiny-unit (4B 147 chars
`b065f37593b3`; 35B-A3B 572 `40e29c2b46b6`; 27B+MTP 89 `392abe9d2bd6`); prefill-logit KLD **0.000000 /
100.000 %**; `SLAB_INIT` at `-c 262144` moved free 17.46 → 22.46 GiB and arena 15.88 → 20.88 GiB.  **And it
regressed**: 35B-A3B `-ncmoe 24 -c 262144` on a 253k-token prompt completes with the region off (1201.2 /
51.2 t/s) and refuses at 41216 cells with it on, because `SLAB_INIT` had already given the freed VRAM to the
arena.  That is the lesson behind "the split is decided before `SLAB_INIT`".

**O3 drafter ring (Goal C, patch `0003`)** — `O3-DRAFTER-RING.md`: pure (text identical, 27B 303 chars
`c437759e06e3`, 35B-A3B 262 `e9db5cb8db0b`); the draft KV shrinks exactly (27B `-c 12288`: 192 → 72 MiB at a
4096 window); **at a usable window it is free** (46k context, 27B `-c 65536`: acceptance 0.5499 → 0.5371,
gen 50.61 → 50.79 t/s) and **at a too-small window it is not** (0.6381 → 0.5712, −7 % at 4096 against a 5 k
context).  Its prize is one draft layer (~1 GiB at `-c 262144`) and the freed VRAM does **not** reach the
arena (the draft KV sits inside the slab's `aux_reserve`).

## 7. Invariants that still hold (from the first session, unchanged)

* **Policy in the memory module, physical from a provider.**  Residency metadata belongs in `llama-kv-cache*`;
  physical comes from P0 (plain pool + pinned master), P1 (a private VMM region) or P2 (a slab carve).
* **The slab stays MoE-only** (`ggml_cuda_slab_enabled()` = `env && g_slab_armed`, armed only by the host-expert
  preflight).  Nothing in this campaign may *require* the slab.
* **Never map/grow anything past what is resident** — reading reserved-but-unmapped VA faults the GPU
  (`VMM-PROBE.md`).
* **A refused residency must never reach `ggml_cuda_device_malloc`'s OOM chain** (it frees the expert arena; the
  r117 identity cliff is 46.7 → 65.0 t/s).  Decline loudly instead.
* **Purity:** residency moves bytes, never values.  Oracle = byte-identical greedy text **and**
  `gate-prefill-logits.sh`.
* The host master is **pinned** (gfx1201 has XNACK off → a kernel cannot read pageable host memory), so the
  spilled part is non-swappable RAM: 4.25 GiB at `-c 131072` q8_0 on the 27B, 8.5 GiB at 262144.  Gate on
  `MemAvailable` (`MOE_HOST_POOL_RAM_PCT` pattern).

## 8. Environment and pitfalls (this box, this tree)

* Build: `~/bin/build-llama-rocm` (ROCm 10.1.0, `build-rocm/`, gfx1201, ccache).  **Never** pass
  `-DCMAKE_HIP_FLAGS`.  A warm `llama-cli` relink is ~5 s; a `.cu` delta 15-30 s; a widely included header
  3-5 min.  Keep ccache ≥ 30 GiB.
* `LD_LIBRARY_PATH=/opt/rocm-10.1.0-gfx120X/lib`; **`HIP_VISIBLE_DEVICES=0`** always (the iGPU is a second HIP
  device the build does not support).
* `llama-cli` prints its INFO/accounting lines **only with `-v`** — and `-v` interleaves log text into the
  generation, so **hash text from a run WITHOUT `-v`**, and use `~/llama-cpp-rdna-boosts/scripts/extract-generated.py`
  (a naive `sed`/`grep` slice does not reproduce the hash; the backspace-correction stream is why).
* **Never** run two GPU jobs at once.  Always `--single-turn --no-display-prompt`, always `timeout`.
* `llama-bench` has no `-c` in this tree.  `llama-cli` has no `--prompt-cache` any more (state save/restore is
  reachable only through the server; `scripts/gate-ckpt-reuse.sh` drives it).
* Streaming a `-v` run into `grep` floods: the whole generation arrives as JSON chunks.  Capture to a file and
  extract the **last** `"draft_n"/"predicted_per_second"` pair (see the O3 session's python snippet pattern).
* `scripts/check-gpu-faults.sh --since "HH:MM" --log FILE` is the fault check (defaults to "today"; the journal
  still holds 10 gfxhub faults from 2026-10-29 15:33:40 that predate this campaign's sessions).
* Models: `/llm/models/Qwen3.8/27B/Q8_0/…` (27.1 GiB, the Goal-A target), `Qwen3.8/27B/Q4_K_XL/…` (**use this one
  for iteration** — Q8_0 leaves no headroom), `Qwen3.6/35B-A3B/Q4_K_M/…` (MoE + slab),
  `Qwen3.5/4B/Q8_0/…` (fast smoke).  Flash-Next (75 GB) does **not** fit one card — Goal B territory.
* `prompts/prose-rdna-boosts.txt` (5256 tokens at the 27B) and `prompts/kld-corpus-35b-in-domain.txt` (23 k
  tokens) are convenient; a 2x/9x `cat` of the latter makes 46k/253k-token contexts.

## 9. Scope discipline (the rule that exists because of three wasted sessions)

1. **The campaign is Goal A, then B/C.  No other option off the `README.md` §5 menu is in scope.**
2. **If the ask cannot be measured on the attached hardware, say so** — then build it for *correctness* with no
   perf claim, or stop.  **Never substitute something that is measurable.**
3. **Do not pick scope.**  Present findings; the maintainer decides.
4. **Surface findings and open questions before a delivery-affecting commit**, not after.
5. Nothing from `wip/` goes into `patches/`, `release.json` or any block without the promotion rule.

## 10. Records

| record | content |
|---|---|
| [`O2-PLAN.md`](O2-PLAN.md) | **the plan of record**: the three goals, the root cause of the drift, the Goal A build contract (§6: opt-in rule, where the split is decided, the corrected mechanism, the A1/A2/A3 order) |
| [`README.md`](README.md) | the campaign: §0 read-first, §4.0 the ask + the OOM evidence, §4.1 the speed rule (Goal B), §4.2 the scope decision, §5 the option menu (**mostly out of scope**) |
| [`INVENTORY.md`](INVENTORY.md) | T1-T5 taxonomy, 4 measured shapes, the link constants, per-token arithmetic |
| [`VMM-PROBE.md`](VMM-PROBE.md) | the 9 driver tests + the two traps (unmapped-read fault, stale fresh mapping) |
| [`STRATA-KV-STREAMING.md`](STRATA-KV-STREAMING.md) | Strata's `kv_stream.hpp` read in full and mapped onto this tree |
| [`IMPLEMENTATION.md`](IMPLEMENTATION.md) | O1 slice 1 + slice 2 (PARKED) with the measured regression |
| [`O3-DRAFTER-RING.md`](O3-DRAFTER-RING.md) | Goal C: design, implementation, measurements, both corrections |
| `patches/` | `0001`-`0003` + `README.md` + `diffstat.txt` — the campaign's code as applied patches (apply to `5bdec04ab`) |
| `bw-probe.cc`, `vmm-probe.cc` | the two standalone probes |

## 11. First 30 minutes of the next session

1. `cd ~/llama-cpp-rdna-boosts && git pull`; read `O2-PLAN.md` (whole file) and this file.
2. `cd ~/llama.cpp && git branch --show-current` — expect **`wip-kv-residency`** at `a93dee7a9`; if it is
   `rdna-boosts`, `git checkout wip-kv-residency` and rebuild.  Verify `git rev-parse --short rdna-boosts` is
   still `5bdec04ab` (**never** commit campaign code there).
3. `~/bin/build-llama-rocm` (expect ~5 s warm), then reproduce the Goal A baseline: the 27B Q8_0 OOM at
   `-c 262144` (`-ctk q8_0 -ctv q8_0`) from §3.  That is the "before" number.
4. Start **A1 step 1** (the op plumbing, inert when `n_tail = 0`) and confirm the default path is unchanged by
   hashing a same-seed text run against the current binary **before** touching any kernel.
5. Do **not** open the kernel edit (step 4) unless steps 1-3 are done and the inert default is proven — a
   half-changed FA path is the one thing worse than no change.
