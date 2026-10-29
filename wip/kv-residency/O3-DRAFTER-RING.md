# O3 — the drafter ring (windowed draft KV): design note

Started 2026-10-29, per the scope decision in [`README.md`](README.md) §4.1 (O1 parked; **O3 first, then
O2**) and `TODO.md` §117.

**Nothing here is delivery work.**  It describes the campaign's next slice before any of it is written.

---

## 1. What O3 is, and why it is first

The MTP draft context carries a KV cache of its own, sized `-c` cells (1 layer): measured 256 MiB at
`-c 65536` on the 27B, 1 GiB at `-c 262144` (`INVENTORY.md` §1.A/§2).  The drafter only ever needs its
**recent** prefix — Strata's `--mtp-window` (default 32768) is exactly this — so O3 replaces
`O(n_ctx)` draft KV with a **fixed window** of `W` cells and a static residency map
(`block -> block % n_slots` in Strata; see `STRATA-KV-STREAMING.md` §1).

It is *approximation* by design, and legal only because **the target verifies every token**: a
window-limited draft can change speed, never an answer.  That makes its cost measurable as a single number
(draft acceptance / accepted-per-step), which is why it is the right rehearsal for O2's policy.

Honest value statement: O3's *standalone* VRAM win on the box we have is modest (256 MiB at `-c 65536`,
1 GiB at `-c 262144` on the 27B — one draft layer).  Its real value is that it is the **smallest complete
instance** of the machinery O2 needs: a fixed window, a residency/placement map, a documentable acceptance
cost, and a policy that must survive rollback (a rejected draft), `seq_rm`, and a prompt-cache resume.
O2's K/V page table is the same three pieces with a bigger payload.

## 2. The finding that should make this cheap: the SWA cache is already a windowed cache

`llama_kv_cache_iswa` sizes its sliding-window cache to the window, not to the context —
`llama-kv-cache-iswa.cpp:73`:

```c
uint32_t size_swa = GGML_PAD(std::min(size_base, hparams.n_swa*(unified ? n_seq_max : 1) + n_ubatch), 256);
```

So `llama_kv_cache` **already** runs correctly with fewer cells than the positions it holds: cells are
recycled as the window slides, positions stay absolute, and the mask is limited by
`llama_hparams::is_masked_swa()` (`llama-kv-cache.cpp:1476`, `:2137`, `:2803`).  That is precisely the ring
O3 wants, and it is already gated, tested and shipped (block 05/12 territory).

**Therefore the first thing to try is not a new ring implementation — it is to construct the draft
context's cache with a sliding window of `W` cells.**  If that holds, O3 becomes a *configuration* change
plus an env knob, and the interesting work moves to proving the acceptance cost and that rollback is
sound.

## 3. Invariants the slice must keep

1. **The target is untouched.**  Only the draft context's cache gets a window.  A wrong draft is a slower
   run, never a wrong answer; the target still verifies every token.
2. **Positions stay absolute.**  A window limits *which cells are attended*, never the position a token is
   placed at (RoPE, and `get_n_kv`'s comparison against the cache's positions, both depend on this).
3. **Rollback must still work.**  A rejected draft is rolled back with `seq_rm` (+ `seq_cp` for the accepted
   prefix).  A windowed cache must not resurrect cells whose position has been re-used by a newer token —
   this is the one place a ring can be silently wrong, and it is what the gate has to check.
4. **The window must be at least the verify width.**  The draft attends the cells it just wrote plus the
   prefix the target's verify step needs; `W < n_max + 1` cannot work.  Clamp to a floor (a few hundred
   cells) rather than trusting the knob.
5. **Default OFF, and no effect when off.**  Same rule as every WIP slice: `LLAMA_SPEC_DRAFT_WINDOW` (0 =
   today's behaviour, byte-identical).
6. **`-ctkd`/`-ctv` interaction (r125).**  The draft KV follows the main `-ctk`/`-ctv` unless pinned.  A
   window changes the cell *count*, so any VRAM arithmetic in the docs/`--fit` must use the windowed size.

## 4. What it must not do

* Do not touch the main (target) cache's `n_swa`/`swa_type` — a dense model's full attention must stay full.
* Do not make the window the default in `wip/` (the promotion rule inverts a default only after the gates).
* Do not claim an acceptance number without the MTP methodology's rules: `-n 3000` (`-n 2000` floor),
  `--reasoning on` for R and off for P/C/K, and the device count quoted with the number
  (`benchmarks/mtp-adaptive-methodology.md` rule 0; AGENTS.md "MTP gates must be long and reasoning-pinned").

## 5. Plan

**Step A — can the draft cache take a window at all?**  Find where the draft context's memory is constructed
(`llama_context::create_memory`, driven by `cparams.ctx_other` / the draft model) and whether `n_swa` can be
forced there for the draft only.  Cheap experiment: force `n_swa` on the draft cache and run a greedy
`--spec-type draft-mtp` arm; the text must be identical and the draft cache log line must show ~`W` cells.

**Step B — the knob and the floor.**  `LLAMA_SPEC_DRAFT_WINDOW=N` (cells, 0 = off), floored at
`n_max + n_ubatch`.  Log the draft cache's size so the VRAM win is visible in the KV log line.

**Step C — correctness of rollback under a window.**  The oracle is not text: run at a window small enough
to overlap constantly (e.g. 512 cells against a 3000-token run) and require (i) no abort, (ii) the target's
greedy text unchanged from the unwindowed arm (it must be — the target verifies), (iii) no cell whose
position was reused is attended after a rollback.  A purpose-built check beats a text diff here; the
`LLAMA_RS_ROLLBACK_DEBUG` / `LLAMA_KV_CACHE_DEBUG=2` style of trapdoor is the precedent.

**Step D — the acceptance cost, measured properly.**  `llama-cli --spec-type draft-mtp -n 3000` on the 27B,
window off vs `W` in {4096, 16384, 32768, 65536}, reporting acceptance and t/s on both split modes, on the
device counts the box allows (**one R9700 today — say so**).  Compare against the recorded reference
(3 devices: layer 51.7 / tensor 90.7 t/s, acceptance 0.62 / 0.60; `AGENTS.md`) only where the device count
matches.

**Step E — the crossover question that parked O1.**  With a window in hand, the O1-vs-O2 question from
`README.md` §4.1 becomes measurable *in principle*: past the window, is `W`-windowed streaming cheaper than
a shrinking arena?  Answering it properly still needs O2's host master, so record the windowed arm's numbers
now and leave the comparison to the O2 session.

## 6. Gates this slice owes

| gate | bar |
|---|---|
| coherence / purity | target greedy text **byte-identical** with the window off and on (the target verifies; nothing may move) |
| MTP Protocol A | `-n 3000`, reasoning-pinned, both split modes, device count quoted; report acceptance and t/s against the unwindowed arm |
| rollback | the Step C oracle: no abort, no reused cell attended after `seq_rm` |
| VRAM | the draft context's `llama_kv_cache: size = …` line scales with `W`, not `-c` |
| prefill logits | not required for a draft-only change (`gate-prefill-logits.sh` measures the *target*'s prefill), but state that explicitly rather than leaving it implicit |

## 7. Open questions for the maintainer (do not guess)

1. **Is the window an approximation we ship on by default** (Strata ships `--mtp-window 32768` default-on and
   reports the cost), or opt-in only?  `README.md` O3 lists it as a shipping candidate; the delivery's
   default-on policy says a beneficial, gated feature is ON — so this needs the gate numbers first.
2. **One window size, or per-model guidance?**  Strata's 32768 is their tuned value; ours would be
   payload- and acceptance-dependent (`blocks 13`'s `--spec-draft-mtp-vocab` had the same shape of answer).
3. **Does the same window belong on a DFlash2 draft** (a separate draft model, `-md`), or is this MTP-only?
