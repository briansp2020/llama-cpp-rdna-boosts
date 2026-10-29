# O2 — windowed / streamed KV: the ask, the only plan, and why the last two sessions drifted

Written 2026-10-29, after the maintainer said plainly: *"we've done everything BUT the thing that I asked
for. All I know about O1 and O3 is that I never asked for them."*

That is correct. This file exists so it cannot happen a third time.

---

## 1. What was asked for

> a **Strata-style windowed KV cache / streaming** — the KV's VRAM footprint is a **fixed window**, not
> `O(-c)`, so a large context stops costing VRAM the expert arena (or anything else) could use.

That is **O2**, and nothing else in this campaign is it.  `README.md` §5's option table (O1-O6) was a menu,
and the two options that got built (`O1 --kv-grow`, `O3` drafter ring) were picked out of that menu by the
agent, never requested.

## 2. Why it kept drifting (the real cause, worth stating)

**The ask cannot be measured on this box right now, and each session quietly substituted something that
could.**

* O2's payoff exists only where the per-token attended set is **bounded** (`README.md` §4).  For full
  attention it is not: a token attends the whole used context, so every non-resident byte crosses the link
  every step — measured, `-nkvo` on a 4B at depth 4096: **100.67 -> 38.71 t/s**.  The eligible families are
  **qwen4exp QSA** (`indexer.top_k 2048 x compress_ratio 4` = 8192 cells of 262144, 3.1 %) and **DSA**, plus
  the drafter.
* The qwen4exp model on this fleet is Flash-Next at **75 GB**, and **one 32 GiB R9700 is attached**.  The
  shape that would *show* the win (small window, arena big enough for the expert set, 2-3 GPUs) cannot be run.
* So session 1 built the VMM probe + O1's primitive and called O1 "the cheap win, do it first"; session 2
  built O1 (the KV region buft) and measured a regression; session 3 built O3 (measurable on one card,
  tiny prize).  All three were measurable.  None was the ask.

**Rule from here: no substitute.**  If the ask cannot be measured, say so and either build it for
*correctness* (explicitly, with no perf claim) or stop.  Do not pick a different option off the menu.

## 3. What O2 actually is

Per `STRATA-KV-STREAMING.md` (§1, §3) mapped onto this tree:

| piece | this tree |
|---|---|
| **host master** — pinned, device-mapped, authoritative K/V (+ indexer keys), laid out exactly like the resident cache | new: we have no host KV at all |
| **writer rule** — write the host copy *always*, the resident slot only if the cell is resident | the KV writers are `SET_ROWS` / fused rope kernels; a streamed writer must do both |
| **residency map** — `cell-run -> slot`, or `-1` | new; `page = compress_ratio` cells for QSA |
| **in-graph resolve** — one kernel per attention call: stamp the touched cells, pick victims for the misses, copy them in | new; it must be **in the graph** (our decode is a captured HIP graph, so a host-side remap between steps is impossible) |
| **the reader** | the whole cost: free for QSA (its reader already walks a block list), expensive for full attention (this is the eligibility rule, not an implementation detail) |
| **VRAM** | a fixed pool of `W` cell-runs, known at context creation -> trivially reserved before the slab, no VMM needed (provider **P0**) |
| **budget cost** | a residency map is ~`n_cells/page x 20 B` = 1.2 MiB at 245760 with a 4-cell page (`INVENTORY.md` §3) |

**Constraint recorded from O3's session:** do *not* extend O2's page table / host master to the MTP draft
context.  The drafter only reads its last `W` cells, so a placement window on a small resident cache is
enough (`O3-DRAFTER-RING.md` §2.2.2) — a ring and a page table on the same cache would double the
indirection for nothing.

## 4. First deliverable (correctness, not performance)

Because the payoff shape is unmeasurable here, the first slice is explicitly a **correctness scaffold**, and
it must say so in its own results:

1. **Host master for one QSA-eligible cache**, pinned and device-mapped, authoritative; every writer writes
   it.  Oracle: `seq_rm` / `seq_cp` / `defrag` / state save-restore only touch the host copy.
2. **Residency map + in-graph resolve** for the target's attention cache, with a window `W` the user can
   force **far below** the context (the way `LLAMA_KV_REGION_MAP_MAX_MIB` was used to force grow refusals).
3. **Gates**: (a) text byte-identical to the fully-resident arm at `W << context` — the window must move
   *bytes*, not values; (b) hit/miss counters proving the residency actually engaged (a run with 0 misses
   tests nothing); (c) no GPU fault; (d) the map's memory cost as budgeted in §3.
4. **No performance claim.**  Record the miss rate and the modelled link cost (`bytes/token ÷ 28.8 GB/s`,
   the measured link constant) instead — that is honest on this box and is exactly what the 2-3 GPU box needs
   to finish the comparison.

Only after that: the payoff measurement, which needs the **full-power hardware** (Flash-Next + 2-3 GPUs) —
and the deferred O1-vs-O2 crossover question (`README.md` §4.1) rides on the same measurement.

## 5. Closed, do not restart

| option | state |
|---|---|
| **O1** `--kv-grow` (lazy physical mapping) | **CLOSED as a detour.** Built (slice 1+2), measured, and it *regresses* a config that uses most of `-c` on a slab-armed MoE model (`IMPLEMENTATION.md` "KNOWN LIMIT").  Not on O2's path.  Keep only its **buffer-type skeleton** as the container O2 needs. |
| **O3** drafter ring | **CLOSED as a detour.** Built (patch `0003`), pure, measured: ~free at a usable window, but the prize is one draft layer (~1 GiB at `-c 262144`) and the freed VRAM does not reach the arena (`O3-DRAFTER-RING.md` §2.2).  Not a step toward O2; revisit only as a *companion* to O2 (§2.2.2), never as a standalone project. |
| **O4** indexer diet, **O5** non-KV metadata, **O6** provider P0 | not started; O6 is partly subsumed by O2's pool. |

Nothing from `wip/kv-residency/` is in `patches/`, `release.json` or any block, and none of it may be
promoted.  The fork branch `wip-kv-residency` is the record.
