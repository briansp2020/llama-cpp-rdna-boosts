# `lfru-host-weight-migration` — LFRU host→GPU slow migration of hot weights (TODO, parked)

> **CLOSED 2026-10-11 — DELIVERED / INTEGRATED, in a different form than described here.**  The maintainer
> confirmed the item is done: the delivery's **device-resident MoE expert cache** (`moe-expert-cache.cu` /
> `MOE_EXPERT_CACHE_MIB`, the movable-boundary slab, the decode/verify band) plus the **bounded pinned host
> pool** (`--host-experts pool`, r35) provide the hot-weight residency/tiering this note describes.  No
> separate LFRU migration is needed.  The `TODO.md` parked entry was removed and `wip/CAMPAIGNS.md` no longer
> lists it.  The text below is the original (unstarted) scoping note, kept for context; **do not act on it.**

The original scoping note follows.

---

**Status: PARKED (seeded 2026-10-11), now CLOSED.** Not part of the delivery; nothing here may be applied to the fork
without the maintainer's go-ahead (see the WIP and promotion rules in `AGENTS.md`). This is the parked TODO
entry *"LFRU host→GPU slow hot-weight migration"* (the survivor of the expert-tiering experiment, dropped
2026-09-05).

> **Cold start.** Read this file, then `archive/work/qwen4exp/LRU_EXPERTS.md`,
> `archive/work/qwen4exp/PHASE0_ROUTING.md` and `archive/work/qwen4exp/HANDOVER-2026-09-04-tiering.md`.
> Because the host-expert machinery changed a lot in r33/r35/r37, re-check the "does this still exist"
> questions below **before** investing. Report findings to the maintainer and wait.

## 0. The one idea left

The expert-tiering experiment was dropped because most of its aims were already covered by current
llama.cpp options. The single surviving idea: an **LFRU-style very slow migration of hot weights from host
to GPU** — a persistent GPU slot cache for the hot experts, with a CPU-computed cold tail (instead of
per-token UVA/demand fetch).

Design notes: `archive/work/qwen4exp/LRU_EXPERTS.md` (reviews the never-merged upstream PR #26563 —
decay-tracked heatmap, per-layer hot store, LUT/cold-mask routing), `PHASE0_ROUTING.md`,
`HANDOVER-2026-09-04-tiering.md`.

## 1. Current reality check (r48) — several of the premises moved

The host-expert path changed substantially after the 2026-09 design notes were written:

- **r33** removed the pageable `--host-experts mmap` mode and the legacy `LLAMA_MMAP_HOST_EXPERTS` env
  (issue #116: a no-XNACK gfx1201 cannot read a pageable host master in a kernel). `--host-experts` now
  selects **`pinned`** (default) or **`pool`**.
- **r35** added the bounded pinned host pool (`--host-experts pool`, `MOE_HOST_POOL_MIB` default 25 % of
  the MoE host expert bytes via `MOE_HOST_POOL_FRAC`). The pool is a **pinned host** L2 over the page
  cache — *not* GPU migration.
- The **MoE expert cache** (`MOE_EXPERT_CACHE_MIB`, `moe-expert-cache.cu`) already provides a
  **device-resident slot cache** over the host master, with the movable-boundary slab and the decode/verify
  band (r31). So "persistent GPU slot cache for the hot experts" is substantially what the delivery
  already ships; the open question is whether an **LFRU-style slow migration** (long-decay heat, migrate
  *weights*, not per-token slots) beats the current per-ubatch/pruned fill for the *cold* tail.
- **r37/r40** made the slab MoE-only and the tensor-split device policy eager/host-routed (the old
  `MOE_EXPERT_CACHE_DEVPOLICY_SPLIT` remap family is gone).

**Re-scope before coding:** read the current `moe-expert-cache.h`/`.cu` and `--host-experts` docs and
write down which LFRU behaviours are *already* delivered. Only then decide whether the remainder (CPU-
computed cold tail vs UVA miss, heat-driven weight migration) is still worth pursuing.

## 2. Why it is parked

- CPU-computes-misses was measured **negative above a ~1.2 GiB arena** (the loss is scheduler dispatches +
  cross-backend copies, not CPU compute) — see `archive/work/moe-expert-cache/WORKLOG.md`.
- Genuine CPU/GPU overlap for misses is a separate open campaign
  ([`moe-cpu-overlap/`](../../moe-cpu-overlap/README.md), TODO #36) and is the more concrete version of the
  same "use the CPU for the cold tail" idea.
- The delivery already has a device-resident slot cache; the LFRU idea's remaining delta is unclear.

## 3. If resumed

- First deliverable is a **one-page "what is already covered"** comparison (LFRU ↔ `moe-expert-cache` /
  `pool` / `moe-cpu-overlap`) and a measured statement of what the migration would add. Only then a design.
- Any prototype stays env-OFF under `wip/`, needs a kill-switch, and is validated on the full feature set
  (default-on policy: never benchmark a partially-enabled path).

## 4. Pointers

- The removed `TODO.md` parked entry (this file is its record).
- `archive/work/qwen4exp/LRU_EXPERTS.md`, `PHASE0_ROUTING.md`, `HANDOVER-2026-09-04-tiering.md`.
- `archive/work/moe-expert-cache/WORKLOG.md` (why CPU-misses lost).
- `src/llama-model-loader.cpp` (`MOE_HOST_POOL_*`), `ggml/src/ggml-cuda/moe-expert-cache.h` (the pool + slot
  cache options), `ENVIRONMENT.md` §§1.2.1/4.
- [`moe-cpu-overlap/README.md`](../../moe-cpu-overlap/README.md) (TODO #36 — the live CPU/GPU-overlap
  campaign; do not duplicate it).
