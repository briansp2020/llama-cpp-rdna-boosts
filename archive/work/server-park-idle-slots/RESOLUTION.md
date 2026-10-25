# server-park-idle-slots — PROMOTED (r136, block 12)

**Status: promoted into the delivery as `v14-3f03b1772-r136` (block 12), 2026-10-25.**  The campaign is
closed; this directory is the record.  Nothing here is applied by `scripts/apply-all.sh` (the delivery
patches are the source), and `wip/` no longer carries the campaign.

## What was promoted

PR #225 (`@overdoingism`) adds a staging API (`llama_memory_seq_park()` / `llama_memory_can_park()`) that
moves a sequence's cells to the top of a unified KV cache, and a server policy that parks the resident idle
slots at task start and moves the slots that run back to the low cells.  A request that runs alone then
attends over the same range at the same cell indices as with `LLAMA_SERVER_LAZY_IDLE_SLOTS=0`, restoring
the sequential reproducibility that #210's resident idle slots had broken.

The change is folded into **block 12**, in these files:

- `src/llama-kv-cache.cpp`, `src/llama-kv-cache.h`, `src/llama-kv-cells.h`
- `src/llama-memory.cpp`, `src/llama-ext.h`
- `common/common.cpp`, `common/common.h`
- `tools/server/server-context.cpp`

New variables: `LLAMA_SERVER_PARK_IDLE_SLOTS` (default on, `0` keeps #210) and `LLAMA_KV_PARK_DEBUG`
(diagnostic).  No block renumbered, merged or split.

## Delivery fix on top of the PR

The PR's fallback saves and clears an idle slot that cannot be parked.  With `--no-cache-idle-slots` there
is no saved copy to restore and the lazy idle slots are meant to stay resident, so the delivery leaves the
slot resident in that case (`SLT_DBG`) instead of throwing its KV away.  With the default
`--cache-idle-slots` the save-and-clear fallback is unchanged.

## Gates (the promotion evidence)

- `scripts/validate-set.sh`: green (checksums, fresh-tarball **strict 14/14** `git am`, applied tree ==
  `release.json` tree).
- `scripts/gate-per-block-build.sh --changed-from pre-r136`: **2/2** (block 12 and the tip; the other 12
  blocks' deltas are unchanged and were skipped).
- `scripts/gate-prefill-logits.sh`: **mean KLD 0.000000 / same-top-p 99.980 %** on the release build.
- Coherence vs the r135 build: dense 4B 1 GPU and 3 GPU `-sm tensor` byte-identical (timing line filtered).
- Server gates: see `EVIDENCE-2026-10-25.md` for the parking tables and the fallback arms.

## Limitations (as shipped)

- Parking restores only the **one-at-a-time** case.  Two slots decoding at the same time still depend on
  placement (`GREEDY-PURITY.md` §38).
- A slot whose cells are not one run of its own (shared through `seq_cp`, e.g. a cached prefix restore)
  cannot be parked and falls back to save-and-clear by default.
- A cache that cannot park (not one stream, cells shared with another cache or a draft, a transposed V,
  a sliding window) keeps #210's behaviour and logs once.  `-fa off` is the common case.
- Not covered by the reporter or by us: MoE with the expert cache, vision/M-RoPE, more than 4 slots,
  Windows; multi-GPU was smoke-tested only (2 GPU `-sm tensor` on a 4B).

## Provenance

The original PR note and patch are preserved as `README.md` and `server-park-idle-slots.patch` in this
directory.  The delivery's folded block 12 is the shipped source; the PR's own patch file is not applied.
Release record: `WORKLOG.md` 261025.006.  Rules: `patches/README.md` (block 12), `ENVIRONMENT.md` §15,
`GREEDY-PURITY.md` §57, `MANIFESTS.md` row 12.
