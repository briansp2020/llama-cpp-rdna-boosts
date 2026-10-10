# HANDOVER — fold the DEVMAP removal + the non-devmap WIP fixes into the delivery blocks (option B)

**For a cold session.  Read end to end before touching the delivery.**  This continues the DEVMAP removal
recorded in [`FINDINGS-devmap-removal.md`](FINDINGS-devmap-removal.md).  The working-tree removal and its
acceptance gate are **done**; what remains is to fold the change into `patches/`/`release.json` **without
re-adding the device-remap code**, and to fold in the **non-devmap WIP fixes** that were validated
alongside it.

---

## 0. Decision (maintainer, 2026-10-10)

> Keep **every non-devmap-specific WIP fix** (they were hard won).  Drop fixes that only exist to support
> the broken device-remap machinery.  The shipped delivery must not contain the device-remap code at all
> ("poisoned code is not even included in the first place"), so the fold is an **amendment to block 06/15**,
> not a new block.

Acceptance gate: the reconstructed `-sm tensor` split path must match or beat the old `DEVMAP=0`
performance, be MTP-pure on qwen4exp (`none == draft-mtp n3`), and fully resolve `TODO.md` #50.

## 1. State as handed over

* `~/llama.cpp` is the **uncommitted WIP tree**, on `rdna-boosts` at the block-15 tip `ff94a733c`:
  * **DEVMAP fully removed** (`remove-devmap.diff`, `grep -c 'DEVMAP\|g_devmap' == 0`).
  * All the WIP fixes since block 15 present (`wip-since-block15.diff`, 4037 lines).
  * Built and validated: `none == n3` on qwen4exp `None`; budget sweep `None`; MTP prefill 1581.7 /
    decode 71.76 / acc 0.77017 vs the old `DEVMAP=0` 1587.0 / 72.56 / 0.77017; plain 1469.8 / 41.60 vs
    1471.9 / 41.80 (within noise); split geometry confirmed (`CACHE_GEOM`: `expert_bytes < host_bytes`,
    `src_off != 0`).
* Delivery repo docs updated and committed (`05c0a66`): `FINDINGS-devmap-removal.md`,
  `ENVIRONMENT.md` REMOVED note, `TODO.md` #50 resolved, `wip/CAMPAIGNS.md`.
* **`patches/`/`release.json` are NOT regenerated** — they still contain the device-remap code.

## 2. Why this is not a `git apply` of `remove-devmap.diff`

The block chain is `base a55e952b8 -> block 00 b4951bf0d ... block 06 4d1e1f99e ... block 15 ff94a733c`.
The device-remap code is created **wholesale in block 06** (the cache file is 5480 lines there and already
has `g_devmap`, `policy_flush`, `promote_host`, `launch_remap`, `tally_prefill`, `get_slot`, …).  Blocks
13 and 15 only tweak it (17 net devmap lines; block 13 adds the KSLOT consumer in `mmvq.cu`).

`remove-devmap.diff` is a diff **against the WIP tree**, so a `git apply --3way` onto block 06 applies the
deletions but **drags in WIP-only changes** from the conflict context (the per-table fusion guard, the
level-3 validator, `GGML_CUDA_CACHE_REDIR_DEBUG`, the `GGML_CUDA_CACHE_STANDDOWN_*` bisection knobs).
That is not acceptable in a "remove devmap" amendment.  **Strip block 06 against its own content.**

## 3. WIP-change classification (from `wip-since-block15.diff`)

**KEEP (non-devmap fixes/features):**
* **narrow-2 slab** (the M1 MTP-draft COMPUTE-aliasing fix): `ggml/include/ggml-backend.h`
  (`ggml_backend_dev_slab_narrow2_floor`, `..._slab_compute_narrow2`), `ggml-cuda-vmm.h`,
  `src/llama-context.cpp` (route the draft context's reserve/alloc to the pinned narrow-2 region),
  `ggml-cuda.cu` (the slab narrow-2 region).
* **per-table fusion guard** (the #50 repair): `ggml-cuda.cu` (`ggml_cuda_cache_blocks_fusion` asks
  `moe_cache_table_serves`), `moe-expert-cache.cu`/`.h`.
* **`-ncmoe` host-expert offload** (`NCMOE=N`): `src/llama-context.cpp` / loader.
* **level-3 slot-content validator**, the `MOE_EXPERT_CACHE_ADMIT`/`_TOUCH`/`_COLD_UVA`/`_DEVPOLICY_SPLIT`
  knobs, `GGML_CUDA_CACHE_GEOM_DEBUG`, and the `test-logits-width-probe` changes.
* `GGML_FORCE_N_RS_SEQ` (diagnostic; keep if harmless).

**DROP (devmap-only — must not ship):** the `res_gen`/`dev_gen` generation guard, `MOE_EXPERT_CACHE_DEV_EAGER`,
`g_devmap_gen_stale`, the eager-path `slot_dev` republish, the `used_dev` zeroing, the `slot_dev` clears,
the **level-4** validator, the `GGML_CUDA_CACHE_STANDDOWN_GATEUP/DOWNFOLD` bisection knobs, the
`GGML_CUDA_CACHE_REDIR_DEBUG` block, and the device prefill tally / lazy seed (the sizing-time host seed
may stay).

> Classify hunk-by-hunk against `wip-since-block15.diff`: keep it if it fixes/improves the **eager**
> path; drop it if it only feeds the device `slot_dev`/device-policy/KSLOT machinery.

## 4. Procedure

1. `git worktree add -f /tmp/rdna-fold ff94a733c` (do **not** disturb `~/llama.cpp`).
2. Reconstruct a clean chain, or work directly on the SCRATCH worktree's `rdna-boosts` commits.  Amend
   **block 06** (`4d1e1f99e`) with a **clean block-06 strip**:
   * `strip_devmap.py` (in this campaign's session notes; brace-matches and deletes the devmap functions
     that exist at block 06) for `moe-expert-cache.cu`, then hand-remove the remaining globals/struct
     members/`parse_env`/`alloc_table_locked`/`stand_down`/`shrink`/`release`/`evict`/`update_host`/
     `get_table`/`table_serves`/`take_over`/`redirect_fused`/`validate`/`report`/`selftest` references
     (the compiler flags what is missed).
   * The plumbing files (`ggml-backend-impl.h`, `ggml-backend-meta.cpp`, `ggml-backend.cpp`,
     `ggml-cuda.cu`, `ggml-cpu.cpp`, `ggml-rpc.cpp`) apply cleanly with the removal diff — but **take
     only the devmap hunks** (do not import the WIP per-table guard / narrow-2 / validator hunks there;
     those belong to the "keep" fold below).
3. **Rebase blocks 07-15** onto the amended 06.  Conflicts are expected only in blocks **13** (KSLOT in
   `mmvq.cu`) and **15**; resolve by **dropping the devmap hunks** (keep the non-devmap changes in those
   blocks).
4. Fold the **KEEP** WIP changes into the topical blocks (block 06/13/15 for the cache/meta/scheduler;
   the narrow-2 work belongs with block 06's slab work; `-ncmoe` offload with block 14/15).
5. Regenerate: `scripts/make-patches.sh <scratch-fork> <base> <new-tip>` then `scripts/make-release.sh`.
6. **Re-validate the DELIVERY tree, not the WIP tree**: fresh clone at `release.json.base`,
   `scripts/apply-all.sh`, build, then the full gate — qwen4exp `none == draft-mtp n3` (`None`), budget
   sweep `None`, ≥1000-token decode + 32k prefill perf vs this session's numbers, `scripts/validate-set.sh`,
   `scripts/gate-prefill-logits.sh`, `test-backend-ops -o MUL_MAT_ID`, coherence.
7. Only then refresh the fork's `rdna-boosts` branch (personal fork only) per `AGENTS.md`.

## 5. Gotchas

* **The acceptance gate was run on the WIP tree.**  The delivery's block 06 still has the **global**
  `moe_cache_has_arena()` fusion guard — the #50 defect — and the WIP per-table guard is not in the
  delivery.  A devmap-only amendment would ship the #50 defect, so the per-table guard fold (step 4) is
  **required** to meet the gate; and the gate must be re-run after the fold.
* Do not import devmap-only diagnostics (§3 DROP) into any block.
* `remove-devmap.diff` and `wip-since-block15.diff` are both **against the WIP tree**; treat them as
  references, not as patches to apply to the block chain.
* Leave `~/llama.cpp`'s WIP tree in place unless the maintainer says otherwise.

## 6. Pointers

* `FINDINGS-devmap-removal.md` — the removal + acceptance record.
* `remove-devmap.diff` — the devmap removal (WIP-tree based).
* `wip-since-block15.diff` — everything since block 15 (WIP-tree based).
* `HANDOVER-devmap-consistency.md`, `FINDINGS-single-seq-rollback.md` §5b-§5g — the superseded device-path
  investigation.
* `TODO.md` #50 (resolved), #51/#52 (separate).
