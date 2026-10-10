# FINDINGS — removal of the DEVMAP (device-remap) path (2026-10-10 session)

**Status: DEVMAP is REMOVED from the working tree, the acceptance gate is green, and the decision is to
never reintroduce it.**  WIP record — the delivery `patches/`/`release.json` still contain the old
device-remap code and must be regenerated to ship this (see §6).

## 1. Decision (maintainer, 2026-10-10)

Drop the device-remap path entirely — code and all associated environment variables — and ship the
eager host-routing path as the one and only `-sm tensor` split path.  Rationale: it has been a
multi-session time sink, it is a fundamentally flawed approach, the eager path is much faster on
prefill and within noise on decode, and the device path's own decode advantage was unproven once its
divergence was understood.

**Acceptance gate (maintainer):** the post-removal `-sm tensor` split path must achieve the same or
better performance than the current `DEVMAP=0` path, be MTP-pure on qwen4exp, and fully resolve
`TODO.md` #50.

## 2. Why it was dropped — the divergence was never the map

The multi-session hunt for the single-sequence `DEVMAP=1` divergence concluded in the previous session
that the device path built its remap from a stale `slot_dev` generation.  A residency-generation repair
was implemented and **did not remove the divergence** (budget sweep still 64, `none` vs `n3` still 9),
and the divergence was invariant across `PREFILL_SEED`/`KSLOT`/`ADMIT`/`DEVPOLICY` and across an eager
promotion with a device read.  The actual cause is a **layout-sensitive numerical flip in the external
stack**:

* `ROCBLAS_USE_HIPBLASLT=0` makes the **budget sweep** and the `n_rs_seq` 0-vs-3 A/B pure — the known
  ROCm hipBLASLt per-solution nondeterminism (issue #67 / ROCm/rocm-libraries#12126).
* It does **not** fix the field `none` vs `draft-mtp n3` path (still 65), and a second layout-sensitive
  mechanism remains there.
* **ROCm 10.1.0 does not fix it** either: `DEVMAP=1` budget sweep 9 (7.14: 64), `none`/`n3` 9; with
  hipBLASLt off it is pure on both, i.e. the new library still has the same hipBLASLt flip.
* Disabling hipBLASLt costs **~12-17 % prefill** on gfx1201/ROCm 7.14 (`1472/1587 -> 1299/1314`) and ~0 %
  plain decode — so it is not a free workaround here (the README's "no measurable cost" was the
  reporter's Windows/ROCm-10 box).

So the device path's reads were correct; the divergence was the library/layout.  Paying a 12-17 %
prefill tax to keep a ~14 % decode path that is *smaller than the prefill loss* is not a trade worth
making — hence the removal.

## 3. What was removed

2684 deletions / 71 insertions across 10 files (`remove-devmap.diff` in this directory).  Removed
entirely:

* **Env vars / globals:** `MOE_EXPERT_CACHE_DEVMAP`, `MOE_EXPERT_CACHE_DEVPOLICY`,
  `MOE_EXPERT_CACHE_DEVPOLICY_SPLIT`, `MOE_EXPERT_CACHE_KSLOT`, `MOE_EXPERT_CACHE_DEV_EAGER`,
  `g_devmap`, `g_devpolicy`, `g_kslot`, `g_dev_eager`, the arming state, and `g_devmap_gen_stale`.
* **State:** `slot_dev`, `used_dev`/`used_host`/`used_host2`/`slot_pin`, the device-policy descriptor
  array and device mirrors (`policy_dev_t`, `slot_expert_dev`, `count_dev`, `ghost_dev`, `last_dev`,
  `slot_prov_dev`, `pool_slot_dev`), `devmap`, `remap_fresh`, `res_gen`/`dev_gen`, and the device
  prefill tally.
* **Functions/kernels:** `moe_cache_build_remap_kernel`/`launch_remap`, `moe_cache_policy_kernel` and
  the whole device-policy pipeline, `moe_cache_promote_host` (deferred promotion),
  `moe_cache_get_slot`/`kslot_active`, `moe_cache_sibling_down`, the device prefill tally/lazy seed
  (`moe_cache_tally_prefill`, `seed_prefill_lazy_locked`, `apply_prefill_seed_rank_locked`), the
  cache-aware whole-table staging / in-place reads (`moe_cache_stage_table`, `moe_cache_inplace_*`,
  `moe_cache_seed_fill_kernel`, `moe_cache_tail_kernel`, `moe_cache_stage_kernel`), `moe_cache_guard_stats`,
  `moe_cache_read_check`, and the device-policy selftest.
* **Plumbing:** the `moe_cache_promote` and `moe_cache_inplace` backend hooks (CUDA / Meta / CPU / RPC),
  the scheduler's `moe_promote_rec` deferred-promotion pass and `need_promote`, the KSLOT in-kernel slot
  lookup in `mmvq.cu`, the MMQ in-place prefill read, and the `moe_cache_get_table` `moe_cache_devmap`
  out-param.

**Kept:** the eager host-routing path (`moe_cache_update_host` → materialized `remap_dev`), the
**identity fast path** (fully-resident tables read the arena with raw ids), the per-table fusion guard
(`moe_cache_table_serves`, the #50 repair), cold UVA reads, the host pool, and the device-side expert
gather.

## 4. Split geometry is unchanged (verified)

The split/mirror decision stays in the meta backend; the removal does not touch it.  Post-removal
`CACHE_GEOM` on qwen4exp `-sm tensor`:

```
ffn_gate_exps  expert_bytes=189440/284160  host_bytes=473600  src_off=0/189440  split_axis=1
ffn_down_exps  expert_bytes=368640/552960  host_bytes=921600  src_off=0/144     split_axis=0 host_pitch=360
```

i.e. `expert_bytes < host_bytes` with non-zero `src_off` — **split, not mirrored**.

## 5. Acceptance gate (passed)

| check | result |
|---|---|
| qwen4exp `-sm tensor` `none` vs `draft-mtp n3` | **None** (pure) |
| budget sweep MIB8000 vs MIB14000 | **None** (pure) |
| MTP n3 perf (1000 tok, 32k, 2 GPU, `ce`, MIB=14000) | prefill **1581.7** t/s, decode **71.76** t/s, acc 0.77017 |
| plain perf | prefill **1469.8** t/s, decode **41.60** t/s |

Against the pre-removal `DEVMAP=0` baseline (MTP prefill 1587.0 / decode 72.56 / acc 0.77017; plain
prefill 1471.9 / decode 41.80) every number is **within run-to-run noise** (<= ~1 %).  `TODO.md` #50 is
resolved: the per-table fusion guard is the only guard, there is no device-remap path to disagree with it.

## 6. Shipping

The working tree is clean of DEVMAP and builds (`llama-server`, `llama-cli`,
`test-logits-width-probe`).  `remove-devmap.diff` is the diff against the pre-removal tree.  To ship it
as the next release, regenerate the block patches from a canonical fork rebuilt at `release.json.base`
with this change applied, update `release.json`, re-run `scripts/validate-set.sh` + the prefill-logit gate,
and refresh the fork's `rdna-boosts` branch (per `AGENTS.md`).  This is a delivery-affecting step and is
**pending the maintainer's go-ahead** (reported, not yet cut).

## 7. Pointers

* `HANDOVER-devmap-consistency.md`, `FINDINGS-single-seq-rollback.md` §5b-§5g, `FINDINGS-devmap-generation-fix.md`
  — the (now-superseded) device-path investigation.
* `remove-devmap.diff` — the removal diff.
* `TODO.md` #50 (resolved), #51/#52 (separate).
