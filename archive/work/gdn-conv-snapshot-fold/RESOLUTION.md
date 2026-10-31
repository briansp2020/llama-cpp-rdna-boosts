# RESOLUTION -- `gdn-conv-snapshot-fold` (PR #177, @briansp2020)

**Status: NOT SHIPPED (reviewed 2026-10-17, for `v12-097cf33d9-r81`) -- rejected on purity, not on
speed.  Re-entry conditions below.**  The contributor's format-patch and write-up are preserved here
unchanged.  Nothing was left behind as a `wip/` stub (the campaign arrived as a PR branch, never on
`main`'s `wip/`).

## What it does

With the direct GDN conv path active, a GDN layer makes three launches per multi-token step
(`gdn_concat_tail`, the batched snapshot `CPY`s, and `gdn_conv_direct_kernel`).  The patch has the direct
conv kernel also write the 3-column recurrent snapshot tails, read straight from `state` / `x`, so the
`CONCAT` and the snapshot `CPY`s launch nothing.  It is taken only after a runtime byte-span check over the
nodes between `CONCAT` and `SSM_CONV` (no writes over `state` / `x`, no reads/writes of a snapshot
destination, destinations disjoint from the inputs and from each other, and a no-op node right after each
skipped `CPY`).  Kill switch `GGML_CUDA_DISABLE_GDN_CONV_SNAP=1`.

## Why it was not taken

**On 35B-A3B the patched path is not run-to-run deterministic, and it does not reproduce the stock
logits.**  The delivery's purity rules do not allow either, regardless of the throughput win.

Reproduced on gfx1201 (3x R9700, ROCm 10.1.0, r80 against r80 + the patch, tree `84ea4301`),
`test-logits-width-probe` on Qwen3.6-35B-A3B Q4_K_M, **1 GPU**, P=256:

| arm | runs | `W8` hash | row-0 hash |
|---|---|---|---|
| stock r80 | 5+ | `dfcb48a5be95c5ac` (stable) | `a44efecb25966236` |
| r80 + #177 | 4 | `48272966cab954fb` x2, **`cb29e653528bddcf` x2** | `907b3e9c40b8c546` / **`361d76c6d5ff877e`** |
| r80 + #177, `GGML_CUDA_DISABLE_GDN_CONV_SNAP=1` | 2 | `dfcb48a5be95c5ac` | `a44efecb25966236` |
| r80 + #177, `GGML_CUDA_DISABLE_GRAPHS=1` | 2 | `48272966cab954fb` | `907b3e9c40b8c546` |

- `width_purity` still reports **maxdiff 0 within a run** (W=1..8 agree), so the widths are consistent
  with each other; the difference is **between runs**, and it is already in row 0, i.e. in the prefill.
- The kill switch reproduces stock exactly, so the patch is the cause.
- Disabling HIP graphs makes the patched arm stable but at a **third** value, so capture is a second,
  independent divergence on top of the base one.
- The dense 27B is identical to stock at both P=256 and P=3584, and the 400-token greedy text on
  35B-A3B is stable and identical to r80 across 3 runs.  So this is a logits-level prefill effect that
  the same-seed text gate cannot see; the prefill-logit KLD gate also passes (0.000000 / 100.000 % on
  both models), which is why the probe -- not the text or KLD -- is the instrument that caught it.

**The speed win is real and should not be lost:** dense 27B Q4_K_XL MTP decode 1 GPU 52.8 -> 53.3-53.4 t/s
(**+1.0 %**, two rounds, 2 runs each), reproducible; MoE 2-GPU tensor tg256 and pp31500 flat.

## Re-entry conditions

1. **Determinism first.**  The probe's `W8` / row-0 hashes must be identical to stock r80 across at least
   5 runs, with HIP graphs on, on 35B-A3B **and** the dense 27B, at P=256 and P=3584.  A same-seed text
   match is not sufficient evidence: the current patch has it.
2. Name the mechanism behind the graph-on divergence.  The prime suspect is the write moving from the
   `CPY` node to the `SSM_CONV` node while the graph allocator's lifetime for the snapshot destination is
   still anchored at the (now empty) `CPY` node, so a captured graph can replay against a different
   `snap.dst` address than the one it was captured with -- the same class as "a Q8_1 arena must outlive
   captured graphs" and "a graph is recaptured when the pool memory it captured is freed".  Confirm or
   refute before re-submitting.
3. Keep the kill switch and the byte-span check.  Both worked.
4. Report the launch counts from the trace (`gdn_concat_tail` / `cb_kernel` gone) next to the t/s, plus
   the probe hashes per run, so determinism is visible in the report.

## Record

`WORKLOG.md` 261017.008 (r81); review reply on the PR.

---

# Second review (2026-10-20, r99 tree) -- still NOT SHIPPED, and the reason has narrowed

Re-reviewed because three later changes reinforced determinism in this area (r90's snapshot of every
split-input upload, r94's `ggml_cuda_graph_mem_freed` on a slab-boundary shrink, r96's second-consumer
split).  The patch was re-cut onto the r99 tree with one conflict resolution (the RDNA3_0 verify-batch
comment in `gdn_conv_check`; everything else applied cleanly) and built with `~/bin/build-llama-rocm`.

**The fold is now proven arithmetically exact on the dense model, and the r81 dense claim holds.**  On
Qwen3.8-27B UD-Q4_K_M, 1 GPU, `test-logits-width-probe` is **bit-identical to stock r99 and stable** at
P=256 (`W8=0e33b420b37beba4`, row-0 `bd6dfe0733110bf7`) and P=3584 (`W8=9c12636638742221`, row-0
`88005e3ac5d99aed`), 2/2 runs each.  The fold fires (`snap=1 n_snap=1`, `tail_from=256`).

**35B-A3B (qwen35moe) is still the blocker, and the bisect says the snapshot write is not the cause.**

| arm (35B-A3B Q4_K_M, 1 GPU, `test-logits-width-probe`, clean release-style build) | P=256 `W8` | P=3584 `W8` |
|---|---|---|
| stock r99 | `97ad459c74d8a694` (3/3) | `d24b26226690e95b` (3/3) |
| r99 + #177 | `29cb3722a842846a` (8/8, row-0 `1b25846f508af2ed`) | **`2b1caf3bd5cfe4e3` x3, `c2336484ebf6dab2` x2** |
| r99 + #177, `GGML_CUDA_DISABLE_GDN_CONV_SNAP=1` | `97ad459c74d8a694` | `d24b26226690e95b` |

* The kill switch reproduces stock exactly, so the patch is the cause.  A debug build (extra host syncs
  and dumps, not shipped) produced yet another P=3584 value (`c9c1dae73dfc53d2`), i.e. at least three
  distinct results overall.
* `GGML_CUDA_ASYNC_INPUT=0` moves the patched arm to a third value at P=256 (`423d3d144ee3ffc9`);
  `GGML_CUDA_DISABLE_GRAPHS=1` leaves `29cb3722a842846a`.  Not the expert cache
  (`MOE_EXPERT_CACHE_MIB=0`), not scheduler staging (`GGML_SCHED_STAGE=0`), not `GGML_SCHED_DEVGATHER=0`,
  not `GGML_CUDA_GDN_GATHER_INDEX=0`, not `GGML_CUDA_FUSE_GDN_STATE_GATHER=0`: all still `29cb3722a842846a`.
* **Keeping the snapshot CPY reproduces stock exactly** (`97ad459c74d8a694`) even though the conv kernel
  also writes the same destination, and a full-region hash of the destination is identical after both
  writes.  So the fold's write is byte-exact; the CPY's write is not what the difference rides on.
* A node-by-node F32 hash of the prefill (debug build) shows layer 0's conv-state gather,
  `conv_output_raw`, `conv_output_silu`, the conv-state write and the SSM state gather all identical, and
  the **first differing node is the `ssm_alpha` LoRA `MUL_MAT` (`node_39`)** -- upstream of every write
  this patch moves.  Adding `gdn_concat_tail` alone (`GGML_CUDA_DBG_NOTAIL`, an extra kernel that writes a
  buffer nothing reads) moves the result to a *third* value, so this model's prefill is sensitive to the
  launch sequence per se, not to what the fold writes.

**What would unblock it.**  A deterministic 35B-A3B result is required before this can ship, whichever
side of the boundary above the cause sits on: the probe's `W8`/row-0 hashes must match stock r99 across 5
runs each at P=256 and P=3584 with graphs on, on 35B-A3B **and** the dense 27B.  (The related open item is
`TODO.md` #75 -- 35B-A3B reproducibility -- whose gfx1201 witness still has to be re-established; see
`WORKLOG.md` 261020.013.  Note that a *terminal* text hash is not a valid instrument for this:
`llama-cli` streams with `\b` corrections, so use `scripts/extract-generated.py`.)  Nothing about the fold
itself needs to change: the kill switch, the byte-span check and the `n_snap`/`snap_ok` guards all behaved
as specified, the dense result is bit-identical, and the 27B +1.0-1.2 % MTP decode win is real.  Record:
`WORKLOG.md` 261020.014 (r100), comment on the PR.

# Third review (2026-10-31, r185 tree) -- CLOSED: superseded, and the blocker is gone

Re-opened for one last integration attempt and closed the same session, because there is nothing left to
integrate.  Three independent reasons, any one of which would be enough:

1. **This PR carries no code.**  Its single commit (2026-10-17, never updated) adds two files under
   `wip/gdn-conv-snapshot-fold/` -- one `format-patch` plus a `README.md` -- i.e. a `wip/` proposal, not a
   fork patch.  Both files are **byte-identical** to the copies already committed in this directory
   (sha256 `21b2337048e53151b3300da73514c8e97ecb7792e3d7063e05b1c1c8c7d7858d` for the patch,
   `65ceb24631190f22a0e145495152414487856239be42a37d91156c46aa3d6510` for the README).  They are here, and
   not in `wip/`, because the campaign closed and the no-stub rule moved them.
2. **The change shipped in r183** as block 10, via the submitter's own PR #256 patch `0001`, which is a
   strict **superset**: same `gdn_conv_snap_safe` byte-span check, same `gdn_conv_snap_args`, same
   `GGML_CUDA_DISABLE_GDN_CONV_SNAP`, *plus* the batched-copy run stop at a folded snapshot `CPY`
   (`cpy-batch.cu: if (ggml_cuda_gdn_conv_snap_try_fuse(cgraph, j)) break;`) that this PR lacked -- without
   it `cpy_batch` can swallow a folded snapshot `CPY`.  All 98 substantive added lines of the PR's patch are
   present in the tree (three of them in extended form, carrying the r183 `norm` / `state_ids` /
   `state_row` arguments), and `patches/0010-...-block-10-....patch` carries the lot.
3. **The blocker recorded above is gone.**  Re-measured on the shipped r185 build (gfx1201, 1 GPU, HIP
   graphs on) with `test-logits-width-probe`, the fold firing per `GGML_CUDA_CONV_DEBUG=1`
   (`GDN_CONV_DIRECT C=8192 T=256 tail_from=256 silu=1 snap=1 n_snap=1`, and `snap=0` with the kill
   switch):

   | model | depth | folds on `W8` / row-0 | vs `GGML_CUDA_DISABLE_GDN_CONV_SNAP=1` | runs |
   |---|---|---|---|---|
   | 35B-A3B UD-Q4_K_M | P=256 | `efb1ac6b4a863719` / `aa6057acd67bd109` | identical | 3/3 |
   | 35B-A3B UD-Q4_K_M | P=3584 | `36026b682d8aa1fc` / `23f86e866c6e5804` | identical | 3/3 |
   | 35B-A3B UD-Q4_K_M | P=4096 | `493a0a43ad3927cd` / `bbbfff452c95a46a` | identical | 2/2 |
   | 27B UD-Q4_K_M (dense) | P=256 | `b43fb4ee01d6b7ab` / `44fd7c8f3b062cfd` | identical | 3/3 |
   | 27B UD-Q4_K_M (dense) | P=3584 | `cb7acb2f9013340d` / `0acd13d1c5084da2` | identical | 3/3 |

   `width_purity=PASS` (worst maxdiff 0) in every arm, both models, all three depths;
   `scripts/check-gpu-faults.sh` clean.  The r100 failure mode (`2b1caf3bd5cfe4e3` x3 /
   `c2336484ebf6dab2` x2 at P=3584) does not reproduce: that was the TODO #75 launch-sequence sensitivity
   of this model's prefill, not this fold, exactly as the node-by-node hash above said.

**What was never used.**  The multi-device data point: `GGML_CUDA_CONV_FUSION_MULTI=1` was bit-identical on
2 x R9700 for 27B tensor/layer, 35B-A3B tensor and the hybrid model.  The direct conv's single-device gate
still stands, because of the 3 x R9700 `-sm tensor` divergence that has never been root-caused; that
divergence, not this PR, is what keeps the r183 folds off the recommended multi-GPU server shape.

**Outcome: CLOSED as superseded** (comment on the PR, credited).  The gate gap it exposed -- r183 shipped
these folds with only text + the `-ub 1` prefill-logit arm, both blind to a launch-order change -- is closed
by `MANIFESTS.md` release gate 14, the block-10 per-block row, `scripts/validate-gpu.sh` step 4c,
`GREEDY-PURITY.md` §73 rule 6 and the `AGENTS.md` r183 bullet.  Record: `WORKLOG.md` 261031.002.
