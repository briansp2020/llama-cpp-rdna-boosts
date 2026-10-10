# FINDINGS — the single-sequence "rewind restores wrong state" hypothesis (2026-10-12 session, second half)

**Status: the rewind hypothesis is REFUTED. The single-sequence plain-vs-MTP divergence is the target's
`n_rs_seq` (0 vs 3) graph × the expert cache, NOT the recurrent rewind.**

**Methodology note (2026-10-12, second half):** every "pure"/"impure" verdict below is on **real
generated content**, verified by dumping the decoded completion text (`tools/probs.sh` → the response
`content`), not a fixed template prefix.  The field prompt's completion is an open-ended table
continuation (the model invents the next `13-fused-moe-glu-mmq.patch` row), and the `none`/`n3` arms
generate the same text until the first-diff token.  The cache-off and 1-GPU "pure" arms likewise generate
full 80-token continuations and match.  The divergences are genuine **argmax** changes (e.g. at the
`n_rs_seq` flip, plain picks tok 80 at `-0.24` while rs=3 picks tok 85 at `-0.35`), not just logit hashes.

All numbers below are on the D2 model (GSQ-IQ3_XXS + the shared Q8_0 MTP head, `-ncmoe 48`), prompt
`/tmp/srr/mixed30k.txt`, `NP=128`, `-ctk/-ctv q8_0 -fa on -t 8 --fit on`, greedy. 2-GPU-text A/Bs force
`GGML_CUDA_ALLREDUCE=ce` (the only deterministic AR on 2 GPUs); 1-GPU and 3-GPU runs need no pin. Tools:
`tools/rrewind.cpp` (new, single-context), `tools/rollback-replay.cpp` (the mstep-style two-context
replay), `tools/arm.sh` (arbitrary server arm with `return_tokens`).

## 1. The instrument: a single-context rewind probe (`rrewind`)

The two-context `mstep`/`rollback-replay` design compares a plain reference *context* against a
verify+rollback *context*; with a partial expert cache the two contexts do not share a cache state, so
the comparison is confounded (see §4).  `rrewind` removes that: **one context**, and at every position it
decodes the same token twice from the same starting state —

* path P (plain): `seq_rm` back to `pos`, decode `[t_k]` → `L_plain`; record the serialized state `S_plain`;
* path V (verify): `seq_rm` back to `pos`, decode `[t_k, junk, junk, junk]` → `L_verify(row 0)`; then the
  MTP rejection rollback `seq_rm(pos+1, -1)` → `S_verify`;
* require `L_verify(row 0) == L_plain` **and** `hash(S_verify) == hash(S_plain)`.

This is the exact single-sequence state transition a rejection causes, isolated from a separate
reference context.  `POLLUTE=N` optionally decodes `N` unrelated tokens (rolled back, so the recurrent
state is unchanged) *between* the two paths, to move the expert-cache resident set exactly as the MTP
draft does.

### Result: the rewind and the verify-width forward are PURE

`W=4 RS=3 N=40 P=2500`, `STATE=1` (serialized-state hashes enabled):

| config | logit mismatches | state mismatches |
|---|---|---|
| 2 GPU, `-sm tensor`, `MOE_EXPERT_CACHE_MIB=14000` (partial, 350/512 slots) | **0** | **0** |
| 2 GPU, `-sm layer`, MIB=14000 (partial) | 0 | 0 |
| 1 GPU, `-sm tensor`, MIB=14000 (partial) | 0 | 0 |
| 2 GPU, `-sm tensor`, cache off | 0 | 0 |
| 2 GPU, `-sm tensor`, MIB=14000, **`POLLUTE=1` / `=3`** | **0** | **0** |

So: the verify batch's row-0 logits equal the plain single-token decode's, and the state after the
rejection rollback is byte-identical to the state after the plain decode — **on the exact configuration
the server divergence needs**.  Moving the cache resident set between the two paths does not change it.
The KV/attention, GDN ssm, GDN conv and QSA-indexer state children all restore correctly; there is
nothing to bisect.

`test-recurrent-state-rollback` passing is therefore not a blind spot here.

## 2. The controlled server matrix (plain `--spec-type none` vs MTP `n-max 3`)

| config | residency | none vs n3 first-diff |
|---|---|---|
| **2 GPU `-sm tensor`, auto** | none 100 %, n3 89.9 % | **9** (n1: 89) |
| 2 GPU `-sm tensor`, cache off | — | **None** |
| 2 GPU `-sm tensor`, MIB=14000 (both arms) | 350/512 slots both | **68** |
| **2 GPU `-sm layer`, MIB=14000 (both arms)** | partial | **None** |
| 1 GPU, auto | none 46.9 %, n3 33.8 % (partial) | **None** |
| **3 GPU `-sm tensor`, auto** | both 100 % | **None** |
| 35B-A3B (non-qwen4exp) 2 GPU tensor, MIB=3000 (partial) | partial | **None** |

Derived conditions:

* **The plain arm is itself cache-sensitive**: `none` cache-on vs cache-off first-diff **103**; `n3`
  cache-on vs cache-off **9**.  So no "plain reference" exists across cache states — this is what makes
  the plain-vs-MTP framing misleading.
* **Full residency removes it** (3 GPU auto; the 2-GPU `none` arm auto grew to 100 %).
* **No split tables removes it** (`-sm layer`, same partial arena).
* **No cache removes it.**
* **It is qwen4exp-specific**: the 35B-A3B MoE is pure at 2-GPU tensor partial, and its plain stream is
  identical full vs partial (`first-diff None`).
* The index moves with width/config (9 / 68 / 89 / 103 / 59) — a near-tie flip, matching the handover's
  "presence, not index, is the signal".

No individual qwen4exp MoE fusion switch removes it on 2 GPU tensor MIB=14000 (all keep `none vs n3 =
68`, acceptance 0.75214 unchanged): `GGML_CUDA_DISABLE_MOE_DOWN_FOLD=1`,
`GGML_CUDA_DISABLE_SHEXP_DOWN_GATE=1`, `GGML_CUDA_DISABLE_WEIGHTED_DOWN=1`,
`GGML_CUDA_DISABLE_HC_FUSION=1`.  (The cache is required, but the *named* qwen4exp fusion hooks are not
the differentiator at this index.)

## 3. Why the earlier "requires rejections / qwen4exp rewind" reduction did not hold

The previous reduction compared `mixed30k` (rejections, diverges) with `prose30k` (acceptance 1.0,
byte-identical) and a **1-GPU** non-qwen4exp control.  The matrix above shows:

* the 1-GPU control was uncontrolled for residency/device count: on **1 GPU the qwen4exp MTP is
  byte-identical** (plain == n3), so that control proved only "1 GPU is pure", not "non-qwen4exp is
  pure";
* prose30k being byte-identical is consistent with a *cache-residency* trigger: the degenerate copy
  prompt holds a different resident set / trajectory, and the divergence is a near-tie, so it is not
  guaranteed to appear on every prompt;
* the "rejections required" reading is explained by #50's trigger, not by the rollback itself: a
  rejection is what makes the MTP arm's verify/decode batch sequence differ from the plain arm's, which
  is what lets the two arms' cache states separate.

The plain arm's own cache-sensitivity (§2, 103) is the single strongest disproof of "the rollback
restores the wrong state": if the rewind were wrong, cache-off would still be wrong (it is pure) and
the plain arm would be cache-invariant (it is not).

## 4. Why the two-context replay was inconclusive

`rollback-replay`/`mstep` runs pass 1 (plain) and pass 2 (verify) as separate contexts.  With a **full**
arena the replay is PURE.  With a **partial** arena it reports a mismatch at the *prefill* row and at
every step — i.e. the two contexts do not share a cache state, so the first row already differs.  That
is cross-context cache drift, not a rollback defect; `rrewind`'s single context is the instrument that
removes it.

## 5. The exact condition (as established)

> **qwen4exp + 2 devices + `-sm tensor` (split expert tables) + a partially-resident MoE expert cache,
> with the plain and MTP arms planning their graphs at different cache/arena states.**  The
> cache-aware path's arithmetic is not invariant to the arena/residency state at graph-plan time, so the
> MTP arm (whose draft context moves the cache) does not reproduce the plain arm's exact trajectory.
> With full residency (3 GPU), `-sm layer`, or the cache off, plain == MTP.

The remaining open question is the precise *code* mechanism (which table/graph flips at plan time — the
`ggml_cuda_cache_blocks_fusion` global gate over a per-table eviction/rearm, per #50, is the documented
candidate).  It is not in the recurrent memory.

## 5b. UPDATE (same session): the plain-vs-MTP divergence is the target's `n_rs_seq` path × the expert cache

After the rewind was ruled out, the remaining structural difference between the two *arms* is the target
context's `n_rs_seq`: the plain arm runs `n_rs_seq = 0`, the MTP arm `n_rs_seq = n_max = 3` (so the GDN ssm
+ qwen4exp conv snapshot path is armed, `n_rs_batch = 4`, and the prefill tail split changes).  A temporary
diagnostic `GGML_FORCE_N_RS_SEQ=<n>` (llama-context.cpp) pins it independently of `--spec-type`, and the
result is decisive (GSQ, 2 GPU, `-sm tensor`, `MOE_EXPERT_CACHE_MIB=14000`, `--spec-type none`, ce):

| arms (everything else identical) | cache | first-diff |
|---|---|---|
| `n_rs_seq=0` vs `n_rs_seq=3` | **on** | **59** (reproducible; A/A and B/B both identical) |
| `n_rs_seq=0` vs `n_rs_seq=3` | off | **None** |
| `n_rs_seq=0` vs `n_rs_seq=3`, `DISABLE_FUSION=1` | on | 65 |
| `n_rs_seq=0` vs `n_rs_seq=3`, `GGML_CUDA_SLAB=0` | on | 9 |

So `n_rs_seq` alone changes the target's forward **only while the expert cache is active**; the change is
independent of the cache-aware fusions and of the slab.  The logits shift is **systematic, not a ULP**: at
the flip (idx 59) the two arms' top-2 differ by ~1.6 nats (plain tok 80 `-0.24` / tok 85 `-3.03`; rs=3 tok
85 `-0.35` / tok 80 `-1.40`), and the runner-up distributions already differ from idx ~57.  Cache
counters confirm both arms take every table over (`takeover=288`), so it is **not** arena-vs-streaming;
the eviction pattern differs (evictions 7462 vs 5770, hits 93.1 % vs 94.5 %), i.e. the `n_rs_seq` graph
moves the per-table residency, which moves the arena/remap arithmetic.

This is the actual `none`-vs-`n3` mechanism: the MTP arm's target is the `n_rs_seq=3` graph, and with the
expert cache active on `-sm tensor` that graph's arithmetic is not bit-identical to the `n_rs_seq=0` plain
graph.  It is **not** the recurrent rewind (`rrewind` proves the rollback restores exact state), and it is
**not** the global fusion guard (the guard decisions are identical; `DISABLE_FUSION` still diverges).

**Why the earlier "rewind + cache" lead was close but wrong in the detail:** the cache is required, and
rejections correlate, but the rollback is exact; what the cache exposes is the *snapshot-armed graph's*
different cache-band arithmetic relative to the snapshot-free graph.  The `n_rs_seq` dependency is a
forward/graph effect, so it is captured by the plain-vs-MTP comparison even with acceptance 1.0 only if
the near-tie happens to hit — which is why prose30k looked pure.

The fast probe (`rreplay` pass-1 plain greedy, cache armed, P=30000, ub=6144) does **not** reproduce the
`n_rs_seq` difference (pass-1 Thash `564c7220336a29cb` identical for rs 0/3), so the effect needs the
server's context shape (large `--fit` context, `n_seq_max > 1`, multi-slot); the reproducing instrument
remains the two server arms with `GGML_FORCE_N_RS_SEQ`.

**Repair direction (this is the `D1` target now):** make the `n_rs_seq > 0` snapshot-armed graph's
cache-band MoE arithmetic bit-identical to the `n_rs_seq = 0` graph's, i.e. find why the expert cache's
residency/remap arithmetic depends on the graph layout (`identity` vs `devmap` vs cold, or a
layout-selected kernel) and pin it.  The per-table fusion guard implemented this session (see §6) is the
directed `#50` repair and is retained, but it is a no-op for this divergence because every table serves in
both arms.

## 5c. UPDATE: why it is qwen4exp-specific — the **split expert-table** cache path

The 35B-A3B (`qwen35moe`) control is **not** a stateless control: it has SSM/GDN layers too (`ssm_conv1d`,
`ssm_a`, ...), and its MoE call is the same `build_moe_ffn` with the same `ffn_gate_up_exps`.  So neither
GDN nor the MoE structure explains the specificity.  The real difference is the **table split geometry**:
with `-sm tensor`, the target's host expert tables are split across devices for qwen4exp but **mirrored**
for the 35B.  Per-table geometry (new `GGML_CUDA_CACHE_GEOM_DEBUG` log, 2 GPU, `-ncmoe`, MIB=14000):

| model / role | `expert_bytes` (per device) | `host_bytes` | `src_off` | `split_axis` |
|---|---|---|---|---|
| qwen4exp `ffn_gate_exps` (dev 0) | 189440 | 473600 | 0 | 1 (contiguous) |
| qwen4exp `ffn_gate_exps` (dev 1) | 284160 | 473600 | 189440 | 1 |
| qwen4exp `ffn_down_exps` (dev 0) | 368640 | 921600 | 0 | **0** (`host_pitch` 360, strided) |
| qwen4exp `ffn_down_exps` (dev 1) | 552960 | 921600 | 144 | 0 |
| qwen35moe `ffn_gate_exps` (both dev) | 589824 | 589824 | 0 | 1 |
| qwen35moe `ffn_down_exps` (both dev) | 720896 | 720896 | 0 | 0 |

So qwen4exp's tables are genuinely **split** (`expert_bytes < host_bytes`, uneven `src_off`, one axis-0
strided slice), while the 35B's are **whole/replicated** (`expert_bytes == host_bytes`, `src_off == 0`).
That is the qwen4exp-specific factor: the divergence lives in the **split-table cache path** (the per-device
arena + remap and the axis-0 2D fill), and qwen4exp is simply the model whose expert tables are split in
this configuration.  Consistently:

* `-sm layer` (tables whole) is pure for qwen4exp;
* the 35B (tables whole) is pure under `-sm tensor`;
* the split-table fill/read geometry is the same across cache budgets, but the **residency** it produces
  differs, which is what the MIB sweep changes;
* **decisive A/B: `GGML_META_SPLIT_COPY=0` (mirror the expert copies, table whole per device) makes
  MIB8000 == MIB14000** (first-diff `None`), while the default `=1` (split tables) gives 59.  So the split
  expert-table cache path is the cause, full stop.

`GGML_META_SPLIT_COPY` (`src/llama-model.cpp:451`) controls this; the default is 1 (a copy inherits its
weight's split), and `=0` keeps every copy mirrored.

## 5d. UPDATE: the split-table **devmap** path — what is and is not the cause

Following §5c, the divergence is confined to **split expert tables** (`GGML_META_SPLIT_COPY=1`, the
default) with the cache in **devmap (partial-residency)** mode:

* the pure cases all reach **identity** mode (`slots >= n_experts`, slot == expert): 3-GPU auto and
  2-GPU auto both size **512 slots/table** = `n_experts`; the divergent budget-pinned cases size
  350 (MIB=14000) or 200 (MIB=8000) → devmap;
* mirrored tables (`GGML_META_SPLIT_COPY=0`) are pure at the same 350/200 slots (whole table per device).

New byte-level validators (`MOE_EXPERT_CACHE_VALIDATE=3/4`, `MOE_EXPERT_CACHE_VALIDATE_LAYER`) were added
to `moe_cache_validate`:

| check | result |
|---|---|
| slot head / full layer-0 slice vs host master, all resident slots | **clean** (0 `SLOT-CONTENT-MISMATCH`) |
| redirected slice strides vs the packed arena (`CACHE_REDIR_DEBUG`) | **match** (`ffn_down_exps` dev0 `nb1=144` == `row=144`, `nb2=368640` == `expert_bytes`; the cold host geometry is `host_pitch=360`/`host_bytes=921600`) |
| device `slot_dev` vs host `expert_slot` | **stale** (`dev=-1` where `host=11`, `slot_dirty=1`) — because the promotion's slot-map H2D is **skipped when the routing list is empty** |
| fusion decisions (`FUSE_TRACE`) and cache-guard state between MIB=8000 and MIB=14000 | **identical** |

A one-line fix was applied for the stale slot map (upload a dirty map even when the promotion has no
routing).  It is correct but **does not remove the divergence** — the genuine cold reads (one-token
promotion lag, by design) remain, and the resident/cold reads are geometrically and byte-identical, so
the MoE read is not the arithmetic source.

**So the MoE read is provably byte-correct (full slice, all rows) and stride-consistent; the fusion and
guard decisions are identical across budgets; yet the logits differ.**  The divergence is therefore a
**graph/memory-layout effect** that the take-over enables and the split geometry exposes — not a wrong
expert, a stale map, a fusion, or a stride.  The take-over is required (`GGML_MOE_CACHE_MAX_TOK=0` makes
budgets agree) and `DISABLE_FUSION` does not fix it (it only moves the index to 9), so if it is an
address-selected kernel it is not a `try_fuse` fusion.

**Confirmed workarounds (pure):** `GGML_META_SPLIT_COPY=0` (mirror the expert copies) or identity mode
(a cache large enough for `slots >= n_experts`); `-sm layer` and cache-off also avoid it.

**Next instrument:** a per-op address/allocation-plan diff between two cache budgets with the take-over on
(the r25 §41 canonical allocation dump), since the arithmetic levers we could name are all identical.

## 5e. Side finding: qwen35moe (35B-A3B) host expert tables are **mirrored**, not split

The `CACHE_GEOM` log shows `qwen35moe` registers every host expert table as **whole**
(`expert_bytes == host_bytes`, `src_off == 0`) under `-sm tensor -ncmoe`, while qwen4exp's are split
(`expert_bytes < host_bytes`, uneven `src_off`).  Splitting the 35B's tables is a performance win (per the
maintainer); it is out of this investigation's scope but is exactly the path this divergence lives in, so
it should not be enabled until §5d is fixed.

## 5f. ROOT CAUSE LOCATED: the **device-remap path** (`MOE_EXPERT_CACHE_DEVMAP=1`, the default)

`MOE_EXPERT_CACHE_DEVMAP=0` — the **eager host-routing path** — makes every divergent pair pure:

| A/B, cache on, split tables | default (`DEVMAP=1`) | `DEVMAP=0` |
|---|---|---|
| 2 GPU, MIB=8000 vs MIB=14000 (plain) | 59 | **None** |
| 2 GPU, `n_rs_seq` 0 vs 3 (MIB=14000) | 59 | **None** |
| 1 GPU, MIB=8000 vs MIB=14000 | 64 | **None** |

So the bug is **entirely in the device-remap path** (`g_devmap`, default ON since session 19: a
device-built remap + a deferred post-graph promotion).  The eager host path builds the remap on the host
with a **two-pass** algorithm — *pass 1* admits every used expert (each protected from the others), *pass
2* builds the remap **from the final `expert_slot` map** — and its comment already records the exact class
of defect: *"A later admission used to be able to evict an earlier one … which is what left an earlier
position pointing at a slot a later fill had refilled; the two-pass remap below fixes the value."*  The
device path builds the remap from the **pre-promotion** `slot_dev` and promotes after the graph, so the
map the remap is built from and the map the fills mutate are not the same generation.

Sub-path probes (all still divergent at the default, so not the fix): `KSLOT=0` (materialized device remap)
59; `DEVPOLICY=0` (host promotion + device remap) 59; `DEVPOLICY_SPLIT=1` 9.  `DEVMAP=0` is the only lever
that makes them agree.

**The mechanism (validator evidence).**  With the host policy authoritative (`MOE_EXPERT_CACHE_DEVPOLICY=0`)
the level-4 validator shows the device `slot_dev` map holding a **positive slot for an expert the
authoritative host map says is not resident**: `SLOT-DEV-MISMATCH layer=0 expert=2 dev=19 host=-1
slot_dirty=0` (3976 occurrences, `dev >= 0` variants exist).  A device-built remap from that map points at
slot 19, which by then holds a *different* expert, so the routed read returns the wrong expert.  This is
exactly the failure mode the eager host path's two-pass comment describes ("an earlier position pointing
at a slot a later fill had refilled"); the device path's remap is built from a `slot_dev` snapshot that is
not the generation the fills mutate.

**Fix direction.**  Make the device `slot_dev` a consistent, authoritative snapshot: clear/`memset` it on
every host-map mutation (the stand-down path at `free_table_buffers_locked` already does), gate the
device-remap takeover on the map being current, and/or add a residency generation the remap kernel checks.
The immediate, validated stand-down is `MOE_EXPERT_CACHE_DEVMAP=0`; it is pure on every case above at the
cost the device path was built to remove (~+22 % for the device path over the eager path at partial
residency, per the source comment).

### The user's hypotheses, tested

* **Host CPU compute on misses:** not present.  `colds=0` / `cold_reaches=0` even at the default `touch`
  admission, `MOE_EXPERT_CACHE_ADMIT=0` and `TOUCH=1` change nothing (identical hit/fill counts), and
  `GGML_SCHED_DEBUG` shows every decode MoE split on `Meta(ROCm0,ROCm1)` with no non-empty CPU split.
* **Admission policy (admit on first touch):** inert here — the rejection path never fires (`colds=0`),
  so changing it cannot help.
* **Force all compute to the GPU:** there is no blanket flag (the CPU backend is the scheduler's last
  resort).  `GGML_OP_OFFLOAD_MIN_BATCH=0` (force every host-weight op onto a GPU) does not remove the
  divergence (59); `=100000` (more CPU op-offload) moves it to 9.  `-ncmoe 0` (all experts GPU-resident)
  is pure but disables the cache.  None of these is the cause; the cause is `DEVMAP`.

The user's instinct — “host-based” vs device work on misses — pointed at exactly the right axis: the
**host** remap path is the correct one; the *device* remap optimization is what regressed it.

## 6. Repair / stand-down recommendation

* The rewind needs **no** repair (and no stand-down): nothing in the recurrent/GDN/QSA state path is
  wrong.
* **Implemented this session (WIP, `~/llama.cpp` working tree): the directed `#50` per-table fusion guard.**
  `ggml_cuda_cache_blocks_fusion` no longer uses the global `moe_cache_has_arena()` (which required ALL
  288 tables) for the two cache-aware patterns; it asks whether THIS table can serve, via a new
  `moe_cache_table_serves()` that mirrors `moe_cache_take_over`'s exact `identity || devmap` condition, and
  the gate+up+GLU triple requires BOTH lanes.  This makes the fusion guard and the scheduler take-over
  agree per table (the code's own `OPEN 2` invariant), so one evicted table no longer flips the
  cache-band arithmetic of every other table.  It builds warning-free and the cache-off width-probe hash
  is unchanged (`d81701810d2a6c34`); `arm.sh`/`probs.sh` gained the instrument.  **It is a no-op for the
  single-sequence divergence**, because in both arms every table serves during the decode (guard
  decisions identical — `GGML_CUDA_FUSE_TRACE`); the divergence is the `n_rs_seq` × cache effect of §5b.
* **The remaining repair target (§5b):** make the `n_rs_seq > 0` snapshot-armed graph's cache-band MoE
  arithmetic bit-identical to the `n_rs_seq = 0` graph's.  The evidence points at the expert cache's
  per-table residency/remap arithmetic (`identity` vs `devmap` vs cold, or a layout-selected kernel)
  differing with the graph's memory layout, which the snapshot ops perturb.  `test-backend-ops -o
  MUL_MAT_ID`, the prefill-logit gate and the width probe are the natural oracles for that change.
* A scoped stand-down, if the §5b repair is deferred, is: **only** when `-sm tensor` + a qwen4exp MoE +
  a partial cache are all true, force the cache-band MoE on the stood-down (non-cache-aware) path for the
  whole run.  Proposed blast radius to prove: pure at full residency (3 GPU), `-sm layer`, 1 GPU, and
  cache off; qwen4exp prefill-logit KLD unchanged; dense golden unchanged; `MUL_MAT_ID` backend ops green.

## 7. Relation to #50, #51, #52

* **#50** (global fusion guard over per-table eviction): this finding is the single-sequence
  plain-vs-MTP symptom of #50.  The fix above is #50's fix.
* **#51** (3-GPU MTP slowness): same code surface (cache-band routed MMVQ + the multi-device split
  table) but a *performance* symptom; the 3-GPU full-residency run here is byte-identical plain==MTP, so
  the correctness-side of #51 is not observed.  Not closed by this session.
* **#52** (multi-sequence non-determinism): **separate root**.  #52 is the §27 invariant — a rejection
  rollback landing while the last decoded batch was a large prefill batch, so the recurrent snapshot
  plane read was never written (the `seq_rm` "rollback crossed a batch boundary" warning).  The
  single-context rewind probe above only exercises rollbacks *inside* a verify batch, which is correct;
  the #52 fix (never present such a rollback, write the pre-batch plane for the chunked prefill, or pin
  the rollback into the batch's own snapshot) does **not** address this divergence, and the #50 fix does
  **not** address #52.

## 8. Reproducers

```bash
cd wip/moe-verify-fusions/tools
export LD_LIBRARY_PATH=/opt/rocm-7.14.1-gfx120X/lib:${LD_LIBRARY_PATH:-}   # NOT /opt/rocm-7.14-gfx120X
export M=/llm/models/Qwen3.8/Flash-Next/GSQ-IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf
export D=/llm/models/Qwen3.8/Flash-Next/IQ4_NL/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf
export PROMPT=/tmp/srr/mixed30k.txt NP=128

# --- the rewind probe (fast, seconds/config) ---
cd ~/llama.cpp && clang++ -O2 -std=c++17 -I include -I ggml/include -I src \
  /home/stew675/llama-cpp-rdna-boosts/wip/moe-verify-fusions/tools/rrewind.cpp -o /tmp/rrewind \
  -Lbuild-rocm-hybrid/bin -lllama -lggml -lggml-base -Wl,-rpath,$PWD/build-rocm-hybrid/bin
HIP_VISIBLE_DEVICES=0,1 W=4 N=40 RS=3 STATE=1 SPLIT=tensor NCMOE=48 CACHE=1 \
  MOE_EXPERT_CACHE_MIB=14000 CTK=q8_0 CTV=q8_0 CTX=4096 \
  /tmp/rrewind $M /tmp/srr/probe9k.txt 2500 2048      # -> PURE (0/0)

# --- the server arms (2 GPU, ce) ---
export GGML_CUDA_ALLREDUCE=ce MOE_EXPERT_CACHE_MIB=14000 SM=tensor GPUS=0,1
./arm.sh ~/llama.cpp/build-rocm-hybrid n3 --spec-type draft-mtp --spec-draft-model "$D" --spec-draft-n-max 3
./arm.sh ~/llama.cpp/build-rocm-hybrid none --spec-type none
python3 - <<'PY'   # first-diff = 68 (2 GPU tensor partial); = None with SM=layer, or GPUS=0, or 3 GPU auto
import json
a=json.load(open('/tmp/srr/none.tokens')); b=json.load(open('/tmp/srr/n3.tokens'))
print(next((i for i in range(min(len(a),len(b))) if a[i]!=b[i]), None))
PY
```

## 9. Pointers

* `HANDOVER-single-seq-rollback.md` — the handover this refutes (§1-§5 lead); §6 still holds for #52.
* `tools/rrewind.cpp` — the single-context rewind probe (new).
* `tools/rollback-replay.cpp` — the two-context mstep-style replay (new; PURE at full residency, confounded
  at partial residency).
* `tools/arm.sh` — arbitrary server arm runner (new).
* `GREEDY-PURITY.md` §27 (rollback bound), §30/§31 (address-selected fusion), §39 (cache band).
* `TODO.md` #50, #51, #52.
