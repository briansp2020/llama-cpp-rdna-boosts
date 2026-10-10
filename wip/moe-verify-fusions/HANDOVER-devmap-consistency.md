# HANDOVER — finish the `DEVMAP=1` (device-remap) consistency fix

**For a cold session.  Read end to end before touching anything.**  This continues
`wip/moe-verify-fusions/` (campaign README) and the 2026-10-12 root-cause work recorded in
`FINDINGS-single-seq-rollback.md` §5b-§5g.

**Status: the single-sequence plain-vs-MTP divergence is ROOT-CAUSED and made non-divergent by a WIP fix
(default `MOE_EXPERT_CACHE_DEVMAP=0`).  The device-remap path itself is still buggy and is left opt-in.
The task is to repair the device path, re-measure it against the new default, and decide whether to
re-enable it.  A second, independent task (35B-A3B Q8_0 split-path validation) is in §9.**

---

## 0. Session kickoff prompt (paste this into the fresh session)

> You are picking up the `wip/moe-verify-fusions` campaign in `/home/stew675/llama-cpp-rdna-boosts` (the
> `rdna-boosts` delivery repo for the `stew675/llama.cpp` fork).  READ FIRST, in order:
> (1) `wip/moe-verify-fusions/HANDOVER-devmap-consistency.md` end to end — it is the self-contained
> handover for this task; (2) `wip/moe-verify-fusions/FINDINGS-single-seq-rollback.md` §5b-§5g (the
> evidence); (3) the "WIP and promotion rules", "Pushing policy" and "Report findings before shipping"
> sections of `AGENTS.md`.
>
> CONTEXT.  The 3-session single-sequence plain-vs-MTP divergence was root-caused: it is the
> **device-remap path** (`MOE_EXPERT_CACHE_DEVMAP=1`, default until this session) building its routed-expert
> remap from a `slot_dev` snapshot that is not the generation the arena fills mutate, so a slot can be
> refilled under the remap and the read returns the wrong expert.  The working `~/llama.cpp` tree already
> defaults `DEVMAP=0` (the eager host-routing path), which makes every divergent pair byte-identical at
> ~3 % MTP-decode / 0 % plain-decode cost.  The tree is uncommitted; the full diff is
> `wip/moe-verify-fusions/session-devmap-fix.diff`.
>
> YOUR TASK, in order:
> (a) Make the device-remap path generation-consistent per `HANDOVER-devmap-consistency.md` §4 (preferred:
>     a residency generation checked at remap build; alternatives in §4).  Keep it scoped to the device
>     path (D1): do not disable the cache or a whole-run fusion.
> (b) Reproduce the bug with the server A/B in §4 before and after the fix, and require the level-4
>     validator (`MOE_EXPERT_CACHE_VALIDATE=4`) to be clean.
> (c) Validate per §4's checklist (2-GPU + 1-GPU budget sweeps, `n_rs_seq` 0-vs-3, width probe, coherence,
>     MTP + prefill-logit gates).
> (d) Re-measure ≥1000-token decode + 32k prefill, `DEVMAP=0` vs fixed `DEVMAP=1`, interleaved and warm,
>     and **recommend** whether to re-enable it — per §8, if the fixed device path does not clearly beat the
>     eager path on both decode and prefill, leave the eager path as the default and the device path opt-in.
> (e) §9: run the qwen35moe (35B-A3B Q8_0) split-path validation (why it is mirrored, force the split,
>     prove `none == n3`, and show the split is faster with no regression).
>
> CONSTRAINTS (binding): WIP is not the delivery — do not touch `patches/`/`release.json` and do not fold
> anything in without an explicit go-ahead; report findings to the maintainer before any delivery-affecting
> commit; never push to upstream `ggml-org/llama.cpp` (only this repo and the personal fork
> `git@github.com:stew675/llama.cpp.git`, and only when asked); leave `~/llama.cpp`'s uncommitted tree in
> place.  Build env and harnesses are in §5/§4.
>
> DELIVERABLE: a repaired, validated device-remap path (or a documented decision to keep it opt-in with
> the re-measurement), the 35B split validation, the numbers recorded in the findings/handover, and a
> recommendation to the maintainer.

---

## 1. TL;DR

* The 3-session plain-vs-MTP divergence was **not** the recurrent rewind, **not** the fusions, **not** the
  cache admission/cold path, and **not** a host CPU compute.
* It is the **device-remap path** (`MOE_EXPERT_CACHE_DEVMAP=1`, default since r-session-19): it builds the
  routed-expert remap from a `slot_dev` snapshot that is not the generation the arena fills mutate, so a
  slot can be refilled under the remap and the routed read returns the **wrong expert**.
* **Fixed in the working tree by defaulting to the eager host-routing remap path** (`DEVMAP=0`, the
  two-pass path whose own source comment documents exactly this class of defect).  Every divergent pair is
  now byte-identical; the cost measured over 1000 decode tokens is **~3 % MTP decode / 0 % plain decode**,
  and **MTP prefill is actually faster** with the eager path.
* **Remaining work:** make `slot_dev` generation-consistent (then re-measure and decide whether the device
  path is worth re-enabling at all — see the open question in §8).

## 2. What is in the working tree (do not lose it)

`~/llama.cpp` is on branch `rdna-boosts` (block-15 tip), **uncommitted**, and contains, besides the
pre-existing narrow-2 work:

* **the fix:** `ggml/src/ggml-cuda/moe-expert-cache.cu` line ~617,
  `g_devmap = env_int("MOE_EXPERT_CACHE_DEVMAP", 0) != 0;` (was `1`), with an explanatory comment;
* **a real related bug fix:** `moe_cache_promote_host` skipped the `slot_dev` H2D upload when the routing
  list was empty (early `used.empty()` return), leaving the device map stale.  It now uploads a dirty map in
  that branch too;
* **opt-in envs added this session:** `MOE_EXPERT_CACHE_ADMIT` (0=always fill / 1=value / 2=touch, default
  2), `MOE_EXPERT_CACHE_TOUCH` (now allows 1), `MOE_EXPERT_CACHE_COLD_UVA` (default 1),
  `GGML_FORCE_N_RS_SEQ` (`src/llama-context.cpp`);
* **validators/diagnostics:** `MOE_EXPERT_CACHE_VALIDATE` levels 1-4 (level 3 = slot contents vs host
  master; level 4 = device `slot_dev` vs host `expert_slot`), `MOE_EXPERT_CACHE_VALIDATE_LAYER`,
  `GGML_CUDA_CACHE_GEOM_DEBUG` (per-table split geometry), `GGML_CUDA_CACHE_REDIR_DEBUG` (redirected
  strides), `GGML_CUDA_FUSE_TRACE`, `GGML_CUDA_CACHE_STANDDOWN_GATEUP/DOWNFOLD`, plus a per-table fusion
  guard `moe_cache_table_serves()` (#50 OPEN-2 repair; a no-op for this divergence);
* the full diff is saved at **`wip/moe-verify-fusions/session-devmap-fix.diff`** (cumulative; supersedes
  `session-narrow2.diff`).

The delivery repo (`~/llama-cpp-rdna-boosts`) has all of this **documented and pushed** on `main`
(`326e256`, `0af1a78`).  `patches/` and `release.json` are untouched — this is WIP.

## 3. The fix already in place, and its numbers

Default now `DEVMAP=0` (eager host-routing remap).  Validated (all with the fix):

| check | result |
|---|---|
| 2 GPU split-table budget sweep MIB8000 vs MIB14000 | **None** (was 59) |
| `n_rs_seq` 0 vs 3, cache on MIB=14000 | **None** (was 59) |
| 1 GPU budget sweep MIB8000 vs MIB14000 | **None** (was 64) |
| field `none` vs `draft-mtp --spec-draft-n-max 3`, 2 GPU `-sm tensor`, 32k prompt | **None** (was idx 9) |
| cache-off width probe, GSQ `NCMOE=48` tensor | `d81701810d2a6c34`, PASS (unchanged) |
| 4B Q8_0 coherence | coherent |

Perf, 1000-token decode + 32k prefill, 2 GPU `-sm tensor`, `GGML_CUDA_ALLREDUCE=ce`, GSQ + shared Q8_0 MTP
head, two runs each:

| config | decode t/s | acceptance | prefill t/s |
|---|---:|---:|---:|
| MTP n3, `DEVMAP=0` (new default) | 72.45 / 72.57 | 0.77017 | 1580 / 1579 |
| MTP n3, `DEVMAP=1` (old) | 74.71 / 74.88 | 0.78676 | 1263 / 1264 |
| plain, `DEVMAP=0` | 54.78 | — | 1844 |
| plain, `DEVMAP=1` | 54.82 | — | 1845 |

(The prefill delta is real and repeatable: the eager path clears `slot_dirty`, so the cache-aware prefill
staging engages under the MTP config.  The acceptance delta is the buggy trajectory, not a quality signal.)

## 4. THE TASK — make the device-remap path generation-consistent

### The exact defect

The device path builds the remap at the start of the graph from `t.slot_dev` (via
`moe_cache_build_remap_kernel` / `moe_cache_launch_remap`, or in-kernel under KSLOT), then promotes
(admits/evicts/fills) **after** the graph.  The remap therefore names slots by a map that the fill/eviction
generation can invalidate.  Under the host policy the validator catches it directly: the device `slot_dev`
holds a **positive slot for an expert the authoritative host `expert_slot` says is not resident**
(`SLOT-DEV-MISMATCH layer=0 expert=2 dev=19 host=-1 slot_dirty=0`, 3976 occurrences per 60-token run).
The read then uses slot 19, which by then holds a different expert.

The eager path is correct because it is **two-pass**: *pass 1* admits every used expert (each protected
from the others), *pass 2* builds the remap from the **final** map (`moe_cache_update_host`, lines
~4631/4642).  Nothing evicts between the remap build and the read.

### Code map

| file:line (working tree) | what |
|---|---|
| `ggml/src/ggml-cuda/moe-expert-cache.cu:617` | the `g_devmap` default (the fix) |
| `:1317` `access_locked` | host LFRU admit/evict/fill; sets `slot_dirty` |
| `:2706` `moe_cache_build_remap_kernel` | device remap: `remap[i] = slot_dev[e]`, cold = `n_res + e` |
| `:2754` `moe_cache_launch_remap` | its launcher |
| `:2810` `moe_cache_policy_kernel` | device LFRU replay + fill (DEVPOLICY); updates `slot`, `slot_expert` |
| `:3632` `moe_cache_validate` | levels 1-4 validators |
| `:4744` `moe_cache_get_table` | returns the devmap (`t.slot_dev`, `t.remap_dev`) at read time |
| `:4864` `moe_cache_take_over` | the scheduler hook; records the deferred promotion |
| `:4910` `moe_cache_promote_host` | deferred promotion; the slot-map H2D is at ~5045 |
| `:5358` `moe_cache_policy_flush` | launches the device policy kernel |
| `ggml/src/ggml-backend.cpp` ~3000 | the deferred promotion pass (after the graph) |
| `:4631/4642` (`moe_cache_update_host`) | the eager two-pass reference implementation |

### Candidate repairs (pick one, prove it)

1. **Generation guard (preferred).**  Add a residency generation incremented on **every** slot-map
   mutation (host `access_locked`, the seed fill, the stand-down/rearm, the device policy kernel), publish
   it alongside `slot_dev`, and make `moe_cache_get_table` / the take-over **reject** (fall back to the
   copy) or the remap kernel **rebuild** when the generation the remap would use is not the current one.
   The invariant: *the map the remap is built from must be the map the arena was last filled for.*
2. **Eager device promotion.**  Move the device policy kernel to *before* the remap build for the current
   token (it needs the current routing — available from the graph's `ids`, which is why the deferred design
   read `used_dev` instead; consider taking the ids tensor directly).  Then the remap uses the current
   residency, matching the eager path's semantics without the host readback.
3. **Clear `slot_dev` on every host-map mutation.**  Audit every `t.slot_dirty = true` site and ensure
   `slot_dev` is either cleared or immediately re-uploaded; the stand-down
   (`free_table_buffers_locked`) already memsets it.  Necessary but probably not sufficient.
4. **Last-resort scoped stand-down.**  Keep `DEVMAP=0` (what is in the tree now) and document the perf
   delta; acceptable only if §8's re-measurement shows the device path is no longer worth it.

**Do not** "fix" it by disabling the whole cache or a whole-run fusion (D1).  Scope any stand-down to
exactly the device-remap path (`DEVMAP=0`), which is already what the tree does.

### How to reproduce the bug fast

The divergence needs the server's context shape, so use the two server arms (a few minutes each), not the
small probes:

```bash
cd wip/moe-verify-fusions/tools
export LD_LIBRARY_PATH=/opt/rocm-7.14.1-gfx120X/lib:${LD_LIBRARY_PATH:-}   # NOT /opt/rocm-7.14.1-gfx120X-> 7.14-gfx120X is stale
export M=/llm/models/Qwen3.8/Flash-Next/GSQ-IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf
export PROMPT=/tmp/srr/mixed30k.txt NP=80 GPUS=0,1 GGML_CUDA_ALLREDUCE=ce SM=tensor
# the failing pair (forces the device path; with the current default you must set DEVMAP=1):
for MIB in 8000 14000; do MOE_EXPERT_CACHE_DEVMAP=1 MOE_EXPERT_CACHE_MIB=$MIB \
  ./probs.sh ~/llama.cpp/build-rocm-hybrid dd_$MIB /tmp/srr/mixed30k.txt 80 --spec-type none; done
python3 - <<'PY'
import json,re
def T(f):
    d={}
    for l in open(f):
        m=re.match(r"i=(\d+) tok=(\d+)",l)
        if m: d[int(m.group(1))]=int(m.group(2))
    return d
a=T("/tmp/srr/dd_8000.probs"); b=T("/tmp/srr/dd_14000.probs")
print("first-diff (expect 59 while buggy, None when fixed):",
      next((i for i in range(min(len(a),len(b))) if a[i]!=b[i]), None))
PY
# validator (level 4) with the host map authoritative — must be clean when fixed:
MOE_EXPERT_CACHE_DEVMAP=1 MOE_EXPERT_CACHE_DEVPOLICY=0 MOE_EXPERT_CACHE_VALIDATE=4 \
  MOE_EXPERT_CACHE_MIB=8000 ./probs.sh ~/llama.cpp/build-rocm-hybrid dim /tmp/srr/mixed30k.txt 60 --spec-type none >/dev/null 2>&1
grep -ac "SLOT-DEV-MISMATCH\|SLOT-CONTENT-MISMATCH" /tmp/srr/dim.err    # expect 0
```

`tools/` also has: `arm.sh` (arbitrary server arm), `probs.sh` (logprobs+content), `perf.sh`
(long-decode timing), `rrewind.cpp` / `rollback-replay.cpp` (teacher-forced replay, cache-armed),
`widthsweep.sh`.  Build the probes with `clang++ -O2 -std=c++17 -I include -I ggml/include -I src <src>
-o /tmp/<name> -Lbuild-rocm-hybrid/bin -lllama -lggml -lggml-base -Wl,-rpath,$PWD/build-rocm-hybrid/bin`.

### Validation required before re-enabling `DEVMAP=1`

* The 2-GPU budget sweep, the 1-GPU budget sweep and the `n_rs_seq` 0-vs-3 A/B all `None`.
* `MOE_EXPERT_CACHE_VALIDATE=4` clean (slot contents, and `slot_dev` == authoritative map, at every
  `after-promote` / `after-rearm`).
* The cache-off width probe unchanged (`d81701810d2a6c34`) and the 4B/dense coherence unchanged.
* The full promo gates: MTP methodology (`benchmarks/mtp-adaptive-methodology.md`), prefill-logit
  (`scripts/gate-prefill-logits.sh`), `test-backend-ops -o MUL_MAT_ID`, `test-recurrent-state-rollback`.
* **A ≥1000-token perf A/B** (`DEVMAP=0` vs fixed `DEVMAP=1`, interleaved, warm) before any decision to
  re-enable — see §8.

## 5. Build / environment

```bash
cd ~/llama.cpp
EXTRA_CMAKE_FLAGS="-DCMAKE_HIP_FLAGS=" ~/bin/build-llama-rocm-714        # fresh configure only
cmake --build build-rocm-hybrid --target llama-server llama-cli test-logits-width-probe -j 16
export LD_LIBRARY_PATH=/opt/rocm-7.14.1-gfx120X/lib:${LD_LIBRARY_PATH:-}   # NOT /opt/rocm-7.14.1-gfx120X
```

`llama-cli` always needs `--single-turn`; `-fa off` is invalid with `-sm tensor`; never run parallel
benches; warm up before recording.  The field prompt is `/tmp/srr/mixed30k.txt` (sha256
`69624f4d207f40bd…`, ~32k tokens, non-degenerate); the GSQ set has no embedded MTP, so
`--spec-draft-model /llm/models/Qwen3.8/Flash-Next/IQ4_NL/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf` is
required.

## 6. Why this was missed for three sessions (so it is not missed again)

* The forward-pass width probe never armed the expert cache, and the cache-off width probe was byte-stable.
* The 1-GPU and 3-GPU controls that "proved" non-qwen4exp purity (a) did not test GDN (the 35B has SSM
  layers too) and (b) reached **identity mode** (512 slots = `n_experts`) or **mirrored tables**, neither
  of which is the devmap/split path.
* The divergence needs **split expert tables** (`GGML_META_SPLIT_COPY=1`) + **partial residency** +
  `-sm tensor`; qwen4exp is simply the model whose host expert tables are actually split.
* The device-path bug is invisible to the text-level gates: it only flips a near-tie after ~59 tokens.

## 7. Constraints (binding, from AGENTS.md)

* **WIP is not the delivery.**  Do not touch `patches/` or `release.json`, and do not fold the fix in
  without the maintainer's explicit go-ahead for that specific item.
* **D1:** purity and the fused paths come first; repair the fused/device path, do not trade it away.  A
  stand-down must be scoped to exactly the condition with a blast-radius matrix.
* **D2:** A/B on GSQ-IQ3_XXS + the shared Q8_0 MTP head; keep IQ4_NL for cross-checks.
* **Report findings to the maintainer before any delivery-affecting commit or release.**
* **Pushing:** the only permitted targets are this repo and the personal fork
  (`git@github.com:stew675/llama.cpp.git`); never upstream `ggml-org/llama.cpp`.  A release must refresh
  the fork's `rdna-boosts` branch.

## 8. Open question — is the device path still worth re-enabling?

The device path was introduced (~session 10) because the eager path's per-op host routing readback +
full device sync stalled partial-residency decode to ~70 t/s; it was "+22 % at every h" **then**.  Chip in:
the eager path is now only **~3 % slower on MTP decode, identical on plain decode, and faster on MTP
prefill** — i.e. roughly a wash overall.  So the honest framing for the maintainer is:

> *Do not re-enable `DEVMAP=1` until it is fixed AND re-measured; and if the fixed device path does not
> clearly beat the eager path at ≥1000 tokens (decode **and** prefill), recommend leaving the eager path as
> the default and keeping the device path as an opt-in.*

This is a decision for the maintainer after §4 + the re-measurement, not a foregone conclusion.

## 9. Second task — qwen35moe (35B-A3B Q8_0) split-path validation

The 35B-A3B currently registers its host expert tables **whole/mirrored** under `-sm tensor -ncmoe`
(`CACHE_GEOM`: `expert_bytes == host_bytes`, `src_off == 0`), while qwen4exp's are split.  Splitting is a
perf win, but it is exactly the path this bug lives in, so it must be validated after §4.

* Model: `/llm/models/Qwen3.6/35B-A3B/Q8_0/Qwen3.6-35B-A3B-Q8_0.gguf` (37.8 GB, embedded MTP head, so no
  `--spec-draft-model`).
* First determine **why** the meta splitter keeps it whole (`GGML_CUDA_CACHE_GEOM_DEBUG=1`; the axis-1
  copies are governed by `GGML_META_SPLIT_COPY` at `src/llama-model.cpp:451`, default 1).  If the weights
  themselves are not split, find the splitter predicate that declines and whether it can be forced.
* Then, with the split active: `none` vs `draft-mtp n3` must be byte-identical (`pure`), and a ≥1000-token
  interleaved perf A/B (split vs mirrored) must show the split is **faster with no regression**.
* Watch-outs: the earlier campaign found the 35B **Q8_0** with `-ncmoe 48` on **1 GPU** crashed MTP with an
  HSA `MEMORY_APERTURE_VIOLATION` (`k_get_rows_float_vec`) — use 2 GPUs, or a smaller `-ncmoe`, or a smaller
  context.  Use `NCMOE=99` to force every expert host-resident when measuring the split geometry, and
  re-check `--fit`/arena sizing.

## 10. Pointers

* `wip/moe-verify-fusions/FINDINGS-single-seq-rollback.md` — §5b (n_rs_seq × cache), §5c (split geometry),
  §5d (validators), §5f (device-remap root cause), §5g (the fix + perf).
* `wip/moe-verify-fusions/session-devmap-fix.diff` — the working-tree diff.
* `wip/moe-verify-fusions/HANDOVER-single-seq-rollback.md` — the earlier handover (rewind hypothesis;
  keep §6 for the separate multi-sequence bug).
* `wip/moe-verify-fusions/README.md` — campaign framing, D1/D2, #50/#51.
* `TODO.md` #52 (multi-sequence, separate), `GREEDY-PURITY.md` §39 (cache band), `ENVIRONMENT.md` §1.3/§7.
* Delivery repo `main` commits `326e256` (root cause) and `0af1a78` (fix record), pushed.
