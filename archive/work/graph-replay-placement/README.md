# Graph replay under the Meta backend: make the decode graph replay (placement identity)

**Status: CLOSED (2026-10-26, r149).**  `TODO #107` is closed and the fix shipped in **block 09**: the graph
cache key now carries the subgraph's weight identity, so a split graph's per-layer subgraphs no longer
share one cache entry and decode replays (2x R9700 x8, Flash-Next IQ4_NL `-sm tensor -ncmoe 48`, 2000-token
decode: 37.7 -> 42.8 t/s, **+13.5 %**; `direct=549499 replay=31696 properties_changed=452439` ->
`direct=3882 replay=574826 properties_changed=28`).  Rule: `patches/README.md` (block 09),
`ENVIRONMENT.md` (`GGML_CUDA_GRAPH_KEY_WEIGHT`), `TODO #109` for the 3-device dense residual.  The campaign
record, including the PR assessment and the measurements that corrected this note's diagnosis, is
[`../graph-replay-keys/README.md`](../graph-replay-keys/README.md).

The rest of this file is the **original scoping note** (2026-10-26, r144), preserved because its diagnosis
section is wrong in an instructive way: read §2 together with the correction in
[`../graph-replay-keys/README.md`](../graph-replay-keys/README.md) §2.  Origin: the O0 measurement of the
retired MoE-miss campaign (`archive/work/moe-cpu-overlap/FINDINGS-2026-10-26.md` §4-§5), which found the
token dispatch-bound with the captured graph never replayed.

Nothing here is part of the delivery.  This note is the handover: it carries the measurement, the exact
diagnosis, the candidate fixes with their costs, the gates, and the traps already paid for.

## 1. The finding in one paragraph

On 2× R9700 (gfx1201, PCIe 5.0 **x8** root ports), Qwen3.8 Flash-Next IQ4_NL (`-sm tensor -ncmoe 48`),
a decode token is **28.4 ms**, of which the GPU is busy **~19 ms** (67 %) and the rest is idle.  The host
makes **297 `ggml_backend_cuda_graph_compute` calls per token** over **12 258 nodes** at **1.1 µs/node =
13.5 ms**, and a HIP API trace shows **~124 000 `hipLaunchKernel`/s (~4 100 per token)** against only
~14 `hipGraphLaunch` per token.  Instrumented counters over 200 tokens:

```
graph_compute calls: direct=56171 replay=3152 capture=21 properties_changed=55321
```

**93 % of calls find `properties_changed`** in `ggml_cuda_graph_update_required`, which resets
`warmup_complete` and runs the **node loop directly** — so the capture is thrown away and the delivery
gets no graph benefit at all here.  Corroboration: `GGML_CUDA_DISABLE_GRAPHS=1` measures the *same*
speed (35.14 vs 35.53 t/s; host 2 863 vs 2 788 ms), and "capture happens" is not the same as "replay
happens" (19 captures were logged over 8 tokens while 93 % of calls still ran directly).

Headroom if the replay worked: ~11-13 ms of host dispatch per 28.4 ms token, i.e. **up to +40 %** before
the GPU itself becomes the limit (~19 ms ⇒ ~50 t/s).

## 2. Why the property check fails (measured, in this order)

`ggml_cuda_graph_update_required` memcpy's each node's whole `ggml_tensor` plus each source's
`data`/`ne`/`nb` and compares.  Printing the first differing member gave three distinct causes:

1. **`src[]` object addresses** (`field=src`, struct offset 152).  Under the Meta (tensor-split) backend
   every per-device subgraph is rebuilt from **fresh simple-tensor objects** on each allocation
   (`ggml_backend_meta_graph_compute` re-maps every node), so a node's source *pointers* move while the
   source *data* pointers, shapes and types are identical.  This is the **same trap the graph *key*** was
   fixed for in r115 (`GGML_CUDA_GRAPH_KEY_DATA` — key on the first node's `data`, not its object
   address); the property comparison was not fixed with it.
2. **`name`** — the rebuild also renames nodes (`hc_combine-0` → `hc_combine-4` → `-7` …), and `name` is
   an inline `char[GGML_MAX_NAME]` *inside* the copied struct, so a purely cosmetic label invalidates
   the replay identity.
3. **A genuine source `data`-pointer move** for a minority of nodes, once (1) and (2) are canonicalised.

Canonicalising (1) and (2) — keep `op`/`type`/`ne`/`nb`/`data`/`view_offs`/`op_params` and every source's
`data`/`ne`/`nb`, blank `src[]`/`view_src`/`buffer`/`extra`/`name` — was implemented and measured:
`replay` rose but the run got **slower** (28.29 vs 35.23 t/s), because the remaining cause (3) then
triggers *recaptures* at ~20/token and a recapture on HIP is `hipGraphExecDestroy` +
`hipGraphInstantiate` (the `GGML_HIP_GRAPH_FORCE_UPDATE` workaround for the ROCm
`hipGraphExecUpdate` executable leak, ROCm/rocm-systems#10713).  So **"relax the comparison" is not the
fix** — the executable has to be *updated* rather than rebuilt, or the placement has to stop moving.

`cgraph->uid` does not help: `ggml_backend_sched_split_graph` assigns a fresh `ggml_graph_next_uid()` to
every split on every rebuild (`ggml/src/ggml-backend.cpp`), so the fast path in
`ggml_cuda_graph_update_required` can never hit.

## 3. What is different about this shape (do not generalise the finding)

r115's own measurement (dense 27B Q8_0, 3 devices, **no `-ncmoe`**) saw only **45 captures for a
300-token decode**, i.e. the property check passed there.  The churn appears with **`-ncmoe` + the
expert-cache arena + the movable-boundary slab** on this box.  Candidates for the moving `data` pointer:

* the slab's **movable work region** — the boundary shrinks at the prefill→decode drop and on arena
  re-reserve, so allocations inside it can be placed differently between otherwise identical graphs
  (r94's `ggml_cuda_graph_mem_freed` handles the *invalidation* half; nothing guarantees *stability*);
* the cache's **`input_cpy` aliasing** and the per-graph activation caches (MMB/BF16 pools are yielded at
  every graph boundary — a cache that changes size changes later placements);
* the meta **split-state / per-device simple-tensor** rebuild itself;
* `GGML_SCHED_STAGE` / the async input ring for the graph inputs (decode never stages expert weights —
  the width gate is 64 tokens — but the ring's slots rotate).

The first task is to *attribute* which one, per changed node, with the "which field differs" probe
already written for this session (see §6).

## 4. Candidate directions, cheapest first

**A. Stabilise the placement (the real fix if the slab is the cause).**  Make the graph's allocations
land at the same addresses for an unchanged graph shape: e.g. reserve the compute buffer / slab work
region for the decode band at a fixed boundary and never shrink it while a captured graph exists; or
key the slab's placement on the graph shape.  Cost: a slab/compute-buffer change (blocks 09/10), no
numerics.  Gate: `properties_changed` ≈ 0 on the census, replay ≈ slots per token, and the existing slab
invariants (r94 graph invalidation, `GGML_CUDA_GRAPH_MEM_GEN`, the issue #187 fault) still hold.

**B. Make a recapture cheap (update instead of instantiate).**  `ggml_cuda_graph_update_executable` uses
`hipGraphExecDestroy`+`Instantiate` unconditionally under `!GGML_HIP_GRAPH_FORCE_UPDATE` because ROCm
≤ 10.0 leaks in `hipGraphExecUpdate` (ROCm/rocm-systems#10713, driver fix PR #11434).  Check whether
ROCm 10.1.0 has the fix; if so, gate the workaround on the ROCm version and measure a
capture+update path per token.  Cost: small, but it needs a *soak* (the leak grew a few KB/token) and a
version gate; it is also the wrong end of the problem if the placement can be stabilised.

**C. Key the executable on the semantic fingerprint, update on any change.**  A replay identity that
ignores object identity (as in §2) plus (B) to apply the pointer delta.  Only worth it if A is
impossible.

**D. Reduce the dispatch count instead.**  Independent of the graph question: 297 `graph_compute` calls
and 12 258 nodes/token is a lot even for a direct run.  The Meta backend hands each device ~41-node
subgraphs; fewer, larger subgraphs (or a per-device graph that the device captures as one unit) would cut
the per-call overhead (46.9 µs/call measured) even without replay.  This is the fallback if A and B both
lose.

**Do not** simply delete the property comparison: an executable whose baked kernel arguments point at
re-placed buffers is a silent wrong-answer bug, not a slow one.

## 5. Gates (all of them, before anything ships)

* `GGML_CUDA_GRAPH_TRACE=1` census: `direct` → ~0 and `properties_changed` → ~0 on a steady decode; the
  replay count should be ≈ the number of captured graphs per token.
* **Coherence**: same-seed greedy text byte-identical pre/post on a *deterministic* arm.  For a fresh
  process on this box that means `ROCBLAS_USE_HIPBLASLT=0` (issue #67 lottery) and comparing the generated
  text only (strip the spinner banner and the timing line — the spinner frame count varies).
* **Prefill-logit KLD gate** (`scripts/gate-prefill-logits.sh`): unchanged (this work must not move
  prefill numerics).
* **`test-backend-ops -o MUL_MAT_ID`** plus one arch sweep, because the MoE path is in the blast radius.
* The slab invariants: issue #187 (a shrink must bump `ggml_cuda_graph_mem_freed`), `GGML_CUDA_GRAPH_MEM_GEN`,
  and the `GGML_CUDA_GRAPH_PROP_ID`-style kill switch for every behaviour change (A/B must be one env var).
* Throughput: 2 000-token decode arms (the arena/heat prior needs the warmup), 2× R9700 **x8** devices
  (`HIP_VISIBLE_DEVICES=0,2`), plus a check that a *dense* and a `-sm layer` config are unchanged.

## 6. Instrumentation this session used (rebuild it, it is not in the tree)

The counters that made this visible **did ship** in r144 (`GGML_CUDA_GRAPH_TRACE=1` now prints
`graph_compute calls: direct=… replay=… capture=… properties_changed=…` at exit).  The transient probes,
not shipped:

* a "which field differs" print inside the `node_props` comparison — it maps the first differing byte
  offset to a `ggml_tensor` member (`op`, `src`, `view_src`, `nb`, `ne`, `type`, `data`, `buffer`, `name`,
  `extra`, `op_params`, `view_offs`, `padding`, `flags`) and prints the per-source `data`/`ne`/`nb` verdict;
* the delivery's own host-side breakdown: `GGML_CUDA_GCDBG=1` (`CUDAGC calls/nodes/host_total/per_node`,
  `CUDAGC_BREAK`, `CUDAGC_OPS`, `CUDAGC_MATMUL`);
* `rocprofv3 --kernel-trace -f csv` (kernel timeline: busy vs idle, per-kernel shares) and
  `rocprofv3 --hip-trace` (`hipLaunchKernel` vs `hipGraphLaunch` per second);
* `GGML_SCHED_DEBUG=1` for the split/node counts (`sched_reserve: graph (pp …, tg bs=1): nodes = … splits = …`).

## 7. Reference numbers to beat (r144, this box)

| arm (2× R9700 x8, Flash-Next IQ4_NL, `-ncmoe 48`, `-n 2000 --ignore-eos`) | t/s | ms/token |
|---|---:|---:|
| `-sm tensor` (auto arena, 85 % residency) | 35.53 | 28.15 |
| `-sm layer` (same) | 25.73 | 38.86 |
| `-sm tensor` + `GGML_CUDA_DISABLE_GRAPHS=1` | 35.14 | 28.46 |
| `-sm tensor`, arena 2048 MiB/device | 34.44 | 29.04 |

Note the third row: **today the graphs are worth nothing on this shape**, so any change that makes
replay real must also make row 3 fall well behind row 1.  Watch for the failure mode of this session's
first attempt: a change that *enables* captures without eliminating the churn is **slower** than doing
nothing.
