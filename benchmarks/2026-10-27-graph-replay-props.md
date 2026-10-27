# 2026-10-27 — the replay guard's per-node cost: in-place compare + a source-packed store (r158)

**`WORKLOG.md` 261027.010** + [`patches/README.md`](../patches/README.md) (block 10, r158 item) —
`ggml_cuda_graph_update_required` is the only guard that a replayed HIP graph still points at the tensors
it was captured from, and it runs on **every** `graph_compute` (the scheduler mints a fresh `cgraph->uid`
per split per compute, so the `uid` early-out cannot fire).  r158 makes it compare the live tensors in
place, over a store packed to the sources a node actually has, and removes two duplications in the same
path (the graph key built 2-3× per call, and a `cudaGetDevice()` per `MUL_MAT_ID` node).

Prototype + all numbers: the long-lived scratch tree (`scripts/scratch-tree.sh`, path `~/llama-scratch`,
`bin-r157` = `release.json` tree of r157, `bin-patched` = r157 + the fold), same build dir, so the FA
instantiation set is identical and text comparisons are valid.  Every GPU arm ran alone;
`ROCBLAS_USE_HIPBLASLT=0` for every timing and text arm (issue #67).

## 1. Why the guard costs anything: the `uid` early-out is dead

`ggml_backend_sched_split_graph` assigns `splits[i].graph.uid = ggml_graph_next_uid()` on every compute
(upstream #21764) and the Meta rebuild assigns fresh child uids (`ggml-backend-meta.cpp`), so
`cgraph->uid == graph->uid` is never true for a live graph.  Verified: `GGML_CUDA_GRAPH_DEBUG=1` on a
decode prints **zero** "CUDA Graph id … reused" lines (57 replays / 2 captures / 0 `properties_changed`).

So the old loop paid, **per node, per call**: `node_properties prop = {}` (1048 B zero-fill) +
`memcpy` of `sizeof(ggml_tensor)` (328 B) + `memcmp` of all 1048 B — even when nothing differed (which is
the normal case: `properties_changed` is in the tens).

## 2. Profile (decode-only window)

`perf record -F 3000 -e cycles:u -D 12000` on 35B-A3B UD-Q4_K_M, 2x R9700 (x8 pair), `-sm tensor
-ncmoe 40 -c 8192 -b 512 -ub 512 -t 12 -C 0x00001ffe -n 1200`; sampling starts after load+prefill, so it
covers decode only.  Census for the window: `direct=2134 replay=287738 capture=1252
properties_changed=866`, 97.9 t/s (10.2 ms/token).  Shares are of all CPU samples in the window and are
diluted by ROCr's spin thread (`AsyncEventsLoop` = 81 %); the comparison is what matters.

| symbol | r157 | r158 prototype |
|---|---:|---:|
| `__memcmp_evex_movbe` (the guard) | **1.41 %** | **0.58 %** |
| `ggml_cuda_graph_get_key` | 0.26 % | 0.15 % |
| `ggml_backend_cuda_graph_compute` (self) | 0.22 % | 0.37 % (compare now inline) |

19.5 ns/node over ~9.8 k nodes/token at r157 matches a standalone microbench of the old loop shape
(18.2 ns/node) and of the in-place form (5.4 ns/node).

## 3. Store size

Largest graph in that run: **67 nodes, 98 non-null sources** (1.4 per node).  Old `node_properties`:
`67 × 1048 = 70,752 B`.  New (`prop_nodes` 328 B/node + `prop_srcs` 72 B/source + 4 B offsets):
**29,840 B (−58 %)**.  The capture count grows sub-linearly with tokens (`2636` at 2000 tokens, `4526` at
6000 on the Flash-Next arm), so at 4526 entries the saving is ~100 MB of host state per context.

## 4. Throughput — no measurable win on this box

| arm | r157 | r158 | Δ |
|---|---:|---:|---:|
| 35B-A3B UD-Q4_K_M, 2 GPU tensor `-ncmoe 40`, n600, 4 interleaved pairs | 97.95 | 98.10 | **+0.15 %** (paired +0.9/−0.6/−0.1/+0.4) |
| Flash-Next IQ4_NL, 2 GPU tensor `-ncmoe 48 -c 4096 -fa 1 -ctk q8_0 -ctv q8_0`, n600, 6 pairs | 40.23 | 40.53 | **+0.75 %** (paired mean +0.30, sd 0.63 — not significant) |
| 35B-A3B + DFlash2 draft, 2 GPU tensor `-ncmoe 40`, n400 | 196.9 | 195.8 | −0.6 % (noise) |
| Gemma4 E4B Q8_0 + its MTP head (`draft-mtp`), 1 GPU, n500, 3 pairs | 207.1 | 207.3 | flat (207.1/207.4/201.8 vs 207.8/207.1/207.3) |
| 27B Q8_0, **3 GPU** tensor, `llama-bench tg128 -r 3` (the TODO #109 shape) | 37.50 ± 1.69 | 37.35 ± 1.80 | flat |
| Qwen3.5-4B Q8_0, 1 GPU, `tg256 -r 3` | 103.57 ± 0.22 | 103.55 ± 0.22 | flat |
| 35B-A3B, 2 GPU, **pinned to 2 CPU cores** (`taskset -c 0,1`, host starved) | 96.2 | 96.9 | +0.7 % (noise) |

**Read it as: the CPU work is real and it is gone (perf), but it is hidden behind the GPU.**  The host
enqueues ~240 `graph_compute` calls per token in ~1-2 ms against 10-28 ms of GPU work, so it runs ahead
and a sub-millisecond host saving does not move the wall clock — even with the host starved to 2 cores.
The first Flash-Next pair read +3.5 % (39.4 -> 40.8) and that was a cold-first-run artifact of the
baseline (39.4 is the lowest of all 12 runs); over 6 pairs it is +0.75 %.  The win is a *reduction in what
a token costs a weaker CPU* (fewer instructions, less bus traffic, ~100 MB less RAM), not a number on
this box.  Do not quote a t/s figure for r158.

## 5. Purity

* Greedy text (same seed, `--temp 0`): **byte-identical** in every pair — 8/8 runs 35B-A3B 2 GPU, 6/6
  Flash-Next 2 GPU, 2/2 35B-A3B + DFlash2, 6/6 Gemma4 E4B + MTP.
* `GGML_CUDA_GRAPH_TRACE` census: **`replay` identical in every pair** (`143506` / `171970` / `31314` /
  `596`); `capture`/`properties_changed` vary run-to-run on **both** builds (the 10 s unused-graph sweep
  depends on wall-clock load time — 5/6 Flash-Next pairs bit-equal, and the same variation appears between
  two runs of one build).
* Draft acceptance is unchanged by construction (single-token text identical) — the E4B MTP arm and the
  DFlash2 arm above.
* `test-backend-ops -o MUL_MAT_ID`: OK.
* `scripts/gate-prefill-logits.sh` on the canonical build: **mean KLD 0.000000, same-top-p 100.000 %**
  (expected — a prefill graph is multi-token and never enters the graph path).
* `scripts/validate-set.sh` green; `scripts/fork-refresh.sh --build --gates` green.

## 6. The guard's equivalence (why this is not a "relaxed compare")

The identity-free compare offered in PR #230's follow-up (`GGML_CUDA_GRAPH_PROPS_RELAXED=1`, blank
`src[]`/`view_src`/`buffer` before the memcmp) is **still not shipped**, and its batch was rejected for a
different reason than before: it treats a *symptom* of the r149 key collision, and it removes the only
check that a replay's baked kernel arguments still point where they were captured.  r158 compares exactly
the same information as before —

* the node header, verbatim, `memcmp`'d against the live `cgraph->nodes[i]` (so `src[]`, `view_src` and
  `buffer` object identity are still compared);
* each **non-null** source's `data`, `ne`, `nb`.

The only thing that changed is *when* the values are materialised (never, on the hit path) and *how much*
is stored (1.4 source entries per node instead of 10 slots).  The packed source run is equivalent because
the node-header compare has already established that the null/non-null layout of `src[]` is unchanged.
`properties_changed`'s counting rule is preserved (a first sight of a shape is not a refused replay).

Record of Brian's follow-up data and the decline: `archive/work/graph-replay-keys/README.md` §4.

## 7. Reproduce

```bash
# the profile (decode-only window; the census prints at exit)
HIP_VISIBLE_DEVICES=0,2 GGML_CUDA_GRAPH_TRACE=1 \
  perf record -F 3000 -e cycles:u -D 12000 -o /tmp/p.data -- <bin>/llama-cli \
    -m ...35B-A3B-UD-Q4_K_M.gguf -ngl 99 -sm tensor -ncmoe 40 -c 8192 -b 512 -ub 512 \
    -t 12 -C 0x00001ffe -p "The capital of France is" -n 1200 --ignore-eos --temp 0 --seed 42 \
    --no-display-prompt --single-turn
perf report -i /tmp/p.data --stdio -g none --sort symbol | grep -Ei "memcmp|get_key|graph_compute"

# the A/B (interleaved pairs, same build dir)
HIP_VISIBLE_DEVICES=0,2 ROCBLAS_USE_HIPBLASLT=0 GGML_CUDA_GRAPH_TRACE=1 \
  <bin-r157|bin-r158>/llama-cli ...same arm...
```
