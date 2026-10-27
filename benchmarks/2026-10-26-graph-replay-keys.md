# 2026-10-26 — the graph-cache key names the layer: decode graphs replay (r149, TODO #107 closed)

**`WORKLOG.md` 261026.007** + [`archive/work/graph-replay-keys/`](../archive/work/graph-replay-keys/README.md)
— `ggml_cuda_graph_key` was `(first node data, n_tokens)`, and under a split mode **every layer's subgraph
can begin on the same activation buffer**, so one cache entry served all of them and each layer was
compared against the previous layer's properties.  `GGML_CUDA_GRAPH_TRACE` over a decode now reads
`direct≈0, replay≈calls, properties_changed` in the tens instead of ~80 % refused.

## Arms

Binary pair: r148 (stock release) and r149 (r148 + the key fix), built from the same source path with the
same `GGML_CUDA_FA_QUANTS` set, so text comparisons are valid.  `HIP_VISIBLE_DEVICES=0,2` is the x8 pair,
`0,1,2` the three x8/x4/x8 devices.  Greedy text comparisons set `ROCBLAS_USE_HIPBLASLT=0` (issue #67).
Every GPU arm was run alone.

## The flagship arm (the shape `TODO #107` was scoped on)

2x R9700 x8, Qwen3.8 Flash-Next IQ4_NL (100.4 GB) + a frozen heat prior, `-sm tensor -ncmoe 48 -c 4096
-b 512 -ub 512 -fa 1 -ctk q8_0 -ctv q8_0 -t 12 -C 0x00001ffe -n 2000 --ignore-eos --temp 0 --seed 42`,
three interleaved pairs:

| pair | r148 t/s | r149 t/s |
|---|---:|---:|
| 1 | 37.7 | 42.5 |
| 2 | 37.7 | 42.9 |
| 3 | 37.6 | 42.8 |
| **mean** | **37.67** | **42.73** (**+13.4 %**) |

Census, identical across repeats and across the two builds' repeats:

| | direct | replay | capture | properties_changed |
|---|---:|---:|---:|---:|
| r148 | 549499 | 31696 | 149 | 452439 |
| r149 | 3882 | 574826 | 2636 | 28 |

Controls: `GGML_CUDA_DISABLE_GRAPHS=1` measures **37.3 t/s on r149** (i.e. the whole delta is the replay,
which is the r144 trap: an arm that captures without replaying is worth nothing); `GGML_CUDA_GRAPH_KEY_WEIGHT=0
GGML_CUDA_GRAPH_PROPS_STRICT=1` on the assessed PR build restores the stock census **and** stock speed
exactly.  `capture` is the number of **distinct keys** (one graph per layer per width), not recaptures: it
is 2636 after 2000 tokens and 4526 after 6000 (sub-linear, and the context sweeps a graph unused for 10 s),
while `properties_changed` stays at 28/1294.

## Full table

| arm | r148 | r149 | delta |
|---|---:|---:|---:|
| Flash-Next IQ4_NL 2 GPU `-ncmoe 48` plain, n2000 | 37.7 | 42.8 | **+13.5 %** |
| Flash-Next IQ4_NL 3 GPU `-ncmoe 48` plain, n1000 | 37.6 | 42.1 | **+12.0 %** |
| Flash-Next 2 GPU `-ncmoe 48` + `draft-mtp-adaptive n3`, n2000 | 111.5 | 113.1 | +1.4 % |
| Flash-Next 3 GPU `-ncmoe 48` + `draft-mtp-adaptive n3`, n1000 | 107.1 | 109.2 | +2.0 % |
| 27B Q8_0 1 GPU Protocol A plain (`-n 3000`, prose) | 29.6 | 29.7 | flat |
| 27B Q8_0 1 GPU Protocol A `draft-mtp` | 70.2 | 70.0 | flat, acceptance 0.80905 both (1449/1791) |
| 27B Q8_0 2 GPU tensor `tg128` | 32.80 | 32.59 | flat |
| 27B Q8_0 3 GPU tensor `tg128` | 42.0 | 38.4 | **-8.6 %** |
| 27B Q8_0 3 GPU tensor `tg128` at depth 16384 | 42.06 | 38.30 | **-8.9 %** |
| 27B UD-Q4_K_XL 3 GPU tensor `tg128` | 52.36 | 49.05 | **-6.3 %** |
| 27B Q8_0 3 GPU tensor + `draft-mtp` (`-n 3000`) | 42.8 | 42.9 | flat, acceptance 0.82386 both (1478/1794) |
| 27B Q8_0 3 GPU `-sm layer` `tg128` | 17.43 | 17.44 | flat |
| 35B-A3B UD-Q4_K_M 1 GPU `tg128` | 101.75 | 101.67 | flat |
| 35B-A3B 1 GPU Protocol A plain / `draft-mtp` | 82.0 / 174.5 | 81.2 / 174.1 | -1.0 % / -0.2 %, acceptance 0.75118 both (1591/2118) |

The one regression is `TODO #109`: the captured graph loses to the direct node loop on a **3-device
tensor-split dense plain decode**.  It is not capture churn (see the capture counts above; the deficit also
survives `-n 512`, where the captures are amortised away), not `GGML_CUDA_GRAPH_CHUNK_SIZE` (512 and 1e6
identical), not `GGML_CUDA_ASYNC_INPUT` and not `GGML_SCHED_EVENTS`.  `GGML_CUDA_ALLREDUCE=internal`
removes it:

| 27B Q8_0 3 GPU tensor `tg128` | r148 | r149 |
|---|---:|---:|
| default (`hybrid`) | 42.07 | 38.3 |
| `GGML_CUDA_ALLREDUCE=internal` | 42.07 | **42.20** |
| `GGML_CUDA_ALLREDUCE=nccl` | 38.49 | 35.60 |
| `GGML_CUDA_AR_SMALL_BYTES=999999999` (all internal) | 42.15 | 38.67 |
| `GGML_CUDA_AR_SMALL_BYTES=0` (all NCCL) | 41.46 | 38.18 |

So the loss is not *which* all-reduce arm runs per call (routing everything to internal does not help); it
is the presence of the hybrid/NCCL comm layer, or the node-level stream overlap that a captured graph cannot
express.  The MoE shapes (the delivery's tuned out-of-VRAM path) are all wins, and a draft hides it.

## Dense, and the comm layer: it is RCCL's *presence*, not the model

The 3-device dense loss above looks like a dense-model cost.  It is not.  Two more dense models and the four
comm-layer arms:

| 27B Q8_0, 3 GPU `-sm tensor`, `tg128` | r148 | r149 |
|---|---:|---:|
| `GGML_CUDA_ALLREDUCE=hybrid` (default) | 41.71 | **38.58** (-7.5 %) |
| `GGML_CUDA_ALLREDUCE=ce` (runs `comm_init_hybrid` first, so RCCL is up) | 40.70 | **38.59** (-5.2 %) |
| `GGML_CUDA_ALLREDUCE=nccl` | 38.49 | **35.60** (-7.5 %) |
| `GGML_CUDA_ALLREDUCE=internal` (RCCL never initialised) | 41.81 | **42.17 (+0.9 %)** |

| dense model | 2 GPU tensor | 3 GPU tensor |
|---|---|---|
| 27B Q8_0 | 32.36 -> 32.52 (+0.5 %, `properties_changed` 71630 -> 1290) | 41.7 -> 38.6 (**-7.5 %**) |
| Gemma4 31B Q8_0 (32.6 GB) | 26.64 -> 26.86 (+0.8 %, `properties_changed` 74690 -> 0) | 34.68 -> 34.58 (flat) |
| Gemma4 E4B Q8_0 (1 GPU) | 71.2 -> 74.8 (flat on a re-run; the census is `capture=1` on **both** builds, so there is no collision to fix on one device) | n/a |

And the same `ce` arm on the flagship shape is unaffected (`-ncmoe 48`, 2 GPU x8, n1000: plain 37.4 -> 41.8,
`draft-mtp n3` 106.9 -> 108.7, text identical in both), as is 27B dense at 2 GPUs with `ce` (31.67 -> 32.53,
+2.7 %, text identical).

So: the per-call all-reduce *choice* is irrelevant (`GGML_CUDA_AR_SMALL_BYTES` 0 and 1e9 both leave the
deficit in place), `ce` regresses exactly like `hybrid` because its init path brings RCCL up too, and only
`internal` (which never touches RCCL) is clean.  That matches the comment already in
`ggml_backend_cuda_comm_init_hybrid`: "the cost is RCCL's presence, which r34 measured as a decode-only
slowdown of the internal pipeline".  r149 does not create the effect; it makes it visible on this arm,
because before r149 this arm never replayed and the direct node loop is immune.  Carried as `TODO #109`.

Note for anyone re-running the 3-device arms: the Gemma4 31B MTP head is a `gemma4-assistant` model and
`LLAMA_SPLIT_MODE_TENSOR not implemented for architecture 'gemma4-assistant'` refuses the load, so that
model's draft arm cannot be measured with `-sm tensor` at all (pre-existing, upstream).

## Purity

`ROCBLAS_USE_HIPBLASLT=0`, greedy, `--seed 42 --temp 0`, `-n 48`: **text byte-identical in 10/10 pairs** —
27B Q8_0 1 GPU, 2 GPU `-sm layer -ts 59,41`, 3 GPU `-sm tensor`; 35B-A3B UD-Q4_K_M 2 GPU `-ncmoe 40`;
Flash-Next IQ4_NL 2 GPU `-ncmoe 48` + a `draft-mtp-adaptive n3` draft; Gemma4 E4B Q8_0 with and without
its MTP draft.  The draft acceptance trajectory is bit-identical (27B 3 GPU 0.82386 with the same
1478/1794, 35B-A3B 0.75118 with the same 1591/2118), which is the stronger statement: a replay runs the
same kernels with the same arguments, so the whole change is a host-side scheduling decision.

Prefill: `scripts/gate-prefill-logits.sh` **mean KLD 0.000000, same-top-p 100.000 %** — as expected, since a
prefill graph is multi-token and skips the graph path, but it is the gate that proves it.

## Reproduce

```bash
# the flagship arm (frozen heat prior so the arena is warm; 2000 tokens, not 200)
HIP_VISIBLE_DEVICES=0,2 MOE_EXPERT_HEATMAP=load MOE_EXPERT_HEATMAP_DIR=<frozen copy> \
  GGML_CUDA_GRAPH_TRACE=1 llama-cli -m ...Flash-Next-IQ4_NL... -ngl 99 -sm tensor -ncmoe 48 \
    -c 4096 -b 512 -ub 512 -fa 1 -ctk q8_0 -ctv q8_0 -t 12 -C 0x00001ffe \
    -p "The capital of France is" -n 2000 --ignore-eos --temp 0 --seed 42 \
    --no-display-prompt --single-turn

# the dense 3-device regression and its all-reduce dependence
HIP_VISIBLE_DEVICES=0,1,2 llama-bench -m ...27B-Q8_0... -p 0 -n 128 -r 3 -t 12 -C 0x1ffe -sm tensor
HIP_VISIBLE_DEVICES=0,1,2 GGML_CUDA_ALLREDUCE=internal llama-bench ... (same)
```
