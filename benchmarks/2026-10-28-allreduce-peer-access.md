# 2026-10-28 — peer access is what costs a replayed decode: the all-reduce arms were re-cut (r165, TODO #109 closed)

**`WORKLOG.md` 261028.007** + [`patches/README.md`](../patches/README.md) (block 01) +
[`GREEDY-PURITY.md`](../GREEDY-PURITY.md) §67.  Root cause, the isolation that proves it, the arm matrix
behind the new default, and the gates.

## The finding

`TODO #109` said a replayed graph loses to the direct node loop "where the RCCL comm layer is
initialised", and asked what RCCL's init leaves behind.  The answer is that **RCCL is not the cause at
all** — `hipDeviceEnablePeerAccess` is:

| 27B Q8_0, 3x R9700, `-sm tensor tg128` (`llama-bench -r 5`) | t/s |
|---|---:|
| `GGML_CUDA_ALLREDUCE=internal` (no peer access anywhere) | **42.25** |
| `GGML_CUDA_ALLREDUCE=internal` + peer access forced on at init, **AR path untouched** | **38.77** |
| the same, peer access enabled and then disabled again at init | 42.24 |
| `hybrid` (RCCL, peer access on) | 38.6–38.9 |
| `hybrid` + `NCCL_P2P_DISABLE=1` | 42.2–42.4 |
| `ce` chain (copy engine, no RCCL, no peer access) | 42.0–42.3 |

One HIP call, with the all-reduce implementation held constant, reproduces the whole of `#109`.

## Why it is the host, not the GPU

* A per-token synced loop removes it entirely: `llama-bench -p 0 -n 1 -r 100` gives **41.39** (hybrid) vs
  **41.09** (internal) — the GPU critical path is identical.
* Under `rocprofv3 --kernel-trace` the decode window has the *same* kernel count per device (24 665), the
  same kernels, and the AR kernel is not slower.
* A 3-device probe (4 GiB streaming read + an FMA loop, then `hipDeviceEnablePeerAccess` on every pair —
  before *and* after the allocation) measures **0.0 %** change in bandwidth, ALU time, and GPU
  per-dispatch cost (3.46 vs 3.51 µs/kernel).  Peer access is not a GPU-side tax.
* The AR profile (`GGML_CUDA_AR_PROFILE`) accounts for only +0.05 ms/token of the ~2.3 ms/token.
* Cost scales with the *number of enabled pairs* (one device's two pairs: 39.90; four pairs: 38.80;
  none: 42.24) and **enable/disable cost microseconds** (0.007 / 0.002 ms for 6 pairs).

So: a device with peer access enabled charges the **host submission path** ~9 % on this box, and it only
shows where the host is the limiter and graphs replay.  The direct node loop is immune because the host is
already saturated there.

## It is not reversible, so it has to be prevented

The obvious fix — let RCCL bring P2P up, then revoke it before decode — does not work: after
`ncclCommInitAll`, every `cudaDeviceDisablePeerAccess()` returns `no error` and the dense decode **stays
at 37.9 t/s**.  Whatever the cost is, it is in the peer mappings RCCL created, not in the enable flag.
`NCCL_P2P_DISABLE` before init is the only reliable lever.

## What was ruled out along the way (all measured, same arm)

Not the per-call arm choice or the hybrid routing thresholds (`GGML_CUDA_AR_SMALL_BYTES` 0 and 1e9; forcing
*every* all-reduce to the internal pipeline with RCCL up still loses: 38.85), not RCCL's presence without
peer access (`hybrid` + `NCCL_P2P_DISABLE` 42.2), not proxy-thread CPU (`NCCL_PROXY_CPUSET=16-23`),
channel count (`NCCL_MAX/MIN_NCHANNELS=1`, `NCCL_MAX_P2P_NCHANNELS=1`), cuMem (`NCCL_CUMEM_ENABLE=0`),
dma-buf (`NCCL_DMABUF_ENABLE=0`), RCCL's graph helper (`NCCL_GRAPH_HELPER_DISABLE`,
`NCCL_GRAPH_MIXING_SUPPORT=0`, `NCCL_GRAPH_REGISTER=0`), the progress append-op
(`NCCL_PROGRESS_APPENDOP_FREQ=0`), lazy connect (`NCCL_RUNTIME_CONNECT=0`), P2P direct
(`NCCL_P2P_DIRECT_DISABLE=1`), P2P memcpy (`NCCL_P2P_USE_CUDA_MEMCPY=0`), capture churn (the
`GGML_CUDA_GRAPH_TRACE` census is **identical** in both arms: `direct=2316 replay=152865 capture=1161
properties_changed=384`), `GGML_CUDA_GRAPH_CHUNK_SIZE`, `GGML_CUDA_ASYNC_INPUT` or `GGML_SCHED_EVENTS`.

## The two large-tensor arms are good at different shapes

With peer access off, the copy-engine (CE) arm and RCCL's ring were compared on the shapes the delivery
targets (`pp2048`/`tg128` unless stated):

| shape | RCCL (peer on, the old default) | RCCL (peer off) | CE (peer off) |
|---|---|---|---|
| 27B Q8_0 dense 3 GPU | 2177 / 38.62 | 2018 / 42.37 | **2178 / 42.2** |
| 27B Q8_0 dense 2 GPU | 2210 / 32.68 | 2091 / 32.99 | **2245 / 32.9** |
| Gemma4 31B Q8_0 dense 3 GPU | 1944 / 34.40 | – | **1998 / 34.91** |
| Gemma4 31B Q8_0 dense 2 GPU | 1831 / 26.76 | – | **1859 / 26.77** |
| Flash-Next GSQ-IQ3_XXS **all-VRAM** 3 GPU | 1491 / 59.92 | – | **1494 / 67.15** |
| Flash-Next IQ4_NL 2 GPU `-ncmoe 20` | **809** / 44.15 | 731–747 / 43.8 | 604 / 44.66 |
| Flash-Next IQ4_NL 2 GPU `-ncmoe 32` | **555** / 39.11 | – | 414 / 39.81 |
| Flash-Next IQ4_NL 2 GPU `-ncmoe 20 -b/-ub 4096` `pp4096` | **2901** / 44.60 | – | 2344 / 44.49 |
| 35B-A3B Q4_K_M 2 GPU `-ncmoe 40` | 1232–1250 / 99.18 | 1257 / 98.99 | **1252 / 99.14** |
| 35B-A3B Q4_K_M 3 GPU `-ncmoe 40` | 607 / 18.99 | – | 597 / **19.33** |
| Gemma4 26B-A4B Q8_0 2 GPU `-ncmoe 20` | 348 / 74.43 | – | 348 / **75.02** |

Two conclusions, both measured at every offload depth:

* **Dense**: CE wins both axes (its `pp8192` on the 27B is 2100 vs RCCL-with-peer 2096, and a staged peer
  copy is as fast as a direct one, so nothing is given up), and peer access is never needed.
* **Host-offloaded MoE**: RCCL's ring wins the prefill and the win is a *PCIe-saturation* effect, not a
  threshold in `-ncmoe`: at `-ncmoe 20`/`32` and at `-ub 4096` RCCL is 24–34 % ahead, and the two arms only
  converge when the model is essentially VRAM-resident (`-ncmoe 8` on 3 GPUs: 1207 vs 1190, then CE is
  marginally ahead at full residency).  A wider ubatch (less expert traffic per token) does not move it.

So the default routes the large arm by *whether the model offloads experts to host memory at all* — the one
signal that separates the regimes — and never enables peer access.

## The shipped default (r165)

Linux/HIP default = `ggml_backend_cuda_comm_init_auto`:

1. internal pipeline for the decode/verify band (unchanged — the small/large crossover is unchanged, so
   the decode and verify paths are bit-identical to r164 by construction);
2. RCCL brought up eagerly (the r34 purity rule) but with `NCCL_P2P_DISABLE` set if the user has not;
3. the copy engine brought up too;
4. the large arm chosen **once**, at the first large all-reduce: `g_comm_host_experts` (set by
   `ggml_cuda_slab_arm()` in the slab block) → RCCL's ring, otherwise → the copy-engine arm.

`GGML_CUDA_ALLREDUCE=hybrid` keeps the old P2P chain for a topology with genuinely cheap peer links;
`=ce` pins the copy-engine half; `GGML_CUDA_AR_CE_PEER=1` opts the copy-engine arm into peer access.

## Gates

* **Multi-device prefill-logit gate** (the AR arm is a prefill-numeric change; the gate could not reach it
  before, so it gained `--sm` and a device *list*):
  `gate-prefill-logits.sh --device 0,1,2 --sm tensor --model Qwen3.5-4B-Q8_0 --record --ab
  "GGML_CUDA_ALLREDUCE=hybrid"` → base arm RCCL, candidate the new default:
  **mean KLD 0.001448, same top-p 98.676 %** → PASS (limits 0.005 / 98 %).
* **Greedy purity**: `plain == draft-mtp` text on 27B Q8_0 3 GPU `-sm tensor` and 35B-A3B 2 GPU
  `-ncmoe 40` (`-n 48`, seed 42, `ROCBLAS_USE_HIPBLASLT=0`); same-arm repeats identical.
* **Protocol A** (27B Q8_0, 3 GPU `-sm tensor`, 100K ctx, prose prompt, same build, the two arms):
  plain **38.3 → 40.7 t/s** (+6.3 %), `draft-mtp` **80.9 → 88.5 t/s** (+9.4 %), acceptance **0.641**
  (1972/3075).  Quote these as a same-build A/B; the `patches/README.md` Protocol A reference figures were
  recorded on a different arm and are not comparable (the old build's plain arm is the one `#109` is about).
* `gate-per-block-build.sh --changed-from <r164 tip>`: **4/4** changed blocks build `all`.
* `validate-set.sh`: green (tree `6d7926b87ab6eff83da91a7c4d5e4b27c681aeef`).

## Reproduce

```bash
# the isolation (peer access alone, AR path untouched)
HIP_VISIBLE_DEVICES=0,1 GPU=1 ... # build with GGML_CUDA_AR_FORCE_PEER=<mode> (the instrument was
# removed after the measurement; it enabled pairs at comm init and logged the enable/disable cost)
# the shipped A/B
HIP_VISIBLE_DEVICES=0,1,2 ROCBLAS_USE_HIPBLASLT=0 <bin>/llama-bench -m Qwen3.8-27B-Q8_0.gguf \
  -ngl 99 -sm tensor -p 0 -n 128 -t 8 -r 5                    # default  ~42.3
HIP_VISIBLE_DEVICES=0,1,2 ROCBLAS_USE_HIPBLASLT=0 GGML_CUDA_ALLREDUCE=hybrid <same>   # ~38.6
```
