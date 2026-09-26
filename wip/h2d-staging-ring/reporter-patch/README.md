# Reporter's patch for issue #50: CUDA-backend H2D staging ring

This is the patch offered in [issue #50](https://github.com/stew675/llama-cpp-rdna-boosts/issues/50), added here so it can be A/B'd against `../h2d-stage.patch` (step 4 of `../README.md`). It corresponds to design option (b) in `../README.md` §4: the ring lives in the CUDA backend, and the scheduler keeps driving the copies.

- `0001-ggml-cuda-overlap-op-offloaded-MoE-expert-uploads-wi.patch`: one commit, about 160 lines in `ggml-cuda.cu`, `common.cuh` and `ggml-backend.cpp`. It applies cleanly with `git am` on top of `scripts/apply-all.sh` for both `v16-84e76d8a2-r9` (tested) and `v16-84e76d8a2-r10` (applies and builds; not re-benchmarked).

## Knobs (all off by default; with none set the build matches the base)

| env | effect |
|---|---|
| `GGML_CUDA_H2D_STAGING_SLOTS=N` | Whole-tensor H2D uploads of at least `GGML_CUDA_H2D_STAGING_MIN_MB` (default 64) go to a ring of N device slots on a dedicated non-blocking copy stream. The tensor is pointed at the slot, and `graph_compute` waits on the slot's upload event. After the compute, every side stream is joined and a free event is recorded, which gates the slot's reuse. If the next slot is still pending, the upload falls back to the in-order path. |
| `GGML_SCHED_MOE_COPY_ALL_MIN_TOKENS=T` | For batches of at least T tokens, the scheduler uploads the whole expert tensor instead of reading the ids back to copy only the used experts. |
| `GGML_SCHED_EVENTS=1` | Creates per-backend events even with one graph copy, so the wait before a split's inputs are overwritten is an in-stream event wait instead of a full synchronize. |

**Coupling to note:** the ring only takes whole-tensor uploads (`offset == 0 && size == nbytes`), and the scheduler's pruned MoE path uploads used experts as partial copies. So for expert tensors the ring effectively needs `COPY_ALL`. That's the same volume-for-overlap trade `../README.md` §5b describes. We never measured the ring without `COPY_ALL`.

## Our box

One R9700 (gfx1201), ROCm 10.0 (`rocm/dev-ubuntu-24.04:10.0.0-full`), Ryzen 9 9900X, DDR5. The link is **PCIe 5.0 x16 at the root port** (`00:01.1` LnkSta 32GT/s x16, and the same through both Navi bridges to the GPU). An H2D microbench measured ~53 GB/s DMA, with pinned equal to pageable. This is the wide-link end of the §4 x4/x16 split.

## Results (base `v16-84e76d8a2-r9`, 2026-09-26)

`llama-bench -ngl 99 -ncmoe 99 -fa 1 -b UB -ub UB -p 8192 -n 32 -r 2`, with ON = `SLOTS=3 COPY_ALL=1024 EVENTS=1`. Columns are r9 / patched with the env vars unset / patched ON:

| model | ub | pp8192 | tg32 |
|---|---|---|---|
| Qwen3.6-35B-A3B Q4_K_M | 2048 | 3025 / 3035 / **5293 (+74 %)** | 33.9 / 33.7 / 33.9 |
| Qwen3.6-35B-A3B Q4_K_M | 4096 | 3976 / 3978 / **6252 (+57 %)** | 33.9 / 33.7 / 33.4 |
| gemma-4-26B-A4B MXFP4 | 2048 | 3108 / 3097 / **4309 (+39 %)** | 17.5 / 17.7 / 17.6 |
| gemma-4-26B-A4B MXFP4 | 4096 | 3405 / 3405 / **3982 (+17 %)** | 17.7 / 17.7 / 17.6 |

`llama-server --cpu-moe -fa on -b 4096 -ub 4096 -c 65536` (same knobs):

- **Greedy text, ring off vs on:** byte-identical on 6 short prompts (512 tokens of reasoning + content) and on two ~20k-token prompts (256 tokens), for both models. Prefill at 20k: 35B 3681 → 5519 t/s; gemma 2392 → 2614 t/s.
- **Soak:** 10 minutes, ring on, 35B, each round a ~20k prefill plus two short concurrent requests. 81/81 OK, VRAM 5337 → 5346 MiB and levelling off, no errors in the log.

Earlier, on Flash-Next UD-Q4_K_XL (an r4-based tree, 128 GB RAM, `-ncmoe 48`, ub 4096, 196k context): prefill went from 1014/1079/917 to 1518/1473/1232 t/s at 27k/64k/190k, with decode unchanged. The needle test at 156k and a 20-minute soak were clean.

## Known gaps (relative to `../README.md` §5)

- **Unbounded:** slots grow to the largest tensor they hold, and a failed `cudaMalloc` aborts through `CUDA_CHECK` instead of falling back. The block-15 `fattn_stage_try_get` pattern would fix this.
- **Always whole-tensor** when used for experts (via `COPY_ALL`). On x4 your measurements show that loses at small ubatches, so it needs your adaptive gate.
- **Direct CUDA backend only:** `-sm tensor` and the meta backend are untested.
- **Graphs:** CUDA/HIP graphs are skipped for any compute that consumes staged uploads (prefill only).
- **Not gated yet:** no `W=1..8` width probe, MTP or deep-context gates on this patch.

We're happy to run your `h2d-stage.patch` on this x16 box alongside ours, to give the A/B a wide-link data point.
