# HANDOVER — Phase 2b tuning of the host-expert pool

Read first: [`README.md`](README.md), [`PARKED-BUG.md`](PARKED-BUG.md), [`PHASE2.md`](PHASE2.md), and the
repo [`AGENTS.md`](../../AGENTS.md).  **Profiling results and the target-by-target design/plan are now
in [`PHASE2B-FINDINGS.md`](PHASE2B-FINDINGS.md)** (2026-10-08): the pool hit rate is 3.8-20 %, target 1
is a 2x win on `-sm layer`, and the prefill seed/tally is dead under the pool.

## Where the work lives

* All campaign code lives on the **`wip/host-expert-pool` branch in `~/llama.cpp`**.  **RE-BASED onto r34
  2026-10-08 (done):** the branch is now `40ce2ab86` (r34 tip / tree `a2536910`) + the 6 campaign
  commits, and the wip `lazy NCCL init` commit was **dropped** (it is already block 12 of r34).  The
  rebased tree is bit-identical to the pre-rebase campaign tree (`dd987030`), so no rebuild/gate was
  needed; `changes.patch` was regenerated with `git -C ~/llama.cpp diff 40ce2ab86`.  A local
  `wip/host-expert-pool-presync` branch (tip `26d743dd4`) preserves the old, un-rebased branch.  Nothing
  of the campaign is in `patches/`/`release.json`.  Build:
  `cd ~/llama.cpp && BUILD_DIR=build-rocm-hybrid EXTRA_CMAKE_FLAGS="-DCMAKE_HIP_FLAGS=" ~/bin/build-llama-rocm-714`
  (or the fast loop `cmake --build build-rocm-hybrid --target llama-cli -j 16`).
* `sudo /usr/local/sbin/reset-amd-gpus` (installed, works) recovers a wedged GPU.  Wrap runs in `timeout`,
  `llama-cli` always `--single-turn`, size `-t 8` to leave the IRQ cores free.
* **WIP rules:** do not touch `patches/`, `release.json` or the blocks, and do not push, without the
  maintainer's explicit go-ahead.

## Done

* **Parked bug fixed** (`PARKED-BUG.md`): the slab granted `cuMemSetAccess` to the owning device only, so
  a cross-device `hipMemcpyPeerAsync` into the slab work region faulted.  Now every peer device is
  granted.  Plus two latent `cudaFree`-of-a-slab-arena fixes and a `mul_mat_vec_q_moe` row-tail clamp.
* **Phase 1**: `MOE_HOST_POOL_MIB` + host-source registry (`moe_cache_set_host_source`, GGUF path/offset
  from `llama_file::path()` in the loader).  Inert when the env is unset.
* **Phase 2**: the bounded pinned pool in `ggml/src/ggml-cuda/moe-expert-cache.cu`.
  * One pool per `(host tensor, device)` (`g_host_pools`), whole host experts, LRU, per-slot
    in-flight `cudaEvent_t`, prewarm.
  * **Buffered fill through the page cache is the default** (the pool is a GPU-readable *bounce
    buffer*); `MOE_HOST_POOL_DIO=1` is a debug fallback that bypasses the page cache (much slower for a
    bounded pool).
  * Wired into `access_locked`'s arena fill ("invisible L2": the GPU reads only the arena).  The full
    pinned master is still present.
  * `MOE_HOST_POOL_PREWARM` (default on) fills every slot when a pool is built.
* **Validated**: output-identical pool on/off for `-sm tensor` and `-sm layer` (35B `359ff4337837`);
  dense 4B `1c5d32ac537d`; `MUL_MAT_ID` 931/931; prefill KLD gate PASS; reporter model end-to-end (2×R9700,
  Flash-Next IQ4_NL, `-ncmoe 48`, 40 GiB pool, 5246-tok prefill + `-n 1000`): `-sm layer` 137.1/18.8 t/s,
  `-sm tensor` 182.2/16.3 t/s, no faults, coherent.

## Open work (Phase 2b tuning targets)

1. **The miss path costs ~0.8–0.9 t/s.**  **DONE (2026-10-08, target 2):** the eviction fetch moved to a
   low-priority background worker (nice 19; reserve/publish under `g_mutex`, the `pread` outside it;
   bidirectional dedupe with the critical path).  `MOE_HOST_POOL_BGFETCH=0` restores the synchronous
   path.  Root cause of the cost (a 3.8–20 % pool hit rate → a `pread` per fill) and the numbers are in
   `PHASE2B-FINDINGS.md`.
2. **The device-side admission policy is disabled for pooled tables.**  **DONE (2026-10-08, target 1):**
   pooled tables now use the device policy (the `!g_pool_enabled` gate is gone) and the in-kernel fill
   sources the bounded host pool: a device `expert -> pool_slot` map is snapshotted from the pool LRU
   each flush, with a master fallback for non-resident experts.  The per-table promote keeps the pool
   warm for the routed experts through the target-2 background worker.  `MOE_HOST_POOL_POLICY` is the
   kill-switch: `0` = original host promotion for pooled tables, `1` = pool-aware (default), `2` =
   device policy filling from the master (the 1b probe).  35B-A3B Q4_K_M `-sm layer -ncmoe 40`,
   `MOE_EXPERT_CACHE_MIB=2048` + `MOE_HOST_POOL_MIB=2048`, prose `-n 128`: host promotion **41.0** ->
   pool-aware device policy **43.1** t/s (pool off device policy **45.4**).  Mode 1 ~= mode 2, i.e. the
   pool-sourced fill is not faster than the master fill -- the win is enabling the policy engine for
   pooled tables.  This is a `-sm layer` change; split tables keep host promotion unless
   `DEVPOLICY_SPLIT=1` (the followup campaign).
3. **Prewarm ranking.**  **DONE (2026-10-08, target 3):** the device prefill tally now drives both the
   arena seed and `pool_rerank_from_tally_locked` (which excludes arena residents — additive), and the
   seed runs outside `if (g_devpolicy)` so it works for the host-promotion/pool path.  `-sm layer`
   pool-on 36.3 -> 41.1 t/s.
4. **Per-device pools duplicate the cache.**  **DONE (2026-10-08, targets 4a+4b):** one pool per host
   tensor with a **process-wide** budget, per-`(slot, device)` in-flight events, `cudaHostAlloc(Portable|
   Mapped)` + a D2D UVA fill, a degenerate-geometry guard, and the **additive** lifecycle (fetch on
   arena eviction, retire on admission once the copy drains).  Note the semantics change:
   `MOE_HOST_POOL_MIB` is now the process-wide TOTAL, not per device.
5. **Pool stats are not visible from `llama-cli`.**  **DONE (2026-10-08):** the pool summary is now
   `GGML_LOG_WARN`; `tools/cli/cli.cpp:36` pins llama-cli's default to `LOG_LEVEL_ERROR`, so the correct
   flags are `-lv 2` (the pool summary) and `-lv 4`/`-v` (the full `moe_cache_report`) — the earlier
   `-lv 1` note was wrong (level 1 is *error*).
6. **Phase 3 (master removal) -- WITHDRAWN, not a goal.**  The maintainer confirmed 2026-10-08 that the
   host pool is an **option** for systems that cannot hold the weights resident; where they can be held
   resident, the usual (fully-resident / pinned-master) behaviour stays.  Do **not** remove the full
   pinned master or the scheduler `tensor->data` fallback.  The original (now moot) plan is kept below for
   the record: remove the full pinned master and the scheduler `tensor->data` fallback for pooled tables,
   so the non-swappable bound is the pool alone -- the `get_cold`/remap re-architecture the README's "hard
   constraint" describes.
7. **The pool's AUTO budget comes from the preflight.**  **DONE (2026-10-08):**  `--host-experts pool`
   selects the pool, and `MOE_HOST_POOL_MIB` (default `auto`) sizes it to `MOE_HOST_POOL_FRAC` (25) % of
   the MoE host
   expert bytes (`moe_host_expert_bytes`, the preflight accumulation -- not the whole model, so qwen4exp's
   PLE is excluded); 4645 MiB pinned for the 35B `-ncmoe 40`.  `pinned` stays the default (no pool); an
   explicit MiB wins.  See
   the long-prompt sweep in `PHASE2B-FINDINGS.md`.
8. **`-sm tensor` slowness** (found while answering the maintainer, 2026-10-08).  Two separate items: a
   **block-12 all-reduce bug** (the default hybrid eagerly calls `ncclCommInitAll`, degrading the internal
   pipeline ~3x; fixed on the wip branch, needs sign-off — [`SM-TENSOR-AR.md`](SM-TENSOR-AR.md)) and the
   **split-table admission-policy choice** (the cache's device policy is hard-disabled for split tables;
   followup campaign [`../cache-split-admission/`](../../wip/cache-split-admission/README.md)).

## Reproduce

```sh
# 35B pool A/B (35B-A3B Q4_K_M), pool 2048 MiB/device, buffered
M=/llm/models/Qwen3.6/35B-A3B/Q4_K_M/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf
HIP_VISIBLE_DEVICES=0,1,2 MOE_EXPERT_CACHE_MIB=2048 MOE_HOST_POOL_MIB=2048 \
  ~/llama.cpp/build-rocm-hybrid/bin/llama-cli -m $M -ngl 99 -sm tensor -ncmoe 40 -fa 1 -t 8 \
  -p "The capital of France is" -n 256 --seed 42 --temp 0 --no-display-prompt --single-turn --reasoning off

# reporter model (2 GPUs, 40 GiB total pool, long prefill + 1000 decode)
FM=/llm/models/Qwen3.8/Flash-Next/IQ4_NL/Qwen3.8-Flash-Next-IQ4_NL-PROJFIX-00001-of-00009.gguf
HIP_VISIBLE_DEVICES=0,1 MOE_HOST_POOL_MIB=20000 \
  ~/llama.cpp/build-rocm-hybrid/bin/llama-cli -m $FM -ngl 99 -ncmoe 48 -sm layer -fa 1 -t 8 -c 16384 \
  -f ~/llama-cpp-rdna-boosts/prompts/prose-rdna-boosts.txt -n 1000 \
  --seed 42 --temp 0 --no-display-prompt --single-turn --reasoning off
```

## Env (pool)

| var | default | meaning |
|---|---|---|
| `--host-experts pool` | (option; default `pinned`) | enable the bounded pinned host pool.  `pinned`/`auto` = no pool; `pool` = on |
| `MOE_HOST_POOL_MIB` | **auto (25 %)** | pool size when enabled: unset/`auto` = `MOE_HOST_POOL_FRAC` % of the MoE host expert bytes; `N` = explicit MiB; `0` disables even with `--host-experts pool` |
| `MOE_HOST_POOL_FRAC` | **25** | auto size as a percentage of the MoE host expert bytes (`moe_host_expert_bytes`) |
| `MOE_HOST_POOL_POLICY` | **1** | pooled tables on the device policy: `1` = pool-aware fill (default), `0` = host promotion (original), `2` = device policy filling from the master (1b probe) |
| `MOE_HOST_POOL_PREFETCH` | **0** | `1` = routing-driven speculative prefill (a net loss unless the pool nears full residency; long-prompt sweep in `PHASE2B-FINDINGS.md`) |
| `MOE_HOST_POOL_PREWARM` | 1 | fill every slot when a pool is built (one-shot, from the prefill tally / arena residents) |
| `MOE_HOST_POOL_BGFETCH` | 1 | `0` = synchronous eviction prefetch (no low-priority worker) |
| `MOE_HOST_POOL_DIO` | **0** | `1` = O_DIRECT fill (debug; bypasses the page cache) |

Cache env (`MOE_EXPERT_CACHE_MIB` etc.) is unchanged; see the cache header.
