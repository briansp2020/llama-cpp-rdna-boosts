# HANDOVER — Phase 2b tuning of the host-expert pool

Read first: [`README.md`](README.md), [`PARKED-BUG.md`](PARKED-BUG.md), [`PHASE2.md`](PHASE2.md), and the
repo [`AGENTS.md`](../../AGENTS.md).  **Profiling results and the target-by-target design/plan are now
in [`PHASE2B-FINDINGS.md`](PHASE2B-FINDINGS.md)** (2026-10-08): the pool hit rate is 3.8-20 %, target 1
is a 2x win on `-sm layer`, and the prefill seed/tally is dead under the pool.

## Where the work lives

* All code lives as **uncommitted changes in `~/llama.cpp`** (working tree = r33 `6e567349c` + the
  campaign changes).  Full diff: [`changes.patch`](changes.patch) (≈760 lines).  Nothing is in
  `patches/`/`release.json` and nothing was pushed.  Build:
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

1. **The miss path costs ~0.8–0.9 t/s** (35B: pool 2048 MiB/device buffered 6.1 vs pool-off 6.9).  The
   pool lookup + `cudaMemcpyAsync` from a slot vs the master.  Profile it (the fill is under `g_mutex`).
2. **The device-side admission policy is disabled for pooled tables** (`t.policy = g_devpolicy &&
   !g_pool_enabled && …`): the in-kernel policy fill reads the full master and cannot refill the pool.
   Making the policy kernel pool-aware (a device `expert -> pool_slot` map, host pre-fills the pool for
   the token's used experts before the flush) would keep the fast path.  This is likely the single
   biggest win on the reporter model.
3. **Prewarm ranking.**  `pool_prepopulate_locked` ranks by the arena's hot set (`t.slot_expert`), then
   the host prefill tally (`t.prefill_count`), then id order.  Under `-sm tensor` the host tally is
   **empty** (the device tally `prefill_count_dev` is consumed by `apply_prefill_seed_rank_locked`), so
   the prewarm is only as good as the arena's seeded residents.  Feed the device tally (or the prefill
   routing) into the prewarm.
4. **Per-device pools duplicate the cache.**  Under `-sm layer`/`-sm tensor` each device caches the same
   or its own experts; the budget is per device, so `MOE_HOST_POOL_MIB=20000` is 40 GiB total.  Decide
   whether the budget should be process-wide (a shared pool needs per-device in-flight tracking).
5. **Pool stats are not visible from `llama-cli`.**  **DONE (2026-10-08):** the pool summary is now
   `GGML_LOG_WARN`; `tools/cli/cli.cpp:36` pins llama-cli's default to `LOG_LEVEL_ERROR`, so the correct
   flags are `-lv 2` (the pool summary) and `-lv 4`/`-v` (the full `moe_cache_report`) — the earlier
   `-lv 1` note was wrong (level 1 is *error*).
6. **Phase 3** (the reason for all of this): remove the full pinned master and the scheduler
   `tensor->data` fallback for pooled tables, so the non-swappable bound is the pool alone.  This is the
   `get_cold`/remap re-architecture the README's "hard constraint" describes; the "invisible L2" route
   chose it deliberately (every pooled access stays cache-owned; `get_cold` would have to return false
   and every used expert must be admitted).
7. The `moe_host_expert_bytes` / `MOE_EXPERT_CACHE_*` sizing and the pool budget are independent today;
   consider whether `MOE_HOST_POOL_MIB` should default from the preflight estimate.

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
| `MOE_HOST_POOL_MIB` | unset/0 = off | pool size in **MiB per device** |
| `MOE_HOST_POOL_PREWARM` | 1 | fill every slot when a pool is built |
| `MOE_HOST_POOL_DIO` | **0** | `1` = O_DIRECT fill (debug; bypasses the page cache) |

Cache env (`MOE_EXPERT_CACHE_MIB` etc.) is unchanged; see the cache header.
