# moe-cache-layer-split-fixes: two fixes for pinned host experts under `-sm layer` at long context

Two small format-patches on top of `v16-a55e952b8-r33` (strict `git am`; also clean on r34, and in either order with
`wip/moe-cache-inplace`).  Both only act when the MoE expert cache holds host-expert tables, so `-ncmoe 0` and
cache-off runs are unchanged.

## 0001: a 512 MiB free-VRAM floor on the workspace pool's first attempt

With host-expert tables in the cache, the workspace pool can take a device down to ~40 MiB free.  In that state the
arena tables in the slab extension get overwritten with live data.  Under `-sm layer -ts 59,41 --n-cpu-moe 48` it
shows as a broken greedy decode right after a 61k-token prefill: the MTP draft acceptance drops to 0/1191 and the
text is wrong.  (The same state is what #116 traced on the pageable path; `-sm tensor` with a pinned master is
protected by the driver refusing allocations earlier, at ~188 MiB free.)

We have not found what writes into the extension.  This patch only keeps the device out of that state: the pool's
first attempt fails like a real OOM when it would leave less than 512 MiB free, so its existing flush + retry runs
instead.  The retry has no floor, so it can never turn into an abort.  `GGML_CUDA_POOL_MIN_FREE_MIB` sets the floor
(0 = off); the default is 512 when `moe_cache_floor_active()` (the cache was sized with host-expert tables), else 0.

## 0002: under `-sm layer`, leave 6 GiB free after the slab extension (default 4 GiB)

Under `-sm layer` every table is unsplit and each device runs its layers' attention over the whole context, so the
later-layer device needs more workspace at long context.  A 255k-token prompt ran device 1 out of VRAM with the
default 4 GiB of headroom (`ggml_cuda_device_malloc: 16.82 MiB allocation failed on device 1 while the
movable-boundary slab holds the VRAM`, then `ROCm error: out of memory`, server exits).  When every cache table is
unsplit and the tables span more than one device, the sizing now asks for at least 6 GiB of headroom before
extending (`ggml_cuda_slab_headroom_at_least_mib`).  An explicit `GGML_CUDA_SLAB_HEADROOM_MIB` still wins.

## Results on r33 (2 x R9700 / gfx1201, ROCm 10.0, Flash-Next GSQ-RCO IQ3_XXS + shared Q8_0 MTP head)

Pinned host experts, `--n-cpu-moe 48`, 256K, q8_0 KV, `-ub 2048`, MTP n-max 3, default cache settings, one run each:

| | r33 | r33 + 0001 + 0002 |
|---|---|---|
| `-sm layer -ts 59,41`: greedy decode after a 61k prefill | broken: sha `43aaeff3595c`, draft 0/1191 | sha `9f826c79de59` (= warm-up), draft 273/377 |
| `-sm layer -ts 59,41`: 255k-token needle (depth 0.5) | device 1 out of memory, server exits | PASS, 1102 t/s; decode after it `9f826c79de59` |
| `-sm layer -ts 59,41`: 2.9k / 61k prefill | 739 / 1236 t/s | 715 / 1232 t/s |
| `-sm tensor`: decode / 2.9k / after-61k / after-255k shas | (unchanged) | `45e68f1bdfeb` / `0a6791b84da2` / `45e68f1bdfeb` / `45e68f1bdfeb` |
| `-sm tensor`: 61k prefill | 1384 t/s (separate run) | 1382 t/s |
| `-sm tensor`: 255k-token needle | not run | PASS, 1095 t/s |

0 GPU faults in every run.

## Not measured

3+ GPUs; other `-ts` ratios; RDNA3 / NVIDIA; how often the floor's flush + retry fires (the pool logs it at debug
level only).
