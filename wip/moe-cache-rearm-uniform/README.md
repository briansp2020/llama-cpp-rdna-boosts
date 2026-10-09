# moe-cache-rearm-uniform: re-arm a stood-down layer's other tables with it, so the layer stays slot-uniform

One format-patch on top of `v16-a55e952b8-r37`, applied with `git am`.  9 lines in
`ggml/src/ggml-cuda/moe-expert-cache.cu`.

## What

When the work pool takes slab chunks, `moe_cache_evict_slab_range` drops only the tables whose storage lay in the
taken range, and `moe_cache_rearm` re-allocates just those.  So a layer can come back with one role re-sized and the
others not.  Sizing already refuses such a layer ("the shared remap would name different experts per role"), and
`moe_cache_redirect_fused` reads the up table's remap for both the gate and up lanes.

The patch makes `moe_cache_rearm` add every allocated table of a stood-down layer to the set it re-arms, so the
existing attempt loop sizes them to one target.

## Seen

A local build (r34 plus a few patches of ours), `-sm layer -ts 59,41`, host experts (`--n-cpu-moe 48`): the
production warm-up's 17.8k-token prefill made the work pool take 302 MiB of slab chunks; `blk.47.ffn_up_exps` was the
one table in that range and came back at 193 slots while `ffn_gate_exps` / `ffn_down_exps` kept 502.  Greedy output
then changed on every request (five different shas, against a stable `9f826c79de59` before the warm-up); with
admissions frozen it settled on a stable but different sha.  `MOE_EXPERT_CACHE_VALIDATE=2` reported every table
consistent.  With the patch: `9f826c79de59` on every request before and after the warm-up (log: `re-armed 3
stood-down expert-cache tables`).

## Stock r37 repro

`-sm tensor --n-cpu-moe 48` (pinned host experts), `-ub 6144 -b 6144 -c 204800`, greedy 400-token request before and
after the 17.8k-token warm-up prompt; reference sha `45e68f1bdfeb` (also the all-VRAM sha).

| build | `GGML_CUDA_SLAB_HEADROOM_MIB` | tables re-armed | greedy after the warm-up |
|---|---|---|---|
| stock r37 | 4096 | 9 | `45e68f1bdfeb`, `45e68f1bdfeb` |
| stock r37 | 3072 | 11 | `3b99183e4a92`, then `02c7fce840cd` |
| r37 + patch | 4096 | 18 | `45e68f1bdfeb`, `45e68f1bdfeb` |
| r37 + patch | 3072 | 18 | stable but not the reference (`992a1e108d05` x2; another run `8237a68abb22` x2) |

At 2048 both builds abort with OOM in the warm-up prefill.

## Not fixed by this patch

At 3072 the output stays wrong with the patch: stable within a run, different between runs (`992a1e108d05`,
`8237a68abb22`, `5480747a87e3`).  `MOE_EXPERT_CACHE_VALIDATE=2` after the re-arm: 288 tables resident, 0
inconsistent.  The only log difference from 4096 is a 600 MiB `cudaMalloc` failure on both devices: the meta
backend's all-reduce tmp buffer (`ggml_backend_meta_graph_compute`, allocation not checked).  It is harmless here:
with that allocation forced to fail at 4096 the output stays `45e68f1bdfeb` and the null buffer is never used (the
butterfly fallback does not run).  So the cause is elsewhere in the tight-headroom eviction / re-arm; not found yet.

In the earlier `-ts 59,41` / `-ts 50,50` / default `-ub 2048` runs on stock r37 the eviction took gate and up of a
layer together (re-armed to one size) and the output stayed correct; `ffn_down_exps` of that layer stayed at 509
slots while gate/up went to 10, so down apparently does not have to match.
