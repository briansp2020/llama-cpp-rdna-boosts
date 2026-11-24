# ours: our production stack on top of the current rdna-boosts release (r351)

This branch holds the patch series we run on top of the latest rdna-boosts release: 2x Radeon AI PRO R9700,
ROCm 10.2, Qwen3.8-Flash-Next UD-IQ4_XS with `-sm tensor` and `--host-experts mmap`. It is regenerated for each
release, and patches the release has taken are dropped. It is not a PR. Everything else on this branch is the
release as published; only `ours/` is added.

**Base:** `v18-919794500-r351` (upstream llama.cpp `919794500` + this repo's release).

```
git checkout 919794500
bash <path-to-this-repo>/scripts/apply-all.sh .     # the r351 release, branch rdna-boosts
git am <path-to-this-repo>/ours/*.patch              # 32 patches (0001-0032)
```

Result: tree `fa393c6d3159`. This is the production build since 2026-11-24 6:40 AM PST.

## How the series is organised

- **0001** is everything outside `moe-expert-cache.cu`.
- **0002-0003** are our `--host-experts mmap` expert cache. The release's `moe-expert-cache.cu` stays as released.
  Our code lives in two included files, `moe-expert-tier.inc` and `moe-expert-tier-state.inc`, and runs only for
  sparse (mmap) tables. Pool and pinned tables run the release's own code. The tier code covers the RAM tier, the
  pin budget, KV lending, the fill fence, batched admissions and the device remap for sparse tables.
- **0004-0032** are one feature each, in the order they were added.
- A fix to one of our patches goes into that patch, not into a new one. A later patch that touches an earlier
  one's code adds a feature of its own, with its own switch.
- Every behaviour change has an environment switch. The table gives the value that turns it off (or the opt-in).
  Patches marked bit-identical give the same output on or off.

## Patches

| | what | switch (off) |
|---|---|---|
| 0001 | everything outside `moe-expert-cache.cu`: scheduler, server, MoE / GDN / shared-expert fusions, dedup, slab allocator, the KV lending hooks in llama, the checkpoint pool (`common_state_data`, pages touched by the pool thread), and the RDNA2 rows-abreast K loop in our MoE kernel. Our strided-upload fallback uses the release's host stream (`ggml_cuda_host_stream()`) | per feature; `GGML_CUDA_MOE_ROWS_ABREAST=0` |
| 0002 | the mmap expert-cache tier code in `moe-expert-tier.inc` / `moe-expert-tier-state.inc`, not yet called. Its namespace-scope env flags are initialised by named functions, not lambdas: clang can give lambdas in different reopened anonymous namespaces of one TU the same symbol, so one flag would read another's variable | per feature |
| 0003 | the hooks: the release's `moe-expert-cache.cu` calls the tier code for sparse tables. Also: the device remap and the tier remap predicate refuse a table whose fill failed; identity (fully resident) tables are served in place above the band (warn-once `MOE_INPLACE_MISS` tripwire); a table counts as pool-backed only when the pool was requested; the release's fused prefill and take-over promise work with tier tables (gate/up lanes with different remaps count as mixed) | per feature; `MOE_EXPERT_PREFILL_IN_PLACE=0`, `GGML_MOE_POOL_TABLE_REQ=0` |
| 0004 | layer split with pipeline parallelism overlaps the devices on ROCm: prompt ubatches rebuild their graph, and host inputs go through a pinned per-(backend, copy) ring uploaded on the split's stream; opt-in `GGML_GALLOC_FP_SHAPES=0`, `LLAMA_KV_N_PAD_PREFILL_POW2=1` | `LLAMA_PP_PREFILL_NO_REUSE=0`, `GGML_SCHED_INPUT_STAGE=0` |
| 0005 | the dual-output mmvq fusion skips weights that live in a compute buffer | `GGML_CUDA_FUSION_CHECK_LEAVES=0` |
| 0006 | CUDA graph eviction counts active time only (idle gaps capped at 1 s); the looked-up graph is never evicted | - |
| 0007 | qwen4exp: PLE page hints (`MADV_WILLNEED`) only when a `mincore` sample says the table is cold (~13 ms per 4096-token ubatch) | `LLAMA_PLE_ADVISE=1` (always advise) |
| 0008 | multi-device: `alloc_splits` skips the drain when no buffer grows | `GGML_SCHED_DEFER_ALLOC_SYNC=0` |
| 0009 | `-sm tensor`: graph inputs in pinned memory (the meta device's first child host buffer type), uploaded asynchronously per device; host compute buffers kept across the drop | `LLAMA_META_HOST_BUFT=0`, `GGML_SCHED_ASYNC_USER_INPUT=0`, `GGML_GALLOC_DROP_HOST=1` |
| 0010 | meta: the split-input copy of a split host weight is sized at the largest device slice, not the whole tensor | `GGML_META_SLICE_ALLOC=0` |
| 0011 | moe cache: a device-remap call does not re-batch a sibling already taken over in the same round | `GGML_MOE_DEVREMAP_REBATCH=1` |
| 0012 | qwen4exp qsa3: Q8_0 cache -> F16 packs in one copy; the gate copy runs before attention on prefill-sized graphs | `LLAMA_QSA3_DIRECT_F16=0`, `LLAMA_GATE_EARLY=0` |
| 0013 | qwen4exp qsa3: the packs read the Q8_0 cache views directly, no F16 copies in the graph | `LLAMA_QSA3_PACK_Q8=0` |
| 0014 | `-sm tensor` ubatch overlap: asynchronous graph-input views, asynchronous CPU-split output upload, deferred input wait | `GGML_SCHED_ASYNC_INPUT_VIEWS=0`, `GGML_SCHED_ASYNC_CPU_OUT=0`, `LLAMA_DEFER_INPUT_WAIT=0` |
| 0015 | decode MoE gate+up+GLU fusion also when the GLU output overlaps only the unfilled weight copy | `GGML_MOE_FUSE_COPY_ALIAS=0` |
| 0016 | mmq: RDNA4 IQ routed-compact experts with 65-96 rows per expert use J=64 (+1-2 % prefill at `-ub 4096`); bit-identical | `GGML_MMQ_IQ_J64_WIDE=0` |
| 0017 | moe cache: the release's in-place prefill stands down when a sparse table exists (only reachable with a GTT arena and mixed tables; inert on discrete GPUs) | - |
| 0018 | KV segment: under `GGML_MOE_KV_LEND=1`, the lent KV rows past the context in use are an expert-cache tier below the arena (heat-seeded, then demand-filled from the RAM tier); every device lends its own K/V slices under `-sm tensor` | opt-in `GGML_MOE_KV_SEG=1` |
| 0019 | KV segment: one take-back per request (the server hints the prompt length; the hint returns to 0 when idle) instead of stepwise takes during long prefills | `GGML_MOE_KV_SEG_TAKE1=0` |
| 0020 | KV segment: small prefill ubatches (decode band < n < 512) read non-resident experts from their KV copies; on by default with the segment; per-source counters in the tier report | `GGML_MOE_KV_SEG_PF=0` |
| 0021 | loader: open the MoE heat map in `--host-experts mmap` mode | `LLAMA_MMAP_HEAT_OPEN=0` |
| 0022 | the persisted-heat seed of a sparse table fills through the tier code (a pinned staging copy, not a pageable copy from the mapping) | - |
| 0023 | a sparse table's staged fill at sizing takes the heat map's hottest experts; the per-expert heat seed leaves sparse tables to it | `GGML_MOE_STAGE_HEAT_RANK=0` |
| 0024 | the release's arena settle skips layers that hold an mmap-tier table (re-sizing such a layer would free the table and turn the RAM tier off) | `GGML_MOE_SETTLE_SPARSE=1` (release behaviour) |
| 0025 | the Q8_1 copy of a layer's input is carried across the routed-MoE graph split, so the routed experts do not quantize it again; bit-identical | `GGML_CUDA_Q8_1_CARRY=0` |
| 0026 | mmq: Q5_0 expert tables on the RDNA4 routed-compact path with the IQ4_NL bands, so prefill reads experts in place instead of gathering the whole table per layer; bit-identical | `GGML_MMQ_Q5_0_COMPACT=0` |
| 0027 | server: when the last prompt chunk is the checkpoint tail and has the verify width (draft n_max + 1), all its rows are outputs, so the first verify reuses the tail's graph (agent-style 2K extensions -3.4 % time) | `LLAMA_CKPT_TAIL_OUTPUTS=0` |
| 0028 | mmvq: the dense gate/up + GLU fusion at the verify width (2-8 tokens, RDNA4), the launch also publishing the down projection's Q8_1 input; the shared expert's 3 launches per layer become 1; bit-identical | `GGML_CUDA_DENSE_GLU_VERIFY=0` |
| 0029 | mmvq-moe (RDNA4): rows per block 8 (8-10 K-blocks) or 4 (11-15) where 2-row blocks leave the last pass partly idle; the Q5_0 MoE down at K 320 goes 48 -> 34 us at 4 tokens; output changes at rounding level (KLD ~0) | `GGML_CUDA_MOE_RPB_K=0` |
| 0030 | server: idle slots are saved to the prompt cache and cleared before the restore of the new request, not after its launch, so a long prompt after other requests starts with the KV room free (259.6k needle after other requests 79 -> 71 s, +11.6 %); output unchanged | `LLAMA_SERVER_IDLE_CLEAR_FIRST=0` |
| 0031 | hyperconn: hc_combine_norm also writes the Q8_1 MMQ activation of its dense consumer, so that matmul skips its own quantize (prefill, 64+ tokens); 31.5k prefill +1.9 %, 259.6k needle +1.4 %; bit-identical | `GGML_CUDA_HC_CN_MMQ_Q8=0` |
| 0032 | llama-batch: ubatch data blocks that no ubatch references any more are reused instead of freshly allocated (capped at 256 MiB, `LLAMA_UBATCH_REUSE_MIB`), so the MTP draft no longer zero-fills and page-faults ~100 MiB of state rows after every long prompt; server-thread page faults -92 %, 4.2k-token request -20..27 ms; output unchanged | `LLAMA_UBATCH_REUSE=0` |

## Production configuration

```
llama-server -m Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00004.gguf -c 262144 -ngl 99 \
  -sm tensor --n-cpu-moe 48 --host-experts mmap -ctk q8_0 -ctv q8_0 \
  -cram 2048 -ub 4096 -b 4096 --reasoning-effort low --n-predict 32768
```

Settings on top of the defaults that matter for this series:

- `LLAMA_TP_FFN_EVEN=1`: the release's opt-in even FFN / shared-expert split.
- `GGML_MOE_KV_LEND=1`, `GGML_MOE_KV_SEG=1`: the KV segment (0018-0020).
- `GGML_MOE_PREFILL_INPLACE=1`, `GGML_MOE_PF_PREFETCH=1`: in-place prefill for tier tables, and RAM-tier experts staged one layer ahead.
- `MOE_HOST_POOL_TRANSPORT=map`: pins the host-read transport.
- `GGML_CUDA_ALLREDUCE=ce`, `GGML_CUDA_AR_P2P_LARGE=1`, `GGML_CUDA_AR_SPIN_TIMEOUT_MS=20000`: the copy-engine BF16
  all-reduce, with a timeout long enough for KFD queue stops.
- Slab and pool sizing: `GGML_CUDA_SLAB_HEADROOM_MIB=3072`, `GGML_CUDA_SLAB_WORK_MIN_MIB=2304`,
  `GGML_CUDA_SLAB_RING_PREFILL=0`, `GGML_CUDA_POOL_CACHE_MB=128`, `GGML_COMPUTE_BUFFER_MARGIN_PCT=3`.
- `LLAMA_SERVER_PREFILL_DUTY=32`: up to 31 decode-only steps between prefill steps, so a slot that is decoding runs
  about 1.6x faster while another slot prefills a long prompt (that prompt's first token comes about 20 % later).
  It only matters when requests overlap.

With 57.6 GiB of host experts, the expert cache holds 68.2 % in VRAM.

## Validation (r351 + 0030-0032, 2026-11-23/24)

- The 7-model comparison with the release passes (nightly gate: 0 problems, 0 warnings, 0 failed loads), with
  speeds equal to the release.
- Production configuration: output identical to the r351 series on every check: greedy, 512-token prompt set,
  4.2k and 30k-token prompts, and the 259.6k-token needle (6,358). A prompt restored from the cache after 0030
  gives the same output as a cold start.
- 0031 is bit-identical: in its verify mode every handed-over activation equals the one the matmul quantizes itself.
- Overnight stress (2 h, 2 GPUs, three configurations, 58 rounds): 0 output mismatches, 0 GPU faults, needle
  correct in every round; cold and heat-map-seeded starts give the production outputs.
- Speed (production configuration, single runs): 31.5k prefill +1.9 %, 259.6k needle +1.4 % over the r351 series;
  decode unchanged (512-token prompt set 151 t/s); a 4.2k-token request finishes 20-27 ms sooner with 0032.

For earlier releases, see this file's history or the `ours-rNNN` branches.
