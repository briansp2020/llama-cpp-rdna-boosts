# ours: our production stack on top of the current rdna-boosts release (r366)

This branch holds the patch series we run on top of the latest rdna-boosts release: 2x Radeon AI PRO R9700,
ROCm 10.2, Qwen3.8-Flash-Next UD-IQ4_XS with `-sm tensor` and `--host-experts mmap`. It is regenerated for each
release, and patches the release has taken are dropped. It is not a PR. Everything else on this branch is the
release as published; only `ours/` is added.

**Base:** `v18-919794500-r366` (upstream llama.cpp `919794500` + this repo's release).

```
git checkout 919794500
bash <path-to-this-repo>/scripts/apply-all.sh .     # the r366 release, branch rdna-boosts
git am <path-to-this-repo>/ours/*.patch              # 40 patches (0001-0040)
```

Result: tree `2ff0e2404dee`. This is our production build from 2026-11-26.

Also in `ours/`: `bench/llbench.py` (the benchmark below), `deploy/docker-compose.yaml` (our production
compose file, local paths replaced by `./models`) and `tools/mkhead.py` (one step of the draft-head recipe).

## Performance (r362 + 36, 2026-11-25)

Everything below is one 2x R9700 box running the production configuration (see *Reproducing it*): 256K
context, 4 slots, MTP speculative decoding (draft n-max 3), greedy requests.

**Coding-agent depth curve.** This is our replica of the local-llm-benchmarks.dev protocol
"deterministic-prime-extend-completions-v1" (`bench/llbench.py`). At each depth D, the first D tokens of a
source-code corpus are put in the slot's cache (untimed). The request then sends those D tokens plus a fresh
2048-token section and generates 256 tokens at temperature 0 with `ignore_eos`. PP is the 2048 uncached tokens
over the time to the first token; TG is generated tokens over decode time. Each row is the mean of 2 measured
runs after 1 warm-up, one slot, one request at a time.

| depth | PP (t/s) | TTFT (s) | TG (t/s) |
|---|---|---|---|
| 0 | 2312 | 0.89 | 129 |
| 8K | 2675 | 0.77 | 127 |
| 16K | 2641 | 0.78 | 140 |
| 24K | 2618 | 0.78 | 154 |
| 32K | 2504 | 0.82 | 151 |
| 40K | 2463 | 0.84 | 145 |
| 48K | 2426 | 0.85 | 141 |
| 56K | 2386 | 0.86 | 151 |
| 64K | 2423 | 0.85 | 153 |
| 128K | 2263 | 0.91 | 132 |
| 192K | 2111 | 0.97 | 130 |
| 240K | 2005 | 1.02 | 126 |

TG is MTP-assisted, so it moves with draft acceptance (48-100 % of drafted tokens accepted per run on this corpus) more than
with depth: single runs at the same depth range from 106 to 158 t/s. The site's own runner is not public; ours
differs in three ways: it uses llama-server's native `/completion` with token-id prompts (not
`/v1/completions`), it sets `ignore_eos`, and the corpus is this repository's source files.

**Other measurements on the same server:**

| | |
|---|---|
| one 150K-token prompt, cold | 3,580 t/s prefill (42 s), then 124 t/s decode |
| one 4.3K-token prompt | 3,035 t/s prefill, 155 t/s decode |
| greedy 512-token answer, short prompt | 132-153 t/s |
| agent turn (append a tool result to a long conversation, short answer) | 773 ms median, 429 ms of it prompt |
| two short requests at once | 100 + 109 t/s (one alone: 121 t/s) |

## Reproducing it

**Hardware.** 2x AMD Radeon AI PRO R9700 (32 GB each, gfx1201), each on PCIe 5.0 x8; AMD Ryzen 9 9900X;
64 GB DDR5-6000 (2x 32 GB); Ubuntu, Linux 7.0. Most of the routed experts live in host RAM and are read through
`--host-experts mmap`, so the model files should sit on a fast NVMe drive and RAM matters: with 64 GB, the
page cache holds part of the 57.6 GiB of host experts.

**Software.** ROCm 10.2.0. We build and run in `rocm/dev-ubuntu-24.04:10.2.0-full` containers:

```
cmake -S . -B build -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1201 -DCMAKE_BUILD_TYPE=Release -DGGML_HIP_GRAPHS=ON \
  -DGGML_NATIVE=ON -DGGML_HIP_RCCL=1 -DGGML_CUDA_NCCL=ON -DGGML_HIP_NO_VMM=ON -DGGML_HIP_MMQ_MFMA=ON -DLLAMA_OPENSSL=ON
cmake --build build -j
cmake --install build --prefix <stage>/usr/local
```

The production image is `FROM rocm/dev-ubuntu-24.04:10.2.0-full`, plus `curl` and `python3`, with
`<stage>/usr/local` copied to `/usr/local`, `ldconfig`, and `LD_LIBRARY_PATH=/opt/rocm/lib`.

**Model.** Unsloth `unsloth/Qwen3.8-Flash-Next-GGUF`, folder `UD-IQ4_XS` (4 shards, UD 3.1), unmodified:

| file | sha256 |
|---|---|
| Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00004.gguf | `93ff23a08e5cd7501f24fa16f5b415d5807d0175e04edd3ed6a7a41b601a4eef` |
| Qwen3.8-Flash-Next-UD-IQ4_XS-00002-of-00004.gguf | `13d5dc8b93d039e4cfade5a9af7037f10c97f7ec4e53f75bd2e9ae34aae54aeb` |
| Qwen3.8-Flash-Next-UD-IQ4_XS-00003-of-00004.gguf | `f6b9b5934908a2043732269dd740155d0f921d0763a1dcdd3452246311651f87` |
| Qwen3.8-Flash-Next-UD-IQ4_XS-00004-of-00004.gguf | `5de8b5dbf89409acc39468decafd4a4a95b13c29d88a14b0356107897829ec36` |

**MTP draft head.** The simplest choice is Unsloth's `MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf` from the same
repository (sha256 `5ff54097406a905cf3a724c709124ceb0e3e10235ee862298969e91c96fa96e6`), used as is. The
numbers above use our variant of it, `mtp-head-expQ4K.gguf` (sha256
`3a8cfd264da3969c2e872e5b1cacd95a80b7cdfa66cd551c43a9d8fda7851893`). In our tests it gives identical output and
4-5 % faster decode, because the draft head's experts take less VRAM and leave more room for the expert cache.
It was made in two steps from the Unsloth head:
1. `tools/mkhead.py` adds a draft-only LM head (`blk.48.nextn.shared_head_head.weight`, the target's
   `output.weight` requantized to Q4_0). Only draft passes use it; verification keeps the target's own head.
2. `llama-quantize --allow-requantize --tensor-type ffn_gate_exps=q4_k --tensor-type ffn_up_exps=q4_k
   --tensor-type ffn_down_exps=q4_k --tensor-type shared_head_head=q4_0 <step-1 file> mtp-head-expQ4K.gguf Q8_0`
   (`ffn_down_exps` falls back to Q5_0: its 640-wide rows are not a multiple of 256).

We made this file some weeks ago and have not re-run the recipe since, so check the sha256. If yours differs, the
Unsloth head is the safe choice; with it, expect decode about 4-5 % lower than in the tables.

**Launch.** `deploy/docker-compose.yaml` is our production service with every setting and environment variable.
The command line is:

```
llama-server -m Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00004.gguf -c 262144 -ngl 99 \
  -sm tensor --n-cpu-moe 48 --host-experts mmap -ctk q8_0 -ctv q8_0 \
  -cram 10240 -ub 4096 -b 4096 --reasoning-effort low --n-predict 32768 \
  --temp 0.7 --top-k 20 --top-p 0.95 --min-p 0 \
  --spec-type draft-mtp -md mtp-head-expQ4K.gguf -ngld 99 --spec-draft-n-max 3 --spec-draft-p-min 0 \
  -ctkd q8_0 -ctvd q8_0
```

The environment variables in the compose file are part of the configuration: several of them turn on features
that are off by default (KV lending, the even FFN split, the copy-engine all-reduce, prefill prefetch). Without
them, the numbers above do not apply. The first requests after a start are slower while the expert cache sizes
itself and fills; run a few requests before measuring.

**Benchmark.** With the server up:

```
python3 ours/bench/llbench.py --url http://127.0.0.1:8080 --src <llama.cpp>/ggml/src --src <llama.cpp>/src \
  --src <llama.cpp>/tools --src <llama.cpp>/common --src <llama.cpp>/examples --src <llama.cpp>/tests \
  --depths 0,8192,16384,24576,32768,40960,49152,57344,65536,131072,196608,245760 --out res.json
```

It needs llama-server's `/tokenize` and `/completion` endpoints, so it runs against llama.cpp-based servers. To
compare another engine, the depth/PP/TG definitions above are the protocol; local-llm-benchmarks.dev publishes
results for several engines on R9700 cards with its own runner.

## How the series is organised

- **0001** is everything outside `moe-expert-cache.cu`.
- **0002-0003** are our `--host-experts mmap` expert cache. The release's `moe-expert-cache.cu` stays as released.
  Our code lives in two included files, `moe-expert-tier.inc` and `moe-expert-tier-state.inc`, and runs only for
  sparse (mmap) tables. Pool and pinned tables run the release's own code. The tier code covers the RAM tier, the
  pin budget, KV lending, the fill fence, batched admissions and the device remap for sparse tables.
- **0004-0040** are one feature each, in the order they were added.
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
| 0033 | server: a checkpoint's state copy is queued before the next batch and finished after it, so the host-side copy overlaps GPU work; byte-identical (verify mode compares every deferred copy with a synchronous one), agent turns -11..-12 ms | `LLAMA_CKPT_ASYNC=0` |
| 0034 | meta (tensor split): cheaper split-state misses -- lean source states, states derived between the allocation and compute lookups, and a walk pruned per buffer; agent-turn prompt time -10 %, outputs identical | `GGML_META_SS_LEAN=0`, `GGML_META_SS_DERIVE=0`, `GGML_META_SS_PRUNE_XBUF=0` |
| 0035 | llama: a state copy larger than the pinned stage (`LLAMA_STATE_ASYNC_MAX_MIB`, 256) runs the batched copy in rounds of the stage size instead of tensor by tensor through pageable memory; 350 MiB prompt-cache save 52-68 -> 20-37 ms, its load -26..-32 ms, same bytes | `LLAMA_STATE_ASYNC_ROUNDS=0` |
| 0036 | server: prompt cache -- a fresh large state buffer (64 MiB or more) gets n/16 of extra capacity, so the next saves of a growing conversation reuse it instead of mapping and faulting in a new one; 120-130K-token agent-turn saves 0.33-0.37 s -> 0.12-0.13 s | `LLAMA_SERVER_CACHE_SLACK=0` |
| 0037 | server: KV full stop. With a unified KV cache, a reply that outgrows its admission reserve ends with a length stop when the cache is full, instead of failing the decode of every generating request with HTTP 500 | `LLAMA_SERVER_KV_FULL_STOP=0` |
| 0038 | cuda: queued (async) row-block reads without one copy per row. The device that allocated the pinned destination copies 2-D; another device gathers the rows on the device, copies each 16 MiB chunk once and scatters it from a stream callback. Since r364 these reads were one copy per row, which under `-sm tensor` made a ~2 GB agent prompt-cache save take 17-20 s; now 0.17-0.38 s | `GGML_CUDA_ROWS_ASYNC_GATHER=0` |
| 0039 | server: the max-min prompt share divides the pass's budget, not what the earlier slots of the pass left (two prompts got 1365 + 682 of a 2048 window instead of 1024 + 1024); two concurrent fresh 39K + 34K / 59K + 59K prefills -17 % wall | `LLAMA_SERVER_SHARE_PASS_BUDGET=0` |
| 0040 | cuda: GDN -- a prefill ubatch holding several sequences with snapshot slots (MTP) takes the chunked prefix + sequential tail per sequence, as a single sequence does, instead of the whole sequential kernel; two concurrent fresh prefills -2.5 to -3.5 % wall, single requests unchanged | `GGML_CUDA_GDN_MULTISEQ_TAIL=0` |

## Settings that matter for this series

- `-cram 10240`: 10 GiB of RAM for the prompt cache, so two long-context agents can take turns on the
  GPU and swap their contexts through RAM instead of re-reading their prompts.
- `LLAMA_SERVER_CACHE_HUGEPAGE=0`: 4 KiB pages for the prompt-cache save buffers. With transparent huge pages in
  `madvise` defrag mode and the page cache filling RAM (the mmapped host experts), huge pages made 120-130K-token
  saves stall 2-11 s in direct compaction on our 59 GB host; with 4 KiB pages they take 0.33-0.37 s (0.12-0.13 s
  when 0036 lets them reuse a buffer).
- `LLAMA_SERVER_KV_ADMIT_RESERVE=16384`: the reply reserve per request of the release's KV admission (its default is
  n_ctx / slots = 64K at 256K and 4 slots), so two long prompts still fit side by side.
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

## Validation (r366, 2026-11-26)

- Rebased from r362 onto r366; two conflicts (the slab seed floor in 0001 with the release's wide placement, and the
  KV segment's idle-slot hint beside the release's deferred-task re-offer). Dropped: our KV admission, which the
  release took in r365. New: 0037-0040.
- The 7-model comparison with the release passes (gate: 0 problems, 0 warnings).
- Production configuration: greedy and the 512-token prompt set give the same output as the r362 series; agent turns
  with prompt-cache saves and restores give the same replies, with 0 GPU faults.
- The overnight stress run has not been done on r366 yet.

For earlier releases, see this file's history or the `ours-rNNN` branches.
