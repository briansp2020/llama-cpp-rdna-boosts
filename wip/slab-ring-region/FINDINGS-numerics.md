# `slab-ring-region` — numerics investigation (2026-10-10)

Two findings, both surfaced by the maintainer's questions.  Neither blocks the campaign conclusion
(prefill 946→992 vs 781; decode unchanged), but both matter for how the field is measured and for purity.

## F1. The §5.5 field prompt is degenerate (acceptance 1.0 is a copy artifact)

`/tmp/srr/prose30k.txt` is `prompts/prose-rdna-boosts.txt` **×6** — the repo's own README repeated
(repetition **6.00× / 1 unique line**).  The 30k-run output is **2989 chars and a verbatim substring of
the prompt** (`copy=True`); the served text is the model regenerating the document it just read.  The MTP
head predicts it exactly: `draft acceptance = 1.00000`, `#acc rate/pos = (1.000, 0.996, 0.992)`.

**MTP is on** (`--spec-type draft-mtp --spec-draft-n-max 3` + the shared Q8_0 MTP head).  The `1.0` is
the workload, not the absence of speculation.

The same repetition also **inflates prefill**: repeated text routes every token to the same few experts
(dense-like MoE GEMMs) at ~1100 t/s/ubatch, while a realistic prompt spreads them (~600 t/s/ubatch).
Expert-cache hit rate is unchanged (93.7 % vs 93.4 %), so it is routing diversity, not cache misses.
Cross-arm comparisons are still valid (every arm used this prompt), but the absolute figures are a
ceiling.

**Fix:** measure on a non-repetitive prompt.  A stopgap built from four *distinct* docs
(`README.md` + `ENVIRONMENT.md` + `CONTAINERS.md` + `archive/docs/baseline-history.md`, 99 174 B,
sha256 `69624f4d207f40bdf7357e74ec978af5bfb4725a0968faf4eea4aab46c2ae241`, 1129 lines / 1062 unique =
1.06×) gives acceptance **0.91**, `copy=False`, and a realistic prefill.
The real fix is a new hash-stable prompt under `prompts/` (never edit a shipped one in place).

## F2. The cold-run prefill anomaly is lazy PLE / host-expert reads from disk

The first mixed-prompt run reported prefill **563**; the identical repeat gave **968** (same acceptance,
so the same compute).  Not contention and not thermals: the model's **PLE** (`per_layer_token_embd`) and
the `-ncmoe 48` host experts are read lazily from disk (`llama-model-loader.h`: "keep PLE / engrams embd
tensors on disk, read them on demand"), so the first pass after a config change pays the read.  Any field
number that follows a change of prompt/config must be discarded unless it is a warm repeat.

## F3. The "different numerics" is the **fusion guard toggling**, not a wrong expert

**No wrong expert.**  The cache is transparent to its own residency: with the ring off, cache
`8192` vs `65536` MiB/device at the cache band (`-ub 4`) gives **mean KLD `0.000000` / same-top-p
100 %**.  The same expert is served whatever is resident.

**The ring's effect is a kernel-path switch.**  `moe_cache_take_over` does not fetch "a host expert": it
redirects the weight tensor to the arena and remaps the ids to slots, so the same kernel runs on the same
data.  What changes is the **cache-aware fusion guard**: the slab hands arena chunks to the work pool,
evicting the tables there, and `moe-expert-cache.cu` says so itself — *"the movable-boundary slab evicts
the tables in the chunks it hands to the work pool, so a PARTIAL cache is the normal state … the only
global decision is the fusion guard (`moe_cache_has_arena()`), which stands the cache-aware fusions
down"*.  The ring-in-slab occupies slab space for the whole prefill, so it evicts far more and the guard
toggles more:

| | ring-in-slab | ring-outside |
|---|---:|---:|
| `moe_cache_evict_slab_range` | **8514 + 9108 MiB** | 2442 + 2970 MiB |
| tables re-armed | **107** | 32 |
| arena after | 29.8 GiB | 41.9 GiB |

**Proof it is the fusion path:** pinning the fusions off makes the two placements identical.

| config (field model, greedy, `--spec-type none`, mixed prompt) | ring-in-slab | ring-outside |
|---|---|---|
| default (fusion on) | `cdd73eb8e3ea` / `233382632053` (per region size) | `ef91112ab636` |
| `GGML_CUDA_DISABLE_FUSION=1` | **`a702e03f374e`** | **`a702e03f374e`** |
| `GGML_CUDA_DISABLE_SHEXP_DOWN_GATE=1` | `233382632053` | `ef91112ab636` |
| cache off (`MOE_EXPERT_CACHE_MIB=0`) | `ef91112ab636` | `ef91112ab636` |

`GGML_MOE_CACHE_STAGE=0` and `GGML_MOE_CACHE_INPLACE=0` leave it unchanged.  `SCHED_STAGE=0` makes the
arms agree but also removes the ring region, consistent with the eviction explanation.

**What it is not.**  No corruption: 0 NaN, 0 `////`, coherent text, and the prefill logits are
bit-identical (KLD `0.000000`) everywhere the eviction does not fire (field model, `-ub 512`, `-ub 2048`
— the staging gate is 1542 — and `-ub 6144` at `-ctx 12288`).  The generated stream is identical for the
first 528 chars; the divergence is a **near-tie at the first novel token** (`"We need answer user."` vs
`"We need answer user's request."`).

### The two real findings this exposed (both pre-existing, neither ring-specific)

1. **The cache-path ↔ no-cache-path gap is material.**  `MOE_EXPERT_CACHE_MIB=0` (the CPU-expert
   fallback) vs the default GPU-cache path at the cache band (`-ub 4`, `-ctx 1024 --chunks 6`) gives
   **mean KLD `0.0259` / same-top-p `96.2 %` — a FAIL of the #113 gate** (limit 0.005 / 98 %).  In kind
   this is the cache doing its job (the fallback computes those layers on the CPU); the magnitude is the
   number to know before anyone treats the cache as a pure perf switch.
2. **The cache-aware fusion guard toggles with the slab's eviction state, and fused ≠ non-fused at
   rounding level.**  A mid-run toggle (any arena change: the r38 G4 refusal, a reserve change, the ring
   region) makes the output depend on the *timing* of the eviction.  The code already carries an
   **OPEN** marker for the related invariant: *"OPEN 2: the wholesale-fallback invariant … must hold for
   EVERY consumer, not just the fusion guard and the take-over hook."*

### Consequence for the campaign

* The ring is numerically **safe**: it does not change the experts and does not perturb the prefill logits;
  it shifts cache residency, which toggles a fusion path the delivery already had.
* The campaign's "0 NaN, acceptance 1.0" on the degenerate prompt **hid** this because the copy is
  saturated.  Future field records must name the prompt **and** its acceptance.
* The two findings above belong in their own `TODO.md` items; they are not this campaign's to fix.

## Reproduce

```bash
M=/llm/models/Qwen3.8/Flash-Next/IQ4_NL/Qwen3.8-Flash-Next-IQ4_NL-PROJFIX-00001-of-00009.gguf
B="HIP_VISIBLE_DEVICES=0,1 LD_LIBRARY_PATH=/opt/rocm-7.14-gfx120X/lib"
CLI=~/llama.cpp/build-rocm-hybrid/bin/llama-cli
run() { env $1 $B $CLI -m $M -sm tensor -ncmoe 48 -ctk q8_0 -ctv q8_0 -fa on -t 8 -c 40960 \
  -b 6144 -ub 6144 -f /tmp/srr/mixed30k.txt -n 16 --seed 42 --temp 0 \
  --no-display-prompt --single-turn; }
GGML_CUDA_ALLREDUCE=ce run                       # cdd73eb8e3ea  (ring in slab)
GGML_CUDA_ALLREDUCE=ce GGML_CUDA_SLAB_RING_MIB=0 run   # ef91112ab636 (ring outside)
GGML_CUDA_ALLREDUCE=ce GGML_MOE_CACHE_MIB=4096 GGML_CUDA_SLAB_RING_MIB=0 run  # 233382632053
```

Prefill-logit A/B (the gate's `--ab` records the fallback arm and compares the default):

```bash
LLAMA_PPL=/tmp/srr/ppl-field2.sh ./scripts/gate-prefill-logits.sh \
  --model "$M" --device 0,1 --ctx 2048 --chunks 10 --base /tmp/field.kld \
  --ab "GGML_CUDA_SLAB_RING_MIB=0" --keep      # mean KLD 0.000000, top-p 100 %
```
(`ppl-field2.sh` = `llama-perplexity -sm tensor -mg 0 -ncmoe 48 -b 2048 -ub 2048`; `-ub 2048` is required
to clear the 1542-token staging width gate — with the default `-ub 512` the ring is never used and the A/B
is vacuous.)
