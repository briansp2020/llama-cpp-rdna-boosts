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

## F3. The "different numerics" is a near-tie flip driven by expert-cache eviction — NOT ring corruption

Reproduce (field model, `-sm tensor -ncmoe 48`, 32 358-token prompt, greedy, `--spec-type none`):

| config | output |
|---|---|
| ring-in-slab, cache ON | `cdd73eb8e3ea` |
| ring-outside, cache ON | `ef91112ab636` |
| cache OFF (either placement) | `ef91112ab636` |
| ring-in-slab, `SCHED_STAGE=0` | `ef91112ab636` |
| ring-outside, `SCHED_STAGE=0` | `ef91112ab636` |

With `GGML_CUDA_ALLREDUCE=ce` this is deterministic per config (the raw default all-reduce is
*not* — it adds run-to-run noise and confounded an earlier comparison).

**It is not corruption.**  No NaN, no `////`, and the **prefill logits are bit-identical** between the two
placements (KLD `0.000000`, same-top-p 100 %) — verified on the field model at `-ctx 512/-ub 512`,
`-ctx 2048/-ub 2048` (staging engaged: the width gate is 1542) and `-ctx 12288/-ub 6144`.  The generated
stream is identical for the first 528 chars (the doc copy); divergence is a **near-tie at the first novel
token** — `"We need answer user."` vs `"We need answer user's request."`.

**Mechanism.**  The ring region occupies slab space for the whole prefill, so the arena is smaller and the
expert cache evicts far more:

| | ring-in-slab | ring-outside |
|---|---:|---:|
| `moe_cache_evict_slab_range` | **8514 + 9108 MiB** | 2442 + 2970 MiB |
| tables re-armed | **107** | 32 |
| arena after | 29.8 GiB | 41.9 GiB |
| prefill hit rate | 64.8 % | 72.3 % |

An expert served from the **cache (device)** is bit-identical to all-VRAM (GREEDY-PURITY §39); an expert
served from the **host** (`-ncmoe` `copy_experts`) is not.  Eviction therefore changes the per-token
device/host mix, and a near-tie flips it.  It is **pre-existing, not ring-specific**: on the *ring-outside*
arm a forced smaller cache already changes the text (`MOE_EXPERT_CACHE_MIB=4096` → `233382632053` vs the
cache-off/host result `ef91112ab636`), and the ring-in-slab merely moves the same knob.

The explicit kernel selectors are **not** involved: `GGML_MOE_CACHE_STAGE=0` and
`GGML_MOE_CACHE_INPLACE=0` leave both placements unchanged.  `SCHED_STAGE=0` makes the arms agree, but it
also removes the ring region from the slab, so it is consistent with the arena/eviction explanation.

### Consequences for the campaign

* The ring is numerically **safe** — it does not perturb the prefill logits; it shifts cache residency.
* The campaign's "0 NaN, acceptance 1.0" on the degenerate prompt **hid** this (every arm produced
  identical text because the copy is saturated).  Future field records must name the prompt **and its
  acceptance**.
* **New, separate finding for the maintainer:** with `-ncmoe`, device-served and host-served experts are
  not bit-identical, so *any* arena change (the r38 G4 cap's refusal, the cache reserve, the ring region)
  can flip a near-tie.  This is the real purity gap; the ring only exposes it.  Worth its own `TODO.md`
  item and a decision on whether the field's purity claims should be stated per-residency.

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
