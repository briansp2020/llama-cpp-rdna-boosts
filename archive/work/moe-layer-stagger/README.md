# `moe-layer-stagger` — interleave the host- and device-resident MoE expert layers to hide H2D upload

> **ARCHIVED 2026-10-15 — CAMPAIGN CLOSED, REFUTED (no code change).**  The stagger hypothesis does
> not work, and its intended win is already covered by the delivery's tail-set host residency.
> * **Layer placement is a no-op.**  With the page-cache confound removed (run from a tmpfs model
>   copy), four disjoint host sets (`0..23`, even, odd, `24..47`) are identical to 0.1 % at `-ncmoe 24`;
>   partial-offload prefill is bound by the H2D of the whole host table per ubatch, so the *schedule*
>   cannot matter — only fewer bytes (`-ub`, a correct gather, residency).  The original "+12 % for
>   stagger" was page-cache warm-up.
> * **The wished-for speedup is now the delivery's own.**  `-ncmoe N` hosts the **last N** trunk layers
>   (rear-fill, **r63**): the tail carries the colder routing and has preceding resident compute to
>   overlap the upload with.  Measured 35B-A3B Q8_0, 2x R9700, `-ncmoe 20`: `pp4096` **4264 vs 4120**,
>   `tg500` **87.5 vs 82.6**, `pp512` **871 vs 635** (`WORKLOG.md` 261014.008).  A per-layer stride
>   generalises nothing on top of that.
> * The chosen basis (`-sm tensor` + `-ncmoe`) corruption found during this campaign was fixed in r62
>   and is unrelated to the stagger question.
>
> Result, data, method and the correctness history: [`FINDINGS-2026-10-14.md`](FINDINGS-2026-10-14.md).
> **Not part of the delivery**; nothing here was promoted (no release change), and nothing may be
> applied to the fork without the maintainer's go-ahead (see `AGENTS.md`).  Seeded from the maintainer's
> observation that the partial-offload split is a contiguous "first N layers to host, the rest to device".
> Resolution: [`RESOLUTION.md`](RESOLUTION.md).

**Headline:** with the page-cache confound removed (run from a tmpfs model copy), four disjoint host
sets (`0..23`, even, odd, `24..47`) are identical to 0.1 % at `-ncmoe 24`; the earlier "+12 % for
stagger" was page-cache warm-up.  Partial-offload prefill is bound by the H2D of the whole host table
per ubatch (~52-67 ms per host layer, over a PCIe **x4** switch link), so the *schedule* cannot matter —
only fewer bytes (`-ub`, a correct gather, residency).  Separately, `-sm tensor -ncmoe > 0` emits
garbage/NaN on 35B-A3B, gemma4 26B-A4B and Flash-Next GSQ unless `GGML_META_SPLIT_COPY=0` or
`-sm layer` is used; this blocks the campaign's tensor-split basis.

## 0. The idea

With partial expert offload (`-ncmoe N`), the host/device assignment is **contiguous**: the first `N`
MoE layers' expert weights live on the host and the remaining layers' experts live on the device.  The
maintainer's proposal: **stagger** the assignment by ratio so that, for a 50:50 split, even layers stay
device-resident and odd layers are host-resident (or vice versa).  The expected benefit is that the
host weights are uploaded **throughout** the prefill rather than in one contiguous host-only stretch, so
the H2D upload of a host layer is overlapped/amortised against the compute of a neighbouring
device-resident layer instead of being exposed as a bandwidth-bound run.

## 1. Current behaviour (evidence)

* `-ncmoe N` builds one CPU buffer-type override per layer index `i` in `0..N-1`:
  `llm_add_n_cpu_ffn_overrides` (`common/common.h:1274`) emits `blk\.<i>\.ffn_(up|down|gate|gate_up)_(ch|)exps`
  (`LLM_FFN_EXPS_REGEX`, `common/common.h:1262`), i.e. **contiguous first N**.
* The loader matches those patterns against each expert tensor name and assigns the CPU buffer type
  (`src/llama-model-loader.cpp:1316-1360`, `moe_host_expert_account` at `:1164`).
* The rest of the layer (attention, norms, shared expert, router) is still on the device (`-ngl`); only
  the routed expert weights move.  The compute for a host-expert layer still runs on the GPU, reading
  the expert bytes uploaded on demand through the op-offload H2D ring (block 06/09).
* There is **no stride/ratio knob**: `-ncmoe` takes an integer count only, and the pattern generator is a
  plain `0..N-1` loop.  So even and odd layers cannot be selected today.

## 2. Why it might help (the hypothesis)

The H2D staging ring already overlaps a layer's expert upload with compute (`GGML_SCHED_STAGE`,
`sched_stage_issue`, the slab ring region).  The hypothesis is that **where** the host layers sit in the
layer sequence still matters:

* **Contiguous (today):** layers `0..N-1` carry all the host bytes.  For that stretch the graph is
  upload-heavy; the device-resident layers come later with no upload to hide against.  If the
  per-layer upload exceeds that layer's compute (bandwidth-bound), the stretch runs at the link rate.
* **Staggered:** every layer has some host bytes, so every layer has an upload to hide behind its own
  (or its neighbour's) compute.  The same total bytes are moved, but spread across the whole prefill.

This is a prefill-overlap argument, not a decode one (decode moves only the routed experts, which is
small).  It is also **not** the same as the ring depth: the ring hides one layer's upload behind the
previous layer's compute; the stagger changes the per-layer upload/compute balance.

## 3. Design options

1. **A stride knob.**  `--ncmoe-stride S` (default 1 = contiguous), so `-ncmoe 24 --ncmoe-stride 2` on a
   48-layer model keeps 24 layers on the host, one every other layer.  Emit the pattern from a stride
   loop (`i = offset; i < n_layer; i += S`) instead of `0..N-1`.  Smallest, most explicit change; needs
   a count-vs-stride semantics decision (which layers are host when `N` is not `n_layer/S`).
2. **A regex/pattern escape hatch.**  Allow `-ncmoe` to accept a pattern (e.g. `odd`, `2/3h`) and pass it
   straight to the override list.  More flexible, but a new user-facing mini-language.
3. **A ratio flag.**  `--moe-host-ratio 0.5` selects `round(ratio * n_layer)` layers spread evenly
   (largest-remainder / Bresenham).  Closest to the maintainer's "according to the ratio" framing, and
   the only one that generalises to non-integer splits cleanly.

Recommendation: **option 3 as the user-facing knob, implemented via option 1's per-layer emission**
(a Bresenham/even-spread selection).  It reads as "keep 50 % of the MoE layers on the host, spread out".

## 4. What to measure

DoD: on a config where the host bytes are a large fraction of a layer's expert bytes and the prefill is
bandwidth-bound, a staggered split beats the contiguous split at the same host byte count.

* Sweep the host fraction (e.g. 25 / 50 / 75 %) at a fixed total, contiguous vs staggered, on
  `-ub 2048` and `-ub 4096`, `llama-bench pp512/pp4096` and a real 5-8k prompt.
* Record the **per-layer upload/compute balance** (the ring's per-slot bytes vs the layer's compute
  time) — if every layer is compute-bound the stagger should be a no-op, and that is a valid result.
* Watch the transient VRAM: a staggered split may change the per-device compute-buffer peak (the
  scheduler's splits follow the host/device boundary), and the slab reserve/boundary interaction is
  `-sm layer`-sensitive.

## 5. Risks / gotchas

* **Purity.**  Which layers take the host-expert path changes which kernels run, and the host-expert
  path is not bit-identical to the device-resident path.  So a staggered config's output is **not**
  expected to equal the contiguous config's; the gate is coherence *within* the staggered config and the
  existing `plain == draft-mtp` purity for it, not cross-config identity.  Document this up front.
* **Layer-count semantics.**  `-ncmoe` currently means "the first N layers"; a stride/ratio makes the
  *set* the contract, and the help text, `--fit` accounting (`llama_model_moe_cache_preflight`'s
  host-byte map) and the MTP draft's `-ncmoed` all read it.
* **Scheduler splits.**  The host/device boundary is where the scheduler inserts the offload split; a
  staggered assignment alternates that boundary per layer, which the block-06 split logic must handle
  (more, smaller splits).  That is the first thing to verify (a correctness/ABI check, not just perf).
* **Shared/other tensors.**  Only the routed-expert tensors move; the draft context's
  `--spec-draft-n-cpu-moe` (`-ncmoed`) has the same shape and would want the same knob.

## 6. Pointers

* `common/common.h:1262-1283` — `LLM_FFN_EXPS_REGEX`, `llm_ffn_block_regex`,
  `llm_add_n_cpu_ffn_overrides`.
* `common/arg.cpp:2783` (`-ncmoe`), `:4162` (`-ncmoed`).
* `src/llama-model-loader.cpp:1164` (`moe_host_expert_account`), `:1316-1360` (override matching),
  `:1481`/`:1523` (the host-expert byte map).
* `ggml/src/ggml-backend.cpp:2365` (`sched_stage_issue`), the slab ring region
  (`ggml/src/ggml-cuda/ggml-cuda-slab.cu`), `patches/README.md` block 09.
* Related campaigns: [`../moe-cpu-overlap/`](../moe-cpu-overlap/README.md) (the CPU-vs-GPU overlap
  question — a different axis: it overlaps CPU *compute* of misses; this one overlaps the H2D *upload*
  against device compute), [`../moe-heatmap-persistence/`](../moe-heatmap-persistence/README.md) (the
  expert cache is orthogonal: it decides which *experts* are resident, not which *layers* are host).
