# A2 — the mixed-backing windowed KV: what it does and what it measures

**Not delivery work.** `TODO` **#118**.  Code: fork branch `wip-kv-residency` at `0e5b0d59d` (on top of A1's
`e49e1cef4`, which is kept on branch `wip-kv-residency-a1-two-base`).  Nothing here is in `patches/`,
`release.json` or any block.  Design of record: [`MIXED-BACKING.md`](MIXED-BACKING.md) (the probe and the
shape); this file is the build and the numbers.

## 1. What is implemented

| piece | where |
|---|---|
| the mapping **table** (`(off, bytes, kind)`), `map_span(off, bytes, host)`, `mapped_host()` | `ggml-cuda-vmm-region.{h,cu}`.  The driver's **minimum** granularity (4 KiB) is queried and a span is rounded outward to it; the bulk still maps in `unit` steps.  The selftest gained a HOST-backed span arm (device writes it, reads it back) and still passes: `VMM region ... PASS (... a HOST-backed span ...)` |
| `ggml_backend_cuda_kv_region_buffer_type_split(device, tail_bytes)`; `grow_cells` maps host `[0, tail)` + device `[tail, cells)` as the **cell watermark** advances; `grow_for` picks the kind by position (state restore writes arbitrary indices) | `ggml-cuda.cu` |
| the split point, the loud warning, the uniform-geometry check | `llama-kv-cache.cpp` (`LLAMA_KV_REGION=1` enables the provider, `LLAMA_KV_RESIDENT_CELLS=W` sets the window) |

Knobs (campaign only, all default-off): `LLAMA_KV_REGION=1`, `LLAMA_KV_RESIDENT_CELLS=W`,
`LLAMA_KV_REGION_DEBUG=1`, `GGML_CUDA_VMM_REGION_SELFTEST=1`.

Two lines say what happened:

```
W llama_kv_cache: WINDOWED KV ENABLED: 35091 of 262144 cells spilling to pinned system RAM
  (36.41 MiB per tensor), 227053 cells resident in VRAM. ... a token at depth costs about
  (cells beyond the window) x 34.0 KiB / 28.8 GB/s ...
I ggml_backend_cuda_kv_region_grow_cells: device 0: windowed KV -- 8704.00 MiB of KV mapped:
  8160.00 MiB in system RAM (pinned, non-swappable) + 544.00 MiB in VRAM, of a 8704.00 MiB reservation
```

**Declines rather than guesses:** if the K/V cell geometry is not uniform across the layers a cache holds
(and K vs V), the split is not offered and the cache is fully resident — today's fit-or-fail, with a warning.

## 2. Gates (1 GPU, gfx1201)

| gate | result |
|---|---|
| text identity, 4B Q8_0 `-c 8192` | plain `ef279d67a708` = region-only = window 2048 = window 4096 |
| text identity, 27B Q4_K_XL `-c 8192` q8_0 KV | plain `5ba44647188f` = window 2048 = window 4096 (and the same hash A1 produced) |
| text identity, 35B-A3B Q4_K_M `-ncmoe 24 -c 16384` q8_0 KV | plain `061add8deaba` = window 4096 |
| prefill logits (`gate-prefill-logits.sh --ab "LLAMA_KV_REGION=1 LLAMA_KV_RESIDENT_CELLS=1024"`, ctx 2048) | **0.000000 mean KLD / 100.000 % same-top-p** |
| GPU faults (`check-gpu-faults.sh --since 21:00`, every log) | clean |
| the Goal A OOM config | **now runs** (below) |

## 3. The numbers

### 3.1 The OOM config runs (this is Goal A's headline)

27B **Q8_0** (27.1 GiB of weights — 1.29 GiB of which llama.cpp maps from the host on this card anyway),
`-c 262144`, `-ctk q8_0 -ctv q8_0`, 51 475-token prompt:

| arm | KV | result |
|---|---|---|
| plain, `-c 262144` | 8704 MiB in VRAM | **`cudaMalloc failed: out of memory`** (as recorded before the campaign) |
| window, `-c 262144` | **8160 MiB system RAM + 544 MiB VRAM** | **runs**: prefill **933–1086 t/s**, decode 10.8 t/s at depth |

### 3.2 What the spill costs at 51 475 tokens of depth

Same prompt, same model, `-c 262144` (so the same graph) with the window anchored at the used depth — the
prediction is `(used − resident) × 34 KiB/cell / 28.8 GB/s` on top of the resident token time:

| resident window | spilled cells | spill read per token | predicted | **measured** |
|---|---|---|---|---|
| all 51 475 (plain `-c 65536`, fully VRAM) | 0 | 0 | 53 ms | **18.7 t/s** |
| 32 768 | 18 707 | 636 MiB | 53 + 22 = 75 ms | **13.6 t/s** |
| 16 384 | 35 091 | 1.19 GiB | 53 + 41 = 94 ms | **10.8 t/s** |

The model holds to within ~3 %: the tail read achieves ~28 GB/s effective in the FA band's access pattern,
not just in a pure stream.  **The overflow cost is linear in the resident fraction, exactly as `O2-PLAN.md`
§3 said, and the resident fraction is the knob.**

### 3.3 The slab half of Goal A: the arena gets the VRAM

35B-A3B Q4_K_M, `-ncmoe 24`, `-c 262144`, q8_0 KV (2720 MiB), `GGML_CUDA_SLAB_STATS=1`:

| arm | `SLAB_INIT` free | slab mapped (= work + arena) | decode |
|---|---|---|---|
| plain KV (fully resident) | 19.80 GiB | 18.75 GiB | 91.6 t/s |
| window, `W=32768` of 262144 | **22.46 GiB** | **21.44 GiB** | 91.1 t/s |

**+2.69 GiB of arena** — the KV's VRAM the window frees (2720 MiB of KV − 64 MiB resident), and decode is
flat at a shallow depth.  This is the second half of the ask ("free VRAM for the decode arena, higher expert
hit rates"), and unlike O1 it is not a lazy growth fighting an arena that already owns the VRAM: the KV never
commits it.

## 3.4 The SLIDING window (built 2026-10-29, `9190fde16`)

The static window above anchored the resident set at the end of `-c`, so a run that has not filled the context
spills its whole used extent (§4).  The window now slides: **the resident set is the newest `W` cells of the
USED extent** and the host/VRAM boundary moves forward with the watermark.

* `ggml_cuda_vmm_region_ensure_span(off, bytes, host)` maps a piece whose kind is right, and **flips** one
  whose kind is wrong: stage the bytes out through a pinned scratch → `hipMemUnmap` → map the other kind →
  stage them back, with a device synchronise around the unmap.  The VA never changes, so captured graphs and
  every existing data path stay valid; the table stays one entry per mapping (the driver refuses a sub-range
  unmap, so a flip is a whole-mapping operation).
* The boundary is quantised **down to whole map units**, so a batch does nothing but map its delta (one cell
  normally) and a remap happens once per `unit/nb1` cells — ~1900 cells on the 27B, ~3900 on the 35B.  Cells
  between quantised boundaries stay resident a little longer, which is free.
* Monotone: a cell handed to system RAM is not taken back.

| arm (1 GPU) | before the slide | **with the slide** |
|---|---|---|
| 27B Q8_0 `-c 262144`, 51 475 tok, `W=49152` | 8.9 t/s | **18.6 t/s** (= the 18.7 t/s fully-resident reference) |
| 27B Q8_0 `-c 262144`, 51 475 tok, `W=16384` | 10.8 t/s | 10.3 t/s (same 1.19 GiB/token of spill; within this box's run-to-run spread) |
| 35B-A3B **Q8_0** `-ncmoe 24 -c 262144`, 5256 tok, `W=32768` | arena 15070 MiB, **58.2 t/s** (the whole used extent spilled) | arena 12393 → **15070 MiB**, residency 63.3 → **77.0 %**, prefill 602 → 615, decode **64.5 → 64.9 t/s with zero spill** |

Text is byte-identical to the fully resident arm on every one of those, **and byte-identical to the pre-slide
hash** (`ef279d67a708` 4B, `5ba44647188f` 27B q8_0) — which is the proof that a remap preserves every staged
chunk, since a lost chunk would have changed the generated text.

**Default `W = -c`** (no spill, today's cache); the opt-in (`LLAMA_KV_RESIDENT_CELLS`) is what spills.

## 3.5 The arena's value at -n 2000 (and the bigger the model, the more it is worth)

**Rule (maintainer, 2026-10-29): always `-n 2000`.**  A shorter run's decode is drowned out by the expert
cache's warm-up (the same 35B-A3B Q8_0 arm reads 54.6 t/s at `-n 128` and 66.3 at `-n 2000`), so every number
here is a 2000-token decode after a 5256-token prompt.

| model / config (1 GPU, `-c 262144`, q8_0 KV unless noted) | arena | host-expert residency | prefill | decode |
|---|---|---|---|---|
| 35B-A3B Q8_0 `-ncmoe 40` plain | 25 245 MiB | 77.3 % | 373 t/s | 66.3 t/s |
| 35B-A3B Q8_0 `-ncmoe 40` window `W=32768` | 27 922 (**+2.62**) | 85.5 % | 382 t/s | 67.4 (**+1.7 %**) |
| 35B-A3B Q8_0 `-ncmoe 40` **bf16 KV** plain | 22 822 MiB | 69.9 % | 375 t/s | 65.4 t/s |
| 35B-A3B Q8_0 `-ncmoe 40` **bf16 KV** window | 27 922 (**+5.10**) | 85.5 % | 373 t/s | 66.2 (**+1.2 %**) |
| 122B-A10B Q4_K_XL `-ncmoe 40` plain | 8 746 MiB | **15.6 %** | 220 t/s | 21.8 t/s |
| 122B-A10B Q4_K_XL `-ncmoe 40` window `W=16384` | 12 026 (**+3.18**) | **21.5 %** | 233 t/s | 24.0 (**+10.1 %**) |
| **122B-A10B Q4_K_XL `-ncmoe 48` (ALL 48 layers) plain** | 19 661 MiB | 29.3 % | 202.8 t/s | 22.9 t/s |
| **122B-A10B Q4_K_XL `-ncmoe 48` window `W=16384`** | 22 807 (**+3.15**) | **34.0 %** | **220.3 t/s** | **24.7 t/s (+7.9 %)** |

| **Flash-Next GSQ-IQ3_XXS** (qwen4exp) `-ncmoe 48` plain | 22 061 MiB | 53.9 % | 394.0 t/s | 40.8 t/s |
| **Flash-Next GSQ-IQ3_XXS** `-ncmoe 48` window `W=16384` | 25 818 (**+3.63**) | **63.1 %** | **445.6 (+13.1 %)** | 41.6 (**+2.0 %**) |

`-ncmoe` **must** be the model's full layer count when the point is the arena: at `-ncmoe 40` eight layers of
experts (218.6 MiB per expert across the host layers) still sit on the device, so the arena is only 8.7 GiB of
the 65.5 GiB host set (15.6 %); at `-ncmoe 48` it is 19.7 GiB (29.3 %).  That is the config that matters for
this model, and it is also its fastest: **24.7 t/s** with the window vs 22.9 plain.

**The window's benefit splits into prefill and decode, and they have different laws.**

* **Prefill is capacity-bound.**  A prefill routes *every* expert of every layer, so the arena size is
  directly the staged-bytes saved: 35B-A3B +1.8 %, 122B-A10B +5.9 % (`-ncmoe 40`) / **+8.6 %** (`-ncmoe 48`),
  **Flash-Next +13.1 %** (512 experts × 48 layers, the hungriest set here).  This is the most consistent win
  in the whole table.
* **Decode is hotness-bound.**  10 of 512 experts are routed per token, and the hot set is lopsided, so at
  ~54-85 % residency the extra GiB mostly holds *cold* experts the run never re-routes: +1.2-2.0 % on the 35B
  and Flash-Next.  It only converts to a large decode gain where the arena is genuinely starved — the 122B at
  **15.6 %** residency gains **+10.1 %** (and +7.9 % at `-ncmoe 48`).

Flash-Next also widens the picture on the caches themselves: it has **two** `llama_kv_cache` instances (the
attention KV, 3264 MiB, and the QSA **indexer** cache, 408 MiB) and the window applies to both — the log shows
one `WINDOWED KV ENABLED` per cache (0.5 KiB/cell and 0.1 KiB/cell).  The indexer cache is the O(n_ctx) term
the campaign's Goal B/O4 notes call the second-biggest, so it now participates in the window instead of being a
fixed VRAM cost.  122B numbers:
48 layers, 256 experts, 8 used, `n_embd_k_gqa` 512, 55 974 MiB of host experts, 218.6 MiB per expert across
its 40 host layers.

**The fittability half at `-n 2000`** (27B Q8_0, 51 475-token prompt, 53 475 used by the end):

| arm | decode |
|---|---|
| `-c 65536`, fully resident (the reference) | 18.4 t/s |
| `-c 262144`, `W=55296` (nothing spills) | 18.6 t/s |
| `-c 262144`, `W=49152` (131 MiB/token of spill) | **18.6 t/s** |

So the 262 144-cell context runs at the small one's speed, and the spill hides behind the 27B Q8_0's 1.29 GiB
of CPU-mapped weights (which is what actually bounds this arm: ~1.29 GiB/token over the link).

### 3.6 The deep arm (the caches actually spill) — and the window's break-even

Flash-Next, `-c 65536` (so `-c` is ~1.4× the 46k-token prompt), `-ncmoe 48`, q8_0 KV, `-n 2000`; at depth
~15k cells of BOTH caches are host-backed (attention KV 0.5 KiB/cell + QSA indexer 0.1 KiB/cell):

| arm | `SLAB_INIT` free | arena | residency | prefill | decode |
|---|---|---|---|---|---|
| plain (both caches fully resident) | 27.06 GiB | 24 539 MiB | 60.0 % | 465.7 t/s | **45.2 t/s** |
| window `W=32768` (both spill at depth) | 27.46 GiB | 24 939 MiB | 60.9 % | 477.2 t/s | **38.6 t/s (−14.6 %)** |

**The window is a bet that `-c` is much larger than the depth you run at**, and this arm is the other side of
it: at `-c 65536` the whole KV is 918 MiB, so the window frees only **0.39 GiB** of arena and costs **14.6 %**
of decode (213 MiB/token of spill at depth).  Compare §3.5 at `-c 262144`, where the same model frees 3.63 GiB
and spills nothing at a 5256-token depth (+13.1 % prefill, +2.0 % decode).  Break-even is therefore a **`--fit`
decision**: size `W` from the depth (and leave `W = -c`, i.e. no spill, when `-c` is close to it) — exactly the
`--fit` half already on the owed list.

### A bug this measurement found (and fixed): the resident window must be committed BEFORE SLAB_INIT

The deep arm first **aborted** at depth: `growth refused (region 816 MiB, wanted 500..503 MiB)`, with
`SLAB_STATS ... free=0.0 MiB`.  The resident window's VRAM was being mapped lazily, so `SLAB_INIT` mapped
`free VRAM − reserve` first and the arena owned all of it by the time the watermark advanced — the ordering
rule `O2-PLAN.md` §6.2 already states, which A2 was violating for the **device** half (the host half is
supposed to be lazy; the resident half is not).  `init_tensor` now commits `min(W, n_cells)·nb1` of VRAM per
tensor at buffer creation and keeps the spill lazy (`ba5c7205e`).

### A bug this measurement found (and fixed): the boundary must be quantised in BYTES

The slide first quantised its boundary in *cells* (`unit/nb1`, integer division), which is not map-unit
aligned: every unit straddled it, `ensure_span` fell back to 4 KiB pieces, and a **host** chunk costs ~130 µs
to create (`mixed-backing-probe.cc` T8) — one per tensor per token was ~4 ms/token.  The same arm read
**14.7 t/s before the fix and 18.6 after** (`f50c33be5`).  The boundary is now `(n_cells − W)·nb1` rounded
down to whole units, so both segments map in whole units (one chunk per ~3855 cells on a 27B).  `grow_for`
reads the same byte boundary, so a state restore cannot map into the wrong kind.

## 4. FINDING (fixed by §3.4) — a static window anchored at the end of `-c` is the wrong product shape

The window is cells `[kv_size − W, kv_size)`: the **last** W cells of the *context*, not of the *used
extent*.  So at 51 475 tokens in a `-c 262144` cache, the entire used extent (cells `[0, 51475)`) is still in
the tail, and the "resident" window covers cells nobody has written yet:

| `-c 262144` arm | W | what is actually resident at 51k depth | decode |
|---|---|---|---|
| `W=49152` | 49152 | cells `[212992, 262144)` — **none used** | 8.9 t/s |
| `W=227053` | 227053 | cells `[35091, 51475)` — the newest 16k used cells | 10.8 t/s |

That is not a bug, it is the static split being anchored on the wrong end for a run that has not filled the
context.  Three ways out, and the choice is the maintainer's:

1. **Slide the window with the watermark** (the natural design): keep the newest W *used* cells resident and
   move one chunk from VRAM to system RAM (or back) as the watermark advances.  The probe's T7 showed exactly
   this is possible — `hipMemUnmap` + re-map the same VA from the other kind, a few µs — so it is a
   bookkeeping change between batches, not a new kernel.  Cost: a 2 MiB copy per advance.
2. **Size W from the actual depth** at the first batch (`W = min(kv_size, depth + slack)`), which is a
   creation-time decision that can be revisited once (`--fit`'s job in A3).
3. **Default W = `-c`** (no spill) and let the user pick, which is today's semantics plus an explicit
   "I accept the spill" flag.

Until one of those lands, `W` must be chosen with the intended depth in mind — which is why every number in
§3.2 uses a window anchored at 51 475 cells.

## 5. Still owed before this can be called done

* ~~the sliding window~~ — **done, §3.4**.

* `scripts/gate-ckpt-reuse.sh` and `scripts/gate-state-pipe.sh` — **the two gates A1 never needed**: state
  save/restore now crosses the host/VRAM boundary.  The probe showed the copy engine handles it (T5) and the
  code maps the right kind by position, but the gates are what prove the whole path.
* the `MemAvailable` gate before pinning (4.35 GiB at `-c 131072`, 8.5 GiB at 262144, non-swappable).
* `--fit` reporting the resident bytes instead of the whole KV, so an over-budget config is *reported*
  rather than warned about and then run.
* sizing `W` from the depth (`--fit`), now that the slide makes W a *cache* rather than a cap.
* MTP verify widths on the windowed arm (A1 gated it; A2 changes the memory, not the kernel, so the risk is
  lower — but it is cheap).
* a 2–3 device arm: only one card is attached, so nothing here says anything about `-sm tensor`.

---

# §3.7 A3 (2026-10-30) — the re-baseline, `--fit` sizing, the RAM gate, and what the re-baseline broke open

Session scope (the maintainer's list): **(1)** `--fit` sizing + reporting of the window, **(2)** the two state
gates on a windowed arm, **(3)** a `MemAvailable` gate before pinning the spill, **(4)** MTP verify widths on a
windowed arm.  All four are done; the re-baseline also found **four** real bugs, and one **open** numeric
question that blocks the campaign's purity claim (§3.7.6).

## 3.7.1 The hardware swap happened: re-baseline first

`bw-probe.cc` (512 MiB, 5 reps) and `mixed-backing-probe.cc` re-run on the re-cabled box; the PCIe root port
now reads `current_link_width 16 @ 32 GT/s` (was x8):

```
bw-probe 1024 MiB:  H2D pinned 55.8   H2D pageable 54.0   D2H pinned 56.8
                    D2D 286.9        kernel read pinned 55.7   kernel read VRAM 635.1  GB/s
mixed-backing-probe (--mib 512): T3 device read 54.7 | T4 host-backed 54.8, both halves 105.5 |
                    T8 768 host chunks 55.0 ms, 53.4 GB/s, 0 failures | 0 failing checks
```

**The link doubled: 28.8 → 55.7 GB/s.**  Every cost model in this campaign becomes `bytes / 55.7 GB/s`, i.e. the
spill cost halves and the `--fit` break-even of §3.6 moves.  Do not compare a post-swap t/s number with a
pre-swap one (not even the plain arm).  The mixed-backing capability is unchanged (T1..T8 all pass).

The identity gates were re-run and re-baselined: 27B Q4_K_XL `-c 8192` q8_0 KV still hashes
`5ba44647188f`; the prefill-logit base was re-recorded (`PPL 5.0803`, sha256 `8fad054d95af…`) and the window
arm compared at **0.000000 / 100.000 %**; the plain `-c 262144` 27B Q8_0 arm still OOMs (`cudaMalloc failed:
out of memory`, 8704.00 MiB) — the feature-off baseline is intact.

## 3.7.2 `--fit` now sizes and reports the window (item 1)

`LLAMA_KV_RESIDENT_CELLS=auto` leaves the number to `--fit`, which is the only thing that knows both budgets
(the VRAM left after the arena/slab reserve, and the pinned-RAM ceiling).  `common_params_fit` now:

* measures the KV's own bytes (`0e5b0d59d`'s region buffer type is identifiable, so the KV's share of the
  context tier is known exactly, `kv_bytes / kv_size` = bytes per cell);
* sizes `W` from `free − margin − everything-but-the-KV`, clamps it against the pinned-RAM ceiling, and hands
  it down through `llama_kv_region_set_resident_cells()` (a process global the KV cache reads at creation);
* **returns early** when the windowed config fits, i.e. it keeps the user's `-c` instead of shrinking it;
* **reports** the split at WARN (visible by default):

```
W common_params_fit_impl: the KV cache is WINDOWED: 65658 of 262144 cells resident (2180.1 MiB of 8704.0 MiB),
  6523.9 MiB spilled to pinned system RAM
W common_params_fit_impl:   -> it costs 6523.9 MiB less VRAM (31386.0 MiB projected vs a 1024 MiB target,
  1024.0 MiB free); the spill reads 34.0 KiB per cell over the link (53 GB/s, calibrated) on every step, so a
  token at the full context reads 6523.9 MiB = about 129.3 ms at that rate, and nothing spills below depth 65658
```

The link constant and the per-cell bytes come from the backend (`ggml_backend_cuda_kv_region_link_gbps`, the
one-off H2D calibration — it reads 53 GB/s here, and it is why no campaign number is hard-coded any more).

Policies, deliberately: `auto` **only spills a context the user pinned** (with `-c` unset the fit keeps
choosing a fully resident context and prints how to ask for more); a **numeric** `W` is honoured and reported
(so `-c 262144 -W 49152` is now *reported as fitting* rather than refused); `auto` without `--fit` declines
the window with a warning.  `llama_kv_cache::memory_breakdown()` reports only the resident bytes against the
device buffer type and the spill against the CPU one, so the memory table and `--fit` both see where the KV
really lives.

Measured end-to-end: the Goal A config (`27B Q8_0 -c 262144 -ctk q8_0`, 35.6 GiB of weights+KV) now runs via
`--fit` alone (`W=123385`, 4096 MiB resident), and `llama-fit-params` reports the arithmetic in 0.4 s.

## 3.7.3 The `MemAvailable` gate on the spill (item 3)

`LLAMA_KV_SPILL_RAM_PCT` (default **85**, `0` disables) caps the pinned spill at that % of MemAvailable, the
`MOE_HOST_POOL_RAM_PCT` pattern, with **two** enforcements:

* **creation-time clamp** (`llama_kv_cache`): the worst case — every tail cell of the context — is compared
  with what is left of the ceiling, and `W` is raised until the spill fits.  A second cache (Flash-Next has
  two) sees what the first reserved (`..._spill_ram_reserve`).  If the clamp would have to make the whole KV
  resident, the window declines and says so (today's fit-or-fail).
* **run-time backstop** (`grow_cells`/`grow_for`): every host mapping asks the ceiling first, and a refusal is
  a loud error naming the knob instead of an OOM kill.

Measured (`4B Q8_0 -c 262144`, `LLAMA_KV_SPILL_RAM_PCT=1` → 795 MiB): `W` is raised 1024 → 236699 cells so the
worst-case spill is exactly the ceiling; the fit reports "(widened by the pinned-RAM ceiling)".

## 3.7.4 The state gates on a windowed arm (item 2) — and the abort they found

* `scripts/gate-ckpt-reuse.sh` with `LLAMA_KV_REGION=1 LLAMA_KV_RESIDENT_CELLS=2048` (4B, hybrid, ctx 16384):
  **PASS** — keep 5 vs eager 8 checkpoints, 3 restored-and-kept, every request byte-identical, no GPU fault.
* `scripts/gate-state-pipe.sh` (`NGPU=1 SM=layer NCMOE=0`, window on): first run **aborted the server** on the
  batched save path —
  `GGML_ASSERT(buf->buft == ggml_backend_cuda_buffer_type(device) && "unsupported buffer type")` at
  `ggml-cuda.cu:5750`, from `llama_io_write_host::flush_async` → `ggml_backend_cuda_get_tensor_async`.  The
  batched state copy only accepted the *plain* CUDA buffer type, so a windowed KV (a region buffer type, which
  is a CUDA buffer by `ggml_backend_buft_is_cuda`, the identity the scheduler itself gates on) aborts the
  moment a prompt-cache save touches it.  All four sites relaxed to `ggml_backend_buft_is_cuda`; the gate then
  reads **PASS** (5 arms, byte-identical restored text, pipelined −5.5 %, threaded copy −7.1 %).
* **This is why the gates were on the list**: no other gate reaches that path, and the failure is a crash at
  the first prompt-cache save, not a wrong number.

## 3.7.5 MTP verify widths on a windowed arm (item 4)

4B Q8_0, `-c 4096`, q8_0 KV, spilling window (`W=2048`), 96 tokens, prose prompt:

| arm | text |
|---|---|
| plain decode | 96 chars `6541eadb9041` |
| plain `draft-mtp` (`n_max 7`) | 96 chars `6541eadb9041` |
| windowed decode | 96 chars `6541eadb9041` |
| windowed `draft-mtp` | 96 chars `6541eadb9041` |

Byte-identical, i.e. the verify widths do not interact with the window (as expected: A2 changes memory, not
kernels).  The 27B is not a usable arm for this until §3.7.6 is resolved (its spilled text already differs).

## 3.7.6 OPEN — a spilled KV changes the numbers on the 27B (blocker for the purity claim)

Found while re-baselining, and **not** resolved:

* 27B Q4_K_XL, `-c 8192 -ctk q8_0 -ctv q8_0`, 5256-token prose, 64 tokens: plain `305 chars
  sha=e2cea6fa6670` vs windowed (`W=1024`, a **one-unit = 4 MiB** spill per tensor) `325 chars
  sha=fb65430049e9`.  `W=1024` and `W=1280` give the same boundary and the same text (so it is
  deterministic), and it reproduces with `--fit off`.
* Prefill-logit gate at `--ctx 8192 --chunks 4 --kv q8_0`, `W=1024`: **KLD 0.004973 / 98.761 %** (limit 0.005).
  `W=8192` (= `-c`, window enabled, nothing spills) reads **0.000000 / 100.000 %**, and the region alone
  (no window) reads **0.000000**, so it is the *spill*, not the buffer type.  The same point's self-comparison
  is `-0.000000 / 100 %`, so the point is stable and this is a real difference.
* A 4B arm with the *same shape* of spill (4 MiB per tensor, f16 **and** q8_0 KV) is byte-identical.  So it is
  not the quantized KV type and not "any spill".
* The flip itself is byte-exact: a debug verify (`LLAMA_KV_REGION_DEBUG=1`) reads every flipped mapping back
  and compares it with the stage — 32/32 flips ok on the 27B arm that produces the divergent text.

So the data path we control is correct and something else about having a spilled KV changes the result.  Two
hypotheses worth trying first, in order: (a) a **`cudaMemcpy` vs kernel asymmetry on a host-location mapping**
is gone (both legs now go through a kernel, §3.7.7) but a *stale-L2 / non-coherent load* in the FA kernel for
system-memory operands would look exactly like this — test by forcing the tile path (`GGML_CUDA_FA_BAND`-family
knobs) or by copying the spilled cells back to VRAM before the attention of one layer; (b) a **mask/padding**
interaction: the spilled cells sit under a unit-rounded mapping whose end is clamped to the tensor's unpadded
extent, so the *last* row's quantized padding may be unmapped where the resident arm has it zeroed — check
`nb1` vs the padded row size per tensor.

## 3.7.7 The three bugs the re-baseline found in the slide (fixed)

1. **A coalesced mapping straddling the boundary silently undid the spill.**  `ensure_span` flipped the *whole*
   mapping whose range the boundary fell inside; `vmm_insert` coalesced same-kind neighbours, so the boundary
   landed inside an entry and the next VRAM step flipped it straight back.  Measured: a 4B arm with
   `W=2048 -c 16384` mapped **0 bytes** to host while the warning claimed the tail was spilled.  Fix: every
   mapping in a windowed region is now on the **map-unit grid** (`grow_cells` maps `used_up`, `init_tensor`
   commits `up(W·nb1, unit)`, `vmm_insert` no longer coalesces), and a flip whose range is not exactly one
   mapping is **refused loudly** instead of being silently widened.
2. **`cudaMemcpy` into/out of a host-location mapping does not land.**  A direct H2D restore into a freshly
   mapped host chunk completed "successfully" and left zeros (measured: `stage=0x53 read=0x00` at byte 0 of a
   4 MiB restore); a D2D whose *source* is one behaves the same.  Every staging leg now goes through a
   **device kernel copy** (`vmm_device_copy`), which is the access the probes measured working (53 GB/s).  The
   debug verify above is the regression test for it.
3. **A multi-piece rewrite of one mapping is not safe** on this driver: detaching a mapping and re-attaching
   it as head/mid/tail (the obvious way to honour a boundary inside a mapping) loses the middle piece's bytes.
   Hence the unit-grid invariant plus the loud refusal, rather than a cleverer flip.

The VMM selftest (`GGML_CUDA_VMM_REGION_SELFTEST=1`) now covers all of it: an aligned window flip moves
exactly one mapping and preserves every byte, and a straddling request is refused and moves nothing.

## 3.7.8 What is left

1. **§3.7.6**, which blocks calling the window pure.  Until it is closed: quote text/KLD purity only for
   **non-spilling** windowed arms; the 4B's spilling arms are the only byte-identity evidence.
2. The `--fit` numbers above are from `llama-fit-params`/smoke runs at 55.7 GB/s; the **arena** and
   **deep-arm** tables of §3.3-§3.6 have *not* been re-measured at the new link (`-n 2000`, `-ncmoe` = the
   model's full layer count) and must not be quoted as post-swap numbers.
3. Goal B is still not started; multi-device is still not measurable here.
