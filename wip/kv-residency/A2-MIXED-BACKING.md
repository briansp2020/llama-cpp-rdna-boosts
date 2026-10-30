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
