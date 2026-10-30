# Mixed backing — Goal A's A2, and the design of record since 2026-10-29

**Not delivery work.** `TODO` **#118**. Code: fork branch `wip-kv-residency` in `~/llama.cpp`; nothing here is in
`patches/`, `release.json` or any block.

This file is the A2 design. It exists because a **probe changed the choice**: A1 built a two-base K/V read
inside the FA kernels (kept, as a record and a fallback, at [`A1-TWO-BASE-KV.md`](A1-TWO-BASE-KV.md)), and the
mixed-backing probe below then showed that the cheaper route — *one* KV tensor with part of its VA backed by
pinned host memory — is available on this driver. That route needs **no kernel change and no op change**, and
it leaves every cell-walking path in the tree untouched.

## 1. The capability (measured, `mixed-backing-probe.cc`)

Build (the `-isystem` is mandatory — otherwise HIP resolves from Fedora's ROCm 7.1 in `/usr/include`):

```bash
hipcc -O3 --offload-arch=gfx1201 -isystem /opt/rocm-10.1.0-gfx120X/include \
      wip/kv-residency/mixed-backing-probe.cc -o /tmp/mixed-probe
HIP_VISIBLE_DEVICES=0 /tmp/mixed-probe [--mib N] [--t8-mib N]
```

gfx1201 / ROCm 10.1.0 / 1× R9700, `pageableAccess=0 directManaged=0` (XNACK off, as recorded), 0 failing
checks at every size tried:

| # | question | answer |
|---|---|---|
| T1 | granularity for a **host-location** `hipMemAllocationProp` | `Host` and `HostNuma` both ok: min **4 KiB**, recommended **2 MiB** |
| T2 | `hipMemCreate(Pinned, hipMemLocationTypeHost)` mapped into a **device** VA reservation, `hipMemSetAccess` device rw | **ok** (`Host`, id 0 — no NUMA variant needed) |
| T3 | device **read** of the host-backed chunk | **28.8 / 28.3 GB/s** (128 MiB / 1 GiB) — the campaign's link constant, confirmed in situ |
| T3 | device **write** (the KV `set_rows` path), D2H read, H2D write + device readback | all ok, values correct in both directions |
| T4 | **one VA range**, `[0,s)` host + `[s,N)` VRAM, one kernel across both | ok; **host 28.4, VRAM 494.1 GB/s**, a single pass over both 55.3 GB/s |
| T5 | copy engine (`hipMemcpy2D`, both directions) into/out of the host chunk | **ok** — `state_write`/`state_read` needs no kernel fallback |
| T6 | fresh host chunk zeroed? | all zero (device chunks were **stale** per `VMM-PROBE.md` T8 — still clear-on-map) |
| T7 | `hipMemUnmap` the host chunk, remap the same VA **from VRAM** | **ok and usable** — the window can *move*, not only be static |
| T8 | the **real shape**: thousands of 2 MiB chunks + a tensor boundary that is *not* a unit multiple (34 KiB/cell ⇒ the boundary lands mid-chunk) | 2048 MiB: 768 chunks (767×2 MiB + 1×2024 KiB), 0 failed, **55 ms**, 28.2 GB/s across all of it, boundary chunk readable.  8704 MiB: **3264 chunks, 0 failed, 451 ms, 26.9 GB/s**, host memory returned on unmap |

Consequences worth stating plainly:

* **28.4 GB/s for the tail read** means `O2-PLAN.md` §3's cost model stands as *measured*, not modelled:
  `(used − W) × bytes_per_cell / 28.8 GB/s` per token.
* **A window can slide** (T7) — a 2 MiB chunk remap is a few µs, so A3's "as much as fits" does not need a
  rebuild.  A2 keeps it static anyway (the split is decided before `SLAB_INIT`, `O2-PLAN.md` §6.2).
* **VRAM saving is automatic**: uncommitted VA costs nothing, so the arena gets the tail's VRAM at `SLAB_INIT`
  by construction.  This is what makes mixed backing *not* O1: O1 grew lazily **after** the arena had taken
  the VRAM (its `KNOWN LIMIT`), while here the tail's VRAM is never committed at all.
* **Creation cost**: ~138 µs per 2 MiB host chunk at scale ⇒ **~0.45 s** for a 6.4 GiB host tail (3264 chunks),
  against a ~10 s load of the 27B.  Acceptable; record it whenever it changes.

## 2. The shape

```
one ggml buffer, one VA range (the KV cache's tensors laid out consecutively, as today)
  per tensor:  [ tail: cells [0,s) .......... host, pinned ] [ head: cells [s,n_cells) .. VRAM ]
                2 MiB host chunks + <=2 small chunks at the per-tensor boundary
```

* The **split point `s`** comes from `llama_kv_cache::get_kv_split_n_tail(n_kv)` (the policy A1 introduced,
  `LLAMA_KV_RESIDENT_CELLS=W`), and is decided **before `SLAB_INIT`** so the arena receives the remainder.
* **Per-tensor boundary alignment.** Each tensor's tail is `s · cell_stride` bytes (34 KiB/cell on the 27B at
  q8_0), which is generally not a 2 MiB multiple, so the bulk is mapped at 2 MiB and the boundary is expressed
  as ≤2 extra small mappings at the 4 KiB minimum (T8).  ~64 extra mappings on a 27B next to ~3300 bulk ones.
* **Nothing else in the tree changes**: the `set_rows` writers, the in-place rope shift that backs
  defrag/`seq_add`, `state_write`/`state_read` (T5), `clear`, `seq_rm`, `--fit`, the kq mask and the FA
  kernels all address the same one VA range.  That is the whole point of choosing this over a second tensor.
* `llama_kv_cache` keeps one **fallback**: if the driver refuses any part of the mapping, the cache falls back
  to the plain device buffer for that layer (no spill, today's fit-or-fail) and warns once — the same fail-soft
  shape O1 slice 2 already has.

## 3. What A2 has to build (in order)

1. **The buffer type**: extend `ggml_backend_cuda_kv_region_buffer_type` (O1 slice 2, on this branch) from
   "reserve VA, grow device chunks lazily" to "reserve VA, map `[host tail | device head]` up front, per
   tensor".  Reuse its registered proc-address selection, its `is_cuda == true` requirement (otherwise the
   scheduler migrates the whole cache to the host every graph), its plain-buft fallback and its
   `memory_breakdown`.  Drop the lazy growth: there is nothing to grow.
2. **The split point through the buffer allocation**: `llama_kv_cache` must know `s` when it allocates, i.e.
   from the explicit knob first (`--kv-resident-mib` / `LLAMA_KV_RESIDENT_CELLS`), auto-sizing later (A3).
3. **`clear` walks spans**, not the whole range (a full-range memset over a 6.4 GiB host tail is ~0.25 s at
   the measured write bandwidth, and the tail holds no live data at creation).
4. **`--fit` reports the split**: the buffer type reports the *resident* bytes, so an over-budget KV is
   reported rather than refused (A3's `--fit` half — but the number has to come from the buft, so it belongs
   to A2's shape).
5. **Gates** (the A1 list, plus the ones mixed backing newly needs):
   * same-seed text byte-identical at several forced window widths (27B q8_0, 27B f16 + tile,
     4B default, 27B MTP verify widths 2..4) — the negative control is now "mis-map one 2 MiB chunk (or
     point it at the wrong cells) and require the text to move";
   * `scripts/gate-prefill-logits.sh` → **0.000000 / 100.000 %**;
   * `scripts/gate-ckpt-reuse.sh` and `scripts/gate-state-pipe.sh` — **new here**: state io now crosses the
     boundary (T5 says the copy engine handles it, but the gates are what proves the whole path);
   * `scripts/check-gpu-faults.sh` over every log;
   * the two Goal A OOM configs now **run** (that is the point), at the measured `(used−W)/28.8 GB/s` cost;
   * `SLAB_INIT` shows the arena gaining exactly what the window frees;
   * the `MemAvailable` gate: the tail is pinned, so it is non-swappable RAM (4.35 GiB at `-c 131072`
     q8_0 on the 27B, 8.5 GiB at 262144) — refuse loudly before `cudaHostAlloc` fails
     (`MOE_HOST_POOL_RAM_PCT` pattern).

## 4. Residual risks

| risk | status |
|---|---|
| mapping **count** at real size for *host* handles | **measured**: 3264 chunks, 0 failures (T8); `VMM-PROBE.md` T6 took 5900 *device* mappings without hazard |
| `hipMemCreate` **handle count** limit | **measured** at 3264 host handles (T8); O1's T3 took 5900 device handles / 24 GiB |
| device **write** into host-backed VA (the KV writer) + visibility | **measured** (T3/T8) |
| copy engine across the boundary (state io, defrag) | **measured** (T5); gated by `gate-ckpt-reuse.sh` / `gate-state-pipe.sh` in A2 |
| concurrent read/write coherence (decode writes the resident cell while reading the tail) | different chunks, no overlap; unmeasured, and the first thing to watch in A2 |
| `ggml_backend_buffer_is_cuda` / scheduler migration | O1 slice 2 already solved it (`is_cuda == true`, proc-address selection) |
| a **moved** window (T7 says it works) | not needed by A2; A3 may use it.  A sliding window is a *different* correctness question (it changes which cells are resident *between* steps) and is not assumed anywhere |
