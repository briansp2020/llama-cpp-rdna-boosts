# HANDOVER — `wip/kv-residency` (TODO **#118**), cold-session start

**Rewritten 2026-10-29 at the end of the sixth session.**  Read this top to bottom before touching anything;
it is self-contained.  Then read [`A2-MIXED-BACKING.md`](A2-MIXED-BACKING.md) (the measured state of the
build) and [`MIXED-BACKING.md`](MIXED-BACKING.md) (the design + the driver probe).

> **The one thing to know:** the ask is a **windowed KV** — Goal A, fittability on one 32 GiB card — and it is
> **built and measured**.  A 27B Q8_0 at `-c 262144` that used to OOM now runs, the spill cost matches
> `(used − resident) × cell_bytes / H2D bandwidth` to a few percent, and slab mode's expert arena gains
> ~3-5 GiB which is worth **+7.9 to +10.1 % decode on the models whose arena is starved** and **+8.6 to
> +13.1 % prefill everywhere**.  What is left is **`--fit` sizing** (the break-even decision) plus three
> robustness/correctness gates.  See §7.
>
> **And: the box is about to change.**  The maintainer is re-doing the PCIe layout for ~**56 GB/s** H2D (from
> 28.8) and a faster model SSD.  Every number in this campaign is at 28.8 GB/s — see §9 before quoting any of
> them again.

---

## 1. State in one screen

| what | where |
|---|---|
| delivery repo | `~/llama-cpp-rdna-boosts`, branch `main`, **pushed** (`origin` = `git@github.com:stew675/llama-cpp-rdna-boosts.git`) |
| delivery release | **r176** (`release.json`) — the campaign is cut against **r174** (`v15-6050b467c-r174`, base `6050b467c`, tip `5bdec04ab`); a promotion re-cuts against the then-current release |
| the campaign's TODO handle | **#118** (`TODO.md` §118 + the table row; the older text says `#117`, which was a dangling reference — see the number note there) |
| campaign code | **fork branch `wip-kv-residency`** in `~/llama.cpp`, **pushed to the personal fork** `git@github.com:stew675/llama.cpp.git`, tip **`ba5c7205e`** |
| the A1 fallback | fork branch `wip-kv-residency-a1-two-base` (`e49e1cef4`), also pushed |
| `rdna-boosts` | **never touched by this campaign.**  Locally `5bdec04ab`; the fork's branch is at r176's tip (moved by the r176 release) |
| patch snapshot | `patches/0001`…`0008` + `patches/README.md` + `patches/diffstat.txt` (2024 insertions, 16 files) — apply **in order** to `5bdec04ab` |
| box | **one R9700 (gfx1201), 32 624 MiB.**  Pin `HIP_VISIBLE_DEVICES=0` on every arm.  Every number here is 1-GPU; say so, never claim multi-GPU |

## 2. What Goal A asked for (the maintainer's words, unchanged)

> a **Windowed KV** … for **fully resident dense/MoE models**, because doing so would allow users to run large
> contexts with a Q8_0 quantization of Qwen3.8-27B, or a Q6_K quantization of 35B-A3B, on a 32 GB card, and
> **not run out of VRAM** … and for **host-resident MoE's (slab mode)**, because it would free up more VRAM
> for the decode arena, allowing for higher expert hit rates, leading to higher decode speeds.

**Opt-in semantics (agreed):** default **off** = byte-identical to today including the OOM; the user opts in
to "the KV's tail may spill into system RAM"; the warning is loud and always printed; `--fit` reports the split
instead of refusing.

## 3. The mechanism as built (do not re-derive)

**One KV tensor, one VA reservation, mixed backing.**  The cells below the window are backed by **pinned system
RAM**, the window by **VRAM**, in the same VA range.  That is why there is **no kernel change and no op change**:
the `set_rows` writers, the in-place rope shift behind defrag/`seq_add`, `state_write`/`state_read`, `clear`, the
kq mask and the FA kernels all keep addressing one contiguous range.  (A1 — a two-base read inside the FA
kernels — was built, gated byte-identical, and **superseded**; kept on its branch as the fallback.)

| piece | where |
|---|---|
| the mapping **table** `(off, bytes, kind)`; `map_span(off, bytes, host)`; `ensure_span(...)` which **flips** a mapping's kind through a pinned scratch; `mapped_host()`; the minimum-granularity query | `ggml/src/ggml-cuda/ggml-cuda-vmm-region.{h,cu}` (extends O1's slice 1/2) |
| `ggml_backend_cuda_kv_region_buffer_type_split(dev, cells)`; `grow_cells` (host + device as the watermark advances); `grow_for` (arbitrary-index writes, state restore); the INFO line that reports RAM vs VRAM bytes | `ggml/src/ggml-cuda/ggml-cuda.cu` |
| the window decision (`LLAMA_KV_RESIDENT_CELLS=W`), the loud warning, the uniform-geometry decline | `src/llama-kv-cache.cpp` |
| the A1 policy half (still used) | `llama_kv_cache::get_kv_split_n_tail` |

**The invariants that cost real time to find — keep them:**

1. **The resident window's VRAM must be committed *before* `SLAB_INIT`** (`init_tensor`).  The slab maps
   `free VRAM − reserve`; a lazily-mapped window has no physical left by the time the watermark advances and the
   batch **aborts** (`growth refused … free=0.0 MiB`, measured on Flash-Next).  The **host** half is meant to be
   lazy (do not pin 7 GiB for a 50k-token run); the resident half is not.  (`ba5c7205e`)
2. **The slide boundary is quantised in BYTES to a whole map unit**, never in cells.  `unit/nb1` is not
   unit-aligned, so a cell-quantised boundary makes every unit straddle it and creates a 4 KiB **host** chunk per
   tensor per token (~130 µs each ≈ 4 ms/token: **14.7 → 18.6 t/s**).  (`f50c33be5`)
3. **The slide is monotone** (a cell handed to system RAM is not taken back) and flips the trailing unit to host
   *before* mapping a new one above, so the window's steady-state VRAM is exactly W cells.
4. **One W for every tensor** ⇒ the K/V cell geometry must be uniform across the layers a cache holds; anything
   else **declines** the window (warn once, no spill, today's fit-or-fail) rather than guess.
5. **Every `llama_kv_cache`** is windowed independently — Flash-Next has two (the attention KV and the QSA
   **indexer** cache) and both take the window, one warning line each.
6. A `remap`/flip destroys a chunk's contents: `ensure_span` stages it through a pinned scratch with a device
   synchronise around the unmap.  A capture/graph holds the **VA**, which never changes, so graphs stay valid.

Knobs (all campaign-only, default off except where noted): `LLAMA_KV_REGION=1` (enables the provider),
`LLAMA_KV_RESIDENT_CELLS=W` (the window; **unset = no window**, i.e. today's cache), `LLAMA_KV_REGION_DEBUG=1`,
`GGML_CUDA_VMM_REGION_SELFTEST=1`, `GGML_CUDA_SLAB_STATS=1`.

## 4. Environment and build

* Build: `~/bin/build-llama-rocm` (ROCm 10.1.0 → `build-rocm/`, gfx1201, ccache).  **Never** pass
  `-DCMAKE_HIP_FLAGS`.  A warm `llama-cli` relink ≈ 5 s; a `.cu` delta 15-30 s; a ubiquitous header
  (`fattn-common.cuh`, `ggml-cuda.cu`) 7-9 min.  Keep ccache ≥ 30 GiB.
* `LD_LIBRARY_PATH=/opt/rocm-10.1.0-gfx120X/lib`; **`HIP_VISIBLE_DEVICES=0`** always.
* `llama-cli` prints its accounting **only with `-v`** — and `-v` interleaves log text into the generation, so
  **hash text from a run WITHOUT `-v`** with `~/llama-cpp-rdna-boosts/scripts/extract-generated.py`.  Read timings
  and the cache/arena lines from a **separate `-v` run** of the same command.
* `--single-turn --no-display-prompt` always; wrap everything in `timeout`; **never two GPU jobs at once**.
* Models: `/llm/models/Qwen3.8/27B/Q8_0` (the Goal A target, 27.1 GiB weights), `…/27B/Q4_K_XL` (iteration),
  `Qwen3.6/35B-A3B/Q8_0` (the Q8_0 MoE arm), `Qwen3.6/35B-A3B/Q4_K_M`, `Qwen3.5/122B-A10B/Q4_K_XL` (48 layers,
  **the arm that shows the arena's value**), `Qwen3.8/Flash-Next/GSQ-IQ3_XXS` (qwen4exp, 512 experts, two KV
  caches; **supported** — qwen4exp is in r174's block 11), `Qwen3.5/4B/Q8_0` (fast smoke).
* Prompts: `prompts/prose-rdna-boosts.txt` (5256 tokens at the 27B), `prompts/kld-corpus-35b-in-domain.txt`
  (23k; `cat` it twice into `/tmp/prompt-46k.txt` for a 51 475-token prompt).

## 5. Measurement rules (each of these cost a session)

1. **`-n 2000` for anything about the expert arena.**  A short run's decode is drowned out by the cache warm-up:
   the same 35B-A3B Q8_0 arm reads 54.6 t/s at `-n 128` and 66.3 at `-n 2000`.  (Maintainer's rule.)
2. **`-ncmoe` must be the model's full layer count** when the point is the arena (35B-A3B 48, 122B-A10B 48,
   Flash-Next 48).  `-ncmoe 40` left eight layers of experts on the device and the arena was half the size.
3. **Read a delta only from arms interleaved back-to-back in one batch** — this box's clocks drift.
4. **Never judge run-to-run determinism on a greedy text hash** (hipBLASLt per-process solving, `AGENTS.md`).
5. Two `llama-cli` runs of the same command differ in text only if something is wrong; the identity gates below
   are the oracle.

## 6. Gates — the exact commands and what they must read

```bash
cd ~/llama.cpp
export LD_LIBRARY_PATH=/opt/rocm-10.1.0-gfx120X/lib HIP_VISIBLE_DEVICES=0
P=~/llama-cpp-rdna-boosts

# 0. the primitive (it now also maps a HOST-backed span and has the device write it)
GGML_CUDA_VMM_REGION_SELFTEST=1 ./build-rocm/bin/llama-cli -v -m /llm/models/Qwen3.5/4B/Q8_0/Qwen3.5-4B-Q8_0.gguf \
  -ngl 99 -c 512 -p hi -n 2 --no-display-prompt --single-turn 2>&1 | grep "selftest.*PASS"

# 1. text identity: plain vs window (must be byte-identical).  Hash from runs WITHOUT -v.
for W in "" 2048 4096; do
  LLAMA_KV_REGION=${W:+1} LLAMA_KV_RESIDENT_CELLS=$W ./build-rocm/bin/llama-cli \
    -m /llm/models/Qwen3.8/27B/Q4_K_XL/Qwen3.8-27B-UD-Q4_K_XL.gguf -ngl 99 -c 8192 -ctk q8_0 -ctv q8_0 \
    -f $P/prompts/prose-rdna-boosts.txt -n 128 --seed 42 --temp 0 --no-display-prompt --single-turn > /tmp/id$W.log 2>&1
  python3 $P/scripts/extract-generated.py /tmp/id$W.log          # all three must print the same 526 chars/sha
done
#   known-good: 27B q8_0 = 526 chars sha=5ba44647188f ; 4B f16 = 591 chars sha=ef279d67a708 ;
#               35B-A3B -ncmoe 24 -c 16384 = 339 chars sha=061add8deaba

# 2. prefill logits (a residency change moves bytes, not values): must read 0.000000 / 100.000 %
$P/scripts/gate-prefill-logits.sh --ctx 2048 --chunks 8 --base /tmp/kld.kld \
  --ab "LLAMA_KV_REGION=1 LLAMA_KV_RESIDENT_CELLS=1024"

# 3. GPU faults (never grep a log for "page fault" by hand)
$P/scripts/check-gpu-faults.sh --since "<session start>" --log /tmp/id.log … 

# 4. the OOM baseline must still OOM with the feature OFF
./build-rocm/bin/llama-cli -m /llm/models/Qwen3.8/27B/Q8_0/Qwen3.8-27B-Q8_0.gguf -ngl 99 -c 262144 \
  -ctk q8_0 -ctv q8_0 -p hi -n 4 … 2>&1 | grep "cudaMalloc failed: out of memory"
```

**Still owed** (no gate exists yet for these — see §7): `scripts/gate-ckpt-reuse.sh`,
`scripts/gate-state-pipe.sh` (state io now crosses a boundary that *moves*), and a `MemAvailable` check before
pinning the spill.

## 7. What remains, in priority order

1. **`--fit` sizing of the window** (the agreed shape: the window slides, and `W` is *sized from the depth*;
   default `W = -c` = no spill).  This is now a **measurable break-even decision**:
   * the window pays when `(kv_size − depth) × cell_bytes` of VRAM freed is worth more than
     `(depth − W) × cell_bytes` of per-token spill;
   * one side of it, measured: Flash-Next at `-c 65536` (≈depth) frees **0.39 GiB** and costs **14.6 %** decode;
     at `-c 262144` (≈4× depth) the same model frees **3.63 GiB** and costs **0 %** (nothing spills at 5256
     tokens) with prefill **+13.1 %**.  See `A2-MIXED-BACKING.md` §3.5–§3.6.
   * `--fit` must also **report** the split (resident MiB of total, spilled MiB, the window in cells, and the
     modelled cost at the current depth) instead of warning and running anyway.
2. **`gate-ckpt-reuse.sh` + `gate-state-pipe.sh`** — the only paths the slide touches that the identity arms do
   not (a restore writes arbitrary cell indices; `grow_for` maps the right kind by position).
3. **The `MemAvailable` gate** before pinning the spill (`MOE_HOST_POOL_RAM_PCT` pattern; up to 8.5 GiB at
   `-c 262144` on a 27B, non-swappable).
4. **MTP verify widths** on a windowed arm (A1 gated them; A2 changes memory, not kernels — lower risk, cheap).
5. **Multi-device**: not measurable here (one card).  Nothing in this campaign says anything about `-sm tensor`
   or `-sm layer` beyond what the delivery already documents.
6. **Goal B** (the sparse attended set, QSA/DSA) is *not* started and is not next; the indexer cache now
   participates in the window, which is where Goal B would begin.

## 8. The numbers (1 GPU; all at **28.8 GB/s** H2D — see §9)

**Fittability** (27B **Q8_0**, 51 475-token prompt → 53 475 used, `-n 2000`):

| arm | decode |
|---|---|
| `-c 65536` fully resident (the reference) | 18.4 t/s |
| `-c 262144`, `W=55296` (nothing spills) | **18.6 t/s** |
| `-c 262144`, `W=49152` (131 MiB/token of spill) | **18.6 t/s** |

The OOM config runs (8704 MiB KV = 8160 MiB system RAM + 544 MiB VRAM), prefill 933-1113 t/s, and the spill
hides behind that arm's 1.29 GiB/token of CPU-mapped weights.

**The arena's value** (`-c 262144`, `-n 2000`, all experts on the host):

| model | arena | residency | prefill | decode |
|---|---|---|---|---|
| 35B-A3B Q8_0 `-ncmoe 40` | 25 245 → 27 922 MiB | 77.3 → 85.5 % | +1.8 % | +1.7 % |
| 122B-A10B Q4_K_XL `-ncmoe 48` | 19 661 → 22 807 | 29.3 → 34.0 % | +8.6 % | **+7.9 %** |
| Flash-Next IQ3_XXS `-ncmoe 48` | 22 061 → 25 818 | 53.9 → 63.1 % | **+13.1 %** | +2.0 % |

**Prefill is capacity-bound** (a prefill routes *every* expert); **decode is hotness-bound** (10 of 512 routed,
and the hot set already fits where residency is high) — it only converts to a big decode gain where the arena
is starved (the 122B at 15.6 % residency gained **+10.1 %**).

## 9. THE BOX IS CHANGING — re-baseline before quoting anything

The maintainer is swapping components: the PCIe layout is being re-done for **~56 GB/s H2D** (from the measured
**28.8 GB/s** this whole campaign is calibrated on) and the model SSD becomes more responsive.  So:

* **Re-measure the link constant first**: `wip/kv-residency/bw-probe.cc` (and `INVENTORY.md` §1.D).  Every cost
  model here is `bytes / 28.8 GB/s`; at 56 GB/s the spill cost **halves**, so a window that was break-even at
  `-c ≈ 1.4 × depth` may become clearly positive, and the `--fit` break-even of §7.1 moves with it.
* **Re-run `mixed-backing-probe.cc`** — it should still pass, but the T3/T4 bandwidth numbers and the T8 chunk
  timing will change (~130 µs per host chunk may improve or worsen).
* **Re-record the prefill-logit base** if the box's *build* is unaffected but the GPU changes; a new GPU or a
  different PCIe slot does not change logits, but a fresh base costs 30 s and removes the question.
* Do **not** compare a post-swap t/s number against a pre-swap one, even for the plain arm.

## 10. Records

| record | content |
|---|---|
| [`A2-MIXED-BACKING.md`](A2-MIXED-BACKING.md) | **the state of the build**: what is implemented, every gate, §3.5 the arena numbers, §3.6 the deep arm + the break-even, and the two bugs the measurements found |
| [`MIXED-BACKING.md`](MIXED-BACKING.md) | the design + the driver probe (one VA range can be backed partly by pinned host memory; host chunks; the real shape at scale) |
| [`A1-TWO-BASE-KV.md`](A1-TWO-BASE-KV.md) | A1 (two-base read in the FA kernels): built, gated, **superseded**, kept as the fallback |
| [`O2-PLAN.md`](O2-PLAN.md) | the plan of record as written (its §6.3 mechanism is now historical — mixed backing won) |
| [`IMPLEMENTATION.md`](IMPLEMENTATION.md) | O1 (the growable KV region) — **parked**; A2 reuses its buffer-type skeleton |
| [`INVENTORY.md`](INVENTORY.md), [`VMM-PROBE.md`](VMM-PROBE.md) | the taxonomy and link constants; the VMM driver facts |
| [`STRATA-KV-STREAMING.md`](STRATA-KV-STREAMING.md), [`O3-DRAFTER-RING.md`](O3-DRAFTER-RING.md) | reference designs; Goal C (done, last priority) |
| `patches/` | `0001`-`0008` + `README.md` + `diffstat.txt` — the campaign's code, apply to `5bdec04ab` in order |
| `mixed-backing-probe.cc`, `vmm-probe.cc`, `bw-probe.cc` | the three standalone probes |

## 11. First 30 minutes of the next session

1. `cd ~/llama-cpp-rdna-boosts && git pull`; read this file, then `A2-MIXED-BACKING.md`.
2. `cd ~/llama.cpp && git branch --show-current` — expect **`wip-kv-residency`** at **`ba5c7205e`**
   (`git log --oneline -6` should show the four A2 commits).  `git rev-parse --short rdna-boosts` must not
   matter to you; never commit campaign code there.
3. `~/bin/build-llama-rocm` (expect ~5 s warm), then run §6 gate 0 and gate 1 — if the text hashes match the
   values in §6, the tree is the one that was measured.
4. **Re-measure the link constant** (§9) if the hardware swap has happened, and note the new value in
   `INVENTORY.md` §1.D and in the top of `A2-MIXED-BACKING.md`.
5. Start §7 item 1 (`--fit` sizing) — that is the only thing between this and a product-shaped opt-in.
