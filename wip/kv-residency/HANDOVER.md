# HANDOVER — `wip/kv-residency` (TODO **#118**), cold-session start

**Rewritten 2026-10-30 at the end of the eighth session (A4).**  Read this top to bottom before touching
anything; it is self-contained.  Then read [`A2-MIXED-BACKING.md`](A2-MIXED-BACKING.md) (the measured state of the
build) and [`MIXED-BACKING.md`](MIXED-BACKING.md) (the design + the driver probe).

> **The one thing to know: the blocker is CLOSED (A4, 2026-10-30).**  A spilled KV is now numerically
> transparent: **prefill-logit KLD -0.000000 / 100.000 %** on a *spilling* 27B arm (it was 0.004973 / 98.761 %),
> and every spilling text arm is back on the plain hash (27B q8_0 and f16, 4B).  The fault was **not** in the KV
> data path or the flip -- it was the **VMM primitive**: `cuMemUnmap` of a device mapping, followed by
> `cuMemCreate`(HOST)/`cuMemMap` at that VA, **zeroes a different, live host-location mapping** in the same
> reservation.  Fixed by a hold taken in `vmm_map` (`LLAMA_KV_REGION_QUARANTINE`, default **on**, `=0` is the A/B
> arm that reproduces the bug).  Isolated reproducer: `hostmap-hazard-probe.cc`.  Full record:
> `A2-MIXED-BACKING.md` **§3.7.8** (the fix) and **§3.7.9** (what the spill costs).  What is left is §7.
>
> **The box changed (2026-10-30).**  The PCIe rework happened: the link is now **55.7 GB/s H2D / 56.8 D2H**
> (root port x16 @ 32 GT/s, was x8) and the probes were re-run (`A2-MIXED-BACKING.md` §3.7.1).  Every
> pre-swap t/s number in this campaign is at 28.8 GB/s and **must not be compared** with a new one; the cost
> model is now `bytes / 55.7 GB/s`.

---

## 1. State in one screen

| what | where |
|---|---|
| delivery repo | `~/llama-cpp-rdna-boosts`, branch `main`, **pushed** (`origin` = `git@github.com:stew675/llama-cpp-rdna-boosts.git`) |
| delivery release | **r176** (`release.json`) — the campaign is cut against **r174** (`v15-6050b467c-r174`, base `6050b467c`, tip `5bdec04ab`); a promotion re-cuts against the then-current release |
| the campaign's TODO handle | **#118** (`TODO.md` §118 + the table row; the older text says `#117`, which was a dangling reference — see the number note there) |
| campaign code | **fork branch `wip-kv-residency`** in `~/llama.cpp`, **pushed to the personal fork** `git@github.com:stew675/llama.cpp.git`, tip **`455d5aa94`** (A4: the hazard hold + the instruments) |
| the A1 fallback | fork branch `wip-kv-residency-a1-two-base` (`e49e1cef4`), also pushed (superseded; unchanged this session) |
| `rdna-boosts` | **never touched by this campaign.**  Locally `5bdec04ab`; the fork's branch is at r176's tip (moved by the r176 release) |
| patch snapshot | `patches/0001`...`0011` + `patches/README.md` + `patches/diffstat.txt` (3118 insertions, 18 files) -- apply **in order** to `5bdec04ab`; verified: the series reproduces the branch tip's tree exactly |
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
7. **Every mapping in a windowed region must sit on the map-unit grid** (r-A3).  A flip is a whole-mapping
   operation (the driver refuses a sub-range unmap, and re-attaching one mapping as head/mid/tail loses the
   middle piece's bytes — measured), so a boundary *inside* a mapping can only be refused.  Hence `grow_cells`
   maps `used_up` (unit-rounded), `init_tensor` commits `up(W·nb1, unit)`, `vmm_insert` does **not** coalesce,
   and `ensure_span` refuses a request that is not exactly one mapping.  Before this, a coalesced entry that
   straddled the boundary was flipped back whole and the window spilled **nothing** while its warning said it
   had.
8. **A staging copy to/from a host-location mapping must be a kernel copy** (`vmm_device_copy`), never
   `cudaMemcpy`: a direct H2D into a freshly mapped host chunk completes "successfully" and leaves zeros, as
   does a D2D whose source is one (measured: `stage=0x53`, `read=0x00`).  `LLAMA_KV_REGION_DEBUG=1` verifies
   every flip against its stage.
9. The spill is **pinned, non-swappable RAM**: `LLAMA_KV_SPILL_RAM_PCT` (85 % of MemAvailable) clamps the
   window at creation (raising `W`) and refuses a run-time mapping loudly rather than being OOM-killed.

10. **A kind flip needs the hazard hold, or the spill is silently wrong (A4, `A2-MIXED-BACKING.md` §3.7.8).**
    `cuMemUnmap`(device) + `cuMemCreate`(HOST)/`cuMemMap` at that VA zeroes a *different, live* host-location
    mapping in the same reservation.  `vmm_map` takes a hold (a `cudaMalloc` of the mapping's size, before the
    create, freed after) on **every** map site.  Never remove it, never take it only on the flip path, and never
    let a failed hold degrade silently -- a refused hold must fail the mapping.

Knobs (all campaign-only): `LLAMA_KV_REGION=1` (enables the provider), `LLAMA_KV_RESIDENT_CELLS=W` (the window)
or `=auto` (**`--fit` sizes it** -- the product shape; unset = no window, i.e. today's cache),
`LLAMA_KV_SPILL_RAM_PCT` (85, the pinned-RAM ceiling), **`LLAMA_KV_REGION_QUARANTINE` (default ON; `=0` is the
A/B arm)** -- the hold that keeps the driver from zeroing a live host mapping when a mapping is created, i.e.
**without it a spill is not numerically transparent**; `LLAMA_KV_REGION_TRACE=1` (every address-level
map/unmap/memset); `LLAMA_KV_REGION_DEBUG=1` (per-growth lines, a byte-exhaustive restore verify, a stage-out
cross-check, and a fingerprint of every host mapping -- **10-40x on a spilling arm, never quote a timing from
it**); `LLAMA_KV_REGION_CANARY=1`, `LLAMA_KV_REGION_DUMP=<path>` (hazard instruments);
`GGML_CUDA_VMM_REGION_SELFTEST=1`, `GGML_CUDA_SLAB_STATS=1`.
**DIAGNOSTIC ONLY:** `LLAMA_KV_REGION_SPILL_SKIP=<tensor indices>` keeps the listed tensors of a windowed buffer
fully resident (it bisected A2 §3.7.6; not shipped behaviour).

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
6. **Never quote a timing from a `LLAMA_KV_REGION_DEBUG` arm.**  Its fingerprint re-reads and hashes every host
   mapping at every growth: a 27B read 397 and 936 ms/token where the product reads 38.5 and 44.8.  (This cost a
   session-half of confusion on 2026-10-30 -- the "slowdown" was the instrument.)

## 6. Gates — the exact commands and what they must read

```bash
cd ~/llama.cpp
export LD_LIBRARY_PATH=/opt/rocm-10.1.0-gfx120X/lib HIP_VISIBLE_DEVICES=0
P=~/llama-cpp-rdna-boosts

# 0. the primitive -- also covers the HOST-backed span, the aligned window flips, and (A4 (ii)) the
#    HOST-MAPPING HAZARD: 8 tensors x 8.5 MiB flipped one at a time, every earlier window re-verified.
#    Run it a second time with LLAMA_KV_REGION_QUARANTINE=0: it MUST then print FAIL (that is what makes it a
#    test rather than a formality).
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

**Added in A3 (all pass, see `A2-MIXED-BACKING.md` §3.7):**

```bash
# the RAM ceiling: a tiny ceiling must widen W instead of being OOM-killed
LLAMA_KV_REGION=1 LLAMA_KV_RESIDENT_CELLS=1024 LLAMA_KV_SPILL_RAM_PCT=1 ... llama-cli -c 262144 ...
# --fit sizes and reports the window (the product shape)
LLAMA_KV_REGION=1 LLAMA_KV_RESIDENT_CELLS=auto ./build-rocm/bin/llama-fit-params -m <27B Q8_0> -c 262144 -ctk q8_0 -ctv q8_0
# the state paths, on a windowed arm (these found an ABORT: the batched save rejected the region buft)
LLAMA_KV_REGION=1 LLAMA_KV_RESIDENT_CELLS=2048 ./scripts/gate-ckpt-reuse.sh
LLAMA_KV_REGION=1 LLAMA_KV_RESIDENT_CELLS=2048 NGPU=1 SM=layer NCMOE=0 ./scripts/gate-state-pipe.sh
# MTP verify widths on a windowed arm (4B, spilling): plain/window x decode/draft-mtp must be one hash
```

**Added in A4 (the hazard, all pass):**

```bash
# the sensitive purity arm -- text alone is not enough, and the §6 trio is NOT sensitive (see below)
ROCBLAS_USE_HIPBLASLT=0 $P/scripts/gate-prefill-logits.sh --record \
  --ab "LLAMA_KV_REGION=1 LLAMA_KV_RESIDENT_CELLS=1024" --ctx 8192 --chunks 4 --kv q8_0 --base /tmp/k.kld
#   must read  -0.000000 / 100.000 %    (with LLAMA_KV_REGION_QUARANTINE=0: 0.004973 / 98.761 %, the old block)
# the text arm that shows it (27B q4_K_XL -c 8192 q8_0, prose prompt, -n 64)
#   plain 305 sha=e2cea6fa6670 ; windowed MUST be the same ; QUARANTINE=0 gives 328 (wrong)
# a 4B arm that REALLY spills (needs -c 16384; at -c 4096 it never reaches a flip)
#   4B -c 16384 -W 1024 : plain 26 sha=158c1671e511, windowed must match
# the isolated reproducer (from the delivery repo; default FAILS, --grab 4 PASSES)
hipcc -O3 --offload-arch=gfx1201 -isystem /opt/rocm-10.1.0-gfx120X/include \
  $P/wip/kv-residency/hostmap-hazard-probe.cc -o /tmp/hh && HIP_VISIBLE_DEVICES=0 /tmp/hh [--grab 4]
```

**A gate that is NOT sensitive, and must not be the regression test:** the §6 identity trio at `W=2048`/`4096`
with `-n 128` **passed while the spill was corrupted**.  Also: **any arm must be checked for `FLIP > 0**
(`LLAMA_KV_REGION_DEBUG=1`) before its result means anything about a spill** -- the campaign's "the 4B is
immune" reading came from arms that never flipped.

**Owed: §7 items 1-3** (the state gates + MTP on provably spilling arms, the Flash-Next/QSA spilling arm, the
hazard as a selftest).  Purity for a spilling arm is now established on 1 GPU for the 27B and the 4B only.

## 7. What remains

### 7.1 CLOSED (A4) — the spilled KV is numerically transparent

The bug, the isolation, the fix and the gate table are in `A2-MIXED-BACKING.md` **§3.7.8**; the cost is in
**§3.7.9**.  In one paragraph: the flip was always byte-faithful (stage-out == an independent copy-engine read,
restore == stage, exhaustively, 32/32), but `cuMemUnmap` of a device mapping followed by
`cuMemCreate`(HOST)/`cuMemMap` at that VA **zeroes a different, live host-location mapping** in the same
reservation -- so the *spilled cells* of an unrelated tensor were wiped after their flip.  `vmm_map` now takes a
hold (a `cudaMalloc` of the mapping's size, taken before the create and freed after) on **every** map site;
`LLAMA_KV_REGION_QUARANTINE=0` is the A/B arm.  A refused hold fails the mapping loudly instead of proceeding
corrupt.

Do not re-derive any of: the quant type (f16 diverges too), `nb1` alignment (the victims are 2 MiB-aligned VA
offsets, in both models), the 4B's "immunity" (its arms never reached a flip), "a lost kernel write" (the
spilled cells are never rewritten; a 0xAB canary placed after the flip is overwritten with zeros), or the flip's
read/restore paths.

### 7.2 Next, in order

1. **The correctness corners the fix reopened.**  `gate-ckpt-reuse.sh`, `gate-state-pipe.sh` and the MTP verify
   widths must run on arms that **provably spill** (`FLIP > 0`).  The A3 passes were 4B arms at `-c 4096` that
   never flipped, so they say nothing about a spill.  The state path is also the other half of the hazard: a
   restore grows the region with `grow_for`, i.e. a mid-run `map-new`, and the `grow#N` fingerprint will report
   it if it bites.
2. **A Flash-Next (qwen4exp) spilling arm.**  The only model with **two** `llama_kv_cache` instances (the
   attention KV and the **QSA indexer** cache -- both take the window, invariant 5) and the §3.7.4 park coupling
   (`llama_memory_hybrid_idx` moves the indexer with the attention cells).  Run `test-backend-ops -o
   FLASH_ATTN_QSA` there too: a silently-zeroed spill is exactly what that coupling would hide.
3. ~~The hazard as a self-test~~ -- **DONE (A4 (ii))**, `GGML_CUDA_VMM_REGION_SELFTEST=1` now includes it and is
   **sensitive** (`LLAMA_KV_REGION_QUARANTINE=0` makes it FAIL).  It only runs under that env var, so keep asking
   for it in every release gate.
4. **The §7.3 re-measurement** at the new link (below).
5. **The promotion conversation** (a block, `ENVIRONMENT.md`, and the default-on call for the window itself) --
   the maintainer's, and it needs 1-3 first.

### 7.3 The re-measurement (owed since the 2026-10-30 re-cable)

The arena tables (`-n 2000`; `-ncmoe` = the model's full layer count; arms interleaved), the deep arm, and the
moved `--fit` break-even.  The cost side is already measured (`§3.7.9`): **~61 GB/s effective, ~16.3 us per
MiB of spill per token, a straight line through the origin**, i.e. 128 MiB/token ~ -6 %, 256 ~ -9 %, 512 ~ -18 %,
640 ~ -22 %, and the window apparatus itself is free (zero-spill arms == plain).

## 8. The numbers (1 GPU; all at **28.8 GB/s** H2D — STALE, see §9)

> **Do not quote anything in this section as a current number.**  The link doubled (55.7 GB/s) and the
> maintainer's decision is that these tables are re-measured **after** §7.1 is closed.  The only post-swap
> numbers are the ones in `A2-MIXED-BACKING.md` §3.7 (fit sizing/reporting, the RAM ceiling, the state gates,
> the MTP arm) and they are correctness/behaviour, not throughput.

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

## 9. The box HAS changed (2026-10-30) — re-baselined

The PCIe rework is done: the root port is x16 @ 32 GT/s, so **H2D 55.7 / D2H 56.8 / kernel read of pinned
host 55.7 GB/s** (`bw-probe.cc`, 512 MiB and 1024 MiB, 5 reps, best of), and `mixed-backing-probe.cc` still
passes every check (T8: 768 host chunks, 53.4 GB/s).  Consequences:

* the cost model is `bytes / 55.7 GB/s` — a spill costs **half** what the campaign's tables assumed, so the
  `--fit` break-even of §3.6 moved;
* the prefill-logit base was re-recorded on the new box (`PPL 5.0803`), and the window arm still compares at
  **0.000000 / 100.000 %** at `--ctx 2048`;
* `llama-fit-params` and the KV warning line now read the **calibrated** link from the backend
  (`ggml_backend_cuda_stage_h2d_gbps`, 53 GB/s here) instead of a hard-coded 28.8;
* `LLAMA_KV_SPILL_RAM_PCT` (85 %) caps the pinned spill; the campaign's "up to 8.5 GiB at `-c 262144`" is now
  enforced rather than assumed.

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
| [`hostmap-hazard-probe.cc`](hostmap-hazard-probe.cc) | **the isolated reproducer of the A4 hazard**: replaying the window flip shows a live host mapping being zeroed; `--grab` passes, `--precreate`/`--no-flip`/`--no-stage`/`--vram-mib` still fail |

## 11. First 30 minutes of the next session

1. `cd ~/llama-cpp-rdna-boosts && git pull`; read this file, then `A2-MIXED-BACKING.md` **§3.7.8** (the fix and
   its gate table), **§3.7.9** (the cost), and §3.7.6's closure banner (the three wrong readings).
2. `cd ~/llama.cpp && git branch --show-current` — expect **`wip-kv-residency`** at **`455d5aa94`** or later
   (`git log --oneline -3` shows the A4 commit).  Never commit campaign code to `rdna-boosts`.
3. `~/bin/build-llama-rocm` (~5 s warm); then gate 0 (VMM selftest, must print PASS) and the §6 identity trio
   (27B Q4_K_XL `-c 8192` q8_0 must hash `5ba44647188f` with the feature off).
4. Do **§7.2 item 1**: the two state gates + MTP on a **provably spilling** arm.  Confirm `FLIP > 0` first
   (`LLAMA_KV_REGION_DEBUG=1`, no `-v`-free arm is trustworthy for this) — and remember DEBUG is 10-40x, so
   never take a timing from the same run.
5. Then §7.2 item 2 (Flash-Next/QSA), item 3 (the selftest case), item 4 (the re-measurement).  Report findings
   to the maintainer before committing any delivery-affecting change, as usual.
