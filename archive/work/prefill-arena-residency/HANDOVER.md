# HANDOVER — `prefill-arena-residency` (R2) → beta-12 integration, gates, delivery prep

**ARCHIVED — this campaign was promoted at r83 (blocks 09/10); see `WORKLOG.md` 261018.003.  The
checkpoint/tasks below are kept as the record of how it was done.**

**For the next session.  Read first:** `AGENTS.md`, then this campaign's `README.md`,
`CONCURRENCY-2026-10-18.md`, then this file.  Nothing here is delivery work yet; nothing may be
committed to `patches/` without the maintainer's go-ahead.

---

## 0. Current checkpoint (2026-10-18)

| item | value |
|---|---|
| campaign HEAD | `d9be7bd` (`wip/prefill-arena-residency: drop the 75/25 freeze; restore the R2-only prototype`) |
| prototype patch | [`r2-prototype.patch`](r2-prototype.patch), **227 insertions / 3 files**, sha256 **`2bfdb8263b0b…`** |
| working tree | `~/llama.cpp` is on `rdna-boosts` (`24380e5c7`), R2 patch applied **uncommitted**, `build-rocm` rebuilt from it |
| freeze | implemented, measured as a net loss with no decode gain, **dropped** (code removed; recoverable at `8134e23`) |
| beta 12 | **integrated + measured + gated (2026-10-18, session 2)** — apply both orders, canonical build, +9 % on top of R2; full matrix green: [`BETA12-INTEGRATION-2026-10-18.md`](BETA12-INTEGRATION-2026-10-18.md) |
| decisions | no mutating devmap; immutable PRS approved; freeze dropped; serialized scheduler is the coexistence limit |

If the `~/llama.cpp` working tree is lost:
```bash
cd ~/llama.cpp && git checkout -- .
git apply ~/llama-cpp-rdna-boosts/wip/prefill-arena-residency/r2-prototype.patch
```

---

## 1. What the R2 patch does (and does not do)

The r40 removal left `ggml_backend_cuda_stage_from_cache` as a `return false` stub.  R2 makes it live,
but **only** for cache-aware whole-table staging, with the **immutable prefill residency snapshot
(PRS)** — no device-remap policy, no `slot_dev` mutation, no deferred in-place read:

1. When the scheduler stages a host expert table (`stage_upload` for `-sm layer`, `stage_gather` for the
   `-sm tensor` meta split), `moe_cache_stage_table` matches it to the registered table by
   `(device, host master, src_off)` and validates the exact copy geometry.
2. It builds/refetches a per-table `expert -> slot` device array (the PRS) from the authoritative host
   `expert_slot`, when the cache write generation (`g_write_seq`) moved or the arena moved.  The map is
   uploaded on the copy stream (ordering is local; no compute-stream wait just for a map build).
3. One `moe_cache_stage_kernel` per table fills the ring slot: resident experts come from their arena
   slot (D2D, the device's compact `expert_bytes` slice), the rest from the pinned host master
   (`host_dev`, 2-D strided for an axis-0 split).  The destination is the ordinary staged table in
   expert order, so every consumer is unchanged.
4. The existing head-pad zero guard and the `unusable`/geometry fallbacks are kept.  A table the slab
   stood down mid-prefill (`slots == 0`, `arena == nullptr`) falls back to the plain upload.
5. `moe_cache_report` prints the staged split (`arena` vs `host` MiB, H2D %, `geom_decline`,
   `unusable`, `nomatch`, PRS rebuilds).

Env: **`GGML_MOE_CACHE_STAGE=1`** enables it; the prototype default is `0`.  For delivery, decide the
default per `AGENTS.md`'s default-on policy (see §5).

**Not in the patch:** the 75/25 freeze, server activity hints, `llama_moe_cache_set_activity`, any
device slot map / `res_gen`/`dev_gen` / promotion, any MMQ in-place read.  Do not reintroduce them.

Files changed (all block-09 territory):
- `ggml/src/ggml-cuda/ggml-cuda-moe-cache-hooks.cu` — the `stage_from_cache` implementation.
- `ggml/src/ggml-cuda/moe-expert-cache.cu` — PRS fields, `moe_cache_stage_table`,
  `moe_cache_stage_kernel`, counters, env parse.
- `ggml/src/ggml-cuda/moe-expert-cache.h` — the declaration.

---

## 2. Measured results to reproduce (single session)

Model: `Qwen3.8-Flash-Next IQ4_NL` (100 GB, 176.9B A3B); 2× R9700 gfx1201; ROCm 10.1.0.
**Field loader flags are mandatory for any prefill number** (`--load-mode none --lazy-mode off`);
without them the 27.8 GiB PLE host gather faults and the baseline is ~329 t/s instead of ~580/1 090.

| prompt | `-ub` | stage 0 | stage 1 (R2) | gain | greedy sha (both) |
|---|---:|---:|---:|---:|---|
| 15 800 tok | 2048 | 581.8 | **838.8** | +44.2 % | `7171139467ce` |
| 15 800 tok | 4096 | 1 091.8 / 1 092.1 | **1 433.1 / 1 431.7** | +31 % | `a60a39e11133` |
| 31 555 tok | 4096 | 1 183.6 | **1 561.3** | +31.9 % | `98800fdba634` |

- H2D staged: **27.1 %** of the table bytes (arena 272.6 GiB vs host 101.6 GiB on the 15.8k/ub2048
  run); `geom_decline=0`.
- Arena at 32k ctx: **47 207.8 MiB = 72.9 %** residency (373/512 slots/table); at 64k ctx:
  **46 195.3 MiB at sizing / 45 712.8 MiB after prefill (~70.5 %)**, decode hit `h=0.7698`.
- Single-session path never freezes; `PRS rebuilds` ≈ one per table per ubatch.

Two-slot concurrency (`-np 2`, `-c 131072`, 64k/slot, `-b/-ub 4096`), full record in
[`CONCURRENCY-2026-10-18.md`](CONCURRENCY-2026-10-18.md):

| scenario | metric | stage 0 | R2 |
|---|---|---:|---:|
| A: 15.7k prefill + 5.3k prefill/decode | long prefill | 754.9 | **889.4** (+17.8 %) |
| A | short prompt | 293.3 | **339.2** |
| B: long decode + 3.9k prefill stream | short prefill | 1 162.8 | **1 479.1** (+27.2 %) |
| B | long decode | 3.76 | 4.17 |

The long decode drops to ~4 t/s under a prefill stream **with R2 off and on** — the two-slot scheduler,
not the cache.  The slab separately evicts/re-arms 54 tables per prefill→decode transition.

---

## 3. Next session — tasks, in order

### 3.1 Beta-12 integration (first; the maintainer asked for this)

**DONE 2026-10-18 (session 2)** — see [`BETA12-INTEGRATION-2026-10-18.md`](BETA12-INTEGRATION-2026-10-18.md)
§1-§5.  Both orders give byte-identical content (no reverse build needed); canonical tree
`~/llama-beta12-r2` = release → beta 12 → R2, build id `ab3af95ee5f5`.  R2 gain on that tree
**+43.8 %** (16k/ub4096) / **+58.3 %** (16k/ub2048) / **+48.0 %** (32k), greedy hashes identical to
the record; **beta 12 adds +8.6 … +9.6 %** on top of R2 (arena 46.6 → 52.3 GiB, residency 71.9 % →
80.7 %, staged H2D 27.1 % → 17.6 %) and costs ~1 % with R2 off.  `-sm layer` + `--host-experts pool`
measured (both byte-identical, +71 % / +40 %).  Two-slot concurrency: R2 +23.6 % (A long) / +39.8 %
(B short), starvation unchanged, re-arm 54 tables with beta vs 60 without.

Historical task text (kept for method):

Beta candidate:
`beta/0012-slab-transient-absorption/0012-rdna-boosts-beta-12-slab-transient-absorption.patch`
(sha256 `9facba49301c0f4f6efa880692bbb48d06bde3e40f4a4c45adc97db98595c5d8`, BETA 2, cut against
`release.json` `v12-097cf33d9-r82`, tree `dc48b4a9…`).

**Already verified (2026-10-18):** on a clean `rdna-boosts` worktree, `git apply --check` succeeds in
**both** orders — `beta → R2` and `R2 → beta`.  The overlap is `ggml-cuda-moe-cache-hooks.cu`, where
beta 12 changes `ggml_backend_cuda_stage_h2d_gbps` (the 512 MiB calibration probe now from the slab
arena) and R2 changes `ggml_backend_cuda_stage_from_cache`; the hunks are disjoint.
`moe-expert-cache.*` is not touched by beta 12.

Still required (build + behaviour, because beta 12 changes slab transient placement and therefore the
arena size / boundary churn our residency depends on):

1. Canonical tree: fresh `scripts/apply-all.sh` checkout at `release.json.base` (`097cf33d9`), then
   `git am -3 beta/...patch` (or `scripts/apply-all.sh --include-beta .`), then
   `git apply wip/prefill-arena-residency/r2-prototype.patch`.
2. Build with `~/bin/build-llama-rocm` (ROCm 10.1.0, canonical).  Do the reverse order too if cheap;
   the canonical delivery order is release → beta → R2.
3. Run the 16k stage 0 / stage 1 A/B (§4) **with beta 12**, then the 15.8k/ub2048 and 31.6k points.
   Compare:
   - greedy hash stage 0 == stage 1 (and to the recorded `a60a39e11133` at 16k/ub4096),
   - the arena size, the staged H2D split, `unusable`/`geom_decline`,
   - the slab `evicted`/`re-armed` counts (beta 12 should reduce the transient churn),
   - prefill t/s (beta 12 may change the arena size; report the delta, do not assume it is neutral).
4. Re-run the concurrency harness (`--scenario a` and `b`, `--stage 0/1`) with beta 12.  The
   scheduler starvation (B long decode) must remain identical with R2 off/on; if beta 12 changes the
   slab churn, record the evict/rearm counts.
5. Check the beta 12 `stage_h2d_gbps` calibration from the slab arena coexists with the R2 arena
   sizing: the calibration runs before `alloc_all_locked`; confirm the log line says "from the slab
   arena" and the arena sizes normally.

If a conflict appears, rebase only the R2 hook hunk; do not edit beta 12.

### 3.2 Gates (with and without beta 12)

**DONE 2026-10-18 (session 2) — all green** on both trees; the table of results is
[`BETA12-INTEGRATION-2026-10-18.md`](BETA12-INTEGRATION-2026-10-18.md) §6.  Highlights: dense + host-expert
coherence identical; prefill-logit **mean KLD -0.000000 / same-top-p 100.0 %** (dense) and **max KLD
5.9e-5 / 100 %** on the host-expert MoE path (`--save-all-logits` / `--kl-divergence`); qwen4exp PASS;
adaptive MTP 4 axes + phase probe identical (0.66461 / 0.51005 / 0.76070 / 0.96262 / 0.62752);
`MUL_MAT_ID` 947/947; redirect mode neutral; GPU faults clean; no `////`.

Historical gate list (kept for method):

Run on the R2 build (stage 1) against stage 0 as the control:

- **Coherence:** dense `Qwen3.5-4B Q8_0` same-seed (known-good build reference), and the Flash-Next
  IQ4_NL host-expert config `stage 0 == stage 1` greedy hash at 16k/ub4096 (recorded
  `a60a39e11133`).  Also `-sm layer -ncmoe` 2-GPU (the R2 hook is reached through `stage_upload`
  there; untested so far).
- **Prefill-logit:** `scripts/gate-prefill-logits.sh` — record a base with the pre-change build, then
  compare; expect **mean KLD 0 / same-top-p 100 %** (the staged bytes are identical).
- **qwen4exp quant coherence:** `scripts/gate-qwen4exp-quant-coherence.sh` (host-expert gather /
  slab interplay; R2 does not touch the gather).
- **Adaptive-MTP:** the 4 axes + phase-switching probe (`benchmarks/mtp-adaptive-methodology.md`),
  `-n 3000`, reasoning pinned; expect identical text/acceptance (no decode kernel changed).
- **MUL_MAT_ID:** `test-backend-ops -o MUL_MAT_ID` (sanity; no kernel change).
- **GPU faults:** `scripts/check-gpu-faults.sh` on the server log; `////` count 0.
- **Performance:** 16k ub2048/ub4096, 31.6k ub4096; stage 0 vs 1; with/without beta; if possible the
  field 131k/MTP config on 2 GPUs (the field uses `-sm tensor -ncmoe 48`, MTP adaptive n7).
- **Mode coverage:** default staging (mode 0) and `GGML_SCHED_STAGE_MODE=1` (redirect); below the
  staging gate is the host/gather path (not R2) — confirm it is unchanged.

### 3.3 Delivery prep (only after 3.1 + 3.2 are green)

**Blocked on the maintainer's go-ahead.**  Decisions to record (from
[`BETA12-INTEGRATION-2026-10-18.md`](BETA12-INTEGRATION-2026-10-18.md) §7): (a) ship the r82 numbers and
note that the extra ~9 % arrives only if beta 12 promotes; (b) drop the `TODO(freeze, phase 2)` comment
from the delivered patch; (c) how to fix the stale `ENVIRONMENT.md` `GGML_MOE_CACHE_INPLACE`/`STAGE` rows
(remove `INPLACE`, rewrite `STAGE` as default-on with `0` as the kill switch); (d) block 09 amendment
(confirmed in session 1) and the new revision number.

Per `AGENTS.md`:

1. **Block:** amend **block 09** (all three files are block-09 territory) rather than add a block;
   confirm with the maintainer.
2. **Default:** flip `GGML_MOE_CACHE_STAGE` to **on** in the same change (default-on policy), with
   `0` as the kill switch; keep `GGML_MOE_CACHE_INPLACE`/`STAGE` documentation honest (the
   `INPLACE` path was removed at r40 — fix the stale `ENVIRONMENT.md` rows while there).
3. Docs: `ENVIRONMENT.md` §2, `patches/README.md` block 09, `README.md` headers, a dated
   `WORKLOG.md` entry (measurements, beta integration, gates).
4. Regenerate from a canonical tree at `release.json.base` via `scripts/apply-all.sh` +
   `scripts/make-patches.sh` / `scripts/make-release.sh`; `scripts/validate-set.sh` must be green
   (checksums + fresh-tarball strict 12/12 + tree).
5. Fork refresh: clean `~/llama.cpp` at `release.json.base`, `apply-all.sh`, build, the gates above,
   then `git push --force-with-lease origin rdna-boosts` to the **personal fork only**; never upstream.
6. Container images: the off-host worker watcher publishes after the tag; do **not** run the full
   image build on the primary box.
7. Report any review findings / open questions to the maintainer **before** the delivery commit/tag.

---

## 4. Exact reproduction commands

Environment:
```bash
export LD_LIBRARY_PATH=/opt/rocm-10.1.0-gfx120X/lib
export HIP_VISIBLE_DEVICES=0,1
export LLAMA_HC_BLK16=1 LLAMA_HC_RES16=1 GGML_CUDA_MMB_DOWN16=1
```

Build:
```bash
cd ~/llama.cpp && cmake --build build-rocm --target llama-cli llama-server -j 16
# or the canonical: ~/bin/build-llama-rocm
```

16k prompt (temp; 3× the shipped prose prompt; **not** a delivery prompt):
```bash
cd ~/llama-cpp-rdna-boosts
python3 - <<'PY'
from pathlib import Path
parts = [Path("prompts/prose-rdna-boosts.txt").read_text() for _ in range(3)]
Path("/tmp/prefill16k.txt").write_text("\n\n===== SECTION =====\n\n".join(parts))
PY
# sha256 d879500312a5c8c1e58a0032289beb4a8c14b4fb07fc3f41942d8d557bf6a817, 48 268 B, 15 800 tok
```
31.6k: `/tmp/prefill32k.txt` = the 16k text + `"\n\n===== PART 2 =====\n\n"` + the 16k text
(96 558 B, 31 555 tok).

16k A/B (the stage-0 control and stage-1 R2 run; `-n 64` gives the hash, hash both logs):
```bash
cd ~/llama.cpp
for st in 0 1; do
  HIP_VISIBLE_DEVICES=0,1 GGML_MOE_CACHE_STAGE=$st \
  LLAMA_HC_BLK16=1 LLAMA_HC_RES16=1 GGML_CUDA_MMB_DOWN16=1 \
  timeout 1800 ./build-rocm/bin/llama-cli \
    -m /llm/models/Qwen3.8/Flash-Next/IQ4_NL/Qwen3.8-Flash-Next-IQ4_NL-PROJFIX-00001-of-00009.gguf \
    -ngl 99 -sm tensor -ncmoe 48 -fa 1 -ctk f16 -ctv f16 \
    -c 32768 -b 4096 -ub 4096 -t 8 -st \
    --load-mode none --lazy-mode off \
    -p "$(cat /tmp/prefill16k.txt)" -n 64 --seed 42 --temp 0 --no-display-prompt \
    > /tmp/r2_16k_stage${st}.log 2>&1
  grep -oE "\[ Prompt: [0-9.]+ t/s \| Generation: [0-9.]+ t/s \]" /tmp/r2_16k_stage${st}.log
  python3 ~/llama-cpp-rdna-boosts/scripts/extract-generated.py /tmp/r2_16k_stage${st}.log
done
```
Expected: stage 0 ≈ 1 092 t/s, stage 1 ≈ 1 432 t/s, both hashes `a60a39e11133`.

Concurrency harness:
```bash
cd ~/llama-cpp-rdna-boosts
python3 wip/prefill-arena-residency/tools/concurrency_check.py --scenario a --stage 1
python3 wip/prefill-arena-residency/tools/concurrency_check.py --scenario b --stage 0
```
(It starts/stops the server itself; logs `/tmp/conc_*_stage*.log`, JSON `/tmp/concurrency_check.json`.)

Prefill-logit gate (record a base first with the pre-change build, then compare):
```bash
cd ~/llama-cpp-rdna-boosts
./scripts/gate-prefill-logits.sh --record   # on the known-good pre-change build
./scripts/gate-prefill-logits.sh            # on the R2 build
```

Beta-12 apply check (both orders, clean tree):
```bash
cd ~/llama.cpp
BETA=~/llama-cpp-rdna-boosts/beta/0012-slab-transient-absorption/0012-rdna-boosts-beta-12-slab-transient-absorption.patch
R2=~/llama-cpp-rdna-boosts/wip/prefill-arena-residency/r2-prototype.patch
git worktree add --detach /tmp/llama-beta-check rdna-boosts
cd /tmp/llama-beta-check
git apply --check "$BETA" && git apply "$BETA" && git apply --check "$R2" && echo "beta->R2 OK"
git checkout -- . && git clean -fdq
git apply "$R2" && git apply --check "$BETA" && echo "R2->beta OK"
```
(Already green on 2026-10-18; re-run after any rebase.)

---

## 5. Known risks / gotchas

- **Pinned host master is required.**  The staging kernel reads `host_dev` in place; a pageable master
  must decline (the table's `host_pinned` gate) — issue #116.
- **`--host-experts pool`** bypasses the in-place cold half (the prior PR's note).  Decide whether to
  decline for pooled tables or read the pool; measure if the pool is the field default.
- **Slab stand-down mid-prefill** is normal (`unusable>0`); the table falls back to the plain upload.
  Do not treat `unusable` as an error.
- **The PRS is rebuilt per table per prefill ubatch** when `g_write_seq` moved; the map is 2 KiB.  The
  cost is the copy-stream ordering, not the map.  A per-table generation (rebuild only if that table's
  slots changed) is a possible refinement; low priority.
- **Do not revive the 75/25 freeze** (net loss; the per-layer pin is a no-op in the serialized
  scheduler).  The serialized scheduler is upstream (`ggml_backend_sched_compute_splits`); the fix for
  coexistence is `archive/work/moe-cpu-overlap/`, not this cache.
- **`-sm tensor` vs `-sm layer`:** both reach the hook, but only `-sm tensor` has been measured.
  `-sm layer` + `-ncmoe` may qualify for upstream `pipeline_parallel`; if so, the scheduler uses
  `n_copies>1` — verify the staging/cache still behaves (it is a different path).
- **The `GGML_SCHED_STAGE_MODE=1` redirect** changes whether the consumer reads the ring slot directly;
  test both.
- **No upstream push.**  Only the personal fork (`git@github.com:stew675/llama.cpp.git`) and the
  delivery repo; `--force-with-lease`, never bare `--force`.

---

## 6. Record / links

- Campaign: [`README.md`](README.md) (design, prior art PR #121, measurements).
- Concurrency + freeze negative result: [`CONCURRENCY-2026-10-18.md`](CONCURRENCY-2026-10-18.md).
- Patch: [`r2-prototype.patch`](r2-prototype.patch) (sha `2bfdb826…`).
- Freeze implementation (dropped, for the future overlap work): campaign commit `8134e23`.
- Beta candidate: [`../../beta/0012-slab-transient-absorption/README.md`](../../beta/0012-slab-transient-absorption/README.md).
- The prior art (r33/r36 PR #121, removed at r40): `git show 670765b`,
  `archive/work/moe-verify-fusions/{FINDINGS-devmap-removal.md,remove-devmap.diff,wip-since-block15.diff}`.
- Serialized scheduler: `ggml_backend_sched_compute_splits` (`ggml/src/ggml-backend.cpp`, upstream at
  `097cf33d9`), `llama_context::decode`/`process_ubatch` (upstream).  The per-layer pin becomes useful
  only once a decode graph can overlap a prefill graph (`archive/work/moe-cpu-overlap/`).
