# `wip/kv-residency/patches` — the campaign's code, self-contained

**Not delivery work.**  These patches belong to the `wip/kv-residency` campaign (`TODO` **#118**; the commit messages inside say
`#117`, which is the handle the campaign used before that number was allocated to another item).  They are not
in `patches/`, not in `release.json`, and no block in the delivery reads them.  Nothing here may be applied to
`~/llama.cpp`'s `rdna-boosts` branch, and folding any of it into the delivery is the promotion rule's job
(`AGENTS.md`, WIP and promotion rules), not a hand-edit.

They are stored here so a cold session can reproduce the exact tree the campaign was measured on, and so the
work is backed up with the record that explains it.

## How to reproduce the tree

The patches apply to the **r174** release tip, `5bdec04ab` (`v15-6050b467c-r174`, base `6050b467c`).
**r176 is now the delivery's current release** (`release.json`); these patches are *not* re-cut against it,
because the campaign is not promoted.  A promotion re-cuts the series from the then-current release's tip.

```bash
cd ~/llama.cpp
git checkout rdna-boosts && git checkout 5bdec04ab        # or: scripts/apply-all.sh after a fresh clone
git checkout -b wip-kv-residency
git am ~/llama-cpp-rdna-boosts/wip/kv-residency/patches/000[123]-*.patch
~/bin/build-llama-rocm
```

If `5bdec04ab` is not available (an old checkout), re-cut it first from this repo:
`./scripts/fork-refresh.sh` (default `LLAMA_REPO=~/llama.cpp`, no `--push`).

## The series

| patch | what | state |
|---|---|---|
| `0001-ggml-cuda-grow-only-VMM-region-primitive.patch` | `ggml/src/ggml-cuda/ggml-cuda-vmm-region.{h,cu}`: one VA reservation, physical mapped per `unit` chunk, grown monotonically, clear-on-map, fail-soft.  The mapped extent is a **set of spans** (`grow_span`/`is_mapped`/`clear` over a per-unit bitmap), because a KV buffer lays its tensors out consecutively and a single prefix watermark would map ~the whole region even for a 4-token run.  Plus a one-shot `GGML_CUDA_VMM_REGION_SELFTEST=1` hook in `ggml_backend_cuda_init`.  Purely additive — no existing path changes. | **applied, building, selftest PASS**; gates green (see `../IMPLEMENTATION.md`) |
| `0002-ggml-cuda-KV-region-buffer-type.patch` | `ggml_backend_cuda_kv_region_buffer_type` (ggml-cuda.cu) + the `llama_kv_cache` selection and the `apply()` grow (src/llama-kv-cache.{h,cpp}).  Env-gated OFF (`LLAMA_KV_REGION=1`), plain device buffer type as the mandatory fallback. | **applied, building**, all gates green — but see the KNOWN LIMIT in `../IMPLEMENTATION.md`: on a slab-armed MoE config the arena keeps the VRAM the KV no longer takes up front, so a run that uses most of `-c` refuses where the plain arm completes |

| `0005-wip-kv-residency-A2-mixed-backing-windowed-KV-one-VA.patch` | **A2, the mixed-backing windowed KV**: one VA reservation, the KV's leading cells backed by pinned system RAM, the rest by VRAM; the mapping-table primitive, the split buffer type, the split point + loud warning.  See `../A2-MIXED-BACKING.md` | **applied, building, gates green** (the design of record) |
| `0006-…-the-SLIDING-window….patch` | the window follows the watermark (the newest W *used* cells stay resident); `ensure_span` flips a mapping's kind through a pinned scratch | applied, measured |
| `0007-…-quantise-the-slide-boundary-i….patch` | the slide boundary is quantised in BYTES to a whole map unit — without it a host chunk was created per tensor per token (**14.7 → 18.6 t/s**) | applied, measured |
| `0008-…-commit-the-resident-WINDOW-a….patch` | the resident window's VRAM is committed at buffer creation, **before `SLAB_INIT`** — without it a deep batch aborts (`growth refused`, `free=0.0 MiB`) | applied, measured |
| `0009-wip-kv-residency-A3-fit-sizing-reporting-of-the-wind.patch` | **A3**: `--fit` sizes + reports the window (`LLAMA_KV_RESIDENT_CELLS=auto`, `llama_kv_region_set_resident_cells`, the KV's bytes-per-cell from the region buffer type, the resident/spill split in `memory_breakdown()`); the pinned-RAM ceiling (`LLAMA_KV_SPILL_RAM_PCT`, creation-time clamp + run-time refusal); the four relaxed async-tensor buffer-type asserts the state gates caught; and the three slide bugs — unit-grid mappings + a refused straddle, kernel-copied staging (a `cudaMemcpy` into a host-location mapping does not land), no coalescing.  See `../A2-MIXED-BACKING.md` §3.7 | **applied, building, gates green** — the state gates + MTP verify widths pass on a windowed arm; **the spilled-KV numeric difference of §3.7.6 is OPEN** |

| (A2-early) | **A2, the mixed-backing windowed KV**: one VA reservation, the KV's leading cells backed by pinned system RAM, the rest by VRAM; the mapping-table primitive, the split buffer type, the split point + loud warning.  Measured: the 27B Q8_0 `-c 262144` OOM config runs; `(used − resident) × cell_bytes / ~28 GB/s` matches to ~3 %; slab mode's arena gains 2.69 GiB and the 35B-A3B **Q8_0** expert residency goes 63.3 % → 77.0 %.  See `../A2-MIXED-BACKING.md` | **applied, building, gates green** (the design of record) |

| `0004-ggml-fa-A1-two-base-KV-read-SUPERSEDED.patch` | **A1, the two-base K/V read** — the op carries `k_tail`/`v_tail` + `n_tail`, and the tile and MMA-f16 kernels select the segment where a KV tile's base is formed (the cell index space stays global).  Gated byte-identical at several window widths on both families, MTP verify widths, and a negative control.  **SUPERSEDED 2026-10-29** by mixed backing (`../MIXED-BACKING.md`: one VA range backed partly by pinned host memory, measured at 28.4 GB/s) — kept as the record and the fallback, see `../A1-TWO-BASE-KV.md` | **applied, building, all gates green** (not the A2 path) |

| `0003-llama-model-MTP-draft-KV-window.patch` | **O3, the drafter ring**: the MTP draft cache can be a sliding window (`LLAMA_SPEC_DRAFT_WINDOW=<cells>`, default off) — one `create_memory` branch plus a helper, reusing `find_slot`'s reclaim and the cache's own `n_swa` for the mask.  **Implemented and measured; a poor product trade** (-6..-9 % decode, +8 % prefill, and the freed VRAM does not reach the arena) — see `../O3-DRAFTER-RING.md` §2.2 | **applied, building, gates green** |

`diffstat.txt` is the net diff of the series against `5bdec04ab`.

The A1 slice is also a **named branch**, so the fallback can be checked out without replaying any patch, and
**both campaign branches are pushed to the maintainer's personal fork** (`git@github.com:stew675/llama.cpp.git`,
the only remote `~/llama.cpp` has) — that is the backup:

| branch on the fork | tip | what |
|---|---|---|
| `wip-kv-residency` | `ba5c7205e` | A2 (mixed backing + the sliding window + both fixes, the design of record) on top of A1 |
| `wip-kv-residency-a1-two-base` | `e49e1cef4` | A1 alone (the two-base K/V read, superseded, kept as the fallback) |

`rdna-boosts` on the fork stays at the release tip and is **not** touched by this campaign.

As slice 2 (the KV buffer type) lands, it is appended as `0002-…`: re-cut with
`git format-patch 5bdec04ab..wip-kv-residency`, replacing the series rather than adding a patch on top of an
already-applied one.

## Verifying an applied tree

```bash
HIP_VISIBLE_DEVICES=0 GGML_CUDA_VMM_REGION_SELFTEST=1 ./build-rocm/bin/llama-cli \
  -m /llm/models/Qwen3.5/4B/Q8_0/Qwen3.5-4B-Q8_0.gguf -ngl 99 -c 4096 -t 8 -p hi -n 4 \
  --temp 0 --no-display-prompt --single-turn -v 2>&1 | grep vmm_region
# device 0: VMM region available (...)
# device 0: PASS (reserved 384 MiB in 6 x 64 MiB units, prefix + span growth, ...)
```

The KV region itself (needs `LLAMA_KV_REGION=1`):

```bash
HIP_VISIBLE_DEVICES=0 LLAMA_KV_REGION=1 LLAMA_KV_REGION_DEBUG=1 ./build-rocm/bin/llama-cli \
  -m /llm/models/Qwen3.6/35B-A3B/Q4_K_M/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf -ngl 99 -ncmoe 24 -c 65536 \
  -t 8 -p hi -n 4 --temp 0 --no-display-prompt --single-turn -v 2>&1 |
  grep -E "KV region reserved|kv_region_grow_cells|KV buffer"
# KV region reserved 1280.00 MiB (VA only, nothing mapped; 4 MiB map unit)
# ROCmKVRegion0 KV buffer size =  1280.00 MiB
```

and confirm the arm is inert by default: a run without the env var must print nothing from this file and
produce byte-identical same-seed text to an r174 build (`../IMPLEMENTATION.md`, "Gates for the slice").
