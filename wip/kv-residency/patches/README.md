# `wip/kv-residency/patches` — the campaign's code, self-contained

**Not delivery work.**  These patches belong to the `wip/kv-residency` campaign (`TODO` #117).  They are not
in `patches/`, not in `release.json`, and no block in the delivery reads them.  Nothing here may be applied to
`~/llama.cpp`'s `rdna-boosts` branch, and folding any of it into the delivery is the promotion rule's job
(`AGENTS.md`, WIP and promotion rules), not a hand-edit.

They are stored here so a cold session can reproduce the exact tree the campaign was measured on, and so the
work is backed up with the record that explains it.

## How to reproduce the tree

The patches apply to the **r174** release tip, `5bdec04ab` (`release.json`: `v15-6050b467c-r174`,
base `6050b467c`).

```bash
cd ~/llama.cpp
git checkout rdna-boosts && git checkout 5bdec04ab        # or: scripts/apply-all.sh after a fresh clone
git checkout -b wip-kv-residency
git am ~/llama-cpp-rdna-boosts/wip/kv-residency/patches/0001-*.patch
~/bin/build-llama-rocm
```

If `5bdec04ab` is not available (an old checkout), re-cut it first from this repo:
`./scripts/fork-refresh.sh` (default `LLAMA_REPO=~/llama.cpp`, no `--push`).

## The series

| patch | what | state |
|---|---|---|
| `0001-ggml-cuda-grow-only-VMM-region-primitive.patch` | `ggml/src/ggml-cuda/ggml-cuda-vmm-region.{h,cu}`: one VA reservation, physical mapped per chunk, grown monotonically, clear-on-map, fail-soft.  Plus a one-shot `GGML_CUDA_VMM_REGION_SELFTEST=1` hook in `ggml_backend_cuda_init`.  Purely additive — no existing path changes. | **applied, building, selftest PASS**; gates green (see `../IMPLEMENTATION.md`) |

`diffstat.txt` is the net diff of the series against `5bdec04ab`.

As slice 2 (the KV buffer type) lands, it is appended as `0002-…`: re-cut with
`git format-patch 5bdec04ab..wip-kv-residency`, replacing the series rather than adding a patch on top of an
already-applied one.

## Verifying an applied tree

```bash
HIP_VISIBLE_DEVICES=0 GGML_CUDA_VMM_REGION_SELFTEST=1 ./build-rocm/bin/llama-cli \
  -m /llm/models/Qwen3.5/4B/Q8_0/Qwen3.5-4B-Q8_0.gguf -ngl 99 -c 4096 -t 8 -p hi -n 4 \
  --temp 0 --no-display-prompt --single-turn -v 2>&1 | grep vmm_region
# device 0: VMM region available (...)
# device 0: PASS (reserved 384 MiB in 6 x 64 MiB units, grew to 3, ...)
```

and confirm the arm is inert by default: a run without the env var must print nothing from this file and
produce byte-identical same-seed text to an r174 build (`../IMPLEMENTATION.md`, "Gates for the slice").
