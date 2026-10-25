# scripts/ — the tooling index

Everything in this directory exists to **apply, regenerate, validate or publish** the delivery.  None of it
is part of the patch set itself.  Each script's header comment is authoritative for its options; this file
is the index.

Conventions: all paths default to this repo (found relative to the script), a `release.json` value is read
for any number the script needs (fork point, tip, tree, block count, hashes) so nothing is hard-coded twice,
and every script is safe to run from any directory.

## The two flows

**Apply + verify (the consumer / developer path)**

```bash
scripts/apply-all.sh /path/to/llama.cpp [this-repo]   # strict 12/12 git am, tree checked vs release.json
scripts/validate-set.sh                               # checksums + a fresh-tarball strict apply (~1 min, no build)
scripts/gate-per-block-build.sh                       # a clean build per block commit (`all`; `--fast` for the release profile)
scripts/gate-prefill-logits.sh                        # prefill-logit KLD gate (prefill changes / every release)
```

**Release work (assessing PRs, A/B arms, the folded tree): ONE checkout, never one worktree per arm**

```bash
scripts/scratch-tree.sh init                     # the scratch tree at the fork point (~/llama-scratch)
scripts/scratch-tree.sh blocks                   # the 14-block delivery tree, built
scripts/scratch-tree.sh arm /tmp/pr218/0014*.patch   # release tree + one PR patch, rebuilt (a delta)
scripts/scratch-tree.sh keep base                # copy build/bin aside for an A/B against a later state
scripts/scratch-tree.sh gate --changed-from <prev-tip>   # per-block gate, reusing this path's ccache
```

**Cut a release (the maintainer path)**

```bash
scripts/make-patches.sh        # re-export the blocks from the canonical fork chain (git format-patch)
scripts/make-release.sh        # rewrite release.json + rdna-boosts-all.patch, hashes included
scripts/validate-set.sh        # must be green before the tag
git tag v<blocks>-<base>-r<N> && git push origin main --tags
# then the worker builds and pushes the container images (see CONTAINERS.md)
```

## Scripts

| script | what it does | when |
|---|---|---|
| `apply-all.sh` | Applies the whole set to a clean llama.cpp checkout: refuses a dirty tree or an existing `rdna-boosts` branch, applies all blocks with **strict `git am`** (one commit each), retries the series once with `git am -3` on drift, and on the strict path asserts the applied tree equals `release.json.tree`. `--include-beta` then applies `beta/*/*.patch` on top of the verified tree (the checkout is then *not* the release tree). | applying the delivery; the only supported apply flow (`git apply` on the concatenated series drops hunks) |
| `validate-set.sh` | The cheap gate that runs on every push: every `patches/*.patch` and `rdna-boosts-all.patch` matches its recorded sha256; the set applies strictly on a **fresh codeload tarball** of `release.json.base`; the base tree and the applied tree/commit count equal what `release.json` records. No compiler, no Docker, ~1 minute. | before every push/tag (CI runs it) |
| `gate-per-block-build.sh` | Builds the target set **once per block commit** from a wiped build directory (a reused cache hides configure/header errors and reports a wrong list). `--changed-from <rev>` rebuilds only the blocks whose delta changed (the tip is always built), `--list-only` prints the matrix, `--fast` builds the release profile (libraries + shipped binaries, no tests/tools) instead of `all`. **Measured (r138): a re-base ladder is 2101 s cold / 95 s warm with `all` and 78 s warm with `--fast`; the cold difference is `tests/test-backend-ops.cpp` at 5m13s per distinct content (~4 per re-base).** | before a release, and after moving a fix between blocks |
| `scratch-tree.sh` | **The release-arm driver: one long-lived worktree + build dir, switched between states** (`init`, `state REV`, `blocks`, `arm PATCH...`, `tip`, `keep TAG`, `gate`, `show`).  Exists because a **new source path is a full cold build** (ccache keys on the absolute paths in the compile command: measured 5m56s / 85 CPU-min for an r132 worktree at a new path, vs **8.6s** for the same content at the same path in a fresh build dir).  A wiped build dir is free; a new worktree path is not, so every arm is a `git checkout`/`git am` inside one tree and a rebuild is only the delta.  Prints each build's wall time. | assessing PRs, beta arms, A/B builds, and any per-release build work |
| `gate-prefill-logits.sh` | The prefill-logit KLD gate (issue #113): runs `llama-perplexity --kl-divergence` against a recorded known-good base and fails unless mean KLD ≤ 0.005 and same-top-p ≥ 98 %. `--record` (re)records the base, `--ab "K=V"` records an exact-fallback A/B arm. | any change to a prefill kernel (GDN/SSM chunked, FA, MMQ/MMVF/MMB, MoE prefill) or a default-on approximate path; every release |
| `gate-qwen4exp-quant-coherence.sh` | Quant-coverage gate for the host-resident-expert device gather: runs qwen4exp over a set of quantizations and fails if the model emits the `////////` corruption (an MMQ over-read landing in unzeroed slot heads). `STRICT=1` also makes a gather-on vs gather-off hash mismatch fatal. | before changing `moe_cache_gather_host` or the expert-slot head zero |
| `check-gpu-faults.sh` | The **only** valid GPU-fault check: reads the kernel journal (`amdgpu`/KFD) and, with `--log FILE`, scans a llama log *after stripping the echoed prompt lines*. Never `grep` a log for `page fault`/`slab` — the prompt echo and the model's own output produce those words in a clean run. | after any run that might have faulted, and on every gate that exercises the GPU |
| `check-windows-paths.sh` | Fails if a tracked path cannot be checked out on Windows (forbidden characters, a component ending in a space/dot, a reserved device name). A single such path makes the repo unclonable there (issue #36 was a `:` in a patch filename). | before pushing a new file whose name is not boring |
| `make-patches.sh` | Re-exports the block commits from the canonical fork chain with `git format-patch --start-number 0` (block 00 → `0000`) and writes `rdna-boosts-all.patch`. Defaults (fork path, base, tip) come from `release.json`. | after amending a block; **never** hand-edit a patch instead |
| `make-release.sh` | Rewrites `release.json` — fork point, base tree, canonical tip/tree, block count and the sha256 of every shipped artifact — plus `rdna-boosts-all.patch`. Values not passed as flags are inherited from the existing `release.json`. | after `make-patches.sh`, before the tag |
| `extract-generated.py` | Prints `<N> chars sha=<sha256[:12]>` for the text a `llama-cli` log actually generated (or the text itself with `--text`). It strips the terminal backspaces and the banner/footer first, which is why a naive `sed`/`grep` slice does not reproduce a reported hash. | comparing same-seed coherence text |
| `moe-trace-replay.py` | The **offline MoE expert-cache policy replayer** (r110, TODO #61 P1): reads an `MOE_EXPERT_TRACE` file and scores the shipped F-LRU, an oracle static top-f, a per-layer capacity quota and pass-start pre-admission against it, reporting `h`, the missing/cold split, expert MiB/token, resident bytes and covered mass, plus the global-top-f static bound. Stdlib only, no model/GPU. `--devices N` converts the traced full host expert size to the per-device arena slice; `--measured-h` prints the model's error against the run's logged `h` (judge policies by their delta). **`--seed none|heat|probe:N|sidecar:<path>`** models the load-time arena seed (`apply_heat_seed_locked`): a real persisted `.moe-heatmap` (**MHT2**: `sidecar:<path>` ranks by its decode counter, the arena's band; an **MHT1** file is read as `dec = all`) for the achievable prior, `heat` for the whole-trace oracle, `probe:N` for the trace's first N passes; **`--churn on|off`** makes a miss take only an EMPTY slot, which separates foreknowledge from eviction (r114, TODO #86). `--score-band all|decode` picks the scored passes (`all` is what the runtime's `h=` counts — use it for calibration only); `--pool-frac`/`--pool-init heat|sidecar:<path>[, -complement]|id` model `--host-experts pool` as a two-tier cache. | deciding a cache-policy/placement change from one GPU trace instead of one benchmark per hypothesis |
| `build-containers.sh` | Builds (and with `--push`, publishes) the ROCm container images: reconstructs the canonical patched tree from `release.json` (asserting its tree hash), builds `full`/`light`/`server` per ROCm line with the CI-equivalent build args and tags, and prunes the registry (`cleanup-ghcr.py`). `--rocm`, `--arch`, `--targets`, `--prepare-only` for a fast smoke. | release container images — normally run by the worker, not on the primary box |
| `container-watch.sh` | The off-host release watcher (cron/systemd on a build worker): polls for the newest release tag (`v<blocks>-<base>-r<N>`) **by creation date** — the pattern must not pin the block count (a `v12-*` glob silently missed every `v13-*` release, r107/r108) (a version sort mis-orders across a re-base), and if it is not the last successfully built tag, checks it out and runs that tag's own `build-containers.sh --push`. `flock` prevents concurrent builds; a state file stops a tag being rebuilt. | one-time worker setup; runs unattended (`CONTAINERS.md`) |
| `cleanup-ghcr.py` | GHCR version retention, called by `build-containers.sh --push`: deletes every package version outside a rolling `--keep-days` window (default 7) while always keeping the current base's `*-<base>` tags and any tag matching `--keep-tag`. A GitHub bulk-delete throttle is a warning, not a failure. | automatically on `--push`; re-run it to finish a throttled prune |

## Notes

* **`release.json` is the only source of truth** for the fork point, tip/tree, block count and hashes.
  `apply-all.sh`, `validate-set.sh`, `make-patches.sh`, `make-release.sh`, `build-containers.sh`,
  `container-watch.sh` and CI all read it; if a script disagrees with the manifest, the manifest wins.
* **Order matters on a release:** `make-patches.sh` → `make-release.sh` → `validate-set.sh` → tag.  A tag
  cut between the first two ships a manifest that does not match `patches/`.
* **The gates are not interchangeable.**  Coherence (same-seed text) is blind to a uniform prefill shift;
  the prefill-logit gate is blind to decode drift; MTP is blind to both.  `MANIFESTS.md` lists which gate a
  given block needs.
* **Run GPU gates alone.**  Two concurrent GPU jobs change the arena/band layout and invent failures.
* `--help` on any script prints its header comment (the option list) without running anything.
