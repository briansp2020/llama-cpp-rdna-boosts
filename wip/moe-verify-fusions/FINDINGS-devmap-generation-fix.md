# FINDINGS — the `DEVMAP=1` generation-consistency repair (2026-10-10 session)

**Status: the residency-generation repair is IMPLEMENTED and the level-4 validator is clean, but it does
NOT make the device-remap path pure.  The residual `DEVMAP=1` divergence is a deeper graph/cold-path
effect, independent of the map.  Recommendation: keep `DEVMAP=0` the default and `DEVMAP=1` opt-in, and
do NOT serve the 35B-A3B expert split until the split-path divergence is fixed.**  WIP only — `patches/`
and `release.json` untouched.

This continues `HANDOVER-devmap-consistency.md` and `FINDINGS-single-seq-rollback.md` §5b-§5g.

---

## 1. What was implemented (working tree `~/llama.cpp`)

The handover §4 preferred repair — **a residency generation checked at remap build** — plus the
consistency gaps it depends on.  In `ggml/src/ggml-cuda/moe-expert-cache.cu`:

* `table_t` gains `res_gen` (authoritative slot-map generation) and `dev_gen` (the generation the device
  `slot_dev` map reflects).  `res_gen` is bumped on **every** authoritative mutation: `access_locked`
  admission, `apply_prefill_seed_locked`, `apply_prefill_seed_rank_locked` (which previously did not even
  set `slot_dirty`), `stand_down_table_locked`, `free_table_buffers_locked`, `moe_cache_shrink_arena`,
  `moe_cache_release_arena`, `moe_cache_evict_slab_range`, and the device-policy kernel launch.
* `dev_gen = res_gen` is set on **every** publication of `slot_dev`: the host-policy upload in
  `moe_cache_promote_host` (both branches), the `build_policy_descs_locked` seed, `alloc_table_locked`'s
  fresh map, the device-policy kernel, and — new — the **eager `moe_cache_update_host` pass**, which
  previously mutated the host map and never republished `slot_dev`, leaving the armed device path to start
  from a stale/`-1` map.
* Every device-remap consumer (`moe_cache_take_over`, `moe_cache_get_table`, `moe_cache_get_slot`,
  `moe_cache_table_serves`, `moe_cache_sibling_down`, `moe_cache_stage_table`, `moe_cache_inplace_ready`)
  now requires `dev_gen == res_gen`.  On a mismatch the take-over declines and the **eager host path**
  serves the op and republishes both.
* `moe_cache_evict_slab_range` now clears `slot_dev` (the per-table `stand_down_table_locked` always did;
  the slab path did not).  `used_dev` is zeroed at allocation (it was read back uninitialized on the
  arming pass).  A `g_devmap_gen_stale` counter logs the first (and every 10000th) guard decline.
* The level-4 validator now skips tables with no arena/slots (never served) and tables the guard declines
  (`dev_gen != res_gen`), and logs `slots`/`res_gen`/`dev_gen`.

The generation guard is a **no-op in steady state** (it fired exactly **once** per server run, on the
arming token) and a hard safety net otherwise.

## 2. Reproduction (before the fix)

`wip/moe-verify-fusions/tools`, GSQ-IQ3_XXS + shared Q8_0 MTP head, `-ncmoe 48`, `-sm tensor`, 2 GPU,
`GGML_CUDA_ALLREDUCE=ce`, `/tmp/srr/mixed30k.txt` (sha256 `69624f4d…`), plain `--spec-type none`,
`NP=80`:

| A/B | before (buggy build, `DEVMAP=1`) |
|---|---|
| MIB8000 vs MIB14000 | first-diff **59** |
| level-4 validator (`DEVPOLICY=0`, MIB=8000) | 3976 lines: 3688 `dev=-1 host>=0` (safe cold) + **288 `dev>=0 host=-1`** at `after-rearm` |

The 288 dangerous lines are on **stood-down or guard-declined** tables (`slots=200`, `dev_gen = res_gen-1`,
`slot_dirty=1`) — i.e. exactly the state the guard was added for.  Level-3 (`SLOT-CONTENT-MISMATCH`): **0**.

## 3. After the fix — the default is pure, the device path is not

| check | `DEVMAP=0` (new default) | `DEVMAP=1` (fixed) |
|---|---|---|
| 2-GPU budget sweep MIB8000 vs MIB14000 | **None** | **64** |
| 1-GPU budget sweep MIB8000 vs MIB14000 | **None** | (59 before) |
| `n_rs_seq` 0 vs 3, MIB=14000 | **None** | **9** |
| field `none` vs `draft-mtp n3` | **None** | **9** |
| level-4 validator (guard-aware) | **0 / 0** | **0 / 0** |

The `DEVMAP=1` divergence is invariant across every sub-knob we tried — it is **not** the map, the cold
path, the seed, or the policy:

| `DEVMAP=1` sub-path | first-diff |
|---|---:|
| default (KSLOT=1, DEVPOLICY=0, seed on) | 64 |
| `PREFILL_SEED=0` | 9 |
| `KSLOT=0` | 64 |
| `ADMIT=0` (always fill; no admission reject) | 64 |
| `DEVPOLICY=0` + `n_rs_seq` 0v3 | 9 |

`ADMIT=0` eliminating the admission rule does not remove it, and level-3 proves the slot bytes are the
host master's; so the residual is the **per-token resident/cold classification** (the device path applies
the current routing one token late, so first-touch experts are cold) exposing a path that is not
bit-identical to the eager two-pass promotion.  §5d's "resident/cold reads are geometrically and
byte-identical" holds at the address level but does not hold for the produced logits.  Making the device
path pure would require eager (pre-graph) promotion — i.e. the host routing readback the device path
exists to remove.

## 4. Perf A/B — `DEVMAP=0` vs fixed `DEVMAP=1` (per §8)

1000-token decode + 32k prefill, GSQ + shared Q8_0 MTP head, 2 GPU `-sm tensor`, `ce`, MIB=14000,
interleaved and warm (two runs each):

| config | decode t/s (rep1/rep2) | prefill t/s (rep1/rep2) | acceptance |
|---|---:|---:|---:|
| MTP n3, `DEVMAP=0` | 71.53 / 72.37 | 1303.8 / 1587.6 | 0.77017 |
| MTP n3, `DEVMAP=1` | 83.26 / 85.66 | 1249.0 / 1250.2 | 0.81776 |
| plain, `DEVMAP=0` | 41.78 / 41.81 | 1470.3 / 1471.5 | — |
| plain, `DEVMAP=1` | 47.84 / 47.94 | 1366.4 / 1368.1 | — |

The `DEVMAP=0` row reproduces the handover (§3/§5g) exactly.  The MTP decode delta is **acceptance-
confounded** (the divergent trajectory accepts 0.8178 vs 0.7702), so the acceptance-immune comparison is
the **plain** row: the device path is ~14 % faster on decode and ~7 % slower on prefill.  **Per §8 it
does not clearly beat the eager path on both**, and it is not pure → keep the eager path the default and
the device path opt-in.

## 5. The 35B-A3B split (handover §9) — the premise is stale and the split diverges

With the current tree, `-sm tensor -ncmoe 99` on the 35B-A3B Q8_0, the host expert tables are **already
split**, contrary to §5e:

```
CACHE_GEOM layer=0 ffn_gate_exps  n_experts=256 expert_bytes=278528 host_bytes=1114112 src_off=0      split_axis=1
CACHE_GEOM layer=0 ffn_gate_exps  n_experts=256 expert_bytes=835584 host_bytes=1114112 src_off=278528 split_axis=1
CACHE_GEOM layer=0 ffn_down_exps  n_experts=256 expert_bytes=278528 host_bytes=1114112 src_off=0      host_pitch=544 split_axis=0
CACHE_GEOM layer=1 ffn_gate_exps  n_experts=256 expert_bytes=557056 host_bytes=1114112 src_off=557056 split_axis=1
```

So there is nothing to force — the splitter already declines nothing for the 35B.  With the split active
the 35B `none` vs `draft-mtp n3` **diverges at 82** (MIB8000 and MIB14000, default `DEVMAP=0`, `ce`),
whereas the handover's mirrored 35B control was `None`.  The `GGML_META_SPLIT_COPY=0` "mirrored" control
is **not a usable reference**: the cache registers **0 tables** (`moe_cache_report: tables=0`), because a
mirrored expert copy is device-resident and never reaches the host-expert cache, and the embedded-MTP
acceptance collapses to **0.01102** (`none` vs `n3` is `None` only because the MTP arm degenerates to
plain).  So the handover §2/§5e 35B control was effectively a cache-off/identity configuration, not a
truthful split-vs-mirrored comparison.  Split vs mirrored perf (1000 tok, MTP n3, `GGML_META_SPLIT_COPY`
1 vs 0): split 2369 prefill / 74.07 decode / 0.71789 acceptance; mirrored 1914.6 / 9.36 / 0.00134 (the
9.36 is just the rejected-draft overhead).

**The 35B `none` vs `n3` divergence is itself CACHE-INDEPENDENT.**  With `MOE_EXPERT_CACHE_MIB=0` (cache
off) it still diverges, at **8** (acceptance 0.79464, `ce`).  So the §9 premise — that the 35B's only
problem is the split and `none == n3` is otherwise a given — is wrong on two counts: the tables are
already split, and the plain-vs-MTP divergence is a separate `n_rs_seq`/MTP defect that the cache does not
cause.  Consequently the 35B **cannot** be validated on purity grounds ("prove `none == n3`") until that
defect is fixed, and its split must not be served.  **Do not serve the 35B split; the mirrored
alternative is also non-functional for this model's embedded MTP.**

## 6. Reproducers

```bash
cd wip/moe-verify-fusions/tools
export LD_LIBRARY_PATH=/opt/rocm-7.14.1-gfx120X/lib:${LD_LIBRARY_PATH:-}
export M=/llm/models/Qwen3.8/Flash-Next/GSQ-IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf
export PROMPT=/tmp/srr/mixed30k.txt NP=80 GPUS=0,1 GGML_CUDA_ALLREDUCE=ce SM=tensor
# default is pure:
for MIB in 8000 14000; do MOE_EXPERT_CACHE_DEVMAP=0 MOE_EXPERT_CACHE_MIB=$MIB \
  ./probs.sh ~/llama.cpp/build-rocm-hybrid d0_$MIB /tmp/srr/mixed30k.txt 80 --spec-type none; done
# device path still diverges:
for MIB in 8000 14000; do MOE_EXPERT_CACHE_DEVMAP=1 MOE_EXPERT_CACHE_MIB=$MIB \
  ./probs.sh ~/llama.cpp/build-rocm-hybrid af_$MIB /tmp/srr/mixed30k.txt 80 --spec-type none; done
# guard-aware level-4 (expect 0/0):
MOE_EXPERT_CACHE_DEVMAP=1 MOE_EXPERT_CACHE_DEVPOLICY=0 MOE_EXPERT_CACHE_VALIDATE=4 \
  MOE_EXPERT_CACHE_MIB=8000 ./probs.sh ~/llama.cpp/build-rocm-hybrid v1_val2 /tmp/srr/mixed30k.txt 60 \
  --spec-type none >/dev/null 2>&1
```

## 7. Pointers

* `HANDOVER-devmap-consistency.md` — the task; §4/§8/§9 amended with this session's result.
* `FINDINGS-single-seq-rollback.md` §5b-§5g — the prior evidence and the `DEVMAP=0` stand-down.
* `session-devmap-fix.diff` — the previous session's diff; `session-devmap-gen-fix.diff` is the
  cumulative working-tree diff including this session's generation-consistency changes (regenerate before
  archiving).
* `TODO.md` #50/#51/#52.
