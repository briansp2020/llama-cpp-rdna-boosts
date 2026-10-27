# graph-replay-keys (PR #230) — the graph cache key must name the layer, not only the activation

**Status: SHIPPED (2026-10-26, r149) as a fold inside block 09.**  Closes `TODO #107`.  The residual
(3-device dense plain decode) is `TODO #109`.

* Submitted as `briansp2020`'s **PR #230** (`wip/graph-replay-keys/`, one patch + a README, offered
  "as data"), assessed and re-cut here.
* The submitter's patch is kept verbatim beside this file
  (`0001-ggml-cuda-graph-cache-keys-that-tell-per-layer-split-subgraphs-apart.patch`) so the assessment
  can be re-derived; it is **not** what shipped.
* Rule: `patches/README.md` (block 09), `ENVIRONMENT.md` (`GGML_CUDA_GRAPH_KEY_WEIGHT`),
  `benchmarks/2026-10-26-graph-replay-keys.md`, `WORKLOG.md` 261026.007.
* Superseded note: `../graph-replay-placement/README.md` (the r144 scoping note this closes).

## 1. What the submitter got right

Their diagnosis is correct, and it corrects the r144 note.  The note's §2 listed three causes in order
(node `src[]` object addresses, then the inline `name`, then "a genuine source `data` move").  The real
single cause is a **cache-key collision**: under a split mode every layer's subgraph begins on the *same*
activation buffer, so the `(first node data, n_tokens)` key sent all of them to one entry; each layer was
then compared against the **previous layer's** properties.  Because `src[]` sits *before* `data` in
`ggml_tensor`, the first differing byte of a collision is the `src` array (offset 152) — which is what the
r144 probe reported and why it read as "the Meta rebuild moves source objects".  The "genuine `data`
move" that appeared once `src`/`name` were canonicalised was the neighbouring layer's weights, i.e. the
same collision again.  Their fix — key on the weight a subgraph reads — is the right one.

Their measurement also reproduces here, on the note's own shape, and larger:

| 2x R9700 x8, Flash-Next IQ4_NL, `-sm tensor -ncmoe 48`, 2000-token decode | stock r148 | + key fix |
|---|---:|---:|
| `graph_compute` census | `direct=549499 replay=31696 capture=149 properties_changed=452439` | `direct=3882 replay=574826 capture=2636 properties_changed=28` |
| generation t/s (3 interleaved pairs) | 37.7 / 37.7 / 37.6 | 42.5 / 42.9 / 42.8 (**+13.5 %**) |
| `GGML_CUDA_DISABLE_GRAPHS=1` (the r144 trap) | 37.3 | 37.3 |
| both kill switches | — | stock exactly (census and t/s) |

## 2. What did not survive assessment

1. **The identity-free property compare is not worth shipping.**  The PR's second half blanks
   `src[]`/`view_src`/`buffer` before the per-node `memcmp`, and it is the half the submitter themselves
   flagged ("fusion checks look at source object identity").  Measured with the key fix held constant
   (i.e. the PR's own `GGML_CUDA_GRAPH_PROPS_STRICT=1` arm against the default): **+0.2-0.3 %**
   (Flash-Next 2 GPU plain 42.6 vs 42.8 t/s, `properties_changed` 340 vs 28; the MTP arm 112.8 vs 113.1,
   `properties_changed` 2054 vs 1588), and on a 3-device dense split it only moves the capture count
   (1161 -> 915) without changing the result.  It is dropped: the key is the fix, and that comparison is
   the replay's only safety check.
2. **`ggml_cuda_graph_mem_freed` coverage was incomplete.**  The PR bumps it at three of the free sites in
   `moe-expert-cache.cu` (arena backing, `remap_dev`, the device move).  The decode-band heat tally
   (`moe_cache_heat_kernel`) receives `t.heat_dev` **and** `t.heat_dec_dev` as raw launch parameters, and
   `free_heat_dev_locked` frees them without a bump — reachable for an *identity* table (no arena free, no
   remap free, so neither of the PR's two bumps fires).  Same for the routing staging (`prs_dev`) and the
   legacy `cudaFree(t.arena)` arm of `moe_cache_release_arena`.  Shipped form: bump in the **helper** that
   returns the memory (`free_arena_backing`, `free_heat_dev_locked`) plus one bump in each teardown/move
   function, so a new free site inherits it.
3. **`GGML_CUDA_GRAPH_PROPS_STRICT` parsed as presence** (`getenv(...) != nullptr`), so `=0` enabled it.
   Moot: the knob is not shipped.
4. `GGML_CUDA_GRAPH_KEY_WEIGHT` was read by two separate `static` lambdas (one per helper).  Shipped as one
   helper.
5. The new helpers were inserted between the existing documentation block for `ggml_cuda_graph_get_key` and
   the function itself, so the existing comment documented the wrong function.  The shipped form keeps the
   comment with its function and appends the new rationale.
6. **Not run by the submitter** (all run here): the prefill-logit KLD gate (**0.000000 / 100.000 %**, as
   predicted — prefill graphs are multi-token and skip the graph path), the `ROCBLAS_USE_HIPBLASLT=0`
   deterministic arm (used for every text comparison), an arch sweep, and a longer run (2000-token arms;
   the submitter's were 400).

## 3. What the assessment found that the PR did not

**Replay is not free, and on one shape it is a loss.**  With the key fix in place, a 3-device
tensor-split **dense** model's plain decode gets **slower**: 27B Q8_0 3 GPU `-sm tensor` `llama-bench
tg128` **42.0 -> 38.4** (-8.6 %), Q4_K_XL 52.4 -> 49.1, depth 16384 the same (42.1 -> 38.3).  It is not
capture churn (the capture count is one-shot per key and the deficit survives `-n 512`), not the chunk
size, not the async input ring and not the scheduler's event path; `GGML_CUDA_ALLREDUCE=internal` removes
it (42.2) while `=nccl` is worse still (35.6), and neither `GGML_CUDA_AR_SMALL_BYTES` arm moves it — so
the interaction is with the hybrid NCCL comm layer (merely having it initialised), or with the node-level
stream overlap a captured graph cannot express.  Carried as `TODO #109`.

Everything else measured flat or better, which is why the key still ships **default-on**:

| arm | stock r148 | + key fix | delta |
|---|---:|---:|---:|
| Flash-Next IQ4_NL 2 GPU x8 `-ncmoe 48` plain, n2000 | 37.7 | 42.8 | **+13.5 %** |
| Flash-Next IQ4_NL 3 GPU `-ncmoe 48` plain, n1000 | 37.6 | 42.1 | **+12.0 %** |
| Flash-Next 2 GPU `-ncmoe 48` + MTP n3, n2000 | 111.5 | 113.1 | +1.4 % |
| Flash-Next 3 GPU `-ncmoe 48` + MTP n3, n1000 | 107.1 | 109.2 | +2.0 % |
| 27B Q8_0 1 GPU Protocol A plain / draft-mtp | 29.6 / 70.2 | 29.7 / 70.0 | flat, acceptance identical |
| 27B Q8_0 2 GPU tensor `tg128` | 32.80 | 32.59 | flat |
| 27B Q8_0 3 GPU `-sm layer` `tg128` | 17.43 | 17.44 | flat |
| 27B Q8_0 3 GPU tensor + MTP n3 | 42.8 | 42.9 | flat, identical text |
| 35B-A3B 1 GPU `tg128` / Protocol A plain / draft-mtp | 101.8 / 82.0 / 174.5 | 101.7 / 81.2 / 174.1 | flat, acceptance identical |
| 27B Q8_0 3 GPU tensor plain `tg128` | 42.0 | 38.4 | **-8.6 %** (`TODO #109`) |

Purity (`ROCBLAS_USE_HIPBLASLT=0`, greedy, seed 42): **text byte-identical in 10/10 config pairs**
(27B 1 GPU, 2 GPU `-sm layer -ts 59,41`, 3 GPU tensor; 35B-A3B 2 GPU `-ncmoe 40`; Flash-Next IQ4_NL 2 GPU
`-ncmoe 48` + MTP; Gemma4 E4B with and without its MTP draft), and the draft acceptance trajectory is
bit-identical (27B 3 GPU MTP 0.82386 = 0.82386 with the same 1478/1794 accepted/generated; 35B-A3B
0.75118 = 0.75118, 1591/2118).

## 4. Gates run for the shipped form

| gate | result |
|---|---|
| same-seed greedy text, 10 config pairs (above) | identical |
| `draft acceptance` / accepted counts, 27B 3 GPU + 35B-A3B | identical |
| `scripts/gate-prefill-logits.sh` | mean KLD **0.000000**, same-top-p **100.000 %** |
| `test-backend-ops -o` FLASH_ATTN_EXT, GATED_DELTA_NET, FLASH_ATTN_QSA, MUL_MAT, MUL_MAT_ID, TOP_K_MASK, HC_MIX, HC_COMBINE, HC_COMBINE_MIX | 9/9 pass (6450/931/55/26/28/2445/30/20/6 cases) |
| `scripts/gate-park-idle-slots.sh` | PASS (park arm byte-identical to lazy-off on all four requests) |
| `scripts/gate-hc-fusions.sh` (Flash-Next IQ4_NL, chain-mixer class) | the five hard HC arms identical; the `dsv4_pre_chain` arm fails **identically on the r148 stock build** (pre-existing, unchanged by this patch) |
| `scripts/check-gpu-faults.sh` (journal + the 2000-token log) | clean |
| Protocol A (`-n 3000`, prose): 27B Q4_K_XL 1 GPU, 35B-A3B 1 GPU | MTP >= plain, acceptance healthy (0.80905 / 0.75118) |
| `scripts/gate-per-block-build.sh --fast --changed-from 29b9d1197` | 3/3 (block 09, 12, tip) |
| `scripts/validate-set.sh` | green (15 blocks, tree `423a59a6`) |

## 5. The shipped diff

Three files, +114/-6, all in **block 09**:

* `ggml/src/ggml-cuda/common.cuh` — `ggml_cuda_graph_key` gains `first_weight_ptr` and `shape_sig`, in
  `operator==` and in the hash.
* `ggml/src/ggml-cuda/ggml-cuda.cu` — `ggml_cuda_graph_key_by_weight()` (one env read),
  `ggml_cuda_graph_first_weight()`, `ggml_cuda_graph_shape_sig()`, and `ggml_cuda_graph_get_key()` fills
  them.
* `ggml/src/ggml-cuda/moe-expert-cache.cu` — `ggml_cuda_graph_mem_freed()` at the helpers and at each
  teardown/stand-down/device-move.

The tree is otherwise r148's.
