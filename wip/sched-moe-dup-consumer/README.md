# wip/sched-moe-dup-consumer

One small scheduler change. When a prompt chunk produces fewer output rows than it has tokens, qwen4exp's last layer has two `MUL_MAT_ID` consumers of the same expert weights: the gathered logits tail and the full-row MTP export tail. When that layer's experts are host-resident, as they are under the default tail-first `-ncmoe` placement, both consumers land in one split. `sched_copy_input` then counts two consumers and copies the layer's whole expert tables for every such chunk.

With the patch, the second consumer starts a split of its own. The existing code that registers the weights as an input of a later split as well then stages them again for that consumer's own routing, which is computed by then. Each consumer goes through the expert cache like every other host-resident layer.

One format-patch on `v12-097cf33d9-r89` (tree `ca9598a2`); applied tree `872e3114`. Patch sha256 `c9914e67…`.

## What changes

`ggml_backend_sched_split_graph` in `ggml-backend.cpp`, +10 / −1:
- In the branch for cache-managed routed ops (a `MUL_MAT_ID` within the backend's `moe_cache_band`, with the cache active), a `MUL_MAT_ID` whose weights are already an input of the current split now sets `need_new_split`, as the first routed op of a new layer does.
- Nothing else changes. A layer's gate, up and down still share one split, so the gate+up+GLU fusion still sees them together. Splits change only where one split would have held two consumers of the same weights.

## Where it shows

Each follow-up turn of a chat with the prompt cache on is processed in up to three chunks (split at the last user message and 4 tokens before the end), and each chunk drops rows. So every short turn paid the full-table copy, once or twice, before its first token. On our hybrid build (IQ3_S, see below), a ~40-token turn spent about 160 of its ~500 ms before the first token in those copies.

## Gates

Setup:
- 2x R9700 (gfx1201), native Linux, HIP runtime 7.15.26333.
- Stock r89 against r89 + the patch, same build dir; only `libggml-base` differs.
- `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS` with the shared Q8_0 MTP head, `-ncmoe 10` (tail-first: layers 38-47 host-resident, expert cache auto, 100 % residency). Exact server command:

  ```
  llama-server -m Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf -c 262144 -ngl 99 -ctk q8_0 -ctv q8_0 \
    -cram 2048 -ub 2048 -b 2048 --reasoning-effort low --temp 0.7 --top-k 20 --top-p 0.95 --min-p 0 \
    --spec-type draft-mtp -md mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf -ngld 99 --spec-draft-n-max 3 \
    --spec-draft-p-min 0 -ctkd q8_0 -ctvd q8_0 \
    -sm tensor --n-cpu-moe 10 -ot per_layer_token_embd.*=CPU -lm none -lzm off
  ```

  with `LLAMA_KQ_MASK_DERIVED=0 GGML_SCHED_SYNC_GRAPH_INPUTS=1 LLAMA_MTP_SPARSE=0 LLAMA_MTP_DRAFT_OP_OFFLOAD=0` (left over in our launcher from older tests) and nothing else. The greedy hashes use `temperature 0`. `-sm layer` and the 35B-A3B runs change only `-sm` and `-m` (no MTP for 35B-A3B).
- `-ncmoe 1` is not useful here: the cache does not fit beside the work buffers, stock disables it and streams the experts, and the patch never fires.

**Greedy hashes** (400 tokens, prose / code), identical to stock r89:

| config | prose / code |
|---|---|
| IQ3_XXS, 2 GPU `-sm tensor`, MTP (4 servers, 2 per arm) | `a7a81f01b3bc` / `4bb43e3339c8` |
| IQ3_XXS, 2 GPU `-sm layer`, MTP | `946ffc37b143` / `e2f7603679a0` |
| Qwen3.6-35B-A3B Q4_K_M, 2 GPU `-sm tensor`, plain decode | `c0cf0739bdab` / `930bcf1f4405` |

MTP draft acceptance on a 400-token greedy request matched within run-to-run variation (patch 273/377, stock 273/377 and 274/374).

**Faults:** none in the `-sm tensor` and 35B-A3B runs. See the note below for `-sm layer`.

## Time to first token

A 4-turn chat with the prompt cache on and the full history resent each turn:
- turn 0: a fresh ~2.4k-token document;
- turns 1 and 2: short follow-ups;
- turn 3: a ~1.6k-token block pasted as a tool result.

Streamed, client-side time to first token. Two documents per server, two servers per arm, alternating order (n = 4 per arm and turn). IQ3_XXS, `-sm tensor`:

| turn | new tokens | stock r89 | + patch | change |
|---|---|---|---|---|
| 0 | ~2.4k | 2072 ms | 2060 ms | −12 ms (−1 %) |
| 1 | ~110 | 757 ms | 731 ms | −26 ms (−3 %) |
| 2 | 32-63 | 580 ms | 469 ms | **−111 ms (−19 %)** |
| 3 | ~1.6k | 1347 ms | 1305 ms | −42 ms (−3 %) |

- Prefill (31.5k tokens): stock 2071 / 2026 t/s, patch 2077 / 2055 t/s; decode unchanged within noise.
- On our hybrid Flash-Next build (an r82 base with our patches, IQ3_S, `-ncmoe 47`, `-sm tensor`, MTP draft), where the last layer's tables are larger, the same change gave turn 1 −82 ms, turn 2 −132 ms and turn 3 −87 ms (8 paired samples each, 7-8 of 8 faster). Decode +0.2 %, prefill +0.5 %, both within noise, with identical output.

## Notes

- **4-concurrent:** not measured on r89. With this config, r87 and r89 abort (r88 not tested) at 4 concurrent streams with `HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION` (stock and patched alike). r86 and r89 with r87's one-line `n_tokens_prev > 8` drop change reverted run clean. Reported separately with the repro.
- **`-sm layer`:** the same abort also hits single-stream on `-sm layer` with r89, in both arms (r86 and the revert: clean). So the `-sm layer` row above has hashes only.
- An alternative we tried first and dropped: running qwen4exp's last-layer tail once on the full rows when a chunk has no output rows at all. It removes only the mid-prompt copy; the final 4-token chunk (one output row) still has two consumers.
