# wip/state-read-batch

Batches the device reads of a context state save. The server's context checkpoint of a recurrent / hybrid model reads one sequence's cells of every recurrent-state tensor, about 75 small ranges. Under `-sm tensor` each range was one synchronous read per device, so one checkpoint paid about 150 device syncs and bounce copies. The server takes a checkpoint before each prompt chunk, so every follow-up turn paid this before its first token.

One format-patch on `v12-097cf33d9-r91` (tree `90507dfb`); applied tree `0f2a1218`. Patch sha256 `55e46bab…`.

## What changes

`src/llama-context.cpp` / `.h`, `ggml/src/ggml-backend-meta.cpp`, +141 / −56:
- `llama_io_write_host` queues every deferred read as `ggml_backend_tensor_get_async` on the backend that owns the tensor's buffer, into one pinned staging buffer kept by the context. It then synchronizes each backend once and copies to the caller's buffer.
- The old synchronous path stays for:
  - a state larger than `LLAMA_STATE_ASYNC_MAX_MIB` (default 256, so a whole-slot save never pins gigabytes);
  - a state with a tensor no context backend owns;
  - `LLAMA_STATE_ASYNC=0`.
- The meta backend's `get_tensor_async` asserted `offset == 0` and a single segment, so it could not read a row range. It now shares the synchronous read's split arithmetic, and queues the same per-device reads on the simple backends. Calls that worked before (offset 0, one segment) issue the same copies as before.
- Restores (`llama_io_read_host`) are unchanged. Batching them measured no faster here: the bytes have to be staged into pinned memory either way.

## Gates

Setup:
- 2x R9700 (gfx1201), native Linux, HIP runtime 7.15.26333.
- Stock r91 against r91 + the patch, same build dir; only `libggml-base` and `libllama` differ.
- Three configs:

| | model | split | other |
|---|---|---|---|
| G1 | `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS` + shared Q8_0 MTP head | 2 GPU `-sm tensor` | `-ncmoe 10` |
| G2 | Qwen3.6-35B-A3B Q4_K_M, no MTP | 1 GPU | `-ncmoe 10` |
| G3 | Qwen3.6-35B-A3B Q4_K_M, no MTP | 2 GPU `-sm tensor` | `-ncmoe 10` |

  G1's command is the one from #186:

  ```
  llama-server -m Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf -c 262144 -ngl 99 -ctk q8_0 -ctv q8_0 \
    -cram 2048 -ub 2048 -b 2048 --reasoning-effort low --temp 0.7 --top-k 20 --top-p 0.95 --min-p 0 \
    --spec-type draft-mtp -md mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf -ngld 99 --spec-draft-n-max 3 \
    --spec-draft-p-min 0 -ctkd q8_0 -ctvd q8_0 \
    -sm tensor --n-cpu-moe 10 -ot per_layer_token_embd.*=CPU -lm none -lzm off
  ```

  with `LLAMA_KQ_MASK_DERIVED=0 GGML_SCHED_SYNC_GRAPH_INPUTS=1 LLAMA_MTP_SPARSE=0 LLAMA_MTP_DRAFT_OP_OFFLOAD=0`. G2 and G3 change `-m`, drop the MTP flags and the `-ot` / `-lm` / `-lzm` options; G2 sets `ROCR_VISIBLE_DEVICES=0`.

**Greedy hashes** (400 tokens, prose / code), identical to stock r91:

| config | prose / code |
|---|---|
| G1 | `a7a81f01b3bc` / `4bb43e3339c8` |
| G2 | `047fd057911b` / `231119399578` |
| G3 | `c0cf0739bdab` / `930bcf1f4405` |

**Checkpoint restore:**
- A ~2.2k-token prompt ends in " apple" (A), then the same prompt ends in " pear" (B), same slot, prompt cache on. B restores the checkpoint A left.
- B's 48 greedy tokens (`ignore_eos`) equal a fresh `cache_prompt: false` run of B in every arm, stock and patched, on all three configs.
- A's output differs from B's on G1 and G3, so the check can tell them apart there.

**Faults:** none in 35 servers.

## Timing

**One checkpoint read** (60.1 MiB on G1, 66.7 MiB on G2). Same patched binary with `LLAMA_STATE_ASYNC=0` / default, timed by a local log around the read:

| config | synchronous | batched |
|---|---|---|
| G1, 2 GPU `-sm tensor` | 11.76 ms | 5.5 ms (5.48-5.64 over 8 servers) |
| G2, 1 GPU | 6.64 ms | 6.91 ms |

On one device there is nothing to gain: the reads there were already cheap. The win is the per-device sync under `-sm tensor`.

**Restored request** (the restore check's B, 4 new tokens): G1 166 → 153 ms, G3 51 → 43 ms, G2 35 → 36 ms.

**Time to first token** on G1:
- A 4-turn chat with the prompt cache on and the full history resent each turn:
  - turn 0: a fresh ~2.4k-token document;
  - turns 1 and 2: short follow-ups;
  - turn 3: a ~1.6k-token block pasted as a tool result.
- Streamed, client-side time to first token. Two documents per server, 7 rounds of one server per arm, alternating order (n = 14 per arm and turn).
- A first round was left out: its stock server started cold (31.5k prefill 1592 t/s, against ~2080 for every other server).

| turn | new tokens | stock r91 | + patch | change |
|---|---|---|---|---|
| 0 | ~2.4k | 2226 ms | 2125 ms | −101 ms (median −39, 9 of 14 faster) |
| 1 | ~110 | 819 ms | 792 ms | −27 ms (8 of 14) |
| 2 | 32-34 | 606 ms | 566 ms | **−40 ms (13 of 14)** |
| 3 | ~1.6k | 1398 ms | 1399 ms | +1 ms (median −22, 9 of 14) |

Turn 3 is bimodal on this config, around 1300 or 1600 ms in both arms, so its mean says little. G2 (one GPU) moved within ±17 ms on every turn.

On our hybrid Flash-Next build (r91 + our patches, IQ3_S, `-ncmoe 47`, `-sm tensor`, MTP draft), the same change gave turns 1-3 −26 / −45 / −47 ms (6 paired samples each), with identical output. Four concurrent streams there went +3.5 % (3 pairs, CI +1.0 to +6.1), and decode and prefill were unchanged.

## Not tested

`-sm layer` and 4 concurrent streams on stock r91: with this config both hit the r87 abort reported in #187.
