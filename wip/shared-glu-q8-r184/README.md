A small one, separate from #258 because it touches the MoE FFN rather than attention. Your 1-token fused MoE gate/up+GLU kernel (`mul_mat_vec_q_moe`) already publishes the routed lanes' GLU output as Q8_1 for the down projection (`fusion.glu_q8`). The shared-expert lane (the last `blockIdx.y`, writing `shared_dst`) did not, so on models with a shared expert the shared down MUL_MAT still ran its own `quantize_q8_1`. This patch has that lane publish too. One patch on stock r184; setting `GGML_CUDA_MOE_SHARED_GLU_Q8=0` turns it off.

**0001: the shared-expert lane publishes its Q8_1** (`common.cuh`, `mmvq.cu`, `mmvq-moe.cu`).
- The shared lane writes into its own `q8_1_cache_get` entry, keyed on the shared GLU node (`shared_dst`), which is the key the shared down projection looks up. The down finds it and skips its quantize.
- Its group counters sit after the routed lanes' (lane index = number of routed lanes), so they are disjoint from them. The host checks they fit in `GGML_CUDA_MOE_GLU_Q8_CNT`.
- Same last-block-of-the-group quantize as the routed lanes: it uses `quantize_q8_1`'s 32-lane reduction and rounding, and writes the zeroed padding blocks the same way.
- 1 token only, contiguous `shared_dst` with the routed rows' width, and only when the cache entry is new; otherwise the stock path runs.

**Gates** (stock r184 vs the patch):
- llama-server greedy, 4 chats x 3 turns: byte-identical replies on Qwen3.6-35B-A3B on both cards (on the R9700 also with the switch off), and on Qwen3-30B-A3B on the 7900 XTX (no shared expert, so the path does not run there).
- wikitext-2 KLD, 24 chunks, `-c 512`, Qwen3.6, base = stock (7900 XTX): KLD 0, PPL 5.911128. Decode-path check on the R9700 (`llama-perplexity -b 1 -ub 1`, 8 chunks, base = switch off): mean KLD 0, max 0.000054, the stored-logit floor that stock also shows when compared against itself.
- 0 GPU faults. Kernel census on the R9700, Qwen3.6 decode: 761 -> 721 launches per token (the shared down's `quantize_q8_1` is gone).

**RX 7900 XTX (gfx1100)**, Qwen3.6-35B-A3B Q4_K_M, F16 KV, llama-bench `-fa 1 -p 0 -n 128 -d 0,4096 -r 10`, three alternating rounds, mean:

| | tg128 | @4096 |
|---|---|---|
| stock r184 | 175.45 | 172.55 |
| patch, switch 0 | 176.30 | 172.63 |
| patch | **179.93 (+2.6 %)** | **175.96 (+2.0 %)** |

**Radeon AI PRO R9700 (gfx1201)**, Qwen3.6-35B-A3B Q4_K_M, tg128 `-r 20`, stock / patch alternated six times each: 141.61 -> **143.42 (+1.3 %)**; every patch run is above every stock run.

Notes:
- With #258 applied underneath, the patch applies without conflicts and the gain holds: Qwen3.6 with q8_0 KV on the 7900 XTX, 167.6 -> 171.2 (+2.1 %), @4096 166.1 -> 169.5 (+2.0 %); replies still byte-identical to stock r184 with F16 and q8_0 KV, KLD 0. (#258's folds do not fire on Qwen3.6, so this is the patch's own gain.)
- Models without a shared expert are untouched.
- Untested: multi-GPU and `-sm tensor`.

Files: `0001-*.patch` (`git format-patch` against r184, apply with `git am`) and this README. No rush; happy to run anything else on either card.

(Claude-assisted, as before.)
