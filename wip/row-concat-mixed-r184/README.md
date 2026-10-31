A small follow-up to the RDNA4 row concat from #252 that you folded into r177. That concat runs the Q / K / V projections of a 1-token decode in one ksplit launch, but only when all three weights have the same type. In the K-quant M files `attn_v` is Q6_K in about half the layers while Q / K are Q4_K, Q5_K or IQ4_XS (Qwen3-30B-A3B Q4_K_M: 24 of 48 layers; Qwen3.6-35B-A3B: 6 of its 10 attention layers; Llama-3.2-1B UD-Q4_K_XL: all 16), so in those layers V still runs as its own launch. This patch lets the concat take such a pair. One patch on stock r184, RDNA4 only; setting `GGML_CUDA_MMVQ_ROW_CONCAT_MIXED=0` turns it off.

**0001: row concat across a K-quant / Q6_K pair** (`common.cuh`, `mmvq.cu`).
- `mul_mat_vec_q_ksplit` becomes a `__device__ __forceinline__` body plus the same `__global__` wrapper. A new `mul_mat_vec_q_ksplit_cat2<type, type_b>` runs the Q6_K body, with that segment's own row stride, for the Q6_K segment's row blocks, and the launch type's body for the rest.
- Both types take the same nwarps on RDNA4, and each row keeps its own one-warp dot product, so every row computes what its standalone launch computes.
- The type is chosen once per block, so the decode grid loop is off for these launches. The matcher takes either graph order (after the #252 reorder V comes first) and requires the launch tensor to be the largest segment, since the row guards use its row count.
- Pairs {Q4_K, Q5_K, IQ4_XS} x Q6_K. On other architectures the cat2 body is compiled out and the stock path runs.

**Gates** (Radeon AI PRO R9700, gfx1201):
- llama-server greedy, 4 chats x 3 turns. Patch vs stock r184: byte-identical on Qwen3-30B-A3B, Qwen3.8-27B Q4_K_M, gemma-4-26B MXFP4_MOE, Qwen3.5-4B Q8_0 and Qwen3.8-27B NVFP4-MTP. Switch on vs off (our working tree): byte-identical on Qwen3-30B-A3B, Qwen3.6-35B-A3B and Llama-3.2-1B.
- Decode-path KLD on Qwen3-30B (`llama-perplexity -b 1 -ub 1`, switch on vs off): mean KLD 0, max 0.000055 (the stored-logit floor).
- 0 GPU faults. Kernel census, Qwen3-30B decode: 24 fewer launches per token, one per Q6_K-V layer.
- RX 7900 XTX (gfx1100): builds cleanly, replies byte-identical to stock r184 on Qwen3-30B-A3B and Qwen3.6-35B-A3B (the path is compiled out there).

**R9700, Qwen3-30B-A3B Q4_K_M**, F16 KV, tg128 `-r 20`, stock / patch alternated six times each: 168.39 -> **170.84 (+1.5 %)**. (The commit message quotes an earlier run on our working tree, 169.74 -> 171.73, +1.2 %.)

With #258 applied underneath, it applies without conflicts and the gain adds on top (Qwen3-30B, R9700, `-n 128 -d 0,4096 -r 10`, three rotated rounds; replies byte-identical to stock r184 with F16 and q8_0 KV):

| Qwen3-30B-A3B | stock r184 | #258 | #258 + this (vs #258) |
|---|---|---|---|
| q8_0 KV tg128 | 150.3 | 167.5 | **169.8 (+1.4 %)** |
| q8_0 KV @4096 | 138.3 | 152.8 | **154.6 (+1.2 %)** |
| F16 KV tg128 (see note) | 169.9 | 169.2 | **172.8 (+2.1 %)** |
| F16 KV @4096 | 149.8 | 151.1 | **153.0 (+1.3 %)** |

(The F16 rows are noisier: stock and #258 read low in the first round; without it the gain over #258 is +1.5 % / +1.0 %.)

Notes:
- Qwen3.6 gets 6 fewer launches per token, too few to show in tg128.
- Untested: multi-GPU and `-sm tensor`.

Files: `0001-*.patch` (`git format-patch` against r184, apply with `git am`) and this README. No rush; happy to run anything else on either card.

(Claude-assisted, as before.)
