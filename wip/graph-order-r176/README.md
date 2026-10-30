Thank you for folding #250 into r175 and for measuring each patch. You noted that the `qwen3moe.cpp` reorder could come back as a backend-side graph ordering. 0001 is that. 0002 and 0003 build on it: 0002 lets the row concat run on RDNA4 as well, and 0003 groups the small GDN projections of Qwen3.6 the same way. All three are bit-identical and have their own env switch.

**0001: V / K projections ahead of the Q projection** (`ggml-cuda.cu`, `ggml_backend_cuda_graph_optimize`; `GGML_CUDA_QK_ROTATE=0` off). This gives the same order the model-file change produced, but in the backend, so no model code changes.

In the decode graph, each layer runs Q MUL_MAT, RESHAPE, RMS_NORM, MUL, ROPE, then the V and K projections, then the K chain. That order blocks two fusions:
- The row concat needs the GEMVs adjacent. Its scan from the Q MUL_MAT stops at the Q RMS_NORM, so only V + K fuse.
- The allocator frees the Q projection after the Q norm and places V / K there, so the overlap test for the Q / K norm+rope pair refuses.

The pass first matches Q MUL_MAT, RESHAPE, RMS_NORM, MUL, ROPE structurally. Then it requires only MUL_MAT / RESHAPE / VIEW nodes that read none of the Q nodes, followed by the K RMS_NORM, MUL, ROPE with the same rope parameters and positions. It then `std::rotate`s those V / K nodes in front of the Q MUL_MAT, in the same function and the same way as your shared-expert rotate. The nodes are the same, on one stream, so the result is bit-identical.
- It is structural only, applies at every ubatch size, runs after the `marks_only` return (so the meta backend's `allocs_only` pass gets it too), and is idempotent.
- V / K now stay allocated across the Q chain, so the compute buffer can grow slightly (at decode, V and K are 512 floats each per layer).

**0002: V, K and Q projections in one launch on RDNA4 too** (`mmvq.cu`; `GGML_CUDA_MMVQ_ROW_CONCAT_RDNA4=0` turns it off on RDNA4 only). This lifts the RDNA3_0-only gate on #250's 0009 row concat.
- **Why RDNA4's row-count dependence does not change a row's result.** #250 kept 0009 off on RDNA4 because `two_rows`, the `GGML_MMVQ_RDNA4_DECODE_GRID` row loop and `ggml_cuda_grid_stall_pad` depend on the row count. These only change which block or loop iteration computes a row. At `ncols_dst == 1`, the warp count (`calc_nwarps_weight`) takes no row argument, `small_m` is RDNA3_0-only, and vdr and the K split are the same for every row. So each row gets the same dot product in the combined launch as in its own launch.
- **One new condition: every segment has an even row count.** The combined launch may take two rows per block, and a 2-row block must not straddle a segment boundary.
- **Needs 0001.** On stock r176, the Q projection and its norm and rope run before V and K, so the scan never sees the three adjacent. Without 0001 this change does not fire (Qwen3-30B: 663 launches per token with or without it).

**0003: GDN beta, alpha and z projections in one launch** (`ggml-cuda.cu`; `GGML_CUDA_MMVQ_CONCAT_REORDER=0` turns it off; RDNA3_0 and RDNA4).
- **What changes.** In the Qwen3.5 / 3.6 GDN layers (measured on Qwen3.6-35B-A3B), `ssm_beta` and `ssm_alpha` (32 rows each) and the `attn_gate` z projection (4096 rows) read the same normed input. Sigmoid, softplus and reshape nodes sit between them, so each runs as its own GEMV launch. The two 32-row launches take about 2 us each, mostly fixed cost. A pass in `ggml_backend_cuda_graph_optimize`, placed right after 0001's V / K rotate, starts at the first quantized MUL_MAT of <= 256 rows. It then finds the later MUL_MATs within 24 nodes that share its src1, weight type, K and row stride (at most one of them larger than 256 rows), and moves up to 2 of them directly after it, so a group is at most 3 MUL_MATs, the row concat's limit. The row concat then runs them as one launch, which gives 60 fewer launches per token on Qwen3.6-35B-A3B.
- **Why bit-identical.** Only nodes whose sources are that activation and a weight move, and they only move earlier. Every op computes exactly what it did before.
- **What it never moves.** It skips any MUL_MAT whose output feeds a CONCAT within 8 nodes. That is attn_qkv: `ggml_cuda_ssm_conv_in_try_fuse` skips the nodes up to its CONCAT, so a node moved in there would never run. It also skips Q8_0 weights, because the GDN prescan matches nodes by position. Qwen3-30B has no quantized weight of <= 256 rows, so the pass changes nothing there.

**Gates** (stock r176 vs the full series, on both cards):
- llama-server greedy, 4 chats x 3 turns: byte-identical replies on both models and both cards, and on the 7900 XTX also with 0001's and 0003's switches off.
- wikitext-2 prefill KLD, 24 chunks, `-c 512`, base = stock r176: KLD 0, PPL identical on both models. 7900 XTX: Qwen3-30B 8.447702, Qwen3.6 5.911128; R9700: 8.464607, 5.925504. Qwen3-30B same top p 99.984 %, the same as stock against its own base.
- 0 GPU faults.
- Two slots decoding together: not usable as a gate on either card. Stock r176 itself gave three different outputs in three runs on both models on the 7900 XTX; the series gave two of stock's outputs and one other on each, as we saw for #250.
- Kernel census, Qwen3-30B decode on the 7900 XTX, per token: 653 -> 558 launches with 0001 (ksplit 169 -> 121, 47 + 47 separate Q / K norm+rope -> 48 `rms_norm_mul_rope_f32_qk_pair`).

**RX 7900 XTX (gfx1100), stock r176** (tree `e990c8611b71`, `-DAMDGPU_TARGETS=gfx1100`, your default env; llama-bench `-fa 1 -p 512 -n 128 -d 0,4096 -r 5`, two alternating rounds after a warm-up, mean of the two; same models):

| | Qwen3-30B tg128 | @4096 | Qwen3.6-35B tg128 | @4096 |
|---|---|---|---|---|
| stock r176 | 187.4 | 166.6 | 155.7 | 153.7 |
| + 0001 | 201.6 (+7.6 %) | 176.8 (+6.1 %) | 156.3 (+0.4 %) | 153.9 (=) |
| + 0001-0003 | 201.5 (+7.5 %) | 176.5 (+5.9 %) | 161.5 (+3.7 %) | 158.5 (+3.1 %) |

0002 is RDNA4-only and measured neutral here (+ 0001 + 0003 without it: 201.5 / 176.8 / 161.3 / 158.3). Prefill (pp512, both depths) is within noise for every arm.

**Radeon AI PRO R9700 (gfx1201), stock r176** (tree `e990c8611b71`, `-DAMDGPU_TARGETS="gfx1100;gfx1201"`, your default env; llama-bench `-fa 1 -p 0 -n 128 -d 0,4096 -r 10`, arms interleaved over 3 rounds, mean of the 3 rounds; Qwen3-30B-A3B and Qwen3.6-35B-A3B Q4_K_M):

| | Qwen3-30B tg128 | @4096 | Qwen3.6-35B tg128 | @4096 |
|---|---|---|---|---|
| stock r176 | 158.4 | 141.2 | 130.3 | 127.5 |
| + 0001 | 163.1 (+3.0 %) | 145.2 (+2.8 %) | 130.4 (=) | 127.5 (=) |
| + 0001-0002 | 167.6 (+5.9 %) | 148.9 (+5.5 %) | 130.3 (=) | 127.5 (=) |
| + 0001-0003 | 167.7 (+5.9 %) | 148.9 (+5.5 %) | 134.2 (+3.0 %) | 131.2 (+2.9 %) |

Greedy server replies (12 multi-turn replies per model) were byte-identical to stock for every arm, with 0 faults. The `+ 0001` row comes from a separate run against the same stock baseline. On this card, two slots decoding together do not give reproducible replies even on stock (several distinct outputs across 3 runs), so we could not use that check as an identity gate here.

Untested: multi-GPU and `-sm tensor` (one card per box right now), and other families with Q / K norm (Qwen3 dense, Gemma 3). Each matcher only fires on its exact chain of nodes.


Files: `0001-*.patch` to `0003-*.patch` (`git format-patch` against the r176 tree, apply in order with `git am`) and this README.
