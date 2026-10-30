**0001: router MUL_MAT + topk-moe in one launch** (`topk-moe.cu`, `topk-moe.cuh`, `ggml-cuda-try-fuse.cu`; `GGML_CUDA_ROUTER_TOPK=0` off). For 1 token, when the router would run as `mul_mat_vec_f` with its 256-thread block, `topk_moe_cuda` computes the logits itself. Each block handles one router row exactly as `mul_mat_vec_f` does: the same thread-to-column map, unroll, `ggml_cuda_mad`, warp reduce and cross-warp butterfly. The last block to finish (a fence, then one atomic on a self-resetting counter) runs the unchanged selection. The router launch goes away.
- The gate re-checks the dispatcher's route: F32 weight and activation, contiguous, 64 / 128 / 256 experts, wave32, not MMB, `ggml_cuda_should_use_mmvf`, and the same block-size choice. Anywhere else the router runs as before.
- It does not fire with a bias or delayed softmax.
- As with the MoE down fold's counter, a launch aborted mid-kernel would leave the counter non-zero.

**Setup:** RX 7900 XTX as in wip/graph-order-r176. Stock r176 + that series' 0001 vs the same + this 0001.

**Gates:** server greedy byte-identical on both models, with the switch on and off; prefill KLD, Qwen3-30B, wikitext-2 8 x 512: -0.000000, same top p 100 %; 0 faults. Census: the 48 `mul_mat_vec_f` router launches per token are gone.

An independent review of the logic found one issue, now fixed. The router phase read the activation before `ggml_cuda_pdl_sync()`. On NVIDIA with programmatic dependent launch, it could read the activation before the norm that writes it had finished. `ggml_cuda_pdl_sync()` compiles to nothing outside CUDA Hopper+ PDL builds, so the HIP numbers above are unaffected. The fold still fires (traced), and the greedy replies were re-checked after the fix (same on both models). Following the same review, the gate also checks that MMB would not take the router, as the dispatcher does.

**Throughput** (two alternating rounds, -r 10, tg128):

| model | stock r176 | + graph-order 0001 | + graph-order 0001 + this 0001 |
|---|---|---|---|
| Qwen3-30B | 189.07 / 188.72 | 203.08 / 202.48 | 206.85 / 206.82 |
| Qwen3.6 | 157.27 / 157.41 | 157.61 / 157.79 | 159.48 / 159.46 |

This 0001 on top of graph-order 0001: +2.0 % / +1.1 %. Prefill flat.


File: `0001-Router-MUL_MAT-1-token-F32-mul_mat_vec_f-256-thread-.patch` (`git format-patch` against the r176 tree, applies on its own, `git am`).
