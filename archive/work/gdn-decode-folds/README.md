Thank you for the careful review of #249's 0002, and especially for spelling out what would change your mind. This is that version. The 1-token direct conv now fires only where `gdn_conv_norm_match() > 0`, and only where the decode-only prescan would not take the conv. The `return 0` in the `ssm_conv_in` stand-down is now `continue`. On its own it is still neutral, as you measured. What it adds is that the snapshot fold (#177) and the conv-state gather fold now run at decode too, and that is where the gain is. Four patches on stock r176, each with its own env switch.

**0001: the direct conv also writes the recurrent snapshot tails** (`gdn-conv.cu`, `gdn-conv.cuh`, `ggml-cuda-try-fuse.cu`, `cpy-batch.cu`; `GGML_CUDA_DISABLE_GDN_CONV_SNAP=1` off). This is #177 ported onto r176. Where the direct conv runs, its blocks also write the 3-column snapshot tails from state / x, so the CONCAT and the snapshot CPYs launch nothing.
- It is taken only when nothing between the CONCAT and the conv writes over state / x or touches a snapshot destination, and the destinations do not overlap each other.
- New since #177: a batched-copy run stops at a snapshot CPY the direct conv already writes, so `cpy_batch` can never swallow one.
- Without 0003, this only changes the 2..255 verify band.

**0002: conv-state gather fold** (`ggml-cuda-try-fuse.cu`, `ggml-cuda-try-fuse.cuh`, `ggml-cuda.cu`, `gdn-conv.cu`; `GGML_CUDA_FUSE_GDN_CONV_GATHER=0` off). `build_rs`'s GET_ROWS of the conv cache, feeding a CONCAT that the direct conv takes with 0001's fold, is deferred, like your GDN state gather. The direct kernel then reads the cache row itself, with the row id from the gather's src[1].
- Each channel's three state values are loaded into registers before the snapshot writes, since one of those writes is the write-back to the same row.
- Conditions: one sequence, at most 16 tokens, the CONCAT is the gather's only reader, and only folded snapshot CPYs write the cache before the conv.
- A CONCAT over a deferred gather that the direct conv does not take aborts, rather than reading an unwritten GET_ROWS.
- It uses the existing gather index (`gdn_gather_index_mode != 0`). `GGML_CUDA_FUSE_GDN_STATE_GATHER=0` turns this fold off too.

**0003: direct conv at 1-token decode, gated** (`gdn-conv.cu`, `gdn-conv.cuh`, `mmvq.cu`, `mmvq.cuh`, `ggml-cuda.cu`; `GGML_CUDA_FUSE_GDN_CONV_DECODE=0` turns off the 1-token path alone). On RDNA4 / RDNA3_0 a 1-token GDN conv takes the direct path only if:
- `gdn_conv_norm_match() > 0`, so it also takes the q / k l2 norm pair; and
- `ggml_cuda_ssm_prescan_match_at()` says the prescan fusion would not take that conv. The prescan launcher is split into that predicate plus the launch.

How the decision stays consistent:
- The gate uses data pointers, so it lives only in the compute-time matchers. Every compute-time site calls them: the CONCAT, the SSM_CONV, the `ssm_conv_in` stand-down, and 0001 / 0002's checks. So all of them decide the same way.
- `graph_optimize` runs before allocation, so it uses the structural match (`ggml_cuda_gdn_conv_pattern_at_concat`). The alloc deps it adds therefore cover every conv the compute-time matchers can take.
- The `ssm_conv_in` epilogue stands down only for the concats the direct conv takes. Elsewhere, for example dense Q8_0 models where the norm match refuses, it keeps writing the CONCAT inside the in-proj GEMV as before.

**0004: shared-expert gate GEMV + SIGMOID / MUL / ADD in one launch** (`mmvf.cu`, `mmvf.cuh`, `common.cuh`, `ggml-cuda-try-fuse.cu`; `GGML_CUDA_FUSE_SHEXP_GATE_SIGMOID=0` off). The chain is `ffn_out = moe_out + ffn_shexp * sigmoid(w . y)`, where `w` is one row. It now runs in the `mul_mat_vec_f` launch the gate would take anyway: the same instantiation and block size. Every block recomputes the gate dot exactly as the plain launch does, then applies `sigmoid_mul_add_f32`'s arithmetic to its slice.
- It is taken only where the dispatcher would pick `mul_mat_vec_f`: not MMB, not FWHT, and BF16 / F32 weights (with F16, `mul_mat_vec_f` can pick a half accumulator, which the fused path does not support).
- It applies to at most 8 tokens.
- The gate, the sigmoid and the product must have no other reader, since none of them is stored.

**Gates** (stock r176 vs all four):
- llama-server greedy, 4 chats x 3 turns: byte-identical replies.
  - 7900 XTX: Qwen3-30B-A3B and Qwen3.6-35B-A3B, with all four switches on and with all four off; Qwen3.6 also with only `GGML_CUDA_FUSE_GDN_CONV_DECODE=0`.
  - R9700: Qwen3-30B-A3B, Qwen3.6-35B-A3B, the dense Qwen3.8-27B and Qwen3.5-4B Q8_0.
- wikitext-2 prefill KLD, 24 chunks, `-c 512`, base = stock r176, 7900 XTX: KLD 0, PPL identical. Qwen3-30B 8.447702, Qwen3.6 5.911128.
- 0 GPU faults.
- Kernel census, Qwen3.6 decode on the 7900 XTX, per token: 912 -> 785 launches. In 29 of the 30 GDN layers, `k_get_rows_float` + `ssm_conv_f32` + `rms_norm_scale_pair_f32` + `cpy_scalar` become one `gdn_conv_direct_kernel<true, 128, true>`. The remaining layer fails the norm match and stays on the stock path. The 40 `sigmoid_mul_add_f32` launches are gone.

**RX 7900 XTX (gfx1100), stock r176** (tree `e990c8611b71`, `-DAMDGPU_TARGETS=gfx1100`, your default env; llama-bench `-fa 1 -p 512 -n 128 -d 0,4096 -r 5`, two alternating rounds after a warm-up, mean of the two). Intermediate arms are the full build with switches off: "0003 alone" = SNAP=1 + SHEXP=0, "0001-0003" = SHEXP=0:

| Qwen3.6-35B-A3B | tg128 | @4096 |
|---|---|---|
| stock r176 | 156.2 | 153.6 |
| 0003 alone | 157.6 (+0.9 %) | 155.4 |
| 0001-0003 | 165.7 (+6.1 %) | 163.0 |
| 0001-0004 | 167.4 (+7.2 %) | 165.0 (+7.4 %) |

Qwen3-30B-A3B (no GDN, no shared-expert gate) is unchanged, at 187.2 in every arm. Prefill is within noise.

**Radeon AI PRO R9700 (gfx1201), stock r176** (`-fa 1 -p 0 -n 128 -d 0,4096 -r 10`, arms interleaved over 3 rounds, mean of the 3 rounds (Qwen3.8-27B: of the last 2; its first round was disturbed)):

| | tg128 |
|---|---|
| Qwen3.6, stock r176 | 129.9 |
| Qwen3.6, 0001-0004 | 137.2 (+5.6 %; @4096 +5.6 %) |
| Qwen3.6, 0001-0004 with `GGML_CUDA_FUSE_GDN_CONV_DECODE=0` | 131.7 (+1.4 %) |
| Qwen3.8-27B Q4_K_M (dense), stock r176 / 0003 alone / 0001-0003 / 0001-0004 | 31.83 / 31.81 / 32.51 / 32.35 (tg64, -r 5) |
| Qwen3.5-4B Q8_0 (your #249 case), stock r176 / 0003 alone / 0001-0004 / 0001-0004 with `..._DECODE=0` | 102.76 / 102.84 / 102.86 / 102.81 (tg256, -r 5) |

On the dense 27B, 0003 alone costs nothing and the full series is +1.6 %. On Qwen3.5-4B Q8_0, your -1.1 % case, every arm is within noise: the 1-token path does not fire there (traced: 494 launches per token with and without the series, no `gdn_conv_direct_kernel`), so the `ssm_conv_in` epilogue keeps its CONCAT as on stock. Greedy replies on the 4B are byte-identical to stock, and its prefill KLD sits at the base file's floor for every arm, stock included.

Notes:
- Host cost: 0001-0003 add a few structural checks per GDN layer per graph evaluation, each with a scan over the graph's readers, and 0004 adds three reader scans per shared-expert layer. This cost is paid when CUDA graphs are off or recapturing. The numbers above include it.
- Switch coupling: `GGML_CUDA_FUSE_GDN_CONV_VERIFY=0` and `GGML_CUDA_FUSE_GDN_CONV_NORM=0` also change the 2..255 band, which is why 0003 has its own switch.
- If you take 0001, #177 can be closed; it is the same fold on r80.
- Untested: multi-GPU and `-sm tensor` (one card per box right now).

Files: `0001-*.patch` to `0004-*.patch` (`git format-patch` against the r176 tree, apply in order with `git am`) and this README.
