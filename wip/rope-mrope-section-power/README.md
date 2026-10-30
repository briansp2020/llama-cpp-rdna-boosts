Thank you for assessing #249 on both of your boxes, and for confirming the `powf` mechanism at the kernel level. As you asked, this PR carries only the rope patch, so the record maps 1:1. Nothing is waiting on you: it is already folded into r174.

`wip/rope-mrope-section-power/0001-...patch` is #249's 0001, byte for byte, on its original base r171 (tree `2f5870713e94`). The README repeats 0001's description, gates and numbers from #249. It also points to your r174 assessment (WORKLOG 261029.005).

---

From #249 (patch 0001 only):

**0001: compute the M-RoPE section power once** (`rope.cu`: `rope_multi` and the fused `rms_norm_mul_rope_f32`). Every section arm computed the same `powf(theta_scale, i0/2)`; only the position index differs. Adjacent lanes take different arms (`sector % 3`), so the divergent arms were serialized within a wave and `powf` executed once per arm taken. It is now computed once before the branch: same input, same value, and when no arm matches `theta_base` still stays 0. Traced on Qwen3.6-35B-A3B decode, the K / Q norm+rope kernels go from 7.96 / 7.71 to 6.16 / 5.97 us.

**Setup:** RX 7900 XTX (gfx1100), 1 GPU, Qwen3.6-35B-A3B Q4_K_M (unsloth). Stock r171 (tree `2f5870713e94`, matches `release.json`) vs the same + 0001, both built with `-DAMDGPU_TARGETS=gfx1100`, your default env.

**Gates** (run on #249's 0001 + 0002 together):
- wikitext-2 perplexity / KLD (24 chunks, `-c 512`, against logits from a plain upstream HIP build): identical to the last digit, PPL 5.920938, KLD 0.009509.
- llama-server greedy, 4 chats x 3 turns: byte-identical replies (743 tokens).
- 2 concurrent requests decoding together: stock r171 itself gave three different outputs over three runs (the batch composition depends on timing); the patched build gave the same three outputs as stock.
- 0 GPU faults.

**Throughput**, `llama-bench -fa 1 -r 5`, t/s, two alternating rounds:

| arm | tg128 | tg128 @ d4096 | pp512 |
|---|---|---|---|
| stock r171 | 153.64 / 153.30 | 151.58 / 150.80 | 5143 / 5078 |
| + 0001 | 154.47 / 154.14 | 152.14 / 151.82 | 5138 / 5097 |

About +0.4 to 0.6 % decode; prefill within noise. Your r174 assessment measured +0.65 % on Qwen3.6-35B-A3B on gfx1100 and noise on RDNA4.

File: `0001-rope-compute-the-M-RoPE-section-power-once-rope_mult.patch` (`git format-patch` against the r171 tree, apply with `git am`).
