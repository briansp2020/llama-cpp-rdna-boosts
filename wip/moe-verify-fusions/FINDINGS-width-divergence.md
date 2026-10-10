# FINDINGS — the plain-vs-MTP divergence (2026-10-12 session)

**Status: the original hypothesis is REFUTED and the search space is sharply reduced.**  The model
forward pass is provably width-pure; the divergence is introduced by the host-expert-cache / MTP
machinery and requires **rejected draft tokens**.  Still WIP.

## 1. The probe: the forward pass is width- and RS-pure

`tests/test-logits-width-probe` (extended this session with `NCMOE` / `SPLIT`) prefills a 256-token
prefix and then decodes a W-token batch for W=1..8, requiring row 0 (and every shared row) to hash
identically across W.  On **GSQ-IQ3_XXS**, `probe9k.txt` (3130 tokens), P=256, ubatch=512:

| config | row0 hash (all W) | width purity |
|---|---|---|
| all-GPU, layer split | `b17cde35665d45ed` | **PASS** (maxdiff 0) |
| `NCMOE=48` host experts, layer split | `b17cde35665d45ed` | **PASS** |
| `NCMOE=48`, **`-sm tensor`** | `d81701810d2a6c34` | **PASS** |
| `NCMOE=48`, tensor, `RS=0/1/3/7/from_w` | identical for every RS | **PASS** |

So the attention/QSA, GDN, MoE and routed-expert kernels are **bit-identical across verify widths**, and
`n_rs_seq` does not touch the logits at all.  The handover §5 candidate ("the cache-band verify-width
kernel arithmetic") is **not** a forward-pass width dependence.

## 2. The divergence needs rejections

Field config (GSQ 2-GPU, `-sm tensor -ncmoe 48`, 32k prompt, `return_tokens`):

| prompt | acceptance | none vs n1 | none vs n3 | none vs n7 |
|---|---:|---|---|---|
| `prose30k.txt` (degenerate, model copies) | **1.00000** | None | None | None |
| `mixed30k.txt` (non-degenerate) | 0.87/0.67/0.45 | idx 9–116 | idx 9–65 | idx 64 / None |

With acceptance 1.0 the context advances exactly as plain and the streams are **byte-identical**; the
divergence appears only once the draft is rejected and the target's KV + recurrent state must be
**rolled back**.  This is why the r39 width gate (`b00fdf534227`) passed — it is an acceptance-1.0
prompt.

## 3. Control matrix (all on GSQ, mixed30k, current build)

| control | none vs n1 | none vs n3 | none vs n7 |
|---|---|---|---|
| narrow-2 default | 89 | 9 | None |
| quick-fix A/B (`GGML_CUDA_SLAB_NARROW2=0`) | 9 | 64 | 64 |
| `GGML_CUDA_DISABLE_FUSION=1` | 116 | 65 | None |
| `GGML_CUDA_GDN_CHUNKED=0` | 112 | 59 | None |
| `MOE_EXPERT_CACHE_MIB=0` (cache off) | **None** | **None** | 147 |
| `GGML_CUDA_SLAB=0` (cache on, slab off) | **None** | (run failed) | 9 |

* The divergence **disappears for n1/n3 when the expert cache is off** → the cache machinery is the
  dominant source.
* Fusions off and chunked GDN off only **move** it → neither is the sole cause (the #50 flip is one
  component, not the whole story).
* The probe tests only the **streaming** host-expert path (it never arms the expert-cache arena), so the
  arena path — `mul_mat_vec_q_moe` reading the cache via the slot remap — is the untested surface.

## 4. What this implies for the fix (recommended next steps)

1. **Extend the probe to run with the expert cache arena armed** (two contexts, or a cache-warmed
   single context), so the arena-vs-streaming path divergence can be reproduced at logits level in
   seconds instead of a 15-minute 32k-token text run.
2. Then decide whether the arena's routed-expert MMVQ is bit-identical to the streaming path at the
   verify widths (`GREEDY-PURITY.md` §39), or whether a partial-residency state makes the two
   paths' **weight bytes** differ (repack/head-pad/scale layout).
3. The recurrent/GDN rollback snapshot is the second suspect (it is what makes rejections matter);
   the `seq_rm` "rollback crossed a batch boundary" warning did **not** fire, so the simple
   stale-snapshot case is not it — the rollback path needs the same controlled treatment.

This reframes item (c): it is **not** a kernel-arithmetic width dependence, it is a cache-path
determinism problem that rejections expose.  If the arena path is made byte-identical to streaming
(§39's claim, currently qualified), the width divergence and a large part of the acceptance spread
should both collapse.
