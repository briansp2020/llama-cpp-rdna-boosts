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

## 3b. The decisive control: the divergence is qwen4exp-specific

Maintainer's control: a **non-qwen4exp MoE with the cache engaged and MTP**, on one GPU.  The
Qwen3.6-35B-A3B Q4_K_M (22.7 GB, embedded `qwen35moe.nextn` MTP head) on one R9700 with `-ncmoe 20`
(so the host-expert cache is live) and the same 32k `mixed30k` prompt:

```
M=/llm/models/Qwen3.6/35B-A3B/Q4_K_M/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf D="" \
GPUS=0 NCMOE=20 CTX=40960 ./widthsweep.sh ~/llama.cpp/build-rocm-hybrid q35q4b
#   none vs n1: first-diff None
#   none vs n3: first-diff None
#   none vs n7: first-diff None        (256 tokens each, acceptance 0.94/0.80/0.57)
```

**Byte-identical.**  So the expert cache, the rejection rollback, the hybrid/spec machinery and the
embedded-MTP path are all sound in general; the divergence lives in the **qwen4exp-specific** layers.

Cross-checks that bound the qwen4exp cause:

* the width-probe (dense **and** sparse QSA, `NCMOE=48`, `-sm tensor`, P=2500 > the 2051 indexer
  threshold) is **PASS** → not a qwen4exp forward-pass width dependence;
* `LLAMA_QSA_SPARSE_FA=0` (dense QSA) still diverges (n1@64, n3@84, n7@112) → not the sparse indexer;
* `GGML_CUDA_GDN_CHUNKED=0` still diverges (n1@112, n3@59) → not the chunked-GDN kernel;
* cache off (`MOE_EXPERT_CACHE_MIB=0`) → n1/n3 pure (n7@147) → the cache is involved;
* `test-recurrent-state-rollback` **PASS** (max diff 0) → the plain recurrent rollback is exact for its
  scenario.

The remaining surface is the qwen4exp **recurrent/conv rewind combined with the live expert cache** —
qwen4exp is the only tested model with recurrent (GDN) layers, and the cache is the only remaining
differentiator the control removes.  (The maintainer notes this exact area — MTP rewind with qwen4exp
and chunked GDN — was historically the hardest to get right.)

## 3c. Multi-sequence control (PR #124's reporter scenario): our tree is unstable too

The PR #124 reporter's test — two concurrent greedy `/completion` requests (`-np 2`, `--kv-unified`),
A long (256 tokens) keeps decoding after B short (96) finishes — run on **our r39+narrow-2 build with no
gather skip** (`tools/multiseq.sh`, GSQ + shared Q8_0 MTP, 2 GPU):

```
A_solo  256 68e7690aa07a     B_solo   96 bb1f48c7d87e
A_conc1 256 ab4c87bdddbc     B_conc1  96 4742a82f994f
A_conc2 256 ca64778b6314     B_conc2  96 5a67e565e819
A_conc3 256 a1446705a718     B_conc3  96 911c61aaabdc
A_conc vs A_solo first-diff: 8/8/8 ; A_conc2 vs A_conc1: 58 ; A_conc3 vs A_conc1: 9
```

Every concurrent repetition differs, and the concurrent output differs from the solo output — i.e. **our
tree reproduces the reporter's instability without PR #124**.  Controls:

* identical under `GGML_CUDA_ALLREDUCE=ce` (the deterministic AR) → not the all-reduce;
* `GGML_CUDA_GDN_CHUNKED=0` does **not** fix it (reps still differ) → not the chunked-GDN kernel alone.

And the run logs the exact condition the reporter saw (pre-existing, not caused by their PR):

```
seq_rm: rollback crossed a batch boundary: seq 1 rollback=2 but the last batch decoded 6136 tokens
        (last pos 18419, n_rs_seq=3, n_rs_batch=4). ... the restored state is wrong
```

So the multi-sequence MTP path has a **pre-existing rollback correctness hole**: a rejection rollback
can land while the last decoded batch was a large (prefill) batch, and the recurrent snapshot path then
reads a snapshot that batch did not write.  `GGML_CUDA_GDN_CHUNKED=0` removes the *substance* of that
hole for the sequential kernel but the instability remains, so there is a second, batching-level part.

This is independent of PR #124 (it is in our tree, which is the reporter's "reference" shape) and it
does **not** by itself explain the single-sequence divergence (the single-sequence width sweeps never
emit this warning).  But it is the same surface the user's hypothesis names: the rewind reads state the
verify did not write.

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
