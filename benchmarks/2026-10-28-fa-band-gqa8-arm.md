# 2026-10-28 — the band's missing head-256 **GQA-8** arm (Qwen3.6-35B-A3B), and a probe that can tokenize a corpus

Box: 3x R9700 gfx1201 (`HIP_VISIBLE_DEVICES=0,1,2`), ROCm 10.1.0, `~/bin/build-llama-rocm` on the release tree
(`test-logits-width-probe` built from it).  Every arm is sequential (no parallel benches).  Closes `TODO.md`
#102 (accepted limitation since r134 / issue #224).

## The arm

`/llm/models/Qwen3.6/35B-A3B/Q8_0/Qwen3.6-35B-A3B-Q8_0.gguf` (`qwen35moe`, 41 layers, 11 of them
full-attention) loads as `n_head = 16`, `n_head_kv = 2`, `n_embd_head_k = n_embd_head_v = 256` — i.e.
**head 256 at GQA ratio 8**, which is exactly the band predicate's unmeasured ratio
(`ggml_cuda_fattn_band_wmma_applies`: `Q->ne[0] == 256 && V->ne[0] == 256 && 4 < gqa_ratio <= 12`, RDNA4,
`n_q <= 8`, native-quant or f16 K/V).  It has been in the fleet all along; the r134 record's "no head-256
GQA-8 model exists" was wrong.

**gemma-4-26B-A4B is not that arm** (the model the Qwen-suggested path pointed at).  Its metadata is
`gemma4.attention.head_count = 16`, `head_count_kv = [8,8,8,8,8,2,…]`, `key_length = 512`,
`key_length_swa = 256`: the SWA layers are head 256 at ratio **2** and the full-attention layers ratio 8 at
head **512**.  Neither enters the band.  The same holds for 31B (32/16 at 256, 32/4 at 512), 12B (16/8 at 256)
and E4B (8/2 = ratio 4 at head 512).

Drafters used (both new to the fleet, from the HF links in the session that closed this item):

| drafter | path | bytes | sha256 (first 16) | geometry |
|---|---|---:|---|---|
| DFlash (block 16) | `Qwen3.6/35B-A3B/DFlash/Q8_0/qwen36-35b-a3b-dflash-Q8_0.gguf` | 421 060 320 | `b9749d1793b8c907` | `dflash`, 6 layers, `block_size 16`, head 32/8 (GQA 4) at head_dim 128, `target_layers [5,6,7,8,9,10,11,12]` |
| DFlash2 (block 8 + selector) | `Qwen3.6/35B-A3B/DFlash2/Q8_0/Qwen3.6-35B-A3B-DFlash2-Q8_0.gguf` | 570 468 544 | `69bc81f5344dfeb6` | `dflash`, 6 layers, `block_size 8`, `selector_rank 256` / `selector_top_k 16`, `conv_kernel_size 2`, same `target_layers` |

Both are `--spec-type draft-dflash` (`-md`); the fork detects the DFlash2 shape from the selector/conv keys.
The target's own embedded MTP head (`qwen35moe.nextn_predict_layers = 1`) covers `draft-mtp` with no second
file.

## Width purity (`test-logits-width-probe`, `P=4096`, `ubatch=512`)

Prompts: `prompts/prose-rdna-boosts.txt` (sha256 `fabdec65…`, 5246 tokens) and
`prompts/kld-corpus-35b-in-domain.txt` (sha256 `0818e79e…`, 25814 tokens — the model's own corpus; usable only
after the token-buffer fix below).

| prompt | `KV` | split | `RS` | verdict |
|---|---|---|---|---|
| prose | f16 | layer | 0 | `PASS`, worst maxdiff **0** |
| prose | q8_0 | layer | 0 | `PASS`, **0** |
| prose | q8_0 | layer | `from_w` | `PASS`, **0** — hashes identical to the `RS=0` arm |
| prose | q8_0 | tensor | 0 | `PASS`, **0** |
| corpus | q8_0 | layer | 0 | `PASS`, **0** |
| corpus | f16 | layer | 0 | `PASS`, **0** |
| corpus | q8_0 | tensor | `from_w` | `PASS`, **0** |

`row0_row1_hashes` are one value across W=1..8 in every arm (e.g. prose/q8_0/layer row 0
`dad0d9bebe9966b8`, row 1 `7631becaba06859d` from W=2 on) — no row-0 split between W=4 and W=5, which is the
failure shape the r126 geometry produced.

**Controls.**  `GGML_HIP_FA_BAND_WMMA=0` (band off, tile kernel) changes every hash (row 0
`dad0d9be…` -> `78b15bea…`) and is also width-pure, which is what proves the default arm was *on the band*.
`GGML_HIP_FA_BAND_WIDE=0` — the two-tile launch instead of the band's own 64-column block — is **bit-identical**
on every arm, which is the r134 `static_assert` observed as a number (both blocks are `np = 4`,
`nbatch_fa/K2/V2` 64, `nstages_target`, `Q_in_reg` alike).

**Reachability of both band blocks** (`rocprofv3 --kernel-trace`, `-c 6144`, `draft-mtp --spec-draft-n-max 7`,
q8_0 KV, `-sm layer`).  Kernel symbols, demangled as `<DKQ, DV, ncols1, ncols2, …, fetch_ahead, band_wide>`:

| kernel | meaning | dispatches |
|---|---|---:|
| `flash_attn_ext_f16<256,256,4,8,f,f,f,t,f>` | band, 32-column block (query widths 1..4) | 70 |
| `flash_attn_ext_f16<256,256,8,8,f,f,f,t,t>` | **band, 64-column block (query widths 5..8)** | **55** |
| `flash_attn_ext_f16<256,256,8,8,f,f,f,f,f>` | prefill 64-column table row | 121 |

So widths 5..8 really run the band's own 16-warp row at `P=4096` (the launcher's gate is `K->ne[1] >= 4096`
and the probe's decode batch pads to `GGML_PAD(4104, 256) = 4352`), and `P=256` — the historical setting —
cannot reach it.

## Text purity (`-n 512`, greedy, `--temp 0 --seed 42`, q8_0 KV, prose prompt)

`plain == draft-*` at GQA 8, the same instrument the MTP purity work uses:

| split | plain | `draft-mtp` | `draft-dflash` (block 16, `n_max 5`) | `draft-dflash` (DFlash2, `n_max 5`) |
|---|---|---|---|---|
| `layer` | 1888 chars `1c5f9e08e5e4` | identical | identical | identical |
| `tensor` | 2011 chars `c5b2c23d064a` | identical | — | identical |

The layer/tensor hashes differ from each other as they must (a split mode is its own numeric reference,
`GREEDY-PURITY.md` §45); the comparison that matters is inside each row.  The drafters are live, not inert:

| arm | acceptance | mean accept len | generation |
|---|---:|---:|---:|
| plain | — | — | 82.9 t/s |
| `draft-mtp` | 0.736 (351/477) | 3.21 | 137.8 t/s |
| `draft-dflash`, block 16 | 0.545 (373/685) | 3.72 | 130.2 t/s |
| `draft-dflash`, DFlash2 | 0.543 (373/687) | 3.70 | 131.9 t/s |

## The probe could not tokenize a corpus (block 06 amendment, r166)

`tests/test-logits-width-probe` tokenized into a fixed `std::vector<llama_token>(8192)`, so the natural prompt
for a model-specific arm — the model's own in-domain KLD corpus (25 814 tokens) — failed with
`tokenize failed (need 25814)` and a 16 KiB-class corpus was silently off limits.  It now asks the tokenizer
for the required length first (`llama_tokenize(..., nullptr, 0, ...)` returns `-n`) and sizes the buffer from
that, which is the whole change.  Nothing about the probe's verdict logic moved.

## Release gates (r166, `v15-6050b467c-r166`)

`validate-set.sh` strict 15/15, tree `cbcb8808…`; `gate-per-block-build.sh --changed-from 809a27419` **2/2**
(`all` targets, block 06 + the tip); `gate-prefill-logits.sh` **mean KLD 0.000000 / same-top-p 100.000 %**;
same-seed coherence unchanged; Protocol A (dense 27B UD-Q4_K_XL, `-c 102400`, bf16 KV, `-n 3000`,
`prompts/prose-rdna-boosts.txt`): `plain == draft-mtp` byte-identical in **both** split modes —
`-sm tensor` 50.9 -> 100.7 t/s (acceptance 0.608, 1937/3185), `-sm layer` 25.0 -> 51.6 t/s, text
11 158 / 11 148 chars.
