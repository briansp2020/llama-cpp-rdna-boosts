# 2026-10-29 — the derived kq mask for batches of several sequences (r172, PR #247)

**Question.** Block 05's derived kq mask replaces the packed `n_kv × n_q` F16 mask for "the shapes it can serve", and
until r172 that meant **one sequence per batch**.  A `--parallel > 1 --kv-unified` server holds several: two slots
prefilling or verifying together, or one slot's prefill next to another's decode.  And because `sched_reserve()` measures
with `n_seq_max` sequences, such a context reserved `n_kv × n_ubatch × 2` bytes on each device **and again pinned on the
host** whether or not a batch ever held two sequences.  The contributor's change gives sequence `k` of the ubatch its own
position range (`k * 2^23`, `KQ_DERIVED_SEQ_STRIDE`) so the single predicate `tok_lo <= cell_pos <= tok_hi` stays exact
and no kernel changes; this record is the independent re-measurement on the release tree.

## What was measured, and on what

| box | device | ROCm | build |
|---|---|---|---|
| `soar` (primary) | 1, 2 and 3x R9700, gfx1201 | 10.1.0-gfx120X | r171 tip + PR #247 (`git am -3`, clean), then the folded r172 tree |

Models: `Qwen3.8/27B/Q8_0` and `Q4_K_XL` (dense, plain attention cache), `Qwen3.5/4B/Q8_0` (hybrid attention + GDN),
`Qwen3.6/35B-A3B/Q3_K_M` (MoE, `-ncmoe 30`), `Gemma4/12B/QAT` and `gpt-oss-20b/mxfp4` (both iSWA), `Qwen3.8/Flash-Next/
GSQ-IQ3_XXS` (qwen4exp hybrid + QSA).  Every arm pins `HIP_VISIBLE_DEVICES`.

## 1. The memory the reserve was wasting (server shape)

`llama-server -c 245760 -ub 2048 -b 2048 --parallel 2 --kv-unified -fa on -ctk q8_0 -ctv q8_0 -ngl 99 -t 8`,
27B UD-Q4_K_XL, 1 GPU, memory read after "listening on":

| arm | device after load | server RSS |
|---|---|---|
| `LLAMA_KQ_MASK_DERIVED_MULTISEQ=0` (the r171 behaviour) | 27 512 999 936 B | 1 964 904 kB |
| default (derived multi-sequence) | **26 405 838 848 B** | **982 872 kB** |
| delta | **-1.031 GiB** | **-0.937 GiB** |

That is the contributor's 1.03 / 0.94 GiB exactly (their box: 28.63 -> 27.60 GiB device, 2.47 -> 1.54 GiB RSS with a
DFlash2 draft attached).  Packed mask size for the shape: `245760 × 2048 × 2 B = 960 MiB` per side.

Under `-sm tensor` (2 GPU, `HIP_VISIBLE_DEVICES=0,2`) the device win is the same (-1.1 GiB on GPU 0) but the **host** win
disappears (RSS 3 199 900 -> 3 183 848 kB, -16 MB): the 0.94 GiB pinned saving is a single-device property of where the
mask's host copy lives.

## 2. Prefill is neutral, not slower — in the three shapes where the batch-wide skip could have cost

| arm | packed | derived |
|---|---|---|
| two concurrent 16K prompts, 1 GPU, wall for the pair | 26.96 s | 27.23 s |
| two concurrent 16K prompts, 2 GPU `-sm tensor` | 17.07 s | 17.09 s |
| 112K-resident decode on slot 0 beside a 16K prefill on slot 1 | 13.97 s | 13.96 s |

The mechanism that could have made it a loss: the derived path's fully-masked-group skip
(`flash_attn_kq_derived_blocks`, §37) is a *batch-wide* window, so with several sequences shifted into their own ranges
the union covers every batch cell and no group is skipped, while the packed path classifies per query tile and can skip
the other slot's cells.  The measurement says the host fill the derived form removes
(`n_kv × n_ubatch × 2` bytes written per ubatch) is the larger term at these sizes, and the contributor's Windows
`-ncmoe` MoE arm (two 8K prompts, 27-31 s -> 9.7-10.1 s) is the same trade at a scale where per-ubatch compute is small
enough for the fill to dominate.

## 3. The oracle: the derived predicate equals the packed fill, cell by cell

`LLAMA_KQ_MASK_DERIVED_VERIFY=1` (the contributor's diagnostic, not for delivery) rebuilds each derived ubatch's mask
with the packed fill (`set_input_kq_mask`) and compares visibility per (token, cell), and recounts the shared-cell
counter against a full scan.  Run over `llama-perplexity -f wiki.test.raw -c 512 -b 2048 -ub 2048 --kv-unified`
(4 sequences per batch):

| model | cache kind | derived multi-sequence ubatches | mismatching cells | PPL (packed arm) |
|---|---|---|---|---|
| 27B Q8_0, 1 GPU | plain | 4 of 4 | **0** | identical per chunk |
| 27B Q8_0, 2 GPU `-sm tensor` | plain (split) | 4 of 4 | **0** | 7.3819 vs 7.3920 (cross-device class) |
| Qwen3.5-4B Q8_0 | hybrid (GDN) | 2 | **0** | 10.0272 both |
| Gemma4-12B QAT | iSWA (both caches) | 4 | **0** | 316.0556 both |
| gpt-oss-20b MXFP4 | iSWA + sinks | 4 | **0** | 436.0274 both |

`llama-parallel -np 4 -pps` (a shared system prompt through `llama_memory_seq_cp`) showed the counter non-zero
(`n_multi = 282`, recount 282), the derived form correctly withheld from the multi-sequence ubatches, and identical
output.  The randomized counter test (`512 cells × 6 sequences`, 400 000 operations of every kind) passes locally:
counter == recount after every operation, shared cells in 385 495 states.

## 4. Logit and text gates (folded build, `0a0c359de` -> r172)

| gate | result |
|---|---|
| `gate-prefill-logits.sh --ctx 512 --chunks 40 --ub 2048 --ab "LLAMA_KQ_MASK_DERIVED_MULTISEQ=0"` (f16) | mean KLD **0.000000**, same-top-p **100.000 %** — PASS |
| the same with `--kv q8_0` | **0.000000 / 100.000 %** — PASS |
| `gate-prefill-logits.sh` (release standard, single sequence, f16) | **0.000000 / 100.000 %** — PASS |
| greedy text on/off, dense 27B / 35B-A3B `-ncmoe 30` / hybrid 4B | byte-identical (only the t/s line differs) |
| `plain == draft-mtp`, 27B Q4_K_XL, 2 GPU `-sm tensor`, `-n 400`, seed 42 | byte-identical (2624 B both) |
| `gate-park-idle-slots.sh` (3 GPU tensor, Flash-Next IQ3_XXS + MTP, 3 slots) | **PASS**, 6 park + 2 unpark moves traced, park == lazyoff on all four requests |

## 5. Where it buys nothing (recorded so the next reader does not re-derive it)

**Flash-Next (qwen4exp)**: `llama-server -sm tensor -c 40960 --parallel 2 --kv-unified`, `-ot per_layer_token_embd.*=CPU
-lm none -lzm off`: RSS 30 932 856 kB (packed) vs 30 933 612 kB (derived), VRAM identical to the byte.  Its QSA layers
use an `allow_derived == false` builder, so the packed mask is created and filled for that graph regardless — the
reserve term is correctly left packed and the win is a dense/plain-cache one.

**A server that shares prefixes** (`llama_memory_seq_cp`) latches `kq_mask_multiseq_seen`, re-reserves once with the
packed mask and then keeps it: correct, one growth, and the 1 GiB is returned.  The latch itself could not be fired on
this box (this `llama-server` never `seq_cp`s across slots while a slot is busy, and a prompt-split child runs as
single-sequence ubatches — the 24696-token split at `--parallel 4` produced 13 single-sequence derived ubatches), so its
firing is documented from the contributor's `-pps` evidence and from code review; `kq_mask_derivable()` re-runs the whole
predicate per ubatch, so correctness does not depend on it.

**Record:** `WORKLOG.md` 261029.003, `GREEDY-PURITY.md` §70, `patches/README.md` (block 05),
`archive/work/kq-derived-multiseq/`, `ENVIRONMENT.md` (`LLAMA_KQ_MASK_DERIVED_MULTISEQ`).
