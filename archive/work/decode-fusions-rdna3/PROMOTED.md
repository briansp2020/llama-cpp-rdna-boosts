# decode-fusions-rdna3 — PROMOTED into block 10 (r175)

Contributor PR [stew675/llama-cpp-rdna-boosts#250](https://github.com/stew675/llama-cpp-rdna-boosts/pull/250)
(`@briansp2020`): seven decode fusions for RDNA3, filed against the r174 tree as
`wip/decode-fusions-rdna3/`.  **Promoted in `v15-6050b467c-r175`** and folded into **block 10** (it owns
the fusion dispatcher, the rope matcher and `mmvq-moe`, and it creates `ggml-cuda-try-fuse.cu` /
`mmvq-moe.cu`, so no earlier block can carry them).

* Numbers, method and the per-patch switch-off deltas: `benchmarks/2026-10-29-rdna3-decode-fusions.md`.
* Release record: `WORKLOG.md` 261029.006.
* Block notes (the invariants a future edit must keep): `patches/README.md`, block 10,
  "The RDNA3 decode fusions (r175, PR #250)".
* Env knobs: `ENVIRONMENT.md` §7.4 — `GGML_CUDA_FA_COMBINE_Q8`, `GGML_CUDA_MOE_GLU_Q8`,
  `GGML_CUDA_ROPE_SET_ROWS_V`, `GGML_CUDA_MOE_RPB_WIDE`, `GGML_CUDA_MWR_FOLD`,
  `GGML_CUDA_ROPE_QK_PAIR`, `GGML_CUDA_MMVQ_ROW_CONCAT`.

**Not taken:** the submitter's `src/models/qwen3moe.cpp` graph reorder (and `LLAMA_QK_ADJACENT`).  It is
worth ~3.3 % of the 30B on top, but it edits upstream model code for every backend to enable a CUDA
fusion, for a retired arch family.  `patches/0008-*.patch` in `patches/` below still carries that hunk —
it is the record of what was submitted, not what ships.

**Fixed at promotion:** the MoE GLU Q8_1 publish now also zeroes each row's padding blocks up to
`MATRIX_ROW_PADDING` (the submitter flagged the gap; the cache arena is reused across graphs).

Nothing here is applied by `patches/` or `scripts/`; this directory is the closed campaign's record.
