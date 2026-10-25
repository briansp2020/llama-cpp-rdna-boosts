# rebase-2026-10b — re-cut the 15-block set onto `6050b467c`

**Status: CLOSED — shipped as `v15-6050b467c-r138` (2026-10-25).**  This directory is the campaign record
(moved out of `wip/` on release, per the WIP rule).  Read it with
[`DECISIONS.md`](DECISIONS.md) (every collision and the call that was made) and
[`../rebase-2026-10/`](../rebase-2026-10/README.md) (the previous re-base, whose procedure and GOTCHAS still
apply — this campaign follows them).

## The move

| | old | new |
|---|---|---|
| base | `3f03b1772` (2026-10-24 18:40) | **`6050b467c`** (2026-10-25 19:46) |
| where | personal fork `master` | the same fork, pulled to current upstream |
| gap | — | **21 commits**, 98 files, +7451/−1226 |
| delivery | `release.json` r137, 15 blocks, tree `5d6a19af` | `v15-6050b467c-r138`, 15 blocks, tree `35090ba36c` |
| blocks | 00-14 (block 14 temporary) | **unchanged numbering, no block merged or split** |

The gap is small (one day of upstream) but it is *dense*: it contains two features that duplicate delivery
blocks.  Treat it as a semantic re-base, exactly as the October re-base was.

## Procedure actually used

```sh
# ~/llama-scratch is the long-lived scratch worktree (scripts/scratch-tree.sh); NEVER a new source path
# (a new path is a full cold build: ccache keys on the absolute source paths).
cd ~/llama-scratch
git branch -f rb2-old e8bd52069                 # r137 chain, for reference
git checkout -B rb2 e8bd52069
git rebase --onto 6050b467c 3f03b1772 rb2       # stops in block 00; resolve, continue, repeat
```

Conflicts, in order: block 00 (`mmq.cuh` tiling geometry), block 04 (`ggml-cuda.cu` rope mode set,
`rope.cu` argument order + BF16 dst, `test-backend-ops.cpp`), block 07 (`mmq.cu` J selector),
block 08 (`unary.cuh`, `mmq.cuh` ×13, `mmq.cu` ×4, `ggml-cuda.cu` ×2), block 10 (`ggml-cuda.cu` the
dispatcher move, `mmq.cu`, `mmq.cuh`, `llama-model.cpp`), block 11 (`qwen4exp.cpp` ×6 +
`llama-context.cpp`).  Blocks 01/02/03/05/06/09/12/13/14 replayed clean.

Then the documented flow: `make-patches.sh ~/llama-scratch 6050b467c <tip>`, `make-release.sh --base …
--base-tree … --tip … --tree … --release v15-6050b467c-r138`, `validate-set.sh` (strict 15/15 on a fresh
tarball, applied tree == `release.json.tree`).

## Gates (gfx1201, ROCm 10.1.0, this box)

* **Per-block build gate: 15/15** (`gate-per-block-build.sh --repo ~/llama-scratch --base 6050b467c
  --tip <tip>`, fresh build dir per commit).  97 s warm; ~35 min on the first (cold) pass.
* **Prefill-logit gate: PASS** — mean KLD **0.000531**, same-top-p **98.863 %** against the *r137-recorded*
  base.  No re-record was needed: the drift is the base move's own rounding, well inside 0.005 / 98 %.
* Op tests (`HIP_VISIBLE_DEVICES=0,1,2`): `FLASH_ATTN_EXT` **6450/6450** (upstream added cases),
  `GATED_DELTA_NET` 55/55, `MUL_MAT` and `MUL_MAT_ID` pass, `FLASH_ATTN_QSA` pass, `TOP_K_MASK` pass,
  `HC_MIX` pass.
* Coherence, `plain == draft-mtp` **byte-identical** on three arms: dense Qwen3.5-4B Q8_0 3 GPU
  `-sm tensor`; Qwen3.6-35B-A3B Q4_K_M 2 GPU `-sm tensor -ncmoe 40` (the host-expert/slab arm);
  Qwen3.8 Flash-Next GSQ-IQ3_XXS 3 GPU `-ncmoe 48 --load-mode none --lazy-mode off` with the MTP head.
* MTP Protocol A (27B UD-Q4_K_XL, 1 GPU, `-n 2000`, `--spec-draft-n-max 3`, reasoning pinned):
  R **1.81×**, P **2.34×**, C **2.47×**, K **2.71×** (r137 recorded 1.81 / 2.41 / 2.47 / 2.70).
* Greedy text **differs from the r137 build** on every arm (e.g. dense 4B Q8_0 3 GPU: 2331 vs 2321 text
  chars).  That is expected on a base move — upstream changed prefill kernels (rope MRoPE, MMQ tiling) —
  and the instrument that decides it is the prefill-logit gate above, not a cross-build text diff.
  Within this build, `plain == draft-mtp` holds everywhere it is promised.

**Not re-validated this campaign:** RDNA3.5 (`halo`, gfx1151) and RDNA3_0; no NVIDIA run (out of scope by
policy, and both fused gate/up paths are arch-disjoint so the NVIDIA arm is upstream's own code).
`TODO` carries the gfx1151 re-validation as an open item.

## Process lessons (both cost real time; the gate caught both)

1. **A fix must be committed by the block that owns the broken code, not by the block where you noticed it.**
   The `rope_multi_cuda` duplicate-parameter fix was applied while resolving block 08's conflicts, so it
   landed in block 08's tree and **blocks 04-07 failed** the per-block gate.  Fix: `git checkout <block>`,
   fix, `git commit --amend`, `git rebase --onto <new> <old> <branch>`, then re-run the gate.
2. **`git add -A` during a conflicted rebase stages unresolved files.**  Block 10's tree ended up carrying
   `<<<<<<< HEAD` markers in `llama-context.cpp` (plus a duplicated `is_dsv4` line in `llama-model.cpp`) and
   only the per-block gate saw it — the tip built fine.  Always `grep -rl '^<<<<<<< '` across *every* block
   tree before declaring the chain done, and stage named files, not `-A`.
3. **A build dir's `GGML_CUDA_FA_QUANTS` is sticky across configures, and it changes greedy text.**
   The re-base gates ran on the scratch build (`q4_0;q8_0;f16;bf16`) while the recorded prefill-logit base
   and the fork-branch build use the canonical 8-pair set; the same source tree therefore produced
   different greedy text with **identical** prefill logits (0.000531 / 98.863 % both times).  The gate
   passed on both, which is the useful part; the text difference is config, not numerics.  Always compare
   text within one build dir, and run the prefill-logit gate with the canonical build.
4. The tip tree was byte-identical before and after both attribution fixes (`35090ba36c`), which is how you
   know an attribution fix did not change the delivery — check it, because the gate's verdict changes while
   the artifact must not.

## Follow-up this campaign produced: the ladder's cost is one test TU

`gate-per-block-build.sh` gained `--fast` (release profile: `ggml-hip llama llama-cli llama-server
llama-bench llama-perplexity`) and the release flow now uses it, because the measurement above showed
**where** the 2101 s goes: `tests/test-backend-ops.cpp` is one 14.4k-line TU that costs **5m13s** cold
(a `src/*.cpp` TU is ~5 s), and a re-base gives the ladder ~4 distinct contents of it.  So ~21 of the
~35 minutes were a test binary that cannot fail on delivery code.  `--fast` is 78 s warm, and the
dropped coverage is restored at the tip (build `all` once, run `test-backend-ops -o …`).  The default
stays `all` (the r2 standard) and the result line names the profile it used.

## Where the work lived

* `~/llama-scratch`, branch `fixchain` (pre-fix chains: `rb2`, and the intermediate amends
  `2fadd32ac`/`210237fc6`); `rb2-old` = the r137 chain `e8bd52069`.
* Build dir: `~/llama-scratch/build-rocm` (the long-lived path, so the r138 content is now warm for the
  next gate run).
