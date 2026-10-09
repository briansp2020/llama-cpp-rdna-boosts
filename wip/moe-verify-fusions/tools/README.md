# `moe-verify-fusions` tools

The field A/B harness this campaign's numbers were measured with, made durable so a cold session does not
have to rebuild it from the prose in `../README.md`.

| script | what it runs |
|---|---|
| `field.sh` | the r38 §5.5 field arm: `llama-server`, `-sm tensor -ncmoe 48 -ub 6144 -b 6144 -c 204800 --no-kv-unified -ctk/-ctv q8_0 -fa on -t 8 --fit on --spec-type draft-mtp --spec-draft-n-max 3`, a small warm-up request, then a ~32 k-token prefill + 1000 MTP tokens.  Forces `GGML_CUDA_ALLREDUCE=ce` (the only **deterministic** AR, 2-GPU-only) |
| `field3.sh` | the same **without** the forced AR — i.e. the 3-GPU production default (hybrid).  Use for `GPUS=0,1,2` |
| `field_nospec.sh` | `field3.sh` with `--spec-type none` and no draft model — the **plain** decode baseline (`#51`'s control) |

All three are env-overridable:

```bash
export M=/llm/models/Qwen3.8/Flash-Next/GSQ-IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf
export D=/llm/models/Qwen3.8/Flash-Next/IQ4_NL/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf
export PROMPT=/tmp/srr/mixed30k.txt
export GPUS=0,1,2          # or 0,1
```

`D` is **required** for the GSQ set: it has no embedded MTP head (verified — `blk.0..47` only, no `blk.48`
and no `nextn.*`; the separate MTP head is itself a `blk.48.*` layer file).  `M` defaults to the IQ4_NL
target, `D` to the shared MTP draft, and `PROMPT` to the **non-degenerate** `mixed30k.txt` — do **not**
fall back to `prose30k.txt`, which is the repo doc ×6 that the model simply copies (acceptance 1.0, inflated
prefill; see `../../archive/work/slab-ring-region/FINDINGS-numerics.md` F1).

## Build the prompt files

Both are derived from the repo, so they are reproducible rather than committed:

```bash
mkdir -p /tmp/srr && cd /home/stew675/llama-cpp-rdna-boosts
# the measured prompt (non-repetitive, ~32 k tokens, sha256 69624f4d207f40bd...)
cat README.md ENVIRONMENT.md CONTAINERS.md archive/docs/baseline-history.md > /tmp/srr/mixed30k.txt
# (historical, do NOT measure on this one) the degenerate r38 field prompt:
#   yes "$(cat prompts/prose-rdna-boosts.txt)" | head -6 > /tmp/srr/prose30k.txt
```

## Run

```bash
cd wip/moe-verify-fusions/tools
PROMPT=/tmp/srr/mixed30k.txt GPUS=0,1,2 ./field3.sh ~/llama.cpp/build-rocm-hybrid t3 run30kfit
```

Arguments are `<build_dir> <tag> <fit|run30k|run30kfit>`; `run30kfit` = a warm-up request + the measured
prefill/decode (`--fit on`), which is what the campaign's tables use.  Outputs land in `$LOGD` (default
`/tmp/srr`): `<tag>.timing` (the `prompt_tps` / `predicted_tps` line), `<tag>.err` (the server log —
`grep -ac nan`, the `draft acceptance` line, the `moe_cache_*` / slab lines), `<tag>.out`.

**Discard the first run after a config change** — it pays the lazy PLE / host-expert disk read (measured
563 vs 968 t/s prefill on an identical config).  Set `M`/`D`/`PROMPT` for the GSQ set before the warm-up run,
not between the warm-up and the measurement.

`PORT` is fixed at 8942; only one harness may run at a time (and per `AGENTS.md`, never run parallel
benches).
