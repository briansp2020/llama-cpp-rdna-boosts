#!/usr/bin/env bash
# GPU validation gate -- the heavy per-push gate that no hosted runner can run.
#
# WHY THIS EXISTS
#   The repo's gating has three tiers, and only two could run anywhere:
#     * tier 1  `.github/workflows/validate.yml`  -- apply + tree/marker checks, ~20 s, no GPU;
#     * tier 2  THIS SCRIPT on the off-host CI worker (radiance, gfx1151) -- apply, the
#               per-block `--fast` ladder, the op-test oracles and the prefill-logit KLD gate;
#     * tier 3  the release flow -- the `all` build, coherence, MTP Protocol A, arch sweeps.
#   Tier 2 is the one that actually catches a bad patch set, because almost every delivery
#   block is a CUDA/HIP kernel change and a hosted runner has no ROCm.  It is the automation
#   behind "a re-base is a decode AND prefill change" (AGENTS.md): it re-applies the set from
#   scratch on the box, so a patch that no longer applies, a block whose tree does not build,
#   a broken op and a shifted prefill all fail here, unattended.
#
# WHAT IT DOES (each step is skippable; see the env list)
#   1. materialises the set: delivery repo checked out at REV, `apply-all.sh` in LLAMA_REPO,
#      applied tree == that rev's `release.json.tree` (strict `git am`);
#   2. `gate-per-block-build.sh --fast` over every block commit (the ladder);
#   3. builds the tip's test-binary and runs the op oracles (`FLASH_ATTN_EXT`,
#      `GATED_DELTA_NET`, `FLASH_ATTN_QSA`, `MUL_MAT`, `MUL_MAT_ID`, `TOP_K_MASK`, `HC_MIX`);
#   4. `gate-prefill-logits.sh` on the tip -- compares against this box's recorded base, or
#      RECORDS one on the first run (a fresh worker box has no base; the gate then exits with
#      the recorded result, and later runs compare);
#   5. writes a one-screen summary to $STATE_DIR/runs/<rev>/summary.txt and, on success,
#      records $STATE_DIR/last-ok.
#
# USAGE
#   validate-gpu.sh [REV]              REV defaults to origin/main
#   validate-gpu.sh --force [REV]      re-run even if REV is the last success
#
# ENV  REPO (delivery checkout)          default the repo this script lives in
#      LLAMA_REPO (long-lived checkout)  default $HOME/llama-ci   <- keep ONE path (ccache keys on it)
#      STATE_DIR                         default $HOME/.cache/rdna-validate
#      JOBS                              default nproc
#      ROCM_DIR / GPU_TARGETS            auto-detected from /opt/rocm-10.1.0-gfxNNNN
#      KLD_MODEL, KLD_CTX (512), KLD_CHUNKS (40), KLD_BASE_DIR
#      SKIP_LADDER=1 / SKIP_OPTESTS=1 / SKIP_KLD=1   narrow the run
#      MAX_ATTEMPTS (3, per rev)         KEEP_RUNS (5)
#      CI_LOCK (the container watcher's lock)  default $HOME/.cache/rdna-containers/.lock
#
# Exit: 0 = validated (or nothing to do / already given up on this rev); 1 = a step failed.
set -uo pipefail

export PATH="$HOME/.local/bin:$HOME/bin:/usr/local/bin:/usr/bin:/bin"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
LLAMA_REPO="${LLAMA_REPO:-$HOME/llama-ci}"
STATE_DIR="${STATE_DIR:-$HOME/.cache/rdna-validate}"
JOBS="${JOBS:-$(nproc)}"
SKIP_LADDER="${SKIP_LADDER:-0}"
SKIP_OPTESTS="${SKIP_OPTESTS:-0}"
SKIP_KLD="${SKIP_KLD:-0}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-3}"
KEEP_RUNS="${KEEP_RUNS:-5}"
FORCE=0
REV=""
CI_LOCK="${CI_LOCK:-$HOME/.cache/rdna-containers/.lock}"

# The KLD arm is slow on an APU; keep the default point small and stable.
KLD_MODEL="${KLD_MODEL:-/llm/models/Qwen3.5/9B/Q8_0/Qwen3.5-9B-Q8_0.gguf}"
KLD_CTX="${KLD_CTX:-512}"
KLD_CHUNKS="${KLD_CHUNKS:-40}"

log() { echo "$(date -Is) $*"; }
die() { log "ERROR: $*"; exit 1; }

while [ $# -gt 0 ]; do
    case "$1" in
        --force) FORCE=1; shift ;;
        -h|--help) sed -n '2,/^set -uo/p' "${BASH_SOURCE[0]}" | sed 's/^# \?//; /^set -uo/d'; exit 0 ;;
        -*) echo "unknown option: $1" >&2; exit 2 ;;
        *)  REV="$1"; shift ;;
    esac
done

for t in git jq rsync; do command -v "$t" >/dev/null || die "$t is required"; done
command -v ccache >/dev/null && : || log "WARN: ccache not found -- the ladder will be slow and cold"

# ROCm: pick the installed 10.1 tree and its target, so this works on any gfx box.
if [ -z "${ROCM_DIR:-}" ]; then
    ROCM_DIR="$(ls -d /opt/rocm-10.1.0-gfx* 2>/dev/null | head -1)"
    [ -n "$ROCM_DIR" ] || die "no /opt/rocm-10.1.0-gfx* install found (set ROCM_DIR)"
fi
[ -n "${GPU_TARGETS:-}" ] || GPU_TARGETS="$(basename "$ROCM_DIR" | sed 's/^.*-//')"

mkdir -p "$STATE_DIR"
exec 9>"$STATE_DIR/.lock"
if ! flock -n 9; then log "another validation run is in progress; skipping"; exit 0; fi

# Never contend with a release image build: that is the worker's other, rarer job and it is
# already slow on an APU.  The watcher holds this lock for the whole build+push.
if [ -e "$CI_LOCK" ] && command -v flock >/dev/null; then
    if ! flock -n "$CI_LOCK" true 2>/dev/null; then
        log "a container build is running; skipping this pass"; exit 0
    fi
fi

cd "$REPO" || die "not a checkout: $REPO"
git fetch --quiet --tags origin || die "git fetch failed in $REPO"

if [ -z "$REV" ]; then
    REV="$(git rev-parse origin/main)"; REV_LABEL="origin/main"
else
    REV="$(git rev-parse --verify -q "${REV}^{commit}")" || die "cannot resolve $REV"
    REV_LABEL="$REV"
fi
short="$(git rev-parse --short "$REV")"

last_ok="$(cat "$STATE_DIR/last-ok" 2>/dev/null || true)"
if [ "$REV" = "$last_ok" ] && [ "$FORCE" -eq 0 ]; then
    log "up to date ($short already validated)"; exit 0
fi

fail_file="$STATE_DIR/failed-$short"
attempt=$(( $(cat "$fail_file" 2>/dev/null || echo 0) + 1 ))
if [ -f "$fail_file" ] && [ "$attempt" -gt "$MAX_ATTEMPTS" ]; then
    log "ERROR: $short has failed $((attempt - 1)) times; not retrying (fix it, then remove $fail_file)"
    exit 0
fi

RUN="$STATE_DIR/runs/$short"
rm -rf "$RUN"; mkdir -p "$RUN"
SUMMARY="$RUN/summary.txt"
{
    echo "rdna-boosts GPU validation"
    echo "rev      : $short ($REV_LABEL)"
    echo "box      : $(hostname), $GPU_TARGETS, ROCm $(basename "$ROCM_DIR"), $(nproc) threads"
    echo "attempt  : $attempt/$MAX_ATTEMPTS"
    echo "started  : $(date -Is)"
    echo
} > "$SUMMARY"
say() { echo "$*" | tee -a "$SUMMARY"; }

fail() {
    echo "$attempt" > "$fail_file"
    say "RESULT   : FAIL -- $*"
    log "ERROR: $short failed (attempt $attempt/$MAX_ATTEMPTS): $*"
    log "summary: $SUMMARY"
    exit 1
}

# --- 1. materialise the set ------------------------------------------------------------------
# The delivery repo must be AT the rev under test before apply-all.sh runs, because apply-all
# reads release.json from its own checkout (a stale checkout would apply the wrong set).
git checkout --quiet --force --detach "$REV" || fail "cannot check out $short in $REPO"
[ "$(git rev-parse HEAD)" = "$REV" ] || fail "HEAD is not $short after checkout"
base="$(jq -r '.base'     release.json)"
tree="$(jq -r '.tree'     release.json)"
n_blocks="$(jq -r '.n_blocks' release.json)"
say "release  : $(jq -r '.release' release.json)  base=$base  blocks=$n_blocks"

if [ ! -d "$LLAMA_REPO/.git" ]; then
    say "clone    : $LLAMA_REPO (first run; keep ONE path -- ccache keys on it)"
    git clone --quiet "$(git -C "$REPO" remote get-url origin | sed 's|llama-cpp-rdna-boosts|llama.cpp|')" "$LLAMA_REPO" \
        || fail "cannot clone the fork into $LLAMA_REPO"
fi
git -C "$LLAMA_REPO" fetch --quiet origin || fail "git fetch failed in $LLAMA_REPO"
git -C "$LLAMA_REPO" cat-file -e "${base}^{commit}" 2>/dev/null || fail "base $base not in $LLAMA_REPO (fetch the fork)"
git -C "$LLAMA_REPO" checkout --quiet --force --detach "$base" || fail "cannot check out base $base"
git -C "$LLAMA_REPO" branch -D rdna-boosts >/dev/null 2>&1 || true
if ! bash "$REPO/scripts/apply-all.sh" "$LLAMA_REPO" >"$RUN/apply.log" 2>&1; then
    fail "apply-all.sh failed (log: $RUN/apply.log)"
fi
applied="$(git -C "$LLAMA_REPO" rev-parse HEAD^{tree})"
[ "$applied" = "$tree" ] || fail "applied tree $applied != release.json.tree $tree"
tip="$(git -C "$LLAMA_REPO" rev-parse HEAD)"
say "apply    : strict, tree $applied == release.json.tree"

# --- 2. per-block build ladder ----------------------------------------------------------------
if [ "$SKIP_LADDER" = 1 ]; then
    say "ladder   : skipped (SKIP_LADDER=1)"
else
    if bash "$REPO/scripts/gate-per-block-build.sh" --repo "$LLAMA_REPO" --base "$base" --tip "$tip" \
            --rocm "$ROCM_DIR" --gpu-target "$GPU_TARGETS" --jobs "$JOBS" --fast --keep-going \
            >"$RUN/ladder.log" 2>&1; then
        say "ladder   : $(grep -E '^==> result:' "$RUN/ladder.log" | tail -1 | sed 's/^==> result: //')"
    else
        cp "$RUN/ladder.log" "$STATE_DIR/last-ladder-fail.log"
        fail "per-block ladder (log: $RUN/ladder.log)"
    fi
fi

BUILD="$LLAMA_REPO/build-per-block"   # the gate's build dir, holding the TIP tree after the ladder
# --- 3. op oracles ----------------------------------------------------------------------------
if [ "$SKIP_OPTESTS" = 1 ]; then
    say "op tests : skipped (SKIP_OPTESTS=1)"
else
    if ! cmake --build "$BUILD" --target test-backend-ops llama-perplexity llama-cli -j "$JOBS" \
            >"$RUN/optest-build.log" 2>&1; then
        fail "tip build for the op tests (log: $RUN/optest-build.log)"
    fi
    opts_failed=""
    for op in FLASH_ATTN_EXT GATED_DELTA_NET FLASH_ATTN_QSA MUL_MAT MUL_MAT_ID TOP_K_MASK HC_MIX; do
        line="$(HIP_VISIBLE_DEVICES=0 LD_LIBRARY_PATH="$BUILD/bin:$ROCM_DIR/lib" \
                timeout 3600 "$BUILD/bin/test-backend-ops" -o "$op" 2>&1 | grep -E 'tests passed|FAIL' | tail -1)"
        say "op       : $(printf '%-16s' "$op") ${line:-<no result>}"
        case "$line" in *passed*) ;; *) opts_failed="$opts_failed $op" ;; esac
    done
    [ -z "$opts_failed" ] || fail "op tests failed:$opts_failed"
fi

# --- 4. prefill-logit -------------------------------------------------------------------------
if [ "$SKIP_KLD" = 1 ]; then
    say "prefill  : skipped (SKIP_KLD=1)"
elif [ ! -f "$KLD_MODEL" ]; then
    say "prefill  : skipped (no model at $KLD_MODEL)"
else
    kld_args=(--ppl "$BUILD/bin/llama-perplexity" --model "$KLD_MODEL" --ctx "$KLD_CTX" --chunks "$KLD_CHUNKS")
    [ -n "${KLD_BASE_DIR:-}" ] && kld_args+=(--base-dir "$KLD_BASE_DIR")
    # A fresh worker has no recorded base: record one, say so, and let the NEXT run compare.
    have_base=0
    if [ -n "${KLD_BASE_DIR:-}" ]; then
        ls "$KLD_BASE_DIR"/prefill-logit-*.kld >/dev/null 2>&1 && have_base=1
    else
        ls "$HOME/.cache/rdna-boosts/gates"/prefill-logit-*.kld >/dev/null 2>&1 && have_base=1
    fi
    if [ "$have_base" -eq 0 ]; then
        kld_args+=(--record)
        say "prefill  : no recorded base on this box -> RECORDING one now (later runs compare)"
    fi
    if LD_LIBRARY_PATH="$BUILD/bin:$ROCM_DIR/lib" bash "$REPO/scripts/gate-prefill-logits.sh" \
            "${kld_args[@]}" >"$RUN/kld.log" 2>&1; then
        say "prefill  : $(grep -E 'mean KLD|same top p' "$RUN/kld.log" | tr -s ' ' | tr '\n' ' ')"
    else
        cp "$RUN/kld.log" "$STATE_DIR/last-kld-fail.log"
        fail "prefill-logit gate (log: $RUN/kld.log)"
    fi
fi

# --- 5. record --------------------------------------------------------------------------------
echo "$REV" > "$STATE_DIR/last-ok"
rm -f "$fail_file"
ls -1d "$STATE_DIR"/runs/*/ 2>/dev/null | sort | head -n -"$KEEP_RUNS" | xargs -r rm -rf
say "RESULT   : PASS  ($(date -Is))"
log "$short validated (summary: $SUMMARY)"
exit 0
