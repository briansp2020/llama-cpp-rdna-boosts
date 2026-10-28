#!/usr/bin/env bash
# run-arm.sh <target-free-MiB|0> [extra llama-cli args...]
#
# Emulates a smaller card by holding VRAM on device 0 with vram_fill, then runs the
# issue-#219 / TODO-#103 config (27B GSQ-RCO-IQ3_S + mmproj + adaptive MTP, -c 65536,
# -b 4096 -ub 1024, q4_0 KV) on that card.
#
# 0 = no holder.  Environment: LOG, BIN, MODEL, MMPROJ, PROMPT, EXTRA (string), NTOK.
# Writes: $LOG (llama-cli -v), $LOG.samples (vram_used,gtt_used every ~300 ms).
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL=${MODEL:-/llm/models/Qwen3.8/27B/GSQ-RCO-IQ3_S/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf}
MMPROJ=${MMPROJ:-/llm/models/Qwen3.8/27B/GSQ-RCO-IQ3_S/mmproj-Qwen3.8-27B-BF16.gguf}
PROMPT=${PROMPT:-$HERE/../prompt-3x-prose.txt}
BIN=${BIN:-$HOME/llama.cpp/build-rocm/bin/llama-cli}
TARGET_FREE_MIB=${1:-0}
shift || true
mkdir -p "$HERE/../logs"
TAG=${TAG:-free${TARGET_FREE_MIB}}
LOG=${LOG:-$HERE/../logs/$TAG.log}

mem_gpu0() {  # "<vram_used_bytes> <gtt_used_bytes>" for device 0
    local v g
    v=$(rocm-smi --showmeminfo vram -d 0 2>/dev/null | awk '/VRAM Total Used Memory \(B\)/{print $NF}')
    g=$(rocm-smi --showmeminfo gtt  -d 0 2>/dev/null | awk '/GTT Total Used Memory \(B\)/{print $NF}')
    echo "${v:-0} ${g:-0}"
}

cleanup() {
    [ -n "${SAMPLER_PID:-}" ] && kill "$SAMPLER_PID" 2>/dev/null
    if [ -n "${HOLDER_PID:-}" ]; then kill "$HOLDER_PID" 2>/dev/null; wait "$HOLDER_PID" 2>/dev/null; fi
    pkill -x vram_fill 2>/dev/null
}
trap cleanup EXIT INT TERM

pkill -x vram_fill 2>/dev/null; sleep 1

if [ "$TARGET_FREE_MIB" != "0" ]; then
    read -r _ NOW _ < <("$HERE/vram_info" 2>/dev/null | sed 's/^//')   # "free N total M"
    NOW=$(HIP_VISIBLE_DEVICES=0 "$HERE/vram_info" | awk '{print $2}')
    HOLD_MIB=$(( NOW - TARGET_FREE_MIB ))
    if [ "$HOLD_MIB" -lt 1 ]; then
        echo "target free ${TARGET_FREE_MIB} MiB but the card already has ${NOW} MiB free" >&2
        exit 2
    fi
    HIP_VISIBLE_DEVICES=0 "$HERE/vram_fill" "$HOLD_MIB" 0 "${CHUNK_MIB:-0}" > "$LOG.holder" 2>&1 &
    HOLDER_PID=$!
    sleep 4
fi
read -r _ FREEB _ < <(HIP_VISIBLE_DEVICES=0 "$HERE/vram_info")

: > "$LOG.samples"
( while :; do read -r v g < <(mem_gpu0); echo "$v $g" >> "$LOG.samples"; sleep 0.3; done ) &
SAMPLER_PID=$!

echo "=== arm $TAG: free before load ${FREEB} MiB (target ${TARGET_FREE_MIB})"
HIP_VISIBLE_DEVICES=0 timeout ${TIMEOUT:-3600} "$BIN" \
    -m "$MODEL" --mmproj "$MMPROJ" \
    -ngl 99 -c 65536 -b 4096 -ub 1024 -fa on -ctk q4_0 -ctv q4_0 \
    -t 8 --seed 42 --temp 0 -n ${NTOK:-32} \
    --spec-type draft-mtp-adaptive \
    -p "$(cat "$PROMPT")" \
    --no-display-prompt --single-turn "$@" -v > "$LOG" 2>&1
RC=$?
sleep 0.5
kill "$SAMPLER_PID" 2>/dev/null; SAMPLER_PID=""
PEAK_V=$(awk 'BEGIN{m=0}{if($1>m)m=$1}END{printf "%d", m/1048576}' "$LOG.samples")
PEAK_G=$(awk 'BEGIN{m=0}{if($2>m)m=$2}END{printf "%d", m/1048576}' "$LOG.samples")
echo "=== llama-cli exit $RC   peak VRAM used ${PEAK_V} MiB, peak GTT used ${PEAK_G} MiB"
grep -E "prompt eval time|eval time|graphs reused|graph_compute calls|refusing|ROCm error|^error" "$LOG" | sed 's/^[0-9.]* [IWE] //' | tail -12
exit $RC
