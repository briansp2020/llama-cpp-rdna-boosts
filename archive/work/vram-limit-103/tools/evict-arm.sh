#!/usr/bin/env bash
# evict-arm.sh <MiB to hold AFTER load|0> [n_predict]
#
# Loads the #103 config in a real llama-server (weights fully resident), then — with
# <MiB> > 0 — starts vram_fill on device 0 so the driver must find that memory by
# evicting pages out of the already-loaded model, exactly what a desktop (or another
# GPU app) does to the reporter's 16 GiB card.  Then it sends one request and reports
# the server's own timings.
#
# This separates "free VRAM is small when the process starts" (load-time placement) from
# "someone takes VRAM away while the model is resident" (run-time eviction).
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL=${MODEL:-/llm/models/Qwen3.8/27B/GSQ-RCO-IQ3_S/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf}
MMPROJ=${MMPROJ:-/llm/models/Qwen3.8/27B/GSQ-RCO-IQ3_S/mmproj-Qwen3.8-27B-BF16.gguf}
PROMPT=${PROMPT:-$HERE/../prompt-3x-prose.txt}
SERVER=${SERVER:-$HOME/llama.cpp/build-rocm/bin/llama-server}
PORT=${PORT:-8199}
HOLD_MIB=${1:-0}
NP=${2:-32}
mkdir -p "$HERE/../logs"
TAG=${TAG:-evict${HOLD_MIB}}
LOG=$HERE/../logs/$TAG.log

cleanup() {
    [ -n "${SPID:-}" ] && kill "$SPID" 2>/dev/null
    [ -n "${HPID:-}" ] && kill "$HPID" 2>/dev/null
    sleep 1
    [ -n "${SPID:-}" ] && kill -9 "$SPID" 2>/dev/null
    [ -n "${HPID:-}" ] && kill -9 "$HPID" 2>/dev/null
    [ -n "${HPID:-}" ] && wait "$HPID" 2>/dev/null
    pkill -x vram_fill 2>/dev/null
}
trap cleanup EXIT INT TERM
pkill -x vram_fill 2>/dev/null; sleep 1

# PRE_HOLD=1: allocate the holder BEFORE the server starts, so the free VRAM at load matches a
# smaller card (the r135 emulation).  Default: hold after the load (models a desktop/other app).
if [ "${PRE_HOLD:-0}" != "0" ] && [ "$HOLD_MIB" != "0" ]; then
    HIP_VISIBLE_DEVICES=0 "$HERE/vram_fill" "$HOLD_MIB" 0 > "$LOG.holder" 2>&1 &
    HPID=$!
    sleep 5
    echo "=== pre-load holder: $(cat "$LOG.holder" | tr '\n' ' ')"
fi

HIP_VISIBLE_DEVICES=0 "$SERVER" -m "$MODEL" --mmproj "$MMPROJ" \
    -ngl 99 -c ${CTX:-65536} -b 4096 -ub ${UB:-1024} -fa on -ctk q4_0 -ctv q4_0 -t 8 \
    --spec-type ${SPEC:-draft-mtp-adaptive} --port "$PORT" --no-webui -v > "$LOG" 2>&1 &
SPID=$!
for _ in $(seq 1 300); do
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
    kill -0 "$SPID" 2>/dev/null || { echo "server died"; tail -5 "$LOG"; exit 1; }
    sleep 1
done
echo "=== server up (PID $SPID); free before hold: $(HIP_VISIBLE_DEVICES=0 "$HERE/vram_info")"

if [ "$HOLD_MIB" != "0" ] && [ "${PRE_HOLD:-0}" = "0" ]; then
    HIP_VISIBLE_DEVICES=0 "$HERE/vram_fill" "$HOLD_MIB" 0 > "$LOG.holder" 2>&1 &
    HPID=$!
    sleep 5
    echo "=== holder $(cat "$LOG.holder" | tr '\n' ' ')"
    echo "=== free with holder: $(HIP_VISIBLE_DEVICES=0 "$HERE/vram_info")"
fi

# KILL_AFTER_LOAD=1: load the model with the card squeezed, then hand the memory back before the
# request.  Separates "the load-time placement/policy froze something" from "moving the holder's
# pages out of VRAM during the request costs the run".
if [ "${KILL_AFTER_LOAD:-0}" != "0" ] && [ -n "${HPID:-}" ]; then
    kill "$HPID" 2>/dev/null; wait "$HPID" 2>/dev/null; HPID=""
    sleep 2
    echo "=== holder stopped; free before request: $(HIP_VISIBLE_DEVICES=0 "$HERE/vram_info")"
fi

: > "$LOG.samples"
( while :; do read -r v g < <(rocm-smi --showmeminfo vram -d 0 2>/dev/null | awk '/VRAM Total Used/{print $NF}'; rocm-smi --showmeminfo gtt -d 0 2>/dev/null | awk '/GTT Total Used/{print $NF}'); echo "$v $g" >> "$LOG.samples"; sleep 0.3; done ) &
SAMPLER=$!

python3 - "$PORT" "$PROMPT" "$NP" "${NREQ:-1}" > "$LOG.response" <<'EOF'
import json, sys, urllib.request
port, prompt_file, np, nreq = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
prompt = open(prompt_file).read()
for r in range(nreq):
    body = {"prompt": prompt, "n_predict": np, "temperature": 0, "top_k": 1, "seed": 42, "cache_prompt": False, "stream": False}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/completion", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    j = json.load(urllib.request.urlopen(req, timeout=3600))
    t = j.get("timings", {})
    print("req %d: prompt_n %s prompt_ms %s prompt_t/s %.1f | predicted_n %s predicted_ms %s predicted_t/s %.1f | draft %s acc %s" % (
        r, t.get("prompt_n"), t.get("prompt_ms"), 1000.0*t.get("prompt_n",0)/max(t.get("prompt_ms",1),1e-9),
        t.get("predicted_n"), t.get("predicted_ms"), 1000.0*t.get("predicted_n",0)/max(t.get("predicted_ms",1),1e-9),
        t.get("draft_n"), t.get("draft_n_accepted")))
EOF
sleep 1
kill "$SAMPLER" 2>/dev/null
PEAK_V=$(awk 'BEGIN{m=0}{if($1>m)m=$1}END{printf "%d", m/1048576}' "$LOG.samples")
PEAK_G=$(awk 'BEGIN{m=0}{if($2>m)m=$2}END{printf "%d", m/1048576}' "$LOG.samples")
echo "--- $(cat "$LOG.response" | tr '\n' '|')"
echo "=== peak VRAM used ${PEAK_V} MiB, peak GTT used ${PEAK_G} MiB"
grep -cE "not enough free device memory" "$LOG" | sed 's/^/FA staging stand-downs: /'
grep -E "graphs reused|graph_compute calls" "$LOG" | tail -3
