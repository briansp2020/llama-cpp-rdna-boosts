#!/bin/bash
# probs.sh <build_dir> <tag> <prompt> <np> <extra server args...>
set -u
BUILD=$1; TAG=$2; PROMPT=$3; NP=$4; shift 4
M=${M:-/llm/models/Qwen3.8/Flash-Next/GSQ-IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf}
PORT=${PORT:-8977}; NCMOE=${NCMOE:-48}; CTX=${CTX:-204800}; UB=${UB:-6144}; GPUS=${GPUS:-0,1}
LOGD=/tmp/srr; mkdir -p $LOGD
export LD_LIBRARY_PATH=/opt/rocm-7.14.1-gfx120X/lib:${LD_LIBRARY_PATH:-}
export HIP_VISIBLE_DEVICES=$GPUS
"$BUILD/bin/llama-server" -v -m "$M" \
  -sm "${SM:-tensor}" -ncmoe "$NCMOE" -ub "$UB" -b "$UB" -c "$CTX" --no-kv-unified \
  -ctk q8_0 -ctv q8_0 -fa on -t 8 --fit on \
  "$@" --host 127.0.0.1 --port $PORT --no-webui --cache-ram 0 \
  > $LOGD/${TAG}.err 2>&1 &
pid=$!
ok=0
for i in $(seq 1 300); do
  kill -0 $pid 2>/dev/null || { echo "$TAG SERVER-DIED"; tail -5 $LOGD/${TAG}.err; exit 1; }
  curl -sf http://127.0.0.1:$PORT/health >/dev/null 2>&1 && { ok=1; break; }
  sleep 1
done
[ $ok -eq 1 ] || { echo "$TAG HEALTH-TIMEOUT"; kill $pid 2>/dev/null; exit 1; }
curl -s http://127.0.0.1:$PORT/completion -H 'Content-Type: application/json' \
  -d '{"prompt":"Hello","n_predict":8,"temperature":0,"cache_prompt":false}' >/dev/null
python3 /tmp/srr/logprobs.py $PORT "$PROMPT" "$NP" > $LOGD/${TAG}.probs 2>&1
kill $pid 2>/dev/null; wait $pid 2>/dev/null
echo "$TAG done"
