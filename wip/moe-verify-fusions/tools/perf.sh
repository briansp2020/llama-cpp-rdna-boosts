#!/bin/bash
# perf.sh <build_dir> <tag> <n_predict> -- <server args...>
# Long decode perf on the field config, with ignore_eos, reporting the response timings.
set -u
BUILD=$1; TAG=$2; NP=$3; shift 3
[ "${1:-}" = "--" ] && shift
M=${M:-/llm/models/Qwen3.8/Flash-Next/GSQ-IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf}
PROMPT=${PROMPT:-/tmp/srr/mixed30k.txt}
PORT=${PORT:-8979}; NCMOE=${NCMOE:-48}; CTX=${CTX:-204800}; UB=${UB:-6144}
GPUS=${GPUS:-0,1}; SM=${SM:-tensor}
LOGD=/tmp/srr; mkdir -p $LOGD
export LD_LIBRARY_PATH=/opt/rocm-7.14.1-gfx120X/lib:${LD_LIBRARY_PATH:-}
export HIP_VISIBLE_DEVICES=$GPUS
"$BUILD/bin/llama-server" -v -m "$M" \
  -sm "$SM" -ncmoe "$NCMOE" -ub "$UB" -b "$UB" -c "$CTX" --no-kv-unified \
  -ctk q8_0 -ctv q8_0 -fa on -t 8 --fit on \
  "$@" --host 127.0.0.1 --port $PORT --no-webui --cache-ram 0 \
  > $LOGD/${TAG}.err 2>&1 &
pid=$!
ok=0
for i in $(seq 1 400); do
  kill -0 $pid 2>/dev/null || { echo "$TAG SERVER-DIED"; tail -4 $LOGD/${TAG}.err; exit 1; }
  curl -sf http://127.0.0.1:$PORT/health >/dev/null 2>&1 && { ok=1; break; }
  sleep 1
done
[ $ok -eq 1 ] || { echo "$TAG HEALTH-TIMEOUT"; kill $pid 2>/dev/null; exit 1; }
curl -s http://127.0.0.1:$PORT/completion -H 'Content-Type: application/json' \
  -d '{"prompt":"Hello","n_predict":8,"temperature":0,"cache_prompt":false}' >/dev/null
python3 - $PORT "$PROMPT" "$NP" <<'PY'
import json,sys,urllib.request
port,prompt_path,np_=sys.argv[1],sys.argv[2],int(sys.argv[3])
prompt=open(prompt_path).read()
req=urllib.request.Request("http://127.0.0.1:%s/completion"%port,
  data=json.dumps({"prompt":prompt,"n_predict":np_,"temperature":0,"cache_prompt":False,
                   "ignore_eos":True,"return_tokens":True}).encode(),
  headers={"Content-Type":"application/json"})
r=json.load(urllib.request.urlopen(req,timeout=3600))
t=r.get("timings",{})
print("prompt_n=%s prompt_tps=%.1f predicted_n=%s predicted_tps=%.2f" % (
    t.get("prompt_n"), t.get("prompt_per_second",0), t.get("predicted_n"), t.get("predicted_per_second",0)))
PY
kill $pid 2>/dev/null; wait $pid 2>/dev/null
grep -aoE "draft acceptance = [0-9.]+" $LOGD/${TAG}.err | tail -1 | sed "s/^/  $TAG /"
