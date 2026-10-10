#!/bin/bash
# Width-purity sweep on the field config: plain (none) vs draft-mtp n-max 1/3/7.
# Captures return_tokens per arm so the streams can be diffed.  Env arm inherited from the caller.
#   widthsweep.sh <build_dir> <tag> [none|n1|n3|n7|all]
set -u
BUILD=$1; TAG=$2; MODE=${3:-all}
M=${M:-/llm/models/Qwen3.8/Flash-Next/GSQ-IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf}
# `D-` (not `:-`): an explicitly EMPTY D means "use the model's embedded MTP head" (no draft model).
D=${D-/llm/models/Qwen3.8/Flash-Next/IQ4_NL/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf}
PROMPT=${PROMPT:-/tmp/srr/mixed30k.txt}
PORT=${PORT:-8963}
NP=${NP:-256}
NCMOE=${NCMOE:-48}
CTX=${CTX:-204800}
UB=${UB:-6144}
SMD=""
[ -n "$D" ] && SMD="--spec-draft-model $D"
LOGD=/tmp/srr; mkdir -p $LOGD
export LD_LIBRARY_PATH=/opt/rocm-7.14.1-gfx120X/lib:${LD_LIBRARY_PATH:-}
export HIP_VISIBLE_DEVICES=${GPUS:-0,1}

run() { # label extra-args...
  local label=$1; shift
  "$BUILD/bin/llama-server" -v -m "$M" \
    -sm tensor -ncmoe "$NCMOE" -ub "$UB" -b "$UB" -c "$CTX" --no-kv-unified \
    -ctk q8_0 -ctv q8_0 -fa on -t 8 --fit on \
    "$@" --host 127.0.0.1 --port $PORT --no-webui --cache-ram 0 \
    > $LOGD/ws_${TAG}_${label}.out 2> $LOGD/ws_${TAG}_${label}.err &
  local pid=$!
  local ok=0
  for i in $(seq 1 300); do
    kill -0 $pid 2>/dev/null || { echo "$label SERVER-DIED"; tail -4 $LOGD/ws_${TAG}_${label}.err; return 1; }
    curl -sf http://127.0.0.1:$PORT/health >/dev/null 2>&1 && { ok=1; break; }
    sleep 1
  done
  [ $ok -eq 1 ] || { echo "$label HEALTH-TIMEOUT"; kill $pid 2>/dev/null; return 1; }
  curl -s http://127.0.0.1:$PORT/completion -H 'Content-Type: application/json' \
    -d '{"prompt":"Hello","n_predict":8,"temperature":0,"cache_prompt":false}' >/dev/null
  python3 - $PORT "$PROMPT" "$NP" > $LOGD/ws_${TAG}_${label}.tokens 2>$LOGD/ws_${TAG}_${label}.tokerr <<'PY'
import json,sys,urllib.request
port,prompt_path,np_=sys.argv[1],sys.argv[2],int(sys.argv[3])
prompt=open(prompt_path).read()
req=urllib.request.Request("http://127.0.0.1:%s/completion"%port,
  data=json.dumps({"prompt":prompt,"n_predict":np_,"temperature":0,"cache_prompt":False,"return_tokens":True}).encode(),
  headers={"Content-Type":"application/json"})
r=json.load(urllib.request.urlopen(req,timeout=1800))
print(json.dumps(r.get("tokens",[])))
PY
  kill $pid 2>/dev/null; wait $pid 2>/dev/null
  echo "$label tokens=$(python3 -c "import json;print(len(json.load(open('$LOGD/ws_${TAG}_${label}.tokens'))))" 2>/dev/null) $(grep -aoE 'draft acceptance = [0-9.]+' $LOGD/ws_${TAG}_${label}.err | tail -1)"
}

[ "$MODE" = all -o "$MODE" = none ] && run none --spec-type none
[ "$MODE" = all -o "$MODE" = n1 ] && run n1 --spec-type draft-mtp $SMD --spec-draft-n-max 1
[ "$MODE" = all -o "$MODE" = n3 ] && run n3 --spec-type draft-mtp $SMD --spec-draft-n-max 3
[ "$MODE" = all -o "$MODE" = n7 ] && run n7 --spec-type draft-mtp $SMD --spec-draft-n-max 7

if [ "$MODE" = all ]; then
python3 - "$TAG" <<'PY'
import json,sys
tag=sys.argv[1]; d="/tmp/srr"
def load(w):
    try: return json.load(open(f"{d}/ws_{tag}_{w}.tokens"))
    except Exception as e: return None
a=load("none")
for w in ["n1","n3","n7"]:
    b=load(w)
    if a is None or b is None:
        print(f"none vs {w}: MISSING"); continue
    n=min(len(a),len(b)); fd=next((i for i in range(n) if a[i]!=b[i]), None)
    print(f"none vs {w}: len {len(a)}/{len(b)} first-diff {fd}")
PY
fi
