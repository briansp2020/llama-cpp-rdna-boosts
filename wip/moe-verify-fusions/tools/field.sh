#!/bin/bash
# §5.5 field A/B arm.  Usage: field.sh <build_dir> <tag> <fit|run30k>
# fit     : --fit on, small request -> fitted n_ctx + resulting arena
# run30k  : --fit off, 30k-token prompt + 1000-token MTP decode -> prefill/decode t/s
set -u
BUILD=$1; TAG=$2; MODE=$3
# Model of choice for this campaign (D2): the GSQ-IQ3_XXS set -- warm in the page cache.
#   M=/llm/models/Qwen3.8/Flash-Next/GSQ-IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf
M=${M:-/llm/models/Qwen3.8/Flash-Next/IQ4_NL/Qwen3.8-Flash-Next-IQ4_NL-PROJFIX-00001-of-00009.gguf}
D=${D:-/llm/models/Qwen3.8/Flash-Next/IQ4_NL/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf}   # GSQ-IQ3_XXS has NO embedded MTP, so this shared head is required
PORT=8942
LOGD=/tmp/srr; mkdir -p $LOGD
export LD_LIBRARY_PATH=/opt/rocm-7.14-gfx120X/lib:${LD_LIBRARY_PATH:-}
export HIP_VISIBLE_DEVICES=${GPUS:-0,1}
export GGML_CUDA_ALLREDUCE=ce

[ -x "$BUILD/bin/llama-server" ] || { echo "no llama-server in $BUILD/bin"; exit 2; }
FIT=on; NPRE=6
case "$MODE" in run30k) FIT=off; NPRE=1000;; run30kfit) FIT=on; NPRE=1000;; esac

"$BUILD/bin/llama-server" -v \
  -m $M --spec-draft-model $D \
  -sm tensor -ncmoe 48 -ub 6144 -b 6144 -c 204800 --no-kv-unified \
  -ctk q8_0 -ctv q8_0 -fa on -t 8 --fit $FIT \
  --spec-type draft-mtp --spec-draft-n-max 3 \
  --host 127.0.0.1 --port $PORT --no-webui --cache-ram 0 \
  > $LOGD/${TAG}.out 2> $LOGD/${TAG}.err &
pid=$!

ok=0
for i in $(seq 1 300); do if curl -sf http://127.0.0.1:$PORT/health >/dev/null 2>&1; then ok=1; break; fi; sleep 1; done
if [ $ok -ne 1 ]; then echo "=== $TAG health timeout ==="; grep -aE 'error|Error|failed|GGML_ASSERT' $LOGD/${TAG}.err | tail -5; kill $pid 2>/dev/null; exit 1; fi

# drive it: small prime (sizes the arena), then the measured prompt
curl -s http://127.0.0.1:$PORT/completion -H 'Content-Type: application/json' \
  -d '{"prompt":"Hello","n_predict":8,"temperature":0,"cache_prompt":false}' >/dev/null

if [ "$MODE" = fit ]; then
  sleep 6
  echo "=== $TAG (fit) ==="
  grep -aE '^[0-9.]+ [IWE] ' $LOGD/${TAG}.err \
    | grep -aE 'tensor split: device [0-9]+ target|MoE expert cache \(auto\)|llama_context: n_ctx  |slab [0-9.]+ GiB|moe_cache_preflight: auto cache' \
    | sed 's/^[0-9. ]*//' | head -14
  kill $pid 2>/dev/null; wait $pid 2>/dev/null
  exit 0
fi

# run30k
python3 - $PORT $NPRE > $LOGD/${TAG}.timing 2>&1 <<'PY'
import json,sys,urllib.request
port,npre=sys.argv[1],int(sys.argv[2])
prompt=open(__import__('os').environ.get('PROMPT','/tmp/srr/mixed30k.txt')).read()  # NOT prose30k: that one is degenerate (the model copies it)
req=urllib.request.Request("http://127.0.0.1:%s/completion"%port,
  data=json.dumps({"prompt":prompt,"n_predict":npre,"temperature":0,"cache_prompt":False}).encode(),
  headers={"Content-Type":"application/json"})
r=json.load(urllib.request.urlopen(req,timeout=1800))
t=r.get("timings",{})
print("prompt_n=%s prompt_tps=%.1f predicted_n=%s predicted_tps=%.1f"%(t.get("prompt_n"),t.get("prompt_per_second",0),t.get("predicted_n"),t.get("predicted_per_second",0)))
PY
echo "=== $TAG (run30k) ==="
cat $LOGD/${TAG}.timing
grep -aoE 'draft acceptance = [0-9.]+ \( *[0-9]+ accepted / *[0-9]+ generated\)' $LOGD/${TAG}.err | tail -1 | sed 's/^/  /'
grep -aE '^[0-9.]+ [IWE] ' $LOGD/${TAG}.err | grep -aE 'MoE expert cache \(auto\)|moe_cache_evict_slab_range|moe_cache_rearm|evicting' | tail -4 | sed 's/^[0-9. ]*//'
grep -aE '^[0-9.]+ E ' $LOGD/${TAG}.err | grep -avE 'http|operator' | tail -3 | sed 's/^[0-9. ]*/  ERROR: /'
kill $pid 2>/dev/null; wait $pid 2>/dev/null
