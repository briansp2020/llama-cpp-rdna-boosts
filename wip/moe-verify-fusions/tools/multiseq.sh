#!/bin/bash
# Multi-sequence GDN-state control (the PR #124 reporter's 2->1 scenario) on OUR build (no gather skip).
# Two concurrent greedy /completion requests A (long) and B (short, finishes first); A keeps decoding
# after B ends.  Each prompt is also run solo.  Repetitions of the concurrent phase must match the solo
# output and each other; any difference between identical repetitions is non-determinism.
#   multiseq.sh <build_dir> <tag>
set -u
BUILD=$1; TAG=$2
M=${M:-/llm/models/Qwen3.8/Flash-Next/GSQ-IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf}
D=${D-/llm/models/Qwen3.8/Flash-Next/IQ4_NL/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf}
PROMPT_A=${PROMPT_A:-/tmp/srr/mixed30k.txt}
PROMPT_B=${PROMPT_B:-/tmp/srr/msqB.txt}
NPA=${NPA:-256}; NPB=${NPB:-96}
PORT=${PORT:-8965}
NCMOE=${NCMOE:-48}
CTX=${CTX:-204800}
KVUNI=${KVUNI:-1}
LOGD=/tmp/srr; mkdir -p $LOGD
export LD_LIBRARY_PATH=/opt/rocm-7.14.1-gfx120X/lib:${LD_LIBRARY_PATH:-}
export HIP_VISIBLE_DEVICES=${GPUS:-0,1}
KVD=""; [ "$KVUNI" = "1" ] && KVD="--kv-unified"; [ "$KVUNI" = "0" ] && KVD="--no-kv-unified"
SMD=""; [ -n "$D" ] && SMD="--spec-draft-model $D"

"$BUILD/bin/llama-server" -v -m "$M" \
  -sm tensor -ncmoe "$NCMOE" -ub 6144 -b 6144 -c "$CTX" $KVD \
  -ctk q8_0 -ctv q8_0 -fa on -t 8 --fit on \
  --spec-type draft-mtp $SMD --spec-draft-n-max 3 \
  --host 127.0.0.1 --port $PORT --no-webui --cache-ram 0 \
  > $LOGD/${TAG}.out 2> $LOGD/${TAG}.err &
pid=$!
ok=0
for i in $(seq 1 400); do
  kill -0 $pid 2>/dev/null || { echo "$TAG SERVER-DIED"; tail -6 $LOGD/${TAG}.err; exit 1; }
  curl -sf http://127.0.0.1:$PORT/health >/dev/null 2>&1 && { ok=1; break; }
  sleep 1
done
[ $ok -eq 1 ] || { echo "$TAG HEALTH-TIMEOUT"; kill $pid 2>/dev/null; exit 1; }
curl -s http://127.0.0.1:$PORT/completion -H 'Content-Type: application/json' \
  -d '{"prompt":"Hello","n_predict":8,"temperature":0,"cache_prompt":false}' >/dev/null

python3 - $PORT "$PROMPT_A" "$PROMPT_B" "$NPA" "$NPB" > $LOGD/${TAG}.seq 2> $LOGD/${TAG}.seqerr <<'PY'
import json, sys, threading, hashlib, urllib.request
port, pA, pB, npa, npb = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
A = open(pA).read(); B = open(pB).read()

def gen(prompt, n):
    body = json.dumps({"prompt": prompt, "n_predict": n, "temperature": 0.0, "top_k": 1, "seed": 1,
                       "cache_prompt": False, "ignore_eos": True, "return_tokens": True}).encode()
    req = urllib.request.Request("http://127.0.0.1:%s/completion" % port, data=body,
                                 headers={"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=7200))
    return r.get("tokens", [])

def h(t):
    return hashlib.sha256(json.dumps(t).encode()).hexdigest()[:12]

a_solo = gen(A, npa); b_solo = gen(B, npb)
print("A_solo %4d %s" % (len(a_solo), h(a_solo)))
print("B_solo %4d %s" % (len(b_solo), h(b_solo)))

res = {}
def run(key, prompt, n):
    res[key] = gen(prompt, n)

for rep in (1, 2, 3):
    t1 = threading.Thread(target=run, args=("A%d" % rep, A, npa))
    t2 = threading.Thread(target=run, args=("B%d" % rep, B, npb))
    t1.start(); t2.start(); t1.join(); t2.join()
    print("A_conc%d %4d %s   B_conc%d %4d %s" % (rep, len(res["A%d" % rep]), h(res["A%d" % rep]),
                                                 rep, len(res["B%d" % rep]), h(res["B%d" % rep])))
# first divergence of each concurrent A vs solo A
def fd(x, y):
    for i in range(min(len(x), len(y))):
        if x[i] != y[i]:
            return i
    return None
for rep in (1, 2, 3):
    print("A_conc%d vs A_solo first-diff: %s" % (rep, fd(res["A%d" % rep], a_solo)))
for rep in (2, 3):
    print("A_conc%d vs A_conc1 first-diff: %s" % (rep, fd(res["A%d" % rep], res["A1"])))
PY
kill $pid 2>/dev/null; wait $pid 2>/dev/null
echo "=== $TAG (multiseq) ==="
cat $LOGD/${TAG}.seq
grep -aoE 'draft acceptance = [0-9.]+' $LOGD/${TAG}.err | tail -2 | sed 's/^/  /'
grep -ac "rollback crossed a batch boundary" $LOGD/${TAG}.err | sed 's/^/  rollback-crossed-warning count: /'
