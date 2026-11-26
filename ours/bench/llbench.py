#!/usr/bin/env python3
"""Replica of local-llm-benchmarks.dev "deterministic-prime-extend-completions-v1"
(coding-agent depth curves, workload "standard"), for llama-server.

Per depth D in 0,8K,...,64K, per repeat (1 warmup + 2 measured):
  1. prime: send the first D corpus tokens (n_predict 1) so they sit in the slot's KV cache (not timed)
  2. extend: same D tokens + a fresh 2048-token section, n_predict 256, temperature 0, streamed
     PP = uncached prompt tokens (server timings.prompt_n) / client time to first token
     TG = (generated tokens - 1) / (last token time - first token time)
Prompts are sent as token-id arrays, so lengths are exact. Every extension section is distinct,
so nothing in it is cached. One slot (id_slot), concurrency 1.

Differences from the site's runner (its source is not public): native /completion instead of
/v1/completions, ignore_eos so all 256 tokens are generated, corpus = local source files.

usage: llbench.py --url http://127.0.0.1:8080 --src <llama.cpp>/ggml/src [--src DIR ...] --out res.json
"""
import argparse, json, os, statistics, sys, time, urllib.request

def post(url, body):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=3600)

def corpus_tokens(base, dirs, need):
    files = []
    for d in dirs:
        for root, _, names in os.walk(os.path.expanduser(d)):
            files += [os.path.join(root, n) for n in names if n.endswith((".c", ".cpp", ".h", ".cu", ".cuh", ".py"))]
    toks = []
    for f in sorted(files):
        text = "// file: %s\n%s\n" % (os.path.basename(f), open(f, errors="replace").read())
        toks += json.load(post(base + "/tokenize", {"content": text}))["tokens"]
        if len(toks) >= need:
            return toks[:need]
    sys.exit("corpus too small: %d of %d tokens; add --src dirs" % (len(toks), need))

def run(base, prompt, n_predict, slot, stream):
    body = {"prompt": prompt, "n_predict": n_predict, "temperature": 0.0, "cache_prompt": True,
            "ignore_eos": True, "id_slot": slot, "stream": stream}
    t0 = time.perf_counter()
    r = post(base + "/completion", body)
    if not stream:
        return json.load(r)
    t_first = t_last = None
    timings = None
    for line in r:
        line = line.strip()
        if not line.startswith(b"data: "):
            continue
        ev = json.loads(line[6:])
        now = time.perf_counter()
        if ev.get("content") or ev.get("tokens"):
            t_first = t_first or now
            t_last = now
        if ev.get("stop"):
            timings = ev.get("timings", {})
    n = timings["predicted_n"]
    return {"ttft_s": t_first - t0, "pp_uncached": timings["prompt_n"],
            "pp_tps": timings["prompt_n"] / (t_first - t0),
            "tg_n": n, "tg_tps": (n - 1) / (t_last - t_first) if n > 1 and t_last > t_first else None,
            "server_prompt_tps": timings.get("prompt_per_second"),
            "server_gen_tps": timings.get("predicted_per_second"),
            "draft_n": timings.get("draft_n"), "draft_accepted": timings.get("draft_n_accepted")}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--src", action="append", required=True)
    ap.add_argument("--depths", default="0,8192,16384,24576,32768,40960,49152,57344,65536")
    ap.add_argument("--pp", type=int, default=2048)
    ap.add_argument("--tg", type=int, default=256)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--warmups", type=int, default=1)
    ap.add_argument("--slot", type=int, default=0)
    ap.add_argument("--out", default="llbench.json")
    a = ap.parse_args()
    depths = [int(x) for x in a.depths.split(",")]
    reps = a.warmups + a.repeats
    # layout: [prefix pool: max depth][one distinct pp section per (depth, repeat)]
    need = max(depths) + len(depths) * reps * a.pp
    toks = corpus_tokens(a.url, a.src, need)
    prefix_pool, sections = toks[:max(depths)], toks[max(depths):]
    results = []
    k = 0
    for d in depths:
        meas = []
        for i in range(reps):
            ext = sections[k * a.pp:(k + 1) * a.pp]; k += 1
            if d:
                run(a.url, prefix_pool[:d], 1, a.slot, False)          # prime (untimed)
            m = run(a.url, prefix_pool[:d] + ext, a.tg, a.slot, True)
            m.update(depth=d, repeat=i, warmup=i < a.warmups)
            results.append(m)
            if i >= a.warmups:
                meas.append(m)
            print("depth %6d %s pp %7.1f t/s (uncached %d, ttft %.2f s)  tg %6.1f t/s (n %d, draft %s/%s)" % (
                d, "warm" if i < a.warmups else "rep%d" % (i - a.warmups), m["pp_tps"], m["pp_uncached"],
                m["ttft_s"], m["tg_tps"] or 0, m["tg_n"], m["draft_accepted"], m["draft_n"]), flush=True)
        print("depth %6d MEAN pp %7.1f  tg %6.1f" % (d, statistics.mean(x["pp_tps"] for x in meas),
                                                     statistics.mean(x["tg_tps"] for x in meas)), flush=True)
    json.dump({"protocol": "prime-extend replica", "args": vars(a), "results": results}, open(a.out, "w"), indent=1)

if __name__ == "__main__":
    main()
