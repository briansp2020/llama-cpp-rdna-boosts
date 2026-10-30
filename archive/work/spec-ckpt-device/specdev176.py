# T0021 v2 on rdna-boosts r176 (for the PR): the target's speculative checkpoint in device memory, decided per slot.
# Same binary (~/lf-work/bin-r176-t21 = plain r176 + T0021 v2), LLAMA_SERVER_SPEC_CKPT_DEVICE=0 (host, the release's
# behaviour) against unset (auto), order 0 A A 0, one server start per arm, greedy, ROCBLAS_USE_HIPBLASLT=0, one request at a
# time.  Two surfaces of the maintainer's: (dense) Qwen3.8-27B UD-Q4_K_XL + DFlash2-Q4_K_M, 1 GPU, ngram-mod (default
# n-max 64) + draft-dflash 7; (moe) Qwen3.6-35B-A3B UD-Q4_K_M + its DFlash Q8_0 drafter at -ncmoe 40 (host experts, the
# expert-cache arena active).  Requests (all public): a whole-file rewrite (n-gram heavy), a zh short story, an English
# continuation of the first 24,000 characters of long-en (llama.cpp's docs/*.md concatenated, MIT).  Per request: t/s, draft / accepted, text md5; per arm: the slot's
# checkpoint decision (log), device memory after load and after the requests.
# Usage: python3 specdev176.py OUT_DIR [dense|moe ...]   (paths at the top are placeholders for the local ones).  Written by Claude (AI), 2026-10-30.
import hashlib, json, os, re, subprocess, sys, time, urllib.request
OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
B = "build/bin"                      # r176 + spec-ckpt-device.patch
R10 = "/opt/rocm/lib"                # the ROCm runtime
MD = "models"
G = "."                               # edit-input.py
TXT = "."                             # long-en.txt (llama.cpp's docs/*.md concatenated)
URL = "http://127.0.0.1:8199"
VRAM = "/sys/class/drm/card0/device/mem_info_vram_used"   # the GPU's sysfs node
SURF = {
    "dense": ["-m", f"{MD}/Qwen3.8-27B-UD-Q4_K_XL.gguf", "-md", f"{MD}/Qwen3.8-27B-DFlash2-Q4_K_M.gguf", "-ngl", "99", "-ngld", "99",
              "-c", "32768", "--parallel", "1", "--spec-type", "ngram-mod,draft-dflash", "--spec-draft-n-max", "7"],
    "moe":   ["-m", f"{MD}/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf", "-md", f"{MD}/qwen36-35b-a3b-dflash-Q8_0.gguf", "-ngl", "99", "-ngld", "99",
              "-ncmoe", "40", "-c", "32768", "--parallel", "1", "--spec-type", "ngram-mod,draft-dflash", "--spec-draft-n-max", "7"],
}

def post(path, body):
    req = urllib.request.Request(URL + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=3600).read())

reqs = {
    "edit": {"messages": [{"role": "user", "content": "Add a one-line comment above every function definition in this Python file "
             "saying what the function does. Change nothing else and output the whole modified file only.\n\n```python\n" +
             open(f"{G}/edit-input.py", encoding="utf-8").read() + "```"}], "max_tokens": 3072},
    "story": {"messages": [{"role": "user", "content": "請寫一篇約八百字的短篇小說，主題是颱風夜裡的一間便利商店，主角是值大夜班的店員。"}],
              "max_tokens": 1536},
    "cont": {"messages": [{"role": "user", "content": "Continue this text for about 600 words, in the same style:\n\n" +
             open(f"{TXT}/long-en.txt", encoding="utf-8").read()[:24000]}], "max_tokens": 1024},
}
for r in reqs.values():
    r.update(temperature=0, chat_template_kwargs={"enable_thinking": False})

allres = {}
for surf in sys.argv[2:] or ["dense", "moe"]:
    for k, dev in enumerate(("0", "A", "A", "0")):
        env = {x: y for x, y in os.environ.items() if not x.startswith(("GGML_", "LLAMA_", "HIP", "ROCBLAS_"))}
        env.update(ROCBLAS_USE_HIPBLASLT="0", LD_LIBRARY_PATH=f"{B}:{R10}")
        if dev == "0":
            env["LLAMA_SERVER_SPEC_CKPT_DEVICE"] = "0"
        tag = f"{surf}-{k}-{dev}"
        log = f"{OUT}/server-{tag}.log"
        p = subprocess.Popen([f"{B}/llama-server"] + SURF[surf] + ["-ub", "2048", "-b", "2048", "-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0",
                              "--jinja", "--host", "127.0.0.1", "--port", "8199", "--no-webui"],
                             cwd=B, env=env, stdout=open(log, "w"), stderr=subprocess.STDOUT, start_new_session=True)
        res = allres[tag] = {}
        try:
            for _ in range(900):
                if p.poll() is not None: raise SystemExit(f"{tag}: server exited")
                try:
                    if b'"ok"' in urllib.request.urlopen(URL + "/health", timeout=2).read(): break
                except Exception: pass
                time.sleep(1)
            time.sleep(2)
            res["vram_load"] = round(int(open(VRAM).read()) / 2**30, 2)
            for name, body in reqs.items():
                out = post("/v1/chat/completions", body)
                t = out["timings"]; txt = out["choices"][0]["message"].get("content") or ""
                res[name] = dict(tps=round(t["predicted_per_second"], 1), n=t["predicted_n"], draft=t.get("draft_n"),
                                 acc=t.get("draft_n_accepted"), md5=hashlib.md5(txt.encode()).hexdigest()[:10])
            res["vram_end"] = round(int(open(VRAM).read()) / 2**30, 2)
        finally:
            os.killpg(p.pid, 15)
            try: p.wait(90)
            except subprocess.TimeoutExpired: os.killpg(p.pid, 9); p.wait()
            time.sleep(3)
        res["decision"] = [re.sub(r"^.*?\| ", "", l.strip())[:160] for l in open(log, errors="replace") if "speculative checkpoints of the target" in l]
        print(tag, json.dumps(res, ensure_ascii=False), flush=True)
json.dump(allres, open(f"{OUT}/specdev176.json", "w"), ensure_ascii=False, indent=1)
for surf in sys.argv[2:] or ["dense", "moe"]:
    for name in reqs:
        row = [(t.split("-")[2], allres[t][name]) for t in allres if t.startswith(surf + "-") and name in allres[t]]
        h = [r["tps"] for d, r in row if d == "0"]; a = [r["tps"] for d, r in row if d == "A"]
        same = len({r["md5"] for d, r in row}) == 1
        if h and a:
            print(f"{surf:5} {name:5} host {sum(h)/len(h):6.1f}  auto {sum(a)/len(a):6.1f}  ({(sum(a)/len(a))/(sum(h)/len(h))-1:+.1%})  text {'same' if same else 'DIFFERENT'}")