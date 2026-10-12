# NInfer concurrency bench: one ninfer-serve with --max-concurrency N, N simultaneous greedy
# requests (distinct prompts, no prefix reuse); reports aggregate and per-request tok/s.
import argparse, json, os, subprocess, threading, time, urllib.request

EXE = r"D:\AI\ninfer-5090-windows\build\apps\ninfer-serve.exe"
PROMPTS = r"D:\AI\engine-compare-20261008\prompts"


def post(port, body, timeout=3600):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                 data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def wait_health(port, proc, limit=900):
    t0 = time.time()
    while time.time() - t0 < limit:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited {proc.returncode}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
                if r.status == 200:
                    return
        except Exception:
            time.sleep(1)
    raise RuntimeError("server never became healthy")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", required=True)
    ap.add_argument("--kv", default="bf16")
    ap.add_argument("--spec", default="dflash2")
    ap.add_argument("--conc", type=int, default=2)
    ap.add_argument("--prompt", default="code_4k")
    ap.add_argument("--max-context", type=int, default=16384)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--port", type=int, default=18090)
    ap.add_argument("--prefix-reuse", action="store_true", help="keep NInfer prefix reuse on (its default)")
    ap.add_argument("--kv-capacity", default="", help="auto | N (default: max_context * conc)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cmd = [EXE, a.artifact, "--port", str(a.port), "--max-context", str(a.max_context),
           "--kv-capacity", a.kv_capacity or str(a.max_context * a.conc), "--max-concurrency", str(a.conc), "--kv-dtype", a.kv,
           "--greedy", "--no-thinking", "--prefill-chunk", "8192"]
    if not a.prefix_reuse:
        cmd += ["--no-prefix-reuse"]
    if a.spec == "dflash2":
        cmd += ["--spec", "dflash2", "--draft-tokens", "7", "--lm-head-draft"]
    log = open(a.out + ".server.log", "w", encoding="utf-8")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=os.path.dirname(EXE))
    try:
        wait_health(a.port, proc)
        with urllib.request.urlopen(f"http://127.0.0.1:{a.port}/v1/models", timeout=10) as r:
            model_id = json.load(r)["data"][0]["id"]
        base = json.load(open(os.path.join(PROMPTS, a.prompt + ".json"), encoding="utf-8"))["messages"]
        bodies = []
        for i in range(a.conc):
            msgs = [dict(m) for m in base]
            msgs[-1]["content"] = f"[agent {i}] " + msgs[-1]["content"]
            bodies.append({"model": model_id, "messages": msgs, "max_tokens": a.max_tokens, "temperature": 0})
        warm = [dict(m) for m in base]
        warm[-1]["content"] = "[agent warm] " + warm[-1]["content"]
        post(a.port, {"model": model_id, "messages": warm, "max_tokens": a.max_tokens, "temperature": 0})  # warm
        results = [None] * a.conc

        def run(i):
            t0 = time.time()
            r = post(a.port, bodies[i])
            results[i] = (time.time() - t0, r["usage"]["completion_tokens"], r.get("timings") or r.get("usage"))

        t0 = time.time()
        threads = [threading.Thread(target=run, args=(i,)) for i in range(a.conc)]
        for t in threads: t.start()
        for t in threads: t.join()
        wall = time.time() - t0
        total = sum(r[1] for r in results)
        out = {"engine": "ninfer", "artifact": os.path.basename(a.artifact), "kv_capacity": a.kv_capacity or "conc", "prefix_reuse": a.prefix_reuse, "kv": a.kv, "spec": a.spec, "conc": a.conc, "prompt": a.prompt, "wall_s": wall,
               "completion_tokens": total, "aggregate_tps": total / wall,
               "per_request": [{"s": r[0], "tokens": r[1], "tps": r[1] / r[0]} for r in results]}
        print(json.dumps(out))
        json.dump(out, open(a.out, "w"), indent=1)
    finally:
        proc.terminate()
        try: proc.wait(30)
        except Exception: proc.kill()


main()
