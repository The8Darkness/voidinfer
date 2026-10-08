# NInfer side of the 1:1 comparison: one ninfer-serve per (artifact, kv, spec, context) config,
# one warm request then `repeats` measured requests per prompt, greedy, thinking off, no prefix reuse.
import argparse, json, os, subprocess, sys, time, urllib.request

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
                    return time.time() - t0
        except Exception:
            time.sleep(1)
    raise RuntimeError("server never became healthy")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--kv", default="bf16")
    ap.add_argument("--spec", default="none")  # none | dflash2
    ap.add_argument("--max-context", type=int, required=True)
    ap.add_argument("--max-tokens", type=int, required=True)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--port", type=int, default=18080)
    ap.add_argument("--prefill-chunk", type=int, default=1024)
    ap.add_argument("--out", required=True)
    ap.add_argument("prompts", nargs="+")
    a = ap.parse_args()
    cmd = [EXE, a.artifact, "--port", str(a.port), "--max-context", str(a.max_context),
           "--kv-capacity", str(a.max_context), "--max-concurrency", "1", "--kv-dtype", a.kv,
           "--no-prefix-reuse", "--greedy", "--no-thinking", "--prefill-chunk", str(a.prefill_chunk)]
    if a.spec == "dflash2":
        cmd += ["--spec", "dflash2", "--draft-tokens", "7", "--lm-head-draft"]
    log = open(a.out + ".server.log", "w", encoding="utf-8")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=os.path.dirname(EXE))
    rows = []
    try:
        load_s = wait_health(a.port, proc)
        with urllib.request.urlopen(f"http://127.0.0.1:{a.port}/v1/models", timeout=10) as r:
            model_id = json.load(r)["data"][0]["id"]
        for name in a.prompts:
            msgs = json.load(open(os.path.join(PROMPTS, name + ".json"), encoding="utf-8"))["messages"]
            body = {"model": model_id, "messages": msgs, "max_tokens": a.max_tokens, "temperature": 0}
            for rep in range(a.repeats + 1):
                r = post(a.port, body)
                t = r["timings"]
                row = {"label": a.label, "kv": a.kv, "spec": a.spec, "prefill_chunk": a.prefill_chunk, "prompt": name, "rep": rep,
                       "prompt_tokens": r["usage"]["prompt_tokens"], "cache_n": t.get("cache_n"),
                       "prefill_ms": t["prompt_ms"], "prefill_tps": t["prompt_per_second"],
                       "out_tokens": t["predicted_n"], "decode_ms": t["predicted_ms"],
                       "decode_tps": t["predicted_per_second"], "draft_n": t.get("draft_n"),
                       "draft_accepted": t.get("draft_n_accepted"),
                       "finish": r["choices"][0].get("finish_reason"),
                       "text_head": r["choices"][0]["message"].get("content", "")[:200]}
                rows.append(row)
                print(json.dumps({k: row[k] for k in ("prompt", "rep", "prompt_tokens", "prefill_tps",
                                                      "out_tokens", "decode_tps", "draft_n", "draft_accepted",
                                                      "finish")}), flush=True)
                if rep == a.repeats:
                    open(a.out + f".{name}.txt", "w", encoding="utf-8").write(
                        r["choices"][0]["message"].get("content", ""))
        json.dump({"load_seconds": load_s, "cmd": cmd, "rows": rows}, open(a.out, "w"), indent=1)
    finally:
        proc.terminate()
        try:
            proc.wait(30)
        except Exception:
            proc.kill()


if __name__ == "__main__":
    main()
