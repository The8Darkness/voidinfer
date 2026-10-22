# Concurrency client shared by both engines: N simultaneous greedy requests (distinct
# "[agent i]" prompts derived from one fixture), after one warm request.
import argparse, json, os, threading, time, urllib.request

PROMPTS = r"D:\AI\engine-compare-20261008\prompts"


def post(port, body, timeout=3600):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                 data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--conc", type=int, default=2)
    ap.add_argument("--prompt", default="code_4k")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--label", default="")
    a = ap.parse_args()
    with urllib.request.urlopen(f"http://127.0.0.1:{a.port}/v1/models", timeout=10) as r:
        model_id = json.load(r)["data"][0]["id"]
    base = json.load(open(os.path.join(PROMPTS, a.prompt + ".json"), encoding="utf-8"))["messages"]
    bodies = []
    for i in range(a.conc):
        msgs = [dict(m) for m in base]
        msgs[-1]["content"] = f"[agent {i + int(os.environ.get('AGENT_OFFSET', '0'))}] " + msgs[-1]["content"]
        bodies.append({"model": model_id, "messages": msgs, "max_tokens": a.max_tokens, "temperature": 0})
    # Warm with a prompt outside the measured set (no prefix-reuse advantage for agent 0).
    warm = [dict(m) for m in base]
    warm[-1]["content"] = "[agent warm] " + warm[-1]["content"]
    post(a.port, {"model": model_id, "messages": warm, "max_tokens": a.max_tokens, "temperature": 0})
    results = [None] * a.conc

    def run(i):
        t0 = time.time()
        r = post(a.port, bodies[i])
        results[i] = (time.time() - t0, r["usage"]["completion_tokens"])

    t0 = time.time()
    threads = [threading.Thread(target=run, args=(i,)) for i in range(a.conc)]
    for t in threads: t.start()
    for t in threads: t.join()
    wall = time.time() - t0
    total = sum(r[1] for r in results)
    print(json.dumps({"label": a.label, "conc": a.conc, "prompt": a.prompt, "wall_s": wall,
                      "completion_tokens": total, "aggregate_tps": total / wall,
                      "per_request_tps": [round(r[1] / r[0], 1) for r in results]}))


main()
