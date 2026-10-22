# Multi-agent long-context client (both engines): N agents with distinct long code prompts
# (disjoint slices of one source file), sent concurrently each turn; later turns append the
# assistant reply and a follow-up question (same conversation -> prefix reuse). Greedy.
# Mirrors the VoidInfer conc_long harness so both engines see identical requests.
import argparse, json, subprocess, threading, time, urllib.request

SRC = r"D:\AI\kvtier\long_code_source.cpp"


def post(port, body, timeout=7200):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                 data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def gpu_used():
    try:
        return int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]).split()[0])
    except Exception:
        return -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--model", default="conc")
    ap.add_argument("--agents", type=int, default=8)
    ap.add_argument("--chars", type=int, default=400000)
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--turns", type=int, default=2)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    src = open(SRC, encoding="utf-8", errors="ignore").read()
    while len(src) < (a.chars + 1000) * a.agents:
        src += src
    convs = []
    for i in range(a.agents):
        text = src[(a.chars + 1000) * i:(a.chars + 1000) * i + a.chars]
        convs.append([{"role": "system", "content": f"You are agent {i}."},
                      {"role": "user", "content": f"Here is source code:\n{text}\n\nWrite a detailed review of this code, function by function."}])
    turns = []
    for turn in range(a.turns):
        results = [None] * a.agents
        peak = [gpu_used()]
        stop = threading.Event()

        def sample():
            while not stop.is_set():
                peak[0] = max(peak[0], gpu_used())
                time.sleep(2)

        def run(i):
            t0 = time.time()
            try:
                r = post(a.port, {"model": a.model, "messages": convs[i], "max_tokens": a.tokens, "temperature": 0})
                results[i] = {"s": time.time() - t0, "prompt_tokens": r["usage"]["prompt_tokens"],
                              "completion_tokens": r["usage"]["completion_tokens"],
                              "text": r["choices"][0]["message"].get("content", "")}
            except Exception as e:
                results[i] = {"s": time.time() - t0, "error": str(e)[:300]}

        sampler = threading.Thread(target=sample); sampler.start()
        t0 = time.time()
        threads = [threading.Thread(target=run, args=(i,)) for i in range(a.agents)]
        for t in threads: t.start()
        for t in threads: t.join()
        wall = time.time() - t0
        stop.set(); sampler.join()
        ok = [r for r in results if "error" not in r]
        rec = {"turn": turn, "wall_s": wall, "ok": len(ok), "errors": [r["error"] for r in results if "error" in r],
               "prompt_tokens": sum(r["prompt_tokens"] for r in ok),
               "completion_tokens": sum(r["completion_tokens"] for r in ok),
               "per_request_s": [round(r["s"], 1) for r in results], "peak_gpu_mib": peak[0]}
        print(json.dumps(rec), flush=True)
        rec["texts"] = [r.get("text", "") for r in results]
        turns.append(rec)
        for i, r in enumerate(results):
            if "text" in r:
                convs[i].append({"role": "assistant", "content": r["text"]})
                convs[i].append({"role": "user", "content": f"Thanks. Now explain the next function in more detail (turn {turn})."})
    json.dump({"agents": a.agents, "chars": a.chars, "tokens": a.tokens, "turns": turns}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
