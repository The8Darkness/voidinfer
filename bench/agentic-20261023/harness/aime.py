"""AIME 2026 (2026-10-25): both engines, current release builds, thinking on, MathArena-style avg@4.

30 problems (AIME I + II) from quality-bench-20261012/data/aime2026.jsonl, prompt = problem + the \\boxed{} instruction,
Qwen thinking sampling (T 1.0, top_p 0.95, top_k 20), 81,920-token output budget, 2 requests in flight on a C2 server
with window 86,016 (NInfer's C2 maximum), 4 samples per problem. Scored with the quality-bench math scorer.

Usage: python -I aime.py --engine ours|nv [--samples 4] [--first 0] [--conc 2 --window 86016 --max-tokens 81920 --tag aime2026]
       -> results/<tag>/<engine>.jsonl (resumable; --first/--samples pick sample indices so engines can alternate runs)
       python -I aime.py --report
"""
import argparse, http.client, json, math, os, random, sys, threading, time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

os.environ["BENCH_THINKING"] = "1"   # servers start without --no-thinking
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bench  # noqa: E402

sys.path.insert(0, os.path.join(HERE, "..", "..", "engine-compare", "tasks"))
from suites import BOXED, compmath_score, last_boxed  # noqa: E402

# AIME 2026 I + II as jsonl with fields problem_idx, problem, answer (MathArena layout)
DATA = os.environ.get("AB_AIME_DATA", os.path.join(HERE, "..", "data", "aime2026.jsonl"))
SAMPLING = {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 0.0}


def items():
    for l in open(DATA, encoding="utf-8"):
        r = json.loads(l)
        yield {"id": f"aime2026-{r['problem_idx']}", "prompt": f"{r['problem']}\n{BOXED}", "gold": str(r["answer"])}


def post(body):
    conn = http.client.HTTPConnection("127.0.0.1", bench.PORT, timeout=7200)
    conn.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
    r = conn.getresponse()
    data = r.read()
    conn.close()
    if r.status != 200:
        raise RuntimeError(f"HTTP {r.status}: {data[:300]!r}")
    return json.loads(data)


def run(a):
    OUT = os.path.join(bench.ROOT, "results", a.tag)
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"{a.engine}.jsonl")
    done = set()
    if os.path.exists(path):
        done = {(r["id"], r["sample"]) for r in map(json.loads, open(path, encoding="utf-8")) if "error" not in r}
    its = list(items())
    todo = [(it, s) for s in range(a.first, a.first + a.samples) for it in its if (it["id"], s) not in done]
    srv = bench.Server(a.engine, a.conc, a.window, os.path.join(OUT, "logs"))
    if not srv.ok:
        print("server failed", srv.log_path); srv.close(); sys.exit(2)
    print(a.engine, "server up in", round(srv.startup_s), "s;", len(todo), "generations to run", flush=True)
    lock, t_start, tok = threading.Lock(), time.time(), [0]
    out = open(path, "a", encoding="utf-8")

    def one(job):
        it, s = job
        body = {"model": "m", "messages": [{"role": "user", "content": it["prompt"]}], "max_tokens": a.max_tokens,
                **SAMPLING, "enable_thinking": True}
        t0 = time.time()
        try:
            r = post(body)
            msg = r["choices"][0]["message"]
            text = msg.get("content") or ""
            rec = {"id": it["id"], "sample": s, "gold": it["gold"], "boxed": last_boxed(text), "correct": compmath_score(it, text) == 1.0,
                   "finish": r["choices"][0].get("finish_reason"), "completion_tokens": (r.get("usage") or {}).get("completion_tokens"),
                   "reasoning_chars": len(msg.get("reasoning_content") or ""), "content_tail": text[-300:], "seconds": round(time.time() - t0, 1)}
        except Exception as e:
            rec = {"id": it["id"], "sample": s, "error": str(e)[:300], "seconds": round(time.time() - t0, 1)}
        with lock:
            out.write(json.dumps(rec, ensure_ascii=False) + "\n"); out.flush()
            tok[0] += rec.get("completion_tokens") or 0
            n = sum(1 for _ in open(path, encoding="utf-8"))
            print(f"{a.engine} {it['id']} s{s}: {'OK ' if rec.get('correct') else 'bad'} {rec.get('boxed')!s:>8} gold {it['gold']:>4} "
                  f"{rec.get('completion_tokens')} tok {rec.get('finish')} {rec['seconds']}s  [{n} rows, "
                  f"{tok[0] / max(1, time.time() - t_start):.0f} tok/s]", flush=True)

    try:
        with ThreadPoolExecutor(a.conc) as pool:
            list(pool.map(one, todo))
    finally:
        out.close()
        srv.close()


def report(a):
    OUT = os.path.join(bench.ROOT, "results", a.tag)
    R = {}
    for e in ("ours", "nv"):
        p = os.path.join(OUT, f"{e}.jsonl")
        if os.path.exists(p):
            rs = [r for r in map(json.loads, open(p, encoding="utf-8")) if "error" not in r]
            R[e] = {(r["id"], r["sample"]): r for r in rs}
    ids = ([it["id"] for it in items()] if os.path.exists(DATA)   # the report also works from the stored results alone
           else sorted({i for rs in R.values() for (i, _) in rs}, key=lambda x: int(x.split("-")[1])))
    lines = [f"### AIME 2026 ({a.tag}: 30 problems, thinking, T 1.0 / top_p 0.95 / top_k 20, {a.max_tokens:,}-token budget, "
             f"C{a.conc} window {a.window:,})", "",
             "| | ours | NInfer |", "|---|---|---|"]
    stat = {}
    for e, rs in R.items():
        per = defaultdict(list)
        for (i, s), r in rs.items():
            per[i].append(r)
        n = len(rs)
        acc = sum(r["correct"] for r in rs.values()) / max(1, n)
        maj = sum(1 for i in ids if per[i] and Counter(str(r["boxed"]) for r in per[i]).most_common(1)[0][0] == per[i][0]["gold"])
        passk = sum(1 for i in ids if any(r["correct"] for r in per[i]))
        toks = [r["completion_tokens"] or 0 for r in rs.values()]
        secs = [r["seconds"] for r in rs.values()]
        stat[e] = {"gens": n, "avg": acc, "maj": maj, "pass": passk, "trunc": sum(r["finish"] == "length" for r in rs.values()),
                   "tok": sum(toks) / max(1, n), "sec": sum(secs) / max(1, n),
                   "per": {i: sum(r["correct"] for r in per[i]) / len(per[i]) for i in ids if per[i]}}
    row = lambda k, f: "| " + k + " | " + " | ".join(f(stat[e]) if e in stat else "-" for e in ("ours", "nv")) + " |"
    lines += [row("generations", lambda s: str(s["gens"])),
              row("avg@k accuracy", lambda s: f"{100 * s['avg']:.1f}% ({s['avg'] * 30:.1f}/30)"),
              row("majority vote /30", lambda s: str(s["maj"])),
              row("pass@k /30", lambda s: str(s["pass"])),
              row("truncated at budget", lambda s: str(s["trunc"])),
              row("mean completion tokens", lambda s: f"{s['tok']:.0f}"),
              row("mean seconds per generation", lambda s: f"{s['sec']:.0f}")]
    if len(stat) == 2:
        common = [i for i in ids if i in stat["ours"]["per"] and i in stat["nv"]["per"]]
        d = [stat["ours"]["per"][i] - stat["nv"]["per"][i] for i in common]
        obs = abs(sum(d))
        rng = random.Random(0)
        hits = sum(abs(sum(x if rng.random() < 0.5 else -x for x in d)) >= obs - 1e-9 for _ in range(20000))
        lines += ["", f"Paired by problem ({len(common)} problems): mean difference {100 * sum(d) / max(1, len(d)):+.1f} points, "
                  f"sign-flip permutation p = {hits / 20000:.3f}; problems where ours is better / worse: "
                  f"{sum(x > 0 for x in d)} / {sum(x < 0 for x in d)}", "",
                  "Per problem (correct samples, ours / NInfer): " + ", ".join(
                      f"{i.split('-')[1]}: {sum(r['correct'] for (j, _), r in R['ours'].items() if j == i)}/"
                      f"{sum(r['correct'] for (j, _), r in R['nv'].items() if j == i)}" for i in common)]
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["ours", "nv"])
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--first", type=int, default=0, help="first sample index")
    ap.add_argument("--conc", type=int, default=2)
    ap.add_argument("--window", type=int, default=86016)
    ap.add_argument("--max-tokens", type=int, default=81920)
    ap.add_argument("--tag", default="aime2026")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    report(a) if a.report else run(a)


if __name__ == "__main__":
    main()
