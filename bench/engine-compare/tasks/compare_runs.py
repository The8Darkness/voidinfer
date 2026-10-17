"""Compares repeated runs of two engine configs on the same suites: totals (accuracy, output tokens,
wall time), run-to-run agreement, and per-problem tables (correct / tokens / seconds per run).

  compare_runs.py --runs D:\\...\\runs_200k_r1 D:\\...\\runs_200k_r2 --configs ours nv.fp8 \\
                  --suites math500 aime2025 --walls queue.log --out report_dir

Wall times come from "Q200K r=<n> <config> <suite> ... wall=<s>s" lines in --walls (the queue output);
without them the per-item seconds are summed (latency, not exclusive time under concurrency).
"""
import argparse, csv, json, math, os, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from suites import SUITES


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def load(path):
    if not os.path.exists(path):
        return {}
    return {r["id"]: r for r in map(json.loads, filter(str.strip, open(path, encoding="utf-8")))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--suites", nargs="+", required=True)
    ap.add_argument("--walls", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    walls = {}
    if a.walls:
        for l in open(a.walls, encoding="utf-8", errors="ignore"):
            m = re.search(r"Q200K r=(\d+) (\S+) (\S+) conc=\d+ rc=\d+ wall=(\d+)s", l)
            if m:
                walls[(int(m[1]), m[2], m[3])] = int(m[4])
    md = ["# Repeated-run comparison", ""]
    for suite in a.suites:
        items_fn, score = SUITES[suite][0], SUITES[suite][1]
        items = {it["id"]: it for it in items_fn()}
        ids = list(items)
        data = {(ri, c): load(os.path.join(run, c, suite + ".jsonl"))
                for ri, run in enumerate(a.runs, 1) for c in a.configs}
        res = {}
        for key, recs in data.items():
            for i in ids:
                r = recs.get(i)
                if r is None or "error" in r:
                    res[key, i] = None
                    continue
                res[key, i] = (bool(score(items[i], r.get("content", ""))), (r.get("usage") or {}).get("completion_tokens", 0),
                               r.get("seconds", 0.0), r.get("finish_reason"))
        md += [f"## {suite} ({len(ids)} problems)", "",
               "| Run | Config | Answered | Correct | Accuracy | Truncated | Errors | Output tokens | Mean tokens | Wall time | Output tok/s |",
               "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for (ri, c) in data:
            got = [res[(ri, c), i] for i in ids if res[(ri, c), i] is not None]
            errs = sum(1 for i in ids if (ri, c) in data and i in data[ri, c] and "error" in data[ri, c][i])
            ok = sum(g[0] for g in got)
            tok = sum(g[1] for g in got)
            wall = walls.get((ri, c, suite)) or sum(g[2] for g in got)
            trunc = sum(g[3] == "length" for g in got)
            md.append(f"| {ri} | {c} | {len(got)} | {ok} | {100 * ok / max(1, len(ids)):.1f}% | {trunc} | {errs} | {tok:,} | "
                      f"{tok / max(1, len(got)):,.0f} | {wall / 60:.1f} min | {tok / max(1, wall):.0f} |")
        md.append("")
        # Pooled per config across runs and paired comparisons.
        md += ["| Config | Mean accuracy | Problems right in every run | Run-to-run flips | Total output tokens | Total wall time |",
               "|---|---:|---:|---:|---:|---:|"]
        for c in a.configs:
            runs = range(1, len(a.runs) + 1)
            accs = [sum(1 for i in ids if res[(ri, c), i] and res[(ri, c), i][0]) / len(ids) for ri in runs]
            allr = sum(1 for i in ids if all(res[(ri, c), i] and res[(ri, c), i][0] for ri in runs))
            flips = sum(1 for i in ids if len({bool(res[(ri, c), i] and res[(ri, c), i][0]) for ri in runs}) > 1)
            tok = sum(res[(ri, c), i][1] for ri in runs for i in ids if res[(ri, c), i])
            wall = sum(walls.get((ri, c, suite)) or 0 for ri in runs)
            md.append(f"| {c} | {100 * sum(accs) / len(accs):.1f}% | {allr} | {flips} | {tok:,} | {wall / 60:.1f} min |")
        md.append("")
        if len(a.configs) == 2:
            c0, c1 = a.configs
            md += [f"| Run | {c0} only right | {c1} only right | McNemar p |", "|---|---:|---:|---:|"]
            for ri in range(1, len(a.runs) + 1):
                b = sum(1 for i in ids if res[(ri, c0), i] and res[(ri, c1), i] and res[(ri, c0), i][0] and not res[(ri, c1), i][0])
                d = sum(1 for i in ids if res[(ri, c0), i] and res[(ri, c1), i] and res[(ri, c1), i][0] and not res[(ri, c0), i][0])
                md.append(f"| {ri} | {b} | {d} | {mcnemar(b, d):.3f} |")
            md.append("")
        # Per-problem CSV (all problems) and markdown (all for small suites, disagreements for large ones).
        cols = [(ri, c) for c in a.configs for ri in range(1, len(a.runs) + 1)]
        with open(os.path.join(a.out, suite + "_per_problem.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["id"] + [f"{c}_r{ri}_{k}" for ri, c in cols for k in ("correct", "tokens", "seconds", "finish")])
            for i in ids:
                row = [i]
                for key in cols:
                    g = res[key, i]
                    row += list(g) if g else ["", "", "", "error"]
                w.writerow(row)
        show = ids if len(ids) <= 60 else [i for i in ids if len({bool(res[k, i] and res[k, i][0]) for k in cols}) > 1]
        title = "all problems" if len(ids) <= 60 else f"problems where any run differs ({len(show)} of {len(ids)}; full table in {suite}_per_problem.csv)"
        md += [f"### {suite}: {title}", "", "| Problem | " + " | ".join(f"{c} r{ri}" for ri, c in cols) + " |",
               "|---|" + "---|" * len(cols)]
        for i in show:
            cells = []
            for key in cols:
                g = res[key, i]
                cells.append("error" if g is None else f"{'✓' if g[0] else '✗'} {g[1]:,} tok {g[2]:.0f}s" + (" (cut)" if g[3] == "length" else ""))
            md.append(f"| {i} | " + " | ".join(cells) + " |")
        md.append("")
    open(os.path.join(a.out, "report.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")
    sys.stdout.reconfigure(encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
