"""Scores RULER / NoLiMa runs and compares engines item-paired.

  score_tasks.py ITEMS.jsonl RUN_DIR_A RUN_DIR_B [...] [--by task|depth|qtype] [--json out]

RULER (RULER eval/synthetic/constants.py): 'all' = mean over references of (ref.lower() in
pred.lower()); 'part' = max over references. NoLiMa ('contains', official): any gold in response
(case-sensitive). Paired difference B-A over items present in both runs: bootstrap 95% CI (10,000
item resamples, seed 1) and an exact two-sided sign test on the discordant items (McNemar).
"""
import argparse, json, math, os
from collections import defaultdict
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
ap = argparse.ArgumentParser()
ap.add_argument("items")
ap.add_argument("runs", nargs="+")
ap.add_argument("--by", default="")
ap.add_argument("--json")
a = ap.parse_args()
P = lambda p: p if os.path.isabs(p) else os.path.join(ROOT, p)
items = {json.loads(l)["id"]: json.loads(l) for l in open(P(a.items), encoding="utf-8")}
name = os.path.splitext(os.path.basename(a.items))[0]


def score(it, pred):
    pred = pred or ""
    if it["metric"] == "contains":
        return float(any(g in pred for g in it["gold"]))
    hits = [float(g.lower() in pred.lower()) for g in it["gold"]]
    return sum(hits) / len(hits) if it["metric"] == "all" else max(hits)


runs = {}
for r in a.runs:
    d = {}
    p = os.path.join(P(r), name + ".jsonl")
    for l in open(p, encoding="utf-8"):
        x = json.loads(l)
        if "error" not in x and x["id"] in items:
            d[x["id"]] = (score(items[x["id"]], x["content"]), x)
    runs[os.path.basename(P(r).rstrip("/\\"))] = d


def mcnemar(a_, b_):
    n01 = sum(1 for x, y in zip(a_, b_) if x < y); n10 = sum(1 for x, y in zip(a_, b_) if x > y)
    n = n01 + n10
    if n == 0:
        return 1.0, n01, n10
    k = min(n01, n10)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n * 2
    return min(1.0, p), n01, n10


rng = np.random.default_rng(1)
report = {"items": name, "groups": {}}
groups = defaultdict(list)
common = [i for i in items if all(i in d for d in runs.values())]
for i in common:
    groups["ALL"].append(i)
    if a.by:
        groups[f"{a.by}={items[i].get(a.by)}"].append(i)
names = list(runs)
for g, ids in groups.items():
    row = {"n": len(ids)}
    line = f"{name:16s} {g:28s} n={len(ids):4d}"
    for nm in names:
        s = np.array([runs[nm][i][0] for i in ids])
        row[nm] = float(s.mean())
        line += f"  {nm} {100 * s.mean():6.2f}"
    for nm in names[1:]:
        x = np.array([runs[names[0]][i][0] for i in ids]); y = np.array([runs[nm][i][0] for i in ids])
        d = y - x
        bs = d[rng.integers(0, len(d), size=(10000, len(d)))].mean(1)
        p, n01, n10 = mcnemar(x, y)
        row[f"{nm}-{names[0]}"] = {"diff": float(d.mean()), "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                                   "mcnemar_p": p, "b_better": n01, "a_better": n10}
        line += f"  | {nm}-{names[0]} {100 * d.mean():+6.2f} [{100 * np.percentile(bs, 2.5):+.2f},{100 * np.percentile(bs, 97.5):+.2f}] p={p:.3f} ({n01}/{n10})"
    report["groups"][g] = row
    print(line)
for nm in names:  # throughput of the run
    sp = os.path.join(P([r for r in a.runs if os.path.basename(P(r).rstrip('/\\')) == nm][0]), name + ".run.json")
    if os.path.exists(sp):
        s = json.load(open(sp))
        print(f"  {nm}: wall {s['wall_s']:.0f}s prompt {s['prompt']} completion {s['completion']} errors {s['errors']} conc {s['conc']}")
        report.setdefault("runtime", {})[nm] = {k: s[k] for k in ("wall_s", "prompt", "completion", "errors", "conc")}
if a.json:
    json.dump(report, open(a.json, "w"), indent=1)
