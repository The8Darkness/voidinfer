"""Paired IFBench comparison from the official per-prompt eval files (results/ifbench/<run>/<run>-eval_results_{loose,strict}.jsonl).
Prompt-level = follow_all_instructions; instruction-level = mean of follow_instruction_list. Paired bootstrap over prompts
(10,000, seed 1) + exact McNemar on prompt-level discordance.   ifbench_pair.py A B [--json out]"""
import json, math, os, sys
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
a, b = sys.argv[1], sys.argv[2]
out = {}
rng = np.random.default_rng(1)
for mode in ("strict", "loose"):
    rec = {}
    for e in (a, b):
        p = os.path.join(ROOT, "results", "ifbench", e, f"{e}-eval_results_{mode}.jsonl")
        rec[e] = {json.loads(l)["prompt"]: json.loads(l) for l in open(p, encoding="utf-8")}
    keys = sorted(set(rec[a]) & set(rec[b]))
    x = np.array([float(rec[a][k]["follow_all_instructions"]) for k in keys])
    y = np.array([float(rec[b][k]["follow_all_instructions"]) for k in keys])
    xi = np.array([np.mean(rec[a][k]["follow_instruction_list"]) for k in keys])
    yi = np.array([np.mean(rec[b][k]["follow_instruction_list"]) for k in keys])
    d = y - x
    idx = rng.integers(0, len(d), size=(10000, len(d)))
    n01, n10 = int(((x == 0) & (y == 1)).sum()), int(((x == 1) & (y == 0)).sum())
    n = n01 + n10
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(min(n01, n10) + 1)) / 2 ** n) if n else 1.0
    di = yi - xi
    r = {"n": len(keys), a: float(x.mean()), b: float(y.mean()), "diff": float(d.mean()),
         "ci": [float(np.percentile(d[idx].mean(1), 2.5)), float(np.percentile(d[idx].mean(1), 97.5))], "mcnemar_p": p,
         f"{b}_better": n01, f"{a}_better": n10, "instr_" + a: float(xi.mean()), "instr_" + b: float(yi.mean()),
         "instr_diff_ci": [float(di.mean()), float(np.percentile(di[idx].mean(1), 2.5)), float(np.percentile(di[idx].mean(1), 97.5))]}
    out[mode] = r
    print(f"IFBench {mode:6s} n={len(keys)} prompt-level {a} {100*x.mean():.2f} {b} {100*y.mean():.2f} | {b}-{a} {100*d.mean():+.2f} "
          f"[{100*r['ci'][0]:+.2f},{100*r['ci'][1]:+.2f}] McNemar p={p:.3f} ({n01}/{n10}) | instruction-level {100*xi.mean():.2f} vs {100*yi.mean():.2f} "
          f"diff {100*di.mean():+.2f} [{100*r['instr_diff_ci'][1]:+.2f},{100*r['instr_diff_ci'][2]:+.2f}]")
if len(sys.argv) > 4 and sys.argv[3] == "--json":
    json.dump(out, open(sys.argv[4], "w"), indent=1)
