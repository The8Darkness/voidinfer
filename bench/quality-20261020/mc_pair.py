"""Paired comparison of two engines on a letter-only multiple-choice suite.

Usage: python mc_pair.py SUITE A.jsonl B.jsonl  (exact two-sided McNemar on one-engine-correct questions)
"""
import json, os, sys, math, collections
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "engine-compare", "tasks"))
from suites import SUITES
suite = sys.argv.pop(1)
gen, score = SUITES[suite][0], SUITES[suite][1]
items = {it["id"]: it for it in gen()}
def load(p):
    return {r["id"]: (score(items[r["id"]], r.get("content", "")) if "error" not in r else None, r.get("content", "").strip())
            for r in (json.loads(l) for l in open(p, encoding="utf-8") if l.strip())}
def mcnemar(x, y):
    n = x + y; k = min(x, y)
    return min(1.0, 2 * sum(math.comb(n, j) for j in range(k + 1)) / 2 ** n) if n else 1.0
a, b = load(sys.argv[1]), load(sys.argv[2])
for name, d in (("A", a), ("B", b)):
    errs = sum(1 for v in d.values() if v[0] is None)
    print(f"{name}: {sum(v[0] or 0 for v in d.values()):.0f}/{len(items)} = {100*sum(v[0] or 0 for v in d.values())/len(items):.2f}%  records={len(d)} errors={errs}")
ids = [i for i in items if a.get(i, (None,))[0] is not None and b.get(i, (None,))[0] is not None]
aw = sum(1 for i in ids if a[i][0] and not b[i][0]); bw = sum(1 for i in ids if b[i][0] and not a[i][0])
same = sum(1 for i in ids if a[i][1] == b[i][1])
print(f"paired n={len(ids)}  identical answers {same} ({100*same/len(ids):.1f}%)")
print(f"discordant {aw+bw}: A-only {aw}, B-only {bw}, diff {100*(aw-bw)/len(ids):+.2f} pts, exact McNemar p={mcnemar(aw, bw):.2g}")
for key in [k for k in ("category", "difficulty", "is_calculation") if k in next(iter(items.values()))]:
    g = collections.defaultdict(lambda: [0, 0, 0, 0])
    for i in ids:
        s = g[items[i][key]]; s[0] += int(a[i][0] and not b[i][0]); s[1] += int(b[i][0] and not a[i][0]); s[2] += 1; s[3] += a[i][0]
    print(f"by {key}:")
    for c, (x, y, n, ac) in sorted(g.items(), key=lambda t: -(t[1][0] - t[1][1])):
        print(f"  {str(c):20s} n={n:5d} A {100*ac/n:5.1f}%  A-only {x:4d} B-only {y:4d} net {x-y:+4d} ({100*(x-y)/n:+.2f} pts) p={mcnemar(x, y):.2g}")
