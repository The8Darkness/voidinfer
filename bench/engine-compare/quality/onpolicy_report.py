"""Summary of onpolicy_bf16.py records per engine (and suite).

agree: answer tokens equal to the BF16 argmax; agree (clear): the same over tokens whose BF16
top-1/top-2 gap exceeds 0.5 nats (near-ties excluded); regret: mean BF16 max_logp - logp(token);
off>1: tokens per 1000 with regret above 1 nat (choices BF16 clearly rejects); nll: mean BF16
NLL of the answer. Confidence intervals by bootstrap over items (paired across engines).
Usage: python onpolicy_report.py RECORDS.json [--by-suite]
"""
import json, random, sys
from collections import defaultdict

recs = json.load(open(sys.argv[1]))
by_suite = '--by-suite' in sys.argv


def stats(rs):
    n = agree = clear = clear_ok = off = 0
    regret = nll = 0.0
    for r in rs:
        for c, b, s, ok in zip(r['chosen'], r['best'], r['second'], r['argmax_ok']):
            n += 1; agree += ok; regret += b - c; nll -= c; off += (b - c) > 1.0
            if b - s > 0.5:
                clear += 1; clear_ok += ok
    return {'tokens': n, 'agree': agree / n, 'agree_clear': clear_ok / max(clear, 1),
            'regret': regret / n, 'off1_per_k': 1000 * off / n, 'nll': nll / n}


def boot(groups, key, configs, ref='ours', reps=1000):
    """Paired bootstrap over items of (config metric - ref metric)."""
    items = sorted(groups)
    out = {}
    rng = random.Random(0)
    for c in configs:
        if c == ref:
            continue
        diffs = []
        for _ in range(reps):
            pick = [items[rng.randrange(len(items))] for _ in items]
            a = stats([groups[i][c] for i in pick if c in groups[i]])[key]
            b = stats([groups[i][ref] for i in pick if ref in groups[i]])[key]
            diffs.append(a - b)
        diffs.sort()
        out[c] = (diffs[int(0.025 * reps)], diffs[int(0.975 * reps)])
    return out


configs = sorted({r['config'] for r in recs}, key=lambda c: (c != 'ours', c))
scopes = sorted({r['suite'] for r in recs}) if by_suite else []
for scope in [None] + scopes:
    rs = [r for r in recs if scope is None or r['suite'] == scope]
    groups = defaultdict(dict)
    for r in rs:
        groups[(r['suite'], r['id'])][r['config']] = r
    print(f"\n== {scope or 'all suites'} ({len(groups)} items)")
    print(f"{'engine':10s} {'tokens':>8s} {'agree':>7s} {'clear':>7s} {'regret':>8s} {'off>1/k':>8s} {'nll':>7s}")
    for c in configs:
        s = stats([g[c] for g in groups.values() if c in g])
        print(f"{c:10s} {s['tokens']:8d} {100*s['agree']:6.2f}% {100*s['agree_clear']:6.2f}% "
              f"{s['regret']:8.4f} {s['off1_per_k']:8.2f} {s['nll']:7.4f}")
    if scope is None and 'ours' in configs:
        for key in ('agree', 'regret', 'off1_per_k'):
            ci = boot(groups, key, configs)
            print(f"  paired 95% CI ({key}, engine - ours): " +
                  ", ".join(f"{c} [{lo:+.4f}, {hi:+.4f}]" for c, (lo, hi) in ci.items()))
