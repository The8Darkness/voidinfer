"""Scores every OUT/<config>/<suite>.jsonl and prints markdown tables: accuracy with a 95% Wilson
interval, truncations, mean output tokens, and paired comparisons against the reference config
(items both engines answered; exact two-sided McNemar test on discordant pairs).

  report.py [--ref ours] [--out report.md]
"""
import argparse, json, math, os, sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from suites import SUITES, hard_detail, ifeval_detail, longctx_detail

OUT = os.environ.get("QB_OUT", r"D:\AI\quality-bench-20261012\runs")


def wilson(k, n):
    if n == 0:
        return (0.0, 0.0)
    z, p = 1.96, k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def mcnemar(b, c):
    b, c = int(b), int(c)
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def load(config, suite):
    path = os.path.join(OUT, config, suite + ".jsonl")
    if not os.path.exists(path):
        return {}
    recs = {}
    for l in open(path, encoding="utf-8"):
        if l.strip():
            r = json.loads(l)
            recs[r["id"]] = r
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="ours")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    configs = sorted(d for d in os.listdir(OUT) if os.path.isdir(os.path.join(OUT, d)))
    configs = [a.ref] + [c for c in configs if c != a.ref] if a.ref in configs else configs
    md = ["| Suite | Config | n | Accuracy | 95% CI | Truncated | Errors | Mean output tokens |",
          "|---|---|---:|---:|---|---:|---:|---:|"]
    paired = ["| Suite | Config vs " + a.ref + " | n paired | " + a.ref + " | config | both right | only " + a.ref +
              " | only config | McNemar p |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    extra = []
    mrcr = ["| Suite | Config vs " + a.ref + " | n | " + a.ref + " mean | config mean | difference [95% CI] |",
            "|---|---|---:|---:|---:|---|"]
    scores = defaultdict(dict)
    for suite, (items_fn, scorer, _, _) in SUITES.items():
        items = {it["id"]: it for it in items_fn()}
        for config in configs:
            recs = load(config, suite)
            if not recs:
                continue
            s, trunc, errs, toks = {}, 0, 0, []
            ife = defaultdict(list)
            keys = []
            for iid, r in recs.items():
                if iid not in items:
                    continue
                if "error" in r:
                    errs += 1
                    s[iid] = 0.0
                    continue
                trunc += r.get("finish_reason") == "length"
                toks.append((r.get("usage") or {}).get("completion_tokens", 0))
                if suite == "ifeval":
                    d = ifeval_detail(items[iid], r["content"])
                    s[iid] = d["strict_prompt"]
                    ife["loose_prompt"].append(d["loose_prompt"])
                    ife["strict_inst"] += d["strict_inst"]
                    ife["loose_inst"] += d["loose_inst"]
                elif suite.startswith(("hard_", "hardx_")):
                    d = hard_detail(items[iid], r["content"])
                    s[iid] = d["all"]
                    keys.append(d["recall"])
                elif suite.startswith("longctx"):
                    d = longctx_detail(items[iid], r["content"])
                    s[iid] = d["all"]
                    keys.append(d["keys"])
                else:
                    s[iid] = scorer(items[iid], r["content"])
            scores[suite][config] = s
            n, k = len(s), sum(s.values())
            lo, hi = wilson(k, n)
            md.append(f"| {suite} | {config} | {n} | {100 * k / n:.1f}% | {100 * lo:.1f}-{100 * hi:.1f} | {trunc} | {errs} | "
                      f"{sum(toks) / max(1, len(toks)):.0f} |")
            if ife:
                extra.append(f"| {suite} | {config} | strict prompt {100 * k / n:.1f}% | loose prompt "
                             f"{100 * sum(ife['loose_prompt']) / n:.1f}% | strict instruction "
                             f"{100 * sum(ife['strict_inst']) / len(ife['strict_inst']):.1f}% | loose instruction "
                             f"{100 * sum(ife['loose_inst']) / len(ife['loose_inst']):.1f}% |")
            if keys:
                extra.append(f"| {suite} | {config} | all correct {100 * k / n:.1f}% | items recalled "
                             f"{100 * sum(keys) / len(keys):.1f}% | | |")
        ref = scores[suite].get(a.ref)
        if not ref:
            continue
        if suite.startswith("mrcr"):
            for config in configs:
                if config == a.ref or config not in scores[suite]:
                    continue
                other = scores[suite][config]
                ids = sorted(set(ref) & set(other))
                d = [ref[i] - other[i] for i in ids]
                m = sum(d) / len(d)
                sd = (sum((x - m) ** 2 for x in d) / max(1, len(d) - 1)) ** 0.5
                h = 1.96 * sd / len(d) ** 0.5
                mrcr.append(f"| {suite} | {config} | {len(ids)} | {sum(ref[i] for i in ids) / len(ids):.3f} | "
                            f"{sum(other[i] for i in ids) / len(ids):.3f} | {m:+.3f} [{m - h:+.3f}, {m + h:+.3f}] |")
            continue
        for config in configs:
            if config == a.ref or config not in scores[suite]:
                continue
            other = scores[suite][config]
            ids = sorted(set(ref) & set(other))
            both = sum(ref[i] and other[i] for i in ids)
            only_ref = sum(ref[i] and not other[i] for i in ids)
            only_other = sum(other[i] and not ref[i] for i in ids)
            paired.append(f"| {suite} | {config} | {len(ids)} | {100 * sum(ref[i] for i in ids) / len(ids):.1f}% | "
                          f"{100 * sum(other[i] for i in ids) / len(ids):.1f}% | {both} | {only_ref} | {only_other} | "
                          f"{mcnemar(only_ref, only_other):.3f} |")
    text = "\n".join(md) + "\n\n" + "\n".join(paired) + "\n\n" + "\n".join(mrcr) + "\n\n" + "\n".join(
        ["| Suite | Config | | | | |", "|---|---|---|---|---|---|"] + extra) + "\n"
    print(text)
    if a.out:
        open(a.out, "w", encoding="utf-8").write(text)


if __name__ == "__main__":
    main()
