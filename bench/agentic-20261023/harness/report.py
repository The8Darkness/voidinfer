"""Aggregates results/<tag>/<engine>.jsonl into results/summary.json and prints Markdown tables."""
import json, math, os, re, statistics as st, sys

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
CONV = ["c01_package", "c02_no_float", "c03_vxerror_codes", "c04_logging", "c05_stdlib", "c06_type_hints",
        "c07_utc_z", "c08_atomic", "c09_exit3", "c10_spdx", "c11_chf", "c12_acc_ids"]


def load(tag, eng):
    p = os.path.join(ROOT, tag, f"{eng}.jsonl")
    if not os.path.exists(p):
        return []
    rows = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
    meta_p = os.path.join(ROOT, tag, f"{eng}.meta.json")
    meta = json.load(open(meta_p)) if os.path.exists(meta_p) else {}
    for r in rows:
        r["_meta"] = meta
    return rows


def mean_ci(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None, None, 0
    m = sum(xs) / len(xs)
    if len(xs) < 2:
        return m, None, len(xs)
    return m, 1.96 * st.stdev(xs) / math.sqrt(len(xs)), len(xs)


def fmt(m, ci, nd=1):
    if m is None:
        return "-"
    return f"{m:.{nd}f}" + (f" ± {ci:.{nd}f}" if ci is not None else "")


def tokens(rows):
    pt = sum(c.get("prompt_tokens") or 0 for r in rows for c in r.get("calls", []))
    ct = sum(c.get("completion_tokens") or 0 for r in rows for c in r.get("calls", []))
    return pt, ct


def mwu_p(a, b):
    """Two-sided Mann-Whitney U p-value (normal approximation, tie-corrected)."""
    a, b = [x for x in a if x is not None], [x for x in b if x is not None]
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return None
    allv = sorted([(v, 0) for v in a] + [(v, 1) for v in b])
    ranks, i = {}, 0
    rank_list = [0.0] * len(allv)
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and allv[j + 1][0] == allv[i][0]:
            j += 1
        for k in range(i, j + 1):
            rank_list[k] = (i + j) / 2 + 1
        i = j + 1
    r1 = sum(r for r, (v, g) in zip(rank_list, allv) if g == 0)
    u = r1 - n1 * (n1 + 1) / 2
    mu = n1 * n2 / 2
    n = n1 + n2
    ties = {}
    for v, _ in allv:
        ties[v] = ties.get(v, 0) + 1
    tc = sum(t ** 3 - t for t in ties.values())
    sd = math.sqrt(n1 * n2 / 12 * ((n + 1) - tc / (n * (n - 1))))
    if sd == 0:
        return 1.0
    z = (abs(u - mu) - 0.5) / sd
    return min(1.0, math.erfc(max(0.0, z) / math.sqrt(2)))


def lh_table(tag):
    out = {}
    lines = [f"### {tag}", "", "| | ours | NInfer | p (MWU) |", "|---|---|---|---|"]
    data = {e: load(tag, e) for e in ("ours", "nv")}
    for e, rows in data.items():
        if not rows:
            continue
        s = {"runs": len(rows), "window": rows[0]["_meta"].get("window"), "conc": rows[0]["_meta"].get("conc"),
             "errors": sum(1 for r in rows if r.get("status") != "ok")}
        s["tests"] = [r.get("tests_passed", 0) for r in rows]
        s["conv"] = [r.get("conv_score", 0) for r in rows]
        s["compactions"] = [r.get("compactions", 0) for r in rows]
        s["wall_min"] = [r.get("wall_s", 0) / 60 for r in rows]
        s["stages_done"] = [sum(1 for x in r.get("stages", []) if x.get("done")) for r in rows]
        rec_all = [h for r in rows for h in r.get("summary_recall", [])]
        s["recall_all"] = (sum(sum(h.values()) for h in rec_all) / (12 * len(rec_all))) if rec_all else None
        last = [r["summary_recall"][-1] for r in rows if r.get("summary_recall")]
        s["recall_last"] = (sum(sum(h.values()) for h in last) / (12 * len(last))) if last else None
        s["n_summaries"] = len(rec_all)
        s["per_conv"] = {c: sum(1 for r in rows if r.get("conventions", {}).get(c)) for c in CONV}
        s["per_conv_recall"] = {c: (sum(1 for h in rec_all if h.get(c)) / len(rec_all)) if rec_all else None for c in CONV}
        pt, ct = tokens(rows)
        s["prompt_tokens"], s["completion_tokens"] = pt, ct
        s["t_llm_min"] = [r.get("t_llm_s", 0) / 60 for r in rows]
        steps = [c for r in rows for c in r.get("calls", []) if c.get("kind") == "step"]
        s["decode_tok_s"] = (sum(c["completion_tokens"] for c in steps) /
                             max(1e-9, sum(c["wall_s"] - c["ttft_s"] for c in steps))) if steps else None
        s["ttft_mean"] = st.mean(c["ttft_s"] for c in steps) if steps else None
        comp = [c for r in rows for c in r.get("calls", []) if c.get("kind") == "compact"]
        s["compact_call_s"] = st.mean(c["wall_s"] for c in comp) if comp else None
        out[e] = s
    o, n = out.get("ours"), out.get("nv")

    def row(label, key, nd=1, test=True):
        vals = []
        for s in (o, n):
            vals.append(fmt(*mean_ci(s[key])[:2], nd=nd) if s else "-")
        p = mwu_p(o[key], n[key]) if (o and n and test) else None
        lines.append(f"| {label} | {vals[0]} | {vals[1]} | {'' if p is None else f'{p:.3f}'} |")

    def row_s(label, key, f=lambda v: f"{v:.2f}"):
        vals = [(f(s[key]) if s and s.get(key) is not None else "-") for s in (o, n)]
        lines.append(f"| {label} | {vals[0]} | {vals[1]} | |")

    row_s("window x lanes", "window", lambda v: str(v))
    lines[-1] = lines[-1].replace("| |", f"| |") if True else ""
    row_s("runs (errors)", "runs", lambda v: str(v))
    row("hidden tests passed /54", "tests")
    row("conventions kept /12", "conv")
    row("compactions per run", "compactions")
    row("stages completed /12", "stages_done")
    row("wall clock per run (min)", "wall_min")
    row_s("summary recall, all summaries", "recall_all", lambda v: f"{100 * v:.1f}%")
    row_s("summary recall, last summary", "recall_last", lambda v: f"{100 * v:.1f}%")
    row_s("summaries written", "n_summaries", lambda v: str(v))
    row_s("decode tok/s per agent (step calls)", "decode_tok_s", lambda v: f"{v:.0f}")
    row_s("mean time to first token (s)", "ttft_mean", lambda v: f"{v:.1f}")
    row_s("mean compaction call (s)", "compact_call_s", lambda v: f"{v:.1f}")
    row_s("prompt tokens (all runs)", "prompt_tokens", lambda v: f"{v:,}")
    row_s("completion tokens (all runs)", "completion_tokens", lambda v: f"{v:,}")
    lines.append("")
    lines.append("Convention kept (runs) / recalled in summaries:")
    lines.append("")
    lines.append("| convention | ours kept | NInfer kept | ours recall | NInfer recall |")
    lines.append("|---|---|---|---|---|")
    for c in CONV:
        cells = []
        for s in (o, n):
            cells.append(f"{s['per_conv'][c]}/{s['runs']}" if s else "-")
        for s in (o, n):
            v = s["per_conv_recall"][c] if s else None
            cells.append("-" if v is None else f"{100 * v:.0f}%")
        lines.append(f"| {c} | " + " | ".join(cells) + " |")
    return out, "\n".join(lines)


def lh2_table(tag, suite="lh2"):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import importlib
    lh2_spec = importlib.import_module(f"{suite}_spec")
    out = {}
    lines = [f"### {tag}", "", "| | ours | NInfer | p (MWU) |", "|---|---|---|---|"]
    data = {e: load(tag, e) for e in ("ours", "nv")}
    for e, rows in data.items():
        if not rows:
            continue
        s = {"runs": len(rows), "window": rows[0]["_meta"].get("window"),
             "errors": sum(1 for r in rows if r.get("status") != "ok")}
        s["rules"] = [r.get("rules_passed", 0) for r in rows]
        s["tests"] = [r.get("tests_passed", 0) for r in rows]
        s["conv"] = [r.get("conv_score", 0) for r in rows]
        s["total"] = [a + b + c for a, b, c in zip(s["rules"], s["tests"], s["conv"])]
        s["compactions"] = [r.get("compactions", 0) for r in rows]
        s["stages_done"] = [sum(1 for x in r.get("stages", []) if x.get("done")) for r in rows]
        s["wall_min"] = [r.get("wall_s", 0) / 60 for r in rows]
        recs = [x for r in rows for x in r.get("recall2", [])]
        hits = [v for x in recs for v in x["hit"].values()]
        s["recall"] = sum(hits) / len(hits) if hits else None
        s["stale"] = sum(v for x in recs for v in x["stale"].values())
        # recall by age = summaries written since the item's latest value arrived (delivered - item stage)
        by_age = {}
        for x in recs:
            for item, v in x["hit"].items():
                by_age.setdefault(x["delivered"] - lh2_spec.ITEM_STAGE[item], []).append(v)
        s["recall_by_age"] = {k: sum(v) / len(v) for k, v in sorted(by_age.items())}
        # lh7: recall of decisions whose feature was not built yet when the summary was written (memory only)
        use = getattr(lh2_spec, "USE_STAGE", None)
        if use:
            pend = [v for x in recs for item, v in x["hit"].items() if item in use and x["delivered"] < use[item]]
            s["recall_pending"] = sum(pend) / len(pend) if pend else None
            s["recall_pending_n"] = len(pend)
        # recall and stale values by compaction number (1 = first summary of the run)
        by_n = {}
        for r in rows:
            for i, x in enumerate(r.get("recall2", []), start=1):
                d = by_n.setdefault(i, [[], 0])
                d[0] += list(x["hit"].values())
                d[1] += sum(x["stale"].values())
        s["recall_by_n"] = {i: (sum(h) / len(h) if h else None, st_) for i, (h, st_) in sorted(by_n.items())}
        sides = [x for r in rows for x in r.get("side", [])]
        s["side_ok"] = (sum(x["ok"] for x in sides) / len(sides)) if sides else None
        s["per_rule"] = {}
        for r in rows:
            for t in r.get("passed", []):
                if "rules" in t:
                    n = t.split("::")[1]
                    s["per_rule"][n] = s["per_rule"].get(n, 0) + 1
        steps = [c for r in rows for c in r.get("calls", []) if c.get("kind") == "step"]
        s["decode_tok_s"] = (sum(c["completion_tokens"] for c in steps) /
                             max(1e-9, sum(c["wall_s"] - c["ttft_s"] for c in steps))) if steps else None
        comp = [c for r in rows for c in r.get("calls", []) if c.get("kind") == "compact"]
        s["compact_call_s"] = st.mean(c["wall_s"] for c in comp) if comp else None
        out[e] = s
    o, n = out.get("ours"), out.get("nv")

    def row(label, key):
        vals = [fmt(*mean_ci(s[key])[:2]) if s else "-" for s in (o, n)]
        p = mwu_p(o[key], n[key]) if (o and n) else None
        lines.append(f"| {label} | {vals[0]} | {vals[1]} | {'' if p is None else f'{p:.3f}'} |")

    def row_s(label, key, f):
        vals = [(f(s[key]) if s and s.get(key) is not None else "-") for s in (o, n)]
        lines.append(f"| {label} | {vals[0]} | {vals[1]} | |")

    row_s("window", "window", str)
    row_s("runs (errors)", "runs", str)
    H = getattr(lh2_spec, "HIDDEN", suite)
    nrule = len(re.findall(r"^def test_", open(os.path.join(ROOT, "..", "hidden", H, "test_rules.py"),
                                                 encoding="utf-8").read(), re.M))
    row(f"decisions honored /{nrule} (rule tests)", "rules")
    # lh2 reuses the v1 functional suite, whose 32 test functions expand to 54 parametrized tests
    nfun = {"lh2": 54}.get(H) or len(re.findall(r"^def test_", open(os.path.join(ROOT, "..", "hidden", H, "test_functional.py"),
                                                                    encoding="utf-8").read(), re.M))
    row(f"functional tests /{nfun}", "tests")
    row("conventions /12", "conv")
    row(f"total /{nfun + nrule + 12}", "total")
    row("stages completed /12", "stages_done")
    row("compactions per run", "compactions")
    row("wall clock per run (min)", "wall_min")
    row_s("summary recall of current items", "recall", lambda v: f"{100 * v:.1f}%")
    row_s("stale superseded values carried", "stale", str)
    row_s("summary recall of decisions not built yet", "recall_pending", lambda v: f"{100 * v:.1f}%")
    row_s("side-task answers correct", "side_ok", lambda v: f"{100 * v:.1f}%")
    row_s("decode tok/s (step calls)", "decode_tok_s", lambda v: f"{v:.0f}")
    row_s("mean compaction call (s)", "compact_call_s", lambda v: f"{v:.1f}")
    lines += ["", "Recall by summaries since the item arrived (0 = first summary after it):", "",
              "| age | ours | NInfer |", "|---|---|---|"]
    ages = sorted(set((o or {}).get("recall_by_age", {})) | set((n or {}).get("recall_by_age", {})))
    for a in ages:
        cells = [f"{100 * s['recall_by_age'][a]:.0f}%" if s and a in s["recall_by_age"] else "-" for s in (o, n)]
        lines.append(f"| {a} | {cells[0]} | {cells[1]} |")
    lines += ["", "Recall of current items / stale values carried, by compaction number (all runs):", "",
              "| compaction | ours | NInfer |", "|---|---|---|"]
    ns = sorted(set((o or {}).get("recall_by_n", {})) | set((n or {}).get("recall_by_n", {})))
    for i in ns:
        cells = []
        for s_ in (o, n):
            v = s_["recall_by_n"].get(i) if s_ else None
            cells.append("-" if v is None or v[0] is None else f"{100 * v[0]:.0f}% / {v[1]}")
        lines.append(f"| {i} | {cells[0]} | {cells[1]} |")
    lines += ["", "Rule tests passed (runs):", "", "| rule | ours | NInfer |", "|---|---|---|"]
    allrules = re.findall(r"^def (test_d\d\d\w*)", open(os.path.join(ROOT, "..", "hidden", H, "test_rules.py"),
                                                             encoding="utf-8").read(), re.M)
    for nm in allrules:
        lines.append(f"| {nm} | {o['per_rule'].get(nm, 0) if o else '-'}/{o['runs'] if o else '-'} | "
                     f"{n['per_rule'].get(nm, 0) if n else '-'}/{n['runs'] if n else '-'} |")
    return out, "\n".join(lines)


def poly_table(tag, data=None):
    out, lines = {}, [f"### {tag}", "", "| | ours | NInfer |", "|---|---|---|"]
    data = data or {e: load(tag, e) for e in ("ours", "nv")}
    for e, rows in data.items():
        if not rows:
            continue
        s = {"tasks": len(rows), "pass1": sum(1 for r in rows if r.get("pass1")), "pass2": sum(1 for r in rows if r.get("pass2")),
             "errors": sum(1 for r in rows if r.get("status") != "ok"),
             "wall_sum_min": sum(r.get("wall_s", 0) for r in rows) / 60,
             "compactions": sum(r.get("compactions", 0) for r in rows)}
        for lang in ("python", "cpp"):
            lr = [r for r in rows if r.get("lang") == lang]
            s[f"{lang}_pass2"] = f"{sum(1 for r in lr if r.get('pass2'))}/{len(lr)}"
        pt, ct = tokens(rows)
        s["prompt_tokens"], s["completion_tokens"] = pt, ct
        s["by_task"] = {r["id"].split(".", 2)[2]: bool(r.get("pass2")) for r in rows}
        s["solved_wall_min"] = s["wall_sum_min"] / max(1, s["pass2"])
        out[e] = s
    o, n = out.get("ours"), out.get("nv")
    for label, key, f in [("tasks", "tasks", str), ("pass@attempt1", "pass1", str), ("pass@attempt2", "pass2", str),
                          ("python pass@2", "python_pass2", str), ("C++ pass@2", "cpp_pass2", str),
                          ("errors", "errors", str), ("compactions", "compactions", str),
                          ("agent-minutes total (4 agents in parallel)", "wall_sum_min", lambda v: f"{v:.1f}"),
                          ("agent-minutes per solved task", "solved_wall_min", lambda v: f"{v:.2f}"),
                          ("completion tokens", "completion_tokens", lambda v: f"{v:,}")]:
        lines.append(f"| {label} | {f(o[key]) if o else '-'} | {f(n[key]) if n else '-'} |")
    if o and n:
        common = set(o["by_task"]) & set(n["by_task"])
        b = sum(1 for t in common if o["by_task"][t] and not n["by_task"][t])
        c = sum(1 for t in common if n["by_task"][t] and not o["by_task"][t])
        p = None
        if b + c:
            k = min(b, c)
            p = min(1.0, 2 * sum(math.comb(b + c, i) for i in range(k + 1)) / 2 ** (b + c))
        lines.append(f"| discordant (ours only / NInfer only), exact McNemar p | {b} | {c} (p={p if p is None else f'{p:.3f}'}) |")
    return out, "\n".join(lines)


if __name__ == "__main__":
    summary, md = {}, []
    for tag in sorted(os.listdir(ROOT)):
        if tag.startswith("smoke") and "--smoke" not in sys.argv:
            continue
        if not os.path.isdir(os.path.join(ROOT, tag)):
            continue
        if tag.startswith(("lh2", "smoke_lh2")):
            s, t = lh2_table(tag)
        elif re.match(r"(smoke_)?lh[3-9]", tag):
            s, t = lh2_table(tag, re.match(r"(?:smoke_)?(lh[3-9])", tag).group(1))
        elif tag.startswith(("lh", "smoke_lh")):
            s, t = lh_table(tag)
        elif tag.startswith(("poly", "smoke_poly")):
            s, t = poly_table(tag)
        else:
            continue
        summary[tag] = s
        md.append(t)
    # final polyglot: Python from poly_c4, C++ from the fixed-harness rerun
    if os.path.isdir(os.path.join(ROOT, "poly_c4_cpp_v2")):
        comb = {e: [r for r in load("poly_c4", e) if r.get("lang") == "python"] +
                   [r for r in load("poly_c4_cpp_v2", e) if r.get("lang") == "cpp"] for e in ("ours", "nv")}
        s, t = poly_table("poly_c4_final (python from poly_c4, C++ from poly_c4_cpp_v2)", comb)
        summary["poly_c4_final"] = s
        md.append(t)
    json.dump(summary, open(os.path.join(ROOT, "summary.json"), "w"), indent=1, default=str)
    print("\n\n".join(md))
