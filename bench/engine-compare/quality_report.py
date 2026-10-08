# Paired quality: identical target tokens [W, W+512) with identical history [512, W) for every engine.
import csv, json, os, sys, pathlib, statistics as st
sys.path.insert(0, r"D:\AI\ninfer-5090-windows\tools\release")
from per_domain_kl import read_record, token_digest

Q = r"D:\AI\engine-compare-20261008\quality"
K = 512
WS = (4096, 16384, 32768)
NINFER = [("gw", "bf16"), ("gw", "fp8"), ("nv", "bf16"), ("nv", "fp8")]
ENGINES = ["ours_fp16", "ours_l0"] + [f"{n}.{k}" for n, k in NINFER]
LABEL = {"ours_fp16": "ours FP16-KV", "ours_l0": "ours L0-OSCAR", "gw.bf16": "NInfer gw/bf16",
         "gw.fp8": "NInfer gw/fp8", "nv.bf16": "NInfer nvfp4/bf16", "nv.fp8": "NInfer nvfp4/fp8"}
out = []
for W in WS:
    d = os.path.join(Q, f"w{W}")
    man = json.load(open(os.path.join(d, "manifest.json")))
    top1 = {}
    for n, k in NINFER:
        p = os.path.join(d, f"ninfer.{n}.{k}.top1")
        if os.path.exists(p):
            top1[f"{n}.{k}"] = {s.id: s for s in read_record(pathlib.Path(p)).streams}
    lps = {}
    for n, k in NINFER:
        p = os.path.join(d, f"ninfer.{n}.{k}.csv")
        if os.path.exists(p):
            m = {}
            for r in csv.DictReader(open(p)):
                m[(r["stream"], int(r["token_index"]))] = float(r["logprob"])
            lps[f"{n}.{k}"] = m
    for s in man["streams"]:
        sid, dom = s["id"], s["domain"]
        ids = list(map(int, open(os.path.join(d, sid + ".ids")).read().split()))
        targets = ids[W:W + K]
        rec = {"W": W, "stream": sid, "domain": dom, "targets": targets}
        for eng, suffix in (("ours_fp16", ".ours.tok"), ("ours_l0", ".ours_l0.tok")):
            p = os.path.join(d, sid + suffix)
            if os.path.exists(p):
                o = [ln.split() for ln in open(p)]
                assert len(o) == K and all(int(x[2]) == t for x, t in zip(o, targets)), (W, sid, eng)
                rec[eng + "_nll"] = [float(x[0]) for x in o]
                rec[eng + "_top1"] = [int(x[1]) for x in o]
        for e, m in lps.items():
            rec[e + "_nll"] = [-m[(sid, W + i)] for i in range(K)]
        for e, streams in top1.items():
            stream = streams[sid]
            # NInfer's own tokenization of the text must equal the ids our engine scored.
            assert stream.token_digest == token_digest(ids), (W, sid, e, "token mismatch")
            rec[e + "_top1"] = [stream.indices[W + i - 1][0] for i in range(K)]
        out.append(rec)

def cell(recs, e):
    if not recs or not all(e + "_nll" in r for r in recs):
        return None
    nll = st.mean(x for r in recs for x in r[e + "_nll"])
    acc = st.mean(1.0 if a == t else 0.0 for r in recs for a, t in zip(r[e + "_top1"], r["targets"]))
    return nll, acc

rows = []
doms = sorted({r["domain"] for r in out})
for W in WS:
    for dom in doms + ["ALL"]:
        recs = [r for r in out if r["W"] == W and (dom == "ALL" or r["domain"] == dom)]
        rows.append((f"{W//1024}K {dom}", {e: cell(recs, e) for e in ENGINES}))
print("mean NLL (nats/token) | top-1 accuracy vs real next token; 2 streams x 512 targets per domain/depth")
print(f"{'slice':26}" + "".join(f"{LABEL[e]:>20}" for e in ENGINES))
for name, cells in rows:
    print(f"{name:26}" + "".join(f"{('%.4f | %4.1f%%' % (c[0], c[1]*100)) if c else '-':>20}" for c in cells.values()))

print("\npaired: mean(NLL_x - NLL_ref) per token [95% CI] and top-1 agreement with ref")
summary = {"table": [{"slice": n, **{e: c for e, c in cells.items()}} for n, cells in rows], "paired": []}
for ref in ("ours_fp16", "ours_l0"):
    for e in ENGINES:
        if e == ref:
            continue
        recs = [r for r in out if ref + "_nll" in r and e + "_nll" in r]
        if not recs:
            continue
        diffs = [a - b for r in recs for a, b in zip(r[e + "_nll"], r[ref + "_nll"])]
        m = st.mean(diffs); se = st.stdev(diffs) / len(diffs) ** 0.5
        agree = st.mean(1.0 if a == b else 0.0 for r in recs for a, b in zip(r[e + "_top1"], r[ref + "_top1"]))
        depths = sorted({r["W"] for r in recs})
        by = "  ".join(f"{W//1024}K {st.mean(a - b for r in recs if r['W'] == W for a, b in zip(r[e + '_nll'], r[ref + '_nll'])):+.4f}" for W in depths)
        print(f"  {LABEL[e]:>18} vs {LABEL[ref]:<14} {m:+.4f} [{m-1.96*se:+.4f},{m+1.96*se:+.4f}] agree {agree*100:5.1f}%  | {by}")
        summary["paired"].append({"x": e, "ref": ref, "mean": m, "ci95": 1.96 * se, "agree": agree, "tokens": len(diffs)})
json.dump(summary, open(os.path.join(Q, "summary.json"), "w"), indent=1)
