"""Runs every analysis from the raw records and writes results/*.json + results/REPORT_TABLES.md.

  make_report.py [--skip-analyze]

Weight panel: reference = FP32 compute with the BF16 weights (refs_fp32); bf16compute transformers
(refs) is reported as a candidate and as a secondary reference. KV long panel: absolute reference
FP32 compute (W <= 32K, refs_long_fp32) or BF16 compute (64K, 128K, refs_long); pairwise within each
engine against its highest-precision KV (NInfer BF16 KV; ours FP16 device KV where it runs (<=16K),
else the exact-history arm). Tasks: score_tasks.py per item file; IFBench from the WSL evaluator.
"""
import argparse, json, os, subprocess, sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
R = os.path.join(ROOT, "results")
PY = [sys.executable, "-I", os.path.join(ROOT, "scripts", "analyze_panel.py")]
ap = argparse.ArgumentParser()
ap.add_argument("--skip-analyze", action="store_true")
a = ap.parse_args()
os.makedirs(R, exist_ok=True)
WINDOWS = [4096, 16384, 32768, 65536, 131072]
OURS = {"fp16": "EXL3 6bpw + FP16 KV", "int4raw": "EXL3 + OSCAR INT4 + hot rows (raw L0)",
        "int2raw": "EXL3 + OSCAR INT2 + hot rows (raw L0)", "int4vc": "EXL3 + INT4 + exact VeriCache (verifier dist.)"}


def run(args, out):
    if a.skip_analyze and os.path.exists(out):
        return json.load(open(out))
    txt = subprocess.run(PY + args + ["--json", out], capture_output=True, text=True, cwd=ROOT)
    open(out[:-5] + ".txt", "w", encoding="utf-8").write(txt.stdout + txt.stderr)
    if txt.returncode:
        print("FAILED", args, txt.stderr[-600:]); return None
    return json.load(open(out))


def exists(d, n=12):
    p = os.path.join(ROOT, d)
    return os.path.isdir(p) and len([f for f in os.listdir(p) if f.endswith(".topk")]) >= n


jobs = {}
jobs["weight_vs_fp32"] = run(["--ref", "ref:refs_fp32", "--cand", "bf16compute=ref:refs", "--cand", "ours=ours:runs/ours_fp16kv",
                              "--cand", "ninfer=ninfer:runs/ninfer_nv_bf16"], os.path.join(R, "weight_vs_fp32.json"))
jobs["weight_vs_bf16"] = run(["--ref", "bf16", "--cand", "ours=ours:runs/ours_fp16kv", "--cand", "ninfer=ninfer:runs/ninfer_nv_bf16"],
                             os.path.join(R, "weight_vs_bf16.json"))
for W in WINDOWS:
    common = ["--panel", f"long/w{W}", "--first", str(W), "--record", f"w{W}.top100", "--csv", f"w{W}.csv"]
    cands = []
    refdir = f"refs_long_fp32/w{W}" if W <= 32768 else f"refs_long/w{W}"
    if W <= 32768 and os.path.isdir(os.path.join(ROOT, f"refs_long/w{W}")):
        cands += ["--cand", f"bf16compute=ref:refs_long/w{W}"]
    for arm in OURS:
        if exists(f"runs/ours_kv_{arm}_w{W}"):
            cands += ["--cand", f"ours_{arm}=ours:runs/ours_kv_{arm}_w{W}"]
    for kv in ("bf16", "fp8"):
        if os.path.exists(os.path.join(ROOT, f"runs/ninfer_nv_{kv}/w{W}.top100")):
            cands += ["--cand", f"ninfer_{kv}=ninfer:runs/ninfer_nv_{kv}"]
    if os.path.isdir(os.path.join(ROOT, refdir)) and len(os.listdir(os.path.join(ROOT, refdir))) >= 24:
        jobs[f"kv_abs_w{W}"] = run(common + ["--ref", f"ref:{refdir}"] + cands, os.path.join(R, f"kv_abs_w{W}.json"))
    if os.path.exists(os.path.join(ROOT, f"runs/ninfer_nv_fp8/w{W}.top100")):
        jobs[f"kv_ninfer_pair_w{W}"] = run(common + ["--ref", "ninfer:runs/ninfer_nv_bf16", "--cand", "ninfer_fp8=ninfer:runs/ninfer_nv_fp8"],
                                           os.path.join(R, f"kv_ninfer_pair_w{W}.json"))
    oref = "fp16" if exists(f"runs/ours_kv_fp16_w{W}") else "int4vc"
    oc = []
    for arm in OURS:
        if arm != oref and exists(f"runs/ours_kv_{arm}_w{W}"):
            oc += ["--cand", f"ours_{arm}=ours:runs/ours_kv_{arm}_w{W}"]
    if exists(f"runs/ours_kv_{oref}_w{W}") and oc:
        jobs[f"kv_ours_pair_w{W}"] = run(common + ["--ref", f"ours:runs/ours_kv_{oref}_w{W}"] + oc, os.path.join(R, f"kv_ours_pair_w{W}.json"))
        jobs[f"kv_ours_pair_w{W}_ref"] = oref

# ---------------- markdown ----------------
L = []
f5 = lambda t: f"{t[0]:.5f} [{t[1]:.5f}, {t[2]:.5f}]"
pc = lambda t: f"{100 * t[0]:.2f}% [{100 * t[1]:.2f}, {100 * t[2]:.2f}]"


def wtable(rep, names, title):
    L.append(f"\n### {title}\n")
    L.append("| slice | tokens | " + " | ".join(f"{n} KL | {n} top-1 | {n} dNLL" for n in names) + " |")
    L.append("|---" * (2 + 3 * len(names)) + "|")
    rows = [("all", rep["all"])] + [(d, v) for d, v in rep["domains"].items()]
    rows += [(f"entropy {k}", v) for k, v in rep.get("entropy", {}).items()]
    if "difficult" in rep:
        rows.append(("difficult (ref top-1 p<0.5)", rep["difficult"]))
    for nm, r in rows:
        cells = []
        for n in names:
            x = r[n]
            cells += [f5(x["kl"]), pc(x["agree"]), f5(x["dnll"])]
        L.append(f"| {nm} | {r[names[0]]['tokens']} | " + " | ".join(cells) + " |")


def dtable(rep, pairs, title):
    L.append(f"\n### {title}\n")
    L.append("| slice | pair | dKL | dTop-1 (pt) | ddNLL | KL ratio | streams worse |")
    L.append("|---|---|---|---|---|---|---|")
    rows = [("all", rep["all"])] + [(d, v) for d, v in rep["domains"].items()]
    if "difficult" in rep:
        rows.append(("difficult", rep["difficult"]))
    for nm, r in rows:
        for y, x in pairs:
            d = r[f"{y}-{x}"]
            ratio = r[y]["kl"][0] / max(r[x]["kl"][0], 1e-12)
            L.append(f"| {nm} | {y} - {x} | {f5(d['kl'])} | {100 * d['agree'][0]:+.2f} [{100 * d['agree'][1]:+.2f}, {100 * d['agree'][2]:+.2f}] |"
                     f" {f5(d['dnll'])} | {ratio:.1f}x | {d['streams_y_worse_kl']} |")


w = jobs["weight_vs_fp32"]
if w:
    L.append("## 1. Weight quantization (FP16/BF16 KV, 70 streams x 1984 scored tokens = 138,880)")
    wtable(w, ["bf16compute", "ours", "ninfer"], "vs FP32-compute reference (BF16 weights)")
    dtable(w, [("ninfer", "ours"), ("ours", "bf16compute"), ("ninfer", "bf16compute")], "paired differences (stream cluster bootstrap 95% CI)")
wb = jobs["weight_vs_bf16"]
if wb:
    wtable(wb, ["ours", "ninfer"], "vs bf16compute reference (transformers)")
L.append("\n## 2. KV precision (12 long streams x 512 scored tokens per window)")
L.append("\n| window | ref | " + " | ".join(["BF16 compute", "ours fp16", "ours int4raw", "ours int2raw", "ours int4vc", "ninfer bf16", "ninfer fp8"]) + " |")
L.append("|---" * 9 + "|")
for W in WINDOWS:
    r = jobs.get(f"kv_abs_w{W}")
    if not r:
        continue
    ref = "FP32" if W <= 32768 else "BF16"
    cells = []
    for n in ["bf16compute", "ours_fp16", "ours_int4raw", "ours_int2raw", "ours_int4vc", "ninfer_bf16", "ninfer_fp8"]:
        x = r["all"].get(n)
        cells.append(f"{x['kl'][0]:.5f} / {100 * x['agree'][0]:.2f}%" if x else "n/a")
    L.append(f"| {W} | {ref} | " + " | ".join(cells) + " |")
L.append("\n(KL vs reference / top-1 agreement; n/a = not run or unsupported: ours FP16 device KV is capped at 32,768 tokens)")
L.append("\n### within-engine pairwise (isolates KV precision; same weights, prompts, settings)\n")
L.append("| window | pair | KL | top-1 agree | dNLL |")
L.append("|---|---|---|---|---|")
for W in WINDOWS:
    r = jobs.get(f"kv_ninfer_pair_w{W}")
    if r:
        x = r["all"]["ninfer_fp8"]
        L.append(f"| {W} | NInfer FP8 KV vs BF16 KV | {f5(x['kl'])} | {pc(x['agree'])} | {f5(x['dnll'])} |")
    r = jobs.get(f"kv_ours_pair_w{W}")
    if r:
        for n, x in r["all"].items():
            if n.startswith("ours_") and "-" not in n and isinstance(x, dict) and "kl" in x:
                L.append(f"| {W} | {n} vs ours {jobs[f'kv_ours_pair_w{W}_ref']} | {f5(x['kl'])} | {pc(x['agree'])} | {f5(x['dnll'])} |")
open(os.path.join(R, "REPORT_TABLES.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
print("\n".join(L))

# ---------------- practical tasks ----------------
import statistics
TASKS = [("ruler_L16384", "task"), ("ruler_L32768", "task"), ("ruler_L65536", "task"), ("ruler_L131072", "task"),
         ("nolima_L4000", "qtype"), ("nolima_L16000", "qtype"), ("nolima_L32000", "qtype")]
T = ["\n## 3. Practical tasks (greedy, thinking off, C1; ours default L0 INT4 + exact VeriCache vs NInfer NVFP4 FP8 KV)\n",
     "| benchmark | slice | n | NInfer | ours | ours - NInfer [95% CI] | McNemar p (ours better/NInfer better) |", "|---|---|---|---|---|---|---|"]
timing = ["\n### runtime (successful requests; per-request wall seconds)\n",
          "| benchmark | engine | requests | median s/request | total s | prompt tok/s (sum prompt / sum s) |", "|---|---|---|---|---|---|"]
for f, by in TASKS:
    out = os.path.join(R, f"task_{f}.json")
    r = subprocess.run([sys.executable, "-I", os.path.join(ROOT, "scripts", "score_tasks.py"), f"items/{f}.jsonl", "runs_tasks/ninfer",
                        "runs_tasks/ours", "--by", by, "--json", out], capture_output=True, text=True, cwd=ROOT)
    if r.returncode:
        print("score failed", f, r.stderr[-300:]); continue
    rep = json.load(open(out))
    for g, row in rep["groups"].items():
        d = row["ours-ninfer"]
        T.append(f"| {f} | {g} | {row['n']} | {100 * row['ninfer']:.2f} | {100 * row['ours']:.2f} | {100 * d['diff']:+.2f} "
                 f"[{100 * d['ci'][0]:+.2f}, {100 * d['ci'][1]:+.2f}] | {d['mcnemar_p']:.3f} ({d['b_better']}/{d['a_better']}) |")
for f in [t[0] for t in TASKS] + ["ifbench"]:
    for e in ("ninfer", "ours"):
        recs = {}
        for l in open(os.path.join(ROOT, "runs_tasks", e, f + ".jsonl"), encoding="utf-8"):
            x = json.loads(l)
            if "error" not in x:
                recs[x["id"]] = x
        secs = [x["seconds"] for x in recs.values()]
        ptok = sum((x.get("usage") or {}).get("prompt_tokens", 0) for x in recs.values())
        timing.append(f"| {f} | {e} | {len(recs)} | {statistics.median(secs):.2f} | {sum(secs):.0f} | {ptok / sum(secs):.0f} |")
ifp = os.path.join(R, "ifbench_pair.json")
if os.path.exists(ifp):
    j = json.load(open(ifp))
    for mode in ("strict", "loose"):
        r = j[mode]
        T.append(f"| IFBench | prompt-level {mode} | {r['n']} | {100 * r['ninfer']:.2f} | {100 * r['ours']:.2f} | {100 * r['diff']:+.2f} "
                 f"[{100 * r['ci'][0]:+.2f}, {100 * r['ci'][1]:+.2f}] | {r['mcnemar_p']:.3f} ({r['ours_better']}/{r['ninfer_better']}) |")
        ic = r["instr_diff_ci"]
        T.append(f"| IFBench | instruction-level {mode} | {r['n']} | {100 * r['instr_ninfer']:.2f} | {100 * r['instr_ours']:.2f} | "
                 f"{100 * ic[0]:+.2f} [{100 * ic[1]:+.2f}, {100 * ic[2]:+.2f}] | - |")
open(os.path.join(R, "REPORT_TABLES.md"), "a", encoding="utf-8").write("\n".join(T + timing) + "\n")
print("\n".join(T + timing))
