# Builds the markdown result tables for docs/ninfer-comparison.md from both engines' raw results.
import glob, json, math, os, statistics as st
R = r"D:\AI\engine-compare-20261008\results"
Q = r"D:\AI\engine-compare-20261008\quality"
groups = {}
def add(key, row):
    groups.setdefault(key, []).append(row)
for ln in open(os.path.join(R, "ours", "results.jsonl"), encoding="utf-8-sig"):
    ln = ln.strip()
    if not ln.startswith("{"):
        continue
    r = json.loads(ln)
    l0 = r.get("l0_oscar"); l0 = l0.get("IsPresent") if isinstance(l0, dict) else l0
    add(("ours", "l0" if l0 else "fp16", r["mode"], r["prompt"]),
        {"pt": r["prompt_tokens"], "pf": r["prefill_tps"], "dec": r["decode_tps"], "out": r["out_tokens"],
         "tpr": r.get("tok_per_round")})
for f in glob.glob(os.path.join(R, "ninfer", "*.*.*.json")):
    d = json.load(open(f))
    for r in d["rows"]:
        if r["rep"] == 0:
            continue
        mode = "base" if r["spec"] == "none" else "dflash2"
        tpr = r["out_tokens"] / (r["draft_n"] / 7) if mode == "dflash2" and r.get("draft_n") else None
        add((r["label"] + "." + r["kv"], r["kv"], mode, r["prompt"]),
            {"pt": r["prompt_tokens"], "pf": r["prefill_tps"], "dec": r["decode_tps"], "out": r["out_tokens"], "tpr": tpr})
def get(eng, mode, prompt, field):
    for (e, kv, m, p), rows in groups.items():
        if m == mode and p == prompt and (e == eng or (eng.startswith("ours") and e == "ours" and kv == eng.split(".")[1])):
            vals = [x[field] for x in rows if x[field] is not None]
            return st.mean(vals) if vals else None
    return None
def fmt(v, nd=1):
    return "—" if v is None else (f"{v:,.0f}" if nd == 0 else f"{v:,.{nd}f}")
CODE = [("code_4k", "4K"), ("code_16k", "16K"), ("code_32k", "32K"), ("code_64k", "64K"), ("code_128k", "128K")]
COLS = [("ours.fp16", "VoidInfer FP16 KV"), ("ours.l0", "VoidInfer L0 OSCAR"), ("nv.fp8", "NInfer NVFP4 / FP8 KV"),
        ("gw.fp8", "NInfer groupwise / FP8 KV"), ("nv.bf16", "NInfer NVFP4 / BF16 KV"), ("gw.bf16", "NInfer groupwise / BF16 KV")]
md = []
def table(title, mode, field, nd, note=""):
    md.append(f"#### {title}\n")
    if note: md.append(note + "\n")
    md.append("| Context (prompt tokens) | " + " | ".join(c[1] for c in COLS) + " |")
    md.append("|---|" + "---:|" * len(COLS))
    for p, lab in CODE:
        pt = next((get(c, mode, p, "pt") for c, _ in COLS if get(c, mode, p, "pt")), None)
        md.append(f"| {lab} ({fmt(pt,0)}) | " + " | ".join(fmt(get(c, mode, p, field), nd) for c, _ in COLS) + " |")
    md.append("")
table("Prefill throughput, target only (tok/s)", "base", "pf", 0)
table("Base (non-speculative) greedy decode (tok/s)", "base", "dec", 1)
table("DFlash2 K=7 greedy decode on the code-review workload (tok/s)", "dflash2", "dec", 1)
table("DFlash2 committed tokens per verification round", "dflash2", "tpr", 2)
table("Prefill throughput with the DFlash2 draft resident (tok/s)", "dflash2", "pf", 0,
      "VoidInfer's Engine prefill also fills the draft's context taps/ring; NInfer's prefill with DFlash2 resident.")
# Scenario table
scen = sorted({p for (_, _, m, p) in groups if p.startswith("scenario_")})
md.append("#### DFlash2 on NInfer's 12 scenario prompts (≈2.1K-token prompts, ≤1,024 outputs, tok/s)\n")
md.append("| Scenario | VoidInfer FP16 KV | NInfer NVFP4 / FP8 KV | NInfer groupwise / FP8 KV | NInfer NVFP4 / BF16 KV | NInfer groupwise / BF16 KV |")
md.append("|---|---:|---:|---:|---:|---:|")
sc = ["ours.fp16", "nv.fp8", "gw.fp8", "nv.bf16", "gw.bf16"]
logs = {c: [] for c in sc}
for p in scen:
    vals = [get(c, "dflash2", p, "dec") for c in sc]
    outs = [get(c, "dflash2", p, "out") for c in sc]
    for c, v in zip(sc, vals):
        if v: logs[c].append(math.log(v))
    md.append(f"| {p.replace('scenario_','')} | " + " | ".join(f"{fmt(v)} ({fmt(o,0)})" for v, o in zip(vals, outs)) + " |")
md.append("| **geometric mean** | " + " | ".join(f"**{math.exp(st.mean(logs[c])):.1f}**" for c in sc) + " |")
md.append("\nParenthesized: generated tokens (both engines stop at EOS; greedy text diverges between weight formats).\n")
open(r"D:\AI\engine-compare-20261008\tables.md", "w", encoding="utf-8").write("\n".join(md))
print("\n".join(md))
