# Builds the markdown tables for docs/ninfer-comparison-2026-10-12.md from results/2026-10-12.
import glob, json, math, os, statistics as st
R = r"D:\AI\engine-compare-20261008\results\2026-10-12"
md = []


def fmt(v, nd=1):
    return "—" if v is None else (f"{v:,.0f}" if nd == 0 else f"{v:,.{nd}f}")


def mean(v):
    v = [x for x in v if x is not None]
    return st.mean(v) if v else None


# ---- single request (C1)
c1 = {}  # (engine, mode, prompt) -> rows
for ln in open(os.path.join(R, "ours", "results.jsonl"), encoding="utf-8-sig"):
    ln = ln.strip()
    if ln.startswith("{"):
        r = json.loads(ln)
        c1.setdefault(("ours", r["mode"], r["prompt"]), []).append(
            {"pt": r["prompt_tokens"], "pf": r["prefill_tps"], "dec": r["decode_tps"], "out": r["out_tokens"],
             "tpr": r.get("tok_per_round"), "ttft": r["prefill_s"]})
for f in glob.glob(os.path.join(R, "ninfer", "*.json")):
    for r in json.load(open(f))["rows"]:
        if r["rep"] == 0:
            continue  # warm request
        mode = "base" if r["spec"] == "none" else "dflash2"
        # NInfer drafts K=7 per round; committed tokens per round = outputs / rounds.
        tpr = r["out_tokens"] / (r["draft_n"] / 7) if mode == "dflash2" and r.get("draft_n") else None
        eng = r["label"] if r["kv"] == "fp8" else f'{r["label"]}.{r["kv"]}'
        c1.setdefault((eng, mode, r["prompt"]), []).append(
            {"pt": r["prompt_tokens"], "pf": r["prefill_tps"], "dec": r["decode_tps"], "out": r["out_tokens"],
             "tpr": tpr, "ttft": r["prefill_ms"] / 1000})


def g(eng, mode, p, f):
    return mean([x[f] for x in c1.get((eng, mode, p), [])])


ENG = [("ours", "VoidInfer (EXL3 6 bpw, L0 OSCAR INT4 KV, exact VeriCache)"), ("nv", "NInfer NVFP4 / FP8 KV"),
       ("gw", "NInfer groupwise / FP8 KV")]
CODE = [("code_4k", "4K"), ("code_16k", "16K"), ("code_32k", "32K"), ("code_64k", "64K"), ("code_128k", "128K"),
        ("code_254k", "254K")]


def table(title, mode, field, nd, prompts=CODE, note=""):
    md.append(f"#### {title}\n")
    if note:
        md.append(note + "\n")
    md.append("| Context (prompt tokens) | " + " | ".join(e[1] for e in ENG) + " |")
    md.append("|---|" + "---:|" * len(ENG))
    for p, lab in prompts:
        pt = next((g(e, mode, p, "pt") for e, _ in ENG if g(e, mode, p, "pt")), None)
        if pt is None:
            continue
        md.append(f"| {lab} ({fmt(pt, 0)}) | " + " | ".join(fmt(g(e, mode, p, field), nd) for e, _ in ENG) + " |")
    md.append("")


table("DFlash2 greedy decode, code-review workload (tok/s, 1,024 outputs)", "dflash2", "dec", 1)
table("DFlash2 committed tokens per verification round", "dflash2", "tpr", 2)
table("Time to first token with the DFlash2 draft resident (s)", "dflash2", "ttft", 2)
table("Prefill throughput with the DFlash2 draft resident (tok/s)", "dflash2", "pf", 0)
table("Base (non-speculative) greedy decode (tok/s, 256 outputs)", "base", "dec", 1,
      [c for c in CODE if c[0] in ("code_4k", "code_32k", "code_128k")])
table("Prefill throughput, target only (tok/s)", "base", "pf", 0,
      [c for c in CODE if c[0] in ("code_4k", "code_32k", "code_128k")])
# 254K: every engine/KV variant that ran
md.append("#### Largest single request: 254K-token prompt (259,483 tokens), DFlash2, 1,024 outputs\n")
md.append("| Engine / KV | Decode (tok/s) | Tokens / round | Prefill (tok/s) | Time to first token (s) |")
md.append("|---|---:|---:|---:|---:|---:|")
LAB = {"ours": "VoidInfer, L0 OSCAR INT4 + exact VeriCache", "nv": "NInfer NVFP4 / FP8 KV",
       "nv.k8v4": "NInfer NVFP4 / k8v4 KV", "nv.nvfp4": "NInfer NVFP4 / NVFP4 KV", "gw": "NInfer groupwise / FP8 KV"}
for e, lab in LAB.items():
    if (e, "dflash2", "code_254k") in c1:
        md.append(f"| {lab} | {fmt(g(e,'dflash2','code_254k','dec'))} | {fmt(g(e,'dflash2','code_254k','tpr'),2)} | "
                  f"{fmt(g(e,'dflash2','code_254k','pf'),0)} | {fmt(g(e,'dflash2','code_254k','ttft'),1)} |")
    else:
        md.append(f"| {lab} | does not start: KV reservation exceeds free VRAM | | | |")
md.append("")
scen = sorted({p for (_, m, p) in c1 if p.startswith("scenario_")})
if scen:
    md.append("#### DFlash2 on NInfer's 12 scenario prompts (~2.1K-token prompts, ≤1,024 outputs, tok/s)\n")
    md.append("| Scenario | " + " | ".join(e[1] for e in ENG) + " |")
    md.append("|---|" + "---:|" * len(ENG))
    logs = {e: [] for e, _ in ENG}
    for p in scen:
        cells = []
        for e, _ in ENG:
            v, o = g(e, "dflash2", p, "dec"), g(e, "dflash2", p, "out")
            if v:
                logs[e].append(math.log(v))
            cells.append(f"{fmt(v)} ({fmt(o, 0)})")
        md.append(f"| {p.replace('scenario_', '')} | " + " | ".join(cells) + " |")
    md.append("| **geometric mean** | " + " | ".join(
        f"**{math.exp(st.mean(logs[e])):.1f}**" if logs[e] else "—" for e, _ in ENG) + " |")
    md.append("\nParenthesized: generated tokens (both engines stop at EOS; greedy text differs between weight formats).\n")

# ---- concurrency
conc = {}
for name in ("ours.jsonl", "ninfer.jsonl", "ninfer.prefix.jsonl"):
    p = os.path.join(R, "conc", name)
    if not os.path.exists(p):
        continue
    for ln in open(p, encoding="utf-8-sig"):
        ln = ln.strip()
        if not ln.startswith("{"):
            continue
        r = json.loads(ln)
        if name.startswith("ours"):
            eng = "ours"
        elif name == "ninfer.jsonl":
            eng = "nv.static"
        else:
            eng = "nv.prefix"
        conc.setdefault((eng, r["prompt"], r["conc"]), []).append(r)


def conc_cell(eng, p, n):
    rs = conc.get((eng, p, n), [])
    v = mean([r["aggregate_tps"] for r in rs])
    per = [x for r in rs for x in (r.get("per_request_tps") or [q["tps"] for q in r.get("per_request", [])])]
    return v, (f"{fmt(v)} [{min(per):.0f}–{max(per):.0f}]" if per else fmt(v))


if conc:
    md.append("#### Concurrent requests: aggregate greedy DFlash2 decode (tok/s, N × 1,024 outputs)\n")
    md.append("Aggregate = all completion tokens / wall time of the batch (prefill included). Mean of 2 runs; "
              "per-request range in brackets. NInfer: NVFP4, FP8 KV, `--kv-capacity` = N × max context (KV reserved per slot), prefix reuse off and on (VoidInfer reuses shared prefixes by default; the "
              "fixtures put the code in a system message all requests share). Where the reservation "
              "exceeds free VRAM the server does not start; a shared `--kv-capacity auto` pool was not measured "
              "(the follow-up run was stopped). Ratio = VoidInfer / faster NInfer.\n")
    md.append("| Prompt | Concurrency | VoidInfer | NInfer NVFP4 / FP8 KV, no prefix reuse | NInfer, prefix reuse | Ratio |")
    md.append("|---|---:|---:|---:|---:|---:|")
    for (p, n) in sorted({(k[1], k[2]) for k in conc}, key=lambda x: (x[0], x[1])):
        ours_v, ours_c = conc_cell("ours", p, n)
        cells, best = [ours_c], None
        for eng in ("nv.static", "nv.prefix"):
            v, c = conc_cell(eng, p, n)
            if v is None and eng == "nv.static" and glob.glob(
                    os.path.join(R, "conc", f"ninfer.{p}.c{n}.rep*.json.server.log")):
                c = "does not start"
            cells.append(c)
            if v and (best is None or v > best):
                best = v
        ratio = f"{ours_v / best:.2f}×" if ours_v and best else "—"
        md.append(f"| {p} | C{n} | " + " | ".join(cells) + f" | {ratio} |")
    md.append("")

# ---- multi-agent long context
ag = {}
for eng in ("ours", "ninfer"):
    p = os.path.join(R, "agents", eng + ".json")
    if os.path.exists(p):
        ag[eng] = json.load(open(p))
if ag:
    md.append("#### Multi-agent long context: 8 agents × 99K–113K-token code prompts (865K tokens), 4 lanes, 2 turns\n")
    md.append("VoidInfer: EXL3 6 bpw, L0 OSCAR INT4 + exact VeriCache, `--max-context 262144`. NInfer: groupwise artifact, FP8 KV, "
              "`--kv-capacity auto` (166,720 tokens on device), `--host-context-mib 48000`, `--max-context 131072` "
              "(the NVFP4 artifact cannot reserve this configuration). 256 output tokens per agent and turn.\n")
    md.append("| Engine | Agents | Turn | Requests OK | Prompt tokens (sum) | Completion tokens | Wall (s) | Peak VRAM (MiB) |")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for eng, d in ag.items():
        for t in d["turns"]:
            md.append(f"| {eng} | {d['agents']} | {t['turn'] + 1} | {t['ok']}/{d['agents']} | {t['prompt_tokens']:,} | "
                      f"{t['completion_tokens']:,} | {t['wall_s']:.0f} | {t['peak_gpu_mib']:,} |")
    md.append("")

# ---- vision
vis = {}
for f in glob.glob(os.path.join(R, "vision", "*.json")):
    for r in json.load(open(f, encoding="utf-8")):
        vis.setdefault((r["label"], r["image"]), []).append(r)
if vis:
    md.append("#### Vision (image + question, greedy, 256 tokens max)\n")
    md.append("| Server | Image | Result | Prompt tokens | Latency, warm (s) | Answer (first 100 chars) |")
    md.append("|---|---|---|---:|---:|---|")
    for (lab, img), rs in sorted(vis.items()):
        r = rs[-1]
        if r["ok"]:
            txt = r["text"].replace("\n", " ").replace("|", "/")[:100]
            md.append(f"| {lab} | {img} | ok | {(r.get('usage') or {}).get('prompt_tokens', '—')} | {r['s']:.2f} | {txt} |")
        else:
            err = (r.get("error") or "").replace("\n", " ").replace("|", "/")[:100]
            md.append(f"| {lab} | {img} | rejected ({r.get('status', '—')}) | — | — | {err} |")
    md.append("")

out = "\n".join(md)
open(os.path.join(R, "tables.md"), "w", encoding="utf-8").write(out)
print(out)
