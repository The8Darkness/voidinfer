"""Converts generated RULER sets and the pinned IFBench items into runner item files (items/*.jsonl).
RULER: user message = RULER input (answer prefix kept out, chat model answers freely), max_tokens 256,
metric string_match_all (niah, vt) or string_match_part (qa), as in RULER's eval/synthetic/constants.py.
IFBench: bench3 items (IFBench_test, 300 prompts, pinned), max_tokens 4096 (non-thinking).
"""
import glob, json, os
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
for d in sorted(glob.glob(os.path.join(ROOT, "ruler", "L*"))):
    L = int(os.path.basename(d)[1:])
    out = open(os.path.join(ROOT, "items", f"ruler_L{L}.jsonl"), "w", encoding="utf-8")
    n = 0
    for task in ["niah_multikey_1", "niah_multikey_2", "niah_multikey_3", "vt", "qa_2"]:
        p = os.path.join(d, task, "validation.jsonl")
        for li, l in enumerate(open(p, encoding="utf-8")):
            r = json.loads(l)
            # NOTE: the runs of 2026-10-16 used f"L{L}_{task}_{r['index']}"; RULER's index is not unique
            # (10 of 750 collided) -> line-based ids from now on.
            out.write(json.dumps({"id": f"L{L}_{task}_{li:03d}_{r['index']}", "task": task, "length": L, "gold": r["outputs"],
                                  "metric": "part" if task.startswith("qa") else "all",
                                  "messages": [{"role": "user", "content": r["input"]}], "max_tokens": 256},
                                 ensure_ascii=False) + "\n"); n += 1
    print(L, n)
src = r"D:\AI\bench3-20261014\items\ifbench.jsonl"
with open(os.path.join(ROOT, "items", "ifbench.jsonl"), "w", encoding="utf-8") as f:
    for l in open(src, encoding="utf-8"):
        r = json.loads(l); r["max_tokens"] = 4096; r["length"] = 4096
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
for L in (4000, 16000, 32000):
    os.replace(os.path.join(ROOT, "nolima", f"L{L}.jsonl"), os.path.join(ROOT, "items", f"nolima_L{L}.jsonl")) if os.path.exists(os.path.join(ROOT, "nolima", f"L{L}.jsonl")) else None
print("ok")
