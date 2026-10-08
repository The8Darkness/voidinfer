# Builds token-exact texts (W+512 tokens) from NInfer's ppl-1m corpus for paired NLL.
import json, os, sys
from tokenizers import Tokenizer
ROOT = r"D:\AI\ninfer-5090-windows\eval\corpora\perplexity-1m"
OUT = os.path.dirname(os.path.abspath(__file__))
tok = Tokenizer.from_file(r"D:\AI\models\Qwen3.8-27B-exl3-6bpw-H6-V6\tokenizer.json")
K = 512
streams = [s for s in json.load(open(os.path.join(ROOT, "manifest.json"), encoding="utf-8"))["streams"]]
pick = {}
for s in streams:
    pick.setdefault(s["domain"], [])
    if len(pick[s["domain"]]) < 2: pick[s["domain"]].append(s)
for W in (31744,):
    d = os.path.join(OUT, f"w{W}"); os.makedirs(d, exist_ok=True)
    man = {"corpus_id": f"compare-w{W}", "streams": []}
    for dom, ss in pick.items():
        for s in ss:
            text = open(os.path.join(ROOT, s["path"]), encoding="utf-8").read()
            ids = tok.encode(text, add_special_tokens=False).ids
            n = W + K
            assert len(ids) >= n, (s["id"], len(ids))
            # cut until decode->encode round trip is exact
            for cut in range(n, n + 64):
                t = tok.decode(ids[:cut], skip_special_tokens=False)
                re = tok.encode(t, add_special_tokens=False).ids
                if len(re) == n: break
            assert len(re) == n, (s["id"], len(re))
            open(os.path.join(d, s["id"] + ".txt"), "w", encoding="utf-8", newline="").write(t)
            open(os.path.join(d, s["id"] + ".ids"), "w").write(" ".join(map(str, re)) + "\n")
            man["streams"].append({"id": s["id"], "domain": dom, "path": s["id"] + ".txt"})
            print(W, s["id"], dom, len(ids), "->", len(re))
    json.dump(man, open(os.path.join(d, "manifest.json"), "w"), indent=1)
