"""Builds the long-context KV panel: 12 streams of >= 131,584 tokens, cut per window W into
prefixes of exactly W+512 tokens (round-trip exact), so the 512 targets [W, W+512) are scored with
the full causal context [0, t). Output: long/w{W}/{id}.txt|.ids + manifest.json (NInfer format).
Windows: 4096, 16384, 32768, 65536, 131072. Deterministic; sources pinned in sources/PROVENANCE.json
and git HEADs recorded in long/streams.json.
"""
import glob, hashlib, json, os, subprocess
import pyarrow.parquet as pq
from tokenizers import Tokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
SRC = os.path.join(ROOT, "sources")
OUT = os.path.join(ROOT, "long")
WINDOWS = [4096, 16384, 32768, 65536, 131072]
S = 512
NEED = max(WINDOWS) + S + 64
TOK_PATH = r"D:\AI\models\Qwen3.8-27B-exl3-6bpw-H6-V6\tokenizer.json"
NIN = r"D:\AI\ninfer-5090-windows"
NIN_DATA = NIN + r"\eval\corpora\perplexity-1m\data"
VOID = r"D:\AI\voidinfer-sol6-high-4000-70-200-20260923"
EXL3 = r"D:\AI\exllamav3-e1"
tok = Tokenizer.from_file(TOK_PATH)


def sha(b):
    return hashlib.sha256(b if isinstance(b, bytes) else b.encode("utf-8")).hexdigest()


def head(repo):
    return subprocess.run(["git", "-C", repo, "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()


def repo_concat(repo, patterns):
    files = sorted(subprocess.run(["git", "-C", repo, "ls-files", *patterns], capture_output=True, text=True,
                                  check=True).stdout.split())
    parts = []
    for f in files:
        data = subprocess.run(["git", "-C", repo, "show", f"HEAD:{f}"], capture_output=True, check=True).stdout
        parts.append(f"// ===== {f} =====\n" + data.decode("utf-8", "replace"))
    return "\n".join(parts), f"{os.path.basename(repo)}@{head(repo)}:{' '.join(patterns)} ({len(files)} files, sorted)"


def gutenberg_body(name):
    raw = open(os.path.join(SRC, name), encoding="utf-8").read()
    return raw[raw.find("\n", raw.find("*** START OF")) + 1: raw.find("*** END OF")], f"sources/{name}"


def ninfer_concat(sub):
    ps = sorted(glob.glob(os.path.join(NIN_DATA, sub, "*.txt")))
    return "\n\n".join(open(p, encoding="utf-8").read() for p in ps), f"ninfer-ppl-1m-v1:data/{sub}/00-03 concatenated"


def s1k_concat():
    sp = glob.glob(os.path.join(SRC, "s1k11_*.parquet"))[0]
    rev = json.load(open(os.path.join(SRC, "PROVENANCE.json"), encoding="utf-8"))[os.path.basename(sp)]["revision"]
    used = {240, 985, 813, 641, 696, 951, 590, 932, 317, 755}  # weight-panel rows
    rows = [r for i, r in enumerate(pq.read_table(sp).to_pylist()) if i not in used]
    text = "\n\n".join(f"Question: {r['question']}\n\nReasoning:\n{r['deepseek_thinking_trajectory']}" for r in rows[:400])
    return text, f"simplescaling/s1K-1.1@{rev} rows in file order excluding weight-panel rows, first 400"


SPECS = {
    "en-middlemarch": lambda: gutenberg_body("lc_gutenberg_145.txt"),
    "en-karamazov": lambda: gutenberg_body("lc_gutenberg_28054.txt"),
    "en-wikitext": lambda: ninfer_concat("wikitext"),
    "de-buddenbrooks": lambda: gutenberg_body("de_gutenberg_34811.txt"),
    "de-effibriest": lambda: gutenberg_body("de_gutenberg_5323.txt"),
    "de-grimm": lambda: gutenberg_body("de_gutenberg_2591.txt"),
    "zh-zhwiki": lambda: ninfer_concat("zhwiki"),
    "zh-hongloumeng": lambda: gutenberg_body("lc_gutenberg_24264.txt"),
    "cpp-voidinfer": lambda: repo_concat(VOID, ["src/exl3/*.cu", "src/exl3/*.h", "src/exl3/*.cuh"]),
    "cpp-ninfer": lambda: repo_concat(NIN, ["src/models/qwen3_5/*.cpp", "src/models/qwen3_5/*.h"]),
    "py-exllamav3": lambda: repo_concat(EXL3, ["exllamav3/*.py"]),
    "rs-s1k": s1k_concat,
}
DOMAIN = {"en": "english", "de": "german", "zh": "chinese", "cpp": "cpp_cuda", "py": "python", "rs": "reasoning"}

meta = []
full = {}
for sid, fn in SPECS.items():
    text, source = fn()
    ids = tok.encode(text, add_special_tokens=False).ids
    assert len(ids) >= NEED, (sid, len(ids))
    full[sid] = ids
    meta.append({"id": sid, "domain": DOMAIN[sid.split("-")[0]], "source": source, "source_sha256": sha(text),
                 "source_tokens": len(ids)})
    print(sid, len(ids), source)

for W in WINDOWS:
    d = os.path.join(OUT, f"w{W}")
    os.makedirs(d, exist_ok=True)
    n = W + S
    man = {"corpus_id": f"voidinfer-kv-long-w{W}", "streams": [], "modes": {}}
    for m in meta:
        ids = full[m["id"]]
        for cut in range(n, n + 64):
            t = tok.decode(ids[:cut], skip_special_tokens=False)
            re = tok.encode(t, add_special_tokens=False).ids
            if len(re) == n:
                break
        assert len(re) == n, (m["id"], W, len(re))
        open(os.path.join(d, m["id"] + ".txt"), "w", encoding="utf-8", newline="").write(t)
        open(os.path.join(d, m["id"] + ".ids"), "w").write(" ".join(map(str, re)) + "\n")
        man["streams"].append({"id": m["id"], "domain": m["domain"], "path": m["id"] + ".txt", "text_sha256": sha(t)})
    man["modes"] = {"full": [s["id"] for s in man["streams"]], "quick": [man["streams"][0]["id"]]}
    json.dump(man, open(os.path.join(d, "manifest.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("w", W, "ok")
json.dump({"windows": WINDOWS, "scored_per_stream": S, "tokenizer_sha256": sha(open(TOK_PATH, "rb").read()),
           "streams": meta}, open(os.path.join(OUT, "streams.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
