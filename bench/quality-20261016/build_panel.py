"""Builds the frozen weight-quantization panel: 70 streams x 2048 tokens (10 per domain, 7 domains).

Each stream is a contiguous source excerpt that starts at a line boundary and is cut so that
decode -> encode round-trips to exactly N tokens with the Qwen3.8 tokenizer, so NInfer (which
tokenizes the text itself) and our engine (which takes the ids) score identical tokens.
Output: panel/{id}.txt, panel/{id}.ids, panel/manifest.json (NInfer corpus format + domain,
source, sha256). Deterministic (seed 20261016).
"""
import glob, hashlib, json, os, random, subprocess
import pyarrow.parquet as pq
from tokenizers import Tokenizer

N = 2048
SEED = 20261016
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
SRC = os.path.join(ROOT, "sources")
OUT = os.path.join(ROOT, "panel")
TOK_PATH = r"D:\AI\models\Qwen3.8-27B-exl3-6bpw-H6-V6\tokenizer.json"
NIN_CORPUS = r"D:\AI\ninfer-5090-windows\eval\corpora\perplexity-1m"
VOID_REPO = r"D:\AI\voidinfer-sol6-high-4000-70-200-20260923"
NIN_REPO = r"D:\AI\ninfer-5090-windows"
EXL3_REPO = r"D:\AI\exllamav3-e1"
tok = Tokenizer.from_file(TOK_PATH)
rng = random.Random(SEED)
os.makedirs(OUT, exist_ok=True)
streams = []


def sha(b):
    return hashlib.sha256(b if isinstance(b, bytes) else b.encode("utf-8")).hexdigest()


def exact(text):
    """Cuts text so that it re-encodes to exactly N tokens; returns (text, ids) or None."""
    ids = tok.encode(text, add_special_tokens=False).ids
    if len(ids) < N + 8:
        return None
    for cut in range(N, N + 64):
        t = tok.decode(ids[:cut], skip_special_tokens=False)
        re = tok.encode(t, add_special_tokens=False).ids
        if len(re) == N:
            return t, re
    return None


def line_start(text, frac_lo, frac_hi):
    """A seeded offset at the start of a line between the two fractions of the text."""
    at = int(len(text) * rng.uniform(frac_lo, frac_hi))
    nl = text.find("\n", at)
    return 0 if nl < 0 else nl + 1


def add(domain, sid, text, source, source_sha):
    got = exact(text)
    assert got, (sid, "too short")
    t, ids = got
    open(os.path.join(OUT, sid + ".txt"), "w", encoding="utf-8", newline="").write(t)
    open(os.path.join(OUT, sid + ".ids"), "w").write(" ".join(map(str, ids)) + "\n")
    streams.append({"id": sid, "domain": domain, "path": sid + ".txt", "source": source,
                    "source_sha256": source_sha, "text_sha256": sha(t), "tokens": len(ids)})
    print(domain, sid, source)


def excerpt(text, lo=0.05, hi=0.6, chars=24000):
    s = line_start(text, lo, hi)
    return text[s:s + chars]


def git_file(repo, rel):
    data = subprocess.run(["git", "-C", repo, "show", f"HEAD:{rel}"], capture_output=True, check=True).stdout
    return data.decode("utf-8")


def git_head(repo):
    return subprocess.run(["git", "-C", repo, "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()


def pick_files(repo, patterns, min_bytes, count, exclude=()):
    files = subprocess.run(["git", "-C", repo, "ls-files", *patterns], capture_output=True, text=True,
                           check=True).stdout.split()
    files = [f for f in files if not any(e in f for e in exclude)]
    files = sorted(f for f in files if os.path.getsize(os.path.join(repo, f)) >= min_bytes)
    return rng.sample(files, count)


# 1. C++/CUDA: 5 VoidInfer sources + 5 NInfer sources (both written in 2026), at HEAD.
vh, nh = git_head(VOID_REPO), git_head(NIN_REPO)
for i, f in enumerate(pick_files(VOID_REPO, ["src/*.cu", "src/*.cpp", "src/*.h", "src/*.cuh"], 30000, 5)):
    t = git_file(VOID_REPO, f)
    add("cpp_cuda", f"cpp-{i:02d}", excerpt(t, 0.05, 0.5), f"voidinfer@{vh}:{f}", sha(t))
for i, f in enumerate(pick_files(NIN_REPO, ["src/*.cu", "src/*.cpp", "src/*.cuh"], 30000, 5)):
    t = git_file(NIN_REPO, f)
    add("cpp_cuda", f"cpp-{i + 5:02d}", excerpt(t, 0.05, 0.5), f"ninfer@{nh}:{f}", sha(t))

# 2. Python: 4 exllamav3 + 3 NInfer tools + 3 VoidInfer bench/test scripts.
eh = git_head(EXL3_REPO)
py = [(EXL3_REPO, eh, f) for f in pick_files(EXL3_REPO, ["*.py"], 20000, 4)]
py += [(NIN_REPO, nh, f) for f in pick_files(NIN_REPO, ["tools/*.py", "bench/*.py", "eval/*.py"], 20000, 3)]
py += [(VOID_REPO, vh, f) for f in pick_files(VOID_REPO, ["*.py"], 14000, 3)]
for i, (repo, h, f) in enumerate(py):
    t = git_file(repo, f)
    add("python", f"py-{i:02d}", excerpt(t, 0.02, 0.4), f"{os.path.basename(repo)}@{h}:{f}", sha(t))

# 3. Math: MATH-500 problem/solution pairs, seeded shuffle, disjoint per stream.
mp = glob.glob(os.path.expanduser(r"~\.cache\huggingface\hub\datasets--HuggingFaceH4--MATH-500\snapshots\*\test.jsonl"))[0]
rows = [json.loads(l) for l in open(mp, encoding="utf-8")]
order = list(range(len(rows)))
rng.shuffle(order)
pos = 0
msha = sha(open(mp, "rb").read())
for i in range(10):
    text, used = "", []
    while len(tok.encode(text, add_special_tokens=False).ids) < N + 64:
        r = rows[order[pos]]; pos += 1; used.append(r["unique_id"])
        text += f"Problem: {r['problem']}\n\nSolution: {r['solution']}\n\n"
    add("math", f"math-{i:02d}", text, f"HuggingFaceH4/MATH-500@{mp.split(os.sep)[-2]}:test.jsonl {used[0]}..+{len(used)}", msha)

# 4. English: wikitext 00-03 and pg19 00-03 (NInfer ppl-1m corpus files), 5 + 5 excerpts.
for i, (name, frac) in enumerate([("wikitext/00", 0.2), ("wikitext/01", 0.5), ("wikitext/02", 0.3), ("wikitext/03", 0.6),
                                  ("wikitext/00", 0.8), ("pg19/00", 0.3), ("pg19/01", 0.5), ("pg19/02", 0.4),
                                  ("pg19/03", 0.6), ("pg19/01", 0.85)]):
    p = os.path.join(NIN_CORPUS, "data", name + ".txt")
    t = open(p, encoding="utf-8").read()
    add("english", f"en-{i:02d}", excerpt(t, frac, frac + 0.05), f"ninfer-ppl-1m-v1:data/{name}.txt", sha(t))

# 5. German: 5 Gutenberg books (body only) + 5 Wikipedia articles.
for i, p in enumerate(sorted(glob.glob(os.path.join(SRC, "de_gutenberg_*.txt")))):
    raw = open(p, encoding="utf-8").read()
    body = raw[raw.find("\n", raw.find("*** START OF")) + 1: raw.find("*** END OF")]
    add("german", f"de-{i:02d}", excerpt(body, 0.1, 0.6), "sources/" + os.path.basename(p), sha(raw))
wiki = sorted(glob.glob(os.path.join(SRC, "de_wiki_*.txt")))
for i, p in enumerate(rng.sample(wiki, 5)):
    t = open(p, encoding="utf-8").read()
    add("german", f"de-{i + 5:02d}", excerpt(t, 0.0, 0.4), "sources/" + os.path.basename(p), sha(t))

# 6. Chinese: zhwiki 00-03, 10 excerpts.
for i, (name, frac) in enumerate([("00", 0.1), ("00", 0.5), ("00", 0.85), ("01", 0.2), ("01", 0.6), ("01", 0.9),
                                  ("02", 0.3), ("02", 0.7), ("03", 0.25), ("03", 0.75)]):
    p = os.path.join(NIN_CORPUS, "data", "zhwiki", name + ".txt")
    t = open(p, encoding="utf-8").read()
    add("chinese", f"zh-{i:02d}", excerpt(t, frac, frac + 0.04, 8000), f"ninfer-ppl-1m-v1:data/zhwiki/{name}.txt", sha(t))

# 7. Reasoning: s1K-1.1 DeepSeek-R1 traces (6 math, 3 science, 1 crossword), question + trace.
sp = glob.glob(os.path.join(SRC, "s1k11_*.parquet"))[0]
tab = pq.read_table(sp).to_pylist()
prov = json.load(open(os.path.join(SRC, "PROVENANCE.json"), encoding="utf-8"))[os.path.basename(sp)]
picked = []
for kind, n in (("math", 6), ("science", 3), ("crossword", 1)):
    cand = [i for i, r in enumerate(tab) if r["cot_type"] == kind and len(r["deepseek_thinking_trajectory"]) > 9000]
    picked += rng.sample(cand, n)
for i, idx in enumerate(picked):
    r = tab[idx]
    text = f"Question: {r['question']}\n\nReasoning:\n{r['deepseek_thinking_trajectory']}"
    add("reasoning", f"rs-{i:02d}", text, f"simplescaling/s1K-1.1@{prov['revision']} row {idx} ({r['cot_type']})", prov["sha256"])

man = {"corpus_id": "voidinfer-wq-panel-v1", "tokens_per_stream": N, "seed": SEED, "tokenizer": TOK_PATH,
       "tokenizer_sha256": sha(open(TOK_PATH, "rb").read()), "streams": streams,
       "modes": {"full": [s["id"] for s in streams], "quick": [s["id"] for s in streams][::10]}}
json.dump(man, open(os.path.join(OUT, "manifest.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print(len(streams), "streams", sum(s["tokens"] for s in streams), "tokens")
