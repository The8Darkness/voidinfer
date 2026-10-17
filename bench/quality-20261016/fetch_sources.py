"""Downloads the external corpus sources once into sources/ and records SHA256 + provenance.

German: Project Gutenberg plain-text books + German Wikipedia plaintext extracts (revision ids
recorded). Reasoning: simplescaling/s1K-1.1 (DeepSeek-R1 traces, off-policy for both engines),
pinned to the dataset commit resolved at fetch time. Re-running skips files that exist.
"""
import hashlib, json, os, urllib.parse, urllib.request

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sources")
os.makedirs(ROOT, exist_ok=True)
UA = {"User-Agent": "voidinfer-quality-bench/1.0 (research; contact via repo)"}
prov = {}


def get(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as r:
        return r.read()


def save(name, data, meta):
    p = os.path.join(ROOT, name)
    if not os.path.exists(p):
        open(p, "wb").write(data)
    data = open(p, "rb").read()
    prov[name] = {**meta, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


GUTENBERG = {  # id: title
    22367: "Kafka - Die Verwandlung", 2229: "Goethe - Faust I", 5323: "Fontane - Effi Briest",
    2591: "Grimm - Kinder- und Hausmärchen", 34811: "Mann - Buddenbrooks",
}
for gid, title in GUTENBERG.items():
    url = f"https://www.gutenberg.org/cache/epub/{gid}/pg{gid}.txt"
    name = f"de_gutenberg_{gid}.txt"
    if not os.path.exists(os.path.join(ROOT, name)):
        data = get(url)
    else:
        data = b""
    save(name, data, {"url": url, "title": title, "license": "Project Gutenberg (public domain in the US)"})

DEWIKI = ["Energiewende", "Photosynthese", "Dreißigjähriger Krieg", "Bundesverfassungsgericht", "Quantenmechanik",
          "Industrialisierung in Deutschland", "Rhein"]
for title in DEWIKI:
    name = "de_wiki_" + title.replace(" ", "_") + ".txt"
    q = urllib.parse.urlencode({"action": "query", "format": "json", "prop": "extracts|revisions", "rvprop": "ids",
                                "explaintext": 1, "titles": title, "redirects": 1})
    url = "https://de.wikipedia.org/w/api.php?" + q
    meta = {"url": url, "license": "CC BY-SA 4.0 (Wikipedia)"}
    p = os.path.join(ROOT, name)
    if not os.path.exists(p):
        page = next(iter(json.loads(get(url))["query"]["pages"].values()))
        open(p, "wb").write(page["extract"].encode("utf-8"))
        meta["revid"] = page["revisions"][0]["revid"]
        open(p + ".revid", "w").write(str(meta["revid"]))
    else:
        meta["revid"] = int(open(p + ".revid").read())
    save(name, b"", meta)

# s1K-1.1: resolve the current dataset commit, then fetch the parquet at that commit.
info = json.loads(get("https://huggingface.co/api/datasets/simplescaling/s1K-1.1"))
sha = info["sha"]
files = [s["rfilename"] for s in info["siblings"] if s["rfilename"].endswith(".parquet")]
for f in files:
    url = f"https://huggingface.co/datasets/simplescaling/s1K-1.1/resolve/{sha}/{f}"
    name = "s1k11_" + os.path.basename(f)
    data = b"" if os.path.exists(os.path.join(ROOT, name)) else get(url)
    save(name, data, {"url": url, "dataset": "simplescaling/s1K-1.1", "revision": sha, "license": "MIT (dataset card)"})

json.dump(prov, open(os.path.join(ROOT, "PROVENANCE.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print(json.dumps(prov, indent=1, ensure_ascii=False)[:3000])
