"""Builds the long-context multi-key retrieval items: windows of real C++/CUDA source cut by exact
Qwen3.8 token counts, with six "deployment key" comments planted at random lines (four are asked
for, two are distractors). Usage: make_longctx.py REPO_ROOT TOKENIZER_JSON OUT_DIR"""
import glob, json, os, random, sys
from tokenizers import Tokenizer

repo, tok_path, out = sys.argv[1:4]
files = sorted(glob.glob(os.path.join(repo, "src", "**", "*.*"), recursive=True))
files = [f for f in files if f.endswith((".cu", ".cuh", ".h", ".cpp"))]
corpus = "".join(f"// ==== {os.path.relpath(f, repo)} ====\n" + open(f, encoding="utf-8", errors="ignore").read() + "\n"
                 for f in files)
tok = Tokenizer.from_file(tok_path)
enc = tok.encode(corpus)
offsets = enc.offsets
print("corpus tokens", len(offsets))
NAMES = ["aurora", "basalt", "cobalt", "dunmore", "elkhorn", "fennel", "garnet", "halcyon", "ironwood", "juniper",
         "kestrel", "larkspur", "meridian", "nimbus", "obsidian", "pinewood", "quarry", "redwood", "saffron", "tamarack"]
for label, window in (("32k", 30500), ("64k", 62000), ("128k", 126000)):
    rng = random.Random(f"longctx-{label}")
    with open(os.path.join(out, f"longctx_{label}.jsonl"), "w", encoding="utf-8") as f:
        for i in range(20):
            start = rng.randrange(0, len(offsets) - window - 1)
            a, b = offsets[start][0], offsets[start + window][0]
            lines = corpus[a:b].split("\n")
            names = rng.sample(NAMES, 6)
            keys = {n: str(rng.randrange(1000000, 9999999)) for n in names}
            spots = sorted(rng.sample(range(1, len(lines) - 1), 6))
            for n, s in zip(reversed(names), reversed(spots)):
                lines.insert(s, f"// deployment key for service {n}: {keys[n]}")
            asked = names[:4]
            rng.shuffle(asked)
            prompt = ("\n".join(lines) + "\n\nAbove is source code from a C++/CUDA project. Some comments in it state "
                      "deployment keys for named services. Report the deployment keys for the services "
                      + ", ".join(asked[:3]) + " and " + asked[3] + ". Answer with one line per service in the form "
                      "NAME=KEY and nothing else.")
            f.write(json.dumps({"id": f"longctx_{label}-{i}", "prompt": prompt,
                                "gold": {n: keys[n] for n in asked}, "depths": [s / len(lines) for s in spots],
                                "prompt_tokens": len(tok.encode(prompt).ids)}) + "\n")
    print(label, "done")
