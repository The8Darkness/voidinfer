"""Harder long-context items that depend on exact attention over the whole history.

  mk   dense multi-key retrieval: 200 near-identical service keys (atlas-17 vs atlas-71) planted as
       comments in real C++/CUDA source; ask for six of them.
  vt   variable tracking (RULER-style): a chain of five assignments to a value plus four distractor
       chains, spread through the source; ask for every variable that carries the value.
  mt   multi-turn agent session: user turns paste source files and update six project settings
       several times (with near-name decoys); ask for the final value of four settings.

Lengths are exact Qwen3.8 token counts of the content (32K, 128K, 200K).
Usage: make_longctx_hard.py REPO_ROOT TOKENIZER_JSON OUT_DIR"""
import glob, json, os, random, string, sys
from tokenizers import Tokenizer

repo, tok_path, out = sys.argv[1:4]
files = sorted(f for f in glob.glob(os.path.join(repo, "src", "**", "*.*"), recursive=True)
               if f.endswith((".cu", ".cuh", ".h", ".cpp")))
corpus = "".join(f"// ==== {os.path.relpath(f, repo)} ====\n" + open(f, encoding="utf-8", errors="ignore").read() + "\n"
                 for f in files)
tok = Tokenizer.from_file(tok_path)
offsets = tok.encode(corpus).offsets
ntok = lambda s: len(tok.encode(s).ids)
WORDS = ["atlas", "beacon", "cinder", "dynamo", "ember", "falcon", "glacier", "harbor", "indigo", "jasper",
         "keystone", "lantern", "monarch", "nebula", "onyx", "prairie", "quartz", "raven", "summit", "tundra"]


def window(rng, tokens):
    start = rng.randrange(0, len(offsets) - tokens - 1)
    return corpus[offsets[start][0]:offsets[start + tokens][0]]


def plant(rng, text, inserts):
    lines = text.split("\n")
    spots = sorted(rng.sample(range(1, len(lines) - 1), len(inserts)))
    for s, line in zip(reversed(spots), reversed(inserts)):
        lines.insert(s, line)
    return "\n".join(lines)


def mk_item(rng, tokens, i, label):
    names = rng.sample([f"{w}-{n}" for w in WORDS for n in range(10, 100)], 200)
    keys = {n: str(rng.randrange(1000000, 9999999)) for n in names}
    order = names[:]
    rng.shuffle(order)
    body = plant(rng, window(rng, tokens - 4000), [f"// deployment key for service {n}: {keys[n]}" for n in order])
    asked = rng.sample(names, 6)
    prompt = (body + "\n\nAbove is source code from a C++/CUDA project. Comments in it state deployment keys for "
              "200 services with similar names. Report the deployment keys for exactly these services: "
              + ", ".join(asked) + ". Answer with one line per service in the form NAME=KEY and nothing else.")
    return {"id": f"hard_mk_{label}-{i}", "prompt": prompt, "gold": {n: keys[n] for n in asked}}


def vt_item(rng, tokens, i, label):
    letters = string.ascii_uppercase
    var = lambda: "".join(rng.choice(letters) for _ in range(5))
    values = rng.sample(range(10000, 99999), 5)
    chains = [[var() for _ in range(5)] for _ in values]
    # Interleave the five chains randomly while keeping each chain in assignment order.
    flat = [[f"// VAR {names[0]} = {v}"] + [f"// VAR {names[k]} = VAR {names[k - 1]}" for k in range(1, 5)]
            for names, v in zip(chains, values)]
    merged = []
    pos = [0] * 5
    while any(p < 5 for p in pos):
        c = rng.choice([k for k in range(5) if pos[k] < 5])
        merged.append(flat[c][pos[c]])
        pos[c] += 1
    body = plant(rng, window(rng, tokens - 600), merged)
    prompt = (body + f"\n\nAbove is source code from a C++/CUDA project. Comments of the form \"VAR X = ...\" assign "
              f"values to variables, either a number or another variable. Find every variable that is assigned the "
              f"value {values[0]}, directly or through other variables. Answer with the variable names separated by "
              f"spaces and nothing else.")
    return {"id": f"hard_vt_{label}-{i}", "prompt": prompt, "gold_set": chains[0]}


def mt_item(rng, tokens, i, label):
    settings = rng.sample(["retry budget", "cache ceiling", "batch window", "flush interval", "shard count",
                           "timeout floor", "queue depth", "replica quota", "page budget", "lease period"], 6)
    decoys = {s: s.replace(" ", " backup ") for s in settings}
    turns, value = 40, {}
    per_turn = (tokens - 3000) // turns
    messages = []
    for t in range(turns):
        notes = []
        for s in rng.sample(settings, 2):
            value[s] = rng.randrange(100, 9999)
            notes.append(f"Set the {s} to {value[s]}.")
        d = rng.choice(settings)
        notes.append(f"Set the {decoys[d]} to {rng.randrange(100, 9999)}.")
        rng.shuffle(notes)
        code = window(rng, per_turn)
        messages.append({"role": "user", "content": f"Here is the next file for the review.\n```cpp\n{code}\n```\n"
                         + " ".join(notes)})
        messages.append({"role": "assistant", "content": "Noted. I reviewed the file and recorded the settings."})
    asked = rng.sample(settings, 4)
    messages.append({"role": "user", "content": "What are the current values of the " + ", ".join(asked[:3]) + " and "
                     + asked[3] + " after all the changes in this conversation? Answer with one line per setting in "
                     "the form NAME=VALUE and nothing else."})
    return {"id": f"hard_mt_{label}-{i}", "messages": messages, "gold": {s: str(value[s]) for s in asked}}


if len(sys.argv) > 4 and sys.argv[4] == "mtx":
    # Larger multi-turn set: 24 items at 64K/128K/200K, independent seeds.
    for label, tokens in (("64k", 63000), ("128k", 126000), ("200k", 200000)):
        rng = random.Random(f"hardx-mt-{label}")
        with open(os.path.join(out, f"hardx_mt_{label}.jsonl"), "w", encoding="utf-8") as f:
            for i in range(24):
                item = mt_item(rng, tokens, i, label)
                item["id"] = f"hardx_mt_{label}-{i}"
                item["content_tokens"] = ntok("".join(m["content"] for m in item["messages"]))
                f.write(json.dumps(item) + "\n")
        print("mtx", label, "done", item["content_tokens"])
    sys.exit(0)

for label, tokens in (("32k", 31500), ("128k", 126000), ("200k", 200000)):
    for task, fn in (("mk", mk_item), ("vt", vt_item), ("mt", mt_item)):
        rng = random.Random(f"hard-{task}-{label}")
        with open(os.path.join(out, f"hard_{task}_{label}.jsonl"), "w", encoding="utf-8") as f:
            for i in range(12):
                item = fn(rng, tokens, i, label)
                text = item.get("prompt") or "".join(m["content"] for m in item["messages"])
                item["content_tokens"] = ntok(text)
                f.write(json.dumps(item) + "\n")
        print(task, label, "done", item["content_tokens"])
