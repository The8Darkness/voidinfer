"""NoLiMa prompt generation (adobe-research/NoLiMa, pinned commit; HF dataset amodaresi/NoLiMa pinned).

Reproduces evaluation/run_tests.py test construction and NoLiMa_Tester.evaluate() placement exactly
(official BookHaystack class, np.random.seed(test seed), the official 26-point depth grid, character
draw per grid point, DEFAULT task template, vLLM-provider default system prompt "You are a helpful
assistant", max_tokens 192, metric 'contains' on the selected character), with the Qwen3.8 tokenizer
as token counter. Subset (representative): needle_set.json (all 58 tests), haystack(s) --books,
depth grid indices --depth-idx (default 0,5,10,15,20,25 = 0/20/40/60/80/100 %).
Output: nolima/L{len}.jsonl, one item per (book, test, depth).
"""
import argparse, json, os, sys
from copy import copy
import numpy as np
from tokenizers import Tokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
REPO = os.path.join(ROOT, "repos", "NoLiMa")
sys.path.insert(0, REPO)
from data.book_haystack import BookHaystack  # noqa: E402

DEFAULT_TASK_TEMPLATE = "You will answer a question based on the following book snippet:\n\n{haystack}\n\nUse the information provided in the book snippet to answer the question. Your answer should be short and based on either explicitly stated facts or strong, logical inferences.\n\nQuestion: {question}\n\n Return only the final answer with no additional explanation or reasoning."
ap = argparse.ArgumentParser()
ap.add_argument("--lengths", type=int, nargs="+", default=[4000, 16000, 32000])
ap.add_argument("--books", nargs="+", default=["rand_book_1.txt"])
ap.add_argument("--haystack-dir", default=os.path.join(REPO, "data", "haystack", "rand_shuffle"))
ap.add_argument("--needle-set", default=os.path.join(REPO, "data", "needlesets", "needle_set.json"))
ap.add_argument("--depth-idx", type=int, nargs="+", default=[0, 5, 10, 15, 20, 25])
ap.add_argument("--intervals", type=int, default=26)
ap.add_argument("--base-seed", type=int, default=42)
a = ap.parse_args()
tok = Tokenizer.from_file(r"D:\AI\models\Qwen3.8-27B-exl3-6bpw-H6-V6\tokenizer.json")
enc = lambda t: tok.encode(t, add_special_tokens=False).ids
dec = lambda ids: tok.decode(list(ids), skip_special_tokens=False)

needle_set = json.load(open(a.needle_set))
tests = []
for exp in needle_set:  # run_tests.py
    for qtype, question in exp["questions"].items():
        for tid, test in exp["tests"].items():
            needle, q = exp["needle"], copy(question)
            for i, arg in enumerate(test["input_args"]):
                ph = "{" + str(i + 1) + "}"
                if ph in question:
                    q = question.replace(ph, arg)
                if ph in needle:
                    needle = needle.replace(ph, arg)
            tests.append({"name": f"{exp['id']}_{tid}_{qtype}", "reasoning_type": exp["reasoning_type"], "qtype": qtype,
                          "seed": a.base_seed + int(exp["id"][:4]), "needle": needle, "question": q,
                          "chars": exp.get("character_set", ""), "template": exp.get("task_template", DEFAULT_TASK_TEMPLATE)})
books = sorted(os.listdir(a.haystack_dir))  # run_tests.py enumerates os.listdir order (sorted on NTFS)
os.makedirs(os.path.join(ROOT, "nolima"), exist_ok=True)
for L in a.lengths:
    out = open(os.path.join(ROOT, "nolima", f"L{L}.jsonl"), "w", encoding="utf-8")
    n = 0
    for book in a.books:
        hno = books.index(book)
        hay = BookHaystack(os.path.join(a.haystack_dir, book))
        for t in tests:
            np.random.seed(t["seed"] + hno)  # NoLiMa_Tester.evaluate
            for di, d in enumerate(np.linspace(0, 100, a.intervals)):
                char = str(np.random.choice(t["chars"]))
                if di not in a.depth_idx:
                    continue
                needle = t["needle"].replace("{CHAR}", char)
                q = t["question"].replace("{CHAR}", char)
                pl = hay.generate_w_needle_placement(needle=needle, token_count_func=lambda s: len(enc(s)),
                                                     encoding_func=enc, decoding_func=dec, context_length=L, depth=d / 100)
                prompt = t["template"].format(haystack=pl["text"], question=q)
                out.write(json.dumps({"id": f"L{L}_{book[:-4]}_{t['name']}_d{int(round(d))}", "length": L, "book": book,
                                      "test": t["name"], "reasoning_type": t["reasoning_type"], "qtype": t["qtype"],
                                      "depth": float(d), "gold": [char], "metric": "contains",
                                      "messages": [{"role": "system", "content": "You are a helpful assistant"},
                                                   {"role": "user", "content": prompt}], "max_tokens": 192},
                                     ensure_ascii=False) + "\n")
                n += 1
    print(L, n, "items")
