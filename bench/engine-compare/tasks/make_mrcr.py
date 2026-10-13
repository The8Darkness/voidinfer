"""Samples OpenAI MRCR (2- and 4-needle) items into the bench format, binned by Qwen3.8 token count of
the conversation (content tokens + 6 per message for the chat template):
(32K, 64K], (64K, 128K], (128K, 196K]; 20 per needle count and bin, seed 0.
Usage: make_mrcr.py MRCR_DIR TOKENIZER_JSON OUT_DIR"""
import glob, json, os, random, sys
import pandas as pd
from tokenizers import Tokenizer

src, tok_path, out = sys.argv[1:4]
tok = Tokenizer.from_file(tok_path)
BINS = (("64k", 32768, 65536), ("128k", 65536, 131072), ("196k", 131072, 196608))
for needles in (2, 4):
    df = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(os.path.join(src, f"{needles}needle", "*.parquet")))])
    rows = []
    for r in df.itertuples():
        messages = json.loads(r.prompt)
        tokens = sum(len(tok.encode(m["content"]).ids) + 6 for m in messages)
        rows.append((tokens, messages, r.answer, r.random_string_to_prepend))
    for label, lo, hi in BINS:
        pick = [x for x in rows if lo < x[0] <= hi]
        pick = random.Random(f"mrcr-{needles}-{label}").sample(pick, min(20, len(pick)))
        with open(os.path.join(out, f"mrcr_{needles}n_{label}.jsonl"), "w", encoding="utf-8") as f:
            for i, (tokens, messages, answer, prefix) in enumerate(pick):
                f.write(json.dumps({"id": f"mrcr_{needles}n_{label}-{i}", "messages": messages, "answer": answer,
                                    "prefix": prefix, "prompt_tokens": tokens}) + "\n")
        print(needles, label, len(pick), "of", sum(lo < x[0] <= hi for x in rows))
