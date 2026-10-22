"""RULER (NVIDIA/RULER, pinned commit in MANIFEST) task generation without prepare.py's shell string.

prepare.py builds one shell command with the multi-line template in double quotes, which cmd.exe
splits at the newlines (prompts came out 26-548 tokens). This wrapper reproduces prepare.py's config
merge (synthetic.yaml + constants.TASKS, 'base' model template, answer prefix appended) and calls the
same official generator scripts with an argument list. Tokenizer: Qwen3.8 HF tokenizer (hf type).
Run with the bench venv (wonderwords, nltk, tenacity, transformers).

  gen_ruler.py --lengths 16384 32768 65536 131072 --samples 50 50 30 20
"""
import argparse, os, subprocess, sys
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
RULER = os.path.join(ROOT, "repos", "RULER", "scripts", "data")
TOK = r"D:/AI/models/Qwen3.8-27B-exl3-6bpw-H6-V6"
TASKS = ["niah_multikey_1", "niah_multikey_2", "niah_multikey_3", "vt", "qa_2"]

ap = argparse.ArgumentParser()
ap.add_argument("--lengths", type=int, nargs="+", default=[16384, 32768, 65536, 131072])
ap.add_argument("--samples", type=int, nargs="+", default=[50, 50, 30, 20])
ap.add_argument("--tasks", nargs="+", default=TASKS)
ap.add_argument("--out", default=os.path.join(ROOT, "ruler"))
ap.add_argument("--seed", type=int, default=42)
a = ap.parse_args()

sys.path.insert(0, os.path.join(RULER, "synthetic"))
from constants import TASKS as BASE  # noqa: E402
custom = yaml.safe_load(open(os.path.join(RULER, "..", "synthetic.yaml")))
for L, n in zip(a.lengths, a.samples):
    for task in a.tasks:
        cfg = dict(custom[task]); cfg.update(BASE[cfg["task"]])
        template = cfg["template"] + cfg.get("answer_prefix", "")
        save = os.path.abspath(os.path.join(a.out, f"L{L}"))
        out = os.path.join(save, task, "validation.jsonl")
        if os.path.exists(out) and sum(1 for _ in open(out, encoding="utf-8")) == n:
            print(L, task, "exists"); continue
        cmd = [sys.executable, os.path.join(RULER, "synthetic", cfg["task"] + ".py"), "--save_dir", save,
               "--save_name", task, "--subset", "validation", "--tokenizer_path", TOK, "--tokenizer_type", "hf",
               "--max_seq_length", str(L), "--tokens_to_generate", str(cfg["tokens_to_generate"]),
               "--num_samples", str(n), "--random_seed", str(a.seed), "--template", template]
        for k, v in cfg["args"].items():
            cmd += [f"--{k}", str(v)]
        if cfg["task"] == "qa":
            cmd += ["--pre_samples", "0"]
        r = subprocess.run(cmd, cwd=os.path.join(RULER, "synthetic"), capture_output=True, text=True)
        print(L, task, "exit", r.returncode, (r.stderr or "")[-400:].strip(), flush=True)
