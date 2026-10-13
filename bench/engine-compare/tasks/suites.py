"""Task suites for the real-world quality comparison: items, prompts and scorers.

Every engine receives the identical user message, greedy decoding, no stop strings (VoidInfer rejects
them) and a per-suite output cap. Scores use only the answer content (not reasoning_content).
"""
import json, os, random, re, subprocess, sys, tempfile

DATA = os.environ.get("QB_DATA", r"D:\AI\quality-bench-20261012\data")
IFEVAL_LIB = os.environ.get("QB_IFEVAL_LIB", r"D:\AI\quality-bench-20261012\ifeval_lib")
BOXED = "Please reason step by step, and put your final answer within \\boxed{}."


def _jsonl(name):
    with open(os.path.join(DATA, name + ".jsonl"), encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def last_boxed(text):
    i = text.rfind("\\boxed")
    if i < 0:
        return None
    j = text.find("{", i)
    if j < 0:
        return None
    depth = 0
    for k in range(j, len(text)):
        if text[k] == "{":
            depth += 1
        elif text[k] == "}":
            depth -= 1
            if depth == 0:
                return text[j + 1:k]
    return None


def _num(s):
    s = s.replace(",", "").replace("$", "").strip()
    m = re.findall(r"-?\d+(?:\.\d+)?", s)
    return float(m[-1]) if m else None


# ---- GSM8K ---------------------------------------------------------------------------------
def gsm8k_items():
    for i, r in enumerate(_jsonl("gsm8k")):
        gold = float(r["answer"].split("####")[-1].replace(",", "").strip())
        yield {"id": f"gsm8k-{i}", "prompt": f"{r['question']}\n{BOXED}", "gold": gold}


def gsm8k_score(item, text):
    b = last_boxed(text)
    v = _num(b) if b is not None else _num(text)
    return float(v is not None and abs(v - item["gold"]) < 1e-6)


# ---- MATH-500 ------------------------------------------------------------------------------
def math500_items():
    for r in _jsonl("math500"):
        yield {"id": "math500-" + r["unique_id"], "prompt": f"{r['problem']}\n{BOXED}", "gold": r["answer"]}


def math500_score(item, text):
    from math_verify import parse, verify
    b = last_boxed(text)
    if b is None:
        return 0.0
    try:
        # Timeouts off: math_verify enforces them with subprocesses, which fail on Windows.
        return float(verify(parse("$" + item["gold"] + "$", parsing_timeout=None),
                            parse("$" + b + "$", parsing_timeout=None), timeout_seconds=None))
    except Exception:
        return 0.0


# ---- MMLU-Pro (1,000-question random sample, seed 0) ----------------------------------------
LETTERS = "ABCDEFGHIJ"


def mmlu_pro_items():
    rows = _jsonl("mmlu_pro")
    for r in random.Random(0).sample(rows, 1000):
        opts = "\n".join(f"{LETTERS[k]}. {o}" for k, o in enumerate(r["options"]))
        prompt = (f"The following is a multiple choice question about {r['category']}. Think step by step "
                  f"and then finish your answer with \"the answer is (X)\" where X is the correct letter choice.\n\n"
                  f"Question: {r['question']}\nOptions:\n{opts}")
        yield {"id": f"mmlu_pro-{r['question_id']}", "prompt": prompt, "gold": r["answer"], "category": r["category"]}


def mmlu_pro_score(item, text):
    m = re.findall(r"answer is \(?([A-J])\)?", text)
    if not m:
        m = re.findall(r"[aA]nswer:\s*\(?([A-J])\)?", text)
    return float(bool(m) and m[-1] == item["gold"])


# ---- IFEval (official google-research checker; prompt-level strict and loose) ---------------
def ifeval_items():
    for r in _jsonl("ifeval"):
        yield {"id": f"ifeval-{r['key']}", "prompt": r["prompt"], "instruction_id_list": r["instruction_id_list"],
               "kwargs": r["kwargs"]}


def _ifeval_follow(item, response):
    if IFEVAL_LIB not in sys.path:
        sys.path.insert(0, IFEVAL_LIB)
    from instruction_following_eval import instructions_registry as reg
    ok = []
    for k, iid in enumerate(item["instruction_id_list"]):
        inst = reg.INSTRUCTION_DICT[iid](iid)
        kw = {a: b for a, b in (item["kwargs"][k] or {}).items() if b is not None}
        inst.build_description(**kw)
        args = inst.get_instruction_args()
        if args and "prompt" in args:
            inst.build_description(prompt=item["prompt"])
        ok.append(bool(response.strip()) and inst.check_following(response))
    return ok


def ifeval_detail(item, text):
    strict = _ifeval_follow(item, text)
    lines = text.split("\n")
    variants = [text, "\n".join(lines[1:]).strip(), "\n".join(lines[:-1]).strip(), "\n".join(lines[1:-1]).strip()]
    variants += [v.replace("*", "") for v in variants]
    loose = [any(_ifeval_follow({**item, "instruction_id_list": [iid], "kwargs": [item["kwargs"][k]]}, v)[0]
                 for v in variants) for k, iid in enumerate(item["instruction_id_list"])]
    return {"strict_prompt": float(all(strict)), "loose_prompt": float(all(loose)),
            "strict_inst": strict, "loose_inst": loose}


def ifeval_score(item, text):
    return ifeval_detail(item, text)["strict_prompt"]


# ---- HumanEval (generated code runs in an isolated subprocess with a timeout) ---------------
def humaneval_items():
    for r in _jsonl("humaneval"):
        prompt = ("Complete the following Python function. Return the complete function (signature, body and any "
                  "imports it needs) in a single ```python code block.\n\n```python\n" + r["prompt"] + "```")
        yield {"id": r["task_id"].replace("/", "-"), "prompt": prompt, "code_prompt": r["prompt"],
               "test": r["test"], "entry_point": r["entry_point"]}


def _code(text):
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, re.S)
    return max(blocks, key=len) if blocks else text


def humaneval_score(item, text):
    code = _code(text)
    if f"def {item['entry_point']}" not in code:
        code = item["code_prompt"] + code
    header = "\n".join(l for l in item["code_prompt"].splitlines() if l.startswith(("import ", "from ")))
    program = header + "\n" + code + "\n\n" + item["test"] + f"\ncheck({item['entry_point']})\n"
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "prog.py")
        open(path, "w", encoding="utf-8").write(program)
        try:
            r = subprocess.run([sys.executable, "-I", path], cwd=d, capture_output=True, timeout=20)
            return float(r.returncode == 0)
        except subprocess.TimeoutExpired:
            return 0.0


# ---- AIME 2025 (thinking) ----------------------------------------------------------------------
def aime_items():
    for r in _jsonl("aime2025"):
        yield {"id": f"aime2025-{r['problem_idx']}", "prompt": f"{r['problem']}\n{BOXED}", "gold": int(r["answer"])}


def aime_score(item, text):
    b = last_boxed(text)
    v = _num(b) if b is not None else None
    return float(v is not None and int(v) == item["gold"])


# ---- Long-context multi-key retrieval in real code (built by make_longctx.py) -----------------
def longctx_items(length):
    for r in _jsonl(f"longctx_{length}"):
        yield r


def longctx_detail(item, text):
    hits = [bool(re.search(rf"\b{re.escape(n)}\s*[=:]\s*`?{v}\b", text, re.I)) for n, v in item["gold"].items()]
    return {"all": float(all(hits)), "keys": sum(hits) / len(hits)}


def longctx_score(item, text):
    return longctx_detail(item, text)["all"]


# ---- Hard long context (built by make_longctx_hard.py) ----------------------------------------------
def hard_items(task, length):
    return lambda: _jsonl(f"hard_{task}_{length}")


def _kv_hits(gold, text):
    def pattern(name):
        return r"[\s_-]+".join(re.escape(w) for w in re.split(r"[\s_-]+", name))
    return [bool(re.search(rf"(?<![\w-]){pattern(n)}\s*[=:]\s*`?{v}\b", text, re.I)) for n, v in gold.items()]


def hard_detail(item, text):
    if "gold_set" in item:
        found = set(re.findall(r"\b[A-Z]{5}\b", text))
        gold = set(item["gold_set"])
        return {"all": float(found == gold), "recall": len(found & gold) / len(gold),
                "precision": len(found & gold) / max(1, len(found))}
    hits = _kv_hits(item["gold"], text)
    return {"all": float(all(hits)), "recall": sum(hits) / len(hits), "precision": None}


def hard_score(item, text):
    return hard_detail(item, text)["all"]


# ---- Competition math with thinking: AIME 2024/2026, HMMT Feb 2025, BRUMO 2025 (MathArena / H4) --
def compmath_items():
    for name in ("aime2024", "aime2026", "hmmt_feb2025", "brumo2025"):
        for r in _jsonl(name):
            yield {"id": f"{name}-{r['problem_idx']}", "prompt": f"{r['problem']}\n{BOXED}", "gold": str(r["answer"])}


def compmath_score(item, text):
    return math500_score(item, text)


# ---- OpenAI MRCR (official grading: hash prefix required, then difflib ratio) ----------------------
def mrcr_score(item, text):
    from difflib import SequenceMatcher
    if not text.startswith(item["prefix"]):
        return 0.0
    return SequenceMatcher(None, text.removeprefix(item["prefix"]), item["answer"].removeprefix(item["prefix"])).ratio()


# name -> (items, scorer, max_tokens, thinking)
SUITES = {
    "gsm8k": (gsm8k_items, gsm8k_score, 4096, False),
    "math500": (math500_items, math500_score, 4096, False),
    "mmlu_pro": (mmlu_pro_items, mmlu_pro_score, 4096, False),
    "ifeval": (ifeval_items, ifeval_score, 4096, False),
    "humaneval": (humaneval_items, humaneval_score, 2048, False),
    "aime2025": (aime_items, aime_score, 32768, True),
    "compmath": (compmath_items, compmath_score, 32768, True),
    "longctx_32k": (lambda: longctx_items("32k"), longctx_score, 256, False),
    "longctx_64k": (lambda: longctx_items("64k"), longctx_score, 256, False),
    "longctx_128k": (lambda: longctx_items("128k"), longctx_score, 256, False),
}
for _task in ("mk", "vt", "mt"):
    for _length in ("32k", "128k", "200k"):
        SUITES[f"hard_{_task}_{_length}"] = (hard_items(_task, _length), hard_score, 512, False)
for _length in ("64k", "128k", "200k"):
    SUITES[f"hardx_mt_{_length}"] = ((lambda l: lambda: _jsonl(f"hardx_mt_{l}"))(_length), hard_score, 4096, False)
for _needles in (2, 4):
    for _length in ("64k", "128k", "196k"):
        SUITES[f"mrcr_{_needles}n_{_length}"] = ((lambda n, l: lambda: _jsonl(f"mrcr_{n}n_{l}"))(_needles, _length),
                                                  mrcr_score, 4096, False)
