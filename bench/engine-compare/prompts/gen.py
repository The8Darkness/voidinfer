# Same system/user messages for both engines: .pmsg (system \x1E user) for ours, .json for NInfer.
import json, os, glob
from tokenizers import Tokenizer
OUT = os.path.dirname(os.path.abspath(__file__))
tok = Tokenizer.from_file(r"D:\AI\models\Qwen3.8-27B-exl3-6bpw-H6-V6\tokenizer.json")
def emit(name, system, user):
    open(os.path.join(OUT, name + ".pmsg"), "w", encoding="utf-8", newline="").write(system + "\x1e" + user)
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
    json.dump({"messages": msgs}, open(os.path.join(OUT, name + ".json"), "w", encoding="utf-8"), ensure_ascii=False)
    n = len(tok.encode(system, add_special_tokens=False).ids) + len(tok.encode(user, add_special_tokens=False).ids)
    print(name, "raw tokens", n)
src = open(r"D:\AI\kvtier\long_code_source.cpp", encoding="utf-8", errors="replace").read()
lines = src.splitlines(keepends=True)
user = ("Write a detailed, numbered code review of the C++ source above. Cover at least forty distinct functions, "
        "state transitions, or failure paths, with a full paragraph for each. Continue consecutively through all "
        "forty items before concluding. Begin item one now.")
head = "Background reference source code:\n\n"
ul = len(tok.encode(user, add_special_tokens=False).ids)
# cumulative token counts per line (approximate; boundary merges are tiny)
cum, acc = [], 0
for ln in lines[:60000]:
    acc += len(tok.encode(ln, add_special_tokens=False).ids); cum.append(acc)
for N in (4096, 16384, 31488, 32768, 65536, 131072):
    budget = N - ul - 40   # chat-template overhead headroom; exact counts are reported by both engines
    k = max(i for i, c in enumerate(cum) if c <= budget) + 1
    emit(f"code_{N//1024}k" if N % 1024 == 0 else f"code_{N//1000}k", head + "".join(lines[:k]), user)
for f in sorted(glob.glob(r"C:\Users\Micha\AppData\Local\Temp\claude\D--AI\7a2592e2-6123-45e2-9a63-a91724763665\scratchpad\ninfer_msgs_padded\*.txt")):
    s = open(f, encoding="utf-8").read()
    system, user2 = s.split("\x1e", 1)
    emit(os.path.basename(f)[:-4], system, user2)
