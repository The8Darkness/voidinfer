# History consistency of a multi-agent run (agents_client.py output with texts): each answer's
# code identifiers (backticked, length >= 6) should occur in the agent's own source slice; an answer
# grounded in another agent's history would cite identifiers only that slice contains.
import json, re, sys
SRC = r"D:\AI\kvtier\long_code_source.cpp"
d = json.load(open(sys.argv[1], encoding="utf-8"))
src = open(SRC, encoding="utf-8", errors="ignore").read()
n, chars = d["agents"], d["chars"]
slices = [src[(chars + 1000) * i:(chars + 1000) * i + chars] for i in range(n)]
for turn in d["turns"]:
    print(f"turn {turn['turn']}:")
    for i, text in enumerate(turn.get("texts", [])):
        ids = {m.strip("()") for m in re.findall(r"`([A-Za-z_][A-Za-z0-9_:]{5,})(?:\(\))?`", text)}
        own = sum(1 for x in ids if x.split("::")[-1] in slices[i])
        foreign = [x for x in ids if x.split("::")[-1] not in slices[i] and
                   any(x.split("::")[-1] in slices[j] for j in range(n) if j != i)]
        print(f"  agent {i}: identifiers {len(ids)}, in own slice {own}, only in other slices {len(foreign)} {foreign[:4]}")
