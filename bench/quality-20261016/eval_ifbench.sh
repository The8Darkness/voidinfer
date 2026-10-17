#!/bin/bash
# IFBench scoring (runs in WSL; official IFBench run_eval at bench3 pin 1c40f0c1) for a runs_tasks dir.
# Usage (Windows): wsl bash /mnt/d/AI/quality-bench-20261016/scripts/eval_ifbench.sh <run-dir-name>
set -e
E=$1; Q=/mnt/d/AI/quality-bench-20261016; B=/mnt/d/AI/bench3-20261014
. ~/bench3/.venv/bin/activate
cd $B/repos/IFBench
mkdir -p $Q/results/ifbench
python - "$E" <<'PY'
import json, sys
Q = "/mnt/d/AI/quality-bench-20261016"; B = "/mnt/d/AI/bench3-20261014"; e = sys.argv[1]
items = {json.loads(l)["id"]: json.loads(l) for l in open(f"{Q}/items/ifbench.jsonl", encoding="utf-8")}
recs = {}
for l in open(f"{Q}/runs_tasks/{e}/ifbench.jsonl", encoding="utf-8"):
    r = json.loads(l)
    if "error" not in r: recs[r["id"]] = r
data = [json.loads(l) for l in open(f"{B}/repos/IFBench/data/IFBench_test.jsonl", encoding="utf-8")]
with open(f"{Q}/results/ifbench/{e}.input.jsonl", "w", encoding="utf-8") as f:
    for d in data:
        if d["key"] in recs: f.write(json.dumps(d, ensure_ascii=False) + "\n")
with open(f"{Q}/results/ifbench/{e}-responses.jsonl", "w", encoding="utf-8") as f:
    for d in data:
        k = d["key"]
        if k in recs:
            f.write(json.dumps({"prompt": items[k]["messages"][0]["content"], "response": recs[k]["content"]}, ensure_ascii=False) + "\n")
print("answered", len(recs))
PY
python -m run_eval --input_data=$Q/results/ifbench/$E.input.jsonl --input_response_data=$Q/results/ifbench/$E-responses.jsonl --output_dir=$Q/results/ifbench/$E 2>&1 | grep -A3 "^prompt-level\|^instruction-level\|=====" | head -20
