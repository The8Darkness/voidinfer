#!/bin/bash
# After queue_all2: retry every item our engine left unanswered. Two engine availability bugs made the
# default server unavailable mid-run (both logged in the server logs):
#   - 128K RULER: "KV spill write" (spill file on the 98%-full C: drive) -> engine unavailable;
#   - NoLiMa 16K: "tap rows left the shared pool (another context claimed it)" -> engine unavailable.
# Retries restart the server with the spill directory on D: and VOIDINFER_EXL3_SHARE_TAPS=0 (tap rows
# kept per context instead of in the shared pool: storage location only, same values/numerics),
# until no item is left (max 15 restarts per file). NInfer is retried the same way (no overrides).
# Then the post jobs (queue_post.sh: NInfer 4K KV reruns, reference validation, long references).
ROOT=/d/AI/quality-bench-20261016
cd $ROOT
until grep -q QUEUE_ALL_DONE logs/queue_all2b.log; do sleep 60; done
missing() {
python -I - "$1" "$2" <<'PY'
import json, os, sys
e, f = sys.argv[1], sys.argv[2]
p = f"runs_tasks/{e}/{f}.jsonl"
ok = set()
if os.path.exists(p):
    for l in open(p, encoding="utf-8"):
        r = json.loads(l)
        if "error" not in r:
            ok.add(r["id"])
items = {json.loads(l)["id"] for l in open(f"items/{f}.jsonl", encoding="utf-8")}
print(len(items - ok))
PY
}
for f in ruler_L16384 ruler_L32768 ruler_L65536 ruler_L131072 nolima_L4000 nolima_L16000 nolima_L32000 ifbench; do
  for e in ours ninfer; do
    for i in $(seq 1 15); do
      n=$(missing $e $f)
      [ "$n" = "0" ] && break
      echo "== $(date +%H:%M:%S) retry $e $f #$i: $n unanswered"
      if [ $e = ours ]; then
        python -I scripts/run_tasks.py --engine ours --items items/$f.jsonl --conc 1 \
          --env "VOIDINFER_EXL3_KV_SPILL_DIR=D:\\AI\\kv-spill" "VOIDINFER_EXL3_SHARE_TAPS=0" >> logs/tasks_retry.log 2>&1
      else
        python -I scripts/run_tasks.py --engine ninfer --items items/$f.jsonl --conc 1 >> logs/tasks_retry.log 2>&1
      fi
      tail -1 logs/tasks_retry.log | cut -c1-200
    done
  done
done
echo "RETRIES_DONE $(date +%H:%M:%S)"
bash scripts/queue_post.sh
