#!/bin/bash
# Master GPU queue after the weight panel. Each step is resumable; logs in logs/.
ROOT=/d/AI/quality-bench-20261016
cd $ROOT
PY="python -I"
VPY=$ROOT/.venv/Scripts/python
step() { echo "== $(date +%H:%M:%S) $*"; }

step "KV benchmark first"

step "KV benchmark (NInfer bf16/fp8 + ours arms)"
bash scripts/queue_kv.sh > logs/queue_kv.log 2>&1; tail -1 logs/queue_kv.log

step "long references: FP32 to 64K, BF16 to 128K"
$PY scripts/bf16_ref_long.py --fp32 --max-len 66048 --out refs_long_fp32 > logs/fp32_long.log 2>&1; tail -1 logs/fp32_long.log
$PY scripts/bf16_ref_long.py --windows 131072 > logs/bf16_long.log 2>&1; tail -1 logs/bf16_long.log

step "practical tasks"
for f in ruler_L16384 ruler_L32768 ruler_L65536 ruler_L131072 nolima_L4000 nolima_L16000 nolima_L32000 ifbench; do
  for e in ours ninfer; do
    $PY scripts/run_tasks.py --engine $e --items items/$f.jsonl --conc 1 >> logs/tasks_$e.log 2>&1
    echo "$e $f: $(tail -1 logs/tasks_$e.log | cut -c1-200)"
  done
done
echo QUEUE_ALL_DONE
