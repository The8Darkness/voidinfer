#!/bin/bash
# After queue_all2: redo the two 4K NInfer KV records cut short by the earlier pause, validate the
# chunked-attention long reference against the FP32 short-panel reference, then build the long
# references (FP32 compute for W <= 64K, BF16 compute for all windows from one 131K forward).
ROOT=/d/AI/quality-bench-20261016
cd $ROOT
true
echo "== $(date +%H:%M:%S) ninfer w4096 reruns"
for KV in bf16 fp8; do
  rm -f runs/ninfer_nv_$KV/w4096.csv runs/ninfer_nv_$KV/w4096.top100
  bash scripts/panel_ninfer.sh nv $KV 4608 $ROOT/long/w4096/manifest.json w4096
done
echo "== $(date +%H:%M:%S) validate chunked attention (FP32, seg 512) vs refs_fp32"
python -I scripts/bf16_ref_long.py --validate --fp32 --val-ref refs_fp32 --only cpp-00 --seg 512 2>&1 | grep -E "VALIDATE|Error"
python -I scripts/bf16_ref_long.py --validate --val-ref refs --only cpp-00 2>&1 | grep -E "VALIDATE cpp|Error"
echo "== $(date +%H:%M:%S) FP32 long reference (<= 64K)"
python -I scripts/bf16_ref_long.py --fp32 --max-len 66048 --out refs_long_fp32 > logs/fp32_long.log 2>&1; tail -1 logs/fp32_long.log
echo "== $(date +%H:%M:%S) BF16 long reference (all windows, 131K)"
python -I scripts/bf16_ref_long.py > logs/bf16_long.log 2>&1; tail -1 logs/bf16_long.log
echo "QUEUE_POST_DONE $(date +%H:%M:%S)"
