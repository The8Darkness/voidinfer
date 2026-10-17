#!/bin/bash
# Long references (after the retry/rerun jobs): FP32 compute for W <= 32K (FP32 attention runs on CUDA
# cores: ~15 min per attention layer at 66K, so 64K is left to BF16), BF16 compute for every window from
# one 131K forward (block-wise masked SDPA). The 4K-32K overlap measures BF16's own noise floor.
cd /d/AI/quality-bench-20261016
echo "== $(date +%H:%M:%S) FP32 long reference (<= 32K)"
python -I scripts/bf16_ref_long.py --fp32 --max-len 33280 --out refs_long_fp32 > logs/fp32_long.log 2>&1; tail -1 logs/fp32_long.log
echo "== $(date +%H:%M:%S) BF16 long reference (all windows)"
python -I scripts/bf16_ref_long.py > logs/bf16_long.log 2>&1; tail -1 logs/bf16_long.log
echo "QUEUE_REFS_DONE $(date +%H:%M:%S)"
