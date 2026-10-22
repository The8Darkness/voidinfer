#!/bin/bash
# NInfer side of the weight panel: ninfer-perplexity, one window per 2048-token stream
# (--context 2048 --stride 2047: one window, scores positions predicting tokens [1, 2048)), top-100 record +
# per-token target log-probs. Usage: panel_ninfer.sh ART_TAG KV [CONTEXT]   (ART_TAG nv|gw, KV bf16|fp8)
set -u
ROOT=/d/AI/quality-bench-20261016
PPL=/d/AI/ninfer-5090-windows/build/apps/ninfer-perplexity.exe
case $1 in
  nv) ART=D:/AI/models/ninfer/Qwen3.8-27B-nvfp4-NInfer/qwen3_8_27b_nvfp4.ninfer ;;
  gw) ART=D:/AI/models/ninfer/Qwen3.8-27B-NInfer/qwen3_8_27b.ninfer ;;
esac
KV=$2; C=${3:-2048}; MAN=${4:-$ROOT/panel/manifest.json}; NAME=${5:-panel}
D=${OUTDIR:-$ROOT/runs/ninfer_$1_$KV}; mkdir -p $D
s=$(date +%s)
$PPL $ART --corpus $MAN --context $C --stride $((C-1)) --kv-dtype $KV --topk-record $D/$NAME.top100 --topk-k 100 \
  > $D/$NAME.topk.stdout 2> $D/$NAME.topk.stderr; echo "topk exit $? $(( $(date +%s)-s ))s"
s=$(date +%s); rm -rf $D/$NAME.report
$PPL $ART --corpus $MAN --context $C --stride $((C-1)) --kv-dtype $KV --per-token-logprobs $D/$NAME.csv --output $D/$NAME.report \
  > $D/$NAME.ppl.stdout 2> $D/$NAME.ppl.stderr; echo "ppl exit $? $(( $(date +%s)-s ))s"
