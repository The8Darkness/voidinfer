#!/bin/bash
# GPU queue for the KV-precision benchmark (run after the weight panel). Sequential, resumable
# (panel_ours.ps1 skips finished streams; NInfer records are rewritten per run).
#   NInfer NVFP4: BF16 KV and FP8 KV, per window W (one window of W+512 tokens per stream).
#   Ours EXL3 6bpw: FP16 device KV | OSCAR INT4 + hot rows (raw L0 decode) | OSCAR INT2 + hot rows |
#   INT4 with exact history (VOIDINFER_EXL3_L0_EXACT=1 = the VeriCache verifier's distribution).
#   Prompt KV for the L0 arms is written from exact history as in serving (L0_EXACT=prefill).
ROOT=/d/AI/quality-bench-20261016
WINDOWS=${WINDOWS:-"4096 16384 32768 65536 131072"}
ARMS=${ARMS:-"fp16 int4raw int2raw int4vc"}
for W in $WINDOWS; do
  for KV in bf16 fp8; do
    if [ ! -s $ROOT/runs/ninfer_nv_$KV/w$W.csv ]; then
      echo "ninfer $KV w$W: $(bash $ROOT/scripts/panel_ninfer.sh nv $KV $((W+512)) $ROOT/long/w$W/manifest.json w$W | tr '\n' ' ')"
    fi
  done
done
for W in $WINDOWS; do
  for ARM in $ARMS; do
    case $ARM in
      fp16) ENV='VOIDINFER_EXL3_L0_OSCAR=0' ;;
      int4raw) ENV='VOIDINFER_EXL3_L0_EXACT=prefill' ;;
      int2raw) ENV='VOIDINFER_EXL3_L0_CODE_BITS=2;VOIDINFER_EXL3_L0_EXACT=prefill' ;;
      int4vc) ENV='VOIDINFER_EXL3_L0_EXACT=1' ;;
    esac
    pwsh -NoProfile -File "D:\\AI\\quality-bench-20261016\\scripts\\panel_ours.ps1" -Tag "kv_${ARM}_w$W" -Env "$ENV" \
      -Panel "D:\\AI\\quality-bench-20261016\\long\\w$W" -Context $W | tail -3
  done
done
echo QUEUE_KV_DONE
