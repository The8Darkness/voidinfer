#!/bin/bash
# NInfer side of the paired NLL: --context W --stride 512 on texts of exactly W+512 tokens;
# window 2 scores tokens [W, W+512) given [512, W), matching our research_exl3_long_nll run.
cd /d/AI/engine-compare-20261008/quality
PPL=/d/AI/ninfer-5090-windows/build/apps/ninfer-perplexity.exe
GW=D:/AI/models/ninfer/Qwen3.8-27B-NInfer/qwen3_8_27b.ninfer
NV=D:/AI/models/ninfer/Qwen3.8-27B-nvfp4-NInfer/qwen3_8_27b_nvfp4.ninfer
for v in gw:$GW nv:$NV; do name=${v%%:*}; art=${v#*:}
  for kv in bf16 fp8; do
    for W in 4096 16384 32768; do
      d=w$W; rm -rf $d/ninfer.$name.$kv.report
      $PPL $art --corpus $d/manifest.json --context $W --stride 512 --kv-dtype $kv --per-token-logprobs $d/ninfer.$name.$kv.csv --output $d/ninfer.$name.$kv.report > $d/ninfer.$name.$kv.stdout 2> $d/ninfer.$name.$kv.stderr
      echo "ppl $name $kv $W exit $?"
      $PPL $art --corpus $d/manifest.json --context $W --stride 512 --kv-dtype $kv --topk-record $d/ninfer.$name.$kv.top1 --topk-k 1 > /dev/null 2> $d/ninfer.$name.$kv.top1.stderr
      echo "top1 $name $kv $W exit $?"
    done
  done
done
echo QUALITY_DONE
