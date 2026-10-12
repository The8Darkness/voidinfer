#!/bin/bash
# NInfer side of the 2026-10-12 comparison (single request, greedy, thinking off, no prefix reuse,
# warm + 2 measured repeats, --prefill-chunk 8192, FP8 E4M3 KV = NInfer's recommended KV).
cd /d/AI/engine-compare-20261008
R=results/2026-10-12/ninfer
GW=D:/AI/models/ninfer/Qwen3.8-27B-NInfer/qwen3_8_27b.ninfer
NV=D:/AI/models/ninfer/Qwen3.8-27B-nvfp4-NInfer/qwen3_8_27b_nvfp4.ninfer
SCEN=$(ls prompts/scenario_*.json | xargs -n1 basename | sed 's/\.json$//' | tr '\n' ' ')
CODE="code_4k code_16k code_32k code_64k code_128k"
for v in nv:$NV gw:$GW; do
  name=${v%%:*}; art=${v#*:}
  python ninfer_bench.py --artifact $art --label $name --kv fp8 --spec dflash2 --max-context 133120 --max-tokens 1024 --repeats 2 --prefill-chunk 8192 --out $R/$name.fp8.dflash2.json $CODE $SCEN > $R/$name.fp8.dflash2.out 2>&1
  echo "done $name dflash2 $(tail -1 $R/$name.fp8.dflash2.out | cut -c1-100)"
  python ninfer_bench.py --artifact $art --label $name --kv fp8 --spec none --max-context 133120 --max-tokens 256 --repeats 2 --prefill-chunk 8192 --out $R/$name.fp8.base.json code_4k code_32k code_128k > $R/$name.fp8.base.out 2>&1
  echo "done $name base $(tail -1 $R/$name.fp8.base.out | cut -c1-100)"
  python ninfer_bench.py --artifact $art --label $name --kv fp8 --spec dflash2 --max-context 262144 --max-tokens 1024 --repeats 1 --prefill-chunk 8192 --out $R/$name.fp8.dflash2.254k.json code_254k > $R/$name.fp8.dflash2.254k.out 2>&1
  echo "done $name 254k $(tail -1 $R/$name.fp8.dflash2.254k.out | cut -c1-100)"
done
echo NINFER_C1_DONE
