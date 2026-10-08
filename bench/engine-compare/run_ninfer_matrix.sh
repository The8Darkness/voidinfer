#!/bin/bash
cd /d/AI/engine-compare-20261008
GW=D:/AI/models/ninfer/Qwen3.8-27B-NInfer/qwen3_8_27b.ninfer
NV=D:/AI/models/ninfer/Qwen3.8-27B-nvfp4-NInfer/qwen3_8_27b_nvfp4.ninfer
SCEN=$(ls prompts/scenario_*.json | xargs -n1 basename | sed 's/\.json$//' | tr '\n' ' ')
for v in gw:$GW nv:$NV; do
  name=${v%%:*}; art=${v#*:}
  for kv in bf16 fp8; do
    if [ $kv = bf16 ]; then ctx=67584; P="code_4k code_16k code_31k code_32k code_64k"; else ctx=133120; P="code_4k code_16k code_31k code_32k code_64k code_128k"; fi
    python ninfer_bench.py --artifact $art --label $name --kv $kv --spec none --max-context $ctx --max-tokens 256 --repeats 2 --prefill-chunk 8192 --out results/ninfer/$name.$kv.base.json $P > results/ninfer/$name.$kv.base.out 2>&1
    echo "done $name $kv base $(tail -1 results/ninfer/$name.$kv.base.out | cut -c1-80)"
    python ninfer_bench.py --artifact $art --label $name --kv $kv --spec dflash2 --max-context $ctx --max-tokens 1024 --repeats 2 --prefill-chunk 8192 --out results/ninfer/$name.$kv.dflash2.json $P $SCEN > results/ninfer/$name.$kv.dflash2.out 2>&1
    echo "done $name $kv dflash2 $(tail -1 results/ninfer/$name.$kv.dflash2.out | cut -c1-80)"
  done
done
echo MATRIX_DONE
