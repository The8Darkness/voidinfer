#!/bin/bash
# NInfer max single-request context on the NVFP4 artifact: FP8 KV cannot reserve 262,144 tokens with DFlash2
# resident (needs 11.27 GB, 9.54 GB available), so try NInfer's smaller KV storages.
cd /d/AI/engine-compare-20261008
R=results/2026-10-12/ninfer
NV=D:/AI/models/ninfer/Qwen3.8-27B-nvfp4-NInfer/qwen3_8_27b_nvfp4.ninfer
for kv in k8v4 nvfp4; do
  python ninfer_bench.py --artifact $NV --label nv --kv $kv --spec dflash2 --max-context 262144 --max-tokens 1024 --repeats 1 --prefill-chunk 8192 --out $R/nv.$kv.dflash2.254k.json code_254k > $R/nv.$kv.dflash2.254k.out 2>&1
  echo "done nv $kv 254k $(tail -1 $R/nv.$kv.dflash2.254k.out | cut -c1-160)"
done
echo NINFER_254K_DONE
