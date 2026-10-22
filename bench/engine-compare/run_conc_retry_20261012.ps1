# NInfer concurrency configurations whose static reservation (KV = N x max context) exceeds free VRAM:
# retried with NInfer's shared --kv-capacity auto pool (NVFP4, then the smaller groupwise artifact).
$R='D:\AI\engine-compare-20261008\results\2026-10-12\conc'
$NV='D:/AI/models/ninfer/Qwen3.8-27B-nvfp4-NInfer/qwen3_8_27b_nvfp4.ninfer'
$GW='D:/AI/models/ninfer/Qwen3.8-27B-NInfer/qwen3_8_27b.ninfer'
$cfg=@(@('code_4k',8192,8),@('code_31k',32768,4),@('code_31k',32768,8))
foreach($art in @(@('nv',$NV),@('gw',$GW))){
  $lab,$path=$art
  foreach($c in $cfg){
    $p,$ctx,$n=$c
    foreach($rep in 1,2){
      python D:\AI\engine-compare-20261008\ninfer_conc.py --artifact $path --kv fp8 --spec dflash2 --conc $n --prompt $p --max-context $ctx --kv-capacity auto --out "$R\ninfer.$lab.auto.$p.c$n.rep$rep.json" 2>&1 | Select -Last 1 | % { $_ | Add-Content "$R\ninfer.auto.jsonl"; "ninfer $lab auto $p C$n rep$rep $_" }
    }
  }
}
'CONC_RETRY_DONE'
