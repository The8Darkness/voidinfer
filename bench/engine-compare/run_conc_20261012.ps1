# Concurrency comparison 2026-10-12: N simultaneous greedy requests (distinct "[agent i]" prompts from one
# fixture, 1,024 output tokens) after one warm request; NInfer nvfp4 / FP8 KV / DFlash2 vs VoidInfer default.
$R='D:\AI\engine-compare-20261008\results\2026-10-12\conc'; New-Item -ItemType Directory -Force $R | Out-Null
$NV='D:/AI/models/ninfer/Qwen3.8-27B-nvfp4-NInfer/qwen3_8_27b_nvfp4.ninfer'
$cfg=@(@('code_4k',8192,1),@('code_4k',8192,2),@('code_4k',8192,4),@('code_4k',8192,8),@('code_31k',32768,2),@('code_31k',32768,4),@('code_31k',32768,8))
foreach($c in $cfg){
  $p,$ctx,$n=$c
  foreach($rep in 1,2){
    $o=& D:\AI\engine-compare-20261008\ours_conc.ps1 -Conc $n -Prompt $p -MaxContext $ctx -Label "ours.$p.c$n.rep$rep" 2>&1 | Select -Last 1
    "$o" | Add-Content "$R\ours.jsonl"; "ours $p C$n rep$rep $o"
    python D:\AI\engine-compare-20261008\ninfer_conc.py --artifact $NV --kv fp8 --spec dflash2 --conc $n --prompt $p --max-context $ctx --out "$R\ninfer.$p.c$n.rep$rep.json" 2>&1 | Select -Last 1 | % { $_ | Add-Content "$R\ninfer.jsonl"; "ninfer $p C$n rep$rep $_" }
  }
}
'CONC_DONE'
