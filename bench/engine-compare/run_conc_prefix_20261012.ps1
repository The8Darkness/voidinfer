# NInfer concurrency with its prefix reuse ON (VoidInfer's server reuses shared prefixes by default; the
# fixtures put the code in a system message shared by all agents). Configurations NInfer can start.
$R='D:\AI\engine-compare-20261008\results\2026-10-12\conc'
$NV='D:/AI/models/ninfer/Qwen3.8-27B-nvfp4-NInfer/qwen3_8_27b_nvfp4.ninfer'
$cfg=@(@('code_4k',8192,1),@('code_4k',8192,2),@('code_4k',8192,4),@('code_31k',32768,2))
foreach($c in $cfg){
  $p,$ctx,$n=$c
  foreach($rep in 1,2){
    python D:\AI\engine-compare-20261008\ninfer_conc.py --artifact $NV --kv fp8 --spec dflash2 --conc $n --prompt $p --max-context $ctx --prefix-reuse --out "$R\ninfer.prefix.$p.c$n.rep$rep.json" 2>&1 | Select -Last 1 | % { $_ | Add-Content "$R\ninfer.prefix.jsonl"; "ninfer prefix $p C$n rep$rep $_" }
  }
}
'CONC_PREFIX_DONE'
