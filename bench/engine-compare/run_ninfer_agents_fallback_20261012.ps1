# NInfer multi-agent run: the default attempt (NVFP4, FP8 KV, 4 lanes, 131,072 context) cannot reserve its
# minimum runtime; try NInfer's next settings in order and keep the first that starts.
$E='D:\AI\engine-compare-20261008'; $A="$E\results\2026-10-12\agents"
$NV='D:/AI/models/ninfer/Qwen3.8-27B-nvfp4-NInfer/qwen3_8_27b_nvfp4.ninfer'
$GW='D:/AI/models/ninfer/Qwen3.8-27B-NInfer/qwen3_8_27b.ninfer'
$tries=@(@('nv.k8v4.c4',$NV,'k8v4',4),@('gw.fp8.c4',$GW,'fp8',4),@('nv.fp8.c2',$NV,'fp8',2))
foreach($t in $tries){
  $lab,$art,$kv,$c=$t
  "try $lab"
  python "$E\ninfer_agents.py" --artifact $art --kv $kv --conc $c --max-context 110592 --out "$A\ninfer.$lab.json" 2>&1 | Select -Last 4
  if(Test-Path "$A\ninfer.$lab.json"){ Copy-Item "$A\ninfer.$lab.json" "$A\ninfer.json"; "ninfer agents ran: $lab"; break }
}
'NINFER_AGENTS_DONE'
