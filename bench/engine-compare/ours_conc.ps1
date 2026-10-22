param([int]$Conc=2,[int]$MaxContext=8192,[int]$Tokens=1024,[string]$Prompt='code_4k',[string]$Env='',[string]$Label='')
# Our side of the concurrency comparison: voidinfer-serve (EXL3, L0 OSCAR, exact VeriCache) with
# --max-concurrency $Conc, driven by conc_client.py exactly like ninfer_conc.py drives NInfer.
. "$PSScriptRoot\ours_env.ps1"
Reset-VoidEnv $Env
$port=18433; $log="D:\AI\kvtier\ours_conc.log"
$a=@('--target',$VoidTarget,'--draft',$VoidDraft,'--host','127.0.0.1','--port',[string]$port,
 '--max-context',[string]$MaxContext,'--max-concurrency',[string]$Conc,'--max-pending-requests','8',
 '--no-thinking','--greedy','--temperature','0','--default-max-tokens',[string]$Tokens,'--max-private-continuations','16','--host-kv-mib','8000','--model-id','conc')
$p=Start-Process -FilePath $VoidServe -ArgumentList $a -PassThru -NoNewWindow -RedirectStandardError $log -RedirectStandardOutput "$log.out"
try {
  $ok=$false
  for($i=0;$i -lt 600;$i++){ Start-Sleep -Milliseconds 1000; if($p.HasExited){break}; try{ Invoke-RestMethod "http://127.0.0.1:$port/v1/models" -TimeoutSec 2 | Out-Null; $ok=$true; break }catch{} }
  if(-not $ok){ "server failed to start"; Get-Content $log -Tail 5; return }
  python "$PSScriptRoot\conc_client.py" --port $port --conc $Conc --prompt $Prompt --max-tokens $Tokens --label $Label
} finally { if(-not $p.HasExited){ Stop-Process -Id $p.Id -Force } }
