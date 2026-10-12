param([int]$Conc=4,[int]$Agents=8,[int]$Chars=400000,[int]$Tokens=256,[int]$Turns=2,[int]$MaxContext=262144,[int]$HostKvMib=20000,[string]$Out,[string]$Env='',[string]$Extra='')
# VoidInfer multi-agent long-context run: voidinfer-serve (EXL3, L0 OSCAR, exact VeriCache, default
# admission), --max-concurrency $Conc lanes, driven by agents_client.py (same requests as NInfer).
. "$PSScriptRoot\ours_env.ps1"
Reset-VoidEnv $Env
$port=18436; $log="$Out.server.log"
$a=@('--target',$VoidTarget,'--draft',$VoidDraft,'--host','127.0.0.1','--port',[string]$port,
 '--max-context',[string]$MaxContext,'--max-concurrency',[string]$Conc,'--max-pending-requests','16',
 '--no-thinking','--greedy','--temperature','0','--default-max-tokens',[string]$Tokens,'--max-shared-prefixes','16',
 '--host-kv-mib',[string]$HostKvMib,'--pending-timeout-ms','3600000','--model-id','conc')+($Extra -split ' ' | ? {$_})
$p=Start-Process -FilePath $VoidServe -ArgumentList $a -PassThru -NoNewWindow -RedirectStandardError $log -RedirectStandardOutput "$log.out"
try {
  $ok=$false; for($i=0;$i -lt 900;$i++){ Start-Sleep 1; if($p.HasExited){break}; try{ Invoke-RestMethod "http://127.0.0.1:$port/v1/models" -TimeoutSec 2 | Out-Null; $ok=$true; break }catch{} }
  if(-not $ok){ "server failed to start"; Get-Content $log -Tail 5; return }
  python "$PSScriptRoot\agents_client.py" --port $port --agents $Agents --chars $Chars --tokens $Tokens --turns $Turns --out $Out
} finally { if(-not $p.HasExited){ Stop-Process -Id $p.Id -Force } }
