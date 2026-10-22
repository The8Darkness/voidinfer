param([int]$Conc=1,[int]$MaxContext=32768,[string]$Env='',[string]$Out='D:\AI\kvtier\vision_try.json',[switch]$NoVision,[switch]$Hold,[switch]$Suite,[int]$Concurrent=0,[string]$Cases='scene,chart,compare,long_prompt,video,text_after')
# Vision: voidinfer-serve (L0 OSCAR, exact VeriCache) with the BF16 Vision tower; a short text
# prompt, then the two NInfer example images (vision_client.py) or the vision suite (-Suite).
. "$PSScriptRoot\ours_env.ps1"
Reset-VoidEnv ('VOIDINFER_EXL3_VISION_STATS=1;VOIDINFER_EXL3_VERICACHE_STATS=1;'+$Env)
$port=18440; $log="D:\AI\kvtier\vision_try.log"
$a=@('--target',$VoidTarget,'--draft',$VoidDraft,'--host','127.0.0.1','--port',[string]$port,
 '--max-context',[string]$MaxContext,'--max-concurrency',[string]$Conc,
 '--no-thinking','--greedy','--temperature','0','--model-id','ours')
if(-not $NoVision){ $a+=@('--vision',$VoidVision) }
$p=Start-Process -FilePath $VoidServe -ArgumentList $a -PassThru -NoNewWindow -RedirectStandardError $log -RedirectStandardOutput "$log.out"
try {
  $base="http://127.0.0.1:$port"; $ok=$false
  for($i=0;$i -lt 900;$i++){ Start-Sleep 1; if($p.HasExited){break}; try{ Invoke-RestMethod "$base/v1/models" -TimeoutSec 2 | Out-Null; $ok=$true; break }catch{} }
  if(-not $ok){ "server failed to start (exit $($p.ExitCode))"; Get-Content $log -Tail 8; return }
  "startup: GPU used $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) MiB"
  foreach($q in @('hi','What is 17*23? Answer with the number only.')){
    $body=@{model='ours';max_tokens=64;temperature=0;messages=@(@{role='user';content=$q})} | ConvertTo-Json -Depth 5
    $sw=[Diagnostics.Stopwatch]::StartNew()
    try{ $r=Invoke-RestMethod "$base/v1/chat/completions" -Method Post -ContentType 'application/json' -Body $body -TimeoutSec 600
      "text '$q': $($sw.ElapsedMilliseconds) ms pt=$($r.usage.prompt_tokens) ct=$($r.usage.completion_tokens) -> $($r.choices[0].message.content)" }
    catch{ "text '$q' FAILED: $($_.ErrorDetails.Message)" }
  }
  if($Suite){ python "$PSScriptRoot\vision_suite.py" --port $port --out $Out --cases $Cases --concurrent $Concurrent }
  elseif(-not $NoVision){ python "$PSScriptRoot\vision_client.py" --port $port --label ours.exl3-vision --out $Out }
  if($Hold){ Read-Host 'hold' }
} finally { if(-not $p.HasExited){ Stop-Process -Id $p.Id -Force; Start-Sleep 2 } ; Get-Content $log | Select-String 'VISION|vision|error|Error|exception' | Select -Last 20 }
